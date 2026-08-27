"""The backfill worker: one channel's existing history, oldest first, through
the extraction chain that already exists.

**Nothing about how a fact is recognised lives here.** The first filter
(aura.extraction.fact_worthiness), the distillation call
(aura.extraction.distiller), the dedup comparison, the supersession proposal and
its cap are all reached exactly as the live path reaches them -- through
distill_facts and stage_distilled_candidates, the same two functions
aura.extraction.pipeline calls, with the same Settings. That is the phase
brief's central constraint and it is structural rather than a promise: this
module contains no threshold, no prompt and no model name. What it contains is
the four things the live path does not need and cannot supply -- where to start,
what order to go in, when to stop, and how to survive being interrupted.

------------------------------------------------------------------------------
BATCHING BY COUNT, NOT BY TIME, AND WHY THAT REMOVES A TABLE
------------------------------------------------------------------------------

Live extraction batches by a five-minute window because it is waiting for
messages that have not been written yet, and something has to hold the ones that
have -- which is what extraction_queue is for, and why it is the one place in
Aura that stores raw message text.

Backfill waits for nothing: the entire history is already written down. So it
fills a batch to EXTRACTION_BATCH_MAX_MESSAGES fact-worthy candidates and sends
it immediately, and it needs no queue table at all, because **Discord is the
durable store and the cursor is the whole of the state.** (cursor_message_id,
until_message_id) describes exactly what is left to do, so a container that dies
mid-batch loses nothing but the work of re-fetching one page.

------------------------------------------------------------------------------
THE ORDERING GUARANTEE, AND WHY IT IS NOT DECORATIVE
------------------------------------------------------------------------------

Supersession judgements are judgements about which of two statements came
later. Feed a channel's history newest-first, or feed a page unsorted, and a
rule that was raised from 3 to 5 and then lowered back to 3 finishes its
backfill claiming 5 -- with a confident model-written reason attached, and
nothing downstream able to tell. So order is enforced in three places rather
than assumed once:

  1. Every page is sorted and bounded by aura.backfill.history.ordered_page,
     independently of discord.py's own (correct) ordering.
  2. A batch is checked with is_strictly_increasing before it is distilled, and
     a batch that fails is re-sorted rather than sent.
  3. The cursor itself is monotonic in SQL: advance_cursor refuses to move
     backwards, so no retry or second worker can replay an older stretch.

------------------------------------------------------------------------------
THE BOUNDARY WITH LIVE EXTRACTION: THREE LAYERS, NOT ONE
------------------------------------------------------------------------------

A message must never be processed by both paths. One bound is not enough to
guarantee that, because the two paths overlap in a narrow window nobody chose:

  1. THE UPPER BOUND. A run's until_message_id is the snowflake of the moment it
     started. Every message written from that instant onward is the live path's,
     and backfill never asks Discord for it.
  2. MESSAGES THE LIVE PATH IS HOLDING RIGHT NOW. A message written four minutes
     before /aura-backfill start is below the upper bound AND sitting in
     extraction_queue waiting for its window to close. Backfill skips exactly
     those (queued_message_ids), because the live path is about to pay for them.
  3. MESSAGES THE LIVE PATH ALREADY FINISHED. A message the live path distilled
     an hour ago is below the upper bound and no longer queued, but already has
     a candidate. Backfill skips those too (staged_message_ids), in any state --
     including a candidate a moderator has already confirmed or discarded, which
     is precisely the one it must not put back in front of them as new work.

Nothing flows the other way: on_message only ever fires for a message being
written now, and the live path's edit/delete handling only ever WITHDRAWS. There
is no path by which live extraction reaches into backfill's territory.

Check 3 also does a second job nobody designed it for and which is worth
recording: it makes backfill's own crash-retry free rather than merely
idempotent. A tick that staged candidates and then died before advancing its
cursor re-fetches that page -- and finds its own candidates already there, so it
skips those messages instead of paying to distill them a second time.

------------------------------------------------------------------------------
WHAT A REFUSED DAILY CAP MEANS HERE
------------------------------------------------------------------------------

The opposite of what it means for live extraction, and this is the one place the
two paths deliberately behave differently. A live batch refused by
EXTRACTION_DAILY_CAP is DROPPED, because holding it would accumulate raw message
text until midnight and then release a flood. A backfill batch refused by
BACKFILL_DAILY_CAP is NOT dropped and NOT held: the cursor simply does not move,
the tick ends, and tomorrow's tick re-fetches the same page from Discord. Losing
history a moderator explicitly asked to be read is not a thing a spend limit is
allowed to do, and backfill has nothing to hold anyway -- its input is not going
anywhere. A run over a large channel therefore takes several days, on purpose,
which is what /aura-backfill status and pause exist to make legible.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime

import aiosqlite
import discord
from fastembed import TextEmbedding

from aura.backfill.gateway import BackfillGateway
from aura.backfill.history import ChannelUnreadable, fetch_history_page
from aura.config import ModelComponent, Settings
from aura.db.backfill_runs import (
    BackfillRun,
    BackfillState,
    advance_cursor,
    get_running_runs,
    set_run_state,
)
from aura.db.backfill_state import (
    count_backfill_calls_on,
    try_acquire_backfill_call_slot,
)
from aura.db.connection import utc_day, utc_now
from aura.db.extraction_queue import QueuedMessage, queued_message_ids
from aura.db.pending_facts import staged_message_ids
from aura.discord_context import channel_display_name
from aura.extraction.distiller import distill_facts
from aura.extraction.pipeline import should_extract, stage_distilled_candidates
from aura.proactive.question_detector import QuestionDetector

logger = logging.getLogger(__name__)

# How many messages one history request asks for. Discord's own maximum, and
# there is no reason to ask for less: a smaller page would mean more requests
# for the same history, which is the opposite of what the pause between pages is
# for.
_PAGE_SIZE = 100

# How many pages one batch may scan before it is flushed regardless of how few
# candidates it found.
#
# This bound is what keeps the restart guarantee TIGHT rather than merely true.
# Without it, a channel with five thousand ordinary messages and three
# fact-worthy ones would scan all five thousand before its batch filled and its
# cursor first moved -- so a restart at message 4,900 would re-read all 4,900.
# With it, the cursor advances at least every ten pages, so a crash costs at
# most a thousand messages of re-reading and never costs a candidate. Ten is
# chosen against what a page actually costs (one request, no LLM call, no
# write): the ceiling is cheap enough to sit well inside a single tick and small
# enough that progress is visible in /aura-backfill status while a long, quiet
# stretch of history is being scanned.
_MAX_PAGES_PER_BATCH = 10


async def advance_due_backfills(
    db: aiosqlite.Connection,
    model: TextEmbedding,
    gateway: BackfillGateway,
    detector: QuestionDetector,
    *,
    settings: Settings,
    now: datetime,
) -> int:
    """Advance every running backfill by one batch. Returns how many actually moved.

    One run at a time, sequentially rather than concurrently, for the reasons
    flush_due_batches and send_due_digests both give: the per-connection lock
    serializes the database work anyway, nothing is waiting on a backfill, and a
    sequential sweep keeps "what did this tick do?" answerable by reading the log
    in order.

    A failure in one run never stops the others. Each is wrapped individually,
    so one channel whose permissions were revoked cannot starve every other run
    on the deployment -- the exact failure shape a single shared try block would
    produce.

    Requires a timezone-aware `now`, injected rather than read here, matching
    every other time-sensitive function in this project: one reading drives the
    daily-cap day key, the cursor timestamp and the run's updated_at for a whole
    tick, so they cannot straddle midnight and disagree.
    """
    if now.tzinfo is None:
        raise ValueError(f"now must be a timezone-aware datetime, got {now!r}")

    advanced = 0
    for run in await get_running_runs(db):
        try:
            if await _advance_one(
                db, model, gateway, detector, run=run, settings=settings, now=now
            ):
                advanced += 1
        except Exception:
            logger.exception(
                "Backfill run %s (channel %s) failed this tick; its cursor is unchanged "
                "and the next tick retries from the same place",
                run.id,
                run.channel_id,
            )
    return advanced


async def _advance_one(
    db: aiosqlite.Connection,
    model: TextEmbedding,
    gateway: BackfillGateway,
    detector: QuestionDetector,
    *,
    run: BackfillRun,
    settings: Settings,
    now: datetime,
) -> bool:
    """Process one batch for one run. Returns whether the cursor moved.

    The ordering here is the load-bearing part of the whole sub-phase, and every
    step fails in a direction that was chosen rather than inherited:

      0. Is there a model configured at all? Without one nothing downstream can
         run, so scanning history would burn requests toward a call that cannot
         happen.
      1. Is there budget left today? A cheap pre-check, so a run whose cap is
         spent costs one indexed read per tick instead of ten history requests.
         It is an optimization and never the decision -- step 4's atomic claim
         is.
      2. Resolve the channel. A permanent failure ends the run; anything else
         propagates to the caller's per-run handler and is retried next tick.
      3. Scan pages until the batch is full, the page budget is spent, or the
         history is exhausted. Read-only with respect to Aura's own state, so it
         is safe to have done this and then abandon it.
      4. CLAIM the spend slot -- before the call it authorizes, never after,
         matching every ledger in this project.
      5. Distill and stage.
      6. ADVANCE THE CURSOR, last. A crash anywhere above leaves it where it
         was, so the next tick re-does one page. Advancing earlier would make
         the same crash skip messages permanently, which is unrecoverable rather
         than merely repeated.
    """
    if not settings.is_llm_configured(ModelComponent.EXTRACTION):
        # Not an error and not worth a log line per tick: a deployment with no
        # extraction model configured simply has no pipeline for a backfill to
        # feed. The run stays exactly where it is.
        return False

    spent = await count_backfill_calls_on(db, guild_id=run.guild_id, day=utc_day(now))
    if spent >= settings.backfill_daily_cap:
        return False

    try:
        channel = await gateway.resolve_channel(run.channel_id)
    except ChannelUnreadable as exc:
        logger.warning(
            "Ending backfill run %s: %s. The cursor is kept as a record; a moderator "
            "can start a new run once the channel is readable again.",
            run.id,
            exc,
        )
        await _fail(db, run=run, now=now)
        return False

    if channel.guild is None or channel.guild.id != run.guild_id:
        # NOT failure handling, and the same check aura.digest.scheduler makes
        # before posting: a run reads MESSAGE CONTENT, so a run whose stored
        # channel id now resolves into a different server would read that
        # server's conversations and stage candidates under this guild's id --
        # a cross-guild data leak, not a mis-render. Reachable through a deleted
        # channel whose id Discord later reuses, and through a hand-edited row,
        # which is exactly why it must not depend on the slash command's own
        # guild scoping.
        logger.error(
            "Ending backfill run %s: channel %s belongs to guild %s, not to guild %s. "
            "Refusing to read another server's history.",
            run.id,
            run.channel_id,
            getattr(channel.guild, "id", None),
            run.guild_id,
        )
        await _fail(db, run=run, now=now)
        return False

    scan = await _scan_for_batch(db, channel, detector, run=run, settings=settings)
    if scan is None:
        # Every fetch attempt failed transiently. The cursor has not moved, so
        # the next tick starts from exactly the same place.
        return False

    if not scan.scanned_any:
        # The first page came back empty, so the history is exhausted: the cursor
        # has reached until_message_id with nothing between them. (The only other
        # way to scan nothing is a transient failure, and that returned None
        # above.) Completing here rather than on the next tick means a finished
        # run stops costing a request per tick forever.
        await _complete(db, run=run, now=now)
        return False

    assert scan.cursor_message_id is not None  # guaranteed by scanned_any
    assert scan.cursor_message_at is not None  # set together with the id above

    staged = 0
    calls_spent = 0
    if scan.candidates:
        outcome = await _distill_and_stage(
            db, model, channel=channel, run=run, candidates=scan.candidates,
            settings=settings, now=now,
        )
        if outcome is None:
            # The daily cap refused the call. The cursor stays put, so this exact
            # page is re-fetched and re-offered tomorrow -- nothing is dropped.
            return False
        staged, calls_spent = outcome

    moved = await advance_cursor(
        db,
        run_id=run.id,
        cursor_message_id=scan.cursor_message_id,
        cursor_message_at=scan.cursor_message_at,
        messages_scanned=scan.messages_scanned,
        candidates_staged=staged,
        calls_spent=calls_spent,
        now=now,
    )
    if not moved:
        # A moderator paused or cancelled this run while the paid call above was
        # in flight. Their decision stands: the cursor is NOT advanced, so if the
        # run is ever resumed this page is processed again -- which re-stages the
        # same candidates idempotently rather than producing new ones. Anything
        # already staged stays staged and reviewable; discarding it would be a
        # write that loses work a moderator asked for before they changed their
        # mind about the rest.
        logger.info(
            "Backfill run %s was paused or cancelled while a batch was in flight; "
            "its cursor was not advanced",
            run.id,
        )
        return False

    logger.info(
        "Backfill run %s (channel %s): scanned %d message(s) up to %s, staged %d "
        "candidate(s) from %d distillation call(s)",
        run.id,
        run.channel_id,
        scan.messages_scanned,
        scan.cursor_message_at.isoformat(),
        staged,
        calls_spent,
    )

    if scan.reached_end:
        await _complete(db, run=run, now=now)
    return True


class _ScanResult:
    """One tick's worth of scanned history: what to distill, and how far it got.

    A small mutable holder rather than a pydantic model, deliberately: it never
    crosses a process boundary, never gets validated, and carries live
    discord.Message-derived values that pydantic would have to be taught about
    for no benefit.

    THE INVARIANT THAT MAKES THE CURSOR SAFE, and the reason this is a class
    rather than a tuple: `cursor_message_id` is always the id of a message this
    scan is DONE with -- scanned, filtered, and either discarded or included in
    `candidates`. Never the id of a message whose candidate was cut for not
    fitting in the batch, because the cursor is committed while that candidate
    is not, and the two must not be able to disagree.
    """

    __slots__ = (
        "candidates",
        "cursor_message_at",
        "cursor_message_id",
        "messages_scanned",
        "reached_end",
    )

    def __init__(self) -> None:
        self.candidates: list[QueuedMessage] = []
        self.cursor_message_id: int | None = None
        self.cursor_message_at: datetime | None = None
        self.messages_scanned: int = 0
        self.reached_end: bool = False

    @property
    def scanned_any(self) -> bool:
        """Whether this scan saw any message at all, and so has a cursor to commit."""
        return self.cursor_message_id is not None

    def cover(self, message: discord.Message) -> None:
        """Record that this scan is finished with `message`, moving the cursor to it."""
        self.cursor_message_id = message.id
        self.cursor_message_at = message.created_at


async def _scan_for_batch(
    db: aiosqlite.Connection,
    channel: discord.TextChannel,
    detector: QuestionDetector,
    *,
    run: BackfillRun,
    settings: Settings,
) -> _ScanResult | None:
    """Read pages until a batch is full, the page budget is spent, or history ends.

    Returns None only if the very first page could not be fetched at all -- a
    partial scan is kept, because the pages that DID arrive are real work and
    throwing them away would mean re-fetching them next tick for nothing.

    The cursor this returns is the last message SCANNED, not the last message
    that survived the filters. That distinction is the difference between a
    backfill that terminates and one that does not: advancing only past
    fact-worthy messages would re-read every rejected message on every
    subsequent tick, forever.

    THE OVERFLOW CASE IS WHERE THIS COULD LOSE HISTORY, so it is handled
    explicitly rather than by truncation. When a page pushes the batch past
    EXTRACTION_BATCH_MAX_MESSAGES, the surplus candidates are dropped from this
    batch AND the cursor is pulled back to the last candidate that fits -- so
    everything after it, fact-worthy or not, is simply re-read on the next tick.
    Truncating the batch while leaving the cursor at the page's end would commit
    to having processed messages whose candidates were thrown away, which is the
    one way this design could silently skip a fact.
    """
    result = _ScanResult()
    after_id = run.resume_after_id
    limit = settings.extraction_batch_max_messages

    for page_number in range(1, _MAX_PAGES_PER_BATCH + 1):
        if settings.backfill_page_pause_seconds > 0 and page_number > 1:
            # Between pages, never before the first: a run that only asks for one
            # page a tick already waits the whole check interval.
            await asyncio.sleep(settings.backfill_page_pause_seconds)

        page = await fetch_history_page(
            channel,
            after_message_id=after_id,
            before_message_id=run.until_message_id,
            limit=_PAGE_SIZE,
        )
        if page is None:
            # Transient failure. Keep whatever earlier pages produced; report
            # None only if this was the first page and there is nothing to keep.
            return result if result.scanned_any else None

        if not page:
            result.reached_end = True
            break

        selected = await _select_candidates(
            db, page, detector, channel_id=channel.id, settings=settings
        )
        room = limit - len(result.candidates)

        if len(selected) <= room:
            result.candidates.extend(selected)
            result.messages_scanned += len(page)
            result.cover(page[-1])
            after_id = page[-1].id
            if len(result.candidates) >= limit:
                break
            continue

        # This page overflows the batch. Keep what fits, stop the cursor at the
        # last message that fits, and count only the messages up to it.
        fitting = selected[:room]
        result.candidates.extend(fitting)
        last_kept_id = fitting[-1].message_id
        covered = [message for message in page if message.id <= last_kept_id]
        result.messages_scanned += len(covered)
        result.cover(covered[-1])
        break

    return result


async def _select_candidates(
    db: aiosqlite.Connection,
    page: list[discord.Message],
    detector: QuestionDetector,
    *,
    channel_id: int,
    settings: Settings,
) -> list[QueuedMessage]:
    """Run one page through the free gates and the live-path boundary checks.

    The gates are the live path's, unchanged and in the same order:
    should_extract first (pure, free), then the fact-worthiness score against
    EXTRACTION_FACT_WORTHINESS_THRESHOLD. There is deliberately no channel-gate
    check here -- /aura-backfill start already refuses a channel extraction is
    not enabled for, and re-reading that switch per page would let a mid-run
    toggle silently truncate a run a moderator is watching.

    The two boundary queries run ONCE for the whole page rather than per message,
    matching how the live path reads active facts once per batch: this is the
    same "batch operations wherever more than one item is processed at once"
    rule CLAUDE.md's Performance section states.
    """
    eligible = [message for message in page if should_extract(message)]
    if not eligible:
        return []

    message_ids = [message.id for message in eligible]
    already_queued = await queued_message_ids(
        db, channel_id=channel_id, message_ids=message_ids
    )
    already_staged = await staged_message_ids(
        db, channel_id=channel_id, message_ids=message_ids
    )
    owned_by_live_path = already_queued | already_staged
    if owned_by_live_path:
        logger.info(
            "Skipping %d message(s) in channel %s that the live extraction path "
            "already owns (%d queued, %d already staged)",
            len(owned_by_live_path),
            channel_id,
            len(already_queued),
            len(already_staged),
        )

    selected: list[QueuedMessage] = []
    for message in eligible:
        if message.id in owned_by_live_path:
            continue
        assert message.guild is not None  # guaranteed by should_extract
        score = await detector.question_likeness(message.content)
        if score < settings.extraction_fact_worthiness_threshold:
            continue
        selected.append(
            QueuedMessage(
                channel_id=message.channel.id,
                message_id=message.id,
                guild_id=message.guild.id,
                channel_name=channel_display_name(message.channel, message.channel.id),
                content=message.content,
                message_created_at=message.created_at,
                # There is no enqueue for a backfilled message -- it never
                # touches extraction_queue -- so this is the moment it was
                # selected. Nothing downstream reads it; it is carried because
                # QueuedMessage is the shape distill_facts and
                # stage_distilled_candidates already speak.
                enqueued_at=utc_now(),
            )
        )
    return selected


async def _distill_and_stage(
    db: aiosqlite.Connection,
    model: TextEmbedding,
    *,
    channel: discord.TextChannel,
    run: BackfillRun,
    candidates: list[QueuedMessage],
    settings: Settings,
    now: datetime,
) -> tuple[int, int] | None:
    """Claim a slot, distill one batch and stage what it produced.

    Returns (candidates staged, calls spent), or None if the daily cap refused
    the call -- which the caller must distinguish from "the call ran and found
    nothing", because only the refusal must leave the cursor where it is.

    The batch is re-checked for strict chronological order immediately before it
    is sent, and re-sorted if it somehow is not. Belt to the braces
    aura.backfill.history.ordered_page already provides: the candidates were
    accumulated across several pages, and the one thing that must be true of the
    list handed to a model that judges what came later is that it is in the order
    things happened.
    """
    ordered = sorted(candidates, key=lambda queued: (queued.message_created_at, queued.message_id))
    if [queued.message_id for queued in ordered] != [
        queued.message_id for queued in candidates
    ]:
        logger.warning(
            "A backfill batch for run %s was assembled out of chronological order; "
            "re-sorting before distillation",
            run.id,
        )

    attempt = await try_acquire_backfill_call_slot(
        db,
        guild_id=run.guild_id,
        run_id=run.id,
        message_count=len(ordered),
        daily_cap=settings.backfill_daily_cap,
        now=now,
    )
    if not attempt.granted:
        logger.info(
            "Backfill run %s is out of budget for today (%d of %d call(s) spent); "
            "it resumes at the next UTC day with its cursor unchanged",
            run.id,
            attempt.daily_count,
            attempt.daily_cap,
        )
        return None

    extraction_model = settings.resolve_model(ModelComponent.EXTRACTION)
    if extraction_model is None:
        # Configuration changed out from under an in-flight batch. The slot is
        # already spent (never refunded, see the ledger). The cursor still
        # advances: re-reading this page tomorrow would spend another slot
        # against the same missing model.
        logger.warning(
            "No extraction model configured; backfill run %s advances past a "
            "%d-message batch without distilling it",
            run.id,
            len(ordered),
        )
        return 0, 1

    distilled = await distill_facts(
        ordered,
        channel_name=channel_display_name(channel, channel.id),
        model=extraction_model,
    )
    if distilled is None:
        # The call failed or its result could not be trusted -- distinct from the
        # model judging the batch empty. The slot stays spent (that is what
        # bounds a reliably-failing model) and the cursor still advances, so a
        # batch the model cannot handle cannot loop forever spending a slot per
        # tick. The messages behind it are reachable through the manual "Add as
        # Aura Fact" context menu, exactly as on the live path.
        logger.warning(
            "Distillation produced no usable result for a %d-message backfill batch "
            "in run %s; moving past it",
            len(ordered),
            run.id,
        )
        return 0, 1

    if not distilled:
        return 0, 1

    staged = await stage_distilled_candidates(
        db,
        model,
        guild_id=run.guild_id,
        batch=ordered,
        distilled=distilled,
        settings=settings,
        now=now,
    )
    return staged, 1


async def _fail(db: aiosqlite.Connection, *, run: BackfillRun, now: datetime) -> None:
    """End a run because its channel cannot (or must not) be read.

    Guarded on the run still being RUNNING like every other transition, so a
    moderator who cancelled it in the same instant keeps their own verdict --
    "ended by a moderator" and "the channel broke" are different things for
    someone reading /aura-backfill status afterwards.
    """
    await set_run_state(
        db,
        run_id=run.id,
        state=BackfillState.FAILED,
        now=now,
        from_states=(BackfillState.RUNNING,),
    )


async def _complete(
    db: aiosqlite.Connection, *, run: BackfillRun, now: datetime
) -> None:
    """Mark a run finished, if it is still the running run it was a moment ago."""
    if await set_run_state(
        db,
        run_id=run.id,
        state=BackfillState.COMPLETED,
        now=now,
        from_states=(BackfillState.RUNNING,),
    ):
        logger.info(
            "Backfill run %s (channel %s) reached the end of its history",
            run.id,
            run.channel_id,
        )


async def run_backfill_worker(
    db: aiosqlite.Connection,
    model: TextEmbedding,
    gateway: BackfillGateway,
    detector: QuestionDetector,
    *,
    settings: Settings,
) -> None:
    """Advance whatever backfills are running, forever. Runs for the process's life.

    The project's third background task, and it follows the shape the extraction
    sweeper set and the digest scheduler confirmed: one `while True` per process,
    created in setup_hook, cancelled in close(), never dying of an exception it
    can survive. A worker that exits silently leaves a bot that looks healthy
    while a moderator's backfill simply never progresses -- the exact failure
    shape CLAUDE.md's non-negotiable principle rules out. CancelledError is a
    BaseException and still propagates, so shutdown works.

    The sleep is deliberately asymmetric. A tick that advanced something sleeps
    only the page pause, so an active run keeps moving batch after batch instead
    of one batch per check interval; a tick that found nothing to do sleeps the
    full interval, so an idle deployment costs one indexed read every half
    minute. That is what makes a several-thousand-message channel finish in
    minutes rather than in as many check intervals as it has batches.
    """
    idle_interval = settings.backfill_check_interval_seconds
    active_interval = settings.backfill_page_pause_seconds
    logger.info(
        "Backfill worker started: cap %d call(s)/guild/UTC-day, %.1fs between history "
        "pages, checking every %.0fs when idle (runs only where a moderator started "
        "one with /aura-backfill)",
        settings.backfill_daily_cap,
        settings.backfill_page_pause_seconds,
        idle_interval,
    )
    while True:
        advanced = 0
        try:
            advanced = await advance_due_backfills(
                db, model, gateway, detector, settings=settings, now=utc_now()
            )
        except Exception:
            logger.exception("Backfill sweep failed; continuing")
        await asyncio.sleep(active_interval if advanced else idle_interval)


__all__ = ["advance_due_backfills", "run_backfill_worker"]
