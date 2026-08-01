"""The automatic extraction path, end to end: message in, staged candidate out.

Five stages, cheapest first, each only reached if the previous one passed:

  0. should_extract -- is this human guild text at all? (pure, free)
  1. LLM configured for extraction (free, no I/O)
  2. channel opted in by a moderator (one indexed DB read)
  3. fact-worthiness above threshold (local CPU, free -- see
     aura.extraction.fact_worthiness)
  4. enqueue into the durable batch (one row)

and then, minutes later and entirely separately, the sweeper closes the batch:

  5. claim a slot from the daily cap (atomic, durable, never refunded)
  6. one distillation call for the whole batch (paid)
  7. dedup hint against existing active facts (local, embedding-based)
  8. stage each candidate for review
  9. for a staged candidate that the dedup check flagged -- and only for those
     -- one supersession-judgment call against its own daily cap (paid; see
     aura.extraction.supersession), then clear the batch

**This path is independent of Trigger 2, deliberately and at every level.** It
has its own channel gate (extraction_channel_config, a separate table from
proactive_channel_config -- reports/phase-3-pre-analysis.md Section 1c found a
real collision risk in sharing one), its own threshold, its own spend ledger,
its own model, and its own entry point called from AuraClient.on_message beside
the proactive one rather than from inside it. Both paths see the same raw
message and neither can observe or affect what the other decided: there is no
shared state between them but the database connection, and no ordering
dependency in either direction. A message can be enqueued for extraction in a
channel where proactive relief is off, and vice versa, which is precisely the
point of keeping the two switches apart.

**Nothing here writes a fact, and nothing here supersedes one.** Stage 8 writes a
CANDIDATE, and stage 9 writes a JUDGMENT ABOUT a candidate. A candidate becomes a
real, citable fact only when a moderator confirms it (see aura.commands.pending),
and an existing fact is retired only when a moderator runs /aura-supersede --
including when stage 9's judgment says "supersession", which is a proposal
rendered in an embed and never a write. This is the first time in Aura's life
that an automatic path proposes knowledge with no human having pointed at the
message and said "that one", which is exactly why both of those gates stay
human.
"""
from __future__ import annotations

import asyncio
import logging
import unicodedata
from datetime import datetime

import aiosqlite
import discord
import numpy as np
from fastembed import TextEmbedding

from aura.config import ModelComponent, Settings
from aura.db.connection import utc_now
from aura.db.extraction_channel_config import is_extraction_enabled
from aura.db.extraction_queue import (
    QueuedMessage,
    clear_batch,
    due_channels,
    enqueue_message,
    read_batch,
    remove_queued_message,
)
from aura.db.extraction_state import try_acquire_extraction_call_slot
from aura.db.fact_variants import FactVariant, get_active_fact_variants
from aura.db.models import Fact
from aura.db.pending_facts import (
    PendingFact,
    record_relationship_judgement,
    stage_pending_fact,
)
from aura.db.repository import get_active_facts
from aura.db.supersession_state import try_acquire_supersession_call_slot
from aura.embeddings import EMBEDDING_DTYPE, best_similarity, embed_texts, group_variants_by_fact
from aura.extraction.distiller import DistilledFact, distill_facts
from aura.extraction.supersession import judge_relationship
from aura.proactive.question_detector import QuestionDetector

logger = logging.getLogger(__name__)

# The same set aura.proactive.listener excludes, arrived at independently and
# kept independently. See should_extract for why this is a deliberate duplicate
# rather than a shared constant.
_EXTRACTABLE_MESSAGE_TYPES = frozenset(
    {discord.MessageType.default, discord.MessageType.reply}
)

# How often the sweeper wakes to look for batches whose window has closed.
# Bounded below so a tiny configured window (tests use zero) does not spin, and
# above so the effective delay past a window's end stays small relative to the
# shipped five-minute window. Half the window, clamped, means a batch is
# distilled within roughly 50% of its window past closing in the worst case,
# which is well inside "nobody is waiting on this".
_MIN_SWEEP_INTERVAL_SECONDS = 1.0
_MAX_SWEEP_INTERVAL_SECONDS = 30.0

# Unicode general categories that render as nothing. Cf is the one that matters
# in practice (zero-width space/joiner, bidi marks, BOM); Cc and the three
# separator categories are included because they are equally invisible and
# equally cheap to exclude. See _has_visible_content.
_INVISIBLE_CATEGORIES = frozenset({"Cf", "Cc", "Zs", "Zl", "Zp"})


def sweep_interval_seconds(window_seconds: float) -> float:
    """The sweeper's wake interval for a given batch window.

    A function rather than a constant so the relationship between the two is
    stated once and testable, instead of a magic number that quietly stops
    making sense if someone configures a ten-second window.
    """
    return min(_MAX_SWEEP_INTERVAL_SECONDS, max(_MIN_SWEEP_INTERVAL_SECONDS, window_seconds / 2))


def should_extract(message: discord.Message) -> bool:
    """Whether message is human-written guild text worth considering for extraction.

    Pure and side-effect free, and a DELIBERATE DUPLICATE of
    aura.proactive.listener.should_classify rather than a call to it. The two
    predicates happen to agree today, and sharing one function would be the
    obvious tidy-up -- but it would silently couple the two triggers' scope to
    each other, which is exactly the class of accidental coupling
    reports/phase-3-pre-analysis.md Section 1c warned about for the channel
    gate. The rules also have genuinely different justifications: proactive
    relief excludes bot messages because a bot is not asking a question, while
    extraction excludes them because a bot's announcement is not a human
    stating something about the server. Those reasons can diverge -- a future
    phase might well want extraction to read an announcement webhook Trigger 2
    must always ignore -- and a shared predicate would make that a change to
    both.
    """
    if message.guild is None:
        # A DM. There is no server for a fact to be about, and no guild to
        # attribute a candidate to.
        return False

    if message.author.bot or message.webhook_id is not None:
        # Other bots, webhook integrations, and Aura itself. Extracting from
        # Aura's own proactive answers would let it launder its own synthesis
        # back into the knowledge model as a new fact -- a feedback loop that
        # would eventually cite itself.
        return False

    if message.interaction_metadata is not None:
        # A response to a slash command or context menu, i.e. a bot reply.
        return False

    if message.type not in _EXTRACTABLE_MESSAGE_TYPES:
        return False

    return _has_visible_content(message.content)


def _has_visible_content(content: str) -> bool:
    """Whether content contains at least one character a human could actually read.

    str.strip() is not enough, and this adversarial pass is where that showed
    up: it removes whitespace as Python defines it, which does NOT include the
    Unicode format characters (category Cf) a Discord message can be composed
    entirely of -- zero-width space, zero-width joiner, the bidi marks, the
    byte-order mark. `"\\u200b\\u200b".strip()` is still truthy, so a message of
    nothing but invisible characters would otherwise reach a paid call.

    Measured before fixing rather than after: the shipped fact-worthiness
    threshold already scores such messages around -0.49, far below the -0.02
    bar, so nothing was actually getting through today. The check is here
    anyway because "a message with no visible characters cannot contain a fact"
    is a structural property and should not depend on where a calibrated,
    explicitly-placeholder threshold happens to sit this month.

    Note for whoever reads this next: aura.proactive.listener.should_classify
    has the same str.strip() gap. It is left alone here on purpose -- Trigger
    2's intake is not this sub-phase's to change, and the same measurement
    applies to its own Stage 1 threshold -- but it is a real, if currently
    harmless, shared edge case and is written up in reports/phase-3a-2.txt.
    """
    return any(
        not character.isspace() and unicodedata.category(character) not in _INVISIBLE_CATEGORIES
        for character in content
    )


def _channel_name(message: discord.Message) -> str:
    """Best-effort human-readable channel name for the distillation prompt.

    Falls back to the channel ID rather than raising or sending an empty
    string: the name is context for the model, not an identifier anything
    depends on, and a channel type without one (or a partial channel object)
    must not be the reason a whole batch is lost.
    """
    name = getattr(message.channel, "name", None)
    return str(name) if name else str(message.channel.id)


async def handle_extraction_message(
    message: discord.Message,
    *,
    db: aiosqlite.Connection,
    detector: QuestionDetector,
    settings: Settings,
) -> None:
    """Run one message through the free gates and enqueue it if it survives them.

    Catches every exception on purpose, and around the filtering as well as the
    scoring and the write. This runs on every message in every channel Aura can
    see, so one malformed message, one embedding failure or one busy database
    must degrade into a log line -- never into an exception travelling back up
    through the gateway's event dispatch, and never into a failure that also
    takes the proactive path down, which shares the same on_message call.

    The failure direction is deliberate: an exception anywhere here means the
    message is not enqueued, so a broken gate can never cause an extraction,
    only a missed one.

    asyncio.CancelledError inherits from BaseException, not Exception, so a
    shutdown cancelling this task still propagates as it should.
    """
    try:
        if not should_extract(message):
            return

        assert message.guild is not None  # guaranteed by should_extract

        # Free, no I/O, and first: with no model configured for extraction
        # there is nothing that could ever drain the queue, so enqueueing would
        # accumulate raw message text forever for a pipeline that cannot run.
        if not settings.is_llm_configured(ModelComponent.EXTRACTION):
            return

        # The cheapest gate that costs a DB read. A channel no moderator has
        # opted in stops here, before any embedding inference -- and this is a
        # DIFFERENT switch from the one proactive relief reads.
        if not await is_extraction_enabled(db, channel_id=message.channel.id):
            return

        score = await detector.question_likeness(message.content)
        if score < settings.extraction_fact_worthiness_threshold:
            return

        await enqueue_message(
            db,
            guild_id=message.guild.id,
            channel_id=message.channel.id,
            message_id=message.id,
            channel_name=_channel_name(message),
            content=message.content,
            message_created_at=message.created_at,
            now=utc_now(),
        )
    except Exception:
        logger.exception("Extraction intake failed for message %s", _log_reference(message))


async def withdraw_message(
    db: aiosqlite.Connection, *, channel_id: int, message_id: int
) -> None:
    """Remove a message from its pending batch after an edit or a deletion.

    The scope of the edit/delete handling this sub-phase builds, and its limit
    is deliberate: a message withdrawn before its batch closes is never
    distilled and never costs anything, which covers the common case (a typo
    corrected, a message thought better of within minutes). Retracting an
    already-staged candidate whose source message is edited days later is
    explicitly out of scope here -- it needs a whole post-processing chain, and
    the review step in front of every candidate already means no unreviewed
    sentence can become a citable fact in the meantime.

    An edit is treated exactly like a deletion rather than re-scored: the
    edited text may say something entirely different from what cleared the
    filter, and re-running the gate on an edit is its own decision with its own
    idempotency questions. Standing down is the conservative direction and
    costs at most one automatically extracted fact that a moderator can still
    add by hand.

    Never raises, for the same reason handle_extraction_message doesn't: it is
    called from a gateway event handler that must survive anything.
    """
    try:
        removed = await remove_queued_message(db, channel_id=channel_id, message_id=message_id)
        if removed:
            logger.info(
                "Withdrew message %s/%s from its pending extraction batch "
                "(edited or deleted before the batch closed)",
                channel_id,
                message_id,
            )
    except Exception:
        logger.exception(
            "Failed to withdraw message %s/%s from the extraction queue", channel_id, message_id
        )


async def flush_due_batches(
    db: aiosqlite.Connection, model: TextEmbedding, *, settings: Settings, now: datetime
) -> int:
    """Distill and stage every batch whose window has closed. Returns how many ran.

    One channel at a time, sequentially rather than concurrently: the
    per-connection lock serializes the database work anyway, the distillation
    calls are not on anyone's critical path, and running them in sequence keeps
    the daily cap's accounting easy to reason about (a burst of concurrent
    claims is safe, but "how many did that sweep spend?" stops being answerable
    by reading the code).

    A failure in one channel's batch never stops the others: each is wrapped
    individually, so one channel with a permanently unparseable batch cannot
    starve every other channel in every guild.

    Requires a timezone-aware `now`, injected rather than read from the clock
    here, matching every other time-sensitive function in this project
    (try_acquire_escalation_slot, evaluate_message, due_channels). One reading
    drives both the window cutoff and the daily-cap day key for the whole
    sweep, so they cannot straddle midnight and disagree -- and "this batch is
    not due yet" becomes testable at the exact moment it matters instead of
    depending on the wall clock at test time.
    """
    channels = await due_channels(
        db, window_seconds=settings.extraction_batch_window_seconds, now=now
    )

    flushed = 0
    for channel_id in channels:
        try:
            if await _flush_channel(
                db, model, channel_id=channel_id, settings=settings, now=now
            ):
                flushed += 1
        except Exception:
            logger.exception("Extraction batch failed for channel %s", channel_id)
    return flushed


async def _flush_channel(
    db: aiosqlite.Connection,
    model: TextEmbedding,
    *,
    channel_id: int,
    settings: Settings,
    now: datetime,
) -> bool:
    """Distill one channel's due batch and stage what it produced. Returns whether it ran.

    The ordering here is the load-bearing part, and it mirrors the proactive
    gate's: claim the spend slot BEFORE the call it authorizes, and clear the
    queue only AFTER the candidates are staged. Each choice fails in the safe
    direction.

    * A crash between claiming and clearing leaves the batch queued, so the
      next sweep re-does it. That costs a second slot and re-stages the same
      candidates, which pending_facts' UNIQUE constraint absorbs -- an
      idempotent repeat rather than duplicates a moderator has to reject twice.
    * Clearing first would make the same crash lose the batch outright, which
      is unrecoverable rather than merely repeated.

    A batch refused by the daily cap is DROPPED, not held. Holding it would
    accumulate raw message text for the rest of the UTC day and then release a
    flood of calls at midnight, which is not what "no more spend today" should
    mean; and the batch window's promise -- a bounded wait, then a decision --
    would quietly become "an unbounded wait". Extraction is best-effort by
    design, and the manual "Add as Aura Fact" context menu is unaffected.
    """
    batch = await read_batch(db, channel_id=channel_id, limit=settings.extraction_batch_max_messages)
    if not batch:
        # Raced with a withdrawal that emptied the channel between due_channels
        # and here. Nothing to do, nothing spent.
        return False

    guild_id = batch[0].guild_id
    message_ids = [message.message_id for message in batch]

    attempt = await try_acquire_extraction_call_slot(
        db,
        guild_id=guild_id,
        channel_id=channel_id,
        message_count=len(batch),
        daily_cap=settings.extraction_daily_cap,
        now=now,
    )
    if not attempt.granted:
        logger.warning(
            "Dropping a %d-message extraction batch in channel %s: guild %s has spent "
            "%d of %d distillation calls today",
            len(batch),
            channel_id,
            guild_id,
            attempt.daily_count,
            attempt.daily_cap,
        )
        await clear_batch(db, channel_id=channel_id, message_ids=message_ids)
        return False

    extraction_model = settings.resolve_model(ModelComponent.EXTRACTION)
    if extraction_model is None:
        # Configuration changed out from under a queued batch. The slot is
        # already spent (never refunded, see the ledger), and the batch is
        # dropped rather than left to accumulate against a pipeline that can no
        # longer run.
        logger.warning(
            "No extraction model configured; dropping a %d-message batch in channel %s",
            len(batch),
            channel_id,
        )
        await clear_batch(db, channel_id=channel_id, message_ids=message_ids)
        return False

    distilled = await distill_facts(
        batch, channel_name=batch[0].channel_name, model=extraction_model
    )
    if distilled is None:
        # The call failed or its result could not be trusted -- distinct from
        # the model judging the batch empty. The slot stays spent (that is what
        # bounds a reliably-failing model), and the batch is cleared rather
        # than retried, so a batch the model cannot handle cannot loop forever
        # spending a slot per sweep.
        logger.warning(
            "Distillation produced no usable result for a %d-message batch in channel %s; "
            "clearing it",
            len(batch),
            channel_id,
        )
        await clear_batch(db, channel_id=channel_id, message_ids=message_ids)
        return True

    if distilled:
        await _stage_distilled(
            db,
            model,
            guild_id=guild_id,
            batch=batch,
            distilled=distilled,
            settings=settings,
            now=now,
        )

    await clear_batch(db, channel_id=channel_id, message_ids=message_ids)
    logger.info(
        "Distilled %d message(s) in channel %s into %d candidate(s) "
        "(%d of %d of today's calls spent)",
        len(batch),
        channel_id,
        len(distilled),
        attempt.daily_count,
        attempt.daily_cap,
    )
    return True


async def _stage_distilled(
    db: aiosqlite.Connection,
    model: TextEmbedding,
    *,
    guild_id: int,
    batch: list[QueuedMessage],
    distilled: list[DistilledFact],
    settings: Settings,
    now: datetime,
) -> None:
    """Embed, dedup-check, stage and (where flagged) judge every candidate from one batch.

    Embeds all candidates in ONE batched inference call rather than one per
    candidate, and reads the guild's active facts ONCE rather than per
    candidate, per CLAUDE.md's Performance section: the naive shape here is N
    calls to find_similar_facts, which is N embeddings and N full table scans
    for a set of sentences that arrived together.

    The dedup check is advisory and says so in what it writes: it records the
    single best-matching active fact and its score when that score clears
    EXTRACTION_DEDUP_SIMILARITY_THRESHOLD, so the moderator reviewing the
    candidate can see "this may replace fact #12" and choose. It never
    supersedes anything and never blocks staging -- a candidate that looks like
    a duplicate is still staged, because deciding that two sentences are the
    same claim is exactly the judgement CLAUDE.md keeps with the human.

    Phase 3a-3 adds one paid call on top of that hint, and its placement is the
    whole of its cost control: it runs INSIDE the `above_threshold` branch and
    only for a candidate this call actually staged. Two consequences worth
    stating rather than leaving to be read out of the control flow:

      * An unflagged candidate -- the overwhelming majority -- never reaches a
        judgment call at all, which is what keeps this a small slice of
        extraction's volume rather than a second per-candidate cost.
      * A candidate that was ALREADY staged (stage_pending_fact returns None on
        the crash-retry path, absorbed by the UNIQUE constraint) is not judged
        again. Its judgment, if it got one, is already stored; re-judging it
        would pay twice for the same answer.
    """
    by_message_id = {message.message_id: message for message in batch}
    contents = [candidate.content for candidate in distilled]
    embeddings = await embed_texts(model, contents)
    active_facts = await get_active_facts(db, guild_id)
    # A freshly distilled candidate has no variants of its own yet -- those
    # only get generated once it is confirmed into a real, active fact (see
    # aura.facts_service) -- so this dedup check compares the candidate's
    # single embedding against each EXISTING active fact's canonical sentence
    # and its variants, never the reverse. Fetched once for the whole batch,
    # like active_facts above, not once per candidate.
    active_variants_by_fact = group_variants_by_fact(
        await get_active_fact_variants(db, guild_id)
    )

    for candidate, embedding in zip(distilled, embeddings, strict=True):
        source = by_message_id.get(candidate.message_id)
        if source is None:
            # Unreachable: _validate_distilled already mapped every message
            # number back through this same batch. Checked rather than asserted
            # because the alternative is an unhandled KeyError that loses the
            # whole batch's other candidates.
            logger.error(
                "Distilled candidate cites message %s, which is not in its own batch; skipping",
                candidate.message_id,
            )
            continue

        similar_fact, similar_score = _best_matching_fact(
            embedding, active_facts, active_variants_by_fact
        )
        above_threshold = (
            similar_fact is not None
            and similar_score >= settings.extraction_dedup_similarity_threshold
        )

        staged = await stage_pending_fact(
            db,
            guild_id=guild_id,
            channel_id=source.channel_id,
            message_id=source.message_id,
            content=candidate.content,
            embedding=embedding.astype(EMBEDDING_DTYPE, copy=False).tobytes(),
            category=candidate.category,
            similar_fact_id=similar_fact.id if above_threshold and similar_fact else None,
            similar_fact_score=similar_score if above_threshold else None,
        )
        if staged is None:
            logger.info(
                "Candidate from message %s/%s was already staged; not staging it twice "
                "and not judging it again",
                source.channel_id,
                source.message_id,
            )
            continue

        if above_threshold and similar_fact is not None:
            await _judge_staged_candidate(
                db,
                candidate=staged,
                predecessor=similar_fact,
                settings=settings,
                now=now,
            )


async def _judge_staged_candidate(
    db: aiosqlite.Connection,
    *,
    candidate: PendingFact,
    predecessor: Fact,
    settings: Settings,
    now: datetime,
) -> None:
    """Ask the model what one dedup hit means, and store the answer beside the candidate.

    Never raises, and never lets its own failure touch anything upstream: the
    candidate is already staged and reviewable before this function is entered,
    so every early return below leaves a moderator with exactly what Phase 3a-2
    gave them -- the plain "this may replace fact #N" hint -- rather than with a
    lost candidate or a half-written row. The failure direction is one-way by
    construction: this can add information to a review, never remove it.

    The three refusals are all ordinary, and each is logged at the level its
    consequence deserves rather than uniformly:

      * no model configured -- routine for a deployment that never set one up;
      * the daily cap is spent -- a bound working as intended, worth a warning
        because it means judgments are silently absent for the rest of the day;
      * the call failed or its answer was unusable -- already logged in full by
        aura.extraction.supersession, so it is not logged twice here.

    The spend slot is claimed BEFORE the call, matching both other ledgers: a
    crash or an API failure spends it rather than earning a free retry.
    """
    model = settings.resolve_model(ModelComponent.SUPERSESSION)
    if model is None or not settings.is_llm_configured(ModelComponent.SUPERSESSION):
        return

    try:
        attempt = await try_acquire_supersession_call_slot(
            db,
            guild_id=candidate.guild_id,
            pending_fact_id=candidate.id,
            daily_cap=settings.supersession_daily_cap,
            now=now,
        )
        if not attempt.granted:
            logger.warning(
                "Not judging candidate %s against fact %s: guild %s has spent %d of %d "
                "supersession judgements today; it stays reviewable with the plain "
                "similarity hint",
                candidate.id,
                predecessor.id,
                candidate.guild_id,
                attempt.daily_count,
                attempt.daily_cap,
            )
            return

        judgement = await judge_relationship(
            predecessor=predecessor.content, candidate=candidate.content, model=model
        )
        if judgement is None:
            return

        applied = await record_relationship_judgement(
            db,
            guild_id=candidate.guild_id,
            pending_id=candidate.id,
            relationship=judgement.relationship,
            reasoning=judgement.reasoning,
        )
        if not applied:
            # A moderator resolved the candidate while the call was in flight.
            # Deliberately not retried and deliberately not forced: a judgment
            # they never saw must not be written onto a decision they already
            # made (see record_relationship_judgement).
            logger.info(
                "Candidate %s was resolved while its judgement was in flight; "
                "the judgement (%s) was discarded rather than backdated",
                candidate.id,
                judgement.relationship.value,
            )
            return

        logger.info(
            "Judged candidate %s against fact %s: %s (%d of %d of today's judgements spent)",
            candidate.id,
            predecessor.id,
            judgement.relationship.value,
            attempt.daily_count,
            attempt.daily_cap,
        )
    except Exception:
        # Never propagates. A batch that produced good candidates must not lose
        # them because an advisory judgement failed in a way its own error
        # handling did not anticipate. CancelledError is a BaseException and
        # still propagates, so shutdown is not swallowed here.
        logger.exception(
            "Supersession judgement failed for candidate %s; it stays reviewable "
            "with the plain similarity hint",
            candidate.id,
        )


def _best_matching_fact(
    embedding: np.ndarray,
    facts: list[Fact],
    variants_by_fact: dict[int, list[FactVariant]],
) -> tuple[Fact | None, float]:
    """Return the active fact most similar to embedding -- via its canonical sentence
    or any of its variants -- and that similarity.

    Returns (None, 0.0) for a guild with no active facts -- the ordinary case
    for a server whose knowledge model is still empty, and the reason this is
    not written as max() over a possibly-empty sequence.

    Skips a fact whose best similarity (across its canonical sentence and
    every variant -- see aura.embeddings.best_similarity) comes back
    non-finite, the same way aura.proactive.gate does: a fact none of whose
    stored vectors can be meaningfully compared to anything must not win or
    lose every comparison silently depending on which way the comparison
    happened to be written.
    """
    best_fact: Fact | None = None
    best_score = 0.0
    for fact in facts:
        score = best_similarity(embedding, fact, variants_by_fact)
        if not np.isfinite(score):
            logger.warning(
                "Fact %s scored non-finite similarity against a new candidate; "
                "excluding it from the dedup check",
                fact.id,
            )
            continue
        if best_fact is None or score > best_score:
            best_fact = fact
            best_score = score
    return best_fact, best_score


async def run_extraction_sweeper(
    db: aiosqlite.Connection, model: TextEmbedding, *, settings: Settings
) -> None:
    """Wake periodically and flush whatever batches are due. Runs for the process's life.

    A single sweeper task for the whole bot rather than a timer per channel,
    which is what makes the batch durable across restarts (see
    aura.db.extraction_queue): there is no per-channel state to lose, only rows,
    and a fresh process's first sweep picks up exactly where the old one
    stopped, including batches that were already overdue while it was down.

    Never dies of a failure it can survive. An exception from one sweep is
    logged and the loop continues, because a sweeper that exits silently leaves
    a bot that looks healthy while every extraction batch accumulates forever --
    the exact failure shape CLAUDE.md's non-negotiable principle rules out.
    CancelledError is a BaseException and still propagates, so shutdown works.
    """
    interval = sweep_interval_seconds(settings.extraction_batch_window_seconds)
    logger.info(
        "Extraction sweeper started: %.0fs window, checking every %.0fs, cap %d call(s)/guild/UTC-day",
        settings.extraction_batch_window_seconds,
        interval,
        settings.extraction_daily_cap,
    )
    while True:
        try:
            await flush_due_batches(db, model, settings=settings, now=utc_now())
        except Exception:
            logger.exception("Extraction sweep failed; continuing")
        await asyncio.sleep(interval)


def _log_reference(message: discord.Message) -> str:
    """Best-effort "channel/message" identifier for a log line, safe on a broken message.

    Called only from a failure path, where raising would defeat the guarantee
    that path exists to provide -- the same reasoning, and the same shape, as
    aura.proactive.listener's own.
    """
    try:
        return f"{message.channel.id}/{message.id}"
    except Exception:
        return "<unidentifiable message>"
