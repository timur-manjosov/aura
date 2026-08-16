"""The periodic digest's clock: deciding which guilds are due, and posting theirs.

**This is the project's second background task, not its first.** Phase 3a-2's
extraction sweeper (aura.extraction.pipeline.run_extraction_sweeper) already
established the shape used here -- one `while True` task per process, created in
setup_hook, cancelled in close(), never dying of an exception it can survive --
and this module follows it deliberately rather than introducing APScheduler
alongside it. CLAUDE.md names APScheduler in the tech stack for exactly this
feature, and it is not used: the whole of what a scheduler library would provide
is "wake up periodically", which is four lines of asyncio here, while everything
that is actually hard about this feature -- surviving a restart mid-interval,
catching up exactly once after downtime, two runners never posting the same
window twice -- is solved in SQL and would be solved in SQL regardless of what
wakes the task up. Adding a dependency that owns the schedule would also move
that state into the scheduler's memory, which is precisely where it must not
live. The deviation is recorded in reports/phase-3e.txt.

**The tick interval is not the digest interval.** The task wakes roughly hourly
(DIGEST_CHECK_INTERVAL_SECONDS) and asks each configured guild whether its own
interval has elapsed. That indirection is what makes a weekly digest survive
restarts at all: there is no timer counting down to next Sunday that a container
restart could reset, only a stored "the last window ended here" that a tick
compares against the clock.

**Exactly one catch-up after downtime.** A guild that was due while the process
was down is due again on the first tick after it comes back, and the digest it
gets covers everything since its last finished window -- one message, however
long the outage. Nothing in this module iterates over missed intervals, and that
is the whole mechanism: see try_claim_digest_run for why "one catch-up" is a
structural property of a half-open window rather than a rule that had to be
implemented.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime

import aiosqlite
import discord

from aura.config import Settings
from aura.db.connection import utc_iso, utc_now
from aura.db.digest_config import DigestConfig, get_enabled_digest_configs
from aura.db.digest_state import (
    DigestRunOutcome,
    due_cutoff,
    last_covered_until,
    mark_digest_run_failed,
    try_claim_digest_run,
)
from aura.digest.builder import DigestContent, build_digest
from aura.digest.formatter import build_digest_embed, digest_locale
from aura.digest.gateway import DigestGateway

logger = logging.getLogger(__name__)


def window_start(config: DigestConfig, last_finished: str | None) -> str:
    """Where this guild's next digest window begins.

    The later of two boundaries, and it needs both:

      * the end of the last window that finished, so consecutive digests neither
        overlap nor leave a gap;
      * the moment digests were switched on, so a guild's first digest starts
        from that decision rather than from the beginning of its history (which
        is the onboarding trigger's job) -- and so re-enabling digests after a
        pause does not open with everything that accumulated while they were
        off.

    Taking the later of the two is what makes the second case work without a
    special case: after a re-enable, enabled_at is newer than any stored run
    (see set_digest_config), so it wins; in steady state the last run is newer,
    so it does.

    Compared as text, like every other timestamp comparison in this project:
    both are fixed-width UTC ISO-8601, where lexicographic order is
    chronological order (aura.db.connection.utc_iso).
    """
    enabled_at = utc_iso(config.enabled_at)
    if last_finished is None:
        return enabled_at
    return max(enabled_at, last_finished)


async def send_due_digests(
    db: aiosqlite.Connection, gateway: DigestGateway, *, now: datetime
) -> int:
    """Post a digest for every guild whose interval has elapsed. Returns how many posted.

    One guild at a time, sequentially rather than concurrently, for the same
    reasons flush_due_batches gives: the per-connection lock serializes the
    database work anyway, nobody is waiting on a digest, and a sequential sweep
    keeps "what did this tick do?" answerable by reading the log in order.

    A failure in one guild never stops the others. Each is wrapped
    individually, so one guild with a deleted channel or a corrupt boundary
    cannot starve every other guild on the deployment -- the exact failure shape
    a single shared try block would produce.

    Takes no Settings, unlike its extraction counterpart: every number this
    sweep needs -- which channel, how often, from when -- is per-guild state
    read from the database, and the one process-wide value (how often to wake)
    belongs to the loop below rather than to a single sweep.

    Requires a timezone-aware `now`, injected rather than read here, matching
    every other time-sensitive function in this project: one reading drives the
    due cutoff, the window end and the run's timestamp for the whole sweep, so
    they cannot disagree with each other, and "the container came back up after
    a missed window" becomes testable at the exact instant it matters.
    """
    if now.tzinfo is None:
        raise ValueError(f"now must be a timezone-aware datetime, got {now!r}")

    posted = 0
    for config in await get_enabled_digest_configs(db):
        try:
            if await _post_guild_digest(db, gateway, config=config, now=now):
                posted += 1
        except Exception:
            logger.exception("Digest failed for guild %s", config.guild_id)
    return posted


async def _post_guild_digest(
    db: aiosqlite.Connection,
    gateway: DigestGateway,
    *,
    config: DigestConfig,
    now: datetime,
) -> bool:
    """Evaluate and, if it is due and has content, post one guild's digest.

    The ordering is the load-bearing part of this whole sub-phase, and every
    step of it fails in a direction that was chosen rather than inherited:

      0. Does the window make sense at all? A clock that moved backwards past
         the last window's end is the one input that can make it not.
      1. Is it due? A cheap read; a guild mid-interval costs one query per tick.
      2. Assemble the window. Read-only, no side effects, so it is safe to have
         done this and then abandon it.
      3. Resolve the target channel. Also read-only, and also before the claim
         (see _resolve_target), so a permanently deleted channel does not write
         and un-write a bookkeeping row every hour forever.
      4. CLAIM the window atomically -- before anything is posted. A second
         runner evaluating the same guild concurrently loses here and stays
         silent, which is what makes a duplicate weekly post impossible rather
         than merely unlikely.
      5. Send. A failure here releases the claim (see mark_digest_run_failed),
         so the window is retried on the next tick instead of being silently
         skipped.

    An empty window claims a run too, recording SKIPPED_EMPTY. That is not
    bookkeeping for its own sake: it consumes a window with nothing in it (there
    is nothing there to lose) and keeps the cadence periodic, so a quiet week is
    followed by a digest a week later rather than by one at whatever arbitrary
    minute the next fact happens to land.

    There is deliberately NO re-read of the channel setting immediately before
    the send, unlike aura.proactive.responder's freshest-setting check. That
    check exists there because a multi-second paid LLM call sits between reading
    the setting and posting, which is a real window for a moderator to change
    their mind in. Here the same span is a handful of local database reads, so a
    moderator disabling digests "just before" a post would have had to do it
    within a millisecond of a post that was already inevitable.
    """
    cutoff = due_cutoff(now, config.interval_seconds)
    since = window_start(config, await last_covered_until(db, guild_id=config.guild_id))
    until = utc_iso(now)

    if since >= until:
        # The window would end at or before it starts, which means the clock
        # went backwards past the last window's end (an NTP correction, a host
        # clock reset) or a stored timestamp is in the future. Checked BEFORE
        # dueness rather than after, so it is reported as the anomaly it is
        # instead of hiding inside the ordinary "not due yet" path -- digests
        # stopping for a week after a clock jump is exactly the kind of silence
        # an operator needs a log line for.
        logger.warning(
            "Skipping the digest for guild %s: its next window would end (%s) at or "
            "before it starts (%s). Has the system clock moved backwards? Digests "
            "resume once the clock passes the last window's end again.",
            config.guild_id,
            until,
            since,
        )
        return False

    if since > cutoff:
        # The ordinary case: this guild is mid-interval. One indexed read per
        # tick and nothing else.
        return False

    content = await build_digest(db, guild_id=config.guild_id, since=since, until=until)

    if content.is_empty:
        claimed = await try_claim_digest_run(
            db,
            guild_id=config.guild_id,
            channel_id=config.channel_id,
            covered_from=since,
            covered_until=until,
            new_fact_count=0,
            milestone_count=0,
            updated_fact_count=0,
            outcome=DigestRunOutcome.SKIPPED_EMPTY,
            cutoff=cutoff,
            now=now,
        )
        if claimed is not None:
            logger.info(
                "Nothing changed in guild %s since %s; no digest posted", config.guild_id, since
            )
        return False

    channel = await _resolve_target(gateway, config=config)
    if channel is None:
        return False

    run_id = await try_claim_digest_run(
        db,
        guild_id=config.guild_id,
        channel_id=config.channel_id,
        covered_from=since,
        covered_until=until,
        new_fact_count=len(content.new_facts),
        milestone_count=len(content.milestones),
        updated_fact_count=len(content.changes),
        outcome=DigestRunOutcome.POSTED,
        cutoff=cutoff,
        now=now,
    )
    if run_id is None:
        # Another evaluation of this guild claimed the same window first. It is
        # posting (or has posted) the same content; this one must not.
        logger.info(
            "Digest window for guild %s was already claimed by a concurrent run; "
            "staying silent",
            config.guild_id,
        )
        return False

    if not await _send(db, channel, config=config, content=content, run_id=run_id):
        return False

    logger.info(
        "Posted a digest to channel %s in guild %s: %d new fact(s), %d milestone(s), "
        "%d update(s) since %s",
        config.channel_id,
        config.guild_id,
        len(content.new_facts),
        len(content.milestones),
        len(content.changes),
        since,
    )
    return True


async def _resolve_target(
    gateway: DigestGateway, *, config: DigestConfig
) -> discord.TextChannel | None:
    """Find the channel this guild's digest may be posted into, or None with a reason.

    Deliberately runs BEFORE the window is claimed, unlike the send itself. Both
    of its refusals are conditions that tend to be permanent -- a deleted
    channel, a revoked invite, a hand-edited configuration row -- and claiming
    first would write, and immediately un-write, one bookkeeping row per tick
    for as long as the condition lasts. Resolving first is read-only, so a guild
    whose channel is simply gone costs one cache lookup an hour and leaves no
    trace but a log line, while the window it never got to post stays open for
    whenever a moderator fixes it.

    This does not weaken the double-post guarantee at all: resolving writes
    nothing and decides nothing, and the claim still sits between it and the
    send.

    The cross-guild check is not failure handling. A digest is built from one
    guild's facts, so posting it into a channel belonging to a different guild
    would publish that guild's entire recent knowledge model in a server that
    never saw any of it. The slash command resolves channels within the invoking
    guild, so this is unreachable through ordinary use -- it is reachable
    through a hand-edited configuration row, and a cross-guild data leak is not
    something to leave depending on the front door.
    """
    channel = await gateway.resolve_channel(config.channel_id)
    if channel is None:
        logger.warning(
            "Digest for guild %s could not be posted: channel %s is unavailable. "
            "The window stays open and will be retried.",
            config.guild_id,
            config.channel_id,
        )
        return None

    if channel.guild.id != config.guild_id:
        logger.error(
            "Refusing to post guild %s's digest into channel %s, which belongs to guild %s",
            config.guild_id,
            config.channel_id,
            channel.guild.id,
        )
        return None

    return channel


async def _send(
    db: aiosqlite.Connection,
    channel: discord.TextChannel,
    *,
    config: DigestConfig,
    content: DigestContent,
    run_id: int,
) -> bool:
    """Render and post one claimed digest, releasing its window if the send fails.

    The only step that happens after the claim, and therefore the only one that
    has to be able to give the window back: a window may stay consumed only if a
    message actually reached Discord.
    """
    locale = digest_locale(channel.guild)
    embed = build_digest_embed(
        content, locale=locale, interval_seconds=config.interval_seconds
    )
    try:
        # Mentions are suppressed explicitly even though Discord does not
        # resolve them inside an embed: a fact's text is written by a server
        # member, and "@everyone" reaching a weekly automated post is the kind
        # of thing that must be impossible by construction rather than by a
        # property of where the text happens to be rendered today.
        await channel.send(embed=embed, allowed_mentions=discord.AllowedMentions.none())
    except Exception:
        # A post can fail for reasons entirely outside Aura's control: the
        # channel was deleted between resolving and sending, send permissions
        # were revoked, Discord returned an error. CancelledError is a
        # BaseException and still propagates, so shutdown is not swallowed.
        logger.exception(
            "Digest post failed in channel %s (guild %s); the window stays open and "
            "will be retried",
            config.channel_id,
            config.guild_id,
        )
        await mark_digest_run_failed(db, run_id=run_id)
        return False

    return True


async def run_digest_scheduler(
    db: aiosqlite.Connection, gateway: DigestGateway, *, settings: Settings
) -> None:
    """Wake periodically and post whatever digests are due. Runs for the process's life.

    Never dies of a failure it can survive, for the same reason the extraction
    sweeper does not: a scheduler task that exits silently leaves a bot that
    looks healthy while its digests simply never arrive again, which is the
    exact failure shape CLAUDE.md's non-negotiable principle rules out.
    CancelledError is a BaseException and still propagates, so shutdown works.

    The first sweep runs immediately rather than after one full interval,
    because that is what makes a restart pick up a window missed during
    downtime promptly instead of up to an hour later.
    """
    interval = settings.digest_check_interval_seconds
    logger.info(
        "Digest scheduler started: checking every %.0fs "
        "(posts only in guilds configured via /aura-digest)",
        interval,
    )
    while True:
        try:
            await send_due_digests(db, gateway, now=utc_now())
        except Exception:
            logger.exception("Digest sweep failed; continuing")
        await asyncio.sleep(interval)
