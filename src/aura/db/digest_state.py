"""The periodic digest's durable bookkeeping: which windows have been reported,
and the atomic claim that stops the same window being reported twice.

The fifth instance in this project of a shape that is now a convention rather
than a coincidence -- one append-only table, one guarded INSERT whose WHERE
clause re-checks the decision at write time, no in-memory state -- and the
reasoning transfers almost unchanged from aura.db.proactive_state, which
introduced it. Two things make it worth restating anyway, because what is being
protected here is not money:

**Durable, not in-memory.** Aura runs in a container with a restart policy, so a
"last digest sent at" held in a Python attribute resets on every restart. For a
weekly digest that is not a small bug: a container that restarts twice a week
would either never reach its interval and go silent forever, or -- with the
opposite default -- post a fresh digest on every boot. The ledger is the only
state there is, so a restart resumes mid-interval with no recovery step, and a
window missed while the process was down is picked up by the next tick after it
comes back (see try_claim_digest_run for why that is one catch-up and not one
per missed interval).

**Race-condition safe, not check-then-set.** The scheduler tick is a background
task; a tick that overruns its own interval, a second process sharing the
database file, or a manual re-run can all put two evaluations of the same guild
in flight at once. "Read the last run, decide it is due, then post" is the
textbook lost-update race, and its prize here is a duplicate weekly digest in a
public channel -- exactly the unwanted interruption CLAUDE.md's conservative
stance rules out. So the run row is CLAIMED by one guarded INSERT before
anything is posted, and a caller whose insert affected no rows has lost the race
and must stay silent.

**Why an outcome, not a boolean.** Two of the three outcomes advance the
schedule and one deliberately does not; see the digest_runs comment in
schema.sql for the full table. The consequence for this module is that every
read of "where does the next digest start" filters on _ADVANCING_OUTCOMES, and
there is exactly one definition of that set, used by both the read and the
claim's guard -- if they could disagree, a failed post would be retried by one
and skipped by the other.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta
from enum import StrEnum

import aiosqlite
from pydantic import BaseModel

from aura.db.connection import connection_lock, utc_iso


class DigestRunOutcome(StrEnum):
    """What became of one evaluated digest window.

    POSTED and SKIPPED_EMPTY both mean "this window is finished with"; only
    POST_FAILED leaves it open for the next tick to retry. See schema.sql for
    why an empty window still counts as finished (there is nothing in it to
    lose, and consuming it keeps the cadence periodic).
    """

    POSTED = "posted"
    SKIPPED_EMPTY = "skipped_empty"
    POST_FAILED = "post_failed"


# The outcomes that move the schedule forward. Defined once and used by both the
# boundary read and the claim's guard, so the two cannot drift into disagreeing
# about whether a failed post has to be retried.
_ADVANCING_OUTCOMES = (DigestRunOutcome.POSTED, DigestRunOutcome.SKIPPED_EMPTY)
_ADVANCING_PLACEHOLDERS = ", ".join("?" for _ in _ADVANCING_OUTCOMES)

_RUN_COLUMNS = (
    "id, guild_id, channel_id, covered_from, covered_until, new_fact_count, "
    "milestone_count, updated_fact_count, outcome, ran_at"
)

# One statement that re-checks "no finished run already covers a window ending
# after the due cutoff" and writes the claim, for the reason this module's
# docstring gives. No ON CONFLICT clause: this table carries no uniqueness
# constraint to conflict on, because the thing being made unique is not a column
# value but a moment in time, which is what the WHERE clause below expresses.
_CLAIM_RUN_SQL = f"""
INSERT INTO digest_runs
    (guild_id, channel_id, covered_from, covered_until, new_fact_count,
     milestone_count, updated_fact_count, outcome, ran_at)
SELECT ?, ?, ?, ?, ?, ?, ?, ?, ?
WHERE NOT EXISTS (
        SELECT 1 FROM digest_runs
        WHERE guild_id = ?
          AND outcome IN ({_ADVANCING_PLACEHOLDERS})
          AND covered_until > ?
    )
"""


class DigestRun(BaseModel):
    """One evaluated digest window, as read back from the database.

    The counts are what the window actually contained at the moment it was
    evaluated, recorded rather than recomputed: the facts behind them keep
    changing afterwards, so this is the only moment at which they are knowable.
    """

    id: int
    guild_id: int
    channel_id: int
    covered_from: datetime
    covered_until: datetime
    new_fact_count: int
    milestone_count: int
    updated_fact_count: int
    outcome: DigestRunOutcome
    ran_at: datetime


def _row_to_run(row: sqlite3.Row) -> DigestRun:
    return DigestRun(
        id=row[0],
        guild_id=row[1],
        channel_id=row[2],
        covered_from=row[3],
        covered_until=row[4],
        new_fact_count=row[5],
        milestone_count=row[6],
        updated_fact_count=row[7],
        outcome=DigestRunOutcome(row[8]),
        ran_at=row[9],
    )


def due_cutoff(now: datetime, interval_seconds: int) -> str:
    """The newest window end that still leaves a guild due for a digest.

    A guild is due when the last window it finished ended at or before this
    moment. Expressed as one function, and as a fixed-width UTC ISO string,
    because the same value is used in two places that must agree exactly: the
    scheduler's own "is this guild due" decision, and the WHERE clause of the
    claim below that re-checks it at write time under concurrency. Two separate
    subtractions would be two chances to disagree by a microsecond.

    Requires a timezone-aware `now`, injected rather than read from the clock
    here, matching every other time-sensitive function in this project -- which
    is also what makes "the container was down across a due window" testable at
    the exact moment it matters instead of by waiting a week.
    """
    if now.tzinfo is None:
        raise ValueError(f"now must be a timezone-aware datetime, got {now!r}")
    if interval_seconds <= 0:
        raise ValueError(f"interval_seconds must be positive, got {interval_seconds}")
    return utc_iso(now - timedelta(seconds=interval_seconds))


async def last_covered_until(conn: aiosqlite.Connection, *, guild_id: int) -> str | None:
    """The end of the newest window this guild has finished, or None if it has none.

    Returned as the raw stored string rather than a datetime, on purpose: its
    only consumers compare it against other fixed-width UTC ISO strings (the
    config's enabled_at, the due cutoff, facts.created_at in SQL), and parsing
    it into a datetime only to format it back would add a failure mode --
    an unparseable timestamp -- to a path that currently cannot have one.

    Ignores POST_FAILED runs, which is the whole point of that outcome existing:
    a window whose send failed has not been reported to anyone, so the next
    digest must still start where it did.
    """
    async with connection_lock(conn):
        async with conn.execute(
            f"""
            SELECT MAX(covered_until) FROM digest_runs
            WHERE guild_id = ? AND outcome IN ({_ADVANCING_PLACEHOLDERS})
            """,
            (guild_id, *_ADVANCING_OUTCOMES),
        ) as cursor:
            row = await cursor.fetchone()
    return row[0] if row is not None else None


async def try_claim_digest_run(
    conn: aiosqlite.Connection,
    *,
    guild_id: int,
    channel_id: int,
    covered_from: str,
    covered_until: str,
    new_fact_count: int,
    milestone_count: int,
    updated_fact_count: int,
    outcome: DigestRunOutcome,
    cutoff: str,
    now: datetime,
) -> int | None:
    """Atomically claim one digest window. Returns the run's id, or None if lost.

    Call this the moment the window's content is known and *before* the message
    it authorizes is posted -- the same ordering, for the same reason, that
    every spend ledger in this project uses. What is being protected differs
    (a public post rather than money) but the direction is identical: a crash
    between the claim and the send consumes the window, whereas claiming
    afterwards would let a crash produce a second digest for a window that was
    already reported.

    **Exactly one catch-up after downtime, by construction rather than by a
    special case.** Nothing here loops over missed intervals: a claim covers
    (covered_from, covered_until] whether that span is one interval or six
    weeks, and once it is written the guild is no longer due. A container that
    was off for a month therefore posts one digest covering the month, not one
    per missed week -- and the code has no notion of "a missed interval" at all,
    which is what makes the guarantee structural.

    `cutoff` must come from due_cutoff() with the same `now` and the same
    interval the caller used to decide the guild was due. The guard re-checks
    that no finished run already ends after it, so two evaluations in flight for
    one guild produce one claim and one silence.

    Never raises for a lost race -- being beaten to a window is an ordinary
    outcome under concurrency, not an error -- and the caller has nothing to
    undo, since nothing has been posted at the point this is called.
    """
    if now.tzinfo is None:
        raise ValueError(f"now must be a timezone-aware datetime, got {now!r}")
    if covered_until <= covered_from:
        raise ValueError(
            f"covered_until ({covered_until!r}) must be after covered_from "
            f"({covered_from!r}); an empty or reversed window has nothing to report"
        )

    async with connection_lock(conn):
        try:
            cursor = await conn.execute(
                _CLAIM_RUN_SQL,
                (
                    guild_id,
                    channel_id,
                    covered_from,
                    covered_until,
                    new_fact_count,
                    milestone_count,
                    updated_fact_count,
                    outcome,
                    utc_iso(now),
                    guild_id,
                    *_ADVANCING_OUTCOMES,
                    cutoff,
                ),
            )
            if cursor.rowcount != 1:
                await conn.rollback()
                return None
            run_id = cursor.lastrowid
            assert run_id is not None  # guaranteed by sqlite after a successful INSERT
            await conn.commit()
        except BaseException:
            await conn.rollback()
            raise
    return run_id


async def mark_digest_run_failed(conn: aiosqlite.Connection, *, run_id: int) -> None:
    """Record that a claimed run's message never made it out, releasing its window.

    The one write in this module that changes an existing row, and it only ever
    moves an outcome in the direction of "this did not happen". A window whose
    send failed -- a deleted channel, a revoked permission, a Discord error --
    was reported to nobody, so leaving it consumed would silently drop a week of
    changes from the only place they were going to be summarized. Releasing it
    means the next tick retries the same window, and keeps retrying while the
    channel stays broken, which costs one log line an hour and self-heals the
    moment a moderator fixes it.

    Guarded on the current outcome being POSTED so it can only ever undo a claim
    this same call made: a run that was already recorded as failed, or that
    somehow reads as skipped_empty, is left exactly as it is rather than
    rewritten.
    """
    async with connection_lock(conn):
        await conn.execute(
            "UPDATE digest_runs SET outcome = ? WHERE id = ? AND outcome = ?",
            (DigestRunOutcome.POST_FAILED, run_id, DigestRunOutcome.POSTED),
        )
        await conn.commit()


async def get_digest_runs(
    conn: aiosqlite.Connection, *, guild_id: int, limit: int
) -> list[DigestRun]:
    """Return a guild's most recent digest runs, newest first.

    Newest first, unlike the pending-review work queue and like every other
    diagnostic read in this project: the question this answers is "what has the
    digest been doing lately", and the useful end of that is the recent one.

    Rejects a negative limit rather than passing it to SQL, where LIMIT -1 means
    no limit at all -- the same trap get_pending_facts and get_recent_signals
    both document.
    """
    if limit < 0:
        raise ValueError(f"limit must not be negative, got {limit}")

    async with connection_lock(conn):
        async with conn.execute(
            f"""
            SELECT {_RUN_COLUMNS} FROM digest_runs
            WHERE guild_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (guild_id, limit),
        ) as cursor:
            rows = await cursor.fetchall()
    return [_row_to_run(row) for row in rows]
