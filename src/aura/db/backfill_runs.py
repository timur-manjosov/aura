"""Backfill's durable bookkeeping: the run row, its state machine, and the
restart-safe cursor that makes a multi-day run survive a container restart.

Modelled directly on aura.db.digest_state, which the phase brief names as the
pattern to follow, and the two share the property that actually matters: **there
is no in-memory state at all.** The digest's "when did the last window end" and
backfill's "which message have we got to" are both single stored values that a
tick reads, acts on, and writes back -- so a restart resumes with no recovery
step, because the row was the only state there ever was. A cursor held in a
Python attribute would reset on every container restart, and for a run that may
legitimately take days that is not a small bug: it would re-read a channel's
entire history from the beginning, re-paying for every batch, on every deploy.

**THE CURSOR IS ADVANCED AFTER THE WORK IT COVERS, NEVER BEFORE.** This is the
one ordering decision in the module and every guarantee rests on it:

  * A crash between distilling a page and advancing past it leaves the cursor
    where it was, so the next tick re-fetches that page and re-processes it.
    That costs one more slot from the daily cap and re-stages the same
    candidates, which pending_facts' UNIQUE constraint absorbs -- an idempotent
    repeat, exactly as live extraction's own crash-retry path behaves.
  * Advancing first would make the same crash SKIP those messages permanently,
    with nothing anywhere recording that it happened. A repeat is recoverable
    and visible; a skip is neither.

So the failure direction is "a message may be looked at twice, never zero
times", and the deliverable's "weder doppelt noch übersprungen" is met where it
is checkable: no message is skipped, and no duplicate CANDIDATE reaches a
moderator.

**Every write that moves a live run is guarded on its current state.** The
worker holds no lock across the paid call it makes between reading a run and
advancing it, so a moderator can pause or cancel in that gap -- and must not
find their decision silently undone a second later. `WHERE state = 'running'`
on the advance is what makes that impossible rather than unlikely: a worker that
affects no rows has been overruled and stops, and the batch it had already
staged simply gets re-processed if the run is ever resumed.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime
from enum import StrEnum

import aiosqlite
from pydantic import BaseModel

from aura.db.connection import connection_lock, utc_iso

# The states a worker may advance. Defined once and used by every read and every
# guard that needs it, so "is this run live" cannot be answered one way by the
# scheduler and another way by the write it authorizes.
_ACTIVE_STATES = ("running", "paused")
_ACTIVE_PLACEHOLDERS = ", ".join("?" for _ in _ACTIVE_STATES)

_RUN_COLUMNS = (
    "id, guild_id, channel_id, state, until_message_id, after_message_id, "
    "cursor_message_id, cursor_message_at, messages_scanned, candidates_staged, "
    "calls_spent, requested_by_id, started_at, updated_at, finished_at"
)


class BackfillState(StrEnum):
    """What a backfill run is currently doing, or how it ended.

    Five values rather than a boolean, because they are not shades of one
    thing: two are live (a worker may or may not touch them) and three are
    terminal for genuinely different reasons a moderator reads differently.

    RUNNING and PAUSED both keep their cursor and both can become the other.
    COMPLETED, CANCELLED and FAILED are terminal: a new /aura-backfill start on
    that channel opens a fresh run rather than resurrecting one of these.
    FAILED is kept distinct from CANCELLED because nobody chose it -- it means
    the channel stopped being readable -- and that is the one terminal state
    with an action attached to it (fix the permission, start again).
    """

    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


# The three states that end a run. Used by the command layer to decide whether
# `start` resumes or opens a new run, and named here so that decision cannot be
# made from a hand-written list somewhere else that forgets one.
TERMINAL_STATES = frozenset(
    {BackfillState.COMPLETED, BackfillState.CANCELLED, BackfillState.FAILED}
)


class BackfillRun(BaseModel):
    """One backfill run over one channel, as read back from the database.

    The three id fields are Discord snowflakes and are compared as integers
    throughout: a snowflake embeds its own creation timestamp, so ordering by id
    IS ordering by time (discord.py derives Message.created_at from the id, not
    the other way round). cursor_message_at carries no ordering weight for that
    reason -- it exists so /aura-backfill status can show a moderator a date
    without anyone having to decode a snowflake by hand.
    """

    id: int
    guild_id: int
    channel_id: int
    state: BackfillState
    until_message_id: int
    after_message_id: int | None
    cursor_message_id: int | None
    cursor_message_at: datetime | None
    messages_scanned: int
    candidates_staged: int
    calls_spent: int
    requested_by_id: int
    started_at: datetime
    updated_at: datetime
    finished_at: datetime | None

    @property
    def resume_after_id(self) -> int | None:
        """The exclusive lower bound the next fetch must start from.

        The cursor once there is one, and the run's original `since:` bound
        before that. Expressed here rather than at the call site because getting
        it the wrong way round is the exact bug that would make a resumed run
        re-read a channel from the beginning, and it should be impossible to
        write that bug twice.
        """
        return self.cursor_message_id if self.cursor_message_id is not None else self.after_message_id


def _row_to_run(row: sqlite3.Row) -> BackfillRun:
    return BackfillRun(
        id=row[0],
        guild_id=row[1],
        channel_id=row[2],
        state=BackfillState(row[3]),
        until_message_id=row[4],
        after_message_id=row[5],
        cursor_message_id=row[6],
        cursor_message_at=row[7],
        messages_scanned=row[8],
        candidates_staged=row[9],
        calls_spent=row[10],
        requested_by_id=row[11],
        started_at=row[12],
        updated_at=row[13],
        finished_at=row[14],
    )


class BackfillAlreadyActiveError(Exception):
    """Raised when a channel already has a run a new one would collide with.

    Carries the existing run so the command layer can tell a moderator whether
    to resume it or cancel it, rather than making them run a second command to
    find out which of the two situations they are in.
    """

    def __init__(self, existing: BackfillRun) -> None:
        self.existing = existing
        super().__init__(
            f"channel {existing.channel_id} already has a {existing.state.value} backfill run"
        )


async def start_backfill_run(
    conn: aiosqlite.Connection,
    *,
    guild_id: int,
    channel_id: int,
    until_message_id: int,
    after_message_id: int | None,
    requested_by_id: int,
    now: datetime,
) -> BackfillRun:
    """Open a new run over channel_id, or raise if one is already live there.

    The uniqueness is enforced by the partial index on (channel_id) WHERE state
    IN ('running', 'paused') -- see schema.sql -- and NOT by the read this
    function does first. That read exists only to hand the caller the existing
    run for its error message; the INSERT is what actually decides, so two
    moderators starting a backfill on one channel in the same second produce one
    run and one clean refusal rather than two cursors advancing over the same
    history and each paying for the other's messages.

    Requires a timezone-aware `now`, injected rather than read here, matching
    every other time-sensitive write in this project.
    """
    if now.tzinfo is None:
        raise ValueError(f"now must be a timezone-aware datetime, got {now!r}")
    if until_message_id <= 0:
        raise ValueError(
            f"until_message_id must be a positive snowflake, got {until_message_id}"
        )
    if after_message_id is not None and after_message_id <= 0:
        # A snowflake is never zero or negative. discord.utils.time_snowflake is
        # arithmetic against Discord's epoch and returns a large negative number
        # for any date before 2015, which Discord rejects as a bad `after` --
        # producing a run that retries and never progresses, with nothing in its
        # status to say why. Refused here so it cannot be stored at all; the
        # command layer clamps such a date to "no lower bound" before it gets
        # this far (see aura.commands.backfill._since_snowflake).
        raise ValueError(
            f"after_message_id must be a positive snowflake or None, got {after_message_id}"
        )
    if after_message_id is not None and after_message_id >= until_message_id:
        raise ValueError(
            f"after_message_id ({after_message_id}) must be below until_message_id "
            f"({until_message_id}); an empty or reversed range has nothing to backfill"
        )

    timestamp = utc_iso(now)
    async with connection_lock(conn):
        try:
            cursor = await conn.execute(
                """
                INSERT INTO backfill_runs
                    (guild_id, channel_id, state, until_message_id, after_message_id,
                     cursor_message_id, cursor_message_at, messages_scanned,
                     candidates_staged, calls_spent, requested_by_id, started_at,
                     updated_at, finished_at)
                VALUES (?, ?, ?, ?, ?, NULL, NULL, 0, 0, 0, ?, ?, ?, NULL)
                """,
                (
                    guild_id,
                    channel_id,
                    BackfillState.RUNNING,
                    until_message_id,
                    after_message_id,
                    requested_by_id,
                    timestamp,
                    timestamp,
                ),
            )
            run_id = cursor.lastrowid
            assert run_id is not None  # guaranteed by sqlite after a successful INSERT
            await conn.commit()
        except sqlite3.IntegrityError:
            await conn.rollback()
            existing = await _active_run_unlocked(conn, channel_id=channel_id)
            if existing is None:
                # The partial index refused the insert, so a live run existed a
                # microsecond ago and has since been resolved. Re-raising the
                # original error would be honest but useless; the caller's own
                # retry is a moderator running the command again.
                raise
            raise BackfillAlreadyActiveError(existing) from None

    return BackfillRun(
        id=run_id,
        guild_id=guild_id,
        channel_id=channel_id,
        state=BackfillState.RUNNING,
        until_message_id=until_message_id,
        after_message_id=after_message_id,
        cursor_message_id=None,
        cursor_message_at=None,
        messages_scanned=0,
        candidates_staged=0,
        calls_spent=0,
        requested_by_id=requested_by_id,
        started_at=now,
        updated_at=now,
        finished_at=None,
    )


async def _active_run_unlocked(
    conn: aiosqlite.Connection, *, channel_id: int
) -> BackfillRun | None:
    """The running or paused run for channel_id. THE CALLER MUST HOLD THE LOCK."""
    async with conn.execute(
        f"""
        SELECT {_RUN_COLUMNS} FROM backfill_runs
        WHERE channel_id = ? AND state IN ({_ACTIVE_PLACEHOLDERS})
        """,
        (channel_id, *_ACTIVE_STATES),
    ) as cursor:
        row = await cursor.fetchone()
    return _row_to_run(row) if row is not None else None


async def get_active_run(
    conn: aiosqlite.Connection, *, channel_id: int
) -> BackfillRun | None:
    """The running or paused run for channel_id, or None if it has neither.

    At most one can exist, guaranteed by the partial unique index rather than by
    this query's shape -- so a second row appearing here would be a schema
    problem, not something callers need to disambiguate.
    """
    async with connection_lock(conn):
        return await _active_run_unlocked(conn, channel_id=channel_id)


async def get_running_runs(conn: aiosqlite.Connection) -> list[BackfillRun]:
    """Every run a worker may advance, oldest first.

    Oldest first because this is a work queue, the same reasoning
    get_pending_facts gives for its own ordering: a run started last week must
    not sit behind one started this morning forever just because the newer one
    keeps producing work.

    Deliberately not scoped to a guild: there is one worker for the process, and
    "which runs are live anywhere" is the question it actually asks.
    """
    async with connection_lock(conn):
        async with conn.execute(
            f"""
            SELECT {_RUN_COLUMNS} FROM backfill_runs
            WHERE state = ?
            ORDER BY id ASC
            """,
            (BackfillState.RUNNING,),
        ) as cursor:
            rows = await cursor.fetchall()
    return [_row_to_run(row) for row in rows]


async def get_recent_runs(
    conn: aiosqlite.Connection, *, guild_id: int, limit: int
) -> list[BackfillRun]:
    """A guild's most recent runs, newest first, for /aura-backfill status.

    Newest first, unlike the work queue above and like every other diagnostic
    read in this project: the question this answers is "what has backfill been
    doing lately", and the useful end of that is the recent one.

    Rejects a negative limit rather than passing it to SQL, where LIMIT -1 means
    no limit at all -- the same trap get_pending_facts and get_digest_runs both
    document.
    """
    if limit < 0:
        raise ValueError(f"limit must not be negative, got {limit}")

    async with connection_lock(conn):
        async with conn.execute(
            f"""
            SELECT {_RUN_COLUMNS} FROM backfill_runs
            WHERE guild_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (guild_id, limit),
        ) as cursor:
            rows = await cursor.fetchall()
    return [_row_to_run(row) for row in rows]


async def advance_cursor(
    conn: aiosqlite.Connection,
    *,
    run_id: int,
    cursor_message_id: int,
    cursor_message_at: datetime,
    messages_scanned: int,
    candidates_staged: int,
    calls_spent: int,
    now: datetime,
) -> bool:
    """Move a running run's cursor forward and add to its counters. Returns whether it moved.

    Two guards, and both are load-bearing rather than defensive:

      * `state = 'running'` -- a moderator who paused or cancelled during the
        paid call this advance follows must not have their decision silently
        undone. A False return means exactly that happened, and the caller stops
        rather than continuing to spend on a run nobody wants any more.
      * `cursor_message_id < ?` -- the cursor is monotonic by construction, so a
        second worker (or a retried tick) that computed an older position cannot
        drag it backwards and cause a stretch of history to be paid for twice.
        The comparison uses IS NULL for a run that has never advanced, since
        NULL < anything is NULL in SQL and would silently refuse every first
        advance.

    Counters are added to rather than assigned, so an advance never has to know
    what the totals were when it started -- which it could not know without
    holding a lock across the LLM call it follows.
    """
    if now.tzinfo is None:
        raise ValueError(f"now must be a timezone-aware datetime, got {now!r}")
    for name, value in (
        ("messages_scanned", messages_scanned),
        ("candidates_staged", candidates_staged),
        ("calls_spent", calls_spent),
    ):
        if value < 0:
            raise ValueError(f"{name} must not be negative, got {value}")

    async with connection_lock(conn):
        cursor = await conn.execute(
            """
            UPDATE backfill_runs
               SET cursor_message_id = ?,
                   cursor_message_at = ?,
                   messages_scanned = messages_scanned + ?,
                   candidates_staged = candidates_staged + ?,
                   calls_spent = calls_spent + ?,
                   updated_at = ?
             WHERE id = ?
               AND state = ?
               AND (cursor_message_id IS NULL OR cursor_message_id < ?)
            """,
            (
                cursor_message_id,
                utc_iso(cursor_message_at),
                messages_scanned,
                candidates_staged,
                calls_spent,
                utc_iso(now),
                run_id,
                BackfillState.RUNNING,
                cursor_message_id,
            ),
        )
        await conn.commit()
    return cursor.rowcount == 1


async def set_run_state(
    conn: aiosqlite.Connection,
    *,
    run_id: int,
    state: BackfillState,
    now: datetime,
    from_states: tuple[BackfillState, ...],
) -> bool:
    """Move one run into `state`, but only from one of `from_states`. Returns whether it moved.

    Every transition in this module's state machine goes through here, and every
    caller has to name what it believes the run currently is. That is deliberate
    friction: an unguarded UPDATE would let a worker mark a run 'completed'
    milliseconds after a moderator cancelled it, or let a second /aura-backfill
    pause re-stamp a run that was already finished -- both of which read, after
    the fact, as the moderator's command having done nothing.

    finished_at is set exactly when the target state is terminal and cleared
    when it is not, so "is this run over" has one answer in the data rather than
    two that can disagree.
    """
    if now.tzinfo is None:
        raise ValueError(f"now must be a timezone-aware datetime, got {now!r}")
    if not from_states:
        raise ValueError("from_states must name at least one state to move from")

    timestamp = utc_iso(now)
    finished_at = timestamp if state in TERMINAL_STATES else None
    placeholders = ", ".join("?" for _ in from_states)
    async with connection_lock(conn):
        cursor = await conn.execute(
            f"""
            UPDATE backfill_runs
               SET state = ?, updated_at = ?, finished_at = ?
             WHERE id = ? AND state IN ({placeholders})
            """,
            (state, timestamp, finished_at, run_id, *from_states),
        )
        await conn.commit()
    return cursor.rowcount == 1
