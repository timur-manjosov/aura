"""Durable budget protection for backfill's distillation calls: the per-guild
daily cap, independent of live extraction's.

The fifth instance of a shape that is a convention rather than a coincidence in
this project -- one append-only ledger, one guarded INSERT whose WHERE clause
re-checks the cap at write time, a stored UTC day key, no in-memory state --
and the reasoning transfers unchanged from aura.db.extraction_state, which is
its nearest twin. Only what is DIFFERENT is argued here.

**Why a separate budget at all, when backfill runs the same distillation call
extraction does.** Because a shared number would leave neither call site with a
bound of its own, and the direction of the damage is not symmetric. Backfill is
bulk work over a fixed, possibly enormous backlog; live extraction is a trickle
over messages members are writing right now. Put them on one budget and the
first backfill of a two-year channel eats the whole day's allowance in its first
minutes, and every message written that day goes unextracted. A guild would
experience that as extraction having stopped, with nothing in the logs to
separate it from extraction being broken -- exactly the failure shape
reports/phase-3a-3.txt Section 4 rejected when it gave the supersession
judgement its own third budget rather than a share of extraction's.

**What a refused claim means here is the opposite of what it means for live
extraction, and this is the one place the twinning genuinely breaks.** When
EXTRACTION_DAILY_CAP binds, the batch is DROPPED: holding it would accumulate
raw message text for the rest of the UTC day and then release a flood at
midnight, and the batch window's promise of a bounded wait would become an
unbounded one. Backfill has no such promise to keep and nothing to hold: its
input is Discord's own history, which is not going anywhere, and its cursor
already records exactly where to resume. So a refusal here PAUSES the run for
the rest of the day instead of discarding a page of history -- see
aura.backfill.worker. Dropping would silently lose messages a moderator
explicitly asked to be read, which is not a thing a spend limit is allowed to
do.

**No idempotency key, same as extraction_calls and for the same reason.** The
trigger is Aura's own worker, not a Discord event that can be redelivered. A
crash between claiming a slot and finishing the batch therefore spends the slot
and re-does the work, which is the only safe direction for a spend limit to err
in; the duplicate CANDIDATES that retry would otherwise produce are prevented
one layer up by pending_facts' UNIQUE constraint.
"""
from __future__ import annotations

from datetime import datetime
from enum import StrEnum

import aiosqlite
from pydantic import BaseModel

from aura.db.connection import connection_lock, utc_day, utc_iso

# One statement that checks the cap and writes the row, for the reason this
# module's docstring gives. ON CONFLICT is absent rather than forgotten: this
# table carries no uniqueness constraint to conflict on.
_ACQUIRE_CALL_SLOT_SQL = """
INSERT INTO backfill_calls (guild_id, run_id, message_count, called_at, call_day)
SELECT ?, ?, ?, ?, ?
WHERE (
        SELECT COUNT(*) FROM backfill_calls
        WHERE guild_id = ? AND call_day = ?
    ) < ?
"""

# Mirrors MAX_DAILY_CAP in aura.db.extraction_state and aura.db.proactive_state,
# and exists for the same reason: the value is bound into SQL, and sqlite3
# refuses a Python int that does not fit a signed 64-bit integer, so a value
# past this would raise on every tick instead of being refused once where an
# operator can see it.
MAX_DAILY_CAP = 1_000_000


class BackfillCallOutcome(StrEnum):
    """Whether a backfill distillation call was allowed to spend, and why not if it wasn't."""

    GRANTED = "granted"
    DAILY_CAP_REACHED = "daily_cap_reached"


class BackfillCallAttempt(BaseModel):
    """The result of one attempt to claim a backfill distillation call.

    Carries the numbers the decision was made on rather than just the verdict,
    so a log line -- and /aura-backfill status -- can say "24 of 30 of today's
    budget is gone" without re-deriving state that has since moved on.
    """

    outcome: BackfillCallOutcome
    # Calls spent on this UTC day INCLUDING this attempt when it was granted,
    # matching how both existing ledgers report their own.
    daily_count: int
    daily_cap: int

    @property
    def granted(self) -> bool:
        """Whether this attempt actually took a slot from the budget."""
        return self.outcome is BackfillCallOutcome.GRANTED


async def count_backfill_calls_on(
    conn: aiosqlite.Connection, *, guild_id: int, day: str
) -> int:
    """Return how many backfill distillation calls guild_id has spent on a UTC day.

    Read-only. Takes the day as a string produced by utc_day so the caller's
    clock, not this function's, defines "today".

    Used by /aura-backfill status to show a moderator why a run is not moving,
    and by the worker as a CHEAP PRE-CHECK before it starts fetching pages for a
    run whose budget is already gone. That pre-check is an optimization and
    never the decision: try_acquire_backfill_call_slot below is what actually
    grants a slot, atomically, and it re-checks the cap at write time regardless
    of what this function returned a moment earlier.
    """
    async with connection_lock(conn):
        async with conn.execute(
            "SELECT COUNT(*) FROM backfill_calls WHERE guild_id = ? AND call_day = ?",
            (guild_id, day),
        ) as cursor:
            row = await cursor.fetchone()
    return int(row[0]) if row else 0


async def try_acquire_backfill_call_slot(
    conn: aiosqlite.Connection,
    *,
    guild_id: int,
    run_id: int,
    message_count: int,
    daily_cap: int,
    now: datetime,
) -> BackfillCallAttempt:
    """Atomically take one slot from the guild's daily backfill budget.

    Call this the moment a batch is assembled and *before* the LLM call it
    authorizes -- the same ordering, for the same reason, every other ledger in
    this project uses. A slot is recorded when it is claimed, not when the work
    it authorizes succeeds, so a crash or an API failure downstream spends the
    slot instead of quietly refunding it.

    Never raises on a normal refusal: being out of budget is the expected way a
    multi-day backfill spends its second day, not an error. A daily_cap of 0 is
    valid and means "no backfill may spend anything today", which is a useful
    off switch that leaves live extraction entirely untouched.

    Requires a timezone-aware `now`, injected rather than read from the clock
    here, so the daily boundary is testable at the exact moment it matters.
    """
    if not 0 <= daily_cap <= MAX_DAILY_CAP:
        raise ValueError(f"daily_cap must be between 0 and {MAX_DAILY_CAP}, got {daily_cap}")
    if message_count < 0:
        raise ValueError(f"message_count must not be negative, got {message_count}")
    if now.tzinfo is None:
        raise ValueError(f"now must be a timezone-aware datetime, got {now!r}")

    # One `now` produces both the timestamp and the day key, so they can never
    # disagree -- two separate clock reads straddling midnight would file a row
    # under one day with a timestamp from another.
    day = utc_day(now)
    called_at = utc_iso(now)

    async with connection_lock(conn):
        try:
            cursor = await conn.execute(
                _ACQUIRE_CALL_SLOT_SQL,
                (guild_id, run_id, message_count, called_at, day, guild_id, day, daily_cap),
            )
            if cursor.rowcount == 1:
                await conn.commit()
                async with conn.execute(
                    "SELECT COUNT(*) FROM backfill_calls WHERE guild_id = ? AND call_day = ?",
                    (guild_id, day),
                ) as count_cursor:
                    row = await count_cursor.fetchone()
                return BackfillCallAttempt(
                    outcome=BackfillCallOutcome.GRANTED,
                    daily_count=int(row[0]) if row else 1,
                    daily_cap=daily_cap,
                )

            # The INSERT's own WHERE clause refused it: the cap is full. Read
            # the count back for the trail, from the same transaction that just
            # declined to add to it.
            await conn.rollback()
            async with conn.execute(
                "SELECT COUNT(*) FROM backfill_calls WHERE guild_id = ? AND call_day = ?",
                (guild_id, day),
            ) as count_cursor:
                row = await count_cursor.fetchone()
            return BackfillCallAttempt(
                outcome=BackfillCallOutcome.DAILY_CAP_REACHED,
                daily_count=int(row[0]) if row else 0,
                daily_cap=daily_cap,
            )
        except BaseException:
            await conn.rollback()
            raise
