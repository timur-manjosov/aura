"""Durable budget protection for variant generation: the per-guild daily cap.

The fourth instance of a shape this project now has three times already
(aura.db.proactive_state, aura.db.extraction_state, aura.db.supersession_state),
and built as a twin of those on purpose rather than as a variation on them: one
append-only ledger, one guarded INSERT that re-checks the cap at write time, a
stored UTC day key, no in-memory state. A reader who has understood any of the
other three has understood this one, and the one difference that does exist is
stated here rather than left to be noticed.

**One episode, one slot.** Variant generation always spends its generation call
and its fidelity-audit call together (see aura.variants_service) -- there is no
call site that generates without immediately auditing, and no call site that
audits without having just generated. A single slot per fact therefore bounds
both calls' cost at once; a second, separate ledger for the audit call would
only add bookkeeping with nothing left for it to protect against on its own.
"""
from __future__ import annotations

from datetime import datetime
from enum import StrEnum

import aiosqlite
from pydantic import BaseModel

from aura.db.connection import connection_lock, utc_day, utc_iso

# One statement that checks the cap and writes the row, so the check cannot be
# separated from the write even by a second process sharing the database file.
_ACQUIRE_VARIANT_SLOT_SQL = """
INSERT INTO variant_calls (guild_id, fact_id, called_at, call_day)
SELECT ?, ?, ?, ?
WHERE (
        SELECT COUNT(*) FROM variant_calls
        WHERE guild_id = ? AND call_day = ?
    ) < ?
"""

# Mirrors MAX_DAILY_CAP in the other three ledgers, for the same reason: the
# value is bound into SQL, and sqlite3 refuses a Python int that does not fit a
# signed 64-bit integer, so a value past this would raise on every fact instead
# of being refused once where an operator can see it.
MAX_DAILY_CAP = 1_000_000


class VariantCallOutcome(StrEnum):
    """Whether a variant-generation episode was allowed to spend, and why not if it wasn't."""

    GRANTED = "granted"
    DAILY_CAP_REACHED = "daily_cap_reached"


class VariantCallAttempt(BaseModel):
    """The result of one attempt to claim a variant-generation slot, with the state behind it.

    Carries the numbers the decision was made on rather than just the verdict,
    so a log line can say "12 of 200 of today's budget is gone" without
    re-deriving state that has since moved on.
    """

    outcome: VariantCallOutcome
    # Episodes spent on this UTC day INCLUDING this attempt when it was
    # granted, so the reading matches how the other three ledgers report
    # their own.
    daily_count: int
    daily_cap: int

    @property
    def granted(self) -> bool:
        """Whether this attempt actually took a slot from the budget."""
        return self.outcome is VariantCallOutcome.GRANTED


async def count_variant_calls_on(
    conn: aiosqlite.Connection, *, guild_id: int, day: str
) -> int:
    """Return how many variant-generation episodes guild_id has already spent on a UTC day.

    Read-only. Takes the day as a string produced by utc_day so the caller's
    clock, not this function's, defines "today".
    """
    async with connection_lock(conn):
        async with conn.execute(
            "SELECT COUNT(*) FROM variant_calls WHERE guild_id = ? AND call_day = ?",
            (guild_id, day),
        ) as cursor:
            row = await cursor.fetchone()
    return int(row[0]) if row else 0


async def try_acquire_variant_call_slot(
    conn: aiosqlite.Connection,
    *,
    guild_id: int,
    fact_id: int,
    daily_cap: int,
    now: datetime,
) -> VariantCallAttempt:
    """Atomically take one slot from the guild's daily variant-generation budget.

    Call this once a fact is known to exist and *before* the generation call it
    authorizes -- the same ordering, for the same reason, as every other
    ledger in this project. A slot is recorded when it is claimed, not when the
    work it authorizes succeeds, so a crash or an API failure downstream spends
    the slot instead of quietly refunding it. Without that direction, a
    reliably-failing model would earn unlimited retries.

    Never raises on a normal refusal: being out of budget is an expected
    outcome, not an error. A daily_cap of 0 is valid and means "never generate
    variants", which leaves fact creation itself fully working.

    Requires a timezone-aware `now`, injected rather than read from the clock
    here, so the daily boundary is testable at the exact moment it matters.
    """
    if not 0 <= daily_cap <= MAX_DAILY_CAP:
        raise ValueError(f"daily_cap must be between 0 and {MAX_DAILY_CAP}, got {daily_cap}")
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
                _ACQUIRE_VARIANT_SLOT_SQL,
                (guild_id, fact_id, called_at, day, guild_id, day, daily_cap),
            )
            if cursor.rowcount == 1:
                await conn.commit()
                async with conn.execute(
                    "SELECT COUNT(*) FROM variant_calls WHERE guild_id = ? AND call_day = ?",
                    (guild_id, day),
                ) as count_cursor:
                    row = await count_cursor.fetchone()
                return VariantCallAttempt(
                    outcome=VariantCallOutcome.GRANTED,
                    daily_count=int(row[0]) if row else 1,
                    daily_cap=daily_cap,
                )

            # The INSERT's own WHERE clause refused it: the cap is full. Read
            # the count back for the trail, from the same transaction that just
            # declined to add to it.
            await conn.rollback()
            async with conn.execute(
                "SELECT COUNT(*) FROM variant_calls WHERE guild_id = ? AND call_day = ?",
                (guild_id, day),
            ) as count_cursor:
                row = await count_cursor.fetchone()
            return VariantCallAttempt(
                outcome=VariantCallOutcome.DAILY_CAP_REACHED,
                daily_count=int(row[0]) if row else 0,
                daily_cap=daily_cap,
            )
        except BaseException:
            await conn.rollback()
            raise
