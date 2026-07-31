"""Durable budget protection for the supersession-judgment call: the per-guild daily cap.

The third instance of a shape this project now has twice already
(aura.db.proactive_state, aura.db.extraction_state), and built as a twin of
those on purpose rather than as a variation on them: one append-only ledger, one
guarded INSERT that re-checks the cap at write time, a stored UTC day key, no
in-memory state. A reader who has understood either of the other two has
understood this one, and the differences that do exist are stated here rather
than left to be noticed.

**Why a third budget at all**, when this call can only ever fire downstream of a
distillation call that already spent one of extraction's slots: because a shared
number would leave neither call site with a bound of its own. A guild that
suddenly produced many dedup-flagged candidates would consume the extraction
budget that produces them, silently converting a judgment ceiling into an
extraction outage -- a failure mode neither operator nor code could distinguish
from "extraction stopped working". Every paid call site in Aura carries its own
independent cost safety net, and this is one.

**Two differences from aura.db.extraction_state, both deliberate:**

*The ledger points at what it bought.* extraction_calls records a message_count
because a distillation call covers a whole batch; a judgment call covers exactly
one staged candidate, so the row references it. The candidate is always staged
before the slot is claimed, and candidates are never deleted, so that reference
is never dangling and stays resolvable for as long as the evidence is worth
having.

*Refusal is cheap here, and that is by design.* When this cap binds, nothing is
dropped and nothing is lost: the candidate is already staged and still gets
reviewed, it simply carries Phase 3a-2's plain similarity hint instead of a
judgment. That is a materially softer failure than the extraction cap's (which
drops a batch), and it is why this cap can be set aggressively low -- including
to 0 -- without breaking anything.
"""
from __future__ import annotations

from datetime import datetime
from enum import StrEnum

import aiosqlite
from pydantic import BaseModel

from aura.db.connection import connection_lock, utc_day, utc_iso

# One statement that checks the cap and writes the row, so the check cannot be
# separated from the write even by a second process sharing the database file.
_ACQUIRE_JUDGEMENT_SLOT_SQL = """
INSERT INTO supersession_calls (guild_id, pending_fact_id, called_at, call_day)
SELECT ?, ?, ?, ?
WHERE (
        SELECT COUNT(*) FROM supersession_calls
        WHERE guild_id = ? AND call_day = ?
    ) < ?
"""

# Mirrors MAX_DAILY_CAP in aura.db.proactive_state and aura.db.extraction_state,
# for the same reason: the value is bound into SQL, and sqlite3 refuses a Python
# int that does not fit a signed 64-bit integer, so a value past this would
# raise on every candidate instead of being refused once where an operator can
# see it.
MAX_DAILY_CAP = 1_000_000


class SupersessionCallOutcome(StrEnum):
    """Whether a judgment call was allowed to spend, and why not if it wasn't."""

    GRANTED = "granted"
    DAILY_CAP_REACHED = "daily_cap_reached"


class SupersessionCallAttempt(BaseModel):
    """The result of one attempt to claim a judgment call, with the state behind it.

    Carries the numbers the decision was made on rather than just the verdict,
    so a log line can say "42 of 50 of today's budget is gone" without
    re-deriving state that has since moved on.
    """

    outcome: SupersessionCallOutcome
    # Calls spent on this UTC day INCLUDING this attempt when it was granted,
    # so the reading matches how the other two ledgers report their own.
    daily_count: int
    daily_cap: int

    @property
    def granted(self) -> bool:
        """Whether this attempt actually took a slot from the budget."""
        return self.outcome is SupersessionCallOutcome.GRANTED


async def count_supersession_calls_on(
    conn: aiosqlite.Connection, *, guild_id: int, day: str
) -> int:
    """Return how many judgment calls guild_id has already spent on a UTC day.

    Read-only. Takes the day as a string produced by utc_day so the caller's
    clock, not this function's, defines "today".
    """
    async with connection_lock(conn):
        async with conn.execute(
            "SELECT COUNT(*) FROM supersession_calls WHERE guild_id = ? AND call_day = ?",
            (guild_id, day),
        ) as cursor:
            row = await cursor.fetchone()
    return int(row[0]) if row else 0


async def try_acquire_supersession_call_slot(
    conn: aiosqlite.Connection,
    *,
    guild_id: int,
    pending_fact_id: int,
    daily_cap: int,
    now: datetime,
) -> SupersessionCallAttempt:
    """Atomically take one slot from the guild's daily judgment budget.

    Call this once a candidate is known to be worth judging and *before* the LLM
    call it authorizes -- the same ordering, for the same reason, as both other
    ledgers. A slot is recorded when it is claimed, not when the work it
    authorizes succeeds, so a crash or an API failure downstream spends the slot
    instead of quietly refunding it. Without that direction, a reliably-failing
    model would earn unlimited retries.

    Never raises on a normal refusal: being out of budget is an expected
    outcome, not an error. A daily_cap of 0 is valid and means "never judge",
    which leaves extraction fully working and every candidate still reviewable.

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
                _ACQUIRE_JUDGEMENT_SLOT_SQL,
                (guild_id, pending_fact_id, called_at, day, guild_id, day, daily_cap),
            )
            if cursor.rowcount == 1:
                await conn.commit()
                async with conn.execute(
                    "SELECT COUNT(*) FROM supersession_calls "
                    "WHERE guild_id = ? AND call_day = ?",
                    (guild_id, day),
                ) as count_cursor:
                    row = await count_cursor.fetchone()
                return SupersessionCallAttempt(
                    outcome=SupersessionCallOutcome.GRANTED,
                    daily_count=int(row[0]) if row else 1,
                    daily_cap=daily_cap,
                )

            # The INSERT's own WHERE clause refused it: the cap is full. Read
            # the count back for the trail, from the same transaction that just
            # declined to add to it.
            await conn.rollback()
            async with conn.execute(
                "SELECT COUNT(*) FROM supersession_calls WHERE guild_id = ? AND call_day = ?",
                (guild_id, day),
            ) as count_cursor:
                row = await count_cursor.fetchone()
            return SupersessionCallAttempt(
                outcome=SupersessionCallOutcome.DAILY_CAP_REACHED,
                daily_count=int(row[0]) if row else 0,
                daily_cap=daily_cap,
            )
        except BaseException:
            await conn.rollback()
            raise
