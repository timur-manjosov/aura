"""Durable budget protection for the distillation call: the per-guild daily cap.

A deliberate structural twin of aura.db.proactive_state, and the twinning is
the point rather than an accident of copying. Both spend limits answer the same
question against the same clock with the same failure modes, so they are built
the same way -- one append-only ledger, one guarded INSERT, a stored UTC day
key, no in-memory state -- and neither has to be re-reasoned about from scratch
when reading the other. Where this module differs from that one, the difference
is stated explicitly below rather than left to be inferred from absence.

**Durable, not in-memory.** Same reasoning verbatim: Aura runs in a container
with a restart policy, and a counter in a Python dict silently resets on every
restart, so a crash loop would hand out unlimited paid calls while every log
line still claimed the cap was being enforced. The ledger is the only state
there is.

**Race-condition safe, not check-then-set.** The sweeper can in principle have
two flushes in flight (different channels of the same guild), and a second
process sharing the database file can always exist, so "read the count, decide,
then write" is the textbook lost-update race. Acquisition is therefore ONE
guarded INSERT whose WHERE clause re-checks the cap at write time, which SQLite
evaluates in a single implicit transaction.

**Two deliberate differences from the proactive ledger:**

*No cooldown.* The batch window (see aura.db.extraction_queue) already bounds
one channel to at most one call per window, which is what a cooldown would have
been for. A second overlapping rate limit would mean two numbers that must be
kept consistent with each other in order for either to mean anything.

*No idempotency key.* proactive_escalations carries UNIQUE (channel_id,
message_id) because its trigger is a Discord event Discord may redeliver. This
ledger's trigger is Aura's own sweeper, so there is no redelivery to absorb; a
crash between claiming a slot and finishing the batch therefore spends the slot
and re-does the work. That is the same conservative direction the proactive
ledger chose -- for a spend limit, erring toward "already spent" is the only
safe way to err -- and the duplicate CANDIDATES such a retry would otherwise
produce are prevented one layer up, by pending_facts' own UNIQUE constraint.
"""
from __future__ import annotations

from datetime import datetime
from enum import StrEnum

import aiosqlite
from pydantic import BaseModel

from aura.db.connection import connection_lock, utc_day, utc_iso

# One statement that checks the cap and writes the row, for the reason this
# module's docstring gives. ON CONFLICT is absent rather than forgotten: this
# table carries no uniqueness constraint to conflict on (see the docstring).
_ACQUIRE_CALL_SLOT_SQL = """
INSERT INTO extraction_calls (guild_id, channel_id, message_count, called_at, call_day)
SELECT ?, ?, ?, ?, ?
WHERE (
        SELECT COUNT(*) FROM extraction_calls
        WHERE guild_id = ? AND call_day = ?
    ) < ?
"""

# Mirrors MAX_DAILY_CAP in aura.db.proactive_state, and exists for the same
# reason: the value is bound into SQL, and sqlite3 refuses a Python int that
# does not fit a signed 64-bit integer, so a value past this would raise on
# every sweep instead of being refused once where an operator can see it.
MAX_DAILY_CAP = 1_000_000


class ExtractionCallOutcome(StrEnum):
    """Whether a distillation call was allowed to spend, and why not if it wasn't."""

    GRANTED = "granted"
    DAILY_CAP_REACHED = "daily_cap_reached"


class ExtractionCallAttempt(BaseModel):
    """The result of one attempt to claim a distillation call, with the state behind it.

    Carries the numbers the decision was made on rather than just the verdict,
    so a log line can say "42 of 50 of today's budget is gone" without
    re-deriving state that has since moved on.
    """

    outcome: ExtractionCallOutcome
    # Calls spent on this UTC day INCLUDING this attempt when it was granted,
    # so the reading matches how the proactive ledger reports its own.
    daily_count: int
    daily_cap: int

    @property
    def granted(self) -> bool:
        """Whether this attempt actually took a slot from the budget."""
        return self.outcome is ExtractionCallOutcome.GRANTED


async def count_extraction_calls_on(
    conn: aiosqlite.Connection, *, guild_id: int, day: str
) -> int:
    """Return how many distillation calls guild_id has already spent on a UTC day.

    Read-only. Takes the day as a string produced by utc_day so the caller's
    clock, not this function's, defines "today".
    """
    async with connection_lock(conn):
        async with conn.execute(
            "SELECT COUNT(*) FROM extraction_calls WHERE guild_id = ? AND call_day = ?",
            (guild_id, day),
        ) as cursor:
            row = await cursor.fetchone()
    return int(row[0]) if row else 0


async def try_acquire_extraction_call_slot(
    conn: aiosqlite.Connection,
    *,
    guild_id: int,
    channel_id: int,
    message_count: int,
    daily_cap: int,
    now: datetime,
) -> ExtractionCallAttempt:
    """Atomically take one slot from the guild's daily distillation budget.

    Call this the moment a batch is known to be worth distilling and *before*
    the LLM call it authorizes -- the same ordering, for the same reason, that
    aura.proactive.gate uses. A slot is recorded when it is claimed, not when
    the work it authorizes succeeds, so a crash or an API failure downstream
    spends the slot instead of quietly refunding it.

    Never raises on a normal refusal: being out of budget is an expected
    outcome, not an error. A daily_cap of 0 is valid and means "never distill",
    which is a useful off switch rather than a misconfiguration.

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
                (guild_id, channel_id, message_count, called_at, day, guild_id, day, daily_cap),
            )
            if cursor.rowcount == 1:
                await conn.commit()
                async with conn.execute(
                    "SELECT COUNT(*) FROM extraction_calls WHERE guild_id = ? AND call_day = ?",
                    (guild_id, day),
                ) as count_cursor:
                    row = await count_cursor.fetchone()
                return ExtractionCallAttempt(
                    outcome=ExtractionCallOutcome.GRANTED,
                    daily_count=int(row[0]) if row else 1,
                    daily_cap=daily_cap,
                )

            # The INSERT's own WHERE clause refused it: the cap is full. Read
            # the count back for the trail, from the same transaction that just
            # declined to add to it.
            await conn.rollback()
            async with conn.execute(
                "SELECT COUNT(*) FROM extraction_calls WHERE guild_id = ? AND call_day = ?",
                (guild_id, day),
            ) as count_cursor:
                row = await count_cursor.fetchone()
            return ExtractionCallAttempt(
                outcome=ExtractionCallOutcome.DAILY_CAP_REACHED,
                daily_count=int(row[0]) if row else 0,
                daily_cap=daily_cap,
            )
        except BaseException:
            await conn.rollback()
            raise
