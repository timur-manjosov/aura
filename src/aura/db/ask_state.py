"""Durable budget protection for /aura-ask: the per-guild and per-member daily caps.

The sixth instance of the ledger shape aura.db.proactive_state introduced and
aura.db.{extraction,supersession,variant,backfill}_state repeat: one append-only
table (ask_calls, see schema.sql), one guarded INSERT that re-checks every
ceiling at write time, a stored UTC day key, no in-memory state. The UTC day
boundary and the reasons for it are documented once, in aura.db.proactive_state.

**Two ceilings, one statement.** Unlike its siblings, this ledger bounds two
things at once: the guild's daily total, and -- on the Free plan -- each
member's share of it, so one member cannot spend a whole server's allowance.
Both live in the same INSERT's WHERE clause, so neither can be checked
separately from the write, and a refusal never takes a slot from either.

**One answer, one slot.** A paid /aura-ask answer is a synthesis call followed
by an independent grounding check (see aura.grounding), always spent together.
One slot bounds both, for the reason aura.db.variant_state gives for its own
generation-plus-audit episode.

Imports only aura.db.connection. Which plan a guild is on, and so which caps
apply, is decided by the caller (aura.commands.ask) through aura.billing's plan
gate, never here.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Final

import aiosqlite
from pydantic import BaseModel

from aura.db.connection import connection_lock, utc_day, utc_iso

# One statement that checks both ceilings and writes the row, so the checks
# cannot be separated from the write even by a second process sharing the
# database file.
_ACQUIRE_ASK_SLOT_SQL: Final = """
INSERT INTO ask_calls (guild_id, user_id, created_at, call_day)
SELECT ?, ?, ?, ?
WHERE (
        SELECT COUNT(*) FROM ask_calls
        WHERE guild_id = ? AND call_day = ?
    ) < ?
  AND (
        SELECT COUNT(*) FROM ask_calls
        WHERE guild_id = ? AND call_day = ? AND user_id = ?
    ) < ?
"""

_COUNT_GUILD_SQL: Final = "SELECT COUNT(*) FROM ask_calls WHERE guild_id = ? AND call_day = ?"
_COUNT_USER_SQL: Final = (
    "SELECT COUNT(*) FROM ask_calls WHERE guild_id = ? AND call_day = ? AND user_id = ?"
)

# Mirrors MAX_DAILY_CAP in the other five ledgers, for the same reason: the
# value is bound into SQL, and sqlite3 refuses a Python int that does not fit a
# signed 64-bit integer, so a value past this would raise on every question
# instead of being refused once where an operator can see it.
MAX_DAILY_CAP: Final = 1_000_000


class AskCallOutcome(StrEnum):
    """Whether a paid /aura-ask answer was allowed to spend, and which ceiling refused it."""

    GRANTED = "granted"
    GUILD_CAP_REACHED = "guild_cap_reached"
    USER_CAP_REACHED = "user_cap_reached"


class AskCallAttempt(BaseModel):
    """The result of one attempt to claim an /aura-ask slot, with the state behind it.

    Attributes
    ----------
    outcome
        Whether the slot was granted, and which ceiling refused it if not.
        When both ceilings are full, the guild's is reported: it is the one
        that would still refuse every other member too.
    guild_count
        Paid answers this guild has spent on this UTC day, INCLUDING this
        attempt when it was granted.
    user_count
        The same, for this member in this guild.
    guild_cap
        The guild ceiling the count was measured against.
    user_cap
        The member ceiling, or None when the plan has no per-member ceiling.
    """

    outcome: AskCallOutcome
    guild_count: int
    user_count: int
    guild_cap: int
    user_cap: int | None

    @property
    def granted(self) -> bool:
        """Report whether this attempt actually took a slot from the budget.

        Returns
        -------
        bool
            True only for a GRANTED outcome.
        """
        return self.outcome is AskCallOutcome.GRANTED


async def _read_counts(
    conn: aiosqlite.Connection, *, guild_id: int, user_id: int, day: str
) -> tuple[int, int]:
    """Return (guild count, member count) for one day. The caller holds the lock."""
    async with conn.execute(_COUNT_GUILD_SQL, (guild_id, day)) as cursor:
        guild_row = await cursor.fetchone()
    async with conn.execute(_COUNT_USER_SQL, (guild_id, day, user_id)) as cursor:
        user_row = await cursor.fetchone()
    return (int(guild_row[0]) if guild_row else 0, int(user_row[0]) if user_row else 0)


async def count_ask_calls_on(
    conn: aiosqlite.Connection, *, guild_id: int, day: str, user_id: int | None = None
) -> int:
    """Return how many paid /aura-ask answers a guild, or one member in it, spent on a UTC day.

    Parameters
    ----------
    conn
        Open database connection.
    guild_id
        Guild whose ledger to count.
    day
        A UTC day key as produced by `utc_day`, so the caller's clock -- not
        this function's -- defines "today".
    user_id
        When given, count only this member's rows in that guild.

    Returns
    -------
    int
        Matching rows in `ask_calls`; 0 if there are none.

    Notes
    -----
    Read-only; takes no slot and changes nothing.
    """
    query: str
    params: tuple[int | str, ...]
    if user_id is None:
        query, params = _COUNT_GUILD_SQL, (guild_id, day)
    else:
        query, params = _COUNT_USER_SQL, (guild_id, day, user_id)
    async with connection_lock(conn):
        async with conn.execute(query, params) as cursor:
            row = await cursor.fetchone()
    return int(row[0]) if row else 0


async def try_acquire_ask_call_slot(
    conn: aiosqlite.Connection,
    *,
    guild_id: int,
    user_id: int,
    guild_cap: int,
    user_cap: int | None,
    now: datetime,
) -> AskCallAttempt:
    """Atomically take one slot from the guild's and the member's daily /aura-ask budget.

    Parameters
    ----------
    conn
        Open database connection.
    guild_id
        Guild whose budget to spend from.
    user_id
        The member asking, counted against `user_cap`.
    guild_cap
        Today's ceiling for the whole guild. 0 is valid and means "never spend".
    user_cap
        Today's ceiling for one member in this guild, or None for no
        per-member ceiling (the guild ceiling alone then applies). 0 is valid
        and means "never spend".
    now
        Timezone-aware moment; supplies both the timestamp and the UTC day
        key, so the two can never straddle midnight in opposite directions.

    Returns
    -------
    AskCallAttempt
        GRANTED with the post-write counts when a slot was taken; otherwise
        GUILD_CAP_REACHED or USER_CAP_REACHED with the current counts.

    Raises
    ------
    ValueError
        If `guild_cap` or `user_cap` is outside [0, MAX_DAILY_CAP], or `now`
        is naive.

    Notes
    -----
    Atomic against concurrent callers, including a second process sharing the
    database file: both ceilings are re-checked inside the INSERT's own WHERE
    clause, so there is no window between deciding and writing.

    Call this only once a paid answer is certain to be attempted -- after
    retrieval found at least one fact, immediately before synthesis -- and
    never refund it. A slot is recorded when it is claimed, not when the
    answer it authorizes succeeds, so a failed or rejected answer still spends
    it; without that direction, a reliably failing model would earn unlimited
    retries.

    The caps are the caller's, read at the moment of the call. A guild whose
    plan changes mid-day is measured against its new caps at once, and the
    rows it already spent today keep counting.
    """
    if not 0 <= guild_cap <= MAX_DAILY_CAP:
        raise ValueError(f"guild_cap must be between 0 and {MAX_DAILY_CAP}, got {guild_cap}")
    if user_cap is not None and not 0 <= user_cap <= MAX_DAILY_CAP:
        raise ValueError(f"user_cap must be between 0 and {MAX_DAILY_CAP}, got {user_cap}")
    if now.tzinfo is None:
        raise ValueError(f"now must be a timezone-aware datetime, got {now!r}")

    # With no per-member ceiling, a member can spend at most what the guild can;
    # binding the guild cap in its place keeps one statement for both plans.
    effective_user_cap = guild_cap if user_cap is None else user_cap

    # One `now` produces both the timestamp and the day key, so they can never
    # disagree.
    day = utc_day(now)
    created_at = utc_iso(now)

    async with connection_lock(conn):
        try:
            cursor = await conn.execute(
                _ACQUIRE_ASK_SLOT_SQL,
                (
                    guild_id,
                    user_id,
                    created_at,
                    day,
                    guild_id,
                    day,
                    guild_cap,
                    guild_id,
                    day,
                    user_id,
                    effective_user_cap,
                ),
            )
            granted = cursor.rowcount == 1
            if granted:
                await conn.commit()
            else:
                await conn.rollback()
            guild_count, user_count = await _read_counts(
                conn, guild_id=guild_id, user_id=user_id, day=day
            )
        except BaseException:
            await conn.rollback()
            raise

    if granted:
        outcome = AskCallOutcome.GRANTED
    elif guild_count >= guild_cap:
        outcome = AskCallOutcome.GUILD_CAP_REACHED
    else:
        outcome = AskCallOutcome.USER_CAP_REACHED
    return AskCallAttempt(
        outcome=outcome,
        guild_count=guild_count,
        user_count=user_count,
        guild_cap=guild_cap,
        user_cap=user_cap,
    )
