"""Onboarding's durable bookkeeping: which joins have already been sent a
summary, and the per-guild daily cap that keeps a mass-join event from
flooding the onboarding channel.

The sixth instance in this project of a shape that is now a convention rather
than a coincidence -- a guarded write whose WHERE clause re-checks the
decision at write time, no in-memory state -- combining two things that are
usually separate ledgers (proactive_escalations' cooldown-and-cap shape, and
digest_runs' claim-before-post shape) into one, because onboarding needs
exactly one guarantee each of them provides and nothing else:

**Exactly-once per join, not per member.** A member who leaves and rejoins
gets a genuinely new join -- a new `joined_at` from Discord -- and a fresh
onboarding message, which is the deliberate product decision documented on
`onboarding_sends` in schema.sql: a returning member is exactly as
context-free as a new one, and the same "here is what's currently true"
summary is exactly as useful the second time. What must NOT happen twice is
the *same* join being reported twice -- Discord redelivering an event after a
resumed gateway session, or two in-process handlers racing on it -- which is
why the claim is keyed on (guild_id, user_id, joined_at) rather than on
(guild_id, user_id) alone.

**A per-guild daily cap, because a join is not a query.** Every other daily
cap in this project bounds LLM spend; this one exists purely to bound how many
embeds a raid, a bot pile-on, or a legitimate invite spike can put into one
channel in one day. No LLM is involved in the base case (see
aura.onboarding.builder), so the cap is not a cost control -- it is the direct
answer to "what happens on a mass join", asked explicitly by this sub-phase's
brief rather than left to be discovered.
"""
from __future__ import annotations

from datetime import datetime
from enum import StrEnum

import aiosqlite

from aura.db.connection import connection_lock, utc_day, utc_iso

# Mirrors MAX_DAILY_CAP in aura.db.proactive_state: the value is bound
# directly into SQL, and an unreasonably large cap defeats the point of having
# one at all while costing nothing to reject up front.
MAX_DAILY_CAP = 1_000_000

# One statement that claims a join's onboarding send, atomically, against BOTH
# guards at once: no row already exists for this exact (guild, user, joined_at)
# triple, and today's count for this guild is still under the cap. Combining
# them in one INSERT ... SELECT ... WHERE, rather than a read-then-write pair
# under the connection lock (the shape aura.db.proactive_state uses), is what
# makes concurrent joins near the cap boundary unable to overshoot it -- the
# same reasoning try_claim_digest_run gives for a single guarded statement
# over a read-then-decide pair.
_CLAIM_SEND_SQL = """
INSERT INTO onboarding_sends
    (guild_id, user_id, joined_at, send_day, fact_count, sent_at)
SELECT ?, ?, ?, ?, ?, ?
WHERE NOT EXISTS (
        SELECT 1 FROM onboarding_sends
        WHERE guild_id = ? AND user_id = ? AND joined_at = ?
    )
    AND (
        SELECT COUNT(*) FROM onboarding_sends WHERE guild_id = ? AND send_day = ?
    ) < ?
"""


class OnboardingSendOutcome(StrEnum):
    """Why a join's onboarding send was granted or refused.

    Distinguishing the two refusal cases is purely for a clear log line --
    both mean "do not send" -- but an operator reading a container log
    benefits from knowing whether a guild's cap is undersized (DAILY_CAP_
    REACHED, repeatedly) or whether Discord redelivered an event
    (ALREADY_SENT, a single one-off).
    """

    CLAIMED = "claimed"
    ALREADY_SENT = "already_sent"
    DAILY_CAP_REACHED = "daily_cap_reached"


async def try_claim_onboarding_send(
    conn: aiosqlite.Connection,
    *,
    guild_id: int,
    user_id: int,
    joined_at: str,
    fact_count: int,
    daily_cap: int,
    now: datetime,
) -> OnboardingSendOutcome:
    """Atomically claim one member's onboarding send, or explain why it was refused.

    Call this the moment the message to send is fully decided (a resolved
    channel, non-empty content) and *before* it is actually posted -- the same
    ordering every claim-before-send ledger in this project uses, so that a
    crash between the claim and the send consumes the slot rather than letting
    a retry (there is none here; see aura.onboarding.listener) produce a
    second message for the same join.

    `joined_at` and `now` are both caller-supplied fixed-width UTC ISO-8601
    text and a timezone-aware datetime respectively, matching every other
    time-sensitive function in this project: the caller's single reading of
    Discord's `joined_at` and the clock drives both the uniqueness key and the
    daily-cap bucket, so the two cannot disagree with each other.
    """
    if now.tzinfo is None:
        raise ValueError(f"now must be a timezone-aware datetime, got {now!r}")
    if not 0 <= daily_cap <= MAX_DAILY_CAP:
        raise ValueError(f"daily_cap must be between 0 and {MAX_DAILY_CAP}, got {daily_cap}")

    day = utc_day(now)
    async with connection_lock(conn):
        try:
            cursor = await conn.execute(
                _CLAIM_SEND_SQL,
                (
                    guild_id,
                    user_id,
                    joined_at,
                    day,
                    fact_count,
                    utc_iso(now),
                    guild_id,
                    user_id,
                    joined_at,
                    guild_id,
                    day,
                    daily_cap,
                ),
            )
            if cursor.rowcount == 1:
                await conn.commit()
                return OnboardingSendOutcome.CLAIMED

            # Lost the claim -- find out which guard refused it, purely so the
            # caller can log something more useful than "no". Read-only, so it
            # is safe to run after rolling the (no-op) insert attempt back.
            await conn.rollback()
            async with conn.execute(
                "SELECT 1 FROM onboarding_sends WHERE guild_id = ? AND user_id = ? "
                "AND joined_at = ?",
                (guild_id, user_id, joined_at),
            ) as check:
                duplicate = await check.fetchone() is not None
        except BaseException:
            await conn.rollback()
            raise

    return (
        OnboardingSendOutcome.ALREADY_SENT
        if duplicate
        else OnboardingSendOutcome.DAILY_CAP_REACHED
    )


async def count_onboarding_sends_on(conn: aiosqlite.Connection, *, guild_id: int, day: str) -> int:
    """Return how many onboarding messages guild_id has already sent on a given UTC day.

    Read-only; not on the send path (the claim above counts atomically for
    itself), but useful for diagnostics and for tests asserting the cap holds.
    """
    async with connection_lock(conn):
        async with conn.execute(
            "SELECT COUNT(*) FROM onboarding_sends WHERE guild_id = ? AND send_day = ?",
            (guild_id, day),
        ) as cursor:
            row = await cursor.fetchone()
    return int(row[0]) if row else 0
