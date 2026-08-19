"""Tests for aura.db.onboarding_state: the atomic claim behind onboarding's
two guarantees -- exactly one message per join, and a per-guild daily cap.

Mirrors tests/test_digest_state.py's organisation (grouped by the promise
being verified, not by function) because the underlying shape is the same
guarded-INSERT convention. Two things are genuinely different from the digest
and get their own classes: the claim key includes joined_at rather than just
guild+user (TestRejoin), and refusal has two distinct causes a caller needs to
tell apart (TestClaiming / TestDailyCap).
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import aiosqlite
import pytest

from aura.db.connection import utc_iso
from aura.db.onboarding_state import (
    OnboardingSendOutcome,
    count_onboarding_sends_on,
    try_claim_onboarding_send,
)
from aura.db.repository import init_schema

GUILD_A = 100000000000000001
GUILD_B = 200000000000000002
USER_A = 700000000000000007
USER_B = 800000000000000008

NOW = datetime(2026, 8, 16, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
async def conn():
    connection = await aiosqlite.connect(":memory:")
    await init_schema(connection)
    yield connection
    await connection.close()


async def _claim(
    conn: aiosqlite.Connection,
    *,
    guild_id: int = GUILD_A,
    user_id: int = USER_A,
    joined_at: datetime = NOW,
    fact_count: int = 3,
    daily_cap: int = 20,
    now: datetime = NOW,
) -> OnboardingSendOutcome:
    return await try_claim_onboarding_send(
        conn,
        guild_id=guild_id,
        user_id=user_id,
        joined_at=utc_iso(joined_at),
        fact_count=fact_count,
        daily_cap=daily_cap,
        now=now,
    )


class TestClaiming:
    async def test_a_first_claim_for_a_join_succeeds(self, conn: aiosqlite.Connection) -> None:
        outcome = await _claim(conn)

        assert outcome is OnboardingSendOutcome.CLAIMED
        assert await count_onboarding_sends_on(conn, guild_id=GUILD_A, day="2026-08-16") == 1

    async def test_the_same_join_claimed_twice_is_refused_the_second_time(
        self, conn: aiosqlite.Connection
    ) -> None:
        first = await _claim(conn)
        second = await _claim(conn)

        assert first is OnboardingSendOutcome.CLAIMED
        assert second is OnboardingSendOutcome.ALREADY_SENT
        assert await count_onboarding_sends_on(conn, guild_id=GUILD_A, day="2026-08-16") == 1

    async def test_a_naive_now_is_refused(self, conn: aiosqlite.Connection) -> None:
        with pytest.raises(ValueError, match="timezone-aware"):
            await try_claim_onboarding_send(
                conn,
                guild_id=GUILD_A,
                user_id=USER_A,
                joined_at=utc_iso(NOW),
                fact_count=1,
                daily_cap=20,
                now=NOW.replace(tzinfo=None),
            )

    @pytest.mark.parametrize("cap", [-1, 1_000_001])
    async def test_a_daily_cap_out_of_range_is_refused(
        self, conn: aiosqlite.Connection, cap: int
    ) -> None:
        with pytest.raises(ValueError, match="daily_cap"):
            await _claim(conn, daily_cap=cap)


class TestRejoin:
    """A member who leaves and rejoins gets a genuinely new joined_at from Discord.

    This is the deliberate product decision documented on onboarding_sends in
    schema.sql: a returning member is treated exactly as context-free as a new
    one, so the claim key includes joined_at rather than being (guild, user)
    alone.
    """

    async def test_a_different_joined_at_for_the_same_member_is_a_fresh_claim(
        self, conn: aiosqlite.Connection
    ) -> None:
        later = NOW + timedelta(days=30)
        first = await _claim(conn, joined_at=NOW, now=NOW)
        second = await _claim(conn, joined_at=later, now=later)

        assert first is OnboardingSendOutcome.CLAIMED
        assert second is OnboardingSendOutcome.CLAIMED
        assert await count_onboarding_sends_on(conn, guild_id=GUILD_A, day="2026-08-16") == 1
        assert await count_onboarding_sends_on(conn, guild_id=GUILD_A, day="2026-09-15") == 1

    async def test_two_different_members_joining_at_the_same_instant_both_claim(
        self, conn: aiosqlite.Connection
    ) -> None:
        first = await _claim(conn, user_id=USER_A, joined_at=NOW)
        second = await _claim(conn, user_id=USER_B, joined_at=NOW)

        assert first is OnboardingSendOutcome.CLAIMED
        assert second is OnboardingSendOutcome.CLAIMED


class TestDailyCap:
    async def test_a_claim_at_the_cap_is_the_last_one_granted(
        self, conn: aiosqlite.Connection
    ) -> None:
        for user in range(3):
            outcome = await _claim(conn, user_id=user, joined_at=NOW + timedelta(seconds=user), daily_cap=3)
            assert outcome is OnboardingSendOutcome.CLAIMED

        over = await _claim(conn, user_id=999, joined_at=NOW + timedelta(seconds=99), daily_cap=3)

        assert over is OnboardingSendOutcome.DAILY_CAP_REACHED

    async def test_a_cap_of_zero_refuses_every_join(self, conn: aiosqlite.Connection) -> None:
        outcome = await _claim(conn, daily_cap=0)

        assert outcome is OnboardingSendOutcome.DAILY_CAP_REACHED

    async def test_the_cap_resets_on_the_next_utc_day(self, conn: aiosqlite.Connection) -> None:
        await _claim(conn, user_id=1, joined_at=NOW, daily_cap=1)
        capped = await _claim(conn, user_id=2, joined_at=NOW + timedelta(minutes=1), daily_cap=1)
        assert capped is OnboardingSendOutcome.DAILY_CAP_REACHED

        tomorrow = NOW + timedelta(days=1)
        outcome = await _claim(conn, user_id=2, joined_at=tomorrow, daily_cap=1, now=tomorrow)

        assert outcome is OnboardingSendOutcome.CLAIMED

    async def test_the_cap_is_per_guild(self, conn: aiosqlite.Connection) -> None:
        await _claim(conn, guild_id=GUILD_A, user_id=1, joined_at=NOW, daily_cap=1)

        outcome = await _claim(conn, guild_id=GUILD_B, user_id=2, joined_at=NOW, daily_cap=1)

        assert outcome is OnboardingSendOutcome.CLAIMED

    async def test_a_mass_join_event_is_bounded_to_exactly_the_cap(
        self, conn: aiosqlite.Connection
    ) -> None:
        # A raid or a bot pile-on: many joins in quick succession, one guild.
        outcomes = [
            await _claim(conn, user_id=user, joined_at=NOW + timedelta(seconds=user), daily_cap=5)
            for user in range(50)
        ]

        claimed = sum(outcome is OnboardingSendOutcome.CLAIMED for outcome in outcomes)
        assert claimed == 5
        assert await count_onboarding_sends_on(conn, guild_id=GUILD_A, day="2026-08-16") == 5


class TestConcurrency:
    async def test_concurrent_joins_never_overshoot_the_cap(
        self, conn: aiosqlite.Connection
    ) -> None:
        # The race the guarded INSERT exists for: many join handlers evaluating
        # at once must never let the cap be exceeded.
        results = await asyncio.gather(
            *(
                _claim(conn, user_id=user, joined_at=NOW + timedelta(seconds=user), daily_cap=5)
                for user in range(20)
            )
        )

        claimed = sum(outcome is OnboardingSendOutcome.CLAIMED for outcome in results)
        assert claimed == 5
        assert await count_onboarding_sends_on(conn, guild_id=GUILD_A, day="2026-08-16") == 5

    async def test_concurrent_claims_of_the_exact_same_join_produce_exactly_one_claim(
        self, conn: aiosqlite.Connection
    ) -> None:
        # Discord redelivering an event after a resumed gateway session, or two
        # in-process handlers racing on it.
        results = await asyncio.gather(*(_claim(conn) for _ in range(8)))

        assert sum(outcome is OnboardingSendOutcome.CLAIMED for outcome in results) == 1
        assert sum(outcome is OnboardingSendOutcome.ALREADY_SENT for outcome in results) == 7

    async def test_concurrent_claims_across_guilds_do_not_interfere(
        self, conn: aiosqlite.Connection
    ) -> None:
        results = await asyncio.gather(
            _claim(conn, guild_id=GUILD_A, joined_at=NOW),
            _claim(conn, guild_id=GUILD_B, joined_at=NOW),
        )

        assert all(outcome is OnboardingSendOutcome.CLAIMED for outcome in results)


class TestCountOnboardingSendsOn:
    async def test_a_day_with_no_sends_counts_zero(self, conn: aiosqlite.Connection) -> None:
        assert await count_onboarding_sends_on(conn, guild_id=GUILD_A, day="2026-08-16") == 0

    async def test_counts_are_isolated_per_guild(self, conn: aiosqlite.Connection) -> None:
        await _claim(conn, guild_id=GUILD_A, joined_at=NOW)
        await _claim(conn, guild_id=GUILD_B, joined_at=NOW)

        assert await count_onboarding_sends_on(conn, guild_id=GUILD_A, day="2026-08-16") == 1
        assert await count_onboarding_sends_on(conn, guild_id=GUILD_B, day="2026-08-16") == 1
