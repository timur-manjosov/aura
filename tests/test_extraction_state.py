"""Tests for aura.db.extraction_state: the distillation call's daily cap.

Deliberately mirrors the cap half of tests/test_proactive_state.py case for
case. The phase brief asks for this cap to be checked against "the same
race-condition class as the existing PROACTIVE_DAILY_CAP", and the honest way
to answer that is to subject it to the same tests rather than to argue from the
resemblance of the code.

The race tests use real asyncio.gather, and the restart tests use a real file
on disk -- an in-memory database cannot demonstrate durability, since closing
it loses the data whether or not the design was durable.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import aiosqlite
import pytest

from aura.db.connection import utc_day
from aura.db.extraction_state import (
    MAX_DAILY_CAP,
    ExtractionCallOutcome,
    count_extraction_calls_on,
    try_acquire_extraction_call_slot,
)
from aura.db.repository import init_schema

GUILD_A = 100000000000000001
GUILD_B = 200000000000000002
CHANNEL_A = 500000000000000005
CHANNEL_B = 600000000000000006

NOON = datetime(2026, 7, 30, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
async def conn():
    connection = await aiosqlite.connect(":memory:")
    await init_schema(connection)
    yield connection
    await connection.close()


async def _acquire(
    conn: aiosqlite.Connection,
    *,
    guild_id: int = GUILD_A,
    channel_id: int = CHANNEL_A,
    daily_cap: int = 3,
    now: datetime = NOON,
):
    return await try_acquire_extraction_call_slot(
        conn,
        guild_id=guild_id,
        channel_id=channel_id,
        message_count=5,
        daily_cap=daily_cap,
        now=now,
    )


class TestAcquisition:
    async def test_the_first_call_of_the_day_is_granted(
        self, conn: aiosqlite.Connection
    ) -> None:
        attempt = await _acquire(conn)
        assert attempt.granted
        assert attempt.outcome is ExtractionCallOutcome.GRANTED
        assert attempt.daily_count == 1
        assert attempt.daily_cap == 3

    async def test_the_count_includes_the_granted_call_itself(
        self, conn: aiosqlite.Connection
    ) -> None:
        # "3 of 50 of today's budget is gone" has to read correctly the moment
        # the third one is granted, not one call later.
        for expected in (1, 2, 3):
            attempt = await _acquire(conn)
            assert attempt.daily_count == expected

    async def test_the_cap_refuses_once_it_is_reached(
        self, conn: aiosqlite.Connection
    ) -> None:
        for _ in range(3):
            assert (await _acquire(conn)).granted

        refused = await _acquire(conn)
        assert not refused.granted
        assert refused.outcome is ExtractionCallOutcome.DAILY_CAP_REACHED
        assert refused.daily_count == 3
        assert await count_extraction_calls_on(conn, guild_id=GUILD_A, day=utc_day(NOON)) == 3

    async def test_a_zero_cap_is_a_valid_off_switch(
        self, conn: aiosqlite.Connection
    ) -> None:
        attempt = await _acquire(conn, daily_cap=0)
        assert not attempt.granted
        assert attempt.outcome is ExtractionCallOutcome.DAILY_CAP_REACHED
        assert await count_extraction_calls_on(conn, guild_id=GUILD_A, day=utc_day(NOON)) == 0

    async def test_a_refusal_writes_nothing(self, conn: aiosqlite.Connection) -> None:
        await _acquire(conn, daily_cap=0)
        async with conn.execute("SELECT COUNT(*) FROM extraction_calls") as cursor:
            assert await cursor.fetchone() == (0,)

    async def test_the_cap_is_per_guild(self, conn: aiosqlite.Connection) -> None:
        for _ in range(3):
            await _acquire(conn, guild_id=GUILD_A)
        assert not (await _acquire(conn, guild_id=GUILD_A)).granted
        # A different guild has its own untouched budget.
        assert (await _acquire(conn, guild_id=GUILD_B)).granted

    async def test_the_cap_is_shared_across_a_guilds_channels(
        self, conn: aiosqlite.Connection
    ) -> None:
        # Deliberately NOT per channel: the point of a guild-level cap is that
        # someone cannot multiply the budget by creating channels or threads.
        assert (await _acquire(conn, channel_id=CHANNEL_A, daily_cap=2)).granted
        assert (await _acquire(conn, channel_id=CHANNEL_B, daily_cap=2)).granted
        assert not (await _acquire(conn, channel_id=CHANNEL_B, daily_cap=2)).granted

    async def test_the_cap_resets_on_the_next_utc_day(
        self, conn: aiosqlite.Connection
    ) -> None:
        for _ in range(3):
            await _acquire(conn)
        assert not (await _acquire(conn)).granted

        tomorrow = NOON + timedelta(days=1)
        fresh = await _acquire(conn, now=tomorrow)
        assert fresh.granted
        assert fresh.daily_count == 1

    async def test_a_non_utc_now_is_filed_under_the_correct_utc_day(
        self, conn: aiosqlite.Connection
    ) -> None:
        # 01:30 in Kolkata is still the previous UTC day; filing it under the
        # local date would hand the guild a second daily budget.
        kolkata = ZoneInfo("Asia/Kolkata")
        late = datetime(2026, 7, 31, 1, 30, tzinfo=kolkata)
        await _acquire(conn, now=late)
        assert await count_extraction_calls_on(
            conn, guild_id=GUILD_A, day="2026-07-30"
        ) == 1
        assert await count_extraction_calls_on(
            conn, guild_id=GUILD_A, day="2026-07-31"
        ) == 0


class TestInputValidation:
    async def test_a_naive_now_is_rejected(self, conn: aiosqlite.Connection) -> None:
        with pytest.raises(ValueError):
            await _acquire(conn, now=datetime(2026, 7, 30, 12, 0))

    async def test_an_absurd_cap_is_refused_where_an_operator_can_see_it(
        self, conn: aiosqlite.Connection
    ) -> None:
        # The value is bound into SQL, and sqlite3 refuses an int that does not
        # fit a signed 64-bit integer -- so an unbounded value would raise on
        # every sweep instead of once, at a readable place.
        with pytest.raises(ValueError):
            await _acquire(conn, daily_cap=MAX_DAILY_CAP + 1)

    async def test_a_negative_cap_is_refused(self, conn: aiosqlite.Connection) -> None:
        with pytest.raises(ValueError):
            await _acquire(conn, daily_cap=-1)

    async def test_a_negative_message_count_is_refused(
        self, conn: aiosqlite.Connection
    ) -> None:
        with pytest.raises(ValueError):
            await try_acquire_extraction_call_slot(
                conn,
                guild_id=GUILD_A,
                channel_id=CHANNEL_A,
                message_count=-1,
                daily_cap=3,
                now=NOON,
            )


class TestCapRaces:
    """The same race class the proactive cap is tested against, applied here.

    Two flushes in flight for different channels of one guild is an ordinary
    state for the sweeper, and a second process sharing the database file is
    always possible. "Read the count, decide, then write" would let both see
    room and both write; the guarded INSERT is what rules that out.
    """

    async def test_concurrent_acquisitions_never_exceed_the_cap(
        self, conn: aiosqlite.Connection
    ) -> None:
        attempts = await asyncio.gather(
            *(_acquire(conn, channel_id=CHANNEL_A + i, daily_cap=3) for i in range(20))
        )

        granted = [attempt for attempt in attempts if attempt.granted]
        assert len(granted) == 3
        assert await count_extraction_calls_on(conn, guild_id=GUILD_A, day=utc_day(NOON)) == 3

    async def test_concurrent_acquisitions_report_a_consistent_running_count(
        self, conn: aiosqlite.Connection
    ) -> None:
        attempts = await asyncio.gather(
            *(_acquire(conn, channel_id=CHANNEL_A + i, daily_cap=5) for i in range(30))
        )
        granted_counts = sorted(a.daily_count for a in attempts if a.granted)
        # Each granted slot reports its own position in the day's budget, so
        # the five winners report 1..5 rather than all reporting the same
        # number -- which is what a lost update would look like.
        assert granted_counts == [1, 2, 3, 4, 5]

    async def test_concurrent_acquisitions_against_a_zero_cap_grant_nothing(
        self, conn: aiosqlite.Connection
    ) -> None:
        attempts = await asyncio.gather(
            *(_acquire(conn, channel_id=CHANNEL_A + i, daily_cap=0) for i in range(20))
        )
        assert not any(attempt.granted for attempt in attempts)
        assert await count_extraction_calls_on(conn, guild_id=GUILD_A, day=utc_day(NOON)) == 0

    async def test_two_guilds_racing_do_not_consume_each_others_budget(
        self, conn: aiosqlite.Connection
    ) -> None:
        attempts = await asyncio.gather(
            *(
                _acquire(conn, guild_id=guild, channel_id=CHANNEL_A + i, daily_cap=2)
                for guild in (GUILD_A, GUILD_B)
                for i in range(10)
            )
        )
        assert len([a for a in attempts if a.granted]) == 4
        assert await count_extraction_calls_on(conn, guild_id=GUILD_A, day=utc_day(NOON)) == 2
        assert await count_extraction_calls_on(conn, guild_id=GUILD_B, day=utc_day(NOON)) == 2


class TestRestartDurability:
    async def test_a_spent_budget_survives_a_restart(self, tmp_path: Path) -> None:
        # A counter in a Python dict resets on every restart, so a crash loop
        # would hand out unlimited paid calls while the logs still claimed the
        # cap was enforced. A real file and a genuinely new connection are the
        # only way to show this is not what happens.
        database = tmp_path / "aura.db"

        first = await aiosqlite.connect(database)
        await init_schema(first)
        for _ in range(3):
            assert (await _acquire(first, daily_cap=3)).granted
        await first.close()

        second = await aiosqlite.connect(database)
        await init_schema(second)
        try:
            assert await count_extraction_calls_on(
                second, guild_id=GUILD_A, day=utc_day(NOON)
            ) == 3
            refused = await _acquire(second, daily_cap=3)
            assert not refused.granted
            assert refused.outcome is ExtractionCallOutcome.DAILY_CAP_REACHED
        finally:
            await second.close()

    async def test_a_slot_claimed_before_a_crash_is_not_refunded(
        self, tmp_path: Path
    ) -> None:
        # The deliberate conservative direction: a crash between claiming a
        # slot and finishing the batch spends the slot. For a spend limit,
        # erring toward "already spent" is the only safe way to err.
        database = tmp_path / "aura.db"

        first = await aiosqlite.connect(database)
        await init_schema(first)
        assert (await _acquire(first, daily_cap=2)).granted
        await first.close()  # crash before the distillation call returns

        second = await aiosqlite.connect(database)
        await init_schema(second)
        try:
            retry = await _acquire(second, daily_cap=2)
            assert retry.granted
            assert retry.daily_count == 2  # the crashed call still counts
            assert not (await _acquire(second, daily_cap=2)).granted
        finally:
            await second.close()
