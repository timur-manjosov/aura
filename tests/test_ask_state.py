"""Tests for aura.db.ask_state: /aura-ask's per-guild and per-member daily caps.

Mirrors tests/test_variant_state.py case for case -- the sixth twin of the same
ledger shape gets the same tests -- plus what is new here: a second ceiling
(the member's) checked in the same statement as the first.

The race tests use real asyncio.gather. The cross-connection race opens several
connections to one database FILE, each with its own connection lock, so what
holds the cap there is the guarded INSERT itself and not the in-process lock.
The restart tests use a real file too: an in-memory database cannot
demonstrate durability.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import aiosqlite
import pytest

from aura.db.ask_state import (
    MAX_DAILY_CAP,
    AskCallAttempt,
    AskCallOutcome,
    count_ask_calls_on,
    try_acquire_ask_call_slot,
)
from aura.db.connection import utc_day
from aura.db.repository import init_schema

GUILD_A = 100000000000000001
GUILD_B = 200000000000000002
ALICE = 111
BOB = 222

NOON = datetime(2026, 10, 2, 12, 0, 0, tzinfo=UTC)
DAY = utc_day(NOON)


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
    user_id: int = ALICE,
    guild_cap: int = 3,
    user_cap: int | None = None,
    now: datetime = NOON,
) -> AskCallAttempt:
    return await try_acquire_ask_call_slot(
        conn, guild_id=guild_id, user_id=user_id, guild_cap=guild_cap, user_cap=user_cap, now=now
    )


async def _rows(conn: aiosqlite.Connection) -> int:
    async with conn.execute("SELECT COUNT(*) FROM ask_calls") as cursor:
        row = await cursor.fetchone()
    assert row is not None
    return int(row[0])


class TestSchema:
    async def test_the_table_has_the_expected_columns(self, conn: aiosqlite.Connection) -> None:
        async with conn.execute("PRAGMA table_info(ask_calls)") as cursor:
            columns = [row[1] for row in await cursor.fetchall()]
        assert columns == ["id", "guild_id", "user_id", "created_at", "call_day"]

    async def test_both_counts_are_served_by_an_index(self, conn: aiosqlite.Connection) -> None:
        for query in (
            "SELECT COUNT(*) FROM ask_calls WHERE guild_id = 1 AND call_day = 'd'",
            "SELECT COUNT(*) FROM ask_calls WHERE guild_id = 1 AND call_day = 'd' AND user_id = 2",
        ):
            async with conn.execute(f"EXPLAIN QUERY PLAN {query}") as cursor:
                plan = " ".join(str(row[3]) for row in await cursor.fetchall())
            assert "idx_ask_calls_guild_day_user" in plan

    async def test_creating_the_schema_twice_is_harmless(self, conn: aiosqlite.Connection) -> None:
        await _acquire(conn)
        await init_schema(conn)
        assert await _rows(conn) == 1


class TestGuildCap:
    async def test_the_first_call_of_the_day_is_granted(self, conn: aiosqlite.Connection) -> None:
        attempt = await _acquire(conn)
        assert attempt.granted
        assert attempt.outcome is AskCallOutcome.GRANTED
        assert (attempt.guild_count, attempt.user_count) == (1, 1)
        assert (attempt.guild_cap, attempt.user_cap) == (3, None)

    async def test_the_count_includes_the_granted_call_itself(
        self, conn: aiosqlite.Connection
    ) -> None:
        for expected in (1, 2, 3):
            assert (await _acquire(conn, user_id=expected)).guild_count == expected

    async def test_the_cap_refuses_once_it_is_reached(self, conn: aiosqlite.Connection) -> None:
        for user_id in (1, 2, 3):
            assert (await _acquire(conn, user_id=user_id)).granted

        refused = await _acquire(conn, user_id=4)
        assert not refused.granted
        assert refused.outcome is AskCallOutcome.GUILD_CAP_REACHED
        assert refused.guild_count == 3
        assert await count_ask_calls_on(conn, guild_id=GUILD_A, day=DAY) == 3

    async def test_a_zero_cap_is_a_valid_off_switch(self, conn: aiosqlite.Connection) -> None:
        attempt = await _acquire(conn, guild_cap=0)
        assert attempt.outcome is AskCallOutcome.GUILD_CAP_REACHED
        assert await _rows(conn) == 0

    async def test_the_cap_is_per_guild(self, conn: aiosqlite.Connection) -> None:
        for user_id in (1, 2, 3):
            await _acquire(conn, user_id=user_id)
        assert not (await _acquire(conn, user_id=4)).granted
        assert (await _acquire(conn, guild_id=GUILD_B, user_id=4)).granted

    async def test_the_cap_resets_at_utc_midnight(self, conn: aiosqlite.Connection) -> None:
        last_second = datetime(2026, 10, 2, 23, 59, 59, tzinfo=UTC)
        midnight = datetime(2026, 10, 3, 0, 0, 0, tzinfo=UTC)
        for user_id in (1, 2, 3):
            assert (await _acquire(conn, user_id=user_id, now=last_second)).granted
        assert not (await _acquire(conn, user_id=4, now=last_second)).granted

        fresh = await _acquire(conn, user_id=4, now=midnight)
        assert fresh.granted
        assert fresh.guild_count == 1

    async def test_a_non_utc_now_is_filed_under_the_utc_day(
        self, conn: aiosqlite.Connection
    ) -> None:
        late = datetime(2026, 10, 3, 1, 30, tzinfo=ZoneInfo("Asia/Kolkata"))
        await _acquire(conn, now=late)
        assert await count_ask_calls_on(conn, guild_id=GUILD_A, day="2026-10-02") == 1
        assert await count_ask_calls_on(conn, guild_id=GUILD_A, day="2026-10-03") == 0

    async def test_a_raised_cap_takes_effect_at_once_and_spent_rows_keep_counting(
        self, conn: aiosqlite.Connection
    ) -> None:
        # A plan change mid-day: Free (cap 3) to Pro (cap 5). The three answers
        # spent on Free count against Pro's cap, so exactly two more fit.
        for user_id in (1, 2, 3):
            await _acquire(conn, user_id=user_id, guild_cap=3)
        assert not (await _acquire(conn, user_id=4, guild_cap=3)).granted

        assert (await _acquire(conn, user_id=4, guild_cap=5)).guild_count == 4
        assert (await _acquire(conn, user_id=5, guild_cap=5)).guild_count == 5
        assert not (await _acquire(conn, user_id=6, guild_cap=5)).granted

    async def test_a_lowered_cap_refuses_at_once(self, conn: aiosqlite.Connection) -> None:
        # Pro to Free mid-day, after more than Free's cap was already spent.
        for user_id in range(1, 6):
            await _acquire(conn, user_id=user_id, guild_cap=25)
        refused = await _acquire(conn, user_id=9, guild_cap=3)
        assert refused.outcome is AskCallOutcome.GUILD_CAP_REACHED
        assert refused.guild_count == 5


class TestMemberCap:
    async def test_one_member_cannot_spend_more_than_their_share(
        self, conn: aiosqlite.Connection
    ) -> None:
        for _ in range(2):
            assert (await _acquire(conn, guild_cap=10, user_cap=2)).granted

        refused = await _acquire(conn, guild_cap=10, user_cap=2)
        assert refused.outcome is AskCallOutcome.USER_CAP_REACHED
        assert (refused.guild_count, refused.user_count) == (2, 2)

    async def test_another_member_keeps_their_share(self, conn: aiosqlite.Connection) -> None:
        for _ in range(2):
            await _acquire(conn, user_id=ALICE, guild_cap=10, user_cap=2)
        assert not (await _acquire(conn, user_id=ALICE, guild_cap=10, user_cap=2)).granted
        assert (await _acquire(conn, user_id=BOB, guild_cap=10, user_cap=2)).granted

    async def test_the_guild_cap_still_binds_below_the_member_cap(
        self, conn: aiosqlite.Connection
    ) -> None:
        await _acquire(conn, user_id=ALICE, guild_cap=2, user_cap=5)
        await _acquire(conn, user_id=BOB, guild_cap=2, user_cap=5)
        refused = await _acquire(conn, user_id=ALICE, guild_cap=2, user_cap=5)
        assert refused.outcome is AskCallOutcome.GUILD_CAP_REACHED

    async def test_when_both_are_full_the_guild_cap_is_reported(
        self, conn: aiosqlite.Connection
    ) -> None:
        await _acquire(conn, guild_cap=1, user_cap=1)
        refused = await _acquire(conn, guild_cap=1, user_cap=1)
        assert refused.outcome is AskCallOutcome.GUILD_CAP_REACHED

    async def test_a_zero_member_cap_refuses_without_writing(
        self, conn: aiosqlite.Connection
    ) -> None:
        refused = await _acquire(conn, guild_cap=10, user_cap=0)
        assert refused.outcome is AskCallOutcome.USER_CAP_REACHED
        assert await _rows(conn) == 0

    async def test_a_member_is_counted_per_guild(self, conn: aiosqlite.Connection) -> None:
        for _ in range(2):
            await _acquire(conn, guild_id=GUILD_A, guild_cap=10, user_cap=2)
        assert not (await _acquire(conn, guild_id=GUILD_A, guild_cap=10, user_cap=2)).granted
        # The same member, another server: a fresh share.
        assert (await _acquire(conn, guild_id=GUILD_B, guild_cap=10, user_cap=2)).granted

    async def test_no_member_cap_means_the_guild_cap_alone(
        self, conn: aiosqlite.Connection
    ) -> None:
        for _ in range(4):
            assert (await _acquire(conn, guild_cap=4, user_cap=None)).granted
        assert not (await _acquire(conn, guild_cap=4, user_cap=None)).granted

    async def test_the_member_count_reads_one_members_rows(
        self, conn: aiosqlite.Connection
    ) -> None:
        await _acquire(conn, user_id=ALICE)
        await _acquire(conn, user_id=BOB)
        await _acquire(conn, user_id=BOB)
        assert await count_ask_calls_on(conn, guild_id=GUILD_A, day=DAY, user_id=BOB) == 2
        assert await count_ask_calls_on(conn, guild_id=GUILD_A, day=DAY) == 3


class TestInputValidation:
    async def test_a_naive_now_is_rejected(self, conn: aiosqlite.Connection) -> None:
        with pytest.raises(ValueError):
            await _acquire(conn, now=datetime(2026, 10, 2, 12, 0))

    @pytest.mark.parametrize("cap", [-1, MAX_DAILY_CAP + 1])
    async def test_an_out_of_range_guild_cap_is_refused(
        self, conn: aiosqlite.Connection, cap: int
    ) -> None:
        with pytest.raises(ValueError):
            await _acquire(conn, guild_cap=cap)

    @pytest.mark.parametrize("cap", [-1, MAX_DAILY_CAP + 1])
    async def test_an_out_of_range_member_cap_is_refused(
        self, conn: aiosqlite.Connection, cap: int
    ) -> None:
        with pytest.raises(ValueError):
            await _acquire(conn, user_cap=cap)

    async def test_a_refused_input_writes_nothing(self, conn: aiosqlite.Connection) -> None:
        with pytest.raises(ValueError):
            await _acquire(conn, guild_cap=-1)
        assert await _rows(conn) == 0


class TestCapRaces:
    @pytest.mark.parametrize("_run", range(5))
    async def test_twenty_simultaneous_callers_get_exactly_the_cap(
        self, conn: aiosqlite.Connection, _run: int
    ) -> None:
        attempts = await asyncio.gather(
            *(_acquire(conn, user_id=user_id, guild_cap=7) for user_id in range(20))
        )
        assert len([a for a in attempts if a.granted]) == 7
        assert sorted(a.guild_count for a in attempts if a.granted) == list(range(1, 8))
        assert await _rows(conn) == 7

    @pytest.mark.parametrize("_run", range(5))
    async def test_one_member_racing_twenty_times_gets_exactly_their_share(
        self, conn: aiosqlite.Connection, _run: int
    ) -> None:
        attempts = await asyncio.gather(
            *(_acquire(conn, guild_cap=10, user_cap=5) for _ in range(20))
        )
        assert len([a for a in attempts if a.granted]) == 5
        refused = {a.outcome for a in attempts if not a.granted}
        assert refused == {AskCallOutcome.USER_CAP_REACHED}

    async def test_two_guilds_racing_do_not_consume_each_others_budget(
        self, conn: aiosqlite.Connection
    ) -> None:
        attempts = await asyncio.gather(
            *(
                _acquire(conn, guild_id=guild, user_id=index, guild_cap=2)
                for index, guild in enumerate([GUILD_A] * 10 + [GUILD_B] * 10)
            )
        )
        assert len([a for a in attempts if a.granted]) == 4
        assert await count_ask_calls_on(conn, guild_id=GUILD_A, day=DAY) == 2
        assert await count_ask_calls_on(conn, guild_id=GUILD_B, day=DAY) == 2

    @pytest.mark.parametrize("_run", range(3))
    async def test_separate_connections_racing_on_one_file_never_exceed_the_cap(
        self, tmp_path: Path, _run: int
    ) -> None:
        # Each connection has its own in-process lock, so nothing but the
        # guarded INSERT stands between these twenty callers and the cap --
        # the shape of a second process sharing the database file.
        database = tmp_path / "aura.db"
        setup = await aiosqlite.connect(database)
        await init_schema(setup)
        await setup.close()

        connections = [await aiosqlite.connect(database, timeout=30) for _ in range(20)]
        try:
            attempts = await asyncio.gather(
                *(
                    _acquire(connection, user_id=index, guild_cap=7)
                    for index, connection in enumerate(connections)
                )
            )
            assert len([a for a in attempts if a.granted]) == 7
            assert await count_ask_calls_on(connections[0], guild_id=GUILD_A, day=DAY) == 7
        finally:
            for connection in connections:
                await connection.close()


class TestRestartDurability:
    async def test_a_spent_budget_survives_a_restart(self, tmp_path: Path) -> None:
        database = tmp_path / "aura.db"

        first = await aiosqlite.connect(database)
        await init_schema(first)
        for _ in range(2):
            assert (await _acquire(first, guild_cap=10, user_cap=2)).granted
        await first.close()

        second = await aiosqlite.connect(database)
        await init_schema(second)
        try:
            refused = await _acquire(second, guild_cap=10, user_cap=2)
            assert refused.outcome is AskCallOutcome.USER_CAP_REACHED
            assert await count_ask_calls_on(second, guild_id=GUILD_A, day=DAY) == 2
        finally:
            await second.close()

    async def test_a_slot_claimed_before_a_crash_is_not_refunded(self, tmp_path: Path) -> None:
        database = tmp_path / "aura.db"

        first = await aiosqlite.connect(database)
        await init_schema(first)
        assert (await _acquire(first, guild_cap=2)).granted
        await first.close()  # crash before synthesis returns

        second = await aiosqlite.connect(database)
        await init_schema(second)
        try:
            retry = await _acquire(second, guild_cap=2)
            assert retry.granted
            assert retry.guild_count == 2  # the crashed call still counts
            assert not (await _acquire(second, guild_cap=2)).granted
        finally:
            await second.close()

    async def test_the_next_day_starts_empty_after_a_restart(self, tmp_path: Path) -> None:
        database = tmp_path / "aura.db"
        first = await aiosqlite.connect(database)
        await init_schema(first)
        await _acquire(first, guild_cap=1)
        await first.close()

        second = await aiosqlite.connect(database)
        await init_schema(second)
        try:
            assert (await _acquire(second, guild_cap=1, now=NOON + timedelta(days=1))).granted
        finally:
            await second.close()
