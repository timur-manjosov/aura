"""Tests for aura.db.variant_state: variant generation's daily cap.

Deliberately mirrors tests/test_supersession_state.py case for case, which
itself mirrors the cap half of tests/test_extraction_state.py and
tests/test_proactive_state.py. The fourth twin of the same shape gets the
same tests, on the theory (already proven three times in this project) that a
fourth copy of a pattern is exactly where a subtle divergence hides.

The race tests use real asyncio.gather, and the restart tests use a real file
on disk: an in-memory database cannot demonstrate durability, since closing it
loses the data whether or not the design was durable.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import aiosqlite
import pytest

from aura.db.connection import utc_day
from aura.db.repository import init_schema
from aura.db.variant_state import (
    MAX_DAILY_CAP,
    VariantCallOutcome,
    count_variant_calls_on,
    try_acquire_variant_call_slot,
)
from aura.facts_service import add_fact

GUILD_A = 100000000000000001
GUILD_B = 200000000000000002

NOON = datetime(2026, 7, 31, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
async def conn():
    connection = await aiosqlite.connect(":memory:")
    await init_schema(connection)
    yield connection
    await connection.close()


async def _fact(
    conn: aiosqlite.Connection, embedding_model, *, guild_id: int = GUILD_A, message_id: int = 1
) -> int:
    """Create a real active fact, so the ledger's foreign key has something to point at."""
    fact = await add_fact(
        conn,
        embedding_model,
        guild_id=guild_id,
        channel_id=1,
        message_id=message_id,
        content=f"fact number {message_id}",
    )
    return fact.id


async def _acquire(
    conn: aiosqlite.Connection,
    *,
    guild_id: int = GUILD_A,
    fact_id: int,
    daily_cap: int = 3,
    now: datetime = NOON,
):
    return await try_acquire_variant_call_slot(
        conn, guild_id=guild_id, fact_id=fact_id, daily_cap=daily_cap, now=now
    )


class TestAcquisition:
    async def test_the_first_call_of_the_day_is_granted(
        self, conn: aiosqlite.Connection, embedding_model
    ) -> None:
        attempt = await _acquire(conn, fact_id=await _fact(conn, embedding_model))
        assert attempt.granted
        assert attempt.outcome is VariantCallOutcome.GRANTED
        assert attempt.daily_count == 1
        assert attempt.daily_cap == 3

    async def test_the_count_includes_the_granted_call_itself(
        self, conn: aiosqlite.Connection, embedding_model
    ) -> None:
        for expected in (1, 2, 3):
            fact_id = await _fact(conn, embedding_model, message_id=expected)
            attempt = await _acquire(conn, fact_id=fact_id)
            assert attempt.daily_count == expected

    async def test_the_cap_refuses_once_it_is_reached(
        self, conn: aiosqlite.Connection, embedding_model
    ) -> None:
        for message_id in range(1, 4):
            fact_id = await _fact(conn, embedding_model, message_id=message_id)
            assert (await _acquire(conn, fact_id=fact_id)).granted

        refused = await _acquire(conn, fact_id=await _fact(conn, embedding_model, message_id=4))
        assert not refused.granted
        assert refused.outcome is VariantCallOutcome.DAILY_CAP_REACHED
        assert refused.daily_count == 3
        assert await count_variant_calls_on(conn, guild_id=GUILD_A, day=utc_day(NOON)) == 3

    async def test_a_zero_cap_is_a_valid_off_switch(
        self, conn: aiosqlite.Connection, embedding_model
    ) -> None:
        attempt = await _acquire(conn, fact_id=await _fact(conn, embedding_model), daily_cap=0)
        assert not attempt.granted
        assert await count_variant_calls_on(conn, guild_id=GUILD_A, day=utc_day(NOON)) == 0

    async def test_a_refusal_writes_nothing(
        self, conn: aiosqlite.Connection, embedding_model
    ) -> None:
        await _acquire(conn, fact_id=await _fact(conn, embedding_model), daily_cap=0)
        async with conn.execute("SELECT COUNT(*) FROM variant_calls") as cursor:
            assert await cursor.fetchone() == (0,)

    async def test_the_cap_is_per_guild(
        self, conn: aiosqlite.Connection, embedding_model
    ) -> None:
        for message_id in range(1, 4):
            fact_id = await _fact(conn, embedding_model, message_id=message_id)
            await _acquire(conn, fact_id=fact_id)
        assert not (
            await _acquire(conn, fact_id=await _fact(conn, embedding_model, message_id=4))
        ).granted

        other = await _fact(conn, embedding_model, guild_id=GUILD_B, message_id=5)
        assert (await _acquire(conn, guild_id=GUILD_B, fact_id=other)).granted

    async def test_the_cap_resets_on_the_next_utc_day(
        self, conn: aiosqlite.Connection, embedding_model
    ) -> None:
        for message_id in range(1, 4):
            fact_id = await _fact(conn, embedding_model, message_id=message_id)
            await _acquire(conn, fact_id=fact_id)
        assert not (
            await _acquire(conn, fact_id=await _fact(conn, embedding_model, message_id=4))
        ).granted

        fresh = await _acquire(
            conn,
            fact_id=await _fact(conn, embedding_model, message_id=5),
            now=NOON + timedelta(days=1),
        )
        assert fresh.granted
        assert fresh.daily_count == 1

    async def test_a_non_utc_now_is_filed_under_the_correct_utc_day(
        self, conn: aiosqlite.Connection, embedding_model
    ) -> None:
        late = datetime(2026, 8, 1, 1, 30, tzinfo=ZoneInfo("Asia/Kolkata"))
        await _acquire(conn, fact_id=await _fact(conn, embedding_model), now=late)
        assert await count_variant_calls_on(conn, guild_id=GUILD_A, day="2026-07-31") == 1
        assert await count_variant_calls_on(conn, guild_id=GUILD_A, day="2026-08-01") == 0

    async def test_the_ledger_records_which_fact_it_paid_for(
        self, conn: aiosqlite.Connection, embedding_model
    ) -> None:
        fact_id = await _fact(conn, embedding_model)
        await _acquire(conn, fact_id=fact_id)
        async with conn.execute("SELECT fact_id FROM variant_calls") as cursor:
            assert await cursor.fetchone() == (fact_id,)


class TestInputValidation:
    async def test_a_naive_now_is_rejected(
        self, conn: aiosqlite.Connection, embedding_model
    ) -> None:
        with pytest.raises(ValueError):
            await _acquire(
                conn,
                fact_id=await _fact(conn, embedding_model),
                now=datetime(2026, 7, 31, 12, 0),
            )

    async def test_an_absurd_cap_is_refused_where_an_operator_can_see_it(
        self, conn: aiosqlite.Connection, embedding_model
    ) -> None:
        with pytest.raises(ValueError):
            await _acquire(
                conn, fact_id=await _fact(conn, embedding_model), daily_cap=MAX_DAILY_CAP + 1
            )

    async def test_a_negative_cap_is_refused(
        self, conn: aiosqlite.Connection, embedding_model
    ) -> None:
        with pytest.raises(ValueError):
            await _acquire(conn, fact_id=await _fact(conn, embedding_model), daily_cap=-1)

    async def test_a_slot_cannot_be_claimed_for_a_fact_that_does_not_exist(
        self, conn: aiosqlite.Connection
    ) -> None:
        with pytest.raises(aiosqlite.IntegrityError):
            await _acquire(conn, fact_id=999999)


class TestCapRaces:
    async def test_concurrent_acquisitions_never_exceed_the_cap(
        self, conn: aiosqlite.Connection, embedding_model
    ) -> None:
        facts = [await _fact(conn, embedding_model, message_id=i) for i in range(1, 21)]
        attempts = await asyncio.gather(
            *(_acquire(conn, fact_id=f, daily_cap=3) for f in facts)
        )

        assert len([a for a in attempts if a.granted]) == 3
        assert await count_variant_calls_on(conn, guild_id=GUILD_A, day=utc_day(NOON)) == 3

    async def test_concurrent_acquisitions_report_a_consistent_running_count(
        self, conn: aiosqlite.Connection, embedding_model
    ) -> None:
        facts = [await _fact(conn, embedding_model, message_id=i) for i in range(1, 31)]
        attempts = await asyncio.gather(
            *(_acquire(conn, fact_id=f, daily_cap=5) for f in facts)
        )
        granted_counts = sorted(a.daily_count for a in attempts if a.granted)
        assert granted_counts == [1, 2, 3, 4, 5]

    async def test_concurrent_acquisitions_against_a_zero_cap_grant_nothing(
        self, conn: aiosqlite.Connection, embedding_model
    ) -> None:
        facts = [await _fact(conn, embedding_model, message_id=i) for i in range(1, 21)]
        attempts = await asyncio.gather(
            *(_acquire(conn, fact_id=f, daily_cap=0) for f in facts)
        )
        assert not any(attempt.granted for attempt in attempts)

    async def test_two_guilds_racing_do_not_consume_each_others_budget(
        self, conn: aiosqlite.Connection, embedding_model
    ) -> None:
        pairs = [
            (guild, await _fact(conn, embedding_model, guild_id=guild, message_id=index))
            for index, guild in enumerate([GUILD_A] * 10 + [GUILD_B] * 10, start=1)
        ]
        attempts = await asyncio.gather(
            *(
                _acquire(conn, guild_id=guild, fact_id=fact_id, daily_cap=2)
                for guild, fact_id in pairs
            )
        )
        assert len([a for a in attempts if a.granted]) == 4
        assert await count_variant_calls_on(conn, guild_id=GUILD_A, day=utc_day(NOON)) == 2
        assert await count_variant_calls_on(conn, guild_id=GUILD_B, day=utc_day(NOON)) == 2


class TestRestartDurability:
    async def test_a_spent_budget_survives_a_restart(
        self, tmp_path: Path, embedding_model
    ) -> None:
        database = tmp_path / "aura.db"

        first = await aiosqlite.connect(database)
        await init_schema(first)
        for message_id in range(1, 4):
            fact_id = await _fact(first, embedding_model, message_id=message_id)
            assert (await _acquire(first, fact_id=fact_id, daily_cap=3)).granted
        await first.close()

        second = await aiosqlite.connect(database)
        await init_schema(second)
        try:
            assert await count_variant_calls_on(second, guild_id=GUILD_A, day=utc_day(NOON)) == 3
            refused = await _acquire(
                second, fact_id=await _fact(second, embedding_model, message_id=4), daily_cap=3
            )
            assert not refused.granted
            assert refused.outcome is VariantCallOutcome.DAILY_CAP_REACHED
        finally:
            await second.close()

    async def test_a_slot_claimed_before_a_crash_is_not_refunded(
        self, tmp_path: Path, embedding_model
    ) -> None:
        database = tmp_path / "aura.db"

        first = await aiosqlite.connect(database)
        await init_schema(first)
        assert (
            await _acquire(first, fact_id=await _fact(first, embedding_model), daily_cap=2)
        ).granted
        await first.close()  # crash before generation/audit return

        second = await aiosqlite.connect(database)
        await init_schema(second)
        try:
            retry = await _acquire(
                second, fact_id=await _fact(second, embedding_model, message_id=2), daily_cap=2
            )
            assert retry.granted
            assert retry.daily_count == 2  # the crashed call still counts
            assert not (
                await _acquire(
                    second, fact_id=await _fact(second, embedding_model, message_id=3), daily_cap=2
                )
            ).granted
        finally:
            await second.close()
