"""Tests for aura.db.supersession_state: the judgement call's daily cap.

Deliberately mirrors tests/test_extraction_state.py case for case, which itself
mirrors the cap half of tests/test_proactive_state.py. The phase brief asks for
this cap to be checked against "the same race-condition class as the existing
daily caps", and the honest way to answer that is to subject it to the same
tests rather than to argue from the resemblance of the code -- a third copy of a
pattern is exactly where a subtle divergence hides.

The race tests use real asyncio.gather, and the restart tests use a real file on
disk: an in-memory database cannot demonstrate durability, since closing it
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
from aura.db.pending_facts import FactCategory, stage_pending_fact
from aura.db.repository import init_schema
from aura.db.supersession_state import (
    MAX_DAILY_CAP,
    SupersessionCallOutcome,
    count_supersession_calls_on,
    try_acquire_supersession_call_slot,
)

GUILD_A = 100000000000000001
GUILD_B = 200000000000000002
CHANNEL_A = 500000000000000005

NOON = datetime(2026, 7, 31, 12, 0, 0, tzinfo=timezone.utc)

EMBEDDING = b"\x00" * 16


@pytest.fixture
async def conn():
    connection = await aiosqlite.connect(":memory:")
    await init_schema(connection)
    yield connection
    await connection.close()


async def _candidate(
    conn: aiosqlite.Connection, *, guild_id: int = GUILD_A, message_id: int = 1
) -> int:
    """Stage a real candidate, so the ledger's foreign key has something to point at."""
    staged = await stage_pending_fact(
        conn,
        guild_id=guild_id,
        channel_id=CHANNEL_A,
        message_id=message_id,
        content=f"A candidate sentence #{message_id}.",
        embedding=EMBEDDING,
        category=FactCategory.RULE,
    )
    assert staged is not None
    return staged.id


async def _acquire(
    conn: aiosqlite.Connection,
    *,
    guild_id: int = GUILD_A,
    pending_fact_id: int,
    daily_cap: int = 3,
    now: datetime = NOON,
):
    return await try_acquire_supersession_call_slot(
        conn,
        guild_id=guild_id,
        pending_fact_id=pending_fact_id,
        daily_cap=daily_cap,
        now=now,
    )


class TestAcquisition:
    async def test_the_first_call_of_the_day_is_granted(
        self, conn: aiosqlite.Connection
    ) -> None:
        attempt = await _acquire(conn, pending_fact_id=await _candidate(conn))
        assert attempt.granted
        assert attempt.outcome is SupersessionCallOutcome.GRANTED
        assert attempt.daily_count == 1
        assert attempt.daily_cap == 3

    async def test_the_count_includes_the_granted_call_itself(
        self, conn: aiosqlite.Connection
    ) -> None:
        for expected in (1, 2, 3):
            candidate = await _candidate(conn, message_id=expected)
            attempt = await _acquire(conn, pending_fact_id=candidate)
            assert attempt.daily_count == expected

    async def test_the_cap_refuses_once_it_is_reached(
        self, conn: aiosqlite.Connection
    ) -> None:
        for message_id in range(1, 4):
            candidate = await _candidate(conn, message_id=message_id)
            assert (await _acquire(conn, pending_fact_id=candidate)).granted

        refused = await _acquire(conn, pending_fact_id=await _candidate(conn, message_id=4))
        assert not refused.granted
        assert refused.outcome is SupersessionCallOutcome.DAILY_CAP_REACHED
        assert refused.daily_count == 3
        assert await count_supersession_calls_on(
            conn, guild_id=GUILD_A, day=utc_day(NOON)
        ) == 3

    async def test_a_zero_cap_is_a_valid_off_switch(
        self, conn: aiosqlite.Connection
    ) -> None:
        attempt = await _acquire(
            conn, pending_fact_id=await _candidate(conn), daily_cap=0
        )
        assert not attempt.granted
        assert await count_supersession_calls_on(
            conn, guild_id=GUILD_A, day=utc_day(NOON)
        ) == 0

    async def test_a_refusal_writes_nothing(self, conn: aiosqlite.Connection) -> None:
        await _acquire(conn, pending_fact_id=await _candidate(conn), daily_cap=0)
        async with conn.execute("SELECT COUNT(*) FROM supersession_calls") as cursor:
            assert await cursor.fetchone() == (0,)

    async def test_the_cap_is_per_guild(self, conn: aiosqlite.Connection) -> None:
        for message_id in range(1, 4):
            candidate = await _candidate(conn, message_id=message_id)
            await _acquire(conn, pending_fact_id=candidate)
        assert not (
            await _acquire(conn, pending_fact_id=await _candidate(conn, message_id=4))
        ).granted

        other = await _candidate(conn, guild_id=GUILD_B, message_id=5)
        assert (await _acquire(conn, guild_id=GUILD_B, pending_fact_id=other)).granted

    async def test_the_cap_is_independent_of_the_extraction_cap(
        self, conn: aiosqlite.Connection
    ) -> None:
        # The reason this is a third budget rather than a share of extraction's:
        # spending every judgement slot must not consume the budget that
        # produces candidates in the first place.
        from aura.db.extraction_state import count_extraction_calls_on

        for message_id in range(1, 4):
            candidate = await _candidate(conn, message_id=message_id)
            await _acquire(conn, pending_fact_id=candidate)

        assert await count_extraction_calls_on(
            conn, guild_id=GUILD_A, day=utc_day(NOON)
        ) == 0

    async def test_the_cap_resets_on_the_next_utc_day(
        self, conn: aiosqlite.Connection
    ) -> None:
        for message_id in range(1, 4):
            candidate = await _candidate(conn, message_id=message_id)
            await _acquire(conn, pending_fact_id=candidate)
        assert not (
            await _acquire(conn, pending_fact_id=await _candidate(conn, message_id=4))
        ).granted

        fresh = await _acquire(
            conn,
            pending_fact_id=await _candidate(conn, message_id=5),
            now=NOON + timedelta(days=1),
        )
        assert fresh.granted
        assert fresh.daily_count == 1

    async def test_a_non_utc_now_is_filed_under_the_correct_utc_day(
        self, conn: aiosqlite.Connection
    ) -> None:
        # 01:30 in Kolkata is still the previous UTC day; filing it under the
        # local date would hand the guild a second daily budget.
        late = datetime(2026, 8, 1, 1, 30, tzinfo=ZoneInfo("Asia/Kolkata"))
        await _acquire(conn, pending_fact_id=await _candidate(conn), now=late)
        assert await count_supersession_calls_on(
            conn, guild_id=GUILD_A, day="2026-07-31"
        ) == 1
        assert await count_supersession_calls_on(
            conn, guild_id=GUILD_A, day="2026-08-01"
        ) == 0

    async def test_the_ledger_records_which_candidate_it_paid_for(
        self, conn: aiosqlite.Connection
    ) -> None:
        candidate = await _candidate(conn)
        await _acquire(conn, pending_fact_id=candidate)
        async with conn.execute("SELECT pending_fact_id FROM supersession_calls") as cursor:
            assert await cursor.fetchone() == (candidate,)


class TestInputValidation:
    async def test_a_naive_now_is_rejected(self, conn: aiosqlite.Connection) -> None:
        with pytest.raises(ValueError):
            await _acquire(
                conn,
                pending_fact_id=await _candidate(conn),
                now=datetime(2026, 7, 31, 12, 0),
            )

    async def test_an_absurd_cap_is_refused_where_an_operator_can_see_it(
        self, conn: aiosqlite.Connection
    ) -> None:
        with pytest.raises(ValueError):
            await _acquire(
                conn, pending_fact_id=await _candidate(conn), daily_cap=MAX_DAILY_CAP + 1
            )

    async def test_a_negative_cap_is_refused(self, conn: aiosqlite.Connection) -> None:
        with pytest.raises(ValueError):
            await _acquire(conn, pending_fact_id=await _candidate(conn), daily_cap=-1)

    async def test_a_slot_cannot_be_claimed_for_a_candidate_that_does_not_exist(
        self, conn: aiosqlite.Connection
    ) -> None:
        # The ledger's REFERENCES constraint, asserted rather than trusted: the
        # PRAGMA that enforces it is per-connection and silently does nothing if
        # it is ever missed, which would leave the spend trail pointing at
        # candidates nobody can look up.
        with pytest.raises(aiosqlite.IntegrityError):
            await _acquire(conn, pending_fact_id=999999)


class TestCapRaces:
    """The same race class both other caps are tested against, applied here.

    One sweep staging several flagged candidates in a row is the ordinary case,
    and a second process sharing the database file is always possible. "Read the
    count, decide, then write" would let both see room and both write; the
    guarded INSERT is what rules that out.
    """

    async def test_concurrent_acquisitions_never_exceed_the_cap(
        self, conn: aiosqlite.Connection
    ) -> None:
        candidates = [await _candidate(conn, message_id=i) for i in range(1, 21)]
        attempts = await asyncio.gather(
            *(_acquire(conn, pending_fact_id=c, daily_cap=3) for c in candidates)
        )

        assert len([a for a in attempts if a.granted]) == 3
        assert await count_supersession_calls_on(
            conn, guild_id=GUILD_A, day=utc_day(NOON)
        ) == 3

    async def test_concurrent_acquisitions_report_a_consistent_running_count(
        self, conn: aiosqlite.Connection
    ) -> None:
        candidates = [await _candidate(conn, message_id=i) for i in range(1, 31)]
        attempts = await asyncio.gather(
            *(_acquire(conn, pending_fact_id=c, daily_cap=5) for c in candidates)
        )
        granted_counts = sorted(a.daily_count for a in attempts if a.granted)
        # Each granted slot reports its own position in the day's budget, so the
        # five winners report 1..5 rather than all reporting the same number --
        # which is what a lost update would look like.
        assert granted_counts == [1, 2, 3, 4, 5]

    async def test_concurrent_acquisitions_against_a_zero_cap_grant_nothing(
        self, conn: aiosqlite.Connection
    ) -> None:
        candidates = [await _candidate(conn, message_id=i) for i in range(1, 21)]
        attempts = await asyncio.gather(
            *(_acquire(conn, pending_fact_id=c, daily_cap=0) for c in candidates)
        )
        assert not any(attempt.granted for attempt in attempts)

    async def test_two_guilds_racing_do_not_consume_each_others_budget(
        self, conn: aiosqlite.Connection
    ) -> None:
        pairs = [
            (guild, await _candidate(conn, guild_id=guild, message_id=index))
            for index, guild in enumerate(
                [GUILD_A] * 10 + [GUILD_B] * 10, start=1
            )
        ]
        attempts = await asyncio.gather(
            *(
                _acquire(conn, guild_id=guild, pending_fact_id=candidate, daily_cap=2)
                for guild, candidate in pairs
            )
        )
        assert len([a for a in attempts if a.granted]) == 4
        assert await count_supersession_calls_on(
            conn, guild_id=GUILD_A, day=utc_day(NOON)
        ) == 2
        assert await count_supersession_calls_on(
            conn, guild_id=GUILD_B, day=utc_day(NOON)
        ) == 2


class TestRestartDurability:
    async def test_a_spent_budget_survives_a_restart(self, tmp_path: Path) -> None:
        # A counter in a Python dict resets on every restart, so a crash loop
        # would hand out unlimited paid calls while the logs still claimed the
        # cap was enforced. A real file and a genuinely new connection are the
        # only way to show this is not what happens.
        database = tmp_path / "aura.db"

        first = await aiosqlite.connect(database)
        await init_schema(first)
        for message_id in range(1, 4):
            candidate = await _candidate(first, message_id=message_id)
            assert (await _acquire(first, pending_fact_id=candidate, daily_cap=3)).granted
        await first.close()

        second = await aiosqlite.connect(database)
        await init_schema(second)
        try:
            assert await count_supersession_calls_on(
                second, guild_id=GUILD_A, day=utc_day(NOON)
            ) == 3
            refused = await _acquire(
                second,
                pending_fact_id=await _candidate(second, message_id=4),
                daily_cap=3,
            )
            assert not refused.granted
            assert refused.outcome is SupersessionCallOutcome.DAILY_CAP_REACHED
        finally:
            await second.close()

    async def test_a_slot_claimed_before_a_crash_is_not_refunded(
        self, tmp_path: Path
    ) -> None:
        # The deliberate conservative direction: a crash between claiming a slot
        # and receiving the judgement spends the slot. For a spend limit, erring
        # toward "already spent" is the only safe way to err -- and the cost of
        # erring that way here is only that one candidate keeps the plain
        # similarity hint.
        database = tmp_path / "aura.db"

        first = await aiosqlite.connect(database)
        await init_schema(first)
        assert (
            await _acquire(first, pending_fact_id=await _candidate(first), daily_cap=2)
        ).granted
        await first.close()  # crash before the judgement call returns

        second = await aiosqlite.connect(database)
        await init_schema(second)
        try:
            retry = await _acquire(
                second,
                pending_fact_id=await _candidate(second, message_id=2),
                daily_cap=2,
            )
            assert retry.granted
            assert retry.daily_count == 2  # the crashed call still counts
            assert not (
                await _acquire(
                    second,
                    pending_fact_id=await _candidate(second, message_id=3),
                    daily_cap=2,
                )
            ).granted
        finally:
            await second.close()
