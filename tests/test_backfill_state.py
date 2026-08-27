"""Tests for aura.db.backfill_state: the backfill daily cap, and its independence.

Subjected to the same suite as the four ledgers before it rather than trusted
because it looks like them -- CLAUDE.md's non-negotiable principle is explicit
that resemblance is not evidence. The concurrency tests use real asyncio.gather
against one connection, not sequential calls dressed up as concurrency, and the
durability tests use a real file-backed database and a genuinely new connection,
because an in-memory database cannot demonstrate durability: closing it loses the
data whether or not the design was durable, so the test would pass for the wrong
reason.

The test that matters most in this file is the last class. Everything else here
is a property this shape already had; "backfill and live extraction cannot touch
each other's budget" is the property this sub-phase adds, and the phase brief
asks for it by name.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import aiosqlite
import pytest

from aura.db.backfill_runs import start_backfill_run
from aura.db.backfill_state import (
    MAX_DAILY_CAP,
    BackfillCallOutcome,
    count_backfill_calls_on,
    try_acquire_backfill_call_slot,
)
from aura.db.connection import utc_day
from aura.db.extraction_state import (
    count_extraction_calls_on,
    try_acquire_extraction_call_slot,
)
from aura.db.repository import init_schema

GUILD_A = 100000000000000001
GUILD_B = 200000000000000002
CHANNEL_A = 300000000000000003
MODERATOR = 4242

NOW = datetime(2026, 8, 26, 12, 0, 0, tzinfo=timezone.utc)
UNTIL = 900000000000000000


@pytest.fixture
async def conn():
    connection = await aiosqlite.connect(":memory:")
    await init_schema(connection)
    yield connection
    await connection.close()


async def _run(conn: aiosqlite.Connection, *, guild_id: int = GUILD_A, channel_id: int = CHANNEL_A):
    """A real run row, so the ledger's REFERENCES constraint has something to point at."""
    return await start_backfill_run(
        conn,
        guild_id=guild_id,
        channel_id=channel_id,
        until_message_id=UNTIL,
        after_message_id=None,
        requested_by_id=MODERATOR,
        now=NOW,
    )


class TestAcquisition:
    async def test_a_granted_slot_is_recorded_and_counted(self, conn) -> None:
        run = await _run(conn)

        attempt = await try_acquire_backfill_call_slot(
            conn, guild_id=GUILD_A, run_id=run.id, message_count=7, daily_cap=30, now=NOW
        )

        assert attempt.granted
        assert attempt.daily_count == 1
        assert attempt.daily_cap == 30
        assert await count_backfill_calls_on(conn, guild_id=GUILD_A, day=utc_day(NOW)) == 1

    async def test_the_cap_refuses_once_it_is_full_and_writes_no_row(self, conn) -> None:
        run = await _run(conn)
        for _ in range(3):
            await try_acquire_backfill_call_slot(
                conn, guild_id=GUILD_A, run_id=run.id, message_count=1, daily_cap=3, now=NOW
            )

        refused = await try_acquire_backfill_call_slot(
            conn, guild_id=GUILD_A, run_id=run.id, message_count=1, daily_cap=3, now=NOW
        )

        assert not refused.granted
        assert refused.outcome is BackfillCallOutcome.DAILY_CAP_REACHED
        assert refused.daily_count == 3
        assert await count_backfill_calls_on(conn, guild_id=GUILD_A, day=utc_day(NOW)) == 3

    async def test_a_zero_cap_grants_nothing(self, conn) -> None:
        run = await _run(conn)

        attempt = await try_acquire_backfill_call_slot(
            conn, guild_id=GUILD_A, run_id=run.id, message_count=1, daily_cap=0, now=NOW
        )

        assert not attempt.granted
        assert await count_backfill_calls_on(conn, guild_id=GUILD_A, day=utc_day(NOW)) == 0

    async def test_the_budget_resets_on_the_next_utc_day(self, conn) -> None:
        run = await _run(conn)
        await try_acquire_backfill_call_slot(
            conn, guild_id=GUILD_A, run_id=run.id, message_count=1, daily_cap=1, now=NOW
        )

        tomorrow = NOW + timedelta(days=1)
        attempt = await try_acquire_backfill_call_slot(
            conn, guild_id=GUILD_A, run_id=run.id, message_count=1, daily_cap=1, now=tomorrow
        )

        assert attempt.granted
        assert attempt.daily_count == 1  # a fresh day, not a running total

    async def test_a_moment_just_before_midnight_utc_still_belongs_to_the_old_day(
        self, conn
    ) -> None:
        run = await _run(conn)
        late = datetime(2026, 8, 26, 23, 59, 59, 999999, tzinfo=timezone.utc)

        await try_acquire_backfill_call_slot(
            conn, guild_id=GUILD_A, run_id=run.id, message_count=1, daily_cap=1, now=late
        )
        refused = await try_acquire_backfill_call_slot(
            conn, guild_id=GUILD_A, run_id=run.id, message_count=1, daily_cap=1, now=late
        )

        assert not refused.granted
        assert await count_backfill_calls_on(conn, guild_id=GUILD_A, day="2026-08-26") == 1

    async def test_a_non_utc_now_is_filed_under_its_utc_day(self, conn) -> None:
        """01:30+05:30 is still the previous UTC day, and must count as one."""
        run = await _run(conn)
        india = timezone(timedelta(hours=5, minutes=30))
        local = datetime(2026, 8, 27, 1, 30, tzinfo=india)  # 2026-08-26 20:00 UTC

        await try_acquire_backfill_call_slot(
            conn, guild_id=GUILD_A, run_id=run.id, message_count=1, daily_cap=5, now=local
        )

        assert await count_backfill_calls_on(conn, guild_id=GUILD_A, day="2026-08-26") == 1
        assert await count_backfill_calls_on(conn, guild_id=GUILD_A, day="2026-08-27") == 0


class TestRejectedInput:
    async def test_a_naive_now_is_rejected(self, conn) -> None:
        run = await _run(conn)
        with pytest.raises(ValueError, match="timezone-aware"):
            await try_acquire_backfill_call_slot(
                conn,
                guild_id=GUILD_A,
                run_id=run.id,
                message_count=1,
                daily_cap=30,
                now=datetime(2026, 8, 26, 12, 0, 0),
            )

    async def test_an_oversized_cap_is_refused_rather_than_bound_into_sql(self, conn) -> None:
        run = await _run(conn)
        with pytest.raises(ValueError, match="between 0 and"):
            await try_acquire_backfill_call_slot(
                conn,
                guild_id=GUILD_A,
                run_id=run.id,
                message_count=1,
                daily_cap=MAX_DAILY_CAP + 1,
                now=NOW,
            )

    async def test_a_negative_cap_is_refused(self, conn) -> None:
        run = await _run(conn)
        with pytest.raises(ValueError, match="between 0 and"):
            await try_acquire_backfill_call_slot(
                conn, guild_id=GUILD_A, run_id=run.id, message_count=1, daily_cap=-1, now=NOW
            )

    async def test_a_negative_message_count_is_refused(self, conn) -> None:
        run = await _run(conn)
        with pytest.raises(ValueError, match="must not be negative"):
            await try_acquire_backfill_call_slot(
                conn, guild_id=GUILD_A, run_id=run.id, message_count=-1, daily_cap=30, now=NOW
            )

    async def test_the_ledgers_reference_constraint_actually_bites(self, conn) -> None:
        """PRAGMA foreign_keys is per-connection and silently does nothing if missed.

        The same check reports/phase-3a-3.txt Section 4 added for
        supersession_calls, for the same reason: a ledger row pointing at a run
        that never existed would make "what did this run spend" silently wrong,
        and nothing else in the codebase would notice.
        """
        with pytest.raises(aiosqlite.IntegrityError):
            await try_acquire_backfill_call_slot(
                conn, guild_id=GUILD_A, run_id=99999, message_count=1, daily_cap=30, now=NOW
            )


class TestConcurrency:
    async def test_twenty_racing_acquisitions_against_a_cap_of_three_grant_exactly_three(
        self, conn
    ) -> None:
        run = await _run(conn)

        attempts = await asyncio.gather(
            *(
                try_acquire_backfill_call_slot(
                    conn, guild_id=GUILD_A, run_id=run.id, message_count=1, daily_cap=3, now=NOW
                )
                for _ in range(20)
            )
        )

        granted = [attempt for attempt in attempts if attempt.granted]
        assert len(granted) == 3
        assert await count_backfill_calls_on(conn, guild_id=GUILD_A, day=utc_day(NOW)) == 3

    async def test_granted_slots_report_distinct_counts_which_is_what_rules_out_a_lost_update(
        self, conn
    ) -> None:
        """All five reporting daily_count=1 is exactly what a lost update looks like."""
        run = await _run(conn)

        attempts = await asyncio.gather(
            *(
                try_acquire_backfill_call_slot(
                    conn, guild_id=GUILD_A, run_id=run.id, message_count=1, daily_cap=5, now=NOW
                )
                for _ in range(5)
            )
        )

        counts = sorted(attempt.daily_count for attempt in attempts if attempt.granted)
        assert counts == [1, 2, 3, 4, 5]

    async def test_two_guilds_racing_consume_only_their_own_budgets(self, conn) -> None:
        run_a = await _run(conn)
        run_b = await _run(conn, guild_id=GUILD_B, channel_id=CHANNEL_A + 1)

        await asyncio.gather(
            *(
                try_acquire_backfill_call_slot(
                    conn, guild_id=GUILD_A, run_id=run_a.id, message_count=1, daily_cap=2, now=NOW
                )
                for _ in range(10)
            ),
            *(
                try_acquire_backfill_call_slot(
                    conn, guild_id=GUILD_B, run_id=run_b.id, message_count=1, daily_cap=2, now=NOW
                )
                for _ in range(10)
            ),
        )

        assert await count_backfill_calls_on(conn, guild_id=GUILD_A, day=utc_day(NOW)) == 2
        assert await count_backfill_calls_on(conn, guild_id=GUILD_B, day=utc_day(NOW)) == 2


class TestRestartDurability:
    async def test_a_spent_budget_survives_a_restart(self, tmp_path) -> None:
        path = tmp_path / "aura.db"

        first = await aiosqlite.connect(path)
        await init_schema(first)
        run = await _run(first)
        for _ in range(3):
            await try_acquire_backfill_call_slot(
                first, guild_id=GUILD_A, run_id=run.id, message_count=1, daily_cap=3, now=NOW
            )
        # How a dying container ends: the connection simply goes away.
        await first.close()

        second = await aiosqlite.connect(path)
        await init_schema(second)
        try:
            attempt = await try_acquire_backfill_call_slot(
                second, guild_id=GUILD_A, run_id=run.id, message_count=1, daily_cap=3, now=NOW
            )
            assert not attempt.granted
            assert attempt.daily_count == 3
        finally:
            await second.close()

    async def test_a_slot_claimed_before_a_crash_is_not_refunded(self, tmp_path) -> None:
        """The whole reason the slot is claimed BEFORE the call it authorizes."""
        path = tmp_path / "aura.db"

        first = await aiosqlite.connect(path)
        await init_schema(first)
        run = await _run(first)
        await try_acquire_backfill_call_slot(
            first, guild_id=GUILD_A, run_id=run.id, message_count=5, daily_cap=30, now=NOW
        )
        await first.close()  # "crash" before the distillation call finished

        second = await aiosqlite.connect(path)
        await init_schema(second)
        try:
            retry = await try_acquire_backfill_call_slot(
                second, guild_id=GUILD_A, run_id=run.id, message_count=5, daily_cap=30, now=NOW
            )
            assert retry.daily_count == 2, "the crashed attempt's slot was refunded"
        finally:
            await second.close()


class TestIndependenceFromExtraction:
    """The property this sub-phase adds, and the phase brief asks for by name.

    'Prüfe, ob BACKFILL_DAILY_CAP und EXTRACTION_DAILY_CAP sich gegenseitig
    nicht beeinflussen -- ein während des Backfills eintreffender Live-Kandidat
    muss weiterhin normal verarbeitet werden.'
    """

    async def test_exhausting_backfills_budget_leaves_extraction_untouched(self, conn) -> None:
        run = await _run(conn)
        for _ in range(30):
            await try_acquire_backfill_call_slot(
                conn, guild_id=GUILD_A, run_id=run.id, message_count=20, daily_cap=30, now=NOW
            )
        assert await count_backfill_calls_on(conn, guild_id=GUILD_A, day=utc_day(NOW)) == 30

        live = await try_acquire_extraction_call_slot(
            conn,
            guild_id=GUILD_A,
            channel_id=CHANNEL_A,
            message_count=3,
            daily_cap=50,
            now=NOW,
        )

        assert live.granted, "a live candidate was refused because a backfill spent its own cap"
        assert live.daily_count == 1

    async def test_exhausting_extractions_budget_leaves_backfill_untouched(self, conn) -> None:
        run = await _run(conn)
        for _ in range(50):
            await try_acquire_extraction_call_slot(
                conn,
                guild_id=GUILD_A,
                channel_id=CHANNEL_A,
                message_count=1,
                daily_cap=50,
                now=NOW,
            )

        attempt = await try_acquire_backfill_call_slot(
            conn, guild_id=GUILD_A, run_id=run.id, message_count=20, daily_cap=30, now=NOW
        )

        assert attempt.granted
        assert attempt.daily_count == 1

    async def test_the_two_ledgers_count_separately_under_concurrency(self, conn) -> None:
        """Interleaved live and backfill claims must not consume each other's slots."""
        run = await _run(conn)

        await asyncio.gather(
            *(
                try_acquire_backfill_call_slot(
                    conn, guild_id=GUILD_A, run_id=run.id, message_count=1, daily_cap=4, now=NOW
                )
                for _ in range(12)
            ),
            *(
                try_acquire_extraction_call_slot(
                    conn,
                    guild_id=GUILD_A,
                    channel_id=CHANNEL_A,
                    message_count=1,
                    daily_cap=6,
                    now=NOW,
                )
                for _ in range(12)
            ),
        )

        assert await count_backfill_calls_on(conn, guild_id=GUILD_A, day=utc_day(NOW)) == 4
        assert await count_extraction_calls_on(conn, guild_id=GUILD_A, day=utc_day(NOW)) == 6
