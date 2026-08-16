"""Tests for aura.db.digest_state: the digest's durable bookkeeping and its
atomic window claim.

This is where the sub-phase's central promises are verified, so the cases are
organised around the promises rather than around the functions:

  * a window is claimed at most once, even by two runners at the same instant;
  * a container restart resumes mid-interval with no recovery step, and a window
    missed during downtime is caught up exactly once -- never twice, never not
    at all;
  * a failed send releases its window instead of silently eating it.

The restart test uses a real file-backed database rather than an in-memory one,
because "survives the process going away" is exactly what an in-memory database
cannot demonstrate.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path

import aiosqlite
import pytest

from aura.db.connection import utc_iso
from aura.db.digest_state import (
    DigestRunOutcome,
    due_cutoff,
    get_digest_runs,
    last_covered_until,
    mark_digest_run_failed,
    try_claim_digest_run,
)
from aura.db.repository import init_schema
from aura.digest.intervals import DigestInterval

GUILD_A = 100000000000000001
GUILD_B = 200000000000000002
CHANNEL_A = 300000000000000003
WEEK = int(DigestInterval.WEEKLY)

# A fixed instant, so every window in these tests is stated in relation to one
# readable moment instead of to whatever the clock says while the suite runs.
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
    since: datetime,
    now: datetime,
    interval: int = WEEK,
    outcome: DigestRunOutcome = DigestRunOutcome.POSTED,
    new_facts: int = 1,
) -> int | None:
    return await try_claim_digest_run(
        conn,
        guild_id=guild_id,
        channel_id=CHANNEL_A,
        covered_from=utc_iso(since),
        covered_until=utc_iso(now),
        new_fact_count=new_facts,
        milestone_count=0,
        updated_fact_count=0,
        outcome=outcome,
        cutoff=due_cutoff(now, interval),
        now=now,
    )


class TestDueCutoff:
    def test_the_cutoff_is_one_interval_back(self) -> None:
        assert due_cutoff(NOW, WEEK) == utc_iso(NOW - timedelta(seconds=WEEK))

    def test_a_naive_datetime_is_refused(self) -> None:
        # Assuming UTC here would move every digest boundary by the host's
        # offset, which on a European host is up to two hours of a window.
        with pytest.raises(ValueError, match="timezone-aware"):
            due_cutoff(NOW.replace(tzinfo=None), WEEK)

    @pytest.mark.parametrize("interval", [0, -1])
    def test_a_non_positive_interval_is_refused(self, interval: int) -> None:
        # An interval of zero would make every guild permanently due.
        with pytest.raises(ValueError, match="positive"):
            due_cutoff(NOW, interval)

    def test_a_cutoff_from_another_timezone_still_compares_correctly(self) -> None:
        # utc_iso normalizes, so a caller in any offset produces a comparable
        # string rather than one that sorts against UTC timestamps wrongly.
        elsewhere = NOW.astimezone(timezone(timedelta(hours=5, minutes=30)))
        assert due_cutoff(elsewhere, WEEK) == due_cutoff(NOW, WEEK)


class TestClaiming:
    async def test_a_first_claim_succeeds_and_is_readable_back(
        self, conn: aiosqlite.Connection
    ) -> None:
        run_id = await _claim(conn, since=NOW - timedelta(days=7), now=NOW)

        assert run_id is not None
        runs = await get_digest_runs(conn, guild_id=GUILD_A, limit=5)
        assert len(runs) == 1
        assert runs[0].id == run_id
        assert runs[0].outcome is DigestRunOutcome.POSTED
        assert runs[0].covered_until == NOW
        assert runs[0].new_fact_count == 1

    async def test_a_claim_advances_the_boundary(self, conn: aiosqlite.Connection) -> None:
        assert await last_covered_until(conn, guild_id=GUILD_A) is None

        await _claim(conn, since=NOW - timedelta(days=7), now=NOW)

        assert await last_covered_until(conn, guild_id=GUILD_A) == utc_iso(NOW)

    async def test_a_second_claim_inside_the_same_interval_is_refused(
        self, conn: aiosqlite.Connection
    ) -> None:
        await _claim(conn, since=NOW - timedelta(days=7), now=NOW)

        # An hour later, well inside the weekly interval.
        later = NOW + timedelta(hours=1)
        assert await _claim(conn, since=NOW, now=later) is None
        assert len(await get_digest_runs(conn, guild_id=GUILD_A, limit=5)) == 1

    async def test_a_claim_after_the_interval_elapses_succeeds(
        self, conn: aiosqlite.Connection
    ) -> None:
        await _claim(conn, since=NOW - timedelta(days=7), now=NOW)

        next_week = NOW + timedelta(days=7, seconds=1)
        assert await _claim(conn, since=NOW, now=next_week) is not None

    async def test_an_empty_window_still_consumes_the_interval(
        self, conn: aiosqlite.Connection
    ) -> None:
        # A quiet week posts nothing but must not leave the guild permanently
        # due, or the next fact would trigger a digest the moment it lands.
        await _claim(
            conn,
            since=NOW - timedelta(days=7),
            now=NOW,
            outcome=DigestRunOutcome.SKIPPED_EMPTY,
            new_facts=0,
        )

        assert await last_covered_until(conn, guild_id=GUILD_A) == utc_iso(NOW)
        assert await _claim(conn, since=NOW, now=NOW + timedelta(hours=1)) is None

    async def test_a_reversed_window_is_refused_rather_than_written(
        self, conn: aiosqlite.Connection
    ) -> None:
        # Reachable only by a clock jumping backwards. A row covering negative
        # time would corrupt every later boundary read.
        with pytest.raises(ValueError, match="covered_until"):
            await try_claim_digest_run(
                conn,
                guild_id=GUILD_A,
                channel_id=CHANNEL_A,
                covered_from=utc_iso(NOW),
                covered_until=utc_iso(NOW - timedelta(hours=1)),
                new_fact_count=1,
                milestone_count=0,
                updated_fact_count=0,
                outcome=DigestRunOutcome.POSTED,
                cutoff=due_cutoff(NOW, WEEK),
                now=NOW,
            )
        assert await get_digest_runs(conn, guild_id=GUILD_A, limit=5) == []

    async def test_a_naive_now_is_refused(self, conn: aiosqlite.Connection) -> None:
        with pytest.raises(ValueError, match="timezone-aware"):
            await try_claim_digest_run(
                conn,
                guild_id=GUILD_A,
                channel_id=CHANNEL_A,
                covered_from=utc_iso(NOW - timedelta(days=7)),
                covered_until=utc_iso(NOW),
                new_fact_count=1,
                milestone_count=0,
                updated_fact_count=0,
                outcome=DigestRunOutcome.POSTED,
                cutoff=due_cutoff(NOW, WEEK),
                now=NOW.replace(tzinfo=None),
            )


class TestFailureReleasesTheWindow:
    async def test_a_failed_post_does_not_advance_the_boundary(
        self, conn: aiosqlite.Connection
    ) -> None:
        run_id = await _claim(conn, since=NOW - timedelta(days=7), now=NOW)
        assert run_id is not None

        await mark_digest_run_failed(conn, run_id=run_id)

        assert await last_covered_until(conn, guild_id=GUILD_A) is None

    async def test_a_failed_post_can_be_retried_immediately(
        self, conn: aiosqlite.Connection
    ) -> None:
        # The point of releasing the window: the next tick, an hour later, must
        # be able to claim the same window again rather than waiting a week.
        run_id = await _claim(conn, since=NOW - timedelta(days=7), now=NOW)
        assert run_id is not None
        await mark_digest_run_failed(conn, run_id=run_id)

        retry = await _claim(conn, since=NOW - timedelta(days=7), now=NOW + timedelta(hours=1))

        assert retry is not None
        runs = await get_digest_runs(conn, guild_id=GUILD_A, limit=5)
        assert [run.outcome for run in runs] == [
            DigestRunOutcome.POSTED,
            DigestRunOutcome.POST_FAILED,
        ]

    async def test_marking_a_skipped_run_failed_does_nothing(
        self, conn: aiosqlite.Connection
    ) -> None:
        # The guard exists so this write can only ever undo a claim its own
        # caller made, never rewrite a differently-decided outcome.
        run_id = await _claim(
            conn,
            since=NOW - timedelta(days=7),
            now=NOW,
            outcome=DigestRunOutcome.SKIPPED_EMPTY,
            new_facts=0,
        )
        assert run_id is not None

        await mark_digest_run_failed(conn, run_id=run_id)

        runs = await get_digest_runs(conn, guild_id=GUILD_A, limit=5)
        assert runs[0].outcome is DigestRunOutcome.SKIPPED_EMPTY

    async def test_marking_an_unknown_run_failed_is_a_no_op(
        self, conn: aiosqlite.Connection
    ) -> None:
        await mark_digest_run_failed(conn, run_id=99999)

        assert await get_digest_runs(conn, guild_id=GUILD_A, limit=5) == []

    async def test_marking_the_same_run_failed_twice_is_idempotent(
        self, conn: aiosqlite.Connection
    ) -> None:
        run_id = await _claim(conn, since=NOW - timedelta(days=7), now=NOW)
        assert run_id is not None

        await mark_digest_run_failed(conn, run_id=run_id)
        await mark_digest_run_failed(conn, run_id=run_id)

        runs = await get_digest_runs(conn, guild_id=GUILD_A, limit=5)
        assert [run.outcome for run in runs] == [DigestRunOutcome.POST_FAILED]


class TestDowntimeCatchUp:
    """The brief's central scheduling question, at the layer that decides it."""

    async def test_a_long_outage_produces_exactly_one_catch_up_claim(
        self, conn: aiosqlite.Connection
    ) -> None:
        # Last digest six weeks ago; the process has been down since. The first
        # tick after it returns must claim ONE window covering all six weeks --
        # not six, and not zero.
        last_run = NOW - timedelta(weeks=6)
        await _claim(conn, since=last_run - timedelta(days=7), now=last_run)

        first_tick = await _claim(conn, since=last_run, now=NOW)
        second_tick = await _claim(conn, since=NOW, now=NOW + timedelta(minutes=1))
        third_tick = await _claim(conn, since=NOW, now=NOW + timedelta(hours=2))

        assert first_tick is not None
        assert second_tick is None
        assert third_tick is None
        runs = await get_digest_runs(conn, guild_id=GUILD_A, limit=10)
        assert len(runs) == 2  # the pre-outage one and exactly one catch-up
        assert runs[0].covered_from == last_run
        assert runs[0].covered_until == NOW

    async def test_the_catch_up_window_covers_the_whole_outage(
        self, conn: aiosqlite.Connection
    ) -> None:
        # Nothing may fall between the last window and the catch-up: the new
        # window starts exactly where the old one ended.
        last_run = NOW - timedelta(weeks=3)
        await _claim(conn, since=last_run - timedelta(days=7), now=last_run)
        boundary = await last_covered_until(conn, guild_id=GUILD_A)
        assert boundary == utc_iso(last_run)

        await _claim(conn, since=last_run, now=NOW)

        runs = await get_digest_runs(conn, guild_id=GUILD_A, limit=1)
        assert utc_iso(runs[0].covered_from) == boundary


class TestRestartDurability:
    async def test_the_schedule_survives_the_process_going_away(
        self, tmp_path: Path
    ) -> None:
        # A file-backed database opened, written, closed and reopened -- the
        # closest a test gets to a container restart. Nothing may need
        # recovering, and the guild must not become due again just because the
        # process is new.
        database = tmp_path / "aura.db"

        first = await aiosqlite.connect(database)
        await init_schema(first)
        await _claim(first, since=NOW - timedelta(days=7), now=NOW)
        await first.close()

        second = await aiosqlite.connect(database)
        await init_schema(second)
        try:
            assert await last_covered_until(second, guild_id=GUILD_A) == utc_iso(NOW)
            # Mid-interval after the restart: still not due.
            assert await _claim(second, since=NOW, now=NOW + timedelta(hours=3)) is None
            # And due again once the interval genuinely elapses.
            assert (
                await _claim(second, since=NOW, now=NOW + timedelta(days=7, minutes=1))
                is not None
            )
        finally:
            await second.close()

    async def test_a_window_missed_during_downtime_is_caught_up_after_a_restart(
        self, tmp_path: Path
    ) -> None:
        database = tmp_path / "aura.db"
        went_down = NOW - timedelta(days=10)

        first = await aiosqlite.connect(database)
        await init_schema(first)
        await _claim(first, since=went_down - timedelta(days=7), now=went_down)
        await first.close()

        second = await aiosqlite.connect(database)
        await init_schema(second)
        try:
            caught_up = await _claim(second, since=went_down, now=NOW)
            repeated = await _claim(second, since=NOW, now=NOW + timedelta(minutes=5))
        finally:
            await second.close()

        assert caught_up is not None
        assert repeated is None


class TestConcurrency:
    async def test_two_simultaneous_claims_produce_exactly_one_run(
        self, conn: aiosqlite.Connection
    ) -> None:
        # The race the whole guarded INSERT exists for: two evaluations of the
        # same guild in flight, whose prize would be a duplicate public post.
        results = await asyncio.gather(
            *(_claim(conn, since=NOW - timedelta(days=7), now=NOW) for _ in range(8))
        )

        assert sum(result is not None for result in results) == 1
        assert len(await get_digest_runs(conn, guild_id=GUILD_A, limit=20)) == 1

    async def test_simultaneous_claims_for_different_guilds_all_succeed(
        self, conn: aiosqlite.Connection
    ) -> None:
        results = await asyncio.gather(
            _claim(conn, guild_id=GUILD_A, since=NOW - timedelta(days=7), now=NOW),
            _claim(conn, guild_id=GUILD_B, since=NOW - timedelta(days=7), now=NOW),
        )

        assert all(result is not None for result in results)


class TestGuildIsolation:
    async def test_one_guilds_run_does_not_satisfy_anothers_schedule(
        self, conn: aiosqlite.Connection
    ) -> None:
        await _claim(conn, guild_id=GUILD_A, since=NOW - timedelta(days=7), now=NOW)

        assert await last_covered_until(conn, guild_id=GUILD_B) is None
        assert await _claim(conn, guild_id=GUILD_B, since=NOW - timedelta(days=7), now=NOW) is not None

    async def test_runs_are_listed_per_guild(self, conn: aiosqlite.Connection) -> None:
        await _claim(conn, guild_id=GUILD_A, since=NOW - timedelta(days=7), now=NOW)
        await _claim(conn, guild_id=GUILD_B, since=NOW - timedelta(days=7), now=NOW)

        assert len(await get_digest_runs(conn, guild_id=GUILD_A, limit=10)) == 1
        assert len(await get_digest_runs(conn, guild_id=GUILD_B, limit=10)) == 1


class TestGetDigestRuns:
    async def test_runs_come_back_newest_first(self, conn: aiosqlite.Connection) -> None:
        await _claim(conn, since=NOW - timedelta(days=14), now=NOW - timedelta(days=7))
        await _claim(conn, since=NOW - timedelta(days=7), now=NOW)

        runs = await get_digest_runs(conn, guild_id=GUILD_A, limit=10)

        assert [run.covered_until for run in runs] == [NOW, NOW - timedelta(days=7)]

    async def test_a_negative_limit_is_refused_rather_than_meaning_unlimited(
        self, conn: aiosqlite.Connection
    ) -> None:
        # LIMIT -1 means "no limit" in SQLite, which would turn a guard into its
        # own opposite.
        with pytest.raises(ValueError, match="limit"):
            await get_digest_runs(conn, guild_id=GUILD_A, limit=-1)

    async def test_a_zero_limit_returns_nothing(self, conn: aiosqlite.Connection) -> None:
        await _claim(conn, since=NOW - timedelta(days=7), now=NOW)

        assert await get_digest_runs(conn, guild_id=GUILD_A, limit=0) == []
