"""Tests for aura.db.backfill_runs: the run lifecycle, and the cursor's guarantees.

Three properties are load-bearing here and each has its own class:

  * at most ONE live run per channel, enforced by the database rather than by a
    prior read -- so two moderators racing produce one run and one refusal;
  * the cursor only ever moves FORWARD, and only while the run is still running
    -- so a pause landing during a paid call is not silently undone, and a stale
    or retried tick cannot drag a run back over history it has already paid for;
  * every state transition names what it believes the run currently is, so a
    worker and a moderator acting in the same instant produce one outcome rather
    than a last-writer-wins mess.

Restart durability is demonstrated on a real file with a genuinely new
connection, not asserted: an in-memory database loses its data on close whether
or not the design was durable, so the test would pass for the wrong reason.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import aiosqlite
import pytest

from aura.db.backfill_runs import (
    TERMINAL_STATES,
    BackfillAlreadyActiveError,
    BackfillState,
    advance_cursor,
    get_active_run,
    get_recent_runs,
    get_running_runs,
    set_run_state,
    start_backfill_run,
)
from aura.db.repository import init_schema

GUILD_A = 100000000000000001
GUILD_B = 200000000000000002
CHANNEL_A = 300000000000000003
CHANNEL_B = 400000000000000004
MODERATOR = 4242
OTHER_MODERATOR = 4343

NOW = datetime(2026, 8, 26, 12, 0, 0, tzinfo=timezone.utc)
LATER = NOW + timedelta(minutes=5)

# Plausible snowflakes: the lower bound is genuinely below the upper one, which
# is what start_backfill_run validates.
SINCE_ID = 700000000000000000
UNTIL_ID = 900000000000000000


@pytest.fixture
async def conn():
    connection = await aiosqlite.connect(":memory:")
    await init_schema(connection)
    yield connection
    await connection.close()


async def _start(
    conn: aiosqlite.Connection,
    *,
    guild_id: int = GUILD_A,
    channel_id: int = CHANNEL_A,
    after_message_id: int | None = None,
    now: datetime = NOW,
):
    return await start_backfill_run(
        conn,
        guild_id=guild_id,
        channel_id=channel_id,
        until_message_id=UNTIL_ID,
        after_message_id=after_message_id,
        requested_by_id=MODERATOR,
        now=now,
    )


class TestStarting:
    async def test_a_new_run_starts_running_with_no_cursor(self, conn) -> None:
        run = await _start(conn)

        assert run.state is BackfillState.RUNNING
        assert run.cursor_message_id is None
        assert run.cursor_message_at is None
        assert run.messages_scanned == 0
        assert run.candidates_staged == 0
        assert run.calls_spent == 0
        assert run.finished_at is None

    async def test_it_is_readable_back_exactly_as_written(self, conn) -> None:
        written = await _start(conn, after_message_id=SINCE_ID)

        read_back = await get_active_run(conn, channel_id=CHANNEL_A)

        assert read_back == written

    async def test_resume_after_id_prefers_the_cursor_over_the_since_bound(self, conn) -> None:
        """Getting this the wrong way round is what would re-read a channel from scratch."""
        run = await _start(conn, after_message_id=SINCE_ID)
        assert run.resume_after_id == SINCE_ID

        await advance_cursor(
            conn,
            run_id=run.id,
            cursor_message_id=SINCE_ID + 500,
            cursor_message_at=NOW,
            messages_scanned=10,
            candidates_staged=1,
            calls_spent=1,
            now=NOW,
        )
        advanced = await get_active_run(conn, channel_id=CHANNEL_A)
        assert advanced is not None
        assert advanced.resume_after_id == SINCE_ID + 500

    async def test_a_run_with_no_since_bound_resumes_from_nothing(self, conn) -> None:
        run = await _start(conn)
        assert run.resume_after_id is None

    async def test_a_reversed_range_is_refused(self, conn) -> None:
        with pytest.raises(ValueError, match="must be below"):
            await start_backfill_run(
                conn,
                guild_id=GUILD_A,
                channel_id=CHANNEL_A,
                until_message_id=SINCE_ID,
                after_message_id=UNTIL_ID,
                requested_by_id=MODERATOR,
                now=NOW,
            )

    async def test_an_empty_range_is_refused(self, conn) -> None:
        with pytest.raises(ValueError, match="must be below"):
            await start_backfill_run(
                conn,
                guild_id=GUILD_A,
                channel_id=CHANNEL_A,
                until_message_id=UNTIL_ID,
                after_message_id=UNTIL_ID,
                requested_by_id=MODERATOR,
                now=NOW,
            )

    async def test_a_naive_now_is_rejected(self, conn) -> None:
        with pytest.raises(ValueError, match="timezone-aware"):
            await _start(conn, now=datetime(2026, 8, 26, 12, 0, 0))  # type: ignore[arg-type]


class TestOneLiveRunPerChannel:
    async def test_starting_a_second_run_on_a_running_channel_is_refused(self, conn) -> None:
        first = await _start(conn)

        with pytest.raises(BackfillAlreadyActiveError) as caught:
            await _start(conn)

        assert caught.value.existing.id == first.id

    async def test_starting_a_second_run_on_a_paused_channel_is_refused(self, conn) -> None:
        first = await _start(conn)
        await set_run_state(
            conn,
            run_id=first.id,
            state=BackfillState.PAUSED,
            now=NOW,
            from_states=(BackfillState.RUNNING,),
        )

        with pytest.raises(BackfillAlreadyActiveError) as caught:
            await _start(conn)

        assert caught.value.existing.state is BackfillState.PAUSED

    @pytest.mark.parametrize("terminal", sorted(TERMINAL_STATES))
    async def test_a_finished_run_does_not_block_a_new_one(self, conn, terminal) -> None:
        first = await _start(conn)
        await set_run_state(
            conn,
            run_id=first.id,
            state=terminal,
            now=NOW,
            from_states=(BackfillState.RUNNING,),
        )

        second = await _start(conn)

        assert second.id != first.id
        assert second.state is BackfillState.RUNNING
        # Both rows survive: the table is append-only, so "this channel has been
        # backfilled before" stays answerable.
        assert len(await get_recent_runs(conn, guild_id=GUILD_A, limit=10)) == 2

    async def test_ten_concurrent_starts_produce_exactly_one_run(self, conn) -> None:
        """The partial unique index is the guarantee, not the read in front of it."""
        results = await asyncio.gather(
            *(_start(conn) for _ in range(10)), return_exceptions=True
        )

        winners = [result for result in results if not isinstance(result, BaseException)]
        losers = [
            result for result in results if isinstance(result, BackfillAlreadyActiveError)
        ]
        assert len(winners) == 1
        assert len(losers) == 9
        assert len(await get_recent_runs(conn, guild_id=GUILD_A, limit=20)) == 1

    async def test_two_channels_do_not_block_each_other(self, conn) -> None:
        await _start(conn, channel_id=CHANNEL_A)
        await _start(conn, channel_id=CHANNEL_B)

        assert len(await get_running_runs(conn)) == 2


class TestCursor:
    async def test_advancing_moves_the_cursor_and_adds_to_the_counters(self, conn) -> None:
        run = await _start(conn)

        moved = await advance_cursor(
            conn,
            run_id=run.id,
            cursor_message_id=800000000000000000,
            cursor_message_at=NOW,
            messages_scanned=100,
            candidates_staged=3,
            calls_spent=1,
            now=LATER,
        )

        assert moved
        after = await get_active_run(conn, channel_id=CHANNEL_A)
        assert after is not None
        assert after.cursor_message_id == 800000000000000000
        assert after.cursor_message_at == NOW
        assert (after.messages_scanned, after.candidates_staged, after.calls_spent) == (100, 3, 1)

    async def test_counters_accumulate_across_advances(self, conn) -> None:
        run = await _start(conn)
        for index in range(1, 4):
            await advance_cursor(
                conn,
                run_id=run.id,
                cursor_message_id=800000000000000000 + index,
                cursor_message_at=NOW,
                messages_scanned=10,
                candidates_staged=2,
                calls_spent=1,
                now=LATER,
            )

        after = await get_active_run(conn, channel_id=CHANNEL_A)
        assert after is not None
        assert (after.messages_scanned, after.candidates_staged, after.calls_spent) == (30, 6, 3)

    async def test_the_cursor_refuses_to_move_backwards(self, conn) -> None:
        """A retried or stale tick must not replay history the run already paid for."""
        run = await _start(conn)
        await advance_cursor(
            conn,
            run_id=run.id,
            cursor_message_id=800000000000000500,
            cursor_message_at=NOW,
            messages_scanned=100,
            candidates_staged=0,
            calls_spent=0,
            now=LATER,
        )

        moved = await advance_cursor(
            conn,
            run_id=run.id,
            cursor_message_id=800000000000000100,
            cursor_message_at=NOW,
            messages_scanned=100,
            candidates_staged=0,
            calls_spent=0,
            now=LATER,
        )

        assert not moved
        after = await get_active_run(conn, channel_id=CHANNEL_A)
        assert after is not None
        assert after.cursor_message_id == 800000000000000500
        assert after.messages_scanned == 100, "a refused advance still added to the counters"

    async def test_the_cursor_refuses_to_move_to_where_it_already_is(self, conn) -> None:
        run = await _start(conn)
        moved = [
            await advance_cursor(
                conn,
                run_id=run.id,
                cursor_message_id=800000000000000500,
                cursor_message_at=NOW,
                messages_scanned=100,
                candidates_staged=0,
                calls_spent=0,
                now=LATER,
            )
            for _ in range(2)
        ]
        assert moved == [True, False]

    async def test_the_first_advance_works_even_though_the_cursor_is_null(self, conn) -> None:
        """`NULL < anything` is NULL in SQL, which would refuse every first advance."""
        run = await _start(conn)

        assert await advance_cursor(
            conn,
            run_id=run.id,
            cursor_message_id=1,
            cursor_message_at=NOW,
            messages_scanned=1,
            candidates_staged=0,
            calls_spent=0,
            now=LATER,
        )

    @pytest.mark.parametrize(
        "state", [BackfillState.PAUSED, BackfillState.CANCELLED, BackfillState.COMPLETED]
    )
    async def test_a_run_that_is_not_running_cannot_be_advanced(self, conn, state) -> None:
        run = await _start(conn)
        await set_run_state(
            conn,
            run_id=run.id,
            state=state,
            now=NOW,
            from_states=(BackfillState.RUNNING,),
        )

        moved = await advance_cursor(
            conn,
            run_id=run.id,
            cursor_message_id=800000000000000000,
            cursor_message_at=NOW,
            messages_scanned=100,
            candidates_staged=5,
            calls_spent=1,
            now=LATER,
        )

        assert not moved

    async def test_a_pause_racing_an_advance_leaves_exactly_one_outcome(self, conn) -> None:
        run = await _start(conn)

        pause, advance = await asyncio.gather(
            set_run_state(
                conn,
                run_id=run.id,
                state=BackfillState.PAUSED,
                now=NOW,
                from_states=(BackfillState.RUNNING,),
            ),
            advance_cursor(
                conn,
                run_id=run.id,
                cursor_message_id=800000000000000000,
                cursor_message_at=NOW,
                messages_scanned=100,
                candidates_staged=0,
                calls_spent=1,
                now=LATER,
            ),
        )

        after = await get_active_run(conn, channel_id=CHANNEL_A)
        assert after is not None
        assert pause, "the moderator's pause must always win its own transition"
        assert after.state is BackfillState.PAUSED
        # Whichever order the two landed in, the cursor and the state agree:
        # either the advance got in before the pause, or it did not happen.
        assert (after.cursor_message_id is not None) == advance

    async def test_negative_counters_are_refused(self, conn) -> None:
        run = await _start(conn)
        with pytest.raises(ValueError, match="must not be negative"):
            await advance_cursor(
                conn,
                run_id=run.id,
                cursor_message_id=1,
                cursor_message_at=NOW,
                messages_scanned=-1,
                candidates_staged=0,
                calls_spent=0,
                now=LATER,
            )

    async def test_a_naive_now_is_rejected(self, conn) -> None:
        run = await _start(conn)
        with pytest.raises(ValueError, match="timezone-aware"):
            await advance_cursor(
                conn,
                run_id=run.id,
                cursor_message_id=1,
                cursor_message_at=NOW,
                messages_scanned=0,
                candidates_staged=0,
                calls_spent=0,
                now=datetime(2026, 8, 26, 12, 0, 0),  # type: ignore[arg-type]
            )


class TestStateTransitions:
    async def test_pausing_and_resuming_round_trips(self, conn) -> None:
        run = await _start(conn)

        assert await set_run_state(
            conn,
            run_id=run.id,
            state=BackfillState.PAUSED,
            now=NOW,
            from_states=(BackfillState.RUNNING,),
        )
        paused = await get_active_run(conn, channel_id=CHANNEL_A)
        assert paused is not None and paused.state is BackfillState.PAUSED

        assert await set_run_state(
            conn,
            run_id=run.id,
            state=BackfillState.RUNNING,
            now=LATER,
            from_states=(BackfillState.PAUSED,),
        )
        resumed = await get_active_run(conn, channel_id=CHANNEL_A)
        assert resumed is not None and resumed.state is BackfillState.RUNNING

    async def test_resuming_clears_finished_at_so_it_cannot_disagree_with_the_state(
        self, conn
    ) -> None:
        run = await _start(conn)
        await set_run_state(
            conn,
            run_id=run.id,
            state=BackfillState.CANCELLED,
            now=NOW,
            from_states=(BackfillState.RUNNING,),
        )
        cancelled = (await get_recent_runs(conn, guild_id=GUILD_A, limit=1))[0]
        assert cancelled.finished_at is not None

        await set_run_state(
            conn,
            run_id=run.id,
            state=BackfillState.RUNNING,
            now=LATER,
            from_states=(BackfillState.CANCELLED,),
        )
        revived = (await get_recent_runs(conn, guild_id=GUILD_A, limit=1))[0]
        assert revived.finished_at is None

    async def test_a_transition_from_the_wrong_state_changes_nothing(self, conn) -> None:
        run = await _start(conn)

        moved = await set_run_state(
            conn,
            run_id=run.id,
            state=BackfillState.RUNNING,
            now=NOW,
            from_states=(BackfillState.PAUSED,),
        )

        assert not moved
        still = await get_active_run(conn, channel_id=CHANNEL_A)
        assert still is not None and still.state is BackfillState.RUNNING

    async def test_two_concurrent_cancels_produce_exactly_one_winner(self, conn) -> None:
        run = await _start(conn)

        results = await asyncio.gather(
            *(
                set_run_state(
                    conn,
                    run_id=run.id,
                    state=BackfillState.CANCELLED,
                    now=NOW,
                    from_states=(BackfillState.RUNNING,),
                )
                for _ in range(5)
            )
        )

        assert sum(results) == 1

    async def test_a_worker_completing_cannot_overwrite_a_moderators_cancel(self, conn) -> None:
        run = await _start(conn)
        await set_run_state(
            conn,
            run_id=run.id,
            state=BackfillState.CANCELLED,
            now=NOW,
            from_states=(BackfillState.RUNNING,),
        )

        completed = await set_run_state(
            conn,
            run_id=run.id,
            state=BackfillState.COMPLETED,
            now=LATER,
            from_states=(BackfillState.RUNNING,),
        )

        assert not completed
        assert (await get_recent_runs(conn, guild_id=GUILD_A, limit=1))[
            0
        ].state is BackfillState.CANCELLED

    async def test_from_states_must_not_be_empty(self, conn) -> None:
        run = await _start(conn)
        with pytest.raises(ValueError, match="at least one state"):
            await set_run_state(
                conn, run_id=run.id, state=BackfillState.PAUSED, now=NOW, from_states=()
            )


class TestReads:
    async def test_only_running_runs_are_offered_to_the_worker(self, conn) -> None:
        running = await _start(conn, channel_id=CHANNEL_A)
        paused = await _start(conn, channel_id=CHANNEL_B)
        await set_run_state(
            conn,
            run_id=paused.id,
            state=BackfillState.PAUSED,
            now=NOW,
            from_states=(BackfillState.RUNNING,),
        )

        offered = await get_running_runs(conn)

        assert [run.id for run in offered] == [running.id]

    async def test_the_worker_gets_runs_oldest_first(self, conn) -> None:
        first = await _start(conn, channel_id=CHANNEL_A)
        second = await _start(conn, channel_id=CHANNEL_B)

        assert [run.id for run in await get_running_runs(conn)] == [first.id, second.id]

    async def test_recent_runs_come_back_newest_first(self, conn) -> None:
        first = await _start(conn, channel_id=CHANNEL_A)
        await set_run_state(
            conn,
            run_id=first.id,
            state=BackfillState.COMPLETED,
            now=NOW,
            from_states=(BackfillState.RUNNING,),
        )
        second = await _start(conn, channel_id=CHANNEL_A)

        assert [run.id for run in await get_recent_runs(conn, guild_id=GUILD_A, limit=10)] == [
            second.id,
            first.id,
        ]

    async def test_recent_runs_are_scoped_to_one_guild(self, conn) -> None:
        await _start(conn, guild_id=GUILD_A, channel_id=CHANNEL_A)
        await _start(conn, guild_id=GUILD_B, channel_id=CHANNEL_B)

        theirs = await get_recent_runs(conn, guild_id=GUILD_B, limit=10)

        assert [run.guild_id for run in theirs] == [GUILD_B]

    async def test_a_negative_limit_is_refused_rather_than_meaning_no_limit(self, conn) -> None:
        """LIMIT -1 in SQLite means 'no limit at all', which is the opposite of the ask."""
        with pytest.raises(ValueError, match="must not be negative"):
            await get_recent_runs(conn, guild_id=GUILD_A, limit=-1)

    async def test_a_channel_with_no_run_reads_back_as_none(self, conn) -> None:
        assert await get_active_run(conn, channel_id=CHANNEL_A) is None


class TestRestartDurability:
    async def test_a_cursor_survives_a_restart_and_resumes_from_exactly_where_it_was(
        self, tmp_path
    ) -> None:
        path = tmp_path / "aura.db"

        first = await aiosqlite.connect(path)
        await init_schema(first)
        run = await _start(first)
        await advance_cursor(
            first,
            run_id=run.id,
            cursor_message_id=800000000000000777,
            cursor_message_at=NOW,
            messages_scanned=250,
            candidates_staged=4,
            calls_spent=2,
            now=LATER,
        )
        await first.close()  # how a dying container ends

        second = await aiosqlite.connect(path)
        await init_schema(second)
        try:
            resumed = await get_active_run(second, channel_id=CHANNEL_A)
            assert resumed is not None
            assert resumed.state is BackfillState.RUNNING
            assert resumed.resume_after_id == 800000000000000777
            assert resumed.messages_scanned == 250
            assert resumed.calls_spent == 2
        finally:
            await second.close()

    async def test_the_one_live_run_rule_survives_a_restart(self, tmp_path) -> None:
        path = tmp_path / "aura.db"

        first = await aiosqlite.connect(path)
        await init_schema(first)
        await _start(first)
        await first.close()

        second = await aiosqlite.connect(path)
        await init_schema(second)
        try:
            with pytest.raises(BackfillAlreadyActiveError):
                await _start(second)
        finally:
            await second.close()
