"""Tests for aura.db.proactive_signals: the Phase 2 decision-trail table.

Real in-memory SQLite throughout -- the behaviour under test (the uniqueness
constraint that absorbs a redelivered gateway event, LIMIT's treatment of a
negative value, guild isolation, nullable stage columns) is SQLite's, and
mocking it away would test nothing.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import aiosqlite
import pytest
from pydantic import ValidationError

from aura.db.proactive_signals import (
    DecisionTrail,
    GateVerdict,
    GracePeriodOutcome,
    OutdatedDiagnosticTableError,
    get_recent_signals,
    record_signal,
    update_grace_outcome,
    update_synthesis_outcome,
    verify_signal_schema,
)
from aura.db.repository import get_active_facts, init_schema

# A Phase 2a-2-shaped proactive_signals table: has `verdict`, but not the
# Phase 2a-3 synthesis columns. Used to prove the additive migration.
_PHASE_2A_2_TABLE = """
CREATE TABLE proactive_signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    guild_id INTEGER NOT NULL,
    channel_id INTEGER NOT NULL,
    message_id INTEGER NOT NULL,
    stage1_score REAL NOT NULL,
    stage1_passed INTEGER NOT NULL,
    stage2_top_score REAL,
    stage2_runner_up_score REAL,
    stage2_gap REAL,
    stage2_passed INTEGER,
    cooldown_seconds_remaining REAL,
    daily_count INTEGER,
    daily_cap INTEGER,
    verdict TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE (channel_id, message_id)
)
"""

# A Phase 2a-3-shaped proactive_signals table: has the synthesis columns, but
# not Phase 2b-1's grace_period_outcome. Used to prove that additive migration.
_PHASE_2A_3_TABLE = """
CREATE TABLE proactive_signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    guild_id INTEGER NOT NULL,
    channel_id INTEGER NOT NULL,
    message_id INTEGER NOT NULL,
    stage1_score REAL NOT NULL,
    stage1_passed INTEGER NOT NULL,
    stage2_top_score REAL,
    stage2_runner_up_score REAL,
    stage2_gap REAL,
    stage2_passed INTEGER,
    cooldown_seconds_remaining REAL,
    daily_count INTEGER,
    daily_cap INTEGER,
    synthesis_answers_question INTEGER,
    synthesis_posted INTEGER,
    verdict TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE (channel_id, message_id)
)
"""

GUILD_A = 100000000000000001
GUILD_B = 200000000000000002

# The trail Phase 2a-1's shape could not express: every stage filled in.
ELIGIBLE_TRAIL = DecisionTrail(
    verdict=GateVerdict.ELIGIBLE,
    stage1_score=0.12,
    stage1_passed=True,
    stage2_top_score=0.81,
    stage2_runner_up_score=0.33,
    stage2_gap=0.48,
    stage2_passed=True,
    cooldown_seconds_remaining=0.0,
    daily_count=3,
    daily_cap=20,
)

# The short-circuited trail: Stage 1 said no, so nothing past it was computed.
REJECTED_TRAIL = DecisionTrail(
    verdict=GateVerdict.STAGE1_REJECTED,
    stage1_score=-0.4,
    stage1_passed=False,
)


@pytest.fixture
async def conn():
    connection = await aiosqlite.connect(":memory:")
    await init_schema(connection)
    yield connection
    await connection.close()


async def _record(
    conn: aiosqlite.Connection,
    *,
    guild_id: int = GUILD_A,
    channel_id: int = 5,
    message_id: int = 1,
    decision: DecisionTrail = REJECTED_TRAIL,
) -> None:
    await record_signal(
        conn,
        guild_id=guild_id,
        channel_id=channel_id,
        message_id=message_id,
        decision=decision,
    )


class TestRecordAndRead:
    async def test_a_full_trail_reads_back_intact(self, conn: aiosqlite.Connection) -> None:
        await _record(conn, channel_id=42, message_id=99, decision=ELIGIBLE_TRAIL)

        [signal] = await get_recent_signals(conn, guild_id=GUILD_A, limit=10)

        assert signal.guild_id == GUILD_A
        assert signal.channel_id == 42
        assert signal.message_id == 99
        assert signal.verdict is GateVerdict.ELIGIBLE
        assert signal.stage1_score == pytest.approx(0.12)
        assert signal.stage1_passed is True
        assert signal.stage2_top_score == pytest.approx(0.81)
        assert signal.stage2_runner_up_score == pytest.approx(0.33)
        assert signal.stage2_gap == pytest.approx(0.48)
        assert signal.stage2_passed is True
        assert signal.cooldown_seconds_remaining == pytest.approx(0.0)
        assert signal.daily_count == 3
        assert signal.daily_cap == 20
        assert signal.created_at.tzinfo is not None  # timezone-aware, not naive

    async def test_an_unevaluated_stage_reads_back_as_none_not_as_zero(
        self, conn: aiosqlite.Connection
    ) -> None:
        # The distinction the whole trail depends on: "never evaluated" and
        # "evaluated and scored zero" call for opposite conclusions when
        # retuning a threshold, and 0.0 would conflate them.
        await _record(conn, decision=REJECTED_TRAIL)

        [signal] = await get_recent_signals(conn, guild_id=GUILD_A, limit=10)

        assert signal.stage1_passed is False
        assert signal.stage2_top_score is None
        assert signal.stage2_runner_up_score is None
        assert signal.stage2_gap is None
        assert signal.stage2_passed is None
        assert signal.cooldown_seconds_remaining is None
        assert signal.daily_count is None
        assert signal.daily_cap is None

    async def test_a_partially_evaluated_trail_round_trips(
        self, conn: aiosqlite.Connection
    ) -> None:
        # Stage 2 ran and failed, so the budget was never consulted.
        trail = DecisionTrail(
            verdict=GateVerdict.NO_MATCHING_FACT,
            stage1_score=0.2,
            stage1_passed=True,
            stage2_top_score=0.3,
            stage2_runner_up_score=None,
            stage2_gap=None,
            stage2_passed=False,
        )
        await _record(conn, decision=trail)

        [signal] = await get_recent_signals(conn, guild_id=GUILD_A, limit=10)

        assert signal.verdict is GateVerdict.NO_MATCHING_FACT
        assert signal.stage2_passed is False
        assert signal.daily_count is None

    async def test_empty_table_returns_an_empty_list_not_an_error(
        self, conn: aiosqlite.Connection
    ) -> None:
        assert await get_recent_signals(conn, guild_id=GUILD_A, limit=10) == []

    async def test_results_are_newest_first(self, conn: aiosqlite.Connection) -> None:
        for message_id in range(5):
            await _record(conn, message_id=message_id)

        signals = await get_recent_signals(conn, guild_id=GUILD_A, limit=10)

        assert [s.message_id for s in signals] == [4, 3, 2, 1, 0]

    async def test_signals_recorded_within_the_same_microsecond_still_order_deterministically(
        self, conn: aiosqlite.Connection
    ) -> None:
        # created_at can tie; insertion order cannot. Ordering by id rather
        # than timestamp is what makes "most recent" mean something here.
        for message_id in range(30):
            await _record(conn, message_id=message_id)

        signals = await get_recent_signals(conn, guild_id=GUILD_A, limit=30)

        assert [s.message_id for s in signals] == list(reversed(range(30)))

    async def test_limit_caps_the_number_of_rows_returned(
        self, conn: aiosqlite.Connection
    ) -> None:
        for message_id in range(10):
            await _record(conn, message_id=message_id)

        assert len(await get_recent_signals(conn, guild_id=GUILD_A, limit=3)) == 3

    async def test_limit_larger_than_the_table_returns_everything_available(
        self, conn: aiosqlite.Connection
    ) -> None:
        await _record(conn, message_id=1)

        assert len(await get_recent_signals(conn, guild_id=GUILD_A, limit=500)) == 1

    async def test_limit_of_zero_returns_nothing(self, conn: aiosqlite.Connection) -> None:
        await _record(conn, message_id=1)

        assert await get_recent_signals(conn, guild_id=GUILD_A, limit=0) == []

    async def test_negative_limit_is_rejected_rather_than_meaning_unlimited(
        self, conn: aiosqlite.Connection
    ) -> None:
        # SQLite reads LIMIT -1 as "no limit at all", so a caller that
        # arrived at a negative number by arithmetic would silently receive
        # the entire table instead of a page.
        for message_id in range(5):
            await _record(conn, message_id=message_id)

        with pytest.raises(ValueError):
            await get_recent_signals(conn, guild_id=GUILD_A, limit=-1)


class TestVerdicts:
    @pytest.mark.parametrize("verdict", list(GateVerdict))
    async def test_every_verdict_round_trips_through_the_database(
        self, conn: aiosqlite.Connection, verdict: GateVerdict
    ) -> None:
        # Stored as text, so a verdict added later must be readable back as
        # its enum member and not degrade into a bare string.
        trail = DecisionTrail(verdict=verdict, stage1_score=0.0, stage1_passed=True)
        await _record(conn, decision=trail)

        [signal] = await get_recent_signals(conn, guild_id=GUILD_A, limit=10)

        assert signal.verdict is verdict
        assert isinstance(signal.verdict, GateVerdict)

    async def test_only_the_eligible_verdict_counts_as_would_escalate(self) -> None:
        for verdict in GateVerdict:
            trail = DecisionTrail(verdict=verdict, stage1_score=0.0, stage1_passed=True)
            assert trail.would_escalate is (verdict is GateVerdict.ELIGIBLE)

    async def test_an_unknown_verdict_string_in_the_database_is_refused_not_guessed(
        self, conn: aiosqlite.Connection
    ) -> None:
        # Only reachable by hand-editing the database. Failing loudly beats
        # returning a row whose verdict silently means nothing.
        await conn.execute(
            """
            INSERT INTO proactive_signals
                (guild_id, channel_id, message_id, stage1_score, stage1_passed, verdict, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (GUILD_A, 1, 1, 0.0, 1, "something_invented", "2026-07-24T00:00:00.000000+00:00"),
        )
        await conn.commit()

        with pytest.raises(ValueError):
            await get_recent_signals(conn, guild_id=GUILD_A, limit=10)


class TestGuildIsolation:
    async def test_one_guild_never_sees_another_guilds_signals(
        self, conn: aiosqlite.Connection
    ) -> None:
        await _record(conn, guild_id=GUILD_A, channel_id=1, message_id=1)
        await _record(conn, guild_id=GUILD_B, channel_id=2, message_id=2)

        [signal] = await get_recent_signals(conn, guild_id=GUILD_A, limit=10)
        assert signal.guild_id == GUILD_A

    async def test_isolation_holds_for_a_guild_with_no_signals_of_its_own(
        self, conn: aiosqlite.Connection
    ) -> None:
        await _record(conn, guild_id=GUILD_A, message_id=1)

        assert await get_recent_signals(conn, guild_id=GUILD_B, limit=10) == []


class TestDuplicateDelivery:
    async def test_the_same_message_recorded_twice_produces_one_row(
        self, conn: aiosqlite.Connection
    ) -> None:
        # Discord's gateway can redeliver an event after a resumed session.
        await _record(conn, channel_id=7, message_id=123)
        await _record(conn, channel_id=7, message_id=123)

        assert len(await get_recent_signals(conn, guild_id=GUILD_A, limit=10)) == 1

    async def test_a_duplicate_does_not_raise(self, conn: aiosqlite.Connection) -> None:
        await _record(conn, channel_id=7, message_id=123)
        await _record(conn, channel_id=7, message_id=123)  # must not raise

    async def test_the_first_trail_is_the_one_kept(self, conn: aiosqlite.Connection) -> None:
        # The first evaluation is the one that decided something and possibly
        # spent a slot. The second can only ever conclude DUPLICATE_DELIVERY,
        # so overwriting would replace a real decision with an artefact of
        # Discord's retry behaviour.
        await _record(conn, channel_id=7, message_id=123, decision=ELIGIBLE_TRAIL)
        duplicate = DecisionTrail(
            verdict=GateVerdict.DUPLICATE_DELIVERY,
            stage1_score=0.12,
            stage1_passed=True,
            stage2_top_score=0.81,
            stage2_passed=True,
            daily_count=3,
            daily_cap=20,
        )
        await _record(conn, channel_id=7, message_id=123, decision=duplicate)

        [signal] = await get_recent_signals(conn, guild_id=GUILD_A, limit=10)
        assert signal.verdict is GateVerdict.ELIGIBLE

    async def test_the_same_message_id_in_different_channels_is_two_observations(
        self, conn: aiosqlite.Connection
    ) -> None:
        await _record(conn, channel_id=1, message_id=555)
        await _record(conn, channel_id=2, message_id=555)

        assert len(await get_recent_signals(conn, guild_id=GUILD_A, limit=10)) == 2


class TestScoreValues:
    @pytest.mark.parametrize("score", [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0])
    async def test_the_whole_contrastive_range_round_trips(
        self, conn: aiosqlite.Connection, score: float
    ) -> None:
        # [-2, 2], not [-1, 1]: the score is a difference of two cosine
        # similarities now, and the floor (-2.0) is a real value the detector
        # returns for unscoreable text.
        trail = DecisionTrail(
            verdict=GateVerdict.STAGE1_REJECTED, stage1_score=score, stage1_passed=False
        )
        await _record(conn, message_id=1, decision=trail)

        [signal] = await get_recent_signals(conn, guild_id=GUILD_A, limit=10)
        assert signal.stage1_score == pytest.approx(score)

    @pytest.mark.parametrize("score", [float("nan"), float("inf"), float("-inf")])
    def test_a_non_finite_score_is_refused_at_the_model_boundary(self, score: float) -> None:
        # SQLite writes a NaN REAL as NULL and accepts +/-inf verbatim, so a
        # non-finite score would become either a lost row or a value that
        # poisons every average later taken from this table. Rejected before
        # it can reach a database at all.
        with pytest.raises(ValidationError):
            DecisionTrail(
                verdict=GateVerdict.STAGE1_REJECTED, stage1_score=score, stage1_passed=False
            )

    @pytest.mark.parametrize(
        "field",
        ["stage2_top_score", "stage2_runner_up_score", "stage2_gap", "cooldown_seconds_remaining"],
    )
    def test_every_optional_float_also_refuses_non_finite_values(self, field: str) -> None:
        with pytest.raises(ValidationError):
            DecisionTrail(
                verdict=GateVerdict.ELIGIBLE,
                stage1_score=0.1,
                stage1_passed=True,
                **{field: float("nan")},  # type: ignore[arg-type]
            )

    async def test_a_non_finite_score_never_reaches_the_table(
        self, conn: aiosqlite.Connection
    ) -> None:
        with pytest.raises(ValidationError):
            await _record(
                conn,
                decision=DecisionTrail(
                    verdict=GateVerdict.STAGE1_REJECTED,
                    stage1_score=float("nan"),
                    stage1_passed=False,
                ),
            )

        assert await get_recent_signals(conn, guild_id=GUILD_A, limit=10) == []

    async def test_a_genuine_constraint_violation_is_not_swallowed_by_the_duplicate_clause(
        self, conn: aiosqlite.Connection
    ) -> None:
        # Proves the ON CONFLICT clause is scoped to the duplicate case only:
        # a NULL score still raises, where INSERT OR IGNORE would have
        # discarded the row in silence.
        with pytest.raises(aiosqlite.IntegrityError):
            await conn.execute(
                """
                INSERT INTO proactive_signals
                    (guild_id, channel_id, message_id, stage1_score, stage1_passed, verdict, created_at)
                VALUES (?, ?, ?, NULL, ?, ?, ?)
                ON CONFLICT (channel_id, message_id) DO NOTHING
                """,
                (GUILD_A, 1, 1, 1, "eligible", "2026-07-24T00:00:00.000000+00:00"),
            )

    async def test_the_boolean_columns_reject_a_value_that_is_neither_true_nor_false(
        self, conn: aiosqlite.Connection
    ) -> None:
        # A CHECK constraint rather than trust: a 2 in stage1_passed would
        # read back as True and quietly misreport what the gate decided.
        with pytest.raises(aiosqlite.IntegrityError):
            await conn.execute(
                """
                INSERT INTO proactive_signals
                    (guild_id, channel_id, message_id, stage1_score, stage1_passed, verdict, created_at)
                VALUES (?, ?, ?, ?, 2, ?, ?)
                """,
                (GUILD_A, 1, 1, 0.0, "eligible", "2026-07-24T00:00:00.000000+00:00"),
            )


class TestConcurrency:
    async def test_a_burst_of_concurrent_writes_records_every_distinct_message(
        self, conn: aiosqlite.Connection
    ) -> None:
        await asyncio.gather(
            *(_record(conn, message_id=message_id) for message_id in range(50))
        )

        signals = await get_recent_signals(conn, guild_id=GUILD_A, limit=100)
        assert len(signals) == 50
        assert {s.message_id for s in signals} == set(range(50))

    async def test_concurrent_duplicates_still_collapse_to_one_row(
        self, conn: aiosqlite.Connection
    ) -> None:
        await asyncio.gather(*(_record(conn, message_id=1) for _ in range(20)))

        assert len(await get_recent_signals(conn, guild_id=GUILD_A, limit=100)) == 1


class TestSchemaVerification:
    async def test_a_freshly_initialized_database_passes(
        self, conn: aiosqlite.Connection
    ) -> None:
        await verify_signal_schema(conn)  # must not raise

    async def test_a_database_with_no_such_table_passes(self) -> None:
        # Nothing outdated to find. Reached only if the table is dropped
        # manually between init_schema and this check.
        connection = await aiosqlite.connect(":memory:")
        try:
            await verify_signal_schema(connection)  # must not raise
        finally:
            await connection.close()

    async def test_a_phase_2a_1_shaped_table_is_rejected_loudly(self, tmp_path: Path) -> None:
        # The failure this exists to prevent: CREATE TABLE IF NOT EXISTS
        # leaves the old table in place, so every INSERT afterwards fails --
        # one logged exception per message, forever, on a bot that otherwise
        # looks healthy.
        database = tmp_path / "legacy.db"
        connection = await aiosqlite.connect(database)
        try:
            await connection.execute(
                """
                CREATE TABLE proactive_signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    guild_id INTEGER NOT NULL,
                    channel_id INTEGER NOT NULL,
                    message_id INTEGER NOT NULL,
                    score REAL NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE (channel_id, message_id)
                )
                """
            )
            await connection.commit()
            # init_schema is a no-op on an existing table, which is the whole
            # problem -- so the check has to catch what it cannot fix.
            await init_schema(connection)

            with pytest.raises(OutdatedDiagnosticTableError, match="DROP TABLE"):
                await verify_signal_schema(connection)
        finally:
            await connection.close()

    async def test_the_rejection_names_the_table_and_the_fix(self, tmp_path: Path) -> None:
        database = tmp_path / "legacy.db"
        connection = await aiosqlite.connect(database)
        try:
            await connection.execute(
                "CREATE TABLE proactive_signals (id INTEGER PRIMARY KEY, score REAL)"
            )
            await connection.commit()

            with pytest.raises(OutdatedDiagnosticTableError) as raised:
                await verify_signal_schema(connection)
        finally:
            await connection.close()

        message = str(raised.value)
        assert "proactive_signals" in message
        assert "sqlite3" in message  # an actionable command, not just a complaint

    async def test_verification_is_idempotent(self, conn: aiosqlite.Connection) -> None:
        for _ in range(3):
            await verify_signal_schema(conn)

    async def test_verification_never_writes_anything(
        self, conn: aiosqlite.Connection
    ) -> None:
        await _record(conn, message_id=1)

        await verify_signal_schema(conn)

        assert len(await get_recent_signals(conn, guild_id=GUILD_A, limit=10)) == 1
        assert conn.in_transaction is False


class TestAdditiveMigration:
    """Phase 2a-2 -> 2a-3 is additive, so the table is migrated in place, not dropped."""

    @staticmethod
    def _columns_of(connection: aiosqlite.Connection):
        async def _run():
            async with connection.execute("PRAGMA table_info(proactive_signals)") as cursor:
                return {row[1] for row in await cursor.fetchall()}

        return _run()

    async def test_a_2a_2_table_gains_the_synthesis_columns(self, tmp_path: Path) -> None:
        connection = await aiosqlite.connect(tmp_path / "legacy.db")
        try:
            await connection.execute(_PHASE_2A_2_TABLE)
            await connection.commit()
            await init_schema(connection)  # a no-op on the existing table

            await verify_signal_schema(connection)

            columns = await self._columns_of(connection)
            assert "synthesis_answers_question" in columns
            assert "synthesis_posted" in columns
        finally:
            await connection.close()

    async def test_the_migration_preserves_existing_rows(self, tmp_path: Path) -> None:
        # Unlike the 2a-1 drop, no data is thrown away: an existing trail keeps
        # its stage numbers and simply gains NULL synthesis columns.
        connection = await aiosqlite.connect(tmp_path / "legacy.db")
        try:
            await connection.execute(_PHASE_2A_2_TABLE)
            await connection.execute(
                """
                INSERT INTO proactive_signals
                    (guild_id, channel_id, message_id, stage1_score, stage1_passed, verdict, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (GUILD_A, 5, 9, 0.123, 1, "stage1_rejected", "2026-01-01T00:00:00.000000+00:00"),
            )
            await connection.commit()
            await init_schema(connection)

            await verify_signal_schema(connection)

            [signal] = await get_recent_signals(connection, guild_id=GUILD_A, limit=10)
            assert signal.stage1_score == pytest.approx(0.123)
            assert signal.synthesis_answers_question is None
            assert signal.synthesis_posted is None
        finally:
            await connection.close()

    async def test_after_migration_the_current_writers_work(self, tmp_path: Path) -> None:
        connection = await aiosqlite.connect(tmp_path / "legacy.db")
        try:
            await connection.execute(_PHASE_2A_2_TABLE)
            await connection.commit()
            await init_schema(connection)
            await verify_signal_schema(connection)

            # The current INSERT (which does not name the synthesis columns) and
            # the new UPDATE both have to work against the migrated table.
            await _record(connection, channel_id=1, message_id=1, decision=ELIGIBLE_TRAIL)
            await update_synthesis_outcome(
                connection, channel_id=1, message_id=1, answers_question=True, posted=True
            )

            [signal] = await get_recent_signals(connection, guild_id=GUILD_A, limit=10)
            assert signal.synthesis_answers_question is True
            assert signal.synthesis_posted is True
        finally:
            await connection.close()

    async def test_migration_is_idempotent(self, tmp_path: Path) -> None:
        connection = await aiosqlite.connect(tmp_path / "legacy.db")
        try:
            await connection.execute(_PHASE_2A_2_TABLE)
            await connection.commit()
            await init_schema(connection)

            for _ in range(3):
                await verify_signal_schema(connection)  # must not raise on the second pass

            columns = await self._columns_of(connection)
            assert {"synthesis_answers_question", "synthesis_posted"} <= columns
        finally:
            await connection.close()


class TestGracePeriodAdditiveMigration:
    """Phase 2a-3 -> 2b-1 is additive too: grace_period_outcome joins in place."""

    @staticmethod
    def _columns_of(connection: aiosqlite.Connection):
        async def _run():
            async with connection.execute("PRAGMA table_info(proactive_signals)") as cursor:
                return {row[1] for row in await cursor.fetchall()}

        return _run()

    async def test_a_2a_3_table_gains_the_grace_column(self, tmp_path: Path) -> None:
        connection = await aiosqlite.connect(tmp_path / "legacy.db")
        try:
            await connection.execute(_PHASE_2A_3_TABLE)
            await connection.commit()
            await init_schema(connection)  # a no-op on the existing table

            await verify_signal_schema(connection)

            columns = await self._columns_of(connection)
            assert "grace_period_outcome" in columns
        finally:
            await connection.close()

    async def test_the_migration_preserves_existing_rows_including_synthesis_data(
        self, tmp_path: Path
    ) -> None:
        connection = await aiosqlite.connect(tmp_path / "legacy.db")
        try:
            await connection.execute(_PHASE_2A_3_TABLE)
            await connection.execute(
                """
                INSERT INTO proactive_signals
                    (guild_id, channel_id, message_id, stage1_score, stage1_passed,
                     synthesis_answers_question, synthesis_posted, verdict, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (GUILD_A, 5, 9, 0.123, 1, 1, 1, "eligible", "2026-01-01T00:00:00.000000+00:00"),
            )
            await connection.commit()
            await init_schema(connection)

            await verify_signal_schema(connection)

            [signal] = await get_recent_signals(connection, guild_id=GUILD_A, limit=10)
            assert signal.stage1_score == pytest.approx(0.123)
            assert signal.synthesis_answers_question is True
            assert signal.synthesis_posted is True
            assert signal.grace_period_outcome is None
        finally:
            await connection.close()

    async def test_after_migration_update_grace_outcome_works(self, tmp_path: Path) -> None:
        connection = await aiosqlite.connect(tmp_path / "legacy.db")
        try:
            await connection.execute(_PHASE_2A_3_TABLE)
            await connection.commit()
            await init_schema(connection)
            await verify_signal_schema(connection)

            await _record(connection, channel_id=1, message_id=1, decision=ELIGIBLE_TRAIL)
            await update_grace_outcome(
                connection,
                channel_id=1,
                message_id=1,
                outcome=GracePeriodOutcome.EXPIRED_AND_PROCEEDED,
            )

            [signal] = await get_recent_signals(connection, guild_id=GUILD_A, limit=10)
            assert signal.grace_period_outcome is GracePeriodOutcome.EXPIRED_AND_PROCEEDED
        finally:
            await connection.close()

    async def test_migration_from_2a_3_is_idempotent(self, tmp_path: Path) -> None:
        connection = await aiosqlite.connect(tmp_path / "legacy.db")
        try:
            await connection.execute(_PHASE_2A_3_TABLE)
            await connection.commit()
            await init_schema(connection)

            for _ in range(3):
                await verify_signal_schema(connection)  # must not raise on the second pass

            columns = await self._columns_of(connection)
            assert "grace_period_outcome" in columns
        finally:
            await connection.close()

    async def test_a_2a_2_table_gains_both_synthesis_and_grace_columns_in_one_pass(
        self, tmp_path: Path
    ) -> None:
        # A table two phases behind must not need two separate upgrade runs.
        connection = await aiosqlite.connect(tmp_path / "legacy.db")
        try:
            await connection.execute(_PHASE_2A_2_TABLE)
            await connection.commit()
            await init_schema(connection)

            await verify_signal_schema(connection)

            columns = await self._columns_of(connection)
            assert {
                "synthesis_answers_question",
                "synthesis_posted",
                "grace_period_outcome",
            } <= columns
        finally:
            await connection.close()


class TestSynthesisOutcome:
    async def test_a_posted_answer_reads_back(self, conn: aiosqlite.Connection) -> None:
        await _record(conn, channel_id=1, message_id=1, decision=ELIGIBLE_TRAIL)

        await update_synthesis_outcome(
            conn, channel_id=1, message_id=1, answers_question=True, posted=True
        )

        [signal] = await get_recent_signals(conn, guild_id=GUILD_A, limit=10)
        assert signal.synthesis_answers_question is True
        assert signal.synthesis_posted is True

    async def test_a_no_result_is_stored_as_null_not_false(
        self, conn: aiosqlite.Connection
    ) -> None:
        # None (no model result) must not be flattened into False (model said
        # no) -- the debug view distinguishes them.
        await _record(conn, channel_id=1, message_id=1, decision=ELIGIBLE_TRAIL)

        await update_synthesis_outcome(
            conn, channel_id=1, message_id=1, answers_question=None, posted=False
        )

        [signal] = await get_recent_signals(conn, guild_id=GUILD_A, limit=10)
        assert signal.synthesis_answers_question is None
        assert signal.synthesis_posted is False

    async def test_an_unposted_but_answered_outcome_reads_back(
        self, conn: aiosqlite.Connection
    ) -> None:
        await _record(conn, channel_id=1, message_id=1, decision=ELIGIBLE_TRAIL)

        await update_synthesis_outcome(
            conn, channel_id=1, message_id=1, answers_question=False, posted=False
        )

        [signal] = await get_recent_signals(conn, guild_id=GUILD_A, limit=10)
        assert signal.synthesis_answers_question is False
        assert signal.synthesis_posted is False

    async def test_it_updates_only_the_targeted_row(self, conn: aiosqlite.Connection) -> None:
        await _record(conn, channel_id=1, message_id=1, decision=ELIGIBLE_TRAIL)
        await _record(conn, channel_id=1, message_id=2, decision=ELIGIBLE_TRAIL)

        await update_synthesis_outcome(
            conn, channel_id=1, message_id=1, answers_question=True, posted=True
        )

        signals = {s.message_id: s for s in await get_recent_signals(conn, guild_id=GUILD_A, limit=10)}
        assert signals[1].synthesis_posted is True
        assert signals[2].synthesis_posted is None  # untouched

    async def test_updating_a_missing_row_is_a_harmless_no_op(
        self, conn: aiosqlite.Connection
    ) -> None:
        # A concurrent duplicate may have kept the ELIGIBLE trail under a
        # different task; an UPDATE that matches nothing must not raise.
        await update_synthesis_outcome(
            conn, channel_id=999, message_id=999, answers_question=True, posted=True
        )
        assert await get_recent_signals(conn, guild_id=GUILD_A, limit=10) == []

    async def test_a_fresh_eligible_row_has_null_synthesis_columns(
        self, conn: aiosqlite.Connection
    ) -> None:
        # record_signal alone leaves the synthesis columns unset; they are only
        # filled by the later update.
        await _record(conn, channel_id=1, message_id=1, decision=ELIGIBLE_TRAIL)

        [signal] = await get_recent_signals(conn, guild_id=GUILD_A, limit=10)
        assert signal.synthesis_answers_question is None
        assert signal.synthesis_posted is None


class TestKnowledgeModelIsolation:
    async def test_recording_signals_never_creates_a_fact(
        self, conn: aiosqlite.Connection
    ) -> None:
        for message_id in range(5):
            await _record(conn, message_id=message_id)

        assert await get_active_facts(conn, GUILD_A) == []

    async def test_the_diagnostic_table_carries_no_message_content_column(
        self, conn: aiosqlite.Connection
    ) -> None:
        # CLAUDE.md's Fact definition: origin is referenced by ID, raw text
        # is never duplicated into Aura's database. The same rule applies to
        # scaffolding that references messages.
        async with conn.execute("PRAGMA table_info(proactive_signals)") as cursor:
            columns = {row[1] for row in await cursor.fetchall()}

        assert columns == {
            "id",
            "guild_id",
            "channel_id",
            "message_id",
            "stage1_score",
            "stage1_passed",
            "stage2_top_score",
            "stage2_runner_up_score",
            "stage2_gap",
            "stage2_passed",
            "cooldown_seconds_remaining",
            "daily_count",
            "daily_cap",
            "synthesis_answers_question",
            "synthesis_posted",
            "grace_period_outcome",
            "verdict",
            "created_at",
        }

    async def test_the_diagnostic_table_has_no_foreign_key_into_the_knowledge_model(
        self, conn: aiosqlite.Connection
    ) -> None:
        async with conn.execute("PRAGMA foreign_key_list(proactive_signals)") as cursor:
            assert await cursor.fetchall() == []
