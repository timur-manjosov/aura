"""The deletion ledger, re-application after a restore, departures and the purge job."""

from __future__ import annotations

import asyncio
import secrets
import sqlite3
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path

import aiosqlite
import pytest

from aura.config import DataPurgeMode, Settings
from aura.db.deletion import MemberDeletionMode
from aura.db.encryption import DatabaseOpenError
from aura.db.guild_departures import (
    clear_departure,
    due_departures,
    get_departures,
    mark_departed,
    reconcile_departures,
)
from aura.privacy.ledger import DeletionKind, DeletionLedger, DeletionReason, LedgerEntry
from aura.privacy.requests import (
    execute_fact_deletion,
    execute_guild_purge,
    execute_member_deletion,
    reapply_ledger,
)
from aura.privacy.sweeper import run_purge_cycle
from tests.privacy_data import (
    AFTER,
    BEFORE,
    GUILD_A,
    GUILD_B,
    MEMBER,
    OTHER,
    REQUEST,
    add_fact,
    open_database,
    populate,
    snapshot,
)


@pytest.fixture
async def db() -> AsyncIterator[aiosqlite.Connection]:
    connection = await open_database()
    yield connection
    await connection.close()


@pytest.fixture
async def ledger(tmp_path: Path) -> AsyncIterator[DeletionLedger]:
    opened = await DeletionLedger.open(str(tmp_path / "ledger.db"), None)
    yield opened
    await opened.close()


async def _count(conn: aiosqlite.Connection, sql: str, *parameters: object) -> int:
    async with conn.execute(sql, parameters) as cursor:
        row = await cursor.fetchone()
    assert row is not None
    return int(row[0])


def _settings(mode: DataPurgeMode = DataPurgeMode.REPORT, **overrides: object) -> Settings:
    return Settings(_env_file=None, discord_token="test", data_purge_mode=mode, **overrides)  # type: ignore[arg-type]


@dataclass
class FakePresence:
    ready: bool = True
    members_of: set[int] = field(default_factory=set)

    def is_ready(self) -> bool:
        return self.ready

    def is_member_of(self, guild_id: int) -> bool:
        return guild_id in self.members_of


class TestLedgerFile:
    async def test_an_entry_round_trips_without_any_content(
        self, ledger: DeletionLedger, tmp_path: Path
    ) -> None:
        entry_id = await ledger.record(
            LedgerEntry(
                kind=DeletionKind.MEMBER,
                reason=DeletionReason.MEMBER_REQUEST,
                requested_at=REQUEST,
                user_id=MEMBER,
                mode=MemberDeletionMode.DELETE_FACTS,
            )
        )
        from aura.db.deletion import DeletionCounts

        await ledger.complete(entry_id, DeletionCounts(rows={"facts": 2}), at=AFTER)
        [entry] = await ledger.entries()
        assert entry.user_id == MEMBER and entry.mode is MemberDeletionMode.DELETE_FACTS
        assert entry.requested_at == REQUEST and entry.completed_at is not None
        assert await ledger.count() == 1

    @pytest.mark.parametrize(
        "entry",
        [
            LedgerEntry(
                kind=DeletionKind.MEMBER, reason=DeletionReason.MEMBER_REQUEST, requested_at=REQUEST
            ),
            LedgerEntry(
                kind=DeletionKind.MEMBER,
                reason=DeletionReason.MEMBER_REQUEST,
                requested_at=REQUEST,
                user_id=0,
                mode=MemberDeletionMode.UNLINK,
            ),
            LedgerEntry(
                kind=DeletionKind.GUILD, reason=DeletionReason.ADMIN_REQUEST, requested_at=REQUEST
            ),
            LedgerEntry(
                kind=DeletionKind.FACT,
                reason=DeletionReason.MODERATOR_REQUEST,
                requested_at=REQUEST,
                guild_id=GUILD_A,
                fact_id=1,
            ),
        ],
    )
    async def test_an_entry_that_could_not_be_reapplied_is_refused(
        self, ledger: DeletionLedger, entry: LedgerEntry
    ) -> None:
        with pytest.raises(sqlite3.IntegrityError):
            await ledger.record(entry)

    async def test_an_encrypted_ledger_needs_its_key(self, tmp_path: Path) -> None:
        pytest.importorskip("sqlcipher3")
        key = secrets.token_hex(32)
        path = str(tmp_path / "ledger.db")
        opened = await DeletionLedger.open(path, key)
        await opened.close()
        assert not (await asyncio.to_thread(Path(path).read_bytes)).startswith(b"SQLite format 3")
        # Bounded: a refusal must come as an error, never as a hang.
        with pytest.raises(DatabaseOpenError):
            await asyncio.wait_for(DeletionLedger.open(path, secrets.token_hex(32)), timeout=10)
        with pytest.raises(DatabaseOpenError):
            await asyncio.wait_for(DeletionLedger.open(path, None), timeout=10)
        reopened = await DeletionLedger.open(path, key)
        assert await reopened.count() == 0
        await reopened.close()


class TestExecutingRequests:
    async def test_a_member_deletion_is_recorded_with_counts(
        self, db: aiosqlite.Connection, ledger: DeletionLedger
    ) -> None:
        await populate(db)
        counts = await execute_member_deletion(
            db,
            ledger,
            user_id=MEMBER,
            guild_id=None,
            mode=MemberDeletionMode.DELETE_FACTS,
            reason=DeletionReason.MEMBER_REQUEST,
            now=REQUEST,
        )
        [entry] = await ledger.entries()
        assert entry.kind is DeletionKind.MEMBER and entry.guild_id is None
        assert entry.completed_at is not None
        assert counts.total > 0

    async def test_a_zero_user_is_refused_before_anything_is_recorded(
        self, db: aiosqlite.Connection, ledger: DeletionLedger
    ) -> None:
        with pytest.raises(ValueError):
            await execute_member_deletion(
                db,
                ledger,
                user_id=0,
                guild_id=None,
                mode=MemberDeletionMode.UNLINK,
                reason=DeletionReason.OPERATOR_REQUEST,
                now=REQUEST,
            )
        assert await ledger.count() == 0

    async def test_the_ledger_holds_no_fact_text(
        self, db: aiosqlite.Connection, ledger: DeletionLedger, tmp_path: Path
    ) -> None:
        canary = "CANARY-p7a-ledger-7f3e"
        fact_id = await add_fact(db, guild_id=GUILD_A, author=MEMBER, when=BEFORE, content=canary)
        await execute_fact_deletion(
            db,
            ledger,
            guild_id=GUILD_A,
            fact_id=fact_id,
            fact_created_at=BEFORE + timedelta(minutes=5),
            reason=DeletionReason.MODERATOR_REQUEST,
            now=REQUEST,
        )
        await execute_member_deletion(
            db,
            ledger,
            user_id=MEMBER,
            guild_id=None,
            mode=MemberDeletionMode.DELETE_FACTS,
            reason=DeletionReason.MEMBER_REQUEST,
            now=REQUEST,
        )
        for file in await asyncio.to_thread(lambda: list(tmp_path.iterdir())):
            assert canary.encode() not in await asyncio.to_thread(file.read_bytes)


class TestReapplicationAfterARestore:
    async def test_a_restored_backup_loses_the_deleted_rows_again(
        self, tmp_path: Path, ledger: DeletionLedger
    ) -> None:
        live_path = tmp_path / "aura.db"
        live = await open_database(str(live_path))
        await populate(live)
        await live.commit()
        backup_path = tmp_path / "backup.db"
        source = sqlite3.connect(live_path)
        target = sqlite3.connect(backup_path)
        source.backup(target)
        source.close()
        target.close()

        await execute_member_deletion(
            live,
            ledger,
            user_id=MEMBER,
            guild_id=None,
            mode=MemberDeletionMode.DELETE_FACTS,
            reason=DeletionReason.MEMBER_REQUEST,
            now=REQUEST,
        )
        await execute_guild_purge(
            live, ledger, guild_id=GUILD_B, reason=DeletionReason.ADMIN_REQUEST, now=REQUEST
        )
        expected = await snapshot(live)
        await live.close()

        restored = await open_database(str(backup_path))
        try:
            removed = await reapply_ledger(restored, ledger)
            assert removed.total > 0
            assert await snapshot(restored) == expected
            # And again: nothing left to do.
            assert (await reapply_ledger(restored, ledger)).total == 0
        finally:
            await restored.close()

    async def test_data_from_after_the_request_is_never_touched(
        self, db: aiosqlite.Connection, ledger: DeletionLedger
    ) -> None:
        await populate(db)
        await execute_member_deletion(
            db,
            ledger,
            user_id=MEMBER,
            guild_id=None,
            mode=MemberDeletionMode.DELETE_FACTS,
            reason=DeletionReason.MEMBER_REQUEST,
            now=REQUEST,
        )
        later = await add_fact(db, guild_id=GUILD_A, author=MEMBER, when=AFTER + timedelta(days=1))
        assert (await reapply_ledger(db, ledger)).total == 0
        assert await _count(db, "SELECT COUNT(*) FROM facts WHERE id = ?", later) == 1

    async def test_an_interrupted_deletion_is_completed_at_the_next_start(
        self, db: aiosqlite.Connection, ledger: DeletionLedger
    ) -> None:
        await populate(db)
        await ledger.record(
            LedgerEntry(
                kind=DeletionKind.GUILD,
                reason=DeletionReason.ADMIN_REQUEST,
                requested_at=REQUEST,
                guild_id=GUILD_A,
            )
        )
        removed = await reapply_ledger(db, ledger)
        assert removed.total > 0
        [entry] = await ledger.entries()
        assert entry.completed_at is not None

    async def test_a_fact_entry_never_hits_a_fact_that_reused_the_id(
        self, db: aiosqlite.Connection, ledger: DeletionLedger
    ) -> None:
        fact_id = await add_fact(db, guild_id=GUILD_A, author=OTHER, when=BEFORE)
        await execute_fact_deletion(
            db,
            ledger,
            guild_id=GUILD_A,
            fact_id=fact_id,
            fact_created_at=BEFORE + timedelta(minutes=5),
            reason=DeletionReason.MODERATOR_REQUEST,
            now=REQUEST,
        )
        # A restored backup whose sequence is older: a new fact gets the same ID.
        await db.execute(
            "INSERT INTO facts (id, guild_id, channel_id, message_id, content, embedding, status, "
            "created_at) VALUES (?, ?, 1, 1, 'new', x'00', 'active', '2026-12-01T00:00:00.000000+00:00')",
            (fact_id, GUILD_A),
        )
        await db.commit()
        assert (await reapply_ledger(db, ledger)).total == 0


class TestDepartures:
    async def test_a_second_mark_keeps_the_first_period(self, db: aiosqlite.Connection) -> None:
        assert await mark_departed(db, guild_id=GUILD_A, now=REQUEST, grace_days=30)
        assert not await mark_departed(db, guild_id=GUILD_A, now=AFTER, grace_days=1)
        [departure] = await get_departures(db)
        assert departure.purge_after.startswith("2026-10-31")

    async def test_due_exactly_at_the_end_of_the_period(self, db: aiosqlite.Connection) -> None:
        await mark_departed(db, guild_id=GUILD_A, now=REQUEST, grace_days=30)
        end = REQUEST + timedelta(days=30)
        assert await due_departures(db, now=end - timedelta(microseconds=1)) == []
        assert [d.guild_id for d in await due_departures(db, now=end)] == [GUILD_A]

    async def test_grace_below_one_day_is_refused(self, db: aiosqlite.Connection) -> None:
        with pytest.raises(ValueError):
            await mark_departed(db, guild_id=GUILD_A, now=REQUEST, grace_days=0)

    async def test_reconciliation_marks_absent_servers_and_clears_returned_ones(
        self, db: aiosqlite.Connection
    ) -> None:
        await mark_departed(db, guild_id=GUILD_B, now=BEFORE, grace_days=30)
        marked, cleared = await reconcile_departures(
            db,
            guilds_with_data={GUILD_A, GUILD_B},
            present_guild_ids={GUILD_B},
            now=REQUEST,
            grace_days=30,
        )
        assert (marked, cleared) == (1, 1)
        assert [d.guild_id for d in await get_departures(db)] == [GUILD_A]
        assert await clear_departure(db, guild_id=GUILD_A)
        assert not await clear_departure(db, guild_id=GUILD_A)

    async def test_a_server_aura_is_in_is_never_marked(self, db: aiosqlite.Connection) -> None:
        marked, cleared = await reconcile_departures(
            db,
            guilds_with_data={GUILD_A},
            present_guild_ids={GUILD_A},
            now=REQUEST,
            grace_days=30,
        )
        assert (marked, cleared) == (0, 0)
        assert await get_departures(db) == []


# A cycle well after every fixture row: what is due, is due completely.
CYCLE = AFTER + timedelta(days=1)


class TestPurgeCycle:
    async def _left(self, db: aiosqlite.Connection) -> None:
        await populate(db)
        await mark_departed(db, guild_id=GUILD_A, now=CYCLE - timedelta(days=31), grace_days=30)

    async def test_report_mode_deletes_nothing(
        self, db: aiosqlite.Connection, ledger: DeletionLedger
    ) -> None:
        await self._left(db)
        before = await snapshot(db)
        report = await run_purge_cycle(db, ledger, FakePresence(), settings=_settings(), now=CYCLE)
        assert await snapshot(db) == before
        assert report.would_purge[GUILD_A].total > 0
        assert await ledger.count() == 0

    async def test_delete_mode_purges_and_records_it(
        self, db: aiosqlite.Connection, ledger: DeletionLedger
    ) -> None:
        await self._left(db)
        report = await run_purge_cycle(
            db, ledger, FakePresence(), settings=_settings(DataPurgeMode.DELETE), now=CYCLE
        )
        assert report.purged[GUILD_A].total > 0
        assert await _count(db, "SELECT COUNT(*) FROM facts WHERE guild_id = ?", GUILD_A) == 0
        assert await _count(db, "SELECT COUNT(*) FROM facts WHERE guild_id = ?", GUILD_B) > 0
        assert await get_departures(db) == []
        [entry] = await ledger.entries()
        assert entry.reason is DeletionReason.LEFT_SERVER

    async def test_report_counts_equal_what_delete_mode_then_removes(
        self, db: aiosqlite.Connection, ledger: DeletionLedger
    ) -> None:
        await self._left(db)
        report = await run_purge_cycle(db, ledger, FakePresence(), settings=_settings(), now=CYCLE)
        purged = await run_purge_cycle(
            db, ledger, FakePresence(), settings=_settings(DataPurgeMode.DELETE), now=CYCLE
        )
        assert report.would_purge[GUILD_A] == purged.purged[GUILD_A]

    async def test_a_server_aura_is_back_in_is_never_purged(
        self, db: aiosqlite.Connection, ledger: DeletionLedger
    ) -> None:
        await self._left(db)
        report = await run_purge_cycle(
            db,
            ledger,
            FakePresence(members_of={GUILD_A}),
            settings=_settings(DataPurgeMode.DELETE),
            now=CYCLE,
        )
        assert report.returned == [GUILD_A] and report.purged == {}
        assert await _count(db, "SELECT COUNT(*) FROM facts WHERE guild_id = ?", GUILD_A) > 0

    async def test_nothing_is_judged_while_the_gateway_is_not_ready(
        self, db: aiosqlite.Connection, ledger: DeletionLedger
    ) -> None:
        await self._left(db)
        report = await run_purge_cycle(
            db,
            ledger,
            FakePresence(ready=False),
            settings=_settings(DataPurgeMode.DELETE),
            now=CYCLE,
        )
        assert report.skipped_not_ready and report.purged == {}
        assert await _count(db, "SELECT COUNT(*) FROM facts WHERE guild_id = ?", GUILD_A) > 0

    async def test_a_server_just_inside_its_period_is_kept(
        self, db: aiosqlite.Connection, ledger: DeletionLedger
    ) -> None:
        await populate(db)
        await mark_departed(db, guild_id=GUILD_A, now=REQUEST - timedelta(days=30), grace_days=30)
        report = await run_purge_cycle(
            db,
            ledger,
            FakePresence(),
            settings=_settings(DataPurgeMode.DELETE),
            now=REQUEST - timedelta(microseconds=1),
        )
        assert report.purged == {}

    async def test_retention_reported_equals_retention_deleted(
        self, db: aiosqlite.Connection, ledger: DeletionLedger
    ) -> None:
        await populate(db)
        settings = _settings(ask_member_id_retention_days=1, onboarding_send_retention_days=1)
        report = await run_purge_cycle(db, ledger, FakePresence(), settings=settings, now=CYCLE)
        deleting = settings.model_copy(update={"data_purge_mode": DataPurgeMode.DELETE})
        real = await run_purge_cycle(db, ledger, FakePresence(), settings=deleting, now=CYCLE)
        assert report.retention == real.retention
        assert real.retention.total > 0

    async def test_the_ledger_is_reapplied_in_report_mode_too(
        self, db: aiosqlite.Connection, ledger: DeletionLedger
    ) -> None:
        await populate(db)
        await ledger.record(
            LedgerEntry(
                kind=DeletionKind.MEMBER,
                reason=DeletionReason.MEMBER_REQUEST,
                requested_at=REQUEST,
                user_id=MEMBER,
                mode=MemberDeletionMode.DELETE_FACTS,
            )
        )
        report = await run_purge_cycle(
            db, ledger, FakePresence(), settings=_settings(), now=REQUEST
        )
        assert report.reapplied.total > 0


class TestNothingIsDeletedOnDowngrade:
    """R4: a cancelled or downgraded plan never deletes anything, in any purge mode."""

    async def test_a_cancelled_subscription_leaves_every_row_and_the_purge_job_deletes_nothing(
        self, db: aiosqlite.Connection, ledger: DeletionLedger
    ) -> None:
        from datetime import UTC, datetime

        from aura.db.subscriptions import (
            SubscriptionSnapshot,
            SubscriptionStatus,
            apply_subscription_snapshot,
        )

        await populate(db)
        knowledge = {
            table: rows
            for table, rows in (await snapshot(db)).items()
            if table not in ("guild_subscriptions", "stripe_processed_events")
        }
        for status in (SubscriptionStatus.ACTIVE, SubscriptionStatus.CANCELED):
            version = await _count(
                db,
                "SELECT COALESCE(MAX(version), 0) FROM guild_subscriptions WHERE subscription_id = ?",
                "sub_downgrade",
            )
            await apply_subscription_snapshot(
                db,
                snapshot=SubscriptionSnapshot(
                    subscription_id="sub_downgrade",
                    guild_id=GUILD_A,
                    customer_id="cus_x",
                    purchaser_user_id=MEMBER,
                    status=status,
                    cancel_at_period_end=False,
                    cancel_at=None,
                    collection_paused=False,
                    latest_invoice_status=None,
                    current_period_start=datetime(2026, 9, 1, tzinfo=UTC),
                    current_period_end=datetime(2026, 10, 1, tzinfo=UTC),
                    livemode=False,
                    on_pro_price=True,
                ),
                event_id=None,
                event_type=None,
                expected_version=version,
                now=REQUEST,
            )
        await run_purge_cycle(
            db,
            ledger,
            FakePresence(members_of={GUILD_A, GUILD_B}),
            settings=_settings(
                DataPurgeMode.DELETE,
                proactive_signal_retention_days=3650,
                ask_member_id_retention_days=3650,
                onboarding_send_retention_days=3650,
            ),
            now=REQUEST,
        )
        after = {
            table: rows
            for table, rows in (await snapshot(db)).items()
            if table not in ("guild_subscriptions", "stripe_processed_events")
        }
        assert after == knowledge
