"""aura.db.subscriptions: the compare-and-swap, the event ledger, and their shared transaction.

These are the two properties the brief's idempotency and race attacks target,
tested against a real SQLite database rather than described:

  * an event delivered twice is applied once -- no second grant, no second
    revocation, no version bump;
  * a stale write never overwrites a newer one, however the writes interleave.
"""

from __future__ import annotations

import asyncio
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import aiosqlite
import pytest

from aura.billing.entitlement import InvoiceStatus, SubscriptionStatus
from aura.db.repository import init_schema
from aura.db.subscriptions import (
    ApplyOutcome,
    SubscriptionSnapshot,
    apply_subscription_snapshot,
    count_processed_events,
    get_sync_state,
    load_subscription_records,
)

NOW = datetime(2026, 9, 13, 12, 0, 0, tzinfo=UTC)
GUILD_A = 100000000000000001
GUILD_B = 200000000000000002


@pytest.fixture
async def conn():
    connection = await aiosqlite.connect(":memory:")
    await init_schema(connection)
    yield connection
    await connection.close()


def snapshot(**overrides: object) -> SubscriptionSnapshot:
    values: dict[str, object] = {
        "subscription_id": "sub_A",
        "guild_id": GUILD_A,
        "customer_id": "cus_A",
        "purchaser_user_id": 5000,
        "status": SubscriptionStatus.ACTIVE,
        "cancel_at_period_end": False,
        "cancel_at": None,
        "collection_paused": False,
        "latest_invoice_status": InvoiceStatus.PAID,
        "current_period_start": NOW,
        "current_period_end": NOW + timedelta(days=30),
        "livemode": False,
    }
    values.update(overrides)
    return SubscriptionSnapshot(**values)  # type: ignore[arg-type]


async def apply(conn, *, expected: int, event: str | None = "evt_1", **overrides: object):
    return await apply_subscription_snapshot(
        conn,
        snapshot=snapshot(**overrides),
        event_id=event,
        event_type=None if event is None else "customer.subscription.updated",
        expected_version=expected,
        now=NOW,
    )


class TestFirstWrite:
    async def test_the_first_snapshot_names_version_zero_and_becomes_version_one(
        self, conn
    ) -> None:
        result = await apply(conn, expected=0)

        assert result.outcome is ApplyOutcome.APPLIED
        assert result.version == 1
        assert result.record is not None and result.record.version == 1
        assert await get_sync_state(conn, subscription_id="sub_A", event_id="evt_1") == (
            await get_sync_state(conn, subscription_id="sub_A", event_id="evt_1")
        )
        state = await get_sync_state(conn, subscription_id="sub_A", event_id="evt_1")
        assert state.event_processed and state.version == 1

    async def test_every_field_round_trips(self, conn) -> None:
        cancel_at = NOW + timedelta(days=5)
        await apply(
            conn,
            expected=0,
            purchaser_user_id=None,
            cancel_at=cancel_at,
            cancel_at_period_end=True,
            collection_paused=True,
            latest_invoice_status=None,
            status=SubscriptionStatus.PAST_DUE,
            livemode=True,
        )

        (stored,) = await load_subscription_records(conn)
        assert stored.purchaser_user_id is None
        assert stored.cancel_at == cancel_at
        assert stored.cancel_at_period_end is True
        assert stored.collection_paused is True
        assert stored.latest_invoice_status is None
        assert stored.status is SubscriptionStatus.PAST_DUE
        assert stored.livemode is True
        assert stored.current_period_end == NOW + timedelta(days=30)

    async def test_an_unknown_subscription_has_version_zero(self, conn) -> None:
        state = await get_sync_state(conn, subscription_id="sub_never", event_id="evt_never")

        assert state.event_processed is False and state.version == 0


class TestIdempotency:
    async def test_the_same_event_twice_is_applied_once(self, conn) -> None:
        await apply(conn, expected=0)
        second = await apply(conn, expected=1, status=SubscriptionStatus.CANCELED)

        assert second.outcome is ApplyOutcome.DUPLICATE_EVENT
        assert second.version == 1
        (stored,) = await load_subscription_records(conn)
        assert stored.status is SubscriptionStatus.ACTIVE
        assert await count_processed_events(conn) == 1

    async def test_a_duplicate_is_reported_as_a_duplicate_even_when_its_version_is_stale(
        self, conn
    ) -> None:
        """An already-applied event tells the caller to stop, not to retry forever."""
        await apply(conn, expected=0)
        await apply(conn, expected=1, event="evt_2")

        again = await apply(conn, expected=0, event="evt_1")

        assert again.outcome is ApplyOutcome.DUPLICATE_EVENT

    async def test_a_redelivery_racing_itself_is_applied_exactly_once(self, conn) -> None:
        results = await asyncio.gather(*(apply(conn, expected=0) for _ in range(10)))

        assert [result.outcome for result in results].count(ApplyOutcome.APPLIED) == 1
        assert await count_processed_events(conn) == 1
        assert (await get_sync_state(conn, subscription_id="sub_A", event_id=None)).version == 1

    async def test_a_reconciliation_write_records_no_event(self, conn) -> None:
        result = await apply(conn, expected=0, event=None)

        assert result.outcome is ApplyOutcome.APPLIED
        assert await count_processed_events(conn) == 0


class TestCompareAndSwap:
    async def test_a_wrong_expected_version_changes_nothing(self, conn) -> None:
        await apply(conn, expected=0)

        conflict = await apply(conn, expected=0, event="evt_2", status=SubscriptionStatus.CANCELED)

        assert conflict.outcome is ApplyOutcome.VERSION_CONFLICT
        assert conflict.version == 1
        (stored,) = await load_subscription_records(conn)
        assert stored.status is SubscriptionStatus.ACTIVE
        assert await count_processed_events(conn) == 1

    async def test_a_slow_stale_fetch_cannot_overwrite_a_newer_one(self, conn) -> None:
        """Sync A reads v1 and fetches "active"; sync B reads v1, fetches "canceled", commits first."""
        await apply(conn, expected=0)
        a_read = (await get_sync_state(conn, subscription_id="sub_A", event_id="evt_a")).version
        b_read = (await get_sync_state(conn, subscription_id="sub_A", event_id="evt_b")).version

        b = await apply(conn, expected=b_read, event="evt_b", status=SubscriptionStatus.CANCELED)
        a = await apply(conn, expected=a_read, event="evt_a", status=SubscriptionStatus.ACTIVE)

        assert b.outcome is ApplyOutcome.APPLIED
        assert a.outcome is ApplyOutcome.VERSION_CONFLICT
        (stored,) = await load_subscription_records(conn)
        assert stored.status is SubscriptionStatus.CANCELED
        # The losing event is NOT marked processed, so its retry re-fetches.
        assert (
            await get_sync_state(conn, subscription_id="sub_A", event_id="evt_a")
        ).event_processed is False

    async def test_ten_concurrent_writers_on_one_version_yield_one_winner(self, conn) -> None:
        results = await asyncio.gather(
            *(apply(conn, expected=0, event=f"evt_{index}") for index in range(10))
        )

        outcomes = [result.outcome for result in results]
        assert outcomes.count(ApplyOutcome.APPLIED) == 1
        assert outcomes.count(ApplyOutcome.VERSION_CONFLICT) == 9
        assert await count_processed_events(conn) == 1


class TestAtomicity:
    async def test_a_failure_recording_the_event_rolls_back_the_snapshot(self, conn) -> None:
        original_execute = conn.execute

        def failing_execute(sql: str, *args: object, **kwargs: object):
            if sql.startswith("INSERT INTO stripe_processed_events"):
                raise sqlite3.OperationalError("disk I/O error (simulated)")
            return original_execute(sql, *args, **kwargs)

        with patch.object(conn, "execute", side_effect=failing_execute):
            with pytest.raises(sqlite3.OperationalError):
                await apply(conn, expected=0)

        assert await load_subscription_records(conn) == []
        assert await count_processed_events(conn) == 0
        # And the connection is usable afterwards -- no transaction left open.
        assert (await apply(conn, expected=0)).outcome is ApplyOutcome.APPLIED

    async def test_the_state_survives_a_restart(self, tmp_path: Path) -> None:
        path = tmp_path / "aura.db"
        first = await aiosqlite.connect(path)
        await init_schema(first)
        await apply(first, expected=0)
        await first.close()

        second = await aiosqlite.connect(path)
        await init_schema(second)
        try:
            state = await get_sync_state(second, subscription_id="sub_A", event_id="evt_1")
            assert state.event_processed and state.version == 1
            assert (await apply(second, expected=1)).outcome is ApplyOutcome.DUPLICATE_EVENT
        finally:
            await second.close()


class TestGuildBinding:
    async def test_moving_a_subscription_to_another_guild_reports_where_it_came_from(
        self, conn
    ) -> None:
        await apply(conn, expected=0)

        moved = await apply(conn, expected=1, event="evt_2", guild_id=GUILD_B)

        assert moved.previous_guild_id == GUILD_A
        (stored,) = await load_subscription_records(conn)
        assert stored.guild_id == GUILD_B

    async def test_an_ordinary_update_reports_no_move(self, conn) -> None:
        await apply(conn, expected=0)

        assert (await apply(conn, expected=1, event="evt_2")).previous_guild_id is None


class TestRefusedInput:
    async def test_an_event_id_without_a_type_is_refused(self, conn) -> None:
        with pytest.raises(ValueError):
            await apply_subscription_snapshot(
                conn,
                snapshot=snapshot(),
                event_id="evt_1",
                event_type=None,
                expected_version=0,
                now=NOW,
            )

    async def test_a_naive_now_is_refused(self, conn) -> None:
        with pytest.raises(ValueError):
            await apply_subscription_snapshot(
                conn,
                snapshot=snapshot(),
                event_id=None,
                event_type=None,
                expected_version=0,
                now=datetime(2026, 9, 13),
            )

    async def test_a_negative_expected_version_is_refused(self, conn) -> None:
        with pytest.raises(ValueError):
            await apply(conn, expected=-1)

    @pytest.mark.parametrize("guild_id", [0, -1, 2**63])
    def test_a_guild_id_the_database_cannot_hold_is_refused(self, guild_id: int) -> None:
        with pytest.raises(ValueError):
            snapshot(guild_id=guild_id)

    def test_an_inverted_period_is_refused(self) -> None:
        with pytest.raises(ValueError):
            snapshot(current_period_end=NOW - timedelta(days=1))

    async def test_the_schema_itself_refuses_an_unknown_status(self, conn) -> None:
        with pytest.raises(sqlite3.IntegrityError):
            await conn.execute(
                "INSERT INTO guild_subscriptions (subscription_id, guild_id, customer_id, status, "
                "cancel_at_period_end, collection_paused, current_period_start, current_period_end, "
                "livemode, version, first_seen_at, confirmed_at) "
                "VALUES ('sub_X', 1, 'cus_X', 'free_forever', 0, 0, 0, 1, 0, 1, 'x', 'x')"
            )
