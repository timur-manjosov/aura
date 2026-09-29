"""Durable subscription state: one snapshot row per Stripe subscription, one row per applied event.

Two operations write here and both are in this module, so the rules that make
them safe sit side by side:

  * COMPARE-AND-SWAP ON `version`. Stripe delivers events out of order, so the
    web backend never applies an event's contents -- it reads this module's
    current version for the subscription, fetches the subscription from Stripe,
    and asks for the snapshot to be stored only if the version is still the one
    it read. A write whose version moved in the meantime is refused, so a slow
    fetch that finishes after a newer one cannot overwrite it, and the stored
    snapshot is always the most recently FETCHED one. No clock is compared
    anywhere: event timestamps are whole seconds and Stripe documents that
    distinct events share them.

  * THE EVENT ID IN THE SAME TRANSACTION. An event's ID is recorded in the same
    commit as the snapshot it caused, so an event is either applied and marked,
    or neither. A redelivery finds it marked and changes nothing -- no second
    grant, no second revocation, no version bump.

Every statement runs under the connection's operation lock (aura.db.connection),
which is what makes "read the version, then write" one indivisible step with
respect to every other coroutine in this process -- and this process is the
only writer of this database, by design.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum

import aiosqlite
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from aura.billing.entitlement import InvoiceStatus, SubscriptionRecord, SubscriptionStatus
from aura.db.connection import connection_lock, utc_iso

# SQLite INTEGER is signed 64-bit; sqlite3 raises when binding anything larger.
_MAX_SQLITE_INTEGER = 2**63 - 1

_RECORD_COLUMNS = (
    "subscription_id, guild_id, customer_id, purchaser_user_id, status, "
    "cancel_at_period_end, cancel_at, collection_paused, latest_invoice_status, "
    "current_period_start, current_period_end, livemode, version, confirmed_at"
)


def _to_unix(moment: datetime) -> int:
    return int(moment.timestamp())


def _from_unix(seconds: int) -> datetime:
    return datetime.fromtimestamp(seconds, tz=timezone.utc)


class SubscriptionSnapshot(BaseModel):
    """What Stripe says about one subscription, as the web backend fetched it."""

    model_config = ConfigDict(frozen=True)

    subscription_id: str
    guild_id: int
    customer_id: str
    purchaser_user_id: int | None
    status: SubscriptionStatus
    cancel_at_period_end: bool
    cancel_at: datetime | None
    collection_paused: bool
    latest_invoice_status: InvoiceStatus | None
    current_period_start: datetime
    current_period_end: datetime
    livemode: bool

    @field_validator("guild_id", "purchaser_user_id")
    @classmethod
    def _fits_the_database(cls, value: int | None) -> int | None:
        if value is not None and not 0 < value <= _MAX_SQLITE_INTEGER:
            raise ValueError(f"a Discord ID must be a positive 64-bit integer, got {value}")
        return value

    @field_validator("cancel_at", "current_period_start", "current_period_end")
    @classmethod
    def _timezone_aware(cls, value: datetime | None) -> datetime | None:
        if value is not None and value.tzinfo is None:
            raise ValueError("subscription times must be timezone-aware")
        return value

    @model_validator(mode="after")
    def _period_is_ordered(self) -> SubscriptionSnapshot:
        if self.current_period_end < self.current_period_start:
            raise ValueError("current_period_end must not precede current_period_start")
        return self


class ApplyOutcome(StrEnum):
    """What happened to one requested snapshot write."""

    APPLIED = "applied"
    DUPLICATE_EVENT = "duplicate_event"
    VERSION_CONFLICT = "version_conflict"


@dataclass(frozen=True)
class ApplyResult:
    """The outcome of apply_subscription_snapshot, with what the caller needs next."""

    outcome: ApplyOutcome
    # The subscription's stored version after this call (0 if it has no row).
    version: int
    # The stored record, only when the write was APPLIED.
    record: SubscriptionRecord | None
    # The guild the subscription belonged to before this write, only when an
    # applied snapshot moved it to a DIFFERENT guild.
    previous_guild_id: int | None


@dataclass(frozen=True)
class SyncState:
    """What the web backend needs before fetching from Stripe."""

    event_processed: bool
    version: int


def _record_from_row(row: aiosqlite.Row | tuple[object, ...]) -> SubscriptionRecord:
    (
        subscription_id,
        guild_id,
        customer_id,
        purchaser_user_id,
        status,
        cancel_at_period_end,
        cancel_at,
        collection_paused,
        latest_invoice_status,
        current_period_start,
        current_period_end,
        livemode,
        version,
        confirmed_at,
    ) = row
    return SubscriptionRecord(
        subscription_id=str(subscription_id),
        guild_id=int(guild_id),  # type: ignore[arg-type]
        customer_id=str(customer_id),
        purchaser_user_id=None if purchaser_user_id is None else int(purchaser_user_id),  # type: ignore[arg-type]
        status=SubscriptionStatus(status),
        cancel_at_period_end=bool(cancel_at_period_end),
        cancel_at=None if cancel_at is None else _from_unix(int(cancel_at)),  # type: ignore[arg-type]
        collection_paused=bool(collection_paused),
        latest_invoice_status=(
            None if latest_invoice_status is None else InvoiceStatus(latest_invoice_status)
        ),
        current_period_start=_from_unix(int(current_period_start)),  # type: ignore[arg-type]
        current_period_end=_from_unix(int(current_period_end)),  # type: ignore[arg-type]
        livemode=bool(livemode),
        version=int(version),  # type: ignore[arg-type]
        confirmed_at=datetime.fromisoformat(str(confirmed_at)),
    )


async def get_sync_state(
    conn: aiosqlite.Connection, *, subscription_id: str, event_id: str | None
) -> SyncState:
    """Whether an event was already applied, and the subscription's current version.

    Version 0 means "no row yet", which is also exactly the version a first
    write must name -- so the very first snapshot for a subscription goes
    through the same compare-and-swap as every later one.
    """
    async with connection_lock(conn):
        event_processed = False
        if event_id is not None:
            async with conn.execute(
                "SELECT 1 FROM stripe_processed_events WHERE event_id = ?", (event_id,)
            ) as cursor:
                event_processed = await cursor.fetchone() is not None
        async with conn.execute(
            "SELECT version FROM guild_subscriptions WHERE subscription_id = ?", (subscription_id,)
        ) as cursor:
            row = await cursor.fetchone()
    return SyncState(event_processed=event_processed, version=int(row[0]) if row else 0)


async def apply_subscription_snapshot(
    conn: aiosqlite.Connection,
    *,
    snapshot: SubscriptionSnapshot,
    event_id: str | None,
    event_type: str | None,
    expected_version: int,
    now: datetime,
) -> ApplyResult:
    """Store a snapshot if its event is new and the version is still the one the caller read.

    `event_id` and `event_type` are both given for a webhook-driven write and
    both None for a reconciliation write, which has no event to deduplicate.

    Checks run in this order, and the order is part of the contract: a
    duplicate event is reported as a duplicate even if the version has since
    moved, because "already applied" is the more useful answer to a redelivery
    -- it tells the caller to stop, where a conflict would tell it to retry.
    """
    if now.tzinfo is None:
        raise ValueError(f"now must be a timezone-aware datetime, got {now!r}")
    if (event_id is None) != (event_type is None):
        raise ValueError("event_id and event_type must be given together or not at all")
    if not 0 <= expected_version < _MAX_SQLITE_INTEGER:
        raise ValueError(f"expected_version out of range: {expected_version}")

    confirmed_at = utc_iso(now)
    async with connection_lock(conn):
        try:
            if event_id is not None:
                async with conn.execute(
                    "SELECT 1 FROM stripe_processed_events WHERE event_id = ?", (event_id,)
                ) as cursor:
                    already_applied = await cursor.fetchone() is not None
                if already_applied:
                    return ApplyResult(
                        ApplyOutcome.DUPLICATE_EVENT,
                        await _stored_version(conn, snapshot.subscription_id),
                        None,
                        None,
                    )

            async with conn.execute(
                "SELECT version, guild_id FROM guild_subscriptions WHERE subscription_id = ?",
                (snapshot.subscription_id,),
            ) as cursor:
                existing = await cursor.fetchone()
            current_version = int(existing[0]) if existing else 0
            if current_version != expected_version:
                return ApplyResult(ApplyOutcome.VERSION_CONFLICT, current_version, None, None)

            new_version = current_version + 1
            values = (
                snapshot.guild_id,
                snapshot.customer_id,
                snapshot.purchaser_user_id,
                snapshot.status.value,
                int(snapshot.cancel_at_period_end),
                None if snapshot.cancel_at is None else _to_unix(snapshot.cancel_at),
                int(snapshot.collection_paused),
                None if snapshot.latest_invoice_status is None else snapshot.latest_invoice_status.value,
                _to_unix(snapshot.current_period_start),
                _to_unix(snapshot.current_period_end),
                int(snapshot.livemode),
                new_version,
                confirmed_at,
            )
            if existing is None:
                await conn.execute(
                    "INSERT INTO guild_subscriptions (guild_id, customer_id, purchaser_user_id, "
                    "status, cancel_at_period_end, cancel_at, collection_paused, "
                    "latest_invoice_status, current_period_start, current_period_end, livemode, "
                    "version, confirmed_at, subscription_id, first_seen_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (*values, snapshot.subscription_id, confirmed_at),
                )
            else:
                cursor = await conn.execute(
                    "UPDATE guild_subscriptions SET guild_id = ?, customer_id = ?, "
                    "purchaser_user_id = ?, status = ?, cancel_at_period_end = ?, cancel_at = ?, "
                    "collection_paused = ?, latest_invoice_status = ?, current_period_start = ?, "
                    "current_period_end = ?, livemode = ?, version = ?, confirmed_at = ? "
                    "WHERE subscription_id = ? AND version = ?",
                    (*values, snapshot.subscription_id, current_version),
                )
                if cursor.rowcount != 1:
                    # Unreachable while every writer holds the connection lock;
                    # raised rather than assumed, so a future writer that does
                    # not cannot silently lose a snapshot.
                    raise RuntimeError(
                        f"subscription {snapshot.subscription_id} changed under the connection lock"
                    )

            if event_id is not None:
                await conn.execute(
                    "INSERT INTO stripe_processed_events (event_id, event_type, subscription_id, "
                    "processed_at) VALUES (?, ?, ?, ?)",
                    (event_id, event_type, snapshot.subscription_id, confirmed_at),
                )
            await conn.commit()
        except BaseException:
            # BaseException, not Exception: a task cancelled between the write
            # and the commit must not leave an open transaction on the one
            # shared connection for the next unrelated operation to commit.
            await conn.rollback()
            raise

    previous_guild_id = int(existing[1]) if existing is not None else None
    record = SubscriptionRecord(
        subscription_id=snapshot.subscription_id,
        guild_id=snapshot.guild_id,
        customer_id=snapshot.customer_id,
        purchaser_user_id=snapshot.purchaser_user_id,
        status=snapshot.status,
        cancel_at_period_end=snapshot.cancel_at_period_end,
        cancel_at=None if snapshot.cancel_at is None else _from_unix(_to_unix(snapshot.cancel_at)),
        collection_paused=snapshot.collection_paused,
        latest_invoice_status=snapshot.latest_invoice_status,
        current_period_start=_from_unix(_to_unix(snapshot.current_period_start)),
        current_period_end=_from_unix(_to_unix(snapshot.current_period_end)),
        livemode=snapshot.livemode,
        version=new_version,
        confirmed_at=datetime.fromisoformat(confirmed_at),
    )
    return ApplyResult(
        ApplyOutcome.APPLIED,
        new_version,
        record,
        previous_guild_id if previous_guild_id not in (None, snapshot.guild_id) else None,
    )


async def _stored_version(conn: aiosqlite.Connection, subscription_id: str) -> int:
    async with conn.execute(
        "SELECT version FROM guild_subscriptions WHERE subscription_id = ?", (subscription_id,)
    ) as cursor:
        row = await cursor.fetchone()
    return int(row[0]) if row else 0


async def load_subscription_records(conn: aiosqlite.Connection) -> list[SubscriptionRecord]:
    """Every stored subscription, for building the runtime plan gate at startup."""
    async with connection_lock(conn):
        async with conn.execute(
            f"SELECT {_RECORD_COLUMNS} FROM guild_subscriptions ORDER BY subscription_id"
        ) as cursor:
            rows = await cursor.fetchall()
    return [_record_from_row(row) for row in rows]


async def count_processed_events(conn: aiosqlite.Connection) -> int:
    """How many Stripe events have been applied -- a diagnostic, and the idempotency tests' witness."""
    async with connection_lock(conn):
        async with conn.execute("SELECT COUNT(*) FROM stripe_processed_events") as cursor:
            row = await cursor.fetchone()
    return int(row[0]) if row else 0
