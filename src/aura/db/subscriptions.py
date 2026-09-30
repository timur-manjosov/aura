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

  * THE PAYMENT-GRACE ANCHOR FROM THE ROW BEING REPLACED. `past_due_since` is
    not something Stripe reports; it is carried from the stored row to the
    snapshot replacing it (aura.billing.entitlement.next_unpaid_since), read
    and written inside the same transaction as the compare-and-swap. Because
    that swap stores snapshots strictly in fetch order, the anchor is a fold
    over exactly the sequence of states the bot has seen, and no reordering of
    webhooks can hand it a different sequence.

Every statement runs under the connection's operation lock (aura.db.connection),
which is what makes "read the version, then write" one indivisible step with
respect to every other coroutine in this process -- and this process is the
only writer of this database, by design.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Final

import aiosqlite
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from aura.billing.entitlement import (
    InvoiceStatus,
    SubscriptionRecord,
    SubscriptionStatus,
    next_unpaid_since,
)
from aura.db.connection import connection_lock, utc_iso

# SQLite INTEGER is signed 64-bit; sqlite3 raises when binding anything larger.
_MAX_SQLITE_INTEGER = 2**63 - 1

_RECORD_COLUMNS = (
    "subscription_id, guild_id, customer_id, purchaser_user_id, status, "
    "cancel_at_period_end, cancel_at, collection_paused, latest_invoice_status, "
    "current_period_start, current_period_end, livemode, version, confirmed_at, "
    "on_pro_price, past_due_since"
)

# The two columns Phase 4c's audit fixes added to guild_subscriptions, with the
# exact definitions schema.sql gives a fresh table. verify_subscriptions_schema
# adds them to a table created before they existed, in this order, so a
# migrated table and a fresh one end up with identical column lists.
#
# on_pro_price defaults to 1 for rows written before the price was checked:
# those rows keep the meaning they were stored with, and the next sync of each
# subscription -- the reconciliation runs a minute after the web backend
# starts -- replaces it with the checked value.
_ADDITIVE_COLUMNS: Final[tuple[tuple[str, str], ...]] = (
    ("on_pro_price", "INTEGER NOT NULL DEFAULT 1 CHECK (on_pro_price IN (0, 1))"),
    ("past_due_since", "INTEGER"),
)


def _to_unix(moment: datetime) -> int:
    return int(moment.timestamp())


def _from_unix(seconds: int) -> datetime:
    return datetime.fromtimestamp(seconds, tz=UTC)


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
    # Every item on the configured Pro price at a quantity of at least one --
    # decided by the web backend, the only side that knows the price.
    on_pro_price: bool

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
        on_pro_price,
        past_due_since,
    ) = row
    return SubscriptionRecord(
        subscription_id=str(subscription_id),
        guild_id=int(guild_id),  # type: ignore[arg-type]
        customer_id=str(customer_id),
        purchaser_user_id=None if purchaser_user_id is None else int(purchaser_user_id),  # type: ignore[arg-type]
        status=SubscriptionStatus(status),  # type: ignore[arg-type]
        cancel_at_period_end=bool(cancel_at_period_end),
        cancel_at=None if cancel_at is None else _from_unix(int(cancel_at)),  # type: ignore[arg-type]
        collection_paused=bool(collection_paused),
        latest_invoice_status=(
            None if latest_invoice_status is None else InvoiceStatus(latest_invoice_status)  # type: ignore[arg-type]
        ),
        current_period_start=_from_unix(int(current_period_start)),  # type: ignore[arg-type]
        current_period_end=_from_unix(int(current_period_end)),  # type: ignore[arg-type]
        livemode=bool(livemode),
        on_pro_price=bool(on_pro_price),
        version=int(version),  # type: ignore[arg-type]
        confirmed_at=datetime.fromisoformat(str(confirmed_at)),
        past_due_since=None if past_due_since is None else _from_unix(int(past_due_since)),  # type: ignore[arg-type]
    )


async def get_sync_state(
    conn: aiosqlite.Connection, *, subscription_id: str, event_id: str | None
) -> SyncState:
    """Report whether an event was already applied, and the current version.

    Parameters
    ----------
    conn
        Open database connection.
    subscription_id
        Stripe's subscription identifier.
    event_id
        The Stripe event about to be processed, or None for a reconciliation
        read that is not driven by an event.

    Returns
    -------
    SyncState
        `event_processed` is True when this exact event has already been
        applied; `version` is the subscription's stored version, 0 for one this
        database has never seen.

    Notes
    -----
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
    """Store a snapshot if its event is new and the version is unchanged.

    Parameters
    ----------
    conn
        Open database connection.
    snapshot
        What Stripe says about the subscription.
    event_id
        The Stripe event driving this write, or None for a reconciliation write
        that records no event.
    event_type
        The event's type, stored alongside it. Must be None exactly when
        `event_id` is.
    expected_version
        The version the caller read before fetching from Stripe. The write
        applies only if the stored version still matches.
    now
        When the write is made.

    Returns
    -------
    ApplyResult
        APPLIED with the new version and record; DUPLICATE_EVENT when this event
        was already processed; VERSION_CONFLICT when the stored version has
        moved on. The last two write nothing.

    Notes
    -----
    Both guards and the write share one transaction, so a redelivered event and
    a concurrent writer are each resolved exactly once. A losing event is NOT
    marked processed, so its retry re-fetches rather than being swallowed.
    `event_id` and `event_type` are both given for a webhook-driven write and
    both None for a reconciliation write, which has no event to deduplicate.

    Checks run in this order, and the order is part of the contract: a
    duplicate event is reported as a duplicate even if the version has since
    moved, because "already applied" is the more useful answer to a redelivery
    -- it tells the caller to stop, where a conflict would tell it to retry.

    The stored `past_due_since` is derived here, from the row being replaced
    and the new snapshot (aura.billing.entitlement.next_unpaid_since), under
    the same lock and in the same transaction as the swap itself.
    """
    if now.tzinfo is None:
        raise ValueError(f"now must be a timezone-aware datetime, got {now!r}")
    if (event_id is None) != (event_type is None):
        raise ValueError("event_id and event_type must be given together or not at all")
    if not 0 <= expected_version < _MAX_SQLITE_INTEGER:
        raise ValueError(f"expected_version out of range: {expected_version}")

    confirmed_at = utc_iso(now)
    # Truncated to the whole seconds the row stores before anything is derived
    # from it, so the anchor carried into the next write is exactly the one
    # read back after a restart.
    period_start = _from_unix(_to_unix(snapshot.current_period_start))
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
                f"SELECT {_RECORD_COLUMNS} FROM guild_subscriptions WHERE subscription_id = ?",
                (snapshot.subscription_id,),
            ) as cursor:
                row = await cursor.fetchone()
            existing = None if row is None else _record_from_row(row)
            current_version = existing.version if existing is not None else 0
            if current_version != expected_version:
                return ApplyResult(ApplyOutcome.VERSION_CONFLICT, current_version, None, None)

            new_version = current_version + 1
            past_due_since = next_unpaid_since(
                existing,
                status=snapshot.status,
                latest_invoice_status=snapshot.latest_invoice_status,
                current_period_start=period_start,
            )
            values = (
                snapshot.guild_id,
                snapshot.customer_id,
                snapshot.purchaser_user_id,
                snapshot.status.value,
                int(snapshot.cancel_at_period_end),
                None if snapshot.cancel_at is None else _to_unix(snapshot.cancel_at),
                int(snapshot.collection_paused),
                None
                if snapshot.latest_invoice_status is None
                else snapshot.latest_invoice_status.value,
                _to_unix(period_start),
                _to_unix(snapshot.current_period_end),
                int(snapshot.livemode),
                new_version,
                confirmed_at,
                int(snapshot.on_pro_price),
                None if past_due_since is None else _to_unix(past_due_since),
            )
            if existing is None:
                await conn.execute(
                    "INSERT INTO guild_subscriptions (guild_id, customer_id, purchaser_user_id, "
                    "status, cancel_at_period_end, cancel_at, collection_paused, "
                    "latest_invoice_status, current_period_start, current_period_end, livemode, "
                    "version, confirmed_at, on_pro_price, past_due_since, subscription_id, "
                    "first_seen_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (*values, snapshot.subscription_id, confirmed_at),
                )
            else:
                cursor = await conn.execute(
                    "UPDATE guild_subscriptions SET guild_id = ?, customer_id = ?, "
                    "purchaser_user_id = ?, status = ?, cancel_at_period_end = ?, cancel_at = ?, "
                    "collection_paused = ?, latest_invoice_status = ?, current_period_start = ?, "
                    "current_period_end = ?, livemode = ?, version = ?, confirmed_at = ?, "
                    "on_pro_price = ?, past_due_since = ? "
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

    previous_guild_id = existing.guild_id if existing is not None else None
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
        current_period_start=period_start,
        current_period_end=_from_unix(_to_unix(snapshot.current_period_end)),
        livemode=snapshot.livemode,
        on_pro_price=snapshot.on_pro_price,
        version=new_version,
        confirmed_at=datetime.fromisoformat(confirmed_at),
        past_due_since=past_due_since,
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
    """Return every stored subscription, for building the plan gate at startup.

    Parameters
    ----------
    conn
        Open database connection.

    Returns
    -------
    list[SubscriptionRecord]
        Every row. Deliberately not guild-scoped: the caller is the process-wide
        plan gate, which needs all of them at once.
    """
    async with connection_lock(conn):
        async with conn.execute(
            f"SELECT {_RECORD_COLUMNS} FROM guild_subscriptions ORDER BY subscription_id"
        ) as cursor:
            rows = await cursor.fetchall()
    return [_record_from_row(row) for row in rows]


async def count_processed_events(conn: aiosqlite.Connection) -> int:
    """Return how many Stripe events have been applied.

    Parameters
    ----------
    conn
        Open database connection.

    Returns
    -------
    int
        Rows in the event ledger. A diagnostic, and the idempotency tests'
        witness that a redelivered event adds nothing.
    """
    async with connection_lock(conn):
        async with conn.execute("SELECT COUNT(*) FROM stripe_processed_events") as cursor:
            row = await cursor.fetchone()
    return int(row[0]) if row else 0


async def verify_subscriptions_schema(conn: aiosqlite.Connection) -> None:
    """Add the audit-fix columns to a guild_subscriptions table created before them.

    Parameters
    ----------
    conn
        Open database connection, after `init_schema`.

    Returns
    -------
    None

    Notes
    -----
    Called once at startup, after init_schema, for the reason
    verify_pending_facts_schema is: `CREATE TABLE IF NOT EXISTS` cannot reshape
    a table that already exists, so a database created by the Phase 4c deploy
    would keep its older shape and every snapshot write would fail on the
    missing columns.

    Purely additive and therefore migrated in place: no existing column
    changes, `past_due_since` is NULL for every pre-existing row -- which
    aura.billing.entitlement.unpaid_since reads exactly as the pre-fix rule did
    -- and `on_pro_price` is 1, which is what every pre-existing row already
    meant (see _ADDITIVE_COLUMNS).

    Idempotent: a table already at the current shape adds nothing, and a
    partially migrated one (a crash between the two ALTERs) is completed in one
    pass. A database with no such table at all passes untouched.
    """
    async with connection_lock(conn):
        async with conn.execute("PRAGMA table_info(guild_subscriptions)") as cursor:
            columns = {row[1] for row in await cursor.fetchall()}
        if not columns:
            return
        missing = [(name, ddl) for name, ddl in _ADDITIVE_COLUMNS if name not in columns]
        for name, ddl in missing:
            await conn.execute(f"ALTER TABLE guild_subscriptions ADD COLUMN {name} {ddl}")
        if missing:
            await conn.commit()
