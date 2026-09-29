"""The bot's internal billing API: how Stripe's news reaches the process that owns the database.

WHY THIS EXISTS AT ALL (Phase 4c's architecture decision, recorded in full in
reports/phase-4c.txt and web/README.md). Subscription state, unlike guild
membership in Phase 4b, is not something Discord can answer -- it has to be
persisted, and the bot has to read it on every message. Two designs were on the
table: let the web backend write a table in data/aura.db directly, or keep this
process the only writer of its own database and give the web backend a narrow
way to hand it snapshots. This module is the second. The web container keeps no
volume and no database handle, so the internet-facing process that receives
Stripe's webhooks cannot reach a single fact; the bot keeps its one connection
and its one lock; and the event ID and the snapshot it caused commit in one
transaction in the database that holds both.

WHAT IT EXPOSES, and nothing more:

  POST /internal/v1/subscriptions/sync-state  -- was this event applied, and
                                                 what version is stored?
  POST /internal/v1/subscriptions/apply       -- store a snapshot, compare-and-swap
  POST /internal/v1/guilds/plans              -- plans for up to 200 guilds

No route deletes anything, lists every guild, or reads a fact.

AUTHENTICATION COMES FIRST. Every request -- including one for a route that does
not exist -- must present the shared secret before anything else happens: no
routing answer, no body read, no validation message. An unauthenticated caller
learns only "401", so this API cannot be mapped from outside it. The listener is
additionally bound only to the internal network in the shipped compose files.

Every error is a short machine-readable code, never an exception text: the
caller is another service, and whatever it needs to diagnose a failure is in
this process's log, not in the response.
"""

from __future__ import annotations

import hmac
import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Any

import aiosqlite
from aiohttp import web
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    StringConstraints,
    ValidationError,
    field_validator,
    model_validator,
)

from aura.billing.apply import apply_snapshot
from aura.billing.entitlement import (
    GuildPlan,
    InvoiceStatus,
    SubscriptionRecord,
    SubscriptionStatus,
)
from aura.billing.plan_gate import PlanGate
from aura.db.connection import utc_now
from aura.db.subscriptions import ApplyOutcome, SubscriptionSnapshot, get_sync_state

logger = logging.getLogger(__name__)

INTERNAL_API_PREFIX = "/internal/v1"

# A snapshot is a few hundred bytes and a 200-guild plan request a few kilobytes.
# 64 KiB is generous for both and still refuses a body big enough to matter.
MAX_REQUEST_BYTES = 64 * 1024

# Discord caps a user at 200 guilds, and the web backend only ever asks about
# the guilds one user manages.
MAX_GUILD_IDS_PER_REQUEST = 200

# datetime's own ceiling (9999-12-31T23:59:59Z). A larger Unix time cannot be
# represented, so it is refused at the boundary instead of raising deep inside.
MAX_UNIX_SECONDS = 253_402_300_799

_MAX_SQLITE_INTEGER = 2**63 - 1

# Stripe object IDs are a prefix and an alphanumeric body. The body is bounded
# so an ID can never become a megabyte of text in a log line or a row.
SUBSCRIPTION_ID_PATTERN = r"^sub_[A-Za-z0-9]{1,200}$"
CUSTOMER_ID_PATTERN = r"^cus_[A-Za-z0-9]{1,200}$"
EVENT_ID_PATTERN = r"^evt_[A-Za-z0-9]{1,200}$"
EVENT_TYPE_PATTERN = r"^[a-z_]{1,50}(\.[a-z_]{1,50}){1,4}$"
SNOWFLAKE_PATTERN = r"^[1-9][0-9]{0,18}$"

_SubscriptionId = Annotated[str, StringConstraints(strict=True, pattern=SUBSCRIPTION_ID_PATTERN)]
_CustomerId = Annotated[str, StringConstraints(strict=True, pattern=CUSTOMER_ID_PATTERN)]
_EventId = Annotated[str, StringConstraints(strict=True, pattern=EVENT_ID_PATTERN)]
_EventType = Annotated[str, StringConstraints(strict=True, pattern=EVENT_TYPE_PATTERN)]
_Snowflake = Annotated[str, StringConstraints(strict=True, pattern=SNOWFLAKE_PATTERN)]
_UnixSeconds = Annotated[StrictInt, Field(ge=0, le=MAX_UNIX_SECONDS)]


class InternalApiErrorCode(StrEnum):
    """Every error code this API returns."""

    UNAUTHORIZED = "unauthorized"
    INVALID_REQUEST = "invalid_request"
    UNSUPPORTED_MEDIA_TYPE = "unsupported_media_type"
    PAYLOAD_TOO_LARGE = "payload_too_large"
    NOT_FOUND = "not_found"
    METHOD_NOT_ALLOWED = "method_not_allowed"
    INTERNAL_ERROR = "internal_error"


def _error(code: InternalApiErrorCode, status: int) -> web.Response:
    return web.json_response({"error": code.value}, status=status)


def _snowflake_to_int(value: str) -> int:
    number = int(value)
    if number > _MAX_SQLITE_INTEGER:
        raise ValueError("a Discord ID must fit a signed 64-bit integer")
    return number


class _RequestModel(BaseModel):
    # extra="forbid": a field this API does not know is a caller that disagrees
    # with it about the contract, and the honest answer to that is a refusal,
    # not a silently ignored key.
    model_config = ConfigDict(extra="forbid", frozen=True)


class SnapshotPayload(_RequestModel):
    """A subscription snapshot on the wire. IDs are strings: snowflakes exceed 2^53."""

    subscription_id: _SubscriptionId
    customer_id: _CustomerId
    guild_id: _Snowflake
    purchaser_user_id: _Snowflake | None
    status: SubscriptionStatus
    cancel_at_period_end: StrictBool
    cancel_at: _UnixSeconds | None
    collection_paused: StrictBool
    latest_invoice_status: InvoiceStatus | None
    current_period_start: _UnixSeconds
    current_period_end: _UnixSeconds
    livemode: StrictBool

    @field_validator("guild_id", "purchaser_user_id")
    @classmethod
    def _fits_the_database(cls, value: str | None) -> str | None:
        if value is not None:
            _snowflake_to_int(value)
        return value

    @model_validator(mode="after")
    def _period_is_ordered(self) -> SnapshotPayload:
        if self.current_period_end < self.current_period_start:
            raise ValueError("current_period_end must not precede current_period_start")
        return self

    def to_snapshot(self) -> SubscriptionSnapshot:
        """Convert the wire shape into the database layer's snapshot.

        Returns
        -------
        SubscriptionSnapshot
            The same subscription with snowflakes parsed to ints and Unix seconds
            parsed to timezone-aware datetimes. Validation has already happened on
            this model, so the conversion cannot fail here.
        """
        return SubscriptionSnapshot(
            subscription_id=self.subscription_id,
            guild_id=int(self.guild_id),
            customer_id=self.customer_id,
            purchaser_user_id=None
            if self.purchaser_user_id is None
            else int(self.purchaser_user_id),
            status=self.status,
            cancel_at_period_end=self.cancel_at_period_end,
            cancel_at=None if self.cancel_at is None else _datetime_from_unix(self.cancel_at),
            collection_paused=self.collection_paused,
            latest_invoice_status=self.latest_invoice_status,
            current_period_start=_datetime_from_unix(self.current_period_start),
            current_period_end=_datetime_from_unix(self.current_period_end),
            livemode=self.livemode,
        )


class SyncStateRequest(_RequestModel):
    """Body of POST /subscriptions/sync-state."""

    subscription_id: _SubscriptionId
    event_id: _EventId | None


class ApplyRequest(_RequestModel):
    """Body of POST /subscriptions/apply. Every key is required, even when its value is null."""

    event_id: _EventId | None
    event_type: _EventType | None
    expected_version: Annotated[StrictInt, Field(ge=0, lt=_MAX_SQLITE_INTEGER)]
    snapshot: SnapshotPayload

    @model_validator(mode="after")
    def _event_fields_travel_together(self) -> ApplyRequest:
        if (self.event_id is None) != (self.event_type is None):
            raise ValueError("event_id and event_type must be given together or not at all")
        return self


class PlansRequest(_RequestModel):
    """Body of POST /guilds/plans."""

    guild_ids: Annotated[
        list[_Snowflake], Field(min_length=1, max_length=MAX_GUILD_IDS_PER_REQUEST)
    ]

    @field_validator("guild_ids")
    @classmethod
    def _unique_and_representable(cls, value: list[str]) -> list[str]:
        if len(set(value)) != len(value):
            raise ValueError("guild_ids must not repeat")
        for guild_id in value:
            _snowflake_to_int(guild_id)
        return value


def _datetime_from_unix(seconds: int) -> datetime:
    return datetime.fromtimestamp(seconds, tz=UTC)


def _unix_or_none(moment: datetime | None) -> int | None:
    return None if moment is None else int(moment.timestamp())


def plan_payload(plan: GuildPlan, records: tuple[SubscriptionRecord, ...]) -> dict[str, Any]:
    """Serialise one guild's plan for the web backend.

    Parameters
    ----------
    plan
        The decided plan and its standing.
    records
        Every subscription known for the guild.

    Returns
    -------
    dict[str, Any]
        A JSON-ready mapping. The subscriptions list carries the customer ID and
        the purchaser, which the web backend needs to open Stripe's billing
        portal for the right person and MUST NOT forward to a browser -- that
        projection is the web backend's job (`aura_web.routes.billing`), stated
        here so it is not mistaken for public data.
    """
    standing = plan.standing
    return {
        "tier": plan.tier.value,
        "basis": plan.basis.value,
        "standing": standing.standing.value,
        "access_until": _unix_or_none(standing.access_until),
        "paid_through": _unix_or_none(standing.paid_through),
        "in_force_subscription_count": len(standing.granting_subscription_ids),
        "subscriptions": [
            {
                "subscription_id": record.subscription_id,
                "customer_id": record.customer_id,
                "purchaser_user_id": (
                    None if record.purchaser_user_id is None else str(record.purchaser_user_id)
                ),
                "status": record.status.value,
                "grants_access": record.subscription_id in standing.granting_subscription_ids,
            }
            for record in records
        ],
    }


@dataclass(frozen=True)
class _Dependencies:
    conn: aiosqlite.Connection
    gate: PlanGate
    clock: Callable[[], datetime]


_DEPENDENCIES = web.AppKey("billing_dependencies", _Dependencies)


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """json.loads hook: a key given twice is refused rather than last-one-wins.

    Two parsers disagreeing about which duplicate counts is the textbook
    request-smuggling shape; refusing both readings removes the question.
    """
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate key {key!r}")
        result[key] = value
    return result


def _reject_non_finite_constant(name: str) -> Any:
    raise ValueError(f"non-standard JSON constant {name}")


async def _read_json_object(request: web.Request) -> dict[str, Any] | web.Response:
    """Read and parse a JSON object body, or return the refusal to send instead."""
    if request.content_type != "application/json":
        return _error(InternalApiErrorCode.UNSUPPORTED_MEDIA_TYPE, 415)
    try:
        raw = await request.read()
    except web.HTTPRequestEntityTooLarge:
        return _error(InternalApiErrorCode.PAYLOAD_TOO_LARGE, 413)
    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_non_finite_constant,
        )
    except (UnicodeDecodeError, ValueError, RecursionError):
        return _error(InternalApiErrorCode.INVALID_REQUEST, 400)
    if not isinstance(payload, dict):
        return _error(InternalApiErrorCode.INVALID_REQUEST, 400)
    return payload


async def _handle_sync_state(request: web.Request) -> web.Response:
    parsed = await _read_json_object(request)
    if isinstance(parsed, web.Response):
        return parsed
    try:
        body = SyncStateRequest.model_validate(parsed)
    except ValidationError:
        return _error(InternalApiErrorCode.INVALID_REQUEST, 400)
    dependencies = request.app[_DEPENDENCIES]
    state = await get_sync_state(
        dependencies.conn, subscription_id=body.subscription_id, event_id=body.event_id
    )
    return web.json_response({"event_processed": state.event_processed, "version": state.version})


async def _handle_apply(request: web.Request) -> web.Response:
    parsed = await _read_json_object(request)
    if isinstance(parsed, web.Response):
        return parsed
    try:
        body = ApplyRequest.model_validate(parsed)
        snapshot = body.snapshot.to_snapshot()
    except ValidationError:
        return _error(InternalApiErrorCode.INVALID_REQUEST, 400)
    dependencies = request.app[_DEPENDENCIES]
    result = await apply_snapshot(
        dependencies.conn,
        dependencies.gate,
        snapshot=snapshot,
        event_id=body.event_id,
        event_type=body.event_type,
        expected_version=body.expected_version,
        now=dependencies.clock(),
    )
    if result.outcome is ApplyOutcome.APPLIED:
        return web.json_response({"outcome": "applied", "version": result.version})
    if result.outcome is ApplyOutcome.DUPLICATE_EVENT:
        return web.json_response({"outcome": "duplicate", "version": result.version})
    return web.json_response({"outcome": "version_conflict", "version": result.version}, status=409)


async def _handle_plans(request: web.Request) -> web.Response:
    parsed = await _read_json_object(request)
    if isinstance(parsed, web.Response):
        return parsed
    try:
        body = PlansRequest.model_validate(parsed)
    except ValidationError:
        return _error(InternalApiErrorCode.INVALID_REQUEST, 400)
    gate = request.app[_DEPENDENCIES].gate
    plans = {
        guild_id: plan_payload(gate.plan_for(int(guild_id)), gate.records_for(int(guild_id)))
        for guild_id in body.guild_ids
    }
    return web.json_response({"plans": plans})


def _authentication_middleware(secret: str) -> Callable[..., Awaitable[web.StreamResponse]]:
    expected = f"Bearer {secret}".encode("ascii")

    @web.middleware
    async def middleware(
        request: web.Request, handler: Callable[[web.Request], Awaitable[web.StreamResponse]]
    ) -> web.StreamResponse:
        """Authenticate one request, then run it with every failure mapped to JSON.

        Parameters
        ----------
        request
            The incoming request.
        handler
            The route handler to run once the secret checks out.

        Returns
        -------
        web.StreamResponse
            The handler's response, or a JSON error body: 401 for a missing or
            wrong secret, 404/405/413 for the routing and size failures aiohttp
            raises, and 500 for anything else, which is logged with a
            traceback.

        Notes
        -----
        The secret is compared with `hmac.compare_digest` against the fully
        encoded header value, so the comparison is constant-time and a header
        that is not valid UTF-8 compares as empty rather than raising. Exactly
        one Authorization header is accepted; zero or several compare as empty.
        """
        presented_values = request.headers.getall("Authorization", [])
        presented = presented_values[0] if len(presented_values) == 1 else ""
        try:
            presented_bytes = presented.encode("utf-8")
        except UnicodeEncodeError:
            presented_bytes = b""
        if not hmac.compare_digest(presented_bytes, expected):
            logger.warning(
                "Refused an internal billing API request from %s: missing or wrong shared secret",
                request.remote,
            )
            return _error(InternalApiErrorCode.UNAUTHORIZED, 401)

        try:
            return await handler(request)
        except web.HTTPNotFound:
            return _error(InternalApiErrorCode.NOT_FOUND, 404)
        except web.HTTPMethodNotAllowed:
            return _error(InternalApiErrorCode.METHOD_NOT_ALLOWED, 405)
        except web.HTTPRequestEntityTooLarge:
            return _error(InternalApiErrorCode.PAYLOAD_TOO_LARGE, 413)
        except web.HTTPException as exc:
            return web.json_response({"error": "http_error"}, status=exc.status)
        except Exception:
            logger.exception("Internal billing API request to %s failed", request.path[:100])
            return _error(InternalApiErrorCode.INTERNAL_ERROR, 500)

    return middleware


def create_internal_api_app(
    conn: aiosqlite.Connection,
    gate: PlanGate,
    *,
    secret: str,
    clock: Callable[[], datetime] = utc_now,
) -> web.Application:
    """Build the aiohttp application, without starting a listener.

    Parameters
    ----------
    conn
        Open database connection.
    gate
        The plan gate to read and write through.
    secret
        The shared secret every request must present.
    clock
        Reads the current time.

    Returns
    -------
    web.Application
        A fully routed application behind the authentication middleware.

    Notes
    -----
    Separate from starting it so tests can drive it directly, with no socket.
    """
    app = web.Application(
        middlewares=[_authentication_middleware(secret)],
        client_max_size=MAX_REQUEST_BYTES,
    )
    app[_DEPENDENCIES] = _Dependencies(conn=conn, gate=gate, clock=clock)
    app.router.add_post(f"{INTERNAL_API_PREFIX}/subscriptions/sync-state", _handle_sync_state)
    app.router.add_post(f"{INTERNAL_API_PREFIX}/subscriptions/apply", _handle_apply)
    app.router.add_post(f"{INTERNAL_API_PREFIX}/guilds/plans", _handle_plans)
    return app


@dataclass
class InternalApiServer:
    """A running internal API listener, owned by the bot client for its lifetime."""

    runner: web.AppRunner

    @property
    def bound_port(self) -> int:
        """Return the port actually bound.

        Returns
        -------
        int
            The listening port. Meaningful when started on port 0, which is what the
            tests do to avoid colliding with anything on the host.
        """
        address = self.runner.addresses[0]
        return int(address[1])

    async def stop(self) -> None:
        """Stop accepting requests and release the socket.

        Returns
        -------
        None

        Notes
        -----
        Idempotent in the sense that aiohttp's cleanup is: calling it on an already
        stopped runner does nothing.
        """
        await self.runner.cleanup()


async def start_internal_api(
    conn: aiosqlite.Connection,
    gate: PlanGate,
    *,
    secret: str,
    host: str,
    port: int,
    clock: Callable[[], datetime] = utc_now,
) -> InternalApiServer:
    """Start listening on the configured address.

    Parameters
    ----------
    conn
        Open database connection.
    gate
        The plan gate to read and write through.
    secret
        The shared secret every request must present.
    host, port
        Address to bind. Port 0 binds an arbitrary free port.
    clock
        Reads the current time.

    Returns
    -------
    InternalApiServer
        The running listener, owned by the bot client for its lifetime.

    Raises
    ------
    OSError
        If the address cannot be bound. Deliberately not swallowed: a bot that
        cannot accept subscription updates should fail visibly at startup rather
        than run looking healthy while every guild silently stays on Free.
    """
    app = create_internal_api_app(conn, gate, secret=secret, clock=clock)
    # aiohttp's access log is off: every request here is service-to-service,
    # and the apply path already logs each state change with its reason.
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    site = web.TCPSite(runner, host=host, port=port)
    try:
        await site.start()
    except BaseException:
        await runner.cleanup()
        raise
    return InternalApiServer(runner=runner)
