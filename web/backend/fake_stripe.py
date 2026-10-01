"""A stand-in for the Stripe API endpoints Phase 4c calls, shaped like API version 2026-08-26.dahlia.

Not a mock of this project's code: a real ASGI application the real
aura_web.stripe_api.StripeClient talks to over HTTP, with the real form
encoding, the real bearer authentication, the real Stripe-Version and
Idempotency-Key headers, and the dahlia object shapes -- in particular the
billing period on each subscription ITEM, which is exactly the kind of detail
a mock written from memory gets wrong.

It also signs webhook bodies, and does so with its own few lines of HMAC
written from Stripe's documentation rather than by calling the stripe SDK the
service under test uses to verify them. A double that shared the verifier's
implementation could not catch that implementation being wrong.

Where the real sandbox run (reports/phase-4c-stripe-verification.md, section 7)
found this stand-in more permissive or simpler than Stripe in a way that could
hide a bug, it now behaves like Stripe: the Managed Payments refusal, the
Invoices (read) permission an expansion needs, cancel_at beside
cancel_at_period_end, a first invoice billed as subscription_create, renewals
that start as a draft, the invoice's copy of the subscription metadata, the
portal URL's shape, and test-clock subscriptions missing from a plain listing.

Excluded from the backend image by .dockerignore, like fake_discord.py.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import parse_qs

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

DAY = 24 * 3600

# The Pro price every stand-in subscription is on unless a test says otherwise.
# The same value as FakeStripeState.price_id, which the web settings are
# configured with, so a subscription created by a completed checkout counts.
DEFAULT_PRICE_ID = "price_auraProMonthlyFake"

# How long after its creation Stripe finalizes a renewal's draft invoice at the
# latest (automatically_finalizes_at in the sandbox: created + 72 h). Stripe
# usually finalizes after about an hour; a test decides when by calling
# FakeSubscription.settle_renewal.
DRAFT_FINALIZATION_WINDOW = 72 * 3600

# Stripe's refusal of a card-only checkout on an account whose default is
# Managed Payments, verbatim as far as the sandbox runs recorded it
# (reports/phase-4c-stripe-verification.txt step A3, and the Phase 4c-F01 fixes
# acceptance run, which captured 400 characters of it). Stripe sends it with a
# type and a request_log_url only -- no code and no param.
MANAGED_PAYMENTS_REFUSAL = (
    "Unsupported parameter: payment_method_types. Managed Payments, which is enabled by "
    "default on your account, handles this parameter for you. Remove payment_method_types, "
    "or pass managed_payments[enabled]=false to disable it for this request. You can "
    "configure whether Managed Payments is enabled by default at "
    "https://dashboard.stripe.com/acct_fakeStandIn/test/settings/managed-payments."
)

# Stripe's permission error, shaped as the sandbox returned it -- including the
# key's last four characters, which is exactly why the service must never log
# Stripe's message text.
INVOICE_READ_REFUSAL = (
    "Permission denied. The provided key 'rk_test_...{key_tail}' does not have the required "
    "permissions for this endpoint on account 'acct_fakeStandIn'. Enabling Invoices Read "
    "('invoice_read') permissions on this key would allow this request to continue."
)


def sign_webhook(payload: bytes, secret: str, *, timestamp: int | None = None) -> str:
    """A Stripe-Signature header value, computed exactly as Stripe documents it.

    HMAC-SHA256 over "<timestamp>.<raw body>" keyed with the endpoint secret,
    sent as t=<timestamp>,v1=<hex digest>.
    """
    moment = int(time.time()) if timestamp is None else timestamp
    digest = hmac.new(
        secret.encode("utf-8"), f"{moment}.".encode() + payload, hashlib.sha256
    ).hexdigest()
    return f"t={moment},v1={digest}"


def _stripe_id(prefix: str) -> str:
    return f"{prefix}_{secrets.token_hex(12)}"


@dataclass
class FakeSubscription:
    """One subscription the stand-in knows about."""

    id: str
    customer: str
    metadata: dict[str, str]
    current_period_start: int
    current_period_end: int
    status: str = "active"
    cancel_at_period_end: bool = False
    cancel_at: int | None = None
    pause_collection: dict[str, Any] | None = None
    latest_invoice_id: str = field(default_factory=lambda: _stripe_id("in"))
    latest_invoice_status: str | None = "paid"
    # A new subscription's first invoice, as Stripe bills it. Renewals switch
    # this to subscription_cycle (see renew).
    latest_invoice_billing_reason: str = "subscription_create"
    # Set while the latest invoice is a renewal draft, as Stripe does.
    latest_invoice_finalizes_at: int | None = None
    livemode: bool = False
    price_id: str = DEFAULT_PRICE_ID
    quantity: int | None = 1
    # A subscription on a test clock is left out of a listing that does not
    # ask for its clock, as in the real sandbox (V-08).
    test_clock: str | None = None

    @property
    def effective_cancel_at(self) -> int | None:
        """cancel_at as Stripe reports it: the period end once cancel_at_period_end is set."""
        if self.cancel_at is not None:
            return self.cancel_at
        return self.current_period_end if self.cancel_at_period_end else None

    def invoice_object(self) -> dict[str, Any]:
        """The latest invoice in the dahlia shape: the subscription only under parent."""
        return {
            "id": self.latest_invoice_id,
            "object": "invoice",
            "status": self.latest_invoice_status,
            "billing_reason": self.latest_invoice_billing_reason,
            "customer": self.customer,
            "automatically_finalizes_at": self.latest_invoice_finalizes_at,
            "parent": {
                "type": "subscription_details",
                "subscription_details": {
                    "subscription": self.id,
                    # Stripe copies the subscription's metadata onto each of
                    # its invoices.
                    "metadata": dict(self.metadata),
                },
                "quote_details": None,
            },
        }

    def renew(self) -> None:
        """Roll into the next period the way Stripe does: a new invoice, still a draft.

        The subscription stays in its status and the new period's invoice is a
        draft until settle_renewal -- the window in which the subscription
        reads in force with an invoice that is neither paid nor failed.
        """
        length = self.current_period_end - self.current_period_start
        self.current_period_start = self.current_period_end
        self.current_period_end = self.current_period_start + length
        self.latest_invoice_id = _stripe_id("in")
        self.latest_invoice_status = "draft"
        self.latest_invoice_billing_reason = "subscription_cycle"
        self.latest_invoice_finalizes_at = self.current_period_start + DRAFT_FINALIZATION_WINDOW

    def settle_renewal(self, *, paid: bool) -> None:
        """Finalize the renewal draft and charge it: paid, or open with the subscription past_due."""
        self.latest_invoice_finalizes_at = None
        if paid:
            self.latest_invoice_status = "paid"
            if self.status == "past_due":
                self.status = "active"
        else:
            self.latest_invoice_status = "open"
            self.status = "past_due"

    def to_object(self, *, expand_invoice: bool) -> dict[str, Any]:
        """The subscription as the dahlia API serialises it."""
        latest_invoice: object = self.latest_invoice_id
        if expand_invoice:
            latest_invoice = self.invoice_object()
        return {
            "id": self.id,
            "object": "subscription",
            "customer": self.customer,
            "status": self.status,
            "metadata": dict(self.metadata),
            "cancel_at_period_end": self.cancel_at_period_end,
            "cancel_at": self.effective_cancel_at,
            "canceled_at": None,
            "pause_collection": self.pause_collection,
            "latest_invoice": latest_invoice,
            "livemode": self.livemode,
            "test_clock": self.test_clock,
            "items": {
                "object": "list",
                "data": [
                    {
                        "id": _stripe_id("si"),
                        "object": "subscription_item",
                        "price": {"id": self.price_id, "object": "price"},
                        "quantity": self.quantity,
                        "current_period_start": self.current_period_start,
                        "current_period_end": self.current_period_end,
                    }
                ],
            },
        }


@dataclass
class FakeStripeState:
    """Everything the stand-in answers with, plus the failure switches tests flip."""

    secret_key: str = "sk_test_fakeStripeSecretKey0000000000000000"
    webhook_secret: str = "whsec_fakeWebhookSigningSecret000000000000"
    price_id: str = DEFAULT_PRICE_ID
    livemode: bool = False

    checkout_sessions: dict[str, dict[str, Any]] = field(default_factory=dict)
    subscriptions: dict[str, FakeSubscription] = field(default_factory=dict)
    portal_sessions: list[dict[str, str]] = field(default_factory=list)
    idempotency: dict[str, tuple[dict[str, str], dict[str, Any]]] = field(default_factory=dict)

    request_log: list[str] = field(default_factory=list)
    received_forms: list[dict[str, str]] = field(default_factory=list)
    received_headers: list[dict[str, str]] = field(default_factory=list)

    fail_checkout_status: int | None = None
    fail_retrieve_status: int | None = None
    fail_list_status: int | None = None
    fail_portal_status: int | None = None
    # The account setting "Managed Payments by default" (on in a sandbox as
    # Stripe delivers it). While on, a checkout naming payment_method_types
    # is refused unless it switches Managed Payments off for itself.
    managed_payments_default: bool = True
    # Whether the key holds Invoices (read). Without it Stripe refuses both
    # GET /v1/invoices and every expansion of latest_invoice (V-02).
    invoice_read_permission: bool = True
    # Replaces the hosted page URL a created session carries, so a test can
    # make "Stripe" hand back a link to somewhere that is not Stripe.
    checkout_url_override: str | None = None
    # Awaited inside every subscription retrieval before answering, so a test
    # can interleave two concurrent syncs at an exact point.
    retrieve_hook: Callable[[str], Awaitable[None]] | None = None

    def add_subscription(
        self,
        *,
        guild_id: str | None,
        purchaser_user_id: str | None,
        now: int,
        period_days: int = 30,
        **overrides: Any,
    ) -> FakeSubscription:
        """Create a subscription directly, as an operator or a completed checkout would."""
        metadata: dict[str, str] = {}
        if guild_id is not None:
            metadata["aura_guild_id"] = guild_id
        if purchaser_user_id is not None:
            metadata["aura_discord_user_id"] = purchaser_user_id
        subscription = FakeSubscription(
            id=_stripe_id("sub"),
            customer=_stripe_id("cus"),
            metadata=metadata,
            current_period_start=now,
            current_period_end=now + period_days * DAY,
            livemode=overrides.pop("livemode", self.livemode),
            price_id=overrides.pop("price_id", self.price_id),
            **overrides,
        )
        self.subscriptions[subscription.id] = subscription
        return subscription

    def complete_checkout(self, session_id: str, *, now: int) -> FakeSubscription:
        """What paying on the hosted page does: a subscription carrying the session's subscription_data."""
        session = self.checkout_sessions[session_id]
        metadata = session["subscription_data_metadata"]
        subscription = self.add_subscription(
            guild_id=metadata.get("aura_guild_id"),
            purchaser_user_id=metadata.get("aura_discord_user_id"),
            now=now,
        )
        session["status"] = "complete"
        session["payment_status"] = "paid"
        session["subscription"] = subscription.id
        session["customer"] = subscription.customer
        return subscription

    def event(
        self, event_type: str, data_object: dict[str, Any], *, event_id: str | None = None
    ) -> dict[str, Any]:
        """A snapshot event envelope, as Stripe delivers it."""
        return {
            "id": event_id or _stripe_id("evt"),
            "object": "event",
            "api_version": "2026-08-26.dahlia",
            "created": int(time.time()),
            "livemode": self.livemode,
            "pending_webhooks": 1,
            "type": event_type,
            "data": {"object": data_object},
        }

    def subscription_event(
        self, event_type: str, subscription_id: str, **kwargs: Any
    ) -> dict[str, Any]:
        """A customer.subscription.* event carrying the subscription as it is now."""
        return self.event(
            event_type,
            self.subscriptions[subscription_id].to_object(expand_invoice=False),
            **kwargs,
        )

    def checkout_completed_event(self, session_id: str, **kwargs: Any) -> dict[str, Any]:
        """checkout.session.completed for a stored session."""
        session = self.checkout_sessions[session_id]
        return self.event("checkout.session.completed", _session_object(session), **kwargs)

    def invoice_event(
        self, event_type: str, subscription_id: str, *, legacy_shape: bool = False, **kwargs: Any
    ) -> dict[str, Any]:
        """An invoice.* event for a subscription's invoice, in the dahlia or the pre-basil shape."""
        invoice = self.subscriptions[subscription_id].invoice_object()
        if legacy_shape:
            del invoice["parent"]
            invoice["subscription"] = subscription_id
        return self.event(event_type, invoice, **kwargs)

    def signed(self, event: dict[str, Any], *, timestamp: int | None = None) -> tuple[bytes, str]:
        """The raw body and a valid Stripe-Signature header for an event."""
        body = json.dumps(event).encode("utf-8")
        return body, sign_webhook(body, self.webhook_secret, timestamp=timestamp)


def _session_object(session: dict[str, Any]) -> dict[str, Any]:
    form = session.get("form", {})
    return {
        "id": session["id"],
        "object": "checkout.session",
        "mode": "subscription",
        "status": session["status"],
        "payment_status": session.get("payment_status", "unpaid"),
        "url": session["url"],
        "client_reference_id": session["client_reference_id"],
        "metadata": dict(session["metadata"]),
        "subscription": session.get("subscription"),
        "customer": session.get("customer"),
        "livemode": session["livemode"],
        # Echoed as Stripe echoes them, so a test can read back what the
        # session was actually created with.
        "managed_payments": {"enabled": session.get("managed_payments_enabled", False)},
        "payment_method_types": [
            value for key, value in sorted(form.items()) if key.startswith("payment_method_types[")
        ],
        "integration_identifier": form.get("integration_identifier"),
    }


def _stripe_error(
    status: int, error_type: str, code: str, *, param: str | None = None
) -> JSONResponse:
    error = {"type": error_type, "code": code, "message": "The stand-in refused this request."}
    if param is not None:
        error["param"] = param
    return JSONResponse({"error": error}, status_code=status)


def _managed_payments_enabled(form: dict[str, str], *, account_default: bool) -> bool | None:
    """Whether Managed Payments applies to this checkout; None for a value Stripe would refuse."""
    requested = form.get("managed_payments[enabled]")
    if requested is None:
        return account_default
    return {"true": True, "false": False}.get(requested)


def _invoice_read_refused(state: FakeStripeState) -> JSONResponse:
    return JSONResponse(
        {
            "error": {
                "type": "invalid_request_error",
                "code": "more_permissions_required",
                "message": INVOICE_READ_REFUSAL.format(key_tail=state.secret_key[-4:]),
            }
        },
        status_code=403,
    )


def create_fake_stripe(state: FakeStripeState) -> FastAPI:
    """Build the ASGI application. Routes mirror Stripe's own paths under /v1."""
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    def authorized(request: Request) -> bool:
        state.received_headers.append(
            {key.lower(): value for key, value in request.headers.items()}
        )
        return request.headers.get("authorization") == f"Bearer {state.secret_key}"

    async def read_form(request: Request) -> dict[str, str] | None:
        raw = (await request.body()).decode("utf-8")
        parsed = parse_qs(raw, keep_blank_values=True)
        if any(len(values) != 1 for values in parsed.values()):
            return None
        form = {key: values[0] for key, values in parsed.items()}
        state.received_forms.append(form)
        return form

    @app.post("/v1/checkout/sessions")
    async def create_checkout_session(request: Request) -> JSONResponse:
        state.request_log.append("POST /v1/checkout/sessions")
        if not authorized(request):
            return _stripe_error(401, "invalid_request_error", "api_key_invalid")
        if state.fail_checkout_status is not None:
            return _stripe_error(state.fail_checkout_status, "api_error", "internal")
        form = await read_form(request)
        if form is None:
            return _stripe_error(400, "invalid_request_error", "parameter_duplicate")
        if form.get("line_items[0][price]") != state.price_id:
            return _stripe_error(
                400, "invalid_request_error", "resource_missing", param="line_items[0][price]"
            )
        managed_payments = _managed_payments_enabled(
            form, account_default=state.managed_payments_default
        )
        if managed_payments is None:
            return _stripe_error(
                400,
                "invalid_request_error",
                "parameter_invalid_boolean",
                param="managed_payments[enabled]",
            )
        if managed_payments and any(key.startswith("payment_method_types[") for key in form):
            # The fields the sandbox's answer carried, and no others: no code
            # and no param, so nothing names the refused parameter but the
            # message the service never logs.
            return JSONResponse(
                {
                    "error": {
                        "type": "invalid_request_error",
                        "message": MANAGED_PAYMENTS_REFUSAL,
                        "request_log_url": "https://dashboard.stripe.com/test/logs/req_fakeStandIn",
                    }
                },
                status_code=400,
            )

        key = request.headers.get("idempotency-key")
        if key is not None and key in state.idempotency:
            previous_form, previous_response = state.idempotency[key]
            if previous_form != form:
                return _stripe_error(400, "idempotency_error", "idempotency_key_in_use")
            return JSONResponse(previous_response)

        session_id = f"cs_test_{secrets.token_hex(16)}"
        session = {
            "id": session_id,
            "status": "open",
            "url": state.checkout_url_override or f"https://checkout.stripe.com/c/pay/{session_id}",
            "client_reference_id": form.get("client_reference_id"),
            "metadata": {
                key[len("metadata[") : -1]: value
                for key, value in form.items()
                if key.startswith("metadata[") and key.endswith("]")
            },
            "subscription_data_metadata": {
                key[len("subscription_data[metadata][") : -1]: value
                for key, value in form.items()
                if key.startswith("subscription_data[metadata][") and key.endswith("]")
            },
            "livemode": state.livemode,
            "managed_payments_enabled": managed_payments,
            "form": form,
        }
        state.checkout_sessions[session_id] = session
        response = _session_object(session)
        if key is not None:
            state.idempotency[key] = (form, response)
        return JSONResponse(response)

    @app.get("/v1/subscriptions/{subscription_id}")
    async def retrieve_subscription(subscription_id: str, request: Request) -> JSONResponse:
        state.request_log.append(f"GET /v1/subscriptions/{subscription_id}")
        if not authorized(request):
            return _stripe_error(401, "invalid_request_error", "api_key_invalid")
        if state.retrieve_hook is not None:
            await state.retrieve_hook(subscription_id)
        if state.fail_retrieve_status is not None:
            return _stripe_error(state.fail_retrieve_status, "api_error", "internal")
        subscription = state.subscriptions.get(subscription_id)
        if subscription is None:
            return _stripe_error(404, "invalid_request_error", "resource_missing")
        expand = request.query_params.getlist("expand[]")
        if "latest_invoice" in expand and not state.invoice_read_permission:
            return _invoice_read_refused(state)
        return JSONResponse(subscription.to_object(expand_invoice="latest_invoice" in expand))

    @app.get("/v1/subscriptions")
    async def list_subscriptions(request: Request) -> JSONResponse:
        state.request_log.append("GET /v1/subscriptions")
        if not authorized(request):
            return _stripe_error(401, "invalid_request_error", "api_key_invalid")
        if state.fail_list_status is not None:
            return _stripe_error(state.fail_list_status, "api_error", "internal")
        if "data.latest_invoice" in request.query_params.getlist("expand[]") and (
            not state.invoice_read_permission
        ):
            return _invoice_read_refused(state)
        limit = int(request.query_params.get("limit", "10"))
        test_clock = request.query_params.get("test_clock")
        ordered = [
            subscription
            for subscription in state.subscriptions.values()
            if subscription.test_clock == test_clock
        ]
        starting_after = request.query_params.get("starting_after")
        if starting_after is not None:
            ids = [subscription.id for subscription in ordered]
            ordered = ordered[ids.index(starting_after) + 1 :] if starting_after in ids else []
        page = ordered[:limit]
        return JSONResponse(
            {
                "object": "list",
                "data": [subscription.to_object(expand_invoice=False) for subscription in page],
                "has_more": len(ordered) > limit,
            }
        )

    @app.get("/v1/invoices")
    async def list_invoices(request: Request) -> JSONResponse:
        state.request_log.append("GET /v1/invoices")
        if not authorized(request):
            return _stripe_error(401, "invalid_request_error", "api_key_invalid")
        if not state.invoice_read_permission:
            return _invoice_read_refused(state)
        limit = int(request.query_params.get("limit", "10"))
        invoices = [subscription.invoice_object() for subscription in state.subscriptions.values()]
        return JSONResponse(
            {"object": "list", "data": invoices[:limit], "has_more": len(invoices) > limit}
        )

    @app.post("/v1/billing_portal/sessions")
    async def create_portal_session(request: Request) -> JSONResponse:
        state.request_log.append("POST /v1/billing_portal/sessions")
        if not authorized(request):
            return _stripe_error(401, "invalid_request_error", "api_key_invalid")
        if state.fail_portal_status is not None:
            return _stripe_error(state.fail_portal_status, "api_error", "internal")
        form = await read_form(request)
        if form is None or form.get("customer") not in {
            sub.customer for sub in state.subscriptions.values()
        }:
            return _stripe_error(400, "invalid_request_error", "resource_missing")
        state.portal_sessions.append(form)
        return JSONResponse(
            {
                "id": _stripe_id("bps"),
                "object": "billing_portal.session",
                "customer": form["customer"],
                "return_url": form.get("return_url"),
                # The real shape: the session's bearer secret in the query.
                "url": f"https://billing.stripe.com/p/session?secret=test_{secrets.token_hex(12)}",
            }
        )

    return app
