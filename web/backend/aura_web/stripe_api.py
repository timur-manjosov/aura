"""Every call this service makes to Stripe's API, and the validation of every answer.

The same shape as aura_web.discord_api, for the same reason: one module where a
third party's response becomes something the rest of the service trusts, field
by field, so an API that changes a type or omits a field produces a clean
refusal here rather than a surprising value three layers up.

httpx directly, not the stripe SDK's HTTP client. The SDK is used in this
service for exactly one thing -- verifying webhook signatures, where running
Stripe's own implementation is the point (aura_web.routes.stripe_webhook).
For the four API calls below, a thin httpx client keeps one validating boundary
per upstream (the Discord one already exists), lets the test suite drive the
real client against a stand-in over ASGI exactly as it drives Discord, and pins
the API version in one visible constant rather than in whatever version an SDK
release defaults to.

The two failure classes are the ones the callers need to tell apart:
StripeUnavailableError means "could not find out" (a network failure, a 5xx, a
429, a body that makes no sense -- worth retrying), StripeRejectedError means
Stripe refused the request itself (a 4xx: a revoked key, an unknown price, a
subscription from another account -- retrying will not help, an operator must
look). Neither carries a response body or a credential in its message.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, TypeGuard
from urllib.parse import urlparse

import httpx

from aura_web.permissions import parse_snowflake

logger = logging.getLogger(__name__)

# Pinned. Every field this module reads was checked against this version's
# documentation -- in particular, since 2025-03-31.basil the billing period
# lives on each subscription ITEM, not on the subscription, and an invoice's
# subscription sits under parent.subscription_details.
STRIPE_API_VERSION = "2026-08-26.dahlia"

# The metadata keys the checkout attaches server-side and every subscription
# event is resolved through. Namespaced, so an unrelated subscription in the
# same Stripe account that happens to carry a "guild_id" is never mistaken for
# an Aura one.
GUILD_METADATA_KEY = "aura_guild_id"
PURCHASER_METADATA_KEY = "aura_discord_user_id"

# Stripe's dashboard label for sessions this integration creates, with the
# random-letter suffix Stripe's guidance asks for so it is unambiguous.
CHECKOUT_INTEGRATION_IDENTIFIER = "aura_guild_pro_checkout_qxmvrtlk"

# The only hosts a redirect this service hands to a browser may point at.
# Checked on every URL Stripe returns, so a misconfigured AURA_WEB_STRIPE_API_BASE
# -- or anything impersonating Stripe on that path -- cannot turn "subscribe" into
# a redirect to an arbitrary page. A deployment using a Stripe custom domain for
# Checkout adds that host here, deliberately.
CHECKOUT_HOSTS = frozenset({"checkout.stripe.com"})
BILLING_PORTAL_HOSTS = frozenset({"billing.stripe.com"})

SUBSCRIPTION_LIST_PAGE_SIZE = 100
# 50 pages of 100 is 5,000 subscriptions -- far past this project's scale, so
# hitting it means a cursor that does not advance, not a large deployment.
MAX_SUBSCRIPTION_LIST_PAGES = 50

KNOWN_SUBSCRIPTION_STATUSES = frozenset(
    {
        "active",
        "trialing",
        "past_due",
        "unpaid",
        "canceled",
        "incomplete",
        "incomplete_expired",
        "paused",
    }
)
KNOWN_INVOICE_STATUSES = frozenset({"draft", "open", "paid", "uncollectible", "void"})

# Only an invoice that bills the subscription's own period -- its first invoice
# or a renewal -- says whether that period was paid. A voided proration or
# one-off invoice (billing_reason subscription_update, manual, ...) says
# nothing about the period, and must not end a paying guild's Pro.
PERIOD_BILLING_REASONS = frozenset({"subscription_create", "subscription_cycle"})

# The bot stores Unix times as datetimes and IDs as signed 64-bit integers;
# anything past either ceiling is refused here rather than there.
MAX_UNIX_SECONDS = 253_402_300_799
_MAX_SQLITE_INTEGER = 2**63 - 1

_STRIPE_ID_BODY = re.compile(r"^[A-Za-z0-9]{1,200}$")


class StripeAPIError(Exception):
    """Base class for every failure talking to Stripe."""


class StripeUnavailableError(StripeAPIError):
    """Stripe could not be reached, or answered in a way this service cannot act on."""


class StripeRejectedError(StripeAPIError):
    """Stripe refused the request itself (HTTP 4xx) -- not transient, an operator must look."""

    def __init__(self, message: str, *, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


def is_stripe_id(value: object, prefix: str) -> TypeGuard[str]:
    """Whether value is a Stripe ID with the given prefix and a plain alphanumeric body.

    Parameters
    ----------
    value
        The candidate, whose JSON type is not guaranteed.
    prefix
        The expected Stripe prefix, e.g. ``sub_``.

    Returns
    -------
    TypeGuard[str]
        True only for a string with that prefix and a plain alphanumeric body,
        which is what makes it safe to interpolate into a URL path.
    """
    return (
        isinstance(value, str)
        and value.startswith(f"{prefix}_")
        and _STRIPE_ID_BODY.match(value[len(prefix) + 1 :]) is not None
    )


def _object_id(value: object, prefix: str) -> str | None:
    """An ID from a field that is either the bare ID or the expanded object carrying it."""
    if isinstance(value, dict):
        value = value.get("id")
    if is_stripe_id(value, prefix):
        return value
    return None


def _strict_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _unix_seconds(value: object, *, field_name: str) -> int:
    seconds = _strict_int(value)
    if seconds is None or not 0 <= seconds <= MAX_UNIX_SECONDS:
        raise StripeUnavailableError(f"Stripe subscription has no usable {field_name}")
    return seconds


@dataclass(frozen=True)
class CheckoutSession:
    """A created Checkout Session: its ID and the Stripe-hosted page to send the browser to."""

    session_id: str
    url: str


@dataclass(frozen=True)
class SubscriptionSnapshot:
    """What Stripe says about one subscription right now, reduced to what Aura decides with."""

    subscription_id: str
    customer_id: str
    # None when the subscription carries no (valid) Aura guild metadata -- it
    # is then not an Aura subscription and is never pushed to the bot.
    guild_id: str | None
    purchaser_user_id: str | None
    status: str
    cancel_at_period_end: bool
    cancel_at: int | None
    collection_paused: bool
    latest_invoice_status: str | None
    current_period_start: int
    current_period_end: int
    livemode: bool

    def internal_api_payload(self) -> dict[str, Any]:
        """The snapshot in the bot internal API's wire shape. Requires a guild.

        Returns
        -------
        dict[str, Any]
            The snapshot in the bot internal API's wire shape.

        Raises
        ------
        ValueError
            If the snapshot carries no guild. A subscription with no guild cannot
            be applied to one, and sending it would be a silent no-op.
        """
        if self.guild_id is None:
            raise ValueError("a snapshot without an Aura guild is never sent to the bot")
        return {
            "subscription_id": self.subscription_id,
            "customer_id": self.customer_id,
            "guild_id": self.guild_id,
            "purchaser_user_id": self.purchaser_user_id,
            "status": self.status,
            "cancel_at_period_end": self.cancel_at_period_end,
            "cancel_at": self.cancel_at,
            "collection_paused": self.collection_paused,
            "latest_invoice_status": self.latest_invoice_status,
            "current_period_start": self.current_period_start,
            "current_period_end": self.current_period_end,
            "livemode": self.livemode,
        }


def _discord_id_from_metadata(
    metadata: dict[str, Any], key: str, subscription_id: str
) -> str | None:
    raw = metadata.get(key)
    if raw is None:
        return None
    # ASCII only: parse_snowflake accepts any Unicode digit, and a guild ID in
    # Arabic-Indic digits normalising to a real guild is not a metadata value
    # this service's own checkout could ever have written.
    parsed = parse_snowflake(raw) if isinstance(raw, str) and raw.isascii() else None
    # Canonical form only: parse_snowflake forgives surrounding whitespace and
    # leading zeros, and a metadata value that is not literally the ID it
    # normalises to was not written by this service's checkout.
    if parsed is None or parsed != raw or int(parsed) <= 0 or int(parsed) > _MAX_SQLITE_INTEGER:
        logger.warning(
            "Stripe subscription %s carries an unusable %s metadata value; treating it as absent",
            subscription_id,
            key,
        )
        return None
    return parsed


def parse_subscription(payload: object) -> SubscriptionSnapshot:
    """Validate a Subscription object (API version 2026-08-26.dahlia) into a snapshot.

    Parameters
    ----------
    payload
        A Stripe Subscription object (API version 2026-08-26.dahlia).

    Returns
    -------
    SubscriptionSnapshot
        The fields Aura stores, with every ID and timestamp validated.

    Raises
    ------
    StripeAPIError
        If any required field is missing, is the wrong type, or carries an ID
        that does not look like a Stripe ID. Nothing is guessed or defaulted.

    Notes
    -----
    An unrecognised status is a refusal, not a guess: a status this code has
    never seen could mean "paid" or "not paid", and the caller answers Stripe
    with a retryable failure so the last known state stays in force while an
    operator reads the log -- the asymmetric rule applied to the unknown.
    """
    if not isinstance(payload, dict) or payload.get("object") != "subscription":
        raise StripeUnavailableError("Stripe did not return a subscription object")

    subscription_id = payload.get("id")
    if not is_stripe_id(subscription_id, "sub"):
        raise StripeUnavailableError("Stripe subscription has no usable id")
    assert isinstance(subscription_id, str)

    status = payload.get("status")
    if status not in KNOWN_SUBSCRIPTION_STATUSES:
        raise StripeUnavailableError(
            f"Stripe subscription {subscription_id} has an unrecognised status"
        )

    customer_id = _object_id(payload.get("customer"), "cus")
    if customer_id is None:
        raise StripeUnavailableError(
            f"Stripe subscription {subscription_id} has no usable customer"
        )

    cancel_at_period_end = payload.get("cancel_at_period_end")
    livemode = payload.get("livemode")
    if not isinstance(cancel_at_period_end, bool) or not isinstance(livemode, bool):
        raise StripeUnavailableError(f"Stripe subscription {subscription_id} has malformed flags")

    raw_cancel_at = payload.get("cancel_at")
    cancel_at = (
        None if raw_cancel_at is None else _unix_seconds(raw_cancel_at, field_name="cancel_at")
    )

    pause_collection = payload.get("pause_collection")
    if pause_collection is not None and not isinstance(pause_collection, dict):
        raise StripeUnavailableError(
            f"Stripe subscription {subscription_id} has malformed pause_collection"
        )

    latest_invoice = payload.get("latest_invoice")
    latest_invoice_status: str | None = None
    if isinstance(latest_invoice, dict):
        raw_invoice_status = latest_invoice.get("status")
        if raw_invoice_status is not None and raw_invoice_status not in KNOWN_INVOICE_STATUSES:
            raise StripeUnavailableError(
                f"Stripe subscription {subscription_id} has an invoice with an unrecognised status"
            )
        if latest_invoice.get("billing_reason") in PERIOD_BILLING_REASONS:
            latest_invoice_status = raw_invoice_status
    elif latest_invoice is not None and not isinstance(latest_invoice, str):
        raise StripeUnavailableError(
            f"Stripe subscription {subscription_id} has a malformed latest_invoice"
        )

    items = payload.get("items")
    item_data = items.get("data") if isinstance(items, dict) else None
    if not isinstance(item_data, list) or not item_data:
        raise StripeUnavailableError(f"Stripe subscription {subscription_id} has no items")
    starts: list[int] = []
    ends: list[int] = []
    for item in item_data:
        if not isinstance(item, dict):
            raise StripeUnavailableError(
                f"Stripe subscription {subscription_id} has a malformed item"
            )
        starts.append(
            _unix_seconds(item.get("current_period_start"), field_name="current_period_start")
        )
        ends.append(_unix_seconds(item.get("current_period_end"), field_name="current_period_end"))
    # With several items the EARLIEST end is taken: access may only rely on the
    # part of the subscription that is paid for longest being paid for at all
    # if every part is. Each item's start precedes its end, so the minimum
    # start always precedes the minimum end.
    period_start = min(starts)
    period_end = min(ends)
    if period_end < period_start:
        raise StripeUnavailableError(
            f"Stripe subscription {subscription_id} has an inverted period"
        )

    metadata = payload.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}

    return SubscriptionSnapshot(
        subscription_id=subscription_id,
        customer_id=customer_id,
        guild_id=_discord_id_from_metadata(metadata, GUILD_METADATA_KEY, subscription_id),
        purchaser_user_id=_discord_id_from_metadata(
            metadata, PURCHASER_METADATA_KEY, subscription_id
        ),
        status=status,
        cancel_at_period_end=cancel_at_period_end,
        cancel_at=cancel_at,
        collection_paused=pause_collection is not None,
        latest_invoice_status=latest_invoice_status,
        current_period_start=period_start,
        current_period_end=period_end,
        livemode=livemode,
    )


def _https_url_on(value: object, hosts: frozenset[str], *, context: str) -> str:
    if not isinstance(value, str):
        raise StripeUnavailableError(f"Stripe's {context} has no URL")
    parsed = urlparse(value)
    if (
        parsed.scheme != "https"
        or parsed.hostname not in hosts
        or parsed.username
        or parsed.password
    ):
        raise StripeUnavailableError(f"Stripe's {context} URL is not on an expected Stripe host")
    return value


def parse_checkout_session(payload: object) -> CheckoutSession:
    """Validate a created Checkout Session down to its ID and its hosted URL.

    Parameters
    ----------
    payload
        A created Stripe Checkout Session object.

    Returns
    -------
    CheckoutSession
        Its ID and its hosted URL, which are the only two fields used.

    Raises
    ------
    StripeAPIError
        If either is missing or malformed.
    """
    if not isinstance(payload, dict) or payload.get("object") != "checkout.session":
        raise StripeUnavailableError("Stripe did not return a checkout session")
    session_id = payload.get("id")
    if not is_stripe_id(session_id, "cs_test") and not is_stripe_id(session_id, "cs_live"):
        raise StripeUnavailableError("Stripe checkout session has no usable id")
    assert isinstance(session_id, str)
    return CheckoutSession(
        session_id=session_id,
        url=_https_url_on(payload.get("url"), CHECKOUT_HOSTS, context="checkout session"),
    )


class StripeClient:
    """Thin, validating wrapper over the four Stripe endpoints Phase 4c needs."""

    def __init__(
        self,
        http: httpx.AsyncClient,
        *,
        api_base: str,
        secret_key: str,
        price_id: str,
        checkout_success_url: str,
        checkout_cancel_url: str,
        portal_return_url: str,
    ) -> None:
        self._http = http
        self._api_base = api_base.rstrip("/")
        self._secret_key = secret_key
        self._price_id = price_id
        self._checkout_success_url = checkout_success_url
        self._checkout_cancel_url = checkout_cancel_url
        self._portal_return_url = portal_return_url

    async def create_checkout_session(
        self, *, guild_id: str, purchaser_user_id: str, idempotency_key: str
    ) -> CheckoutSession:
        """Create a Stripe-hosted subscription checkout bound to one guild, server-side.

        Parameters
        ----------
        guild_id
            The guild the subscription will be bound to, written into Stripe
            metadata server-side so a browser cannot choose it.
        purchaser_user_id
            Who is paying, recorded for the billing portal.
        idempotency_key
            Stripe's own replay guard, so a double-submitted checkout creates one
            session rather than two.

        Returns
        -------
        CheckoutSession
            The created session's ID and hosted URL.

        Raises
        ------
        StripeAPIError
            For a transport failure, an error status, or a body Stripe returned in
            a shape this code refuses to guess at.

        Notes
        -----
        Everything that decides what is bought and for whom is set here, from
        values the caller already validated: the price comes from configuration,
        the guild and the purchaser from the session and the authorization
        check. Nothing in the request body reaches this call except the guild
        ID that check approved.

        The guild is written in two places on purpose. The session's own
        metadata and client_reference_id identify the checkout; the
        SUBSCRIPTION's metadata (subscription_data) is what every later event
        -- renewal, failure, cancellation -- is resolved through, and it cannot
        be changed by the customer, only by the operator in Stripe.

        No payment_method_types: Stripe chooses eligible methods from the
        dashboard's settings. Delayed methods (bank debits) are handled by the
        entitlement rules, which treat a voided invoice as unpaid.
        """
        payload = await self._request(
            "POST",
            "/v1/checkout/sessions",
            data={
                "mode": "subscription",
                "line_items[0][price]": self._price_id,
                "line_items[0][quantity]": "1",
                "success_url": self._checkout_success_url,
                "cancel_url": self._checkout_cancel_url,
                "client_reference_id": guild_id,
                f"metadata[{GUILD_METADATA_KEY}]": guild_id,
                f"metadata[{PURCHASER_METADATA_KEY}]": purchaser_user_id,
                f"subscription_data[metadata][{GUILD_METADATA_KEY}]": guild_id,
                f"subscription_data[metadata][{PURCHASER_METADATA_KEY}]": purchaser_user_id,
                "integration_identifier": CHECKOUT_INTEGRATION_IDENTIFIER,
            },
            idempotency_key=idempotency_key,
            context="checkout session creation",
        )
        return parse_checkout_session(payload)

    async def retrieve_subscription(self, subscription_id: str) -> SubscriptionSnapshot:
        """Fetch one subscription's current state, with its latest invoice expanded.

        Parameters
        ----------
        subscription_id
            Stripe's subscription identifier.

        Returns
        -------
        SubscriptionSnapshot
            Its current state, with the latest invoice expanded so the payment
            status is known in the same round trip.

        Raises
        ------
        StripeAPIError
            For a transport failure, an error status, or a body Stripe returned in
            a shape this code refuses to guess at.
        """
        if not is_stripe_id(subscription_id, "sub"):
            raise ValueError("not a subscription ID")
        payload = await self._request(
            "GET",
            f"/v1/subscriptions/{subscription_id}",
            params=[("expand[]", "latest_invoice")],
            context="subscription retrieval",
        )
        snapshot = parse_subscription(payload)
        if snapshot.subscription_id != subscription_id:
            raise StripeUnavailableError(
                "Stripe returned a different subscription than was asked for"
            )
        return snapshot

    async def list_aura_subscription_ids(self) -> list[str]:
        """Every subscription in the account that carries Aura guild metadata, for reconciliation.

        Returns
        -------
        list[str]
            Every subscription in the account carrying Aura guild metadata, for
            reconciliation. Paged through in full.

        Raises
        ------
        StripeAPIError
            For a transport failure, an error status, or a body Stripe returned in
            a shape this code refuses to guess at.

        Notes
        -----
        Only IDs: the reconciler re-fetches each one individually, through the
        same compare-and-swap an event uses, so a listing that is minutes old by
        the time a page is processed can never be what gets stored.
        """
        found: list[str] = []
        starting_after: str | None = None
        for _ in range(MAX_SUBSCRIPTION_LIST_PAGES):
            params = [("status", "all"), ("limit", str(SUBSCRIPTION_LIST_PAGE_SIZE))]
            if starting_after is not None:
                params.append(("starting_after", starting_after))
            payload = await self._request(
                "GET", "/v1/subscriptions", params=params, context="subscription listing"
            )
            if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
                raise StripeUnavailableError("Stripe's subscription listing was not a list")
            entries = payload["data"]
            last_id: str | None = None
            for entry in entries:
                if not isinstance(entry, dict) or not is_stripe_id(entry.get("id"), "sub"):
                    continue
                last_id = entry["id"]
                metadata = entry.get("metadata")
                if isinstance(metadata, dict) and GUILD_METADATA_KEY in metadata:
                    found.append(entry["id"])
            if payload.get("has_more") is not True or last_id is None or last_id == starting_after:
                return found
            starting_after = last_id
        logger.warning(
            "Stripe's subscription listing exceeded %d pages; reconciling what was collected",
            MAX_SUBSCRIPTION_LIST_PAGES,
        )
        return found

    async def create_portal_session(self, *, customer_id: str) -> str:
        """Create a Stripe billing portal session for one customer and return its URL.

        Parameters
        ----------
        customer_id
            The Stripe customer to open the portal for.

        Returns
        -------
        str
            The portal URL to redirect the browser to. Single-use and short-lived,
            which is why it is never cached.

        Raises
        ------
        StripeAPIError
            For a transport failure, an error status, or a body Stripe returned in
            a shape this code refuses to guess at.
        """
        if not is_stripe_id(customer_id, "cus"):
            raise ValueError("not a customer ID")
        payload = await self._request(
            "POST",
            "/v1/billing_portal/sessions",
            data={"customer": customer_id, "return_url": self._portal_return_url},
            context="billing portal session creation",
        )
        if not isinstance(payload, dict) or payload.get("object") != "billing_portal.session":
            raise StripeUnavailableError("Stripe did not return a billing portal session")
        return _https_url_on(
            payload.get("url"), BILLING_PORTAL_HOSTS, context="billing portal session"
        )

    async def _request(
        self,
        method: str,
        path: str,
        *,
        context: str,
        data: dict[str, str] | None = None,
        params: list[tuple[str, str]] | None = None,
        idempotency_key: str | None = None,
    ) -> object:
        headers = {
            "Authorization": f"Bearer {self._secret_key}",
            "Stripe-Version": STRIPE_API_VERSION,
        }
        if idempotency_key is not None:
            headers["Idempotency-Key"] = idempotency_key
        try:
            response = await self._http.request(
                method,
                f"{self._api_base}{path}",
                data=data,
                params=tuple(params) if params is not None else None,
                headers=headers,
            )
        except httpx.HTTPError as exc:
            # The exception type only: an httpx error's text can include the
            # request URL, and this module never lets anything request-shaped
            # reach a log line.
            raise StripeUnavailableError(
                f"Could not reach Stripe for {context} ({type(exc).__name__})"
            ) from exc

        if response.status_code == 429 or response.status_code >= 500:
            raise StripeUnavailableError(
                f"Stripe returned HTTP {response.status_code} for {context}"
            )
        if response.status_code >= 400:
            raise StripeRejectedError(
                f"Stripe rejected {context} (HTTP {response.status_code}{_error_summary(response)})",
                status_code=response.status_code,
            )
        if response.status_code >= 300:
            raise StripeUnavailableError(
                f"Stripe answered {context} with a redirect, which is never followed"
            )
        try:
            return response.json()
        except ValueError as exc:
            raise StripeUnavailableError(f"Stripe returned a non-JSON body for {context}") from exc


def _error_summary(response: httpx.Response) -> str:
    """Stripe's error type and code, for the operator's log -- never its free-text message.

    Stripe's human-readable error message can quote request parameters back,
    and this module's rule is that nothing request-shaped reaches a log line.
    The type and code ("invalid_request_error", "resource_missing") are enough
    to diagnose, and are fixed vocabulary.
    """
    try:
        body = response.json()
    except ValueError:
        return ""
    error = body.get("error") if isinstance(body, dict) else None
    if not isinstance(error, dict):
        return ""
    parts = [
        str(error[key])
        for key in ("type", "code")
        if isinstance(error.get(key), str) and re.fullmatch(r"[a-z_]{1,64}", error[key])
    ]
    return f": {'/'.join(parts)}" if parts else ""
