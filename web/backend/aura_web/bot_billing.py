"""The web backend's client for the bot's internal billing API.

This service never opens Aura's database (web/README.md, Phase 4b). Subscription
state is handed to the bot process, which stays the only writer of its own
file: this module is the whole of that conversation. Three calls, each
answered by a strictly validated shape -- the bot is a trusted peer, but a
peer on the other side of a network hop, and "trusted" is not a reason to let
a malformed answer become a plausible-looking plan.

Every failure -- unreachable, a 5xx, a refused secret, a body that makes no
sense -- is one exception type. The callers' answer is the same in each case:
do not act on billing state this service could not establish.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import httpx

from aura_web.stripe_api import SubscriptionSnapshot

logger = logging.getLogger(__name__)

INTERNAL_API_PREFIX = "/internal/v1"

_KNOWN_TIERS = frozenset({"free", "pro"})
_KNOWN_BASES = frozenset({"billing_not_enforced", "complimentary", "subscription"})
_KNOWN_STANDINGS = frozenset(
    {"no_subscription", "ended", "active", "renewal_pending", "canceling", "payment_grace"}
)


class BotBillingError(Exception):
    """The bot's billing state could not be read or changed."""


class ApplyOutcome(StrEnum):
    """What the bot did with one snapshot."""

    APPLIED = "applied"
    DUPLICATE = "duplicate"
    VERSION_CONFLICT = "version_conflict"


@dataclass(frozen=True)
class SyncState:
    """Whether an event was already applied, and the subscription's stored version."""

    event_processed: bool
    version: int


@dataclass(frozen=True)
class ApplyResult:
    """The bot's answer to one apply request."""

    outcome: ApplyOutcome
    version: int


@dataclass(frozen=True)
class SubscriptionView:
    """One of a guild's subscriptions, as the bot reports it. Never sent to a browser as is."""

    subscription_id: str
    customer_id: str
    purchaser_user_id: str | None
    status: str
    grants_access: bool


@dataclass(frozen=True)
class GuildPlanView:
    """One guild's plan, as the bot decided it."""

    tier: str
    basis: str
    standing: str
    access_until: int | None
    paid_through: int | None
    in_force_subscription_count: int
    subscriptions: tuple[SubscriptionView, ...]


def _strict_int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _parse_plan(raw: object) -> GuildPlanView:
    if not isinstance(raw, dict):
        raise BotBillingError("a plan in the bot's answer was not an object")
    tier, basis, standing = raw.get("tier"), raw.get("basis"), raw.get("standing")
    if tier not in _KNOWN_TIERS or basis not in _KNOWN_BASES or standing not in _KNOWN_STANDINGS:
        raise BotBillingError(
            "a plan in the bot's answer has an unrecognised tier, basis or standing"
        )
    count = _strict_int(raw.get("in_force_subscription_count"))
    if count is None or count < 0:
        raise BotBillingError("a plan in the bot's answer has no usable subscription count")
    access_until = raw.get("access_until")
    paid_through = raw.get("paid_through")
    if (access_until is not None and _strict_int(access_until) is None) or (
        paid_through is not None and _strict_int(paid_through) is None
    ):
        raise BotBillingError("a plan in the bot's answer has a malformed date")
    raw_subscriptions = raw.get("subscriptions")
    if not isinstance(raw_subscriptions, list):
        raise BotBillingError("a plan in the bot's answer has no subscription list")
    subscriptions: list[SubscriptionView] = []
    for entry in raw_subscriptions:
        if not isinstance(entry, dict):
            raise BotBillingError("a subscription in the bot's answer was not an object")
        subscription_id, customer_id = entry.get("subscription_id"), entry.get("customer_id")
        purchaser, status, grants = (
            entry.get("purchaser_user_id"),
            entry.get("status"),
            entry.get("grants_access"),
        )
        if (
            not isinstance(subscription_id, str)
            or not isinstance(customer_id, str)
            or not (purchaser is None or isinstance(purchaser, str))
            or not isinstance(status, str)
            or not isinstance(grants, bool)
        ):
            raise BotBillingError("a subscription in the bot's answer is malformed")
        subscriptions.append(
            SubscriptionView(subscription_id, customer_id, purchaser, status, grants)
        )
    return GuildPlanView(
        tier=tier,
        basis=basis,
        standing=standing,
        access_until=access_until,
        paid_through=paid_through,
        in_force_subscription_count=count,
        subscriptions=tuple(subscriptions),
    )


class BotBillingClient:
    """Calls the bot's internal billing API with the shared secret."""

    def __init__(self, http: httpx.AsyncClient, *, base_url: str, secret: str) -> None:
        self._http = http
        self._base_url = base_url.rstrip("/")
        self._secret = secret

    async def get_sync_state(self, *, subscription_id: str, event_id: str | None) -> SyncState:
        """Ask whether an event was applied and which version of the subscription is stored.

        Parameters
        ----------
        subscription_id
            Stripe's subscription identifier.
        event_id
            The event about to be processed, or None for a reconciliation read.

        Returns
        -------
        SyncState
            Whether that event was already applied, and the version the bot has
            stored.

        Raises
        ------
        BotBillingError
            For a transport failure, a rejected shared secret, or a body the bot's
            internal API returned in an unexpected shape.
        """
        body = await self._post(
            "/subscriptions/sync-state",
            {"subscription_id": subscription_id, "event_id": event_id},
            accepted_statuses=(200,),
        )
        event_processed = body.get("event_processed")
        version = _strict_int(body.get("version"))
        if not isinstance(event_processed, bool) or version is None or version < 0:
            raise BotBillingError("the bot's sync state answer is malformed")
        return SyncState(event_processed=event_processed, version=version)

    async def apply_snapshot(
        self,
        *,
        event_id: str | None,
        event_type: str | None,
        expected_version: int,
        snapshot: SubscriptionSnapshot,
    ) -> ApplyResult:
        """Hand the bot one snapshot, to be stored only if the version is still expected_version.

        Parameters
        ----------
        event_id, event_type
            The Stripe event driving this write, or None for reconciliation.
        expected_version
            The version read before fetching from Stripe. The bot stores the
            snapshot only if its stored version still matches.
        snapshot
            What Stripe says about the subscription.

        Returns
        -------
        ApplyResult
            Applied, duplicate, or version conflict -- the bot's own verdict,
            relayed unchanged.

        Raises
        ------
        BotBillingError
            For a transport failure, a rejected shared secret, or a body the bot's
            internal API returned in an unexpected shape.
        """
        body = await self._post(
            "/subscriptions/apply",
            {
                "event_id": event_id,
                "event_type": event_type,
                "expected_version": expected_version,
                "snapshot": snapshot.internal_api_payload(),
            },
            accepted_statuses=(200, 409),
        )
        outcome = body.get("outcome")
        version = _strict_int(body.get("version"))
        if (
            outcome not in {member.value for member in ApplyOutcome}
            or version is None
            or version < 0
        ):
            raise BotBillingError("the bot's apply answer is malformed")
        return ApplyResult(outcome=ApplyOutcome(outcome), version=version)

    async def get_guild_plans(self, guild_ids: list[str]) -> dict[str, GuildPlanView]:
        """Plans for the given guilds (at most 200), keyed by guild ID.

        Parameters
        ----------
        guild_ids
            The guilds to ask about. At most 200, which is the bot's own limit.

        Returns
        -------
        dict[str, GuildPlanView]
            One entry per requested guild, keyed by guild ID.

        Raises
        ------
        BotBillingError
            For a transport failure, a rejected shared secret, or a body the bot's
            internal API returned in an unexpected shape.
        """
        if not guild_ids:
            return {}
        body = await self._post("/guilds/plans", {"guild_ids": guild_ids}, accepted_statuses=(200,))
        raw_plans = body.get("plans")
        if not isinstance(raw_plans, dict) or set(raw_plans) != set(guild_ids):
            raise BotBillingError(
                "the bot's plan answer does not cover exactly the guilds asked about"
            )
        return {guild_id: _parse_plan(raw_plans[guild_id]) for guild_id in guild_ids}

    async def _post(
        self, path: str, payload: dict[str, Any], *, accepted_statuses: tuple[int, ...]
    ) -> dict[str, Any]:
        try:
            response = await self._http.post(
                f"{self._base_url}{INTERNAL_API_PREFIX}{path}",
                json=payload,
                headers={"Authorization": f"Bearer {self._secret}"},
            )
        except httpx.HTTPError as exc:
            raise BotBillingError(
                f"Could not reach the bot's billing API ({type(exc).__name__})"
            ) from exc

        if response.status_code == 401:
            # Not an outage: the two services disagree about the shared secret.
            # Logged at ERROR because nothing but an operator fixes it.
            logger.error(
                "The bot's billing API refused this service's shared secret: "
                "AURA_WEB_BOT_INTERNAL_API_SECRET must equal the bot's INTERNAL_API_SECRET"
            )
            raise BotBillingError("the bot's billing API refused the shared secret")
        if response.status_code not in accepted_statuses:
            raise BotBillingError(f"the bot's billing API returned HTTP {response.status_code}")
        try:
            body = response.json()
        except ValueError as exc:
            raise BotBillingError("the bot's billing API returned a non-JSON body") from exc
        if not isinstance(body, dict):
            raise BotBillingError("the bot's billing API returned a non-object body")
        return body
