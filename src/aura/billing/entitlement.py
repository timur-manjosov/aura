"""Whether a guild is on Pro right now, decided from subscription records alone.

Pure functions over already-loaded records and an injected clock: no database,
no network, no Discord. CLAUDE.md's Testing section asks for exactly this
separation for fact logic, and it matters more here -- the boundaries below
("Pro until this exact second, Free from the next one") are the part of Phase
4c that must be provable to the second, and a function that reads the clock
itself cannot be tested at the second that matters.

THE ASYMMETRY THIS MODULE ENCODES. A mistake here goes one of two ways, and they
are not equally bad:

  * granting Pro that was not paid for costs the operator a bounded amount of
    LLM spend -- every Pro trigger is capped per guild per day already;
  * moving a paying guild to Free is a broken promise to a customer.

So every rule leans toward the paying guild exactly as far as a documented,
finite bound allows, and no further:

  * An active subscription stays Pro for BILLING_RENEWAL_GRACE_HOURS past the
    date it is paid through, because the confirmation of a renewal can be
    delayed by an outage Stripe itself is still retrying. It does not stay Pro
    past that: "we have not heard" is trusted for a fixed time, not forever.
  * A failed renewal (past_due) stays Pro for BILLING_PAYMENT_GRACE_DAYS from
    the start of the unpaid period, because cards fail for harmless reasons.
    Anchored to Stripe's period start, so neither further failed retries nor a
    late webhook can stretch it.
  * A subscription that is set to end (cancel at period end, or a cancel date)
    ends exactly then. There is nothing uncertain about a date the customer
    chose, so there is no grace to add.
  * Everything Stripe itself treats as not paying -- canceled, unpaid,
    incomplete, incomplete_expired, paused, collection paused, or an invoice
    that was voided or written off -- is Free immediately.

Access is a half-open interval: Pro while `now < access_until`, Free from
`access_until` on. That single comparison is what makes "exactly at the
boundary" a defined answer rather than an accident of `<` versus `<=`.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, field_validator, model_validator


class SubscriptionStatus(StrEnum):
    """Stripe's subscription statuses, all of them, as of API version 2026-08-26.dahlia."""

    ACTIVE = "active"
    TRIALING = "trialing"
    PAST_DUE = "past_due"
    UNPAID = "unpaid"
    CANCELED = "canceled"
    INCOMPLETE = "incomplete"
    INCOMPLETE_EXPIRED = "incomplete_expired"
    PAUSED = "paused"


class InvoiceStatus(StrEnum):
    """Stripe's invoice statuses, used only to catch a voided or written-off period."""

    DRAFT = "draft"
    OPEN = "open"
    PAID = "paid"
    UNCOLLECTIBLE = "uncollectible"
    VOID = "void"


# The statuses under which Stripe itself considers the subscription in force.
_IN_FORCE_STATUSES = frozenset({SubscriptionStatus.ACTIVE, SubscriptionStatus.TRIALING})

# An invoice in one of these states will never be paid through the normal flow.
# A subscription can still read `active` beside one: Stripe documents that a
# delayed payment method (a bank debit) failing after activation voids the
# invoice and leaves the subscription active.
_WRITTEN_OFF_INVOICE_STATUSES = frozenset({InvoiceStatus.VOID, InvoiceStatus.UNCOLLECTIBLE})


def _require_aware(moment: datetime, name: str) -> datetime:
    if moment.tzinfo is None:
        raise ValueError(f"{name} must be a timezone-aware datetime, got {moment!r}")
    return moment


class GracePolicy(BaseModel):
    """The two bounded grace periods, validated once instead of at every decision."""

    model_config = ConfigDict(frozen=True)

    renewal_grace: timedelta
    payment_failure_grace: timedelta

    @field_validator("renewal_grace", "payment_failure_grace")
    @classmethod
    def _not_negative(cls, value: timedelta) -> timedelta:
        if value < timedelta(0):
            raise ValueError(f"a grace period cannot be negative, got {value}")
        return value


class SubscriptionRecord(BaseModel):
    """One stored subscription snapshot, as the rules below read it."""

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
    version: int
    confirmed_at: datetime

    @field_validator("cancel_at", "current_period_start", "current_period_end", "confirmed_at")
    @classmethod
    def _timezone_aware(cls, value: datetime | None) -> datetime | None:
        # A naive datetime compared against an aware one raises; compared
        # against another naive one it is silently wrong by the host's UTC
        # offset. Refusing it here keeps both out of the decision below.
        return None if value is None else _require_aware(value, "subscription times")

    @model_validator(mode="after")
    def _period_is_ordered(self) -> SubscriptionRecord:
        if self.current_period_end < self.current_period_start:
            raise ValueError("current_period_end must not precede current_period_start")
        return self


class Standing(StrEnum):
    """Where a guild's subscriptions leave it right now, for both the gate and the admin."""

    NO_SUBSCRIPTION = "no_subscription"
    ENDED = "ended"
    ACTIVE = "active"
    RENEWAL_PENDING = "renewal_pending"
    CANCELING = "canceling"
    PAYMENT_GRACE = "payment_grace"


_GRANTING_STANDINGS = frozenset(
    {Standing.ACTIVE, Standing.RENEWAL_PENDING, Standing.CANCELING, Standing.PAYMENT_GRACE}
)

# When a guild has more than one subscription granting access (two admins who
# both paid), the one shown to the admin is the most reassuring TRUE one: a
# renewing subscription before one that is ending, and either before one that
# is only alive on payment grace. Access itself does not depend on this order.
_DISPLAY_PRIORITY = {
    Standing.ACTIVE: 0,
    Standing.RENEWAL_PENDING: 0,
    Standing.CANCELING: 1,
    Standing.PAYMENT_GRACE: 2,
}


@dataclass(frozen=True)
class AccessWindow:
    """How long one subscription grants Pro, independent of the current time."""

    record: SubscriptionRecord
    standing: Standing
    access_until: datetime


def access_window(record: SubscriptionRecord, policy: GracePolicy) -> AccessWindow | None:
    """Return the interval during which one subscription grants Pro.

    Parameters
    ----------
    record
        The stored subscription.
    policy
        The grace periods to extend the paid period by.

    Returns
    -------
    AccessWindow or None
        The window and the standing it implies, or None if this subscription
        never grants Pro -- collection paused, or its latest invoice written
        off.

    Notes
    -----
    Deliberately time-independent: the window is a property of the record and the
    policy, and "is it open right now" is a single comparison the caller makes.
    That split is what lets a test pin the exact second a window closes without
    faking a clock inside this function.
    """
    if record.collection_paused:
        # Stripe voids or holds the invoices of a subscription whose collection
        # is paused, while its status still reads active. A paused collection
        # is not a paid period, so it grants nothing; an operator who wants to
        # keep a guild on Pro without payment lists it as complimentary.
        return None
    if record.latest_invoice_status in _WRITTEN_OFF_INVOICE_STATUSES:
        return None

    if record.status in _IN_FORCE_STATUSES:
        access_until = record.current_period_end + policy.renewal_grace
        ending = record.cancel_at_period_end or record.cancel_at is not None
        if record.cancel_at_period_end:
            # A known end date: no renewal is expected, so there is nothing
            # unconfirmed to wait for.
            access_until = record.current_period_end
        if record.cancel_at is not None:
            access_until = min(access_until, record.cancel_at)
        return AccessWindow(
            record=record,
            standing=Standing.CANCELING if ending else Standing.ACTIVE,
            access_until=access_until,
        )

    if record.status is SubscriptionStatus.PAST_DUE:
        return AccessWindow(
            record=record,
            standing=Standing.PAYMENT_GRACE,
            access_until=record.current_period_start + policy.payment_failure_grace,
        )

    return None


@dataclass(frozen=True)
class SubscriptionStanding:
    """A guild's subscription position at one instant."""

    standing: Standing
    # When Pro ends if nothing new is heard. None when nothing grants access.
    access_until: datetime | None
    # The period end of the subscription shown -- "paid through" for display.
    paid_through: datetime | None
    # The subscription the standing above describes.
    shown_subscription_id: str | None
    # Every subscription granting access right now; more than one means a
    # guild is paying twice.
    granting_subscription_ids: frozenset[str]

    @property
    def grants_access(self) -> bool:
        """Report whether any subscription grants Pro at this instant.

        Returns
        -------
        bool
            True for ACTIVE, RENEWAL_PENDING, CANCELING and PAYMENT_GRACE. A
            subscription that is ending still grants access until it actually ends.
        """
        return self.standing in _GRANTING_STANDINGS


def resolve_standing(
    records: Iterable[SubscriptionRecord], *, now: datetime, policy: GracePolicy
) -> SubscriptionStanding:
    """Combine a guild's subscriptions into one standing at a given instant.

    Parameters
    ----------
    records
        Every subscription known for the guild. May be empty.
    now
        The instant to decide at. Must be timezone-aware.
    policy
        The grace periods.

    Returns
    -------
    SubscriptionStanding
        One standing for the guild, the subscription it describes, and every
        subscription granting access right now -- more than one means the guild
        is paying twice.

    Raises
    ------
    ValueError
        If `now` is naive.

    Notes
    -----
    Any one open window grants Pro. A guild with records but no open window has
    ENDED rather than NO_SUBSCRIPTION, so an admin whose subscription lapsed is
    told that, not told they never had one.
    """
    _require_aware(now, "now")
    record_list = list(records)
    open_windows = [
        window
        for record in record_list
        if (window := access_window(record, policy)) is not None and now < window.access_until
    ]
    if not open_windows:
        return SubscriptionStanding(
            standing=Standing.ENDED if record_list else Standing.NO_SUBSCRIPTION,
            access_until=None,
            paid_through=None,
            shown_subscription_id=None,
            granting_subscription_ids=frozenset(),
        )

    shown = min(
        open_windows,
        key=lambda window: (
            _DISPLAY_PRIORITY[window.standing],
            -window.access_until.timestamp(),
            window.record.subscription_id,
        ),
    )
    standing = shown.standing
    if standing is Standing.ACTIVE and now >= shown.record.current_period_end:
        # Inside the renewal grace: still Pro, but the admin should be able to
        # see that the renewal has not been confirmed rather than read
        # "active" past the date they know the period ended.
        standing = Standing.RENEWAL_PENDING
    return SubscriptionStanding(
        standing=standing,
        access_until=shown.access_until,
        paid_through=shown.record.current_period_end,
        shown_subscription_id=shown.record.subscription_id,
        granting_subscription_ids=frozenset(
            window.record.subscription_id for window in open_windows
        ),
    )


class PlanTier(StrEnum):
    """The two plans. There is deliberately no third."""

    FREE = "free"
    PRO = "pro"


class PlanBasis(StrEnum):
    """Why a guild has the tier it has."""

    BILLING_NOT_ENFORCED = "billing_not_enforced"
    COMPLIMENTARY = "complimentary"
    SUBSCRIPTION = "subscription"


@dataclass(frozen=True)
class GuildPlan:
    """The decision every Pro trigger and every plan display reads."""

    guild_id: int
    tier: PlanTier
    basis: PlanBasis
    standing: SubscriptionStanding

    @property
    def is_pro(self) -> bool:
        """Report whether Pro-only triggers may run for this guild right now.

        Returns
        -------
        bool
            Whether the decided tier is PRO, whatever the basis for it.
        """
        return self.tier is PlanTier.PRO


def decide_plan(
    *,
    guild_id: int,
    records: Iterable[SubscriptionRecord],
    now: datetime,
    policy: GracePolicy,
    enforced: bool,
    complimentary: bool,
) -> GuildPlan:
    """Decide one guild's plan.

    Parameters
    ----------
    guild_id
        Guild to decide for.
    records
        Every subscription known for it.
    now
        The instant to decide at. Must be timezone-aware.
    policy
        The grace periods.
    enforced
        Whether Free is an answer this deployment can give at all.
    complimentary
        Whether the operator has put this guild on Pro by decision.

    Returns
    -------
    GuildPlan
        The tier, the basis for it, and the full standing -- computed in every
        case, including when billing is not enforced.

    Raises
    ------
    ValueError
        If `now` is naive, via `resolve_standing`.

    Notes
    -----
    Computing the standing even when billing is not enforced is what lets a guild
    that subscribed before enforcement was switched on see its subscription, and
    what lets the web backend refuse a second checkout for a guild that is
    already paying regardless of the deployment's mode.
    """
    standing = resolve_standing(records, now=now, policy=policy)
    if not enforced:
        return GuildPlan(guild_id, PlanTier.PRO, PlanBasis.BILLING_NOT_ENFORCED, standing)
    if complimentary:
        return GuildPlan(guild_id, PlanTier.PRO, PlanBasis.COMPLIMENTARY, standing)
    tier = PlanTier.PRO if standing.grants_access else PlanTier.FREE
    return GuildPlan(guild_id, tier, PlanBasis.SUBSCRIPTION, standing)
