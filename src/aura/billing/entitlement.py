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
    the start of the OLDEST period still unpaid, because cards fail for
    harmless reasons. That start is recorded once, when the database layer
    first stores the subscription past_due, and carried forward until a
    payment is actually seen (next_unpaid_since) -- so neither further failed
    retries, nor a late webhook, nor Stripe rolling an unpaid subscription into
    its next period can grant a second grace.
  * A subscription that is set to end (cancel at period end, or a cancel date)
    ends exactly then, whether it is paid up or on payment grace. There is
    nothing uncertain about a date the customer chose, so there is no grace to
    add.
  * Everything Stripe itself treats as not paying -- canceled, unpaid,
    incomplete, incomplete_expired, paused, collection paused, or an invoice
    that was voided or written off -- is Free immediately. So is a subscription
    that is not on the Pro price at a quantity of at least one: whatever it
    pays for, it is not Pro.
  * A subscription Stripe reports in force while an unpaid period is still
    recorded against it (the anchor above) grants nothing until a payment is
    seen. Stripe returns a written-off subscription to `active`, and the next
    period's invoice sits unpaid beside it for an hour or more before it is
    charged; "in force" then means "not yet failed again", not "paid".

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
    """One stored subscription snapshot, as the rules below read it.

    Attributes
    ----------
    on_pro_price
        Whether every item of the subscription is on the configured Pro price
        at a quantity of at least one. Decided by the web backend, which alone
        knows the price; False never grants.
    past_due_since
        The start of the oldest period still unpaid, as the database layer
        recorded it (see `next_unpaid_since`). None when no unpaid period is
        recorded -- including for a row written before this column existed,
        which `unpaid_since` then reads as its own period start.
    """

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
    on_pro_price: bool
    version: int
    confirmed_at: datetime
    past_due_since: datetime | None = None

    @field_validator(
        "cancel_at", "current_period_start", "current_period_end", "confirmed_at", "past_due_since"
    )
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
    PAYMENT_PENDING = "payment_pending"
    CANCELING = "canceling"
    PAYMENT_GRACE = "payment_grace"


_GRANTING_STANDINGS = frozenset(
    {
        Standing.ACTIVE,
        Standing.RENEWAL_PENDING,
        Standing.PAYMENT_PENDING,
        Standing.CANCELING,
        Standing.PAYMENT_GRACE,
    }
)

# When a guild has more than one subscription granting access (two admins who
# both paid), the one shown to the admin is the most reassuring TRUE one: a
# renewing subscription before one that is ending, and either before one that
# is only alive on payment grace -- and, within each, a paid period before one
# whose payment is not yet confirmed. Access itself does not depend on this order.
_DISPLAY_PRIORITY = {
    Standing.ACTIVE: 0,
    Standing.RENEWAL_PENDING: 0,
    Standing.PAYMENT_PENDING: 0,
    Standing.CANCELING: 1,
    Standing.PAYMENT_GRACE: 2,
}


def unpaid_since(record: SubscriptionRecord) -> datetime | None:
    """Return the start of the oldest period this record shows as unpaid.

    Parameters
    ----------
    record
        A stored subscription.

    Returns
    -------
    datetime or None
        The recorded `past_due_since` when there is one; otherwise the record's
        own period start if it is past_due, and None if it is not.

    Notes
    -----
    The fallback is not a guess. A past_due record proves its own period is
    unpaid and nothing about earlier ones, so its own period start is the
    earliest unpaid moment it can vouch for. It is only reached for a row
    written before `past_due_since` existed (and for records a test builds
    without history): every row stored since then carries the value
    explicitly, set by `next_unpaid_since`.
    """
    if record.past_due_since is not None:
        return record.past_due_since
    if record.status is SubscriptionStatus.PAST_DUE:
        return record.current_period_start
    return None


def next_unpaid_since(
    previous: SubscriptionRecord | None,
    *,
    status: SubscriptionStatus,
    latest_invoice_status: InvoiceStatus | None,
    current_period_start: datetime,
) -> datetime | None:
    """Carry the payment-grace anchor from the stored record to the snapshot replacing it.

    Parameters
    ----------
    previous
        The record currently stored for this subscription, or None for one
        seen for the first time.
    status, latest_invoice_status, current_period_start
        The same fields of the snapshot about to replace it.

    Returns
    -------
    datetime or None
        None once the snapshot shows the subscription in force with its period
        invoice paid -- a payment has been seen, so nothing is owed any more.
        Otherwise the earlier of the anchor carried from `previous` and, for a
        past_due snapshot, the snapshot's own period start; None when neither
        exists.

    Raises
    ------
    ValueError
        If `current_period_start` is naive.

    Notes
    -----
    Only a payment clears the anchor, never merely a status that is not
    past_due. Stripe documents that, under its default status resolution,
    VOIDING the latest invoice or MARKING IT UNCOLLECTIBLE also returns a
    past_due subscription to `active` -- and its retry settings can do the
    latter automatically once the retries run out. If any non-past_due status
    reset the anchor, that write-off would clear it, the next period's failed
    invoice would anchor afresh, and an unpaid subscription would again earn a
    new grace every period: the defect this function exists to close. The
    written-off period grants nothing meanwhile (`access_window`), nor does
    the next period's invoice until it is paid, and the anchor simply
    survives both.

    Taking the minimum, rather than keeping the first value blindly, makes the
    anchor monotonic: whatever order snapshots are stored in, it can only move
    earlier while the subscription stays unpaid, never later. It is always at
    or before the current period start, so the grace it yields is never longer
    than the one the period start alone would give.

    The anchor is only as complete as the snapshots the bot saw. A subscription
    first seen past_due in its second unpaid period is anchored at that
    period's start, and a payment made and missed entirely before the next
    failure is not seen as one. Both need the web backend to have missed every
    webhook retry and every six-hourly reconciliation for a whole billing
    period.
    """
    period_start = _require_aware(current_period_start, "current_period_start")
    if status in _IN_FORCE_STATUSES and latest_invoice_status is InvoiceStatus.PAID:
        return None
    carried = None if previous is None else unpaid_since(previous)
    if status is SubscriptionStatus.PAST_DUE:
        return period_start if carried is None else min(carried, period_start)
    return carried


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
        never grants Pro -- not on the Pro price, collection paused, its latest
        invoice written off, in force but with its period invoice unpaid while
        an unpaid period is recorded (`past_due_since`), or a status Stripe
        treats as not paying.

    Notes
    -----
    Deliberately time-independent: the window is a property of the record and the
    policy, and "is it open right now" is a single comparison the caller makes.
    That split is what lets a test pin the exact second a window closes without
    faking a clock inside this function.
    """
    if not record.on_pro_price:
        # Aura's metadata on a subscription for another price, or for none of
        # it (quantity 0), is not a Pro subscription -- a portal plan switch or
        # a hand-made subscription must not buy Pro at whatever it costs.
        return None
    if record.collection_paused:
        # Stripe voids or holds the invoices of a subscription whose collection
        # is paused, while its status still reads active. A paused collection
        # is not a paid period, so it grants nothing; an operator who wants to
        # keep a guild on Pro without payment lists it as complimentary.
        return None
    if record.latest_invoice_status in _WRITTEN_OFF_INVOICE_STATUSES:
        return None

    if record.status in _IN_FORCE_STATUSES:
        if (
            record.past_due_since is not None
            and record.latest_invoice_status is not InvoiceStatus.PAID
        ):
            # In force on paper, owing in fact (V-05). The anchor survives
            # only while no payment has been seen since a period went unpaid,
            # so an in-force record that still carries one reads `active`
            # because Stripe wrote an invoice off -- and the unpaid invoice
            # beside it is the next period's, which Stripe is only now about
            # to charge. Nothing was paid since the anchor; nothing is granted
            # until something is.
            return None
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
        anchor = unpaid_since(record)
        assert anchor is not None  # unpaid_since never returns None for past_due
        access_until = anchor + policy.payment_failure_grace
        # A known end date binds here exactly as it does for a paid-up
        # subscription: payment grace is for a failed card, not a way past a
        # date the customer chose.
        if record.cancel_at_period_end:
            access_until = min(access_until, record.current_period_end)
        if record.cancel_at is not None:
            access_until = min(access_until, record.cancel_at)
        return AccessWindow(
            record=record, standing=Standing.PAYMENT_GRACE, access_until=access_until
        )

    return None


@dataclass(frozen=True)
class SubscriptionStanding:
    """A guild's subscription position at one instant."""

    standing: Standing
    # When Pro ends if nothing new is heard. None when nothing grants access.
    access_until: datetime | None
    # The period end of the subscription shown, and ONLY when that period's
    # invoice is paid -- None otherwise, so no surface can say "paid through"
    # a date nobody has paid for (a payment still processing, a failed one).
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
            True for ACTIVE, RENEWAL_PENDING, PAYMENT_PENDING, CANCELING and
            PAYMENT_GRACE. A subscription that is ending still grants access until
            it actually ends.
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
            window.record.latest_invoice_status is not InvoiceStatus.PAID,
            -window.access_until.timestamp(),
            window.record.subscription_id,
        ),
    )
    period_paid = shown.record.latest_invoice_status is InvoiceStatus.PAID
    standing = shown.standing
    if standing is Standing.ACTIVE and now >= shown.record.current_period_end:
        # Inside the renewal grace: still Pro, but the admin should be able to
        # see that the renewal has not been confirmed rather than read
        # "active" past the date they know the period ended.
        standing = Standing.RENEWAL_PENDING
    elif standing is Standing.ACTIVE and not period_paid:
        # In force, but this period's invoice is not (yet) paid: a renewal
        # between its draft and its charge, or a payment that is still
        # processing. Still Pro -- the invoice has not failed, and nothing
        # older is owed (access_window refuses the record otherwise) -- but
        # not "paid through" anything.
        standing = Standing.PAYMENT_PENDING
    return SubscriptionStanding(
        standing=standing,
        access_until=shown.access_until,
        paid_through=shown.record.current_period_end if period_paid else None,
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
    Computing the standing on every basis is what lets `/aura-plan` and the
    dashboard still show a subscription whose guild's plan it does not decide:
    one that predates a switch to unenforced billing, or one a complimentary
    guild is paying anyway.
    """
    standing = resolve_standing(records, now=now, policy=policy)
    if not enforced:
        return GuildPlan(guild_id, PlanTier.PRO, PlanBasis.BILLING_NOT_ENFORCED, standing)
    if complimentary:
        return GuildPlan(guild_id, PlanTier.PRO, PlanBasis.COMPLIMENTARY, standing)
    tier = PlanTier.PRO if standing.grants_access else PlanTier.FREE
    return GuildPlan(guild_id, tier, PlanBasis.SUBSCRIPTION, standing)
