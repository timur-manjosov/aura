"""aura.billing.entitlement: every standing, and every boundary proven to the microsecond.

The brief for Phase 4c asks for the grace periods to be verified "exactly --
neither too early nor unbounded". Every boundary below is therefore tested as a
pair: one tick before it (still Pro) and exactly at it (Free). A rule that
answered one tick early or one tick late would fail one half of its pair.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

import pytest

from aura.billing.entitlement import (
    GracePolicy,
    InvoiceStatus,
    PlanBasis,
    PlanTier,
    Standing,
    SubscriptionRecord,
    SubscriptionStatus,
    access_window,
    decide_plan,
    next_unpaid_since,
    resolve_standing,
    unpaid_since,
)

NOW = datetime(2026, 9, 13, 12, 0, 0, tzinfo=UTC)
TICK = timedelta(microseconds=1)
POLICY = GracePolicy(renewal_grace=timedelta(hours=72), payment_failure_grace=timedelta(days=7))
GUILD = 100000000000000001
PERIOD_START = NOW - timedelta(days=10)
PERIOD_END = NOW + timedelta(days=20)


def record(**overrides: object) -> SubscriptionRecord:
    values: dict[str, object] = {
        "subscription_id": "sub_A",
        "guild_id": GUILD,
        "customer_id": "cus_A",
        "purchaser_user_id": 5000,
        "status": SubscriptionStatus.ACTIVE,
        "cancel_at_period_end": False,
        "cancel_at": None,
        "collection_paused": False,
        "latest_invoice_status": InvoiceStatus.PAID,
        "current_period_start": PERIOD_START,
        "current_period_end": PERIOD_END,
        "livemode": False,
        "on_pro_price": True,
        "version": 1,
        "confirmed_at": PERIOD_START,
    }
    values.update(overrides)
    return SubscriptionRecord(**values)  # type: ignore[arg-type]


def at(moment: datetime, *records: SubscriptionRecord, policy: GracePolicy = POLICY):
    return resolve_standing(records, now=moment, policy=policy)


class TestActiveSubscription:
    def test_is_pro_and_reports_the_paid_through_date(self) -> None:
        standing = at(NOW, record())

        assert standing.standing is Standing.ACTIVE
        assert standing.grants_access
        assert standing.paid_through == PERIOD_END
        assert standing.access_until == PERIOD_END + timedelta(hours=72)

    def test_past_the_paid_through_date_it_is_still_pro_but_says_renewal_pending(self) -> None:
        standing = at(PERIOD_END, record())

        assert standing.standing is Standing.RENEWAL_PENDING
        assert standing.grants_access

    def test_the_renewal_grace_ends_exactly_seventy_two_hours_after_the_period(self) -> None:
        boundary = PERIOD_END + timedelta(hours=72)

        assert at(boundary - TICK, record()).grants_access
        assert not at(boundary, record()).grants_access
        assert at(boundary, record()).standing is Standing.ENDED

    def test_hearing_nothing_for_a_year_does_not_extend_trust(self) -> None:
        """The "status channel unreachable" case: the stored snapshot ages, the bound does not move."""
        assert not at(PERIOD_END + timedelta(days=365), record()).grants_access

    def test_a_confirmation_long_ago_does_not_shorten_a_paid_period(self) -> None:
        """confirmed_at is diagnostic: a guild paid through a date keeps Pro until it."""
        old = record(confirmed_at=PERIOD_START - timedelta(days=300))

        assert at(PERIOD_END - TICK, old).grants_access

    def test_trialing_follows_the_same_rule(self) -> None:
        trialing = record(status=SubscriptionStatus.TRIALING)

        assert at(PERIOD_END + timedelta(hours=72) - TICK, trialing).grants_access
        assert not at(PERIOD_END + timedelta(hours=72), trialing).grants_access

    def test_with_no_renewal_grace_pro_ends_exactly_at_the_period_end(self) -> None:
        strict = GracePolicy(renewal_grace=timedelta(0), payment_failure_grace=timedelta(0))

        assert at(PERIOD_END - TICK, record(), policy=strict).grants_access
        assert not at(PERIOD_END, record(), policy=strict).grants_access


class TestSubscriptionSetToEnd:
    def test_cancel_at_period_end_ends_exactly_at_the_period_end_with_no_grace(self) -> None:
        ending = record(cancel_at_period_end=True)

        assert at(PERIOD_END - TICK, ending).standing is Standing.CANCELING
        assert not at(PERIOD_END, ending).grants_access

    def test_a_cancel_date_inside_the_period_ends_pro_on_that_date(self) -> None:
        cancel_at = NOW + timedelta(days=3)
        ending = record(cancel_at=cancel_at)

        assert at(cancel_at - TICK, ending).grants_access
        assert not at(cancel_at, ending).grants_access

    def test_a_cancel_date_after_the_renewal_grace_does_not_extend_past_it(self) -> None:
        """A far-future cancel date means "renews until then", not "paid until then"."""
        ending = record(cancel_at=PERIOD_END + timedelta(days=90))

        assert not at(PERIOD_END + timedelta(hours=72), ending).grants_access

    def test_a_cancel_date_inside_the_renewal_grace_wins(self) -> None:
        cancel_at = PERIOD_END + timedelta(hours=10)
        ending = record(cancel_at=cancel_at)

        assert at(cancel_at - TICK, ending).grants_access
        assert not at(cancel_at, ending).grants_access

    def test_a_cancel_date_already_in_the_past_is_free_even_if_status_still_says_active(
        self,
    ) -> None:
        """The deletion webhook has not arrived yet; the known end date still binds."""
        assert not at(NOW, record(cancel_at=NOW - timedelta(seconds=1))).grants_access


class TestPaymentFailureGrace:
    def test_past_due_is_pro_for_exactly_seven_days_from_the_unpaid_period_start(self) -> None:
        failed = record(
            status=SubscriptionStatus.PAST_DUE,
            current_period_start=NOW,
            current_period_end=NOW + timedelta(days=30),
        )
        boundary = NOW + timedelta(days=7)

        assert at(boundary - TICK, failed).standing is Standing.PAYMENT_GRACE
        assert not at(boundary, failed).grants_access

    def test_more_failed_retries_do_not_extend_the_grace(self) -> None:
        """A later snapshot of the same unpaid period carries the same period start."""
        first_failure = record(
            status=SubscriptionStatus.PAST_DUE,
            current_period_start=NOW,
            version=1,
            confirmed_at=NOW,
        )
        fifth_failure = first_failure.model_copy(
            update={"version": 5, "confirmed_at": NOW + timedelta(days=6)}
        )

        assert access_window(first_failure, POLICY) == access_window(first_failure, POLICY)
        assert (
            access_window(fifth_failure, POLICY).access_until
            == access_window(first_failure, POLICY).access_until
        )  # type: ignore[union-attr]
        assert not at(NOW + timedelta(days=7), fifth_failure).grants_access

    def test_a_failure_first_heard_about_late_is_not_given_a_fresh_week(self) -> None:
        stale_failure = record(
            status=SubscriptionStatus.PAST_DUE,
            current_period_start=NOW - timedelta(days=8),
            confirmed_at=NOW,
        )

        assert not at(NOW, stale_failure).grants_access

    def test_payment_recovered_back_to_active_is_pro_again(self) -> None:
        recovered = record(
            status=SubscriptionStatus.ACTIVE, current_period_start=NOW - timedelta(days=9)
        )

        assert at(NOW, recovered).standing is Standing.ACTIVE


class TestStatusesThatNeverGrant:
    @pytest.mark.parametrize(
        "status",
        [
            SubscriptionStatus.UNPAID,
            SubscriptionStatus.CANCELED,
            SubscriptionStatus.INCOMPLETE,
            SubscriptionStatus.INCOMPLETE_EXPIRED,
            SubscriptionStatus.PAUSED,
        ],
    )
    def test_is_free_immediately_even_mid_period(self, status: SubscriptionStatus) -> None:
        assert access_window(record(status=status), POLICY) is None
        assert not at(NOW, record(status=status)).grants_access

    def test_paused_collection_is_free_although_the_status_reads_active(self) -> None:
        assert not at(NOW, record(collection_paused=True)).grants_access

    @pytest.mark.parametrize("invoice", [InvoiceStatus.VOID, InvoiceStatus.UNCOLLECTIBLE])
    def test_a_written_off_invoice_is_free_although_the_status_reads_active(
        self, invoice: InvoiceStatus
    ) -> None:
        """Stripe: a delayed payment method failing after activation voids the invoice, status stays active."""
        assert not at(NOW, record(latest_invoice_status=invoice)).grants_access

    @pytest.mark.parametrize(
        "invoice", [InvoiceStatus.DRAFT, InvoiceStatus.OPEN, InvoiceStatus.PAID, None]
    )
    def test_an_invoice_still_in_flight_does_not_revoke(
        self, invoice: InvoiceStatus | None
    ) -> None:
        """A renewal invoice is a draft for about an hour, then open until charged."""
        assert at(NOW, record(latest_invoice_status=invoice)).grants_access


class TestSeveralSubscriptions:
    def test_no_records_is_no_subscription_and_ended_records_are_ended(self) -> None:
        assert at(NOW).standing is Standing.NO_SUBSCRIPTION
        assert at(NOW, record(status=SubscriptionStatus.CANCELED)).standing is Standing.ENDED

    def test_one_granting_subscription_is_enough(self) -> None:
        standing = at(
            NOW,
            record(subscription_id="sub_old", status=SubscriptionStatus.CANCELED),
            record(subscription_id="sub_new"),
        )

        assert standing.grants_access
        assert standing.granting_subscription_ids == frozenset({"sub_new"})

    def test_two_paying_subscriptions_are_both_counted(self) -> None:
        standing = at(NOW, record(subscription_id="sub_1"), record(subscription_id="sub_2"))

        assert standing.granting_subscription_ids == frozenset({"sub_1", "sub_2"})

    def test_the_shown_standing_prefers_a_renewing_subscription_over_payment_grace(self) -> None:
        standing = at(
            NOW,
            record(
                subscription_id="sub_failing",
                status=SubscriptionStatus.PAST_DUE,
                current_period_start=NOW,
            ),
            record(subscription_id="sub_fine"),
        )

        assert standing.standing is Standing.ACTIVE
        assert standing.shown_subscription_id == "sub_fine"


class TestDecidePlan:
    def test_not_enforced_is_pro_for_everyone_and_still_reports_the_standing(self) -> None:
        plan = decide_plan(
            guild_id=GUILD,
            records=[record(status=SubscriptionStatus.CANCELED)],
            now=NOW,
            policy=POLICY,
            enforced=False,
            complimentary=False,
        )

        assert plan.tier is PlanTier.PRO
        assert plan.basis is PlanBasis.BILLING_NOT_ENFORCED
        assert plan.standing.standing is Standing.ENDED

    def test_complimentary_is_pro_without_a_subscription(self) -> None:
        plan = decide_plan(
            guild_id=GUILD, records=[], now=NOW, policy=POLICY, enforced=True, complimentary=True
        )

        assert plan.is_pro
        assert plan.basis is PlanBasis.COMPLIMENTARY

    def test_enforced_without_a_granting_subscription_is_free(self) -> None:
        plan = decide_plan(
            guild_id=GUILD, records=[], now=NOW, policy=POLICY, enforced=True, complimentary=False
        )

        assert plan.tier is PlanTier.FREE
        assert not plan.is_pro

    def test_enforced_with_a_granting_subscription_is_pro(self) -> None:
        plan = decide_plan(
            guild_id=GUILD,
            records=[record()],
            now=NOW,
            policy=POLICY,
            enforced=True,
            complimentary=False,
        )

        assert plan.is_pro
        assert plan.basis is PlanBasis.SUBSCRIPTION


class TestRefusesAmbiguousInput:
    def test_a_naive_now_is_refused(self) -> None:
        with pytest.raises(ValueError):
            resolve_standing([record()], now=datetime(2026, 9, 13, 12, 0), policy=POLICY)

    def test_a_record_with_naive_times_is_refused(self) -> None:
        with pytest.raises(ValueError):
            record(current_period_end=datetime(2026, 10, 1))

    def test_an_inverted_period_is_refused(self) -> None:
        with pytest.raises(ValueError):
            record(current_period_start=PERIOD_END, current_period_end=PERIOD_START)

    def test_a_negative_grace_is_refused(self) -> None:
        with pytest.raises(ValueError):
            GracePolicy(renewal_grace=timedelta(hours=-1), payment_failure_grace=timedelta(days=7))

    def test_a_non_utc_zone_is_compared_by_instant_not_by_wall_clock(self) -> None:
        """The same instant written in UTC+05:30 must give the same answer."""
        india = timezone(timedelta(hours=5, minutes=30))
        boundary = (PERIOD_END + timedelta(hours=72)).astimezone(india)

        assert at(boundary - TICK, record()).grants_access
        assert not at(boundary, record()).grants_access


# --- Phase 4c audit fixes -----------------------------------------------------


def unpaid(**overrides: object) -> SubscriptionRecord:
    """A past_due record with an explicit anchor, as the database layer writes one."""
    fields: dict[str, object] = {
        "status": SubscriptionStatus.PAST_DUE,
        "latest_invoice_status": InvoiceStatus.OPEN,
    }
    fields.update(overrides)
    return record(**fields)


class TestThePaymentGraceAnchor:
    """next_unpaid_since: one grace per lapsed payment, never one per period (F-04)."""

    def test_the_first_past_due_snapshot_anchors_at_its_own_period_start(self) -> None:
        assert (
            next_unpaid_since(
                None,
                status=SubscriptionStatus.PAST_DUE,
                latest_invoice_status=InvoiceStatus.OPEN,
                current_period_start=PERIOD_START,
            )
            == PERIOD_START
        )

    def test_a_later_unpaid_period_keeps_the_earlier_anchor(self) -> None:
        stored = unpaid(past_due_since=PERIOD_START)

        carried = next_unpaid_since(
            stored,
            status=SubscriptionStatus.PAST_DUE,
            latest_invoice_status=InvoiceStatus.OPEN,
            current_period_start=PERIOD_END,
        )

        assert carried == PERIOD_START

    def test_the_anchor_only_ever_moves_earlier(self) -> None:
        """Monotonic: a snapshot with an earlier period start (never real) cannot push it later."""
        stored = unpaid(
            current_period_start=PERIOD_END,
            current_period_end=PERIOD_END + timedelta(days=30),
            past_due_since=PERIOD_END,
        )

        carried = next_unpaid_since(
            stored,
            status=SubscriptionStatus.PAST_DUE,
            latest_invoice_status=InvoiceStatus.OPEN,
            current_period_start=PERIOD_START,
        )

        assert carried == PERIOD_START

    @pytest.mark.parametrize("status", [SubscriptionStatus.ACTIVE, SubscriptionStatus.TRIALING])
    def test_a_paid_period_in_force_clears_it(self, status: SubscriptionStatus) -> None:
        assert (
            next_unpaid_since(
                unpaid(past_due_since=PERIOD_START),
                status=status,
                latest_invoice_status=InvoiceStatus.PAID,
                current_period_start=PERIOD_START,
            )
            is None
        )

    @pytest.mark.parametrize(
        ("status", "invoice"),
        [
            # Stripe: voiding or marking the latest invoice uncollectible returns
            # a past_due subscription to `active` -- without anything being paid.
            (SubscriptionStatus.ACTIVE, InvoiceStatus.VOID),
            (SubscriptionStatus.ACTIVE, InvoiceStatus.UNCOLLECTIBLE),
            # The next period's renewal between its draft and its charge.
            (SubscriptionStatus.ACTIVE, InvoiceStatus.DRAFT),
            (SubscriptionStatus.ACTIVE, InvoiceStatus.OPEN),
            (SubscriptionStatus.ACTIVE, None),
            (SubscriptionStatus.PAST_DUE, InvoiceStatus.PAID),
            (SubscriptionStatus.UNPAID, InvoiceStatus.OPEN),
            (SubscriptionStatus.CANCELED, InvoiceStatus.OPEN),
            (SubscriptionStatus.PAUSED, None),
            (SubscriptionStatus.INCOMPLETE, InvoiceStatus.OPEN),
        ],
    )
    def test_anything_but_a_paid_period_in_force_keeps_it(
        self, status: SubscriptionStatus, invoice: InvoiceStatus | None
    ) -> None:
        carried = next_unpaid_since(
            unpaid(past_due_since=PERIOD_START),
            status=status,
            latest_invoice_status=invoice,
            current_period_start=PERIOD_END,
        )

        assert carried == PERIOD_START

    def test_a_subscription_never_unpaid_has_no_anchor(self) -> None:
        for invoice in (InvoiceStatus.DRAFT, InvoiceStatus.OPEN, None):
            assert (
                next_unpaid_since(
                    record(),
                    status=SubscriptionStatus.ACTIVE,
                    latest_invoice_status=invoice,
                    current_period_start=PERIOD_END,
                )
                is None
            )

    def test_a_row_from_before_the_column_reads_as_its_own_period_start(self) -> None:
        """Legacy rows keep exactly the pre-fix meaning, and carry it forward."""
        legacy = unpaid(past_due_since=None)

        assert unpaid_since(legacy) == PERIOD_START
        assert access_window(legacy, POLICY).access_until == PERIOD_START + timedelta(days=7)  # type: ignore[union-attr]
        assert (
            next_unpaid_since(
                legacy,
                status=SubscriptionStatus.PAST_DUE,
                latest_invoice_status=InvoiceStatus.OPEN,
                current_period_start=PERIOD_END,
            )
            == PERIOD_START
        )

    def test_a_naive_period_start_is_refused(self) -> None:
        with pytest.raises(ValueError):
            next_unpaid_since(
                None,
                status=SubscriptionStatus.PAST_DUE,
                latest_invoice_status=InvoiceStatus.OPEN,
                current_period_start=datetime(2026, 9, 1),
            )

    def test_a_naive_anchor_is_refused(self) -> None:
        with pytest.raises(ValueError):
            unpaid(past_due_since=datetime(2026, 9, 1))


class TestPaymentGraceFromTheAnchor:
    def test_an_unpaid_run_rolling_into_its_next_period_ends_seven_days_after_the_first(
        self,
    ) -> None:
        second_period = unpaid(
            current_period_start=PERIOD_END,
            current_period_end=PERIOD_END + timedelta(days=30),
            past_due_since=PERIOD_START,
        )
        boundary = PERIOD_START + timedelta(days=7)

        assert at(boundary - TICK, second_period).standing is Standing.PAYMENT_GRACE
        assert not at(boundary, second_period).grants_access
        assert not at(PERIOD_END + TICK, second_period).grants_access

    def test_the_boundary_is_exact_to_the_second(self) -> None:
        failing = unpaid(past_due_since=PERIOD_START)
        boundary = PERIOD_START + timedelta(days=7)

        answers = [
            at(boundary + offset, failing).grants_access
            for offset in (-timedelta(seconds=1), timedelta(0), timedelta(seconds=1))
        ]

        assert answers == [True, False, False]


class TestPaymentGraceMeetsAnEndDate:
    """F-14: a date the customer chose binds inside the payment grace too."""

    def test_a_cancel_date_inside_the_grace_ends_pro_on_that_date(self) -> None:
        cancel_at = PERIOD_START + timedelta(days=2)
        failing = unpaid(past_due_since=PERIOD_START, cancel_at=cancel_at)

        assert at(cancel_at - TICK, failing).grants_access
        assert not at(cancel_at, failing).grants_access

    def test_a_cancel_date_after_the_grace_does_not_extend_it(self) -> None:
        failing = unpaid(past_due_since=PERIOD_START, cancel_at=PERIOD_END)

        assert not at(PERIOD_START + timedelta(days=7), failing).grants_access

    def test_cancel_at_period_end_caps_a_grace_longer_than_the_period(self) -> None:
        generous = GracePolicy(
            renewal_grace=timedelta(hours=72), payment_failure_grace=timedelta(days=60)
        )
        failing = unpaid(past_due_since=PERIOD_START, cancel_at_period_end=True)

        assert at(PERIOD_END - TICK, failing, policy=generous).grants_access
        assert not at(PERIOD_END, failing, policy=generous).grants_access

    def test_the_standing_still_says_the_payment_failed(self) -> None:
        failing = unpaid(past_due_since=PERIOD_START, cancel_at=PERIOD_START + timedelta(days=2))

        standing = at(PERIOD_START + timedelta(days=1), failing)

        assert standing.standing is Standing.PAYMENT_GRACE
        assert standing.access_until == PERIOD_START + timedelta(days=2)


class TestOnlyTheProPriceGrants:
    """F-07: a subscription not on the Pro price at quantity >= 1 grants nothing, whatever else."""

    @pytest.mark.parametrize(
        "status",
        [SubscriptionStatus.ACTIVE, SubscriptionStatus.TRIALING, SubscriptionStatus.PAST_DUE],
    )
    def test_a_status_that_would_grant_does_not(self, status: SubscriptionStatus) -> None:
        off_price = record(status=status, on_pro_price=False)

        assert access_window(off_price, POLICY) is None
        assert at(NOW, off_price).standing is Standing.ENDED

    def test_an_enforced_guild_on_the_wrong_price_is_free(self) -> None:
        plan = decide_plan(
            guild_id=GUILD,
            records=[record(on_pro_price=False)],
            now=NOW,
            policy=POLICY,
            enforced=True,
            complimentary=False,
        )

        assert plan.tier is PlanTier.FREE

    def test_it_does_not_hide_a_second_subscription_on_the_right_price(self) -> None:
        standing = at(
            NOW,
            record(subscription_id="sub_cheap", on_pro_price=False),
            record(subscription_id="sub_pro"),
        )

        assert standing.granting_subscription_ids == frozenset({"sub_pro"})

    def test_the_field_is_required(self) -> None:
        values = record().model_dump()
        del values["on_pro_price"]

        with pytest.raises(ValueError):
            SubscriptionRecord.model_validate(values)


class TestPaymentPending:
    """F-05: in force but not yet paid -- still Pro, never "paid through"."""

    @pytest.mark.parametrize("invoice", [InvoiceStatus.DRAFT, InvoiceStatus.OPEN, None])
    def test_an_unpaid_period_is_payment_pending_with_no_paid_through_date(
        self, invoice: InvoiceStatus | None
    ) -> None:
        standing = at(NOW, record(latest_invoice_status=invoice))

        assert standing.standing is Standing.PAYMENT_PENDING
        assert standing.grants_access
        assert standing.paid_through is None
        assert standing.access_until == PERIOD_END + timedelta(hours=72)

    def test_a_paid_period_is_active_and_paid_through_its_end(self) -> None:
        standing = at(NOW, record(latest_invoice_status=InvoiceStatus.PAID))

        assert standing.standing is Standing.ACTIVE
        assert standing.paid_through == PERIOD_END

    def test_past_the_period_end_it_is_the_renewal_that_is_pending(self) -> None:
        standing = at(PERIOD_END, record(latest_invoice_status=InvoiceStatus.OPEN))

        assert standing.standing is Standing.RENEWAL_PENDING
        assert standing.paid_through is None

    @pytest.mark.parametrize("invoice", [InvoiceStatus.OPEN, InvoiceStatus.DRAFT])
    def test_no_standing_carries_a_paid_through_date_for_an_unpaid_period(
        self, invoice: InvoiceStatus
    ) -> None:
        for candidate in (
            record(latest_invoice_status=invoice),
            record(latest_invoice_status=invoice, cancel_at_period_end=True),
            unpaid(latest_invoice_status=invoice, current_period_start=NOW),
        ):
            assert at(NOW, candidate).paid_through is None

    def test_of_two_granting_subscriptions_the_paid_one_is_shown(self) -> None:
        standing = at(
            NOW,
            record(
                subscription_id="sub_a_pending",
                latest_invoice_status=InvoiceStatus.OPEN,
                current_period_end=PERIOD_END + timedelta(days=5),
            ),
            record(subscription_id="sub_z_paid"),
        )

        assert standing.standing is Standing.ACTIVE
        assert standing.shown_subscription_id == "sub_z_paid"
        assert standing.granting_subscription_ids == frozenset({"sub_a_pending", "sub_z_paid"})
