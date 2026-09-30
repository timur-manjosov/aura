"""Phase 4c audit: when Pro starts and ends, proven at the second, through the runtime gate.

Written by the post-hoc audit of commit 9d0aa23 (reports/phase-4c-audit.md).
tests/test_billing_entitlement.py already pins the pure rules; these tests
drive the same boundaries through `PlanGate` -- the object every Pro trigger
actually asks -- with an injected clock, one second before, exactly at, and one
second after each boundary. They also pin what `BILLING_MODE=disabled` means at
runtime, and the grace-period behaviours the audit found.

A test marked ``xfail(strict=True)`` pins a defect the audit reported: it fails
today, and the day the defect is fixed it passes -- which strict mode turns
into a failure until the marker is removed.
"""

from __future__ import annotations

import inspect
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from typing import Final

import aiosqlite
import pytest

from aura.billing import (
    GracePolicy,
    PlanBasis,
    PlanGate,
    PlanTier,
    Standing,
    SubscriptionRecord,
    SubscriptionStatus,
)
from aura.billing.apply import apply_snapshot
from aura.billing.entitlement import InvoiceStatus
from aura.billing.plan_gate import grace_policy_from_settings
from aura.config import BillingMode, Settings
from aura.db.connection import utc_now
from aura.db.repository import init_schema
from aura.db.subscriptions import SubscriptionSnapshot

GUILD: Final = 100000000000000001
OTHER_GUILD: Final = 200000000000000002
PERIOD_START: Final = datetime(2026, 9, 1, 12, 0, 0, tzinfo=UTC)
PERIOD_END: Final = PERIOD_START + timedelta(days=30)
ONE_SECOND: Final = timedelta(seconds=1)
SECRET: Final = "audit-internal-api-secret-" + "x" * 20
POLICY: Final = GracePolicy(
    renewal_grace=timedelta(hours=72), payment_failure_grace=timedelta(days=7)
)


class MovableClock:
    """A clock a test sets explicitly; returns an aware UTC instant."""

    def __init__(self, moment: datetime) -> None:
        self.moment = moment

    def __call__(self) -> datetime:
        return self.moment


def record(**overrides: object) -> SubscriptionRecord:
    fields: dict[str, object] = {
        "subscription_id": "sub_Audit1",
        "guild_id": GUILD,
        "customer_id": "cus_Audit1",
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
    fields.update(overrides)
    return SubscriptionRecord.model_validate(fields)


def enforced_gate(clock: MovableClock, *records: SubscriptionRecord) -> PlanGate:
    return PlanGate(
        enforced=True,
        policy=POLICY,
        complimentary_guild_ids=frozenset(),
        records=records,
        clock=clock,
    )


def answers_around(gate: PlanGate, clock: MovableClock, boundary: datetime) -> list[bool]:
    """The gate's answer one second before, exactly at, and one second after `boundary`."""
    answers = []
    for moment in (boundary - ONE_SECOND, boundary, boundary + ONE_SECOND):
        clock.moment = moment
        answers.append(gate.allows_pro(GUILD))
    return answers


class TestBoundariesThroughTheGate:
    """Pro while ``now < access_until``; Free from that instant on, for every kind of end."""

    def test_an_active_subscription_ends_exactly_seventy_two_hours_after_its_period(
        self,
    ) -> None:
        clock = MovableClock(PERIOD_START)
        gate = enforced_gate(clock, record())

        assert answers_around(gate, clock, PERIOD_END + timedelta(hours=72)) == [
            True,
            False,
            False,
        ]

    def test_inside_the_renewal_grace_the_standing_says_the_renewal_is_pending(self) -> None:
        clock = MovableClock(PERIOD_END - ONE_SECOND)
        gate = enforced_gate(clock, record())

        before = gate.plan_for(GUILD).standing.standing
        clock.moment = PERIOD_END
        at = gate.plan_for(GUILD).standing.standing

        assert (before, at) == (Standing.ACTIVE, Standing.RENEWAL_PENDING)

    def test_cancel_at_period_end_ends_exactly_at_the_period_end(self) -> None:
        clock = MovableClock(PERIOD_START)
        gate = enforced_gate(clock, record(cancel_at_period_end=True))

        assert answers_around(gate, clock, PERIOD_END) == [True, False, False]

    def test_a_cancel_date_inside_the_period_ends_exactly_then(self) -> None:
        cancel_at = PERIOD_START + timedelta(days=10)
        clock = MovableClock(PERIOD_START)
        gate = enforced_gate(clock, record(cancel_at=cancel_at))

        assert answers_around(gate, clock, cancel_at) == [True, False, False]

    def test_a_failed_renewal_ends_exactly_seven_days_after_the_unpaid_period_began(
        self,
    ) -> None:
        clock = MovableClock(PERIOD_START)
        gate = enforced_gate(
            clock,
            record(status=SubscriptionStatus.PAST_DUE, latest_invoice_status=InvoiceStatus.OPEN),
        )

        assert answers_around(gate, clock, PERIOD_START + timedelta(days=7)) == [
            True,
            False,
            False,
        ]

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
    def test_a_status_stripe_treats_as_not_paying_is_free_at_the_first_second_of_the_period(
        self, status: SubscriptionStatus
    ) -> None:
        clock = MovableClock(PERIOD_START)
        gate = enforced_gate(clock, record(status=status))

        assert gate.allows_pro(GUILD) is False
        assert gate.plan_for(GUILD).standing.standing is Standing.ENDED

    def test_a_subscription_of_another_guild_grants_this_guild_nothing(self) -> None:
        clock = MovableClock(PERIOD_START)
        gate = enforced_gate(clock, record(guild_id=OTHER_GUILD))

        assert gate.allows_pro(GUILD) is False
        assert gate.allows_pro(OTHER_GUILD) is True


class TestTheClock:
    def test_the_production_clock_is_utc_and_aware(self) -> None:
        moment = utc_now()

        assert moment.tzinfo is not None
        assert moment.utcoffset() == timedelta(0)

    def test_the_gate_defaults_to_the_production_clock_and_accepts_an_injected_one(self) -> None:
        default = inspect.signature(PlanGate.__init__).parameters["clock"].default

        assert default is utc_now

    def test_a_naive_clock_is_refused_rather_than_compared_by_wall_time(self) -> None:
        gate = PlanGate(
            enforced=True,
            policy=POLICY,
            complimentary_guild_ids=frozenset(),
            records=[record()],
            clock=lambda: datetime(2026, 9, 2, 12, 0, 0),
        )

        with pytest.raises(ValueError):
            gate.allows_pro(GUILD)


class TestUnreachableStatusSource:
    """The web backend or Stripe gone quiet: the last snapshot applies, bounded, then Free."""

    def test_nothing_heard_after_an_active_period_is_trusted_for_seventy_two_hours_only(
        self,
    ) -> None:
        clock = MovableClock(PERIOD_END + timedelta(hours=71, minutes=59, seconds=59))
        gate = enforced_gate(clock, record())
        assert gate.allows_pro(GUILD) is True

        for later in (timedelta(hours=72), timedelta(days=30), timedelta(days=3650)):
            clock.moment = PERIOD_END + later
            assert gate.allows_pro(GUILD) is False

    def test_degrading_means_free_with_an_explained_standing_never_an_unknown_state(
        self,
    ) -> None:
        clock = MovableClock(PERIOD_END + timedelta(days=365))
        gate = enforced_gate(clock, record())

        plan = gate.plan_for(GUILD)

        assert plan.tier is PlanTier.FREE
        assert plan.basis is PlanBasis.SUBSCRIPTION
        assert plan.standing.standing is Standing.ENDED


class TestWhatDisabledMeans:
    """BILLING_MODE=disabled: every guild keeps every Pro feature, whatever is stored."""

    def test_the_shipped_default_is_disabled_and_the_gate_never_answers_free(self) -> None:
        settings = Settings(_env_file=None, discord_token="t")  # type: ignore[call-arg, arg-type]
        ended = record(status=SubscriptionStatus.CANCELED)
        lapsed = record(
            subscription_id="sub_Audit2",
            guild_id=OTHER_GUILD,
            current_period_start=PERIOD_START - timedelta(days=400),
            current_period_end=PERIOD_START - timedelta(days=370),
        )

        gate = PlanGate.from_settings(settings, records=[ended, lapsed])

        assert settings.billing_mode is BillingMode.DISABLED
        assert gate.enforced is False
        for guild_id in (GUILD, OTHER_GUILD, 1, 2**63 - 1):
            assert gate.allows_pro(guild_id) is True
            assert gate.plan_for(guild_id).basis is PlanBasis.BILLING_NOT_ENFORCED

    def test_enforced_is_only_ever_an_explicit_setting(self) -> None:
        settings = Settings(
            _env_file=None,  # type: ignore[call-arg]
            discord_token="t",  # type: ignore[arg-type]
            billing_mode=BillingMode.ENFORCED,
            internal_api_secret=SECRET,  # type: ignore[arg-type]
        )

        gate = PlanGate.from_settings(settings, records=[])

        assert gate.enforced is True
        assert gate.allows_pro(GUILD) is False
        assert grace_policy_from_settings(settings) == POLICY


@pytest.fixture
async def conn() -> AsyncIterator[aiosqlite.Connection]:
    connection = await aiosqlite.connect(":memory:")
    await init_schema(connection)
    yield connection
    await connection.close()


def snapshot(**overrides: object) -> SubscriptionSnapshot:
    fields: dict[str, object] = {
        "subscription_id": "sub_Audit1",
        "guild_id": GUILD,
        "customer_id": "cus_Audit1",
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
    }
    fields.update(overrides)
    return SubscriptionSnapshot.model_validate(fields)


class TestPaymentGraceAcrossPeriods:
    """The 7-day payment grace is documented as anchored and not extendable."""

    async def test_retries_inside_one_unpaid_period_cannot_extend_the_grace(
        self, conn: aiosqlite.Connection
    ) -> None:
        clock = MovableClock(PERIOD_START)
        gate = enforced_gate(clock)
        for version, event in enumerate(("evt_Fail1", "evt_Fail2", "evt_Fail3")):
            await apply_snapshot(
                conn,
                gate,
                snapshot=snapshot(
                    status=SubscriptionStatus.PAST_DUE, latest_invoice_status=InvoiceStatus.OPEN
                ),
                event_id=event,
                event_type="invoice.payment_failed",
                expected_version=version,
                now=PERIOD_START + timedelta(days=version * 3),
            )

        clock.moment = PERIOD_START + timedelta(days=7)
        assert gate.allows_pro(GUILD) is False

    async def test_an_unpaid_subscription_rolling_into_its_next_period_earns_no_new_grace(
        self, conn: aiosqlite.Connection
    ) -> None:
        clock = MovableClock(PERIOD_START)
        gate = enforced_gate(clock)
        await apply_snapshot(
            conn,
            gate,
            snapshot=snapshot(
                status=SubscriptionStatus.PAST_DUE, latest_invoice_status=InvoiceStatus.OPEN
            ),
            event_id="evt_FirstFailure",
            event_type="invoice.payment_failed",
            expected_version=0,
            now=PERIOD_START,
        )
        clock.moment = PERIOD_START + timedelta(days=8)
        assert gate.allows_pro(GUILD) is False  # the grace did end, as documented

        # Stripe's retry window is longer than the period (1 or 2 months), or the
        # account is set to "leave the subscription past-due": the next cycle
        # starts, a new invoice is created, and the subscription is STILL unpaid.
        next_start = PERIOD_END
        await apply_snapshot(
            conn,
            gate,
            snapshot=snapshot(
                status=SubscriptionStatus.PAST_DUE,
                latest_invoice_status=InvoiceStatus.OPEN,
                current_period_start=next_start,
                current_period_end=next_start + timedelta(days=30),
            ),
            event_id="evt_NextCycle",
            event_type="customer.subscription.updated",
            expected_version=1,
            now=next_start,
        )
        clock.moment = next_start + timedelta(days=1)

        assert gate.allows_pro(GUILD) is False


class TestDelayedFirstPayment:
    """A bank debit (SEPA) activates the subscription before the money arrives.

    Stripe documents that, for payment methods with delayed confirmation, the
    subscription goes straight to ``active`` while the first invoice stays
    ``open`` and its payment ``processing``; if the debit later fails, Stripe
    voids the invoice and leaves the subscription ``active``. Finding F-05 was
    decided in the remediation: Checkout offers cards only, so this state is an
    edge case rather than a path, and its GRANTING is deliberately unchanged
    -- but it must never be described as paid. These tests pin both halves.
    """

    def test_pro_is_granted_while_the_first_debit_is_still_processing(self) -> None:
        clock = MovableClock(PERIOD_START + timedelta(hours=1))
        gate = enforced_gate(clock, record(latest_invoice_status=InvoiceStatus.OPEN))

        plan = gate.plan_for(GUILD)

        assert plan.is_pro is True
        # /aura-plan and the dashboard say "payment pending", never "paid through".
        assert plan.standing.standing is Standing.PAYMENT_PENDING
        assert plan.standing.paid_through is None

    def test_a_failed_debit_voids_the_invoice_and_ends_pro_immediately(self) -> None:
        clock = MovableClock(PERIOD_START + timedelta(days=5))
        gate = enforced_gate(clock, record(latest_invoice_status=InvoiceStatus.VOID))

        assert gate.allows_pro(GUILD) is False


class TestACancelDateOnAnUnpaidSubscription:
    def test_a_cancel_date_inside_the_payment_grace_still_ends_pro_on_that_date(self) -> None:
        cancel_at = PERIOD_START + timedelta(days=2)
        clock = MovableClock(cancel_at + ONE_SECOND)
        gate = enforced_gate(
            clock,
            record(
                status=SubscriptionStatus.PAST_DUE,
                latest_invoice_status=InvoiceStatus.OPEN,
                cancel_at=cancel_at,
            ),
        )

        assert gate.allows_pro(GUILD) is False
