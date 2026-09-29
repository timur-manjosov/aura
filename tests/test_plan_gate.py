"""aura.billing.plan_gate: the runtime Free/Pro answer, its modes, and its in-memory view."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from aura.billing import (
    GracePolicy,
    PlanBasis,
    PlanGate,
    PlanTier,
    SubscriptionRecord,
    SubscriptionStatus,
)
from aura.billing.entitlement import InvoiceStatus
from aura.config import BillingMode, Settings

NOW = datetime(2026, 9, 13, 12, 0, 0, tzinfo=UTC)
POLICY = GracePolicy(renewal_grace=timedelta(hours=72), payment_failure_grace=timedelta(days=7))
GUILD_A = 100000000000000001
GUILD_B = 200000000000000002


def record(**overrides: object) -> SubscriptionRecord:
    values: dict[str, object] = {
        "subscription_id": "sub_A",
        "guild_id": GUILD_A,
        "customer_id": "cus_A",
        "purchaser_user_id": 5000,
        "status": SubscriptionStatus.ACTIVE,
        "cancel_at_period_end": False,
        "cancel_at": None,
        "collection_paused": False,
        "latest_invoice_status": InvoiceStatus.PAID,
        "current_period_start": NOW - timedelta(days=1),
        "current_period_end": NOW + timedelta(days=29),
        "livemode": False,
        "version": 1,
        "confirmed_at": NOW,
    }
    values.update(overrides)
    return SubscriptionRecord(**values)  # type: ignore[arg-type]


class Clock:
    def __init__(self, moment: datetime) -> None:
        self.moment = moment

    def __call__(self) -> datetime:
        return self.moment


def gate(
    *records: SubscriptionRecord,
    clock: Clock | None = None,
    complimentary: frozenset[int] = frozenset(),
) -> PlanGate:
    return PlanGate(
        enforced=True,
        policy=POLICY,
        complimentary_guild_ids=complimentary,
        records=records,
        clock=clock or Clock(NOW),
    )


class TestUnenforced:
    def test_every_guild_may_use_pro(self) -> None:
        unenforced = PlanGate.unenforced()

        assert unenforced.allows_pro(GUILD_A)
        assert unenforced.plan_for(GUILD_A).basis is PlanBasis.BILLING_NOT_ENFORCED

    def test_the_shipped_default_settings_are_unenforced(self) -> None:
        settings = Settings(_env_file=None, discord_token="t")  # type: ignore[call-arg]

        assert PlanGate.from_settings(settings, records=[]).enforced is False


class TestEnforced:
    def test_a_guild_without_a_subscription_is_free(self) -> None:
        assert not gate().allows_pro(GUILD_A)
        assert gate().plan_for(GUILD_A).tier is PlanTier.FREE

    def test_a_guild_with_an_active_subscription_is_pro_and_another_guild_is_not(self) -> None:
        paying = gate(record())

        assert paying.allows_pro(GUILD_A)
        assert not paying.allows_pro(GUILD_B)

    def test_the_answer_changes_exactly_when_the_clock_crosses_the_boundary(self) -> None:
        clock = Clock(NOW)
        paying = gate(record(), clock=clock)
        boundary = record().current_period_end + timedelta(hours=72)

        clock.moment = boundary - timedelta(microseconds=1)
        assert paying.allows_pro(GUILD_A)
        clock.moment = boundary
        assert not paying.allows_pro(GUILD_A)

    def test_a_complimentary_guild_is_pro_without_paying(self) -> None:
        comped = gate(complimentary=frozenset({GUILD_B}))

        assert comped.allows_pro(GUILD_B)
        assert comped.plan_for(GUILD_B).basis is PlanBasis.COMPLIMENTARY
        assert not comped.allows_pro(GUILD_A)


class TestRecordApplied:
    def test_a_newer_version_replaces_the_view(self) -> None:
        view = gate(record(version=1))

        assert view.record_applied(record(version=2, status=SubscriptionStatus.CANCELED)) is True
        assert not view.allows_pro(GUILD_A)

    @pytest.mark.parametrize("version", [1, 0])
    def test_an_equal_or_older_version_is_ignored(self, version: int) -> None:
        view = gate(record(version=1))

        assert (
            view.record_applied(record(version=version, status=SubscriptionStatus.CANCELED))
            is False
        )
        assert view.allows_pro(GUILD_A)

    def test_out_of_order_bookkeeping_cannot_move_the_view_backwards(self) -> None:
        view = gate()
        view.record_applied(record(version=3, status=SubscriptionStatus.CANCELED))
        view.record_applied(record(version=2, status=SubscriptionStatus.ACTIVE))

        assert not view.allows_pro(GUILD_A)

    def test_a_subscription_moved_to_another_guild_leaves_the_first_one(self) -> None:
        view = gate(record(version=1))

        view.record_applied(record(version=2, guild_id=GUILD_B))

        assert not view.allows_pro(GUILD_A)
        assert view.allows_pro(GUILD_B)
        assert view.records_for(GUILD_A) == ()

    def test_records_are_listed_newest_period_first(self) -> None:
        older = record(subscription_id="sub_old", current_period_end=NOW + timedelta(days=1))
        newer = record(subscription_id="sub_new", current_period_end=NOW + timedelta(days=40))

        assert [r.subscription_id for r in gate(older, newer).records_for(GUILD_A)] == [
            "sub_new",
            "sub_old",
        ]


class TestFromSettings:
    def test_mode_grace_periods_and_complimentary_guilds_come_from_settings(self) -> None:
        settings = Settings(
            _env_file=None,  # type: ignore[call-arg]
            discord_token="t",
            billing_mode=BillingMode.ENFORCED,
            internal_api_secret="s" * 40,
            billing_renewal_grace_hours=1.5,
            billing_payment_grace_days=2,
            billing_complimentary_guild_ids=f" {GUILD_B}, ,",
        )

        configured = PlanGate.from_settings(settings, records=[])

        assert configured.enforced
        assert configured.policy.renewal_grace == timedelta(hours=1.5)
        assert configured.policy.payment_failure_grace == timedelta(days=2)
        assert configured.complimentary_guild_ids == frozenset({GUILD_B})
