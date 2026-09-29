"""The runtime answer to "may this guild use Pro features right now?".

Every Pro trigger asks this object, and nothing else, so there is exactly one
place where Free and Pro are told apart -- the same "one seam" shape
Settings.resolve_model gives model selection.

AN IN-MEMORY VIEW OF THE SUBSCRIPTION TABLE, WRITTEN THROUGH, NEVER RE-READ.
The gate is consulted on every message in every proactive- or extraction-enabled
channel, so it must be cheap, and it must not be able to fail. Both follow
from the architecture Phase 4c chose: this process is the only writer of its
database, and the only writer of subscription rows is aura.billing.apply, which
updates this view in the same step it commits. The view therefore cannot drift
from the table while the process runs, and a subscription check never touches
the database -- so "the subscription store was unreachable" is not a state a
Pro trigger can observe. The one channel that CAN be unreachable is the one
that brings Stripe's news to this process, and its bounded grace lives in the
rules themselves (aura.billing.entitlement), where it is tested to the second.

Versions only move forward in this view: a record replaces the one it holds
only if its version is higher. The database already guarantees writes commit in
version order; this makes the view hold that property even if two applies ever
finished their bookkeeping in the opposite order to their commits.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable
from datetime import datetime, timedelta

from aura.billing.entitlement import GracePolicy, GuildPlan, SubscriptionRecord, decide_plan
from aura.config import BillingMode, Settings
from aura.db.connection import utc_now


def grace_policy_from_settings(settings: Settings) -> GracePolicy:
    """Build the grace policy from the two configured periods."""
    return GracePolicy(
        renewal_grace=timedelta(hours=settings.billing_renewal_grace_hours),
        payment_failure_grace=timedelta(days=settings.billing_payment_grace_days),
    )


class PlanGate:
    """Decides each guild's plan from the subscriptions this process knows about."""

    def __init__(
        self,
        *,
        enforced: bool,
        policy: GracePolicy,
        complimentary_guild_ids: frozenset[int],
        records: Iterable[SubscriptionRecord],
        clock: Callable[[], datetime] = utc_now,
    ) -> None:
        self._enforced = enforced
        self._policy = policy
        self._complimentary_guild_ids = complimentary_guild_ids
        self._clock = clock
        self._by_subscription: dict[str, SubscriptionRecord] = {}
        self._by_guild: dict[int, dict[str, SubscriptionRecord]] = {}
        for record in records:
            self.record_applied(record)

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        *,
        records: Iterable[SubscriptionRecord],
        clock: Callable[[], datetime] = utc_now,
    ) -> PlanGate:
        """The production gate: mode, grace periods and complimentary guilds from settings."""
        return cls(
            enforced=settings.billing_mode is BillingMode.ENFORCED,
            policy=grace_policy_from_settings(settings),
            complimentary_guild_ids=settings.complimentary_guild_ids,
            records=records,
            clock=clock,
        )

    @classmethod
    def unenforced(cls) -> PlanGate:
        """A gate that allows every Pro feature -- what BILLING_MODE=disabled means."""
        return cls(
            enforced=False,
            policy=GracePolicy(renewal_grace=timedelta(0), payment_failure_grace=timedelta(0)),
            complimentary_guild_ids=frozenset(),
            records=(),
        )

    @property
    def enforced(self) -> bool:
        """Whether this gate can ever answer Free."""
        return self._enforced

    @property
    def policy(self) -> GracePolicy:
        """The grace periods this gate decides with."""
        return self._policy

    @property
    def complimentary_guild_ids(self) -> frozenset[int]:
        """Guilds on Pro by the operator's decision rather than a subscription."""
        return self._complimentary_guild_ids

    def record_applied(self, record: SubscriptionRecord) -> bool:
        """Adopt a newly committed record. Returns whether the view changed.

        A record whose version is not newer than the one already held is
        ignored, so the view can only ever move forward.
        """
        existing = self._by_subscription.get(record.subscription_id)
        if existing is not None and existing.version >= record.version:
            return False
        if existing is not None and existing.guild_id != record.guild_id:
            previous_guild = self._by_guild.get(existing.guild_id, {})
            previous_guild.pop(record.subscription_id, None)
            if not previous_guild:
                self._by_guild.pop(existing.guild_id, None)
        self._by_subscription[record.subscription_id] = record
        self._by_guild.setdefault(record.guild_id, {})[record.subscription_id] = record
        return True

    def records_for(self, guild_id: int) -> tuple[SubscriptionRecord, ...]:
        """Every known subscription for one guild, newest period first."""
        records = self._by_guild.get(guild_id, {}).values()
        return tuple(
            sorted(records, key=lambda record: (record.current_period_end, record.subscription_id), reverse=True)
        )

    def plan_for(self, guild_id: int) -> GuildPlan:
        """The guild's plan at this instant, with its full standing for display."""
        return decide_plan(
            guild_id=guild_id,
            records=self._by_guild.get(guild_id, {}).values(),
            now=self._clock(),
            policy=self._policy,
            enforced=self._enforced,
            complimentary=guild_id in self._complimentary_guild_ids,
        )

    def allows_pro(self, guild_id: int) -> bool:
        """Whether Pro-only triggers may run for this guild right now."""
        if not self._enforced:
            return True
        return self.plan_for(guild_id).is_pro
