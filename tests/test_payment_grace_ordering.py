"""The payment-grace anchor under every delivery order: one grace per lapsed payment (F-04).

The anchor (`past_due_since`) is the one piece of subscription state that is
NOT a copy of what Stripe says -- it is folded, snapshot by snapshot, out of
the sequence of states the bot has stored. So the attack on it is the
Stripe-side disorder the rest of Phase 4c already survives: every permutation
of a lifecycle's webhook deliveries, each delivered twice at once, with Stripe
advancing between them.

The chain under test is the production one minus the sockets: the web
backend's real `sync_subscription` (read version, re-fetch, compare-and-swap,
retry), the real Stripe parser over the stand-in's dahlia objects, the bot's
real wire model (`ApplyRequest`), and the real `apply_snapshot` into a real
SQLite database and plan gate. tests/test_billing_contract.py drives the same
lifecycle through the two real HTTP services.

Two delivery models, because they prove different things:

  * RECONCILED: the six-hourly reconciliation runs at least once while Stripe
    sits in each state (every state here lasts days). Then the stored anchor
    must equal what Stripe's history implies, whatever order the webhooks
    arrive in -- the property the fix promises.
  * WEBHOOKS ONLY: nothing but the (reordered, duplicated) webhooks. Then the
    bot can only know the states some delivery happened to observe; the anchor
    must be exactly the fold over those, and never later than Stripe's current
    period start -- so even a bot that missed most of a lifecycle grants no more
    than the pre-fix rule would have.
"""

from __future__ import annotations

import asyncio
import itertools
import random
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, Final

import aiosqlite
import pytest
import pytest_asyncio

from aura.billing import GracePolicy, PlanGate, Standing
from aura.billing.apply import apply_snapshot
from aura.billing.internal_api import ApplyRequest
from aura.db.repository import init_schema
from aura.db.subscriptions import ApplyOutcome as BotApplyOutcome
from aura.db.subscriptions import count_processed_events, get_sync_state, load_subscription_records
from aura_web import bot_billing
from aura_web.billing_sync import SyncConflictError, sync_subscription
from aura_web.stripe_api import SubscriptionSnapshot as WebSnapshot
from aura_web.stripe_api import parse_subscription
from fake_stripe import DEFAULT_PRICE_ID, FakeSubscription

POLICY: Final = GracePolicy(
    renewal_grace=timedelta(hours=72), payment_failure_grace=timedelta(days=7)
)
T0: Final = 1_780_000_000
PERIOD_SECONDS: Final = 30 * 24 * 3600
ONE_SECOND: Final = timedelta(seconds=1)


def period_start(number: int) -> int:
    return T0 + (number - 1) * PERIOD_SECONDS


def moment(unix_seconds: int) -> datetime:
    return datetime.fromtimestamp(unix_seconds, tz=UTC)


@dataclass(frozen=True)
class StripeState:
    """What Stripe says about the subscription after one lifecycle step."""

    status: str
    invoice: str
    period: int


@dataclass(frozen=True)
class Lifecycle:
    """A lifecycle: each step's event type and the state Stripe is in after it."""

    steps: tuple[tuple[str, StripeState], ...]
    # What Stripe's full history implies for the final state -- written out by
    # hand, not computed by the code under test.
    expected_anchor_period: int | None


# Two rollovers: a failed renewal that stays unpaid into the next period, a
# recovery, and a second failure afterwards. The recovery must reset the anchor,
# so the second failure earns its own seven days -- once.
TWO_ROLLOVERS: Final = Lifecycle(
    steps=(
        ("invoice.paid", StripeState("active", "paid", 1)),
        ("invoice.payment_failed", StripeState("past_due", "open", 2)),
        ("customer.subscription.updated", StripeState("past_due", "open", 3)),
        ("invoice.paid", StripeState("active", "paid", 3)),
        ("invoice.payment_failed", StripeState("past_due", "open", 4)),
    ),
    expected_anchor_period=4,
)

# Three rollovers, and the write-off loop: after the retries run out Stripe
# marks the invoice uncollectible and (under its default status resolution)
# answers with `active` -- without anything being paid. The next period's
# renewal is drafted, then fails. Nothing was paid since period 2, so no new
# grace is due.
THREE_ROLLOVERS_WITH_A_WRITE_OFF: Final = Lifecycle(
    steps=(
        ("invoice.paid", StripeState("active", "paid", 1)),
        ("invoice.payment_failed", StripeState("past_due", "open", 2)),
        ("customer.subscription.updated", StripeState("past_due", "open", 3)),
        ("invoice.marked_uncollectible", StripeState("active", "uncollectible", 3)),
        ("invoice.created", StripeState("active", "draft", 4)),
        ("invoice.payment_failed", StripeState("past_due", "open", 4)),
    ),
    expected_anchor_period=2,
)

# The same history stopped inside the next renewal's draft window (V-05): Stripe
# reads `active` again after the write-off, and the new period's invoice is a
# draft that has not been charged yet. Nothing has been paid since period 2, so
# nothing may be granted -- not even for the hour before the charge fails.
INTO_THE_DRAFT_WINDOW_AFTER_A_WRITE_OFF: Final = Lifecycle(
    steps=THREE_ROLLOVERS_WITH_A_WRITE_OFF.steps[:5],
    expected_anchor_period=2,
)


def reference_anchor(observed: list[StripeState]) -> int | None:
    """The rule as the report states it, applied to exactly the states the bot saw.

    Written from the specification, independently of next_unpaid_since: a paid
    period in force clears the anchor, a past_due state sets it to the earliest
    unpaid period start seen since, anything else leaves it alone.
    """
    anchor: int | None = None
    for state in observed:
        if state.status in {"active", "trialing"} and state.invoice == "paid":
            anchor = None
        elif state.status == "past_due":
            start = period_start(state.period)
            anchor = start if anchor is None else min(anchor, start)
    return anchor


class MovableClock:
    def __init__(self) -> None:
        self.moment = moment(T0)

    def __call__(self) -> datetime:
        return self.moment


class InProcessStripe:
    """Stripe's GET /v1/subscriptions/{id}, answered through the real parser."""

    def __init__(self, rng: random.Random) -> None:
        self.subscriptions: dict[str, FakeSubscription] = {}
        self._rng = rng

    async def retrieve_subscription(self, subscription_id: str) -> WebSnapshot:
        # Yield (sometimes more than once) so concurrent syncs interleave the
        # way network round trips make them.
        for _ in range(self._rng.randint(0, 2)):
            await asyncio.sleep(0)
        payload = self.subscriptions[subscription_id].to_object(expand_invoice=True)
        return parse_subscription(payload, pro_price_id=DEFAULT_PRICE_ID)


class InProcessBot:
    """The bot's internal API contract, minus the socket: the same validation, the same apply."""

    def __init__(self, conn: aiosqlite.Connection, gate: PlanGate, clock: MovableClock) -> None:
        self._conn = conn
        self._gate = gate
        self._clock = clock

    async def get_sync_state(self, *, subscription_id: str, event_id: str | None) -> Any:
        state = await get_sync_state(self._conn, subscription_id=subscription_id, event_id=event_id)
        return bot_billing.SyncState(event_processed=state.event_processed, version=state.version)

    async def apply_snapshot(
        self,
        *,
        event_id: str | None,
        event_type: str | None,
        expected_version: int,
        snapshot: WebSnapshot,
    ) -> Any:
        request = ApplyRequest.model_validate(
            {
                "event_id": event_id,
                "event_type": event_type,
                "expected_version": expected_version,
                "snapshot": snapshot.internal_api_payload(),
            }
        )
        result = await apply_snapshot(
            self._conn,
            self._gate,
            snapshot=request.snapshot.to_snapshot(),
            event_id=request.event_id,
            event_type=request.event_type,
            expected_version=request.expected_version,
            now=self._clock(),
        )
        outcome = {
            BotApplyOutcome.APPLIED: bot_billing.ApplyOutcome.APPLIED,
            BotApplyOutcome.DUPLICATE_EVENT: bot_billing.ApplyOutcome.DUPLICATE,
            BotApplyOutcome.VERSION_CONFLICT: bot_billing.ApplyOutcome.VERSION_CONFLICT,
        }[result.outcome]
        return bot_billing.ApplyResult(outcome=outcome, version=result.version)


@dataclass
class World:
    conn: aiosqlite.Connection
    gate: PlanGate
    clock: MovableClock
    stripe: InProcessStripe
    bot: InProcessBot
    delivered_event_ids: set[str] = field(default_factory=set)


@pytest_asyncio.fixture
async def world() -> AsyncIterator[World]:
    conn = await aiosqlite.connect(":memory:")
    await init_schema(conn)
    clock = MovableClock()
    gate = PlanGate(
        enforced=True, policy=POLICY, complimentary_guild_ids=frozenset(), records=[], clock=clock
    )
    stripe = InProcessStripe(random.Random(4))
    try:
        yield World(conn, gate, clock, stripe, InProcessBot(conn, gate, clock))
    finally:
        await conn.close()


async def sync_until_settled(
    world: World, subscription_id: str, event: tuple[str, str] | None
) -> None:
    """One delivery (or one reconciliation), retried the way Stripe retries a 503."""
    event_id, event_type = event if event is not None else (None, None)
    for _ in range(20):
        try:
            await sync_subscription(
                stripe=world.stripe,  # type: ignore[arg-type]
                bot=world.bot,  # type: ignore[arg-type]
                subscription_id=subscription_id,
                event_id=event_id,
                event_type=event_type,
                live_mode=False,
            )
        except SyncConflictError:
            continue
        if event_id is not None:
            world.delivered_event_ids.add(event_id)
        return
    raise AssertionError(f"{subscription_id} never settled")


def apply_state(subscription: FakeSubscription, state: StripeState) -> None:
    subscription.status = state.status
    subscription.latest_invoice_status = state.invoice
    subscription.current_period_start = period_start(state.period)
    subscription.current_period_end = period_start(state.period) + PERIOD_SECONDS


async def run_ordering(
    world: World,
    lifecycle: Lifecycle,
    order: tuple[int, ...],
    *,
    label: str,
    reconcile_each_state: bool,
) -> tuple[str, list[StripeState]]:
    """Advance Stripe lazily and deliver events in `order`, each twice at once.

    An event can only be delivered once Stripe has emitted it, i.e. after its
    own state change, so Stripe is advanced to step i before event i goes out.
    Returns the subscription and the states the bot observed, in order.
    """
    subscription_id = f"sub_{label}"
    subscription = FakeSubscription(
        id=subscription_id,
        customer=f"cus_{label}",
        metadata={"aura_guild_id": str(1000 + len(world.stripe.subscriptions))},
        current_period_start=period_start(1),
        current_period_end=period_start(2),
    )
    world.stripe.subscriptions[subscription_id] = subscription
    observed: list[StripeState] = []
    reached = -1

    async def advance_to(step: int) -> None:
        nonlocal reached
        while reached < step:
            reached += 1
            state = lifecycle.steps[reached][1]
            apply_state(subscription, state)
            world.clock.moment = moment(period_start(state.period)) + timedelta(hours=reached)
            if reconcile_each_state:
                await sync_until_settled(world, subscription_id, None)
                observed.append(state)

    for index in order:
        await advance_to(index)
        event = (f"evt_{label}x{index}", lifecycle.steps[index][0])
        await asyncio.gather(
            sync_until_settled(world, subscription_id, event),
            sync_until_settled(world, subscription_id, event),
        )
        observed.append(lifecycle.steps[reached][1])
    return subscription_id, observed


async def stored_record(world: World, subscription_id: str) -> Any:
    (record,) = [
        record
        for record in await load_subscription_records(world.conn)
        if record.subscription_id == subscription_id
    ]
    return record


def orderings(lifecycle: Lifecycle) -> list[tuple[int, ...]]:
    return list(itertools.permutations(range(len(lifecycle.steps))))


class TestReconciledEveryOrderEndsOnStripesHistory:
    @pytest.mark.parametrize(
        "lifecycle",
        [TWO_ROLLOVERS, THREE_ROLLOVERS_WITH_A_WRITE_OFF],
        ids=["two-rollovers-recovery-second-failure", "three-rollovers-write-off"],
    )
    async def test_every_order_with_duplicates_stores_the_anchor_stripes_history_implies(
        self, world: World, lifecycle: Lifecycle
    ) -> None:
        expected_anchor = (
            None
            if lifecycle.expected_anchor_period is None
            else moment(period_start(lifecycle.expected_anchor_period))
        )
        final = lifecycle.steps[-1][1]
        for number, order in enumerate(orderings(lifecycle)):
            subscription_id, _ = await run_ordering(
                world, lifecycle, order, label=f"R{number}", reconcile_each_state=True
            )

            record = await stored_record(world, subscription_id)
            assert record.status.value == final.status, order
            assert record.current_period_start == moment(period_start(final.period)), order
            assert record.past_due_since == expected_anchor, order

        assert await count_processed_events(world.conn) == len(world.delivered_event_ids)

    async def test_the_second_failure_earns_exactly_one_grace_to_the_second(
        self, world: World
    ) -> None:
        subscription_id, _ = await run_ordering(
            world,
            TWO_ROLLOVERS,
            (4, 0, 3, 1, 2),
            label="Boundary",
            reconcile_each_state=True,
        )
        record = await stored_record(world, subscription_id)
        boundary = moment(period_start(4)) + timedelta(days=7)

        answers = []
        for instant in (boundary - ONE_SECOND, boundary, boundary + ONE_SECOND):
            world.clock.moment = instant
            answers.append(world.gate.allows_pro(record.guild_id))

        assert answers == [True, False, False]

    async def test_after_a_write_off_the_next_failure_is_free_at_once(self, world: World) -> None:
        subscription_id, _ = await run_ordering(
            world,
            THREE_ROLLOVERS_WITH_A_WRITE_OFF,
            (5, 4, 3, 2, 1, 0),
            label="WriteOff",
            reconcile_each_state=True,
        )
        record = await stored_record(world, subscription_id)
        world.clock.moment = moment(period_start(4)) + timedelta(hours=6)

        plan = world.gate.plan_for(record.guild_id)

        assert plan.is_pro is False
        assert plan.standing.standing is Standing.ENDED


class TestTheDraftWindowAfterAWriteOff:
    async def test_every_order_stores_the_old_anchor_and_grants_nothing_before_the_charge(
        self, world: World
    ) -> None:
        lifecycle = INTO_THE_DRAFT_WINDOW_AFTER_A_WRITE_OFF
        for number, order in enumerate(orderings(lifecycle)):
            subscription_id, _ = await run_ordering(
                world, lifecycle, order, label=f"D{number}", reconcile_each_state=True
            )
            record = await stored_record(world, subscription_id)
            world.clock.moment = moment(period_start(4)) + timedelta(minutes=30)

            plan = world.gate.plan_for(record.guild_id)

            assert (record.status.value, record.latest_invoice_status.value) == (
                "active",
                "draft",
            ), order
            assert record.past_due_since == moment(period_start(2)), order
            assert plan.is_pro is False, order
            assert plan.standing.standing is Standing.ENDED, order

    async def test_with_webhooks_only_it_is_free_exactly_when_an_unpaid_period_was_seen(
        self, world: World
    ) -> None:
        # Whatever order the deliveries come in, the bot can only have folded
        # the states it saw. Seen any past_due at all: nothing is granted.
        # Seen none (every failure's delivery lost): the draft reads like any
        # renewal in progress, which is the bounded pre-V-05 answer.
        lifecycle = INTO_THE_DRAFT_WINDOW_AFTER_A_WRITE_OFF
        for number, order in enumerate(orderings(lifecycle)):
            subscription_id, observed = await run_ordering(
                world, lifecycle, order, label=f"DW{number}", reconcile_each_state=False
            )
            record = await stored_record(world, subscription_id)
            world.clock.moment = moment(period_start(4)) + timedelta(minutes=30)

            plan = world.gate.plan_for(record.guild_id)

            assert record.latest_invoice_status.value == "draft", order
            assert plan.is_pro is (reference_anchor(observed) is None), order


class TestTheSandboxWriteOffSequenceThroughTheStack:
    """C10-C12 of the real sandbox run, step for step, parser to gate.

    Anchored at the sandbox's first failed renewal; the periods are a fixed 31
    days rather than calendar months, which changes no decision below.

    Stripe's side is the stand-in driven the way the sandbox behaved: renewals
    start as drafts (renew), then are charged (settle_renewal), and marking an
    invoice uncollectible leaves the subscription `active`.
    """

    FIRST_FAILURE = 1_793_368_865  # 2026-10-30 14:01:05 UTC, clock B's first failed renewal

    async def test_each_step_decides_what_the_sandbox_run_recorded_and_the_draft_is_free(
        self, world: World
    ) -> None:
        month = 31 * 24 * 3600
        subscription = FakeSubscription(
            id="sub_sandboxClockB",
            customer="cus_sandboxClockB",
            metadata={"aura_guild_id": "1200", "aura_discord_user_id": "5000"},
            current_period_start=self.FIRST_FAILURE - month,
            current_period_end=self.FIRST_FAILURE,
        )
        world.stripe.subscriptions[subscription.id] = subscription
        decisions: list[tuple[str, str, bool]] = []

        async def observe(label: str, *, minutes_after_period_start: int = 30) -> None:
            await sync_until_settled(world, subscription.id, None)
            world.clock.moment = moment(subscription.current_period_start) + timedelta(
                minutes=minutes_after_period_start
            )
            plan = world.gate.plan_for(1200)
            decisions.append((label, plan.standing.standing.value, plan.is_pro))

        subscription.renew()
        subscription.settle_renewal(paid=False)
        await observe("C10a failed renewal")
        subscription.renew()
        subscription.settle_renewal(paid=False)
        await observe("C10c rolled over unpaid")
        subscription.latest_invoice_status = "uncollectible"
        subscription.status = "active"
        await observe("C11a written off")
        subscription.renew()
        await observe("C11b draft")
        subscription.settle_renewal(paid=False)
        await observe("C11b charge failed")
        subscription.settle_renewal(paid=True)
        await observe("C12a paid")
        subscription.renew()
        await observe("C12b-1 draft after a payment")
        subscription.settle_renewal(paid=False)
        await observe("C12b later failure")

        assert decisions == [
            ("C10a failed renewal", "payment_grace", True),
            ("C10c rolled over unpaid", "ended", False),
            ("C11a written off", "ended", False),
            ("C11b draft", "ended", False),
            ("C11b charge failed", "ended", False),
            ("C12a paid", "active", True),
            # A draft on a subscription that owes nothing is still Pro (V-05
            # only withholds it while an unpaid period is recorded).
            ("C12b-1 draft after a payment", "payment_pending", True),
            ("C12b later failure", "payment_grace", True),
        ]
        record = await stored_record(world, subscription.id)
        assert record.past_due_since == moment(subscription.current_period_start)


class TestWebhooksOnlyTheAnchorIsTheFoldOverWhatWasSeen:
    @pytest.mark.parametrize(
        "lifecycle",
        [TWO_ROLLOVERS, THREE_ROLLOVERS_WITH_A_WRITE_OFF],
        ids=["two-rollovers-recovery-second-failure", "three-rollovers-write-off"],
    )
    async def test_every_order_with_duplicates_stores_exactly_the_fold_and_never_more_grace(
        self, world: World, lifecycle: Lifecycle
    ) -> None:
        final = lifecycle.steps[-1][1]
        final_start = moment(period_start(final.period))
        for number, order in enumerate(orderings(lifecycle)):
            subscription_id, observed = await run_ordering(
                world, lifecycle, order, label=f"W{number}", reconcile_each_state=False
            )

            record = await stored_record(world, subscription_id)
            expected = reference_anchor(observed)
            assert record.past_due_since == (None if expected is None else moment(expected)), order
            # Never later than Stripe's current period start: whatever the bot
            # missed, the grace it grants is at most the pre-fix rule's.
            assert record.past_due_since is None or record.past_due_since <= final_start, order
            assert record.status.value == final.status, order
