"""aura_web.billing_sync: the one path by which a plan changes, and the reconciler that heals lost events."""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
from unittest.mock import patch

import httpx
import pytest
import pytest_asyncio
from fake_bot_billing import FakeBotBillingState, create_fake_bot_billing
from fake_stripe import FakeStripeState, create_fake_stripe

from aura_web.billing_sync import (
    MAX_SYNC_ATTEMPTS,
    LivemodeMismatchError,
    SyncConflictError,
    SyncOutcome,
    reconcile_subscriptions,
    run_reconciler,
    sync_subscription,
)
from aura_web.bot_billing import BotBillingClient, BotBillingError
from aura_web.stripe_api import StripeClient, StripeUnavailableError

NOW = 1_757_764_800


@dataclass
class Clients:
    stripe: StripeClient
    bot: BotBillingClient


@pytest_asyncio.fixture
async def clients(stripe_state: FakeStripeState, bot_billing_state: FakeBotBillingState) -> AsyncIterator[Clients]:
    async with (
        httpx.AsyncClient(transport=httpx.ASGITransport(app=create_fake_stripe(stripe_state))) as stripe_http,
        httpx.AsyncClient(transport=httpx.ASGITransport(app=create_fake_bot_billing(bot_billing_state))) as bot_http,
    ):
        yield Clients(
            stripe=StripeClient(
                stripe_http, api_base="https://stripe.test", secret_key=stripe_state.secret_key,
                price_id=stripe_state.price_id, checkout_success_url="https://f/", checkout_cancel_url="https://f/",
                portal_return_url="https://f/",
            ),
            bot=BotBillingClient(bot_http, base_url="https://bot.test", secret=bot_billing_state.secret),
        )


async def sync(clients: Clients, subscription_id: str, event_id: str | None = "evt_1", live_mode: bool = False) -> SyncOutcome:
    return await sync_subscription(
        stripe=clients.stripe, bot=clients.bot, subscription_id=subscription_id, event_id=event_id,
        event_type=None if event_id is None else "customer.subscription.updated", live_mode=live_mode,
    )


class TestSync:
    async def test_applies_then_recognises_the_duplicate(self, clients, stripe_state, bot_billing_state) -> None:
        subscription = stripe_state.add_subscription(guild_id="1000", purchaser_user_id="5000", now=NOW)

        assert await sync(clients, subscription.id) is SyncOutcome.APPLIED
        assert await sync(clients, subscription.id) is SyncOutcome.DUPLICATE
        assert bot_billing_state.versions[subscription.id] == 1

    async def test_a_subscription_from_the_other_mode_is_refused_before_the_bot_hears_of_it(self, clients, stripe_state, bot_billing_state) -> None:
        subscription = stripe_state.add_subscription(guild_id="1000", purchaser_user_id="5000", now=NOW, livemode=True)

        with pytest.raises(LivemodeMismatchError):
            await sync(clients, subscription.id)

        assert "apply" not in bot_billing_state.request_log

    async def test_a_lost_race_is_retried_and_then_wins(self, clients, stripe_state, bot_billing_state) -> None:
        subscription = stripe_state.add_subscription(guild_id="1000", purchaser_user_id="5000", now=NOW)
        lost = 0

        async def lose_once(body):
            nonlocal lost
            if lost == 0:
                lost += 1
                bot_billing_state.commit_competing_write(subscription.id)

        bot_billing_state.before_apply = lose_once

        assert await sync(clients, subscription.id) is SyncOutcome.APPLIED
        assert bot_billing_state.request_log.count("apply") == 2

    async def test_racing_forever_raises_after_a_bounded_number_of_attempts(self, clients, stripe_state, bot_billing_state) -> None:
        subscription = stripe_state.add_subscription(guild_id="1000", purchaser_user_id="5000", now=NOW)

        async def always_lose(body):
            bot_billing_state.commit_competing_write(subscription.id)

        bot_billing_state.before_apply = always_lose

        with pytest.raises(SyncConflictError):
            await sync(clients, subscription.id)
        assert bot_billing_state.request_log.count("apply") == MAX_SYNC_ATTEMPTS

    async def test_stripe_being_down_never_reaches_the_bot(self, clients, stripe_state, bot_billing_state) -> None:
        subscription = stripe_state.add_subscription(guild_id="1000", purchaser_user_id="5000", now=NOW)
        stripe_state.fail_retrieve_status = 503

        with pytest.raises(StripeUnavailableError):
            await sync(clients, subscription.id)
        assert "apply" not in bot_billing_state.request_log

    async def test_a_malformed_bot_answer_is_an_error_not_a_guess(self, stripe_state, bot_billing_state) -> None:
        subscription = stripe_state.add_subscription(guild_id="1000", purchaser_user_id="5000", now=NOW)
        weird = httpx.MockTransport(lambda _: httpx.Response(200, json={"event_processed": "no", "version": 0}))
        async with httpx.AsyncClient(transport=weird) as bot_http, httpx.AsyncClient(
            transport=httpx.ASGITransport(app=create_fake_stripe(stripe_state))
        ) as stripe_http:
            broken = Clients(
                stripe=StripeClient(stripe_http, api_base="https://s", secret_key=stripe_state.secret_key, price_id="price_x",
                                    checkout_success_url="https://f/", checkout_cancel_url="https://f/", portal_return_url="https://f/"),
                bot=BotBillingClient(bot_http, base_url="https://b", secret="x" * 40),
            )
            with pytest.raises(BotBillingError):
                await sync(broken, subscription.id)


class TestReconciliation:
    async def test_every_aura_subscription_is_re_synced_and_others_are_skipped(self, clients, stripe_state, bot_billing_state) -> None:
        paying = [stripe_state.add_subscription(guild_id=str(1000 + i), purchaser_user_id="5000", now=NOW) for i in range(3)]
        stripe_state.add_subscription(guild_id=None, purchaser_user_id=None, now=NOW)

        report = await reconcile_subscriptions(stripe=clients.stripe, bot=clients.bot, live_mode=False)

        assert report.examined == 3
        assert report.applied == 3
        assert set(bot_billing_state.snapshots) == {subscription.id for subscription in paying}
        # Reconciliation records no event IDs: there is no event to deduplicate.
        assert bot_billing_state.processed_events == {}

    async def test_an_event_lost_while_the_web_backend_was_down_is_healed(self, clients, stripe_state, bot_billing_state) -> None:
        subscription = stripe_state.add_subscription(guild_id="1000", purchaser_user_id="5000", now=NOW)
        await sync(clients, subscription.id)
        subscription.status = "canceled"  # the deletion event never arrived

        await reconcile_subscriptions(stripe=clients.stripe, bot=clients.bot, live_mode=False)

        assert bot_billing_state.snapshots[subscription.id]["status"] == "canceled"

    async def test_one_broken_subscription_does_not_stop_the_others(self, clients, stripe_state, bot_billing_state) -> None:
        good = stripe_state.add_subscription(guild_id="1000", purchaser_user_id="5000", now=NOW)
        bad = stripe_state.add_subscription(guild_id="2000", purchaser_user_id="5000", now=NOW, status="from_the_future")

        report = await reconcile_subscriptions(stripe=clients.stripe, bot=clients.bot, live_mode=False)

        assert report.failed == [bad.id]
        assert good.id in bot_billing_state.snapshots

    async def test_the_loop_survives_a_failing_pass(self, clients) -> None:
        passes = 0

        async def flaky(**_kwargs):
            nonlocal passes
            passes += 1
            if passes == 1:
                raise RuntimeError("boom")
            from aura_web.billing_sync import ReconciliationReport
            return ReconciliationReport()

        with patch("aura_web.billing_sync.reconcile_subscriptions", flaky):
            task = asyncio.create_task(
                run_reconciler(stripe=clients.stripe, bot=clients.bot, live_mode=False, interval_seconds=0.001, first_delay_seconds=0)
            )
            for _ in range(200):
                if passes >= 3:
                    break
                await asyncio.sleep(0.005)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        assert passes >= 3
