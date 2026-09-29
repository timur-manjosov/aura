"""The web backend and the bot, together: the real client against the real internal API, over a real socket.

web/backend/tests exercises the web backend against fake_bot_billing.py, and
tests/test_internal_api.py exercises the bot's API on its own. Neither proves
the two AGREE. This file does, in two ways:

  * The contract: the same scripted sequence of calls is made through the web
    backend's production BotBillingClient against the stand-in and against the
    real aiohttp API on a real port, and the answers must be identical -- so the
    stand-in cannot quietly drift from what it stands in for. And a snapshot
    produced by the web backend's own Stripe parser must be accepted by the
    bot's own validator, byte for byte.

  * The whole path: a signed Stripe webhook enters the real web application,
    the real sync fetches the subscription from the Stripe stand-in, and the
    real bot API commits it to a real SQLite database whose plan gate then
    answers Pro -- and Free again after the subscription is canceled, with the
    brief's attacks (a forged signature, a duplicate delivery, contradictory
    events racing) run against the real end rather than a double.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import timedelta

import aiosqlite
import httpx
import pytest
import pytest_asyncio

from aura.billing import GracePolicy, PlanGate
from aura.billing.internal_api import InternalApiServer, start_internal_api
from aura.db.repository import init_schema
from aura.db.subscriptions import count_processed_events, load_subscription_records
from aura_web.app import create_app
from aura_web.bot_billing import ApplyOutcome, BotBillingClient, BotBillingError
from aura_web.config import WebSettings
from aura_web.discord_api import DiscordClient
from aura_web.stripe_api import StripeClient, parse_subscription
from fake_bot_billing import FakeBotBillingState, create_fake_bot_billing
from fake_discord import (
    PERMISSION_MANAGE_GUILD,
    FakeDiscordState,
    FakeGuild,
    FakeUser,
    create_fake_discord,
)
from fake_stripe import FakeStripeState, create_fake_stripe, sign_webhook

SECRET = "contract-test-internal-api-secret-" + "0" * 20
FRONTEND = "https://frontend.test"
POLICY = GracePolicy(renewal_grace=timedelta(hours=72), payment_failure_grace=timedelta(days=7))


@dataclass
class RealBot:
    conn: aiosqlite.Connection
    gate: PlanGate
    server: InternalApiServer

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.server.bound_port}"


@pytest_asyncio.fixture
async def real_bot() -> AsyncIterator[RealBot]:
    conn = await aiosqlite.connect(":memory:")
    await init_schema(conn)
    gate = PlanGate(enforced=True, policy=POLICY, complimentary_guild_ids=frozenset(), records=[])
    bot = RealBot(
        conn, gate, await start_internal_api(conn, gate, secret=SECRET, host="127.0.0.1", port=0)
    )
    try:
        yield bot
    finally:
        # The attribute, not the first server: a test may have restarted it.
        await bot.server.stop()
        await conn.close()


def stripe_snapshot(stripe_state: FakeStripeState, **overrides):
    subscription = stripe_state.add_subscription(
        guild_id="1000", purchaser_user_id="5000", now=int(time.time()), **overrides
    )
    return parse_subscription(subscription.to_object(expand_invoice=True))


async def scripted_sequence(client: BotBillingClient, snapshot) -> list[tuple[str, int]]:
    """The same calls, in the same order, against whichever implementation the client points at."""
    observed: list[tuple[str, int]] = []
    state = await client.get_sync_state(
        subscription_id=snapshot.subscription_id, event_id="evt_Contract1"
    )
    observed.append((f"processed={state.event_processed}", state.version))
    for event_id, expected in (
        ("evt_Contract1", 0),
        ("evt_Contract1", 1),
        ("evt_Contract2", 0),
        ("evt_Contract2", 1),
    ):
        result = await client.apply_snapshot(
            event_id=event_id,
            event_type="customer.subscription.updated",
            expected_version=expected,
            snapshot=snapshot,
        )
        observed.append((result.outcome.value, result.version))
    state = await client.get_sync_state(
        subscription_id=snapshot.subscription_id, event_id="evt_Contract2"
    )
    observed.append((f"processed={state.event_processed}", state.version))
    reconciliation = await client.apply_snapshot(
        event_id=None, event_type=None, expected_version=2, snapshot=snapshot
    )
    observed.append((reconciliation.outcome.value, reconciliation.version))
    return observed


class TestContract:
    async def test_the_stand_in_and_the_real_api_answer_the_same_sequence_identically(
        self, real_bot
    ) -> None:
        stripe_state = FakeStripeState()
        snapshot = stripe_snapshot(stripe_state)
        fake_state = FakeBotBillingState(secret=SECRET)

        async with httpx.AsyncClient() as real_http:
            real = await scripted_sequence(
                BotBillingClient(real_http, base_url=real_bot.base_url, secret=SECRET), snapshot
            )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=create_fake_bot_billing(fake_state))
        ) as fake_http:
            fake = await scripted_sequence(
                BotBillingClient(fake_http, base_url="https://bot.test", secret=SECRET), snapshot
            )

        assert real == fake
        assert real == [
            ("processed=False", 0),
            ("applied", 1),
            ("duplicate", 1),
            ("version_conflict", 1),
            ("applied", 2),
            ("processed=True", 2),
            ("applied", 3),
        ]

    @pytest.mark.parametrize(
        "overrides",
        [
            {},
            {"status": "past_due", "latest_invoice_status": "open"},
            {"cancel_at_period_end": True, "cancel_at": int(time.time()) + 86400},
            {"pause_collection": {"behavior": "void"}},
            {"latest_invoice_status": None},
        ],
    )
    async def test_every_snapshot_the_web_parser_produces_is_accepted_by_the_bot_validator(
        self, real_bot, overrides
    ) -> None:
        snapshot = stripe_snapshot(FakeStripeState(), **overrides)

        async with httpx.AsyncClient() as http:
            client = BotBillingClient(http, base_url=real_bot.base_url, secret=SECRET)
            result = await client.apply_snapshot(
                event_id=None, event_type=None, expected_version=0, snapshot=snapshot
            )

        assert result.outcome is ApplyOutcome.APPLIED

    async def test_the_plan_the_bot_serialises_is_one_the_web_client_can_read(
        self, real_bot
    ) -> None:
        snapshot = stripe_snapshot(FakeStripeState())
        async with httpx.AsyncClient() as http:
            client = BotBillingClient(http, base_url=real_bot.base_url, secret=SECRET)
            await client.apply_snapshot(
                event_id=None, event_type=None, expected_version=0, snapshot=snapshot
            )
            plans = await client.get_guild_plans(["1000", "2000"])

        assert plans["1000"].tier == "pro"
        assert plans["1000"].subscriptions[0].purchaser_user_id == "5000"
        assert plans["2000"].tier == "free"

    async def test_a_mismatched_secret_is_a_client_error_not_a_silent_free_plan(
        self, real_bot
    ) -> None:
        async with httpx.AsyncClient() as http:
            client = BotBillingClient(
                http, base_url=real_bot.base_url, secret="not-the-secret-" + "0" * 30
            )
            with pytest.raises(BotBillingError):
                await client.get_guild_plans(["1000"])


class StallingBotClient(BotBillingClient):
    """The production client, able to pause one chosen apply after its fetch -- the race window."""

    stall_event_id: str | None = None
    stalled = asyncio.Event
    release = asyncio.Event

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.stalled = asyncio.Event()
        self.release = asyncio.Event()

    async def apply_snapshot(self, **kwargs):
        if kwargs["event_id"] is not None and kwargs["event_id"] == self.stall_event_id:
            self.stall_event_id = None
            self.stalled.set()
            await self.release.wait()
        return await super().apply_snapshot(**kwargs)


@dataclass
class Stack:
    client: httpx.AsyncClient
    stripe_state: FakeStripeState
    bot: RealBot
    bot_client: StallingBotClient


@pytest_asyncio.fixture
async def stack(real_bot: RealBot) -> AsyncIterator[Stack]:
    discord_state = FakeDiscordState()
    discord_state.guilds = {"1000": FakeGuild(id="1000", name="Aura Test Server")}
    discord_state.bot_guild_ids = {"1000"}
    discord_state.users = {
        "5000": FakeUser(
            id="5000", username="moderator", guild_permissions={"1000": PERMISSION_MANAGE_GUILD}
        )
    }
    stripe_state = FakeStripeState()
    settings = WebSettings(
        _env_file=None,  # type: ignore[call-arg]
        discord_client_id=discord_state.client_id,
        discord_client_secret=discord_state.client_secret,
        discord_bot_token=discord_state.bot_token,
        discord_api_base="https://discord.test/api/v10",
        oauth_redirect_uri=f"{FRONTEND}/api/auth/callback",
        post_login_redirect_url=f"{FRONTEND}/",
        stripe_secret_key=stripe_state.secret_key,
        stripe_webhook_secret=stripe_state.webhook_secret,
        stripe_price_id=stripe_state.price_id,
        stripe_api_base="https://stripe.test",
        bot_internal_api_url=real_bot.base_url,
        bot_internal_api_secret=SECRET,
    )
    holder: dict[str, StallingBotClient] = {}
    async with (
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=create_fake_discord(discord_state))
        ) as discord_http,
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=create_fake_stripe(stripe_state))
        ) as stripe_http,
        httpx.AsyncClient() as bot_http,
    ):

        def bot_factory(_http, configured: WebSettings) -> StallingBotClient:
            holder["client"] = StallingBotClient(
                bot_http,
                base_url=configured.bot_internal_api_url,
                secret=configured.bot_internal_api_secret,
            )
            return holder["client"]

        app = create_app(
            settings,
            discord_client_factory=lambda _http, s: DiscordClient(
                discord_http,
                api_base=s.discord_api_base,
                client_id=s.discord_client_id,
                client_secret=s.discord_client_secret,
                bot_token=s.discord_bot_token,
            ),
            stripe_client_factory=lambda _http, s: StripeClient(
                stripe_http,
                api_base=s.stripe_api_base,
                secret_key=s.stripe_secret_key,
                price_id=s.stripe_price_id,
                checkout_success_url=s.checkout_success_url,
                checkout_cancel_url=s.checkout_cancel_url,
                portal_return_url=s.billing_portal_return_url,
            ),
            bot_billing_client_factory=bot_factory,
        )
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="https://testserver"
            ) as client:
                state = await client.get("/api/auth/login")
                login_state = httpx.URL(state.headers["location"]).params["state"]
                await client.get(
                    "/api/auth/callback",
                    params={"code": discord_state.issue_code("5000"), "state": login_state},
                )
                yield Stack(client, stripe_state, real_bot, holder["client"])


async def deliver(stack: Stack, event: dict) -> httpx.Response:
    body, signature = stack.stripe_state.signed(event)
    return await stack.client.post(
        "/api/stripe/webhook", content=body, headers={"Stripe-Signature": signature}
    )


class TestTheWholePath:
    async def test_checkout_webhook_database_gate_and_back_to_free(self, stack: Stack) -> None:
        assert not stack.bot.gate.allows_pro(1000)
        checkout = await stack.client.post(
            "/api/billing/checkout",
            content=b'{"guild_id": "1000"}',
            headers={"Content-Type": "application/json", "Origin": FRONTEND},
        )
        assert checkout.status_code == 200
        session_id = next(iter(stack.stripe_state.checkout_sessions))
        subscription = stack.stripe_state.complete_checkout(session_id, now=int(time.time()))

        paid = await deliver(stack, stack.stripe_state.checkout_completed_event(session_id))

        assert paid.json() == {"status": "applied"}
        assert stack.bot.gate.allows_pro(1000)
        (stored,) = await load_subscription_records(stack.bot.conn)
        assert (stored.guild_id, stored.purchaser_user_id) == (1000, 5000)
        dashboard = (await stack.client.get("/api/billing/guilds")).json()
        assert (
            dashboard[0]["plan"]["tier"] == "pro"
            and dashboard[0]["plan"]["is_billing_owner"] is True
        )

        subscription.status = "canceled"
        canceled = await deliver(
            stack,
            stack.stripe_state.subscription_event("customer.subscription.deleted", subscription.id),
        )

        assert canceled.json() == {"status": "applied"}
        assert not stack.bot.gate.allows_pro(1000)
        assert (await stack.client.get("/api/billing/guilds")).json()[0]["plan"][
            "standing"
        ] == "ended"

    async def test_a_forged_signature_changes_nothing_in_the_real_database(
        self, stack: Stack
    ) -> None:
        subscription = stack.stripe_state.add_subscription(
            guild_id="1000", purchaser_user_id="5000", now=int(time.time())
        )
        body = json.dumps(
            stack.stripe_state.subscription_event("customer.subscription.created", subscription.id)
        ).encode()

        for signature in (None, "", sign_webhook(body, "whsec_forged")):
            headers = {} if signature is None else {"Stripe-Signature": signature}
            response = await stack.client.post("/api/stripe/webhook", content=body, headers=headers)
            assert response.status_code == 400

        assert await load_subscription_records(stack.bot.conn) == []
        assert await count_processed_events(stack.bot.conn) == 0
        assert not stack.bot.gate.allows_pro(1000)

    async def test_a_duplicate_delivery_is_recorded_once_in_the_real_database(
        self, stack: Stack
    ) -> None:
        subscription = stack.stripe_state.add_subscription(
            guild_id="1000", purchaser_user_id="5000", now=int(time.time())
        )
        event = stack.stripe_state.subscription_event(
            "customer.subscription.created", subscription.id
        )

        first, second = await deliver(stack, event), await deliver(stack, event)

        assert (first.json(), second.json()) == ({"status": "applied"}, {"status": "duplicate"})
        assert await count_processed_events(stack.bot.conn) == 1
        (stored,) = await load_subscription_records(stack.bot.conn)
        assert stored.version == 1

    async def test_renewed_and_canceled_racing_end_canceled_in_the_real_database(
        self, stack: Stack
    ) -> None:
        subscription = stack.stripe_state.add_subscription(
            guild_id="1000", purchaser_user_id="5000", now=int(time.time())
        )
        renewed = stack.stripe_state.subscription_event(
            "customer.subscription.updated", subscription.id
        )
        canceled = stack.stripe_state.subscription_event(
            "customer.subscription.deleted", subscription.id
        )
        stack.bot_client.stall_event_id = renewed["id"]

        async def cancel_while_the_renewal_is_stalled() -> httpx.Response:
            await stack.bot_client.stalled.wait()
            subscription.status = "canceled"
            response = await deliver(stack, canceled)
            stack.bot_client.release.set()
            return response

        renewal_response, cancel_response = await asyncio.gather(
            deliver(stack, renewed), cancel_while_the_renewal_is_stalled()
        )

        assert cancel_response.json() == {"status": "applied"}
        assert renewal_response.json() == {"status": "applied"}
        (stored,) = await load_subscription_records(stack.bot.conn)
        assert stored.status.value == "canceled"
        assert stored.version == 2
        assert not stack.bot.gate.allows_pro(1000)

    async def test_a_bot_that_is_down_defers_the_event_and_a_redelivery_after_restart_applies_it(
        self, stack: Stack
    ) -> None:
        """The brief's unreachable status channel, end to end: nothing changes, nothing is lost."""
        subscription = stack.stripe_state.add_subscription(
            guild_id="1000", purchaser_user_id="5000", now=int(time.time())
        )
        event = stack.stripe_state.subscription_event(
            "customer.subscription.created", subscription.id
        )
        port = stack.bot.server.bound_port
        await stack.bot.server.stop()

        deferred = await deliver(stack, event)

        assert deferred.status_code == 503
        assert deferred.json() == {"error": "billing_unavailable"}
        assert await load_subscription_records(stack.bot.conn) == []
        assert await count_processed_events(stack.bot.conn) == 0

        stack.bot.server = await start_internal_api(
            stack.bot.conn, stack.bot.gate, secret=SECRET, host="127.0.0.1", port=port
        )
        redelivered = await deliver(stack, event)

        assert redelivered.json() == {"status": "applied"}
        assert stack.bot.gate.allows_pro(1000)
        assert await count_processed_events(stack.bot.conn) == 1

    async def test_a_failed_renewal_keeps_pro_on_payment_grace_in_the_real_gate(
        self, stack: Stack
    ) -> None:
        subscription = stack.stripe_state.add_subscription(
            guild_id="1000",
            purchaser_user_id="5000",
            now=int(time.time()),
            status="past_due",
            latest_invoice_status="open",
        )

        response = await deliver(
            stack, stack.stripe_state.invoice_event("invoice.payment_failed", subscription.id)
        )

        assert response.json() == {"status": "applied"}
        plan = stack.bot.gate.plan_for(1000)
        assert plan.is_pro and plan.standing.standing.value == "payment_grace"
