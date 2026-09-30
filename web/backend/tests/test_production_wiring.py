"""create_app with NO client factories: the wiring production runs, end to end.

Every other test hands create_app its own client factories, so the default
branch -- the one that builds DiscordClient, StripeClient and BotBillingClient
from the settings in production -- was exercised by nothing: a setting dropped
there (the portal configuration, a credential) would pass the whole suite. Here
the application builds its real clients itself, and only the network under
them is replaced: each request is routed by host to the matching stand-in.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from unittest.mock import patch

import httpx
import pytest_asyncio

from aura_web.app import create_app
from aura_web.config import WebSettings
from fake_bot_billing import FakeBotBillingState, create_fake_bot_billing
from fake_discord import FakeDiscordState, create_fake_discord
from fake_stripe import FakeStripeState, create_fake_stripe
from helpers import (
    FAKE_BOT_BASE,
    FAKE_DISCORD_BASE,
    FAKE_STRIPE_BASE,
    FRONTEND_BASE,
    complete_login,
)

PORTAL_CONFIGURATION = "bpc_productionWiringTest"


class HostRoutingTransport(httpx.AsyncBaseTransport):
    """Sends each request to the stand-in that owns its host."""

    def __init__(self, routes: dict[str, httpx.AsyncBaseTransport]) -> None:
        self._routes = routes

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        return await self._routes[request.url.host].handle_async_request(request)


@pytest_asyncio.fixture
async def browser(
    web_settings: WebSettings,
    discord_state: FakeDiscordState,
    stripe_state: FakeStripeState,
    bot_billing_state: FakeBotBillingState,
) -> AsyncIterator[httpx.AsyncClient]:
    routes: dict[str, httpx.AsyncBaseTransport] = {
        httpx.URL(FAKE_DISCORD_BASE).host: httpx.ASGITransport(
            app=create_fake_discord(discord_state)
        ),
        httpx.URL(FAKE_STRIPE_BASE).host: httpx.ASGITransport(app=create_fake_stripe(stripe_state)),
        httpx.URL(FAKE_BOT_BASE).host: httpx.ASGITransport(
            app=create_fake_bot_billing(bot_billing_state)
        ),
    }
    real_client = httpx.AsyncClient

    def routed_client(*args: object, **kwargs: object) -> httpx.AsyncClient:
        return real_client(*args, transport=HostRoutingTransport(routes), **kwargs)  # type: ignore[arg-type]

    settings = web_settings.model_copy(
        update={"stripe_portal_configuration_id": PORTAL_CONFIGURATION}
    )
    app = create_app(settings)
    with patch("aura_web.app.httpx.AsyncClient", routed_client):
        async with app.router.lifespan_context(app):
            async with real_client(
                transport=httpx.ASGITransport(app=app), base_url="https://testserver"
            ) as client:
                yield client


async def post_json(client: httpx.AsyncClient, path: str, body: bytes) -> httpx.Response:
    return await client.post(
        path,
        content=body,
        headers={"Content-Type": "application/json", "Origin": FRONTEND_BASE},
    )


class TestTheDefaultClientsAreBuiltFromTheSettings:
    async def test_login_checkout_and_portal_reach_the_services_as_configured(
        self,
        browser: httpx.AsyncClient,
        discord_state: FakeDiscordState,
        stripe_state: FakeStripeState,
        bot_billing_state: FakeBotBillingState,
    ) -> None:
        # Login: the real DiscordClient, with the configured client secret and
        # bot token (the stand-in refuses anything else).
        login = await complete_login(browser, discord_state, "5000")
        assert login.status_code == 303

        # Checkout: the real StripeClient, with the configured key and price,
        # cards only; the plan came from the real BotBillingClient.
        checkout = await post_json(browser, "/api/billing/checkout", b'{"guild_id": "1000"}')
        assert checkout.status_code == 200
        (form,) = stripe_state.received_forms
        assert form["line_items[0][price]"] == stripe_state.price_id
        assert form["payment_method_types[0]"] == "card"
        assert stripe_state.received_headers[-1]["authorization"] == (
            f"Bearer {stripe_state.secret_key}"
        )
        assert "plans" in bot_billing_state.request_log

        # Portal: opened with the configured portal configuration.
        subscription = stripe_state.add_subscription(
            guild_id="1000", purchaser_user_id="5000", now=0
        )
        bot_billing_state.plans["1000"] = {
            "tier": "pro",
            "basis": "subscription",
            "standing": "active",
            "access_until": 1_800_000_000,
            "paid_through": 1_799_740_800,
            "in_force_subscription_count": 1,
            "subscriptions": [
                {
                    "subscription_id": subscription.id,
                    "customer_id": subscription.customer,
                    "purchaser_user_id": "5000",
                    "status": "active",
                    "grants_access": True,
                }
            ],
        }
        portal = await post_json(browser, "/api/billing/portal", b'{"guild_id": "1000"}')

        assert portal.status_code == 200
        assert stripe_state.portal_sessions[-1]["configuration"] == PORTAL_CONFIGURATION
