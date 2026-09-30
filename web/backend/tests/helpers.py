"""Shared helpers for the web backend's tests.

A module of its own rather than functions on conftest: importing conftest by
name works only by accident of pytest's sys.path handling, and breaks the
moment the suite is run from a different rootdir.
"""

from __future__ import annotations

import httpx

from aura_web.app import create_app
from aura_web.bot_billing import BotBillingClient
from aura_web.config import WebSettings
from aura_web.discord_api import DiscordClient
from aura_web.stripe_api import StripeClient
from fake_discord import FakeDiscordState

FAKE_DISCORD_BASE = "https://discord.test/api/v10"
FAKE_STRIPE_BASE = "https://stripe.test"
FAKE_BOT_BASE = "https://aura-bot.test"
FRONTEND_BASE = "https://frontend.test"


def build_discord_client_factory(discord_http: httpx.AsyncClient):
    """Return a factory that binds the real DiscordClient to the fake transport.

    Everything about the client is what production builds; only the socket
    underneath it is redirected.
    """

    def factory(_: httpx.AsyncClient, settings: WebSettings) -> DiscordClient:
        return DiscordClient(
            discord_http,
            api_base=settings.discord_api_base,
            client_id=settings.discord_client_id,
            client_secret=settings.discord_client_secret,
            bot_token=settings.discord_bot_token,
        )

    return factory


def build_app(
    web_settings: WebSettings,
    discord_http: httpx.AsyncClient,
    stripe_http: httpx.AsyncClient | None = None,
    bot_http: httpx.AsyncClient | None = None,
):
    """Build the production application against the fake transports that are given.

    Stripe and the bot's billing API are optional so the Phase 4b tests that
    never touch billing keep their exact shape; a test that does touch billing
    passes all three, and every client is still the production class.
    """

    def stripe_factory(_: httpx.AsyncClient, settings: WebSettings) -> StripeClient:
        assert stripe_http is not None
        return StripeClient(
            stripe_http,
            api_base=settings.stripe_api_base,
            secret_key=settings.stripe_secret_key,
            price_id=settings.stripe_price_id,
            checkout_success_url=settings.checkout_success_url,
            checkout_cancel_url=settings.checkout_cancel_url,
            portal_return_url=settings.billing_portal_return_url,
            portal_configuration_id=settings.stripe_portal_configuration_id,
        )

    def bot_factory(_: httpx.AsyncClient, settings: WebSettings) -> BotBillingClient:
        assert bot_http is not None
        return BotBillingClient(
            bot_http,
            base_url=settings.bot_internal_api_url,
            secret=settings.bot_internal_api_secret,
        )

    return create_app(
        web_settings,
        discord_client_factory=build_discord_client_factory(discord_http),
        stripe_client_factory=stripe_factory if stripe_http is not None else None,
        bot_billing_client_factory=bot_factory if bot_http is not None else None,
    )


async def start_login(client: httpx.AsyncClient) -> str:
    """Drive GET /api/auth/login and return the ``state`` Discord would receive."""
    response = await client.get("/api/auth/login")
    assert response.status_code == 307, response.text
    return httpx.URL(response.headers["location"]).params["state"]


async def complete_login(
    client: httpx.AsyncClient, discord_state: FakeDiscordState, user_id: str
) -> httpx.Response:
    """Run a full, honest login for `user_id` and return the callback's response.

    The client's cookie jar carries the state cookie from the login leg to the
    callback leg by itself, exactly as a browser would -- so a regression that
    broke the cookie binding would surface here as a failing login, not as a
    quietly skipped check.
    """
    state = await start_login(client)
    code = discord_state.issue_code(user_id)
    return await client.get("/api/auth/callback", params={"code": code, "state": state})


def set_cookie_header(response: httpx.Response, name: str) -> str:
    """Return the raw Set-Cookie header line that sets `name`, or fail the test."""
    for header_value in response.headers.get_list("set-cookie"):
        if header_value.split("=", 1)[0].strip() == name:
            return header_value
    raise AssertionError(f"no Set-Cookie for {name!r} in {response.headers.get_list('set-cookie')}")
