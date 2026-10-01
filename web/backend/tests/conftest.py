"""Fixtures that wire the real backend to a fake Discord over real HTTP.

Nothing in aura_web is mocked. The application under test is the one
create_app builds for production, the requests against it are real HTTP
carried by httpx, and the Discord it talks to is a separate ASGI app speaking
the documented wire protocol (see web/backend/fake_discord.py). What is
substituted is Discord itself -- exactly the one thing a test cannot call.

The base URL is https, not http, and that is load-bearing rather than
cosmetic: the session cookie is issued with Secure, and Python's cookie jar
correctly refuses to send a Secure cookie back over http. A test suite on
http would therefore pass only if the Secure flag were missing -- it would
quietly reward the bug it is supposed to catch.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import httpx
import pytest
import pytest_asyncio

from aura_web.config import WebSettings
from fake_bot_billing import FakeBotBillingState, create_fake_bot_billing
from fake_discord import (
    PERMISSION_MANAGE_GUILD,
    FakeDiscordState,
    FakeGuild,
    FakeUser,
    create_fake_discord,
)
from fake_stripe import FakeStripeState, create_fake_stripe
from helpers import FAKE_BOT_BASE, FAKE_DISCORD_BASE, FAKE_STRIPE_BASE, FRONTEND_BASE, build_app


@pytest.fixture(autouse=True)
def hermetic_web_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the developer's web/.env and shell out of every test in this tree.

    The same rule as tests/conftest.hermetic_settings_environment: every
    AURA_WEB_* variable WebSettings reads is removed from the environment (a
    real test-mode Stripe key exported from running the backend by hand must
    not reach a test that did not set it), and WebSettings' own ``web/.env``,
    which it resolves against the working directory, is not read. Derived from
    WebSettings, so a new setting is covered.
    """
    prefix = WebSettings.model_config.get("env_prefix", "")
    for name in WebSettings.model_fields:
        monkeypatch.delenv(f"{prefix}{name}".upper(), raising=False)
    monkeypatch.setitem(WebSettings.model_config, "env_file", None)


@pytest.fixture
def discord_state() -> FakeDiscordState:
    """A fake Discord pre-loaded with one moderator, one plain member, three guilds.

    The guild set is chosen so every arm of the two-condition filter is
    represented by real data rather than by a test that constructs only the
    case it is checking:

      * 1000 -- user can manage, Aura is in it            -> must appear
      * 2000 -- user can manage, Aura is NOT in it        -> must not appear
      * 3000 -- user cannot manage, Aura IS in it         -> must not appear
    """
    state = FakeDiscordState()
    state.guilds = {
        "1000": FakeGuild(id="1000", name="Aura Test Server", icon="a1b2c3"),
        "2000": FakeGuild(id="2000", name="Server Without Aura", icon=None),
        "3000": FakeGuild(id="3000", name="Server Where I Am A Member", icon=None),
    }
    state.bot_guild_ids = {"1000", "3000"}
    state.users = {
        "5000": FakeUser(
            id="5000",
            username="moderator",
            global_name="The Moderator",
            avatar="deadbeef",
            guild_permissions={
                "1000": PERMISSION_MANAGE_GUILD,
                "2000": PERMISSION_MANAGE_GUILD,
                "3000": 2048,
            },
        ),
        "6000": FakeUser(
            id="6000",
            username="plainmember",
            global_name=None,
            avatar=None,
            # Present in a guild Aura runs on, with no management rights.
            guild_permissions={"1000": 2048, "3000": 2048},
        ),
    }
    return state


@pytest.fixture
def stripe_state() -> FakeStripeState:
    """A Stripe stand-in in test mode with no subscriptions yet."""
    return FakeStripeState()


@pytest.fixture
def bot_billing_state() -> FakeBotBillingState:
    """The bot's billing API stand-in, every guild on Free until a test says otherwise."""
    return FakeBotBillingState()


@pytest.fixture
def web_settings(
    discord_state: FakeDiscordState,
    stripe_state: FakeStripeState,
    bot_billing_state: FakeBotBillingState,
) -> WebSettings:
    """Production settings, pointed at the stand-ins instead of the real services."""
    return WebSettings(
        _env_file=None,  # type: ignore[call-arg]
        discord_client_id=discord_state.client_id,
        discord_client_secret=discord_state.client_secret,
        discord_bot_token=discord_state.bot_token,
        discord_api_base=FAKE_DISCORD_BASE,
        oauth_redirect_uri=f"{FRONTEND_BASE}/api/auth/callback",
        post_login_redirect_url=f"{FRONTEND_BASE}/",
        # One second, so a test can let the cache expire without sleeping for
        # the production minute.
        bot_guilds_cache_ttl_seconds=1.0,
        bot_guilds_stale_tolerance_seconds=0.0,
        stripe_secret_key=stripe_state.secret_key,
        stripe_webhook_secret=stripe_state.webhook_secret,
        stripe_price_id=stripe_state.price_id,
        stripe_api_base=FAKE_STRIPE_BASE,
        checkout_success_url=f"{FRONTEND_BASE}/?checkout=success",
        checkout_cancel_url=f"{FRONTEND_BASE}/?checkout=cancelled",
        billing_portal_return_url=f"{FRONTEND_BASE}/",
        bot_internal_api_url=FAKE_BOT_BASE,
        bot_internal_api_secret=bot_billing_state.secret,
    )


@pytest_asyncio.fixture
async def app_client(
    web_settings: WebSettings,
    discord_state: FakeDiscordState,
    stripe_state: FakeStripeState,
    bot_billing_state: FakeBotBillingState,
) -> AsyncIterator[httpx.AsyncClient]:
    """The real application, with a real cookie jar, talking to the three stand-ins."""
    async with (
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=create_fake_discord(discord_state)),
            base_url="https://discord.test",
        ) as discord_http,
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=create_fake_stripe(stripe_state)),
            base_url=FAKE_STRIPE_BASE,
        ) as stripe_http,
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=create_fake_bot_billing(bot_billing_state)),
            base_url=FAKE_BOT_BASE,
        ) as bot_http,
    ):
        app = build_app(web_settings, discord_http, stripe_http, bot_http)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="https://testserver"
            ) as client:
                yield client
