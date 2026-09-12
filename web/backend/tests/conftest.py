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
from fake_discord import (
    PERMISSION_MANAGE_GUILD,
    FakeDiscordState,
    FakeGuild,
    FakeUser,
    create_fake_discord,
)
from helpers import FAKE_DISCORD_BASE, FRONTEND_BASE, build_app

from aura_web.config import WebSettings


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
def web_settings(discord_state: FakeDiscordState) -> WebSettings:
    """Production settings, pointed at the fake Discord instead of the real one."""
    return WebSettings(
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
    )


@pytest_asyncio.fixture
async def app_client(
    web_settings: WebSettings, discord_state: FakeDiscordState
) -> AsyncIterator[httpx.AsyncClient]:
    """The real application, with a real cookie jar, talking to the fake Discord."""
    fake_discord = create_fake_discord(discord_state)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=fake_discord), base_url="https://discord.test"
    ) as discord_http:
        app = build_app(web_settings, discord_http)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="https://testserver"
            ) as client:
                yield client
