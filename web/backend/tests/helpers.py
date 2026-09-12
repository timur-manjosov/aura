"""Shared helpers for the web backend's tests.

A module of its own rather than functions on conftest: importing conftest by
name works only by accident of pytest's sys.path handling, and breaks the
moment the suite is run from a different rootdir.
"""
from __future__ import annotations

import httpx
from fake_discord import FakeDiscordState

from aura_web.app import create_app
from aura_web.config import WebSettings
from aura_web.discord_api import DiscordClient

FAKE_DISCORD_BASE = "https://discord.test/api/v10"
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


def build_app(web_settings: WebSettings, discord_http: httpx.AsyncClient):
    """Build the production application against the fake Discord transport."""
    return create_app(
        web_settings, discord_client_factory=build_discord_client_factory(discord_http)
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
    raise AssertionError(
        f"no Set-Cookie for {name!r} in {response.headers.get_list('set-cookie')}"
    )
