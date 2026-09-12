"""ASGI factories for the Phase 4b live verification run.

Two factories, each started by uvicorn in its own process by
scripts/verify_oauth_flow.py, so the OAuth2 flow runs over real TCP sockets
with real HTTP rather than through an in-process transport. Everything the
backend does -- form-encoding the token exchange, sending Basic auth, setting
cookies, parsing JSON -- happens exactly as it would against Discord.

The dataset is the same shape as the test suite's: one moderator who manages
one Aura guild and one guild Aura is absent from, and one plain member with no
management rights anywhere. That combination is what makes the guild filter's
two conditions observable in the transcript rather than asserted in prose.
"""
from __future__ import annotations

import os

from fake_discord import (
    PERMISSION_MANAGE_GUILD,
    PERMISSION_SEND_MESSAGES,
    FakeDiscordState,
    FakeGuild,
    FakeUser,
    create_fake_discord,
)

from aura_web.app import create_app
from aura_web.config import WebSettings

CLIENT_ID = "123456789012345678"
CLIENT_SECRET = "verification-client-secret-DO-NOT-REUSE"
BOT_TOKEN = "verification-bot-token-DO-NOT-REUSE"

MODERATOR_ID = "5000"
PLAIN_MEMBER_ID = "6000"


def build_state() -> FakeDiscordState:
    """The fixture dataset, identical in both processes that need to know it."""
    state = FakeDiscordState(
        client_id=CLIENT_ID, client_secret=CLIENT_SECRET, bot_token=BOT_TOKEN
    )
    state.guilds = {
        "1000": FakeGuild(id="1000", name="Aura Test Server", icon="a1b2c3d4"),
        "2000": FakeGuild(id="2000", name="Server Without Aura", icon=None),
        "3000": FakeGuild(id="3000", name="Server Where I Am A Member", icon=None),
    }
    # Aura is in 1000 and 3000, absent from 2000.
    state.bot_guild_ids = {"1000", "3000"}
    state.users = {
        MODERATOR_ID: FakeUser(
            id=MODERATOR_ID,
            username="moderator",
            global_name="The Moderator",
            avatar="deadbeef",
            guild_permissions={
                "1000": PERMISSION_MANAGE_GUILD,
                "2000": PERMISSION_MANAGE_GUILD,
                "3000": PERMISSION_SEND_MESSAGES,
            },
        ),
        PLAIN_MEMBER_ID: FakeUser(
            id=PLAIN_MEMBER_ID,
            username="plainmember",
            global_name=None,
            avatar=None,
            guild_permissions={
                "1000": PERMISSION_SEND_MESSAGES,
                "3000": PERMISSION_SEND_MESSAGES,
            },
        ),
    }
    return state


def build_fake():
    """uvicorn factory for the Discord stand-in."""
    return create_fake_discord(build_state())


def build_backend():
    """uvicorn factory for the real backend, pointed at the stand-in.

    Reads the fake's base URL and the frontend origin from the environment the
    parent script sets, so nothing about the application itself is special-
    cased for verification -- this is create_app with production settings.
    """
    return create_app(
        WebSettings(
            _env_file=None,
            discord_client_id=CLIENT_ID,
            discord_client_secret=CLIENT_SECRET,
            discord_bot_token=BOT_TOKEN,
            discord_api_base=os.environ["VERIFY_DISCORD_API_BASE"],
            oauth_redirect_uri=os.environ["VERIFY_REDIRECT_URI"],
            post_login_redirect_url=os.environ["VERIFY_POST_LOGIN_URL"],
            # Left at the production default (True) on purpose: the point of
            # this run is to see the real flags on the real wire.
            session_cookie_secure=True,
        )
    )
