"""A stand-in for Discord's OAuth2 and user endpoints, faithful to the documented schema.

Not a mock of this project's own code: a real ASGI application that speaks
Discord's wire protocol, so the service under test makes genuine HTTP
requests with genuine headers and parses genuine JSON. Everything a mock of
DiscordClient would have hidden -- the Basic-auth credential check, the
form encoding, the string-typed permission bitmask, the ``after`` cursor
pagination -- is exercised for real.

It lives outside the aura_web package because two different consumers import
it: the pytest suite in web/backend/tests, and scripts/verify_oauth_flow.py,
which runs it and the real backend as separate uvicorn processes on real
sockets so the raw HTTP can be captured with curl. Keeping one double for
both means the schema in the tests and the schema in the report's verification
run cannot drift apart.

Field shapes follow Discord's documentation as of API v10:
  * ``GET /users/@me`` -> a user object; ``id`` is a snowflake string.
  * ``GET /users/@me/guilds`` -> an array of partial guilds whose
    ``permissions`` is a DECIMAL STRING, not an integer (v8+).
  * ``POST /oauth2/token`` -> form-encoded in, JSON out, client credentials
    in HTTP Basic auth.
"""
from __future__ import annotations

import base64
import secrets
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import parse_qs

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

# Discord's real permission bits, repeated here rather than imported from
# aura_web.permissions: a double that shares constants with the code it tests
# cannot catch that code getting a constant wrong.
PERMISSION_MANAGE_GUILD = 32
PERMISSION_ADMINISTRATOR = 8
PERMISSION_SEND_MESSAGES = 2048


@dataclass
class FakeUser:
    """One Discord account this double knows about."""

    id: str
    username: str
    global_name: str | None = None
    avatar: str | None = None
    # guild id -> permission bitmask the user holds there
    guild_permissions: dict[str, int] = field(default_factory=dict)


@dataclass
class FakeGuild:
    """One Discord server this double knows about."""

    id: str
    name: str
    icon: str | None = None


@dataclass
class FakeDiscordState:
    """Everything the double will answer with, plus the failure switches.

    The failure switches are what make the adversarial tests possible without
    monkeypatching: flipping ``fail_bot_guilds`` makes Discord's membership
    route return 503 the way a real outage would, and the service under test
    has no idea it was arranged.
    """

    client_id: str = "123456789012345678"
    client_secret: str = "test-client-secret"
    bot_token: str = "test-bot-token"

    users: dict[str, FakeUser] = field(default_factory=dict)
    guilds: dict[str, FakeGuild] = field(default_factory=dict)
    # Which guilds the BOT is in. Deliberately independent of which guilds the
    # users are in, because the whole point of the filter under test is that
    # those two sets differ.
    bot_guild_ids: set[str] = field(default_factory=set)

    # code -> user id, populated by issue_code()
    authorization_codes: dict[str, str] = field(default_factory=dict)
    # access token -> user id
    access_tokens: dict[str, str] = field(default_factory=dict)
    # refresh token -> user id
    refresh_tokens: dict[str, str] = field(default_factory=dict)
    revoked_tokens: set[str] = field(default_factory=set)

    # The scopes the next exchange will report as granted. Lets a test
    # simulate a user who hand-edited the authorize URL.
    granted_scopes: str = "identify guilds"
    token_lifetime_seconds: int = 604800

    fail_token_exchange_status: int | None = None
    fail_current_user_status: int | None = None
    fail_user_guilds_status: int | None = None
    fail_bot_guilds_status: int | None = None
    # Forces /users/@me/guilds to page, so the cursor logic is exercised.
    page_size_override: int | None = None

    request_log: list[str] = field(default_factory=list)

    def issue_code(self, user_id: str) -> str:
        """Mint an authorization code for a user, as the consent screen would."""
        code = secrets.token_urlsafe(16)
        self.authorization_codes[code] = user_id
        return code


async def _read_form(request: Request) -> dict[str, str]:
    """Decode an application/x-www-form-urlencoded body without Starlette's parser.

    Starlette's request.form() requires python-multipart even for urlencoded
    bodies. Parsing the body here keeps this double -- and therefore the test
    suite and the verification script -- free of a dependency that the service
    under test does not have and does not need.
    """
    raw = (await request.body()).decode("utf-8")
    return {key: values[0] for key, values in parse_qs(raw, keep_blank_values=True).items()}


def _basic_auth_ok(request: Request, state: FakeDiscordState) -> bool:
    header = request.headers.get("Authorization", "")
    if not header.startswith("Basic "):
        return False
    try:
        decoded = base64.b64decode(header[len("Basic ") :]).decode("utf-8")
    except (ValueError, UnicodeDecodeError):
        return False
    client_id, _, client_secret = decoded.partition(":")
    # Discord url-encodes the Basic credentials; httpx does not for plain
    # ASCII, and the test credentials are plain ASCII, so a direct compare is
    # correct here.
    return client_id == state.client_id and client_secret == state.client_secret


def _bearer_user(request: Request, state: FakeDiscordState) -> str | None:
    header = request.headers.get("Authorization", "")
    if not header.startswith("Bearer "):
        return None
    token = header[len("Bearer ") :]
    if token in state.revoked_tokens:
        return None
    return state.access_tokens.get(token)


def _is_bot(request: Request, state: FakeDiscordState) -> bool:
    header = request.headers.get("Authorization", "")
    return header == f"Bot {state.bot_token}"


def create_fake_discord(state: FakeDiscordState) -> FastAPI:
    """Build the ASGI application. All routes live under /api/v10, as Discord's do."""
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    @app.post("/api/v10/oauth2/token")
    async def token(request: Request) -> JSONResponse:
        state.request_log.append("POST /oauth2/token")
        if state.fail_token_exchange_status is not None:
            return JSONResponse(
                {"error": "server_error"}, status_code=state.fail_token_exchange_status
            )
        if not _basic_auth_ok(request, state):
            return JSONResponse({"error": "invalid_client"}, status_code=401)

        form = await _read_form(request)
        grant_type = form.get("grant_type")

        if grant_type == "authorization_code":
            code = form.get("code", "")
            user_id = state.authorization_codes.pop(code, None)
            if user_id is None:
                # Discord answers a reused or forged code with 400
                # invalid_grant, not 401 -- reproduced exactly, because the
                # service under test has a branch that depends on it.
                return JSONResponse({"error": "invalid_grant"}, status_code=400)
        elif grant_type == "refresh_token":
            refresh = form.get("refresh_token", "")
            user_id = state.refresh_tokens.get(refresh)
            if user_id is None:
                return JSONResponse({"error": "invalid_grant"}, status_code=400)
        else:
            return JSONResponse({"error": "unsupported_grant_type"}, status_code=400)

        access_token = "access-" + secrets.token_urlsafe(16)
        refresh_token = "refresh-" + secrets.token_urlsafe(16)
        state.access_tokens[access_token] = user_id
        state.refresh_tokens[refresh_token] = user_id
        return JSONResponse(
            {
                "access_token": access_token,
                "token_type": "Bearer",
                "expires_in": state.token_lifetime_seconds,
                "refresh_token": refresh_token,
                "scope": state.granted_scopes,
            }
        )

    @app.post("/api/v10/oauth2/token/revoke")
    async def revoke(request: Request) -> JSONResponse:
        state.request_log.append("POST /oauth2/token/revoke")
        if not _basic_auth_ok(request, state):
            return JSONResponse({"error": "invalid_client"}, status_code=401)
        form = await _read_form(request)
        state.revoked_tokens.add(form.get("token", ""))
        return JSONResponse({})

    @app.get("/api/v10/users/@me")
    async def current_user(request: Request) -> JSONResponse:
        state.request_log.append("GET /users/@me")
        if state.fail_current_user_status is not None:
            return JSONResponse({"message": "nope"}, status_code=state.fail_current_user_status)
        user_id = _bearer_user(request, state)
        if user_id is None:
            return JSONResponse({"message": "401: Unauthorized"}, status_code=401)
        user = state.users[user_id]
        return JSONResponse(
            {
                "id": user.id,
                "username": user.username,
                "discriminator": "0",
                "global_name": user.global_name,
                "avatar": user.avatar,
                "bot": False,
                "mfa_enabled": True,
                "locale": "en-US",
                "flags": 0,
                "premium_type": 0,
                "public_flags": 0,
            }
        )

    @app.post("/__fake__/issue-code")
    async def issue_code(request: Request) -> JSONResponse:
        """Stand in for the consent screen, which has no API to drive.

        Real Discord hands the browser an authorization code after a human
        clicks Authorize on discord.com -- a page this double cannot host and
        the verification script cannot click. This endpoint is the seam that
        replaces that click, so the script can run the rest of the flow
        (state check, code exchange, session creation) against real sockets.

        Under /__fake__/ rather than /api/, so it is unmistakably not part of
        the Discord surface being imitated. The file it lives in never ships:
        web/backend/.dockerignore excludes it from the image.
        """
        payload = await request.json()
        return JSONResponse({"code": state.issue_code(str(payload["user_id"]))})

    @app.get("/__fake__/issued-tokens")
    async def issued_tokens() -> JSONResponse:
        """Report every token this double has handed out during the run.

        The leak scan in scripts/verify_oauth_flow.py needs the ACTUAL token
        values to search for. Reading them from Discord's side rather than
        guessing at their shape is what stops that scan from passing
        vacuously against strings no response could have contained.
        """
        return JSONResponse(
            {
                "access_tokens": sorted(state.access_tokens),
                "refresh_tokens": sorted(state.refresh_tokens),
            }
        )

    @app.get("/api/v10/users/@me/guilds")
    async def user_guilds(request: Request) -> JSONResponse:
        limit = int(request.query_params.get("limit", "200"))
        after = request.query_params.get("after")

        if _is_bot(request, state):
            state.request_log.append("GET /users/@me/guilds (bot)")
            if state.fail_bot_guilds_status is not None:
                return JSONResponse(
                    {"message": "nope"}, status_code=state.fail_bot_guilds_status
                )
            entries: list[dict[str, Any]] = [
                {
                    "id": guild_id,
                    "name": state.guilds[guild_id].name if guild_id in state.guilds else guild_id,
                    "icon": state.guilds[guild_id].icon if guild_id in state.guilds else None,
                    # A bot's own entry reports the bot's permissions; the
                    # service must ignore this field on this route, and a
                    # deliberately misleading value here proves it does.
                    "permissions": str(PERMISSION_SEND_MESSAGES),
                    "features": [],
                    "owner": False,
                }
                for guild_id in sorted(state.bot_guild_ids, key=int)
            ]
        else:
            state.request_log.append("GET /users/@me/guilds (user)")
            if state.fail_user_guilds_status is not None:
                return JSONResponse(
                    {"message": "nope"}, status_code=state.fail_user_guilds_status
                )
            user_id = _bearer_user(request, state)
            if user_id is None:
                return JSONResponse({"message": "401: Unauthorized"}, status_code=401)
            user = state.users[user_id]
            entries = [
                {
                    "id": guild_id,
                    "name": state.guilds[guild_id].name if guild_id in state.guilds else guild_id,
                    "icon": state.guilds[guild_id].icon if guild_id in state.guilds else None,
                    "permissions": str(permissions),
                    "features": [],
                    "owner": False,
                }
                for guild_id, permissions in sorted(
                    user.guild_permissions.items(), key=lambda item: int(item[0])
                )
            ]

        if after is not None:
            entries = [entry for entry in entries if int(entry["id"]) > int(after)]
        effective_limit = state.page_size_override or limit
        return JSONResponse(entries[:effective_limit])

    return app
