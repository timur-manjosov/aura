"""No Discord credential ever reaches the browser. Checked on the wire, not in the code.

The sub-phase's brief asks for this specifically at the network level. So
these tests do not inspect response models or assert that some function was
not called: they complete a real login, collect every secret that exists in
the system at that moment, then sweep every endpoint and search the RAW
BYTES of each response -- status line, every header including Set-Cookie, and
body -- for any of them.

That framing is what makes the test durable. A future handler that adds
``"access_token": ...`` to a response, or a debug header, or an error body
echoing an upstream payload, fails here without anyone having remembered to
extend a list of forbidden field names.
"""
from __future__ import annotations

import httpx
import pytest
from fake_discord import FakeDiscordState
from helpers import complete_login, start_login

# Endpoints reachable by a browser, with the method each accepts. Logout is
# last so the session survives for the reads above it.
BROWSER_REACHABLE = [
    ("GET", "/api/health"),
    ("GET", "/api/me"),
    ("GET", "/api/guilds"),
    ("GET", "/api/auth/login"),
    ("POST", "/api/auth/logout"),
]


def secrets_in_play(discord_state: FakeDiscordState) -> dict[str, str]:
    """Every value that must never appear in a response, labelled for the failure message."""
    secrets: dict[str, str] = {
        "client_secret": discord_state.client_secret,
        "bot_token": discord_state.bot_token,
    }
    for index, access_token in enumerate(discord_state.access_tokens):
        secrets[f"access_token[{index}]"] = access_token
    for index, refresh_token in enumerate(discord_state.refresh_tokens):
        secrets[f"refresh_token[{index}]"] = refresh_token
    return secrets


def raw_response_bytes(response: httpx.Response) -> bytes:
    """Reassemble what actually went over the wire, headers included.

    Set-Cookie is part of this on purpose: it is the one header that carries a
    secret by design, and the whole question is whether the secret it carries
    is our opaque session identifier or Discord's token.
    """
    head = f"HTTP/1.1 {response.status_code}\r\n".encode()
    for name, value in response.headers.multi_items():
        head += f"{name}: {value}\r\n".encode()
    return head + b"\r\n" + response.content


class TestNoTokenReachesTheBrowser:
    async def test_no_endpoint_emits_a_discord_credential(
        self, app_client: httpx.AsyncClient, discord_state: FakeDiscordState
    ) -> None:
        callback_response = await complete_login(app_client, discord_state, "5000")
        assert callback_response.status_code == 303

        forbidden = secrets_in_play(discord_state)
        assert forbidden, "the login must have produced tokens for this test to mean anything"

        for method, path in BROWSER_REACHABLE:
            response = await app_client.request(method, path)
            wire = raw_response_bytes(response)
            for label, secret in forbidden.items():
                assert secret.encode() not in wire, (
                    f"{label} leaked in the {method} {path} response: {wire!r}"
                )

    async def test_the_callback_redirect_itself_carries_no_token(
        self, app_client: httpx.AsyncClient, discord_state: FakeDiscordState
    ) -> None:
        """The riskiest single response: it is the one that has just handled tokens."""
        response = await complete_login(app_client, discord_state, "5000")

        wire = raw_response_bytes(response)
        for label, secret in secrets_in_play(discord_state).items():
            assert secret.encode() not in wire, f"{label} leaked in the callback response"

    async def test_the_redirect_location_has_no_query_string_at_all(
        self, app_client: httpx.AsyncClient, discord_state: FakeDiscordState
    ) -> None:
        """Anything in the Location query lands in the browser's history and the referrer.

        Asserting the exact configured URL, rather than "no token in it", keeps
        a later convenience parameter from quietly becoming a leak channel.
        """
        response = await complete_login(app_client, discord_state, "5000")

        assert response.headers["location"] == "https://frontend.test/"

    async def test_the_session_cookie_value_is_not_derived_from_any_token(
        self, app_client: httpx.AsyncClient, discord_state: FakeDiscordState
    ) -> None:
        """Not equal, and not a prefix or suffix -- an encoding is still a leak."""
        await complete_login(app_client, discord_state, "5000")
        cookie = app_client.cookies["aura_session"]

        for secret in secrets_in_play(discord_state).values():
            assert secret not in cookie
            assert cookie not in secret

    async def test_an_error_response_does_not_echo_upstream_detail(
        self, app_client: httpx.AsyncClient, discord_state: FakeDiscordState
    ) -> None:
        """A 503 must be a code, not a relayed Discord body or status.

        Upstream error text is where credentials and internal hostnames leak
        in practice, and it is exactly what a helpful-looking error handler
        passes through.
        """
        discord_state.fail_token_exchange_status = 500
        state = await start_login(app_client)
        code = discord_state.issue_code("5000")

        response = await app_client.get(
            "/api/auth/callback", params={"code": code, "state": state}
        )

        assert response.json() == {"error": "discord_unavailable"}
        assert b"500" not in response.content
        assert b"discord" not in response.content.lower().replace(b"discord_unavailable", b"")

    async def test_the_authorize_redirect_carries_no_client_secret(
        self, app_client: httpx.AsyncClient, discord_state: FakeDiscordState
    ) -> None:
        """The client ID is public and belongs in the URL; the secret never is."""
        response = await app_client.get("/api/auth/login")

        assert discord_state.client_id in response.headers["location"]
        assert discord_state.client_secret not in response.headers["location"]
        assert discord_state.bot_token not in response.headers["location"]

    @pytest.mark.parametrize(
        "path", ["/api/me", "/api/guilds", "/api/health", "/api/auth/login"]
    )
    async def test_no_response_carries_a_debug_header_naming_a_secret(
        self, app_client: httpx.AsyncClient, discord_state: FakeDiscordState, path: str
    ) -> None:
        await complete_login(app_client, discord_state, "5000")

        response = await app_client.get(path)

        header_blob = "\n".join(f"{name}: {value}" for name, value in response.headers.items())
        for secret in secrets_in_play(discord_state).values():
            assert secret not in header_blob
