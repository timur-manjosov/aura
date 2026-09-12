"""The OAuth2 login flow end to end, and every way it must refuse to complete.

These drive the real application over real HTTP with a real cookie jar. Where
a test asserts on a header it asserts on the header the client actually
received, not on the argument some function was called with -- the brief for
this sub-phase asks for the flags to be checked in the HTTP response rather
than in the source, and that distinction is the point of the whole file.
"""
from __future__ import annotations

import httpx
import pytest
from fake_discord import FakeDiscordState
from helpers import (
    FRONTEND_BASE,
    build_app,
    complete_login,
    set_cookie_header,
    start_login,
)

from aura_web.discord_api import REQUIRED_SCOPES


class TestLoginRedirect:
    async def test_redirects_to_discord_with_the_minimum_scopes(
        self, app_client: httpx.AsyncClient
    ) -> None:
        response = await app_client.get("/api/auth/login")

        assert response.status_code == 307
        location = httpx.URL(response.headers["location"])
        assert location.host == "discord.com"
        assert location.path == "/oauth2/authorize"
        assert location.params["response_type"] == "code"
        assert set(location.params["scope"].split()) == set(REQUIRED_SCOPES)

    async def test_requests_no_scope_beyond_identify_and_guilds(
        self, app_client: httpx.AsyncClient
    ) -> None:
        """The brief's "only the minimum scopes" requirement, asserted as an exact set.

        A superset assertion would pass if someone added `email` or `bot`
        later, which is the failure this is here to catch.
        """
        response = await app_client.get("/api/auth/login")

        scopes = set(httpx.URL(response.headers["location"]).params["scope"].split())
        assert scopes == {"identify", "guilds"}

    async def test_sets_the_state_cookie_with_every_protective_flag(
        self, app_client: httpx.AsyncClient
    ) -> None:
        response = await app_client.get("/api/auth/login")

        raw = set_cookie_header(response, "aura_oauth_state").lower()
        assert "httponly" in raw
        assert "secure" in raw
        assert "samesite=lax" in raw
        assert "path=/" in raw

    async def test_the_state_in_the_url_matches_the_state_in_the_cookie(
        self, app_client: httpx.AsyncClient
    ) -> None:
        response = await app_client.get("/api/auth/login")

        state_in_url = httpx.URL(response.headers["location"]).params["state"]
        assert app_client.cookies["aura_oauth_state"] == state_in_url

    async def test_two_logins_mint_two_different_states(
        self, app_client: httpx.AsyncClient
    ) -> None:
        first = await start_login(app_client)
        second = await start_login(app_client)

        assert first != second


class TestSuccessfulCallback:
    async def test_opens_a_session_and_redirects_to_the_frontend(
        self, app_client: httpx.AsyncClient, discord_state: FakeDiscordState
    ) -> None:
        response = await complete_login(app_client, discord_state, "5000")

        assert response.status_code == 303
        assert response.headers["location"] == f"{FRONTEND_BASE}/"
        assert app_client.cookies.get("aura_session")

    async def test_the_session_cookie_carries_every_protective_flag(
        self, app_client: httpx.AsyncClient, discord_state: FakeDiscordState
    ) -> None:
        """The brief's explicit check: httpOnly, Secure and SameSite in the real response."""
        response = await complete_login(app_client, discord_state, "5000")

        raw = set_cookie_header(response, "aura_session").lower()
        assert "httponly" in raw
        assert "secure" in raw
        assert "samesite=lax" in raw
        assert "path=/" in raw
        assert "max-age=" in raw

    async def test_the_spent_state_cookie_is_cleared(
        self, app_client: httpx.AsyncClient, discord_state: FakeDiscordState
    ) -> None:
        await complete_login(app_client, discord_state, "5000")

        assert not app_client.cookies.get("aura_oauth_state")

    async def test_the_session_identifier_is_not_the_discord_token(
        self, app_client: httpx.AsyncClient, discord_state: FakeDiscordState
    ) -> None:
        await complete_login(app_client, discord_state, "5000")

        session_cookie = app_client.cookies["aura_session"]
        assert session_cookie not in discord_state.access_tokens
        assert session_cookie not in discord_state.refresh_tokens

    async def test_responses_are_marked_uncacheable(
        self, app_client: httpx.AsyncClient, discord_state: FakeDiscordState
    ) -> None:
        response = await complete_login(app_client, discord_state, "5000")

        assert response.headers["cache-control"] == "no-store"


class TestStateRejection:
    """Every way the ``state`` check must refuse, per the brief's first attack."""

    async def test_a_callback_with_no_state_is_refused(
        self, app_client: httpx.AsyncClient, discord_state: FakeDiscordState
    ) -> None:
        await start_login(app_client)
        code = discord_state.issue_code("5000")

        response = await app_client.get("/api/auth/callback", params={"code": code})

        assert response.status_code == 400
        assert response.json() == {"error": "invalid_state"}
        assert not app_client.cookies.get("aura_session")

    async def test_a_tampered_state_is_refused(
        self, app_client: httpx.AsyncClient, discord_state: FakeDiscordState
    ) -> None:
        state = await start_login(app_client)
        code = discord_state.issue_code("5000")
        tampered = state[:-1] + ("A" if state[-1] != "A" else "B")

        response = await app_client.get(
            "/api/auth/callback", params={"code": code, "state": tampered}
        )

        assert response.status_code == 400
        assert response.json() == {"error": "invalid_state"}
        assert not app_client.cookies.get("aura_session")

    async def test_a_never_issued_state_is_refused(
        self, app_client: httpx.AsyncClient, discord_state: FakeDiscordState
    ) -> None:
        code = discord_state.issue_code("5000")

        response = await app_client.get(
            "/api/auth/callback", params={"code": code, "state": "not-a-state-we-minted"}
        )

        assert response.status_code == 400
        assert not app_client.cookies.get("aura_session")

    async def test_a_state_cannot_be_replayed(
        self, app_client: httpx.AsyncClient, discord_state: FakeDiscordState
    ) -> None:
        """Single-use: the second callback with the same state must fail."""
        state = await start_login(app_client)
        first_code = discord_state.issue_code("5000")
        second_code = discord_state.issue_code("5000")

        first = await app_client.get(
            "/api/auth/callback", params={"code": first_code, "state": state}
        )
        second = await app_client.get(
            "/api/auth/callback", params={"code": second_code, "state": state}
        )

        assert first.status_code == 303
        assert second.status_code == 400
        assert second.json() == {"error": "invalid_state"}

    async def test_an_attackers_valid_state_is_refused_in_a_victims_browser(
        self, web_settings, discord_state: FakeDiscordState
    ) -> None:
        """The CSRF attack the state parameter exists for, run end to end.

        The attacker starts a real login in their own browser and gets a
        state this service genuinely issued, plus an authorization code for
        their own Discord account. They then lure a victim's browser to the
        callback carrying both. If only the server-side store were checked,
        the state would validate and the VICTIM's browser would end up holding
        a session for the ATTACKER's account. The cookie binding is what makes
        this a 400.
        """
        from fake_discord import create_fake_discord

        fake_discord = create_fake_discord(discord_state)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=fake_discord), base_url="https://discord.test"
        ) as discord_http:
            app = build_app(web_settings, discord_http)
            async with app.router.lifespan_context(app):
                transport = httpx.ASGITransport(app=app)
                # Two clients means two cookie jars, i.e. two browsers.
                async with httpx.AsyncClient(
                    transport=transport, base_url="https://testserver"
                ) as attacker, httpx.AsyncClient(
                    transport=transport, base_url="https://testserver"
                ) as victim:
                    attacker_state = await start_login(attacker)
                    attacker_code = discord_state.issue_code("6000")
                    # The victim has their own pending login, hence their own
                    # state cookie -- the realistic case, and the harder one.
                    await start_login(victim)

                    response = await victim.get(
                        "/api/auth/callback",
                        params={"code": attacker_code, "state": attacker_state},
                    )

                    assert response.status_code == 400
                    assert response.json() == {"error": "invalid_state"}
                    assert not victim.cookies.get("aura_session")

    async def test_a_state_without_its_cookie_is_refused(
        self, app_client: httpx.AsyncClient, discord_state: FakeDiscordState
    ) -> None:
        """A genuinely issued state, presented by a browser that never got the cookie."""
        state = await start_login(app_client)
        app_client.cookies.delete("aura_oauth_state")
        code = discord_state.issue_code("5000")

        response = await app_client.get(
            "/api/auth/callback", params={"code": code, "state": state}
        )

        assert response.status_code == 400
        assert not app_client.cookies.get("aura_session")

    async def test_an_empty_state_is_refused(
        self, app_client: httpx.AsyncClient, discord_state: FakeDiscordState
    ) -> None:
        await start_login(app_client)
        code = discord_state.issue_code("5000")

        response = await app_client.get(
            "/api/auth/callback", params={"code": code, "state": ""}
        )

        assert response.status_code == 400

    async def test_a_denial_is_only_read_after_the_state_checks_out(
        self, app_client: httpx.AsyncClient
    ) -> None:
        """?error= from a forged link must not short-circuit the state check.

        Reported as invalid_state, not oauth_denied: the request never proved
        it came from a login this service started, so nothing in it -- the
        error parameter included -- is trusted enough to be reported back.
        """
        response = await app_client.get(
            "/api/auth/callback", params={"error": "access_denied", "state": "forged"}
        )

        assert response.status_code == 400
        assert response.json() == {"error": "invalid_state"}

    async def test_a_real_user_denial_is_reported_as_a_denial(
        self, app_client: httpx.AsyncClient
    ) -> None:
        state = await start_login(app_client)

        response = await app_client.get(
            "/api/auth/callback", params={"error": "access_denied", "state": state}
        )

        assert response.status_code == 400
        assert response.json() == {"error": "oauth_denied"}
        assert not app_client.cookies.get("aura_session")

    async def test_a_valid_state_with_no_code_is_refused(
        self, app_client: httpx.AsyncClient
    ) -> None:
        state = await start_login(app_client)

        response = await app_client.get("/api/auth/callback", params={"state": state})

        assert response.status_code == 400
        assert response.json() == {"error": "oauth_failed"}
        assert not app_client.cookies.get("aura_session")


class TestCodeRejection:
    async def test_a_forged_authorization_code_is_refused(
        self, app_client: httpx.AsyncClient
    ) -> None:
        state = await start_login(app_client)

        response = await app_client.get(
            "/api/auth/callback", params={"code": "forged-code", "state": state}
        )

        assert response.status_code == 400
        assert response.json() == {"error": "oauth_failed"}
        assert not app_client.cookies.get("aura_session")

    async def test_a_reused_authorization_code_is_refused(
        self, app_client: httpx.AsyncClient, discord_state: FakeDiscordState
    ) -> None:
        code = discord_state.issue_code("5000")
        first_state = await start_login(app_client)
        await app_client.get("/api/auth/callback", params={"code": code, "state": first_state})

        second_state = await start_login(app_client)
        response = await app_client.get(
            "/api/auth/callback", params={"code": code, "state": second_state}
        )

        assert response.status_code == 400
        assert response.json() == {"error": "oauth_failed"}

    async def test_discord_being_down_during_the_exchange_is_a_503_not_a_login(
        self, app_client: httpx.AsyncClient, discord_state: FakeDiscordState
    ) -> None:
        discord_state.fail_token_exchange_status = 502
        state = await start_login(app_client)
        code = discord_state.issue_code("5000")

        response = await app_client.get(
            "/api/auth/callback", params={"code": code, "state": state}
        )

        assert response.status_code == 503
        assert response.json() == {"error": "discord_unavailable"}
        assert not app_client.cookies.get("aura_session")

    async def test_a_token_granted_without_the_guilds_scope_is_refused(
        self, app_client: httpx.AsyncClient, discord_state: FakeDiscordState
    ) -> None:
        """A user who hand-edits the authorize URL to drop a scope gets no session.

        Caught at the exchange rather than as a confusing 403 on the first
        guild listing some minutes later.
        """
        discord_state.granted_scopes = "identify"
        state = await start_login(app_client)
        code = discord_state.issue_code("5000")

        response = await app_client.get(
            "/api/auth/callback", params={"code": code, "state": state}
        )

        assert response.status_code == 400
        assert response.json() == {"error": "oauth_failed"}
        assert not app_client.cookies.get("aura_session")


class TestLogout:
    async def test_logout_clears_the_cookie_and_the_server_side_session(
        self, app_client: httpx.AsyncClient, discord_state: FakeDiscordState
    ) -> None:
        await complete_login(app_client, discord_state, "5000")

        response = await app_client.post("/api/auth/logout")

        assert response.status_code == 204
        assert not app_client.cookies.get("aura_session")
        assert (await app_client.get("/api/me")).status_code == 401

    async def test_logout_revokes_the_token_at_discord(
        self, app_client: httpx.AsyncClient, discord_state: FakeDiscordState
    ) -> None:
        await complete_login(app_client, discord_state, "5000")

        await app_client.post("/api/auth/logout")

        assert "POST /oauth2/token/revoke" in discord_state.request_log
        assert discord_state.revoked_tokens

    async def test_logout_without_a_session_is_still_a_204(
        self, app_client: httpx.AsyncClient
    ) -> None:
        """No oracle: a stolen-cookie probe learns nothing from the status code."""
        response = await app_client.post("/api/auth/logout")

        assert response.status_code == 204

    async def test_logout_with_an_unknown_identifier_is_indistinguishable(
        self, app_client: httpx.AsyncClient
    ) -> None:
        app_client.cookies.set("aura_session", "definitely-not-a-real-session", domain="testserver")

        response = await app_client.post("/api/auth/logout")

        assert response.status_code == 204

    @pytest.mark.parametrize("method", ["get", "put", "delete"])
    async def test_logout_is_not_reachable_by_any_other_method(
        self, app_client: httpx.AsyncClient, method: str
    ) -> None:
        """GET-able logout is CSRF-able logout, via an <img> tag on any page."""
        response = await getattr(app_client, method)("/api/auth/logout")

        assert response.status_code == 405

    async def test_a_session_deleted_server_side_cannot_be_resurrected_by_the_cookie(
        self, app_client: httpx.AsyncClient, discord_state: FakeDiscordState
    ) -> None:
        await complete_login(app_client, discord_state, "5000")
        stolen = app_client.cookies["aura_session"]
        await app_client.post("/api/auth/logout")

        app_client.cookies.set("aura_session", stolen, domain="testserver")
        response = await app_client.get("/api/me")

        assert response.status_code == 401
