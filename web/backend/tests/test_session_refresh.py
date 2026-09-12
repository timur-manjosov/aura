"""What happens when a Discord access token expires while a session is still alive.

Sessions outlive access tokens by design (seven days against Discord's
lifetime), so the refresh path is not an edge case -- it is what every
long-lived session eventually does. The three outcomes it must distinguish:
refreshed (carry on), refused (log out), unreachable (fail this request but
keep the login).
"""
from __future__ import annotations

import httpx
from fake_discord import FakeDiscordState
from helpers import complete_login


class TestAutomaticRefresh:
    async def test_an_expired_access_token_is_refreshed_transparently(
        self, app_client: httpx.AsyncClient, discord_state: FakeDiscordState
    ) -> None:
        # Shorter than the 60-second refresh leeway, so the very next request
        # treats the token as expired without any waiting.
        discord_state.token_lifetime_seconds = 10
        await complete_login(app_client, discord_state, "5000")
        discord_state.request_log.clear()

        response = await app_client.get("/api/me")

        assert response.status_code == 200
        assert response.json()["id"] == "5000"
        assert discord_state.request_log.count("POST /oauth2/token") == 1

    async def test_the_refreshed_session_keeps_its_identifier(
        self, app_client: httpx.AsyncClient, discord_state: FakeDiscordState
    ) -> None:
        """Rotating on refresh would log out every other tab of the same browser."""
        discord_state.token_lifetime_seconds = 10
        await complete_login(app_client, discord_state, "5000")
        before = app_client.cookies["aura_session"]

        await app_client.get("/api/me")

        assert app_client.cookies["aura_session"] == before

    async def test_a_refreshed_session_can_still_list_guilds(
        self, app_client: httpx.AsyncClient, discord_state: FakeDiscordState
    ) -> None:
        discord_state.token_lifetime_seconds = 10
        await complete_login(app_client, discord_state, "5000")

        response = await app_client.get("/api/guilds")

        assert response.status_code == 200
        assert [guild["id"] for guild in response.json()] == ["1000"]

    async def test_no_refreshed_token_reaches_the_browser(
        self, app_client: httpx.AsyncClient, discord_state: FakeDiscordState
    ) -> None:
        discord_state.token_lifetime_seconds = 10
        await complete_login(app_client, discord_state, "5000")

        response = await app_client.get("/api/me")

        body_and_headers = response.text + "".join(
            f"{name}{value}" for name, value in response.headers.items()
        )
        for token in list(discord_state.access_tokens) + list(discord_state.refresh_tokens):
            assert token not in body_and_headers


class TestRefusedRefresh:
    async def test_a_refresh_discord_rejects_ends_the_session(
        self, app_client: httpx.AsyncClient, discord_state: FakeDiscordState
    ) -> None:
        """A dead credential must log the user out, not leave a page that fails forever."""
        discord_state.token_lifetime_seconds = 10
        await complete_login(app_client, discord_state, "5000")
        discord_state.refresh_tokens.clear()

        response = await app_client.get("/api/me")

        assert response.status_code == 401
        assert response.json() == {"error": "not_authenticated"}

    async def test_the_ended_session_is_gone_server_side_too(
        self, app_client: httpx.AsyncClient, discord_state: FakeDiscordState
    ) -> None:
        discord_state.token_lifetime_seconds = 10
        await complete_login(app_client, discord_state, "5000")
        discord_state.refresh_tokens.clear()
        await app_client.get("/api/me")

        discord_state.refresh_tokens["anything"] = "5000"
        assert (await app_client.get("/api/me")).status_code == 401


class TestUnreachableDuringRefresh:
    async def test_an_outage_during_refresh_is_a_503_not_a_logout(
        self, app_client: httpx.AsyncClient, discord_state: FakeDiscordState
    ) -> None:
        """Unavailability is not a credential problem; the login must survive it."""
        discord_state.token_lifetime_seconds = 10
        await complete_login(app_client, discord_state, "5000")
        discord_state.fail_token_exchange_status = 503

        response = await app_client.get("/api/me")

        assert response.status_code == 503
        assert response.json() == {"error": "discord_unavailable"}

    async def test_the_session_still_works_once_discord_returns(
        self, app_client: httpx.AsyncClient, discord_state: FakeDiscordState
    ) -> None:
        discord_state.token_lifetime_seconds = 10
        await complete_login(app_client, discord_state, "5000")
        discord_state.fail_token_exchange_status = 503
        await app_client.get("/api/me")

        discord_state.fail_token_exchange_status = None
        response = await app_client.get("/api/me")

        assert response.status_code == 200

    async def test_an_outage_during_refresh_also_fails_the_guild_list_closed(
        self, app_client: httpx.AsyncClient, discord_state: FakeDiscordState
    ) -> None:
        discord_state.token_lifetime_seconds = 10
        await complete_login(app_client, discord_state, "5000")
        discord_state.fail_token_exchange_status = 503

        response = await app_client.get("/api/guilds")

        assert response.status_code == 503
        assert response.json() == {"error": "discord_unavailable"}
