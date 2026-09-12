"""/api/me and /api/guilds: what a logged-in browser may read, and what it may not.

The filtering assertions here are deliberately end-to-end rather than unit
tests of select_manageable_guilds (those live in test_guild_selection.py).
The property that matters is not "the pure function filters correctly" but
"the guild a user sees on the page is the intersection of two conditions,
after a real login, against a real permission bitmask Discord serialised as a
string" -- which only the whole stack can demonstrate.
"""
from __future__ import annotations

import asyncio

import httpx
import pytest
from fake_discord import (
    PERMISSION_ADMINISTRATOR,
    PERMISSION_MANAGE_GUILD,
    FakeDiscordState,
    FakeGuild,
    FakeUser,
)
from helpers import complete_login


class TestMe:
    async def test_returns_the_logged_in_identity(
        self, app_client: httpx.AsyncClient, discord_state: FakeDiscordState
    ) -> None:
        await complete_login(app_client, discord_state, "5000")

        response = await app_client.get("/api/me")

        assert response.status_code == 200
        assert response.json() == {
            "id": "5000",
            "username": "moderator",
            "global_name": "The Moderator",
            "avatar": "deadbeef",
        }

    async def test_without_a_session_it_is_a_401_error_code(
        self, app_client: httpx.AsyncClient
    ) -> None:
        response = await app_client.get("/api/me")

        assert response.status_code == 401
        assert response.json() == {"error": "not_authenticated"}

    async def test_a_forged_session_cookie_is_a_401(
        self, app_client: httpx.AsyncClient
    ) -> None:
        app_client.cookies.set("aura_session", "a" * 43, domain="testserver")

        response = await app_client.get("/api/me")

        assert response.status_code == 401

    async def test_one_browsers_session_is_not_visible_to_another(
        self, app_client: httpx.AsyncClient, discord_state: FakeDiscordState
    ) -> None:
        """Two logins, two identifiers, no crossover."""
        await complete_login(app_client, discord_state, "5000")
        moderator_cookie = app_client.cookies["aura_session"]

        app_client.cookies.clear()
        await complete_login(app_client, discord_state, "6000")
        member_cookie = app_client.cookies["aura_session"]

        assert moderator_cookie != member_cookie
        assert (await app_client.get("/api/me")).json()["id"] == "6000"


class TestGuildFiltering:
    async def test_shows_only_guilds_the_user_manages_and_aura_is_in(
        self, app_client: httpx.AsyncClient, discord_state: FakeDiscordState
    ) -> None:
        await complete_login(app_client, discord_state, "5000")

        response = await app_client.get("/api/guilds")

        assert response.status_code == 200
        assert response.json() == [
            {"id": "1000", "name": "Aura Test Server", "icon": "a1b2c3"}
        ]

    async def test_a_guild_the_user_manages_but_aura_is_absent_from_is_excluded(
        self, app_client: httpx.AsyncClient, discord_state: FakeDiscordState
    ) -> None:
        await complete_login(app_client, discord_state, "5000")

        returned_ids = {guild["id"] for guild in (await app_client.get("/api/guilds")).json()}

        assert "2000" not in returned_ids

    async def test_a_guild_aura_is_in_but_the_user_cannot_manage_is_excluded(
        self, app_client: httpx.AsyncClient, discord_state: FakeDiscordState
    ) -> None:
        await complete_login(app_client, discord_state, "5000")

        returned_ids = {guild["id"] for guild in (await app_client.get("/api/guilds")).json()}

        assert "3000" not in returned_ids

    async def test_a_user_who_manages_nothing_sees_an_empty_list_not_an_error(
        self, app_client: httpx.AsyncClient, discord_state: FakeDiscordState
    ) -> None:
        """The brief's third attack, stated exactly: empty list, HTTP 200, no leakage."""
        await complete_login(app_client, discord_state, "6000")

        response = await app_client.get("/api/guilds")

        assert response.status_code == 200
        assert response.json() == []

    async def test_a_user_in_no_guilds_at_all_sees_an_empty_list(
        self, app_client: httpx.AsyncClient, discord_state: FakeDiscordState
    ) -> None:
        discord_state.users["7000"] = FakeUser(id="7000", username="lonely")

        await complete_login(app_client, discord_state, "7000")
        response = await app_client.get("/api/guilds")

        assert response.status_code == 200
        assert response.json() == []

    async def test_an_administrator_without_the_manage_guild_bit_still_qualifies(
        self, app_client: httpx.AsyncClient, discord_state: FakeDiscordState
    ) -> None:
        """Discord does not fold ADMINISTRATOR into the other bits.

        Checking MANAGE_GUILD alone would lock owners and admins out of their
        own dashboard -- the most obvious possible bug, and the easiest to
        ship.
        """
        discord_state.users["8000"] = FakeUser(
            id="8000", username="owner", guild_permissions={"1000": PERMISSION_ADMINISTRATOR}
        )

        await complete_login(app_client, discord_state, "8000")
        response = await app_client.get("/api/guilds")

        assert [guild["id"] for guild in response.json()] == ["1000"]

    async def test_the_bots_own_permissions_in_a_guild_are_never_used_as_the_users(
        self, app_client: httpx.AsyncClient, discord_state: FakeDiscordState
    ) -> None:
        """The bot's membership listing also carries a permissions field.

        The fake sets it to a value that grants nothing, so a service that
        accidentally read permissions off the BOT's entry would return an
        empty list for a user who genuinely manages the guild.
        """
        await complete_login(app_client, discord_state, "5000")

        assert [guild["id"] for guild in (await app_client.get("/api/guilds")).json()] == ["1000"]

    async def test_guilds_are_name_sorted_regardless_of_discords_order(
        self, app_client: httpx.AsyncClient, discord_state: FakeDiscordState
    ) -> None:
        discord_state.guilds["4000"] = FakeGuild(id="4000", name="Alpha Server")
        discord_state.guilds["5000"] = FakeGuild(id="5000", name="zulu server")
        discord_state.bot_guild_ids |= {"4000", "5000"}
        discord_state.users["5000"].guild_permissions.update(
            {"4000": PERMISSION_MANAGE_GUILD, "5000": PERMISSION_MANAGE_GUILD}
        )

        await complete_login(app_client, discord_state, "5000")
        names = [guild["name"] for guild in (await app_client.get("/api/guilds")).json()]

        assert names == ["Alpha Server", "Aura Test Server", "zulu server"]

    async def test_without_a_session_the_guild_list_is_a_401(
        self, app_client: httpx.AsyncClient
    ) -> None:
        response = await app_client.get("/api/guilds")

        assert response.status_code == 401
        assert response.json() == {"error": "not_authenticated"}

    async def test_pagination_across_more_than_one_page_is_followed(
        self, app_client: httpx.AsyncClient, discord_state: FakeDiscordState
    ) -> None:
        """Aura in 250 guilds: the ``after`` cursor must fetch all of them.

        250 is chosen to straddle Discord's 200-per-page maximum, so the
        second page is a real page rather than a synthetic one.
        """
        for index in range(250):
            guild_id = str(100_000 + index)
            discord_state.guilds[guild_id] = FakeGuild(id=guild_id, name=f"Server {index:03d}")
            discord_state.bot_guild_ids.add(guild_id)
        # The user manages only the very last one, which is reachable only if
        # the second page was actually fetched.
        last_guild = str(100_000 + 249)
        discord_state.users["5000"].guild_permissions[last_guild] = PERMISSION_MANAGE_GUILD

        await complete_login(app_client, discord_state, "5000")
        returned_ids = {guild["id"] for guild in (await app_client.get("/api/guilds")).json()}

        assert last_guild in returned_ids


class TestDiscordOutages:
    async def test_a_failing_membership_lookup_fails_closed_rather_than_unfiltered(
        self, app_client: httpx.AsyncClient, discord_state: FakeDiscordState
    ) -> None:
        """The dangerous failure mode: showing every guild because we could not filter.

        A 503 is correct; a 200 listing guild 2000 (which Aura is not in) would
        be the bug this test exists to make impossible to ship.
        """
        await complete_login(app_client, discord_state, "5000")
        discord_state.fail_bot_guilds_status = 503

        response = await app_client.get("/api/guilds")

        assert response.status_code == 503
        assert response.json() == {"error": "discord_unavailable"}

    async def test_a_failing_user_guild_lookup_is_a_503(
        self, app_client: httpx.AsyncClient, discord_state: FakeDiscordState
    ) -> None:
        await complete_login(app_client, discord_state, "5000")
        discord_state.fail_user_guilds_status = 500

        response = await app_client.get("/api/guilds")

        assert response.status_code == 503

    async def test_an_outage_does_not_destroy_the_session(
        self, app_client: httpx.AsyncClient, discord_state: FakeDiscordState
    ) -> None:
        """Unavailability is not a credential problem; the login must survive it."""
        await complete_login(app_client, discord_state, "5000")
        discord_state.fail_bot_guilds_status = 503
        await app_client.get("/api/guilds")

        discord_state.fail_bot_guilds_status = None
        response = await app_client.get("/api/guilds")

        assert response.status_code == 200
        assert [guild["id"] for guild in response.json()] == ["1000"]

    async def test_a_revoked_user_token_ends_the_session(
        self, app_client: httpx.AsyncClient, discord_state: FakeDiscordState
    ) -> None:
        """Revoking access at Discord must log the user out here, not loop on errors."""
        await complete_login(app_client, discord_state, "5000")
        discord_state.revoked_tokens.update(discord_state.access_tokens)

        response = await app_client.get("/api/guilds")

        assert response.status_code == 401
        assert (await app_client.get("/api/me")).status_code == 401


class TestConcurrency:
    async def test_simultaneous_guild_requests_do_not_stampede_discord(
        self, app_client: httpx.AsyncClient, discord_state: FakeDiscordState
    ) -> None:
        """Ten concurrent page loads must cost ONE bot-membership lookup, not ten.

        /users/@me/guilds is the route Discord throttles hardest; a cache
        without a refresh lock turns every cold start into a burst against it.
        """
        await complete_login(app_client, discord_state, "5000")
        discord_state.request_log.clear()

        responses = await asyncio.gather(
            *(app_client.get("/api/guilds") for _ in range(10))
        )

        assert all(response.status_code == 200 for response in responses)
        bot_lookups = discord_state.request_log.count("GET /users/@me/guilds (bot)")
        assert bot_lookups == 1, discord_state.request_log

    async def test_simultaneous_logins_produce_distinct_sessions(
        self, web_settings, discord_state: FakeDiscordState
    ) -> None:
        from fake_discord import create_fake_discord
        from helpers import build_app, complete_login as run_login

        fake_discord = create_fake_discord(discord_state)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=fake_discord), base_url="https://discord.test"
        ) as discord_http:
            app = build_app(web_settings, discord_http)
            async with app.router.lifespan_context(app):
                transport = httpx.ASGITransport(app=app)
                clients = [
                    httpx.AsyncClient(transport=transport, base_url="https://testserver")
                    for _ in range(8)
                ]
                try:
                    await asyncio.gather(
                        *(run_login(client, discord_state, "5000") for client in clients)
                    )
                    cookies = {client.cookies["aura_session"] for client in clients}
                    assert len(cookies) == 8
                finally:
                    await asyncio.gather(*(client.aclose() for client in clients))


class TestHealth:
    async def test_health_needs_no_session_and_no_discord(
        self, app_client: httpx.AsyncClient, discord_state: FakeDiscordState
    ) -> None:
        discord_state.fail_bot_guilds_status = 503

        response = await app_client.get("/api/health")

        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


class TestSecurityHeaders:
    @pytest.mark.parametrize("path", ["/api/health", "/api/me", "/api/guilds"])
    async def test_every_response_is_marked_uncacheable(
        self, app_client: httpx.AsyncClient, path: str
    ) -> None:
        """A shared proxy caching /api/guilds would hand one user's servers to the next."""
        response = await app_client.get(path)

        assert response.headers["cache-control"] == "no-store"
        assert response.headers["x-content-type-options"] == "nosniff"
        assert response.headers["x-frame-options"] == "DENY"
