"""A page load spends Discord's per-user budget once, however many endpoints it calls.

The production failure this pins down: the frontend loads the guild list and
the billing view at the same moment, each asked Discord for the same user's
guilds, and Discord refused the second call with a 429 -- so a signed-in admin
saw "Plan information isn't available right now." The Discord double here
enforces a per-token budget of one request, which is what turns that failure
into a test.
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from fake_discord import FakeDiscordState
from helpers import FRONTEND_BASE, complete_login


@pytest.fixture
async def moderator_on_a_tight_budget(
    app_client: httpx.AsyncClient, discord_state: FakeDiscordState
) -> httpx.AsyncClient:
    discord_state.user_guilds_requests_per_token = 1
    await complete_login(app_client, discord_state, "5000")
    return app_client


def user_guild_requests(discord_state: FakeDiscordState) -> int:
    return discord_state.request_log.count("GET /users/@me/guilds (user)")


class TestOnePageLoad:
    async def test_the_parallel_page_load_succeeds_with_one_discord_call(
        self, moderator_on_a_tight_budget: httpx.AsyncClient, discord_state: FakeDiscordState
    ) -> None:
        responses = await asyncio.gather(
            moderator_on_a_tight_budget.get("/api/me"),
            moderator_on_a_tight_budget.get("/api/guilds"),
            moderator_on_a_tight_budget.get("/api/billing/guilds"),
        )
        assert [response.status_code for response in responses] == [200, 200, 200]
        assert [guild["id"] for guild in responses[1].json()] == ["1000"]
        assert [guild["id"] for guild in responses[2].json()] == ["1000"]
        assert user_guild_requests(discord_state) == 1

    async def test_a_reload_and_a_checkout_right_after_still_cost_one_call(
        self, moderator_on_a_tight_budget: httpx.AsyncClient, discord_state: FakeDiscordState
    ) -> None:
        client = moderator_on_a_tight_budget
        for _ in range(3):
            assert (await client.get("/api/guilds")).status_code == 200
            assert (await client.get("/api/billing/guilds")).status_code == 200
        checkout = await client.post(
            "/api/billing/checkout",
            content=json.dumps({"guild_id": "1000"}),
            headers={"Content-Type": "application/json", "Origin": FRONTEND_BASE},
        )
        assert checkout.status_code == 200
        assert user_guild_requests(discord_state) == 1

    async def test_the_cached_list_still_refuses_a_guild_the_user_cannot_manage(
        self, moderator_on_a_tight_budget: httpx.AsyncClient
    ) -> None:
        client = moderator_on_a_tight_budget
        await client.get("/api/guilds")
        for guild_id in ("2000", "3000"):
            response = await client.post(
                "/api/billing/checkout",
                content=json.dumps({"guild_id": guild_id}),
                headers={"Content-Type": "application/json", "Origin": FRONTEND_BASE},
            )
            assert response.status_code == 403


class TestSessionsStaySeparate:
    async def test_another_user_s_page_load_gets_their_own_guilds(
        self,
        app_client: httpx.AsyncClient,
        discord_state: FakeDiscordState,
    ) -> None:
        discord_state.user_guilds_requests_per_token = 1
        await complete_login(app_client, discord_state, "5000")
        assert [g["id"] for g in (await app_client.get("/api/guilds")).json()] == ["1000"]
        assert (await app_client.post("/api/auth/logout")).status_code == 204
        await complete_login(app_client, discord_state, "6000")
        assert (await app_client.get("/api/guilds")).json() == []
        assert user_guild_requests(discord_state) == 2

    async def test_logout_then_login_again_asks_discord_afresh(
        self,
        app_client: httpx.AsyncClient,
        discord_state: FakeDiscordState,
    ) -> None:
        await complete_login(app_client, discord_state, "5000")
        await app_client.get("/api/guilds")
        await app_client.post("/api/auth/logout")
        discord_state.users["5000"].guild_permissions["1000"] = 2048
        await complete_login(app_client, discord_state, "5000")
        assert (await app_client.get("/api/guilds")).json() == []
