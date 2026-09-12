"""DiscordClient against responses Discord should never send, and sometimes does.

Driven through httpx.MockTransport rather than the fake Discord app, because
the point here is the malformed, truncated and hostile payload -- shapes a
well-behaved double will not produce. Everything the client does with a
response is exercised over real HTTP machinery; only the bytes are chosen.

The guiding rule these pin down: a bad response must become a typed error,
never a plausible-looking value. An empty guild list invented from a broken
payload is indistinguishable, downstream, from a correct one.
"""
from __future__ import annotations

import json
from collections.abc import Callable

import httpx
import pytest

from aura_web.discord_api import (
    MAX_GUILD_PAGES,
    DiscordAuthError,
    DiscordClient,
    DiscordUnavailableError,
)

API_BASE = "https://discord.test/api/v10"


def client_with(handler: Callable[[httpx.Request], httpx.Response]) -> DiscordClient:
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return DiscordClient(
        http,
        api_base=API_BASE,
        client_id="123456789012345678",
        client_secret="secret",
        bot_token="bot-token",
    )


def responding(status_code: int, payload: object = None, *, text: str | None = None):
    def handler(request: httpx.Request) -> httpx.Response:
        if text is not None:
            return httpx.Response(status_code, text=text)
        return httpx.Response(status_code, json=payload)

    return handler


class TestTokenExchange:
    async def test_a_well_formed_exchange_returns_tokens(self) -> None:
        discord = client_with(
            responding(
                200,
                {
                    "access_token": "at",
                    "refresh_token": "rt",
                    "expires_in": 604800,
                    "scope": "identify guilds",
                    "token_type": "Bearer",
                },
            )
        )

        tokens = await discord.exchange_code("code", "https://frontend.test/cb")

        assert tokens.access_token == "at"
        assert tokens.refresh_token == "rt"

    async def test_the_client_secret_travels_in_basic_auth_not_the_body(self) -> None:
        """The body is what gets logged by proxies and debug patches; the header is not."""
        seen: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["auth"] = request.headers.get("Authorization", "")
            seen["body"] = request.content.decode()
            seen["content_type"] = request.headers.get("Content-Type")
            return httpx.Response(
                200,
                json={"access_token": "at", "refresh_token": "rt", "expires_in": 10, "scope": "identify guilds"},
            )

        await client_with(handler).exchange_code("code", "https://frontend.test/cb")

        assert str(seen["auth"]).startswith("Basic ")
        assert "secret" not in str(seen["body"])
        assert seen["content_type"] == "application/x-www-form-urlencoded"

    async def test_a_400_invalid_grant_is_an_auth_error_not_an_outage(self) -> None:
        """Discord answers a reused or forged code with 400, not 401.

        Misclassifying it as unavailability would turn a replayed callback
        into a retryable 503 instead of a refusal.
        """
        discord = client_with(responding(400, {"error": "invalid_grant"}))

        with pytest.raises(DiscordAuthError):
            await discord.exchange_code("code", "https://frontend.test/cb")

    async def test_a_401_is_an_auth_error(self) -> None:
        discord = client_with(responding(401, {"error": "invalid_client"}))

        with pytest.raises(DiscordAuthError):
            await discord.exchange_code("code", "https://frontend.test/cb")

    @pytest.mark.parametrize("status", [500, 502, 503, 504])
    async def test_a_server_error_is_unavailability(self, status: int) -> None:
        discord = client_with(responding(status, {"message": "oops"}))

        with pytest.raises(DiscordUnavailableError):
            await discord.exchange_code("code", "https://frontend.test/cb")

    async def test_a_rate_limit_is_unavailability_and_is_not_retried(self) -> None:
        """Retrying inside a handler amplifies a limit Discord already imposed."""
        calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            return httpx.Response(429, json={"retry_after": 5}, headers={"Retry-After": "5"})

        with pytest.raises(DiscordUnavailableError):
            await client_with(handler).exchange_code("code", "https://frontend.test/cb")
        assert calls == 1

    async def test_a_network_failure_is_unavailability(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("no route to host")

        with pytest.raises(DiscordUnavailableError):
            await client_with(handler).exchange_code("code", "https://frontend.test/cb")

    @pytest.mark.parametrize(
        "body", ["", "not json", "<html>503</html>", "{", "null", "[]", '"a string"']
    )
    async def test_a_non_json_or_wrongly_shaped_body_is_unavailability(self, body: str) -> None:
        discord = client_with(responding(200, text=body))

        with pytest.raises(DiscordUnavailableError):
            await discord.exchange_code("code", "https://frontend.test/cb")

    @pytest.mark.parametrize(
        "payload",
        [
            {"refresh_token": "rt", "expires_in": 10, "scope": "identify guilds"},
            {"access_token": "", "expires_in": 10, "scope": "identify guilds"},
            {"access_token": None, "expires_in": 10, "scope": "identify guilds"},
            {"access_token": 12345, "expires_in": 10, "scope": "identify guilds"},
        ],
    )
    async def test_a_missing_or_unusable_access_token_is_unavailability(
        self, payload: dict[str, object]
    ) -> None:
        discord = client_with(responding(200, payload))

        with pytest.raises(DiscordUnavailableError):
            await discord.exchange_code("code", "https://frontend.test/cb")

    @pytest.mark.parametrize("scope", ["identify", "guilds", "", "email", "identify email"])
    async def test_a_grant_missing_a_required_scope_is_refused(self, scope: str) -> None:
        discord = client_with(
            responding(200, {"access_token": "at", "expires_in": 10, "scope": scope})
        )

        with pytest.raises(DiscordAuthError):
            await discord.exchange_code("code", "https://frontend.test/cb")

    async def test_an_extra_granted_scope_does_not_break_the_exchange(self) -> None:
        """Discord may return more than was asked for; only the minimum is required."""
        discord = client_with(
            responding(
                200, {"access_token": "at", "expires_in": 10, "scope": "identify guilds email"}
            )
        )

        assert (await discord.exchange_code("c", "https://frontend.test/cb")).access_token == "at"

    @pytest.mark.parametrize("expires_in", [None, "soon", -5, 0, [], {}, True])
    async def test_an_unreadable_expiry_falls_back_to_a_short_lifetime(
        self, expires_in: object
    ) -> None:
        """The dangerous direction is a too-LONG life; every bad value reads as short."""
        discord = client_with(
            responding(
                200,
                {"access_token": "at", "expires_in": expires_in, "scope": "identify guilds"},
            )
        )

        tokens = await discord.exchange_code("c", "https://frontend.test/cb")

        from aura_web.sessions import utc_now

        assert (tokens.expires_at - utc_now()).total_seconds() <= 3601

    async def test_an_absurd_expiry_is_clamped(self) -> None:
        discord = client_with(
            responding(
                200,
                {"access_token": "at", "expires_in": 10**12, "scope": "identify guilds"},
            )
        )

        tokens = await discord.exchange_code("c", "https://frontend.test/cb")

        from aura_web.sessions import utc_now

        assert (tokens.expires_at - utc_now()).days <= 30

    async def test_a_missing_refresh_token_is_tolerated_as_none(self) -> None:
        discord = client_with(
            responding(200, {"access_token": "at", "expires_in": 60, "scope": "identify guilds"})
        )

        assert (await discord.exchange_code("c", "https://frontend.test/cb")).refresh_token is None


class TestRefresh:
    async def test_a_refresh_does_not_re_require_scopes(self) -> None:
        """Discord's refresh response need not repeat the scope field.

        Failing on it would log a user out for something that cannot have
        changed since the exchange already proved it.
        """
        discord = client_with(responding(200, {"access_token": "new", "expires_in": 60}))

        assert (await discord.refresh_tokens("rt")).access_token == "new"

    async def test_a_rejected_refresh_token_is_an_auth_error(self) -> None:
        discord = client_with(responding(400, {"error": "invalid_grant"}))

        with pytest.raises(DiscordAuthError):
            await discord.refresh_tokens("rt")


class TestRevoke:
    async def test_revocation_never_raises_on_a_failure(self) -> None:
        """Logout must not fail on somebody else's availability."""
        discord = client_with(responding(500, {"error": "down"}))

        await discord.revoke_token("at")

    async def test_revocation_never_raises_on_a_network_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectTimeout("timed out")

        await client_with(handler).revoke_token("at")


class TestCurrentUser:
    async def test_a_well_formed_user_is_parsed(self) -> None:
        discord = client_with(
            responding(
                200,
                {"id": "5000", "username": "mod", "global_name": "Mod", "avatar": "abc123"},
            )
        )

        user = await discord.fetch_current_user("at")

        assert (user.id, user.username, user.global_name, user.avatar) == (
            "5000",
            "mod",
            "Mod",
            "abc123",
        )

    async def test_the_bearer_token_is_sent_in_the_authorization_header(self) -> None:
        seen: dict[str, str] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["auth"] = request.headers.get("Authorization", "")
            return httpx.Response(200, json={"id": "1", "username": "u"})

        await client_with(handler).fetch_current_user("the-token")

        assert seen["auth"] == "Bearer the-token"

    @pytest.mark.parametrize(
        "payload",
        [{}, {"id": None}, {"id": "not-a-snowflake"}, {"id": -1}, {"id": ["1"]}, []],
    )
    async def test_an_unusable_identity_is_unavailability(self, payload: object) -> None:
        discord = client_with(responding(200, payload))

        with pytest.raises(DiscordUnavailableError):
            await discord.fetch_current_user("at")

    async def test_a_missing_username_falls_back_rather_than_raising(self) -> None:
        """An identity with a usable id is still a usable identity."""
        discord = client_with(responding(200, {"id": "5000"}))

        user = await discord.fetch_current_user("at")

        assert user.id == "5000"
        assert user.username == "user-5000"

    async def test_a_hostile_username_is_sanitised(self) -> None:
        discord = client_with(
            responding(200, {"id": "5000", "username": "mod\x00\r\nSet-Cookie: x=y"})
        )

        user = await discord.fetch_current_user("at")

        assert "\r" not in user.username and "\n" not in user.username

    async def test_a_junk_avatar_hash_becomes_none(self) -> None:
        discord = client_with(
            responding(200, {"id": "5000", "username": "u", "avatar": "../../secret"})
        )

        assert (await discord.fetch_current_user("at")).avatar is None

    async def test_a_401_ends_as_an_auth_error(self) -> None:
        discord = client_with(responding(401, {"message": "401: Unauthorized"}))

        with pytest.raises(DiscordAuthError):
            await discord.fetch_current_user("at")


class TestGuildListing:
    async def test_a_well_formed_page_is_parsed(self) -> None:
        discord = client_with(
            responding(200, [{"id": "1", "name": "One", "icon": "aa", "permissions": "32"}])
        )

        guilds = await discord.fetch_user_guilds("at")

        assert [(g.id, g.name, g.icon, g.permissions) for g in guilds] == [
            ("1", "One", "aa", "32")
        ]

    async def test_the_bot_token_is_sent_for_the_membership_lookup(self) -> None:
        seen: dict[str, str] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["auth"] = request.headers.get("Authorization", "")
            return httpx.Response(200, json=[])

        await client_with(handler).fetch_bot_guild_ids()

        assert seen["auth"] == "Bot bot-token"

    @pytest.mark.parametrize("payload", [{}, "a string", None, 42, {"guilds": []}])
    async def test_a_non_array_response_is_unavailability(self, payload: object) -> None:
        """An object where an array belongs must not silently read as "no guilds"."""
        discord = client_with(responding(200, payload))

        with pytest.raises(DiscordUnavailableError):
            await discord.fetch_user_guilds("at")

    async def test_unusable_entries_are_skipped_not_fatal(self) -> None:
        """One broken row must not cost the user every other guild they manage."""
        discord = client_with(
            responding(
                200,
                [
                    "not an object",
                    {"name": "no id"},
                    {"id": "bad-id", "name": "x"},
                    {"id": "1", "name": "Good", "permissions": "32"},
                ],
            )
        )

        guilds = await discord.fetch_user_guilds("at")

        assert [guild.id for guild in guilds] == ["1"]

    async def test_an_integer_permission_field_is_normalised_to_a_string(self) -> None:
        discord = client_with(responding(200, [{"id": "1", "name": "One", "permissions": 32}]))

        guilds = await discord.fetch_user_guilds("at")

        assert guilds[0].permissions == "32"

    async def test_a_missing_permission_field_yields_no_access(self) -> None:
        from aura_web.permissions import has_manage_guild

        discord = client_with(responding(200, [{"id": "1", "name": "One"}]))

        guilds = await discord.fetch_user_guilds("at")

        assert has_manage_guild(guilds[0].permissions) is False

    async def test_pagination_follows_the_after_cursor(self) -> None:
        pages = [
            [{"id": str(index), "name": f"G{index}", "permissions": "32"} for index in range(1, 201)],
            [{"id": "201", "name": "G201", "permissions": "32"}],
        ]
        seen_queries: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen_queries.append(str(request.url.params))
            return httpx.Response(200, json=pages[len(seen_queries) - 1] if len(seen_queries) <= 2 else [])

        guilds = await client_with(handler).fetch_user_guilds("at")

        assert len(guilds) == 201
        assert "after=200" in seen_queries[1]

    async def test_pagination_stops_on_a_short_page(self) -> None:
        calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            return httpx.Response(200, json=[{"id": "1", "name": "One", "permissions": "32"}])

        await client_with(handler).fetch_user_guilds("at")

        assert calls == 1

    async def test_a_cursor_that_never_advances_does_not_loop_forever(self) -> None:
        """A hostile or broken upstream must not hang a request handler."""
        calls = 0
        page = [
            {"id": str(index), "name": f"G{index}", "permissions": "32"} for index in range(200)
        ]

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            # Always the same page, so the cursor cannot move past it.
            return httpx.Response(200, json=page)

        guilds = await client_with(handler).fetch_bot_guild_ids()

        assert calls <= MAX_GUILD_PAGES + 1
        assert len(guilds) == 200

    async def test_duplicate_guilds_across_pages_are_deduplicated(self) -> None:
        first = [{"id": str(index), "name": "G", "permissions": "32"} for index in range(1, 201)]
        second = [{"id": "200", "name": "G", "permissions": "32"}]
        responses = [first, second, []]
        index = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal index
            payload = responses[min(index, len(responses) - 1)]
            index += 1
            return httpx.Response(200, json=payload)

        guilds = await client_with(handler).fetch_user_guilds("at")

        assert len({guild.id for guild in guilds}) == len(guilds)

    async def test_an_empty_list_is_an_empty_result_not_an_error(self) -> None:
        discord = client_with(responding(200, []))

        assert await discord.fetch_user_guilds("at") == []
        assert await discord.fetch_bot_guild_ids() == frozenset()

    async def test_a_huge_guild_name_is_truncated_before_it_is_stored(self) -> None:
        discord = client_with(
            responding(200, [{"id": "1", "name": "x" * 1_000_000, "permissions": "32"}])
        )

        guilds = await discord.fetch_user_guilds("at")

        assert len(guilds[0].name) == 100


class TestRedirectsAreNotFollowed:
    async def test_a_redirect_is_not_followed_to_another_host(self) -> None:
        """Following one would forward a bearer token wherever it pointed.

        The production client sets follow_redirects=False; this asserts the
        behaviour rather than the flag, by pointing a 302 at an attacker host
        and checking the token never arrives there.
        """
        visited: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            visited.append(str(request.url))
            if "evil" in str(request.url):
                return httpx.Response(200, json={"id": "1", "username": "pwned"})
            return httpx.Response(302, headers={"Location": "https://evil.test/steal"})

        http = httpx.AsyncClient(transport=httpx.MockTransport(handler), follow_redirects=False)
        discord = DiscordClient(
            http, api_base=API_BASE, client_id="1", client_secret="s", bot_token="b"
        )

        with pytest.raises(DiscordUnavailableError):
            await discord.fetch_current_user("at")
        assert not any("evil" in url for url in visited)


class TestOversizedPayloads:
    async def test_a_deeply_nested_json_body_does_not_crash_the_parser(self) -> None:
        """Rejected as a bad shape, not as a stack overflow inside a handler."""
        nested: object = {"id": "1"}
        for _ in range(200):
            nested = {"nested": nested}
        discord = client_with(responding(200, text=json.dumps(nested)))

        with pytest.raises(DiscordUnavailableError):
            await discord.fetch_user_guilds("at")
