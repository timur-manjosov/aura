"""A genuine attempt to break the running service with hostile input.

CLAUDE.md's Non-Negotiable Principle asks for this specifically, and asks for
it as tests rather than as a one-off session: malformed, empty, oversized,
duplicated and unicode-hostile input, aimed at the endpoints an unauthenticated
stranger can reach. Each of these was run against the service first and written
down second, so the assertions record observed behaviour rather than hoped-for
behaviour.

The bar throughout is the same: a hostile request must produce a clean refusal
-- never a 500, never a session, never a crash, never a header this service did
not intend to send.
"""
from __future__ import annotations

import httpx
import pytest
from fake_discord import FakeDiscordState
from helpers import complete_login, set_cookie_header, start_login


class TestHostileStateParameter:
    @pytest.mark.parametrize(
        "state",
        [
            "",
            " ",
            "\x00",
            "../../etc/passwd",
            "<script>alert(1)</script>",
            "' OR 1=1 --",
            "%00",
            "🙈🙉🙊",
            "ｆｕｌｌｗｉｄｔｈ",
            "a" * 4000,
        ],
    )
    async def test_a_hostile_state_is_refused_cleanly(
        self, app_client: httpx.AsyncClient, discord_state: FakeDiscordState, state: str
    ) -> None:
        await start_login(app_client)
        code = discord_state.issue_code("5000")

        response = await app_client.get(
            "/api/auth/callback", params={"code": code, "state": state}
        )

        assert response.status_code == 400
        assert response.json() == {"error": "invalid_state"}
        assert not app_client.cookies.get("aura_session")

    async def test_a_state_carrying_crlf_cannot_inject_a_header(
        self, app_client: httpx.AsyncClient, discord_state: FakeDiscordState
    ) -> None:
        """The classic response-splitting probe, aimed at the cookie we echo back."""
        await start_login(app_client)

        response = await app_client.get(
            "/api/auth/callback",
            params={"code": "x", "state": "a\r\nX-Injected: yes\r\n\r\nb"},
        )

        assert response.status_code == 400
        assert "x-injected" not in {name.lower() for name in response.headers}

    @pytest.mark.parametrize("order", ["forged-first", "valid-first"])
    async def test_a_repeated_state_parameter_is_refused_outright(
        self, app_client: httpx.AsyncClient, discord_state: FakeDiscordState, order: str
    ) -> None:
        """Parameter pollution: neither ordering may complete the callback.

        Starlette resolves a repeated parameter to its LAST occurrence, so the
        first version of this test passed one ordering and failed the other --
        which is how the duplicate-parameter hole was found. A repeat is now
        refused whichever value would have won, because Discord never sends
        one and there is no reading of a duplicate that is worth guessing at.
        """
        valid_state = await start_login(app_client)
        code = discord_state.issue_code("5000")
        query = (
            f"code={code}&state=forged&state={valid_state}"
            if order == "forged-first"
            else f"code={code}&state={valid_state}&state=forged"
        )

        response = await app_client.get(f"/api/auth/callback?{query}")

        assert response.status_code == 400
        assert response.json() == {"error": "invalid_state"}
        assert not app_client.cookies.get("aura_session")

    async def test_a_repeated_code_parameter_is_refused(
        self, app_client: httpx.AsyncClient, discord_state: FakeDiscordState
    ) -> None:
        valid_state = await start_login(app_client)
        real_code = discord_state.issue_code("5000")

        response = await app_client.get(
            f"/api/auth/callback?code=forged&code={real_code}&state={valid_state}"
        )

        assert response.status_code == 400
        assert response.json() == {"error": "oauth_failed"}
        assert not app_client.cookies.get("aura_session")

    async def test_a_repeated_error_parameter_still_reads_as_a_denial(
        self, app_client: httpx.AsyncClient
    ) -> None:
        """Duplicates must not turn a denial into an attempted code exchange."""
        state = await start_login(app_client)

        response = await app_client.get(
            f"/api/auth/callback?error=access_denied&error=access_denied&state={state}"
        )

        assert response.status_code == 400
        assert response.json() == {"error": "oauth_denied"}


class TestHostileCookies:
    @pytest.mark.parametrize(
        "value",
        ["", " ", "null", "undefined", "0", "a" * 8000, "🙈", "../../x", "%2e%2e"],
    )
    async def test_a_hostile_session_cookie_is_just_unauthenticated(
        self, app_client: httpx.AsyncClient, value: str
    ) -> None:
        app_client.cookies.set("aura_session", value, domain="testserver")

        response = await app_client.get("/api/me")

        assert response.status_code == 401
        assert response.json() == {"error": "not_authenticated"}

    async def test_a_hostile_state_cookie_cannot_validate_a_forged_state(
        self, app_client: httpx.AsyncClient, discord_state: FakeDiscordState
    ) -> None:
        """Setting both halves to the same attacker-chosen value must still fail.

        This is the check that proves the server-side store is load-bearing:
        with only the cookie comparison, an attacker who controls both the
        query and the cookie would pass.
        """
        app_client.cookies.set("aura_oauth_state", "attacker-chosen", domain="testserver")
        code = discord_state.issue_code("5000")

        response = await app_client.get(
            "/api/auth/callback", params={"code": code, "state": "attacker-chosen"}
        )

        assert response.status_code == 400
        assert response.json() == {"error": "invalid_state"}
        assert not app_client.cookies.get("aura_session")


class TestHostileAuthorizationCode:
    @pytest.mark.parametrize(
        "code", ["", " ", "\x00", "a" * 5000, "🙈", "code&grant_type=client_credentials"]
    )
    async def test_a_hostile_code_never_produces_a_session(
        self, app_client: httpx.AsyncClient, code: str
    ) -> None:
        """The last one matters: the code is form-encoded into the token request.

        An unescaped `&` would add a parameter to that POST body. httpx encodes
        it, and this pins that down rather than trusting it.
        """
        state = await start_login(app_client)

        response = await app_client.get(
            "/api/auth/callback", params={"code": code, "state": state}
        )

        assert response.status_code in (400, 503)
        assert not app_client.cookies.get("aura_session")

    async def test_a_code_cannot_smuggle_an_extra_form_field_into_the_exchange(
        self, app_client: httpx.AsyncClient, discord_state: FakeDiscordState
    ) -> None:
        state = await start_login(app_client)
        real_code = discord_state.issue_code("5000")

        response = await app_client.get(
            "/api/auth/callback",
            params={"code": f"{real_code}&grant_type=refresh_token", "state": state},
        )

        # The whole string is sent as one `code` value, so Discord simply does
        # not recognise it -- rather than the grant type being overridden.
        assert response.status_code == 400
        assert not app_client.cookies.get("aura_session")


class TestHostileDiscordIdentity:
    async def test_a_username_that_is_pure_control_characters_still_logs_in(
        self, app_client: httpx.AsyncClient, discord_state: FakeDiscordState
    ) -> None:
        """Sanitising to nothing must fall back, not produce a blank identity."""
        discord_state.users["9000"] = type(discord_state.users["5000"])(
            id="9000", username="\x00\x01\x02", global_name=None, avatar=None
        )

        await complete_login(app_client, discord_state, "9000")
        response = await app_client.get("/api/me")

        assert response.status_code == 200
        assert response.json()["username"] == "user-9000"

    async def test_a_guild_name_of_bidi_overrides_is_neutralised(
        self, app_client: httpx.AsyncClient, discord_state: FakeDiscordState
    ) -> None:
        """A right-to-left override in a server name misrenders every consumer.

        The character is written as an escape rather than pasted literally: a
        raw U+202E in a source file makes the line itself render misleadingly
        in editors and review tools, which would be an odd thing to introduce
        in the test that exists to strip it.
        """
        from fake_discord import PERMISSION_MANAGE_GUILD, FakeGuild

        discord_state.guilds["4242"] = FakeGuild(id="4242", name="safe\u202ereversed")
        discord_state.bot_guild_ids.add("4242")
        discord_state.users["5000"].guild_permissions["4242"] = PERMISSION_MANAGE_GUILD

        await complete_login(app_client, discord_state, "5000")
        names = [guild["name"] for guild in (await app_client.get("/api/guilds")).json()]

        assert all("\u202e" not in name for name in names), names

    async def test_an_emoji_guild_name_survives_intact(
        self, app_client: httpx.AsyncClient, discord_state: FakeDiscordState
    ) -> None:
        """Sanitising must not mangle the names real servers actually use."""
        from fake_discord import PERMISSION_MANAGE_GUILD, FakeGuild

        discord_state.guilds["4243"] = FakeGuild(id="4243", name="🎮 ゲーム鯖 • Süß")
        discord_state.bot_guild_ids.add("4243")
        discord_state.users["5000"].guild_permissions["4243"] = PERMISSION_MANAGE_GUILD

        await complete_login(app_client, discord_state, "5000")
        names = [guild["name"] for guild in (await app_client.get("/api/guilds")).json()]

        assert "🎮 ゲーム鯖 • Süß" in names

    async def test_a_guild_reported_twice_by_discord_appears_once(
        self, app_client: httpx.AsyncClient, discord_state: FakeDiscordState
    ) -> None:
        """Discord listing the bot's membership twice must not duplicate a card."""
        discord_state.bot_guild_ids.add("1000")

        await complete_login(app_client, discord_state, "5000")
        guilds = (await app_client.get("/api/guilds")).json()

        assert len({guild["id"] for guild in guilds}) == len(guilds)


class TestUnexpectedMethodsAndPaths:
    @pytest.mark.parametrize(
        ("method", "path"),
        [
            ("POST", "/api/auth/login"),
            ("POST", "/api/auth/callback"),
            ("DELETE", "/api/me"),
            ("PUT", "/api/guilds"),
            ("PATCH", "/api/health"),
        ],
    )
    async def test_a_wrong_method_is_405_not_a_crash(
        self, app_client: httpx.AsyncClient, method: str, path: str
    ) -> None:
        response = await app_client.request(method, path)

        assert response.status_code == 405

    @pytest.mark.parametrize(
        "path",
        [
            "/api/me/",
            "/api/../api/me",
            "/api/me/../guilds",
            "/openapi.json",
            "/docs",
            "/redoc",
            "/api/auth/",
        ],
    )
    async def test_an_unrouted_path_does_not_leak(
        self, app_client: httpx.AsyncClient, discord_state: FakeDiscordState, path: str
    ) -> None:
        """The docs routes are off deliberately: they describe an authed API to anyone."""
        await complete_login(app_client, discord_state, "5000")

        response = await app_client.get(path)

        # 200 is acceptable for the traversal spellings: the client normalises
        # "/api/../api/me" to "/api/me" before it is ever sent, so what arrives
        # is an ordinary authenticated request for the caller's OWN identity --
        # no traversal reached the server, and nothing was bypassed.
        assert response.status_code in (200, 307, 404, 405), response.text
        assert "access_token" not in response.text
        for token in discord_state.access_tokens:
            assert token not in response.text

    async def test_a_head_request_returns_no_body_and_no_secret(
        self, app_client: httpx.AsyncClient, discord_state: FakeDiscordState
    ) -> None:
        await complete_login(app_client, discord_state, "5000")

        response = await app_client.head("/api/me")

        assert response.content == b""
        for token in discord_state.access_tokens:
            assert token not in str(response.headers)


class TestSessionIsolationUnderAbuse:
    async def test_a_flood_of_logins_cannot_hand_out_a_reused_identifier(
        self, app_client: httpx.AsyncClient, discord_state: FakeDiscordState
    ) -> None:
        """A collision would put one person into another person's session."""
        identifiers = set()
        for _ in range(40):
            app_client.cookies.clear()
            response = await complete_login(app_client, discord_state, "5000")
            identifiers.add(set_cookie_header(response, "aura_session").split(";")[0])

        assert len(identifiers) == 40

    async def test_one_users_session_never_returns_another_users_guilds(
        self, app_client: httpx.AsyncClient, discord_state: FakeDiscordState
    ) -> None:
        await complete_login(app_client, discord_state, "5000")
        moderator_cookie = app_client.cookies["aura_session"]

        app_client.cookies.clear()
        await complete_login(app_client, discord_state, "6000")
        member_cookie = app_client.cookies["aura_session"]

        app_client.cookies.clear()
        member_view = (
            await app_client.get("/api/guilds", headers={"Cookie": f"aura_session={member_cookie}"})
        ).json()
        moderator_view = (
            await app_client.get(
                "/api/guilds", headers={"Cookie": f"aura_session={moderator_cookie}"}
            )
        ).json()

        assert member_view == []
        assert [guild["id"] for guild in moderator_view] == ["1000"]

    async def test_truncating_a_valid_identifier_by_one_character_fails(
        self, app_client: httpx.AsyncClient, discord_state: FakeDiscordState
    ) -> None:
        """No prefix matching anywhere in the lookup path."""
        await complete_login(app_client, discord_state, "5000")
        valid = app_client.cookies["aura_session"]
        # Sent as an explicit header rather than through httpx's cookie jar:
        # the jar keeps a manually-set cookie alongside the server-set one
        # under a different domain spelling and sends BOTH, so a jar-based
        # version of this test proves nothing about the truncated value.
        app_client.cookies.clear()

        response = await app_client.get(
            "/api/me", headers={"Cookie": f"aura_session={valid[:-1]}"}
        )

        assert response.status_code == 401
