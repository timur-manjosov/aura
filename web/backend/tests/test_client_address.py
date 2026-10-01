"""The trust rule for X-Forwarded-For, attacked at every hop of the deployed chain.

The chain is browser -> host Caddy -> frontend (Next.js) -> backend. What each
hop does to the forwarding headers was measured against the deployed versions
(see aura_web.client_address); these tests encode that measurement and then
forge what each party could forge:

  * a client, through Caddy: any X-Forwarded-For it sends is replaced, so the
    backend sees one entry -- its real address -- with the frontend as peer;
  * a client, through Caddy, with Forwarded or X-Real-IP: passed through
    untouched, so the backend must never read them;
  * anything that reaches the backend NOT from the frontend (a process on the
    host, the bot's network): its X-Forwarded-For must count for nothing.

The application-level tests drive the real app through httpx's ASGI transport,
whose `client` argument is exactly the connection peer uvicorn would report.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import socket
from collections.abc import AsyncIterator
from ipaddress import IPv4Address, IPv6Address, ip_address
from typing import Final

import httpx
import pytest
import pytest_asyncio
import uvicorn

from aura_web.__main__ import uvicorn_config
from aura_web.client_address import (
    IGNORED_FORWARDING_HEADERS,
    MAX_FORWARDED_HOPS,
    parse_address,
    resolve_client_address,
)
from aura_web.config import WebSettings
from fake_discord import FakeDiscordState, create_fake_discord
from helpers import build_app

FRONTEND: Final = "172.16.86.3"
GATEWAY: Final = "172.16.86.1"
TRUSTED: Final[frozenset[IPv4Address | IPv6Address]] = frozenset({ip_address(FRONTEND)})
CLIENT: Final = "203.0.113.7"
OTHER_CLIENT: Final = "198.51.100.23"
FORGED: Final = "192.0.2.66"


class TestResolveClientAddress:
    def test_an_untrusted_peer_is_the_client_whatever_it_forwards(self) -> None:
        assert resolve_client_address(GATEWAY, [FORGED], TRUSTED) == GATEWAY

    def test_the_trusted_proxy_s_single_entry_is_the_client(self) -> None:
        assert resolve_client_address(FRONTEND, [CLIENT], TRUSTED) == CLIENT

    def test_the_rightmost_entry_wins_over_anything_to_its_left(self) -> None:
        """Only the rightmost entry was written by a party this service trusts."""
        assert resolve_client_address(FRONTEND, [f"{FORGED}, {CLIENT}"], TRUSTED) == CLIENT

    def test_several_header_lines_are_one_list_in_order(self) -> None:
        assert resolve_client_address(FRONTEND, [FORGED, CLIENT], TRUSTED) == CLIENT

    def test_trusted_hops_are_skipped_from_the_right(self) -> None:
        assert resolve_client_address(FRONTEND, [f"{CLIENT}, {FRONTEND}"], TRUSTED) == CLIENT

    def test_a_chain_of_only_trusted_hops_falls_back_to_the_peer(self) -> None:
        assert resolve_client_address(FRONTEND, [FRONTEND], TRUSTED) == FRONTEND

    def test_no_forwarding_header_from_the_proxy_falls_back_to_the_peer(self) -> None:
        assert resolve_client_address(FRONTEND, [], TRUSTED) == FRONTEND

    @pytest.mark.parametrize(
        "forwarded",
        [
            "not-an-address",
            "",
            f"{CLIENT}:443",
            f"[{CLIENT}]",
            "[2001:db8::1]",
            "fe80::1%eth0",
            f"{CLIENT},",
            f"{CLIENT}, ,",
            "unknown",
            "١٢٣.0.0.1",
            "1.2.3.4\x00",
            "999.1.1.1",
        ],
    )
    def test_an_unparseable_entry_stops_the_walk_at_the_peer(self, forwarded: str) -> None:
        """A malformed chain is not evidence; it earns the proxy's shared identity."""
        assert resolve_client_address(FRONTEND, [forwarded], TRUSTED) == FRONTEND

    def test_garbage_left_of_the_answer_is_never_read(self) -> None:
        """Only the rightmost untrusted entry matters; what precedes it is the client's own."""
        assert (
            resolve_client_address(FRONTEND, [f"garbage, , {FORGED}x, {CLIENT}"], TRUSTED) == CLIENT
        )

    def test_a_chain_past_the_hop_limit_is_not_walked(self) -> None:
        hops = ", ".join([CLIENT] * (MAX_FORWARDED_HOPS + 1))
        assert resolve_client_address(FRONTEND, [hops], TRUSTED) == FRONTEND

    def test_a_chain_at_the_hop_limit_is_walked(self) -> None:
        hops = ", ".join([FORGED] * (MAX_FORWARDED_HOPS - 1) + [CLIENT])
        assert resolve_client_address(FRONTEND, [hops], TRUSTED) == CLIENT

    def test_an_ipv4_mapped_peer_is_recognised_as_the_proxy(self) -> None:
        assert resolve_client_address(f"::ffff:{FRONTEND}", [CLIENT], TRUSTED) == CLIENT

    def test_an_ipv4_mapped_entry_is_reported_as_plain_ipv4(self) -> None:
        assert resolve_client_address(FRONTEND, [f"::ffff:{CLIENT}"], TRUSTED) == CLIENT

    def test_an_ipv6_client_is_returned_in_canonical_form(self) -> None:
        assert resolve_client_address(FRONTEND, ["2001:0DB8:0000::0001"], TRUSTED) == "2001:db8::1"

    def test_a_peer_that_is_not_an_address_is_returned_unchanged(self) -> None:
        assert resolve_client_address("testclient", [CLIENT], TRUSTED) == "testclient"

    def test_no_peer_at_all_resolves_to_nothing(self) -> None:
        assert resolve_client_address(None, [CLIENT], TRUSTED) is None

    def test_nothing_is_trusted_by_default(self) -> None:
        assert resolve_client_address(FRONTEND, [CLIENT], frozenset()) == FRONTEND


class TestParseAddress:
    @pytest.mark.parametrize(
        "value", ["", " ", "localhost", "1.2.3", "1.2.3.4.5", "::g", "1.2.3.4/32", "fe80::1%1"]
    )
    def test_anything_but_one_plain_address_is_none(self, value: str) -> None:
        assert parse_address(value) is None

    def test_a_mapped_address_folds_to_ipv4(self) -> None:
        assert parse_address("::ffff:203.0.113.7") == ip_address("203.0.113.7")


# --- The application, with the frontend as the connection peer ---------------


async def _app_client(
    settings: WebSettings, discord_state: FakeDiscordState, peer: str
) -> AsyncIterator[httpx.AsyncClient]:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=create_fake_discord(discord_state)),
        base_url="https://discord.test",
    ) as discord_http:
        app = build_app(settings, discord_http)
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app, client=(peer, 40000)),
                base_url="https://testserver",
            ) as client,
        ):
            yield client


@pytest.fixture
def proxied_settings(web_settings: WebSettings) -> WebSettings:
    """The suite's settings, trusting the frontend's pinned address, with a small global bucket."""
    return web_settings.model_copy(
        update={
            "trusted_proxy_addresses": FRONTEND,
            "rate_limit_global_burst": 3,
            "rate_limit_global_per_minute": 1.0,
        }
    )


@pytest_asyncio.fixture
async def via_frontend(
    proxied_settings: WebSettings, discord_state: FakeDiscordState
) -> AsyncIterator[httpx.AsyncClient]:
    """Requests arriving from the frontend container, as every proxied request does."""
    async for client in _app_client(proxied_settings, discord_state, FRONTEND):
        yield client


@pytest_asyncio.fixture
async def via_gateway(
    proxied_settings: WebSettings, discord_state: FakeDiscordState
) -> AsyncIterator[httpx.AsyncClient]:
    """Requests arriving from the Docker gateway: any process on the host."""
    async for client in _app_client(proxied_settings, discord_state, GATEWAY):
        yield client


async def _exhaust(client: httpx.AsyncClient, headers: dict[str, str]) -> int:
    """Send requests under one identity until the global bucket refuses; return how many passed."""
    for sent in range(10):
        if (await client.get("/api/health", headers=headers)).status_code == 429:
            return sent
    return 10


class TestTheChainThroughTheApplication:
    async def test_each_client_behind_the_frontend_has_its_own_limit(
        self, via_frontend: httpx.AsyncClient
    ) -> None:
        assert await _exhaust(via_frontend, {"X-Forwarded-For": CLIENT}) == 3
        # A different client behind the same proxy is untouched by the first.
        response = await via_frontend.get("/api/health", headers={"X-Forwarded-For": OTHER_CLIENT})
        assert response.status_code == 200

    async def test_a_client_cannot_pick_a_fresh_identity_by_prepending_entries(
        self, via_frontend: httpx.AsyncClient
    ) -> None:
        """Caddy strips this in production; the rule holds even if it did not."""
        assert await _exhaust(via_frontend, {"X-Forwarded-For": CLIENT}) == 3
        for forged in (FORGED, OTHER_CLIENT, "10.0.0.1"):
            response = await via_frontend.get(
                "/api/health", headers={"X-Forwarded-For": f"{forged}, {CLIENT}"}
            )
            assert response.status_code == 429

    @pytest.mark.parametrize("header", sorted(name.decode() for name in IGNORED_FORWARDING_HEADERS))
    async def test_forwarded_and_x_real_ip_never_choose_the_identity(
        self, via_frontend: httpx.AsyncClient, header: str
    ) -> None:
        """Caddy passes both through as the client sent them."""
        value = f"for={FORGED}" if header == "forwarded" else FORGED
        assert await _exhaust(via_frontend, {"X-Forwarded-For": CLIENT}) == 3
        response = await via_frontend.get(
            "/api/health", headers={"X-Forwarded-For": CLIENT, header: value}
        )
        assert response.status_code == 429

    async def test_an_untrusted_peer_s_forwarded_for_counts_for_nothing(
        self, via_gateway: httpx.AsyncClient
    ) -> None:
        """Every request from the host shares the host's one identity, whatever it claims."""
        sent = 0
        for index in range(10):
            response = await via_gateway.get(
                "/api/health", headers={"X-Forwarded-For": f"192.0.2.{index + 1}"}
            )
            if response.status_code == 429:
                break
            sent += 1
        assert sent == 3

    async def test_a_malformed_chain_from_the_proxy_shares_the_proxy_s_identity(
        self, via_frontend: httpx.AsyncClient
    ) -> None:
        assert await _exhaust(via_frontend, {"X-Forwarded-For": "garbage"}) == 3
        # Another malformed spelling lands in the same, already empty, bucket.
        response = await via_frontend.get("/api/health", headers={"X-Forwarded-For": "nonsense"})
        assert response.status_code == 429
        # A well-formed client is unaffected.
        response = await via_frontend.get("/api/health", headers={"X-Forwarded-For": CLIENT})
        assert response.status_code == 200

    async def test_the_resolved_client_is_what_the_log_names(
        self, via_frontend: httpx.AsyncClient, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level(logging.WARNING, logger="aura_web.rate_limit")
        await _exhaust(via_frontend, {"X-Forwarded-For": f"{FORGED}, {CLIENT}"})
        refusals = [
            r.getMessage() for r in caplog.records if "Rate limit reached" in r.getMessage()
        ]
        assert len(refusals) == 1
        assert CLIENT in refusals[0]
        assert FORGED not in refusals[0]
        assert FRONTEND not in refusals[0]


# --- The real server: uvicorn must not apply a rule of its own -----------------


def _free_port() -> int:
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as probe:
        probe.bind(("127.0.0.1", 0))
        port: int = probe.getsockname()[1]
        return port


@contextlib.asynccontextmanager
async def _serve(settings: WebSettings, discord_state: FakeDiscordState) -> AsyncIterator[str]:
    """Serve the application with real uvicorn, configured exactly as aura_web.__main__ does."""
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=create_fake_discord(discord_state)),
        base_url="https://discord.test",
    ) as discord_http:
        app = build_app(settings, discord_http)
        port = _free_port()
        server = uvicorn.Server(uvicorn_config(app, host="127.0.0.1", port=port))
        task = asyncio.create_task(server.serve())
        try:
            for _ in range(500):
                if server.started:
                    break
                await asyncio.sleep(0.01)
            assert server.started
            yield f"http://127.0.0.1:{port}"
        finally:
            server.should_exit = True
            await task


def _small_global_bucket(settings: WebSettings, **update: object) -> WebSettings:
    return settings.model_copy(
        update={"rate_limit_global_burst": 3, "rate_limit_global_per_minute": 1.0, **update}
    )


class TestTheRealServer:
    def test_uvicorn_s_own_proxy_header_handling_is_off(self) -> None:
        config = uvicorn_config(object(), host="127.0.0.1", port=1)  # type: ignore[arg-type]
        assert config.proxy_headers is False

    async def test_a_forged_forwarded_for_from_loopback_is_not_believed(
        self,
        web_settings: WebSettings,
        discord_state: FakeDiscordState,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """127.0.0.1 is the one peer uvicorn's own handling would trust; it must not."""
        caplog.set_level(logging.INFO)
        statuses = []
        async with (
            _serve(_small_global_bucket(web_settings), discord_state) as base_url,
            httpx.AsyncClient(base_url=base_url) as client,
        ):
            for index in range(6):
                response = await client.get(
                    "/api/health", headers={"X-Forwarded-For": f"192.0.2.{index + 1}"}
                )
                statuses.append(response.status_code)
        # One shared identity: three pass, the rest are refused.
        assert statuses == [200, 200, 200, 429, 429, 429]
        access = [r.getMessage() for r in caplog.records if r.name == "uvicorn.access"]
        assert access, "uvicorn's access log must stay on"
        assert all(line.startswith("127.0.0.1:") for line in access)
        assert not any("192.0.2." in line for line in access)

    async def test_the_access_log_names_the_client_the_rule_resolved(
        self,
        web_settings: WebSettings,
        discord_state: FakeDiscordState,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """With the peer trusted, the rewritten address is what uvicorn logs."""
        caplog.set_level(logging.INFO)
        settings = _small_global_bucket(web_settings, trusted_proxy_addresses="127.0.0.1")
        async with (
            _serve(settings, discord_state) as base_url,
            httpx.AsyncClient(base_url=base_url) as client,
        ):
            response = await client.get(
                "/api/health", headers={"X-Forwarded-For": f"{FORGED}, {CLIENT}"}
            )
        assert response.status_code == 200
        access = [r.getMessage() for r in caplog.records if r.name == "uvicorn.access"]
        assert len(access) == 1
        assert access[0].startswith(f"{CLIENT}:0 ")
