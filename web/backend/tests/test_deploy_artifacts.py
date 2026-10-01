"""What the deployment files promise, checked without deploying anything.

web/docker-compose.yml and web/deploy/Caddyfile.aura decide what the internet
can reach and which address the backend believes a request came from. Neither
is executed by the suite, so the promises that matter are read out of the files
themselves: what is published and where, which address is trusted and whether
it is really the frontend's, what Caddy forwards and refuses, and that the block
cannot change how any other site on the same Caddy behaves.
"""

from __future__ import annotations

import re
from ipaddress import IPv4Address, IPv4Network, ip_address
from pathlib import Path
from typing import Any, Final

import pytest
import yaml

from aura_web.config import parse_trusted_proxy_addresses
from aura_web.routes.stripe_webhook import MAX_WEBHOOK_BODY_BYTES

WEB: Final = Path(__file__).resolve().parents[2]
COMPOSE_FILE: Final = WEB / "docker-compose.yml"
CADDY_BLOCK: Final = WEB / "deploy" / "Caddyfile.aura"
FRONTEND_DOCKERFILE: Final = WEB / "frontend" / "Dockerfile"

SITE_ADDRESS: Final = "aura.timurmanjosov.com"
PUBLISHED_FRONTEND_PORT: Final = re.compile(
    r"^127\.0\.0\.1:\$\{AURA_WEB_FRONTEND_PORT:-(?P<default>\d+)\}:3000$"
)
CADDY_SIZE_UNITS: Final = {"KiB": 1024, "MiB": 1024**2}
IPV4_LITERAL: Final = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")
IPV6_LITERAL: Final = re.compile(r"\[?[0-9A-Fa-f]{0,4}(?::[0-9A-Fa-f]{0,4}){2,7}\]?")


@pytest.fixture(scope="module")
def compose() -> dict[str, Any]:
    loaded = yaml.safe_load(COMPOSE_FILE.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


@pytest.fixture(scope="module")
def caddy_lines() -> list[str]:
    """The block's directives: comments and blank lines removed, whitespace collapsed."""
    lines = []
    for raw in CADDY_BLOCK.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            lines.append(" ".join(line.split()))
    return lines


def service(compose: dict[str, Any], name: str) -> dict[str, Any]:
    entry = compose["services"][name]
    assert isinstance(entry, dict)
    return entry


def pinned_address(compose: dict[str, Any], name: str) -> IPv4Address:
    address = ip_address(service(compose, name)["networks"]["aura-web"]["ipv4_address"])
    assert isinstance(address, IPv4Address)
    return address


def frontend_port_default(compose: dict[str, Any]) -> int:
    (published,) = service(compose, "frontend")["ports"]
    match = PUBLISHED_FRONTEND_PORT.match(published)
    assert match is not None, published
    return int(match["default"])


class TestWhatIsPublished:
    def test_the_frontend_is_published_on_loopback_only(self, compose: dict[str, Any]) -> None:
        ports = service(compose, "frontend")["ports"]
        assert len(ports) == 1
        assert PUBLISHED_FRONTEND_PORT.match(ports[0]), ports[0]

    def test_the_backend_publishes_nothing(self, compose: dict[str, Any]) -> None:
        assert "ports" not in service(compose, "backend")

    def test_no_service_escapes_its_network_or_holds_the_bot_s_data(
        self, compose: dict[str, Any]
    ) -> None:
        for name, entry in compose["services"].items():
            assert "network_mode" not in entry, name
            assert not entry.get("privileged", False), name
            assert "volumes" not in entry, name

    def test_every_log_is_bounded(self, compose: dict[str, Any]) -> None:
        for name, entry in compose["services"].items():
            logging_options = entry["logging"]
            assert logging_options["driver"] == "json-file", name
            assert logging_options["options"]["max-size"], name
            assert int(logging_options["options"]["max-file"]) >= 1, name


class TestTheTrustedProxyIsTheFrontend:
    def test_the_backend_trusts_exactly_the_frontend_s_pinned_address(
        self, compose: dict[str, Any]
    ) -> None:
        configured = service(compose, "backend")["environment"]["AURA_WEB_TRUSTED_PROXY_ADDRESSES"]
        assert parse_trusted_proxy_addresses(configured) == frozenset(
            {pinned_address(compose, "frontend")}
        )

    def test_both_pins_are_ordinary_hosts_of_the_pinned_subnet(
        self, compose: dict[str, Any]
    ) -> None:
        (ipam,) = compose["networks"]["aura-web"]["ipam"]["config"]
        subnet = IPv4Network(ipam["subnet"])
        assert subnet.is_private
        gateway = next(iter(subnet.hosts()))
        frontend = pinned_address(compose, "frontend")
        backend = pinned_address(compose, "backend")
        for address in (frontend, backend):
            assert address in subnet
            assert address not in (subnet.network_address, subnet.broadcast_address, gateway)
        assert frontend != backend

    def test_the_subnet_stays_outside_docker_s_automatic_pools(
        self, compose: dict[str, Any]
    ) -> None:
        """So no network Docker creates on its own can ever be handed an overlapping range."""
        (ipam,) = compose["networks"]["aura-web"]["ipam"]["config"]
        subnet = IPv4Network(ipam["subnet"])
        automatic_pools = [IPv4Network(f"172.{second}.0.0/16") for second in range(17, 32)]
        automatic_pools.append(IPv4Network("192.168.0.0/16"))
        assert not any(subnet.overlaps(pool) for pool in automatic_pools)


class TestTheFrontendImage:
    def test_it_runs_the_production_server(self) -> None:
        dockerfile = FRONTEND_DOCKERFILE.read_text(encoding="utf-8")
        runtime_stage = dockerfile.rsplit("FROM ", 1)[1]
        assert "NODE_ENV=production" in runtime_stage
        assert 'CMD ["node", "server.js"]' in runtime_stage
        assert "next dev" not in dockerfile
        assert "npm run dev" not in dockerfile


class TestTheCaddyBlock:
    def test_it_is_one_site_block_for_one_hostname_and_nothing_else(
        self, caddy_lines: list[str]
    ) -> None:
        """No global options block, no snippet, no second site: nothing that reaches other sites."""
        assert caddy_lines[0] == f"{SITE_ADDRESS} {{"
        depth = 0
        top_level_openings = 0
        for line in caddy_lines:
            if depth == 0 and line.endswith("{"):
                top_level_openings += 1
            depth += line.count("{") - line.count("}")
            assert depth >= 0
        assert depth == 0
        assert top_level_openings == 1

    @pytest.mark.parametrize("directive", ["import", "trusted_proxies", "tls ", "servers", "bind"])
    def test_it_uses_nothing_that_configures_beyond_this_site(
        self, caddy_lines: list[str], directive: str
    ) -> None:
        assert not any(line.startswith(directive) for line in caddy_lines)

    def test_it_proxies_to_the_frontend_s_published_port_by_hostname(
        self, caddy_lines: list[str], compose: dict[str, Any]
    ) -> None:
        proxies = [line for line in caddy_lines if line.startswith("reverse_proxy")]
        assert proxies == [f"reverse_proxy localhost:{frontend_port_default(compose)}"]

    def test_no_address_literal_appears_anywhere(self) -> None:
        text = CADDY_BLOCK.read_text(encoding="utf-8")
        assert not IPV4_LITERAL.search(text)
        assert not [match for match in IPV6_LITERAL.findall(text) if match.count(":") >= 2]

    def test_its_body_ceiling_is_the_backend_s_largest_limit(self, caddy_lines: list[str]) -> None:
        (line,) = [line for line in caddy_lines if line.startswith("max_size ")]
        match = re.fullmatch(r"max_size (\d+)(KiB|MiB)", line)
        assert match is not None, line
        assert int(match[1]) * CADDY_SIZE_UNITS[match[2]] == MAX_WEBHOOK_BODY_BYTES

    def test_hsts_never_speaks_for_another_hostname(self, caddy_lines: list[str]) -> None:
        (line,) = [line for line in caddy_lines if line.startswith("Strict-Transport-Security")]
        assert "includesubdomains" not in line.lower()
        assert "preload" not in line.lower()
        assert re.search(r"max-age=\d+", line)

    def test_the_site_is_not_indexed(self, caddy_lines: list[str]) -> None:
        assert 'X-Robots-Tag "noindex"' in caddy_lines

    def test_header_rules_never_override_what_the_application_chose(
        self, caddy_lines: list[str]
    ) -> None:
        assert "defer" in caddy_lines
        for name in ("X-Content-Type-Options", "Referrer-Policy", "X-Frame-Options"):
            assert any(line.startswith(f"?{name} ") for line in caddy_lines), name

    def test_the_access_log_drops_the_signature_and_the_oauth_parameters(
        self, caddy_lines: list[str]
    ) -> None:
        assert "request>headers>Stripe-Signature delete" in caddy_lines
        assert "delete code" in caddy_lines
        assert "delete state" in caddy_lines
