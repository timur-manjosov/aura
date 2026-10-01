"""Per-client request-rate limits: every bucket, its boundary, its memory, and its abuse.

Two layers. The limiter itself is a pure object on a fake clock, so a boundary
is a number rather than a sleep. The application tests then drive the real
service -- the production middleware order, the production routes -- with that
same fake clock injected, and check what a client actually receives: the 429,
its Retry-After, its body, that the route behind it was never reached, and
that nothing it should not was read or logged.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

import httpx
import pytest
import pytest_asyncio
from pydantic import ValidationError
from starlette.requests import Request

from aura_web.config import (
    MAX_RATE_LIMIT_BURST,
    MAX_RATE_LIMIT_PER_MINUTE,
    MAX_RATE_LIMIT_TRACKED_CLIENTS,
    MAX_TRUSTED_PROXY_ADDRESSES,
    WebSettings,
    parse_trusted_proxy_addresses,
)
from aura_web.rate_limit import (
    WEBHOOK_REFUND_STATE_KEY,
    BucketPolicy,
    RateLimitBucket,
    RateLimiter,
    classify_path,
    client_key,
    rate_limiter_from_settings,
    refund_webhook_allowance,
)
from fake_bot_billing import FakeBotBillingState, create_fake_bot_billing
from fake_discord import FakeDiscordState, create_fake_discord
from fake_stripe import FakeStripeState, create_fake_stripe, sign_webhook
from helpers import FAKE_BOT_BASE, FAKE_STRIPE_BASE, FRONTEND_BASE, build_app, complete_login

WEB_ENV_EXAMPLE: Final = Path(__file__).resolve().parents[2] / ".env.example"
WEBHOOK: Final = "/api/stripe/webhook"
CLIENT: Final = "203.0.113.7"
ATTACKER: Final = "198.51.100.66"
STRIPE: Final = "192.0.2.15"


@dataclass
class FakeClock:
    """A monotonic clock that moves only when told to."""

    now: float = 1_000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def limiter(
    clock: FakeClock,
    *,
    burst: int = 2,
    per_minute: float = 6.0,
    max_tracked_clients: int = 100,
) -> RateLimiter:
    """A limiter with one policy for every bucket; 6 a minute is one token every 10 s."""
    policy = BucketPolicy(burst=burst, per_minute=per_minute)
    return RateLimiter(
        dict.fromkeys(RateLimitBucket, policy),
        max_tracked_clients=max_tracked_clients,
        clock=clock,
    )


# --- The limiter ---------------------------------------------------------------


class TestTheTokenBucket:
    def test_exactly_the_burst_passes_and_the_next_request_is_refused(self) -> None:
        clock = FakeClock()
        limits = limiter(clock, burst=3)
        decisions = [limits.acquire(RateLimitBucket.AUTH, CLIENT) for _ in range(4)]
        assert [d.allowed for d in decisions] == [True, True, True, False]

    def test_retry_after_is_the_wait_for_one_token(self) -> None:
        clock = FakeClock()
        limits = limiter(clock)
        limits.acquire(RateLimitBucket.AUTH, CLIENT)
        limits.acquire(RateLimitBucket.AUTH, CLIENT)
        assert limits.acquire(RateLimitBucket.AUTH, CLIENT).retry_after_seconds == 10
        clock.advance(4)
        assert limits.acquire(RateLimitBucket.AUTH, CLIENT).retry_after_seconds == 6

    def test_one_token_is_back_exactly_at_the_refill_interval_and_not_before(self) -> None:
        clock = FakeClock()
        limits = limiter(clock)
        limits.acquire(RateLimitBucket.AUTH, CLIENT)
        limits.acquire(RateLimitBucket.AUTH, CLIENT)
        clock.advance(9.999)
        assert not limits.acquire(RateLimitBucket.AUTH, CLIENT).allowed
        clock.advance(0.001)
        assert limits.acquire(RateLimitBucket.AUTH, CLIENT).allowed
        assert not limits.acquire(RateLimitBucket.AUTH, CLIENT).allowed

    def test_retry_after_is_never_below_one_second(self) -> None:
        clock = FakeClock()
        limits = limiter(clock, burst=1, per_minute=6_000)
        limits.acquire(RateLimitBucket.AUTH, CLIENT)
        assert limits.acquire(RateLimitBucket.AUTH, CLIENT).retry_after_seconds == 1

    def test_a_long_idle_refills_to_the_burst_and_no_further(self) -> None:
        clock = FakeClock()
        limits = limiter(clock, burst=3)
        for _ in range(3):
            limits.acquire(RateLimitBucket.AUTH, CLIENT)
        clock.advance(24 * 3600)
        decisions = [limits.acquire(RateLimitBucket.AUTH, CLIENT) for _ in range(4)]
        assert [d.allowed for d in decisions] == [True, True, True, False]

    def test_a_refused_request_does_not_push_the_wait_further_out(self) -> None:
        clock = FakeClock()
        limits = limiter(clock)
        limits.acquire(RateLimitBucket.AUTH, CLIENT)
        limits.acquire(RateLimitBucket.AUTH, CLIENT)
        for _ in range(50):
            limits.acquire(RateLimitBucket.AUTH, CLIENT)
        clock.advance(10)
        assert limits.acquire(RateLimitBucket.AUTH, CLIENT).allowed

    def test_a_clock_stepping_backwards_refills_nothing(self) -> None:
        clock = FakeClock()
        limits = limiter(clock)
        limits.acquire(RateLimitBucket.AUTH, CLIENT)
        limits.acquire(RateLimitBucket.AUTH, CLIENT)
        clock.advance(-3600)
        assert not limits.acquire(RateLimitBucket.AUTH, CLIENT).allowed
        clock.advance(3600 + 10)
        assert limits.acquire(RateLimitBucket.AUTH, CLIENT).allowed

    def test_clients_and_buckets_are_independent(self) -> None:
        clock = FakeClock()
        limits = limiter(clock, burst=1)
        assert limits.acquire(RateLimitBucket.AUTH, CLIENT).allowed
        assert not limits.acquire(RateLimitBucket.AUTH, CLIENT).allowed
        assert limits.acquire(RateLimitBucket.AUTH, ATTACKER).allowed
        for bucket in (RateLimitBucket.BILLING, RateLimitBucket.WEBHOOK, RateLimitBucket.GLOBAL):
            assert limits.acquire(bucket, CLIENT).allowed

    def test_a_refund_returns_one_token_and_never_exceeds_the_burst(self) -> None:
        clock = FakeClock()
        limits = limiter(clock, burst=2)
        limits.acquire(RateLimitBucket.WEBHOOK, CLIENT)
        for _ in range(10):
            limits.refund(RateLimitBucket.WEBHOOK, CLIENT)
        decisions = [limits.acquire(RateLimitBucket.WEBHOOK, CLIENT) for _ in range(3)]
        assert [d.allowed for d in decisions] == [True, True, False]

    def test_a_refund_for_a_client_never_seen_creates_nothing(self) -> None:
        limits = limiter(FakeClock())
        limits.refund(RateLimitBucket.WEBHOOK, CLIENT)
        assert limits.tracked_clients(RateLimitBucket.WEBHOOK) == 0

    def test_a_refusal_episode_is_reported_once_at_its_start_and_once_at_its_end(self) -> None:
        clock = FakeClock()
        limits = limiter(clock, burst=1)
        limits.acquire(RateLimitBucket.AUTH, CLIENT)
        refusals = [limits.acquire(RateLimitBucket.AUTH, CLIENT) for _ in range(5)]
        assert [d.refusal_episode_started for d in refusals] == [True, False, False, False, False]
        clock.advance(10)
        lifted = limits.acquire(RateLimitBucket.AUTH, CLIENT)
        assert lifted.allowed
        assert lifted.refusals_in_ended_episode == 5
        clock.advance(10)
        assert limits.acquire(RateLimitBucket.AUTH, CLIENT).refusals_in_ended_episode == 0


class TestTheWebhookRefundHandle:
    def test_it_hands_back_at_most_one_token_however_often_it_is_called(self) -> None:
        calls: list[None] = []
        request = Request(
            {
                "type": "http",
                "headers": [],
                "state": {WEBHOOK_REFUND_STATE_KEY: lambda: calls.append(None)},
            }
        )
        for _ in range(5):
            refund_webhook_allowance(request)
        assert len(calls) == 1

    def test_a_request_the_limiter_never_saw_has_nothing_to_hand_back(self) -> None:
        refund_webhook_allowance(Request({"type": "http", "headers": []}))
        refund_webhook_allowance(Request({"type": "http", "headers": [], "state": {}}))


class TestBoundedMemory:
    def test_a_bucket_never_remembers_more_than_its_bound(self) -> None:
        limits = limiter(FakeClock(), max_tracked_clients=50)
        for index in range(5_000):
            limits.acquire(RateLimitBucket.GLOBAL, f"10.0.{index // 256}.{index % 256}")
        assert limits.tracked_clients(RateLimitBucket.GLOBAL) == 50

    def test_the_least_recently_seen_client_is_forgotten_first(self) -> None:
        limits = limiter(FakeClock(), burst=1, max_tracked_clients=2)
        limits.acquire(RateLimitBucket.GLOBAL, "a")
        limits.acquire(RateLimitBucket.GLOBAL, "b")
        # "a" is used again (refused), so "b" is now the least recent.
        assert not limits.acquire(RateLimitBucket.GLOBAL, "a").allowed
        limits.acquire(RateLimitBucket.GLOBAL, "c")
        assert not limits.acquire(RateLimitBucket.GLOBAL, "a").allowed
        assert limits.acquire(RateLimitBucket.GLOBAL, "b").allowed

    def test_each_bucket_is_bounded_on_its_own(self) -> None:
        limits = limiter(FakeClock(), max_tracked_clients=3)
        for index in range(10):
            limits.acquire(RateLimitBucket.WEBHOOK, f"w{index}")
        limits.acquire(RateLimitBucket.AUTH, "a")
        assert limits.tracked_clients(RateLimitBucket.WEBHOOK) == 3
        assert limits.tracked_clients(RateLimitBucket.AUTH) == 1


class TestTheLimiterRefusesNonsense:
    def test_every_bucket_needs_a_policy(self) -> None:
        with pytest.raises(ValueError, match="no rate-limit policy"):
            RateLimiter(
                {RateLimitBucket.AUTH: BucketPolicy(burst=1, per_minute=1)},
                max_tracked_clients=1,
            )

    @pytest.mark.parametrize(("burst", "per_minute"), [(0, 1.0), (1, 0.0), (1, -1.0)])
    def test_a_policy_must_be_positive(self, burst: int, per_minute: float) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            RateLimiter(
                dict.fromkeys(RateLimitBucket, BucketPolicy(burst=burst, per_minute=per_minute)),
                max_tracked_clients=1,
            )

    def test_the_memory_bound_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="max_tracked_clients"):
            limiter(FakeClock(), max_tracked_clients=0)


class TestClientKeys:
    def test_an_ipv6_client_is_one_identity_across_its_slash_64(self) -> None:
        assert client_key("2001:db8:1:2::1") == client_key("2001:db8:1:2:ffff:ffff:ffff:ffff")

    def test_neighbouring_ipv6_slash_64s_are_different_clients(self) -> None:
        assert client_key("2001:db8:1:2::1") != client_key("2001:db8:1:3::1")

    def test_an_ipv4_mapped_address_is_the_ipv4_client(self) -> None:
        assert client_key("::ffff:203.0.113.7") == client_key("203.0.113.7") == "203.0.113.7"

    def test_no_address_is_one_shared_identity(self) -> None:
        assert client_key(None) == "unknown"

    def test_a_non_address_peer_is_cut_short(self) -> None:
        assert len(client_key("x" * 10_000)) == 64


class TestPathClassification:
    @pytest.mark.parametrize(
        ("path", "bucket"),
        [
            ("/api/auth/login", RateLimitBucket.AUTH),
            ("/api/auth/callback", RateLimitBucket.AUTH),
            ("/api/auth/logout", RateLimitBucket.AUTH),
            ("/api/auth/anything-else", RateLimitBucket.AUTH),
            ("/api/billing/checkout", RateLimitBucket.BILLING),
            ("/api/billing/portal", RateLimitBucket.BILLING),
            ("/api/billing/guilds", RateLimitBucket.GLOBAL),
            ("/api/billing/checkout/", RateLimitBucket.GLOBAL),
            ("/api/stripe/webhook", RateLimitBucket.WEBHOOK),
            ("/api/stripe/webhook/", RateLimitBucket.WEBHOOK),
            ("/api/me", RateLimitBucket.GLOBAL),
            ("/api/guilds", RateLimitBucket.GLOBAL),
            ("/api/health", RateLimitBucket.GLOBAL),
            ("/", RateLimitBucket.GLOBAL),
            ("/API/auth/login", RateLimitBucket.GLOBAL),
            ("//api/auth/login", RateLimitBucket.GLOBAL),
        ],
    )
    def test_each_path_has_its_bucket(self, path: str, bucket: RateLimitBucket) -> None:
        assert classify_path(path) is bucket


# --- Configuration ---------------------------------------------------------------

RATE_LIMIT_FIELDS: Final = sorted(
    name for name in WebSettings.model_fields if name.startswith("rate_limit_")
)


def documented_defaults() -> dict[str, str]:
    """Every `# AURA_WEB_RATE_LIMIT_...=value` line of web/.env.example."""
    pattern = re.compile(r"^#\s*AURA_WEB_(RATE_LIMIT_[A-Z_]+)=(\S+)$", re.MULTILINE)
    return {
        name.lower(): value
        for name, value in pattern.findall(WEB_ENV_EXAMPLE.read_text(encoding="utf-8"))
    }


class TestConfiguration:
    def test_the_documented_defaults_are_the_real_defaults(self) -> None:
        documented = documented_defaults()
        assert sorted(documented) == RATE_LIMIT_FIELDS
        for name in RATE_LIMIT_FIELDS:
            default = WebSettings.model_fields[name].default
            assert type(default)(documented[name]) == default, name

    @pytest.mark.parametrize("name", RATE_LIMIT_FIELDS)
    @pytest.mark.parametrize("value", [0, -1])
    def test_a_limit_that_would_switch_itself_off_is_refused(
        self, web_settings: WebSettings, name: str, value: int
    ) -> None:
        with pytest.raises(ValidationError):
            WebSettings.model_validate({**web_settings.model_dump(), name: value})

    @pytest.mark.parametrize(
        ("name", "value"),
        [
            ("rate_limit_auth_burst", MAX_RATE_LIMIT_BURST + 1),
            ("rate_limit_global_per_minute", MAX_RATE_LIMIT_PER_MINUTE + 1),
            ("rate_limit_max_tracked_clients", MAX_RATE_LIMIT_TRACKED_CLIENTS + 1),
        ],
    )
    def test_an_absurdly_large_limit_is_refused(
        self, web_settings: WebSettings, name: str, value: float
    ) -> None:
        with pytest.raises(ValidationError):
            WebSettings.model_validate({**web_settings.model_dump(), name: value})

    def test_the_settings_build_the_limiter_they_describe(self, web_settings: WebSettings) -> None:
        settings = web_settings.model_copy(
            update={"rate_limit_billing_burst": 4, "rate_limit_billing_per_minute": 3.0}
        )
        policy = rate_limiter_from_settings(settings).policy(RateLimitBucket.BILLING)
        assert policy == BucketPolicy(burst=4, per_minute=3.0)


class TestTrustedProxyAddresses:
    def test_blank_trusts_nothing(self) -> None:
        assert parse_trusted_proxy_addresses("") == frozenset()
        assert parse_trusted_proxy_addresses(" , ") == frozenset()

    def test_addresses_are_parsed_and_mapped_forms_folded(self) -> None:
        parsed = parse_trusted_proxy_addresses(" 172.16.86.3 , ::ffff:10.0.0.1, 2001:db8::1 ")
        assert {str(address) for address in parsed} == {"172.16.86.3", "10.0.0.1", "2001:db8::1"}

    @pytest.mark.parametrize(
        "value",
        [
            "frontend",
            "172.16.86.0/28",
            "172.16.86.3:3000",
            "0.0.0.0",
            "::",
            "172.16.86.3; 10.0.0.1",
            "*",
        ],
    )
    def test_anything_but_single_addresses_is_refused(self, value: str) -> None:
        with pytest.raises(ValueError, match="TRUSTED_PROXY_ADDRESSES"):
            parse_trusted_proxy_addresses(value)

    def test_too_many_addresses_are_refused(self) -> None:
        value = ",".join(f"10.0.0.{index}" for index in range(1, MAX_TRUSTED_PROXY_ADDRESSES + 2))
        with pytest.raises(ValueError, match="at most"):
            parse_trusted_proxy_addresses(value)

    def test_a_bad_value_refuses_the_whole_configuration(self, web_settings: WebSettings) -> None:
        with pytest.raises(ValidationError):
            WebSettings.model_validate(
                {**web_settings.model_dump(), "trusted_proxy_addresses": "10.0.0.0/8"}
            )

    def test_the_default_trusts_no_proxy(self, web_settings: WebSettings) -> None:
        assert web_settings.trusted_proxies == frozenset()


# --- The running service -----------------------------------------------------------


@dataclass
class Service:
    """The application on a fake clock, with one HTTP client per connection peer."""

    app: object
    clock: FakeClock
    limiter: RateLimiter
    clients: dict[str, httpx.AsyncClient] = field(default_factory=dict)


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def production_limits(web_settings: WebSettings) -> WebSettings:
    """The suite's settings with every rate-limit setting at its production default."""
    defaults = {name: WebSettings.model_fields[name].default for name in RATE_LIMIT_FIELDS}
    return web_settings.model_copy(update=defaults)


@pytest_asyncio.fixture
async def service(
    production_limits: WebSettings,
    clock: FakeClock,
    discord_state: FakeDiscordState,
    stripe_state: FakeStripeState,
    bot_billing_state: FakeBotBillingState,
) -> AsyncIterator[Service]:
    """The real application with production limits on a fake clock, reachable from three peers."""
    rate_limiter = rate_limiter_from_settings(production_limits, clock=clock)
    async with (
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=create_fake_discord(discord_state)),
            base_url="https://discord.test",
        ) as discord_http,
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=create_fake_stripe(stripe_state)),
            base_url=FAKE_STRIPE_BASE,
        ) as stripe_http,
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=create_fake_bot_billing(bot_billing_state)),
            base_url=FAKE_BOT_BASE,
        ) as bot_http,
    ):
        app = build_app(
            production_limits, discord_http, stripe_http, bot_http, rate_limiter=rate_limiter
        )
        async with app.router.lifespan_context(app):
            clients = {
                peer: httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app, client=(peer, 40000)),
                    base_url="https://testserver",
                )
                for peer in (CLIENT, ATTACKER, STRIPE)
            }
            try:
                yield Service(app=app, clock=clock, limiter=rate_limiter, clients=clients)
            finally:
                for client in clients.values():
                    await client.aclose()


def defaults(name: str) -> int:
    value = WebSettings.model_fields[name].default
    assert isinstance(value, int)
    return value


AUTH_BURST: Final = defaults("rate_limit_auth_burst")
BILLING_BURST: Final = defaults("rate_limit_billing_burst")
WEBHOOK_BURST: Final = defaults("rate_limit_webhook_burst")
GLOBAL_BURST: Final = defaults("rate_limit_global_burst")


def assert_refused(response: httpx.Response, *, retry_after: int | None = None) -> None:
    """A 429 carrying the documented body and headers, and nothing else of interest."""
    assert response.status_code == 429
    assert response.json() == {"error": "rate_limited"}
    assert int(response.headers["retry-after"]) >= 1
    if retry_after is not None:
        assert response.headers["retry-after"] == str(retry_after)
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-content-type-options"] == "nosniff"
    assert "set-cookie" not in response.headers


@dataclass
class BodyProbe:
    """A request body that records whether anyone read it."""

    read: bool = False

    async def stream(self) -> AsyncIterator[bytes]:
        self.read = True
        yield b'{"guild_id": "1000"}'


class TestTheAuthBucket:
    async def test_the_burst_then_a_429_with_its_retry_after(self, service: Service) -> None:
        client = service.clients[CLIENT]
        for _ in range(AUTH_BURST):
            assert (await client.get("/api/auth/login")).status_code == 307
        # 10 a minute: one token every six seconds.
        assert_refused(await client.get("/api/auth/login"), retry_after=6)

    async def test_a_refused_login_issues_no_state(self, service: Service) -> None:
        client = service.clients[CLIENT]
        for _ in range(AUTH_BURST):
            await client.get("/api/auth/login")
        context = service.app.state.context  # type: ignore[attr-defined]
        issued = len(context.oauth_states)
        for _ in range(50):
            await client.get("/api/auth/login")
        assert len(context.oauth_states) == issued

    async def test_login_callback_and_logout_share_one_bucket(self, service: Service) -> None:
        client = service.clients[CLIENT]
        for index in range(AUTH_BURST):
            path = ("/api/auth/login", "/api/auth/callback")[index % 2]
            await client.get(path)
        assert_refused(await client.post("/api/auth/logout"))

    async def test_the_bucket_refills_with_time(self, service: Service) -> None:
        client = service.clients[CLIENT]
        for _ in range(AUTH_BURST):
            await client.get("/api/auth/login")
        assert (await client.get("/api/auth/login")).status_code == 429
        service.clock.advance(6)
        assert (await client.get("/api/auth/login")).status_code == 307
        assert (await client.get("/api/auth/login")).status_code == 429

    async def test_one_client_s_flood_leaves_another_client_alone(self, service: Service) -> None:
        for _ in range(AUTH_BURST * 3):
            await service.clients[ATTACKER].get("/api/auth/login")
        assert (await service.clients[CLIENT].get("/api/auth/login")).status_code == 307


class TestTheBillingBucket:
    async def test_the_strict_burst_then_a_429_before_the_body_is_read(
        self, service: Service, stripe_state: FakeStripeState
    ) -> None:
        client = service.clients[CLIENT]
        for _ in range(BILLING_BURST):
            response = await client.post(
                "/api/billing/checkout",
                json={"guild_id": "1000"},
                headers={"Origin": FRONTEND_BASE},
            )
            assert response.status_code == 401
        probe = BodyProbe()
        response = await client.post(
            "/api/billing/checkout",
            content=probe.stream(),
            headers={"Content-Type": "application/json", "Origin": FRONTEND_BASE},
        )
        # Two a minute: one token every thirty seconds.
        assert_refused(response, retry_after=30)
        assert probe.read is False
        assert stripe_state.request_log == []

    async def test_a_signed_in_admin_is_refused_before_stripe_is_called(
        self,
        service: Service,
        discord_state: FakeDiscordState,
        stripe_state: FakeStripeState,
    ) -> None:
        client = service.clients[CLIENT]
        await complete_login(client, discord_state, "5000")
        for _ in range(BILLING_BURST):
            await client.post(
                "/api/billing/portal", json={"guild_id": "1000"}, headers={"Origin": FRONTEND_BASE}
            )
        calls_before = len(stripe_state.request_log)
        response = await client.post(
            "/api/billing/checkout", json={"guild_id": "1000"}, headers={"Origin": FRONTEND_BASE}
        )
        assert_refused(response)
        assert len(stripe_state.request_log) == calls_before

    async def test_reading_plans_does_not_spend_the_billing_bucket(self, service: Service) -> None:
        client = service.clients[CLIENT]
        for _ in range(BILLING_BURST * 2):
            assert (await client.get("/api/billing/guilds")).status_code == 401
        response = await client.post(
            "/api/billing/checkout", json={"guild_id": "1000"}, headers={"Origin": FRONTEND_BASE}
        )
        assert response.status_code == 401


def ignorable_event(stripe_state: FakeStripeState) -> tuple[bytes, str]:
    """A genuinely signed delivery for an event that concerns no subscription."""
    return stripe_state.signed(
        stripe_state.event("customer.created", {"id": "cus_rate_limit", "object": "customer"})
    )


async def deliver(
    client: httpx.AsyncClient, body: bytes | AsyncIterator[bytes], signature: str
) -> httpx.Response:
    return await client.post(
        WEBHOOK,
        content=body,
        headers={"Content-Type": "application/json", "Stripe-Signature": signature},
    )


class TestTheWebhookBucket:
    async def test_forgeries_drain_it_then_are_refused_without_the_body_read(
        self, service: Service
    ) -> None:
        client = service.clients[ATTACKER]
        for _ in range(WEBHOOK_BURST):
            response = await deliver(client, b"{}", "t=1,v1=" + "0" * 64)
            assert response.status_code == 400
        probe = BodyProbe()
        assert_refused(await deliver(client, probe.stream(), "t=1,v1=" + "0" * 64))
        assert probe.read is False

    async def test_genuine_deliveries_are_never_refused_however_many(
        self, service: Service, stripe_state: FakeStripeState
    ) -> None:
        """Three times the burst, back to back, with no time passing at all."""
        client = service.clients[STRIPE]
        for _ in range(WEBHOOK_BURST * 3):
            body, signature = ignorable_event(stripe_state)
            response = await deliver(client, body, signature)
            assert response.status_code == 200, response.text

    async def test_a_genuine_subscription_delivery_is_processed_past_the_burst(
        self,
        service: Service,
        stripe_state: FakeStripeState,
        bot_billing_state: FakeBotBillingState,
    ) -> None:
        client = service.clients[STRIPE]
        for _ in range(WEBHOOK_BURST):
            await deliver(client, *ignorable_event(stripe_state))
        subscription = stripe_state.add_subscription(
            guild_id="1000", purchaser_user_id="5000", now=int(service.clock.now)
        )
        body, signature = stripe_state.signed(
            stripe_state.subscription_event("customer.subscription.created", subscription.id)
        )
        response = await deliver(client, body, signature)
        assert response.status_code == 200
        assert subscription.id in bot_billing_state.snapshots

    async def test_a_genuine_delivery_passes_during_another_address_s_flood(
        self, service: Service, stripe_state: FakeStripeState
    ) -> None:
        attacker = service.clients[ATTACKER]
        for _ in range(WEBHOOK_BURST * 2):
            await deliver(attacker, b"{}", "t=1,v1=" + "0" * 64)
        assert (await deliver(attacker, b"{}", "t=1,v1=00")).status_code == 429
        response = await deliver(service.clients[STRIPE], *ignorable_event(stripe_state))
        assert response.status_code == 200

    async def test_genuine_deliveries_interleaved_with_forgeries_from_one_address(
        self, service: Service, stripe_state: FakeStripeState
    ) -> None:
        """Only the forgeries count: half the burst of them leaves every genuine one through."""
        client = service.clients[STRIPE]
        for _ in range(WEBHOOK_BURST // 2):
            assert (await deliver(client, b"{}", "t=1,v1=00")).status_code == 400
            assert (await deliver(client, *ignorable_event(stripe_state))).status_code == 200

    async def test_a_delivery_signed_with_another_secret_counts_as_a_forgery(
        self, service: Service, stripe_state: FakeStripeState
    ) -> None:
        """The relay case: another Stripe account pointing its endpoint at this URL."""
        client = service.clients[ATTACKER]
        foreign_secret = "whsec_" + "x" * 32
        for _ in range(WEBHOOK_BURST):
            body = json.dumps(
                stripe_state.event("customer.created", {"id": "cus_x", "object": "customer"})
            ).encode()
            response = await deliver(client, body, sign_webhook(body, foreign_secret))
            assert response.status_code == 400
        assert (await deliver(client, b"{}", "t=1,v1=00")).status_code == 429


class TestTheGlobalBucket:
    async def test_everything_else_shares_one_ceiling_per_client(self, service: Service) -> None:
        client = service.clients[CLIENT]
        paths = ["/api/me", "/api/guilds", "/api/billing/guilds", "/api/health", "/no-such-route"]
        for index in range(GLOBAL_BURST):
            await client.get(paths[index % len(paths)])
        # Sixty a minute: one token a second.
        assert_refused(await client.get("/api/health"), retry_after=1)

    async def test_the_global_ceiling_leaves_the_other_buckets_alone(
        self, service: Service
    ) -> None:
        client = service.clients[CLIENT]
        for _ in range(GLOBAL_BURST + 5):
            await client.get("/api/me")
        assert (await client.get("/api/auth/login")).status_code == 307


class TestConcurrency:
    async def test_simultaneous_requests_get_exactly_the_burst(self, service: Service) -> None:
        client = service.clients[CLIENT]
        responses = await asyncio.gather(
            *(client.get("/api/health") for _ in range(GLOBAL_BURST + 25))
        )
        statuses = [response.status_code for response in responses]
        assert statuses.count(200) == GLOBAL_BURST
        assert statuses.count(429) == 25

    async def test_simultaneous_forgeries_cannot_overdraw_the_webhook_bucket(
        self, service: Service
    ) -> None:
        client = service.clients[ATTACKER]
        responses = await asyncio.gather(
            *(deliver(client, b"{}", "t=1,v1=00") for _ in range(WEBHOOK_BURST + 15))
        )
        statuses = [response.status_code for response in responses]
        assert statuses.count(400) == WEBHOOK_BURST
        assert statuses.count(429) == 15


class TestWhatIsLogged:
    async def test_one_warning_per_episode_and_one_line_when_it_lifts(
        self, service: Service, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level(logging.INFO, logger="aura_web.rate_limit")
        records = caplog.records
        client = service.clients[CLIENT]
        for _ in range(GLOBAL_BURST + 40):
            await client.get("/api/health")
        warnings = [r for r in records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        assert "global" in warnings[0].getMessage()
        assert CLIENT in warnings[0].getMessage()
        service.clock.advance(1)
        await client.get("/api/health")
        lifted = [r.getMessage() for r in records if "Rate limit lifted" in r.getMessage()]
        assert len(lifted) == 1
        assert "40 refused" in lifted[0]

    async def test_no_query_string_cookie_or_signature_reaches_a_log_line(
        self, service: Service, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level(logging.DEBUG)
        client = service.clients[ATTACKER]
        client.cookies.set("aura_session", "COOKIE-CANARY-1f3a")
        for _ in range(AUTH_BURST + 3):
            await client.get(
                "/api/auth/callback", params={"code": "CODE-CANARY-9c1e", "state": "STATE-CANARY"}
            )
        for _ in range(WEBHOOK_BURST + 3):
            await deliver(client, b"{}", "t=1,v1=SIGNATURE-CANARY")
        # The service's own loggers; httpx logs the TEST client's requests here too.
        text = "\n".join(
            record.getMessage() for record in caplog.records if record.name.startswith("aura_web")
        )
        assert "Rate limit reached" in text
        for canary in ("COOKIE-CANARY", "CODE-CANARY", "STATE-CANARY", "SIGNATURE-CANARY"):
            assert canary not in text


class TestLogInjection:
    async def test_a_newline_in_the_path_cannot_start_a_forged_log_line(
        self, service: Service, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level(logging.WARNING, logger="aura_web.rate_limit")
        client = service.clients[ATTACKER]
        path = "/api/auth/x%0a2026-10-01 00:00:00 CRITICAL aura_web.app: forged%0d%0a"
        for _ in range(AUTH_BURST + 1):
            await client.get(path)
        (warning,) = [
            r.getMessage() for r in caplog.records if "Rate limit reached" in r.getMessage()
        ]
        assert "\n" not in warning
        assert "\r" not in warning
        assert "/api/auth/x%0A2026" in warning


class TestMemoryThroughTheService:
    async def test_a_flood_of_distinct_clients_keeps_the_limiter_bounded(
        self,
        web_settings: WebSettings,
        discord_state: FakeDiscordState,
    ) -> None:
        """Distinct clients arrive the only way they can: behind the trusted frontend."""
        settings = web_settings.model_copy(
            update={"trusted_proxy_addresses": "172.16.86.3", "rate_limit_max_tracked_clients": 25}
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=create_fake_discord(discord_state)),
            base_url="https://discord.test",
        ) as discord_http:
            app = build_app(settings, discord_http)
            async with (
                app.router.lifespan_context(app),
                httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app, client=("172.16.86.3", 40000)),
                    base_url="https://testserver",
                ) as client,
            ):
                for index in range(1_000):
                    await client.get(
                        "/api/health",
                        headers={"X-Forwarded-For": f"2001:db8:{index:x}::1"},
                    )
                limiter_in_use: RateLimiter = app.state.rate_limiter
                assert limiter_in_use.tracked_clients(RateLimitBucket.GLOBAL) == 25


FRONTEND_PEER: Final = "172.16.86.3"


@pytest_asyncio.fixture
async def behind_the_frontend(
    production_limits: WebSettings, clock: FakeClock, discord_state: FakeDiscordState
) -> AsyncIterator[httpx.AsyncClient]:
    """The real application with production limits, reached only through the trusted frontend."""
    settings = production_limits.model_copy(update={"trusted_proxy_addresses": FRONTEND_PEER})
    rate_limiter = rate_limiter_from_settings(settings, clock=clock)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=create_fake_discord(discord_state)),
        base_url="https://discord.test",
    ) as discord_http:
        app = build_app(settings, discord_http, rate_limiter=rate_limiter)
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app, client=(FRONTEND_PEER, 40000)),
                base_url="https://testserver",
            ) as client,
        ):
            yield client


async def login_as(client: httpx.AsyncClient, visitor: str) -> httpx.Response:
    return await client.get("/api/auth/login", headers={"X-Forwarded-For": visitor})


class TestVisitorsBehindTheFrontend:
    async def test_rotating_the_interface_id_inside_one_slash_64_shares_one_bucket(
        self, behind_the_frontend: httpx.AsyncClient
    ) -> None:
        for index in range(AUTH_BURST):
            response = await login_as(behind_the_frontend, f"2001:db8:aa:bb::{index + 1:x}")
            assert response.status_code == 307
        assert_refused(await login_as(behind_the_frontend, "2001:db8:aa:bb:dead:beef:cafe:1"))

    async def test_a_neighbouring_slash_64_keeps_its_own_bucket(
        self, behind_the_frontend: httpx.AsyncClient
    ) -> None:
        for _ in range(AUTH_BURST * 2):
            await login_as(behind_the_frontend, "2001:db8:aa:bb::1")
        assert (await login_as(behind_the_frontend, "2001:db8:aa:bc::1")).status_code == 307

    async def test_two_visitors_on_different_ipv4_addresses_are_counted_apart(
        self, behind_the_frontend: httpx.AsyncClient
    ) -> None:
        laptop, phone = CLIENT, ATTACKER
        for _ in range(AUTH_BURST):
            assert (await login_as(behind_the_frontend, laptop)).status_code == 307
        assert_refused(await login_as(behind_the_frontend, laptop))
        for _ in range(AUTH_BURST):
            assert (await login_as(behind_the_frontend, phone)).status_code == 307
        assert_refused(await login_as(behind_the_frontend, phone))

    async def test_an_ipv6_visitor_and_an_ipv4_visitor_are_counted_apart(
        self, behind_the_frontend: httpx.AsyncClient
    ) -> None:
        for _ in range(AUTH_BURST * 2):
            await login_as(behind_the_frontend, "2001:db8:aa:bb::1")
        assert (await login_as(behind_the_frontend, CLIENT)).status_code == 307
