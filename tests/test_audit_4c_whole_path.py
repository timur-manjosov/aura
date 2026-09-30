"""Phase 4c audit, Attacks 2-4 and 9: break atomicity, race it, reorder it -- against the real ends.

Written by the post-hoc audit of commit 9d0aa23 (reports/phase-4c-audit.md).
Every test here runs the production web application (FastAPI, the real sync,
the real Stripe and bot clients) against a Stripe stand-in, and the production
bot internal API (aiohttp on a real socket) over a real SQLite FILE with the
real plan gate -- so a "restart" is a new connection and a new gate built from
what is on disk, exactly as `AuraClient.setup_hook` builds it.

What is asserted is always the end state Stripe itself holds, and the gate's
answer for it, never merely an HTTP status.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import random
import socket
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Final
from unittest.mock import patch

import aiosqlite
import httpx
import pytest
import pytest_asyncio
from pydantic import SecretStr

from aura.billing import GracePolicy, PlanGate
from aura.billing.internal_api import (
    InternalApiServer,
    create_internal_api_app,
    start_internal_api,
)
from aura.config import Settings
from aura.db.repository import init_schema
from aura.db.subscriptions import count_processed_events, load_subscription_records
from aura_web.app import create_app
from aura_web.bot_billing import BotBillingClient
from aura_web.config import WebSettings
from aura_web.discord_api import DiscordClient
from aura_web.stripe_api import StripeClient
from fake_discord import (
    PERMISSION_MANAGE_GUILD,
    FakeDiscordState,
    FakeGuild,
    FakeUser,
    create_fake_discord,
)
from fake_stripe import FakeStripeState, FakeSubscription, create_fake_stripe

SECRET: Final = "audit-whole-path-internal-secret-" + "z" * 20
FRONTEND: Final = "https://frontend.test"
POLICY: Final = GracePolicy(
    renewal_grace=timedelta(hours=72), payment_failure_grace=timedelta(days=7)
)
RACE_RUNS: Final = 10


@dataclass
class RealBot:
    """The bot's side: a file database, its plan gate and the internal API on a real port."""

    path: Path
    conn: aiosqlite.Connection
    gate: PlanGate
    server: InternalApiServer
    port: int

    async def restart(self) -> None:
        """Stop everything and come back the way setup_hook does: from what is on disk."""
        await self.server.stop()
        await self.conn.close()
        self.conn = await aiosqlite.connect(self.path)
        await init_schema(self.conn)
        self.gate = PlanGate(
            enforced=True,
            policy=POLICY,
            complimentary_guild_ids=frozenset(),
            records=await load_subscription_records(self.conn),
        )
        self.server = await start_internal_api(
            self.conn, self.gate, secret=SecretStr(SECRET), host="127.0.0.1", port=self.port
        )


@dataclass
class Stack:
    web: httpx.AsyncClient
    stripe: FakeStripeState
    bot: RealBot
    discord: FakeDiscordState


@pytest_asyncio.fixture
async def bot(tmp_path: Path) -> AsyncIterator[RealBot]:
    path = tmp_path / "aura.db"
    conn = await aiosqlite.connect(path)
    await init_schema(conn)
    gate = PlanGate(enforced=True, policy=POLICY, complimentary_guild_ids=frozenset(), records=[])
    server = await start_internal_api(
        conn, gate, secret=SecretStr(SECRET), host="127.0.0.1", port=0
    )
    real = RealBot(path, conn, gate, server, server.bound_port)
    try:
        yield real
    finally:
        await real.server.stop()
        await real.conn.close()


@pytest_asyncio.fixture
async def stack(bot: RealBot) -> AsyncIterator[Stack]:
    discord_state = FakeDiscordState()
    discord_state.guilds = {"1000": FakeGuild(id="1000", name="Aura Test Server")}
    discord_state.bot_guild_ids = {"1000"}
    discord_state.users = {
        "5000": FakeUser(
            id="5000", username="moderator", guild_permissions={"1000": PERMISSION_MANAGE_GUILD}
        )
    }
    stripe_state = FakeStripeState()
    settings = WebSettings(
        _env_file=None,  # type: ignore[call-arg]
        discord_client_id=discord_state.client_id,
        discord_client_secret=discord_state.client_secret,  # type: ignore[arg-type]
        discord_bot_token=discord_state.bot_token,  # type: ignore[arg-type]
        discord_api_base="https://discord.test/api/v10",
        oauth_redirect_uri=f"{FRONTEND}/api/auth/callback",
        post_login_redirect_url=f"{FRONTEND}/",
        stripe_secret_key=stripe_state.secret_key,  # type: ignore[arg-type]
        stripe_webhook_secret=stripe_state.webhook_secret,  # type: ignore[arg-type]
        stripe_price_id=stripe_state.price_id,
        stripe_api_base="https://stripe.test",
        bot_internal_api_url=f"http://127.0.0.1:{bot.port}",
        bot_internal_api_secret=SECRET,  # type: ignore[arg-type]
    )
    async with (
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=create_fake_discord(discord_state))
        ) as discord_http,
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=create_fake_stripe(stripe_state))
        ) as stripe_http,
        httpx.AsyncClient(timeout=10.0) as bot_http,
    ):
        app = create_app(
            settings,
            discord_client_factory=lambda _unused_http, s: DiscordClient(
                discord_http,
                api_base=s.discord_api_base,
                client_id=s.discord_client_id,
                client_secret=s.discord_client_secret,
                bot_token=s.discord_bot_token,
            ),
            stripe_client_factory=lambda _unused_http, s: StripeClient(
                stripe_http,
                api_base=s.stripe_api_base,
                secret_key=s.stripe_secret_key,
                price_id=s.stripe_price_id,
                checkout_success_url=s.checkout_success_url,
                checkout_cancel_url=s.checkout_cancel_url,
                portal_return_url=s.billing_portal_return_url,
            ),
            bot_billing_client_factory=lambda _unused_http, s: BotBillingClient(
                bot_http, base_url=s.bot_internal_api_url, secret=s.bot_internal_api_secret
            ),
        )
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="https://testserver"
            ) as web:
                yield Stack(web, stripe_state, bot, discord_state)


async def deliver(stack: Stack, event: dict[str, Any]) -> httpx.Response:
    body, signature = stack.stripe.signed(event)
    return await stack.web.post(
        "/api/stripe/webhook", content=body, headers={"Stripe-Signature": signature}
    )


async def deliver_until_acknowledged(stack: Stack, event: dict[str, Any]) -> httpx.Response:
    """Redeliver on any non-2xx, as Stripe does, a bounded number of times."""
    for _ in range(20):
        response = await deliver(stack, event)
        if response.status_code < 300:
            return response
        await asyncio.sleep(0)
    raise AssertionError(f"event {event['id']} was never acknowledged")


async def stored(stack: Stack, subscription_id: str) -> Any:
    records = [
        record
        for record in await load_subscription_records(stack.bot.conn)
        if record.subscription_id == subscription_id
    ]
    return records[0] if records else None


# --- Attack 4: reorder it ----------------------------------------------------

Step = Callable[[FakeSubscription], None]


def _set(**fields: Any) -> Step:
    def apply(subscription: FakeSubscription) -> None:
        for name, value in fields.items():
            setattr(subscription, name, value)

    return apply


# Each timeline is the sequence of (event type, the change Stripe made just
# before emitting it). The final Stripe state is what every ordering must end on.
ENDS_CANCELED: Final[list[tuple[str, Step]]] = [
    ("customer.subscription.created", _set()),
    (
        "invoice.payment_failed",
        _set(status="past_due", latest_invoice_status="open"),
    ),
    ("invoice.paid", _set(status="active", latest_invoice_status="paid")),
    ("customer.subscription.updated", _set(cancel_at_period_end=True)),
    ("customer.subscription.deleted", _set(status="canceled")),
]
ENDS_PAID: Final[list[tuple[str, Step]]] = [
    ("customer.subscription.created", _set()),
    ("invoice.payment_failed", _set(status="past_due", latest_invoice_status="open")),
    ("invoice.payment_failed", _set()),
    ("invoice.paid", _set(status="active", latest_invoice_status="paid")),
]


async def _run_ordering(
    stack: Stack,
    guild_id: str,
    timeline: list[tuple[str, Step]],
    order: tuple[int, ...],
) -> FakeSubscription:
    """Advance Stripe through the timeline lazily, delivering events in `order`.

    An event can only be delivered after Stripe emitted it, i.e. after its own
    state change -- so before delivering event i, Stripe is advanced to step i
    if it is not there yet. Everything else about the order is adversarial.
    """
    subscription = stack.stripe.add_subscription(
        guild_id=guild_id, purchaser_user_id="5000", now=int(time.time())
    )
    events: list[dict[str, Any] | None] = [None] * len(timeline)
    reached = -1

    def advance_to(step: int) -> None:
        nonlocal reached
        while reached < step:
            reached += 1
            event_type, change = timeline[reached]
            change(subscription)
            if event_type.startswith("invoice."):
                events[reached] = stack.stripe.invoice_event(event_type, subscription.id)
            else:
                events[reached] = stack.stripe.subscription_event(event_type, subscription.id)

    for index in order:
        advance_to(index)
        event = events[index]
        assert event is not None
        await deliver_until_acknowledged(stack, event)
    advance_to(len(timeline) - 1)
    return subscription


class TestReorderIt:
    async def test_every_delivery_order_of_a_lifecycle_ending_canceled_ends_free(
        self, stack: Stack
    ) -> None:
        orderings = list(itertools.permutations(range(len(ENDS_CANCELED))))
        for number, order in enumerate(orderings):
            guild_id = str(1000 + number)
            subscription = await _run_ordering(stack, guild_id, ENDS_CANCELED, order)

            record = await stored(stack, subscription.id)
            assert record is not None, order
            assert record.status.value == "canceled", order
            assert stack.bot.gate.allows_pro(int(guild_id)) is False, order
        assert len(orderings) == 120

    async def test_payment_failed_arriving_after_payment_succeeded_still_ends_on_pro(
        self, stack: Stack
    ) -> None:
        for number, order in enumerate(itertools.permutations(range(len(ENDS_PAID)))):
            guild_id = str(5000 + number)
            subscription = await _run_ordering(stack, guild_id, ENDS_PAID, order)

            record = await stored(stack, subscription.id)
            assert record is not None and record.status.value == "active", order
            assert stack.bot.gate.allows_pro(int(guild_id)) is True, order

    async def test_an_old_updated_event_replayed_after_the_deletion_changes_nothing(
        self, stack: Stack
    ) -> None:
        subscription = stack.stripe.add_subscription(
            guild_id="1000", purchaser_user_id="5000", now=int(time.time())
        )
        stale_update = stack.stripe.subscription_event(
            "customer.subscription.updated", subscription.id
        )
        subscription.status = "canceled"
        deletion = stack.stripe.subscription_event("customer.subscription.deleted", subscription.id)

        assert (await deliver(stack, deletion)).json() == {"status": "applied"}
        # The stale event's own payload says "active"; it is only a signal to re-fetch.
        assert (await deliver(stack, stale_update)).json() == {"status": "applied"}

        record = await stored(stack, subscription.id)
        assert record.status.value == "canceled"
        assert stack.bot.gate.allows_pro(1000) is False


# --- Attack 3: race it --------------------------------------------------------


class TestRaceIt:
    @pytest.mark.parametrize("run", range(RACE_RUNS))
    async def test_one_event_delivered_twenty_times_at_once_is_applied_exactly_once(
        self, stack: Stack, run: int
    ) -> None:
        rng = random.Random(run)

        async def jitter(_: str) -> None:
            await asyncio.sleep(rng.random() / 200)

        stack.stripe.retrieve_hook = jitter
        subscription = stack.stripe.add_subscription(
            guild_id="1000", purchaser_user_id="5000", now=int(time.time())
        )
        event = stack.stripe.subscription_event("customer.subscription.created", subscription.id)

        responses = await asyncio.gather(*(deliver(stack, event) for _ in range(20)))

        statuses = sorted(response.json().get("status", "error") for response in responses)
        assert statuses.count("applied") == 1, statuses
        assert set(statuses) <= {"applied", "duplicate"}, statuses
        assert await count_processed_events(stack.bot.conn) == 1
        assert (await stored(stack, subscription.id)).version == 1

    @pytest.mark.parametrize("run", range(RACE_RUNS))
    async def test_contradictory_events_at_once_always_converge_on_stripes_final_state(
        self, stack: Stack, run: int
    ) -> None:
        rng = random.Random(1000 + run)
        subscription = stack.stripe.add_subscription(
            guild_id="1000", purchaser_user_id="5000", now=int(time.time())
        )
        renewed = stack.stripe.invoice_event("invoice.paid", subscription.id)
        failed = stack.stripe.invoice_event("invoice.payment_failed", subscription.id)
        updated = stack.stripe.subscription_event("customer.subscription.updated", subscription.id)
        fetches = 0

        async def jitter_and_cancel_midway(_: str) -> None:
            # Stripe's state changes WHILE the syncs are in flight: the
            # subscription is canceled at a random point among the fetches.
            nonlocal fetches
            fetches += 1
            if fetches == cancel_at_fetch:
                subscription.status = "canceled"
            await asyncio.sleep(rng.random() / 100)

        cancel_at_fetch = rng.randint(1, 4)
        stack.stripe.retrieve_hook = jitter_and_cancel_midway
        deleted = stack.stripe.subscription_event("customer.subscription.deleted", subscription.id)
        events = [renewed, failed, updated, deleted]
        rng.shuffle(events)

        await asyncio.gather(*(deliver_until_acknowledged(stack, event) for event in events))
        # Stripe's own guarantee: the deletion event is emitted AFTER the
        # cancellation. If the cancellation happened after its delivery in this
        # simulation, a real Stripe would still be delivering it -- so redeliver.
        subscription.status = "canceled"
        await deliver_until_acknowledged(
            stack, stack.stripe.subscription_event("customer.subscription.deleted", subscription.id)
        )

        record = await stored(stack, subscription.id)
        assert record.status.value == "canceled"
        assert stack.bot.gate.allows_pro(1000) is False
        assert await count_processed_events(stack.bot.conn) == 5
        assert record.version == 5

    @pytest.mark.parametrize("run", range(5))
    async def test_the_reconciler_racing_webhooks_leaves_the_gate_equal_to_the_disk(
        self, stack: Stack, run: int
    ) -> None:
        from aura_web.billing_sync import reconcile_subscriptions

        rng = random.Random(2000 + run)

        async def jitter(_: str) -> None:
            await asyncio.sleep(rng.random() / 100)

        stack.stripe.retrieve_hook = jitter
        subscriptions = [
            stack.stripe.add_subscription(
                guild_id=str(1000 + index), purchaser_user_id="5000", now=int(time.time())
            )
            for index in range(5)
        ]
        events = [
            stack.stripe.subscription_event("customer.subscription.updated", sub.id)
            for sub in subscriptions
        ]
        for index, sub in enumerate(subscriptions):
            sub.status = ["canceled", "past_due", "active", "unpaid", "active"][index]

        async with httpx.AsyncClient() as bot_http:
            reconciler_bot = BotBillingClient(
                bot_http, base_url=f"http://127.0.0.1:{stack.bot.port}", secret=SecretStr(SECRET)
            )
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=create_fake_stripe(stack.stripe))
            ) as stripe_http:
                reconciler_stripe = StripeClient(
                    stripe_http,
                    api_base="https://stripe.test",
                    secret_key=SecretStr(stack.stripe.secret_key),
                    price_id=stack.stripe.price_id,
                    checkout_success_url=f"{FRONTEND}/",
                    checkout_cancel_url=f"{FRONTEND}/",
                    portal_return_url=f"{FRONTEND}/",
                )
                await asyncio.gather(
                    *(deliver_until_acknowledged(stack, event) for event in events),
                    reconcile_subscriptions(
                        stripe=reconciler_stripe, bot=reconciler_bot, live_mode=False
                    ),
                )

        on_disk = {r.subscription_id: r for r in await load_subscription_records(stack.bot.conn)}
        for sub in subscriptions:
            assert on_disk[sub.id].status.value == sub.status
            fresh_gate = PlanGate(
                enforced=True,
                policy=POLICY,
                complimentary_guild_ids=frozenset(),
                records=on_disk.values(),
            )
            guild = int(sub.metadata["aura_guild_id"])
            assert stack.bot.gate.allows_pro(guild) == fresh_gate.allows_pro(guild)


# --- Attack 2: break atomicity ------------------------------------------------


class _FailOnce:
    """Wraps one aiosqlite method so its first matching call raises."""

    def __init__(self, original: Callable[..., Any], *, when: Callable[..., bool]) -> None:
        self.original = original
        self.when = when
        self.fired = False

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if not self.fired and self.when(*args, **kwargs):
            self.fired = True
            raise RuntimeError("injected failure")
        return self.original(*args, **kwargs)


class TestBreakAtomicity:
    async def _new_subscription_event(self, stack: Stack) -> tuple[FakeSubscription, dict]:
        subscription = stack.stripe.add_subscription(
            guild_id="1000", purchaser_user_id="5000", now=int(time.time())
        )
        return subscription, stack.stripe.subscription_event(
            "customer.subscription.created", subscription.id
        )

    async def test_a_failure_writing_the_snapshot_marks_nothing_and_the_retry_applies(
        self, stack: Stack
    ) -> None:
        subscription, event = await self._new_subscription_event(stack)
        conn = stack.bot.conn
        failing = _FailOnce(
            conn.execute, when=lambda sql, *_a, **_k: "INSERT INTO guild_subscriptions" in sql
        )

        with patch.object(conn, "execute", failing):
            first = await deliver(stack, event)

        assert first.status_code == 503
        assert await load_subscription_records(conn) == []
        assert await count_processed_events(conn) == 0
        assert stack.bot.gate.allows_pro(1000) is False
        assert (await deliver(stack, event)).json() == {"status": "applied"}
        assert stack.bot.gate.allows_pro(1000) is True
        assert (await stored(stack, subscription.id)).version == 1

    async def test_a_failure_marking_the_event_rolls_the_snapshot_back(self, stack: Stack) -> None:
        _, event = await self._new_subscription_event(stack)
        conn = stack.bot.conn
        failing = _FailOnce(
            conn.execute, when=lambda sql, *_a, **_k: "INSERT INTO stripe_processed_events" in sql
        )

        with patch.object(conn, "execute", failing):
            first = await deliver(stack, event)

        assert first.status_code == 503
        assert await load_subscription_records(conn) == []
        assert await count_processed_events(conn) == 0
        assert stack.bot.gate.allows_pro(1000) is False
        assert (await deliver(stack, event)).json() == {"status": "applied"}

    async def test_a_failing_commit_leaves_neither_half_and_the_retry_applies(
        self, stack: Stack
    ) -> None:
        _, event = await self._new_subscription_event(stack)
        conn = stack.bot.conn
        failing = _FailOnce(conn.commit, when=lambda *_a, **_k: True)

        with patch.object(conn, "commit", failing):
            first = await deliver(stack, event)

        assert first.status_code == 503
        assert await load_subscription_records(conn) == []
        assert await count_processed_events(conn) == 0
        assert (await deliver(stack, event)).json() == {"status": "applied"}
        assert stack.bot.gate.allows_pro(1000) is True

    async def test_a_crash_after_the_commit_is_healed_by_the_restart_not_by_stripes_retry(
        self, stack: Stack
    ) -> None:
        """Pins current behaviour (finding F-12): the ledger can say "applied" while the gate did not move.

        The commit and the gate update are separate steps. If anything fails
        between them, the database holds the snapshot and the event, the gate
        does not -- and Stripe's retry is answered "duplicate", so it never
        repairs the gate. Only a restart (which rebuilds the gate from disk)
        does. Unreachable with today's code short of a process crash, which
        restarts anyway; recorded so a future change to record_applied knows.
        """
        _, event = await self._new_subscription_event(stack)
        gate = stack.bot.gate
        failing = _FailOnce(gate.record_applied, when=lambda *_a, **_k: True)

        with patch.object(gate, "record_applied", failing):
            first = await deliver(stack, event)

        assert first.status_code == 503
        assert len(await load_subscription_records(stack.bot.conn)) == 1
        assert await count_processed_events(stack.bot.conn) == 1
        assert gate.allows_pro(1000) is False  # the stale window
        assert (await deliver(stack, event)).json() == {"status": "duplicate"}
        assert gate.allows_pro(1000) is False  # the retry does not repair it

        await stack.bot.restart()

        assert stack.bot.gate.allows_pro(1000) is True

    async def test_a_process_death_right_after_the_commit_loses_nothing(self, stack: Stack) -> None:
        subscription, event = await self._new_subscription_event(stack)
        assert (await deliver(stack, event)).json() == {"status": "applied"}

        await stack.bot.restart()

        assert stack.bot.gate.allows_pro(1000) is True
        assert (await deliver(stack, event)).json() == {"status": "duplicate"}
        assert (await stored(stack, subscription.id)).version == 1


# --- The internal API at its edges -------------------------------------------


def _raw_request(port: int, request: bytes) -> bytes:
    with socket.create_connection(("127.0.0.1", port), timeout=5) as sock:
        sock.sendall(request)
        chunks = []
        while True:
            chunk = sock.recv(65536)
            if not chunk:
                break
            chunks.append(chunk)
    return b"".join(chunks)


def _plans_request(authorization: bytes | None) -> bytes:
    body = json.dumps({"guild_ids": ["1000"]}).encode()
    lines = [
        b"POST /internal/v1/guilds/plans HTTP/1.1",
        b"Host: 127.0.0.1",
        b"Content-Type: application/json",
        f"Content-Length: {len(body)}".encode(),
        b"Connection: close",
    ]
    if authorization is not None:
        lines.append(b"Authorization: " + authorization)
    return b"\r\n".join(lines) + b"\r\n\r\n" + body


class TestInternalApiEdges:
    async def test_a_header_that_is_not_utf8_is_a_401_not_a_crash(self, bot: RealBot) -> None:
        response = await asyncio.to_thread(
            _raw_request, bot.port, _plans_request(b"Bearer \xff\xfe" + SECRET.encode())
        )

        assert response.startswith(b"HTTP/1.1 401")

    async def test_two_authorization_headers_are_refused_even_when_the_genuine_one_comes_first(
        self, bot: RealBot
    ) -> None:
        """The committed test sends the wrong header first, so "use the first" also passes it (F-02)."""
        request = _plans_request(b"Bearer " + SECRET.encode()).replace(
            b"\r\n\r\n", b"\r\nAuthorization: Bearer wrong\r\n\r\n", 1
        )

        response = await asyncio.to_thread(_raw_request, bot.port, request)

        assert response.startswith(b"HTTP/1.1 401")

    async def test_the_bearer_scheme_is_matched_exactly(self, bot: RealBot) -> None:
        for presented in (
            b"bearer " + SECRET.encode(),
            SECRET.encode(),
            b"Bearer  " + SECRET.encode(),
        ):
            response = await asyncio.to_thread(_raw_request, bot.port, _plans_request(presented))
            assert response.startswith(b"HTTP/1.1 401"), presented

    @pytest.mark.parametrize(
        "weak_secret",
        ["", " ", " " * 40, "\t" * 32, "x" * 31, "Bearer " + "x" * 24],
        ids=["empty", "one-space", "forty-spaces", "tabs", "31-chars", "31-chars-with-space"],
    )
    async def test_no_listener_can_be_built_around_an_empty_or_weak_secret(
        self, tmp_path: Path, weak_secret: str
    ) -> None:
        """Defence in depth (F-08): the listener refuses a weak secret itself.

        The configuration refuses a short secret and main.py starts no API
        without one, so production could not reach this even before the fix.
        But the listener, handed an empty secret, expected exactly ``Bearer ``
        -- which aiohttp delivers intact -- and served every guild's customer
        and purchaser IDs to whoever sent it. Now no construction path accepts
        such a secret, and a refused start leaves nothing listening.
        """
        conn = await aiosqlite.connect(tmp_path / "empty.db")
        await init_schema(conn)
        gate = PlanGate(
            enforced=True, policy=POLICY, complimentary_guild_ids=frozenset(), records=[]
        )
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        try:
            with pytest.raises(ValueError):
                create_internal_api_app(conn, gate, secret=SecretStr(weak_secret))
            with pytest.raises(ValueError):
                await start_internal_api(
                    conn, gate, secret=SecretStr(weak_secret), host="127.0.0.1", port=port
                )
            with pytest.raises(OSError):
                socket.create_connection(("127.0.0.1", port), timeout=1).close()
        finally:
            await conn.close()

    async def test_a_plain_string_secret_is_refused_rather_than_used(self, tmp_path: Path) -> None:
        """Only a SecretStr is accepted, so a secret can never reach the listener unmasked."""
        conn = await aiosqlite.connect(tmp_path / "plain.db")
        await init_schema(conn)
        gate = PlanGate(
            enforced=True, policy=POLICY, complimentary_guild_ids=frozenset(), records=[]
        )
        try:
            for secret in ("", SECRET):
                with pytest.raises(TypeError):
                    create_internal_api_app(conn, gate, secret=secret)  # type: ignore[arg-type]
                with pytest.raises(TypeError):
                    await start_internal_api(
                        conn,
                        gate,
                        secret=secret,  # type: ignore[arg-type]
                        host="127.0.0.1",
                        port=0,
                    )
        finally:
            await conn.close()

    def test_a_blank_or_short_secret_never_reaches_the_listener_through_configuration(
        self,
    ) -> None:
        for blank in ("", "   ", "\t"):
            settings = Settings(_env_file=None, discord_token="t", internal_api_secret=blank)  # type: ignore[call-arg, arg-type]
            assert settings.internal_api_secret is None
        with pytest.raises(ValueError):
            Settings(_env_file=None, discord_token="t", internal_api_secret="x" * 31)  # type: ignore[call-arg, arg-type]


# --- H: secrets must not print by accident -------------------------------------


class TestBotSecretsDoNotPrint:
    def test_the_settings_repr_does_not_contain_the_internal_api_secret(self) -> None:
        settings = Settings(_env_file=None, discord_token="t", internal_api_secret=SECRET)  # type: ignore[call-arg, arg-type]

        assert SECRET not in repr(settings)
        assert SECRET not in str(settings)


# --- D: a guild that subscribes again --------------------------------------------


async def sign_in(stack: Stack) -> None:
    login = await stack.web.get("/api/auth/login")
    state = httpx.URL(login.headers["location"]).params["state"]
    callback = await stack.web.get(
        "/api/auth/callback", params={"code": stack.discord.issue_code("5000"), "state": state}
    )
    assert callback.status_code == 303


async def checkout(stack: Stack) -> httpx.Response:
    return await stack.web.post(
        "/api/billing/checkout",
        content=b'{"guild_id": "1000"}',
        headers={"Content-Type": "application/json", "Origin": FRONTEND},
    )


class TestSubscribingAgain:
    async def test_a_guild_can_subscribe_again_after_a_cancellation_and_only_the_new_one_grants(
        self, stack: Stack
    ) -> None:
        await sign_in(stack)
        assert (await checkout(stack)).status_code == 200
        first_session = next(iter(stack.stripe.checkout_sessions))
        first = stack.stripe.complete_checkout(first_session, now=int(time.time()))
        await deliver(stack, stack.stripe.checkout_completed_event(first_session))
        assert stack.bot.gate.allows_pro(1000) is True

        first.status = "canceled"
        await deliver(
            stack, stack.stripe.subscription_event("customer.subscription.deleted", first.id)
        )
        assert stack.bot.gate.allows_pro(1000) is False

        # Outside the double-click window, so Stripe creates a fresh session.
        with patch("aura_web.routes.billing.time.time", return_value=time.time() + 3600):
            again = await checkout(stack)
        assert again.status_code == 200
        second_session = next(
            session_id
            for session_id in stack.stripe.checkout_sessions
            if session_id != first_session
        )
        second = stack.stripe.complete_checkout(second_session, now=int(time.time()))
        await deliver(stack, stack.stripe.checkout_completed_event(second_session))

        plan = stack.bot.gate.plan_for(1000)
        assert plan.is_pro is True
        assert plan.standing.granting_subscription_ids == frozenset({second.id})
        assert {r.subscription_id for r in await load_subscription_records(stack.bot.conn)} == {
            first.id,
            second.id,
        }

    async def test_while_a_cancellation_is_pending_nobody_can_start_a_second_subscription(
        self, stack: Stack
    ) -> None:
        """Characterisation: CANCELING still grants, so the guild counts as already paying."""
        await sign_in(stack)
        subscription = stack.stripe.add_subscription(
            guild_id="1000",
            purchaser_user_id="5000",
            now=int(time.time()),
            cancel_at_period_end=True,
        )
        await deliver(
            stack, stack.stripe.subscription_event("customer.subscription.updated", subscription.id)
        )

        refused = await checkout(stack)

        assert refused.status_code == 409
        assert refused.json() == {"error": "already_subscribed"}
