"""Phase 4c audit, Attacks 1, 7 and 8: forge the webhook, CSRF the checkout, scan for secrets by value.

Written by the post-hoc audit of commit 9d0aa23 (reports/phase-4c-audit.md).
Complements test_stripe_webhook.py, test_billing_routes.py and
test_no_stripe_secret_leakage.py rather than repeating them: the forged-header
cases they do not cover, the empty-signing-secret case below the
configuration, the exact body ceiling, the cross-site variants a real browser
can produce, and a secret scan at DEBUG over the FAILURE paths -- a bot that
cannot be reached, Stripe refusing the key, a configuration that is refused at
startup.

A test marked ``xfail(strict=True)`` pins a defect the audit reported.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import time
from collections.abc import AsyncIterator
from typing import Any, Final

import httpx
import pytest
import pytest_asyncio
from pydantic import SecretStr, ValidationError
from test_no_token_leakage import raw_response_bytes

from aura_web.config import WebConfigurationError, WebSettings, load_web_settings
from aura_web.routes.billing import CHECKOUT_IDEMPOTENCY_WINDOW_SECONDS, checkout_idempotency_key
from aura_web.routes.stripe_webhook import MAX_WEBHOOK_BODY_BYTES
from fake_bot_billing import FakeBotBillingState, create_fake_bot_billing
from fake_discord import FakeDiscordState, create_fake_discord
from fake_stripe import FakeStripeState, create_fake_stripe, sign_webhook
from helpers import FAKE_BOT_BASE, FAKE_STRIPE_BASE, FRONTEND_BASE, build_app, complete_login

WEBHOOK: Final = "/api/stripe/webhook"
CHECKOUT: Final = "/api/billing/checkout"
PORTAL: Final = "/api/billing/portal"
# A marker in every forged body, so a log line that echoes the body is found.
BODY_MARKER: Final = "AUDITBODYMARKERqz"


def _hex_hmac(secret: str, timestamp: int, body: bytes) -> str:
    return hmac.new(
        secret.encode("utf-8"), f"{timestamp}.".encode() + body, hashlib.sha256
    ).hexdigest()


def event_body(stripe_state: FakeStripeState) -> bytes:
    subscription = stripe_state.add_subscription(
        guild_id="1000", purchaser_user_id="5000", now=int(time.time())
    )
    event = stripe_state.subscription_event("customer.subscription.created", subscription.id)
    event["data"]["object"]["metadata"]["marker"] = BODY_MARKER
    return json.dumps(event).encode()


def untouched(stripe_state: FakeStripeState, bot_state: FakeBotBillingState) -> bool:
    return (
        bot_state.request_log == [] and stripe_state.request_log == [] and bot_state.snapshots == {}
    )


def forged_headers(body: bytes, secret: str) -> dict[str, list[str] | None]:
    now = int(time.time())
    good = _hex_hmac(secret, now, body)
    return {
        "missing": None,
        "empty": [""],
        "only-whitespace": ["   "],
        "no-timestamp": [f"v1={good}"],
        "no-v1-scheme": [f"t={now}"],
        "only-a-v0-signature": [f"t={now},v0={good}"],
        "empty-v1": [f"t={now},v1="],
        "non-numeric-timestamp": [f"t=soon,v1={good}"],
        "uppercase-hex-digest": [f"t={now},v1={good.upper()}"],
        "space-after-comma": [f"t={now}, v1={good}"],
        "truncated-digest": [f"t={now},v1={good[:-1]}"],
        "digest-of-another-timestamp": [f"t={now - 1},v1={good}"],
        "first-t-is-forged": [f"t={now - 1},t={now},v1={good}"],
        "wrong-secret": [sign_webhook(body, "whsec_attackerChosenSecret")],
        "empty-key-hmac": [f"t={now},v1={_hex_hmac('', now, body)}"],
        "stale-301s": [f"t={now - 301},v1={_hex_hmac(secret, now - 301, body)}"],
        "stale-a-year": [f"t={now - 31_536_000},v1={_hex_hmac(secret, now - 31_536_000, body)}"],
        "two-headers-one-genuine": [f"t={now},v1={good}", "t=1,v1=deadbeef"],
        "garbage": ["éèê"],
        "very-long": ["t=1,v1=" + "a" * 8000],
    }


def _header_list(values: list[str] | None) -> list[tuple[bytes, bytes]]:
    # Latin-1, so a header carrying non-ASCII bytes reaches the server as bytes.
    headers = [(b"Content-Type", b"application/json")]
    for value in values or []:
        headers.append((b"Stripe-Signature", value.encode("latin-1")))
    return headers


class TestForgeEverything:
    @pytest.mark.parametrize(
        "case",
        [
            "missing",
            "empty",
            "only-whitespace",
            "no-timestamp",
            "no-v1-scheme",
            "only-a-v0-signature",
            "empty-v1",
            "non-numeric-timestamp",
            "uppercase-hex-digest",
            "space-after-comma",
            "truncated-digest",
            "digest-of-another-timestamp",
            "first-t-is-forged",
            "wrong-secret",
            "empty-key-hmac",
            "stale-301s",
            "stale-a-year",
            "two-headers-one-genuine",
            "garbage",
            "very-long",
        ],
    )
    async def test_the_forgery_is_refused_touches_nothing_and_logs_nothing_it_was_sent(
        self,
        app_client: httpx.AsyncClient,
        stripe_state: FakeStripeState,
        bot_billing_state: FakeBotBillingState,
        caplog: pytest.LogCaptureFixture,
        case: str,
    ) -> None:
        body = event_body(stripe_state)
        values = forged_headers(body, stripe_state.webhook_secret)[case]

        with caplog.at_level(logging.DEBUG):
            response = await app_client.post(WEBHOOK, content=body, headers=_header_list(values))

        assert response.status_code == 400
        assert response.json() == {"error": "invalid_signature"}
        assert untouched(stripe_state, bot_billing_state)
        logged = "\n".join(record.getMessage() for record in caplog.records)
        assert BODY_MARKER not in logged
        assert stripe_state.webhook_secret not in logged
        for value in values or []:
            if len(value) > 12:
                assert value not in logged

    async def test_a_body_that_is_not_utf8_but_correctly_signed_is_refused(
        self,
        app_client: httpx.AsyncClient,
        stripe_state: FakeStripeState,
        bot_billing_state: FakeBotBillingState,
    ) -> None:
        body = b'{"object": "event", "id": "evt_\xff\xfe"}'
        response = await app_client.post(
            WEBHOOK,
            content=body,
            headers={"Stripe-Signature": sign_webhook(body, stripe_state.webhook_secret)},
        )

        assert response.status_code == 400
        assert untouched(stripe_state, bot_billing_state)


class TestWhatAValidSignatureStillAllows:
    """Characterisation: behaviours of Stripe's own verifier this service inherits."""

    async def test_several_v1_signatures_pass_when_one_is_genuine_as_stripe_rotation_requires(
        self, app_client: httpx.AsyncClient, stripe_state: FakeStripeState
    ) -> None:
        body = event_body(stripe_state)
        now = int(time.time())
        header = f"t={now},v1={'0' * 64},v1={_hex_hmac(stripe_state.webhook_secret, now, body)}"

        response = await app_client.post(
            WEBHOOK, content=body, headers={"Stripe-Signature": header}
        )

        assert response.status_code == 200

    async def test_a_future_timestamp_signed_with_the_secret_is_accepted(
        self, app_client: httpx.AsyncClient, stripe_state: FakeStripeState
    ) -> None:
        """Only an OLD timestamp is refused (finding F-16, a note): forging one needs the secret."""
        body = event_body(stripe_state)
        future = int(time.time()) + 365 * 24 * 3600
        header = f"t={future},v1={_hex_hmac(stripe_state.webhook_secret, future, body)}"

        response = await app_client.post(
            WEBHOOK, content=body, headers={"Stripe-Signature": header}
        )

        assert response.status_code == 200

    async def test_a_genuine_delivery_replayed_inside_the_tolerance_is_a_duplicate(
        self,
        app_client: httpx.AsyncClient,
        stripe_state: FakeStripeState,
        bot_billing_state: FakeBotBillingState,
    ) -> None:
        body = event_body(stripe_state)
        headers = {"Stripe-Signature": sign_webhook(body, stripe_state.webhook_secret)}

        first = await app_client.post(WEBHOOK, content=body, headers=headers)
        replay = await app_client.post(WEBHOOK, content=body, headers=headers)

        assert (first.json(), replay.json()) == ({"status": "applied"}, {"status": "duplicate"})
        assert len(bot_billing_state.applied_snapshots) == 1


class TestTheBodyCeiling:
    async def _post(
        self, client: httpx.AsyncClient, stripe_state: FakeStripeState, size: int, *, chunked: bool
    ) -> httpx.Response:
        base = event_body(stripe_state)
        body = base[:-1] + b"," + b'"pad":"' + b"a" * (size - len(base) - 9) + b'"}'
        assert len(body) == size
        headers = {"Stripe-Signature": sign_webhook(body, stripe_state.webhook_secret)}
        if not chunked:
            return await client.post(WEBHOOK, content=body, headers=headers)

        async def stream() -> AsyncIterator[bytes]:
            for start in range(0, len(body), 65536):
                yield body[start : start + 65536]

        return await client.post(WEBHOOK, content=stream(), headers=headers)

    async def test_exactly_the_ceiling_is_read_and_verified(
        self, app_client: httpx.AsyncClient, stripe_state: FakeStripeState
    ) -> None:
        response = await self._post(app_client, stripe_state, MAX_WEBHOOK_BODY_BYTES, chunked=False)

        assert response.status_code == 200

    @pytest.mark.parametrize("chunked", [False, True])
    async def test_one_byte_over_is_refused_before_verification(
        self,
        app_client: httpx.AsyncClient,
        stripe_state: FakeStripeState,
        bot_billing_state: FakeBotBillingState,
        chunked: bool,
    ) -> None:
        response = await self._post(
            app_client, stripe_state, MAX_WEBHOOK_BODY_BYTES + 1, chunked=chunked
        )

        assert response.status_code == 413
        assert untouched(stripe_state, bot_billing_state)


class TestAnEmptySigningSecretNeverRunsOpen:
    @pytest.mark.parametrize("secret", ["", "   ", "whsec_", "whsec_ spaced", "sk_test_x"])
    def test_the_configuration_refuses_it(self, web_settings: WebSettings, secret: str) -> None:
        values = web_settings.model_dump() | {"stripe_webhook_secret": secret}

        with pytest.raises(ValidationError):
            WebSettings(_env_file=None, **values)  # type: ignore[call-arg]

    @pytest_asyncio.fixture
    async def open_app(
        self,
        web_settings: WebSettings,
        discord_state: FakeDiscordState,
        stripe_state: FakeStripeState,
        bot_billing_state: FakeBotBillingState,
    ) -> AsyncIterator[httpx.AsyncClient]:
        """The real app with the signing secret forced empty, past the validator."""
        unvalidated = web_settings.model_copy(update={"stripe_webhook_secret": SecretStr("")})
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
            app = build_app(unvalidated, discord_http, stripe_http, bot_http)
            async with app.router.lifespan_context(app):
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app), base_url="https://testserver"
                ) as client:
                    yield client

    async def test_below_the_configuration_an_empty_key_signature_is_still_refused(
        self,
        open_app: httpx.AsyncClient,
        stripe_state: FakeStripeState,
        bot_billing_state: FakeBotBillingState,
    ) -> None:
        body = event_body(stripe_state)
        now = int(time.time())

        response = await open_app.post(
            WEBHOOK,
            content=body,
            headers={"Stripe-Signature": f"t={now},v1={_hex_hmac('', now, body)}"},
        )

        assert response.status_code == 400
        assert untouched(stripe_state, bot_billing_state)


class TestCsrfTheBillingRoutes:
    """Attack 7. Every variant a hostile page can make a browser send, against both POST routes."""

    @pytest_asyncio.fixture
    async def signed_in(
        self, app_client: httpx.AsyncClient, discord_state: FakeDiscordState
    ) -> httpx.AsyncClient:
        response = await complete_login(app_client, discord_state, "5000")
        assert response.status_code == 303
        return app_client

    @pytest.mark.parametrize("path", [CHECKOUT, PORTAL])
    @pytest.mark.parametrize(
        ("headers", "body", "expected"),
        [
            pytest.param(
                {
                    "Content-Type": "application/x-www-form-urlencoded",
                    "Origin": "https://evil.example",
                    "Sec-Fetch-Site": "cross-site",
                },
                b"guild_id=1000",
                415,
                id="html-form-urlencoded",
            ),
            pytest.param(
                {"Content-Type": "text/plain", "Origin": "https://evil.example"},
                b'{"guild_id": "1000"}',
                415,
                id="html-form-text-plain-carrying-json",
            ),
            pytest.param(
                {
                    "Content-Type": "multipart/form-data; boundary=x",
                    "Origin": "https://evil.example",
                },
                b'--x\r\nContent-Disposition: form-data; name="guild_id"\r\n\r\n1000\r\n--x--',
                415,
                id="html-form-multipart",
            ),
            pytest.param(
                {"Content-Type": "application/json", "Origin": "https://evil.example"},
                b'{"guild_id": "1000"}',
                403,
                id="json-foreign-origin",
            ),
            pytest.param(
                {"Content-Type": "application/json", "Origin": "null"},
                b'{"guild_id": "1000"}',
                403,
                id="json-opaque-origin",
            ),
            pytest.param(
                {"Content-Type": "application/json", "Origin": "http://frontend.test"},
                b'{"guild_id": "1000"}',
                403,
                id="json-scheme-downgraded-origin",
            ),
            pytest.param(
                {
                    "Content-Type": "application/json",
                    "Origin": "https://frontend.test.evil.example",
                },
                b'{"guild_id": "1000"}',
                403,
                id="json-suffix-origin",
            ),
            pytest.param(
                {"Content-Type": "application/json", "Origin": "https://frontend.test:8443"},
                b'{"guild_id": "1000"}',
                403,
                id="json-other-port-origin",
            ),
            pytest.param(
                {"Content-Type": "application/json", "Sec-Fetch-Site": "cross-site"},
                b'{"guild_id": "1000"}',
                403,
                id="json-no-origin-marked-cross-site",
            ),
            pytest.param(
                {
                    "Content-Type": "application/json",
                    "Origin": "https://sibling.frontend.test",
                    "Sec-Fetch-Site": "same-site",
                },
                b'{"guild_id": "1000"}',
                403,
                id="json-same-site-sibling-origin",
            ),
        ],
    )
    async def test_a_cross_site_request_is_refused_before_stripe_is_contacted(
        self,
        signed_in: httpx.AsyncClient,
        stripe_state: FakeStripeState,
        path: str,
        headers: dict[str, str],
        body: bytes,
        expected: int,
    ) -> None:
        response = await signed_in.post(path, content=body, headers=headers)

        assert response.status_code == expected
        assert stripe_state.request_log == []

    async def test_two_origin_headers_are_refused(
        self, signed_in: httpx.AsyncClient, stripe_state: FakeStripeState
    ) -> None:
        response = await signed_in.post(
            CHECKOUT,
            content=b'{"guild_id": "1000"}',
            headers=[
                ("Content-Type", "application/json"),
                ("Origin", FRONTEND_BASE),
                ("Origin", "https://evil.example"),
            ],
        )

        assert response.status_code == 403
        assert stripe_state.request_log == []

    async def test_the_checkout_cannot_be_reached_by_a_get(
        self, signed_in: httpx.AsyncClient, stripe_state: FakeStripeState
    ) -> None:
        response = await signed_in.get(CHECKOUT, params={"guild_id": "1000"})

        assert response.status_code == 405
        assert stripe_state.request_log == []

    async def test_the_legitimate_same_origin_request_passes(
        self, signed_in: httpx.AsyncClient
    ) -> None:
        response = await signed_in.post(
            CHECKOUT,
            content=b'{"guild_id": "1000"}',
            headers={
                "Content-Type": "application/json; charset=utf-8",
                "Origin": FRONTEND_BASE.upper(),
                "Sec-Fetch-Site": "same-origin",
            },
        )

        assert response.status_code == 200
        assert response.json()["url"].startswith("https://checkout.stripe.com/")

    async def test_the_session_cookie_is_withheld_from_cross_site_posts_by_samesite_lax(
        self, app_client: httpx.AsyncClient, discord_state: FakeDiscordState
    ) -> None:
        response = await complete_login(app_client, discord_state, "5000")

        session_cookie = next(
            value
            for value in response.headers.get_list("set-cookie")
            if value.startswith("aura_session=")
        ).lower()
        assert "samesite=lax" in session_cookie
        assert "httponly" in session_cookie
        assert "secure" in session_cookie


class TestWebSecretsDoNotPrint:
    def test_the_settings_repr_contains_no_secret(
        self,
        web_settings: WebSettings,
        discord_state: FakeDiscordState,
        stripe_state: FakeStripeState,
        bot_billing_state: FakeBotBillingState,
    ) -> None:
        printed = repr(web_settings) + str(web_settings)

        for secret in (
            stripe_state.secret_key,
            stripe_state.webhook_secret,
            bot_billing_state.secret,
            discord_state.bot_token,
            discord_state.client_secret,
        ):
            assert secret not in printed


class _UnreachableTransport(httpx.AsyncBaseTransport):
    """A transport whose every request fails the way a stopped bot container does."""

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(f"All connection attempts failed for {request.url}")


class TestSecretLeakScanOverFailurePaths:
    """Attack 8: the actual secret values, searched in every response byte and every DEBUG log line."""

    async def test_no_failure_path_puts_a_secret_on_the_wire_or_in_a_log(
        self,
        web_settings: WebSettings,
        discord_state: FakeDiscordState,
        stripe_state: FakeStripeState,
        bot_billing_state: FakeBotBillingState,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        secrets = {
            "stripe_secret_key": stripe_state.secret_key,
            "stripe_webhook_secret": stripe_state.webhook_secret,
            "bot_internal_api_secret": bot_billing_state.secret,
            "discord_client_secret": discord_state.client_secret,
            "discord_bot_token": discord_state.bot_token,
        }
        responses: list[httpx.Response] = []
        json_headers = {"Content-Type": "application/json", "Origin": FRONTEND_BASE}

        with caplog.at_level(logging.DEBUG):
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
                    transport=_UnreachableTransport(), base_url=FAKE_BOT_BASE
                ) as dead_bot_http,
            ):
                app = build_app(web_settings, discord_http, stripe_http, dead_bot_http)
                async with app.router.lifespan_context(app):
                    async with httpx.AsyncClient(
                        transport=httpx.ASGITransport(app=app), base_url="https://testserver"
                    ) as client:
                        responses.append(await complete_login(client, discord_state, "5000"))
                        # The bot cannot be reached at all.
                        responses.append(await client.get("/api/billing/guilds"))
                        responses.append(
                            await client.post(
                                CHECKOUT, content=b'{"guild_id": "1000"}', headers=json_headers
                            )
                        )
                        responses.append(
                            await client.post(
                                PORTAL, content=b'{"guild_id": "1000"}', headers=json_headers
                            )
                        )
                        body = event_body(stripe_state)
                        signed = {
                            "Stripe-Signature": sign_webhook(body, stripe_state.webhook_secret)
                        }
                        responses.append(await client.post(WEBHOOK, content=body, headers=signed))
                        # Stripe refusing the key, rate limiting, erroring, not finding it.
                        for status in (401, 403, 404, 429, 500):
                            stripe_state.fail_retrieve_status = status
                            fresh = event_body(stripe_state)
                            responses.append(
                                await client.post(
                                    WEBHOOK,
                                    content=fresh,
                                    headers={
                                        "Stripe-Signature": sign_webhook(
                                            fresh, stripe_state.webhook_secret
                                        )
                                    },
                                )
                            )
                        stripe_state.fail_retrieve_status = None
                        # A body over the ceiling, refused before verification.
                        responses.append(
                            await client.post(
                                WEBHOOK,
                                content=b"x" * (MAX_WEBHOOK_BODY_BYTES + 5),
                                headers={"Stripe-Signature": "t=1,v1=00"},
                            )
                        )

        assert len(responses) == 11
        assert any(response.status_code == 503 for response in responses)
        for response in responses:
            wire = raw_response_bytes(response)
            for label, secret in secrets.items():
                assert secret.encode() not in wire, f"{label} in a response to {response.url}"

        logged = "\n".join(
            record.getMessage()
            + (logging.Formatter().formatException(record.exc_info) if record.exc_info else "")
            + (record.stack_info or "")
            for record in caplog.records
        )
        assert "billing" in logged.lower()
        for label, secret in secrets.items():
            assert secret not in logged, f"{label} was written to a log record"

    def test_a_refused_configuration_names_the_problem_and_never_the_secrets(
        self,
        web_settings: WebSettings,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        # The real values, not SecretStr's mask: the environment must hold the
        # actual secrets for "never printed" to mean anything.
        values: dict[str, Any] = {
            name: value.get_secret_value() if isinstance(value, SecretStr) else value
            for name, value in web_settings.model_dump().items()
            if value is not None
        }
        # A fake live key, assembled at runtime so that no key-shaped literal
        # sits in the source for secret scanners to (rightly) refuse.
        live_key = "sk_live_" + "auditNeverPrintThis0000000000"
        values |= {
            "stripe_secret_key": live_key,
            "bot_internal_api_secret": "short",
            "checkout_success_url": "not a url",
        }
        for name, value in values.items():
            monkeypatch.setenv(f"AURA_WEB_{name.upper()}", str(value))
        monkeypatch.chdir("/")

        with caplog.at_level(logging.DEBUG), pytest.raises(WebConfigurationError) as raised:
            load_web_settings()

        rendered = " ".join(
            (str(raised.value), repr(raised.value), str(raised.value.__cause__), caplog.text)
        )
        for secret in (
            live_key,
            web_settings.stripe_webhook_secret.get_secret_value(),
            web_settings.discord_bot_token.get_secret_value(),
            web_settings.discord_client_secret.get_secret_value(),
        ):
            assert secret not in rendered


class TestInterruptedCheckouts:
    async def test_a_web_backend_restart_mid_checkout_loses_the_login_not_the_subscription(
        self,
        app_client: httpx.AsyncClient,
        web_settings: WebSettings,
        discord_state: FakeDiscordState,
        stripe_state: FakeStripeState,
        bot_billing_state: FakeBotBillingState,
    ) -> None:
        await complete_login(app_client, discord_state, "5000")
        started = await app_client.post(
            CHECKOUT,
            content=b'{"guild_id": "1000"}',
            headers={"Content-Type": "application/json", "Origin": FRONTEND_BASE},
        )
        assert started.status_code == 200
        session_id = next(iter(stripe_state.checkout_sessions))
        stripe_state.complete_checkout(session_id, now=int(time.time()))

        # A second process: every in-memory session is gone.
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
            restarted = build_app(web_settings, discord_http, stripe_http, bot_http)
            async with restarted.router.lifespan_context(restarted):
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=restarted), base_url="https://testserver"
                ) as fresh_browser:
                    assert (await fresh_browser.get("/api/me")).status_code == 401
                    body, signature = stripe_state.signed(
                        stripe_state.checkout_completed_event(session_id)
                    )
                    delivered = await fresh_browser.post(
                        WEBHOOK, content=body, headers={"Stripe-Signature": signature}
                    )

        assert delivered.json() == {"status": "applied"}
        (applied,) = bot_billing_state.applied_snapshots
        assert applied["snapshot"]["guild_id"] == "1000"
        assert applied["snapshot"]["purchaser_user_id"] == "5000"

    def test_two_clicks_either_side_of_a_window_boundary_get_two_different_keys(self) -> None:
        """Characterisation (finding F-17, a note): the double-click guard is a fixed time bucket."""
        boundary = 10 * CHECKOUT_IDEMPOTENCY_WINDOW_SECONDS

        before = checkout_idempotency_key(user_id="5000", guild_id="1000", now=boundary - 0.001)
        after = checkout_idempotency_key(user_id="5000", guild_id="1000", now=boundary)

        assert before != after


class TestWhatASubscriptionMustBeToCount:
    async def test_a_subscription_on_another_price_is_not_pushed_to_the_bot_as_pro(
        self,
        app_client: httpx.AsyncClient,
        stripe_state: FakeStripeState,
        bot_billing_state: FakeBotBillingState,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """F-07: pushed, but as a subscription that grants nothing.

        The remediation chose "pushed as non-granting" over "not pushed": a
        subscription the bot already holds would otherwise keep its last Pro
        copy and go on granting until that copy expired.
        """
        from fake_stripe import FakeSubscription

        original = FakeSubscription.to_object

        def on_a_cheaper_price(self: FakeSubscription, *, expand_invoice: bool) -> dict[str, Any]:
            payload = original(self, expand_invoice=expand_invoice)
            for item in payload["items"]["data"]:
                item["price"] = {"id": "price_someOtherCheaperProduct", "object": "price"}
                item["quantity"] = 0
            return payload

        monkeypatch.setattr(FakeSubscription, "to_object", on_a_cheaper_price)
        body = event_body(stripe_state)

        response = await app_client.post(
            WEBHOOK,
            content=body,
            headers={"Stripe-Signature": sign_webhook(body, stripe_state.webhook_secret)},
        )

        assert response.status_code == 200
        (applied,) = bot_billing_state.applied_snapshots
        assert applied["snapshot"]["on_pro_price"] is False
