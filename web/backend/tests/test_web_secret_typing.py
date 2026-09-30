"""Every web credential is a SecretStr until the moment a request is authenticated with it (F-10).

Two halves, both needed. A SecretStr that is never unwrapped where it is used
sends "**********" instead of the credential -- a silent outage -- so the first
half proves each consumer still puts the PLAIN value on the wire, byte for
byte, by capturing the real request the real client builds. The second half
proves the settings, the service context and the per-user token objects print
nothing secret.
"""

from __future__ import annotations

import base64
import contextlib
from datetime import UTC, datetime, timedelta

import httpx
import pytest
import stripe
from pydantic import SecretStr

from aura_web.bot_billing import BotBillingClient
from aura_web.config import WebSettings
from aura_web.discord_api import DiscordAPIError, DiscordClient
from aura_web.routes.billing import _Caller
from aura_web.sessions import DiscordTokens, DiscordUser, Session
from aura_web.stripe_api import StripeClient
from fake_bot_billing import FakeBotBillingState, create_fake_bot_billing
from fake_discord import FakeDiscordState, create_fake_discord
from fake_stripe import FakeStripeState, create_fake_stripe, sign_webhook
from helpers import FAKE_BOT_BASE, FAKE_STRIPE_BASE, build_app, complete_login

MASK = "**********"
SECRET_FIELDS = (
    "discord_client_secret",
    "discord_bot_token",
    "stripe_secret_key",
    "stripe_webhook_secret",
    "bot_internal_api_secret",
)


class CapturingTransport(httpx.AsyncBaseTransport):
    """Answers every request with a fixed body and keeps the request."""

    def __init__(self, body: object) -> None:
        self.requests: list[httpx.Request] = []
        self._body = body

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return httpx.Response(200, json=self._body)


class TestTheSettingsTypeEveryCredential:
    @pytest.mark.parametrize("field", SECRET_FIELDS)
    def test_each_credential_is_a_secret_str(self, web_settings: WebSettings, field: str) -> None:
        assert isinstance(getattr(web_settings, field), SecretStr)

    def test_the_public_values_stay_plain(self, web_settings: WebSettings) -> None:
        assert isinstance(web_settings.discord_client_id, str)
        assert isinstance(web_settings.stripe_price_id, str)


class TestEveryConsumerSendsThePlainValue:
    async def test_the_stripe_client_sends_the_key_itself(self) -> None:
        transport = CapturingTransport({"object": "list", "data": [], "has_more": False})
        async with httpx.AsyncClient(transport=transport) as http:
            client = StripeClient(
                http,
                api_base="https://stripe.test",
                secret_key=SecretStr("sk_test_plainValueOnTheWire"),
                price_id="price_x",
                checkout_success_url="https://f/",
                checkout_cancel_url="https://f/",
                portal_return_url="https://f/",
            )
            await client.list_aura_subscription_ids()

        (request,) = transport.requests
        assert request.headers["authorization"] == "Bearer sk_test_plainValueOnTheWire"

    async def test_the_bot_billing_client_sends_the_secret_itself(self) -> None:
        transport = CapturingTransport({"event_processed": False, "version": 0})
        secret = "plain-internal-secret-" + "q" * 20
        async with httpx.AsyncClient(transport=transport) as http:
            client = BotBillingClient(http, base_url="https://bot.test", secret=SecretStr(secret))
            await client.get_sync_state(subscription_id="sub_A", event_id=None)

        (request,) = transport.requests
        assert request.headers["authorization"] == f"Bearer {secret}"

    async def test_the_discord_client_sends_the_bot_token_itself(self) -> None:
        transport = CapturingTransport([])
        async with httpx.AsyncClient(transport=transport) as http:
            client = DiscordClient(
                http,
                api_base="https://discord.test/api/v10",
                client_id="1",
                client_secret=SecretStr("plain-client-secret"),
                bot_token=SecretStr("plain-bot-token"),
            )
            await client.fetch_bot_guild_ids()

        assert transport.requests[0].headers["authorization"] == "Bot plain-bot-token"

    async def test_the_discord_client_authenticates_the_token_exchange_with_the_secret_itself(
        self,
    ) -> None:
        transport = CapturingTransport(
            {"access_token": "a", "token_type": "Bearer", "expires_in": 3600, "scope": "x"}
        )
        async with httpx.AsyncClient(transport=transport) as http:
            client = DiscordClient(
                http,
                api_base="https://discord.test/api/v10",
                client_id="1",
                client_secret=SecretStr("plain-client-secret"),
                bot_token=SecretStr("plain-bot-token"),
            )
            # Whatever the client makes of the answer, only the request it built matters.
            with contextlib.suppress(DiscordAPIError):
                await client.exchange_code("code", "https://frontend.test/api/auth/callback")

        expected = base64.b64encode(b"1:plain-client-secret").decode()
        assert transport.requests[0].headers["authorization"] == f"Basic {expected}"

    async def test_the_webhook_verifies_against_the_signing_secret_itself(
        self,
        app_client: httpx.AsyncClient,
        stripe_state: FakeStripeState,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        seen: list[object] = []
        original = stripe.WebhookSignature.verify_header

        def spy(payload: bytes, header: str, secret: object, tolerance: int) -> bool:
            seen.append(secret)
            return original(payload, header, secret, tolerance)  # type: ignore[arg-type]

        monkeypatch.setattr(stripe.WebhookSignature, "verify_header", spy)
        subscription = stripe_state.add_subscription(
            guild_id="1000", purchaser_user_id="5000", now=int(datetime.now(UTC).timestamp())
        )
        body, signature = stripe_state.signed(
            stripe_state.subscription_event("customer.subscription.updated", subscription.id)
        )

        response = await app_client.post(
            "/api/stripe/webhook", content=body, headers={"Stripe-Signature": signature}
        )

        assert response.status_code == 200
        assert seen == [stripe_state.webhook_secret]
        assert type(seen[0]) is str

    async def test_a_signature_made_with_the_mask_is_refused(
        self, app_client: httpx.AsyncClient, stripe_state: FakeStripeState
    ) -> None:
        """Were the SecretStr itself interpolated, "**********" would be the key that verifies."""
        subscription = stripe_state.add_subscription(
            guild_id="1000", purchaser_user_id="5000", now=int(datetime.now(UTC).timestamp())
        )
        body, _ = stripe_state.signed(
            stripe_state.subscription_event("customer.subscription.updated", subscription.id)
        )

        response = await app_client.post(
            "/api/stripe/webhook",
            content=body,
            headers={"Stripe-Signature": sign_webhook(body, MASK)},
        )

        assert response.status_code == 400


class TestNothingPrintsACredential:
    def test_the_settings_print_only_masks(
        self,
        web_settings: WebSettings,
        discord_state: FakeDiscordState,
        stripe_state: FakeStripeState,
        bot_billing_state: FakeBotBillingState,
    ) -> None:
        printed = "".join((repr(web_settings), str(web_settings), str(web_settings.model_dump())))

        for secret in (
            discord_state.client_secret,
            discord_state.bot_token,
            stripe_state.secret_key,
            stripe_state.webhook_secret,
            bot_billing_state.secret,
        ):
            assert secret not in printed
        assert MASK in printed

    async def test_the_service_context_prints_no_credential_even_with_users_signed_in(
        self,
        web_settings: WebSettings,
        discord_state: FakeDiscordState,
        stripe_state: FakeStripeState,
        bot_billing_state: FakeBotBillingState,
    ) -> None:
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
            app = build_app(web_settings, discord_http, stripe_http, bot_http)
            async with app.router.lifespan_context(app):
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app), base_url="https://testserver"
                ) as browser:
                    await complete_login(browser, discord_state, "5000")
                context = app.state.context
                printed = repr(context) + str(context)

        user_tokens = [*discord_state.access_tokens, *discord_state.refresh_tokens]
        assert user_tokens
        for secret in (
            discord_state.client_secret,
            discord_state.bot_token,
            stripe_state.secret_key,
            stripe_state.webhook_secret,
            bot_billing_state.secret,
            *user_tokens,
        ):
            assert secret not in printed

    def test_a_session_prints_its_expiry_but_not_the_users_tokens(self) -> None:
        tokens = DiscordTokens(
            access_token="user-access-token-value",
            refresh_token="user-refresh-token-value",
            expires_at=datetime(2026, 10, 1, tzinfo=UTC),
        )
        session = Session(
            user=DiscordUser(id="5000", username="moderator", global_name=None, avatar=None),
            tokens=tokens,
            created_at=datetime(2026, 9, 30, tzinfo=UTC),
            expires_at=datetime(2026, 9, 30, tzinfo=UTC) + timedelta(days=7),
        )

        printed = repr(tokens) + repr(session)

        assert "user-access-token-value" not in printed
        assert "user-refresh-token-value" not in printed
        assert "2026" in printed

    def test_a_billing_caller_prints_no_session_cookie(self) -> None:
        caller = _Caller(session=None, session_token="the-session-cookie-value", manageable=[])  # type: ignore[arg-type]

        assert "the-session-cookie-value" not in repr(caller)
