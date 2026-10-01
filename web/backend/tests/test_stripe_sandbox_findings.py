"""What the real Stripe sandbox run found (V-01, V-02, V-12), pinned so it cannot come back.

reports/phase-4c-stripe-verification.md ran the whole billing path against
Stripe itself. Three of its findings live on this side of the service:

  * V-01: an account whose default is Managed Payments refuses the card-only
    checkout outright, and the log said only "HTTP 400". The app now switches
    Managed Payments off for its own sessions, and a refusal names the
    parameter Stripe objected to.
  * V-02: expanding `latest_invoice` needs the Invoices (read) permission; a
    key without it fails every sync. The service now says so once, at startup.
  * V-12: the stand-in was more permissive or simpler than Stripe in ways that
    could hide a bug. Each correction is asserted here, so the stand-in cannot
    drift back without a test going red.

Every test drives the real StripeClient (or the whole application) against the
stand-in over HTTP; nothing in aura_web is mocked.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager
from typing import Final

import httpx
import pytest
import pytest_asyncio
from pydantic import SecretStr

from aura_web.stripe_api import (
    CHECKOUT_MANAGED_PAYMENTS_FIELD,
    CheckoutSession,
    StripeClient,
    StripeRejectedError,
    parse_subscription,
)
from fake_discord import FakeDiscordState
from fake_stripe import (
    DAY,
    DRAFT_FINALIZATION_WINDOW,
    MANAGED_PAYMENTS_REFUSAL,
    FakeStripeState,
    create_fake_stripe,
)
from helpers import FRONTEND_BASE, complete_login

NOW: Final = 1_790_000_000
CHECKOUT: Final = "/api/billing/checkout"
STRIPE_BASE: Final = "https://stripe.test"


def client_for(
    http: httpx.AsyncClient, state: FakeStripeState, *, key: str | None = None
) -> StripeClient:
    return StripeClient(
        http,
        api_base=STRIPE_BASE,
        secret_key=SecretStr(key if key is not None else state.secret_key),
        price_id=state.price_id,
        checkout_success_url="https://frontend.test/ok",
        checkout_cancel_url="https://frontend.test/no",
        portal_return_url="https://frontend.test/",
    )


@asynccontextmanager
async def stripe_http(state: FakeStripeState) -> AsyncGenerator[httpx.AsyncClient]:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=create_fake_stripe(state)), base_url=STRIPE_BASE
    ) as http:
        yield http


@pytest_asyncio.fixture
async def stripe_client(stripe_state: FakeStripeState) -> AsyncIterator[StripeClient]:
    async with stripe_http(stripe_state) as http:
        yield client_for(http, stripe_state)


def log_text(caplog: pytest.LogCaptureFixture) -> str:
    return "\n".join(record.getMessage() for record in caplog.records)


async def create_checkout(client: StripeClient) -> CheckoutSession:
    return await client.create_checkout_session(
        guild_id="1000", purchaser_user_id="5000", idempotency_key="k" * 64
    )


# --- V-01: Managed Payments ---------------------------------------------------


class TestTheCheckoutSwitchesManagedPaymentsOff:
    async def test_the_form_sends_the_flag_beside_the_card_only_list(
        self, stripe_client: StripeClient, stripe_state: FakeStripeState
    ) -> None:
        await create_checkout(stripe_client)

        (form,) = stripe_state.received_forms
        assert form[CHECKOUT_MANAGED_PAYMENTS_FIELD] == "false"
        assert CHECKOUT_MANAGED_PAYMENTS_FIELD == "managed_payments[enabled]"
        assert form["payment_method_types[0]"] == "card"

    @pytest.mark.parametrize("account_default", [True, False], ids=["default-on", "default-off"])
    async def test_the_apps_exact_form_is_accepted_whatever_the_account_default(
        self, stripe_client: StripeClient, stripe_state: FakeStripeState, account_default: bool
    ) -> None:
        stripe_state.managed_payments_default = account_default

        session = await create_checkout(stripe_client)

        created = stripe_state.checkout_sessions[session.session_id]
        assert created["managed_payments_enabled"] is False
        assert created["form"]["payment_method_types[0]"] == "card"

    async def test_the_stand_in_ships_with_managed_payments_on_like_a_new_sandbox(self) -> None:
        assert FakeStripeState().managed_payments_default is True


class TestTheStandInRefusesWhatStripeRefuses:
    """The stand-in's half of V-01: without these, the app's fix would be untested."""

    @staticmethod
    async def app_form(stripe_client: StripeClient, state: FakeStripeState) -> dict[str, str]:
        await create_checkout(stripe_client)
        return dict(state.received_forms[-1])

    async def test_a_card_only_form_without_the_flag_is_refused_under_the_default(
        self, stripe_client: StripeClient, stripe_state: FakeStripeState
    ) -> None:
        form = await self.app_form(stripe_client, stripe_state)
        del form[CHECKOUT_MANAGED_PAYMENTS_FIELD]

        async with stripe_http(stripe_state) as http:
            response = await http.post(
                "/v1/checkout/sessions",
                data=form,
                headers={"Authorization": f"Bearer {stripe_state.secret_key}"},
            )

        assert response.status_code == 400
        error = response.json()["error"]
        assert set(error) == {"type", "message", "request_log_url"}
        assert error["type"] == "invalid_request_error"
        assert error["message"] == MANAGED_PAYMENTS_REFUSAL

    async def test_the_same_form_is_accepted_once_the_default_is_off(
        self, stripe_client: StripeClient, stripe_state: FakeStripeState
    ) -> None:
        form = await self.app_form(stripe_client, stripe_state)
        del form[CHECKOUT_MANAGED_PAYMENTS_FIELD]
        stripe_state.managed_payments_default = False

        async with stripe_http(stripe_state) as http:
            response = await http.post(
                "/v1/checkout/sessions",
                data=form,
                headers={"Authorization": f"Bearer {stripe_state.secret_key}"},
            )

        assert response.status_code == 200
        assert response.json()["managed_payments"] == {"enabled": False}

    @pytest.mark.parametrize("account_default", [True, False])
    async def test_switching_managed_payments_on_explicitly_with_a_card_list_is_refused(
        self, stripe_client: StripeClient, stripe_state: FakeStripeState, account_default: bool
    ) -> None:
        form = await self.app_form(stripe_client, stripe_state)
        form[CHECKOUT_MANAGED_PAYMENTS_FIELD] = "true"
        stripe_state.managed_payments_default = account_default

        async with stripe_http(stripe_state) as http:
            response = await http.post(
                "/v1/checkout/sessions",
                data=form,
                headers={"Authorization": f"Bearer {stripe_state.secret_key}"},
            )

        assert response.status_code == 400

    @pytest.mark.parametrize("value", ["False", "0", "", "no"])
    async def test_a_flag_that_is_not_a_boolean_is_refused(
        self, stripe_client: StripeClient, stripe_state: FakeStripeState, value: str
    ) -> None:
        form = await self.app_form(stripe_client, stripe_state)
        form[CHECKOUT_MANAGED_PAYMENTS_FIELD] = value

        async with stripe_http(stripe_state) as http:
            response = await http.post(
                "/v1/checkout/sessions",
                data=form,
                headers={"Authorization": f"Bearer {stripe_state.secret_key}"},
            )

        assert response.status_code == 400
        assert response.json()["error"]["param"] == CHECKOUT_MANAGED_PAYMENTS_FIELD


class TestTheSessionEchoesWhatItWasCreatedWith:
    async def test_managed_payments_the_method_list_and_the_integration_identifier(
        self, stripe_state: FakeStripeState
    ) -> None:
        async with stripe_http(stripe_state) as http:
            response = await http.post(
                "/v1/checkout/sessions",
                data={
                    "mode": "subscription",
                    "line_items[0][price]": stripe_state.price_id,
                    "payment_method_types[0]": "card",
                    CHECKOUT_MANAGED_PAYMENTS_FIELD: "false",
                    "integration_identifier": "aura_guild_pro_checkout_qxmvrtlk",
                },
                headers={"Authorization": f"Bearer {stripe_state.secret_key}"},
            )

        body = response.json()
        assert body["managed_payments"] == {"enabled": False}
        assert body["payment_method_types"] == ["card"]
        assert body["integration_identifier"] == "aura_guild_pro_checkout_qxmvrtlk"


# --- V-01: a refusal names the parameter, never the message --------------------


def rejecting_client(error: object, *, status: int = 400) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(status, json={"error": error}))
    )


class TestARefusalNamesTheParameter:
    async def test_the_code_and_the_param_reach_the_error_but_not_the_message(
        self, stripe_state: FakeStripeState
    ) -> None:
        error = {
            "type": "invalid_request_error",
            "code": "parameter_unknown",
            "param": "payment_method_types",
            "message": "Unsupported parameter: payment_method_types. rk_test_...ab12",
        }
        async with rejecting_client(error) as http:
            with pytest.raises(StripeRejectedError) as raised:
                await create_checkout(client_for(http, stripe_state))

        text = str(raised.value)
        assert text.endswith(
            "(HTTP 400: invalid_request_error/parameter_unknown, param payment_method_types)"
        )
        assert "Unsupported" not in text and "ab12" not in text

    async def test_the_real_managed_payments_refusal_logs_its_type_and_no_message(
        self, stripe_state: FakeStripeState
    ) -> None:
        # The sandbox's answer carries no code and no param (acceptance run of
        # the Phase 4c-F01 fixes), so the type is all that can be logged.
        error = {
            "type": "invalid_request_error",
            "message": MANAGED_PAYMENTS_REFUSAL,
            "request_log_url": "https://dashboard.stripe.com/test/logs/req_x",
        }
        async with rejecting_client(error) as http:
            with pytest.raises(StripeRejectedError) as raised:
                await create_checkout(client_for(http, stripe_state))

        assert str(raised.value) == (
            "Stripe rejected checkout session creation (HTTP 400: invalid_request_error)"
        )

    async def test_a_param_without_a_code_is_still_named(
        self, stripe_state: FakeStripeState
    ) -> None:
        error = {"type": "invalid_request_error", "param": "line_items[0][price]"}
        async with rejecting_client(error) as http:
            with pytest.raises(StripeRejectedError) as raised:
                await create_checkout(client_for(http, stripe_state))

        assert str(raised.value).endswith(
            "(HTTP 400: invalid_request_error, param line_items[0][price])"
        )

    async def test_a_param_alone_is_named(self, stripe_state: FakeStripeState) -> None:
        async with rejecting_client({"param": "success_url"}) as http:
            with pytest.raises(StripeRejectedError) as raised:
                await create_checkout(client_for(http, stripe_state))

        assert str(raised.value).endswith("(HTTP 400: param success_url)")

    @pytest.mark.parametrize(
        "param",
        [
            "payment_method_types' OR 1=1",
            "metadata[aura_guild_id] sk_test_leak",
            "x" * 129,
            "PAYMENT_METHOD_TYPES",
            "line_items[0][price]\nFORGED LOG LINE",
            "",
            42,
            ["payment_method_types"],
            None,
        ],
    )
    async def test_a_param_outside_stripes_vocabulary_is_dropped_not_escaped(
        self, stripe_state: FakeStripeState, param: object
    ) -> None:
        error = {"type": "invalid_request_error", "code": "parameter_invalid", "param": param}
        async with rejecting_client(error) as http:
            with pytest.raises(StripeRejectedError) as raised:
                await create_checkout(client_for(http, stripe_state))

        assert str(raised.value).endswith("(HTTP 400: invalid_request_error/parameter_invalid)")

    async def test_the_route_logs_the_parameter_of_a_refused_checkout(
        self,
        app_client: httpx.AsyncClient,
        discord_state: FakeDiscordState,
        stripe_state: FakeStripeState,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        await complete_login(app_client, discord_state, "5000")
        # The configured Price no longer exists at "Stripe": the refusal real
        # Stripe gives is resource_missing on line_items[0][price].
        stripe_state.price_id = "price_somethingElse"

        with caplog.at_level(logging.INFO):
            response = await app_client.post(
                CHECKOUT,
                content=b'{"guild_id": "1000"}',
                headers={"Content-Type": "application/json", "Origin": FRONTEND_BASE},
            )

        assert response.status_code == 502
        assert response.json() == {"error": "payment_provider_error"}
        refusals = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert len(refusals) == 1
        assert (
            refusals[0]
            .getMessage()
            .endswith(
                "(HTTP 400: invalid_request_error/resource_missing, param line_items[0][price])"
            )
        )
        assert "The stand-in refused" not in log_text(caplog)
        assert stripe_state.secret_key not in log_text(caplog)


# --- V-12: stand-in fidelity --------------------------------------------------


class TestCancelAtBesideCancelAtPeriodEnd:
    async def test_stripe_sets_cancel_at_to_the_period_end(
        self, stripe_client: StripeClient, stripe_state: FakeStripeState
    ) -> None:
        subscription = stripe_state.add_subscription(
            guild_id="1000", purchaser_user_id="5000", now=NOW, cancel_at_period_end=True
        )

        snapshot = await stripe_client.retrieve_subscription(subscription.id)

        assert snapshot.cancel_at_period_end is True
        assert snapshot.cancel_at == snapshot.current_period_end == NOW + 30 * DAY

    async def test_an_explicit_cancel_date_is_kept(self, stripe_state: FakeStripeState) -> None:
        subscription = stripe_state.add_subscription(
            guild_id="1000", purchaser_user_id="5000", now=NOW, cancel_at=NOW + DAY
        )

        assert subscription.to_object(expand_invoice=False)["cancel_at"] == NOW + DAY

    async def test_no_cancellation_means_no_cancel_date(
        self, stripe_state: FakeStripeState
    ) -> None:
        subscription = stripe_state.add_subscription(
            guild_id="1000", purchaser_user_id="5000", now=NOW
        )

        assert subscription.to_object(expand_invoice=False)["cancel_at"] is None


class TestInvoicesAsStripeBillsThem:
    async def test_a_new_subscriptions_first_invoice_is_subscription_create(
        self, stripe_state: FakeStripeState
    ) -> None:
        added = stripe_state.add_subscription(guild_id="1000", purchaser_user_id="5000", now=NOW)

        invoice = added.to_object(expand_invoice=True)["latest_invoice"]

        assert invoice["billing_reason"] == "subscription_create"
        assert invoice["status"] == "paid"
        assert invoice["automatically_finalizes_at"] is None

    async def test_a_completed_checkout_bills_its_first_invoice_as_subscription_create(
        self, stripe_client: StripeClient, stripe_state: FakeStripeState
    ) -> None:
        session = await create_checkout(stripe_client)

        subscription = stripe_state.complete_checkout(session.session_id, now=NOW)

        assert subscription.latest_invoice_billing_reason == "subscription_create"

    async def test_a_renewal_starts_as_a_draft_of_the_next_period(
        self, stripe_client: StripeClient, stripe_state: FakeStripeState
    ) -> None:
        subscription = stripe_state.add_subscription(
            guild_id="1000", purchaser_user_id="5000", now=NOW
        )
        first_invoice = subscription.latest_invoice_id

        subscription.renew()
        snapshot = await stripe_client.retrieve_subscription(subscription.id)

        invoice = subscription.to_object(expand_invoice=True)["latest_invoice"]
        assert invoice["id"] != first_invoice
        assert invoice["status"] == "draft"
        assert invoice["billing_reason"] == "subscription_cycle"
        assert invoice["automatically_finalizes_at"] == (NOW + 30 * DAY + DRAFT_FINALIZATION_WINDOW)
        assert snapshot.status == "active"
        assert snapshot.latest_invoice_status == "draft"
        assert (snapshot.current_period_start, snapshot.current_period_end) == (
            NOW + 30 * DAY,
            NOW + 60 * DAY,
        )

    @pytest.mark.parametrize(
        ("paid", "status", "invoice"), [(True, "active", "paid"), (False, "past_due", "open")]
    )
    async def test_the_draft_finalizes_into_a_charge_that_succeeds_or_fails(
        self,
        stripe_client: StripeClient,
        stripe_state: FakeStripeState,
        paid: bool,
        status: str,
        invoice: str,
    ) -> None:
        subscription = stripe_state.add_subscription(
            guild_id="1000", purchaser_user_id="5000", now=NOW
        )
        subscription.renew()

        subscription.settle_renewal(paid=paid)
        snapshot = await stripe_client.retrieve_subscription(subscription.id)

        assert (snapshot.status, snapshot.latest_invoice_status) == (status, invoice)
        assert subscription.latest_invoice_finalizes_at is None

    async def test_paying_a_past_due_renewal_returns_the_subscription_to_active(
        self, stripe_state: FakeStripeState
    ) -> None:
        subscription = stripe_state.add_subscription(
            guild_id="1000", purchaser_user_id="5000", now=NOW
        )
        subscription.renew()
        subscription.settle_renewal(paid=False)

        subscription.settle_renewal(paid=True)

        assert (subscription.status, subscription.latest_invoice_status) == ("active", "paid")

    async def test_an_invoice_carries_a_copy_of_the_subscription_metadata(
        self, stripe_state: FakeStripeState
    ) -> None:
        subscription = stripe_state.add_subscription(
            guild_id="1000", purchaser_user_id="5000", now=NOW
        )

        invoice = stripe_state.invoice_event("invoice.paid", subscription.id)["data"]["object"]

        assert "subscription" not in invoice
        assert invoice["parent"] == {
            "type": "subscription_details",
            "subscription_details": {
                "subscription": subscription.id,
                "metadata": {"aura_guild_id": "1000", "aura_discord_user_id": "5000"},
            },
            "quote_details": None,
        }

    async def test_the_copy_is_a_copy(self, stripe_state: FakeStripeState) -> None:
        subscription = stripe_state.add_subscription(
            guild_id="1000", purchaser_user_id="5000", now=NOW
        )

        invoice = subscription.invoice_object()
        invoice["parent"]["subscription_details"]["metadata"]["aura_guild_id"] = "9999"

        assert subscription.metadata["aura_guild_id"] == "1000"

    async def test_the_pre_basil_shape_has_no_parent(self, stripe_state: FakeStripeState) -> None:
        subscription = stripe_state.add_subscription(
            guild_id="1000", purchaser_user_id="5000", now=NOW
        )

        invoice = stripe_state.invoice_event("invoice.paid", subscription.id, legacy_shape=True)

        assert invoice["data"]["object"]["subscription"] == subscription.id
        assert "parent" not in invoice["data"]["object"]

    async def test_the_expanded_invoice_parses_like_before(
        self, stripe_state: FakeStripeState
    ) -> None:
        subscription = stripe_state.add_subscription(
            guild_id="1000", purchaser_user_id="5000", now=NOW
        )

        snapshot = parse_subscription(
            subscription.to_object(expand_invoice=True), pro_price_id=stripe_state.price_id
        )

        assert snapshot.latest_invoice_status == "paid"
        assert snapshot.on_pro_price is True


class TestThePortalURLHasTheRealShape:
    async def test_the_session_secret_is_in_the_query_and_the_host_check_still_passes(
        self, stripe_state: FakeStripeState
    ) -> None:
        subscription = stripe_state.add_subscription(
            guild_id="1000", purchaser_user_id="5000", now=NOW
        )
        async with stripe_http(stripe_state) as http:
            url = httpx.URL(
                await client_for(http, stripe_state).create_portal_session(
                    customer_id=subscription.customer
                )
            )

        assert url.host == "billing.stripe.com"
        assert url.path == "/p/session"
        assert url.params["secret"].startswith("test_")


class TestTestClockSubscriptionsAreNotListed:
    async def test_a_plain_listing_omits_them_so_reconciliation_cannot_see_them(
        self, stripe_client: StripeClient, stripe_state: FakeStripeState
    ) -> None:
        plain = stripe_state.add_subscription(guild_id="1000", purchaser_user_id="5000", now=NOW)
        stripe_state.add_subscription(
            guild_id="1001", purchaser_user_id="5000", now=NOW, test_clock="clock_Abc"
        )

        assert await stripe_client.list_aura_subscription_ids() == [plain.id]

    async def test_asking_for_the_clock_lists_only_its_subscriptions(
        self, stripe_state: FakeStripeState
    ) -> None:
        stripe_state.add_subscription(guild_id="1000", purchaser_user_id="5000", now=NOW)
        clocked = stripe_state.add_subscription(
            guild_id="1001", purchaser_user_id="5000", now=NOW, test_clock="clock_Abc"
        )
        async with stripe_http(stripe_state) as http:
            response = await http.get(
                "/v1/subscriptions",
                params=[("status", "all"), ("test_clock", "clock_Abc")],
                headers={"Authorization": f"Bearer {stripe_state.secret_key}"},
            )

        assert [entry["id"] for entry in response.json()["data"]] == [clocked.id]
        assert response.json()["data"][0]["test_clock"] == "clock_Abc"

    async def test_a_clock_subscription_can_still_be_retrieved_directly(
        self, stripe_client: StripeClient, stripe_state: FakeStripeState
    ) -> None:
        clocked = stripe_state.add_subscription(
            guild_id="1001", purchaser_user_id="5000", now=NOW, test_clock="clock_Abc"
        )

        snapshot = await stripe_client.retrieve_subscription(clocked.id)

        assert snapshot.guild_id == "1001"
