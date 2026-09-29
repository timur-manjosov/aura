"""aura_web.stripe_api: Stripe's responses validated field by field, failures classified, secrets kept out.

Written against hostile and malformed shapes the same way test_discord_api.py
treats Discord: a changed type, a missing field or an impersonated host must be
a clean refusal here, never a plausible value three layers up.
"""
from __future__ import annotations

from collections.abc import AsyncIterator

import httpx
import pytest
import pytest_asyncio
from fake_stripe import FakeStripeState, FakeSubscription, create_fake_stripe

from aura_web.stripe_api import (
    StripeClient,
    StripeRejectedError,
    StripeUnavailableError,
    parse_checkout_session,
    parse_subscription,
)

NOW = 1_757_764_800


def subscription_object(**overrides: object) -> dict:
    base = FakeSubscription(
        id="sub_Abc123",
        customer="cus_Abc123",
        metadata={"aura_guild_id": "1000", "aura_discord_user_id": "5000"},
        current_period_start=NOW,
        current_period_end=NOW + 30 * 86400,
    ).to_object(expand_invoice=True)
    base.update(overrides)
    return base


class TestParseSubscription:
    def test_reads_the_dahlia_shape_with_the_period_on_the_item(self) -> None:
        snapshot = parse_subscription(subscription_object())

        assert snapshot.subscription_id == "sub_Abc123"
        assert snapshot.customer_id == "cus_Abc123"
        assert snapshot.guild_id == "1000"
        assert snapshot.purchaser_user_id == "5000"
        assert snapshot.current_period_start == NOW
        assert snapshot.current_period_end == NOW + 30 * 86400
        assert snapshot.latest_invoice_status == "paid"
        assert snapshot.collection_paused is False

    def test_several_items_use_the_earliest_end(self) -> None:
        items = {"data": [
            {"current_period_start": NOW, "current_period_end": NOW + 365 * 86400},
            {"current_period_start": NOW + 10, "current_period_end": NOW + 30 * 86400},
        ]}

        snapshot = parse_subscription(subscription_object(items=items))

        assert snapshot.current_period_start == NOW
        assert snapshot.current_period_end == NOW + 30 * 86400

    @pytest.mark.parametrize(
        "overrides",
        [
            {"object": "customer"},
            {"id": "sub_"},
            {"id": "sub_abc/../x"},
            {"status": "suspended"},
            {"status": None},
            {"customer": None},
            {"customer": {"id": "acct_x"}},
            {"cancel_at_period_end": "false"},
            {"livemode": 0},
            {"cancel_at": True},
            {"cancel_at": -1},
            {"cancel_at": 10**15},
            {"pause_collection": "void"},
            {"latest_invoice": {"status": "refunded"}},
            {"latest_invoice": 5},
            {"items": {"data": []}},
            {"items": None},
            {"items": {"data": ["si_x"]}},
            {"items": {"data": [{"current_period_start": NOW, "current_period_end": None}]}},
            {"items": {"data": [{"current_period_start": True, "current_period_end": NOW}]}},
            {"items": {"data": [{"current_period_start": NOW + 10, "current_period_end": NOW}]}},
            {"items": {"data": [{"current_period_start": "1757764800", "current_period_end": NOW}]}},
        ],
    )
    def test_a_malformed_subscription_is_refused(self, overrides) -> None:
        with pytest.raises(StripeUnavailableError):
            parse_subscription(subscription_object(**overrides))

    @pytest.mark.parametrize("payload", [None, [], "sub_x", 1])
    def test_a_non_object_is_refused(self, payload) -> None:
        with pytest.raises(StripeUnavailableError):
            parse_subscription(payload)

    def test_an_expanded_customer_is_read_by_its_id(self) -> None:
        assert parse_subscription(subscription_object(customer={"id": "cus_Expanded1"})).customer_id == "cus_Expanded1"

    @pytest.mark.parametrize("reason", ["subscription_create", "subscription_cycle"])
    @pytest.mark.parametrize("status", ["void", "uncollectible", "open", "paid"])
    def test_the_invoice_billing_the_period_is_reported(self, reason: str, status: str) -> None:
        invoice = {"id": "in_1", "object": "invoice", "status": status, "billing_reason": reason}

        assert parse_subscription(subscription_object(latest_invoice=invoice)).latest_invoice_status == status

    @pytest.mark.parametrize("reason", ["subscription_update", "manual", "subscription_threshold", None, "invented"])
    def test_a_voided_invoice_that_does_not_bill_the_period_is_not_reported(self, reason) -> None:
        """An operator voiding a proration or one-off invoice must not end a paid period's Pro."""
        invoice = {"id": "in_1", "object": "invoice", "status": "void", "billing_reason": reason}

        assert parse_subscription(subscription_object(latest_invoice=invoice)).latest_invoice_status is None

    def test_an_unexpanded_invoice_leaves_its_status_unknown(self) -> None:
        assert parse_subscription(subscription_object(latest_invoice="in_123")).latest_invoice_status is None

    def test_paused_collection_is_reported(self) -> None:
        assert parse_subscription(subscription_object(pause_collection={"behavior": "void"})).collection_paused is True

    @pytest.mark.parametrize(
        "value", ["abc", "-1", "0", "9223372036854775808", "١٠٠٠", " 1000", 1000, None, ""]
    )
    def test_unusable_guild_metadata_means_not_an_aura_subscription(self, value) -> None:
        metadata = {"aura_guild_id": value} if value is not None else {}

        assert parse_subscription(subscription_object(metadata=metadata)).guild_id is None

    def test_an_unrelated_guild_id_key_is_not_mistaken_for_aura_metadata(self) -> None:
        assert parse_subscription(subscription_object(metadata={"guild_id": "1000"})).guild_id is None


class TestParseCheckoutSession:
    @pytest.mark.parametrize(
        "url",
        [
            None,
            "",
            "http://checkout.stripe.com/c/pay/cs_test_1",
            "https://checkout.stripe.com.evil.example/c/pay",
            "https://evil.example/checkout.stripe.com",
            "https://user@checkout.stripe.com/c/pay",
            "//checkout.stripe.com/c/pay",
            "javascript://checkout.stripe.com/%0aalert(1)",
        ],
    )
    def test_a_url_that_is_not_https_on_stripe_is_refused(self, url) -> None:
        with pytest.raises(StripeUnavailableError):
            parse_checkout_session({"object": "checkout.session", "id": "cs_test_abc", "url": url})

    def test_a_proper_session_is_accepted(self) -> None:
        session = parse_checkout_session(
            {"object": "checkout.session", "id": "cs_test_abc", "url": "https://checkout.stripe.com/c/pay/cs_test_abc"}
        )

        assert session.session_id == "cs_test_abc"


@pytest_asyncio.fixture
async def stripe_client(stripe_state: FakeStripeState) -> AsyncIterator[StripeClient]:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=create_fake_stripe(stripe_state)), base_url="https://stripe.test"
    ) as http:
        yield StripeClient(
            http, api_base="https://stripe.test", secret_key=stripe_state.secret_key, price_id=stripe_state.price_id,
            checkout_success_url="https://frontend.test/ok", checkout_cancel_url="https://frontend.test/no",
            portal_return_url="https://frontend.test/",
        )


class TestClientFailureClassification:
    async def test_a_revoked_key_is_a_rejection_not_an_outage(self, stripe_state) -> None:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_fake_stripe(stripe_state))) as http:
            client = StripeClient(
                http, api_base="https://stripe.test", secret_key="sk_test_revoked", price_id="price_x",
                checkout_success_url="https://f/", checkout_cancel_url="https://f/", portal_return_url="https://f/",
            )
            with pytest.raises(StripeRejectedError) as raised:
                await client.retrieve_subscription("sub_Abc123")

        assert raised.value.status_code == 401
        assert "sk_test_revoked" not in str(raised.value)

    @pytest.mark.parametrize("status", [500, 502, 503, 429])
    async def test_server_errors_and_rate_limits_are_unavailability(self, stripe_client, stripe_state, status) -> None:
        stripe_state.fail_retrieve_status = status

        with pytest.raises(StripeUnavailableError):
            await stripe_client.retrieve_subscription("sub_Abc123")

    async def test_an_unknown_subscription_is_a_rejection_that_names_only_the_error_code(self, stripe_client) -> None:
        with pytest.raises(StripeRejectedError) as raised:
            await stripe_client.retrieve_subscription("sub_DoesNotExist")

        assert "resource_missing" in str(raised.value)
        assert "The stand-in refused" not in str(raised.value)

    async def test_a_network_failure_is_unavailability_without_the_url(self, stripe_state) -> None:
        def explode(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError(f"cannot connect to {request.url}")

        async with httpx.AsyncClient(transport=httpx.MockTransport(explode)) as http:
            client = StripeClient(
                http, api_base="https://stripe.test", secret_key=stripe_state.secret_key, price_id="price_x",
                checkout_success_url="https://f/", checkout_cancel_url="https://f/", portal_return_url="https://f/",
            )
            with pytest.raises(StripeUnavailableError) as raised:
                await client.retrieve_subscription("sub_Abc123")

        assert "stripe.test" not in str(raised.value)

    @pytest.mark.parametrize("response", [httpx.Response(200, content=b"<html>"), httpx.Response(302, headers={"location": "https://evil/"})])
    async def test_a_non_json_or_redirecting_answer_is_unavailability(self, stripe_state, response) -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: response)) as http:
            client = StripeClient(
                http, api_base="https://stripe.test", secret_key=stripe_state.secret_key, price_id="price_x",
                checkout_success_url="https://f/", checkout_cancel_url="https://f/", portal_return_url="https://f/",
            )
            with pytest.raises(StripeUnavailableError):
                await client.retrieve_subscription("sub_Abc123")

    async def test_a_different_subscription_than_requested_is_refused(self, stripe_state) -> None:
        other = subscription_object(id="sub_Other999")
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(200, json=other))) as http:
            client = StripeClient(
                http, api_base="https://stripe.test", secret_key=stripe_state.secret_key, price_id="price_x",
                checkout_success_url="https://f/", checkout_cancel_url="https://f/", portal_return_url="https://f/",
            )
            with pytest.raises(StripeUnavailableError):
                await client.retrieve_subscription("sub_Abc123")

    @pytest.mark.parametrize("subscription_id", ["", "sub_", "../v1/customers", "sub_abc?expand[]=x", "cus_abc"])
    async def test_an_id_that_is_not_a_subscription_id_never_becomes_a_request(self, stripe_client, stripe_state, subscription_id) -> None:
        with pytest.raises(ValueError):
            await stripe_client.retrieve_subscription(subscription_id)

        assert stripe_state.request_log == []


class TestListing:
    async def test_follows_pagination_and_returns_only_aura_subscriptions(self, stripe_client, stripe_state) -> None:
        aura = [stripe_state.add_subscription(guild_id=str(1000 + index), purchaser_user_id="5000", now=NOW) for index in range(150)]
        stripe_state.add_subscription(guild_id=None, purchaser_user_id=None, now=NOW)

        found = await stripe_client.list_aura_subscription_ids()

        assert found == [subscription.id for subscription in aura]
        assert stripe_state.request_log.count("GET /v1/subscriptions") == 2
