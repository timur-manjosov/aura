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
from pydantic import SecretStr

from aura_web.stripe_api import (
    StripeClient,
    StripeRejectedError,
    StripeUnavailableError,
    SubscriptionSnapshot,
    parse_checkout_session,
    parse_subscription,
)
from fake_stripe import FakeStripeState, FakeSubscription, create_fake_stripe

NOW = 1_757_764_800
PRO_PRICE = FakeStripeState().price_id
# What every item of a Pro subscription carries besides its period.
PRO_ITEM = {"price": {"id": PRO_PRICE, "object": "price"}, "quantity": 1}


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


def parse(payload: object) -> SubscriptionSnapshot:
    return parse_subscription(payload, pro_price_id=PRO_PRICE)


class TestParseSubscription:
    def test_reads_the_dahlia_shape_with_the_period_on_the_item(self) -> None:
        snapshot = parse(subscription_object())

        assert snapshot.subscription_id == "sub_Abc123"
        assert snapshot.customer_id == "cus_Abc123"
        assert snapshot.guild_id == "1000"
        assert snapshot.purchaser_user_id == "5000"
        assert snapshot.current_period_start == NOW
        assert snapshot.current_period_end == NOW + 30 * 86400
        assert snapshot.latest_invoice_status == "paid"
        assert snapshot.collection_paused is False

    def test_several_items_use_the_earliest_end(self) -> None:
        items = {
            "data": [
                {**PRO_ITEM, "current_period_start": NOW, "current_period_end": NOW + 365 * 86400},
                {
                    **PRO_ITEM,
                    "current_period_start": NOW + 10,
                    "current_period_end": NOW + 30 * 86400,
                },
            ]
        }

        snapshot = parse(subscription_object(items=items))

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
            {
                "items": {
                    "data": [{**PRO_ITEM, "current_period_start": NOW, "current_period_end": None}]
                }
            },
            {
                "items": {
                    "data": [{**PRO_ITEM, "current_period_start": True, "current_period_end": NOW}]
                }
            },
            {
                "items": {
                    "data": [
                        {**PRO_ITEM, "current_period_start": NOW + 10, "current_period_end": NOW}
                    ]
                }
            },
            {
                "items": {
                    "data": [
                        {
                            **PRO_ITEM,
                            "current_period_start": "1757764800",
                            "current_period_end": NOW,
                        }
                    ]
                }
            },
        ],
    )
    def test_a_malformed_subscription_is_refused(self, overrides) -> None:
        with pytest.raises(StripeUnavailableError):
            parse(subscription_object(**overrides))

    @pytest.mark.parametrize("payload", [None, [], "sub_x", 1])
    def test_a_non_object_is_refused(self, payload) -> None:
        with pytest.raises(StripeUnavailableError):
            parse(payload)

    def test_an_expanded_customer_is_read_by_its_id(self) -> None:
        assert (
            parse(subscription_object(customer={"id": "cus_Expanded1"})).customer_id
            == "cus_Expanded1"
        )

    @pytest.mark.parametrize("reason", ["subscription_create", "subscription_cycle"])
    @pytest.mark.parametrize("status", ["void", "uncollectible", "open", "paid"])
    def test_the_invoice_billing_the_period_is_reported(self, reason: str, status: str) -> None:
        invoice = {"id": "in_1", "object": "invoice", "status": status, "billing_reason": reason}

        assert parse(subscription_object(latest_invoice=invoice)).latest_invoice_status == status

    @pytest.mark.parametrize(
        "reason", ["subscription_update", "manual", "subscription_threshold", None, "invented"]
    )
    def test_a_voided_invoice_that_does_not_bill_the_period_is_not_reported(self, reason) -> None:
        """An operator voiding a proration or one-off invoice must not end a paid period's Pro."""
        invoice = {"id": "in_1", "object": "invoice", "status": "void", "billing_reason": reason}

        assert parse(subscription_object(latest_invoice=invoice)).latest_invoice_status is None

    def test_an_unexpanded_invoice_leaves_its_status_unknown(self) -> None:
        assert parse(subscription_object(latest_invoice="in_123")).latest_invoice_status is None

    def test_paused_collection_is_reported(self) -> None:
        assert (
            parse(subscription_object(pause_collection={"behavior": "void"})).collection_paused
            is True
        )

    @pytest.mark.parametrize(
        "value", ["abc", "-1", "0", "9223372036854775808", "١٠٠٠", " 1000", 1000, None, ""]
    )
    def test_unusable_guild_metadata_means_not_an_aura_subscription(self, value) -> None:
        metadata = {"aura_guild_id": value} if value is not None else {}

        assert parse(subscription_object(metadata=metadata)).guild_id is None

    def test_an_unrelated_guild_id_key_is_not_mistaken_for_aura_metadata(self) -> None:
        assert parse(subscription_object(metadata={"guild_id": "1000"})).guild_id is None


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
            {
                "object": "checkout.session",
                "id": "cs_test_abc",
                "url": "https://checkout.stripe.com/c/pay/cs_test_abc",
            }
        )

        assert session.session_id == "cs_test_abc"


@pytest_asyncio.fixture
async def stripe_client(stripe_state: FakeStripeState) -> AsyncIterator[StripeClient]:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=create_fake_stripe(stripe_state)),
        base_url="https://stripe.test",
    ) as http:
        yield StripeClient(
            http,
            api_base="https://stripe.test",
            secret_key=SecretStr(stripe_state.secret_key),
            price_id=stripe_state.price_id,
            checkout_success_url="https://frontend.test/ok",
            checkout_cancel_url="https://frontend.test/no",
            portal_return_url="https://frontend.test/",
        )


class TestClientFailureClassification:
    async def test_a_revoked_key_is_a_rejection_not_an_outage(self, stripe_state) -> None:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=create_fake_stripe(stripe_state))
        ) as http:
            client = StripeClient(
                http,
                api_base="https://stripe.test",
                secret_key=SecretStr("sk_test_revoked"),
                price_id="price_x",
                checkout_success_url="https://f/",
                checkout_cancel_url="https://f/",
                portal_return_url="https://f/",
            )
            with pytest.raises(StripeRejectedError) as raised:
                await client.retrieve_subscription("sub_Abc123")

        assert raised.value.status_code == 401
        assert "sk_test_revoked" not in str(raised.value)

    @pytest.mark.parametrize("status", [500, 502, 503, 429])
    async def test_server_errors_and_rate_limits_are_unavailability(
        self, stripe_client, stripe_state, status
    ) -> None:
        stripe_state.fail_retrieve_status = status

        with pytest.raises(StripeUnavailableError):
            await stripe_client.retrieve_subscription("sub_Abc123")

    async def test_an_unknown_subscription_is_a_rejection_that_names_only_the_error_code(
        self, stripe_client
    ) -> None:
        with pytest.raises(StripeRejectedError) as raised:
            await stripe_client.retrieve_subscription("sub_DoesNotExist")

        assert "resource_missing" in str(raised.value)
        assert "The stand-in refused" not in str(raised.value)

    async def test_a_network_failure_is_unavailability_without_the_url(self, stripe_state) -> None:
        def explode(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError(f"cannot connect to {request.url}")

        async with httpx.AsyncClient(transport=httpx.MockTransport(explode)) as http:
            client = StripeClient(
                http,
                api_base="https://stripe.test",
                secret_key=SecretStr(stripe_state.secret_key),
                price_id="price_x",
                checkout_success_url="https://f/",
                checkout_cancel_url="https://f/",
                portal_return_url="https://f/",
            )
            with pytest.raises(StripeUnavailableError) as raised:
                await client.retrieve_subscription("sub_Abc123")

        assert "stripe.test" not in str(raised.value)

    @pytest.mark.parametrize(
        "response",
        [
            httpx.Response(200, content=b"<html>"),
            httpx.Response(302, headers={"location": "https://evil/"}),
        ],
    )
    async def test_a_non_json_or_redirecting_answer_is_unavailability(
        self, stripe_state, response
    ) -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: response)) as http:
            client = StripeClient(
                http,
                api_base="https://stripe.test",
                secret_key=SecretStr(stripe_state.secret_key),
                price_id="price_x",
                checkout_success_url="https://f/",
                checkout_cancel_url="https://f/",
                portal_return_url="https://f/",
            )
            with pytest.raises(StripeUnavailableError):
                await client.retrieve_subscription("sub_Abc123")

    async def test_a_different_subscription_than_requested_is_refused(self, stripe_state) -> None:
        other = subscription_object(id="sub_Other999")
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda _: httpx.Response(200, json=other))
        ) as http:
            client = StripeClient(
                http,
                api_base="https://stripe.test",
                secret_key=SecretStr(stripe_state.secret_key),
                price_id="price_x",
                checkout_success_url="https://f/",
                checkout_cancel_url="https://f/",
                portal_return_url="https://f/",
            )
            with pytest.raises(StripeUnavailableError):
                await client.retrieve_subscription("sub_Abc123")

    @pytest.mark.parametrize(
        "subscription_id", ["", "sub_", "../v1/customers", "sub_abc?expand[]=x", "cus_abc"]
    )
    async def test_an_id_that_is_not_a_subscription_id_never_becomes_a_request(
        self, stripe_client, stripe_state, subscription_id
    ) -> None:
        with pytest.raises(ValueError):
            await stripe_client.retrieve_subscription(subscription_id)

        assert stripe_state.request_log == []


class TestListing:
    async def test_follows_pagination_and_returns_only_aura_subscriptions(
        self, stripe_client, stripe_state
    ) -> None:
        aura = [
            stripe_state.add_subscription(
                guild_id=str(1000 + index), purchaser_user_id="5000", now=NOW
            )
            for index in range(150)
        ]
        stripe_state.add_subscription(guild_id=None, purchaser_user_id=None, now=NOW)

        found = await stripe_client.list_aura_subscription_ids()

        assert found == [subscription.id for subscription in aura]
        assert stripe_state.request_log.count("GET /v1/subscriptions") == 2


# --- Phase 4c audit fixes -----------------------------------------------------


def with_items(*items: dict) -> dict:
    return subscription_object(
        items={
            "object": "list",
            "data": [
                {"current_period_start": NOW, "current_period_end": NOW + 30 * 86400, **item}
                for item in items
            ],
        }
    )


def pro_item(**overrides: object) -> dict:
    return {**PRO_ITEM, **overrides}


class TestOnlyTheProPriceCounts:
    """Attack 5: a subscription with the right metadata but the wrong price or quantity (F-07)."""

    def test_one_item_on_the_pro_price_at_quantity_one_counts(self) -> None:
        assert parse(with_items(pro_item())).on_pro_price is True

    @pytest.mark.parametrize("quantity", [2, 10_000])
    def test_a_larger_quantity_still_counts(self, quantity: int) -> None:
        assert parse(with_items(pro_item(quantity=quantity))).on_pro_price is True

    def test_every_item_on_the_pro_price_counts(self) -> None:
        assert parse(with_items(pro_item(), pro_item(quantity=3))).on_pro_price is True

    def test_a_price_given_as_a_bare_id_is_read(self) -> None:
        assert parse(with_items(pro_item(price=PRO_PRICE))).on_pro_price is True

    @pytest.mark.parametrize(
        "item",
        [
            pro_item(price={"id": "price_someOtherCheaperProduct", "object": "price"}),
            pro_item(price={"id": PRO_PRICE + "x", "object": "price"}),
            pro_item(price="price_someOtherCheaperProduct"),
            pro_item(quantity=0),
            pro_item(quantity=-1),
            pro_item(quantity=None),
            pro_item(price={"id": "price_someOtherCheaperProduct", "object": "price"}, quantity=0),
        ],
        ids=[
            "another-price",
            "a-price-that-only-starts-like-pro",
            "another-price-as-bare-id",
            "quantity-zero",
            "quantity-negative",
            "metered-no-quantity",
            "another-price-at-quantity-zero",
        ],
    )
    def test_a_single_wrong_item_does_not_count(self, item: dict) -> None:
        assert parse(with_items(item)).on_pro_price is False

    @pytest.mark.parametrize("wrong_position", [0, 1, 2])
    def test_one_wrong_item_among_right_ones_does_not_count(self, wrong_position: int) -> None:
        items = [pro_item(), pro_item(), pro_item()]
        items[wrong_position] = pro_item(price={"id": "price_addOn", "object": "price"})

        assert parse(with_items(*items)).on_pro_price is False

    def test_the_metadata_does_not_make_a_wrong_price_count(self) -> None:
        wrong = with_items(pro_item(price={"id": "price_cheap", "object": "price"}))

        snapshot = parse(wrong)

        assert snapshot.guild_id == "1000"
        assert snapshot.on_pro_price is False
        assert snapshot.internal_api_payload()["on_pro_price"] is False

    @pytest.mark.parametrize(
        "item",
        [
            {"quantity": 1},
            pro_item(price=None),
            pro_item(price=5),
            pro_item(price={"object": "price"}),
            pro_item(price={"id": "prod_NotAPrice", "object": "product"}),
            pro_item(price={"id": "price_bad/../id", "object": "price"}),
            pro_item(price=""),
            pro_item(quantity=True),
            pro_item(quantity="1"),
            pro_item(quantity=1.0),
        ],
        ids=[
            "no-price",
            "null-price",
            "numeric-price",
            "price-without-id",
            "a-product-not-a-price",
            "path-in-the-id",
            "empty-price",
            "boolean-quantity",
            "string-quantity",
            "float-quantity",
        ],
    )
    def test_a_malformed_price_or_quantity_is_refused_not_guessed(self, item: dict) -> None:
        with pytest.raises(StripeUnavailableError):
            parse(with_items(item))

    def test_a_malformed_item_is_refused_even_after_a_wrong_one(self) -> None:
        with pytest.raises(StripeUnavailableError):
            parse(
                with_items(
                    pro_item(price={"id": "price_cheap", "object": "price"}),
                    pro_item(quantity="1"),
                )
            )

    def test_the_price_is_the_one_the_parser_is_given(self) -> None:
        payload = with_items(pro_item())

        assert parse_subscription(payload, pro_price_id="price_somethingElse").on_pro_price is False
        assert parse_subscription(payload, pro_price_id=PRO_PRICE).on_pro_price is True

    async def test_the_client_checks_against_its_configured_price(
        self, stripe_state: FakeStripeState
    ) -> None:
        subscription = stripe_state.add_subscription(
            guild_id="1000", purchaser_user_id="5000", now=NOW, price_id="price_cheap"
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=create_fake_stripe(stripe_state))
        ) as http:
            client = StripeClient(
                http,
                api_base="https://stripe.test",
                secret_key=SecretStr(stripe_state.secret_key),
                price_id=stripe_state.price_id,
                checkout_success_url="https://f/",
                checkout_cancel_url="https://f/",
                portal_return_url="https://f/",
            )

            snapshot = await client.retrieve_subscription(subscription.id)

        assert snapshot.on_pro_price is False


class TestCheckoutOffersCardsOnly:
    """F-05: no delayed payment method, whatever the dashboard enables."""

    async def test_card_is_the_one_and_only_payment_method_type_sent(
        self, stripe_client: StripeClient, stripe_state: FakeStripeState
    ) -> None:
        await stripe_client.create_checkout_session(
            guild_id="1000", purchaser_user_id="5000", idempotency_key="k" * 64
        )

        (form,) = stripe_state.received_forms
        assert {key: value for key, value in form.items() if "payment_method" in key} == {
            "payment_method_types[0]": "card"
        }


class TestThePortalConfiguration:
    async def _portal_form(
        self, stripe_state: FakeStripeState, configuration: str | None
    ) -> dict[str, str]:
        subscription = stripe_state.add_subscription(
            guild_id="1000", purchaser_user_id="5000", now=NOW
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=create_fake_stripe(stripe_state))
        ) as http:
            client = StripeClient(
                http,
                api_base="https://stripe.test",
                secret_key=SecretStr(stripe_state.secret_key),
                price_id=stripe_state.price_id,
                checkout_success_url="https://f/",
                checkout_cancel_url="https://f/",
                portal_return_url="https://f/back",
                portal_configuration_id=configuration,
            )
            await client.create_portal_session(customer_id=subscription.customer)
        (form,) = stripe_state.portal_sessions
        return form

    async def test_a_configured_portal_configuration_is_sent(self, stripe_state) -> None:
        form = await self._portal_form(stripe_state, "bpc_noPlanSwitching")

        assert form["configuration"] == "bpc_noPlanSwitching"
        assert form["return_url"] == "https://f/back"

    async def test_without_one_the_account_default_applies(self, stripe_state) -> None:
        form = await self._portal_form(stripe_state, None)

        assert "configuration" not in form
