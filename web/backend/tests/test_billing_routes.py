"""The billing routes: who may start a checkout or open a portal, and what a browser may learn.

The brief's third attack -- a checkout for a guild the user may not manage --
is answered here end to end: a real login, a real guild list from the Discord
stand-in, a real request carrying a guild ID the user does not manage, and the
assertion that Stripe was never contacted at all, not merely that the response
was a 403.
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from fake_bot_billing import FakeBotBillingState, free_plan
from fake_discord import FakeDiscordState
from fake_stripe import FakeStripeState
from helpers import FRONTEND_BASE, complete_login

CHECKOUT = "/api/billing/checkout"
PORTAL = "/api/billing/portal"


async def post_json(
    client: httpx.AsyncClient, path: str, body: object, **headers: str
) -> httpx.Response:
    return await client.post(
        path,
        content=json.dumps(body) if not isinstance(body, bytes) else body,
        headers={"Content-Type": "application/json", "Origin": FRONTEND_BASE, **headers},
    )


def paying_plan(*, purchaser: str, customer: str = "cus_payer", count: int = 1) -> dict:
    return {
        "tier": "pro",
        "basis": "subscription",
        "standing": "active",
        "access_until": 1_800_000_000,
        "paid_through": 1_799_740_800,
        "in_force_subscription_count": count,
        "subscriptions": [
            {
                "subscription_id": "sub_paid",
                "customer_id": customer,
                "purchaser_user_id": purchaser,
                "status": "active",
                "grants_access": True,
            }
        ],
    }


@pytest.fixture
async def moderator(
    app_client: httpx.AsyncClient, discord_state: FakeDiscordState
) -> httpx.AsyncClient:
    await complete_login(app_client, discord_state, "5000")
    return app_client


class TestCheckoutHappyPath:
    async def test_returns_a_stripe_hosted_url_bound_server_side_to_the_guild_and_the_payer(
        self, moderator, stripe_state: FakeStripeState
    ) -> None:
        response = await post_json(moderator, CHECKOUT, {"guild_id": "1000"})

        assert response.status_code == 200
        url = httpx.URL(response.json()["url"])
        assert url.scheme == "https" and url.host == "checkout.stripe.com"
        (form,) = stripe_state.received_forms
        assert form["mode"] == "subscription"
        assert form["line_items[0][price]"] == stripe_state.price_id
        assert form["line_items[0][quantity]"] == "1"
        assert form["client_reference_id"] == "1000"
        assert form["metadata[aura_guild_id]"] == "1000"
        assert form["metadata[aura_discord_user_id]"] == "5000"
        assert form["subscription_data[metadata][aura_guild_id]"] == "1000"
        assert form["subscription_data[metadata][aura_discord_user_id]"] == "5000"
        assert form["success_url"] == f"{FRONTEND_BASE}/?checkout=success"
        assert "payment_method_types[0]" not in form and not any(
            key.startswith("payment_method_types") for key in form
        )

    async def test_pins_the_api_version_and_sends_an_idempotency_key(
        self, moderator, stripe_state
    ) -> None:
        await post_json(moderator, CHECKOUT, {"guild_id": "1000"})

        headers = stripe_state.received_headers[-1]
        assert headers["stripe-version"] == "2026-08-26.dahlia"
        assert len(headers["idempotency-key"]) == 64
        assert headers["authorization"] == f"Bearer {stripe_state.secret_key}"

    async def test_a_double_click_returns_the_same_checkout_not_a_second_one(
        self, moderator, stripe_state
    ) -> None:
        first = await post_json(moderator, CHECKOUT, {"guild_id": "1000"})
        second = await post_json(moderator, CHECKOUT, {"guild_id": "1000"})

        assert first.json()["url"] == second.json()["url"]
        assert len(stripe_state.checkout_sessions) == 1

    async def test_five_simultaneous_clicks_still_open_one_checkout(
        self, moderator, stripe_state
    ) -> None:
        responses = await asyncio.gather(
            *(post_json(moderator, CHECKOUT, {"guild_id": "1000"}) for _ in range(5))
        )

        assert {response.json()["url"] for response in responses} == {responses[0].json()["url"]}
        assert len(stripe_state.checkout_sessions) == 1


class TestCheckoutAuthorization:
    @pytest.mark.parametrize(
        "guild_id",
        [
            "2000",  # the user manages it, but Aura is not in it
            "3000",  # Aura is in it, but the user is only a member
            "4000",  # does not exist anywhere
            "123456789012345678",
        ],
    )
    async def test_a_guild_the_user_may_not_manage_is_refused_before_stripe_is_contacted(
        self, moderator, stripe_state, bot_billing_state, guild_id: str
    ) -> None:
        response = await post_json(moderator, CHECKOUT, {"guild_id": guild_id})

        assert response.status_code == 403
        assert response.json() == {"error": "guild_not_manageable"}
        assert stripe_state.request_log == []
        assert "plans" not in bot_billing_state.request_log

    async def test_a_plain_member_cannot_subscribe_the_guild_they_are_in(
        self, app_client, discord_state, stripe_state
    ) -> None:
        await complete_login(app_client, discord_state, "6000")

        response = await post_json(app_client, CHECKOUT, {"guild_id": "1000"})

        assert response.status_code == 403
        assert stripe_state.request_log == []

    async def test_without_a_session_it_is_a_401_and_stripe_is_untouched(
        self, app_client, stripe_state
    ) -> None:
        response = await post_json(app_client, CHECKOUT, {"guild_id": "1000"})

        assert response.status_code == 401
        assert stripe_state.request_log == []

    async def test_permission_revoked_since_login_is_honoured_immediately(
        self, moderator, discord_state, stripe_state
    ) -> None:
        """The guild list is fetched fresh for the decision, never taken from the login moment."""
        discord_state.users["5000"].guild_permissions["1000"] = 2048

        response = await post_json(moderator, CHECKOUT, {"guild_id": "1000"})

        assert response.status_code == 403
        assert stripe_state.request_log == []

    async def test_a_discord_outage_fails_closed(
        self, moderator, discord_state, stripe_state
    ) -> None:
        discord_state.fail_user_guilds_status = 503

        response = await post_json(moderator, CHECKOUT, {"guild_id": "1000"})

        assert response.status_code == 503
        assert stripe_state.request_log == []


class TestCheckoutBodyTampering:
    @pytest.mark.parametrize(
        "body",
        [
            {"guild_id": 1000},
            {"guild_id": "01000"},
            {"guild_id": " 1000"},
            {"guild_id": "1000 "},
            {"guild_id": "١٠٠٠"},
            {"guild_id": "-1000"},
            {"guild_id": "0"},
            {"guild_id": ["1000"]},
            {"guild_id": None},
            {},
            {"guild_id": "1000", "price": "price_free"},
            {"guild_id": "1000", "purchaser_user_id": "9999"},
            {"guild_id": "1000", "metadata": {"aura_guild_id": "2000"}},
            ["1000"],
            "1000",
        ],
    )
    async def test_anything_but_exactly_one_canonical_guild_id_is_refused(
        self, moderator, stripe_state, body
    ) -> None:
        response = await post_json(moderator, CHECKOUT, body)

        assert response.status_code == 400
        assert response.json() == {"error": "invalid_request"}
        assert stripe_state.request_log == []

    @pytest.mark.parametrize(
        "raw",
        [
            b"",
            b"not json",
            b'{"guild_id": "2000", "guild_id": "1000"}',
            b'{"guild_id": NaN}',
            b"\xff\xfe",
            b"[" * 3000 + b"]" * 3000,
        ],
    )
    async def test_malformed_or_ambiguous_json_is_refused(
        self, moderator, stripe_state, raw: bytes
    ) -> None:
        response = await post_json(moderator, CHECKOUT, raw)

        assert response.status_code in (400, 413)
        assert stripe_state.request_log == []

    async def test_an_oversized_body_is_refused(self, moderator, stripe_state) -> None:
        response = await post_json(moderator, CHECKOUT, {"guild_id": "1000", "pad": "x" * 5000})

        assert response.status_code == 413
        assert stripe_state.request_log == []


class TestCheckoutCrossSite:
    @pytest.mark.parametrize(
        "content_type",
        ["application/x-www-form-urlencoded", "multipart/form-data", "text/plain", ""],
    )
    async def test_a_body_a_cross_site_form_could_send_is_refused(
        self, moderator, stripe_state, content_type
    ) -> None:
        response = await moderator.post(
            CHECKOUT,
            content=b'{"guild_id": "1000"}',
            headers={"Content-Type": content_type, "Origin": FRONTEND_BASE},
        )

        assert response.status_code == 415
        assert stripe_state.request_log == []

    @pytest.mark.parametrize(
        "origin",
        [
            "https://evil.example",
            "null",
            "https://frontend.test.evil.example",
            "http://frontend.test",
        ],
    )
    async def test_a_foreign_origin_is_refused(self, moderator, stripe_state, origin: str) -> None:
        response = await post_json(moderator, CHECKOUT, {"guild_id": "1000"}, Origin=origin)

        assert response.status_code == 403
        assert response.json() == {"error": "forbidden_origin"}
        assert stripe_state.request_log == []

    async def test_a_request_the_browser_marks_cross_site_is_refused(
        self, moderator, stripe_state
    ) -> None:
        response = await post_json(
            moderator, CHECKOUT, {"guild_id": "1000"}, **{"Sec-Fetch-Site": "cross-site"}
        )

        assert response.status_code == 403
        assert stripe_state.request_log == []


class TestCheckoutBillingState:
    async def test_a_guild_that_is_already_paying_is_refused_a_second_subscription(
        self, moderator, stripe_state, bot_billing_state: FakeBotBillingState
    ) -> None:
        bot_billing_state.plans["1000"] = paying_plan(purchaser="7777")

        response = await post_json(moderator, CHECKOUT, {"guild_id": "1000"})

        assert response.status_code == 409
        assert response.json() == {"error": "already_subscribed"}
        assert stripe_state.request_log == []

    async def test_an_unreachable_bot_fails_closed_before_stripe(
        self, moderator, stripe_state, bot_billing_state
    ) -> None:
        bot_billing_state.fail_status = 503

        response = await post_json(moderator, CHECKOUT, {"guild_id": "1000"})

        assert response.status_code == 503
        assert response.json() == {"error": "billing_unavailable"}
        assert stripe_state.request_log == []

    @pytest.mark.parametrize(
        "status, expected_status, code",
        [(500, 503, "payment_provider_unavailable"), (400, 502, "payment_provider_error")],
    )
    async def test_stripe_failures_are_codes_not_relayed_errors(
        self, moderator, stripe_state, status, expected_status, code
    ) -> None:
        stripe_state.fail_checkout_status = status

        response = await post_json(moderator, CHECKOUT, {"guild_id": "1000"})

        assert response.status_code == expected_status
        assert response.json() == {"error": code}

    @pytest.mark.parametrize(
        "override",
        [
            "https://evil.example/c/pay/cs_test_x",
            "http://checkout.stripe.com/c/pay/cs_test_x",
            "https://checkout.stripe.com.evil.example/c/pay",
            "https://user:pass@checkout.stripe.com/c/pay",
            "javascript:alert(1)",
        ],
    )
    async def test_a_checkout_url_not_on_stripe_is_never_handed_to_the_browser(
        self, moderator, stripe_state, override
    ) -> None:
        stripe_state.checkout_url_override = override

        response = await post_json(moderator, CHECKOUT, {"guild_id": "1000"})

        assert response.status_code == 503
        assert override not in response.text


class TestPortal:
    async def test_the_payer_gets_a_billing_portal_session_for_their_own_customer(
        self, moderator, stripe_state, bot_billing_state
    ) -> None:
        subscription = stripe_state.add_subscription(
            guild_id="1000", purchaser_user_id="5000", now=0
        )
        bot_billing_state.plans["1000"] = paying_plan(
            purchaser="5000", customer=subscription.customer
        )

        response = await post_json(moderator, PORTAL, {"guild_id": "1000"})

        assert response.status_code == 200
        assert httpx.URL(response.json()["url"]).host == "billing.stripe.com"
        assert stripe_state.portal_sessions[-1]["customer"] == subscription.customer
        assert stripe_state.portal_sessions[-1]["return_url"] == f"{FRONTEND_BASE}/"

    async def test_another_admin_of_the_same_guild_cannot_open_the_payers_portal(
        self, moderator, stripe_state, bot_billing_state
    ) -> None:
        bot_billing_state.plans["1000"] = paying_plan(purchaser="7777")

        response = await post_json(moderator, PORTAL, {"guild_id": "1000"})

        assert response.status_code == 403
        assert response.json() == {"error": "not_billing_owner"}
        assert stripe_state.request_log == []

    async def test_no_subscription_and_someone_elses_subscription_are_indistinguishable(
        self, moderator, bot_billing_state
    ) -> None:
        bot_billing_state.plans["1000"] = free_plan()
        bot_billing_state.plans["3000"] = paying_plan(purchaser="7777")

        none = await post_json(moderator, PORTAL, {"guild_id": "1000"})
        someone_elses = await post_json(moderator, PORTAL, {"guild_id": "3000"})

        assert (none.status_code, none.content) == (
            someone_elses.status_code,
            someone_elses.content,
        )

    async def test_a_payer_who_lost_manage_permission_can_still_reach_their_own_billing(
        self, moderator, discord_state, stripe_state, bot_billing_state
    ) -> None:
        subscription = stripe_state.add_subscription(
            guild_id="1000", purchaser_user_id="5000", now=0
        )
        bot_billing_state.plans["1000"] = paying_plan(
            purchaser="5000", customer=subscription.customer
        )
        discord_state.users["5000"].guild_permissions["1000"] = 2048

        response = await post_json(moderator, PORTAL, {"guild_id": "1000"})

        assert response.status_code == 200

    async def test_the_portal_refuses_cross_site_requests_too(
        self, moderator, stripe_state
    ) -> None:
        response = await post_json(
            moderator, PORTAL, {"guild_id": "1000"}, Origin="https://evil.example"
        )

        assert response.status_code == 403
        assert stripe_state.request_log == []

    async def test_without_a_session_it_is_a_401(self, app_client) -> None:
        assert (await post_json(app_client, PORTAL, {"guild_id": "1000"})).status_code == 401


class TestBillingGuilds:
    async def test_returns_plans_only_for_the_dashboards_guilds_and_nothing_private(
        self, moderator, bot_billing_state
    ) -> None:
        bot_billing_state.plans["1000"] = paying_plan(
            purchaser="5000", customer="cus_secretCustomer"
        )
        bot_billing_state.plans["3000"] = paying_plan(purchaser="5000")

        response = await moderator.get("/api/billing/guilds")

        assert response.status_code == 200
        body = response.json()
        assert [entry["id"] for entry in body] == ["1000"]
        plan = body[0]["plan"]
        assert plan["tier"] == "pro"
        assert plan["is_billing_owner"] is True
        assert plan["can_subscribe"] is False
        assert "cus_secretCustomer" not in response.text
        assert "sub_paid" not in response.text
        assert "purchaser" not in response.text

    async def test_a_guild_with_no_subscription_can_subscribe(self, moderator) -> None:
        plan = (await moderator.get("/api/billing/guilds")).json()[0]["plan"]

        assert plan["can_subscribe"] is True
        assert plan["is_billing_owner"] is False

    async def test_an_unreachable_bot_is_a_503(self, moderator, bot_billing_state) -> None:
        bot_billing_state.fail_status = 500

        assert (await moderator.get("/api/billing/guilds")).status_code == 503

    async def test_a_bot_answer_missing_a_guild_is_treated_as_unavailable(
        self, moderator, bot_billing_state
    ) -> None:
        async def drop_plans(_):  # pragma: no cover - replaced below
            return None

        bot_billing_state.plans["1000"] = {"tier": "pro"}

        assert (await moderator.get("/api/billing/guilds")).status_code == 503

    async def test_without_a_session_it_is_a_401(self, app_client) -> None:
        assert (await app_client.get("/api/billing/guilds")).status_code == 401
