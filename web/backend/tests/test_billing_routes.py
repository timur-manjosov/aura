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
import logging

import httpx
import pytest

from aura_web.config import WebSettings
from fake_bot_billing import FakeBotBillingState, create_fake_bot_billing, free_plan
from fake_discord import FakeDiscordState, create_fake_discord
from fake_stripe import FakeStripeState, create_fake_stripe
from helpers import FAKE_BOT_BASE, FAKE_STRIPE_BASE, FRONTEND_BASE, build_app, complete_login

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
        # Cards only, sent explicitly (F-05): the dashboard's payment method
        # settings cannot add a delayed method that grants Pro before it pays.
        assert {key: value for key, value in form.items() if "payment_method" in key} == {
            "payment_method_types[0]": "card"
        }

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


# --- Phase 4c audit fixes -----------------------------------------------------


def unpaid_plan(**overrides: object) -> dict:
    """A guild on Pro with no subscription behind it."""
    return {
        "tier": "pro",
        "basis": "subscription",
        "standing": "no_subscription",
        "access_until": None,
        "paid_through": None,
        "in_force_subscription_count": 0,
        "subscriptions": [],
    } | overrides


class TestSubscribingIsOfferedOnlyWhereItChangesSomething:
    """F-15: no "Upgrade to Pro" where paying buys nothing."""

    @pytest.mark.parametrize("basis", ["billing_not_enforced", "complimentary"])
    async def test_a_guild_already_on_pro_by_another_basis_is_not_offered_a_checkout(
        self, moderator, bot_billing_state, basis: str
    ) -> None:
        bot_billing_state.plans["1000"] = unpaid_plan(basis=basis)

        plan = (await moderator.get("/api/billing/guilds")).json()[0]["plan"]

        assert plan["can_subscribe"] is False

    async def test_a_guild_whose_plan_a_subscription_decides_is_offered_one(
        self, moderator, bot_billing_state
    ) -> None:
        bot_billing_state.plans["1000"] = unpaid_plan(tier="free", standing="ended")

        plan = (await moderator.get("/api/billing/guilds")).json()[0]["plan"]

        assert plan["can_subscribe"] is True

    async def test_a_complimentary_guild_that_is_also_paying_is_not_offered_a_second(
        self, moderator, bot_billing_state
    ) -> None:
        bot_billing_state.plans["1000"] = paying_plan(purchaser="5000") | {"basis": "complimentary"}

        plan = (await moderator.get("/api/billing/guilds")).json()[0]["plan"]

        assert plan["can_subscribe"] is False
        assert plan["is_billing_owner"] is True


class TestPaymentPendingReachesTheBrowser:
    async def test_the_standing_is_passed_through_without_a_paid_through_date(
        self, moderator, bot_billing_state
    ) -> None:
        bot_billing_state.plans["1000"] = paying_plan(purchaser="5000") | {
            "standing": "payment_pending",
            "paid_through": None,
        }

        plan = (await moderator.get("/api/billing/guilds")).json()[0]["plan"]

        assert plan["standing"] == "payment_pending"
        assert plan["paid_through"] is None
        assert plan["tier"] == "pro"


class TestThePortalConfigurationReachesStripe:
    async def test_the_configured_portal_configuration_is_what_the_route_opens(
        self,
        web_settings: WebSettings,
        discord_state: FakeDiscordState,
        stripe_state: FakeStripeState,
        bot_billing_state: FakeBotBillingState,
    ) -> None:
        configured = web_settings.model_copy(
            update={"stripe_portal_configuration_id": "bpc_noPlanSwitching"}
        )
        subscription = stripe_state.add_subscription(
            guild_id="1000", purchaser_user_id="5000", now=0
        )
        bot_billing_state.plans["1000"] = paying_plan(
            purchaser="5000", customer=subscription.customer
        )
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
            app = build_app(configured, discord_http, stripe_http, bot_http)
            async with app.router.lifespan_context(app):
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app), base_url="https://testserver"
                ) as browser:
                    await complete_login(browser, discord_state, "5000")
                    response = await post_json(browser, PORTAL, {"guild_id": "1000"})

        assert response.status_code == 200
        assert stripe_state.portal_sessions[-1]["configuration"] == "bpc_noPlanSwitching"

    async def test_without_one_no_configuration_is_sent(
        self, moderator, stripe_state, bot_billing_state
    ) -> None:
        subscription = stripe_state.add_subscription(
            guild_id="1000", purchaser_user_id="5000", now=0
        )
        bot_billing_state.plans["1000"] = paying_plan(
            purchaser="5000", customer=subscription.customer
        )

        await post_json(moderator, PORTAL, {"guild_id": "1000"})

        assert "configuration" not in stripe_state.portal_sessions[-1]


# --- Enforced-billing test: no checkout where paying buys nothing --------------


class TestNothingToBuy:
    """A checkout only where a subscription decides the plan; every refusal before Stripe."""

    @pytest.mark.parametrize("basis", ["billing_not_enforced", "complimentary"])
    async def test_a_guild_whose_plan_no_subscription_decides_is_refused_before_stripe(
        self, moderator, stripe_state, bot_billing_state, basis: str
    ) -> None:
        bot_billing_state.plans["1000"] = unpaid_plan(basis=basis)

        response = await post_json(moderator, CHECKOUT, {"guild_id": "1000"})

        assert response.status_code == 409
        assert response.json() == {"error": "nothing_to_buy"}
        assert stripe_state.request_log == []
        assert stripe_state.checkout_sessions == {}

    @pytest.mark.parametrize("basis", ["billing_not_enforced", "complimentary"])
    async def test_it_is_refused_on_that_basis_even_when_someone_is_already_paying(
        self, moderator, stripe_state, bot_billing_state, basis: str
    ) -> None:
        bot_billing_state.plans["1000"] = paying_plan(purchaser="7777") | {"basis": basis}

        response = await post_json(moderator, CHECKOUT, {"guild_id": "1000"})

        assert response.status_code == 409
        assert response.json() == {"error": "nothing_to_buy"}
        assert stripe_state.request_log == []

    @pytest.mark.parametrize("standing", ["no_subscription", "ended"])
    async def test_under_enforcement_a_free_guild_still_gets_its_checkout(
        self, moderator, stripe_state, bot_billing_state, standing: str
    ) -> None:
        bot_billing_state.plans["1000"] = unpaid_plan(tier="free", standing=standing)

        response = await post_json(moderator, CHECKOUT, {"guild_id": "1000"})

        assert response.status_code == 200
        assert httpx.URL(response.json()["url"]).host == "checkout.stripe.com"
        assert len(stripe_state.checkout_sessions) == 1

    async def test_under_enforcement_a_paying_guild_is_still_already_subscribed(
        self, moderator, stripe_state, bot_billing_state
    ) -> None:
        bot_billing_state.plans["1000"] = paying_plan(purchaser="7777")

        response = await post_json(moderator, CHECKOUT, {"guild_id": "1000"})

        assert (response.status_code, response.json()) == (409, {"error": "already_subscribed"})
        assert stripe_state.request_log == []

    async def test_a_guild_the_user_cannot_manage_learns_nothing_about_its_basis(
        self, moderator, stripe_state, bot_billing_state
    ) -> None:
        """Aura is in 3000 but the user is only a member: 403 first, the bot never asked."""
        bot_billing_state.plans["3000"] = unpaid_plan(basis="billing_not_enforced")

        response = await post_json(moderator, CHECKOUT, {"guild_id": "3000"})

        assert (response.status_code, response.json()) == (403, {"error": "guild_not_manageable"})
        assert "plans" not in bot_billing_state.request_log
        assert stripe_state.request_log == []

    @pytest.mark.parametrize(
        "break_the_bot",
        [
            pytest.param(lambda state: setattr(state, "fail_status", 503), id="bot-5xx"),
            pytest.param(lambda state: setattr(state, "secret", "x" * 48), id="secret-refused"),
            pytest.param(
                lambda state: state.plans.__setitem__("1000", unpaid_plan(basis="enforced")),
                id="unknown-basis",
            ),
            pytest.param(
                lambda state: state.plans.__setitem__("1000", unpaid_plan(basis=None)),
                id="no-basis",
            ),
        ],
    )
    async def test_when_the_bot_cannot_say_it_fails_closed_rather_than_assuming_enforcement(
        self, moderator, stripe_state, bot_billing_state, break_the_bot
    ) -> None:
        break_the_bot(bot_billing_state)

        response = await post_json(moderator, CHECKOUT, {"guild_id": "1000"})

        assert (response.status_code, response.json()) == (503, {"error": "billing_unavailable"})
        assert stripe_state.request_log == []

    async def test_an_unreachable_bot_fails_closed_before_stripe(
        self,
        web_settings: WebSettings,
        discord_state: FakeDiscordState,
        stripe_state: FakeStripeState,
    ) -> None:
        def connection_refused(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection refused", request=request)

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
                transport=httpx.MockTransport(connection_refused), base_url=FAKE_BOT_BASE
            ) as bot_http,
        ):
            app = build_app(web_settings, discord_http, stripe_http, bot_http)
            async with app.router.lifespan_context(app):
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app), base_url="https://testserver"
                ) as browser:
                    await complete_login(browser, discord_state, "5000")
                    response = await post_json(browser, CHECKOUT, {"guild_id": "1000"})

        assert (response.status_code, response.json()) == (503, {"error": "billing_unavailable"})
        assert stripe_state.request_log == []

    async def test_the_bot_is_asked_afresh_for_every_checkout(
        self, moderator, stripe_state, bot_billing_state
    ) -> None:
        """Enforcement switched off between two clicks: the second is refused, not served stale."""
        first = await post_json(moderator, CHECKOUT, {"guild_id": "1000"})
        bot_billing_state.plans["1000"] = unpaid_plan(basis="billing_not_enforced")
        second = await post_json(moderator, CHECKOUT, {"guild_id": "1000"})

        assert first.status_code == 200
        assert (second.status_code, second.json()) == (409, {"error": "nothing_to_buy"})
        assert len(stripe_state.checkout_sessions) == 1
        assert bot_billing_state.request_log.count("plans") == 2

    @pytest.mark.parametrize("basis", ["billing_not_enforced", "complimentary"])
    async def test_the_payer_can_still_open_the_portal_on_that_basis(
        self, moderator, stripe_state, bot_billing_state, basis: str
    ) -> None:
        subscription = stripe_state.add_subscription(
            guild_id="1000", purchaser_user_id="5000", now=0
        )
        bot_billing_state.plans["1000"] = paying_plan(
            purchaser="5000", customer=subscription.customer
        ) | {"basis": basis}

        response = await post_json(moderator, PORTAL, {"guild_id": "1000"})

        assert response.status_code == 200
        assert httpx.URL(response.json()["url"]).host == "billing.stripe.com"

    @pytest.mark.parametrize("basis", ["billing_not_enforced", "complimentary", "subscription"])
    @pytest.mark.parametrize("paying", [False, True])
    async def test_the_route_refuses_exactly_what_the_dashboard_does_not_offer(
        self, moderator, stripe_state, bot_billing_state, basis: str, paying: bool
    ) -> None:
        plan = paying_plan(purchaser="7777") if paying else unpaid_plan(tier="free")
        bot_billing_state.plans["1000"] = plan | {"basis": basis}

        offered = (await moderator.get("/api/billing/guilds")).json()[0]["plan"]["can_subscribe"]
        response = await post_json(moderator, CHECKOUT, {"guild_id": "1000"})

        assert offered is (response.status_code == 200)
        assert offered is (len(stripe_state.checkout_sessions) == 1)

    async def test_the_refusal_is_logged_with_the_basis_and_nothing_else_reaches_stripe(
        self, moderator, stripe_state, bot_billing_state, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level(logging.WARNING, logger="aura_web.routes.billing")
        bot_billing_state.plans["1000"] = unpaid_plan(basis="complimentary")

        await post_json(moderator, CHECKOUT, {"guild_id": "1000"})

        (line,) = [r.getMessage() for r in caplog.records if r.name == "aura_web.routes.billing"]
        assert "guild 1000" in line and "user 5000" in line and "complimentary" in line
