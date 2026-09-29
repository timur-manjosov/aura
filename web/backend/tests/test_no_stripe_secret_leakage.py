"""No Stripe or internal-API secret reaches a browser or a log line. Checked on the wire and in the logs.

The same method as test_no_token_leakage.py applied to Phase 4c's secrets: run
every billing flow that exists -- a checkout, a portal session, the plan list,
genuine and forged webhooks, every failure path -- capture the raw bytes of
every response and every log record emitted at any level, and search both for
the actual secret values. Searching for values rather than field names is what
keeps this test honest when a future handler adds a debug header or echoes an
upstream error.
"""
from __future__ import annotations

import json
import logging

import httpx
from fake_bot_billing import FakeBotBillingState
from fake_discord import FakeDiscordState
from fake_stripe import FakeStripeState, sign_webhook
from helpers import FRONTEND_BASE, complete_login
from test_no_token_leakage import raw_response_bytes


async def test_no_billing_flow_leaks_a_secret_to_the_wire_or_the_logs(
    app_client: httpx.AsyncClient,
    discord_state: FakeDiscordState,
    stripe_state: FakeStripeState,
    bot_billing_state: FakeBotBillingState,
    caplog,
) -> None:
    secrets = {
        "stripe_secret_key": stripe_state.secret_key,
        "stripe_webhook_secret": stripe_state.webhook_secret,
        "bot_internal_api_secret": bot_billing_state.secret,
        "discord_client_secret": discord_state.client_secret,
        "discord_bot_token": discord_state.bot_token,
    }
    json_headers = {"Content-Type": "application/json", "Origin": FRONTEND_BASE}
    responses: list[httpx.Response] = []

    with caplog.at_level(logging.DEBUG):
        responses.append(await complete_login(app_client, discord_state, "5000"))
        responses.append(await app_client.get("/api/billing/guilds"))
        responses.append(await app_client.post("/api/billing/checkout", content=b'{"guild_id": "1000"}', headers=json_headers))
        responses.append(await app_client.post("/api/billing/checkout", content=b'{"guild_id": "2000"}', headers=json_headers))

        session_id = next(iter(stripe_state.checkout_sessions))
        subscription = stripe_state.complete_checkout(session_id, now=1_757_764_800)
        body, signature = stripe_state.signed(stripe_state.checkout_completed_event(session_id))
        responses.append(await app_client.post("/api/stripe/webhook", content=body, headers={"Stripe-Signature": signature}))
        responses.append(await app_client.post("/api/stripe/webhook", content=body, headers={"Stripe-Signature": signature}))
        responses.append(await app_client.post("/api/stripe/webhook", content=body, headers={"Stripe-Signature": sign_webhook(body, "whsec_wrong")}))
        responses.append(await app_client.post("/api/stripe/webhook", content=body))

        bot_billing_state.plans["1000"] = {
            "tier": "pro", "basis": "subscription", "standing": "active", "access_until": 1, "paid_through": 1,
            "in_force_subscription_count": 1,
            "subscriptions": [{"subscription_id": subscription.id, "customer_id": subscription.customer, "purchaser_user_id": "5000", "status": "active", "grants_access": True}],
        }
        responses.append(await app_client.post("/api/billing/portal", content=b'{"guild_id": "1000"}', headers=json_headers))

        stripe_state.fail_checkout_status = 400
        bot_billing_state.plans.pop("1000")
        responses.append(await app_client.post("/api/billing/checkout", content=b'{"guild_id": "1000"}', headers=json_headers))
        stripe_state.fail_retrieve_status = 500
        responses.append(await app_client.post("/api/stripe/webhook", content=body.replace(b"evt_", b"evt_x"), headers={"Stripe-Signature": sign_webhook(body.replace(b"evt_", b"evt_x"), stripe_state.webhook_secret)}))
        bot_billing_state.secret = "rotated-on-one-side-only-" + "0" * 30
        stripe_state.fail_retrieve_status = None
        responses.append(await app_client.get("/api/billing/guilds"))

    assert len(responses) == 12
    assert any(response.status_code == 200 and "checkout.stripe.com" in response.text for response in responses)
    for response in responses:
        wire = raw_response_bytes(response)
        for label, secret in secrets.items():
            assert secret.encode() not in wire, f"{label} leaked in a response to {response.request.url}"

    logged = "\n".join(
        record.getMessage() + (logging.Formatter().formatException(record.exc_info) if record.exc_info else "")
        for record in caplog.records
    )
    assert logged, "the flows must have logged something for this scan to mean anything"
    for label, secret in secrets.items():
        assert secret not in logged, f"{label} was written to a log record"


async def test_the_browser_facing_plan_never_carries_stripe_identifiers(
    app_client: httpx.AsyncClient, discord_state: FakeDiscordState, bot_billing_state: FakeBotBillingState
) -> None:
    await complete_login(app_client, discord_state, "5000")
    bot_billing_state.plans["1000"] = {
        "tier": "pro", "basis": "subscription", "standing": "active", "access_until": 1, "paid_through": 1,
        "in_force_subscription_count": 1,
        "subscriptions": [{"subscription_id": "sub_Private1", "customer_id": "cus_Private1", "purchaser_user_id": "7777", "status": "active", "grants_access": True}],
    }

    body = (await app_client.get("/api/billing/guilds")).text

    for private in ("sub_Private1", "cus_Private1", "7777"):
        assert private not in body
    assert json.loads(body)[0]["plan"]["is_billing_owner"] is False
