"""POST /api/stripe/webhook: the signature check is absolute, and processing is idempotent and race-free.

The brief names this endpoint's signature verification as the control to be
examined most intensively in the whole of Phase 4c, so the first class below
attacks it from every direction a request can arrive from -- no header, an
empty one, a wrong one, a right one for a different body, a right one that has
expired, two headers, a body that is not text -- and asserts the same three
things every time: the request is refused, the bot is never contacted, and
Stripe is never asked about the subscription. "No state change" is proven by
the absence of any call that could have caused one.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time

import httpx
import pytest

from fake_bot_billing import FakeBotBillingState
from fake_stripe import FakeStripeState, sign_webhook

WEBHOOK = "/api/stripe/webhook"
NOW = int(time.time())


async def deliver(
    client: httpx.AsyncClient, body: bytes, signature: str | None, **extra_headers: str
) -> httpx.Response:
    headers = {"Content-Type": "application/json", **extra_headers}
    if signature is not None:
        headers["Stripe-Signature"] = signature
    return await client.post(WEBHOOK, content=body, headers=headers)


def aura_subscription(stripe_state: FakeStripeState, **overrides):
    return stripe_state.add_subscription(
        guild_id="1000", purchaser_user_id="5000", now=NOW, **overrides
    )


def nothing_was_touched(stripe_state: FakeStripeState, bot_state: FakeBotBillingState) -> bool:
    return (
        bot_state.request_log == [] and stripe_state.request_log == [] and bot_state.snapshots == {}
    )


class TestTheSignatureIsAbsolute:
    @pytest.fixture
    def event_body(self, stripe_state: FakeStripeState) -> bytes:
        subscription = aura_subscription(stripe_state)
        return json.dumps(
            stripe_state.subscription_event("customer.subscription.created", subscription.id)
        ).encode()

    async def test_a_missing_signature_is_refused_without_any_state_change(
        self, app_client, stripe_state, bot_billing_state, event_body
    ) -> None:
        response = await deliver(app_client, event_body, None)

        assert response.status_code == 400
        assert response.json() == {"error": "invalid_signature"}
        assert nothing_was_touched(stripe_state, bot_billing_state)

    @pytest.mark.parametrize("signature", ["", " ", "\t"])
    async def test_an_empty_signature_is_refused_without_any_state_change(
        self, app_client, stripe_state, bot_billing_state, event_body, signature
    ) -> None:
        response = await deliver(app_client, event_body, signature)

        assert response.status_code == 400
        assert response.json() == {"error": "invalid_signature"}
        assert nothing_was_touched(stripe_state, bot_billing_state)

    async def test_a_signature_made_with_the_wrong_secret_is_refused(
        self, app_client, stripe_state, bot_billing_state, event_body
    ) -> None:
        forged = sign_webhook(event_body, "whsec_attackerGuessedSecret")

        response = await deliver(app_client, event_body, forged)

        assert response.status_code == 400
        assert nothing_was_touched(stripe_state, bot_billing_state)

    async def test_a_valid_signature_over_a_different_body_is_refused(
        self, app_client, stripe_state, bot_billing_state, event_body
    ) -> None:
        """The attacker keeps Stripe's genuine header and swaps the body underneath it."""
        genuine_signature = sign_webhook(event_body, stripe_state.webhook_secret)
        tampered = event_body.replace(b'"1000"', b'"2000"')
        assert tampered != event_body

        response = await deliver(app_client, tampered, genuine_signature)

        assert response.status_code == 400
        assert nothing_was_touched(stripe_state, bot_billing_state)

    async def test_whitespace_added_to_a_signed_body_is_refused(
        self, app_client, stripe_state, bot_billing_state, event_body
    ) -> None:
        """Verification is over exact bytes -- a re-serialised body is a different body."""
        signature = sign_webhook(event_body, stripe_state.webhook_secret)

        response = await deliver(app_client, event_body + b"\n", signature)

        assert response.status_code == 400
        assert nothing_was_touched(stripe_state, bot_billing_state)

    async def test_a_correctly_signed_request_older_than_five_minutes_is_refused(
        self, app_client, stripe_state, bot_billing_state, event_body
    ) -> None:
        """A captured request replayed later: the signature is genuine, the moment is not."""
        stale = sign_webhook(
            event_body, stripe_state.webhook_secret, timestamp=int(time.time()) - 301
        )

        response = await deliver(app_client, event_body, stale)

        assert response.status_code == 400
        assert nothing_was_touched(stripe_state, bot_billing_state)

    @pytest.mark.parametrize(
        "mangle",
        [
            lambda header: header.replace("v1=", "v0="),
            lambda header: header.split(",")[1],
            lambda header: header.split(",")[0],
            lambda header: header.replace("t=", "t=abc"),
            lambda header: header.replace("t=", "t=99999999999999999999999"),
            lambda header: "v1=" + header.split("v1=")[1] + ",t=" + header.split(",")[0][2:] + "x",
            lambda header: header.upper(),
            lambda header: header + "\x00",
        ],
    )
    async def test_a_malformed_or_downgraded_header_is_refused(
        self, app_client, stripe_state, bot_billing_state, event_body, mangle
    ) -> None:
        header = mangle(sign_webhook(event_body, stripe_state.webhook_secret))

        try:
            response = await deliver(app_client, event_body, header)
        except (httpx.LocalProtocolError, ValueError):
            # A header the HTTP client itself refuses to send never arrives,
            # which is the same outcome from the server's side.
            assert nothing_was_touched(stripe_state, bot_billing_state)
            return

        assert response.status_code == 400
        assert nothing_was_touched(stripe_state, bot_billing_state)

    async def test_two_signature_headers_are_refused_even_if_one_is_genuine(
        self, app_client, stripe_state, bot_billing_state, event_body
    ) -> None:
        genuine = sign_webhook(event_body, stripe_state.webhook_secret)

        response = await app_client.post(
            WEBHOOK,
            content=event_body,
            headers=[
                ("Content-Type", "application/json"),
                ("Stripe-Signature", "t=1,v1=00"),
                ("Stripe-Signature", genuine),
            ],
        )

        assert response.status_code == 400
        assert nothing_was_touched(stripe_state, bot_billing_state)

    async def test_a_signed_body_that_is_not_utf8_is_refused_as_a_signature_failure(
        self, app_client, stripe_state, bot_billing_state
    ) -> None:
        body = b"\xff\xfe{}"
        response = await deliver(app_client, body, sign_webhook(body, stripe_state.webhook_secret))

        assert response.status_code == 400
        assert nothing_was_touched(stripe_state, bot_billing_state)

    async def test_an_oversized_body_is_refused_before_verification(
        self, app_client, stripe_state, bot_billing_state
    ) -> None:
        body = b"{" + b" " * (1024 * 1024 + 1) + b"}"

        response = await deliver(app_client, body, sign_webhook(body, stripe_state.webhook_secret))

        assert response.status_code == 413
        assert nothing_was_touched(stripe_state, bot_billing_state)

    async def test_a_rejected_signature_logs_neither_the_header_nor_the_body(
        self, app_client, stripe_state, event_body, caplog
    ) -> None:
        forged = sign_webhook(event_body, "whsec_attackerGuessedSecret")

        with caplog.at_level(logging.DEBUG):
            await deliver(app_client, event_body, forged)

        logged = "\n".join(record.getMessage() for record in caplog.records)
        assert forged not in logged
        assert "customer.subscription.created" not in logged


class TestVerifiedButUnusable:
    @pytest.mark.parametrize(
        "body",
        [
            b"not json",
            b"[]",
            b'{"object": "event"}',
            b'{"object": "event", "id": "evt_1", "type": "customer.subscription.created", "livemode": false}',
            b'{"object": "event", "id": "evt_1", "id": "evt_2", "type": "invoice.paid", "livemode": false, "data": {"object": {}}}',
            b'{"object": "event", "id": "ch_1", "type": "invoice.paid", "livemode": false, "data": {"object": {}}}',
            b'{"object": "event", "id": "evt_1", "type": "invoice.paid", "livemode": "false", "data": {"object": {}}}',
        ],
    )
    async def test_a_correctly_signed_malformed_event_is_refused_without_state_change(
        self, app_client, stripe_state, bot_billing_state, body
    ) -> None:
        response = await deliver(app_client, body, sign_webhook(body, stripe_state.webhook_secret))

        assert response.status_code == 400
        assert response.json() == {"error": "invalid_event"}
        assert nothing_was_touched(stripe_state, bot_billing_state)

    async def test_a_live_event_reaching_a_test_mode_deployment_is_refused(
        self, app_client, stripe_state, bot_billing_state
    ) -> None:
        subscription = aura_subscription(stripe_state)
        event = stripe_state.subscription_event("customer.subscription.updated", subscription.id)
        event["livemode"] = True
        stripe_state.request_log.clear()

        body, signature = stripe_state.signed(event)
        response = await deliver(app_client, body, signature)

        assert response.status_code == 400
        assert response.json() == {"error": "livemode_mismatch"}
        assert nothing_was_touched(stripe_state, bot_billing_state)


class TestProcessing:
    async def test_a_subscription_event_stores_what_stripe_says_now(
        self, app_client, stripe_state, bot_billing_state
    ) -> None:
        subscription = aura_subscription(stripe_state)
        body, signature = stripe_state.signed(
            stripe_state.subscription_event("customer.subscription.created", subscription.id)
        )

        response = await deliver(app_client, body, signature)

        assert response.status_code == 200
        assert response.json() == {"status": "applied"}
        stored = bot_billing_state.snapshots[subscription.id]
        assert stored["guild_id"] == "1000"
        assert stored["purchaser_user_id"] == "5000"
        assert stored["status"] == "active"
        assert stored["latest_invoice_status"] == "paid"

    async def test_the_event_payload_is_never_used_as_state(
        self, app_client, stripe_state, bot_billing_state
    ) -> None:
        """An event claiming "active" for a subscription Stripe now reports canceled stores canceled."""
        subscription = aura_subscription(stripe_state)
        event = stripe_state.subscription_event("customer.subscription.updated", subscription.id)
        subscription.status = "canceled"

        body, signature = stripe_state.signed(event)
        await deliver(app_client, body, signature)

        assert bot_billing_state.snapshots[subscription.id]["status"] == "canceled"

    async def test_checkout_completed_resolves_the_subscription_it_created(
        self, app_client, stripe_state, bot_billing_state
    ) -> None:
        subscription = aura_subscription(stripe_state)
        event = stripe_state.event(
            "checkout.session.completed",
            {
                "object": "checkout.session",
                "id": "cs_test_x",
                "mode": "subscription",
                "subscription": subscription.id,
            },
        )

        body, signature = stripe_state.signed(event)
        response = await deliver(app_client, body, signature)

        assert response.json() == {"status": "applied"}
        assert subscription.id in bot_billing_state.snapshots

    @pytest.mark.parametrize("legacy_shape", [False, True])
    async def test_invoice_events_resolve_their_subscription_in_both_api_shapes(
        self, app_client, stripe_state, bot_billing_state, legacy_shape
    ) -> None:
        subscription = aura_subscription(
            stripe_state, status="past_due", latest_invoice_status="open"
        )
        event = stripe_state.invoice_event(
            "invoice.payment_failed", subscription.id, legacy_shape=legacy_shape
        )

        body, signature = stripe_state.signed(event)
        response = await deliver(app_client, body, signature)

        assert response.json() == {"status": "applied"}
        assert bot_billing_state.snapshots[subscription.id]["status"] == "past_due"

    @pytest.mark.parametrize(
        "event_type, data_object",
        [
            ("charge.succeeded", {"object": "charge", "id": "ch_1"}),
            (
                "checkout.session.completed",
                {
                    "object": "checkout.session",
                    "id": "cs_test_1",
                    "mode": "payment",
                    "subscription": None,
                },
            ),
            ("invoice.paid", {"object": "invoice", "id": "in_1", "parent": None}),
            ("customer.subscription.updated", {"object": "charge", "id": "sub_disguised"}),
        ],
    )
    async def test_an_event_concerning_no_subscription_is_acknowledged_without_action(
        self, app_client, stripe_state, bot_billing_state, event_type, data_object
    ) -> None:
        body, signature = stripe_state.signed(stripe_state.event(event_type, data_object))

        response = await deliver(app_client, body, signature)

        assert response.status_code == 200
        assert response.json() == {"status": "ignored"}
        assert nothing_was_touched(stripe_state, bot_billing_state)

    async def test_a_subscription_without_aura_metadata_is_not_pushed_to_the_bot(
        self, app_client, stripe_state, bot_billing_state
    ) -> None:
        foreign = stripe_state.add_subscription(guild_id=None, purchaser_user_id=None, now=NOW)
        body, signature = stripe_state.signed(
            stripe_state.subscription_event("customer.subscription.created", foreign.id)
        )

        response = await deliver(app_client, body, signature)

        assert response.json() == {"status": "not_aura"}
        assert "apply" not in bot_billing_state.request_log


class TestIdempotency:
    async def test_the_same_event_delivered_twice_is_applied_once(
        self, app_client, stripe_state, bot_billing_state
    ) -> None:
        subscription = aura_subscription(stripe_state)
        event = stripe_state.subscription_event("customer.subscription.created", subscription.id)

        first = await deliver(app_client, *stripe_state.signed(event))
        second = await deliver(app_client, *stripe_state.signed(event))

        assert first.json() == {"status": "applied"}
        assert second.status_code == 200
        assert second.json() == {"status": "duplicate"}
        assert bot_billing_state.request_log.count("apply") == 1
        assert bot_billing_state.versions[subscription.id] == 1
        # The redelivery stopped before asking Stripe anything.
        assert stripe_state.request_log.count(f"GET /v1/subscriptions/{subscription.id}") == 1

    async def test_a_duplicate_does_not_revoke_what_a_later_event_granted(
        self, app_client, stripe_state, bot_billing_state
    ) -> None:
        subscription = aura_subscription(stripe_state, status="past_due")
        failed = stripe_state.subscription_event("customer.subscription.updated", subscription.id)
        await deliver(app_client, *stripe_state.signed(failed))

        subscription.status = "active"
        recovered = stripe_state.subscription_event(
            "customer.subscription.updated", subscription.id
        )
        await deliver(app_client, *stripe_state.signed(recovered))
        await deliver(app_client, *stripe_state.signed(failed))

        assert bot_billing_state.snapshots[subscription.id]["status"] == "active"
        assert bot_billing_state.versions[subscription.id] == 2

    async def test_ten_simultaneous_deliveries_of_one_event_apply_it_once(
        self, app_client, stripe_state, bot_billing_state
    ) -> None:
        subscription = aura_subscription(stripe_state)
        event = stripe_state.subscription_event("customer.subscription.created", subscription.id)

        responses = await asyncio.gather(
            *(deliver(app_client, *stripe_state.signed(event)) for _ in range(10))
        )

        assert all(response.status_code == 200 for response in responses)
        assert [response.json()["status"] for response in responses].count("applied") == 1
        assert len(bot_billing_state.processed_events) == 1


class TestUnavailableCounterparts:
    async def test_an_unreachable_bot_defers_the_event_and_a_redelivery_applies_it(
        self, app_client, stripe_state, bot_billing_state
    ) -> None:
        subscription = aura_subscription(stripe_state)
        event = stripe_state.subscription_event("customer.subscription.created", subscription.id)
        bot_billing_state.fail_status = 503

        deferred = await deliver(app_client, *stripe_state.signed(event))

        assert deferred.status_code == 503
        assert deferred.json() == {"error": "billing_unavailable"}
        assert bot_billing_state.snapshots == {}

        bot_billing_state.fail_status = None
        redelivered = await deliver(app_client, *stripe_state.signed(event))
        assert redelivered.json() == {"status": "applied"}

    async def test_a_bot_refusing_the_shared_secret_defers_and_is_logged_as_an_error(
        self, app_client, stripe_state, bot_billing_state, caplog
    ) -> None:
        subscription = aura_subscription(stripe_state)
        bot_billing_state.secret = "a-different-secret-than-the-web-backend-holds-000"

        with caplog.at_level(logging.ERROR):
            response = await deliver(
                app_client,
                *stripe_state.signed(
                    stripe_state.subscription_event(
                        "customer.subscription.created", subscription.id
                    )
                ),
            )

        assert response.status_code == 503
        assert any(
            "shared secret" in record.getMessage()
            for record in caplog.records
            if record.levelno >= logging.ERROR
        )

    @pytest.mark.parametrize(
        "status, code",
        [
            (500, "payment_provider_unavailable"),
            (429, "payment_provider_unavailable"),
            (404, "payment_provider_error"),
        ],
    )
    async def test_stripe_failing_the_fetch_defers_without_touching_the_bot(
        self, app_client, stripe_state, bot_billing_state, status, code
    ) -> None:
        subscription = aura_subscription(stripe_state)
        stripe_state.fail_retrieve_status = status

        response = await deliver(
            app_client,
            *stripe_state.signed(
                stripe_state.subscription_event("customer.subscription.created", subscription.id)
            ),
        )

        assert response.status_code == 503
        assert response.json() == {"error": code}
        assert "apply" not in bot_billing_state.request_log

    async def test_a_status_this_code_has_never_seen_keeps_the_last_known_state(
        self, app_client, stripe_state, bot_billing_state
    ) -> None:
        subscription = aura_subscription(stripe_state)
        await deliver(
            app_client,
            *stripe_state.signed(
                stripe_state.subscription_event("customer.subscription.created", subscription.id)
            ),
        )

        subscription.status = "suspended_by_a_future_api"
        response = await deliver(
            app_client,
            *stripe_state.signed(
                stripe_state.subscription_event("customer.subscription.updated", subscription.id)
            ),
        )

        assert response.status_code == 503
        assert bot_billing_state.snapshots[subscription.id]["status"] == "active"


class TestContradictoryEventsAtOnce:
    async def test_renewed_then_canceled_racing_ends_canceled_whatever_the_interleaving(
        self, app_client, stripe_state, bot_billing_state
    ) -> None:
        """The brief's race: "renewed" and "canceled" shortly after one another.

        Delivery A (the renewal) fetches while the subscription is still active,
        then stalls before it writes. Meanwhile the subscription is canceled and
        delivery B is processed completely. When A resumes, its write carries the
        version it read before B committed and is refused; A retries, re-fetches,
        and stores what Stripe says now. The stale "active" never lands.
        """
        subscription = aura_subscription(stripe_state)
        renewed = stripe_state.subscription_event("customer.subscription.updated", subscription.id)
        canceled = stripe_state.subscription_event("customer.subscription.deleted", subscription.id)

        a_is_stalled = asyncio.Event()
        release_a = asyncio.Event()
        stalls = 0

        async def stall_the_first_write(body):
            nonlocal stalls
            if body["event_id"] == renewed["id"] and stalls == 0:
                stalls += 1
                a_is_stalled.set()
                await release_a.wait()

        bot_billing_state.before_apply = stall_the_first_write

        async def run_b_while_a_is_stalled() -> httpx.Response:
            await a_is_stalled.wait()
            subscription.status = "canceled"
            response = await deliver(app_client, *stripe_state.signed(canceled))
            release_a.set()
            return response

        a_response, b_response = await asyncio.gather(
            deliver(app_client, *stripe_state.signed(renewed)), run_b_while_a_is_stalled()
        )

        assert b_response.json() == {"status": "applied"}
        assert a_response.json() == {"status": "applied"}
        assert bot_billing_state.snapshots[subscription.id]["status"] == "canceled"
        assert bot_billing_state.versions[subscription.id] == 2
        assert set(bot_billing_state.processed_events) == {renewed["id"], canceled["id"]}

    async def test_a_sync_that_keeps_losing_races_gives_the_event_back_to_stripe(
        self, app_client, stripe_state, bot_billing_state
    ) -> None:
        subscription = aura_subscription(stripe_state)

        async def always_lose(body):
            bot_billing_state.commit_competing_write(subscription.id)

        bot_billing_state.before_apply = always_lose

        response = await deliver(
            app_client,
            *stripe_state.signed(
                stripe_state.subscription_event("customer.subscription.updated", subscription.id)
            ),
        )

        assert response.status_code == 503
        assert bot_billing_state.request_log.count("apply") == 3
        assert bot_billing_state.processed_events == {}
