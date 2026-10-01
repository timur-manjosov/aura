"""POST /api/stripe/webhook: Stripe's events, and nothing is trusted before the signature is.

THE ORDER IS THE SECURITY PROPERTY, so it is written out once here and followed
exactly below:

  1. Exactly one non-empty Stripe-Signature header, or refuse. Nothing read yet.
  2. Read the raw body, bounded. Bytes only -- nothing is parsed.
  3. Verify the signature over those exact bytes with Stripe's own
     implementation (stripe.WebhookSignature.verify_header, the verification
     half of stripe.Webhook.construct_event), against the endpoint's signing
     secret, with Stripe's 5-minute timestamp tolerance. ANY failure -- a wrong
     signature, a missing v1 scheme, a malformed header, a body that is not
     UTF-8, an old timestamp, or an exception nobody anticipated -- refuses
     with 400 and touches no state. The failure branch catches Exception, not a
     list of expected types, on purpose: the one outcome that must be
     impossible is an unverified body reaching step 4 because verification
     failed in a way this code did not predict.
     A delivery that verifies hands its rate-limit token back here
     (aura_web.rate_limit): only failed verifications count against a client,
     so Stripe's own deliveries can never exhaust the webhook's limit.
  4. Only now is the body parsed, and only into an event ID, a type, a mode and
     a subscription reference (aura_web.stripe_events).
  5. The subscription is re-fetched from Stripe and handed to the bot through
     the compare-and-swap in aura_web.billing_sync. The event ID is recorded by
     the bot in the same transaction as the state, so a redelivery is a no-op.

construct_event itself is not called because it additionally builds a
StripeObject bound to the SDK's module-global API key -- a deprecated pattern
this service does not use -- and step 4 needs a validated shape rather than a
dictionary-like object anyway. The verification call is identical.

Responses: 2xx means "handled, do not redeliver" (applied, a duplicate, or an
event that concerns no Aura subscription). 400 means the request itself is not
acceptable. 503 means "could not process it now, redeliver" -- Stripe retries
for up to three days in live mode, and every retry is safe.
"""

from __future__ import annotations

import logging

import stripe
from fastapi import APIRouter, Depends, Request, Response
from fastapi.responses import JSONResponse

from aura_web.billing_sync import LivemodeMismatchError, SyncConflictError, sync_subscription
from aura_web.bot_billing import BotBillingError
from aura_web.context import ServiceContext, get_context
from aura_web.errors import ErrorCode, error_response
from aura_web.rate_limit import refund_webhook_allowance
from aura_web.request_bodies import BodyTooLargeError, read_bounded_body
from aura_web.stripe_api import StripeAPIError, StripeRejectedError
from aura_web.stripe_events import InvalidEventError, parse_verified_event

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/stripe", tags=["stripe"])

# Subscription, checkout and invoice events are a few kilobytes; an invoice with
# many line items can reach tens of kilobytes. A megabyte is far past anything
# Stripe sends for the events this endpoint handles, and still refuses a body
# big enough to be a memory lever against an unauthenticated route.
MAX_WEBHOOK_BODY_BYTES = 1024 * 1024

# Stripe's own default. Stripe signs every delivery attempt with a fresh
# timestamp, so a legitimate retry is never older than this; a captured request
# replayed later is. Within the window a replay is still refused as a duplicate.
SIGNATURE_TOLERANCE_SECONDS = 300


@router.post("/webhook")
async def stripe_webhook(
    request: Request, context: ServiceContext = Depends(get_context)
) -> Response:
    """Verify, then process, one Stripe webhook delivery. See the module docstring for the order.

    Parameters
    ----------
    request
        The incoming request.
    context
        The shared context: settings, stores and clients.

    Returns
    -------
    Response
        200 once the delivery has been handled or deliberately ignored; a 4xx
        only for a body that failed signature verification or could not be
        parsed. Anything Stripe should retry is answered with 5xx.
    """
    signature_headers = request.headers.getlist("stripe-signature")
    if len(signature_headers) != 1 or not signature_headers[0].strip():
        logger.warning(
            "Rejected a Stripe webhook: expected exactly one non-empty Stripe-Signature header, got %d",
            len(signature_headers),
        )
        return error_response(ErrorCode.INVALID_SIGNATURE, status_code=400)

    try:
        payload = await read_bounded_body(request, MAX_WEBHOOK_BODY_BYTES)
    except BodyTooLargeError:
        logger.warning("Rejected a Stripe webhook: body exceeds %d bytes", MAX_WEBHOOK_BODY_BYTES)
        return error_response(ErrorCode.PAYLOAD_TOO_LARGE, status_code=413)

    try:
        stripe.WebhookSignature.verify_header(
            payload,
            signature_headers[0],
            context.settings.stripe_webhook_secret.get_secret_value(),
            SIGNATURE_TOLERANCE_SECONDS,
        )
    except Exception as exc:
        # The exception's class only. SignatureVerificationError carries the
        # header and the payload, and neither belongs in a log line.
        logger.warning(
            "Rejected a Stripe webhook: signature verification failed (%s)", type(exc).__name__
        )
        return error_response(ErrorCode.INVALID_SIGNATURE, status_code=400)

    # Signed by the endpoint's own secret: whatever happens next, this delivery
    # came from Stripe, and Stripe's deliveries never count against the limit.
    refund_webhook_allowance(request)

    try:
        event = parse_verified_event(payload)
    except InvalidEventError as exc:
        logger.warning(
            "Rejected a correctly signed Stripe webhook that is not a usable event: %s", exc
        )
        return error_response(ErrorCode.INVALID_EVENT, status_code=400)

    if event.livemode != context.settings.stripe_live_mode:
        logger.error(
            "Rejected Stripe event %s: livemode=%s, but this service is configured for livemode=%s",
            event.event_id,
            event.livemode,
            context.settings.stripe_live_mode,
        )
        return error_response(ErrorCode.LIVEMODE_MISMATCH, status_code=400)

    if event.subscription_id is None:
        logger.info(
            "Acknowledged Stripe event %s (%s): it concerns no subscription this service acts on",
            event.event_id,
            event.event_type,
        )
        return JSONResponse({"status": "ignored"})

    try:
        outcome = await sync_subscription(
            stripe=context.stripe,
            bot=context.bot_billing,
            subscription_id=event.subscription_id,
            event_id=event.event_id,
            event_type=event.event_type,
            live_mode=context.settings.stripe_live_mode,
        )
    except LivemodeMismatchError as exc:
        logger.error("Rejected Stripe event %s: %s", event.event_id, exc)
        return error_response(ErrorCode.LIVEMODE_MISMATCH, status_code=400)
    except SyncConflictError as exc:
        logger.warning("Deferred Stripe event %s to Stripe's retry: %s", event.event_id, exc)
        return error_response(ErrorCode.BILLING_UNAVAILABLE, status_code=503)
    except StripeRejectedError as exc:
        logger.error("Could not process Stripe event %s: %s", event.event_id, exc)
        return error_response(ErrorCode.PAYMENT_PROVIDER_ERROR, status_code=503)
    except StripeAPIError as exc:
        logger.warning("Deferred Stripe event %s to Stripe's retry: %s", event.event_id, exc)
        return error_response(ErrorCode.PAYMENT_PROVIDER_UNAVAILABLE, status_code=503)
    except BotBillingError as exc:
        logger.warning("Deferred Stripe event %s to Stripe's retry: %s", event.event_id, exc)
        return error_response(ErrorCode.BILLING_UNAVAILABLE, status_code=503)

    logger.info(
        "Processed Stripe event %s (%s) for subscription %s: %s",
        event.event_id,
        event.event_type,
        event.subscription_id,
        outcome.value,
    )
    return JSONResponse({"status": outcome.value})
