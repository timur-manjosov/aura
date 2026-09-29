"""Billing actions a signed-in admin can take: see plans, start a checkout, open the billing portal.

WHO MAY DO WHAT, decided here and nowhere else:

  * GET  /api/billing/guilds   -- plans for the guilds on this user's dashboard,
    i.e. exactly the guilds 4b's two-condition filter already selects.
  * POST /api/billing/checkout -- a subscription for ONE of those guilds. The
    guild ID in the body is only a request: it is checked, server-side and
    against Discord's live answer, to be a guild this user manages and Aura is
    in, before anything is sent to Stripe. The ID that reaches Stripe's
    metadata is the checked one; nothing else from the body goes anywhere.
  * POST /api/billing/portal   -- Stripe's billing portal for a subscription
    THIS user paid for. Managing the guild is not enough: another admin of the
    same server must not see the payer's card, address or invoices. Having paid
    is also sufficient on its own -- a payer who has since lost the Manage
    Server permission can still reach their own billing to cancel it.

CROSS-SITE REQUESTS. Both POST routes change state at Stripe, so both refuse
anything a third-party page could send with the user's cookie: the body must be
application/json (an HTML form cannot send that, and a cross-origin fetch that
sets it needs a CORS preflight this service never answers), an Origin header,
when present, must be this service's own frontend origin, and a browser that
labels the request cross-site is refused outright. SameSite=Lax on the session
cookie already withholds it from cross-site POSTs; these checks hold even if a
future cookie change weakens that.

Every refusal is a code, never prose, and none reveals whether a guild the user
cannot see has a subscription.
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass

from fastapi import APIRouter, Depends, Request, Response
from fastapi.responses import JSONResponse

from aura_web.bot_billing import BotBillingError, GuildPlanView
from aura_web.context import ServiceContext, get_context, resolve_active_session
from aura_web.discord_api import DiscordAPIError, DiscordAuthError
from aura_web.errors import ErrorCode, error_response
from aura_web.guild_selection import ManageableGuild, select_manageable_guilds
from aura_web.permissions import parse_snowflake
from aura_web.request_bodies import BodyTooLargeError, parse_json_object, read_bounded_body
from aura_web.sessions import Session
from aura_web.stripe_api import StripeAPIError, StripeRejectedError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/billing", tags=["billing"])

# {"guild_id": "<20 digits>"} is under 40 bytes. 4 KiB leaves room for nothing
# but whitespace, and refuses anything else.
MAX_BILLING_REQUEST_BYTES = 4096

# Two clicks on "Subscribe" within this window -- a double click, a second tab,
# a retry after a slow response -- reuse ONE Checkout Session through Stripe's
# idempotency key instead of opening two payable pages for the same guild.
CHECKOUT_IDEMPOTENCY_WINDOW_SECONDS = 600


def _refuse_cross_site(request: Request, context: ServiceContext) -> Response | None:
    """The cross-site request checks every state-changing billing route runs first."""
    content_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    if content_type != "application/json":
        return error_response(ErrorCode.UNSUPPORTED_MEDIA_TYPE, status_code=415)
    origins = request.headers.getlist("origin")
    if len(origins) > 1 or (
        origins and origins[0].strip().lower() != context.settings.frontend_origin
    ):
        logger.warning("Refused a billing request from a foreign origin")
        return error_response(ErrorCode.FORBIDDEN_ORIGIN, status_code=403)
    if request.headers.get("sec-fetch-site", "").strip().lower() == "cross-site":
        logger.warning("Refused a billing request the browser marked cross-site")
        return error_response(ErrorCode.FORBIDDEN_ORIGIN, status_code=403)
    return None


async def _read_guild_id(request: Request) -> str | Response:
    """The one field these routes accept, in its one canonical form, or the refusal to send."""
    try:
        raw = await read_bounded_body(request, MAX_BILLING_REQUEST_BYTES)
    except BodyTooLargeError:
        return error_response(ErrorCode.PAYLOAD_TOO_LARGE, status_code=413)
    body = parse_json_object(raw)
    if body is None or set(body) != {"guild_id"} or not isinstance(body["guild_id"], str):
        return error_response(ErrorCode.INVALID_REQUEST, status_code=400)
    raw_guild_id = body["guild_id"]
    parsed = parse_snowflake(raw_guild_id)
    # Canonical form only: " 123", "0123" and "١٢٣" all normalise to 123 in
    # parse_snowflake, and a value that is not literally the ID it normalises
    # to is a caller that is not the frontend.
    if parsed is None or parsed != raw_guild_id or int(parsed) <= 0:
        return error_response(ErrorCode.INVALID_REQUEST, status_code=400)
    return parsed


@dataclass(frozen=True)
class _Caller:
    session: Session
    session_token: str
    manageable: list[ManageableGuild]


async def _resolve_caller(request: Request, context: ServiceContext) -> _Caller | Response:
    """The signed-in user and, freshly from Discord, the guilds they may manage with Aura in them."""
    try:
        resolved = await resolve_active_session(request, context)
        if resolved is None:
            return error_response(ErrorCode.NOT_AUTHENTICATED, status_code=401)
        token, session = resolved
        user_guilds = await context.discord.fetch_user_guilds(session.tokens.access_token)
        bot_guild_ids = await context.bot_guilds.get()
    except DiscordAuthError:
        context.sessions.delete(request.cookies.get(context.settings.session_cookie_name))
        return error_response(ErrorCode.NOT_AUTHENTICATED, status_code=401)
    except DiscordAPIError:
        # Fails closed for the same reason /api/guilds does: without Discord's
        # answer there is no way to know which guilds this user may act for.
        return error_response(ErrorCode.DISCORD_UNAVAILABLE, status_code=503)
    return _Caller(session, token, select_manageable_guilds(user_guilds, bot_guild_ids))


def browser_plan(plan: GuildPlanView, *, user_id: str) -> dict[str, object]:
    """What a browser may see of a guild's plan: the standing, never who paid or with which account.

    Parameters
    ----------
    plan
        The guild's plan as the bot reported it.
    user_id
        The caller, used only to decide whether they may open the portal.

    Returns
    -------
    dict[str, object]
        What a browser may see: the standing, and whether this caller can
        manage the subscription. Never the customer ID, never who paid.
    """
    return {
        "tier": plan.tier,
        "basis": plan.basis,
        "standing": plan.standing,
        "access_until": plan.access_until,
        "paid_through": plan.paid_through,
        "active_subscription_count": plan.in_force_subscription_count,
        "can_subscribe": plan.in_force_subscription_count == 0,
        "is_billing_owner": any(
            subscription.purchaser_user_id == user_id for subscription in plan.subscriptions
        ),
    }


def checkout_idempotency_key(*, user_id: str, guild_id: str, now: float) -> str:
    """One Stripe idempotency key per user, guild and time window.

    Parameters
    ----------
    user_id, guild_id
        Who is paying for which guild.
    now
        Current Unix time, bucketed into a window so a double-submit inside it
        reuses one key.

    Returns
    -------
    str
        A key stable for one user, guild and window -- so a repeated submission
        creates one Stripe session rather than two.

    Notes
    -----
    Hashed so the key Stripe stores is not a readable pairing of a Discord user
    and a guild, and namespaced so it cannot collide with a key any other code
    path might ever send.
    """
    window = int(now // CHECKOUT_IDEMPOTENCY_WINDOW_SECONDS)
    material = f"aura-checkout:v1:{user_id}:{guild_id}:{window}".encode()
    return hashlib.sha256(material).hexdigest()


@router.get("/guilds")
async def billing_guilds(
    request: Request, context: ServiceContext = Depends(get_context)
) -> Response:
    """Plans for every guild on the caller's dashboard. Possibly none.

    Parameters
    ----------
    request
        The incoming request.
    context
        The shared context: settings, stores and clients.

    Returns
    -------
    Response
        Plans for every guild on the caller's dashboard, possibly none; 401
        without a session.
    """
    caller = await _resolve_caller(request, context)
    if isinstance(caller, Response):
        return caller
    guild_ids = [guild.id for guild in caller.manageable]
    try:
        plans = await context.bot_billing.get_guild_plans(guild_ids)
    except BotBillingError as exc:
        logger.warning("Could not read plans from the bot: %s", exc)
        return error_response(ErrorCode.BILLING_UNAVAILABLE, status_code=503)
    return JSONResponse(
        [
            {"id": guild_id, "plan": browser_plan(plans[guild_id], user_id=caller.session.user.id)}
            for guild_id in guild_ids
        ]
    )


@router.post("/checkout")
async def create_checkout(
    request: Request, context: ServiceContext = Depends(get_context)
) -> Response:
    """Start a Stripe-hosted Pro checkout for one guild the caller may manage.

    Parameters
    ----------
    request
        The incoming request.
    context
        The shared context: settings, stores and clients.

    Returns
    -------
    Response
        The hosted checkout URL for the browser to follow, or a JSON error:
        401 without a session, 403 for a guild the caller may not manage, 409
        when the guild already has a subscription the bot knows about.
    """
    refusal = _refuse_cross_site(request, context)
    if refusal is not None:
        return refusal
    guild_id = await _read_guild_id(request)
    if isinstance(guild_id, Response):
        return guild_id
    caller = await _resolve_caller(request, context)
    if isinstance(caller, Response):
        return caller

    if guild_id not in {guild.id for guild in caller.manageable}:
        # One answer for "does not exist", "you do not manage it" and "Aura is
        # not in it", so this route cannot be used to learn which is true.
        logger.warning(
            "Refused a checkout for guild %s by user %s: not a guild they manage with Aura in it",
            guild_id,
            caller.session.user.id,
        )
        return error_response(ErrorCode.GUILD_NOT_MANAGEABLE, status_code=403)

    try:
        plan = (await context.bot_billing.get_guild_plans([guild_id]))[guild_id]
    except BotBillingError as exc:
        # Fails closed: without the bot's answer there is no way to know whether
        # this guild is already paying, and a second subscription is a real
        # charge to a real card.
        logger.warning("Refused a checkout for guild %s: plans unavailable (%s)", guild_id, exc)
        return error_response(ErrorCode.BILLING_UNAVAILABLE, status_code=503)
    if plan.in_force_subscription_count > 0:
        return error_response(ErrorCode.ALREADY_SUBSCRIBED, status_code=409)

    try:
        checkout = await context.stripe.create_checkout_session(
            guild_id=guild_id,
            purchaser_user_id=caller.session.user.id,
            idempotency_key=checkout_idempotency_key(
                user_id=caller.session.user.id, guild_id=guild_id, now=time.time()
            ),
        )
    except StripeRejectedError as exc:
        logger.error("Stripe refused a checkout for guild %s: %s", guild_id, exc)
        return error_response(ErrorCode.PAYMENT_PROVIDER_ERROR, status_code=502)
    except StripeAPIError as exc:
        logger.warning("Could not create a checkout for guild %s: %s", guild_id, exc)
        return error_response(ErrorCode.PAYMENT_PROVIDER_UNAVAILABLE, status_code=503)

    logger.info(
        "Created checkout session %s for guild %s by user %s",
        checkout.session_id,
        guild_id,
        caller.session.user.id,
    )
    return JSONResponse({"url": checkout.url})


@router.post("/portal")
async def open_billing_portal(
    request: Request, context: ServiceContext = Depends(get_context)
) -> Response:
    """Open Stripe's billing portal for a subscription the caller paid for.

    Parameters
    ----------
    request
        The incoming request.
    context
        The shared context: settings, stores and clients.

    Returns
    -------
    Response
        A single-use Stripe portal URL, or a JSON error: 401 without a
        session, 403 unless the caller is the purchaser of record.
    """
    refusal = _refuse_cross_site(request, context)
    if refusal is not None:
        return refusal
    guild_id = await _read_guild_id(request)
    if isinstance(guild_id, Response):
        return guild_id
    try:
        resolved = await resolve_active_session(request, context)
    except DiscordAPIError:
        return error_response(ErrorCode.DISCORD_UNAVAILABLE, status_code=503)
    if resolved is None:
        return error_response(ErrorCode.NOT_AUTHENTICATED, status_code=401)
    _, session = resolved

    try:
        plan = (await context.bot_billing.get_guild_plans([guild_id]))[guild_id]
    except BotBillingError as exc:
        logger.warning(
            "Refused a billing portal session for guild %s: plans unavailable (%s)", guild_id, exc
        )
        return error_response(ErrorCode.BILLING_UNAVAILABLE, status_code=503)

    owned = sorted(
        (
            subscription
            for subscription in plan.subscriptions
            if subscription.purchaser_user_id == session.user.id
        ),
        key=lambda subscription: not subscription.grants_access,
    )
    if not owned:
        # The same answer whether the guild has no subscription or someone
        # else's: this route must not reveal which guilds are paying.
        return error_response(ErrorCode.NOT_BILLING_OWNER, status_code=403)

    try:
        url = await context.stripe.create_portal_session(customer_id=owned[0].customer_id)
    except StripeRejectedError as exc:
        logger.error("Stripe refused a billing portal session for guild %s: %s", guild_id, exc)
        return error_response(ErrorCode.PAYMENT_PROVIDER_ERROR, status_code=502)
    except StripeAPIError as exc:
        logger.warning("Could not create a billing portal session for guild %s: %s", guild_id, exc)
        return error_response(ErrorCode.PAYMENT_PROVIDER_UNAVAILABLE, status_code=503)
    return JSONResponse({"url": url})
