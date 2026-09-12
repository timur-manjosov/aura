"""The OAuth2 authorization-code flow: start it, finish it, end it.

The CSRF defence is two halves that only work together, so both are written
here where a reader sees them at once:

  1. A ``state`` value minted server-side, stored single-use with a TTL
     (aura_web.sessions.OAuthStateStore), and required to come back intact.
  2. The same value in an httpOnly cookie, compared against the returned one.

Half 1 alone proves the state came from this service -- but an attacker can
start their own login and get a perfectly valid state, then trick a victim's
browser into completing the callback with it, binding the victim's browser to
the ATTACKER's Discord account. Half 2 is what closes that: the victim's
browser carries the victim's state cookie, which will not match. Half 2 alone
would be no better, since a cookie the attacker sets is a cookie the attacker
knows. Both, or neither.
"""
from __future__ import annotations

import logging
from urllib.parse import urlencode

from fastapi import APIRouter, Depends, Request, Response
from fastapi.responses import RedirectResponse

from aura_web.context import (
    ServiceContext,
    get_context,
    read_session_cookie,
    session_cookie_max_age,
)
from aura_web.discord_api import (
    REQUIRED_SCOPES,
    DiscordAPIError,
    DiscordAuthError,
)
from aura_web.errors import ErrorCode, error_response
from aura_web.sessions import constant_time_equals

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/auth", tags=["auth"])

# Discord's authorize page lives on the web origin, not the API base, so it is
# not built from settings.discord_api_base.
DISCORD_AUTHORIZE_URL = "https://discord.com/oauth2/authorize"

# Sorted so the scope string is identical on every request -- a stable URL is
# easier to compare against what is registered on the Discord application.
SCOPE_PARAMETER = " ".join(sorted(REQUIRED_SCOPES))


def _exactly_one_query_param(request: Request, name: str) -> str | None:
    """Return a query parameter's value only if it appears exactly once.

    Starlette resolves a repeated parameter to its LAST occurrence, so
    ``?state=forged&state=real`` reads as the real one. That is not itself
    exploitable here -- the second half of the CSRF check compares against an
    httpOnly cookie an attacker cannot read, so supplying a valid state means
    already having one -- but "not exploitable given the rest of the design"
    is a grey area rather than an answer, and parameter pollution is
    specifically the class of bug where two parsers disagreeing turns a safe
    design unsafe later.

    Discord never sends a parameter twice. A repeat is therefore malformed or
    hostile, and the honest response to both is to refuse rather than to pick
    one. Found by this sub-phase's adversarial pass, which asserted the wrong
    thing first and then found the right thing.
    """
    values = request.query_params.getlist(name)
    return values[0] if len(values) == 1 else None


def _set_cookie(
    response: Response,
    *,
    name: str,
    value: str,
    max_age: int,
    context: ServiceContext,
) -> None:
    """Set one of this service's cookies with the full flag set.

    Every cookie goes through here so the three flags that matter are decided
    once. httpOnly keeps script (including anything injected into the
    frontend) from reading the identifier; Secure keeps it off plaintext
    transport; SameSite keeps it off cross-site sub-requests, which is what
    CSRF against the logout and future mutating endpoints would ride on.
    Path is pinned to "/" so a later route added under another prefix does not
    silently stop seeing the session.
    """
    response.set_cookie(
        key=name,
        value=value,
        max_age=max_age,
        httponly=True,
        secure=context.settings.session_cookie_secure,
        samesite=context.settings.session_cookie_samesite,
        path="/",
    )


def _clear_cookie(response: Response, *, name: str, context: ServiceContext) -> None:
    """Expire one of this service's cookies, repeating every flag it was set with.

    Browsers match a deletion to an existing cookie by name, domain and path;
    omitting the flags here would leave the original in place on some clients
    while looking, in the response, as though it had been cleared.
    """
    response.delete_cookie(
        key=name,
        path="/",
        httponly=True,
        secure=context.settings.session_cookie_secure,
        samesite=context.settings.session_cookie_samesite,
    )


@router.get("/login")
async def login(context: ServiceContext = Depends(get_context)) -> Response:
    """Begin a login: mint a state, bind it to this browser, redirect to Discord."""
    state = context.oauth_states.issue()

    response = RedirectResponse(
        url=DISCORD_AUTHORIZE_URL
        + "?"
        + urlencode(
            {
                "client_id": context.settings.discord_client_id,
                "response_type": "code",
                "scope": SCOPE_PARAMETER,
                "redirect_uri": context.settings.oauth_redirect_uri,
                "state": state,
            }
        ),
        # 307, not FastAPI's default 302: a 302 lets a client turn the
        # follow-up into a GET, which is harmless here but becomes a real
        # surprise the moment any redirect in this service is issued from a
        # POST. One convention, chosen once.
        status_code=307,
    )
    _set_cookie(
        response,
        name=context.settings.oauth_state_cookie_name,
        value=state,
        max_age=context.settings.oauth_state_ttl_seconds,
        context=context,
    )
    return response


@router.get("/callback")
async def callback(request: Request, context: ServiceContext = Depends(get_context)) -> Response:
    """Finish a login: verify the state both ways, exchange the code, open a session.

    Every rejection returns a status code and an error code rather than
    redirecting to the frontend. A refused callback is a security event, and
    bouncing the browser onward would make it look like an ordinary
    navigation to both the user and whoever is reading the logs.
    """
    settings = context.settings
    state_from_query = _exactly_one_query_param(request, "state")
    state_from_cookie = request.cookies.get(settings.oauth_state_cookie_name)

    # Consumed before anything else can return early, so a state is burnt by
    # its first use whatever happens next -- otherwise a callback that failed
    # later for an unrelated reason would leave a replayable state behind.
    state_was_issued = context.oauth_states.consume(state_from_query)
    state_matches_browser = constant_time_equals(state_from_query, state_from_cookie)

    if not state_was_issued or not state_matches_browser:
        logger.warning(
            "Rejected an OAuth callback: state issued=%s, bound-to-this-browser=%s",
            state_was_issued,
            state_matches_browser,
        )
        response = error_response(ErrorCode.INVALID_STATE, status_code=400)
        _clear_cookie(response, name=settings.oauth_state_cookie_name, context=context)
        return response

    # Only checked after the state is verified. Discord sends ?error=access_denied
    # when a user clicks "Cancel", and trusting that parameter before proving the
    # callback is genuine would let anyone drive this branch's logging and
    # response from a crafted link.
    #
    # ANY occurrence counts, unlike state and code: a repeated error parameter
    # still means the flow did not produce a usable grant, and treating a
    # duplicate as absent would send us on to exchange a code that is not there.
    if request.query_params.getlist("error"):
        logger.info("User declined the Discord authorization prompt")
        response = error_response(ErrorCode.OAUTH_DENIED, status_code=400)
        _clear_cookie(response, name=settings.oauth_state_cookie_name, context=context)
        return response

    code = _exactly_one_query_param(request, "code")
    if not code:
        logger.warning(
            "Rejected an OAuth callback with a valid state but no single usable code"
        )
        response = error_response(ErrorCode.OAUTH_FAILED, status_code=400)
        _clear_cookie(response, name=settings.oauth_state_cookie_name, context=context)
        return response

    try:
        tokens = await context.discord.exchange_code(code, settings.oauth_redirect_uri)
        user = await context.discord.fetch_current_user(tokens.access_token)
    except DiscordAuthError as exc:
        logger.warning("Discord rejected the authorization code exchange: %s", exc)
        response = error_response(ErrorCode.OAUTH_FAILED, status_code=400)
        _clear_cookie(response, name=settings.oauth_state_cookie_name, context=context)
        return response
    except DiscordAPIError as exc:
        logger.warning("Could not complete the OAuth exchange with Discord: %s", exc)
        response = error_response(ErrorCode.DISCORD_UNAVAILABLE, status_code=503)
        _clear_cookie(response, name=settings.oauth_state_cookie_name, context=context)
        return response

    session_token = context.sessions.create(user, tokens)
    logger.info("Opened a session for Discord user %s", user.id)

    response = RedirectResponse(url=settings.post_login_redirect_url, status_code=303)
    _set_cookie(
        response,
        name=settings.session_cookie_name,
        value=session_token,
        max_age=session_cookie_max_age(settings),
        context=context,
    )
    # The state cookie has done its single job; leaving it would keep a spent
    # secret in the browser for its full TTL.
    _clear_cookie(response, name=settings.oauth_state_cookie_name, context=context)
    return response


@router.post("/logout")
async def logout(request: Request, context: ServiceContext = Depends(get_context)) -> Response:
    """End the session, clear the cookie, and ask Discord to revoke the token.

    POST rather than GET, so a cross-site image or link cannot log a user out;
    combined with SameSite the cookie would not travel on such a request
    anyway, which is belt and braces rather than redundancy -- the two protect
    against different browser configurations.

    Always 204, whether or not a session existed. A logout endpoint that
    answered differently for a valid and an invalid identifier would be a free
    oracle for testing stolen cookies.
    """
    token = read_session_cookie(request, context.settings)
    session = context.sessions.delete(token)

    if session is not None:
        # Deletion first, revocation second: the session is gone regardless of
        # whether Discord answers (revoke_token never raises), so the user's
        # logout cannot fail on somebody else's availability.
        await context.discord.revoke_token(session.tokens.access_token)
        logger.info("Closed the session for Discord user %s", session.user.id)

    response = Response(status_code=204)
    _clear_cookie(response, name=context.settings.session_cookie_name, context=context)
    return response
