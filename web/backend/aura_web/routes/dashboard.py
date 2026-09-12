"""What a logged-in browser is allowed to read: who it is, and which guilds it manages.

Both handlers exist to be small. The interesting decisions -- which guilds
qualify, what a Discord outage means -- live in aura_web.guild_selection and
aura_web.bot_guilds, where they are unit-testable without a request. What is
decided *here*, and nowhere else, is the response shape, and its one
non-negotiable property: no Discord access token, refresh token, or client
secret appears in any field. The session record holds those; these responses
project only what the shell renders.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, Request, Response
from fastapi.responses import JSONResponse

from aura_web.context import ServiceContext, get_context, resolve_active_session
from aura_web.discord_api import DiscordAPIError, DiscordAuthError
from aura_web.errors import ErrorCode, error_response
from aura_web.guild_selection import select_manageable_guilds

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["dashboard"])


@router.get("/health")
async def health() -> Response:
    """Liveness for the container healthcheck. Reaches nothing external on purpose.

    A healthcheck that called Discord would report this service as unhealthy
    during a Discord outage and let the orchestrator restart a process that is
    working perfectly -- turning somebody else's downtime into our own.
    """
    return JSONResponse({"status": "ok"})


@router.get("/me")
async def me(request: Request, context: ServiceContext = Depends(get_context)) -> Response:
    """Return the signed-in user's public identity, or 401.

    The avatar is returned as Discord's raw hash plus the user ID rather than
    as an assembled CDN URL, so the browser does the assembling and this
    service is not in the business of emitting third-party URLs it would then
    have to keep correct.
    """
    try:
        resolved = await resolve_active_session(request, context)
    except DiscordAPIError:
        return error_response(ErrorCode.DISCORD_UNAVAILABLE, status_code=503)

    if resolved is None:
        return error_response(ErrorCode.NOT_AUTHENTICATED, status_code=401)

    _, session = resolved
    return JSONResponse(
        {
            "id": session.user.id,
            "username": session.user.username,
            "global_name": session.user.global_name,
            "avatar": session.user.avatar,
        }
    )


@router.get("/guilds")
async def guilds(request: Request, context: ServiceContext = Depends(get_context)) -> Response:
    """Return the guilds this user may manage AND Aura is in. Possibly none.

    An empty list is a normal, successful answer -- a user who moderates
    nothing, or whose servers have not invited Aura, gets ``[]`` and HTTP 200.
    Reporting that as an error would be both wrong and, for anyone probing,
    a hint that something was withheld.
    """
    try:
        resolved = await resolve_active_session(request, context)
        if resolved is None:
            return error_response(ErrorCode.NOT_AUTHENTICATED, status_code=401)

        _, session = resolved
        user_guilds = await context.discord.fetch_user_guilds(session.tokens.access_token)
        bot_guild_ids = await context.bot_guilds.get()
    except DiscordAuthError as exc:
        # The access token was accepted at login and is refused now: it was
        # revoked, or the user removed the authorization. The session is dead
        # either way, so it is ended rather than left to fail on every load.
        logger.info("Discord refused a user's token while listing guilds (%s); ending the session", exc)
        context.sessions.delete(request.cookies.get(context.settings.session_cookie_name))
        return error_response(ErrorCode.NOT_AUTHENTICATED, status_code=401)
    except DiscordAPIError as exc:
        # Fail closed. Without a trustworthy membership list there is no way
        # to apply the second half of the filter, and answering with the
        # user's full guild list would show servers Aura is not in.
        logger.warning("Could not build the guild list from Discord (%s); refusing the request", exc)
        return error_response(ErrorCode.DISCORD_UNAVAILABLE, status_code=503)

    manageable = select_manageable_guilds(user_guilds, bot_guild_ids)
    return JSONResponse(
        [{"id": guild.id, "name": guild.name, "icon": guild.icon} for guild in manageable]
    )
