"""The service's long-lived collaborators, built once and shared by every request.

Held on ``app.state`` and reached through a FastAPI dependency rather than
imported as module globals. Two reasons, both learned from the bot process:
a module global outlives the event loop that created it (the trap
aura.db.connection documents at length for asyncio locks), and a global makes
"construct the whole service against a fake Discord" impossible without
patching, which is exactly the kind of test this sub-phase's verification
needs to be able to write.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import timedelta

from fastapi import Request

from aura_web.bot_guilds import BotGuildCache
from aura_web.config import WebSettings
from aura_web.discord_api import DiscordAPIError, DiscordAuthError, DiscordClient
from aura_web.sessions import (
    OAuthStateStore,
    Session,
    SessionStore,
    utc_now,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ServiceContext:
    """Everything a request handler needs, with nothing request-scoped in it."""

    settings: WebSettings
    discord: DiscordClient
    sessions: SessionStore
    oauth_states: OAuthStateStore
    bot_guilds: BotGuildCache


def get_context(request: Request) -> ServiceContext:
    """FastAPI dependency resolving the shared context off the application."""
    context: ServiceContext = request.app.state.context
    return context


def read_session_cookie(request: Request, settings: WebSettings) -> str | None:
    """Read the raw session identifier out of the request's cookies.

    One place, so no handler reaches for ``request.cookies`` with a literal
    name and quietly misses a configured rename.
    """
    return request.cookies.get(settings.session_cookie_name)


async def resolve_active_session(
    request: Request, context: ServiceContext
) -> tuple[str, Session] | None:
    """Resolve the caller's session, refreshing its Discord token if needed.

    Returns None for every way a caller can fail to be logged in -- no
    cookie, an unknown or expired identifier, or a refresh Discord refuses --
    so callers have one "not authenticated" branch instead of four.

    A session whose refresh is rejected is deleted rather than left in place:
    it can no longer do anything, and keeping it would let the user sit on a
    page that appears logged in while every action behind it fails.
    """
    token = read_session_cookie(request, context.settings)
    session = context.sessions.get(token)
    if token is None or session is None:
        return None

    if not session.tokens.is_expired(now=utc_now()):
        return token, session

    if session.tokens.refresh_token is None:
        logger.info("Session's access token expired with no refresh token; ending it")
        context.sessions.delete(token)
        return None

    try:
        refreshed = await context.discord.refresh_tokens(session.tokens.refresh_token)
    except DiscordAuthError as exc:
        logger.info("Discord refused a token refresh (%s); ending the session", exc)
        context.sessions.delete(token)
        return None
    except DiscordAPIError as exc:
        # Unavailability is NOT a reason to destroy a session -- the
        # credential is probably still good and we simply could not ask.
        # The request fails; the login survives the outage.
        logger.warning("Could not refresh a Discord token (%s); failing this request", exc)
        raise

    context.sessions.replace_tokens(token, refreshed)
    return token, session


def session_cookie_max_age(settings: WebSettings) -> int:
    """The session cookie's lifetime, in seconds, matched to the server-side TTL.

    Kept equal on purpose. A cookie outliving its record leaves the browser
    presenting an identifier the server has already forgotten; a record
    outliving its cookie leaks memory the user can never reclaim by logging
    out.
    """
    return int(timedelta(seconds=settings.session_ttl_seconds).total_seconds())
