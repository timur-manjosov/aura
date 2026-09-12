"""FastAPI application factory and process lifetime.

The factory takes its settings and, optionally, its Discord client as
arguments. That is what lets the verification this sub-phase's brief asks for
exist at all: the same application object, wired to a stand-in Discord, can be
driven through a complete OAuth2 round trip over real HTTP, with real cookies
and real headers to inspect -- rather than a test that reads the code and
believes it.

There is deliberately no CORS middleware. The frontend reaches this service
through a same-origin path (Next.js rewrites /api/* to this container), so the
browser never makes a cross-origin request and no cross-origin request needs
to be allowed. That is not a shortcut: it is what lets the session cookie stay
SameSite=Lax instead of SameSite=None, which is the flag that actually keeps a
third-party page from riding the session.
"""
from __future__ import annotations

import logging
import sys
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from aura_web.bot_guilds import BotGuildCache
from aura_web.config import WebConfigurationError, WebSettings, load_web_settings
from aura_web.context import ServiceContext
from aura_web.discord_api import DiscordClient
from aura_web.routes import auth_router, dashboard_router
from aura_web.sessions import OAuthStateStore, SessionStore

logger = logging.getLogger(__name__)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add the response headers every answer from this service should carry.

    ``Cache-Control: no-store`` is the load-bearing one. /api/me and
    /api/guilds are per-session answers behind a cookie; without it a shared
    proxy, or the browser's own back/forward cache, may hand one user's guild
    list to the next person on the same machine. The other two are cheap
    hardening: nosniff stops a JSON body being re-interpreted as script, and
    DENY keeps the page out of a framing clickjack.
    """

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        response = await call_next(request)
        response.headers.setdefault("Cache-Control", "no-store")
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        return response


def create_app(
    settings: WebSettings,
    *,
    discord_client_factory: Callable[[httpx.AsyncClient, WebSettings], DiscordClient] | None = None,
) -> FastAPI:
    """Build the application, deferring every network-owning object to startup.

    Nothing that holds a socket or an asyncio primitive is constructed here.
    The httpx pool and the guild cache's refresh lock are created inside the
    lifespan, i.e. inside the running event loop, for the reason
    aura.db.connection spells out for the bot: an asyncio object built under
    one loop and used under another fails late, intermittently, and in tests
    before production.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(settings.http_timeout_seconds),
            # Redirects are not followed: every URL here is a fixed Discord
            # endpoint, so a redirect means something has been substituted for
            # it, and following one would forward a bearer token to wherever
            # it pointed.
            follow_redirects=False,
        ) as http:
            discord = (
                discord_client_factory(http, settings)
                if discord_client_factory is not None
                else DiscordClient(
                    http,
                    api_base=settings.discord_api_base,
                    client_id=settings.discord_client_id,
                    client_secret=settings.discord_client_secret,
                    bot_token=settings.discord_bot_token,
                )
            )
            app.state.context = ServiceContext(
                settings=settings,
                discord=discord,
                sessions=SessionStore(
                    ttl_seconds=settings.session_ttl_seconds,
                    max_sessions=settings.max_sessions,
                ),
                oauth_states=OAuthStateStore(
                    ttl_seconds=settings.oauth_state_ttl_seconds,
                    max_states=settings.max_pending_states,
                ),
                bot_guilds=BotGuildCache(
                    discord,
                    ttl_seconds=settings.bot_guilds_cache_ttl_seconds,
                    stale_tolerance_seconds=settings.bot_guilds_stale_tolerance_seconds,
                ),
            )
            logger.info(
                "Aura web backend ready: redirect_uri=%s, cookie=%s (secure=%s, samesite=%s), "
                "session TTL %ds, bot-guild cache %.0fs",
                settings.oauth_redirect_uri,
                settings.session_cookie_name,
                settings.session_cookie_secure,
                settings.session_cookie_samesite,
                settings.session_ttl_seconds,
                settings.bot_guilds_cache_ttl_seconds,
            )
            yield

    app = FastAPI(
        title="Aura Web Backend",
        version="4b",
        lifespan=lifespan,
        # The interactive docs and the OpenAPI schema are off. They describe
        # an authenticated API to anyone who asks, and nothing in 4b needs
        # them; a deployment that wants them should have to turn them on.
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.add_middleware(SecurityHeadersMiddleware)
    app.include_router(auth_router)
    app.include_router(dashboard_router)
    return app


def build_app() -> FastAPI:
    """Load settings from the environment and build the application.

    The uvicorn entry point. Configuration failures exit with a readable line
    rather than a pydantic traceback surfacing through the ASGI loader, the
    same contract aura.main gives the bot process.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )
    try:
        settings = load_web_settings()
    except WebConfigurationError as exc:
        logger.critical("Startup aborted: %s", exc)
        sys.exit(1)

    logging.getLogger().setLevel(settings.log_level.upper())
    return create_app(settings)
