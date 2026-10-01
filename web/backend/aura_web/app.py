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

import asyncio
import logging
import sys
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager, suppress

import httpx
from fastapi import FastAPI
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from aura_web.billing_sync import run_reconciler
from aura_web.bot_billing import BotBillingClient
from aura_web.bot_guilds import BotGuildCache
from aura_web.config import WebConfigurationError, WebSettings, load_web_settings
from aura_web.context import ServiceContext
from aura_web.discord_api import DiscordClient
from aura_web.routes import auth_router, billing_router, dashboard_router, stripe_webhook_router
from aura_web.sessions import OAuthStateStore, SessionStore
from aura_web.stripe_api import StripeClient

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

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Run one request and add the security headers to its response.

        Parameters
        ----------
        request
            The incoming request.
        call_next
            The rest of the middleware chain.

        Returns
        -------
        Response
            The handler's response with the four headers added. Each is set
            with `setdefault`, so a route that deliberately chose its own
            value keeps it.
        """
        response = await call_next(request)
        response.headers.setdefault("Cache-Control", "no-store")
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        return response


async def check_stripe_key(stripe_client: StripeClient) -> None:
    """Run the Stripe key's permission probe, and never let it fail the service.

    Parameters
    ----------
    stripe_client
        The client whose configured key is probed.

    Returns
    -------
    None
        The finding is logged by StripeClient.check_key_permissions; any
        unexpected exception is logged here by type only and swallowed.

    Notes
    -----
    The probe already turns every Stripe failure into a log line. This wrapper
    is for everything else: an exception escaping a background task would only
    surface as "Task exception was never retrieved" at shutdown, with a
    traceback, instead of as one line at the time it happened.
    """
    try:
        await stripe_client.check_key_permissions()
    except Exception as exc:
        logger.warning(
            "Stripe key self-check failed unexpectedly (%s); startup continues",
            type(exc).__name__,
        )


def create_app(
    settings: WebSettings,
    *,
    discord_client_factory: Callable[[httpx.AsyncClient, WebSettings], DiscordClient] | None = None,
    stripe_client_factory: Callable[[httpx.AsyncClient, WebSettings], StripeClient] | None = None,
    bot_billing_client_factory: Callable[[httpx.AsyncClient, WebSettings], BotBillingClient]
    | None = None,
    check_stripe_key_at_startup: bool = True,
) -> FastAPI:
    """Build the application, deferring every network-owning object to startup.

    Parameters
    ----------
    settings
        Validated web configuration.
    discord_client_factory, stripe_client_factory, bot_billing_client_factory
        Build each network-owning client from the shared httpx client and the
        settings. Each defaults to the real implementation; tests pass fakes, so
        the whole application can be driven with no socket.
    check_stripe_key_at_startup
        Whether to probe the Stripe key's permissions once, in the background,
        right after startup (StripeClient.check_key_permissions). On in
        production; tests that count Stripe requests turn it off, and the tests
        of the probe itself turn it on.

    Returns
    -------
    FastAPI
        The application, with no connection opened yet -- every client is
        built in the lifespan handler, at startup.

    Notes
    -----
    Nothing that holds a socket or an asyncio primitive is constructed here.
    The httpx pool and the guild cache's refresh lock are created inside the
    lifespan, i.e. inside the running event loop, for the reason
    aura.db.connection spells out for the bot: an asyncio object built under
    one loop and used under another fails late, intermittently, and in tests
    before production.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        """Build every network-owning object at startup and close it at shutdown.

        Parameters
        ----------
        app
            The application to attach the shared context to.

        Yields
        ------
        None
            For the life of the running service.

        Notes
        -----
        Everything constructed here -- the httpx pool, the guild cache's
        refresh lock, the background reconciler -- is created inside the
        running event loop and torn down when it exits, which is what keeps
        an asyncio object from being built under one loop and used under
        another.
        """
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
            stripe_client = (
                stripe_client_factory(http, settings)
                if stripe_client_factory is not None
                else StripeClient(
                    http,
                    api_base=settings.stripe_api_base,
                    secret_key=settings.stripe_secret_key,
                    price_id=settings.stripe_price_id,
                    checkout_success_url=settings.checkout_success_url,
                    checkout_cancel_url=settings.checkout_cancel_url,
                    portal_return_url=settings.billing_portal_return_url,
                    portal_configuration_id=settings.stripe_portal_configuration_id,
                )
            )
            bot_billing = (
                bot_billing_client_factory(http, settings)
                if bot_billing_client_factory is not None
                else BotBillingClient(
                    http,
                    base_url=settings.bot_internal_api_url,
                    secret=settings.bot_internal_api_secret,
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
                stripe=stripe_client,
                bot_billing=bot_billing,
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
            # Never the key itself: its mode and the Price are all an operator
            # needs to confirm the right configuration is live.
            logger.info(
                "Stripe billing ready: %s mode, price %s, portal configuration %s, "
                "reconciliation every %.0fs",
                "LIVE" if settings.stripe_live_mode else "test",
                settings.stripe_price_id,
                settings.stripe_portal_configuration_id or "account default",
                settings.stripe_reconcile_interval_seconds,
            )
            reconciler = asyncio.create_task(
                run_reconciler(
                    stripe=stripe_client,
                    bot=bot_billing,
                    live_mode=settings.stripe_live_mode,
                    interval_seconds=settings.stripe_reconcile_interval_seconds,
                )
            )
            # A background task, not an awaited call: the probe is advisory,
            # and a Stripe that is slow or unreachable at the moment the
            # container starts must not hold the service back from serving.
            key_check = (
                asyncio.create_task(check_stripe_key(stripe_client))
                if check_stripe_key_at_startup
                else None
            )
            app.state.stripe_key_check = key_check
            try:
                yield
            finally:
                background = [reconciler] if key_check is None else [reconciler, key_check]
                for task in background:
                    task.cancel()
                    with suppress(asyncio.CancelledError):
                        await task

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
    app.include_router(billing_router)
    app.include_router(stripe_webhook_router)
    return app


def build_app() -> FastAPI:
    """Load settings from the environment and build the application.

    Returns
    -------
    FastAPI
        The application, configured from the process environment.

    Raises
    ------
    WebConfigurationError
        If the environment is missing or contradicts itself. Deliberately
        not caught: a misconfigured service should refuse to start.

    Notes
    -----
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
