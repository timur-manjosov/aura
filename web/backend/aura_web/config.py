"""Configuration for the web service, deliberately separate from the bot's.

`aura.config.Settings` is not imported here, and that is the point. The bot
process and this one are different containers with different secrets and
different failure modes: the bot needs DISCORD_TOKEN, an LLM key and a
writable database path; this service needs an OAuth2 client secret and
nothing else the bot has. Sharing one Settings class would mean every
deployment of either service had to satisfy the other's required fields, and
a web container would be handed credentials it has no use for -- the opposite
of the container isolation this sub-phase exists to establish.

Every variable is read with an ``AURA_WEB_`` prefix so that mounting the
bot's own ``.env`` into this container by accident configures nothing rather
than configuring something subtly wrong.
"""
from __future__ import annotations

from typing import Literal
from urllib.parse import urlparse

from pydantic import Field, ValidationError, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ENV_EXAMPLE_HINT = "Copy web/.env.example to web/.env and fill in the required values."

# Exactly the three values Starlette's set_cookie accepts. Declared as the
# config field's type rather than checked in a validator body, so the value
# reaches set_cookie without a cast and a fourth spelling cannot be introduced
# without the type system noticing.
SameSitePolicy = Literal["lax", "strict", "none"]

# Discord's own documented ceiling for GET /users/@me/guilds. Used as the page
# size for every guild listing so the common case (a bot in fewer than 200
# guilds, a user in fewer than 200 guilds) costs exactly one request against a
# route Discord rate-limits tightly.
DISCORD_GUILD_PAGE_SIZE = 200


class WebConfigurationError(Exception):
    """Raised when the web service's configuration is missing or invalid.

    Mirrors aura.config.ConfigurationError's role in the bot process: one
    application-specific exception the entry point can catch to print an
    actionable message, instead of a pydantic error structure surfacing
    through uvicorn's startup traceback.
    """


def _require_absolute_http_url(value: str, field_name: str) -> str:
    """Reject anything that is not an absolute http(s) URL with a host.

    Applied to both configured URLs because a relative or scheme-less value
    fails in two different unhelpful ways -- Discord rejects the authorize
    request for redirect_uri, and the browser resolves a scheme-less redirect
    against our own origin for the post-login target -- neither of which
    points at the actual typo.
    """
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(
            f"{field_name} must be an absolute http(s) URL including a host, got {value!r}. "
            + ENV_EXAMPLE_HINT
        )
    return value


class WebSettings(BaseSettings):
    """Typed, validated configuration for the OAuth2 web backend."""

    model_config = SettingsConfigDict(
        env_prefix="AURA_WEB_",
        env_file="web/.env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # No defaults on the three credentials: a blank value and a missing one
    # must be indistinguishable failures, both caught by the validator below,
    # rather than one of them starting a service that 401s on every login.
    discord_client_id: str = Field(default="", validate_default=True)
    discord_client_secret: str = Field(default="", validate_default=True)
    # The bot's own token, used for exactly one read-only question: which
    # guilds is Aura actually in. See aura_web.discord_api.DiscordClient.
    # fetch_bot_guild_ids for why that question is asked of Discord rather
    # than of Aura's database, and web/README.md for the blast-radius
    # trade-off this choice accepts.
    discord_bot_token: str = Field(default="", validate_default=True)

    # Must byte-for-byte match one of the redirect URIs registered on the
    # Discord application, and is sent twice (authorize, then token exchange)
    # because Discord verifies it on both legs.
    oauth_redirect_uri: str = "http://localhost:3000/api/auth/callback"
    # Where the browser is sent after a successful callback. Config-only, and
    # deliberately NOT overridable by a query parameter: letting the caller
    # choose the post-login destination is the textbook open-redirect hole in
    # an OAuth callback, and there is no 4b feature that needs it.
    post_login_redirect_url: str = "http://localhost:3000/"

    session_cookie_name: str = "aura_session"
    oauth_state_cookie_name: str = "aura_oauth_state"
    # Secure by default. http://localhost is a trustworthy origin in current
    # browsers, so Secure cookies work in local development without loosening
    # this; it exists as a switch only for a plain-HTTP host that is not
    # localhost, which should be a conscious act rather than a default.
    session_cookie_secure: bool = True
    # Lax, not Strict: the OAuth callback is a top-level cross-site navigation
    # from discord.com, and Strict would withhold the state cookie exactly
    # there -- breaking the CSRF check it exists to perform. Lax still blocks
    # the cross-site sub-requests that CSRF actually rides on.
    session_cookie_samesite: SameSitePolicy = "lax"

    session_ttl_seconds: int = Field(default=7 * 24 * 3600, gt=0)
    oauth_state_ttl_seconds: int = Field(default=600, gt=0)
    # Both stores are in-memory and grow on unauthenticated input (anyone can
    # hit /api/auth/login), so both need a ceiling. See aura_web.sessions for
    # the eviction order these bounds trigger.
    max_sessions: int = Field(default=10_000, gt=0)
    max_pending_states: int = Field(default=10_000, gt=0)

    discord_api_base: str = "https://discord.com/api/v10"
    http_timeout_seconds: float = Field(default=10.0, gt=0)
    # How long the bot's guild-membership list is reused before refetching.
    # /users/@me/guilds is the route Discord rate-limits hardest for dashboard
    # workloads, and bot membership changes on human timescales, so a short
    # cache costs nothing in freshness and removes one Discord round trip from
    # every page load.
    bot_guilds_cache_ttl_seconds: float = Field(default=60.0, gt=0)
    # How far past the TTL a cached list may still be served when a refresh
    # fails. Serving a five-minute-old membership list through a Discord
    # hiccup is strictly better than showing a logged-in user an error page;
    # past that the list is dropped and the request fails closed rather than
    # answering from something stale enough to be wrong.
    bot_guilds_stale_tolerance_seconds: float = Field(default=300.0, ge=0)

    log_level: str = "INFO"

    @field_validator("discord_client_id", "discord_client_secret", "discord_bot_token")
    @classmethod
    def _reject_blank_credentials(cls, value: str, info: object) -> str:
        field_name = getattr(info, "field_name", "credential")
        if not value.strip():
            raise ValueError(
                f"{field_name.upper()} is required (set AURA_WEB_{field_name.upper()}). "
                + ENV_EXAMPLE_HINT
            )
        return value.strip()

    @field_validator("discord_client_id")
    @classmethod
    def _client_id_is_a_snowflake(cls, value: str) -> str:
        """Reject a client ID that is not a bare snowflake.

        It is interpolated into the authorize URL the browser is redirected
        to; a value carrying a ``&`` or ``#`` would silently become extra
        query parameters on that URL rather than a wrong client ID, which is
        a far harder thing to diagnose from the Discord error page.
        """
        if not value.isdigit():
            raise ValueError(
                f"DISCORD_CLIENT_ID must be a numeric Discord snowflake, got {value!r}. "
                + ENV_EXAMPLE_HINT
            )
        return value

    @field_validator("oauth_redirect_uri")
    @classmethod
    def _redirect_uri_is_absolute(cls, value: str) -> str:
        return _require_absolute_http_url(value, "OAUTH_REDIRECT_URI")

    @field_validator("post_login_redirect_url")
    @classmethod
    def _post_login_url_is_absolute(cls, value: str) -> str:
        return _require_absolute_http_url(value, "POST_LOGIN_REDIRECT_URL")

    @field_validator("discord_api_base")
    @classmethod
    def _api_base_is_absolute(cls, value: str) -> str:
        """Validate and normalise the API base, trailing slash included.

        Every call site builds its URL as ``f"{api_base}/oauth2/token"``; a
        configured trailing slash would produce a double slash that Discord
        answers with a 404 rather than a useful error.
        """
        _require_absolute_http_url(value, "DISCORD_API_BASE")
        return value.rstrip("/")

    @field_validator("session_cookie_samesite", mode="before")
    @classmethod
    def _normalise_samesite(cls, value: object) -> object:
        """Lower-case and trim before the Literal above decides.

        Runs in "before" mode so `SAMESITE=Lax` from a .env is accepted and
        normalised, while anything outside the three legal values is still
        rejected by the type rather than by a hand-written membership check --
        which is what lets the value be passed straight to Starlette's
        set_cookie, whose own parameter is that same Literal.
        """
        return value.strip().lower() if isinstance(value, str) else value

    @field_validator("session_cookie_name", "oauth_state_cookie_name")
    @classmethod
    def _cookie_name_is_a_token(cls, value: str) -> str:
        """Reject cookie names containing characters that would break Set-Cookie.

        A name with a space, semicolon or equals sign does not fail loudly --
        it emits a header the browser silently discards or, worse, parses as
        two attributes, which would look like "the session just never sticks."
        """
        stripped = value.strip()
        if not stripped or not all(
            character.isalnum() or character in "_-." for character in stripped
        ):
            raise ValueError(
                f"cookie names must be non-empty and alphanumeric/._- only, got {value!r}"
            )
        return stripped


def load_web_settings() -> WebSettings:
    """Load and validate web settings, raising WebConfigurationError on failure.

    The entry point production code should use, for the same reason
    aura.config.load_settings exists: it flattens pydantic's error structure
    into one readable line so a misconfigured container fails immediately with
    an actionable cause instead of a traceback from inside uvicorn.
    """
    try:
        return WebSettings()
    except ValidationError as exc:
        messages = [
            str(error.get("ctx", {}).get("error", error["msg"])) for error in exc.errors()
        ]
        raise WebConfigurationError(" ".join(messages)) from exc
