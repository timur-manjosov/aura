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

from pydantic import Field, ValidationError, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ENV_EXAMPLE_HINT = "Copy web/.env.example to web/.env and fill in the required values."

# Exactly the three values Starlette's set_cookie accepts. Declared as the
# config field's type rather than checked in a validator body, so the value
# reaches set_cookie without a cast and a fourth spelling cannot be introduced
# without the type system noticing.
SameSitePolicy = Literal["lax", "strict", "none"]

# The Stripe key prefixes this service recognises. Only the test-mode pair is
# accepted unless AURA_WEB_STRIPE_ALLOW_LIVE_MODE is set: Phase 4c is built and
# verified against Stripe's test mode exclusively, and going live is the
# operator's own separate, deliberate step -- a flag they set, never a key
# pasted into the wrong line.
STRIPE_TEST_KEY_PREFIXES = ("sk_test_", "rk_test_")
STRIPE_LIVE_KEY_PREFIXES = ("sk_live_", "rk_live_")

# The shortest shared secret accepted for the bot's internal billing API --
# the same floor the bot itself enforces (aura.config).
MIN_BOT_INTERNAL_API_SECRET_LENGTH = 32

# Stripe object IDs and secrets are ASCII letters, digits and underscores. A
# value carrying anything else is a copy-paste accident (a quote, a space, a
# newline), and one that would travel into a header or a form body.
_STRIPE_TOKEN_CHARACTERS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"
)

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


def _require_stripe_token(value: str, *, field_name: str, prefixes: tuple[str, ...]) -> str:
    """Reject a Stripe credential or ID with the wrong prefix or a stray character.

    The value itself is never echoed into the error: a secret key pasted into
    the wrong variable must not end up in a startup log line.
    """
    matched = next((prefix for prefix in prefixes if value.startswith(prefix)), None)
    if matched is None or len(value) == len(matched) or len(value) > 255:
        raise ValueError(
            f"{field_name} must be one of {', '.join(prefixes)} followed by the rest of the value. "
            + ENV_EXAMPLE_HINT
        )
    if not set(value) <= _STRIPE_TOKEN_CHARACTERS:
        raise ValueError(
            f"{field_name} contains characters a Stripe value never has (check for quotes, "
            "spaces or a trailing newline). " + ENV_EXAMPLE_HINT
        )
    return value


class WebSettings(BaseSettings):
    """Typed, validated configuration for the OAuth2 web backend."""

    model_config = SettingsConfigDict(
        env_prefix="AURA_WEB_",
        env_file="web/.env",
        env_file_encoding="utf-8",
        extra="ignore",
        # A refused value is never echoed into the error. Pydantic's default is
        # to include `input_value` -- and for a model-level validator, every
        # input at once -- so without this a single `logger.exception` around
        # settings loading would write the Stripe key, the webhook secret and
        # the bot token into a log line.
        hide_input_in_errors=True,
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

    # --- Stripe (Phase 4c) ---------------------------------------------------
    # The API key this service calls Stripe with. A restricted key (rk_) with
    # only the permissions listed in web/.env.example is recommended over a
    # full secret key (sk_). Never sent to the browser, never logged.
    stripe_secret_key: str = Field(default="", validate_default=True)
    # The signing secret of the webhook endpoint (whsec_...). Every webhook
    # request is verified against it before its body is read as an event.
    stripe_webhook_secret: str = Field(default="", validate_default=True)
    # The recurring Price a Pro subscription is created for. Config-only: the
    # browser chooses a guild, never what it is charged.
    stripe_price_id: str = Field(default="", validate_default=True)
    # Off by default, and refused at startup when a live key is configured
    # without it. See STRIPE_LIVE_KEY_PREFIXES above for why.
    stripe_allow_live_mode: bool = False
    stripe_api_base: str = "https://api.stripe.com"

    # Where Stripe sends the browser back to. Config-only for the same reason
    # post_login_redirect_url is: a caller-chosen return target is an open
    # redirect with a payment page in the middle of it.
    checkout_success_url: str = "http://localhost:3000/?checkout=success"
    checkout_cancel_url: str = "http://localhost:3000/?checkout=cancelled"
    billing_portal_return_url: str = "http://localhost:3000/"

    # How often every subscription is re-fetched from Stripe and pushed to the
    # bot, independent of webhooks. Stripe retries an undelivered webhook for
    # three days in live mode and a few hours in a sandbox; an event lost past
    # that would otherwise leave a paying guild on Free until its next
    # renewal. Six hours bounds that to well inside one renewal grace period.
    stripe_reconcile_interval_seconds: float = Field(
        default=6 * 3600.0, ge=60.0, le=7 * 24 * 3600.0
    )

    # --- The bot's internal billing API (Phase 4c) --------------------------
    # This service never opens Aura's database (see web/README.md); it hands
    # subscription snapshots to the bot process, which stays the only writer.
    bot_internal_api_url: str = Field(default="", validate_default=True)
    bot_internal_api_secret: str = Field(default="", validate_default=True)

    log_level: str = "INFO"

    @field_validator(
        "discord_client_id",
        "discord_client_secret",
        "discord_bot_token",
        "stripe_secret_key",
        "stripe_webhook_secret",
        "stripe_price_id",
        "bot_internal_api_url",
        "bot_internal_api_secret",
    )
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

    @field_validator("stripe_secret_key")
    @classmethod
    def _stripe_secret_key_shape(cls, value: str) -> str:
        return _require_stripe_token(
            value,
            field_name="STRIPE_SECRET_KEY",
            prefixes=STRIPE_TEST_KEY_PREFIXES + STRIPE_LIVE_KEY_PREFIXES,
        )

    @field_validator("stripe_webhook_secret")
    @classmethod
    def _stripe_webhook_secret_shape(cls, value: str) -> str:
        return _require_stripe_token(
            value, field_name="STRIPE_WEBHOOK_SECRET", prefixes=("whsec_",)
        )

    @field_validator("stripe_price_id")
    @classmethod
    def _stripe_price_id_shape(cls, value: str) -> str:
        return _require_stripe_token(value, field_name="STRIPE_PRICE_ID", prefixes=("price_",))

    @field_validator("checkout_success_url")
    @classmethod
    def _checkout_success_url_is_absolute(cls, value: str) -> str:
        return _require_absolute_http_url(value, "CHECKOUT_SUCCESS_URL")

    @field_validator("checkout_cancel_url")
    @classmethod
    def _checkout_cancel_url_is_absolute(cls, value: str) -> str:
        return _require_absolute_http_url(value, "CHECKOUT_CANCEL_URL")

    @field_validator("billing_portal_return_url")
    @classmethod
    def _portal_return_url_is_absolute(cls, value: str) -> str:
        return _require_absolute_http_url(value, "BILLING_PORTAL_RETURN_URL")

    @field_validator("stripe_api_base")
    @classmethod
    def _stripe_api_base_is_absolute(cls, value: str) -> str:
        _require_absolute_http_url(value, "STRIPE_API_BASE")
        return value.rstrip("/")

    @field_validator("bot_internal_api_url")
    @classmethod
    def _bot_internal_api_url_is_absolute(cls, value: str) -> str:
        _require_absolute_http_url(value, "BOT_INTERNAL_API_URL")
        return value.rstrip("/")

    @field_validator("bot_internal_api_secret")
    @classmethod
    def _bot_internal_api_secret_is_strong(cls, value: str) -> str:
        if len(value) < MIN_BOT_INTERNAL_API_SECRET_LENGTH:
            raise ValueError(
                f"BOT_INTERNAL_API_SECRET must be at least {MIN_BOT_INTERNAL_API_SECRET_LENGTH} "
                "characters and equal to INTERNAL_API_SECRET in the bot's .env. " + ENV_EXAMPLE_HINT
            )
        if not all(
            character.isascii() and character.isprintable() and not character.isspace()
            for character in value
        ):
            raise ValueError(
                "BOT_INTERNAL_API_SECRET may contain only printable ASCII characters without spaces."
            )
        return value

    @model_validator(mode="after")
    def _live_mode_is_a_deliberate_decision(self) -> WebSettings:
        """Refuse a live Stripe key unless live mode was switched on explicitly."""
        if (
            self.stripe_secret_key.startswith(STRIPE_LIVE_KEY_PREFIXES)
            and not self.stripe_allow_live_mode
        ):
            raise ValueError(
                "STRIPE_SECRET_KEY is a LIVE key, and STRIPE_ALLOW_LIVE_MODE is not set. Live "
                "payments are a separate, deliberate step: use a test key (sk_test_/rk_test_) "
                "or set AURA_WEB_STRIPE_ALLOW_LIVE_MODE=true on purpose."
            )
        return self

    @property
    def stripe_live_mode(self) -> bool:
        """Whether the configured key is a live-mode key (events must then be live too).

        Returns
        -------
        bool
            True when the configured secret key is a live-mode key, in which case
            only live events are accepted.
        """
        return self.stripe_secret_key.startswith(STRIPE_LIVE_KEY_PREFIXES)

    @property
    def frontend_origin(self) -> str:
        """The single browser origin this service serves, derived from the post-login URL.

        Returns
        -------
        str
            The single scheme-and-host origin this service serves, derived from the
            post-login URL so the two cannot disagree.

        Notes
        -----
        Mutating billing requests must come from here. Derived rather than
        configured separately so the two can never disagree.
        """
        parsed = urlparse(self.post_login_redirect_url)
        return f"{parsed.scheme}://{parsed.netloc}".lower()

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

    Returns
    -------
    WebSettings
        A fully validated configuration.

    Raises
    ------
    WebConfigurationError
        On any validation failure, carrying one plain-text message rather
        than pydantic's structured error.

    Notes
    -----
    The entry point production code should use, for the same reason
    aura.config.load_settings exists: it flattens pydantic's error structure
    into one readable line so a misconfigured container fails immediately with
    an actionable cause instead of a traceback from inside uvicorn.
    """
    try:
        return WebSettings()
    except ValidationError as exc:
        # include_input=False: the flattened message is built from the
        # validators' own text, and the raw values -- secrets among them --
        # are not even handed to this loop.
        messages = [
            str(error.get("ctx", {}).get("error", error["msg"]))
            for error in exc.errors(include_input=False)
        ]
        raise WebConfigurationError(" ".join(messages)) from exc
