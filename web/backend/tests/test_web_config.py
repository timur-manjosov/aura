"""Configuration validation: the misconfigurations that must fail at startup, loudly.

A web service that starts with a blank client secret does not fail at
startup -- it fails on every login, hours later, as a Discord error page
nobody can trace back to a missing environment variable. Each of these turns
one of those into a refusal at construction.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from aura_web.config import WebConfigurationError, WebSettings, load_web_settings

VALID = {
    "discord_client_id": "123456789012345678",
    "discord_client_secret": "a-secret",
    "discord_bot_token": "a-bot-token",
    "stripe_secret_key": "sk_test_aValidLookingTestKey",
    "stripe_webhook_secret": "whsec_aValidLookingSecret",
    "stripe_price_id": "price_aValidLookingPrice",
    "bot_internal_api_url": "http://aura:8081",
    "bot_internal_api_secret": "an-internal-api-secret-that-is-long-enough-000",
}


class TestRequiredCredentials:
    @pytest.mark.parametrize("field", list(VALID))
    def test_a_missing_credential_is_rejected(self, field: str) -> None:
        values = dict(VALID)
        values.pop(field)

        with pytest.raises(ValidationError):
            WebSettings(_env_file=None, **values)

    @pytest.mark.parametrize("field", list(VALID))
    @pytest.mark.parametrize("blank", ["", "   ", "\t\n"])
    def test_a_blank_credential_is_rejected_like_a_missing_one(
        self, field: str, blank: str
    ) -> None:
        """Otherwise `AURA_WEB_DISCORD_CLIENT_SECRET=` reads as "configured"."""
        values = dict(VALID) | {field: blank}

        with pytest.raises(ValidationError):
            WebSettings(_env_file=None, **values)

    def test_surrounding_whitespace_is_stripped(self) -> None:
        """A trailing newline from a copy-paste must not become part of the secret."""
        settings = WebSettings(_env_file=None, **(dict(VALID) | {"discord_client_secret": " s \n"}))

        assert settings.discord_client_secret == "s"


class TestClientId:
    @pytest.mark.parametrize(
        "client_id",
        ["abc", "123&redirect_uri=evil", "123 456", "12.3", "-1", "123#x", "<script>"],
    )
    def test_a_non_snowflake_client_id_is_rejected(self, client_id: str) -> None:
        """It is interpolated into the authorize URL; a `&` would add query parameters."""
        with pytest.raises(ValidationError):
            WebSettings(_env_file=None, **(dict(VALID) | {"discord_client_id": client_id}))


class TestUrls:
    @pytest.mark.parametrize(
        "url",
        ["", "/api/auth/callback", "frontend.test/cb", "ftp://x/y", "javascript:alert(1)", "://x"],
    )
    def test_a_non_absolute_redirect_uri_is_rejected(self, url: str) -> None:
        with pytest.raises(ValidationError):
            WebSettings(_env_file=None, **(dict(VALID) | {"oauth_redirect_uri": url}))

    @pytest.mark.parametrize("url", ["", "/", "javascript:alert(1)", "data:text/html,x"])
    def test_a_non_absolute_post_login_url_is_rejected(self, url: str) -> None:
        with pytest.raises(ValidationError):
            WebSettings(_env_file=None, **(dict(VALID) | {"post_login_redirect_url": url}))

    def test_a_trailing_slash_on_the_api_base_is_normalised_away(self) -> None:
        """Every call site builds f"{base}/oauth2/token"; a double slash 404s."""
        settings = WebSettings(
            _env_file=None, **(dict(VALID) | {"discord_api_base": "https://discord.com/api/v10/"})
        )

        assert settings.discord_api_base == "https://discord.com/api/v10"


class TestCookieSettings:
    def test_secure_is_the_default(self) -> None:
        assert WebSettings(_env_file=None, **VALID).session_cookie_secure is True

    def test_lax_is_the_default_samesite(self) -> None:
        """Strict would withhold the state cookie on the callback navigation."""
        assert WebSettings(_env_file=None, **VALID).session_cookie_samesite == "lax"

    @pytest.mark.parametrize("value", ["LAX", "Strict", "none"])
    def test_a_known_samesite_is_normalised(self, value: str) -> None:
        settings = WebSettings(_env_file=None, **(dict(VALID) | {"session_cookie_samesite": value}))

        assert settings.session_cookie_samesite == value.lower()

    @pytest.mark.parametrize("value", ["", "sameorigin", "true", "laxx"])
    def test_an_unknown_samesite_is_rejected(self, value: str) -> None:
        with pytest.raises(ValidationError):
            WebSettings(_env_file=None, **(dict(VALID) | {"session_cookie_samesite": value}))

    @pytest.mark.parametrize(
        "name", ["", "   ", "aura session", "aura;session", "aura=session", "aura\nsession"]
    )
    def test_a_cookie_name_that_would_break_set_cookie_is_rejected(self, name: str) -> None:
        """A bad name does not fail loudly -- it looks like "the session never sticks"."""
        with pytest.raises(ValidationError):
            WebSettings(_env_file=None, **(dict(VALID) | {"session_cookie_name": name}))


class TestBounds:
    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("session_ttl_seconds", 0),
            ("session_ttl_seconds", -1),
            ("oauth_state_ttl_seconds", 0),
            ("max_sessions", 0),
            ("max_pending_states", 0),
            ("http_timeout_seconds", 0),
            ("bot_guilds_cache_ttl_seconds", 0),
            ("bot_guilds_stale_tolerance_seconds", -1),
        ],
    )
    def test_a_nonsense_bound_is_rejected(self, field: str, value: int) -> None:
        with pytest.raises(ValidationError):
            WebSettings(_env_file=None, **(dict(VALID) | {field: value}))


class TestLoadWebSettings:
    def test_a_configuration_failure_becomes_one_readable_message(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        for name in list(VALID):
            monkeypatch.delenv(f"AURA_WEB_{name.upper()}", raising=False)
        monkeypatch.chdir("/")

        with pytest.raises(WebConfigurationError) as exc_info:
            load_web_settings()

        assert "DISCORD_CLIENT_ID" in str(exc_info.value)

    def test_a_complete_environment_loads(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for name, value in VALID.items():
            monkeypatch.setenv(f"AURA_WEB_{name.upper()}", value)

        settings = load_web_settings()

        assert settings.discord_client_id == VALID["discord_client_id"]


class TestStripeSettings:
    @pytest.mark.parametrize("key", ["sk_test_abc", "rk_test_abc"])
    def test_a_test_mode_key_is_accepted(self, key: str) -> None:
        settings = WebSettings(_env_file=None, **(dict(VALID) | {"stripe_secret_key": key}))

        assert settings.stripe_live_mode is False

    @pytest.mark.parametrize("key", ["sk_live_realMoney123", "rk_live_realMoney123"])
    def test_a_live_key_is_refused_unless_live_mode_is_switched_on_deliberately(
        self, key: str
    ) -> None:
        with pytest.raises(ValidationError) as raised:
            WebSettings(_env_file=None, **(dict(VALID) | {"stripe_secret_key": key}))

        # The model-level refusal must not carry any input with it -- neither
        # the key nor the other secrets validated alongside it.
        # `.errors()` returns the raw input by design; what must be clean is
        # the rendered error and the input-free error list the loader uses.
        for secret in (key, VALID["discord_client_secret"], VALID["bot_internal_api_secret"]):
            assert secret not in str(raised.value)
            assert secret not in repr(raised.value)
            assert secret not in repr(raised.value.errors(include_input=False))

    def test_the_startup_path_refuses_a_live_key_without_logging_it(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        for name, value in (VALID | {"stripe_secret_key": "sk_live_neverLogThis"}).items():
            monkeypatch.setenv(f"AURA_WEB_{name.upper()}", value)
        monkeypatch.chdir("/")

        with pytest.raises(WebConfigurationError) as raised:
            load_web_settings()

        assert "LIVE" in str(raised.value)
        assert "sk_live_neverLogThis" not in str(raised.value)
        assert "sk_live_neverLogThis" not in str(raised.value.__cause__)

    def test_a_live_key_with_the_explicit_flag_is_live_mode(self) -> None:
        settings = WebSettings(
            _env_file=None,
            **(dict(VALID) | {"stripe_secret_key": "sk_live_x", "stripe_allow_live_mode": True}),
        )

        assert settings.stripe_live_mode is True

    @pytest.mark.parametrize(
        "key",
        [
            "pk_test_publishable",
            "sk_abc",
            "whsec_abc",
            "sk_test_has space",
            'sk_test_"quoted"',
            "sk_test_new\nline",
            "sk_test_" + "a" * 300,
        ],
    )
    def test_anything_that_is_not_a_secret_or_restricted_key_is_refused(self, key: str) -> None:
        with pytest.raises(ValidationError) as raised:
            WebSettings(_env_file=None, **(dict(VALID) | {"stripe_secret_key": key}))

        assert key.strip() not in str(raised.value)

    @pytest.mark.parametrize("secret", ["sk_test_abc", "whsec", "whsec_", "whsec_a b"])
    def test_the_webhook_secret_must_be_a_signing_secret(self, secret: str) -> None:
        with pytest.raises(ValidationError):
            WebSettings(_env_file=None, **(dict(VALID) | {"stripe_webhook_secret": secret}))

    @pytest.mark.parametrize(
        "price", ["prod_abc", "price_", "price_abc&mode=payment", "12", "price"]
    )
    def test_the_price_must_be_a_price_id(self, price: str) -> None:
        with pytest.raises(ValidationError):
            WebSettings(_env_file=None, **(dict(VALID) | {"stripe_price_id": price}))

    @pytest.mark.parametrize(
        "field",
        [
            "checkout_success_url",
            "checkout_cancel_url",
            "billing_portal_return_url",
            "stripe_api_base",
            "bot_internal_api_url",
        ],
    )
    @pytest.mark.parametrize("url", ["/relative", "javascript:alert(1)", "ftp://x/y"])
    def test_every_billing_url_must_be_absolute(self, field: str, url: str) -> None:
        with pytest.raises(ValidationError):
            WebSettings(_env_file=None, **(dict(VALID) | {field: url}))

    @pytest.mark.parametrize(
        "secret",
        [
            "short",
            "x" * 31,
            "has a space in it and is long enough 0000",
            "unicodé-secret-that-is-long-enough-0000000",
        ],
    )
    def test_the_internal_api_secret_must_be_long_and_header_safe(self, secret: str) -> None:
        with pytest.raises(ValidationError):
            WebSettings(_env_file=None, **(dict(VALID) | {"bot_internal_api_secret": secret}))

    def test_the_frontend_origin_is_derived_from_the_post_login_url(self) -> None:
        settings = WebSettings(
            _env_file=None,
            **(dict(VALID) | {"post_login_redirect_url": "https://Aura.Example:8443/app/"}),
        )

        assert settings.frontend_origin == "https://aura.example:8443"

    def test_reconciliation_cannot_be_set_to_hammer_stripe(self) -> None:
        with pytest.raises(ValidationError):
            WebSettings(_env_file=None, **(dict(VALID) | {"stripe_reconcile_interval_seconds": 1}))


class TestNoBleedFromTheBotsEnvironment:
    def test_the_bots_own_variables_do_not_configure_this_service(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The AURA_WEB_ prefix exists so a mis-mounted .env configures nothing.

        Handing the web container the bot's file must fail at startup, not
        half-configure a service that then behaves unpredictably.
        """
        for name in list(VALID):
            monkeypatch.delenv(f"AURA_WEB_{name.upper()}", raising=False)
        monkeypatch.setenv("DISCORD_TOKEN", "the-bots-token")
        monkeypatch.setenv("DISCORD_CLIENT_ID", "123456789012345678")
        monkeypatch.setenv("DISCORD_CLIENT_SECRET", "not-ours")
        monkeypatch.chdir("/")

        with pytest.raises(WebConfigurationError):
            load_web_settings()
