"""Phase 4c's settings: safe defaults, and every misconfiguration refused at startup with a readable reason."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from aura.config import BillingMode, ConfigurationError, Settings, load_settings

REPO_ROOT = Path(__file__).resolve().parent.parent


def settings(**overrides: object) -> Settings:
    return Settings(_env_file=None, discord_token="fake-token", **overrides)  # type: ignore[call-arg]


class TestDefaults:
    def test_billing_is_not_enforced_until_an_operator_says_so(self) -> None:
        configured = settings()

        assert configured.billing_mode is BillingMode.DISABLED
        assert configured.internal_api_secret is None
        assert configured.internal_api_host == "127.0.0.1"
        assert configured.internal_api_port == 8081
        assert configured.billing_renewal_grace_hours == 72.0
        assert configured.billing_payment_grace_days == 7.0
        assert configured.complimentary_guild_ids == frozenset()

    def test_the_shipped_env_example_is_itself_a_valid_configuration(self) -> None:
        """A deployment that copies .env.example must start, billing lines included."""
        configured = Settings(_env_file=REPO_ROOT / ".env.example", discord_token="fake-token")  # type: ignore[call-arg]

        assert configured.billing_mode is BillingMode.DISABLED


class TestEnforcement:
    def test_enforcement_without_the_internal_api_is_refused_with_the_reason(
        self, monkeypatch
    ) -> None:
        monkeypatch.setenv("DISCORD_TOKEN", "fake-token")
        monkeypatch.setenv("BILLING_MODE", "enforced")
        monkeypatch.delenv("INTERNAL_API_SECRET", raising=False)
        monkeypatch.chdir("/")

        with pytest.raises(ConfigurationError) as raised:
            load_settings()

        assert "INTERNAL_API_SECRET" in str(raised.value)

    def test_enforcement_with_the_internal_api_is_accepted(self) -> None:
        assert (
            settings(billing_mode="enforced", internal_api_secret="s" * 32).billing_mode
            is BillingMode.ENFORCED
        )

    @pytest.mark.parametrize("mode", ["on", "ENFORCED ", "true", ""])
    def test_an_unknown_mode_is_refused(self, mode: str) -> None:
        with pytest.raises(ValidationError):
            settings(billing_mode=mode)


class TestInternalApiSecret:
    @pytest.mark.parametrize("blank", ["", "   "])
    def test_a_blank_secret_means_unset_rather_than_a_crash(self, blank: str) -> None:
        assert settings(internal_api_secret=blank).internal_api_secret is None

    @pytest.mark.parametrize(
        "secret",
        [
            "short",
            "s" * 31,
            "has a space and is otherwise long enough 00",
            "ünïcödé-secret-long-enough-000000000",
            "tab\tseparated-secret-long-enough-00000",
        ],
    )
    def test_a_weak_or_header_unsafe_secret_is_refused(self, secret: str) -> None:
        with pytest.raises(ValidationError) as raised:
            settings(internal_api_secret=secret)

        assert secret not in str(raised.value) or len(secret) < 8


class TestNoSecretInErrors:
    def test_a_refused_configuration_does_not_carry_any_secret_in_its_error(self) -> None:
        """Enforcement refused at the model level must not echo the other secrets validated with it."""
        with pytest.raises(ValidationError) as raised:
            settings(
                billing_mode="enforced",
                llm_api_key="sk-or-never-echo-this",
                internal_api_secret=None,
            )

        assert "sk-or-never-echo-this" not in str(raised.value)
        assert "fake-token" not in str(raised.value)


class TestComplimentaryGuilds:
    def test_entries_are_trimmed_and_empty_entries_ignored(self) -> None:
        configured = settings(
            billing_complimentary_guild_ids=" 100000000000000001, ,200000000000000002,"
        )

        assert configured.complimentary_guild_ids == frozenset(
            {100000000000000001, 200000000000000002}
        )

    @pytest.mark.parametrize(
        "entry", ["abc", "0", "-5", "1.5", "9223372036854775808", "١٢٣", "1e18"]
    )
    def test_an_entry_that_is_not_a_guild_id_refuses_startup_and_is_named(self, entry: str) -> None:
        with pytest.raises(ValidationError) as raised:
            settings(billing_complimentary_guild_ids=f"100000000000000001,{entry}")

        # Named by the validator's own message -- the one part of the error
        # that survives hide_input_in_errors, and the part an operator reads.
        assert entry in str(raised.value)


class TestOtherBillingSettings:
    @pytest.mark.parametrize("url", ["/dashboard", "aura.example", "javascript:alert(1)"])
    def test_a_dashboard_url_must_be_absolute(self, url: str) -> None:
        with pytest.raises(ValidationError):
            settings(billing_dashboard_url=url)

    def test_a_blank_dashboard_url_means_unset(self) -> None:
        assert settings(billing_dashboard_url=" ").billing_dashboard_url is None

    @pytest.mark.parametrize(
        "field, value",
        [
            ("billing_renewal_grace_hours", -1),
            ("billing_renewal_grace_hours", float("nan")),
            ("billing_renewal_grace_hours", 24 * 31),
            ("billing_payment_grace_days", -0.1),
            ("billing_payment_grace_days", float("inf")),
            ("billing_payment_grace_days", 61),
            ("internal_api_port", 0),
            ("internal_api_port", 65536),
            ("internal_api_host", "  "),
        ],
    )
    def test_a_nonsense_value_is_refused(self, field: str, value: object) -> None:
        with pytest.raises(ValidationError):
            settings(**{field: value})
