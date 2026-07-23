"""Tests for aura.config: settings loading and validation.

Every test clears the relevant environment variables first (clean_env,
below) so behavior is deterministic regardless of what's actually exported
in the shell running the tests, or whether a real .env happens to exist in
the current working directory.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from aura.config import ConfigurationError, Settings, load_settings

_ENV_KEYS = (
    "DISCORD_TOKEN",
    "LLM_PROVIDER",
    "LLM_API_KEY",
    "DATABASE_PATH",
    "EMBEDDING_MODEL",
    "LOG_LEVEL",
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in _ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


def _settings(**overrides: str) -> Settings:
    """Build Settings from explicit values only, ignoring any real .env file on disk."""
    return Settings(_env_file=None, **overrides)  # type: ignore[call-arg]


class TestDiscordTokenValidation:
    def test_valid_token_is_accepted_and_stripped(self) -> None:
        settings = _settings(discord_token="  abc123  ")
        assert settings.discord_token == "abc123"

    def test_missing_token_raises(self) -> None:
        with pytest.raises(ValueError, match="DISCORD_TOKEN"):
            _settings()

    def test_missing_token_error_points_to_env_example(self) -> None:
        with pytest.raises(ValueError, match=r"\.env\.example"):
            _settings()

    def test_empty_string_token_raises(self) -> None:
        with pytest.raises(ValueError, match="DISCORD_TOKEN"):
            _settings(discord_token="")

    def test_whitespace_only_token_raises(self) -> None:
        with pytest.raises(ValueError, match="DISCORD_TOKEN"):
            _settings(discord_token="   \t\n  ")


class TestDefaults:
    def test_default_values(self) -> None:
        settings = _settings(discord_token="valid-token")
        assert settings.llm_provider == "anthropic"
        assert settings.llm_api_key is None
        assert settings.database_path == "data/aura.db"
        assert settings.embedding_model == "BAAI/bge-small-en-v1.5"
        assert settings.log_level == "INFO"

    def test_overrides_are_respected(self) -> None:
        settings = _settings(
            discord_token="valid-token",
            llm_provider="openai",
            log_level="DEBUG",
        )
        assert settings.llm_provider == "openai"
        assert settings.log_level == "DEBUG"


class TestLoadSettings:
    """Integration-level tests for the production entry point, load_settings()."""

    def test_missing_token_raises_configuration_error(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)  # no .env file here
        with pytest.raises(ConfigurationError, match="DISCORD_TOKEN"):
            load_settings()

    def test_configuration_error_mentions_env_example(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        with pytest.raises(ConfigurationError, match=r"\.env\.example"):
            load_settings()

    def test_whitespace_only_env_var_raises_configuration_error(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("DISCORD_TOKEN", "   ")
        with pytest.raises(ConfigurationError, match="DISCORD_TOKEN"):
            load_settings()

    def test_succeeds_with_real_env_var_and_no_dotenv_file(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("DISCORD_TOKEN", "real-token-value")
        settings = load_settings()
        assert settings.discord_token == "real-token-value"

    def test_reads_token_from_dotenv_file(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".env").write_text("DISCORD_TOKEN=from-dotenv-file\n", encoding="utf-8")
        settings = load_settings()
        assert settings.discord_token == "from-dotenv-file"

    def test_real_env_var_takes_precedence_over_dotenv_file(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".env").write_text("DISCORD_TOKEN=from-file\n", encoding="utf-8")
        monkeypatch.setenv("DISCORD_TOKEN", "from-real-env")
        settings = load_settings()
        assert settings.discord_token == "from-real-env"
