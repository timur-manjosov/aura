"""Application configuration loaded from environment variables and `.env`."""
from __future__ import annotations

from pydantic import Field, ValidationError, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ENV_EXAMPLE_HINT = "Copy .env.example to .env and fill in the required values."


class ConfigurationError(Exception):
    """Raised when application configuration is missing or invalid.

    Kept distinct from pydantic's ValidationError so callers (main.py) can
    catch one application-specific exception and print an actionable
    message, instead of parsing pydantic's internal error structure.
    """


class Settings(BaseSettings):
    """Typed, validated application settings.

    Values are sourced from real process environment variables first, then
    from a `.env` file (see `.env.example`); environment variables take
    precedence. Field names are matched to environment variables
    case-insensitively (e.g. `discord_token` <-> `DISCORD_TOKEN`).
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )

    # No default (would make a blank/missing token indistinguishable from a
    # deliberate empty value); validate_default=True forces the validator
    # below to run even when the variable is absent entirely.
    discord_token: str = Field(default="", validate_default=True)
    llm_provider: str = "anthropic"
    llm_api_key: str | None = None
    database_path: str = "data/aura.db"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    log_level: str = "INFO"

    @field_validator("discord_token")
    @classmethod
    def _require_non_blank_token(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError(f"DISCORD_TOKEN is missing or blank. {ENV_EXAMPLE_HINT}")
        return stripped


def load_settings() -> Settings:
    """Load and validate settings, raising ConfigurationError on failure.

    This is the entry point production code (main.py) should use. It
    translates pydantic's ValidationError into a single plain-text message
    so a misconfigured deployment fails immediately with a readable cause
    instead of a traceback surfacing three layers down inside discord.py.
    """
    try:
        return Settings()
    except ValidationError as exc:
        messages = [
            str(error.get("ctx", {}).get("error", error["msg"])) for error in exc.errors()
        ]
        raise ConfigurationError(" ".join(messages)) from exc
