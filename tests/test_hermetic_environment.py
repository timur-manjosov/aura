"""The suite does not depend on the developer's .env or shell (V-04).

`import litellm` used to copy this repository's root .env into os.environ, so a
test building Settings(_env_file=None) still saw INTERNAL_API_SECRET -- and
four tests failed, one of them by binding a real port, the day that variable
was added to .env as DEPLOYMENT.md instructs. tests/conftest.py now closes both
routes: LITELLM_MODE before litellm is imported, and an autouse fixture that
removes every variable the settings classes read. These tests pin both.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from aura.config import ConfigurationError, Settings, load_settings
from aura_web.config import WebSettings
from tests.conftest import RUN_REAL_LLM_ENV, settings_environment_names

REPO_ROOT = Path(__file__).resolve().parents[1]

# litellm, imported in a fresh interpreter with load_dotenv replaced by a
# recorder, reports whether its import would have loaded a .env.
LOAD_DOTENV_PROBE = """
import dotenv
calls = []
dotenv.load_dotenv = lambda *args, **kwargs: calls.append(1) or True
import litellm
print(len(calls))
"""


def litellm_import_loads_dotenv(litellm_mode: str | None) -> bool:
    environment = {key: value for key, value in os.environ.items() if key != "LITELLM_MODE"}
    if litellm_mode is not None:
        environment["LITELLM_MODE"] = litellm_mode
    completed = subprocess.run(
        [sys.executable, "-c", LOAD_DOTENV_PROBE],
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    )
    return completed.stdout.strip().splitlines()[-1] != "0"


class TestLitellmCannotLoadADotenv:
    @pytest.mark.skipif(bool(os.environ.get(RUN_REAL_LLM_ENV)), reason="opt-in run uses .env")
    def test_the_test_process_runs_litellm_in_production_mode(self) -> None:
        assert os.environ.get("LITELLM_MODE") == "PRODUCTION"

    def test_production_mode_skips_the_dotenv_load(self) -> None:
        assert litellm_import_loads_dotenv("PRODUCTION") is False

    def test_the_default_mode_would_load_it(self) -> None:
        """The positive control: without it the test above could pass for a wrong reason."""
        assert litellm_import_loads_dotenv(None) is True


class TestNoSettingReachesATestUnlessItSetsIt:
    @pytest.mark.parametrize(
        "name",
        [
            "INTERNAL_API_SECRET",
            "DISCORD_TOKEN",
            "LLM_API_KEY",
            "SYNTHESIS_MODEL",
            "BILLING_MODE",
            "BILLING_COMPLIMENTARY_GUILD_IDS",
            "DATABASE_PATH",
            "AURA_WEB_STRIPE_SECRET_KEY",
            "AURA_WEB_BOT_INTERNAL_API_SECRET",
            "AURA_WEB_DISCORD_BOT_TOKEN",
        ],
    )
    def test_the_scrub_covers_the_credentials_and_the_billing_switches(self, name: str) -> None:
        assert name in settings_environment_names(keep_real_llm_settings=False)
        assert name not in os.environ

    def test_every_bot_setting_is_covered_so_a_new_one_cannot_be_forgotten(self) -> None:
        names = settings_environment_names(keep_real_llm_settings=False)

        assert {field.upper() for field in Settings.model_fields} <= names

    def test_the_scrub_runs_for_every_test(self, request: pytest.FixtureRequest) -> None:
        assert "hermetic_settings_environment" in request.fixturenames

    def test_nothing_the_settings_read_is_visible_to_a_test(self) -> None:
        leaked = settings_environment_names(keep_real_llm_settings=False) & set(os.environ)

        assert leaked == set()

    def test_the_real_llm_opt_in_keeps_only_the_provider_credentials_and_models(self) -> None:
        kept = settings_environment_names(keep_real_llm_settings=False) - (
            settings_environment_names(keep_real_llm_settings=True)
        )

        assert "LLM_API_KEY" in kept and "SYNTHESIS_MODEL" in kept
        assert all(name.startswith("LLM_") or name.endswith("_MODEL") for name in kept)
        assert "EMBEDDING_MODEL" not in kept
        assert "INTERNAL_API_SECRET" not in kept

    def test_settings_without_a_file_see_no_internal_api_secret(self) -> None:
        settings = Settings(_env_file=None, discord_token="fake-token")  # type: ignore[call-arg]

        assert settings.internal_api_secret is None

    def test_a_value_a_test_sets_itself_is_seen(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("INTERNAL_API_SECRET", "s" * 32)

        settings = Settings(_env_file=None, discord_token="fake-token")  # type: ignore[call-arg]

        assert settings.internal_api_secret is not None


class TestNoDotenvFileIsReadUnlessATestAsks:
    """The second route (V-04): Settings reads ./.env itself, and production code calls it."""

    def test_the_settings_classes_read_no_file_by_default(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".env").write_text("DISCORD_TOKEN=from-a-stray-file\n", encoding="utf-8")
        (tmp_path / "web").mkdir()
        (tmp_path / "web" / ".env").write_text(
            "AURA_WEB_STRIPE_SECRET_KEY=rk_test_strayFileValue\n", encoding="utf-8"
        )

        with pytest.raises(ConfigurationError):
            load_settings()
        assert Settings.model_config.get("env_file") is None
        assert WebSettings.model_config.get("env_file") is None

    @pytest.mark.reads_dotenv_file
    def test_a_marked_test_keeps_the_real_file_reading(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".env").write_text("DISCORD_TOKEN=from-the-file\n", encoding="utf-8")

        assert load_settings().discord_token.get_secret_value() == "from-the-file"


class TestTheBotImage:
    def test_litellm_runs_in_production_mode_there_too(self) -> None:
        """No implicit .env loader in the container; its configuration is compose's env_file."""
        dockerfile = (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")

        assert "LITELLM_MODE=PRODUCTION" in dockerfile
        assert ".env" not in [
            line.split()[1] for line in dockerfile.splitlines() if line.startswith("COPY ")
        ]
