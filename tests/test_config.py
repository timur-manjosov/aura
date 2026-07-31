"""Tests for aura.config: settings loading and validation.

Every test clears the relevant environment variables first (clean_env,
below) so behavior is deterministic regardless of what's actually exported
in the shell running the tests, or whether a real .env happens to exist in
the current working directory.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from aura.config import ConfigurationError, ModelComponent, Settings, load_settings

_ENV_KEYS = (
    "DISCORD_TOKEN",
    "LLM_PROVIDER",
    "LLM_API_KEY",
    "SYNTHESIS_MODEL",
    "PROACTIVE_MODEL",
    "SIMILARITY_THRESHOLD",
    "PROACTIVE_QUESTION_THRESHOLD",
    "PROACTIVE_SIMILARITY_THRESHOLD",
    "PROACTIVE_CONFIDENCE_GAP",
    "PROACTIVE_COOLDOWN_SECONDS",
    "PROACTIVE_DAILY_CAP",
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
        assert settings.llm_provider is None
        assert settings.llm_api_key is None
        assert settings.synthesis_model is None
        assert settings.similarity_threshold == 0.4
        assert settings.database_path == "data/aura.db"
        assert settings.embedding_model == "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        assert settings.log_level == "INFO"

    def test_proactive_defaults(self) -> None:
        # Phase 2b-3 recalibration -- see config.py for the per-field
        # evidence from the phase-2b-2 synthetic-corpus simulation.
        settings = _settings(discord_token="valid-token")
        assert settings.proactive_question_threshold == -0.15
        assert settings.proactive_similarity_threshold == 0.30
        assert settings.proactive_confidence_gap == 0.05
        assert settings.proactive_cooldown_seconds == 900.0
        assert settings.proactive_daily_cap == 60

    def test_extraction_default(self) -> None:
        # Phase 3a-1b calibrated -0.02 (supersedes 3a-1's -0.03) -- see
        # config.py for the full sweep evidence. Currently overridden to
        # -0.04 for the single-member testing phase (same sweep, a
        # different, more recall-friendly row); see config.py's
        # extraction_fact_worthiness_threshold comment and
        # reports/testing-threshold-note.txt. Revert this assertion to
        # -0.02 alongside config.py and .env.example when that override is
        # reverted.
        settings = _settings(discord_token="valid-token")
        assert settings.extraction_fact_worthiness_threshold == -0.04

    def test_extraction_pipeline_defaults(self) -> None:
        # Phase 3a-2's knobs. Pinned here so a change to any of them is a
        # decision someone made rather than a drift nobody noticed -- these
        # bound real spending.
        settings = _settings(discord_token="valid-token")
        assert settings.extraction_batch_window_seconds == 300.0
        assert settings.extraction_batch_max_messages == 20
        assert settings.extraction_daily_cap == 50
        # Extraction-dedup-threshold-calibration report: 0.60, down from 0.70,
        # see config.py for the full sweep evidence.
        assert settings.extraction_dedup_similarity_threshold == 0.60
        # Unset by default, like every other model: a hardcoded model with no
        # key would look configured in code while failing on every call.
        assert settings.extraction_model is None

    def test_supersession_defaults(self) -> None:
        # Phase 3a-3's own budget, deliberately independent of extraction's so
        # that spending every judgement cannot consume the extraction calls that
        # produce candidates in the first place.
        settings = _settings(discord_token="valid-token")
        assert settings.supersession_daily_cap == 50
        assert settings.supersession_model is None

    def test_the_proactive_gate_bar_may_now_be_looser_than_the_direct_query_bar(self) -> None:
        # Phase 2b-3 deliberately inverted the ordering this test asserted
        # through Phase 2b-2: proactive_similarity_threshold (0.30) is now
        # BELOW similarity_threshold (0.4). That is safe, not a regression,
        # because the two thresholds serve different purposes and always
        # have: proactive_similarity_threshold is only the gate's "is this
        # worth attempting" bar (aura.proactive.gate), while the content
        # actually handed to the LLM is independently re-filtered by
        # similarity_threshold in aura.proactive.responder.respond_with_
        # synthesis, regardless of how loose the gate is -- see
        # test_proactive_responder.py::test_a_fact_between_the_gate_bar_and_
        # the_direct_query_bar_still_never_synthesizes for the proof. This
        # test just pins the new, intentional ordering so a future change
        # doesn't silently drift back without a decision behind it.
        settings = _settings(discord_token="valid-token")
        assert settings.proactive_similarity_threshold < settings.similarity_threshold

    def test_overrides_are_respected(self) -> None:
        settings = _settings(
            discord_token="valid-token",
            llm_provider="openai",
            log_level="DEBUG",
        )
        assert settings.llm_provider == "openai"
        assert settings.log_level == "DEBUG"


class TestProactiveSettingsValidation:
    """A typo in .env must abort startup, not silently disable a safety net."""

    @pytest.mark.parametrize(
        ("key", "value"),
        [
            # -2.0 is the score question_likeness returns for text it cannot
            # score at all, so a threshold there would admit empty messages.
            ("PROACTIVE_QUESTION_THRESHOLD", "-2.0"),
            ("PROACTIVE_QUESTION_THRESHOLD", "-3"),
            ("PROACTIVE_QUESTION_THRESHOLD", "2.5"),
            ("PROACTIVE_QUESTION_THRESHOLD", "not-a-number"),
            # Outside the range a cosine similarity can occupy, so it would
            # either never match or always match.
            ("PROACTIVE_SIMILARITY_THRESHOLD", "1.5"),
            ("PROACTIVE_SIMILARITY_THRESHOLD", "-1.5"),
            ("PROACTIVE_CONFIDENCE_GAP", "-0.1"),
            ("PROACTIVE_CONFIDENCE_GAP", "3"),
            ("PROACTIVE_COOLDOWN_SECONDS", "-1"),
            # A negative cap is the dangerous one: read as "no cap" by a naive
            # comparison, it would remove the daily limit entirely.
            ("PROACTIVE_DAILY_CAP", "-1"),
            ("PROACTIVE_DAILY_CAP", "1.5"),
            ("EXTRACTION_FACT_WORTHINESS_THRESHOLD", "-2.0"),
            ("EXTRACTION_FACT_WORTHINESS_THRESHOLD", "2.5"),
            ("EXTRACTION_FACT_WORTHINESS_THRESHOLD", "not-a-number"),
        ],
    )
    def test_an_out_of_range_value_is_refused_with_a_readable_error(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, key: str, value: str
    ) -> None:
        monkeypatch.chdir(tmp_path)  # no .env file here
        monkeypatch.setenv("DISCORD_TOKEN", "valid-token")
        monkeypatch.setenv(key, value)

        with pytest.raises(ConfigurationError):
            load_settings()

    @pytest.mark.parametrize(
        ("key", "value", "attribute", "expected"),
        [
            ("PROACTIVE_QUESTION_THRESHOLD", "0.05", "proactive_question_threshold", 0.05),
            ("PROACTIVE_SIMILARITY_THRESHOLD", "0.5", "proactive_similarity_threshold", 0.5),
            ("PROACTIVE_CONFIDENCE_GAP", "0.0", "proactive_confidence_gap", 0.0),
            ("PROACTIVE_COOLDOWN_SECONDS", "0", "proactive_cooldown_seconds", 0.0),
            # Zero is a deliberate off switch for proactive relief, not a
            # misconfiguration, so it has to be accepted.
            ("PROACTIVE_DAILY_CAP", "0", "proactive_daily_cap", 0),
            ("PROACTIVE_DAILY_CAP", "500", "proactive_daily_cap", 500),
            (
                "EXTRACTION_FACT_WORTHINESS_THRESHOLD",
                "0.1",
                "extraction_fact_worthiness_threshold",
                0.1,
            ),
        ],
    )
    def test_a_valid_override_is_read_from_the_environment(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        key: str,
        value: str,
        attribute: str,
        expected: float,
    ) -> None:
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("DISCORD_TOKEN", "valid-token")
        monkeypatch.setenv(key, value)

        assert getattr(load_settings(), attribute) == expected


class TestIsLlmConfigured:
    def test_false_when_neither_key_nor_model_set(self) -> None:
        settings = _settings(discord_token="valid-token")
        assert settings.is_llm_configured(ModelComponent.SYNTHESIS) is False

    def test_false_when_only_api_key_set(self) -> None:
        settings = _settings(discord_token="valid-token", llm_api_key="sk-fake")
        assert settings.is_llm_configured(ModelComponent.SYNTHESIS) is False

    def test_false_when_only_synthesis_model_set(self) -> None:
        settings = _settings(discord_token="valid-token", synthesis_model="openrouter/foo/bar")
        assert settings.is_llm_configured(ModelComponent.SYNTHESIS) is False

    def test_true_when_both_are_set(self) -> None:
        settings = _settings(
            discord_token="valid-token", llm_api_key="sk-fake", synthesis_model="openrouter/foo/bar"
        )
        assert settings.is_llm_configured(ModelComponent.SYNTHESIS) is True

    def test_false_when_api_key_is_blank_string(self) -> None:
        settings = _settings(
            discord_token="valid-token", llm_api_key="", synthesis_model="openrouter/foo/bar"
        )
        assert settings.is_llm_configured(ModelComponent.SYNTHESIS) is False

    def test_proactive_is_configured_off_synthesis_model_alone(self) -> None:
        # PROACTIVE_MODEL falls back to the synthesis model, so a deployment
        # that set only SYNTHESIS_MODEL has a configured proactive trigger too.
        settings = _settings(
            discord_token="valid-token", llm_api_key="sk-fake", synthesis_model="openrouter/foo/bar"
        )
        assert settings.is_llm_configured(ModelComponent.PROACTIVE) is True

    def test_proactive_needs_the_api_key_like_synthesis_does(self) -> None:
        settings = _settings(
            discord_token="valid-token", proactive_model="openrouter/foo/bar"
        )
        assert settings.is_llm_configured(ModelComponent.PROACTIVE) is False


class TestResolveModel:
    """The one seam every LLM-calling component resolves its model through."""

    def test_synthesis_resolves_to_the_synthesis_model(self) -> None:
        settings = _settings(discord_token="valid-token", synthesis_model="a/b")
        assert settings.resolve_model(ModelComponent.SYNTHESIS) == "a/b"

    def test_proactive_uses_its_own_model_when_set(self) -> None:
        settings = _settings(
            discord_token="valid-token", synthesis_model="a/b", proactive_model="c/d"
        )
        assert settings.resolve_model(ModelComponent.PROACTIVE) == "c/d"

    def test_proactive_falls_back_to_the_synthesis_model_when_unset(self) -> None:
        # "Not assumed to differ from SYNTHESIS_MODEL by default" (CLAUDE.md):
        # a single configured model powers both triggers.
        settings = _settings(discord_token="valid-token", synthesis_model="a/b")
        assert settings.proactive_model is None
        assert settings.resolve_model(ModelComponent.PROACTIVE) == "a/b"

    def test_proactive_model_env_override_is_read(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("DISCORD_TOKEN", "valid-token")
        monkeypatch.setenv("PROACTIVE_MODEL", "openrouter/some/model")
        assert load_settings().proactive_model == "openrouter/some/model"

    def test_extraction_uses_its_own_model_when_set(self) -> None:
        settings = _settings(
            discord_token="valid-token", synthesis_model="a/b", extraction_model="e/f"
        )
        assert settings.resolve_model(ModelComponent.EXTRACTION) == "e/f"

    def test_extraction_falls_back_to_the_synthesis_model_when_unset(self) -> None:
        settings = _settings(discord_token="valid-token", synthesis_model="a/b")
        assert settings.extraction_model is None
        assert settings.resolve_model(ModelComponent.EXTRACTION) == "a/b"

    def test_extraction_model_env_override_is_read(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("DISCORD_TOKEN", "valid-token")
        monkeypatch.setenv("EXTRACTION_MODEL", "openrouter/some/extractor")
        assert load_settings().extraction_model == "openrouter/some/extractor"

    def test_extraction_is_configured_off_synthesis_model_alone(self) -> None:
        settings = _settings(
            discord_token="valid-token",
            llm_api_key="sk-fake",
            synthesis_model="openrouter/foo/bar",
        )
        assert settings.is_llm_configured(ModelComponent.EXTRACTION) is True

    def test_extraction_needs_the_api_key_like_the_others(self) -> None:
        settings = _settings(discord_token="valid-token", extraction_model="openrouter/foo/bar")
        assert settings.is_llm_configured(ModelComponent.EXTRACTION) is False

    def test_supersession_uses_its_own_model_when_set(self) -> None:
        settings = _settings(
            discord_token="valid-token", synthesis_model="a/b", supersession_model="s/j"
        )
        assert settings.resolve_model(ModelComponent.SUPERSESSION) == "s/j"

    def test_supersession_does_not_inherit_the_extraction_model(self) -> None:
        # Two separate call sites with two separate bake-off answers behind
        # them: EXTRACTION_MODEL carries Phase 2's result as an admitted
        # assumption, while SUPERSESSION_MODEL was measured for its own task.
        # Chaining one to the other would re-point a measured choice whenever
        # the assumed one changed.
        settings = _settings(
            discord_token="valid-token", synthesis_model="a/b", extraction_model="e/f"
        )
        assert settings.resolve_model(ModelComponent.SUPERSESSION) == "a/b"

    def test_supersession_falls_back_to_the_synthesis_model_when_unset(self) -> None:
        settings = _settings(discord_token="valid-token", synthesis_model="a/b")
        assert settings.supersession_model is None
        assert settings.resolve_model(ModelComponent.SUPERSESSION) == "a/b"

    def test_supersession_model_env_override_is_read(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("DISCORD_TOKEN", "valid-token")
        monkeypatch.setenv("SUPERSESSION_MODEL", "openrouter/some/judge")
        assert load_settings().supersession_model == "openrouter/some/judge"

    def test_supersession_needs_the_api_key_like_the_others(self) -> None:
        settings = _settings(
            discord_token="valid-token", supersession_model="openrouter/foo/bar"
        )
        assert settings.is_llm_configured(ModelComponent.SUPERSESSION) is False

    def test_every_component_is_resolvable(self) -> None:
        # resolve_model's match statement has no default arm, so a member added
        # to ModelComponent without one silently returns None instead of
        # failing -- and a component whose model resolves to None never runs.
        settings = _settings(discord_token="valid-token", synthesis_model="a/b")
        for component in ModelComponent:
            assert settings.resolve_model(component) == "a/b"

    def test_all_unset_resolves_to_none_for_every_component(self) -> None:
        settings = _settings(discord_token="valid-token")
        for component in ModelComponent:
            assert settings.resolve_model(component) is None


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
