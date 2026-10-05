"""The background calls' request parameters (P5): output ceilings, routes, usage lines, cut-offs.

Fact extraction, the supersession judge and the two variant calls carried no
`max_tokens` until P5, and none of the three functions could be sent a provider
route. These tests pin what each call now sends -- the ceiling from its own
setting, a route only when one is configured, nothing extra otherwise -- and
that a reply stopped at the ceiling takes the existing failure path even when
what arrived happens to parse. Proactive relief's route is covered here too,
in both answer formats, together with the rule that `/aura-ask` never gets it.

Hermetic: litellm.acompletion is mocked in every test.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from litellm.types.utils import ModelResponse
from pydantic import ValidationError

from aura.config import Settings
from aura.db.extraction_queue import QueuedMessage
from aura.extraction.distiller import distill_facts
from aura.extraction.supersession import judge_relationship
from aura.variants_service import _audit_variants, _generate_variants

OPENROUTER_MODEL = "openrouter/deepseek/deepseek-v4.1-flash"
DIRECT_MODEL = "anthropic/claude-haiku-4.5"
NOW = datetime(2026, 10, 4, 12, 0, tzinfo=UTC)


def _settings(**overrides: object) -> Settings:
    values: dict[str, object] = {"discord_token": "fake-token", "llm_api_key": "fake-key"}
    values.update(overrides)
    return Settings(_env_file=None, **values)  # type: ignore[arg-type]


def _response(payload: object, *, finish_reason: str = "stop") -> MagicMock:
    response = MagicMock(spec=ModelResponse)
    choice = MagicMock()
    choice.message.content = json.dumps(payload)
    choice.finish_reason = finish_reason
    response.choices = [choice]
    response.usage = MagicMock(prompt_tokens=1500, completion_tokens=90)
    return response


def _queued(message_id: int, content: str) -> QueuedMessage:
    return QueuedMessage(
        channel_id=500,
        message_id=message_id,
        guild_id=100,
        channel_name="ankündigungen",
        content=content,
        message_created_at=NOW,
        enqueued_at=NOW,
    )


_FACTS_REPLY: dict[str, Any] = {
    "facts": [
        {
            "message": 1,
            "language": "German",
            "content": "Der Spieleabend ist am 7. Oktober 2026 um 20 Uhr.",
            "category": "event",
        }
    ]
}

_JUDGEMENT_REPLY: dict[str, str] = {
    "change_signal": "wurde verlegt",
    "shared_subject": "Spieleabend",
    "category": "supersession",
    "language": "German",
    "reasoning": "Die Uhrzeit des Spieleabends wurde verlegt.",
}


@pytest.fixture(autouse=True)
def _environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_API_KEY", "fake-key")
    monkeypatch.setenv("DISCORD_TOKEN", "fake-token")


class TestTheNewSettings:
    def test_defaults_bound_every_call_and_send_no_route(self) -> None:
        settings = _settings()

        assert settings.extraction_max_output_tokens == 4096
        assert settings.supersession_max_output_tokens == 1024
        assert settings.variant_max_output_tokens == 1024
        assert settings.variant_audit_max_output_tokens == 1024
        assert settings.extraction_verify_max_output_tokens == 2048
        for name in (
            "extraction_providers",
            "extraction_reasoning",
            "supersession_providers",
            "supersession_reasoning",
            "proactive_providers",
            "proactive_reasoning",
            "extraction_verify_providers",
            "extraction_verify_reasoning",
        ):
            assert getattr(settings, name) == ""
        assert settings.extraction_deny_data_collection is False
        assert settings.supersession_deny_data_collection is False
        assert settings.proactive_deny_data_collection is False
        assert settings.extraction_verify_model is None
        assert settings.proactive_max_output_tokens is None

    def test_the_proactive_ceiling_is_bounded_when_set(self) -> None:
        assert _settings(proactive_max_output_tokens=256).proactive_max_output_tokens == 256
        assert _settings(proactive_max_output_tokens=16384).proactive_max_output_tokens == 16384
        for value in (255, 16385):
            with pytest.raises(ValidationError):
                _settings(proactive_max_output_tokens=value)

    @pytest.mark.parametrize(
        ("field", "lowest", "highest"),
        [
            ("extraction_max_output_tokens", 1024, 16384),
            ("supersession_max_output_tokens", 256, 8192),
            ("variant_max_output_tokens", 256, 8192),
            ("variant_audit_max_output_tokens", 256, 8192),
            ("extraction_verify_max_output_tokens", 512, 8192),
        ],
    )
    def test_ceilings_are_bounded_exactly_at_their_limits(
        self, field: str, lowest: int, highest: int
    ) -> None:
        assert getattr(_settings(**{field: lowest}), field) == lowest
        assert getattr(_settings(**{field: highest}), field) == highest
        with pytest.raises(ValidationError):
            _settings(**{field: lowest - 1})
        with pytest.raises(ValidationError):
            _settings(**{field: highest + 1})

    @pytest.mark.parametrize(
        "field",
        [
            "extraction_reasoning",
            "supersession_reasoning",
            "proactive_reasoning",
            "extraction_verify_reasoning",
        ],
    )
    def test_an_unknown_reasoning_level_is_refused(self, field: str) -> None:
        with pytest.raises(ValidationError):
            _settings(**{field: "maximum"})

    def test_the_settings_are_read_from_the_environment(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("EXTRACTION_MAX_OUTPUT_TOKENS", "3000")
        monkeypatch.setenv("PROACTIVE_PROVIDERS", "Google")
        monkeypatch.setenv("PROACTIVE_REASONING", "low")
        monkeypatch.setenv("SUPERSESSION_DENY_DATA_COLLECTION", "true")

        settings = Settings(_env_file=None)  # type: ignore[call-arg]

        assert settings.extraction_max_output_tokens == 3000
        assert settings.proactive_providers == "Google"
        assert settings.proactive_reasoning == "low"
        assert settings.supersession_deny_data_collection is True


class TestFactExtractionCall:
    async def test_it_sends_the_ceiling_and_no_route_by_default(self) -> None:
        mock = AsyncMock(return_value=_response(_FACTS_REPLY))
        with patch("aura.extraction.distiller.litellm.acompletion", mock):
            facts = await distill_facts(
                [_queued(1, "Morgen um 20 Uhr ist Spieleabend.")],
                channel_name="events",
                model=OPENROUTER_MODEL,
            )

        assert facts is not None and len(facts) == 1
        kwargs = mock.call_args.kwargs
        assert kwargs["max_tokens"] == 4096
        assert "extra_body" not in kwargs

    async def test_it_sends_the_configured_route(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("EXTRACTION_PROVIDERS", "DeepInfra, Together")
        monkeypatch.setenv("EXTRACTION_REASONING", "off")
        monkeypatch.setenv("EXTRACTION_DENY_DATA_COLLECTION", "true")
        monkeypatch.setenv("EXTRACTION_MAX_OUTPUT_TOKENS", "2000")
        mock = AsyncMock(return_value=_response(_FACTS_REPLY))
        with patch("aura.extraction.distiller.litellm.acompletion", mock):
            await distill_facts([_queued(1, "x")], channel_name="events", model=OPENROUTER_MODEL)

        kwargs = mock.call_args.kwargs
        assert kwargs["max_tokens"] == 2000
        assert kwargs["extra_body"] == {
            "provider": {
                "order": ["DeepInfra", "Together"],
                "allow_fallbacks": False,
                "data_collection": "deny",
            },
            "reasoning": {"enabled": False},
        }

    async def test_a_model_outside_openrouter_never_gets_the_route(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("EXTRACTION_PROVIDERS", "DeepInfra")
        mock = AsyncMock(return_value=_response(_FACTS_REPLY))
        with patch("aura.extraction.distiller.litellm.acompletion", mock):
            await distill_facts([_queued(1, "x")], channel_name="events", model=DIRECT_MODEL)

        assert "extra_body" not in mock.call_args.kwargs

    @pytest.mark.parametrize("finish_reason", ["length", "max_tokens"])
    async def test_a_cut_off_reply_stages_nothing_even_when_it_parses(
        self, finish_reason: str
    ) -> None:
        mock = AsyncMock(return_value=_response(_FACTS_REPLY, finish_reason=finish_reason))
        with patch("aura.extraction.distiller.litellm.acompletion", mock):
            facts = await distill_facts(
                [_queued(1, "x")], channel_name="events", model=OPENROUTER_MODEL
            )

        assert facts is None

    async def test_one_usage_line_without_content(self, caplog: pytest.LogCaptureFixture) -> None:
        mock = AsyncMock(return_value=_response(_FACTS_REPLY))
        with (
            caplog.at_level(logging.INFO, logger="aura.llm_usage"),
            patch("aura.extraction.distiller.litellm.acompletion", mock),
        ):
            await distill_facts(
                [_queued(1, "Geheimer Inhalt der Nachricht")],
                channel_name="events",
                model=OPENROUTER_MODEL,
            )

        lines = [r.getMessage() for r in caplog.records if r.name == "aura.llm_usage"]
        assert lines == [
            f"LLM usage: purpose=extraction model={OPENROUTER_MODEL} prompt_tokens=1500 "
            "completion_tokens=90 finish_reason=stop"
        ]
        assert "Geheimer" not in caplog.text
        assert "Spieleabend" not in caplog.text


class TestSupersessionCall:
    async def test_it_sends_the_ceiling_and_no_route_by_default(self) -> None:
        mock = AsyncMock(return_value=_response(_JUDGEMENT_REPLY))
        with patch("aura.extraction.supersession.litellm.acompletion", mock):
            judgement = await judge_relationship(
                predecessor="a", candidate="b", model=OPENROUTER_MODEL
            )

        assert judgement is not None
        assert mock.call_args.kwargs["max_tokens"] == 1024
        assert "extra_body" not in mock.call_args.kwargs

    async def test_it_sends_the_configured_route(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SUPERSESSION_PROVIDERS", "Google")
        monkeypatch.setenv("SUPERSESSION_REASONING", "low")
        mock = AsyncMock(return_value=_response(_JUDGEMENT_REPLY))
        with patch("aura.extraction.supersession.litellm.acompletion", mock):
            await judge_relationship(predecessor="a", candidate="b", model=OPENROUTER_MODEL)

        assert mock.call_args.kwargs["extra_body"] == {
            "provider": {"order": ["Google"], "allow_fallbacks": False},
            "reasoning": {"effort": "low"},
        }

    @pytest.mark.parametrize("finish_reason", ["length", "max_tokens"])
    async def test_a_cut_off_judgement_is_not_judged(self, finish_reason: str) -> None:
        mock = AsyncMock(return_value=_response(_JUDGEMENT_REPLY, finish_reason=finish_reason))
        with patch("aura.extraction.supersession.litellm.acompletion", mock):
            judgement = await judge_relationship(
                predecessor="a", candidate="b", model=OPENROUTER_MODEL
            )

        assert judgement is None

    async def test_one_usage_line(self, caplog: pytest.LogCaptureFixture) -> None:
        mock = AsyncMock(return_value=_response(_JUDGEMENT_REPLY))
        with (
            caplog.at_level(logging.INFO, logger="aura.llm_usage"),
            patch("aura.extraction.supersession.litellm.acompletion", mock),
        ):
            await judge_relationship(predecessor="a", candidate="b", model=OPENROUTER_MODEL)

        assert [r.getMessage() for r in caplog.records if r.name == "aura.llm_usage"] == [
            f"LLM usage: purpose=supersession model={OPENROUTER_MODEL} prompt_tokens=1500 "
            "completion_tokens=90 finish_reason=stop"
        ]


class TestVariantCalls:
    async def test_generation_sends_its_ceiling(self) -> None:
        mock = AsyncMock(return_value=_response({"variants": ["Eine andere Fassung."]}))
        with patch("aura.variants_service.litellm.acompletion", mock):
            variants = await _generate_variants("Ein Fakt.", count=1, model=OPENROUTER_MODEL)

        assert variants == ["Eine andere Fassung."]
        assert mock.call_args.kwargs["max_tokens"] == 1024

    async def test_a_cut_off_generation_stores_nothing(self) -> None:
        mock = AsyncMock(
            return_value=_response({"variants": ["Eine Fassung."]}, finish_reason="length")
        )
        with patch("aura.variants_service.litellm.acompletion", mock):
            variants = await _generate_variants("Ein Fakt.", count=1, model=OPENROUTER_MODEL)

        assert variants is None

    async def test_the_audit_sends_its_ceiling_and_a_cut_off_audit_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("VARIANT_AUDIT_MAX_OUTPUT_TOKENS", "512")
        verdicts = {"verdicts": [{"index": 1, "faithful": True, "reasoning": "Gleich."}]}
        ok = AsyncMock(return_value=_response(verdicts))
        with patch("aura.variants_service.litellm.acompletion", ok):
            passed = await _audit_variants(
                canonical="Ein Fakt.", variants=["Ein Fakt, anders."], model=OPENROUTER_MODEL
            )
        cut = AsyncMock(return_value=_response(verdicts, finish_reason="length"))
        with patch("aura.variants_service.litellm.acompletion", cut):
            failed = await _audit_variants(
                canonical="Ein Fakt.", variants=["Ein Fakt, anders."], model=OPENROUTER_MODEL
            )

        assert passed is not None
        assert ok.call_args.kwargs["max_tokens"] == 512
        assert failed is None
