"""Tests for the cost bounds on the two calls behind every answer: synthesis and grounding.

Both functions are shared by /aura-ask and proactive relief, so every bound
here applies to both triggers. litellm is always mocked; what is under test is
what Aura sends to it and what Aura does with a response cut off at the limit.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from litellm.types.utils import Choices, Message, ModelResponse, Usage

from aura.config import Settings
from aura.db.models import Fact, FactStatus
from aura.grounding import (
    PROACTIVE_GROUNDING_TIMEOUT_SECONDS,
    GroundingOutcome,
    verify_answer_grounded,
)
from aura.llm_usage import log_llm_usage, was_cut_off
from aura.synthesis import MAX_PROMPT_FACT_CHARS, _build_messages, synthesize_answer

GUILD = 100000000000000001
MODEL = "openrouter/fake/model"
QUESTION = "When does the private event start?"
FACT_TEXT = "The private event starts on Saturday at 18:00."

_GROUNDED = {
    "has_unsupported_claim": False,
    "unsupported_claim": "",
    "has_contradicted_claim": False,
    "contradicted_claim": "",
    "has_invented_source": False,
    "invented_source": "",
    "grounded": True,
    "reasoning": "every claim maps to a cited fact",
}


def _fact(content: str = FACT_TEXT, fact_id: int = 1) -> Fact:
    return Fact(
        id=fact_id,
        guild_id=GUILD,
        channel_id=11,
        message_id=101,
        content=content,
        embedding=b"",
        status=FactStatus.ACTIVE,
        created_at=datetime(2026, 10, 1, 12, 0, tzinfo=UTC),
    )


def _settings(**overrides: object) -> Settings:
    defaults: dict[str, object] = {
        "discord_token": "fake-token",
        "llm_api_key": "fake-key",
        "synthesis_model": MODEL,
        "grounding_check_model": "openrouter/fake/checker",
    }
    defaults.update(overrides)
    return Settings(_env_file=None, **defaults)  # type: ignore[arg-type]


def _response(
    content: str | None,
    *,
    finish_reason: str = "stop",
    usage: Usage | None = None,
) -> ModelResponse:
    response = ModelResponse(
        choices=[
            Choices(
                finish_reason=finish_reason,
                index=0,
                message=Message(content=content, role="assistant"),
            )
        ]
    )
    if usage is not None:
        response.usage = usage  # type: ignore[attr-defined]
    return response


def _synthesis_body() -> str:
    return json.dumps(
        {"answer": "Saturday at 18:00.", "used_fact_numbers": [1], "answers_question": True}
    )


@pytest.fixture
def settings(monkeypatch: pytest.MonkeyPatch) -> Settings:
    configured = _settings()
    monkeypatch.setattr("aura.synthesis.load_settings", lambda: configured)
    return configured


async def _synthesize(response: ModelResponse) -> tuple[object, AsyncMock]:
    completion = AsyncMock(return_value=response)
    with patch("aura.synthesis.litellm.acompletion", completion):
        result = await synthesize_answer([_fact()], QUESTION, "en-US", model=MODEL)
    return result, completion


async def _ground(
    response: ModelResponse, settings: Settings
) -> tuple[GroundingOutcome, AsyncMock]:
    completion = AsyncMock(return_value=response)
    with patch("litellm.acompletion", completion):
        outcome = await verify_answer_grounded(
            answer="Saturday at 18:00.",
            cited_facts=[_fact()],
            settings=settings,
            timeout_seconds=PROACTIVE_GROUNDING_TIMEOUT_SECONDS,
        )
    return outcome, completion


class TestSynthesisOutputCeiling:
    async def test_the_default_ceiling_reaches_the_provider(self, settings: Settings) -> None:
        result, completion = await _synthesize(_response(_synthesis_body()))
        assert result is not None
        assert completion.call_args.kwargs["max_tokens"] == 700

    async def test_a_configured_ceiling_reaches_the_provider(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        configured = _settings(ask_synthesis_max_output_tokens=300)
        monkeypatch.setattr("aura.synthesis.load_settings", lambda: configured)
        _, completion = await _synthesize(_response(_synthesis_body()))
        assert completion.call_args.kwargs["max_tokens"] == 300

    async def test_the_other_call_parameters_are_unchanged(self, settings: Settings) -> None:
        _, completion = await _synthesize(_response(_synthesis_body()))
        kwargs = completion.call_args.kwargs
        assert set(kwargs) == {
            "model",
            "api_key",
            "messages",
            "response_format",
            "timeout",
            "temperature",
            "max_tokens",
        }
        assert kwargs["temperature"] == 0.0
        assert kwargs["response_format"] == {"type": "json_object"}

    @pytest.mark.parametrize("reason", ["length", "max_tokens"])
    async def test_a_cut_off_answer_is_not_used_even_if_it_parses(
        self, settings: Settings, reason: str
    ) -> None:
        result, _ = await _synthesize(_response(_synthesis_body(), finish_reason=reason))
        assert result is None

    async def test_a_cut_off_body_takes_the_malformed_path(
        self, settings: Settings, caplog: pytest.LogCaptureFixture
    ) -> None:
        truncated = _synthesis_body()[:25]
        with caplog.at_level(logging.ERROR, logger="aura.synthesis"):
            result, _ = await _synthesize(_response(truncated, finish_reason="length"))
        assert result is None
        assert any("malformed" in record.getMessage() for record in caplog.records)

    async def test_a_fenced_answer_cut_before_its_closing_fence_is_not_used(
        self, settings: Settings
    ) -> None:
        body = "```json\n" + _synthesis_body() + "\n"
        result, _ = await _synthesize(_response(body, finish_reason="length"))
        assert result is None


class TestGroundingOutputCeiling:
    async def test_the_default_ceiling_reaches_the_provider(self) -> None:
        outcome, completion = await _ground(_response(json.dumps(_GROUNDED)), _settings())
        assert outcome is GroundingOutcome.GROUNDED
        assert completion.call_args.kwargs["max_tokens"] == 300

    async def test_a_configured_ceiling_reaches_the_provider(self) -> None:
        _, completion = await _ground(
            _response(json.dumps(_GROUNDED)), _settings(grounding_max_output_tokens=500)
        )
        assert completion.call_args.kwargs["max_tokens"] == 500

    @pytest.mark.parametrize("reason", ["length", "max_tokens"])
    async def test_a_cut_off_verdict_fails_closed_even_if_it_parses(self, reason: str) -> None:
        outcome, _ = await _ground(
            _response(json.dumps(_GROUNDED), finish_reason=reason), _settings()
        )
        assert outcome is GroundingOutcome.CHECK_FAILED

    async def test_a_cut_off_body_fails_closed(self) -> None:
        outcome, _ = await _ground(
            _response(json.dumps(_GROUNDED)[:40], finish_reason="length"), _settings()
        )
        assert outcome is GroundingOutcome.CHECK_FAILED


class TestUsageLine:
    async def test_synthesis_logs_real_token_counts_and_no_content(
        self, settings: Settings, caplog: pytest.LogCaptureFixture
    ) -> None:
        usage = Usage(prompt_tokens=1156, completion_tokens=98, total_tokens=1254)
        with caplog.at_level(logging.INFO, logger="aura.llm_usage"):
            await _synthesize(_response(_synthesis_body(), usage=usage))

        lines = [r.getMessage() for r in caplog.records if r.name == "aura.llm_usage"]
        assert lines == [
            f"LLM usage: purpose=synthesis model={MODEL} prompt_tokens=1156 "
            "completion_tokens=98 finish_reason=stop"
        ]
        assert caplog.records[-1].levelno == logging.INFO

    async def test_grounding_logs_real_token_counts(self, caplog: pytest.LogCaptureFixture) -> None:
        usage = Usage(prompt_tokens=1490, completion_tokens=73, total_tokens=1563)
        with caplog.at_level(logging.INFO, logger="aura.llm_usage"):
            await _ground(_response(json.dumps(_GROUNDED), usage=usage), _settings())

        lines = [r.getMessage() for r in caplog.records if r.name == "aura.llm_usage"]
        assert lines == [
            "LLM usage: purpose=grounding model=openrouter/fake/checker prompt_tokens=1490 "
            "completion_tokens=73 finish_reason=stop"
        ]

    async def test_a_cut_off_call_is_still_measured(
        self, settings: Settings, caplog: pytest.LogCaptureFixture
    ) -> None:
        usage = Usage(prompt_tokens=1200, completion_tokens=700, total_tokens=1900)
        with caplog.at_level(logging.INFO, logger="aura.llm_usage"):
            await _synthesize(_response("{", finish_reason="length", usage=usage))
        assert any(
            "completion_tokens=700 finish_reason=length" in r.getMessage() for r in caplog.records
        )

    async def test_no_line_carries_the_question_the_fact_or_the_answer(
        self, settings: Settings, caplog: pytest.LogCaptureFixture
    ) -> None:
        usage = Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2)
        with caplog.at_level(logging.DEBUG, logger="aura.llm_usage"):
            await _synthesize(_response(_synthesis_body(), usage=usage))
            await _ground(_response(json.dumps(_GROUNDED), usage=usage), _settings())
        text = "\n".join(r.getMessage() for r in caplog.records if r.name == "aura.llm_usage")
        for secret in ("private", "Saturday", "18:00", str(GUILD)[:6]):
            assert secret not in text

    def test_missing_usage_is_logged_as_missing(self, caplog: pytest.LogCaptureFixture) -> None:
        response = _response("{}")  # litellm sets no usage unless the provider sent one
        assert getattr(response, "usage", None) is None
        with caplog.at_level(logging.INFO, logger="aura.llm_usage"):
            log_llm_usage(response, purpose="synthesis", model=MODEL)
        assert caplog.records[-1].getMessage() == (
            f"LLM usage: purpose=synthesis model={MODEL} usage=missing finish_reason=stop"
        )

    @pytest.mark.parametrize(
        "usage",
        [
            # litellm's own Usage coerces these; a provider object passed through
            # unconverted would not, so the raw shapes are what is tested here.
            SimpleNamespace(prompt_tokens=None, completion_tokens=5),
            SimpleNamespace(prompt_tokens=True, completion_tokens=5),
            SimpleNamespace(prompt_tokens="12", completion_tokens=5),
            SimpleNamespace(completion_tokens=5),
        ],
    )
    def test_a_usage_without_integer_counts_is_logged_as_missing(
        self, usage: SimpleNamespace, caplog: pytest.LogCaptureFixture
    ) -> None:
        response = _response("{}")
        response.usage = usage  # type: ignore[attr-defined]
        with caplog.at_level(logging.INFO, logger="aura.llm_usage"):
            log_llm_usage(response, purpose="grounding", model=MODEL)
        assert "usage=missing" in caplog.records[-1].getMessage()

    def test_an_absurd_finish_reason_is_bounded(self, caplog: pytest.LogCaptureFixture) -> None:
        response = _response("{}")
        # Set after construction: litellm maps an unknown reason to "stop" on the way in.
        response.choices[0].finish_reason = "x" * 500
        with caplog.at_level(logging.INFO, logger="aura.llm_usage"):
            log_llm_usage(response, purpose="synthesis", model=MODEL)
        assert caplog.records[-1].getMessage().endswith("finish_reason=" + "x" * 32)

    def test_no_choices_is_neither_cut_off_nor_an_error(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        response = ModelResponse(choices=[])
        assert was_cut_off(response) is False
        with caplog.at_level(logging.INFO, logger="aura.llm_usage"):
            log_llm_usage(response, purpose="synthesis", model=MODEL)
        assert caplog.records[-1].getMessage().endswith("finish_reason=none")


def _user_prompt(facts: list[Fact], *, with_channels: bool) -> str:
    messages = _build_messages(
        facts,
        QUESTION,
        "en-US",
        question_channel_name="general" if with_channels else None,
        question_asked_at=datetime(2026, 10, 2, 9, 0, tzinfo=UTC) if with_channels else None,
        fact_channel_names={11: "events"} if with_channels else None,
    )
    return messages[1]["content"]


class TestFactTextInTheSynthesisPrompt:
    @pytest.mark.parametrize("with_channels", [False, True])
    @pytest.mark.parametrize(
        ("content", "expected"),
        [
            ("x" * 999, "x" * 999),
            ("x" * 1000, "x" * 1000),
            ("x" * 1001, "x" * 1000),
            ("x" * 4000, "x" * 1000),
            ("日" * 1001, "日" * 1000),
            ("ä" * 999 + "🎉🎉", "ä" * 999 + "🎉"),
        ],
        ids=["999", "1000", "1001", "4000", "cjk-1001", "multibyte-boundary"],
    )
    def test_each_fact_is_cut_to_the_bound(
        self, content: str, expected: str, with_channels: bool
    ) -> None:
        prompt = _user_prompt([_fact(content)], with_channels=with_channels)
        fact_line = prompt.split("\nFacts:\n", 1)[1]
        assert fact_line.endswith(" " + expected)
        assert content[len(expected) :] == "" or content not in prompt

    def test_the_bound_matches_the_grounding_checks(self) -> None:
        from aura.grounding import _MAX_FACT_CHARS

        assert MAX_PROMPT_FACT_CHARS == _MAX_FACT_CHARS == 1000

    def test_every_fact_is_cut_independently(self) -> None:
        facts = [_fact("a" * 1500, fact_id=1), _fact("short fact", fact_id=2)]
        prompt = _user_prompt(facts, with_channels=False)
        assert f"[1] {'a' * 1000}\n[2] short fact" in prompt

    def test_the_question_is_not_cut_here(self) -> None:
        # /aura-ask cuts the question itself (aura.commands.ask); proactive
        # relief's message reaches synthesis unchanged, as before.
        question = "q" * 3000
        messages = _build_messages(
            [_fact()],
            question,
            "en-US",
            question_channel_name=None,
            question_asked_at=None,
            fact_channel_names=None,
        )
        assert f"<<<MESSAGE\n{question}\nMESSAGE" in messages[1]["content"]
