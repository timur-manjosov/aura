"""Tests for aura.llm_request_options and its use by the two v2 calls.

What must hold: no option set means no extra field on the call at all (a call
exactly as before); a model not routed through OpenRouter never gets
OpenRouter's fields; and when the operator configures a route, both v2 calls send
exactly that route.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from litellm.types.utils import ModelResponse

from aura.answer_check import build_statements, verify_answer_v2
from aura.answer_contract import synthesize_contract_answer
from aura.config import Settings
from aura.db.models import Fact, FactStatus
from aura.llm_request_options import openrouter_extra_body, parse_provider_list

FACT = Fact(
    id=1,
    guild_id=1,
    channel_id=1,
    message_id=1,
    content="The cup starts at 18:00.",
    embedding=b"",
    status=FactStatus.ACTIVE,
    created_at=datetime(2026, 8, 1, tzinfo=UTC),
)
CONTRACT = {
    "request_reading": "r",
    "fact_notes": [],
    "relations": [],
    "not_covered_topics": [],
    "tone": "neutral",
    "lead": "The cup starts at 18:00.",
    "points": [],
    "used_fact_numbers": [1],
    "answers_question": True,
}
VERDICT = {"statements": [{"id": "L", "issues": [], "supported": True}]}


def _settings(**overrides: object) -> Settings:
    defaults: dict[str, object] = {
        "discord_token": "t",
        "llm_api_key": "k",
        "synthesis_model": "openrouter/a/synth",
        "grounding_check_model": "openrouter/b/check",
    }
    defaults.update(overrides)
    return Settings(_env_file=None, **defaults)  # type: ignore[arg-type]


def _response(payload: object) -> MagicMock:
    response = MagicMock(spec=ModelResponse)
    choice = MagicMock()
    choice.message.content = json.dumps(payload)
    choice.finish_reason = "stop"
    response.choices = [choice]
    response.usage = MagicMock(prompt_tokens=1, completion_tokens=1)
    return response


class TestTheOptions:
    def test_nothing_set_is_nothing_sent(self) -> None:
        assert (
            openrouter_extra_body(
                "openrouter/x", providers=(), deny_data_collection=False, reasoning=""
            )
            is None
        )

    def test_a_model_outside_openrouter_gets_no_openrouter_fields(self) -> None:
        assert (
            openrouter_extra_body(
                "anthropic/claude", providers=("A",), deny_data_collection=True, reasoning="low"
            )
            is None
        )

    def test_every_option_maps_to_openrouters_field(self) -> None:
        body = openrouter_extra_body(
            "openrouter/x",
            providers=("DeepInfra", "Together"),
            deny_data_collection=True,
            reasoning="off",
        )

        assert body == {
            "provider": {
                "order": ["DeepInfra", "Together"],
                "allow_fallbacks": False,
                "data_collection": "deny",
            },
            "reasoning": {"enabled": False},
        }

    @pytest.mark.parametrize("effort", ["low", "medium", "high"])
    def test_an_effort_level_is_passed_as_such(self, effort: str) -> None:
        body = openrouter_extra_body(
            "openrouter/x", providers=(), deny_data_collection=False, reasoning=effort
        )  # type: ignore[arg-type]

        assert body == {"reasoning": {"effort": effort}}

    @pytest.mark.parametrize(
        ("raw", "expected"), [("", ()), (" , ", ()), ("A", ("A",)), (" A , B,", ("A", "B"))]
    )
    def test_the_provider_list_is_parsed_leniently(
        self, raw: str, expected: tuple[str, ...]
    ) -> None:
        assert parse_provider_list(raw) == expected

    def test_an_unknown_reasoning_level_is_refused_by_the_settings(self) -> None:
        with pytest.raises(ValueError):
            _settings(answer_v2_reasoning="extreme")


class TestTheCalls:
    async def test_without_options_the_contract_call_carries_no_extra_body(self) -> None:
        completion = AsyncMock(return_value=_response(CONTRACT))
        with patch("aura.answer_contract.litellm.acompletion", completion):
            await synthesize_contract_answer(
                [FACT], "q", "en-US", model="openrouter/a/synth", settings=_settings()
            )

        assert completion.await_args is not None
        assert "extra_body" not in completion.await_args.kwargs

    async def test_the_contract_call_sends_the_configured_route(self) -> None:
        completion = AsyncMock(return_value=_response(CONTRACT))
        settings = _settings(
            answer_v2_providers="DeepInfra",
            answer_v2_reasoning="off",
            answer_v2_deny_data_collection=True,
        )
        with patch("aura.answer_contract.litellm.acompletion", completion):
            await synthesize_contract_answer(
                [FACT], "q", "en-US", model="openrouter/a/synth", settings=settings
            )

        assert completion.await_args is not None
        assert completion.await_args.kwargs["extra_body"] == {
            "provider": {
                "order": ["DeepInfra"],
                "allow_fallbacks": False,
                "data_collection": "deny",
            },
            "reasoning": {"enabled": False},
        }

    async def test_the_check_call_sends_its_own_route_not_the_synthesis_route(self) -> None:
        completion = AsyncMock(return_value=_response(VERDICT))
        settings = _settings(
            answer_v2_providers="DeepInfra",
            answer_v2_reasoning="off",
            answer_v2_check_providers="Together",
            answer_v2_check_reasoning="low",
        )
        with patch("aura.answer_check.litellm.acompletion", completion):
            await verify_answer_v2(
                build_statements("The cup starts at 18:00.", [], (1,)),
                [FACT],
                settings=settings,
                timeout_seconds=5.0,
            )

        assert completion.await_args is not None
        assert completion.await_args.kwargs["extra_body"] == {
            "provider": {"order": ["Together"], "allow_fallbacks": False},
            "reasoning": {"effort": "low"},
        }

    async def test_without_options_the_check_call_carries_no_extra_body(self) -> None:
        completion = AsyncMock(return_value=_response(VERDICT))
        with patch("aura.answer_check.litellm.acompletion", completion):
            await verify_answer_v2(
                build_statements("The cup starts at 18:00.", [], (1,)),
                [FACT],
                settings=_settings(),
                timeout_seconds=5.0,
            )

        assert completion.await_args is not None
        assert "extra_body" not in completion.await_args.kwargs
