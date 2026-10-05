"""Proactive relief's own provider route (P5), at the real litellm boundary.

PROACTIVE_PROVIDERS, PROACTIVE_REASONING and PROACTIVE_DENY_DATA_COLLECTION
describe PROACTIVE_MODEL. They go with proactive relief's answer in both
formats and with nothing else: `/aura-ask` keeps its own (ANSWER_V2_*) route in
v2 and none in legacy, and the v2 check keeps ANSWER_V2_CHECK_*. Unset, every
call is exactly as before. The checks here patch litellm.acompletion itself,
so what is asserted is what would leave the process.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import aiosqlite
import pytest
from litellm.types.utils import ModelResponse

from aura.config import Settings
from aura.db.repository import init_schema
from aura.grounding import GroundingOutcome
from aura.synthesis import synthesize_answer
from tests.test_answer_v2_paths import (
    _add,
    _ask,
    _interaction,
    _MatchingModel,
    _proactive_setup,
    _respond,
)


@pytest.fixture
async def conn():
    """A fresh in-memory database with Aura's schema."""
    connection = await aiosqlite.connect(":memory:")
    await init_schema(connection)
    yield connection
    await connection.close()


PROACTIVE_ROUTE = {
    "provider": {"order": ["Google"], "allow_fallbacks": False, "data_collection": "deny"},
    "reasoning": {"effort": "low"},
}
LEGACY_REPLY = {
    "answer": "The rules are in #welcome.",
    "used_fact_numbers": [1],
    "answers_question": True,
}
PROACTIVE_REPLY_KIND = {"message_kind": "sincere_request"}
CONTRACT_REPLY = {
    "request_reading": "r",
    "fact_notes": [],
    "relations": [],
    "not_covered_topics": [],
    "tone": "neutral",
    "lead": "The rules are in #welcome.",
    "points": [],
    "used_fact_numbers": [1],
    "answers_question": True,
}


def _settings(**overrides: object) -> Settings:
    defaults: dict[str, object] = {
        "discord_token": "fake-token",
        "llm_api_key": "fake-key",
        "synthesis_model": "openrouter/fake/synth",
        "proactive_model": "openrouter/fake/proactive",
        "grounding_check_model": "openrouter/fake/check",
        "similarity_threshold": 0.0,
    }
    defaults.update(overrides)
    return Settings(_env_file=None, **defaults)  # type: ignore[arg-type]


def _routed(**overrides: object) -> Settings:
    return _settings(
        proactive_providers="Google",
        proactive_reasoning="low",
        proactive_deny_data_collection=True,
        **overrides,
    )


def _response(payload: object) -> MagicMock:
    response = MagicMock(spec=ModelResponse)
    choice = MagicMock()
    choice.message.content = json.dumps(payload)
    choice.finish_reason = "stop"
    response.choices = [choice]
    response.usage = MagicMock(prompt_tokens=10, completion_tokens=5)
    return response


def _extra_bodies(mock: AsyncMock) -> list[Any]:
    return [call.kwargs.get("extra_body") for call in mock.await_args_list]


class TestProactiveRelief:
    async def test_legacy_sends_the_proactive_route(self, conn: aiosqlite.Connection) -> None:
        _, message = await _proactive_setup(conn)
        llm = AsyncMock(return_value=_response(LEGACY_REPLY))
        with (
            patch("aura.synthesis.litellm.acompletion", llm),
            patch(
                "aura.proactive.responder.verify_answer_grounded",
                AsyncMock(return_value=GroundingOutcome.GROUNDED),
            ),
            patch("aura.synthesis.load_settings", return_value=_routed()),
        ):
            outcome = await _respond(conn, message, _routed())

        assert outcome.posted is True
        assert _extra_bodies(llm) == [PROACTIVE_ROUTE]
        assert llm.await_args is not None
        assert llm.await_args.kwargs["model"] == "openrouter/fake/proactive"

    async def test_legacy_sends_nothing_extra_by_default(self, conn: aiosqlite.Connection) -> None:
        _, message = await _proactive_setup(conn)
        llm = AsyncMock(return_value=_response(LEGACY_REPLY))
        with (
            patch("aura.synthesis.litellm.acompletion", llm),
            patch(
                "aura.proactive.responder.verify_answer_grounded",
                AsyncMock(return_value=GroundingOutcome.GROUNDED),
            ),
            patch("aura.synthesis.load_settings", return_value=_settings()),
        ):
            await _respond(conn, message, _settings())

        assert llm.await_args is not None
        assert "extra_body" not in llm.await_args.kwargs

    async def test_v2_sends_the_proactive_route_and_never_the_answer_route(
        self, conn: aiosqlite.Connection
    ) -> None:
        _, message = await _proactive_setup(conn)
        llm = AsyncMock(return_value=_response({**CONTRACT_REPLY, **PROACTIVE_REPLY_KIND}))
        settings = _routed(
            proactive_answer_format="v2",
            answer_v2_providers="DeepInfra",
            answer_v2_reasoning="off",
        )
        with (
            patch("aura.answer_contract.litellm.acompletion", llm),
            patch(
                "aura.proactive.responder.verify_answer_v2",
                AsyncMock(return_value=GroundingOutcome.GROUNDED),
            ),
        ):
            outcome = await _respond(conn, message, settings)

        assert outcome.posted is True
        assert _extra_bodies(llm) == [PROACTIVE_ROUTE]

    async def test_a_proactive_model_outside_openrouter_gets_no_route(
        self, conn: aiosqlite.Connection
    ) -> None:
        _, message = await _proactive_setup(conn)
        llm = AsyncMock(return_value=_response({**CONTRACT_REPLY, **PROACTIVE_REPLY_KIND}))
        settings = _routed(proactive_answer_format="v2", proactive_model="anthropic/direct")
        with (
            patch("aura.answer_contract.litellm.acompletion", llm),
            patch(
                "aura.proactive.responder.verify_answer_v2",
                AsyncMock(return_value=GroundingOutcome.GROUNDED),
            ),
        ):
            await _respond(conn, message, settings)

        assert llm.await_args is not None
        assert "extra_body" not in llm.await_args.kwargs


class TestAskNeverGetsTheProactiveRoute:
    async def test_legacy_ask(self, conn: aiosqlite.Connection) -> None:
        model = _MatchingModel()
        await _add(conn, model, "The event starts at 18:00.")
        settings = _routed()
        interaction = _interaction(conn, model, settings)
        llm = AsyncMock(return_value=_response({**LEGACY_REPLY, "answer": "At 18:00."}))
        with (
            patch("aura.synthesis.litellm.acompletion", llm),
            patch("aura.synthesis.load_settings", return_value=settings),
            patch(
                "aura.commands.ask.verify_answer_grounded",
                AsyncMock(return_value=GroundingOutcome.GROUNDED),
            ),
        ):
            await _ask(interaction)

        assert llm.await_args is not None
        assert "extra_body" not in llm.await_args.kwargs

    async def test_v2_ask_sends_only_the_answer_route(self, conn: aiosqlite.Connection) -> None:
        model = _MatchingModel()
        await _add(conn, model, "The event starts at 18:00.")
        settings = _routed(answer_format="v2", answer_v2_providers="DeepInfra")
        interaction = _interaction(conn, model, settings)
        llm = AsyncMock(return_value=_response({**CONTRACT_REPLY, "lead": "At 18:00."}))
        with (
            patch("aura.answer_contract.litellm.acompletion", llm),
            patch(
                "aura.commands.ask.verify_answer_v2",
                AsyncMock(return_value=GroundingOutcome.GROUNDED),
            ),
        ):
            await _ask(interaction)

        assert _extra_bodies(llm) == [
            {"provider": {"order": ["DeepInfra"], "allow_fallbacks": False}}
        ]


class TestTheLegacyFunctionItself:
    async def test_an_explicit_route_is_sent_and_none_is_nothing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from datetime import UTC, datetime

        from aura.db.models import Fact, FactStatus

        monkeypatch.setenv("LLM_API_KEY", "fake-key")
        monkeypatch.setenv("DISCORD_TOKEN", "fake-token")
        fact = Fact(
            id=1,
            guild_id=1,
            channel_id=1,
            message_id=1,
            content="x",
            embedding=b"",
            status=FactStatus.ACTIVE,
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
        )
        llm = AsyncMock(return_value=_response(LEGACY_REPLY))
        with patch("aura.synthesis.litellm.acompletion", llm):
            await synthesize_answer([fact], "q", "en-US", model="openrouter/a/b")
            await synthesize_answer(
                [fact],
                "q",
                "en-US",
                model="openrouter/a/b",
                extra_body={"reasoning": {"effort": "low"}},
            )

        assert _extra_bodies(llm) == [None, {"reasoning": {"effort": "low"}}]


class TestTheProactiveVariantInTheResponder:
    async def test_v2_uses_the_variant_with_the_posting_date_and_its_own_ceiling(
        self, conn: aiosqlite.Connection
    ) -> None:
        from aura.answer_contract import build_proactive_contract_messages

        fact, message = await _proactive_setup(conn)
        llm = AsyncMock(return_value=_response({**CONTRACT_REPLY, **PROACTIVE_REPLY_KIND}))
        settings = _settings(proactive_answer_format="v2", proactive_max_output_tokens=4000)
        with (
            patch("aura.answer_contract.litellm.acompletion", llm),
            patch(
                "aura.proactive.responder.verify_answer_v2",
                AsyncMock(return_value=GroundingOutcome.GROUNDED),
            ),
        ):
            outcome = await _respond(conn, message, settings)

        assert outcome.posted is True
        assert llm.await_args is not None
        assert llm.await_args.kwargs["max_tokens"] == 4000
        assert llm.await_args.kwargs["messages"] == build_proactive_contract_messages(
            [fact], message.content, "en-US", posted_at=message.created_at
        )

    async def test_without_its_own_ceiling_it_keeps_the_v2_one(
        self, conn: aiosqlite.Connection
    ) -> None:
        _, message = await _proactive_setup(conn)
        llm = AsyncMock(return_value=_response({**CONTRACT_REPLY, **PROACTIVE_REPLY_KIND}))
        with (
            patch("aura.answer_contract.litellm.acompletion", llm),
            patch(
                "aura.proactive.responder.verify_answer_v2",
                AsyncMock(return_value=GroundingOutcome.GROUNDED),
            ),
        ):
            await _respond(conn, message, _settings(proactive_answer_format="v2"))

        assert llm.await_args is not None
        assert llm.await_args.kwargs["max_tokens"] == 1000

    @pytest.mark.parametrize(
        "kind",
        [
            "request_to_a_person",
            "rhetorical_or_sarcastic",
            "venting_or_opinion",
            "statement_or_banter",
            "steering_attempt",
            "needs_earlier_conversation",
        ],
    )
    async def test_anything_but_a_sincere_request_is_silence(
        self, conn: aiosqlite.Connection, kind: str
    ) -> None:
        _, message = await _proactive_setup(conn)
        llm = AsyncMock(return_value=_response({**CONTRACT_REPLY, "message_kind": kind}))
        check = AsyncMock(side_effect=AssertionError("no check for a reply that is not posted"))
        with (
            patch("aura.answer_contract.litellm.acompletion", llm),
            patch("aura.proactive.responder.verify_answer_v2", check),
        ):
            outcome = await _respond(conn, message, _settings(proactive_answer_format="v2"))

        assert outcome.posted is False
        assert outcome.answers_question is True
        message.channel.send.assert_not_awaited()

    async def test_a_reply_without_the_kind_is_silence(self, conn: aiosqlite.Connection) -> None:
        _, message = await _proactive_setup(conn)
        llm = AsyncMock(return_value=_response(CONTRACT_REPLY))
        with patch("aura.answer_contract.litellm.acompletion", llm):
            outcome = await _respond(conn, message, _settings(proactive_answer_format="v2"))

        assert outcome.posted is False
        assert outcome.answers_question is None
