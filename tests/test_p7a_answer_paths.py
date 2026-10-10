"""The AI label and the privacy line on the real answer paths: present when on, absent when off."""

from __future__ import annotations

from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, patch

import aiosqlite
import pytest
from fastembed import TextEmbedding

from aura.db.repository import init_schema
from aura.grounding import GroundingOutcome
from aura.synthesis import SynthesisResult
from tests.test_answer_v2_paths import (
    _add,
    _ask,
    _contract_for,
    _interaction,
    _proactive_setup,
    _respond,
    _sent,
    _settings,
    _tripwire,
    _v2,
)


@pytest.fixture
async def conn() -> AsyncIterator[aiosqlite.Connection]:
    connection = await aiosqlite.connect(":memory:")
    await init_schema(connection)
    yield connection
    await connection.close()


LABELS = {
    "ai_label_enabled": True,
    "privacy_info_enabled": True,
    "privacy_policy_url": "https://example.org/privacy",
    "privacy_contact": "privacy@example.org",
}


async def _ask_v2(
    conn: aiosqlite.Connection, model: TextEmbedding, **overrides: object
) -> dict[str, object]:
    fact = await _add(conn, model, "The event starts on Saturday at 18:00.")
    with (
        patch(
            "aura.commands.ask.synthesize_contract_answer",
            AsyncMock(return_value=_contract_for(fact)),
        ),
        patch(
            "aura.commands.ask.verify_answer_v2", AsyncMock(return_value=GroundingOutcome.GROUNDED)
        ),
    ):
        interaction = _interaction(conn, model, _v2(**overrides))
        await _ask(interaction)
    return _sent(interaction)


class TestAskV2:
    async def test_switched_on_the_card_carries_the_label_and_the_privacy_line(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        embed = (await _ask_v2(conn, embedding_model, **LABELS))["embed"]
        assert embed.author.name.startswith("🤖 AI-generated · ")  # type: ignore[attr-defined]
        assert embed.footer.text.endswith(" · Privacy: /aura-privacy")  # type: ignore[attr-defined]

    async def test_switched_off_neither_appears(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        embed = (await _ask_v2(conn, embedding_model))["embed"]
        assert "AI-generated" not in (embed.author.name or "")  # type: ignore[attr-defined]
        assert "aura-privacy" not in (embed.footer.text or "")  # type: ignore[attr-defined]


class TestAskLegacy:
    async def test_the_legacy_embed_is_labelled_too(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        fact = await _add(conn, embedding_model, "The event starts on Saturday at 18:00.")
        result = SynthesisResult(answer="Saturday.", used_fact_ids=[fact.id], answers_question=True)
        with (
            patch("aura.commands.ask.synthesize_answer", AsyncMock(return_value=result)),
            patch(
                "aura.commands.ask.verify_answer_grounded",
                AsyncMock(return_value=GroundingOutcome.GROUNDED),
            ),
        ):
            interaction = _interaction(conn, embedding_model, _settings(**LABELS))
            await _ask(interaction)
        embed = _sent(interaction)["embed"]
        assert embed.author.name == "🤖 AI-generated"  # type: ignore[attr-defined]
        assert embed.footer.text == "Privacy: /aura-privacy"  # type: ignore[attr-defined]


class TestProactive:
    async def test_v2_card_is_labelled_when_switched_on(self, conn: aiosqlite.Connection) -> None:
        fact, message = await _proactive_setup(conn)
        with (
            patch(
                "aura.proactive.responder.synthesize_contract_answer",
                AsyncMock(return_value=_contract_for(fact, lead="The rules are in #welcome.")),
            ),
            patch(
                "aura.proactive.responder.verify_answer_v2",
                AsyncMock(return_value=GroundingOutcome.GROUNDED),
            ),
        ):
            outcome = await _respond(
                conn,
                message,
                _settings(proactive_answer_format="v2", proactive_model="pro/active", **LABELS),
            )
        assert outcome.posted is True
        embed = message.channel.send.await_args.kwargs["embed"]  # type: ignore[union-attr]
        assert embed.author.name.startswith("🤖 AI-generated · 💡")
        assert embed.footer.text.endswith("Privacy: /aura-privacy")

    async def test_legacy_embed_is_labelled_when_switched_on(
        self, conn: aiosqlite.Connection
    ) -> None:
        fact, message = await _proactive_setup(conn)
        legacy = SynthesisResult(
            answer="In #welcome.", used_fact_ids=[fact.id], answers_question=True
        )
        with (
            patch("aura.proactive.responder.synthesize_answer", AsyncMock(return_value=legacy)),
            patch(
                "aura.proactive.responder.verify_answer_grounded",
                AsyncMock(return_value=GroundingOutcome.GROUNDED),
            ),
            patch("aura.proactive.responder.synthesize_contract_answer", _tripwire("v2")),
        ):
            await _respond(conn, message, _settings(**LABELS))
        embed = message.channel.send.await_args.kwargs["embed"]  # type: ignore[union-attr]
        assert embed.author.name.startswith("🤖 AI-generated · ")
        assert embed.footer.text.endswith(" · Privacy: /aura-privacy")
