"""Tests for the v2 answer format's wiring: settings, /aura-ask, proactive relief, the preview.

What must hold:

* With the defaults (ANSWER_FORMAT=legacy, PROACTIVE_ANSWER_FORMAT=legacy) no v2
  code runs on either trigger -- every v2 entry point is a tripwire -- and the
  legacy replies are unchanged.
* With ANSWER_FORMAT=v2 (in tests only), /aura-ask answers through the contract,
  checks every displayed statement, and sends cards; the slot, the caps and the
  free answer's rules are the legacy ones.
* Proactive relief in v2 stays silent on conflicts and "unclear if same" pairs.
* The ANSWER_V2_* route goes with /aura-ask's answer only, never with proactive
  relief's model; the check route goes with every check (issue #11).
* The operator preview is operator-only, ephemeral, and touches no model, no
  ledger and no database.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import aiosqlite
import discord
import numpy as np
import pytest
from discord import app_commands
from fastembed import TextEmbedding
from litellm.types.utils import ModelResponse

from aura.answer_contract import validate_contract
from aura.billing import PlanGate
from aura.commands.ask import ask_command
from aura.commands.operator import (
    _handle_operator_preview_error,
    _is_operator,
    operator_preview_command,
)
from aura.commands.preview_samples import preview_samples
from aura.config import (
    AnswerFormat,
    CardStyle,
    ConfigurationError,
    ModelComponent,
    Settings,
    load_settings,
)
from aura.db.models import Fact
from aura.db.proactive_channel_config import set_channel_enabled
from aura.db.repository import init_schema
from aura.facts_service import add_fact
from aura.grounding import GroundingOutcome
from aura.i18n import t
from aura.proactive.responder import respond_with_synthesis
from aura.synthesis import SynthesisResult
from aura.theme import ACCENT_COLORS, MessageKind

GUILD = 100000000000000001
OPERATOR = 4242


@pytest.fixture
async def conn():
    connection = await aiosqlite.connect(":memory:")
    await init_schema(connection)
    yield connection
    await connection.close()


def _settings(**overrides: object) -> Settings:
    defaults: dict[str, object] = {
        "discord_token": "fake-token",
        "llm_api_key": "fake-key",
        "synthesis_model": "openrouter/fake/synth",
        "grounding_check_model": "openrouter/fake/check",
        "similarity_threshold": 0.0,
    }
    defaults.update(overrides)
    return Settings(_env_file=None, **defaults)  # type: ignore[arg-type]


def _v2(**overrides: object) -> Settings:
    return _settings(answer_format="v2", **overrides)


def _gate(*, pro: bool = True) -> MagicMock:
    gate = MagicMock(spec=PlanGate)
    gate.allows_pro = MagicMock(return_value=pro)
    return gate


def _interaction(
    conn: aiosqlite.Connection, model: Any, settings: Settings, *, pro: bool = True
) -> MagicMock:
    interaction = MagicMock(spec=discord.Interaction)
    interaction.locale = "en-US"
    interaction.guild_id = GUILD
    interaction.guild = None
    interaction.channel_id = 42
    interaction.user = MagicMock()
    interaction.user.id = 111
    interaction.created_at = datetime.now(UTC)
    interaction.client = MagicMock()
    interaction.client.db = conn
    interaction.client.embedding_model = model
    interaction.client.settings = settings
    interaction.client.plan_gate = _gate(pro=pro)
    interaction.response = MagicMock()
    interaction.response.defer = AsyncMock()
    interaction.response.is_done = MagicMock(return_value=True)
    interaction.followup = MagicMock()
    interaction.followup.send = AsyncMock()
    interaction.delete_original_response = AsyncMock()
    return interaction


async def _ask(interaction: MagicMock, question: str = "When does the event start?") -> None:
    await ask_command.callback(interaction, question)  # type: ignore[call-arg, arg-type]  # pyright: ignore


async def _add(conn: aiosqlite.Connection, model: Any, content: str, message_id: int = 1) -> Fact:
    return await add_fact(
        conn, model, guild_id=GUILD, channel_id=11, message_id=message_id, content=content
    )


def _contract_for(fact: Fact, **overrides: Any) -> Any:
    reply: dict[str, Any] = {
        "request_reading": "r",
        "fact_notes": [{"n": 1, "covers": "c"}],
        "relations": [],
        "not_covered_topics": [],
        "tone": "neutral",
        "lead": "The event starts on Saturday at 18:00.",
        "points": [],
        "used_fact_numbers": [1],
        "answers_question": True,
    }
    reply.update(overrides)
    return validate_contract(reply, [fact])


def _tripwire(name: str) -> AsyncMock:
    return AsyncMock(side_effect=AssertionError(f"{name} must not run here"))


def _sent(interaction: MagicMock) -> dict[str, Any]:
    interaction.followup.send.assert_awaited_once()
    call = interaction.followup.send.await_args
    assert call is not None
    return {"args": call.args, **call.kwargs}


# --- settings -------------------------------------------------------------------


class TestSettings:
    def test_the_defaults_ship_dark(self) -> None:
        settings = _settings()

        assert settings.answer_format is AnswerFormat.LEGACY
        assert settings.proactive_answer_format is AnswerFormat.LEGACY
        assert settings.answer_card_style is CardStyle.EMBED
        assert settings.answer_v2_max_output_tokens == 1000
        assert settings.answer_v2_check_max_output_tokens == 600

    def test_they_are_read_from_the_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for name, value in {
            "DISCORD_TOKEN": "x",
            "LLM_API_KEY": "k",
            "GROUNDING_CHECK_MODEL": "g/c",
            "ANSWER_FORMAT": "v2",
            "PROACTIVE_ANSWER_FORMAT": "v2",
            "ANSWER_CARD_STYLE": "container",
            "ANSWER_V2_MODEL": "a/m",
            "ANSWER_V2_CHECK_MODEL": "a/c",
        }.items():
            monkeypatch.setenv(name, value)
        settings = Settings(_env_file=None)  # type: ignore[call-arg]

        assert settings.answer_format is AnswerFormat.V2
        assert settings.proactive_answer_format is AnswerFormat.V2
        assert settings.answer_card_style is CardStyle.CONTAINER
        assert settings.resolve_model(ModelComponent.ANSWER_V2) == "a/m"
        assert settings.resolve_model(ModelComponent.ANSWER_V2_CHECK) == "a/c"

    @pytest.mark.parametrize("field", ["answer_format", "proactive_answer_format"])
    def test_v2_without_any_checker_model_refuses_to_start(
        self, field: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("DISCORD_TOKEN", "x")
        monkeypatch.setenv(field.upper(), "v2")
        with pytest.raises(ConfigurationError, match="never sent unchecked"):
            load_settings()

    @pytest.mark.parametrize("value", ["V3", "true", ""])
    def test_an_unknown_format_or_style_is_refused(self, value: str) -> None:
        with pytest.raises(ValueError):
            _settings(answer_format=value)
        with pytest.raises(ValueError):
            _settings(answer_card_style=value)

    def test_the_v2_model_falls_back_to_the_synthesis_model(self) -> None:
        assert _settings().resolve_model(ModelComponent.ANSWER_V2) == "openrouter/fake/synth"

    def test_the_v2_checker_falls_back_to_the_grounding_checker_never_to_synthesis(self) -> None:
        assert _settings().resolve_model(ModelComponent.ANSWER_V2_CHECK) == "openrouter/fake/check"
        assert (
            _settings(grounding_check_model=None).resolve_model(ModelComponent.ANSWER_V2_CHECK)
            is None
        )

    @pytest.mark.parametrize(
        ("field", "low", "high"),
        [
            ("answer_v2_max_output_tokens", 255, 8193),
            ("answer_v2_check_max_output_tokens", 127, 4097),
        ],
    )
    def test_the_output_ceilings_are_bounded(self, field: str, low: int, high: int) -> None:
        for value in (low, high):
            with pytest.raises(ValueError):
                _settings(**{field: value})


# --- /aura-ask, legacy default ---------------------------------------------------


class TestAskLegacyIsUntouched:
    async def test_a_paid_answer_runs_no_v2_code_and_sends_the_legacy_embed(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        fact = await _add(conn, embedding_model, "The event starts on Saturday at 18:00.")
        synth = AsyncMock(
            return_value=SynthesisResult(
                answer="Saturday.", used_fact_ids=[fact.id], answers_question=True
            )
        )
        with (
            patch("aura.commands.ask.synthesize_answer", synth),
            patch(
                "aura.commands.ask.verify_answer_grounded",
                AsyncMock(return_value=GroundingOutcome.GROUNDED),
            ),
            patch("aura.commands.ask.synthesize_contract_answer", _tripwire("v2 synthesis")),
            patch("aura.commands.ask.verify_answer_v2", _tripwire("v2 check")),
        ):
            interaction = _interaction(conn, embedding_model, _settings())
            await _ask(interaction)

        sent = _sent(interaction)
        embed = sent["embed"]
        assert set(sent) == {"args", "embed"}
        assert embed.description == "Saturday."
        assert embed.colour is None
        assert embed.author.name is None
        assert embed.fields[0].value == (
            f"https://discord.com/channels/{GUILD}/{fact.channel_id}/{fact.message_id}"
        )

    async def test_no_information_is_the_legacy_plain_line(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        interaction = _interaction(conn, embedding_model, _settings(similarity_threshold=0.99))
        with patch("aura.commands.ask.build_notice_card", MagicMock(side_effect=AssertionError)):
            await _ask(interaction, "zzzz qqqq")

        assert _sent(interaction)["args"] == (t("ask_no_info", "en-US"),)

    async def test_not_configured_is_the_legacy_plain_line(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        interaction = _interaction(conn, embedding_model, _settings(synthesis_model=None))
        await _ask(interaction)

        assert _sent(interaction)["args"] == (t("ask_not_configured", "en-US"),)


# --- /aura-ask, v2 ----------------------------------------------------------------


class TestAskV2:
    async def test_a_grounded_answer_is_sent_as_an_answer_card(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        fact = await _add(conn, embedding_model, "The event starts on Saturday at 18:00.")
        synth = AsyncMock(return_value=_contract_for(fact))
        check = AsyncMock(return_value=GroundingOutcome.GROUNDED)
        with (
            patch("aura.commands.ask.synthesize_contract_answer", synth),
            patch("aura.commands.ask.verify_answer_v2", check),
            patch("aura.commands.ask.synthesize_answer", _tripwire("legacy synthesis")),
            patch("aura.commands.ask.verify_answer_grounded", _tripwire("legacy check")),
        ):
            interaction = _interaction(conn, embedding_model, _v2(answer_v2_model="v2/model"))
            await _ask(interaction)

        assert synth.await_args is not None
        assert synth.await_args.kwargs["model"] == "v2/model"
        statements, cited = check.await_args.args  # type: ignore[union-attr]
        assert [s.statement_id for s in statements] == ["L"]
        assert statements[0].text == "The event starts on Saturday at 18:00."
        assert [f.id for f in cited] == [fact.id]
        sent = _sent(interaction)
        assert sent["embed"].colour.value == ACCENT_COLORS[MessageKind.ANSWER]
        assert sent["allowed_mentions"].everyone is False
        async with conn.execute("SELECT COUNT(*) FROM ask_calls") as cursor:
            assert (await cursor.fetchone()) == (1,)

    async def test_the_container_style_sends_a_view_with_mentions_disabled(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        fact = await _add(conn, embedding_model, "The event starts on Saturday at 18:00.")
        with (
            patch(
                "aura.commands.ask.synthesize_contract_answer",
                AsyncMock(return_value=_contract_for(fact)),
            ),
            patch(
                "aura.commands.ask.verify_answer_v2",
                AsyncMock(return_value=GroundingOutcome.GROUNDED),
            ),
        ):
            interaction = _interaction(conn, embedding_model, _v2(answer_card_style="container"))
            await _ask(interaction)

        sent = _sent(interaction)
        assert isinstance(sent["view"], discord.ui.LayoutView)
        assert "embed" not in sent
        assert sent["allowed_mentions"].users is False

    @pytest.mark.parametrize(
        ("outcome", "key"),
        [
            (GroundingOutcome.UNGROUNDED, "ask_grounding_rejected"),
            (GroundingOutcome.CHECK_FAILED, "ask_grounding_unverified"),
        ],
    )
    async def test_a_refused_check_sends_the_honest_notice_not_the_answer(
        self,
        conn: aiosqlite.Connection,
        embedding_model: TextEmbedding,
        outcome: GroundingOutcome,
        key: str,
    ) -> None:
        fact = await _add(conn, embedding_model, "The event starts on Saturday at 18:00.")
        with (
            patch(
                "aura.commands.ask.synthesize_contract_answer",
                AsyncMock(return_value=_contract_for(fact)),
            ),
            patch("aura.commands.ask.verify_answer_v2", AsyncMock(return_value=outcome)),
        ):
            interaction = _interaction(conn, embedding_model, _v2())
            await _ask(interaction)

        description = _sent(interaction)["embed"].description
        assert t(key, "en-US") in description
        assert "18:00" not in description

    async def test_a_failed_synthesis_sends_the_error_card_and_never_checks(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        await _add(conn, embedding_model, "The event starts on Saturday at 18:00.")
        with (
            patch("aura.commands.ask.synthesize_contract_answer", AsyncMock(return_value=None)),
            patch("aura.commands.ask.verify_answer_v2", _tripwire("v2 check")),
        ):
            interaction = _interaction(conn, embedding_model, _v2())
            await _ask(interaction)

        embed = _sent(interaction)["embed"]
        assert t("ask_error", "en-US") in embed.description
        assert embed.colour.value == ACCENT_COLORS[MessageKind.ERROR]

    async def test_an_answer_citing_nothing_shows_the_template_not_the_models_words(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        fact = await _add(conn, embedding_model, "The event starts on Saturday at 18:00.")
        contract = _contract_for(
            fact, lead="MODEL-WORDS-CANARY", used_fact_numbers=[], answers_question=False
        )
        with (
            patch("aura.commands.ask.synthesize_contract_answer", AsyncMock(return_value=contract)),
            patch("aura.commands.ask.verify_answer_v2", _tripwire("v2 check")),
        ):
            interaction = _interaction(conn, embedding_model, _v2())
            await _ask(interaction)

        description = _sent(interaction)["embed"].description
        assert t("ask_no_info", "en-US") in description
        assert "CANARY" not in description

    async def test_nothing_found_is_a_no_information_card_without_a_model_or_slot(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        with patch("aura.commands.ask.synthesize_contract_answer", _tripwire("v2 synthesis")):
            interaction = _interaction(conn, embedding_model, _v2(similarity_threshold=0.99))
            await _ask(interaction, "zzzz qqqq")

        embed = _sent(interaction)["embed"]
        assert t("ask_no_info", "en-US") in embed.description
        assert embed.colour.value == ACCENT_COLORS[MessageKind.RELATED]
        async with conn.execute("SELECT COUNT(*) FROM ask_calls") as cursor:
            assert (await cursor.fetchone()) == (0,)

    async def test_a_capped_question_gets_an_ephemeral_limit_card_without_a_model(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        await _add(conn, embedding_model, "The event starts on Saturday at 18:00.")
        with patch("aura.commands.ask.synthesize_contract_answer", _tripwire("v2 synthesis")):
            interaction = _interaction(conn, embedding_model, _v2(ask_daily_cap_pro=0))
            await _ask(interaction)

        interaction.delete_original_response.assert_awaited_once()
        sent = _sent(interaction)
        assert sent["ephemeral"] is True
        assert sent["embed"].colour.value == ACCENT_COLORS[MessageKind.LIMIT]
        assert "18:00" in sent["embed"].description

    async def test_not_configured_is_an_error_card(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        interaction = _interaction(conn, embedding_model, _v2(synthesis_model=None))
        await _ask(interaction)

        assert t("ask_not_configured", "en-US") in _sent(interaction)["embed"].description

    async def test_a_card_discord_refuses_goes_out_as_plain_text(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        fact = await _add(conn, embedding_model, "The event starts on Saturday at 18:00.")
        response = MagicMock(status=400, reason="Bad Request")
        refused = discord.HTTPException(response, "Invalid Form Body")
        with (
            patch(
                "aura.commands.ask.synthesize_contract_answer",
                AsyncMock(return_value=_contract_for(fact)),
            ),
            patch(
                "aura.commands.ask.verify_answer_v2",
                AsyncMock(return_value=GroundingOutcome.GROUNDED),
            ),
        ):
            interaction = _interaction(conn, embedding_model, _v2())
            interaction.followup.send = AsyncMock(side_effect=[refused, None])
            await _ask(interaction)

        second = interaction.followup.send.await_args_list[1]
        assert "18:00" in second.args[0]
        assert second.kwargs["allowed_mentions"].everyone is False

    async def test_any_other_send_failure_propagates_as_in_the_legacy_format(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        fact = await _add(conn, embedding_model, "The event starts on Saturday at 18:00.")
        forbidden = discord.HTTPException(
            MagicMock(status=403, reason="Forbidden"), "Missing Access"
        )
        with (
            patch(
                "aura.commands.ask.synthesize_contract_answer",
                AsyncMock(return_value=_contract_for(fact)),
            ),
            patch(
                "aura.commands.ask.verify_answer_v2",
                AsyncMock(return_value=GroundingOutcome.GROUNDED),
            ),
        ):
            interaction = _interaction(conn, embedding_model, _v2())
            interaction.followup.send = AsyncMock(side_effect=[forbidden, None])
            with pytest.raises(discord.HTTPException):
                await _ask(interaction)

        assert interaction.followup.send.await_count == 1

    async def test_the_outcome_log_line_carries_no_content(
        self,
        conn: aiosqlite.Connection,
        embedding_model: TextEmbedding,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        fact = await _add(conn, embedding_model, "The event starts on Saturday at 18:00.")
        with (
            patch(
                "aura.commands.ask.synthesize_contract_answer",
                AsyncMock(return_value=_contract_for(fact)),
            ),
            patch(
                "aura.commands.ask.verify_answer_v2",
                AsyncMock(return_value=GroundingOutcome.GROUNDED),
            ),
            caplog.at_level(logging.INFO),
        ):
            await _ask(_interaction(conn, embedding_model, _v2()), "QUESTION-CANARY when?")

        lines = [r.getMessage() for r in caplog.records if "v2 answer" in r.getMessage()]
        assert lines == [
            "/aura-ask v2 answer in guild 1000…: grounded (0 point(s), 1 source(s), answers_question=True)"
        ]
        assert "CANARY" not in caplog.text
        assert str(GUILD) not in caplog.text


# --- proactive relief ---------------------------------------------------------------


class _MatchingModel:
    def embed(self, documents: list[str], **_kwargs: object):
        for _ in documents:
            yield np.ones(4, dtype=np.float32)


async def _proactive_setup(conn: aiosqlite.Connection) -> tuple[Fact, MagicMock]:
    fact = await add_fact(
        conn,
        _MatchingModel(),
        guild_id=GUILD,
        channel_id=1,
        message_id=1,
        content="Rules are in #welcome.",  # type: ignore[arg-type]
    )
    await set_channel_enabled(conn, guild_id=GUILD, channel_id=555, enabled=True, updated_by_id=1)
    message = MagicMock(spec=discord.Message)
    message.content = "where are the rules?"
    message.guild = MagicMock()
    message.guild.id = GUILD
    message.guild.preferred_locale = "en-US"
    message.channel = MagicMock()
    message.channel.id = 555
    message.channel.send = AsyncMock()
    return fact, message


async def _respond(conn: aiosqlite.Connection, message: MagicMock, settings: Settings) -> Any:
    return await respond_with_synthesis(message, db=conn, model=_MatchingModel(), settings=settings)  # type: ignore[arg-type]


class TestProactive:
    async def test_the_default_runs_no_v2_code(self, conn: aiosqlite.Connection) -> None:
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
            patch("aura.proactive.responder.synthesize_contract_answer", _tripwire("v2 synthesis")),
            patch("aura.proactive.responder.verify_answer_v2", _tripwire("v2 check")),
        ):
            outcome = await _respond(conn, message, _settings())

        assert outcome.posted is True
        assert message.channel.send.await_args.kwargs["embed"].description == "In #welcome."  # type: ignore[union-attr]

    async def test_v2_posts_a_proactive_card_with_mentions_disabled(
        self, conn: aiosqlite.Connection
    ) -> None:
        fact, message = await _proactive_setup(conn)
        synth = AsyncMock(return_value=_contract_for(fact, lead="The rules are in #welcome."))
        with (
            patch("aura.proactive.responder.synthesize_contract_answer", synth),
            patch(
                "aura.proactive.responder.verify_answer_v2",
                AsyncMock(return_value=GroundingOutcome.GROUNDED),
            ),
            patch("aura.proactive.responder.synthesize_answer", _tripwire("legacy synthesis")),
        ):
            outcome = await _respond(
                conn, message, _settings(proactive_answer_format="v2", proactive_model="pro/active")
            )

        assert outcome.posted is True
        assert synth.await_args is not None
        assert synth.await_args.kwargs["model"] == "pro/active"
        kwargs = message.channel.send.await_args.kwargs  # type: ignore[union-attr]
        assert kwargs["embed"].colour.value == ACCENT_COLORS[MessageKind.PROACTIVE]
        assert kwargs["allowed_mentions"].everyone is False

    @pytest.mark.parametrize(
        "overrides",
        [
            {
                "relations": [{"facts": [1, 2], "kind": "unclear_if_same"}],
                "used_fact_numbers": [1, 2],
            },
            {
                "relations": [{"facts": [1, 2], "kind": "same_detail_conflict"}],
                "used_fact_numbers": [1, 2],
            },
            {"answers_question": False},
            {"used_fact_numbers": []},
        ],
    )
    async def test_v2_stays_silent_unless_the_answer_answers_unprompted(
        self, conn: aiosqlite.Connection, overrides: dict[str, Any]
    ) -> None:
        fact, message = await _proactive_setup(conn)
        second = await add_fact(
            conn,
            _MatchingModel(),
            guild_id=GUILD,
            channel_id=1,
            message_id=2,
            content="Rules are in #rules.",  # type: ignore[arg-type]
        )
        reply = {
            "request_reading": "r",
            "fact_notes": [],
            "relations": [],
            "not_covered_topics": [],
            "tone": "neutral",
            "lead": "The rules are pinned.",
            "points": [],
            "used_fact_numbers": [1],
            "answers_question": True,
            **overrides,
        }
        contract = validate_contract(reply, [fact, second])
        with (
            patch(
                "aura.proactive.responder.synthesize_contract_answer",
                AsyncMock(return_value=contract),
            ),
            patch("aura.proactive.responder.verify_answer_v2", _tripwire("v2 check")),
        ):
            outcome = await _respond(conn, message, _settings(proactive_answer_format="v2"))

        assert outcome.posted is False
        message.channel.send.assert_not_called()

    @pytest.mark.parametrize(
        "verdict", [GroundingOutcome.UNGROUNDED, GroundingOutcome.CHECK_FAILED]
    )
    async def test_v2_stays_silent_when_the_check_refuses(
        self,
        conn: aiosqlite.Connection,
        verdict: GroundingOutcome,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        fact, message = await _proactive_setup(conn)
        with (
            patch(
                "aura.proactive.responder.synthesize_contract_answer",
                AsyncMock(return_value=_contract_for(fact)),
            ),
            patch("aura.proactive.responder.verify_answer_v2", AsyncMock(return_value=verdict)),
            caplog.at_level(logging.WARNING),
        ):
            outcome = await _respond(conn, message, _settings(proactive_answer_format="v2"))

        assert outcome.posted is False
        message.channel.send.assert_not_called()
        assert "v2 answer check returned" in caplog.text

    async def test_v2_obeys_a_channel_disabled_mid_flight(self, conn: aiosqlite.Connection) -> None:
        fact, message = await _proactive_setup(conn)

        async def disable_then_pass(*_args: object, **_kwargs: object) -> GroundingOutcome:
            await set_channel_enabled(
                conn, guild_id=GUILD, channel_id=555, enabled=False, updated_by_id=1
            )
            return GroundingOutcome.GROUNDED

        with (
            patch(
                "aura.proactive.responder.synthesize_contract_answer",
                AsyncMock(return_value=_contract_for(fact)),
            ),
            patch(
                "aura.proactive.responder.verify_answer_v2",
                AsyncMock(side_effect=disable_then_pass),
            ),
        ):
            outcome = await _respond(conn, message, _settings(proactive_answer_format="v2"))

        assert outcome.posted is False
        message.channel.send.assert_not_called()

    async def test_v2_container_style_posts_a_view(self, conn: aiosqlite.Connection) -> None:
        fact, message = await _proactive_setup(conn)
        with (
            patch(
                "aura.proactive.responder.synthesize_contract_answer",
                AsyncMock(return_value=_contract_for(fact)),
            ),
            patch(
                "aura.proactive.responder.verify_answer_v2",
                AsyncMock(return_value=GroundingOutcome.GROUNDED),
            ),
        ):
            await _respond(
                conn,
                message,
                _settings(proactive_answer_format="v2", answer_card_style="container"),
            )

        assert isinstance(message.channel.send.await_args.kwargs["view"], discord.ui.LayoutView)  # type: ignore[union-attr]


# --- the request route per trigger (issue #11) -----------------------------------------

# A route for ANSWER_V2_MODEL pinned to providers that may not serve any other
# model, plus a separate route for the checker.
_ROUTED: dict[str, object] = {
    "answer_v2_model": "openrouter/fake/answer",
    "answer_v2_providers": "DeepInfra,Together",
    "answer_v2_reasoning": "off",
    "answer_v2_deny_data_collection": True,
    "answer_v2_check_providers": "Google",
    "answer_v2_check_reasoning": "low",
}
_ANSWER_ROUTE: dict[str, object] = {
    "provider": {
        "order": ["DeepInfra", "Together"],
        "allow_fallbacks": False,
        "data_collection": "deny",
    },
    "reasoning": {"enabled": False},
}
_CHECK_ROUTE: dict[str, object] = {
    "provider": {"order": ["Google"], "allow_fallbacks": False, "data_collection": "deny"},
    "reasoning": {"effort": "low"},
}


def _model_reply(payload: object) -> MagicMock:
    response = MagicMock(spec=ModelResponse)
    choice = MagicMock()
    choice.message.content = json.dumps(payload)
    choice.finish_reason = "stop"
    response.choices = [choice]
    response.usage = MagicMock(prompt_tokens=1, completion_tokens=1)
    return response


def _contract_reply(lead: str) -> MagicMock:
    return _model_reply(
        {
            "request_reading": "r",
            "fact_notes": [{"n": 1, "covers": "c"}],
            "relations": [],
            "not_covered_topics": [],
            "tone": "neutral",
            "lead": lead,
            "points": [],
            "used_fact_numbers": [1],
            "answers_question": True,
        }
    )


_GROUNDED_REPLY: dict[str, object] = {"statements": [{"id": "L", "issues": [], "supported": True}]}


def _two_calls(lead: str) -> AsyncMock:
    # aura.answer_contract and aura.answer_check share one litellm module, so
    # one mock serves both calls; the synthesis comes first, the check second.
    return AsyncMock(side_effect=[_contract_reply(lead), _model_reply(_GROUNDED_REPLY)])


def _calls_by_model(completion: AsyncMock) -> dict[str, dict[str, Any]]:
    assert completion.await_count == 2
    return {call.kwargs["model"]: call.kwargs for call in completion.await_args_list}


class TestRoutePerTrigger:
    async def test_ask_sends_the_answer_route_and_the_check_route(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        await _add(conn, embedding_model, "The event starts on Saturday at 18:00.")
        completion = _two_calls("The event starts on Saturday at 18:00.")
        with patch("aura.answer_contract.litellm.acompletion", completion):
            interaction = _interaction(conn, embedding_model, _v2(**_ROUTED))
            await _ask(interaction)

        calls = _calls_by_model(completion)
        assert calls["openrouter/fake/answer"]["extra_body"] == _ANSWER_ROUTE
        assert calls["openrouter/fake/check"]["extra_body"] == _CHECK_ROUTE
        assert _sent(interaction)["embed"].colour.value == ACCENT_COLORS[MessageKind.ANSWER]

    async def test_proactive_sends_its_own_model_without_the_answer_route(
        self, conn: aiosqlite.Connection
    ) -> None:
        _, message = await _proactive_setup(conn)
        completion = _two_calls("The rules are in #welcome.")
        settings = _settings(
            proactive_answer_format="v2", proactive_model="openrouter/fake/proactive", **_ROUTED
        )
        with patch("aura.answer_contract.litellm.acompletion", completion):
            outcome = await _respond(conn, message, settings)

        assert outcome.posted is True
        calls = _calls_by_model(completion)
        assert "extra_body" not in calls["openrouter/fake/proactive"]
        assert calls["openrouter/fake/check"]["extra_body"] == _CHECK_ROUTE

    async def test_ask_without_route_settings_sends_no_extra_fields(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        await _add(conn, embedding_model, "The event starts on Saturday at 18:00.")
        completion = _two_calls("The event starts on Saturday at 18:00.")
        with patch("aura.answer_contract.litellm.acompletion", completion):
            await _ask(_interaction(conn, embedding_model, _v2()))

        calls = _calls_by_model(completion)
        assert "extra_body" not in calls["openrouter/fake/synth"]
        assert "extra_body" not in calls["openrouter/fake/check"]


# --- the operator preview -------------------------------------------------------------


class _NoDatabase:
    def __getattr__(self, name: str) -> Any:
        raise AssertionError(f"the preview touched the database ({name})")


def _preview_interaction(*, user_id: int = OPERATOR, operator: int | None = OPERATOR) -> MagicMock:
    interaction = MagicMock(spec=discord.Interaction)
    interaction.locale = "de"
    interaction.guild_id = GUILD
    interaction.user = MagicMock()
    interaction.user.id = user_id
    interaction.client = MagicMock()
    interaction.client.settings = _settings(operator_discord_user_id=operator)
    interaction.client.db = _NoDatabase()
    interaction.response = MagicMock()
    interaction.response.defer = AsyncMock()
    interaction.response.is_done = MagicMock(return_value=False)
    interaction.response.send_message = AsyncMock()
    interaction.followup = MagicMock()
    interaction.followup.send = AsyncMock()
    return interaction


async def _preview(interaction: MagicMock, style: str | None = None) -> None:
    choice = app_commands.Choice(name=style, value=style) if style else None
    await operator_preview_command.callback(interaction, choice)  # type: ignore[call-arg, arg-type]  # pyright: ignore


class TestOperatorPreview:
    def test_only_the_configured_operator_passes_the_check(self) -> None:
        assert _is_operator(_preview_interaction()) is True
        assert _is_operator(_preview_interaction(user_id=1)) is False
        assert _is_operator(_preview_interaction(operator=None)) is False
        assert operator_preview_command.guild_only is True

    async def test_everyone_else_gets_the_localized_refusal_ephemerally(self) -> None:
        interaction = _preview_interaction(user_id=1)
        await _handle_operator_preview_error(interaction, app_commands.CheckFailure())

        interaction.response.send_message.assert_awaited_once_with(
            t("operator_budget_permission_error", "de"), ephemeral=True
        )

    async def test_both_styles_are_shown_ephemerally_with_no_model_and_no_database(self) -> None:
        interaction = _preview_interaction()
        with patch("litellm.acompletion", _tripwire("a model")):
            await _preview(interaction)

        interaction.response.defer.assert_awaited_once_with(ephemeral=True, thinking=True)
        calls = interaction.followup.send.await_args_list
        assert len(calls) == 1 + 7 * 2
        assert all(call.kwargs["ephemeral"] is True for call in calls)
        assert sum("embed" in call.kwargs for call in calls) == 7
        assert sum("view" in call.kwargs for call in calls) == 7
        assert all(call.kwargs["allowed_mentions"].everyone is False for call in calls[1:])

    @pytest.mark.parametrize(("style", "key"), [("embed", "embed"), ("container", "view")])
    async def test_one_style_shows_only_that_style(self, style: str, key: str) -> None:
        interaction = _preview_interaction()
        await _preview(interaction, style)

        calls = interaction.followup.send.await_args_list[1:]
        assert len(calls) == 7
        assert all(key in call.kwargs for call in calls)

    async def test_a_sample_discord_refuses_is_reported_and_the_rest_still_shown(self) -> None:
        interaction = _preview_interaction()
        refused = discord.HTTPException(
            MagicMock(status=400, reason="Bad Request"), "Invalid Form Body"
        )
        interaction.followup.send = AsyncMock(side_effect=[None, refused, *([None] * 20)])
        await _preview(interaction, "embed")

        texts = [c.args[0] for c in interaction.followup.send.await_args_list if c.args]
        assert any("refused this sample (HTTP 400)" in text for text in texts)
        assert interaction.followup.send.await_count == 1 + 1 + 1 + 6

    def test_the_samples_cover_every_kind_through_the_real_validator(self) -> None:
        samples = preview_samples("de", guild_id=GUILD, now=datetime(2026, 10, 4, tzinfo=UTC))

        assert len(samples) == 7
        assert {sample.card.kind for sample in samples} == set(MessageKind)
        assert samples[2].card.notes[-1].startswith("Nicht vermerkt:")
        assert samples[1].card.notes == (t("answer_note_conflict", "de"),)
        english = preview_samples("en-US", guild_id=GUILD, now=datetime(2026, 10, 4, tzinfo=UTC))
        assert "Movie night" in english[0].card.paragraphs[0]


class TestComponentsV2Unavailable:
    async def test_the_container_style_falls_back_to_an_embed(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        fact = await _add(conn, embedding_model, "The event starts on Saturday at 18:00.")
        with (
            patch(
                "aura.commands.ask.synthesize_contract_answer",
                AsyncMock(return_value=_contract_for(fact)),
            ),
            patch(
                "aura.commands.ask.verify_answer_v2",
                AsyncMock(return_value=GroundingOutcome.GROUNDED),
            ),
            patch("aura.commands.ask.components_v2_available", return_value=False),
        ):
            interaction = _interaction(conn, embedding_model, _v2(answer_card_style="container"))
            await _ask(interaction)

        sent = _sent(interaction)
        assert "view" not in sent
        assert isinstance(sent["embed"], discord.Embed)

    async def test_the_preview_says_so_instead_of_failing(self) -> None:
        interaction = _preview_interaction()
        with patch("aura.commands.operator.components_v2_available", return_value=False):
            await _preview(interaction, "container")

        texts = [c.args[0] for c in interaction.followup.send.await_args_list[1:]]
        assert len(texts) == 7
        assert all("no Components V2 support" in text for text in texts)

    def test_the_installed_discord_py_supports_it(self) -> None:
        from aura.answer_card import components_v2_available

        assert components_v2_available()
