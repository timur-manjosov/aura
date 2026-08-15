"""Tests for aura.commands.ask: /aura-ask, the direct-query trigger.

Exercises the command callback, the cooldown check, and the error handler
directly against mocked discord.Interaction objects and a real in-memory
database -- never a live Discord connection, matching the rest of this
project's testing philosophy. synthesize_answer and litellm are always
mocked here too (test_synthesis.py already covers the LLM call itself in
depth) -- zero real API calls, zero cost, per this phase's hard constraint.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import aiosqlite
import discord
import pytest
from discord import app_commands
from fastembed import TextEmbedding

from aura.commands.ask import _handle_ask_command_error, ask_command
from aura.config import Settings
from aura.db.repository import init_schema
from aura.facts_service import add_fact
from aura.grounding import GroundingOutcome
from aura.synthesis import SynthesisResult

GUILD_A = 100000000000000001
GUILD_B = 200000000000000002


@pytest.fixture
async def conn():
    connection = await aiosqlite.connect(":memory:")
    await init_schema(connection)
    yield connection
    await connection.close()


def _fake_settings(**overrides: object) -> Settings:
    defaults: dict[str, object] = {
        "discord_token": "fake-token",
        "llm_api_key": "fake-key",
        "synthesis_model": "openrouter/fake/model",
        "similarity_threshold": 0.4,
    }
    defaults.update(overrides)
    return Settings(_env_file=None, **defaults)  # type: ignore[arg-type]


def _make_interaction(
    *,
    db: aiosqlite.Connection | None,
    embedding_model: TextEmbedding | None,
    settings: Settings,
    locale: str = "en-US",
    guild_id: int = GUILD_A,
    user_id: int = 111,
) -> MagicMock:
    """A mock Interaction exposing just what /aura-ask actually touches."""
    interaction = MagicMock(spec=discord.Interaction)
    interaction.locale = locale
    interaction.guild_id = guild_id
    interaction.user = MagicMock()
    interaction.user.id = user_id
    interaction.created_at = datetime.now(timezone.utc)
    interaction.client = MagicMock()
    interaction.client.db = db
    interaction.client.embedding_model = embedding_model
    interaction.client.settings = settings
    interaction.response = MagicMock()
    interaction.response.defer = AsyncMock()
    interaction.response.send_message = AsyncMock()
    interaction.response.is_done = MagicMock(return_value=False)
    interaction.followup = MagicMock()
    interaction.followup.send = AsyncMock()
    interaction.command = MagicMock()
    interaction.command.name = "aura-ask"
    return interaction


async def _invoke_ask(interaction: discord.Interaction, question: str) -> None:
    """Call ask_command's callback directly, bypassing its checks.

    Same discord.py CommandCallback union-typing gap as facts.py's
    _invoke_list_facts (see that docstring): at runtime this callback only
    ever takes (interaction, question), confirmed directly against the
    installed discord.py, but Pyright can't rule out a group/cog-bound
    union member from the stub alone.
    """
    await ask_command.callback(interaction, question)  # pyright: ignore[reportCallIssue]


async def _call_cooldown_check(interaction: discord.Interaction) -> bool:
    """Invoke ask_command's one check -- the cooldown predicate -- directly.

    .checks is typed as a broader union that also covers synchronous
    predicates; app_commands.checks.cooldown's specific predicate is a
    coroutine function at runtime (confirmed via inspect.getsource), so
    it's genuinely awaitable even though Pyright can't tell that from the
    general Check type alone.
    """
    check = ask_command.checks[0]
    return await check(interaction)  # pyright: ignore[reportGeneralTypeIssues]


class TestNotConfigured:
    async def test_replies_cleanly_and_does_no_further_work(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        settings = _fake_settings(llm_api_key=None, synthesis_model=None)
        interaction = _make_interaction(db=conn, embedding_model=embedding_model, settings=settings)

        mock_find = AsyncMock()
        mock_synth = AsyncMock()
        with (
            patch("aura.commands.ask.find_similar_facts", mock_find),
            patch("aura.commands.ask.synthesize_answer", mock_synth),
        ):
            await _invoke_ask(interaction, "any question")

        interaction.response.defer.assert_awaited_once()
        mock_find.assert_not_awaited()  # not even the embedding lookup runs
        mock_synth.assert_not_awaited()
        interaction.followup.send.assert_awaited_once()
        args, kwargs = interaction.followup.send.call_args
        assert "set up" in args[0].lower() or "configure" in args[0].lower()
        assert "embed" not in kwargs


class TestBelowThresholdAndZeroFacts:
    async def test_zero_facts_shows_no_info_reply_and_never_calls_synthesize(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        settings = _fake_settings()
        interaction = _make_interaction(db=conn, embedding_model=embedding_model, settings=settings)

        mock_synth = AsyncMock()
        with patch("aura.commands.ask.synthesize_answer", mock_synth):
            await _invoke_ask(interaction, "any question")

        mock_synth.assert_not_awaited()
        interaction.followup.send.assert_awaited_once()
        args, kwargs = interaction.followup.send.call_args
        assert "don't have" in args[0].lower() or "no information" in args[0].lower()
        assert "embed" not in kwargs

    async def test_below_threshold_shows_no_info_reply_and_never_calls_synthesize(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        await add_fact(
            conn, embedding_model, guild_id=GUILD_A, channel_id=1, message_id=1,
            content="we sell homemade candles on weekends",
        )
        # An all-but-impossible bar to clear with any real cosine score,
        # so the below-threshold branch is forced deterministically rather
        # than relying on a delicate embedding-score margin.
        settings = _fake_settings(similarity_threshold=0.99)
        interaction = _make_interaction(db=conn, embedding_model=embedding_model, settings=settings)

        mock_synth = AsyncMock()
        with patch("aura.commands.ask.synthesize_answer", mock_synth):
            await _invoke_ask(
                interaction, "completely unrelated question about martian weather"
            )

        mock_synth.assert_not_awaited()
        interaction.followup.send.assert_awaited_once()
        args, _ = interaction.followup.send.call_args
        assert "don't have" in args[0].lower() or "no information" in args[0].lower()


class TestDeferralTiming:
    async def test_defer_happens_before_find_similar_facts(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        # The one thing in this phase that silently breaks the entire
        # feature if missed: Discord's 3-second initial-response window is
        # essentially never fast enough for an LLM call, so defer() must
        # run before any of the slow work starts, not after.
        call_order: list[str] = []

        async def tracked_defer(*_args: object, **_kwargs: object) -> None:
            call_order.append("defer")

        async def tracked_find(*_args: object, **_kwargs: object) -> list[object]:
            call_order.append("find_similar_facts")
            return []

        settings = _fake_settings()
        interaction = _make_interaction(db=conn, embedding_model=embedding_model, settings=settings)
        interaction.response.defer = AsyncMock(side_effect=tracked_defer)

        with patch("aura.commands.ask.find_similar_facts", AsyncMock(side_effect=tracked_find)):
            await _invoke_ask(interaction, "any question")

        assert call_order == ["defer", "find_similar_facts"]

    async def test_defer_happens_even_when_llm_is_not_configured(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        settings = _fake_settings(llm_api_key=None, synthesis_model=None)
        interaction = _make_interaction(db=conn, embedding_model=embedding_model, settings=settings)

        await _invoke_ask(interaction, "any question")

        interaction.response.defer.assert_awaited_once()


class TestCooldown:
    """Cooldown state lives in a closure created once at import time (see
    discord.py's _create_cooldown_decorator), so every test here uses its
    own never-reused user IDs to avoid cross-test interference."""

    async def test_second_call_within_window_from_same_user_is_rejected(self) -> None:
        cooldown_check = ask_command.checks[0]
        now = datetime.now(timezone.utc)

        first = MagicMock(spec=discord.Interaction)
        first.user = MagicMock(id=900001)
        first.created_at = now

        second = MagicMock(spec=discord.Interaction)
        second.user = MagicMock(id=900001)  # same user
        second.created_at = now + timedelta(seconds=1)  # still well within the 30s window

        assert await _call_cooldown_check(first) is True
        with pytest.raises(app_commands.CommandOnCooldown):
            await _call_cooldown_check(second)

    async def test_different_user_is_unaffected_by_first_users_cooldown(self) -> None:
        cooldown_check = ask_command.checks[0]
        now = datetime.now(timezone.utc)

        first = MagicMock(spec=discord.Interaction)
        first.user = MagicMock(id=900002)
        first.created_at = now

        second = MagicMock(spec=discord.Interaction)
        second.user = MagicMock(id=900003)  # different user
        second.created_at = now + timedelta(seconds=1)

        assert await _call_cooldown_check(first) is True
        assert await _call_cooldown_check(second) is True  # not rejected


class TestCooldownErrorHandler:
    async def test_command_on_cooldown_replies_ephemerally_and_localized(self) -> None:
        interaction = _make_interaction(db=None, embedding_model=None, settings=_fake_settings())
        cooldown = app_commands.checks.Cooldown(1, 30.0)
        error = app_commands.CommandOnCooldown(cooldown, retry_after=12.3)

        await _handle_ask_command_error(interaction, error)

        interaction.response.send_message.assert_awaited_once()
        args, kwargs = interaction.response.send_message.call_args
        assert "12" in args[0]  # seconds shown, rounded
        assert kwargs.get("ephemeral") is True

    async def test_command_on_cooldown_uses_followup_if_already_responded(self) -> None:
        interaction = _make_interaction(db=None, embedding_model=None, settings=_fake_settings())
        interaction.response.is_done = MagicMock(return_value=True)
        cooldown = app_commands.checks.Cooldown(1, 30.0)
        error = app_commands.CommandOnCooldown(cooldown, retry_after=5.0)

        await _handle_ask_command_error(interaction, error)

        interaction.followup.send.assert_awaited_once()
        interaction.response.send_message.assert_not_awaited()

    async def test_unexpected_errors_are_logged_and_not_surfaced_to_the_user(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        interaction = _make_interaction(db=None, embedding_model=None, settings=_fake_settings())
        error = app_commands.CommandInvokeError(MagicMock(), ValueError("boom"))

        with caplog.at_level(logging.ERROR):
            await _handle_ask_command_error(interaction, error)

        interaction.response.send_message.assert_not_awaited()
        interaction.followup.send.assert_not_awaited()
        assert any(record.levelno == logging.ERROR for record in caplog.records)


class TestGuildIsolation:
    async def test_facts_fed_to_synthesis_are_guild_scoped(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        identical_content = "the server rules were updated last week"
        await add_fact(
            conn, embedding_model, guild_id=GUILD_A, channel_id=1, message_id=1,
            content=identical_content,
        )
        await add_fact(
            conn, embedding_model, guild_id=GUILD_B, channel_id=1, message_id=1,
            content=identical_content,
        )

        settings = _fake_settings(similarity_threshold=0.0)  # accept everything retrieved
        interaction = _make_interaction(
            db=conn, embedding_model=embedding_model, settings=settings, guild_id=GUILD_A
        )

        mock_synth = AsyncMock(return_value=SynthesisResult(answer="ans", used_fact_ids=[], answers_question=True))
        with patch("aura.commands.ask.synthesize_answer", mock_synth):
            await _invoke_ask(interaction, identical_content)

        mock_synth.assert_awaited_once()
        facts_arg = mock_synth.call_args[0][0]
        assert len(facts_arg) == 1
        assert facts_arg[0].guild_id == GUILD_A


class TestSuccessPath:
    async def test_answer_shown_and_only_cited_facts_become_sources(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        cited = await add_fact(
            conn, embedding_model, guild_id=GUILD_A, channel_id=11, message_id=101,
            content="the server was founded in 2020",
        )
        uncited = await add_fact(
            conn, embedding_model, guild_id=GUILD_A, channel_id=22, message_id=202,
            content="the server's founding year is 2020",
        )

        settings = _fake_settings(similarity_threshold=0.0)  # accept both retrieved facts
        interaction = _make_interaction(db=conn, embedding_model=embedding_model, settings=settings)

        # Only `cited` is in used_fact_ids, even though both facts were
        # retrieved and handed to synthesis -- only what the model actually
        # says it drew from becomes a source, not everything retrieved.
        fake_result = SynthesisResult(
            answer="The server was founded in 2020.", used_fact_ids=[cited.id], answers_question=True
        )
        with patch("aura.commands.ask.synthesize_answer", AsyncMock(return_value=fake_result)):
            await _invoke_ask(interaction, "When was the server founded?")

        interaction.followup.send.assert_awaited_once()
        _, kwargs = interaction.followup.send.call_args
        assert kwargs.get("ephemeral", False) is False  # visible to everyone, not ephemeral

        embed = kwargs["embed"]
        assert embed.description == "The server was founded in 2020."
        sources_value = embed.fields[0].value
        assert f"/{cited.channel_id}/{cited.message_id}" in sources_value
        assert f"/{uncited.channel_id}/{uncited.message_id}" not in sources_value

    async def test_no_citations_omits_the_sources_field(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        await add_fact(
            conn, embedding_model, guild_id=GUILD_A, channel_id=1, message_id=1,
            content="a relevant fact",
        )
        settings = _fake_settings(similarity_threshold=0.0)
        interaction = _make_interaction(db=conn, embedding_model=embedding_model, settings=settings)

        fake_result = SynthesisResult(
            answer="I could not find a clear answer.", used_fact_ids=[], answers_question=False
        )
        with patch("aura.commands.ask.synthesize_answer", AsyncMock(return_value=fake_result)):
            await _invoke_ask(interaction, "an unrelated question")

        _, kwargs = interaction.followup.send.call_args
        embed = kwargs["embed"]
        assert len(embed.fields) == 0


class TestSynthesisFailure:
    async def test_synthesis_returning_none_shows_localized_error(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        await add_fact(
            conn, embedding_model, guild_id=GUILD_A, channel_id=1, message_id=1,
            content="a relevant fact",
        )
        settings = _fake_settings(similarity_threshold=0.0)
        interaction = _make_interaction(db=conn, embedding_model=embedding_model, settings=settings)

        with patch("aura.commands.ask.synthesize_answer", AsyncMock(return_value=None)):
            await _invoke_ask(interaction, "a question")

        interaction.followup.send.assert_awaited_once()
        args, kwargs = interaction.followup.send.call_args
        assert "embed" not in kwargs
        assert "went wrong" in args[0].lower() or "error" in args[0].lower()


class TestLocalePassedThrough:
    async def test_interaction_locale_reaches_synthesize_answer(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        await add_fact(
            conn, embedding_model, guild_id=GUILD_A, channel_id=1, message_id=1,
            content="a relevant fact",
        )
        settings = _fake_settings(similarity_threshold=0.0)
        interaction = _make_interaction(
            db=conn, embedding_model=embedding_model, settings=settings, locale="de"
        )

        mock_synth = AsyncMock(return_value=SynthesisResult(answer="ans", used_fact_ids=[], answers_question=True))
        with patch("aura.commands.ask.synthesize_answer", mock_synth):
            await _invoke_ask(interaction, "a question")

        mock_synth.assert_awaited_once()
        # positional args: (facts, question, locale)
        assert mock_synth.call_args[0][2] == "de"

    async def test_the_synthesis_model_is_resolved_through_the_seam_and_passed_in(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        # /aura-ask no longer lets synthesize_answer read the model itself; it
        # resolves SYNTHESIS through Settings.resolve_model and passes it in, so
        # there is one model-resolution convention across both triggers.
        await add_fact(
            conn, embedding_model, guild_id=GUILD_A, channel_id=1, message_id=1,
            content="a relevant fact",
        )
        settings = _fake_settings(
            similarity_threshold=0.0, synthesis_model="openrouter/the/synth-model"
        )
        interaction = _make_interaction(db=conn, embedding_model=embedding_model, settings=settings)

        mock_synth = AsyncMock(
            return_value=SynthesisResult(answer="ans", used_fact_ids=[], answers_question=True)
        )
        with patch("aura.commands.ask.synthesize_answer", mock_synth):
            await _invoke_ask(interaction, "a question")

        assert mock_synth.call_args.kwargs["model"] == "openrouter/the/synth-model"


def _grounding(outcome: GroundingOutcome) -> AsyncMock:
    """Patchable stand-in for the grounding check, forced to one verdict."""
    return AsyncMock(return_value=outcome)


class TestGroundingCheck:
    """The independent check between synthesis and the followup that sends it.

    Every test here mocks the check itself -- aura.grounding's own suite covers
    the call. What is under test is /aura-ask's policy around it: what gets sent,
    what does not, and that no verdict can influence the text of an answer that
    does get sent.
    """

    async def _ask_with(
        self,
        conn: aiosqlite.Connection,
        embedding_model: TextEmbedding,
        *,
        outcome: GroundingOutcome,
        answer: str = "The server was founded in 2020.",
        cited: bool = True,
    ) -> MagicMock:
        fact = await add_fact(
            conn, embedding_model, guild_id=GUILD_A, channel_id=11, message_id=101,
            content="the server was founded in 2020",
        )
        settings = _fake_settings(
            similarity_threshold=0.0, grounding_check_model="openrouter/other/vendor"
        )
        interaction = _make_interaction(db=conn, embedding_model=embedding_model, settings=settings)
        result = SynthesisResult(
            answer=answer,
            used_fact_ids=[fact.id] if cited else [],
            answers_question=True,
        )
        with (
            patch("aura.commands.ask.synthesize_answer", AsyncMock(return_value=result)),
            patch("aura.commands.ask.verify_answer_grounded", _grounding(outcome)),
        ):
            await _invoke_ask(interaction, "When was the server founded?")
        return interaction

    async def test_a_grounded_answer_is_sent_unchanged(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        interaction = await self._ask_with(
            conn, embedding_model, outcome=GroundingOutcome.GROUNDED
        )
        _, kwargs = interaction.followup.send.call_args
        assert kwargs["embed"].description == "The server was founded in 2020."

    async def test_an_unconfigured_check_sends_exactly_as_before(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        # The one deliberate not-fail-closed path: an operator who never set
        # GROUNDING_CHECK_MODEL keeps the pre-feature behaviour rather than a
        # bot that answers nothing after a `git pull`.
        interaction = await self._ask_with(
            conn, embedding_model, outcome=GroundingOutcome.NOT_CONFIGURED
        )
        _, kwargs = interaction.followup.send.call_args
        assert kwargs["embed"].description == "The server was founded in 2020."

    async def test_an_ungrounded_answer_is_never_sent(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        interaction = await self._ask_with(
            conn, embedding_model, outcome=GroundingOutcome.UNGROUNDED
        )
        _, kwargs = interaction.followup.send.call_args
        assert "embed" not in kwargs  # the answer itself never left

    async def test_a_failed_check_is_never_sent_either(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        # The fail-closed case that matters most: the check broke, so the answer
        # is unverified -- which carries exactly the risk a rejected one does.
        interaction = await self._ask_with(
            conn, embedding_model, outcome=GroundingOutcome.CHECK_FAILED
        )
        _, kwargs = interaction.followup.send.call_args
        assert "embed" not in kwargs

    async def test_the_two_refusals_say_different_honest_things(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        # Same consequence, different truth: "the facts did not back it" and "I
        # could not check it" are not the same statement, and telling a user the
        # wrong one is its own small dishonesty.
        rejected = await self._ask_with(
            conn, embedding_model, outcome=GroundingOutcome.UNGROUNDED
        )
        unverified = await self._ask_with(
            conn, embedding_model, outcome=GroundingOutcome.CHECK_FAILED
        )
        rejected_text = rejected.followup.send.call_args[0][0]
        unverified_text = unverified.followup.send.call_args[0][0]
        assert rejected_text != unverified_text
        assert rejected_text.strip() and unverified_text.strip()

    async def test_a_refusal_still_replies_rather_than_falling_silent(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        # Somebody asked. A deferred interaction with no followup shows "the
        # application did not respond" -- worse than an honest refusal.
        for outcome in (GroundingOutcome.UNGROUNDED, GroundingOutcome.CHECK_FAILED):
            interaction = await self._ask_with(conn, embedding_model, outcome=outcome)
            interaction.followup.send.assert_awaited_once()

    async def test_the_check_receives_only_the_cited_facts(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        cited = await add_fact(
            conn, embedding_model, guild_id=GUILD_A, channel_id=11, message_id=101,
            content="the server was founded in 2020",
        )
        uncited = await add_fact(
            conn, embedding_model, guild_id=GUILD_A, channel_id=22, message_id=202,
            content="the server's founding year is 2020",
        )
        settings = _fake_settings(
            similarity_threshold=0.0, grounding_check_model="openrouter/other/vendor"
        )
        interaction = _make_interaction(db=conn, embedding_model=embedding_model, settings=settings)
        result = SynthesisResult(answer="2020.", used_fact_ids=[cited.id], answers_question=True)

        check = _grounding(GroundingOutcome.GROUNDED)
        with (
            patch("aura.commands.ask.synthesize_answer", AsyncMock(return_value=result)),
            patch("aura.commands.ask.verify_answer_grounded", check),
        ):
            await _invoke_ask(interaction, "When was the server founded?")

        # A claim supported only by a retrieved-but-uncited fact is still one the
        # reader cannot verify from the sources shown, so the check sees exactly
        # what the reader will.
        passed_facts = check.call_args.kwargs["cited_facts"]
        assert [fact.id for fact in passed_facts] == [cited.id]
        assert uncited.id not in [fact.id for fact in passed_facts]

    async def test_the_check_receives_the_exact_answer_that_would_be_sent(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        # If the checked string and the sent string could ever differ, the whole
        # feature would be theatre.
        answer = "Gegründet 2020 — „laut den Fakten“. 🎉"
        fact = await add_fact(
            conn, embedding_model, guild_id=GUILD_A, channel_id=11, message_id=101,
            content="the server was founded in 2020",
        )
        settings = _fake_settings(
            similarity_threshold=0.0, grounding_check_model="openrouter/other/vendor"
        )
        interaction = _make_interaction(db=conn, embedding_model=embedding_model, settings=settings)
        result = SynthesisResult(answer=answer, used_fact_ids=[fact.id], answers_question=True)

        check = _grounding(GroundingOutcome.GROUNDED)
        with (
            patch("aura.commands.ask.synthesize_answer", AsyncMock(return_value=result)),
            patch("aura.commands.ask.verify_answer_grounded", check),
        ):
            await _invoke_ask(interaction, "a question")

        assert check.call_args.kwargs["answer"] == answer
        _, kwargs = interaction.followup.send.call_args
        assert kwargs["embed"].description == answer

    async def test_the_check_runs_after_synthesis_and_before_the_send(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        order: list[str] = []
        fact = await add_fact(
            conn, embedding_model, guild_id=GUILD_A, channel_id=11, message_id=101,
            content="a relevant fact",
        )
        settings = _fake_settings(
            similarity_threshold=0.0, grounding_check_model="openrouter/other/vendor"
        )
        interaction = _make_interaction(db=conn, embedding_model=embedding_model, settings=settings)
        interaction.followup.send = AsyncMock(side_effect=lambda *a, **k: order.append("send"))

        async def synth(*_a: object, **_k: object) -> SynthesisResult:
            order.append("synthesize")
            return SynthesisResult(answer="ans", used_fact_ids=[fact.id], answers_question=True)

        async def check(*_a: object, **_k: object) -> GroundingOutcome:
            order.append("grounding")
            return GroundingOutcome.GROUNDED

        with (
            patch("aura.commands.ask.synthesize_answer", AsyncMock(side_effect=synth)),
            patch("aura.commands.ask.verify_answer_grounded", AsyncMock(side_effect=check)),
        ):
            await _invoke_ask(interaction, "a question")

        assert order == ["synthesize", "grounding", "send"]

    async def test_no_check_runs_when_synthesis_produced_nothing(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        # Nothing to check, and nothing to pay for.
        await add_fact(
            conn, embedding_model, guild_id=GUILD_A, channel_id=1, message_id=1,
            content="a relevant fact",
        )
        settings = _fake_settings(
            similarity_threshold=0.0, grounding_check_model="openrouter/other/vendor"
        )
        interaction = _make_interaction(db=conn, embedding_model=embedding_model, settings=settings)
        check = _grounding(GroundingOutcome.GROUNDED)

        with (
            patch("aura.commands.ask.synthesize_answer", AsyncMock(return_value=None)),
            patch("aura.commands.ask.verify_answer_grounded", check),
        ):
            await _invoke_ask(interaction, "a question")

        check.assert_not_awaited()

    async def test_an_uncited_answer_is_still_checked(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        # /aura-ask shows an answer whether or not the model cited anything, so
        # a zero-citation answer is the most dangerous state there is -- it must
        # be checked against an empty fact set, not waved past for lack of facts.
        check = _grounding(GroundingOutcome.GROUNDED)
        await add_fact(
            conn, embedding_model, guild_id=GUILD_A, channel_id=1, message_id=1,
            content="a relevant fact",
        )
        settings = _fake_settings(
            similarity_threshold=0.0, grounding_check_model="openrouter/other/vendor"
        )
        interaction = _make_interaction(db=conn, embedding_model=embedding_model, settings=settings)
        result = SynthesisResult(answer="I have nothing on that.", used_fact_ids=[], answers_question=False)

        with (
            patch("aura.commands.ask.synthesize_answer", AsyncMock(return_value=result)),
            patch("aura.commands.ask.verify_answer_grounded", check),
        ):
            await _invoke_ask(interaction, "a question")

        check.assert_awaited_once()
        assert check.call_args.kwargs["cited_facts"] == []

    async def test_the_ask_path_uses_the_ask_time_limit(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        from aura.grounding import ASK_GROUNDING_TIMEOUT_SECONDS

        check = _grounding(GroundingOutcome.GROUNDED)
        fact = await add_fact(
            conn, embedding_model, guild_id=GUILD_A, channel_id=1, message_id=1,
            content="a relevant fact",
        )
        settings = _fake_settings(
            similarity_threshold=0.0, grounding_check_model="openrouter/other/vendor"
        )
        interaction = _make_interaction(db=conn, embedding_model=embedding_model, settings=settings)
        result = SynthesisResult(answer="ans", used_fact_ids=[fact.id], answers_question=True)

        with (
            patch("aura.commands.ask.synthesize_answer", AsyncMock(return_value=result)),
            patch("aura.commands.ask.verify_answer_grounded", check),
        ):
            await _invoke_ask(interaction, "a question")

        assert check.call_args.kwargs["timeout_seconds"] == ASK_GROUNDING_TIMEOUT_SECONDS


class TestNoRegressionWithTheCheckAlwaysPassing:
    """Every pre-existing /aura-ask behaviour, re-run with the check configured.

    The check may only ever remove answers that fail it. With it configured and
    forced to "grounded", /aura-ask must behave EXACTLY as it did before this
    feature existed -- these mirror the assertions in the classes above, run
    through the new code path.
    """

    async def test_sources_still_come_only_from_cited_facts(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        cited = await add_fact(
            conn, embedding_model, guild_id=GUILD_A, channel_id=11, message_id=101,
            content="the server was founded in 2020",
        )
        uncited = await add_fact(
            conn, embedding_model, guild_id=GUILD_A, channel_id=22, message_id=202,
            content="the server's founding year is 2020",
        )
        settings = _fake_settings(
            similarity_threshold=0.0, grounding_check_model="openrouter/other/vendor"
        )
        interaction = _make_interaction(db=conn, embedding_model=embedding_model, settings=settings)
        result = SynthesisResult(
            answer="The server was founded in 2020.", used_fact_ids=[cited.id], answers_question=True
        )

        with (
            patch("aura.commands.ask.synthesize_answer", AsyncMock(return_value=result)),
            patch(
                "aura.commands.ask.verify_answer_grounded",
                _grounding(GroundingOutcome.GROUNDED),
            ),
        ):
            await _invoke_ask(interaction, "When was the server founded?")

        _, kwargs = interaction.followup.send.call_args
        assert kwargs.get("ephemeral", False) is False
        embed = kwargs["embed"]
        assert embed.description == "The server was founded in 2020."
        sources = embed.fields[0].value
        assert f"/{cited.channel_id}/{cited.message_id}" in sources
        assert f"/{uncited.channel_id}/{uncited.message_id}" not in sources

    async def test_no_citations_still_omits_the_sources_field(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        await add_fact(
            conn, embedding_model, guild_id=GUILD_A, channel_id=1, message_id=1,
            content="a relevant fact",
        )
        settings = _fake_settings(
            similarity_threshold=0.0, grounding_check_model="openrouter/other/vendor"
        )
        interaction = _make_interaction(db=conn, embedding_model=embedding_model, settings=settings)
        result = SynthesisResult(
            answer="I could not find a clear answer.", used_fact_ids=[], answers_question=False
        )

        with (
            patch("aura.commands.ask.synthesize_answer", AsyncMock(return_value=result)),
            patch(
                "aura.commands.ask.verify_answer_grounded",
                _grounding(GroundingOutcome.GROUNDED),
            ),
        ):
            await _invoke_ask(interaction, "an unrelated question")

        assert len(interaction.followup.send.call_args[1]["embed"].fields) == 0

    async def test_an_overlong_answer_is_still_truncated_to_the_embed_limit(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        fact = await add_fact(
            conn, embedding_model, guild_id=GUILD_A, channel_id=1, message_id=1,
            content="a relevant fact",
        )
        settings = _fake_settings(
            similarity_threshold=0.0, grounding_check_model="openrouter/other/vendor"
        )
        interaction = _make_interaction(db=conn, embedding_model=embedding_model, settings=settings)
        result = SynthesisResult(answer="x" * 5000, used_fact_ids=[fact.id], answers_question=True)

        with (
            patch("aura.commands.ask.synthesize_answer", AsyncMock(return_value=result)),
            patch(
                "aura.commands.ask.verify_answer_grounded",
                _grounding(GroundingOutcome.GROUNDED),
            ),
        ):
            await _invoke_ask(interaction, "a question")

        assert len(interaction.followup.send.call_args[1]["embed"].description) <= 4096

    async def test_the_no_info_path_never_reaches_the_check(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        settings = _fake_settings(grounding_check_model="openrouter/other/vendor")
        interaction = _make_interaction(db=conn, embedding_model=embedding_model, settings=settings)
        check = _grounding(GroundingOutcome.GROUNDED)

        with patch("aura.commands.ask.verify_answer_grounded", check):
            await _invoke_ask(interaction, "any question")

        check.assert_not_awaited()
        args, _ = interaction.followup.send.call_args
        assert "don't have" in args[0].lower() or "no information" in args[0].lower()

    async def test_a_failed_synthesis_still_shows_the_generic_error(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        # Not the grounding message: synthesis failing and the check failing are
        # different situations and must stay distinguishable to the user.
        await add_fact(
            conn, embedding_model, guild_id=GUILD_A, channel_id=1, message_id=1,
            content="a relevant fact",
        )
        settings = _fake_settings(
            similarity_threshold=0.0, grounding_check_model="openrouter/other/vendor"
        )
        interaction = _make_interaction(db=conn, embedding_model=embedding_model, settings=settings)

        with (
            patch("aura.commands.ask.synthesize_answer", AsyncMock(return_value=None)),
            patch(
                "aura.commands.ask.verify_answer_grounded",
                _grounding(GroundingOutcome.CHECK_FAILED),
            ),
        ):
            await _invoke_ask(interaction, "a question")

        args, _ = interaction.followup.send.call_args
        assert "went wrong" in args[0].lower() or "error" in args[0].lower()
