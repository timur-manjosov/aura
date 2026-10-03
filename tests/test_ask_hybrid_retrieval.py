"""/aura-ask with hybrid retrieval, end to end: what it finds now, and everything around it that must not move.

Real embedding model, real in-memory database, the real command callback;
synthesis and the grounding check are mocked (or armed as tripwires) exactly
as in test_ask_command.py and test_ask_cost_bounds.py. No model is ever called.
"""

from __future__ import annotations

import logging
import re
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import aiosqlite
import discord
import pytest
from fastembed import TextEmbedding

import aura.retrieval.hybrid as hybrid
from ask_retrieval_eval.cases import Metrics, compute_metrics, load_case_file
from ask_retrieval_eval.production_path import QuestionOutcome, run_questions, seed_invented_guild
from aura.billing import PlanGate
from aura.commands.ask import ask_command
from aura.config import Settings
from aura.db.models import Fact
from aura.db.repository import create_fact, init_schema, supersede_fact
from aura.embeddings import EMBEDDING_DTYPE, SYNTHESIS_FACT_LIMIT, embed_text
from aura.grounding import GroundingOutcome
from aura.retrieval.hybrid import HybridRetrievalConfig
from aura.retrieval.index_cache import LexicalIndexCache
from aura.synthesis import SynthesisResult
from tests.conftest import EMBEDDING_MODEL_NAME

GUILD_A = 100012345678901234
ALICE = 111
PUBLIC_SET = load_case_file(
    __import__("pathlib").Path(__file__).parent / "fixtures" / "ask_retrieval_public_set.json"
)

MATH_FACT = "Die Klausuren der Mathematik werden in der letzten Februarwoche geschrieben."
MATH_QUESTIONS = (
    "Matheklausuren",
    "Klausuren Mathe",
    "Wann sind die Mathe-Klausuren?",
    "Mathe Klasuren",
    "Matheklausruen",
    "matheklausuren?",
    "MATHEKLAUSUREN",
)
MENTORIAT_FACTS = (
    "Das Mentoriat für Erstsemester findet jeden Dienstag um 16 Uhr statt.",
    "Ein weiteres Mentoriat gibt es jeden Donnerstag um 10 Uhr im Sprachkanal.",
)
MENTORIAT_QUESTIONS = (
    "Mentoriate",
    "mentoriate",
    "MENTORIATE",
    "Mentoriate?",
    "Mentoriate!!",
    "Was ist mit Mentoriaten?",
    "was ist mit mentoriaten",
    "WAS IST MIT MENTORIATEN?!",
    "Wann finden Mentoriate statt?",
    "wann finden mentoriate statt",
    "Wann finden Mentoriate statt",
    "WANN FINDEN MENTORIATE STATT?",
    "wann finden mentoriate statt??",
)


@pytest.fixture
async def conn():
    connection = await aiosqlite.connect(":memory:")
    await init_schema(connection)
    yield connection
    await connection.close()


@pytest.fixture(autouse=True)
def fresh_shared_cache():
    """Each test starts with an empty process-wide index cache, as a fresh process would."""
    hybrid.SHARED_INDEX_CACHE.clear()
    yield
    hybrid.SHARED_INDEX_CACHE.clear()


def _settings(**overrides: object) -> Settings:
    defaults: dict[str, object] = {
        "discord_token": "fake-token",
        "llm_api_key": "fake-key",
        "synthesis_model": "openrouter/fake/model",
    }
    defaults.update(overrides)
    return Settings(_env_file=None, **defaults)  # type: ignore[arg-type]


def _interaction(
    conn: aiosqlite.Connection,
    model: TextEmbedding,
    *,
    settings: Settings | None = None,
    locale: str = "en-US",
) -> MagicMock:
    interaction = MagicMock(spec=discord.Interaction)
    interaction.locale = locale
    interaction.guild_id = GUILD_A
    interaction.channel_id = 42
    interaction.guild = None
    interaction.user = MagicMock()
    interaction.user.id = ALICE
    interaction.created_at = datetime.now(UTC)
    interaction.client = MagicMock()
    interaction.client.db = conn
    interaction.client.embedding_model = model
    interaction.client.settings = settings or _settings()
    interaction.client.plan_gate = PlanGate.unenforced()
    interaction.response = MagicMock()
    interaction.response.defer = AsyncMock()
    interaction.response.is_done = MagicMock(return_value=True)
    interaction.followup = MagicMock()
    interaction.followup.send = AsyncMock()
    interaction.delete_original_response = AsyncMock()
    return interaction


async def _add(
    conn: aiosqlite.Connection, model: TextEmbedding, content: str, *, message_id: int = 1
) -> Fact:
    """Store a fact with its real embedding, without the background variant step."""
    vector = await embed_text(model, content)
    return await create_fact(
        conn,
        guild_id=GUILD_A,
        channel_id=11,
        message_id=message_id,
        content=content,
        embedding=vector.astype(EMBEDDING_DTYPE, copy=False).tobytes(),
    )


async def _seed_public_facts(conn: aiosqlite.Connection, model: TextEmbedding) -> list[Fact]:
    return [
        await _add(conn, model, fact.content, message_id=1000 + fact.fact_id)
        for fact in PUBLIC_SET.facts
    ]


def _answer(fact_ids: list[int] | None = None) -> AsyncMock:
    return AsyncMock(
        return_value=SynthesisResult(
            answer="An answer.", used_fact_ids=fact_ids or [], answers_question=True
        )
    )


async def _ask(interaction: MagicMock, question: str, synth: AsyncMock | None = None) -> AsyncMock:
    """Ask, with synthesis and grounding mocked; return the synthesis mock."""
    synth = synth or _answer()
    with (
        patch("aura.commands.ask.synthesize_answer", synth),
        patch(
            "aura.commands.ask.verify_answer_grounded",
            AsyncMock(return_value=GroundingOutcome.GROUNDED),
        ),
    ):
        await ask_command.callback(interaction, question)  # type: ignore[call-arg, arg-type]  # pyright: ignore
    return synth


def _synthesized_ids(synth: AsyncMock) -> list[int]:
    assert synth.await_args is not None, "synthesis was not reached"
    return [fact.id for fact in synth.await_args.args[0]]


async def _paid_rows(conn: aiosqlite.Connection) -> int:
    async with conn.execute("SELECT COUNT(*) FROM ask_calls") as cursor:
        row = await cursor.fetchone()
    assert row is not None
    return int(row[0])


@pytest.fixture(scope="session")
async def public_outcomes(embedding_model: TextEmbedding) -> list[QuestionOutcome]:
    """The invented evaluation set, run once through /aura-ask's own retrieval calls."""
    connection = await aiosqlite.connect(":memory:")
    try:
        await init_schema(connection)
        id_map = await seed_invented_guild(
            connection, embedding_model, PUBLIC_SET.facts, guild_id=GUILD_A
        )
        settings = _settings()
        return await run_questions(
            connection,
            embedding_model,
            PUBLIC_SET.cases,
            guild_id=GUILD_A,
            config=HybridRetrievalConfig.from_settings(settings, fact_limit=SYNTHESIS_FACT_LIMIT),
            cache=LexicalIndexCache(),
            fact_id_map=id_map,
        )
    finally:
        await connection.close()


def _metrics(outcomes: list[QuestionOutcome], *, hybrid_selection: bool) -> Metrics:
    return compute_metrics(
        PUBLIC_SET.cases,
        {o.case_id: (o.hybrid if hybrid_selection else o.baseline) for o in outcomes},
    )


class TestThePublicEvaluationSet:
    """The quality diagnosis' invented set (14 facts, 48 questions) through the production path."""

    def test_the_set_is_the_one_the_diagnosis_measured(self) -> None:
        assert len(PUBLIC_SET.facts) == 14
        kinds = [case.kind for case in PUBLIC_SET.cases]
        assert (kinds.count("positive"), kinds.count("negative"), kinds.count("adjacent")) == (
            34,
            10,
            4,
        )

    def test_the_previous_selection_reproduces_the_diagnosis_baseline(
        self, public_outcomes: list[QuestionOutcome]
    ) -> None:
        before = _metrics(public_outcomes, hybrid_selection=False)
        assert (before.hits, before.full, before.negative_false_hits) == (22, 18, 3)
        assert before.precision == pytest.approx(0.924, abs=0.005)

    def test_hybrid_retrieval_reaches_at_least_the_diagnosis_numbers(
        self, public_outcomes: list[QuestionOutcome]
    ) -> None:
        after = _metrics(public_outcomes, hybrid_selection=True)
        assert after.hits >= 31
        assert after.full >= 27
        assert after.negative_false_hits <= 3
        assert after.precision >= 0.90
        assert after.adjacent_other == 0

    def test_no_question_the_previous_selection_answered_is_lost(
        self, public_outcomes: list[QuestionOutcome]
    ) -> None:
        by_id = {case.case_id: case for case in PUBLIC_SET.cases}
        for outcome in public_outcomes:
            relevant = by_id[outcome.case_id].relevant
            if set(outcome.baseline) & relevant:
                assert set(outcome.hybrid) & relevant, outcome.case_id


class TestTimursCasesAsInventedAnalogs:
    """The real failures, rebuilt on invented facts among the public set's 14."""

    @pytest.mark.parametrize("question", MATH_QUESTIONS)
    async def test_an_abbreviated_compound_finds_the_math_exams(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding, question: str
    ) -> None:
        await _seed_public_facts(conn, embedding_model)
        exams = await _add(conn, embedding_model, MATH_FACT, message_id=2001)
        synth = await _ask(_interaction(conn, embedding_model), question)
        assert exams.id in _synthesized_ids(synth)

    @pytest.mark.parametrize("question", MENTORIAT_QUESTIONS)
    async def test_every_spelling_finds_both_mentoriat_facts(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding, question: str
    ) -> None:
        await _seed_public_facts(conn, embedding_model)
        tuesday = await _add(conn, embedding_model, MENTORIAT_FACTS[0], message_id=2002)
        thursday = await _add(conn, embedding_model, MENTORIAT_FACTS[1], message_id=2003)
        synth = await _ask(_interaction(conn, embedding_model), question)
        assert {tuesday.id, thursday.id} <= set(_synthesized_ids(synth))


class TestTheAnswerPath:
    async def test_facts_reach_synthesis_in_ranking_order(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        # Controlled similarities: the shape-alike fact scores higher by
        # embedding, the fact containing the asked word wins on the blend.
        shape_alike = await _add(conn, embedding_model, MENTORIAT_FACTS[0], message_id=1)
        right_topic = await _add(
            conn, embedding_model, "Die Serverwartung ist jeden Donnerstag.", message_id=2
        )
        find = AsyncMock(return_value=[(shape_alike, 0.472), (right_topic, 0.445)])
        with patch("aura.commands.ask.find_similar_facts", find):
            synth = await _ask(_interaction(conn, embedding_model), "Wann ist Wartung?")
        assert _synthesized_ids(synth) == [right_topic.id, shape_alike.id]
        assert find.await_args is not None
        assert find.await_args.kwargs["top_k"] > 10**9  # every active fact is scored

    async def test_a_paid_answer_found_by_words_claims_exactly_one_slot(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        await _seed_public_facts(conn, embedding_model)
        await _add(conn, embedding_model, MENTORIAT_FACTS[0], message_id=2002)
        synth = await _ask(_interaction(conn, embedding_model), "Mentoriate")
        assert synth.await_count == 1
        assert await _paid_rows(conn) == 1

    async def test_the_capped_free_answer_lists_what_the_words_found(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        await _seed_public_facts(conn, embedding_model)
        mentoriat = await _add(conn, embedding_model, MENTORIAT_FACTS[0], message_id=2002)
        interaction = _interaction(conn, embedding_model, settings=_settings(ask_daily_cap_pro=0))
        synth = AsyncMock(side_effect=AssertionError("synthesis reached"))
        await _ask(interaction, "Mentoriate", synth)
        synth.assert_not_awaited()
        _, kwargs = interaction.followup.send.call_args
        assert kwargs["ephemeral"] is True
        assert f"/{mentoriat.channel_id}/{mentoriat.message_id}" in kwargs["embed"].description
        assert await _paid_rows(conn) == 0

    async def test_a_question_containing_a_facts_full_text_retrieves_it_first(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        facts = await _seed_public_facts(conn, embedding_model)
        for fact in facts:
            synth = await _ask(_interaction(conn, embedding_model), fact.content)
            assert _synthesized_ids(synth)[0] == fact.id

    async def test_a_new_fact_is_found_by_the_very_next_question(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        await _seed_public_facts(conn, embedding_model)
        first = _interaction(conn, embedding_model)
        await _ask(first, "Mentoriate")
        assert first.followup.send.call_args.args == (
            "I don't have any information about that yet.",
        )
        added = await _add(conn, embedding_model, MENTORIAT_FACTS[0], message_id=2002)
        synth = await _ask(_interaction(conn, embedding_model), "Mentoriate")
        assert added.id in _synthesized_ids(synth)

    async def test_a_superseded_fact_is_not_found_by_its_words_any_more(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        await _seed_public_facts(conn, embedding_model)
        old = await _add(conn, embedding_model, MENTORIAT_FACTS[0], message_id=2002)
        await _ask(_interaction(conn, embedding_model), "Mentoriate")  # warms the cache
        vector = await embed_text(embedding_model, "Das Tutorium ist jetzt mittwochs.")
        await supersede_fact(
            conn,
            old_fact_id=old.id,
            guild_id=GUILD_A,
            channel_id=11,
            message_id=2004,
            content="Das Tutorium ist jetzt mittwochs.",
            embedding=vector.astype(EMBEDDING_DTYPE, copy=False).tobytes(),
        )
        interaction = _interaction(conn, embedding_model)
        synth = AsyncMock(side_effect=AssertionError("synthesis reached"))
        await _ask(interaction, "Mentoriate", synth)
        synth.assert_not_awaited()


class TestNothingQualifies:
    async def test_possibly_related_facts_are_listed_without_any_model(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        await _seed_public_facts(conn, embedding_model)
        interaction = _interaction(conn, embedding_model, locale="de")
        synth = AsyncMock(side_effect=AssertionError("synthesis reached"))
        with patch("litellm.acompletion", AsyncMock(side_effect=AssertionError("litellm"))):
            # "Owner" is in no fact, "Server" in two: a quarter of the subject.
            await _ask(interaction, "Wer ist der Owner vom Server?", synth)
        synth.assert_not_awaited()
        _, kwargs = interaction.followup.send.call_args
        description = kwargs["embed"].description
        assert description.startswith("Dazu ist nichts Genaues gespeichert. Vielleicht verwandt:")
        assert description.count("• [") <= 3 and "Server" in description
        assert kwargs.get("ephemeral", False) is False  # as visible as the plain reply
        interaction.delete_original_response.assert_not_awaited()
        assert await _paid_rows(conn) == 0

    async def test_a_related_fact_cannot_break_out_of_its_link(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        facts = await _seed_public_facts(conn, embedding_model)
        hostile = await _add(
            conn, embedding_model, "Server [hier](https://evil.example) klicken", message_id=9
        )
        # Every fact just above the floor and below the bar, so only words decide.
        find = AsyncMock(return_value=[(fact, 0.1) for fact in [*facts, hostile]])
        interaction = _interaction(conn, embedding_model)
        with patch("aura.commands.ask.find_similar_facts", find):
            await _ask(
                interaction, "Wer ist der Owner vom Server?", AsyncMock(side_effect=AssertionError)
            )
        _, kwargs = interaction.followup.send.call_args
        description = kwargs["embed"].description
        assert "\\[hier\\]" in description
        # Only an unescaped bracket could end the label and start a link.
        assert not re.search(r"(?<!\\)\]\(https://evil", description)

    async def test_an_unrelated_question_gets_the_plain_reply(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        await _seed_public_facts(conn, embedding_model)
        interaction = _interaction(conn, embedding_model)
        await _ask(
            interaction, "Wie wird das Wetter morgen?", AsyncMock(side_effect=AssertionError)
        )
        assert interaction.followup.send.call_args.args == (
            "I don't have any information about that yet.",
        )
        assert await _paid_rows(conn) == 0


class TestFailSafe:
    async def test_without_word_matching_it_answers_from_similarity_and_warns_once(
        self,
        conn: aiosqlite.Connection,
        embedding_model: TextEmbedding,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        facts = await _seed_public_facts(conn, embedding_model)
        with (
            patch.object(hybrid, "shipped_stopwords", side_effect=RuntimeError("stopwords gone")),
            caplog.at_level(logging.INFO),
        ):
            synth = await _ask(_interaction(conn, embedding_model), "Wann ist die Serverwartung?")
        maintenance = next(fact for fact in facts if "Serverwartung" in fact.content)
        assert maintenance.id in _synthesized_ids(synth)
        warnings = [record for record in caplog.records if record.levelno >= logging.WARNING]
        assert len(warnings) == 1 and "word matching failed" in warnings[0].getMessage()
        assert any("word matching unavailable" in r.getMessage() for r in caplog.records)
        # Nothing of the question or a fact reaches the log on this path either.
        everything = " ".join(record.getMessage() for record in caplog.records)
        assert "Serverwartung" not in everything and "Wann ist" not in everything

    async def test_without_word_matching_a_keyword_gets_the_old_no_info_reply(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        await _seed_public_facts(conn, embedding_model)
        await _add(conn, embedding_model, MENTORIAT_FACTS[0], message_id=2002)
        interaction = _interaction(conn, embedding_model)
        with patch.object(hybrid, "shipped_stopwords", side_effect=RuntimeError):
            await _ask(interaction, "Mentoriate", AsyncMock(side_effect=AssertionError))
        assert interaction.followup.send.call_args.args == (
            "I don't have any information about that yet.",
        )


class TestTheRetrievalLogLine:
    async def test_it_has_counts_only(
        self,
        conn: aiosqlite.Connection,
        embedding_model: TextEmbedding,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        await _seed_public_facts(conn, embedding_model)
        await _add(conn, embedding_model, MENTORIAT_FACTS[0], message_id=2002)
        await _add(conn, embedding_model, MENTORIAT_FACTS[1], message_id=2003)
        with caplog.at_level(logging.DEBUG, logger="aura.commands.ask"):
            await _ask(_interaction(conn, embedding_model), "Was ist mit Mentoriaten?")
        [line] = [r.getMessage() for r in caplog.records if "retrieval in guild" in r.getMessage()]
        assert re.fullmatch(
            r"/aura-ask retrieval in guild 1000…: 2 of 16 active fact\(s\) selected, "
            r"2 by words alone, 0 possibly related",
            line,
        ), line
        everything = " ".join(record.getMessage() for record in caplog.records)
        assert "Mentoriat" not in everything
        assert str(GUILD_A) not in everything


def test_the_suite_uses_the_production_embedding_model() -> None:
    assert _settings().embedding_model == EMBEDDING_MODEL_NAME
