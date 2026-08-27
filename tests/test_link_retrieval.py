"""End to end: a linked fact reaching a real answer, through both answering triggers.

The before/after evidence for CLAUDE.md's fourth knowledge-model component,
run against the REAL embedding model rather than a stub -- the claim being
proven is precisely that similarity search cannot find the second fact, so a
fake model that scores everything alike would prove nothing at all.

The scenario is the one from the phase brief. Two facts a moderator considers
one topic:

    A: "The community tournament starts on Saturday at 18:00 UTC."
    B: "The winner receives one month of Discord Nitro."

and the question "When does the tournament start?". Measured against
paraphrase-multilingual-MiniLM-L12-v2, A scores ~0.77 and B ~0.35 -- so B sits
below the 0.40 direct-query bar and similarity alone never sees it. That gap is
the feature's entire reason to exist, and it is asserted here rather than
assumed, so this file fails loudly if a future embedding model closes it and
makes the fixture meaningless.

Four things are proven, in order: the gap is real; a link closes it; the gate
is NOT widened by a link; and a fact cited only through a link survives all the
way to the grounding check and the source permalinks a reader clicks.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import aiosqlite
import discord
import numpy as np
import pytest
from fastembed import TextEmbedding

from aura.commands.ask import ask_command
from aura.config import Settings
from aura.db.models import Fact
from aura.db.proactive_channel_config import set_channel_enabled
from aura.db.repository import (
    create_fact,
    init_schema,
    link_facts,
    supersede_fact_with_existing_successor,
)
from aura.embeddings import EMBEDDING_DTYPE, embed_text, find_similar_facts
from aura.grounding import GroundingOutcome
from aura.links_service import expand_with_linked_facts
from aura.db.proactive_signals import GateVerdict
from aura.proactive.gate import ProactiveGateConfig, evaluate_message
from aura.proactive.question_detector import QuestionDetector
from aura.proactive.responder import respond_with_synthesis
from aura.synthesis import SynthesisResult

GUILD_A = 100000000000000001
CHANNEL = 555

QUESTION = "When does the tournament start?"
FACT_A = "The community tournament starts on Saturday at 18:00 UTC."
FACT_B = "The winner receives one month of Discord Nitro."

# The direct-query bar (Settings.similarity_threshold's shipped default),
# stated here so the arithmetic below is readable without opening config.py.
SIMILARITY_THRESHOLD = 0.40


@pytest.fixture
async def conn():
    connection = await aiosqlite.connect(":memory:")
    await init_schema(connection)
    yield connection
    await connection.close()


@pytest.fixture(scope="session")
async def detector(embedding_model: TextEmbedding) -> QuestionDetector:
    """Session-scoped for the same reason embedding_model is: the exemplar embedding is the cost."""
    return await QuestionDetector.create(embedding_model)


async def _add(
    conn: aiosqlite.Connection,
    model: TextEmbedding,
    *,
    content: str,
    channel_id: int = 1,
    message_id: int = 1,
) -> Fact:
    """Insert a fact with a real embedding, without the variant-generation side trip.

    Deliberately not aura.facts_service.add_fact: that schedules a background
    LLM call for variant generation, which these tests neither need nor may
    make. The row it writes is identical.
    """
    embedding = await embed_text(model, content)
    return await create_fact(
        conn,
        guild_id=GUILD_A,
        channel_id=channel_id,
        message_id=message_id,
        content=content,
        embedding=embedding.astype(EMBEDDING_DTYPE, copy=False).tobytes(),
    )


async def _seed_scenario(
    conn: aiosqlite.Connection, model: TextEmbedding
) -> tuple[Fact, Fact]:
    fact_a = await _add(conn, model, content=FACT_A, channel_id=10, message_id=100)
    fact_b = await _add(conn, model, content=FACT_B, channel_id=20, message_id=200)
    return fact_a, fact_b


async def _relevant(
    conn: aiosqlite.Connection, model: TextEmbedding, *, threshold: float = SIMILARITY_THRESHOLD
) -> list[Fact]:
    results = await find_similar_facts(conn, model, guild_id=GUILD_A, query=QUESTION)
    return [fact for fact, score in results if score >= threshold]


class TestTheGapIsReal:
    """Without this, everything below would be proving nothing."""

    async def test_similarity_finds_the_date_fact_and_misses_the_prize_fact(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        fact_a, fact_b = await _seed_scenario(conn, embedding_model)

        scores = {
            fact.id: score
            for fact, score in await find_similar_facts(
                conn, embedding_model, guild_id=GUILD_A, query=QUESTION
            )
        }
        assert scores[fact_a.id] >= SIMILARITY_THRESHOLD
        assert scores[fact_b.id] < SIMILARITY_THRESHOLD
        assert [fact.id for fact in await _relevant(conn, embedding_model)] == [fact_a.id]

    async def test_the_two_facts_are_not_paraphrases_of_each_other(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        # If they were, multi-representation indexing would already cover this
        # case and a link would be redundant. They are not: this is the
        # relationship only a human can assert.
        from aura.embeddings import cosine_similarity

        vector_a = await embed_text(embedding_model, FACT_A)
        vector_b = await embed_text(embedding_model, FACT_B)
        assert cosine_similarity(vector_a, vector_b) < SIMILARITY_THRESHOLD


class TestBeforeAndAfter:
    async def test_before_linking_only_one_fact_reaches_synthesis(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        fact_a, _ = await _seed_scenario(conn, embedding_model)
        relevant = await _relevant(conn, embedding_model)

        expanded = await expand_with_linked_facts(conn, guild_id=GUILD_A, facts=relevant)
        assert [fact.id for fact in expanded] == [fact_a.id]

    async def test_after_linking_both_facts_reach_synthesis(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        fact_a, fact_b = await _seed_scenario(conn, embedding_model)
        await link_facts(conn, guild_id=GUILD_A, fact_id_1=fact_a.id, fact_id_2=fact_b.id)

        relevant = await _relevant(conn, embedding_model)
        expanded = await expand_with_linked_facts(conn, guild_id=GUILD_A, facts=relevant)

        assert [fact.id for fact in expanded] == [fact_a.id, fact_b.id]
        assert [fact.content for fact in expanded] == [FACT_A, FACT_B]

    async def test_unlinking_returns_retrieval_to_its_previous_behaviour(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        from aura.db.repository import unlink_facts

        fact_a, fact_b = await _seed_scenario(conn, embedding_model)
        await link_facts(conn, guild_id=GUILD_A, fact_id_1=fact_a.id, fact_id_2=fact_b.id)
        await unlink_facts(conn, guild_id=GUILD_A, fact_id_1=fact_a.id, fact_id_2=fact_b.id)

        relevant = await _relevant(conn, embedding_model)
        expanded = await expand_with_linked_facts(conn, guild_id=GUILD_A, facts=relevant)
        assert [fact.id for fact in expanded] == [fact_a.id]

    async def test_a_link_pointing_at_a_replaced_fact_delivers_the_replacement(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        # The same before/after, one supersession later: the prize changed, a
        # moderator ran /aura-supersede, and nobody touched the link. The
        # answer must carry the CURRENT prize, not the retired one.
        fact_a, fact_b = await _seed_scenario(conn, embedding_model)
        await link_facts(conn, guild_id=GUILD_A, fact_id_1=fact_a.id, fact_id_2=fact_b.id)
        replacement = await _add(
            conn,
            embedding_model,
            content="The winner receives three months of Discord Nitro.",
            channel_id=20,
            message_id=201,
        )
        await supersede_fact_with_existing_successor(
            conn, old_fact_id=fact_b.id, new_fact_id=replacement.id, guild_id=GUILD_A
        )

        relevant = await _relevant(conn, embedding_model)
        expanded = await expand_with_linked_facts(conn, guild_id=GUILD_A, facts=relevant)

        assert [fact.id for fact in expanded] == [fact_a.id, replacement.id]
        assert FACT_B not in [fact.content for fact in expanded]


class TestTheGateIsNotWidened:
    """A link may widen an answer Aura was already giving. It may never authorize one."""

    async def test_a_link_does_not_make_an_unmatched_message_eligible(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding, detector: QuestionDetector
    ) -> None:
        # Stage 2 scores the message against facts by similarity alone. Here
        # the message matches only the prize fact's TOPIC weakly and nothing
        # strongly, so it must fail Stage 2 whether or not a link exists --
        # the eligibility decision, and with it the escalation budget, stays a
        # pure similarity question.
        fact_a, fact_b = await _seed_scenario(conn, embedding_model)
        await link_facts(conn, guild_id=GUILD_A, fact_id_1=fact_a.id, fact_id_2=fact_b.id)

        config = ProactiveGateConfig(
            question_threshold=-2.0 + 1e-9,  # Stage 1 always passes, so Stage 2 decides
            similarity_threshold=0.95,  # nothing here can clear this
            cooldown_seconds=0.0,
            daily_cap=20,
        )
        from datetime import datetime, timezone

        trail = await evaluate_message(
            conn,
            embedding_model,
            detector,
            guild_id=GUILD_A,
            channel_id=CHANNEL,
            message_id=777,
            content="what is the weather like today?",
            config=config,
            now=datetime.now(timezone.utc),
        )
        assert trail.verdict is not GateVerdict.ELIGIBLE
        assert trail.stage2_passed is False

    def test_the_gate_module_cannot_reach_link_expansion(self) -> None:
        # Structural, not behavioural: the gate must not even import the
        # expansion, so the property above cannot quietly regress later.
        import ast
        from pathlib import Path

        import aura.proactive.gate as gate_module

        tree = ast.parse(Path(gate_module.__file__).read_text(encoding="utf-8"))
        imported = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in node.names
        }
        assert "expand_with_linked_facts" not in imported
        assert "get_linked_fact_ids" not in imported


def _settings(**overrides: object) -> Settings:
    defaults: dict[str, object] = {
        "discord_token": "fake-token",
        "llm_api_key": "fake-key",
        "synthesis_model": "openrouter/fake/model",
        "similarity_threshold": SIMILARITY_THRESHOLD,
    }
    defaults.update(overrides)
    return Settings(_env_file=None, **defaults)  # type: ignore[arg-type]


def _ask_interaction(conn: aiosqlite.Connection, model: TextEmbedding) -> MagicMock:
    from datetime import datetime, timezone

    interaction = MagicMock(spec=discord.Interaction)
    interaction.locale = "en-US"
    interaction.guild_id = GUILD_A
    interaction.channel_id = CHANNEL
    interaction.created_at = datetime.now(timezone.utc)
    interaction.guild = None  # channel names fall back to IDs; irrelevant here
    interaction.user = MagicMock()
    interaction.user.id = 1
    interaction.client = MagicMock()
    interaction.client.db = conn
    interaction.client.embedding_model = model
    interaction.client.settings = _settings()
    interaction.response = MagicMock()
    interaction.response.defer = AsyncMock()
    interaction.followup = MagicMock()
    interaction.followup.send = AsyncMock()
    return interaction


class TestLinkedFactsSurviveTheWholeAskPath:
    async def test_synthesis_receives_both_facts(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        fact_a, fact_b = await _seed_scenario(conn, embedding_model)
        await link_facts(conn, guild_id=GUILD_A, fact_id_1=fact_a.id, fact_id_2=fact_b.id)
        interaction = _ask_interaction(conn, embedding_model)
        synth = AsyncMock(
            return_value=SynthesisResult(
                answer="Saturday, 18:00 UTC. The winner gets a month of Nitro.",
                used_fact_ids=[fact_a.id, fact_b.id],
                answers_question=True,
            )
        )

        with (
            patch("aura.commands.ask.synthesize_answer", synth),
            patch(
                "aura.commands.ask.verify_answer_grounded",
                AsyncMock(return_value=GroundingOutcome.GROUNDED),
            ),
        ):
            await ask_command.callback(interaction, QUESTION)  # pyright: ignore[reportCallIssue]

        assert synth.await_args is not None
        assert [fact.id for fact in synth.await_args.args[0]] == [fact_a.id, fact_b.id]

    async def test_a_fact_cited_only_through_a_link_reaches_the_grounding_check(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        # The verification the brief asks for: grounding's logic is unchanged
        # -- it still checks the answer against exactly the facts the answer
        # claims to have used -- and this proves a link-discovered citation is
        # inside that set rather than invisible to it. A missing fact here
        # would make the check reject every answer that used a link, which
        # would be a silent, total regression of Trigger 2.
        fact_a, fact_b = await _seed_scenario(conn, embedding_model)
        await link_facts(conn, guild_id=GUILD_A, fact_id_1=fact_a.id, fact_id_2=fact_b.id)
        interaction = _ask_interaction(conn, embedding_model)
        grounding = AsyncMock(return_value=GroundingOutcome.GROUNDED)

        with (
            patch(
                "aura.commands.ask.synthesize_answer",
                AsyncMock(
                    return_value=SynthesisResult(
                        answer="The winner gets a month of Nitro.",
                        used_fact_ids=[fact_b.id],  # ONLY the linked fact
                        answers_question=True,
                    )
                ),
            ),
            patch("aura.commands.ask.verify_answer_grounded", grounding),
        ):
            await ask_command.callback(interaction, QUESTION)  # pyright: ignore[reportCallIssue]

        assert grounding.await_args is not None
        cited = grounding.await_args.kwargs["cited_facts"]
        assert [fact.id for fact in cited] == [fact_b.id]

    async def test_a_fact_cited_only_through_a_link_is_shown_as_a_source(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        # A citation a reader cannot click is not a citation.
        fact_a, fact_b = await _seed_scenario(conn, embedding_model)
        await link_facts(conn, guild_id=GUILD_A, fact_id_1=fact_a.id, fact_id_2=fact_b.id)
        interaction = _ask_interaction(conn, embedding_model)

        with (
            patch(
                "aura.commands.ask.synthesize_answer",
                AsyncMock(
                    return_value=SynthesisResult(
                        answer="The winner gets a month of Nitro.",
                        used_fact_ids=[fact_b.id],
                        answers_question=True,
                    )
                ),
            ),
            patch(
                "aura.commands.ask.verify_answer_grounded",
                AsyncMock(return_value=GroundingOutcome.GROUNDED),
            ),
        ):
            await ask_command.callback(interaction, QUESTION)  # pyright: ignore[reportCallIssue]

        _, kwargs = interaction.followup.send.call_args
        [sources_field] = kwargs["embed"].fields
        assert f"/{fact_b.channel_id}/{fact_b.message_id}" in sources_field.value


class TestLinkedFactsSurviveTheWholeProactivePath:
    def _message(self) -> MagicMock:
        message = MagicMock(spec=discord.Message)
        message.content = QUESTION
        message.guild = MagicMock()
        message.guild.id = GUILD_A
        message.guild.preferred_locale = "en-US"
        message.guild.get_channel = MagicMock(return_value=None)
        message.channel = MagicMock()
        message.channel.id = CHANNEL
        message.channel.send = AsyncMock()
        message.id = 777
        from datetime import datetime, timezone

        message.created_at = datetime.now(timezone.utc)
        return message

    def _settings(self) -> Settings:
        # The shipped proactive threshold is currently 0.20, a deliberately
        # loose single-member testing value that would admit the prize fact on
        # similarity alone and make this test prove nothing. Pinned to the
        # direct-query bar instead, so what is being measured here is the
        # link, not the threshold of the day.
        return _settings(
            proactive_model=None, proactive_similarity_threshold=SIMILARITY_THRESHOLD
        )

    async def test_synthesis_receives_the_linked_fact_too(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        fact_a, fact_b = await _seed_scenario(conn, embedding_model)
        await link_facts(conn, guild_id=GUILD_A, fact_id_1=fact_a.id, fact_id_2=fact_b.id)
        await set_channel_enabled(
            conn, guild_id=GUILD_A, channel_id=CHANNEL, enabled=True, updated_by_id=1
        )
        synth = AsyncMock(
            return_value=SynthesisResult(
                answer="Saturday, 18:00 UTC.", used_fact_ids=[fact_a.id], answers_question=True
            )
        )

        with (
            patch("aura.proactive.responder.synthesize_answer", synth),
            patch(
                "aura.proactive.responder.verify_answer_grounded",
                AsyncMock(return_value=GroundingOutcome.GROUNDED),
            ),
        ):
            outcome = await respond_with_synthesis(
                self._message(), db=conn, model=embedding_model, settings=self._settings()
            )

        assert outcome.posted is True
        assert synth.await_args is not None
        assert [fact.id for fact in synth.await_args.args[0]] == [fact_a.id, fact_b.id]

    async def test_a_link_discovered_citation_reaches_grounding_and_the_posted_sources(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        fact_a, fact_b = await _seed_scenario(conn, embedding_model)
        await link_facts(conn, guild_id=GUILD_A, fact_id_1=fact_a.id, fact_id_2=fact_b.id)
        await set_channel_enabled(
            conn, guild_id=GUILD_A, channel_id=CHANNEL, enabled=True, updated_by_id=1
        )
        message = self._message()
        grounding = AsyncMock(return_value=GroundingOutcome.GROUNDED)

        with (
            patch(
                "aura.proactive.responder.synthesize_answer",
                AsyncMock(
                    return_value=SynthesisResult(
                        answer="Saturday at 18:00 UTC, and the winner gets a month of Nitro.",
                        used_fact_ids=[fact_a.id, fact_b.id],
                        answers_question=True,
                    )
                ),
            ),
            patch("aura.proactive.responder.verify_answer_grounded", grounding),
        ):
            outcome = await respond_with_synthesis(
                message, db=conn, model=embedding_model, settings=self._settings()
            )

        assert outcome.posted is True
        assert grounding.await_args is not None
        assert [fact.id for fact in grounding.await_args.kwargs["cited_facts"]] == [
            fact_a.id,
            fact_b.id,
        ]

        _, kwargs = message.channel.send.call_args
        [sources_field] = kwargs["embed"].fields
        assert f"/{fact_a.channel_id}/{fact_a.message_id}" in sources_field.value
        assert f"/{fact_b.channel_id}/{fact_b.message_id}" in sources_field.value


class TestVariantsAndLinksDoNotInterfere:
    async def test_a_variant_found_fact_still_brings_its_links(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        # Multi-representation indexing and links are different mechanisms on
        # the same read path; a fact found through a VARIANT must expand its
        # links exactly like one found through its canonical sentence.
        from aura.db.fact_variants import store_fact_variants

        fact_a, fact_b = await _seed_scenario(conn, embedding_model)
        await link_facts(conn, guild_id=GUILD_A, fact_id_1=fact_a.id, fact_id_2=fact_b.id)

        # A paraphrase that matches the question far better than the canonical
        # sentence would on its own -- the variant is why the fact is found.
        variant_text = "When does the tournament start? It begins Saturday at 18:00 UTC."
        variant_embedding = await embed_text(embedding_model, variant_text)
        await store_fact_variants(
            conn,
            fact_id=fact_a.id,
            contents=[variant_text],
            embeddings=[variant_embedding.astype(EMBEDDING_DTYPE, copy=False).tobytes()],
        )

        relevant = await _relevant(conn, embedding_model)
        expanded = await expand_with_linked_facts(conn, guild_id=GUILD_A, facts=relevant)
        assert [fact.id for fact in expanded] == [fact_a.id, fact_b.id]

    async def test_a_link_never_changes_the_text_of_a_cited_fact(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        # A variant is only ever the reason a fact was found, never what gets
        # displayed -- and the same must hold for a link.
        fact_a, fact_b = await _seed_scenario(conn, embedding_model)
        await link_facts(conn, guild_id=GUILD_A, fact_id_1=fact_a.id, fact_id_2=fact_b.id)

        relevant = await _relevant(conn, embedding_model)
        expanded = await expand_with_linked_facts(conn, guild_id=GUILD_A, facts=relevant)
        assert [fact.content for fact in expanded] == [FACT_A, FACT_B]
        assert all(
            isinstance(fact.embedding, bytes)
            and np.frombuffer(fact.embedding, dtype=EMBEDDING_DTYPE).size > 0
            for fact in expanded
        )
