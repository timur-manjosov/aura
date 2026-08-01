"""Tests for aura.embeddings: semantic embedding and similarity search.

Uses the real embedding model throughout (the session-scoped embedding_model
fixture in conftest.py), not a mock: several of the tests below specifically
verify real semantic behavior -- that paraphrases score higher than
unrelated text, that concurrent inference doesn't corrupt results -- which a
mock can't meaningfully exercise. cosine_similarity's own unit tests are the
one exception, since they only need plain numpy arrays.
"""
from __future__ import annotations

import asyncio

import aiosqlite
import numpy as np
import pytest
from fastembed import TextEmbedding

from aura.db.fact_variants import get_active_fact_variants, store_fact_variants
from aura.db.repository import init_schema, supersede_fact
from aura.embeddings import (
    EMBEDDING_DTYPE,
    best_similarity,
    cosine_similarity,
    embed_text,
    embed_texts,
    find_similar_facts,
    group_variants_by_fact,
)
from aura.facts_service import add_fact

GUILD_A = 100000000000000001
GUILD_B = 200000000000000002

_MODEL_DIM = 384


@pytest.fixture
async def conn():
    connection = await aiosqlite.connect(":memory:")
    await init_schema(connection)
    yield connection
    await connection.close()


class TestCosineSimilarity:
    def test_identical_vectors_score_one(self) -> None:
        v = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        assert cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors_score_zero(self) -> None:
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0], dtype=np.float32)
        assert cosine_similarity(a, b) == pytest.approx(0.0)

    def test_opposite_vectors_score_negative_one(self) -> None:
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        assert cosine_similarity(a, -a) == pytest.approx(-1.0)

    def test_non_unit_length_vectors_are_normalized_before_scoring(self) -> None:
        # Same direction, wildly different magnitude -- cosine similarity
        # must ignore magnitude entirely.
        a = np.array([1.0, 0.0], dtype=np.float32)
        scaled = a * 100.0
        assert cosine_similarity(a, scaled) == pytest.approx(1.0)

    def test_zero_vector_on_one_side_returns_zero_not_nan_or_raise(self) -> None:
        zero = np.zeros(4, dtype=np.float32)
        other = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        result = cosine_similarity(zero, other)
        assert result == 0.0
        assert not np.isnan(result)

    def test_zero_vector_on_both_sides_returns_zero_not_nan_or_raise(self) -> None:
        zero = np.zeros(4, dtype=np.float32)
        result = cosine_similarity(zero, zero)
        assert result == 0.0
        assert not np.isnan(result)


class TestEmbedText:
    async def test_returns_float32_vector_of_the_models_dimension(
        self, embedding_model: TextEmbedding
    ) -> None:
        vector = await embed_text(embedding_model, "the server rules were updated last week")
        assert vector.dtype == EMBEDDING_DTYPE
        assert vector.shape == (_MODEL_DIM,)

    async def test_single_character_does_not_crash(self, embedding_model: TextEmbedding) -> None:
        vector = await embed_text(embedding_model, "a")
        assert vector.shape == (_MODEL_DIM,)

    async def test_single_emoji_does_not_crash(self, embedding_model: TextEmbedding) -> None:
        vector = await embed_text(embedding_model, "🎉")
        assert vector.shape == (_MODEL_DIM,)

    async def test_empty_string_does_not_crash(self, embedding_model: TextEmbedding) -> None:
        vector = await embed_text(embedding_model, "")
        assert vector.shape == (_MODEL_DIM,)
        assert vector.dtype == EMBEDDING_DTYPE

    async def test_whitespace_only_text_does_not_crash(self, embedding_model: TextEmbedding) -> None:
        vector = await embed_text(embedding_model, "   \n\t  ")
        assert vector.shape == (_MODEL_DIM,)

    async def test_text_far_beyond_the_models_token_limit_does_not_crash(
        self, embedding_model: TextEmbedding
    ) -> None:
        # The model truncates at 512 tokens; this is roughly 2x that in
        # words, so truncation is guaranteed to actually kick in.
        long_text = ("server rules and community guidelines " * 200).strip()
        vector = await embed_text(embedding_model, long_text)
        assert vector.shape == (_MODEL_DIM,)

    async def test_byte_round_trip_is_bit_identical(self, embedding_model: TextEmbedding) -> None:
        original = await embed_text(embedding_model, "the server was founded in 2020")
        serialized = original.tobytes()
        deserialized = np.frombuffer(serialized, dtype=EMBEDDING_DTYPE)

        assert np.array_equal(original, deserialized)
        assert deserialized.tobytes() == serialized  # bit-identical, not just "close enough"

    async def test_concurrent_embedding_calls_match_a_sequential_baseline(
        self, embedding_model: TextEmbedding
    ) -> None:
        # embed_text's asyncio.to_thread wrapping means the same model
        # object can get called from multiple threads at once -- this
        # verifies the installed fastembed version's inference is actually
        # safe under that, rather than assuming it.
        texts = [f"distinct fact number {i} about topic {i}" for i in range(20)]

        sequential = [await embed_text(embedding_model, text) for text in texts]
        concurrent = await asyncio.gather(*(embed_text(embedding_model, text) for text in texts))

        for seq_vec, conc_vec in zip(sequential, concurrent):
            assert np.array_equal(seq_vec, conc_vec)


class TestEmbedTexts:
    async def test_returns_one_vector_per_text_in_input_order(
        self, embedding_model: TextEmbedding
    ) -> None:
        texts = ["the server was founded in 2020", "we sell candles", "où sont les règles ?"]

        batched = await embed_texts(embedding_model, texts)

        assert len(batched) == len(texts)
        for text, vector in zip(texts, batched):
            individually = await embed_text(embedding_model, text)
            assert np.array_equal(vector, individually)

    async def test_vectors_carry_the_fixed_dtype(self, embedding_model: TextEmbedding) -> None:
        [vector] = await embed_texts(embedding_model, ["one text"])
        assert vector.dtype == EMBEDDING_DTYPE

    async def test_an_empty_batch_is_rejected_rather_than_silently_returning_nothing(
        self, embedding_model: TextEmbedding
    ) -> None:
        with pytest.raises(ValueError):
            await embed_texts(embedding_model, [])

    async def test_a_batch_of_hostile_strings_does_not_crash(
        self, embedding_model: TextEmbedding
    ) -> None:
        texts = ["", "   ", "🎉", "a\x00b", "Hello 世界 Привет مرحبا", "x" * 20000]

        batched = await embed_texts(embedding_model, texts)

        assert len(batched) == len(texts)
        assert all(vector.shape == (_MODEL_DIM,) for vector in batched)

    async def test_a_batch_is_not_reordered_by_duplicate_content(
        self, embedding_model: TextEmbedding
    ) -> None:
        texts = ["same", "different", "same"]
        batched = await embed_texts(embedding_model, texts)
        assert np.array_equal(batched[0], batched[2])
        assert not np.array_equal(batched[0], batched[1])


class TestFindSimilarFacts:
    async def test_zero_facts_in_guild_returns_empty_list_not_an_error(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        results = await find_similar_facts(conn, embedding_model, guild_id=GUILD_A, query="anything")
        assert results == []

    async def test_top_k_larger_than_available_facts_does_not_error(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        await add_fact(
            conn, embedding_model, guild_id=GUILD_A, channel_id=1, message_id=1, content="only fact"
        )
        results = await find_similar_facts(
            conn, embedding_model, guild_id=GUILD_A, query="only fact", top_k=50
        )
        assert len(results) == 1

    async def test_top_k_limits_the_number_of_results(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        for i in range(8):
            await add_fact(
                conn, embedding_model, guild_id=GUILD_A, channel_id=1, message_id=i,
                content=f"distinct fact number {i}",
            )
        results = await find_similar_facts(
            conn, embedding_model, guild_id=GUILD_A, query="distinct fact", top_k=3
        )
        assert len(results) == 3

    async def test_results_are_sorted_descending_by_score(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        await add_fact(
            conn, embedding_model, guild_id=GUILD_A, channel_id=1, message_id=1,
            content="the server was founded in 2020",
        )
        await add_fact(
            conn, embedding_model, guild_id=GUILD_A, channel_id=1, message_id=2,
            content="we sell homemade candles on weekends",
        )
        await add_fact(
            conn, embedding_model, guild_id=GUILD_A, channel_id=1, message_id=3,
            content="the server's founding year is 2020",
        )

        results = await find_similar_facts(
            conn, embedding_model, guild_id=GUILD_A, query="when was the server founded?", top_k=10
        )
        scores = [score for _, score in results]
        assert scores == sorted(scores, reverse=True)

    async def test_guild_isolation_even_with_identical_content(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        # The strongest possible adversarial version of guild isolation:
        # byte-identical content in both guilds, so if isolation were only
        # approximate (e.g. a forgotten guild_id filter that happens not to
        # matter for dissimilar content), this is where it would surface.
        identical_content = "the server rules were updated last week"
        await add_fact(
            conn, embedding_model, guild_id=GUILD_A, channel_id=1, message_id=1,
            content=identical_content,
        )
        await add_fact(
            conn, embedding_model, guild_id=GUILD_B, channel_id=1, message_id=1,
            content=identical_content,
        )

        results = await find_similar_facts(
            conn, embedding_model, guild_id=GUILD_A, query=identical_content, top_k=10
        )
        assert len(results) == 1
        fact, score = results[0]
        assert fact.guild_id == GUILD_A
        assert score == pytest.approx(1.0, abs=1e-4)

    async def test_paraphrases_score_higher_than_unrelated_content(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        # The one test in this phase that's really testing whether the
        # feature works, not just whether the code runs.
        paraphrase = "Die Serverregeln wurden letzte Woche aktualisiert."  # German paraphrase
        unrelated = "We're getting a new pizza place downtown next month."

        await add_fact(
            conn, embedding_model, guild_id=GUILD_A, channel_id=1, message_id=1, content=paraphrase
        )
        await add_fact(
            conn, embedding_model, guild_id=GUILD_A, channel_id=1, message_id=2, content=unrelated
        )

        results = await find_similar_facts(
            conn, embedding_model, guild_id=GUILD_A,
            query="The server rules were updated last week.", top_k=10,
        )
        scores_by_content = {fact.content: score for fact, score in results}

        paraphrase_score = scores_by_content[paraphrase]
        unrelated_score = scores_by_content[unrelated]
        assert paraphrase_score > unrelated_score
        assert paraphrase_score - unrelated_score > 0.2  # a real gap, not marginal

    async def test_concurrent_add_fact_calls_produce_correct_uncorrupted_embeddings(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        contents = [f"distinct fact number {i} about server topic {i}" for i in range(10)]

        await asyncio.gather(
            *(
                add_fact(
                    conn, embedding_model, guild_id=GUILD_A, channel_id=1, message_id=i, content=c
                )
                for i, c in enumerate(contents)
            )
        )

        # Querying with a fact's own exact content must rank that fact
        # first: the model is deterministic, so the query embedding and the
        # stored embedding for the matching fact should be identical
        # vectors (score ~1.0). If concurrent to_thread calls had corrupted
        # or cross-assigned an embedding to the wrong row, this is where it
        # would show up as the wrong fact (or a suspiciously low score)
        # coming back first.
        for content in contents:
            results = await find_similar_facts(
                conn, embedding_model, guild_id=GUILD_A, query=content, top_k=1
            )
            assert len(results) == 1
            fact, score = results[0]
            assert fact.content == content
            assert score == pytest.approx(1.0, abs=1e-4)


async def _store_variant(
    conn: aiosqlite.Connection, embedding_model: TextEmbedding, *, fact_id: int, content: str
) -> None:
    """Store one real-embedded variant directly, bypassing the LLM generation/audit call.

    aura.variants_service is tested on its own (with litellm mocked); what
    Part 2 needs here is a variant that already passed that pipeline and is
    sitting in fact_variants with a real embedding, so this writes straight to
    storage the way generate_variants_for_fact would have after a successful
    generation and audit.
    """
    [embedding] = await embed_texts(embedding_model, [content])
    await store_fact_variants(
        conn,
        fact_id=fact_id,
        contents=[content],
        embeddings=[embedding.astype(EMBEDDING_DTYPE, copy=False).tobytes()],
    )


class TestBestSimilarity:
    """Unit tests for the Part 2 max-over-variants helper, isolated from the DB."""

    async def test_with_no_variants_argument_falls_back_to_the_canonical_vector_alone(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        fact = await add_fact(
            conn, embedding_model, guild_id=GUILD_A, channel_id=1, message_id=1,
            content="the sky is blue",
        )
        query_embedding = await embed_text(embedding_model, "the sky is blue")
        assert best_similarity(query_embedding, fact) == pytest.approx(
            cosine_similarity(
                query_embedding, np.frombuffer(fact.embedding, dtype=EMBEDDING_DTYPE)
            )
        )

    async def test_a_variant_that_matches_better_than_canonical_wins_the_max(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        fact = await add_fact(
            conn, embedding_model, guild_id=GUILD_A, channel_id=1, message_id=1,
            content="the sky is blue",
        )
        await _store_variant(
            conn, embedding_model, fact_id=fact.id, content="the exact query text"
        )
        variants_by_fact = group_variants_by_fact(
            await get_active_fact_variants(conn, GUILD_A)
        )
        query_embedding = await embed_text(embedding_model, "the exact query text")

        canonical_only = best_similarity(query_embedding, fact)
        with_variant = best_similarity(query_embedding, fact, variants_by_fact)

        assert with_variant > canonical_only
        assert with_variant == pytest.approx(1.0, abs=1e-4)

    def test_non_finite_canonical_vector_is_skipped_not_allowed_to_win(self) -> None:
        from datetime import datetime, timezone

        from aura.db.models import Fact, FactStatus

        nan_bytes = np.full(4, np.nan, dtype=EMBEDDING_DTYPE).tobytes()
        fact = Fact(
            id=1, guild_id=GUILD_A, channel_id=1, message_id=1, content="x",
            embedding=nan_bytes, status=FactStatus.ACTIVE, superseded_by_id=None,
            created_at=datetime.now(timezone.utc), superseded_at=None,
        )
        query_embedding = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        result = best_similarity(query_embedding, fact)
        assert np.isnan(result)

    def test_a_non_finite_variant_does_not_poison_a_finite_canonical_score(self) -> None:
        from datetime import datetime, timezone

        from aura.db.fact_variants import FactVariant
        from aura.db.models import Fact, FactStatus

        good = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        nan_bytes = np.full(4, np.nan, dtype=EMBEDDING_DTYPE).tobytes()
        fact = Fact(
            id=1, guild_id=GUILD_A, channel_id=1, message_id=1, content="x",
            embedding=good.tobytes(), status=FactStatus.ACTIVE, superseded_by_id=None,
            created_at=datetime.now(timezone.utc), superseded_at=None,
        )
        variant = FactVariant(
            id=1, fact_id=1, content="bad variant", embedding=nan_bytes,
            created_at=datetime.now(timezone.utc),
        )
        result = best_similarity(good, fact, {1: [variant]})
        assert result == pytest.approx(1.0)


class TestFindSimilarFactsWithVariants:
    """Multi-Representation Indexing Part 2: find_similar_facts via a fact's variants.

    Real before/after comparisons throughout, not just an assertion about the
    end state -- the same content is scored twice, once before any variant is
    stored and once after, so a genuine ranking flip is what proves the wiring
    rather than a claim about it.
    """

    # A clause-rich exception fact, the shape reports/variant-indexing-part1.txt
    # Section 4 found variants help most on (a value, a channel and a day
    # qualifier -- plenty of structural room to reword), unlike a bare,
    # minimal statement.
    _CANONICAL = (
        "The #trading channel enforces a 5MB upload limit, except on "
        "Saturdays when the limit is lifted."
    )
    _VARIANT = (
        "Every day but Saturday, files posted in #trading may not exceed "
        "5MB; on Saturdays that cap does not apply."
    )
    _DISTRACTOR = "Saturday is when #trading sees the most messages and the most new members."
    _QUERY = "Can I post bigger files in trading on Saturday without the size cap?"

    async def test_a_query_unreachable_via_canonical_wording_is_found_via_its_variant(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        target = await add_fact(
            conn, embedding_model, guild_id=GUILD_A, channel_id=1, message_id=1,
            content=self._CANONICAL,
        )
        distractor = await add_fact(
            conn, embedding_model, guild_id=GUILD_A, channel_id=1, message_id=2,
            content=self._DISTRACTOR,
        )

        # BEFORE: only the canonical vector exists. The topically-adjacent
        # distractor (shares "Saturday" and "#trading") outranks the fact that
        # actually answers the question, because the canonical wording and the
        # query wording share little vocabulary.
        before = await find_similar_facts(
            conn, embedding_model, guild_id=GUILD_A, query=self._QUERY, top_k=10
        )
        before_scores = {fact.id: score for fact, score in before}
        assert before[0][0].id == distractor.id
        assert before_scores[target.id] < before_scores[distractor.id]

        # AFTER: store the audited variant that happens to phrase the same
        # fact closer to how the query phrases its question.
        await _store_variant(conn, embedding_model, fact_id=target.id, content=self._VARIANT)

        after = await find_similar_facts(
            conn, embedding_model, guild_id=GUILD_A, query=self._QUERY, top_k=10
        )
        after_scores = {fact.id: score for fact, score in after}

        # The fact that actually answers the question now wins, and its score
        # strictly improved -- this is not noise, it is the variant's vector.
        assert after[0][0].id == target.id
        assert after_scores[target.id] > before_scores[target.id]
        assert after_scores[target.id] > after_scores[distractor.id]

        # CLAUDE.md's Fact component: what gets shown is always the one
        # distilled canonical sentence, never the variant that won the match.
        assert after[0][0].content == self._CANONICAL

    async def test_a_retired_facts_variant_does_not_surface_it(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        old = await add_fact(
            conn, embedding_model, guild_id=GUILD_A, channel_id=1, message_id=1,
            content=self._CANONICAL,
        )
        await _store_variant(conn, embedding_model, fact_id=old.id, content=self._VARIANT)

        replacement_embedding = await embed_text(
            embedding_model, "The #trading channel now has no upload limit on any day."
        )
        await supersede_fact(
            conn,
            old_fact_id=old.id,
            guild_id=GUILD_A,
            channel_id=1,
            message_id=2,
            content="The #trading channel now has no upload limit on any day.",
            embedding=replacement_embedding.astype(EMBEDDING_DTYPE, copy=False).tobytes(),
        )

        # Querying with wording that matches the RETIRED fact's variant almost
        # exactly must not surface the retired fact -- the join against
        # facts.status='active' (aura.db.fact_variants.get_active_fact_variants)
        # has to actually exclude it here, not just be schema-possible.
        results = await find_similar_facts(
            conn, embedding_model, guild_id=GUILD_A, query=self._VARIANT, top_k=10
        )
        assert all(fact.id != old.id for fact, _ in results)

    async def test_top_k_ranking_with_variants_is_stable_and_deterministic(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        # Regression check: with no variants stored anywhere, behaviour is
        # byte-for-byte what it was before Part 2 -- this is the same
        # assertion test_results_are_sorted_descending_by_score above already
        # makes, repeated here explicitly under the new code path so a
        # regression in the shared helper's zero-variant fallback fails loudly
        # in this class rather than only in the older one.
        for i in range(5):
            await add_fact(
                conn, embedding_model, guild_id=GUILD_A, channel_id=1, message_id=i,
                content=f"distinct fact number {i} about server topic {i}",
            )
        first = await find_similar_facts(
            conn, embedding_model, guild_id=GUILD_A, query="distinct fact", top_k=10
        )
        second = await find_similar_facts(
            conn, embedding_model, guild_id=GUILD_A, query="distinct fact", top_k=10
        )
        assert [f.id for f, _ in first] == [f.id for f, _ in second]
        scores = [score for _, score in first]
        assert scores == sorted(scores, reverse=True)
