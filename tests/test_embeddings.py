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

from aura.db.repository import init_schema
from aura.embeddings import (
    EMBEDDING_DTYPE,
    cosine_similarity,
    embed_text,
    embed_texts,
    find_similar_facts,
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
