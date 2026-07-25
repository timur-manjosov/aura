"""Tests for aura.facts_service: the seam between Discord commands and the repository."""
from __future__ import annotations

import aiosqlite
import numpy as np
import pytest
from fastembed import TextEmbedding

from aura.db.models import FactStatus
from aura.db.repository import get_active_facts, init_schema
from aura.embeddings import EMBEDDING_DTYPE
from aura.facts_service import add_fact

GUILD_A = 100000000000000001


@pytest.fixture
async def conn():
    connection = await aiosqlite.connect(":memory:")
    await init_schema(connection)
    yield connection
    await connection.close()


async def test_add_fact_creates_an_active_fact(
    conn: aiosqlite.Connection, embedding_model: TextEmbedding
) -> None:
    fact = await add_fact(
        conn, embedding_model, guild_id=GUILD_A, channel_id=42, message_id=99, content="the sky is blue"
    )
    assert fact.guild_id == GUILD_A
    assert fact.channel_id == 42
    assert fact.message_id == 99
    assert fact.content == "the sky is blue"
    assert fact.status == FactStatus.ACTIVE
    assert fact.superseded_by_id is None


async def test_add_fact_is_visible_via_get_active_facts(
    conn: aiosqlite.Connection, embedding_model: TextEmbedding
) -> None:
    created = await add_fact(
        conn, embedding_model, guild_id=GUILD_A, channel_id=1, message_id=1, content="x"
    )
    [readback] = await get_active_facts(conn, GUILD_A)
    assert readback.id == created.id
    assert readback.content == "x"


async def test_add_fact_computes_and_stores_a_real_non_degenerate_embedding(
    conn: aiosqlite.Connection, embedding_model: TextEmbedding
) -> None:
    fact = await add_fact(
        conn,
        embedding_model,
        guild_id=GUILD_A,
        channel_id=1,
        message_id=1,
        content="the server rules were updated last week",
    )
    vector = np.frombuffer(fact.embedding, dtype=EMBEDDING_DTYPE)
    assert vector.shape == (384,)
    # A real embedding of non-empty content is never the zero vector --
    # this is the cheapest possible signal that real inference happened,
    # not a stub or an accidentally-skipped computation.
    assert np.linalg.norm(vector) > 0.0


async def test_add_fact_embedding_survives_the_round_trip_through_storage(
    conn: aiosqlite.Connection, embedding_model: TextEmbedding
) -> None:
    fact = await add_fact(
        conn, embedding_model, guild_id=GUILD_A, channel_id=1, message_id=1, content="hello world"
    )
    [readback] = await get_active_facts(conn, GUILD_A)
    assert readback.embedding == fact.embedding
