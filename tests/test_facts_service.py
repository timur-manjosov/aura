"""Tests for aura.facts_service: the seam between Discord commands and the repository."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import aiosqlite
import numpy as np
import pytest
from fastembed import TextEmbedding

from aura.db.models import FactStatus
from aura.db.pending_facts import FactCategory, confirm_pending_fact, stage_pending_fact
from aura.db.repository import get_active_facts, init_schema
from aura.embeddings import EMBEDDING_DTYPE
from aura.facts_service import add_fact, confirm_fact

GUILD_A = 100000000000000001
CHANNEL = 500000000000000005
MODERATOR = 111


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


class TestVariantGenerationTrigger:
    """add_fact and confirm_fact are the two -- and only two -- places a fact becomes active.

    Multi-Representation Indexing Part 1's design requires exactly one shared
    hook covering both origins (manual and automatically-confirmed) rather
    than two independently-maintained ones. These tests prove the hook fires
    from both, using a patched generate_variants_for_fact so no real
    background task (and no real LLM call) is needed to prove the wiring.
    """

    async def test_add_fact_schedules_variant_generation(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        with patch(
            "aura.facts_service.generate_variants_for_fact", AsyncMock(return_value=[])
        ) as generate:
            fact = await add_fact(
                conn, embedding_model, guild_id=GUILD_A, channel_id=1, message_id=1, content="x"
            )
            # The scheduled task runs on a future loop iteration, not
            # synchronously inside add_fact -- give the loop one turn.
            await asyncio.sleep(0)

        generate.assert_awaited_once_with(conn, embedding_model, fact)

    async def test_confirm_fact_schedules_variant_generation(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        staged = await stage_pending_fact(
            conn,
            guild_id=GUILD_A,
            channel_id=CHANNEL,
            message_id=1,
            content="a distilled candidate",
            embedding=b"\x00" * 16,
            category=FactCategory.ANNOUNCEMENT,
        )
        assert staged is not None

        with patch(
            "aura.facts_service.generate_variants_for_fact", AsyncMock(return_value=[])
        ) as generate:
            fact = await confirm_fact(
                conn,
                embedding_model,
                guild_id=GUILD_A,
                pending_id=staged.id,
                resolved_by_id=MODERATOR,
            )
            await asyncio.sleep(0)

        generate.assert_awaited_once_with(conn, embedding_model, fact)

    async def test_confirm_fact_produces_the_exact_same_fact_as_confirm_pending_fact_would(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        # confirm_fact is a thin wrapper, not a reimplementation: it must not
        # change what confirming a candidate actually produces.
        staged = await stage_pending_fact(
            conn,
            guild_id=GUILD_A,
            channel_id=CHANNEL,
            message_id=1,
            content="a distilled candidate",
            embedding=b"\x00" * 16,
            category=FactCategory.RULE,
        )
        assert staged is not None

        with patch("aura.facts_service.generate_variants_for_fact", AsyncMock(return_value=[])):
            fact = await confirm_fact(
                conn,
                embedding_model,
                guild_id=GUILD_A,
                pending_id=staged.id,
                resolved_by_id=MODERATOR,
            )
            await asyncio.sleep(0)

        assert fact.content == staged.content
        assert fact.embedding == staged.embedding
        assert fact.status == FactStatus.ACTIVE
        [readback] = await get_active_facts(conn, GUILD_A)
        assert readback.id == fact.id

    async def test_confirm_fact_propagates_already_resolved_without_scheduling_anything(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        from aura.db.pending_facts import PendingFactAlreadyResolvedError

        staged = await stage_pending_fact(
            conn,
            guild_id=GUILD_A,
            channel_id=CHANNEL,
            message_id=1,
            content="a distilled candidate",
            embedding=b"\x00" * 16,
            category=FactCategory.RULE,
        )
        assert staged is not None
        await confirm_pending_fact(
            conn, guild_id=GUILD_A, pending_id=staged.id, resolved_by_id=MODERATOR
        )

        with patch(
            "aura.facts_service.generate_variants_for_fact", AsyncMock(return_value=[])
        ) as generate:
            with pytest.raises(PendingFactAlreadyResolvedError):
                await confirm_fact(
                    conn,
                    embedding_model,
                    guild_id=GUILD_A,
                    pending_id=staged.id,
                    resolved_by_id=MODERATOR,
                )
            await asyncio.sleep(0)

        generate.assert_not_awaited()

    async def test_scheduled_variant_generation_does_not_delay_the_callers_return(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        # The whole reason this is fire-and-forget rather than awaited inline:
        # add_fact must return as soon as the fact itself is committed, not
        # after variant generation (two LLM round trips) also completes.
        blocked = asyncio.Event()

        async def _slow_generation(*_args, **_kwargs):
            await blocked.wait()
            return []

        with patch(
            "aura.facts_service.generate_variants_for_fact", AsyncMock(side_effect=_slow_generation)
        ):
            fact = await asyncio.wait_for(
                add_fact(
                    conn, embedding_model, guild_id=GUILD_A, channel_id=1, message_id=1, content="x"
                ),
                timeout=1.0,
            )
            assert fact.content == "x"
            blocked.set()  # let the still-pending background task finish cleanly
            await asyncio.sleep(0)
