"""Tests for aura.db.fact_variants: storage for Multi-Representation Indexing Part 1.

Purely a data-access layer -- no LLM call in this file, matching CLAUDE.md's
"test fact-extraction and matching logic as pure functions/units, independent
of Discord" for the same reason: the generation and audit logic that decides
WHAT gets stored here lives in aura.variants_service and has its own tests.
This file only proves the storage contract: writes land together, reads
come back correctly, and the join against facts.status='active' is the
schema property Part 2 is designed to rely on.
"""
from __future__ import annotations

import aiosqlite
import pytest

from aura.db.fact_variants import (
    get_active_fact_variants,
    get_variants_for_fact,
    store_fact_variants,
)
from aura.db.repository import init_schema, supersede_fact
from aura.facts_service import add_fact

GUILD_A = 100000000000000001
GUILD_B = 200000000000000002

EMBEDDING = b"\x00" * 16
OTHER_EMBEDDING = b"\x01" * 16


@pytest.fixture
async def conn():
    connection = await aiosqlite.connect(":memory:")
    await init_schema(connection)
    yield connection
    await connection.close()


async def _fact_id(conn: aiosqlite.Connection, embedding_model, *, guild_id: int = GUILD_A) -> int:
    fact = await add_fact(
        conn, embedding_model, guild_id=guild_id, channel_id=1, message_id=1, content="the sky is blue"
    )
    return fact.id


class TestStoreFactVariants:
    async def test_stores_every_variant_and_returns_them(
        self, conn: aiosqlite.Connection, embedding_model
    ) -> None:
        fact_id = await _fact_id(conn, embedding_model)
        stored = await store_fact_variants(
            conn,
            fact_id=fact_id,
            contents=["variant one", "variant two"],
            embeddings=[EMBEDDING, OTHER_EMBEDDING],
        )
        assert [v.content for v in stored] == ["variant one", "variant two"]
        assert all(v.fact_id == fact_id for v in stored)
        assert stored[0].id != stored[1].id

    async def test_stored_variants_are_readable_back(
        self, conn: aiosqlite.Connection, embedding_model
    ) -> None:
        fact_id = await _fact_id(conn, embedding_model)
        await store_fact_variants(
            conn, fact_id=fact_id, contents=["x", "y"], embeddings=[EMBEDDING, OTHER_EMBEDDING]
        )
        readback = await get_variants_for_fact(conn, fact_id)
        assert [v.content for v in readback] == ["x", "y"]
        assert readback[0].embedding == EMBEDDING
        assert readback[1].embedding == OTHER_EMBEDDING

    async def test_empty_lists_are_rejected(
        self, conn: aiosqlite.Connection, embedding_model
    ) -> None:
        fact_id = await _fact_id(conn, embedding_model)
        with pytest.raises(ValueError):
            await store_fact_variants(conn, fact_id=fact_id, contents=[], embeddings=[])

    async def test_mismatched_lengths_are_rejected(
        self, conn: aiosqlite.Connection, embedding_model
    ) -> None:
        fact_id = await _fact_id(conn, embedding_model)
        with pytest.raises(ValueError):
            await store_fact_variants(
                conn, fact_id=fact_id, contents=["a", "b"], embeddings=[EMBEDDING]
            )

    async def test_nothing_is_written_when_the_length_check_fails(
        self, conn: aiosqlite.Connection, embedding_model
    ) -> None:
        fact_id = await _fact_id(conn, embedding_model)
        with pytest.raises(ValueError):
            await store_fact_variants(
                conn, fact_id=fact_id, contents=["a", "b"], embeddings=[EMBEDDING]
            )
        assert await get_variants_for_fact(conn, fact_id) == []

    async def test_a_nonexistent_fact_id_is_rejected_by_the_foreign_key(
        self, conn: aiosqlite.Connection
    ) -> None:
        # The PRAGMA that enforces this is per-connection and silently does
        # nothing if a connection ever skips init_schema -- asserted rather
        # than trusted, the same reasoning test_supersession_state.py applies
        # to its own ledger's foreign key.
        with pytest.raises(aiosqlite.IntegrityError):
            await store_fact_variants(
                conn, fact_id=999999, contents=["x"], embeddings=[EMBEDDING]
            )

    async def test_one_fact_can_have_several_variant_rows(
        self, conn: aiosqlite.Connection, embedding_model
    ) -> None:
        # No UNIQUE constraint on fact_id, unlike fact_links' one-row-per-pair
        # shape: a fact legitimately produces several variants.
        fact_id = await _fact_id(conn, embedding_model)
        await store_fact_variants(
            conn, fact_id=fact_id, contents=["a"], embeddings=[EMBEDDING]
        )
        await store_fact_variants(
            conn, fact_id=fact_id, contents=["b", "c"], embeddings=[EMBEDDING, OTHER_EMBEDDING]
        )
        assert len(await get_variants_for_fact(conn, fact_id)) == 3


class TestGetActiveFactVariants:
    """The join property schema.sql's fact_variants comment is built around."""

    async def test_variants_of_an_active_fact_are_returned(
        self, conn: aiosqlite.Connection, embedding_model
    ) -> None:
        fact_id = await _fact_id(conn, embedding_model)
        await store_fact_variants(
            conn, fact_id=fact_id, contents=["a variant"], embeddings=[EMBEDDING]
        )
        active = await get_active_fact_variants(conn, GUILD_A)
        assert [v.content for v in active] == ["a variant"]

    async def test_variants_of_a_superseded_fact_drop_out_automatically(
        self, conn: aiosqlite.Connection, embedding_model
    ) -> None:
        # The whole reason fact_variants has no status column of its own:
        # superseding the fact is the only write needed for its variants to
        # stop surfacing here.
        old_fact_id = await _fact_id(conn, embedding_model)
        await store_fact_variants(
            conn, fact_id=old_fact_id, contents=["stale variant"], embeddings=[EMBEDDING]
        )
        await supersede_fact(
            conn,
            old_fact_id=old_fact_id,
            guild_id=GUILD_A,
            channel_id=1,
            message_id=2,
            content="the sky is grey now",
            embedding=OTHER_EMBEDDING,
        )
        active = await get_active_fact_variants(conn, GUILD_A)
        assert active == []

    async def test_scoped_to_one_guild(
        self, conn: aiosqlite.Connection, embedding_model
    ) -> None:
        fact_a = await _fact_id(conn, embedding_model, guild_id=GUILD_A)
        fact_b = await _fact_id(conn, embedding_model, guild_id=GUILD_B)
        await store_fact_variants(conn, fact_id=fact_a, contents=["a"], embeddings=[EMBEDDING])
        await store_fact_variants(conn, fact_id=fact_b, contents=["b"], embeddings=[EMBEDDING])

        assert [v.content for v in await get_active_fact_variants(conn, GUILD_A)] == ["a"]
        assert [v.content for v in await get_active_fact_variants(conn, GUILD_B)] == ["b"]

    async def test_no_variants_is_an_empty_list_not_an_error(
        self, conn: aiosqlite.Connection, embedding_model
    ) -> None:
        await _fact_id(conn, embedding_model)
        assert await get_active_fact_variants(conn, GUILD_A) == []
