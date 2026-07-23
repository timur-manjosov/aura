"""Tests for aura.facts_service: the seam between Discord commands and the repository."""
from __future__ import annotations

import aiosqlite
import pytest

from aura.db.models import FactStatus
from aura.db.repository import get_active_facts, init_schema
from aura.facts_service import add_fact

GUILD_A = 100000000000000001


@pytest.fixture
async def conn():
    connection = await aiosqlite.connect(":memory:")
    await init_schema(connection)
    yield connection
    await connection.close()


async def test_add_fact_creates_an_active_fact(conn: aiosqlite.Connection) -> None:
    fact = await add_fact(
        conn, guild_id=GUILD_A, channel_id=42, message_id=99, content="the sky is blue"
    )
    assert fact.guild_id == GUILD_A
    assert fact.channel_id == 42
    assert fact.message_id == 99
    assert fact.content == "the sky is blue"
    assert fact.status == FactStatus.ACTIVE
    assert fact.superseded_by_id is None


async def test_add_fact_is_visible_via_get_active_facts(conn: aiosqlite.Connection) -> None:
    created = await add_fact(conn, guild_id=GUILD_A, channel_id=1, message_id=1, content="x")
    [readback] = await get_active_facts(conn, GUILD_A)
    assert readback.id == created.id
    assert readback.content == "x"
