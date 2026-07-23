"""Tests for aura.db.repository: the knowledge model's data-access layer."""
from __future__ import annotations

import asyncio
import sqlite3
from datetime import datetime
from pathlib import Path

import aiosqlite
import pytest

from aura.db.models import Fact, FactLink, FactStatus
from aura.db.repository import (
    CrossGuildLinkError,
    FactAlreadySupersededError,
    FactNotFoundError,
    SelfLinkError,
    create_fact,
    get_active_facts,
    get_linked_facts,
    init_schema,
    link_facts,
    supersede_fact,
)

GUILD_A = 100000000000000001
GUILD_B = 200000000000000002


@pytest.fixture
async def conn():
    connection = await aiosqlite.connect(":memory:")
    await init_schema(connection)
    yield connection
    await connection.close()


async def _make_fact(
    conn: aiosqlite.Connection,
    *,
    guild_id: int = GUILD_A,
    channel_id: int = 1,
    message_id: int = 1,
    content: str = "test fact",
) -> Fact:
    return await create_fact(
        conn, guild_id=guild_id, channel_id=channel_id, message_id=message_id, content=content
    )


class TestModels:
    """Minimal sanity checks -- most model coverage happens implicitly via repository tests."""

    def test_fact_link_validates_with_plausible_data(self) -> None:
        created_at = datetime.fromisoformat("2026-01-01T00:00:00.000000+00:00")
        link = FactLink(fact_a_id=1, fact_b_id=2, created_at=created_at)
        assert link.fact_a_id == 1
        assert link.fact_b_id == 2

    def test_fact_status_members_equal_their_db_string_values(self) -> None:
        assert FactStatus.ACTIVE == "active"
        assert FactStatus.SUPERSEDED == "superseded"


class TestInitSchema:
    async def test_foreign_keys_pragma_is_actually_on(self, conn: aiosqlite.Connection) -> None:
        # Per CLAUDE.md: don't just trust that the pragma was set, query it back.
        async with conn.execute("PRAGMA foreign_keys") as cursor:
            row = await cursor.fetchone()
        assert row is not None
        assert row[0] == 1

    async def test_foreign_key_violation_is_rejected_at_the_schema_level(
        self, conn: aiosqlite.Connection
    ) -> None:
        # Deliberately bypasses link_facts() -- this proves the *schema itself*
        # rejects orphaned references, independent of any application-level
        # check, so any future code path that writes to fact_links directly
        # is still protected.
        with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
            await conn.execute(
                "INSERT INTO fact_links (fact_a_id, fact_b_id, created_at) VALUES (?, ?, ?)",
                (999998, 999999, "2026-01-01T00:00:00.000000+00:00"),
            )

    async def test_journal_mode_is_wal_for_a_real_file_backed_database(
        self, tmp_path: Path
    ) -> None:
        # :memory: databases silently stay in 'memory' journal mode no matter
        # what this pragma requests, so WAL specifically needs a real file.
        db_path = tmp_path / "test.db"
        connection = await aiosqlite.connect(str(db_path))
        try:
            await init_schema(connection)
            async with connection.execute("PRAGMA journal_mode") as cursor:
                row = await cursor.fetchone()
            assert row is not None
            assert row[0].lower() == "wal"
        finally:
            await connection.close()

    async def test_running_init_schema_twice_does_not_raise(
        self, conn: aiosqlite.Connection
    ) -> None:
        await init_schema(conn)


class TestCreateFact:
    async def test_creates_an_active_fact_with_all_fields_set(
        self, conn: aiosqlite.Connection
    ) -> None:
        fact = await create_fact(
            conn, guild_id=GUILD_A, channel_id=42, message_id=99, content="the sky is blue"
        )
        assert fact.guild_id == GUILD_A
        assert fact.channel_id == 42
        assert fact.message_id == 99
        assert fact.content == "the sky is blue"
        assert fact.status == FactStatus.ACTIVE
        assert fact.superseded_by_id is None
        assert fact.superseded_at is None
        assert fact.created_at is not None

    async def test_persisted_and_readable_back(self, conn: aiosqlite.Connection) -> None:
        created = await _make_fact(conn)
        [readback] = await get_active_facts(conn, GUILD_A)
        assert readback.id == created.id
        assert readback.content == created.content

    async def test_two_facts_get_distinct_ids(self, conn: aiosqlite.Connection) -> None:
        a = await _make_fact(conn, content="a")
        b = await _make_fact(conn, content="b")
        assert a.id != b.id

    async def test_realistic_snowflake_sized_ids_round_trip_without_truncation(
        self, conn: aiosqlite.Connection
    ) -> None:
        snowflake_guild = 881234567890123456
        snowflake_channel = 991234567890123456
        snowflake_message = 771234567890123456
        fact = await create_fact(
            conn,
            guild_id=snowflake_guild,
            channel_id=snowflake_channel,
            message_id=snowflake_message,
            content="realistic ids",
        )
        assert fact.guild_id == snowflake_guild
        assert fact.channel_id == snowflake_channel
        assert fact.message_id == snowflake_message

    async def test_unicode_content_round_trips_correctly(self, conn: aiosqlite.Connection) -> None:
        content = "サーバーのルールは 日本語 でも読めます 🎉 مرحبا"
        fact = await create_fact(
            conn, guild_id=GUILD_A, channel_id=1, message_id=1, content=content
        )
        [readback] = await get_active_facts(conn, GUILD_A)
        assert fact.content == content
        assert readback.content == content

    async def test_sql_metacharacters_in_content_are_stored_literally(
        self, conn: aiosqlite.Connection
    ) -> None:
        malicious = "'; DROP TABLE facts; --"
        await create_fact(conn, guild_id=GUILD_A, channel_id=1, message_id=1, content=malicious)
        active = await get_active_facts(conn, GUILD_A)
        assert [f.content for f in active] == [malicious]

    async def test_empty_string_content_is_accepted(self, conn: aiosqlite.Connection) -> None:
        # Nothing in schema.sql or this phase's spec rejects empty content --
        # deciding what counts as a fact worth storing is fact-extraction's
        # job (a later phase), not the data layer's. This documents that as
        # a deliberate choice rather than an untested gap.
        fact = await create_fact(conn, guild_id=GUILD_A, channel_id=1, message_id=1, content="")
        assert fact.content == ""

    async def test_concurrent_creates_both_succeed_with_distinct_ids(
        self, conn: aiosqlite.Connection
    ) -> None:
        results = await asyncio.gather(
            _make_fact(conn, content="concurrent-a"),
            _make_fact(conn, content="concurrent-b"),
        )
        assert results[0].id != results[1].id
        active = await get_active_facts(conn, GUILD_A)
        assert len(active) == 2


class TestSupersedeFact:
    async def test_old_fact_becomes_superseded_pointing_at_new_fact(
        self, conn: aiosqlite.Connection
    ) -> None:
        old = await _make_fact(conn, content="old content")
        new = await supersede_fact(
            conn,
            old_fact_id=old.id,
            guild_id=GUILD_A,
            channel_id=1,
            message_id=2,
            content="new content",
        )
        assert new.status == FactStatus.ACTIVE
        assert new.id != old.id

        active = await get_active_facts(conn, GUILD_A)
        assert [f.id for f in active] == [new.id]

        async with conn.execute(
            "SELECT status, superseded_by_id, superseded_at FROM facts WHERE id = ?", (old.id,)
        ) as cursor:
            row = await cursor.fetchone()
        assert row is not None
        status, superseded_by_id, superseded_at = row
        assert status == FactStatus.SUPERSEDED
        assert superseded_by_id == new.id
        assert superseded_at is not None

    async def test_nonexistent_old_fact_id_raises_and_leaves_nothing_behind(
        self, conn: aiosqlite.Connection
    ) -> None:
        with pytest.raises(FactAlreadySupersededError):
            await supersede_fact(
                conn,
                old_fact_id=999999,
                guild_id=GUILD_A,
                channel_id=1,
                message_id=1,
                content="should not be created",
            )
        assert await get_active_facts(conn, GUILD_A) == []

    async def test_already_superseded_fact_raises_and_leaves_nothing_behind(
        self, conn: aiosqlite.Connection
    ) -> None:
        old = await _make_fact(conn)
        await supersede_fact(
            conn, old_fact_id=old.id, guild_id=GUILD_A, channel_id=1, message_id=2, content="v2"
        )
        with pytest.raises(FactAlreadySupersededError):
            await supersede_fact(
                conn,
                old_fact_id=old.id,
                guild_id=GUILD_A,
                channel_id=1,
                message_id=3,
                content="v3-should-not-exist",
            )
        active = await get_active_facts(conn, GUILD_A)
        assert [f.content for f in active] == ["v2"]

    async def test_wrong_guild_id_fails_the_same_way_as_already_superseded(
        self, conn: aiosqlite.Connection
    ) -> None:
        # The caller shouldn't be able to supersede a fact by guessing its ID
        # and claiming a different guild -- this is a variant of "wrong ID
        # from this guild's point of view" and gets the same error.
        fact = await _make_fact(conn, guild_id=GUILD_A)
        with pytest.raises(FactAlreadySupersededError):
            await supersede_fact(
                conn,
                old_fact_id=fact.id,
                guild_id=GUILD_B,
                channel_id=1,
                message_id=2,
                content="should not be created",
            )
        assert await get_active_facts(conn, GUILD_B) == []
        [still_active] = await get_active_facts(conn, GUILD_A)
        assert still_active.id == fact.id

    async def test_concurrent_supersede_exactly_one_wins(
        self, conn: aiosqlite.Connection
    ) -> None:
        old = await _make_fact(conn, content="original")

        async def attempt(content: str) -> Fact:
            return await supersede_fact(
                conn,
                old_fact_id=old.id,
                guild_id=GUILD_A,
                channel_id=1,
                message_id=2,
                content=content,
            )

        results = await asyncio.gather(
            attempt("racer-A"), attempt("racer-B"), return_exceptions=True
        )

        successes: list[Fact] = []
        failures: list[BaseException] = []
        for result in results:
            if isinstance(result, Fact):
                successes.append(result)
            else:
                failures.append(result)

        assert len(successes) == 1
        assert len(failures) == 1
        assert isinstance(failures[0], FactAlreadySupersededError)

        active = await get_active_facts(conn, GUILD_A)
        assert len(active) == 1
        assert active[0].id == successes[0].id
        assert active[0].content == successes[0].content

    async def test_three_concurrent_supersedes_exactly_one_wins(
        self, conn: aiosqlite.Connection
    ) -> None:
        # Same property, higher contention -- two racers is the minimum bar,
        # this checks the lock doesn't just happen to work for exactly two.
        old = await _make_fact(conn, content="original")

        async def attempt(content: str) -> Fact:
            return await supersede_fact(
                conn,
                old_fact_id=old.id,
                guild_id=GUILD_A,
                channel_id=1,
                message_id=2,
                content=content,
            )

        results = await asyncio.gather(
            attempt("racer-A"),
            attempt("racer-B"),
            attempt("racer-C"),
            return_exceptions=True,
        )
        successes = [r for r in results if isinstance(r, Fact)]
        failures = [r for r in results if isinstance(r, FactAlreadySupersededError)]
        assert len(successes) == 1
        assert len(failures) == 2

        active = await get_active_facts(conn, GUILD_A)
        assert len(active) == 1


class TestLinkFacts:
    async def test_link_then_get_linked_facts_is_symmetric(
        self, conn: aiosqlite.Connection
    ) -> None:
        a = await _make_fact(conn, content="a")
        b = await _make_fact(conn, content="b")
        await link_facts(conn, a.id, b.id)

        linked_from_a = await get_linked_facts(conn, a.id)
        linked_from_b = await get_linked_facts(conn, b.id)
        assert [f.id for f in linked_from_a] == [b.id]
        assert [f.id for f in linked_from_b] == [a.id]

    async def test_argument_order_does_not_create_duplicate_rows(
        self, conn: aiosqlite.Connection
    ) -> None:
        a = await _make_fact(conn, content="a")
        b = await _make_fact(conn, content="b")
        await link_facts(conn, a.id, b.id)
        await link_facts(conn, b.id, a.id)  # reversed order

        async with conn.execute("SELECT COUNT(*) FROM fact_links") as cursor:
            row = await cursor.fetchone()
        assert row is not None
        assert row[0] == 1

    async def test_linking_already_linked_facts_is_a_silent_no_op(
        self, conn: aiosqlite.Connection
    ) -> None:
        a = await _make_fact(conn, content="a")
        b = await _make_fact(conn, content="b")
        await link_facts(conn, a.id, b.id)
        await link_facts(conn, a.id, b.id)  # must not raise
        assert len(await get_linked_facts(conn, a.id)) == 1

    async def test_self_link_is_rejected(self, conn: aiosqlite.Connection) -> None:
        fact = await _make_fact(conn)
        with pytest.raises(SelfLinkError):
            await link_facts(conn, fact.id, fact.id)

    async def test_linking_nonexistent_fact_raises_fact_not_found(
        self, conn: aiosqlite.Connection
    ) -> None:
        real = await _make_fact(conn)
        with pytest.raises(FactNotFoundError):
            await link_facts(conn, real.id, 999999)

    async def test_linking_two_nonexistent_facts_raises_fact_not_found(
        self, conn: aiosqlite.Connection
    ) -> None:
        with pytest.raises(FactNotFoundError):
            await link_facts(conn, 999998, 999999)

    async def test_cross_guild_link_is_rejected_and_nothing_is_inserted(
        self, conn: aiosqlite.Connection
    ) -> None:
        a = await _make_fact(conn, guild_id=GUILD_A)
        b = await _make_fact(conn, guild_id=GUILD_B)
        with pytest.raises(CrossGuildLinkError):
            await link_facts(conn, a.id, b.id)
        async with conn.execute("SELECT COUNT(*) FROM fact_links") as cursor:
            row = await cursor.fetchone()
        assert row is not None
        assert row[0] == 0

    async def test_concurrent_link_calls_in_both_orders_produce_one_row(
        self, conn: aiosqlite.Connection
    ) -> None:
        a = await _make_fact(conn, content="a")
        b = await _make_fact(conn, content="b")
        results = await asyncio.gather(
            link_facts(conn, a.id, b.id),
            link_facts(conn, b.id, a.id),
            return_exceptions=True,
        )
        # Neither ordering is an error case -- both should complete cleanly.
        assert all(r is None for r in results), results
        async with conn.execute("SELECT COUNT(*) FROM fact_links") as cursor:
            row = await cursor.fetchone()
        assert row is not None
        assert row[0] == 1


class TestGuildScoping:
    async def test_get_active_facts_never_returns_another_guilds_facts(
        self, conn: aiosqlite.Connection
    ) -> None:
        await _make_fact(conn, guild_id=GUILD_A, content="guild A fact")
        await _make_fact(conn, guild_id=GUILD_B, content="guild B fact")

        guild_a_facts = await get_active_facts(conn, GUILD_A)
        guild_b_facts = await get_active_facts(conn, GUILD_B)

        assert [f.content for f in guild_a_facts] == ["guild A fact"]
        assert [f.content for f in guild_b_facts] == ["guild B fact"]

    async def test_get_active_facts_for_a_guild_with_no_facts_returns_empty(
        self, conn: aiosqlite.Connection
    ) -> None:
        await _make_fact(conn, guild_id=GUILD_A)
        assert await get_active_facts(conn, guild_id=999999) == []

    async def test_get_linked_facts_cannot_cross_a_guild_boundary(
        self, conn: aiosqlite.Connection
    ) -> None:
        a1 = await _make_fact(conn, guild_id=GUILD_A, content="a1")
        a2 = await _make_fact(conn, guild_id=GUILD_A, content="a2")
        b1 = await _make_fact(conn, guild_id=GUILD_B, content="b1")

        await link_facts(conn, a1.id, a2.id)
        with pytest.raises(CrossGuildLinkError):
            await link_facts(conn, a1.id, b1.id)

        linked = await get_linked_facts(conn, a1.id)
        assert [f.id for f in linked] == [a2.id]
        assert b1.id not in [f.id for f in linked]


class TestGetActiveFacts:
    async def test_superseded_facts_are_excluded(self, conn: aiosqlite.Connection) -> None:
        old = await _make_fact(conn, content="old")
        new = await supersede_fact(
            conn, old_fact_id=old.id, guild_id=GUILD_A, channel_id=1, message_id=2, content="new"
        )
        active = await get_active_facts(conn, GUILD_A)
        assert [f.id for f in active] == [new.id]

    async def test_no_facts_returns_empty_list(self, conn: aiosqlite.Connection) -> None:
        assert await get_active_facts(conn, GUILD_A) == []


class TestGetLinkedFacts:
    async def test_fact_with_no_links_returns_empty_list(self, conn: aiosqlite.Connection) -> None:
        fact = await _make_fact(conn)
        assert await get_linked_facts(conn, fact.id) == []

    async def test_returns_multiple_linked_facts(self, conn: aiosqlite.Connection) -> None:
        hub = await _make_fact(conn, content="hub")
        leaf_1 = await _make_fact(conn, content="leaf1")
        leaf_2 = await _make_fact(conn, content="leaf2")
        await link_facts(conn, hub.id, leaf_1.id)
        await link_facts(conn, hub.id, leaf_2.id)

        linked = await get_linked_facts(conn, hub.id)
        assert {f.id for f in linked} == {leaf_1.id, leaf_2.id}

    async def test_nonexistent_fact_id_returns_empty_list_not_an_error(
        self, conn: aiosqlite.Connection
    ) -> None:
        assert await get_linked_facts(conn, 999999) == []
