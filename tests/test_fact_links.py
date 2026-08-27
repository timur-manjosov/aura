"""The link surface the link phase added to aura.db.repository, and its adversarial edges.

The original Phase 1b contract (symmetry, normalization, self-link rejection,
guild isolation) is still asserted in test_repository.py, where the repository
is owned. What lives here is everything wiring fact_links to a command and to
retrieval required: unlinking, the active-facts-only rule, the batched
neighbour read, and supersession-chain resolution -- plus a deliberate attempt
to break each of them.

Every test here is a pure data-layer test against a real in-memory SQLite
database: no Discord connection, no LLM, no embedding model, per CLAUDE.md's
testing philosophy.
"""
from __future__ import annotations

import asyncio
import logging

import aiosqlite
import pytest

from aura.db.models import Fact, FactStatus
from aura.db.repository import (
    FactNotActiveError,
    FactNotFoundError,
    SelfLinkError,
    create_fact,
    get_linked_fact_ids,
    get_linked_facts,
    init_schema,
    link_facts,
    resolve_active_successors,
    supersede_fact_with_existing_successor,
    unlink_facts,
)

GUILD_A = 100000000000000001
GUILD_B = 200000000000000002

_FAKE_EMBEDDING = bytes(384 * 4)


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
    content: str = "test fact",
) -> Fact:
    return await create_fact(
        conn,
        guild_id=guild_id,
        channel_id=1,
        message_id=1,
        content=content,
        embedding=_FAKE_EMBEDDING,
    )


async def _supersede(conn: aiosqlite.Connection, old: Fact, new: Fact) -> None:
    await supersede_fact_with_existing_successor(
        conn, old_fact_id=old.id, new_fact_id=new.id, guild_id=old.guild_id
    )


async def _link_row_count(conn: aiosqlite.Connection) -> int:
    async with conn.execute("SELECT COUNT(*) FROM fact_links") as cursor:
        row = await cursor.fetchone()
    assert row is not None
    return row[0]


class TestLinkingRequiresActiveFacts:
    """A link is a claim about what is true together, so both ends must be true."""

    async def test_superseded_fact_cannot_be_linked(self, conn: aiosqlite.Connection) -> None:
        old = await _make_fact(conn, content="old")
        new = await _make_fact(conn, content="new")
        other = await _make_fact(conn, content="other")
        await _supersede(conn, old, new)

        with pytest.raises(FactNotActiveError):
            await link_facts(conn, guild_id=GUILD_A, fact_id_1=old.id, fact_id_2=other.id)
        assert await _link_row_count(conn) == 0

    async def test_the_rejected_end_may_be_either_argument(
        self, conn: aiosqlite.Connection
    ) -> None:
        # Both orders, because the pair is sorted internally -- a check that
        # only looked at the first argument would pass one of these by luck.
        old = await _make_fact(conn, content="old")
        new = await _make_fact(conn, content="new")
        other = await _make_fact(conn, content="other")
        await _supersede(conn, old, new)

        with pytest.raises(FactNotActiveError):
            await link_facts(conn, guild_id=GUILD_A, fact_id_1=other.id, fact_id_2=old.id)
        assert await _link_row_count(conn) == 0

    async def test_both_ends_superseded_is_also_rejected(
        self, conn: aiosqlite.Connection
    ) -> None:
        old_1 = await _make_fact(conn, content="old 1")
        old_2 = await _make_fact(conn, content="old 2")
        new_1 = await _make_fact(conn, content="new 1")
        new_2 = await _make_fact(conn, content="new 2")
        await _supersede(conn, old_1, new_1)
        await _supersede(conn, old_2, new_2)

        with pytest.raises(FactNotActiveError):
            await link_facts(conn, guild_id=GUILD_A, fact_id_1=old_1.id, fact_id_2=old_2.id)
        assert await _link_row_count(conn) == 0

    async def test_a_missing_fact_outranks_an_inactive_one(
        self, conn: aiosqlite.Connection
    ) -> None:
        # Both wrong at once: the nonexistent ID must win, because that is the
        # answer that reveals nothing about which IDs exist.
        old = await _make_fact(conn, content="old")
        new = await _make_fact(conn, content="new")
        await _supersede(conn, old, new)

        with pytest.raises(FactNotFoundError):
            await link_facts(conn, guild_id=GUILD_A, fact_id_1=old.id, fact_id_2=999999)

    async def test_a_link_survives_its_end_being_superseded_afterwards(
        self, conn: aiosqlite.Connection
    ) -> None:
        # The whole point of resolving forward at read time: nothing rewrites
        # or deletes the row when one of its facts is retired.
        a = await _make_fact(conn, content="a")
        b = await _make_fact(conn, content="b")
        successor = await _make_fact(conn, content="b, corrected")
        await link_facts(conn, guild_id=GUILD_A, fact_id_1=a.id, fact_id_2=b.id)

        await _supersede(conn, b, successor)

        assert await _link_row_count(conn) == 1
        assert [f.id for f in await get_linked_facts(conn, guild_id=GUILD_A, fact_id=a.id)] == [
            b.id
        ]


class TestUnlinkFacts:
    async def test_unlink_removes_the_row_and_reports_it(
        self, conn: aiosqlite.Connection
    ) -> None:
        a = await _make_fact(conn, content="a")
        b = await _make_fact(conn, content="b")
        await link_facts(conn, guild_id=GUILD_A, fact_id_1=a.id, fact_id_2=b.id)

        assert await unlink_facts(conn, guild_id=GUILD_A, fact_id_1=a.id, fact_id_2=b.id) is True
        assert await _link_row_count(conn) == 0
        assert await get_linked_facts(conn, guild_id=GUILD_A, fact_id=a.id) == []

    async def test_unlink_is_indifferent_to_argument_order(
        self, conn: aiosqlite.Connection
    ) -> None:
        a = await _make_fact(conn, content="a")
        b = await _make_fact(conn, content="b")
        await link_facts(conn, guild_id=GUILD_A, fact_id_1=a.id, fact_id_2=b.id)

        assert await unlink_facts(conn, guild_id=GUILD_A, fact_id_1=b.id, fact_id_2=a.id) is True
        assert await _link_row_count(conn) == 0

    async def test_unlinking_what_was_never_linked_reports_false_not_an_error(
        self, conn: aiosqlite.Connection
    ) -> None:
        a = await _make_fact(conn, content="a")
        b = await _make_fact(conn, content="b")
        assert await unlink_facts(conn, guild_id=GUILD_A, fact_id_1=a.id, fact_id_2=b.id) is False

    async def test_unlinking_twice_is_idempotent(self, conn: aiosqlite.Connection) -> None:
        a = await _make_fact(conn, content="a")
        b = await _make_fact(conn, content="b")
        await link_facts(conn, guild_id=GUILD_A, fact_id_1=a.id, fact_id_2=b.id)

        assert await unlink_facts(conn, guild_id=GUILD_A, fact_id_1=a.id, fact_id_2=b.id) is True
        assert await unlink_facts(conn, guild_id=GUILD_A, fact_id_1=a.id, fact_id_2=b.id) is False

    async def test_a_superseded_fact_can_still_be_unlinked(
        self, conn: aiosqlite.Connection
    ) -> None:
        # Deliberately permitted, unlike linking: the links most worth cleaning
        # up are exactly the ones whose facts have since moved on.
        a = await _make_fact(conn, content="a")
        b = await _make_fact(conn, content="b")
        successor = await _make_fact(conn, content="b, corrected")
        await link_facts(conn, guild_id=GUILD_A, fact_id_1=a.id, fact_id_2=b.id)
        await _supersede(conn, b, successor)

        assert await unlink_facts(conn, guild_id=GUILD_A, fact_id_1=a.id, fact_id_2=b.id) is True
        assert await _link_row_count(conn) == 0

    async def test_self_unlink_is_rejected(self, conn: aiosqlite.Connection) -> None:
        fact = await _make_fact(conn)
        with pytest.raises(SelfLinkError):
            await unlink_facts(conn, guild_id=GUILD_A, fact_id_1=fact.id, fact_id_2=fact.id)

    async def test_another_guild_cannot_unlink_this_guilds_link(
        self, conn: aiosqlite.Connection
    ) -> None:
        # The attack this guards: guild B's moderator guesses guild A's fact
        # IDs and tears down its links. fact_links has no guild column, so the
        # DELETE has to prove both ends belong to the caller's guild.
        a = await _make_fact(conn, guild_id=GUILD_A, content="a")
        b = await _make_fact(conn, guild_id=GUILD_A, content="b")
        await link_facts(conn, guild_id=GUILD_A, fact_id_1=a.id, fact_id_2=b.id)

        assert await unlink_facts(conn, guild_id=GUILD_B, fact_id_1=a.id, fact_id_2=b.id) is False
        assert await _link_row_count(conn) == 1

    async def test_concurrent_unlinks_report_exactly_one_removal(
        self, conn: aiosqlite.Connection
    ) -> None:
        a = await _make_fact(conn, content="a")
        b = await _make_fact(conn, content="b")
        await link_facts(conn, guild_id=GUILD_A, fact_id_1=a.id, fact_id_2=b.id)

        results = await asyncio.gather(
            unlink_facts(conn, guild_id=GUILD_A, fact_id_1=a.id, fact_id_2=b.id),
            unlink_facts(conn, guild_id=GUILD_A, fact_id_1=b.id, fact_id_2=a.id),
        )
        assert sorted(results, key=repr) == [False, True]
        assert await _link_row_count(conn) == 0

    async def test_link_and_unlink_racing_leave_a_consistent_row_count(
        self, conn: aiosqlite.Connection
    ) -> None:
        # Whichever order the lock grants, the table must never end up with a
        # duplicate row or a half-written one -- only 0 or 1.
        a = await _make_fact(conn, content="a")
        b = await _make_fact(conn, content="b")

        await asyncio.gather(
            link_facts(conn, guild_id=GUILD_A, fact_id_1=a.id, fact_id_2=b.id),
            unlink_facts(conn, guild_id=GUILD_A, fact_id_1=a.id, fact_id_2=b.id),
            link_facts(conn, guild_id=GUILD_A, fact_id_1=b.id, fact_id_2=a.id),
            unlink_facts(conn, guild_id=GUILD_A, fact_id_1=b.id, fact_id_2=a.id),
        )
        assert await _link_row_count(conn) in (0, 1)


class TestGetLinkedFactIds:
    async def test_returns_neighbours_of_several_facts_in_one_call(
        self, conn: aiosqlite.Connection
    ) -> None:
        a = await _make_fact(conn, content="a")
        b = await _make_fact(conn, content="b")
        c = await _make_fact(conn, content="c")
        d = await _make_fact(conn, content="d")
        await link_facts(conn, guild_id=GUILD_A, fact_id_1=a.id, fact_id_2=b.id)
        await link_facts(conn, guild_id=GUILD_A, fact_id_1=c.id, fact_id_2=d.id)

        assert await get_linked_fact_ids(conn, guild_id=GUILD_A, fact_ids=[a.id, c.id]) == {
            a.id: [b.id],
            c.id: [d.id],
        }

    async def test_both_ends_of_one_row_are_reported_when_both_are_requested(
        self, conn: aiosqlite.Connection
    ) -> None:
        a = await _make_fact(conn, content="a")
        b = await _make_fact(conn, content="b")
        await link_facts(conn, guild_id=GUILD_A, fact_id_1=a.id, fact_id_2=b.id)

        assert await get_linked_fact_ids(conn, guild_id=GUILD_A, fact_ids=[a.id, b.id]) == {
            a.id: [b.id],
            b.id: [a.id],
        }

    async def test_neighbours_are_sorted_and_deduplicated(
        self, conn: aiosqlite.Connection
    ) -> None:
        hub = await _make_fact(conn, content="hub")
        leaves = [await _make_fact(conn, content=f"leaf {i}") for i in range(3)]
        for leaf in reversed(leaves):  # inserted out of order on purpose
            await link_facts(conn, guild_id=GUILD_A, fact_id_1=hub.id, fact_id_2=leaf.id)

        result = await get_linked_fact_ids(
            conn, guild_id=GUILD_A, fact_ids=[hub.id, hub.id, hub.id]
        )
        assert result == {hub.id: sorted(leaf.id for leaf in leaves)}

    async def test_a_fact_with_no_links_is_absent_rather_than_empty(
        self, conn: aiosqlite.Connection
    ) -> None:
        fact = await _make_fact(conn)
        assert await get_linked_fact_ids(conn, guild_id=GUILD_A, fact_ids=[fact.id]) == {}

    async def test_no_ids_asks_the_database_nothing(self, conn: aiosqlite.Connection) -> None:
        statements: list[str] = []
        await conn.set_trace_callback(statements.append)
        try:
            assert await get_linked_fact_ids(conn, guild_id=GUILD_A, fact_ids=[]) == {}
        finally:
            await conn.set_trace_callback(None)  # pyright: ignore[reportArgumentType]
        assert statements == []

    async def test_superseded_neighbours_are_still_returned(
        self, conn: aiosqlite.Connection
    ) -> None:
        # This read reports what is linked; deciding what a link MEANS now is
        # resolve_active_successors' job, and filtering here would take that
        # decision away from it.
        a = await _make_fact(conn, content="a")
        b = await _make_fact(conn, content="b")
        successor = await _make_fact(conn, content="b, corrected")
        await link_facts(conn, guild_id=GUILD_A, fact_id_1=a.id, fact_id_2=b.id)
        await _supersede(conn, b, successor)

        assert await get_linked_fact_ids(conn, guild_id=GUILD_A, fact_ids=[a.id]) == {
            a.id: [b.id]
        }

    async def test_another_guilds_links_are_never_returned(
        self, conn: aiosqlite.Connection
    ) -> None:
        a = await _make_fact(conn, guild_id=GUILD_A, content="a")
        b1 = await _make_fact(conn, guild_id=GUILD_B, content="b1")
        b2 = await _make_fact(conn, guild_id=GUILD_B, content="b2")
        await link_facts(conn, guild_id=GUILD_B, fact_id_1=b1.id, fact_id_2=b2.id)

        assert await get_linked_fact_ids(conn, guild_id=GUILD_A, fact_ids=[a.id, b1.id]) == {}

    async def test_a_hand_written_cross_guild_link_is_filtered_out(
        self, conn: aiosqlite.Connection
    ) -> None:
        # link_facts cannot create this row; a hand edit can. The read must
        # refuse it anyway rather than trusting the writer -- otherwise one
        # bad row leaks another server's fact into an answer.
        a = await _make_fact(conn, guild_id=GUILD_A, content="a")
        b = await _make_fact(conn, guild_id=GUILD_B, content="b")
        low, high = sorted((a.id, b.id))
        await conn.execute(
            "INSERT INTO fact_links (fact_a_id, fact_b_id, created_at) VALUES (?, ?, ?)",
            (low, high, "2026-01-01T00:00:00+00:00"),
        )
        await conn.commit()

        assert await get_linked_fact_ids(conn, guild_id=GUILD_A, fact_ids=[a.id]) == {}
        assert await get_linked_facts(conn, guild_id=GUILD_A, fact_id=a.id) == []

    async def test_more_ids_than_one_chunk_can_carry_still_works(
        self, conn: aiosqlite.Connection
    ) -> None:
        # The bound-parameter ceiling this query binds each ID twice against.
        # 600 requested IDs is past _LINK_QUERY_CHUNK_SIZE (250), so this only
        # passes if the chunking is real.
        hub = await _make_fact(conn, content="hub")
        leaf = await _make_fact(conn, content="leaf")
        await link_facts(conn, guild_id=GUILD_A, fact_id_1=hub.id, fact_id_2=leaf.id)

        requested = [hub.id, *range(900000, 900600)]
        assert await get_linked_fact_ids(conn, guild_id=GUILD_A, fact_ids=requested) == {
            hub.id: [leaf.id]
        }


class TestResolveActiveSuccessors:
    async def test_an_active_fact_resolves_to_itself(self, conn: aiosqlite.Connection) -> None:
        fact = await _make_fact(conn, content="a")
        resolved = await resolve_active_successors(conn, guild_id=GUILD_A, fact_ids=[fact.id])
        assert resolved[fact.id].id == fact.id

    async def test_one_supersession_resolves_to_the_successor(
        self, conn: aiosqlite.Connection
    ) -> None:
        old = await _make_fact(conn, content="old")
        new = await _make_fact(conn, content="new")
        await _supersede(conn, old, new)

        resolved = await resolve_active_successors(conn, guild_id=GUILD_A, fact_ids=[old.id])
        assert resolved[old.id].id == new.id
        assert resolved[old.id].status is FactStatus.ACTIVE

    async def test_a_repeatedly_superseded_fact_resolves_to_the_last_link_not_a_middle_one(
        self, conn: aiosqlite.Connection
    ) -> None:
        # The explicit attack from the brief: v1 -> v2 -> v3 -> v4. Resolving
        # to v2 or v3 would cite something that has itself already been
        # retired, which is exactly the "never state something outdated" rule.
        versions = [await _make_fact(conn, content=f"v{i}") for i in range(1, 5)]
        for older, newer in zip(versions, versions[1:]):
            await _supersede(conn, older, newer)

        resolved = await resolve_active_successors(
            conn, guild_id=GUILD_A, fact_ids=[versions[0].id]
        )
        assert resolved[versions[0].id].id == versions[-1].id
        assert resolved[versions[0].id].content == "v4"

    async def test_every_step_of_a_chain_resolves_to_the_same_head(
        self, conn: aiosqlite.Connection
    ) -> None:
        versions = [await _make_fact(conn, content=f"v{i}") for i in range(1, 5)]
        for older, newer in zip(versions, versions[1:]):
            await _supersede(conn, older, newer)

        resolved = await resolve_active_successors(
            conn, guild_id=GUILD_A, fact_ids=[v.id for v in versions]
        )
        assert {origin: fact.id for origin, fact in resolved.items()} == {
            version.id: versions[-1].id for version in versions
        }

    async def test_a_nonexistent_id_is_absent_rather_than_guessed_at(
        self, conn: aiosqlite.Connection
    ) -> None:
        assert await resolve_active_successors(conn, guild_id=GUILD_A, fact_ids=[999999]) == {}

    async def test_another_guilds_fact_is_absent(self, conn: aiosqlite.Connection) -> None:
        fact = await _make_fact(conn, guild_id=GUILD_B, content="b")
        assert await resolve_active_successors(conn, guild_id=GUILD_A, fact_ids=[fact.id]) == {}

    async def test_no_ids_returns_empty_without_touching_the_database(
        self, conn: aiosqlite.Connection
    ) -> None:
        statements: list[str] = []
        await conn.set_trace_callback(statements.append)
        try:
            assert await resolve_active_successors(conn, guild_id=GUILD_A, fact_ids=[]) == {}
        finally:
            await conn.set_trace_callback(None)  # pyright: ignore[reportArgumentType]
        assert statements == []

    async def test_a_chain_running_off_the_end_resolves_to_nothing(
        self, conn: aiosqlite.Connection
    ) -> None:
        # Superseded with no successor recorded: only a hand edit produces
        # this, and the answer must be "nothing", never the dead fact itself.
        fact = await _make_fact(conn, content="orphan")
        await conn.execute(
            "UPDATE facts SET status = 'superseded', superseded_at = ? WHERE id = ?",
            ("2026-01-01T00:00:00+00:00", fact.id),
        )
        await conn.commit()

        assert await resolve_active_successors(conn, guild_id=GUILD_A, fact_ids=[fact.id]) == {}

    async def test_a_chain_pointing_into_another_guild_resolves_to_nothing(
        self, conn: aiosqlite.Connection
    ) -> None:
        fact = await _make_fact(conn, guild_id=GUILD_A, content="a")
        foreign = await _make_fact(conn, guild_id=GUILD_B, content="b")
        await conn.execute(
            "UPDATE facts SET status = 'superseded', superseded_by_id = ?, superseded_at = ? "
            "WHERE id = ?",
            (foreign.id, "2026-01-01T00:00:00+00:00", fact.id),
        )
        await conn.commit()

        assert await resolve_active_successors(conn, guild_id=GUILD_A, fact_ids=[fact.id]) == {}

    async def test_a_cycle_terminates_and_resolves_to_nothing(
        self, conn: aiosqlite.Connection, caplog: pytest.LogCaptureFixture
    ) -> None:
        # Unreachable through supersede_fact_with_existing_successor (it
        # refuses an inactive successor), so this is a hand-edited database.
        # The requirement is simply that it terminates instead of hanging the
        # event loop, and answers "nothing".
        a = await _make_fact(conn, content="a")
        b = await _make_fact(conn, content="b")
        await conn.execute(
            "UPDATE facts SET status = 'superseded', superseded_by_id = ?, superseded_at = ? "
            "WHERE id = ?",
            (b.id, "2026-01-01T00:00:00+00:00", a.id),
        )
        await conn.execute(
            "UPDATE facts SET status = 'superseded', superseded_by_id = ?, superseded_at = ? "
            "WHERE id = ?",
            (a.id, "2026-01-01T00:00:00+00:00", b.id),
        )
        await conn.commit()

        with caplog.at_level(logging.WARNING, logger="aura.db.repository"):
            resolved = await asyncio.wait_for(
                resolve_active_successors(conn, guild_id=GUILD_A, fact_ids=[a.id, b.id]),
                timeout=10,
            )
        assert resolved == {}
        assert "cycles" in caplog.text

    async def test_a_self_referential_supersession_resolves_to_nothing(
        self, conn: aiosqlite.Connection
    ) -> None:
        # The one-node cycle, which a naive "follow until active" loop spins on
        # forever.
        fact = await _make_fact(conn, content="a")
        await conn.execute(
            "UPDATE facts SET status = 'superseded', superseded_by_id = ?, superseded_at = ? "
            "WHERE id = ?",
            (fact.id, "2026-01-01T00:00:00+00:00", fact.id),
        )
        await conn.commit()

        resolved = await asyncio.wait_for(
            resolve_active_successors(conn, guild_id=GUILD_A, fact_ids=[fact.id]), timeout=10
        )
        assert resolved == {}

    async def test_two_ids_resolving_to_the_same_fact_both_report_it(
        self, conn: aiosqlite.Connection
    ) -> None:
        old_1 = await _make_fact(conn, content="old 1")
        old_2 = await _make_fact(conn, content="old 2")
        head = await _make_fact(conn, content="head")
        await _supersede(conn, old_1, head)
        await _supersede(conn, old_2, head)

        resolved = await resolve_active_successors(
            conn, guild_id=GUILD_A, fact_ids=[old_1.id, old_2.id]
        )
        assert {origin: fact.id for origin, fact in resolved.items()} == {
            old_1.id: head.id,
            old_2.id: head.id,
        }

    async def test_a_chain_longer_than_the_hop_limit_resolves_to_nothing_loudly(
        self, conn: aiosqlite.Connection, caplog: pytest.LogCaptureFixture
    ) -> None:
        # Fail closed, and say so: an operator has to be able to see that a
        # link was dropped rather than silently ignored.
        from aura.db.repository import _MAX_SUPERSESSION_HOPS

        versions = [
            await _make_fact(conn, content=f"v{i}") for i in range(_MAX_SUPERSESSION_HOPS + 2)
        ]
        for older, newer in zip(versions, versions[1:]):
            await _supersede(conn, older, newer)

        with caplog.at_level(logging.WARNING, logger="aura.db.repository"):
            resolved = await resolve_active_successors(
                conn, guild_id=GUILD_A, fact_ids=[versions[0].id]
            )
        assert versions[0].id not in resolved
        assert "exceed" in caplog.text
