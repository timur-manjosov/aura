"""aura.links_service: which linked facts join a synthesis call, and which never do.

This is the read side of CLAUDE.md's fourth knowledge-model component. The
happy path is one assertion; almost everything here is the adversarial half --
the flooding, cycling, cross-guild and stale-chain shapes that decide whether
"a link makes a fact available" is safe or is a hole.

No Discord, no LLM, no embedding model: expansion is pure data-layer logic over
a real in-memory database, and is tested as such.
"""
from __future__ import annotations

import asyncio

import aiosqlite
import pytest

from aura.db.models import Fact
from aura.db.repository import (
    create_fact,
    init_schema,
    link_facts,
    supersede_fact_with_existing_successor,
)
from aura.links_service import LINKED_FACT_LIMIT, expand_with_linked_facts

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
    conn: aiosqlite.Connection, *, guild_id: int = GUILD_A, content: str = "fact"
) -> Fact:
    return await create_fact(
        conn,
        guild_id=guild_id,
        channel_id=1,
        message_id=1,
        content=content,
        embedding=_FAKE_EMBEDDING,
    )


async def _link(conn: aiosqlite.Connection, a: Fact, b: Fact, *, guild_id: int = GUILD_A) -> None:
    await link_facts(conn, guild_id=guild_id, fact_id_1=a.id, fact_id_2=b.id)


async def _supersede(conn: aiosqlite.Connection, old: Fact, new: Fact) -> None:
    await supersede_fact_with_existing_successor(
        conn, old_fact_id=old.id, new_fact_id=new.id, guild_id=old.guild_id
    )


async def _expand(conn: aiosqlite.Connection, facts: list[Fact], **kwargs: int) -> list[int]:
    expanded = await expand_with_linked_facts(
        conn, guild_id=GUILD_A, facts=facts, **kwargs  # type: ignore[arg-type]
    )
    return [fact.id for fact in expanded]


class TestTheThingItExistsFor:
    async def test_a_linked_fact_becomes_a_candidate(self, conn: aiosqlite.Connection) -> None:
        # The whole feature in one assertion: similarity found the tournament
        # date, and the prize -- which shares no vocabulary with it -- comes
        # along because a moderator said it belongs there.
        found = await _make_fact(conn, content="The tournament starts on Saturday.")
        linked = await _make_fact(conn, content="The winner gets a month of Nitro.")
        await _link(conn, found, linked)

        assert await _expand(conn, [found]) == [found.id, linked.id]

    async def test_the_similarity_hits_come_first_and_unchanged(
        self, conn: aiosqlite.Connection
    ) -> None:
        # Ranking order is the caller's decision; expansion only appends.
        first = await _make_fact(conn, content="first")
        second = await _make_fact(conn, content="second")
        linked = await _make_fact(conn, content="linked")
        await _link(conn, second, linked)

        assert await _expand(conn, [second, first]) == [second.id, first.id, linked.id]


class TestSparseAndEmptyInput:
    async def test_no_facts_returns_no_facts(self, conn: aiosqlite.Connection) -> None:
        assert await _expand(conn, []) == []

    async def test_no_facts_asks_the_database_nothing(self, conn: aiosqlite.Connection) -> None:
        statements: list[str] = []
        await conn.set_trace_callback(statements.append)
        try:
            await expand_with_linked_facts(conn, guild_id=GUILD_A, facts=[])
        finally:
            await conn.set_trace_callback(None)  # pyright: ignore[reportArgumentType]
        assert statements == []

    async def test_facts_with_no_links_pass_through_untouched(
        self, conn: aiosqlite.Connection
    ) -> None:
        facts = [await _make_fact(conn, content=f"fact {i}") for i in range(3)]
        assert await _expand(conn, facts) == [fact.id for fact in facts]

    async def test_a_zero_limit_disables_expansion_without_querying(
        self, conn: aiosqlite.Connection
    ) -> None:
        found = await _make_fact(conn, content="found")
        linked = await _make_fact(conn, content="linked")
        await _link(conn, found, linked)

        statements: list[str] = []
        await conn.set_trace_callback(statements.append)
        try:
            expanded = await expand_with_linked_facts(
                conn, guild_id=GUILD_A, facts=[found], limit=0
            )
        finally:
            await conn.set_trace_callback(None)  # pyright: ignore[reportArgumentType]
        assert [fact.id for fact in expanded] == [found.id]
        assert statements == []

    async def test_the_input_list_is_never_mutated(self, conn: aiosqlite.Connection) -> None:
        # The caller still uses its own list afterwards (for the fallback
        # paths in /aura-ask); appending in place would corrupt it.
        found = await _make_fact(conn, content="found")
        linked = await _make_fact(conn, content="linked")
        await _link(conn, found, linked)
        seeds = [found]

        await _expand(conn, seeds)
        assert [fact.id for fact in seeds] == [found.id]


class TestSupersessionResolution:
    async def test_a_link_into_a_superseded_fact_resolves_to_its_successor(
        self, conn: aiosqlite.Connection
    ) -> None:
        found = await _make_fact(conn, content="found")
        old = await _make_fact(conn, content="old")
        new = await _make_fact(conn, content="new")
        await _link(conn, found, old)
        await _supersede(conn, old, new)

        assert await _expand(conn, [found]) == [found.id, new.id]

    async def test_a_repeatedly_superseded_link_resolves_to_the_current_fact(
        self, conn: aiosqlite.Connection
    ) -> None:
        # The brief's explicit attack: v1 was replaced by v2, v2 by v3, v3 by
        # v4. A link drawn at v1 must land on v4 -- never on an intermediate
        # state, which is a fact retrieval is forbidden to cite.
        found = await _make_fact(conn, content="found")
        versions = [await _make_fact(conn, content=f"v{i}") for i in range(1, 5)]
        await _link(conn, found, versions[0])
        for older, newer in zip(versions, versions[1:]):
            await _supersede(conn, older, newer)

        expanded = await expand_with_linked_facts(conn, guild_id=GUILD_A, facts=[found])
        assert [fact.id for fact in expanded] == [found.id, versions[-1].id]
        assert expanded[1].content == "v4"

    async def test_a_link_resolving_onto_a_seed_fact_adds_nothing(
        self, conn: aiosqlite.Connection
    ) -> None:
        # Similarity already found the successor; the link points at its
        # predecessor. Resolution lands on a fact that is already in the list,
        # and it must not appear twice.
        old = await _make_fact(conn, content="old")
        new = await _make_fact(conn, content="new")
        other = await _make_fact(conn, content="other")
        await _link(conn, other, old)
        await _supersede(conn, old, new)

        assert await _expand(conn, [other, new]) == [other.id, new.id]

    async def test_two_links_resolving_onto_one_successor_contribute_it_once(
        self, conn: aiosqlite.Connection
    ) -> None:
        found = await _make_fact(conn, content="found")
        old_1 = await _make_fact(conn, content="old 1")
        old_2 = await _make_fact(conn, content="old 2")
        head = await _make_fact(conn, content="head")
        await _link(conn, found, old_1)
        await _link(conn, found, old_2)
        await _supersede(conn, old_1, head)
        await _supersede(conn, old_2, head)

        assert await _expand(conn, [found]) == [found.id, head.id]

    async def test_a_link_whose_chain_runs_off_the_end_contributes_nothing(
        self, conn: aiosqlite.Connection
    ) -> None:
        found = await _make_fact(conn, content="found")
        dangling = await _make_fact(conn, content="dangling")
        await _link(conn, found, dangling)
        await conn.execute(
            "UPDATE facts SET status = 'superseded', superseded_at = ? WHERE id = ?",
            ("2026-01-01T00:00:00+00:00", dangling.id),
        )
        await conn.commit()

        assert await _expand(conn, [found]) == [found.id]

    async def test_a_cyclic_chain_terminates_and_contributes_nothing(
        self, conn: aiosqlite.Connection
    ) -> None:
        found = await _make_fact(conn, content="found")
        a = await _make_fact(conn, content="a")
        b = await _make_fact(conn, content="b")
        await _link(conn, found, a)
        for fact, successor in ((a, b), (b, a)):
            await conn.execute(
                "UPDATE facts SET status = 'superseded', superseded_by_id = ?, "
                "superseded_at = ? WHERE id = ?",
                (successor.id, "2026-01-01T00:00:00+00:00", fact.id),
            )
        await conn.commit()

        expanded = await asyncio.wait_for(
            expand_with_linked_facts(conn, guild_id=GUILD_A, facts=[found]), timeout=10
        )
        assert [fact.id for fact in expanded] == [found.id]


class TestFloodingIsBounded:
    async def test_a_hub_with_more_links_than_the_limit_is_capped(
        self, conn: aiosqlite.Connection
    ) -> None:
        # The brief's flooding attack in its worst realistic shape: one fact a
        # moderator (or a mistake) linked to far more than a prompt should
        # carry.
        hub = await _make_fact(conn, content="hub")
        leaves = [await _make_fact(conn, content=f"leaf {i}") for i in range(50)]
        for leaf in leaves:
            await _link(conn, hub, leaf)

        expanded = await _expand(conn, [hub])
        assert len(expanded) == 1 + LINKED_FACT_LIMIT
        assert expanded[0] == hub.id

    async def test_the_cap_holds_across_several_seed_facts(
        self, conn: aiosqlite.Connection
    ) -> None:
        # Five seeds each with five links is 25 candidates; the ceiling is on
        # the total added, not per seed.
        seeds = [await _make_fact(conn, content=f"seed {i}") for i in range(5)]
        for seed in seeds:
            for index in range(5):
                await _link(conn, seed, await _make_fact(conn, content=f"leaf {seed.id}-{index}"))

        expanded = await _expand(conn, seeds)
        assert len(expanded) == len(seeds) + LINKED_FACT_LIMIT

    async def test_a_long_chain_contributes_only_its_direct_neighbours(
        self, conn: aiosqlite.Connection
    ) -> None:
        # The brief's chain attack: A-B-C-D-... Expansion is ONE hop, so a
        # seed at A gets B and nothing further, however long the chain runs.
        # This is the structural half of the flood defence -- the cap above is
        # the backstop, this is why it is rarely the thing that binds.
        chain = [await _make_fact(conn, content=f"chain {i}") for i in range(20)]
        for earlier, later in zip(chain, chain[1:]):
            await _link(conn, earlier, later)

        assert await _expand(conn, [chain[0]]) == [chain[0].id, chain[1].id]
        # ...and from the middle of the chain, exactly its two neighbours.
        assert await _expand(conn, [chain[5]]) == [chain[5].id, chain[4].id, chain[6].id]

    async def test_the_cap_counts_added_facts_not_total_facts(
        self, conn: aiosqlite.Connection
    ) -> None:
        # A full similarity result (SYNTHESIS_FACT_LIMIT seeds) must not
        # starve link expansion -- that would make the feature invisible in
        # exactly the guilds that have the most documented.
        seeds = [await _make_fact(conn, content=f"seed {i}") for i in range(5)]
        linked = await _make_fact(conn, content="linked")
        await _link(conn, seeds[0], linked)

        assert await _expand(conn, seeds) == [*[seed.id for seed in seeds], linked.id]


class TestOrderingIsDeterministic:
    async def test_neighbours_of_the_best_ranked_seed_come_first(
        self, conn: aiosqlite.Connection
    ) -> None:
        top = await _make_fact(conn, content="top")
        runner_up = await _make_fact(conn, content="runner up")
        top_leaf = await _make_fact(conn, content="top leaf")
        runner_up_leaf = await _make_fact(conn, content="runner up leaf")
        # Linked in the order that would produce the WRONG answer if the
        # implementation sorted by fact id alone rather than by seed rank.
        await _link(conn, runner_up, runner_up_leaf)
        await _link(conn, top, top_leaf)

        assert await _expand(conn, [top, runner_up]) == [
            top.id,
            runner_up.id,
            top_leaf.id,
            runner_up_leaf.id,
        ]

    async def test_repeated_calls_produce_the_identical_list(
        self, conn: aiosqlite.Connection
    ) -> None:
        # Two identical calls must build two identical prompts, or an odd
        # answer cannot be reproduced -- the same reason find_similar_facts
        # breaks its ties explicitly.
        hub = await _make_fact(conn, content="hub")
        for index in range(20):
            await _link(conn, hub, await _make_fact(conn, content=f"leaf {index}"))

        first = await _expand(conn, [hub])
        second = await _expand(conn, [hub])
        assert first == second

    async def test_a_seed_linked_to_another_seed_is_not_appended_again(
        self, conn: aiosqlite.Connection
    ) -> None:
        # The common case once a guild links a lot: both ends of one link are
        # already citation candidates.
        a = await _make_fact(conn, content="a")
        b = await _make_fact(conn, content="b")
        await _link(conn, a, b)

        assert await _expand(conn, [a, b]) == [a.id, b.id]


class TestGuildIsolation:
    async def test_another_guilds_facts_are_never_added(
        self, conn: aiosqlite.Connection
    ) -> None:
        found = await _make_fact(conn, guild_id=GUILD_A, content="a")
        foreign_1 = await _make_fact(conn, guild_id=GUILD_B, content="b1")
        foreign_2 = await _make_fact(conn, guild_id=GUILD_B, content="b2")
        await _link(conn, foreign_1, foreign_2, guild_id=GUILD_B)

        assert await _expand(conn, [found]) == [found.id]

    async def test_a_hand_written_cross_guild_link_adds_nothing(
        self, conn: aiosqlite.Connection
    ) -> None:
        # link_facts cannot write this row. If something else ever does, the
        # read path must still refuse it: a leaked fact here would be cited,
        # with a permalink, into the wrong server.
        found = await _make_fact(conn, guild_id=GUILD_A, content="a")
        foreign = await _make_fact(conn, guild_id=GUILD_B, content="b")
        low, high = sorted((found.id, foreign.id))
        await conn.execute(
            "INSERT INTO fact_links (fact_a_id, fact_b_id, created_at) VALUES (?, ?, ?)",
            (low, high, "2026-01-01T00:00:00+00:00"),
        )
        await conn.commit()

        assert await _expand(conn, [found]) == [found.id]


class TestQueryCost:
    async def test_expansion_costs_a_bounded_number_of_queries(
        self, conn: aiosqlite.Connection
    ) -> None:
        # CLAUDE.md's batching rule, asserted rather than assumed: expanding
        # five seeds must not be five neighbour lookups. One neighbour query
        # plus one resolution hop is the shape; the assertion is deliberately
        # loose about the exact number and strict about it not scaling with
        # the number of seeds.
        seeds = [await _make_fact(conn, content=f"seed {i}") for i in range(5)]
        for seed in seeds:
            await _link(conn, seed, await _make_fact(conn, content=f"leaf {seed.id}"))

        statements: list[str] = []
        await conn.set_trace_callback(statements.append)
        try:
            await expand_with_linked_facts(conn, guild_id=GUILD_A, facts=seeds)
        finally:
            await conn.set_trace_callback(None)  # pyright: ignore[reportArgumentType]

        selects = [s for s in statements if s.lstrip().upper().startswith("SELECT")]
        assert len(selects) <= 3, selects

    async def test_expansion_issues_no_mutating_statement(
        self, conn: aiosqlite.Connection
    ) -> None:
        # Retrieval reads the knowledge model; it never writes it. Asserted at
        # the SQL level so the claim rests on what the database actually saw.
        found = await _make_fact(conn, content="found")
        linked = await _make_fact(conn, content="linked")
        await _link(conn, found, linked)

        statements: list[str] = []
        await conn.set_trace_callback(statements.append)
        try:
            await expand_with_linked_facts(conn, guild_id=GUILD_A, facts=[found])
        finally:
            await conn.set_trace_callback(None)  # pyright: ignore[reportArgumentType]

        for statement in statements:
            assert statement.lstrip().upper().startswith("SELECT"), statement
