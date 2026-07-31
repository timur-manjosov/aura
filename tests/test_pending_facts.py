"""Tests for aura.db.pending_facts: the staging table and its resolution races.

The interesting half of this module is the race, and it is the reason the phase
brief calls it out by name: two moderators pressing Confirm on the same
candidate, or one confirming while another discards, must produce exactly one
outcome and at most one active fact. Those tests use real asyncio.gather over a
real in-memory database, never sequential calls dressed up as concurrency --
the same standard tests/test_repository.py holds supersession to.

A real database throughout, never a live gateway connection, per CLAUDE.md's
testing philosophy.
"""
from __future__ import annotations

import asyncio

import aiosqlite
import pytest

from aura.db.models import FactStatus
from aura.db.pending_facts import (
    FactCategory,
    PendingFactAlreadyResolvedError,
    PendingFactNotFoundError,
    PendingFactStatus,
    SupersessionRelationship,
    confirm_pending_fact,
    count_pending_facts,
    discard_pending_fact,
    get_pending_fact,
    get_pending_facts,
    record_relationship_judgement,
    stage_pending_fact,
    verify_pending_facts_schema,
)
from aura.db.repository import create_fact, get_active_facts, init_schema

GUILD_A = 100000000000000001
GUILD_B = 200000000000000002
CHANNEL = 500000000000000005
MOD_A = 11111
MOD_B = 22222

EMBEDDING = b"\x00" * 16


@pytest.fixture
async def conn():
    connection = await aiosqlite.connect(":memory:")
    await init_schema(connection)
    yield connection
    await connection.close()


async def _stage(
    conn: aiosqlite.Connection,
    *,
    guild_id: int = GUILD_A,
    message_id: int = 1,
    content: str = "The server is down for maintenance today at 14:00 UTC.",
    category: FactCategory = FactCategory.STATUS_CHANGE,
    similar_fact_id: int | None = None,
    similar_fact_score: float | None = None,
):
    return await stage_pending_fact(
        conn,
        guild_id=guild_id,
        channel_id=CHANNEL,
        message_id=message_id,
        content=content,
        embedding=EMBEDDING,
        category=category,
        similar_fact_id=similar_fact_id,
        similar_fact_score=similar_fact_score,
    )


class TestStaging:
    async def test_a_staged_candidate_is_pending_and_readable(
        self, conn: aiosqlite.Connection
    ) -> None:
        staged = await _stage(conn)
        assert staged is not None
        assert staged.status is PendingFactStatus.PENDING
        assert staged.confirmed_fact_id is None

        read_back = await get_pending_fact(conn, guild_id=GUILD_A, pending_id=staged.id)
        assert read_back is not None
        assert read_back.content == staged.content
        assert read_back.category is FactCategory.STATUS_CHANGE

    async def test_staging_never_creates_a_fact(self, conn: aiosqlite.Connection) -> None:
        # The load-bearing property of the whole staging design: an automatic
        # path may propose, never publish.
        await _stage(conn)
        assert await get_active_facts(conn, GUILD_A) == []

    async def test_restaging_the_same_sentence_from_the_same_message_is_a_no_op(
        self, conn: aiosqlite.Connection
    ) -> None:
        # What makes a re-distilled batch safe after a crash: the same messages
        # produce the same sentences, and those must land as the same
        # candidates rather than as duplicates a moderator rejects one by one.
        first = await _stage(conn)
        second = await _stage(conn)

        assert first is not None
        assert second is None
        assert await count_pending_facts(conn, guild_id=GUILD_A) == 1

    async def test_one_message_may_assert_two_different_things(
        self, conn: aiosqlite.Connection
    ) -> None:
        # The UNIQUE constraint is (channel, message, content) and not
        # (channel, message) precisely so this stays possible.
        await _stage(conn, content="Maintenance runs today at 14:00 UTC.")
        await _stage(conn, content="The tournament starts on Saturday at 18:00.")
        assert await count_pending_facts(conn, guild_id=GUILD_A) == 2

    async def test_the_same_sentence_from_a_different_message_is_staged_separately(
        self, conn: aiosqlite.Connection
    ) -> None:
        await _stage(conn, message_id=1)
        await _stage(conn, message_id=2)
        assert await count_pending_facts(conn, guild_id=GUILD_A) == 2

    async def test_an_unknown_category_is_rejected_by_the_schema(
        self, conn: aiosqlite.Connection
    ) -> None:
        # The CHECK constraint is the backstop behind the distiller's own
        # enum validation: a category that is not in the closed vocabulary must
        # never reach a row, whichever layer let it through.
        with pytest.raises(aiosqlite.IntegrityError):
            await conn.execute(
                """
                INSERT INTO pending_facts
                    (guild_id, channel_id, message_id, content, embedding, category,
                     status, created_at)
                VALUES (?, ?, ?, ?, ?, 'gossip', 'pending', '2026-07-30T00:00:00.000000+00:00')
                """,
                (GUILD_A, CHANNEL, 99, "something", EMBEDDING),
            )


class TestGuildIsolation:
    async def test_another_guilds_candidate_cannot_be_read(
        self, conn: aiosqlite.Connection
    ) -> None:
        staged = await _stage(conn, guild_id=GUILD_B)
        assert staged is not None
        assert await get_pending_fact(conn, guild_id=GUILD_A, pending_id=staged.id) is None

    async def test_another_guilds_candidate_cannot_be_confirmed(
        self, conn: aiosqlite.Connection
    ) -> None:
        staged = await _stage(conn, guild_id=GUILD_B)
        assert staged is not None
        with pytest.raises(PendingFactNotFoundError):
            await confirm_pending_fact(
                conn, guild_id=GUILD_A, pending_id=staged.id, resolved_by_id=MOD_A
            )
        # And nothing leaked into the wrong guild, or the right one.
        assert await get_active_facts(conn, GUILD_A) == []
        assert await get_active_facts(conn, GUILD_B) == []

    async def test_another_guilds_candidate_cannot_be_discarded(
        self, conn: aiosqlite.Connection
    ) -> None:
        staged = await _stage(conn, guild_id=GUILD_B)
        assert staged is not None
        with pytest.raises(PendingFactNotFoundError):
            await discard_pending_fact(
                conn, guild_id=GUILD_A, pending_id=staged.id, resolved_by_id=MOD_A
            )
        still_pending = await get_pending_fact(conn, guild_id=GUILD_B, pending_id=staged.id)
        assert still_pending is not None
        assert still_pending.status is PendingFactStatus.PENDING

    async def test_the_review_queue_only_shows_its_own_guild(
        self, conn: aiosqlite.Connection
    ) -> None:
        await _stage(conn, guild_id=GUILD_A, message_id=1)
        await _stage(conn, guild_id=GUILD_B, message_id=2)
        assert len(await get_pending_facts(conn, guild_id=GUILD_A, limit=10)) == 1
        assert await count_pending_facts(conn, guild_id=GUILD_B) == 1


class TestConfirmation:
    async def test_confirming_creates_exactly_one_active_fact(
        self, conn: aiosqlite.Connection
    ) -> None:
        staged = await _stage(conn)
        assert staged is not None

        fact = await confirm_pending_fact(
            conn, guild_id=GUILD_A, pending_id=staged.id, resolved_by_id=MOD_A
        )

        active = await get_active_facts(conn, GUILD_A)
        assert len(active) == 1
        assert active[0].id == fact.id
        assert active[0].content == staged.content
        assert active[0].status is FactStatus.ACTIVE
        # The confirmed fact carries the SOURCE message's coordinates, not the
        # candidate's own id, so its Discord permalink resolves to the message
        # a human can check it against.
        assert active[0].channel_id == CHANNEL
        assert active[0].message_id == staged.message_id

    async def test_confirming_records_who_did_it_and_what_it_became(
        self, conn: aiosqlite.Connection
    ) -> None:
        staged = await _stage(conn)
        assert staged is not None
        fact = await confirm_pending_fact(
            conn, guild_id=GUILD_A, pending_id=staged.id, resolved_by_id=MOD_A
        )

        resolved = await get_pending_fact(conn, guild_id=GUILD_A, pending_id=staged.id)
        assert resolved is not None
        assert resolved.status is PendingFactStatus.CONFIRMED
        assert resolved.resolved_by_id == MOD_A
        assert resolved.resolved_at is not None
        assert resolved.confirmed_fact_id == fact.id

    async def test_a_confirmed_candidate_leaves_the_review_queue(
        self, conn: aiosqlite.Connection
    ) -> None:
        staged = await _stage(conn)
        assert staged is not None
        await confirm_pending_fact(
            conn, guild_id=GUILD_A, pending_id=staged.id, resolved_by_id=MOD_A
        )
        assert await count_pending_facts(conn, guild_id=GUILD_A) == 0
        assert await get_pending_facts(conn, guild_id=GUILD_A, limit=10) == []

    async def test_confirming_twice_raises_and_creates_no_second_fact(
        self, conn: aiosqlite.Connection
    ) -> None:
        staged = await _stage(conn)
        assert staged is not None
        await confirm_pending_fact(
            conn, guild_id=GUILD_A, pending_id=staged.id, resolved_by_id=MOD_A
        )

        with pytest.raises(PendingFactAlreadyResolvedError):
            await confirm_pending_fact(
                conn, guild_id=GUILD_A, pending_id=staged.id, resolved_by_id=MOD_B
            )

        assert len(await get_active_facts(conn, GUILD_A)) == 1

    async def test_confirming_a_nonexistent_candidate_raises(
        self, conn: aiosqlite.Connection
    ) -> None:
        with pytest.raises(PendingFactNotFoundError):
            await confirm_pending_fact(
                conn, guild_id=GUILD_A, pending_id=4242, resolved_by_id=MOD_A
            )


class TestDiscarding:
    async def test_discarding_writes_no_fact_and_keeps_the_record(
        self, conn: aiosqlite.Connection
    ) -> None:
        staged = await _stage(conn)
        assert staged is not None
        await discard_pending_fact(
            conn, guild_id=GUILD_A, pending_id=staged.id, resolved_by_id=MOD_B
        )

        assert await get_active_facts(conn, GUILD_A) == []
        assert await count_pending_facts(conn, guild_id=GUILD_A) == 0
        # Kept, not deleted: a rejection is the only evidence a later phase has
        # about whether extraction earns its cost.
        resolved = await get_pending_fact(conn, guild_id=GUILD_A, pending_id=staged.id)
        assert resolved is not None
        assert resolved.status is PendingFactStatus.DISCARDED
        assert resolved.resolved_by_id == MOD_B
        assert resolved.confirmed_fact_id is None

    async def test_discarding_twice_raises(self, conn: aiosqlite.Connection) -> None:
        staged = await _stage(conn)
        assert staged is not None
        await discard_pending_fact(
            conn, guild_id=GUILD_A, pending_id=staged.id, resolved_by_id=MOD_A
        )
        with pytest.raises(PendingFactAlreadyResolvedError):
            await discard_pending_fact(
                conn, guild_id=GUILD_A, pending_id=staged.id, resolved_by_id=MOD_B
            )

    async def test_confirming_an_already_discarded_candidate_raises_and_writes_nothing(
        self, conn: aiosqlite.Connection
    ) -> None:
        staged = await _stage(conn)
        assert staged is not None
        await discard_pending_fact(
            conn, guild_id=GUILD_A, pending_id=staged.id, resolved_by_id=MOD_A
        )

        with pytest.raises(PendingFactAlreadyResolvedError):
            await confirm_pending_fact(
                conn, guild_id=GUILD_A, pending_id=staged.id, resolved_by_id=MOD_B
            )

        assert await get_active_facts(conn, GUILD_A) == []


class TestResolutionRaces:
    """The phase brief's named attack: two moderators resolving one candidate at once.

    Real concurrency via asyncio.gather, not sequential calls. The property
    under test is never "who wins" -- with two genuinely simultaneous clicks
    there is no correct winner to prefer -- but that exactly one call takes
    effect, the other fails cleanly, and no interleaving produces two active
    facts or a fact alongside a discard.
    """

    async def test_two_concurrent_confirmations_produce_exactly_one_fact(
        self, conn: aiosqlite.Connection
    ) -> None:
        staged = await _stage(conn)
        assert staged is not None

        results = await asyncio.gather(
            confirm_pending_fact(
                conn, guild_id=GUILD_A, pending_id=staged.id, resolved_by_id=MOD_A
            ),
            confirm_pending_fact(
                conn, guild_id=GUILD_A, pending_id=staged.id, resolved_by_id=MOD_B
            ),
            return_exceptions=True,
        )

        winners = [r for r in results if not isinstance(r, BaseException)]
        losers = [r for r in results if isinstance(r, PendingFactAlreadyResolvedError)]
        assert len(winners) == 1
        assert len(losers) == 1

        active = await get_active_facts(conn, GUILD_A)
        assert len(active) == 1
        assert active[0].id == winners[0].id

    async def test_five_concurrent_confirmations_still_produce_exactly_one_fact(
        self, conn: aiosqlite.Connection
    ) -> None:
        staged = await _stage(conn)
        assert staged is not None

        results = await asyncio.gather(
            *(
                confirm_pending_fact(
                    conn, guild_id=GUILD_A, pending_id=staged.id, resolved_by_id=mod
                )
                for mod in range(5)
            ),
            return_exceptions=True,
        )

        assert len([r for r in results if not isinstance(r, BaseException)]) == 1
        assert (
            len([r for r in results if isinstance(r, PendingFactAlreadyResolvedError)]) == 4
        )
        assert len(await get_active_facts(conn, GUILD_A)) == 1

    async def test_a_concurrent_confirm_and_discard_leave_one_definite_outcome(
        self, conn: aiosqlite.Connection
    ) -> None:
        # The mixed race the brief asks about specifically. Either outcome is
        # acceptable; what is not acceptable is both, or a fact created
        # alongside a candidate marked discarded.
        staged = await _stage(conn)
        assert staged is not None

        results = await asyncio.gather(
            confirm_pending_fact(
                conn, guild_id=GUILD_A, pending_id=staged.id, resolved_by_id=MOD_A
            ),
            discard_pending_fact(
                conn, guild_id=GUILD_A, pending_id=staged.id, resolved_by_id=MOD_B
            ),
            return_exceptions=True,
        )

        failures = [r for r in results if isinstance(r, BaseException)]
        assert len(failures) == 1
        assert isinstance(failures[0], PendingFactAlreadyResolvedError)

        resolved = await get_pending_fact(conn, guild_id=GUILD_A, pending_id=staged.id)
        assert resolved is not None
        assert resolved.status in (PendingFactStatus.CONFIRMED, PendingFactStatus.DISCARDED)

        active = await get_active_facts(conn, GUILD_A)
        if resolved.status is PendingFactStatus.CONFIRMED:
            # Confirmed means exactly one fact, and the candidate points at it.
            assert len(active) == 1
            assert resolved.confirmed_fact_id == active[0].id
        else:
            # Discarded means no fact at all -- the loser's insert must have
            # been rolled back, not merely unlinked.
            assert active == []
            assert resolved.confirmed_fact_id is None

    async def test_concurrent_staging_of_the_same_candidate_leaves_one_row(
        self, conn: aiosqlite.Connection
    ) -> None:
        # Two sweeps racing on a re-distilled batch (or two processes sharing
        # the file) must not produce two identical candidates.
        results = await asyncio.gather(*(_stage(conn) for _ in range(10)))

        assert len([r for r in results if r is not None]) == 1
        assert await count_pending_facts(conn, guild_id=GUILD_A) == 1


class TestReviewQueueOrder:
    async def test_candidates_come_back_oldest_first(
        self, conn: aiosqlite.Connection
    ) -> None:
        # A work queue reviewed newest-first would let the oldest candidate sit
        # at the bottom forever while newer ones keep landing on top of it.
        first = await _stage(conn, message_id=1, content="First.")
        second = await _stage(conn, message_id=2, content="Second.")
        third = await _stage(conn, message_id=3, content="Third.")
        assert first is not None and second is not None and third is not None

        queue = await get_pending_facts(conn, guild_id=GUILD_A, limit=10)
        assert [candidate.id for candidate in queue] == [first.id, second.id, third.id]

    async def test_a_negative_limit_is_rejected_rather_than_meaning_no_limit(
        self, conn: aiosqlite.Connection
    ) -> None:
        # LIMIT -1 in SQL means "every row", so a caller that arrived at -1 by
        # arithmetic would silently get the whole table.
        with pytest.raises(ValueError):
            await get_pending_facts(conn, guild_id=GUILD_A, limit=-1)

    async def test_a_zero_limit_returns_nothing_without_erroring(
        self, conn: aiosqlite.Connection
    ) -> None:
        await _stage(conn)
        assert await get_pending_facts(conn, guild_id=GUILD_A, limit=0) == []


class TestDedupHint:
    async def test_the_hint_is_stored_and_read_back(self, conn: aiosqlite.Connection) -> None:
        existing = await create_fact(
            conn,
            guild_id=GUILD_A,
            channel_id=CHANNEL,
            message_id=77,
            content="Maintenance is on Tuesdays.",
            embedding=EMBEDDING,
        )
        staged = await _stage(conn, similar_fact_id=existing.id, similar_fact_score=0.83)
        assert staged is not None

        read_back = await get_pending_fact(conn, guild_id=GUILD_A, pending_id=staged.id)
        assert read_back is not None
        assert read_back.similar_fact_id == existing.id
        assert read_back.similar_fact_score == pytest.approx(0.83)

    async def test_a_flagged_candidate_still_confirms_into_an_independent_fact(
        self, conn: aiosqlite.Connection
    ) -> None:
        # The hint is advisory: it must never supersede anything by itself.
        # Phase 3a-3 owns automatic supersession; this phase leaves the old
        # fact active and lets the moderator decide.
        existing = await create_fact(
            conn,
            guild_id=GUILD_A,
            channel_id=CHANNEL,
            message_id=77,
            content="Maintenance is on Tuesdays.",
            embedding=EMBEDDING,
        )
        staged = await _stage(
            conn,
            content="Maintenance now runs on Wednesdays.",
            similar_fact_id=existing.id,
            similar_fact_score=0.91,
        )
        assert staged is not None

        await confirm_pending_fact(
            conn, guild_id=GUILD_A, pending_id=staged.id, resolved_by_id=MOD_A
        )

        active = await get_active_facts(conn, GUILD_A)
        assert len(active) == 2
        assert all(fact.status is FactStatus.ACTIVE for fact in active)
        assert all(fact.superseded_by_id is None for fact in active)


class TestRelationshipJudgement:
    """Phase 3a-3's two columns: what may be written into them, and when.

    The judgement is advisory in exactly the way the hint above it is, and the
    last test in this class is the one that matters most -- the strongest
    possible verdict still leaves both facts active, because nothing in this
    module has ever been able to supersede anything.
    """

    @staticmethod
    async def _flagged(conn: aiosqlite.Connection, *, score: float = 0.83):
        existing = await create_fact(
            conn,
            guild_id=GUILD_A,
            channel_id=CHANNEL,
            message_id=77,
            content="Maintenance is on Tuesdays.",
            embedding=EMBEDDING,
        )
        staged = await _stage(
            conn,
            content="Maintenance now runs on Wednesdays.",
            similar_fact_id=existing.id,
            similar_fact_score=score,
        )
        assert staged is not None
        return existing, staged

    async def test_a_fresh_candidate_has_no_judgement(
        self, conn: aiosqlite.Connection
    ) -> None:
        # NULL means "never judged", and it is an ordinary state: no dedup hit,
        # the cap refused the call, or the call failed.
        staged = await _stage(conn)
        assert staged is not None
        assert staged.relationship is None
        assert staged.relationship_reasoning is None

    @pytest.mark.parametrize("relationship", list(SupersessionRelationship))
    async def test_every_relationship_round_trips(
        self, conn: aiosqlite.Connection, relationship: SupersessionRelationship
    ) -> None:
        _, staged = await self._flagged(conn)
        assert await record_relationship_judgement(
            conn,
            guild_id=GUILD_A,
            pending_id=staged.id,
            relationship=relationship,
            reasoning="Beide nennen denselben Wartungstag.",
        )

        read_back = await get_pending_fact(conn, guild_id=GUILD_A, pending_id=staged.id)
        assert read_back is not None
        assert read_back.relationship is relationship
        assert read_back.relationship_reasoning == "Beide nennen denselben Wartungstag."

    async def test_a_judgement_cannot_be_attached_to_an_unflagged_candidate(
        self, conn: aiosqlite.Connection
    ) -> None:
        # The data-level half of the cost bound: this judgement exists only to
        # explain a dedup hit, and a candidate without one has nothing for it to
        # explain -- regardless of what any future call site tries.
        staged = await _stage(conn)
        assert staged is not None
        assert not await record_relationship_judgement(
            conn,
            guild_id=GUILD_A,
            pending_id=staged.id,
            relationship=SupersessionRelationship.SUPERSESSION,
            reasoning="Nothing to compare against.",
        )

        read_back = await get_pending_fact(conn, guild_id=GUILD_A, pending_id=staged.id)
        assert read_back is not None
        assert read_back.relationship is None

    @pytest.mark.parametrize("resolve", [confirm_pending_fact, discard_pending_fact])
    async def test_an_already_resolved_candidate_is_not_annotated_afterwards(
        self, conn: aiosqlite.Connection, resolve
    ) -> None:
        # The realistic race: a moderator reviews the candidate while the
        # judgement call is still in flight. Writing it afterwards would make
        # the stored row claim they decided with information that arrived later.
        _, staged = await self._flagged(conn)
        await resolve(conn, guild_id=GUILD_A, pending_id=staged.id, resolved_by_id=MOD_A)

        assert not await record_relationship_judgement(
            conn,
            guild_id=GUILD_A,
            pending_id=staged.id,
            relationship=SupersessionRelationship.CONTRADICTION,
            reasoning="Arrived too late.",
        )

        read_back = await get_pending_fact(conn, guild_id=GUILD_A, pending_id=staged.id)
        assert read_back is not None
        assert read_back.relationship is None

    async def test_another_guild_cannot_annotate_this_guilds_candidate(
        self, conn: aiosqlite.Connection
    ) -> None:
        _, staged = await self._flagged(conn)
        assert not await record_relationship_judgement(
            conn,
            guild_id=GUILD_B,
            pending_id=staged.id,
            relationship=SupersessionRelationship.SUPERSESSION,
            reasoning="Wrong guild.",
        )

    async def test_a_nonexistent_candidate_is_a_clean_false(
        self, conn: aiosqlite.Connection
    ) -> None:
        assert not await record_relationship_judgement(
            conn,
            guild_id=GUILD_A,
            pending_id=999999,
            relationship=SupersessionRelationship.SUPERSESSION,
            reasoning="Nobody home.",
        )

    async def test_an_invented_relationship_is_refused_by_the_schema(
        self, conn: aiosqlite.Connection
    ) -> None:
        # The CHECK constraint, asserted rather than trusted: the enum guards
        # the code path, and this guards the column against anything else.
        _, staged = await self._flagged(conn)
        with pytest.raises(aiosqlite.IntegrityError):
            await conn.execute(
                "UPDATE pending_facts SET relationship = ? WHERE id = ?",
                ("delete_the_old_one", staged.id),
            )

    async def test_a_judged_candidate_still_confirms_into_an_independent_fact(
        self, conn: aiosqlite.Connection
    ) -> None:
        # The property the whole sub-phase rests on: a "supersession" verdict is
        # a proposal a moderator reads. Confirming the candidate creates a
        # second active fact and retires nothing -- only /aura-supersede does
        # that, and only when a human runs it.
        existing, staged = await self._flagged(conn)
        await record_relationship_judgement(
            conn,
            guild_id=GUILD_A,
            pending_id=staged.id,
            relationship=SupersessionRelationship.SUPERSESSION,
            reasoning="The maintenance day changed.",
        )
        await confirm_pending_fact(
            conn, guild_id=GUILD_A, pending_id=staged.id, resolved_by_id=MOD_A
        )

        active = await get_active_facts(conn, GUILD_A)
        assert len(active) == 2
        assert all(fact.superseded_by_id is None for fact in active)
        assert existing.id in {fact.id for fact in active}


class TestSchemaMigration:
    """A pending_facts table created by Phase 3a-2 must gain the two new columns.

    Purely additive, so it migrates in place rather than refusing to start: no
    existing column changes meaning, and every pre-existing candidate keeps
    rendering exactly as it did, with the plain similarity hint.
    """

    @staticmethod
    async def _phase_3a2_table(conn: aiosqlite.Connection) -> None:
        """Recreate the table as Phase 3a-2 shipped it -- without the new columns."""
        await conn.execute("DROP TABLE pending_facts")
        await conn.execute(
            """
            CREATE TABLE pending_facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                guild_id INTEGER NOT NULL,
                channel_id INTEGER NOT NULL,
                message_id INTEGER NOT NULL,
                content TEXT NOT NULL,
                embedding BLOB NOT NULL,
                category TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                similar_fact_id INTEGER REFERENCES facts(id),
                similar_fact_score REAL,
                confirmed_fact_id INTEGER REFERENCES facts(id),
                created_at TEXT NOT NULL,
                resolved_at TEXT,
                resolved_by_id INTEGER,
                UNIQUE (channel_id, message_id, content)
            )
            """
        )
        await conn.commit()

    @staticmethod
    async def _columns(conn: aiosqlite.Connection) -> set[str]:
        async with conn.execute("PRAGMA table_info(pending_facts)") as cursor:
            return {row[1] for row in await cursor.fetchall()}

    async def test_an_older_table_is_migrated_in_place(
        self, conn: aiosqlite.Connection
    ) -> None:
        await self._phase_3a2_table(conn)
        assert "relationship" not in await self._columns(conn)

        await verify_pending_facts_schema(conn)

        assert {"relationship", "relationship_reasoning"} <= await self._columns(conn)

    async def test_a_candidate_staged_before_the_migration_still_reads_back(
        self, conn: aiosqlite.Connection
    ) -> None:
        await self._phase_3a2_table(conn)
        await conn.execute(
            """
            INSERT INTO pending_facts
                (guild_id, channel_id, message_id, content, embedding, category,
                 status, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                GUILD_A,
                CHANNEL,
                1,
                "An older candidate.",
                EMBEDDING,
                FactCategory.RULE,
                PendingFactStatus.PENDING,
                "2026-07-30T12:00:00.000000+00:00",
            ),
        )
        await conn.commit()

        await verify_pending_facts_schema(conn)

        candidates = await get_pending_facts(conn, guild_id=GUILD_A, limit=10)
        assert len(candidates) == 1
        assert candidates[0].content == "An older candidate."
        assert candidates[0].relationship is None

    async def test_the_migration_is_idempotent(self, conn: aiosqlite.Connection) -> None:
        # Run twice on an already-current table, and once on a half-migrated one
        # (a crash between the two ALTERs), which must be completed in one pass.
        await verify_pending_facts_schema(conn)
        await verify_pending_facts_schema(conn)

        await self._phase_3a2_table(conn)
        await conn.execute("ALTER TABLE pending_facts ADD COLUMN relationship TEXT")
        await conn.commit()
        await verify_pending_facts_schema(conn)
        assert {"relationship", "relationship_reasoning"} <= await self._columns(conn)

    async def test_a_database_without_the_table_passes_untouched(
        self, conn: aiosqlite.Connection
    ) -> None:
        await conn.execute("DROP TABLE pending_facts")
        await conn.commit()
        await verify_pending_facts_schema(conn)
        assert await self._columns(conn) == set()

    async def test_the_migrated_column_still_rejects_an_invented_value(
        self, conn: aiosqlite.Connection
    ) -> None:
        # The CHECK constraint has to survive the ALTER path too, or a migrated
        # deployment would be the one place an arbitrary string could land in
        # the column.
        await self._phase_3a2_table(conn)
        await verify_pending_facts_schema(conn)
        await conn.execute(
            """
            INSERT INTO pending_facts
                (id, guild_id, channel_id, message_id, content, embedding, category,
                 status, created_at)
            VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                GUILD_A,
                CHANNEL,
                1,
                "An older candidate.",
                EMBEDDING,
                FactCategory.RULE,
                PendingFactStatus.PENDING,
                "2026-07-30T12:00:00.000000+00:00",
            ),
        )
        with pytest.raises(aiosqlite.IntegrityError):
            await conn.execute(
                "UPDATE pending_facts SET relationship = 'whatever' WHERE id = 1"
            )
