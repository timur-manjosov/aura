"""Tests for aura.digest.builder: what a digest actually says about a window.

Four of these classes exist because the phase brief asked the questions
directly, and each answer is a deliberate decision rather than whatever the
implementation happened to do:

  * TestFactsCreatedAndRetiredInTheSameWindow -- what a digest shows when
    everything new was already replaced before it ran.
  * TestCollapsedChains -- what it shows for a fact replaced several times over.
  * TestMilestones -- the first use of the category Phase 3a-2 designed for this.
  * TestBrokenChains -- what a hand-edited or impossible chain does to it.

Facts are inserted through raw SQL rather than through create_fact, for one
reason: create_fact timestamps from the clock, and every question here is about
which side of a boundary a timestamp falls on. Nothing else about the rows
differs from what create_fact writes.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import aiosqlite
import pytest

from aura.db.connection import utc_iso, utc_now
from aura.db.pending_facts import (
    FactCategory,
    confirm_pending_fact,
    stage_pending_fact,
)
from aura.db.repository import (
    create_fact,
    init_schema,
    supersede_fact_with_existing_successor,
)
from aura.digest.builder import build_digest

GUILD_A = 100000000000000001
GUILD_B = 200000000000000002
CHANNEL = 300000000000000003

NOW = datetime(2026, 8, 16, 12, 0, 0, tzinfo=timezone.utc)
LAST_DIGEST = NOW - timedelta(days=7)
BEFORE_EVERYTHING = NOW - timedelta(days=90)

_next_message_id = iter(range(500000000000000000, 500000000000001000))


@pytest.fixture
async def conn():
    connection = await aiosqlite.connect(":memory:")
    await init_schema(connection)
    yield connection
    await connection.close()


async def add_fact(
    conn: aiosqlite.Connection,
    *,
    content: str,
    created_at: datetime,
    guild_id: int = GUILD_A,
) -> int:
    """Insert one active fact with an exact creation timestamp. Returns its ID."""
    cursor = await conn.execute(
        """
        INSERT INTO facts
            (guild_id, channel_id, message_id, content, embedding, status, created_at)
        VALUES (?, ?, ?, ?, ?, 'active', ?)
        """,
        (guild_id, CHANNEL, next(_next_message_id), content, b"\x00\x00\x00\x00",
         utc_iso(created_at)),
    )
    await conn.commit()
    assert cursor.lastrowid is not None
    return cursor.lastrowid


async def supersede(
    conn: aiosqlite.Connection, *, old_id: int, new_id: int | None, at: datetime
) -> None:
    """Retire old_id in favour of new_id at an exact instant, as supersede_fact would."""
    await conn.execute(
        "UPDATE facts SET status = 'superseded', superseded_by_id = ?, superseded_at = ? "
        "WHERE id = ?",
        (new_id, utc_iso(at), old_id),
    )
    await conn.commit()


async def mark_as_milestone_candidate(
    conn: aiosqlite.Connection,
    *,
    fact_id: int,
    guild_id: int = GUILD_A,
    category: str = "milestone",
    status: str = "confirmed",
) -> None:
    """Record that fact_id was confirmed from a candidate of the given category."""
    await conn.execute(
        """
        INSERT INTO pending_facts
            (guild_id, channel_id, message_id, content, embedding, category, status,
             confirmed_fact_id, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (guild_id, CHANNEL, next(_next_message_id), f"candidate for {fact_id}",
         b"\x00\x00\x00\x00", category, status,
         fact_id if status == "confirmed" else None, utc_iso(NOW)),
    )
    await conn.commit()


async def build(conn: aiosqlite.Connection, *, guild_id: int = GUILD_A, since: datetime = LAST_DIGEST):
    return await build_digest(
        conn, guild_id=guild_id, since=utc_iso(since), until=utc_iso(NOW)
    )


class TestNewFacts:
    async def test_a_fact_created_in_the_window_is_reported_as_new(
        self, conn: aiosqlite.Connection
    ) -> None:
        await add_fact(conn, content="Movie night is on Fridays.", created_at=NOW - timedelta(days=2))

        content = await build(conn)

        assert [fact.content for fact in content.new_facts] == ["Movie night is on Fridays."]
        assert content.is_empty is False

    async def test_a_fact_from_before_the_window_is_not_reported(
        self, conn: aiosqlite.Connection
    ) -> None:
        await add_fact(conn, content="Old news.", created_at=BEFORE_EVERYTHING)

        content = await build(conn)

        assert content.new_facts == []
        assert content.is_empty is True

    async def test_the_window_is_half_open_at_its_start(
        self, conn: aiosqlite.Connection
    ) -> None:
        # A fact created at exactly the previous window's end was already
        # reported by that digest; reporting it again would duplicate it.
        await add_fact(conn, content="Right on the boundary.", created_at=LAST_DIGEST)

        content = await build(conn)

        assert content.new_facts == []

    async def test_the_window_is_closed_at_its_end(self, conn: aiosqlite.Connection) -> None:
        # A fact created at exactly `until` belongs to this window, so that
        # nothing can fall between two consecutive digests.
        await add_fact(conn, content="Right at the end.", created_at=NOW)

        content = await build(conn)

        assert [fact.content for fact in content.new_facts] == ["Right at the end."]

    async def test_facts_are_ordered_oldest_first(self, conn: aiosqlite.Connection) -> None:
        await add_fact(conn, content="second", created_at=NOW - timedelta(days=2))
        await add_fact(conn, content="first", created_at=NOW - timedelta(days=5))
        await add_fact(conn, content="third", created_at=NOW - timedelta(hours=1))

        content = await build(conn)

        assert [fact.content for fact in content.new_facts] == ["first", "second", "third"]

    async def test_another_guilds_facts_never_appear(self, conn: aiosqlite.Connection) -> None:
        await add_fact(
            conn, content="Other server's business.", created_at=NOW - timedelta(days=1),
            guild_id=GUILD_B,
        )

        content = await build(conn)

        assert content.is_empty is True


class TestChanges:
    async def test_a_supersession_in_the_window_is_reported_as_a_change(
        self, conn: aiosqlite.Connection
    ) -> None:
        old_id = await add_fact(conn, content="Meetings are on Monday.", created_at=BEFORE_EVERYTHING)
        new_id = await add_fact(
            conn, content="Meetings are on Thursday.", created_at=NOW - timedelta(days=1)
        )
        await supersede(conn, old_id=old_id, new_id=new_id, at=NOW - timedelta(days=1))

        content = await build(conn)

        assert len(content.changes) == 1
        change = content.changes[0]
        assert change.previous.content == "Meetings are on Monday."
        assert change.current.content == "Meetings are on Thursday."
        assert change.collapsed_steps == 0
        # The successor is ALSO genuinely new, and appears in both sections on
        # purpose: a reader skimming only "new" should still see it.
        assert [fact.content for fact in content.new_facts] == ["Meetings are on Thursday."]

    async def test_a_supersession_from_before_the_window_is_not_reported(
        self, conn: aiosqlite.Connection
    ) -> None:
        old_id = await add_fact(conn, content="Ancient.", created_at=BEFORE_EVERYTHING)
        new_id = await add_fact(conn, content="Less ancient.", created_at=BEFORE_EVERYTHING)
        await supersede(conn, old_id=old_id, new_id=new_id, at=BEFORE_EVERYTHING + timedelta(days=1))

        content = await build(conn)

        assert content.is_empty is True

    async def test_the_successor_may_predate_the_window(
        self, conn: aiosqlite.Connection
    ) -> None:
        # /aura-supersede lets a moderator retire a fact in favour of an
        # existing one. The change happened in this window even though nothing
        # was created in it.
        old_id = await add_fact(conn, content="Draft rule.", created_at=BEFORE_EVERYTHING)
        new_id = await add_fact(conn, content="Final rule.", created_at=BEFORE_EVERYTHING)
        await supersede(conn, old_id=old_id, new_id=new_id, at=NOW - timedelta(hours=3))

        content = await build(conn)

        assert content.new_facts == []
        assert len(content.changes) == 1
        assert content.changes[0].current.content == "Final rule."

    async def test_changes_are_ordered_oldest_first(self, conn: aiosqlite.Connection) -> None:
        for index, days_ago in enumerate((1, 5, 3)):
            old_id = await add_fact(conn, content=f"old {index}", created_at=BEFORE_EVERYTHING)
            new_id = await add_fact(conn, content=f"new {index}", created_at=NOW - timedelta(days=days_ago))
            await supersede(conn, old_id=old_id, new_id=new_id, at=NOW - timedelta(days=days_ago))

        content = await build(conn)

        assert [change.previous.content for change in content.changes] == [
            "old 1",
            "old 2",
            "old 0",
        ]


class TestFactsCreatedAndRetiredInTheSameWindow:
    """The brief's third attack: everything new was already replaced again.

    The decision, taken deliberately: a digest states the NET change over its
    window. A fact nobody was ever told about, which is no longer true either,
    is not news -- so it appears in neither section, and the reader is told only
    about the version that currently holds.
    """

    async def test_a_fact_created_and_retired_in_the_window_is_not_reported_as_new(
        self, conn: aiosqlite.Connection
    ) -> None:
        first_id = await add_fact(conn, content="Event at 18:00.", created_at=NOW - timedelta(days=3))
        second_id = await add_fact(conn, content="Event at 20:00.", created_at=NOW - timedelta(days=2))
        await supersede(conn, old_id=first_id, new_id=second_id, at=NOW - timedelta(days=2))

        content = await build(conn)

        assert [fact.content for fact in content.new_facts] == ["Event at 20:00."]

    async def test_it_is_not_reported_as_a_change_either(
        self, conn: aiosqlite.Connection
    ) -> None:
        # "Event at 18:00 -> Event at 20:00" would be telling the reader about
        # the retirement of something they were never told existed.
        first_id = await add_fact(conn, content="Event at 18:00.", created_at=NOW - timedelta(days=3))
        second_id = await add_fact(conn, content="Event at 20:00.", created_at=NOW - timedelta(days=2))
        await supersede(conn, old_id=first_id, new_id=second_id, at=NOW - timedelta(days=2))

        content = await build(conn)

        assert content.changes == []

    async def test_everything_new_being_retired_leaves_only_the_survivor(
        self, conn: aiosqlite.Connection
    ) -> None:
        # Three facts created this window, two of them already replaced.
        first_id = await add_fact(conn, content="v1", created_at=NOW - timedelta(days=6))
        second_id = await add_fact(conn, content="v2", created_at=NOW - timedelta(days=4))
        third_id = await add_fact(conn, content="v3", created_at=NOW - timedelta(days=2))
        await supersede(conn, old_id=first_id, new_id=second_id, at=NOW - timedelta(days=4))
        await supersede(conn, old_id=second_id, new_id=third_id, at=NOW - timedelta(days=2))

        content = await build(conn)

        assert [fact.content for fact in content.new_facts] == ["v3"]
        assert content.changes == []
        assert content.total_items == 1

    async def test_a_new_fact_retired_in_favour_of_an_older_one_reports_nothing(
        self, conn: aiosqlite.Connection
    ) -> None:
        # Net effect for the reader: nothing changed. The fact they knew is
        # still the fact that holds.
        established_id = await add_fact(conn, content="The long-standing rule.", created_at=BEFORE_EVERYTHING)
        mistake_id = await add_fact(conn, content="A mistaken rule.", created_at=NOW - timedelta(days=2))
        await supersede(conn, old_id=mistake_id, new_id=established_id, at=NOW - timedelta(days=1))

        content = await build(conn)

        assert content.is_empty is True


class TestCollapsedChains:
    """The brief's fourth attack: a fact replaced several times in one window."""

    async def test_a_three_step_chain_is_reported_as_start_and_current_state(
        self, conn: aiosqlite.Connection
    ) -> None:
        a_id = await add_fact(conn, content="A", created_at=BEFORE_EVERYTHING)
        b_id = await add_fact(conn, content="B", created_at=NOW - timedelta(days=5))
        c_id = await add_fact(conn, content="C", created_at=NOW - timedelta(days=3))
        d_id = await add_fact(conn, content="D", created_at=NOW - timedelta(days=1))
        await supersede(conn, old_id=a_id, new_id=b_id, at=NOW - timedelta(days=5))
        await supersede(conn, old_id=b_id, new_id=c_id, at=NOW - timedelta(days=3))
        await supersede(conn, old_id=c_id, new_id=d_id, at=NOW - timedelta(days=1))

        content = await build(conn)

        assert len(content.changes) == 1
        change = content.changes[0]
        assert change.previous.content == "A"
        assert change.current.content == "D"
        # The intermediate steps are named as a count rather than hidden.
        assert change.collapsed_steps == 2
        # ...and the intermediates are not offered as "new" either: only the
        # version that currently holds is.
        assert [fact.content for fact in content.new_facts] == ["D"]

    async def test_the_change_is_dated_from_the_first_step_of_the_chain(
        self, conn: aiosqlite.Connection
    ) -> None:
        a_id = await add_fact(conn, content="A", created_at=BEFORE_EVERYTHING)
        b_id = await add_fact(conn, content="B", created_at=NOW - timedelta(days=5))
        c_id = await add_fact(conn, content="C", created_at=NOW - timedelta(days=1))
        await supersede(conn, old_id=a_id, new_id=b_id, at=NOW - timedelta(days=5))
        await supersede(conn, old_id=b_id, new_id=c_id, at=NOW - timedelta(days=1))

        content = await build(conn)

        assert content.changes[0].changed_at == NOW - timedelta(days=5)

    async def test_a_chain_continued_from_a_previous_window_starts_at_this_windows_step(
        self, conn: aiosqlite.Connection
    ) -> None:
        # A was already replaced by B before the last digest (so the reader
        # knows B), and B was replaced by C during this window. The change they
        # need is B -> C, not A -> C.
        a_id = await add_fact(conn, content="A", created_at=BEFORE_EVERYTHING)
        b_id = await add_fact(conn, content="B", created_at=BEFORE_EVERYTHING + timedelta(days=1))
        c_id = await add_fact(conn, content="C", created_at=NOW - timedelta(days=1))
        await supersede(conn, old_id=a_id, new_id=b_id, at=BEFORE_EVERYTHING + timedelta(days=1))
        await supersede(conn, old_id=b_id, new_id=c_id, at=NOW - timedelta(days=1))

        content = await build(conn)

        assert len(content.changes) == 1
        assert content.changes[0].previous.content == "B"
        assert content.changes[0].current.content == "C"

    async def test_two_facts_replaced_by_one_are_reported_as_two_changes(
        self, conn: aiosqlite.Connection
    ) -> None:
        # A merge is genuinely two changes for a reader who knew both.
        first_id = await add_fact(conn, content="Rule about images.", created_at=BEFORE_EVERYTHING)
        second_id = await add_fact(conn, content="Rule about links.", created_at=BEFORE_EVERYTHING)
        merged_id = await add_fact(
            conn, content="One rule about attachments.", created_at=NOW - timedelta(days=1)
        )
        await supersede(conn, old_id=first_id, new_id=merged_id, at=NOW - timedelta(days=1))
        await supersede(conn, old_id=second_id, new_id=merged_id, at=NOW - timedelta(days=1))

        content = await build(conn)

        assert len(content.changes) == 2
        assert {change.current.content for change in content.changes} == {
            "One rule about attachments."
        }


class TestMilestones:
    async def test_a_milestone_fact_is_held_apart_from_the_other_new_facts(
        self, conn: aiosqlite.Connection
    ) -> None:
        milestone_id = await add_fact(
            conn, content="The server reached 500 members.", created_at=NOW - timedelta(days=1)
        )
        await mark_as_milestone_candidate(conn, fact_id=milestone_id)
        await add_fact(conn, content="Movie night is on Fridays.", created_at=NOW - timedelta(days=1))

        content = await build(conn)

        assert [fact.content for fact in content.milestones] == [
            "The server reached 500 members."
        ]
        assert [fact.content for fact in content.new_facts] == ["Movie night is on Fridays."]

    @pytest.mark.parametrize("category", ["announcement", "rule", "decision", "event", "status_change"])
    async def test_every_other_category_stays_in_the_ordinary_section(
        self, conn: aiosqlite.Connection, category: str
    ) -> None:
        fact_id = await add_fact(conn, content="Something happened.", created_at=NOW - timedelta(days=1))
        await mark_as_milestone_candidate(conn, fact_id=fact_id, category=category)

        content = await build(conn)

        assert content.milestones == []
        assert len(content.new_facts) == 1

    async def test_a_hand_entered_fact_is_never_a_milestone(
        self, conn: aiosqlite.Connection
    ) -> None:
        # The documented limitation: a fact typed in through "Add as Aura Fact"
        # was never categorised by anything, so it cannot reach the milestone
        # section however milestone-like it reads.
        await add_fact(
            conn, content="We just hit 1000 members!", created_at=NOW - timedelta(days=1)
        )

        content = await build(conn)

        assert content.milestones == []
        assert len(content.new_facts) == 1

    async def test_a_discarded_milestone_candidate_does_not_promote_anything(
        self, conn: aiosqlite.Connection
    ) -> None:
        fact_id = await add_fact(conn, content="Unrelated fact.", created_at=NOW - timedelta(days=1))
        await mark_as_milestone_candidate(conn, fact_id=fact_id, status="discarded")

        content = await build(conn)

        assert content.milestones == []

    async def test_another_guilds_milestone_candidate_is_not_consulted(
        self, conn: aiosqlite.Connection
    ) -> None:
        fact_id = await add_fact(conn, content="Ours.", created_at=NOW - timedelta(days=1))
        await mark_as_milestone_candidate(conn, fact_id=fact_id, guild_id=GUILD_B)

        content = await build(conn)

        assert content.milestones == []
        assert len(content.new_facts) == 1

    async def test_a_milestone_alone_still_makes_a_digest_worth_posting(
        self, conn: aiosqlite.Connection
    ) -> None:
        milestone_id = await add_fact(
            conn, content="Reached 100 posts.", created_at=NOW - timedelta(days=1)
        )
        await mark_as_milestone_candidate(conn, fact_id=milestone_id)

        content = await build(conn)

        assert content.is_empty is False
        assert content.total_items == 1


class TestAgainstTheRealConfirmationPath:
    """The milestone section, verified against the code that really writes those rows.

    Every other test in this file stages pending_facts by hand, which would keep
    passing if the real confirmation path wrote a different shape -- a
    confirmed_fact_id left unset, say, or a status spelled differently. This one
    goes through aura.db.pending_facts itself, so the join the digest depends on
    is checked against its actual producer rather than against the test's idea
    of it.
    """

    async def test_a_confirmed_milestone_candidate_reaches_the_milestone_section(
        self, conn: aiosqlite.Connection
    ) -> None:
        started = utc_now() - timedelta(minutes=5)
        candidate = await stage_pending_fact(
            conn,
            guild_id=GUILD_A,
            channel_id=CHANNEL,
            message_id=next(_next_message_id),
            content="The server reached 500 members on 2026-08-15.",
            embedding=b"\x00\x00\x00\x00",
            category=FactCategory.MILESTONE,
        )
        assert candidate is not None
        fact = await confirm_pending_fact(
            conn, guild_id=GUILD_A, pending_id=candidate.id, resolved_by_id=7
        )

        content = await build_digest(
            conn,
            guild_id=GUILD_A,
            since=utc_iso(started),
            until=utc_iso(utc_now() + timedelta(minutes=5)),
        )

        assert [milestone.id for milestone in content.milestones] == [fact.id]
        assert content.new_facts == []

    async def test_a_real_supersession_is_reported_as_a_change(
        self, conn: aiosqlite.Connection
    ) -> None:
        # The same argument as the milestone case, applied to the other half of
        # the digest: the "Updated" section reads facts.superseded_at and
        # superseded_by_id, and every other test in this file writes those
        # columns by hand. This one drives /aura-supersede's actual repository
        # call, so the digest is checked against the writer it depends on.
        old = await create_fact(
            conn,
            guild_id=GUILD_A,
            channel_id=CHANNEL,
            message_id=next(_next_message_id),
            content="Meetings are on Monday.",
            embedding=b"\x00\x00\x00\x00",
        )
        new = await create_fact(
            conn,
            guild_id=GUILD_A,
            channel_id=CHANNEL,
            message_id=next(_next_message_id),
            content="Meetings are on Thursday.",
            embedding=b"\x00\x00\x00\x00",
        )
        # The window opens AFTER both facts exist, so only the supersession
        # falls inside it -- the shape a real mid-week /aura-supersede takes,
        # and the one where a "before -> now" pair is what the reader needs.
        started = utc_now()
        await supersede_fact_with_existing_successor(
            conn, old_fact_id=old.id, new_fact_id=new.id, guild_id=GUILD_A
        )

        content = await build_digest(
            conn,
            guild_id=GUILD_A,
            since=utc_iso(started),
            until=utc_iso(utc_now() + timedelta(minutes=5)),
        )

        assert len(content.changes) == 1
        assert content.changes[0].previous.id == old.id
        assert content.changes[0].current.id == new.id

    async def test_a_confirmed_candidate_of_another_category_does_not(
        self, conn: aiosqlite.Connection
    ) -> None:
        started = utc_now() - timedelta(minutes=5)
        candidate = await stage_pending_fact(
            conn,
            guild_id=GUILD_A,
            channel_id=CHANNEL,
            message_id=next(_next_message_id),
            content="Posting links needs level 5.",
            embedding=b"\x00\x00\x00\x00",
            category=FactCategory.RULE,
        )
        assert candidate is not None
        await confirm_pending_fact(
            conn, guild_id=GUILD_A, pending_id=candidate.id, resolved_by_id=7
        )

        content = await build_digest(
            conn,
            guild_id=GUILD_A,
            since=utc_iso(started),
            until=utc_iso(utc_now() + timedelta(minutes=5)),
        )

        assert content.milestones == []
        assert len(content.new_facts) == 1


class TestBrokenChains:
    """Hand-edited or impossible chain data must degrade, never hang or crash."""

    async def test_a_superseded_fact_with_no_successor_is_left_out_with_a_warning(
        self, conn: aiosqlite.Connection, caplog: pytest.LogCaptureFixture
    ) -> None:
        orphan_id = await add_fact(conn, content="Retired into nothing.", created_at=BEFORE_EVERYTHING)
        await supersede(conn, old_id=orphan_id, new_id=None, at=NOW - timedelta(days=1))

        with caplog.at_level(logging.WARNING):
            content = await build(conn)

        assert content.changes == []
        assert any("no successor" in record.getMessage() for record in caplog.records)

    async def test_a_dangling_successor_pointer_reports_the_chain_as_ending_there(
        self, conn: aiosqlite.Connection, caplog: pytest.LogCaptureFixture
    ) -> None:
        # The foreign key on facts.superseded_by_id makes this unwritable while
        # PRAGMA foreign_keys is on -- which init_schema guarantees for every
        # connection Aura opens. Turned off here on purpose: the question is
        # what the digest does with a database that HAS been corrupted (an
        # operator's manual DELETE, a restore from a partial dump), not whether
        # Aura itself can produce one.
        old_id = await add_fact(conn, content="Points nowhere.", created_at=BEFORE_EVERYTHING)
        await conn.execute("PRAGMA foreign_keys = OFF")
        await conn.execute(
            "UPDATE facts SET status = 'superseded', superseded_by_id = 999999, "
            "superseded_at = ? WHERE id = ?",
            (utc_iso(NOW - timedelta(days=1)), old_id),
        )
        await conn.commit()
        await conn.execute("PRAGMA foreign_keys = ON")

        with caplog.at_level(logging.WARNING):
            content = await build(conn)

        # The chain ends at the fact itself, so there is no pair to report --
        # but the digest is built rather than failing.
        assert content.changes == []
        assert any("not readable" in record.getMessage() for record in caplog.records)

    async def test_a_successor_in_another_guild_is_not_followed(
        self, conn: aiosqlite.Connection
    ) -> None:
        # Following it would print another server's fact into this one's digest.
        foreign_id = await add_fact(conn, content="Foreign fact.", created_at=BEFORE_EVERYTHING, guild_id=GUILD_B)
        old_id = await add_fact(conn, content="Ours.", created_at=BEFORE_EVERYTHING)
        await supersede(conn, old_id=old_id, new_id=foreign_id, at=NOW - timedelta(days=1))

        content = await build(conn)

        assert content.changes == []

    async def test_a_chain_that_loops_terminates(
        self, conn: aiosqlite.Connection, caplog: pytest.LogCaptureFixture
    ) -> None:
        # Structurally impossible through supersede_fact (a successor must be
        # active, so a chain can only ever grow at its head), which makes this a
        # hand-edited database. It must not spin the scheduler task forever.
        # A -> B -> C -> B, with A the only chain start.
        a_id = await add_fact(conn, content="A", created_at=BEFORE_EVERYTHING)
        b_id = await add_fact(conn, content="B", created_at=BEFORE_EVERYTHING)
        c_id = await add_fact(conn, content="C", created_at=BEFORE_EVERYTHING)
        await supersede(conn, old_id=a_id, new_id=b_id, at=NOW - timedelta(days=3))
        await supersede(conn, old_id=b_id, new_id=c_id, at=NOW - timedelta(days=2))
        await supersede(conn, old_id=c_id, new_id=b_id, at=NOW - timedelta(days=1))

        with caplog.at_level(logging.ERROR):
            content = await build(conn)

        assert any("loops back" in record.getMessage() for record in caplog.records)
        # Whatever it reports, it reports something finite and returns.
        assert len(content.changes) == 1

    async def test_a_two_fact_cycle_reports_no_change_rather_than_a_pair(
        self, conn: aiosqlite.Connection
    ) -> None:
        # The degenerate hand-edited case where every retired fact is also
        # somebody's successor: there is no chain START, so there is nothing to
        # report -- and, importantly, no infinite walk either.
        first_id = await add_fact(conn, content="A", created_at=BEFORE_EVERYTHING)
        second_id = await add_fact(conn, content="B", created_at=BEFORE_EVERYTHING)
        await supersede(conn, old_id=first_id, new_id=second_id, at=NOW - timedelta(days=2))
        await supersede(conn, old_id=second_id, new_id=first_id, at=NOW - timedelta(days=1))

        content = await build(conn)

        assert content.changes == []

    async def test_a_chain_longer_than_the_depth_cap_stops_and_says_so(
        self, conn: aiosqlite.Connection, caplog: pytest.LogCaptureFixture
    ) -> None:
        ids = [
            await add_fact(conn, content=f"step {index}", created_at=BEFORE_EVERYTHING)
            for index in range(40)
        ]
        for index in range(len(ids) - 1):
            await supersede(
                conn,
                old_id=ids[index],
                new_id=ids[index + 1],
                at=NOW - timedelta(days=6) + timedelta(minutes=index),
            )

        with caplog.at_level(logging.WARNING):
            content = await build(conn)

        assert len(content.changes) == 1
        assert content.changes[0].previous.content == "step 0"
        assert any("longer than" in record.getMessage() for record in caplog.records)


class TestEmptyWindow:
    async def test_a_window_with_nothing_in_it_is_empty(
        self, conn: aiosqlite.Connection
    ) -> None:
        content = await build(conn)

        assert content.is_empty is True
        assert content.total_items == 0

    async def test_the_reported_window_bounds_are_carried_through(
        self, conn: aiosqlite.Connection
    ) -> None:
        content = await build(conn)

        assert content.covered_from == LAST_DIGEST
        assert content.covered_until == NOW
        assert content.guild_id == GUILD_A

    async def test_building_a_digest_writes_nothing(self, conn: aiosqlite.Connection) -> None:
        # Assembly must be free of consequences: a retry, or two runners racing,
        # must be able to build the same digest without changing anything.
        await add_fact(conn, content="A fact.", created_at=NOW - timedelta(days=1))
        async with conn.execute("SELECT COUNT(*) FROM facts") as cursor:
            before = await cursor.fetchone()

        await build(conn)
        await build(conn)

        async with conn.execute("SELECT COUNT(*) FROM facts") as cursor:
            assert await cursor.fetchone() == before
        async with conn.execute("SELECT COUNT(*) FROM digest_runs") as cursor:
            assert await cursor.fetchone() == (0,)
