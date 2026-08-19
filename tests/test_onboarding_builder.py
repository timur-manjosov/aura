"""Tests for aura.onboarding.builder: what a brand-new member should be shown.

Mirrors tests/test_digest_builder.py's shape (facts inserted through raw SQL so
timestamps are exact, a helper to mark a candidate's category) but asks a
different set of questions, because onboarding is not windowed:

  * TestCategoryPriority -- CLAUDE.md's third trigger reads the same knowledge
    model the digest does, but orders and filters it differently (rules and
    status changes first, milestones excluded entirely).
  * TestTheCap -- the global item limit, spent in priority order, and what
    omitted_count reports about what did not fit.
  * TestAgainstTheRealConfirmationPath -- the same defence-in-depth the digest
    builder tests take: verified against the actual writers, not just against
    hand-inserted rows shaped like their output.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import aiosqlite
import pytest

from aura.db.connection import utc_iso
from aura.db.pending_facts import FactCategory, confirm_pending_fact, stage_pending_fact
from aura.db.repository import init_schema, supersede_fact_with_existing_successor
from aura.onboarding.builder import build_onboarding_content

GUILD_A = 100000000000000001
GUILD_B = 200000000000000002
CHANNEL = 300000000000000003

NOW = datetime(2026, 8, 16, 12, 0, 0, tzinfo=timezone.utc)

_next_message_id = iter(range(600000000000000000, 600000000000001000))


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
    created_at: datetime = NOW,
    guild_id: int = GUILD_A,
    status: str = "active",
) -> int:
    cursor = await conn.execute(
        """
        INSERT INTO facts
            (guild_id, channel_id, message_id, content, embedding, status, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (guild_id, CHANNEL, next(_next_message_id), content, b"\x00\x00\x00\x00",
         status, utc_iso(created_at)),
    )
    await conn.commit()
    assert cursor.lastrowid is not None
    return cursor.lastrowid


async def categorize(
    conn: aiosqlite.Connection,
    *,
    fact_id: int,
    category: str,
    guild_id: int = GUILD_A,
) -> None:
    """Record fact_id as having been confirmed from a candidate of the given category."""
    await conn.execute(
        """
        INSERT INTO pending_facts
            (guild_id, channel_id, message_id, content, embedding, category, status,
             confirmed_fact_id, created_at)
        VALUES (?, ?, ?, ?, ?, ?, 'confirmed', ?, ?)
        """,
        (guild_id, CHANNEL, next(_next_message_id), f"candidate for {fact_id}",
         b"\x00\x00\x00\x00", category, fact_id, utc_iso(NOW)),
    )
    await conn.commit()


async def build(conn: aiosqlite.Connection, *, guild_id: int = GUILD_A, limit: int = 15):
    return await build_onboarding_content(conn, guild_id=guild_id, limit=limit)


class TestEmptyGuild:
    async def test_a_guild_with_no_facts_is_empty(self, conn: aiosqlite.Connection) -> None:
        content = await build(conn)

        assert content.is_empty is True
        assert content.shown_count == 0
        assert content.omitted_count == 0

    async def test_a_superseded_fact_alone_is_still_empty(self, conn: aiosqlite.Connection) -> None:
        await add_fact(conn, content="Old rule.", status="superseded")

        content = await build(conn)

        assert content.is_empty is True

    async def test_another_guilds_facts_never_appear(self, conn: aiosqlite.Connection) -> None:
        await add_fact(conn, content="Not ours.", guild_id=GUILD_B)

        content = await build(conn)

        assert content.is_empty is True


class TestCategoryPriority:
    async def test_rules_and_status_changes_are_bucketed_separately_from_other(
        self, conn: aiosqlite.Connection
    ) -> None:
        rule_id = await add_fact(conn, content="No spoilers outside #spoilers.")
        await categorize(conn, fact_id=rule_id, category=FactCategory.RULE)
        status_id = await add_fact(conn, content="Voice chat is currently muted server-wide.")
        await categorize(conn, fact_id=status_id, category=FactCategory.STATUS_CHANGE)
        announcement_id = await add_fact(conn, content="A tournament is happening next week.")
        await categorize(conn, fact_id=announcement_id, category=FactCategory.ANNOUNCEMENT)

        content = await build(conn)

        assert [f.content for f in content.rules] == ["No spoilers outside #spoilers."]
        assert [f.content for f in content.status_changes] == [
            "Voice chat is currently muted server-wide."
        ]
        assert [f.content for f in content.other] == ["A tournament is happening next week."]

    async def test_milestones_are_excluded_entirely_not_routed_to_other(
        self, conn: aiosqlite.Connection
    ) -> None:
        milestone_id = await add_fact(conn, content="The server reached 500 members.")
        await categorize(conn, fact_id=milestone_id, category=FactCategory.MILESTONE)
        ordinary_id = await add_fact(conn, content="Movie night is on Fridays.")
        await categorize(conn, fact_id=ordinary_id, category=FactCategory.EVENT)

        content = await build(conn)

        assert content.rules == []
        assert content.status_changes == []
        assert [f.content for f in content.other] == ["Movie night is on Fridays."]
        assert content.total_eligible == 1

    async def test_a_guild_of_only_milestones_is_empty(self, conn: aiosqlite.Connection) -> None:
        milestone_id = await add_fact(conn, content="1000 members!")
        await categorize(conn, fact_id=milestone_id, category=FactCategory.MILESTONE)

        content = await build(conn)

        assert content.is_empty is True

    async def test_a_hand_entered_fact_with_no_category_lands_in_other(
        self, conn: aiosqlite.Connection
    ) -> None:
        # Never categorised by anything (added through "Add as Aura Fact"), so
        # it cannot be routed to rules or status_changes -- only "other" reads
        # a missing category as a valid bucket.
        await add_fact(conn, content="We just hit 1000 members!")

        content = await build(conn)

        assert content.rules == []
        assert content.status_changes == []
        assert [f.content for f in content.other] == ["We just hit 1000 members!"]

    async def test_within_a_bucket_facts_are_newest_first(
        self, conn: aiosqlite.Connection
    ) -> None:
        old_id = await add_fact(conn, content="older rule", created_at=NOW - timedelta(days=5))
        await categorize(conn, fact_id=old_id, category=FactCategory.RULE)
        new_id = await add_fact(conn, content="newer rule", created_at=NOW - timedelta(days=1))
        await categorize(conn, fact_id=new_id, category=FactCategory.RULE)

        content = await build(conn)

        assert [f.content for f in content.rules] == ["newer rule", "older rule"]

    async def test_another_guilds_category_is_not_consulted(
        self, conn: aiosqlite.Connection
    ) -> None:
        rule_id = await add_fact(conn, content="ours")
        await categorize(conn, fact_id=rule_id, category=FactCategory.RULE, guild_id=GUILD_B)

        content = await build(conn)

        # Categorized for the wrong guild -- our guild sees it as uncategorized.
        assert content.rules == []
        assert [f.content for f in content.other] == ["ours"]


class TestTheCap:
    async def test_the_limit_is_spent_in_priority_order_across_sections(
        self, conn: aiosqlite.Connection
    ) -> None:
        for index in range(3):
            rule_id = await add_fact(conn, content=f"rule {index}")
            await categorize(conn, fact_id=rule_id, category=FactCategory.RULE)
        for index in range(3):
            status_id = await add_fact(conn, content=f"status {index}")
            await categorize(conn, fact_id=status_id, category=FactCategory.STATUS_CHANGE)
        for index in range(3):
            await add_fact(conn, content=f"other {index}")

        content = await build(conn, limit=4)

        assert len(content.rules) == 3
        assert len(content.status_changes) == 1
        assert content.other == []
        assert content.shown_count == 4
        assert content.total_eligible == 9
        assert content.omitted_count == 5

    async def test_a_limit_of_zero_shows_nothing_but_still_counts_eligible(
        self, conn: aiosqlite.Connection
    ) -> None:
        rule_id = await add_fact(conn, content="a rule")
        await categorize(conn, fact_id=rule_id, category=FactCategory.RULE)

        content = await build(conn, limit=0)

        assert content.shown_count == 0
        assert content.total_eligible == 1
        assert content.omitted_count == 1
        assert content.is_empty is True

    async def test_a_negative_limit_is_refused(self, conn: aiosqlite.Connection) -> None:
        with pytest.raises(ValueError, match="negative"):
            await build(conn, limit=-1)

    async def test_everything_fitting_leaves_no_omission(
        self, conn: aiosqlite.Connection
    ) -> None:
        rule_id = await add_fact(conn, content="a rule")
        await categorize(conn, fact_id=rule_id, category=FactCategory.RULE)

        content = await build(conn, limit=15)

        assert content.omitted_count == 0


class TestPurity:
    async def test_building_content_writes_nothing(self, conn: aiosqlite.Connection) -> None:
        rule_id = await add_fact(conn, content="a rule")
        await categorize(conn, fact_id=rule_id, category=FactCategory.RULE)
        async with conn.execute("SELECT COUNT(*) FROM facts") as cursor:
            before = await cursor.fetchone()

        await build(conn)
        await build(conn)

        async with conn.execute("SELECT COUNT(*) FROM facts") as cursor:
            assert await cursor.fetchone() == before
        async with conn.execute("SELECT COUNT(*) FROM onboarding_sends") as cursor:
            assert await cursor.fetchone() == (0,)


class TestAgainstTheRealConfirmationPath:
    """Verified against the code that really writes categorized, confirmed facts.

    Every other test in this file stages pending_facts by hand, which would
    keep passing if the real confirmation path wrote a different shape. This
    drives stage_pending_fact + confirm_pending_fact directly.
    """

    async def test_a_confirmed_rule_reaches_the_rules_section(
        self, conn: aiosqlite.Connection
    ) -> None:
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
        fact = await confirm_pending_fact(
            conn, guild_id=GUILD_A, pending_id=candidate.id, resolved_by_id=7
        )

        content = await build_onboarding_content(conn, guild_id=GUILD_A, limit=15)

        assert [f.id for f in content.rules] == [fact.id]

    async def test_a_confirmed_milestone_is_excluded(self, conn: aiosqlite.Connection) -> None:
        candidate = await stage_pending_fact(
            conn,
            guild_id=GUILD_A,
            channel_id=CHANNEL,
            message_id=next(_next_message_id),
            content="The server reached 500 members.",
            embedding=b"\x00\x00\x00\x00",
            category=FactCategory.MILESTONE,
        )
        assert candidate is not None
        await confirm_pending_fact(
            conn, guild_id=GUILD_A, pending_id=candidate.id, resolved_by_id=7
        )

        content = await build_onboarding_content(conn, guild_id=GUILD_A, limit=15)

        assert content.is_empty is True

    async def test_a_superseded_fact_from_a_real_supersession_is_not_shown(
        self, conn: aiosqlite.Connection
    ) -> None:
        old = await add_fact(conn, content="Meetings are on Monday.")
        new = await add_fact(conn, content="Meetings are on Thursday.")
        await supersede_fact_with_existing_successor(
            conn, old_fact_id=old, new_fact_id=new, guild_id=GUILD_A
        )

        content = await build_onboarding_content(conn, guild_id=GUILD_A, limit=15)

        contents = [f.content for f in content.other]
        assert "Meetings are on Thursday." in contents
        assert "Meetings are on Monday." not in contents
