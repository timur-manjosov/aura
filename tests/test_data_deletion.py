"""The P7a deletion rules remove exactly what they should, in one step, and nothing else."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import timedelta
from pathlib import Path

import aiosqlite
import pytest

from aura.db.connection import utc_iso
from aura.db.deletion import (
    DeletionCounts,
    MemberDeletionMode,
    RetentionPolicy,
    apply_retention,
    forget_fact,
    forget_member,
    guilds_with_data,
    latest_snowflake_at,
    purge_guild,
)
from aura.db.models import Fact, FactStatus
from aura.db.repository import create_fact, get_fact_by_id
from aura.rendering import source_link
from tests.privacy_data import (
    AFTER,
    BEFORE,
    BILLING_TABLES,
    GUILD_A,
    GUILD_B,
    MEMBER,
    MODERATOR,
    OTHER,
    REQUEST,
    add_fact,
    open_database,
    populate,
    snapshot,
    snowflake,
    supersede,
)


@pytest.fixture
async def conn() -> AsyncIterator[aiosqlite.Connection]:
    connection = await open_database()
    yield connection
    await connection.close()


async def _scalar(conn: aiosqlite.Connection, sql: str, *parameters: object) -> object:
    async with conn.execute(sql, parameters) as cursor:
        row = await cursor.fetchone()
    assert row is not None
    return row[0]


async def _fact_ids(conn: aiosqlite.Connection) -> set[int]:
    async with conn.execute("SELECT id FROM facts") as cursor:
        return {row[0] for row in await cursor.fetchall()}


class TestSnowflakeBound:
    def test_a_message_written_at_the_moment_is_inside_the_bound(self) -> None:
        assert snowflake(REQUEST) <= latest_snowflake_at(REQUEST)

    def test_a_message_one_millisecond_later_is_outside(self) -> None:
        assert snowflake(REQUEST + timedelta(milliseconds=1)) > latest_snowflake_at(REQUEST)

    def test_before_discords_epoch_nothing_matches(self) -> None:
        from datetime import UTC, datetime

        assert latest_snowflake_at(datetime(2014, 1, 1, tzinfo=UTC)) == 0

    def test_a_naive_moment_is_refused(self) -> None:
        from datetime import datetime

        with pytest.raises(ValueError):
            latest_snowflake_at(datetime(2026, 1, 1))


class TestForgetMemberDeletingFacts:
    async def test_everything_of_the_member_goes_and_nothing_else(
        self, conn: aiosqlite.Connection
    ) -> None:
        ids = await populate(conn)
        before = await snapshot(conn)

        counts = await forget_member(
            conn,
            user_id=MEMBER,
            guild_id=None,
            mode=MemberDeletionMode.DELETE_FACTS,
            before=REQUEST,
        )

        after = await snapshot(conn)
        remaining = await _fact_ids(conn)
        assert ids["a_member_fact"] not in remaining and ids["b_member_fact"] not in remaining
        # Written after the request: kept, in both servers.
        assert {ids["a_member_late_fact"], ids["b_member_late_fact"]} <= remaining
        assert {ids["a_other_fact"], ids["b_other_fact"], ids["a_unknown_fact"]} <= remaining
        assert counts.rows["facts"] == 2
        assert counts.rows["pending_facts"] == 2  # the member's own candidates
        assert counts.rows["extraction_queue"] == 2
        assert counts.rows["ask_calls.anonymized"] == 2
        assert counts.rows["onboarding_sends"] == 2
        for table in BILLING_TABLES:
            assert after[table] == before[table]
        assert await _scalar(conn, "SELECT COUNT(*) FROM ask_calls WHERE user_id = ?", MEMBER) == 0
        assert await _scalar(conn, "SELECT COUNT(*) FROM ask_calls WHERE user_id = ?", OTHER) == 2
        assert await _scalar(conn, "SELECT COUNT(*) FROM ask_calls") == 4  # counts stay
        assert (
            await _scalar(conn, "SELECT COUNT(*) FROM onboarding_sends WHERE user_id = ?", OTHER)
            == 2
        )
        assert (
            await _scalar(conn, "SELECT COUNT(*) FROM extraction_queue WHERE author_id = ?", OTHER)
            == 2
        )

    async def test_the_member_fact_takes_its_links_variants_and_spend_rows(
        self, conn: aiosqlite.Connection
    ) -> None:
        ids = await populate(conn)
        await forget_member(
            conn,
            user_id=MEMBER,
            guild_id=None,
            mode=MemberDeletionMode.DELETE_FACTS,
            before=REQUEST,
        )
        assert await _scalar(conn, "SELECT COUNT(*) FROM fact_links") == 0
        assert await _scalar(conn, "SELECT COUNT(*) FROM fact_variants") == 0
        assert await _scalar(conn, "SELECT COUNT(*) FROM variant_calls") == 0
        # The other member's candidate pointed at the deleted fact: the hint and
        # the model's reasoning (which may quote it) are gone, the candidate stays.
        row = await (
            await conn.execute(
                "SELECT similar_fact_id, similar_fact_score, relationship, relationship_reasoning "
                "FROM pending_facts WHERE id = ?",
                (ids["a_other_candidate"],),
            )
        ).fetchone()
        assert row == (None, None, None, None)

    async def test_a_member_in_two_servers_can_limit_it_to_one(
        self, conn: aiosqlite.Connection
    ) -> None:
        ids = await populate(conn)
        await forget_member(
            conn,
            user_id=MEMBER,
            guild_id=GUILD_A,
            mode=MemberDeletionMode.DELETE_FACTS,
            before=REQUEST,
        )
        remaining = await _fact_ids(conn)
        assert ids["a_member_fact"] not in remaining
        assert ids["b_member_fact"] in remaining
        assert (
            await _scalar(
                conn,
                "SELECT COUNT(*) FROM ask_calls WHERE user_id = ? AND guild_id = ?",
                MEMBER,
                GUILD_B,
            )
            == 1
        )
        assert (
            await _scalar(
                conn,
                "SELECT COUNT(*) FROM pending_facts WHERE source_author_id = ? AND guild_id = ?",
                MEMBER,
                GUILD_B,
            )
            == 1
        )

    async def test_moderator_ids_become_zero_only_for_that_person(
        self, conn: aiosqlite.Connection
    ) -> None:
        await populate(conn)
        counts = await forget_member(
            conn,
            user_id=MODERATOR,
            guild_id=None,
            mode=MemberDeletionMode.DELETE_FACTS,
            before=REQUEST,
        )
        for table, column in (
            ("proactive_channel_config", "updated_by_id"),
            ("extraction_channel_config", "updated_by_id"),
            ("digest_config", "updated_by_id"),
            ("onboarding_config", "updated_by_id"),
            ("backfill_runs", "requested_by_id"),
            ("pending_facts", "resolved_by_id"),
        ):
            assert (
                await _scalar(conn, f"SELECT COUNT(*) FROM {table} WHERE {column} = ?", MODERATOR)
                == 0
            ), table
        assert counts.rows["proactive_channel_config.updated_by_id_removed"] == 2
        assert "facts" not in counts.rows  # the moderator wrote no fact

    async def test_another_members_id_never_matches(self, conn: aiosqlite.Connection) -> None:
        await populate(conn)
        stranger = 900_000_000_000_000_099
        before = await snapshot(conn)
        counts = await forget_member(
            conn,
            user_id=stranger,
            guild_id=None,
            mode=MemberDeletionMode.DELETE_FACTS,
            before=REQUEST,
        )
        assert counts.total == 0
        assert await snapshot(conn) == before

    @pytest.mark.parametrize("user_id", [0, -1])
    async def test_a_non_id_is_refused(self, conn: aiosqlite.Connection, user_id: int) -> None:
        await populate(conn)
        with pytest.raises(ValueError):
            await forget_member(
                conn,
                user_id=user_id,
                guild_id=None,
                mode=MemberDeletionMode.DELETE_FACTS,
                before=REQUEST,
            )


class TestForgetMemberUnlinking:
    async def test_the_facts_stay_without_their_link_or_author(
        self, conn: aiosqlite.Connection
    ) -> None:
        ids = await populate(conn)
        counts = await forget_member(
            conn, user_id=MEMBER, guild_id=None, mode=MemberDeletionMode.UNLINK, before=REQUEST
        )
        fact = await get_fact_by_id(conn, guild_id=GUILD_A, fact_id=ids["a_member_fact"])
        assert fact is not None
        assert (fact.channel_id, fact.message_id) == (0, 0)
        assert source_link(fact) == f"https://discord.com/channels/{GUILD_A}"
        assert (
            await _scalar(
                conn, "SELECT source_author_id FROM facts WHERE id = ?", ids["a_member_fact"]
            )
            is None
        )
        assert counts.rows["facts.unlinked"] == 2
        assert "facts" not in counts.rows
        # Candidates from the member's messages always go.
        assert counts.rows["pending_facts"] == 2

    async def test_a_second_run_finds_nothing(self, conn: aiosqlite.Connection) -> None:
        await populate(conn)
        await forget_member(
            conn, user_id=MEMBER, guild_id=None, mode=MemberDeletionMode.UNLINK, before=REQUEST
        )
        again = await forget_member(
            conn, user_id=MEMBER, guild_id=None, mode=MemberDeletionMode.UNLINK, before=REQUEST
        )
        assert again.total == 0


class TestDryRuns:
    @pytest.mark.parametrize("mode", list(MemberDeletionMode))
    async def test_member_dry_run_counts_equal_the_real_run_and_change_nothing(
        self, conn: aiosqlite.Connection, mode: MemberDeletionMode
    ) -> None:
        await populate(conn)
        before = await snapshot(conn)
        dry = await forget_member(
            conn, user_id=MEMBER, guild_id=None, mode=mode, before=REQUEST, dry_run=True
        )
        assert await snapshot(conn) == before
        real = await forget_member(conn, user_id=MEMBER, guild_id=None, mode=mode, before=REQUEST)
        assert dry == real
        assert real.total > 0

    async def test_purge_dry_run_counts_equal_the_real_purge(
        self, conn: aiosqlite.Connection
    ) -> None:
        await populate(conn)
        before = await snapshot(conn)
        dry = await purge_guild(conn, guild_id=GUILD_A, before=REQUEST, dry_run=True)
        assert await snapshot(conn) == before
        real = await purge_guild(conn, guild_id=GUILD_A, before=REQUEST)
        assert dry == real

    async def test_retention_dry_run_counts_equal_the_real_run(
        self, conn: aiosqlite.Connection
    ) -> None:
        await populate(conn)
        policy = RetentionPolicy(
            proactive_signal_days=1, ask_member_id_days=1, onboarding_send_days=1
        )
        before = await snapshot(conn)
        dry = await apply_retention(conn, now=REQUEST, policy=policy, dry_run=True)
        assert await snapshot(conn) == before
        assert dry == await apply_retention(conn, now=REQUEST, policy=policy)


class TestPurgeGuild:
    async def test_every_row_of_the_server_goes_except_billing(
        self, conn: aiosqlite.Connection
    ) -> None:
        await populate(conn)
        before = await snapshot(conn)
        tables = (
            "facts",
            "pending_facts",
            "proactive_signals",
            "proactive_escalations",
            "proactive_channel_config",
            "extraction_channel_config",
            "extraction_queue",
            "extraction_calls",
            "supersession_calls",
            "variant_calls",
            "ask_calls",
            "digest_config",
            "digest_runs",
            "onboarding_config",
            "onboarding_sends",
            "backfill_runs",
            "backfill_calls",
        )
        other_server = {
            table: await _scalar(conn, f"SELECT COUNT(*) FROM {table} WHERE guild_id = ?", GUILD_B)
            for table in tables
        }
        await purge_guild(conn, guild_id=GUILD_A, before=AFTER + timedelta(days=1))
        after = await snapshot(conn)
        for table in BILLING_TABLES:
            assert after[table] == before[table]
        for table in tables:
            assert (
                await _scalar(conn, f"SELECT COUNT(*) FROM {table} WHERE guild_id = ?", GUILD_A)
                == 0
            ), table
            assert (
                await _scalar(conn, f"SELECT COUNT(*) FROM {table} WHERE guild_id = ?", GUILD_B)
                == other_server[table]
            ), table

    async def test_the_other_server_is_byte_for_byte_untouched(
        self, conn: aiosqlite.Connection
    ) -> None:
        await populate(conn)
        reference = await open_database()
        try:
            await populate(reference)
            await purge_guild(conn, guild_id=GUILD_A, before=AFTER + timedelta(days=1))
            for table in ("facts", "pending_facts", "ask_calls", "onboarding_sends"):
                async with conn.execute(
                    f"SELECT COUNT(*) FROM {table} WHERE guild_id = ?", (GUILD_B,)
                ) as cursor:
                    kept = (await cursor.fetchone())[0]  # type: ignore[index]
                async with reference.execute(
                    f"SELECT COUNT(*) FROM {table} WHERE guild_id = ?", (GUILD_B,)
                ) as cursor:
                    original = (await cursor.fetchone())[0]  # type: ignore[index]
                assert kept == original, table
            assert await _scalar(conn, "SELECT COUNT(*) FROM fact_links") == 1
        finally:
            await reference.close()

    async def test_data_from_after_the_moment_survives(self, conn: aiosqlite.Connection) -> None:
        ids = await populate(conn)
        await purge_guild(conn, guild_id=GUILD_A, before=REQUEST)
        assert await _fact_ids(conn) >= {ids["a_member_late_fact"]}
        assert ids["a_member_fact"] not in await _fact_ids(conn)

    async def test_a_fact_from_an_old_message_recorded_later_still_goes(
        self, conn: aiosqlite.Connection
    ) -> None:
        # A batch in flight while the server was purged: the message is old,
        # the row is new. The message's own time decides.
        fact_id = await add_fact(conn, guild_id=GUILD_A, author=OTHER, when=BEFORE)
        await conn.execute(
            "UPDATE facts SET created_at = ? WHERE id = ?", (utc_iso(AFTER), fact_id)
        )
        await conn.commit()
        await purge_guild(conn, guild_id=GUILD_A, before=REQUEST)
        assert fact_id not in await _fact_ids(conn)

    async def test_an_unlinked_fact_recorded_after_the_moment_survives(
        self, conn: aiosqlite.Connection
    ) -> None:
        # message_id 0 (link removed on request) must not read as "an old message".
        fact_id = await add_fact(conn, guild_id=GUILD_A, author=None, when=AFTER)
        await conn.execute(
            "UPDATE facts SET channel_id = 0, message_id = 0 WHERE id = ?", (fact_id,)
        )
        await conn.commit()
        await purge_guild(conn, guild_id=GUILD_A, before=REQUEST)
        assert fact_id in await _fact_ids(conn)

    async def test_a_purge_that_fails_half_way_changes_nothing(
        self, conn: aiosqlite.Connection, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        await populate(conn)
        before = await snapshot(conn)
        import aura.db.deletion as deletion

        original = deletion._run
        calls = 0

        async def failing_run(*args: object, **kwargs: object) -> None:
            nonlocal calls
            calls += 1
            if calls == 6:
                raise RuntimeError("disk full")
            await original(*args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(deletion, "_run", failing_run)
        with pytest.raises(RuntimeError):
            await purge_guild(conn, guild_id=GUILD_A, before=AFTER + timedelta(days=1))
        monkeypatch.setattr(deletion, "_run", original)
        assert await snapshot(conn) == before
        # And it can simply be run again.
        assert (await purge_guild(conn, guild_id=GUILD_A, before=AFTER + timedelta(days=1))).total

    async def test_a_second_purge_finds_nothing(self, conn: aiosqlite.Connection) -> None:
        await populate(conn)
        await purge_guild(conn, guild_id=GUILD_A, before=REQUEST)
        assert (await purge_guild(conn, guild_id=GUILD_A, before=REQUEST)).total == 0


class TestChainRepair:
    async def _chain(self, conn: aiosqlite.Connection) -> tuple[int, int, int]:
        first = await add_fact(conn, guild_id=GUILD_A, author=OTHER, when=BEFORE)
        second = await add_fact(conn, guild_id=GUILD_A, author=MEMBER, when=BEFORE)
        third = await add_fact(conn, guild_id=GUILD_A, author=OTHER, when=BEFORE)
        await supersede(conn, first, second, BEFORE)
        await supersede(conn, second, third, BEFORE)
        return first, second, third

    async def _successor(self, conn: aiosqlite.Connection, fact_id: int) -> object:
        return await _scalar(conn, "SELECT superseded_by_id FROM facts WHERE id = ?", fact_id)

    async def test_deleting_the_middle_repoints_the_predecessor(
        self, conn: aiosqlite.Connection
    ) -> None:
        first, second, third = await self._chain(conn)
        await forget_fact(conn, guild_id=GUILD_A, fact_id=second)
        assert await self._successor(conn, first) == third
        assert await _scalar(conn, "SELECT status FROM facts WHERE id = ?", first) == "superseded"

    async def test_deleting_the_current_fact_leaves_the_old_one_retired(
        self, conn: aiosqlite.Connection
    ) -> None:
        _, second, third = await self._chain(conn)
        await forget_fact(conn, guild_id=GUILD_A, fact_id=third)
        assert await self._successor(conn, second) is None
        # Never back in answers: still superseded.
        assert await _scalar(conn, "SELECT status FROM facts WHERE id = ?", second) == "superseded"

    async def test_deleting_two_links_in_a_row_skips_both(self, conn: aiosqlite.Connection) -> None:
        first, _, third = await self._chain(conn)  # the middle one is the member's
        fourth = await add_fact(conn, guild_id=GUILD_A, author=OTHER, when=BEFORE)
        await supersede(conn, third, fourth, BEFORE)
        await conn.execute("UPDATE facts SET source_author_id = ? WHERE id = ?", (MEMBER, third))
        await conn.commit()
        await forget_member(
            conn,
            user_id=MEMBER,
            guild_id=GUILD_A,
            mode=MemberDeletionMode.DELETE_FACTS,
            before=REQUEST,
        )
        assert await self._successor(conn, first) == fourth

    async def test_a_hand_made_cycle_ends_instead_of_looping(
        self, conn: aiosqlite.Connection
    ) -> None:
        first = await add_fact(conn, guild_id=GUILD_A, author=OTHER, when=BEFORE)
        second = await add_fact(conn, guild_id=GUILD_A, author=MEMBER, when=BEFORE)
        third = await add_fact(conn, guild_id=GUILD_A, author=MEMBER, when=BEFORE)
        await supersede(conn, first, second, BEFORE)
        await supersede(conn, second, third, BEFORE)
        await supersede(conn, third, second, BEFORE)  # a cycle nobody should make
        await asyncio.wait_for(
            forget_member(
                conn,
                user_id=MEMBER,
                guild_id=GUILD_A,
                mode=MemberDeletionMode.DELETE_FACTS,
                before=REQUEST,
            ),
            timeout=5,
        )
        assert await self._successor(conn, first) is None


class TestForgetFact:
    async def test_another_servers_fact_is_never_matched(self, conn: aiosqlite.Connection) -> None:
        ids = await populate(conn)
        counts = await forget_fact(conn, guild_id=GUILD_B, fact_id=ids["a_member_fact"])
        assert counts.total == 0
        assert ids["a_member_fact"] in await _fact_ids(conn)

    async def test_a_different_timestamp_is_not_the_same_fact(
        self, conn: aiosqlite.Connection
    ) -> None:
        ids = await populate(conn)
        counts = await forget_fact(
            conn, guild_id=GUILD_A, fact_id=ids["a_member_fact"], created_at="2001-01-01T00:00:00"
        )
        assert counts.total == 0

    async def test_the_candidate_it_was_confirmed_from_goes_too(
        self, conn: aiosqlite.Connection
    ) -> None:
        from tests.privacy_data import add_candidate

        fact_id = await add_fact(conn, guild_id=GUILD_A, author=OTHER, when=BEFORE)
        candidate = await add_candidate(
            conn, guild_id=GUILD_A, author=OTHER, when=BEFORE, confirmed_fact_id=fact_id
        )
        counts = await forget_fact(conn, guild_id=GUILD_A, fact_id=fact_id)
        assert counts.rows == {"facts": 1, "pending_facts": 1, "supersession_calls": 1}
        assert (
            await _scalar(conn, "SELECT COUNT(*) FROM pending_facts WHERE id = ?", candidate) == 0
        )


class TestRetention:
    async def test_each_rule_respects_its_period(self, conn: aiosqlite.Connection) -> None:
        await populate(conn)  # every row dated BEFORE = REQUEST - 3 days
        keep_all = RetentionPolicy(
            proactive_signal_days=4, ask_member_id_days=4, onboarding_send_days=4
        )
        assert (await apply_retention(conn, now=REQUEST, policy=keep_all)).total == 0
        drop_all = RetentionPolicy(
            proactive_signal_days=2, ask_member_id_days=2, onboarding_send_days=2
        )
        counts = await apply_retention(conn, now=REQUEST, policy=drop_all)
        assert counts.rows == {
            "proactive_signals": 2,
            "proactive_escalations": 2,
            "ask_calls.anonymized": 4,
            "onboarding_sends": 4,
        }
        assert await _scalar(conn, "SELECT COUNT(*) FROM ask_calls") == 4

    @pytest.mark.parametrize(
        "field", ["proactive_signal_days", "ask_member_id_days", "onboarding_send_days"]
    )
    def test_a_period_below_one_day_is_refused(self, field: str) -> None:
        values = {"proactive_signal_days": 1, "ask_member_id_days": 1, "onboarding_send_days": 1}
        values[field] = 0
        with pytest.raises(ValueError):
            RetentionPolicy(**values)

    async def test_todays_rows_are_never_touched(self, conn: aiosqlite.Connection) -> None:
        await populate(conn)
        policy = RetentionPolicy(
            proactive_signal_days=1, ask_member_id_days=1, onboarding_send_days=1
        )
        counts = await apply_retention(conn, now=BEFORE + timedelta(hours=23), policy=policy)
        assert counts.total == 0


class TestConcurrency:
    async def test_a_deletion_and_new_writes_interleave_safely(
        self, conn: aiosqlite.Connection
    ) -> None:
        await populate(conn)

        async def keep_writing() -> None:
            # The production write path, which holds the connection lock.
            for _ in range(20):
                await create_fact(
                    conn,
                    guild_id=GUILD_A,
                    channel_id=1,
                    message_id=snowflake(AFTER),
                    content="written while the deletion runs",
                    embedding=bytes(384 * 4),
                    source_author_id=MEMBER,
                )

        await asyncio.gather(
            keep_writing(),
            forget_member(
                conn,
                user_id=MEMBER,
                guild_id=None,
                mode=MemberDeletionMode.DELETE_FACTS,
                before=REQUEST,
            ),
        )
        # Everything written after the request survives; nothing from before does.
        assert (
            await _scalar(
                conn,
                "SELECT COUNT(*) FROM facts WHERE source_author_id = ? AND message_id <= ?",
                MEMBER,
                latest_snowflake_at(REQUEST),
            )
            == 0
        )
        assert (
            await _scalar(conn, "SELECT COUNT(*) FROM facts WHERE source_author_id = ?", MEMBER)
            == 22
        )

    async def test_two_requests_at_once_delete_once(self, conn: aiosqlite.Connection) -> None:
        await populate(conn)
        first, second = await asyncio.gather(
            forget_member(
                conn,
                user_id=MEMBER,
                guild_id=None,
                mode=MemberDeletionMode.DELETE_FACTS,
                before=REQUEST,
            ),
            forget_member(
                conn,
                user_id=MEMBER,
                guild_id=None,
                mode=MemberDeletionMode.DELETE_FACTS,
                before=REQUEST,
            ),
        )
        assert first.total > 0
        assert second.total == 0


class TestGuildsWithData:
    async def test_a_server_known_only_from_billing_is_not_listed(
        self, conn: aiosqlite.Connection
    ) -> None:
        from tests.privacy_data import add_billing_rows

        await add_billing_rows(conn, guild_id=333)
        await add_fact(conn, guild_id=GUILD_A, author=OTHER, when=BEFORE)
        assert await guilds_with_data(conn) == {GUILD_A}


class TestCounts:
    def test_merging_adds_and_drops_zeros(self) -> None:
        merged = DeletionCounts(rows={"facts": 1}).merged(DeletionCounts(rows={"facts": 2, "x": 1}))
        assert merged.rows == {"facts": 3, "x": 1}
        assert merged.total == 4
        assert DeletionCounts().summary() == "nothing"
        assert merged.summary() == "facts=3, x=1"


class TestSecureDelete:
    async def test_the_main_connection_overwrites_deleted_bytes(
        self, conn: aiosqlite.Connection
    ) -> None:
        assert await _scalar(conn, "PRAGMA secure_delete") == 1


def _fact(message_id: int) -> Fact:
    from datetime import UTC, datetime

    return Fact(
        id=1,
        guild_id=GUILD_A,
        channel_id=5,
        message_id=message_id,
        content="x",
        embedding=b"",
        status=FactStatus.ACTIVE,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


class TestSourceLink:
    def test_a_normal_fact_links_its_message(self) -> None:
        assert source_link(_fact(7)) == f"https://discord.com/channels/{GUILD_A}/5/7"

    def test_a_fact_whose_link_was_removed_links_the_server(self) -> None:
        assert source_link(_fact(0)) == f"https://discord.com/channels/{GUILD_A}"


class TestWalIsCleared:
    async def test_a_deletion_leaves_no_copy_of_the_text_in_the_wal(self, tmp_path: Path) -> None:
        path = tmp_path / "aura.db"
        conn = await open_database(str(path))
        try:
            canary = "CANARY-p7a-wal-55ee"
            await add_fact(conn, guild_id=GUILD_A, author=MEMBER, when=BEFORE, content=canary)
            await forget_member(
                conn,
                user_id=MEMBER,
                guild_id=None,
                mode=MemberDeletionMode.DELETE_FACTS,
                before=REQUEST,
            )
            wal = Path(f"{path}-wal")
            wal_exists = await asyncio.to_thread(wal.exists)
            wal_bytes = await asyncio.to_thread(wal.read_bytes) if wal_exists else b""
            assert canary.encode() not in wal_bytes
            assert canary.encode() not in await asyncio.to_thread(path.read_bytes)
        finally:
            await conn.close()
