"""Tests for aura.digest.scheduler: which guilds are due, and what actually posts.

End to end over a real in-memory database and a fake gateway, with `now`
injected at every call -- so a six-week outage, a clock jumping backwards and
two guilds on different cadences are all one function call rather than a wait.

The fake gateway is the only stand-in. It records what was sent and can be told
to fail in each of the ways a real one does (channel gone, send raises, wrong
guild), which is what makes the release-and-retry behaviour observable at all:
the interesting property is not "a digest was posted" but "the window is still
open afterwards".
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import cast
from unittest.mock import patch

import aiosqlite
import discord
import pytest

from aura.config import Settings
from aura.db.connection import utc_iso
from aura.db.digest_config import DigestConfig, set_digest_config
from aura.db.digest_state import DigestRunOutcome, get_digest_runs
from aura.db.repository import init_schema
from aura.digest.intervals import DigestInterval
from aura.digest.scheduler import run_digest_scheduler, send_due_digests, window_start

GUILD_A = 100000000000000001
GUILD_B = 200000000000000002
CHANNEL_A = 300000000000000003
CHANNEL_B = 400000000000000004
MODERATOR = 4242

NOW = datetime(2026, 8, 16, 12, 0, 0, tzinfo=timezone.utc)
WEEK = int(DigestInterval.WEEKLY)
DAY = int(DigestInterval.DAILY)

_next_message_id = iter(range(500000000000000000, 500000000000001000))


@dataclass
class SentMessage:
    channel_id: int
    embed: discord.Embed
    allowed_mentions: discord.AllowedMentions | None


class FakeChannel:
    """The smallest thing the scheduler needs a channel to be."""

    def __init__(self, channel_id: int, guild_id: int, sink: list[SentMessage], *, locale: str = "en-US") -> None:
        self.id = channel_id
        self.guild = FakeGuild(guild_id, locale)
        self._sink = sink
        self.raises: Exception | None = None

    async def send(self, *, embed: discord.Embed, allowed_mentions=None) -> None:
        if self.raises is not None:
            raise self.raises
        self._sink.append(SentMessage(self.id, embed, allowed_mentions))


class FakeGuild:
    def __init__(self, guild_id: int, locale: str) -> None:
        self.id = guild_id
        self.preferred_locale = locale


@dataclass
class FakeGateway:
    """A DigestGateway that resolves from a dict and records every send."""

    channels: dict[int, FakeChannel] = field(default_factory=dict)
    sent: list[SentMessage] = field(default_factory=list)
    resolve_calls: int = 0

    def add_channel(self, channel_id: int, guild_id: int, *, locale: str = "en-US") -> FakeChannel:
        channel = FakeChannel(channel_id, guild_id, self.sent, locale=locale)
        self.channels[channel_id] = channel
        return channel

    async def resolve_channel(self, channel_id: int) -> discord.TextChannel | None:
        self.resolve_calls += 1
        # A real TextChannel cannot be constructed without a gateway
        # connection, and the scheduler only ever touches `.guild` and
        # `.send()` -- so the stand-in is cast rather than faked in full.
        return cast("discord.TextChannel | None", self.channels.get(channel_id))


@pytest.fixture
async def conn():
    connection = await aiosqlite.connect(":memory:")
    await init_schema(connection)
    yield connection
    await connection.close()


async def configure(
    conn: aiosqlite.Connection,
    *,
    guild_id: int = GUILD_A,
    channel_id: int = CHANNEL_A,
    interval: int = WEEK,
    enabled: bool = True,
) -> None:
    await set_digest_config(
        conn,
        guild_id=guild_id,
        channel_id=channel_id,
        interval_seconds=interval,
        enabled=enabled,
        updated_by_id=MODERATOR,
    )


async def backdate_enabled_at(
    conn: aiosqlite.Connection, *, guild_id: int = GUILD_A, moment: datetime
) -> None:
    """Move a guild's baseline into the past, as if digests had been on since then.

    set_digest_config timestamps from the clock, and every question here is
    about elapsed intervals -- so the baseline is written and then moved,
    rather than the test waiting a week.
    """
    await conn.execute(
        "UPDATE digest_config SET enabled_at = ? WHERE guild_id = ?",
        (utc_iso(moment), guild_id),
    )
    await conn.commit()


async def add_fact(
    conn: aiosqlite.Connection,
    *,
    content: str = "Movie night is on Fridays.",
    created_at: datetime,
    guild_id: int = GUILD_A,
) -> int:
    cursor = await conn.execute(
        """
        INSERT INTO facts
            (guild_id, channel_id, message_id, content, embedding, status, created_at)
        VALUES (?, ?, ?, ?, ?, 'active', ?)
        """,
        (guild_id, CHANNEL_A, next(_next_message_id), content, b"\x00\x00\x00\x00",
         utc_iso(created_at)),
    )
    await conn.commit()
    assert cursor.lastrowid is not None
    return cursor.lastrowid


async def ready_guild(
    conn: aiosqlite.Connection,
    gateway: FakeGateway,
    *,
    guild_id: int = GUILD_A,
    channel_id: int = CHANNEL_A,
    interval: int = WEEK,
    enabled_since: datetime | None = None,
    with_content: bool = True,
) -> FakeChannel:
    """A guild that is configured, past due, and (by default) has something to say."""
    await configure(conn, guild_id=guild_id, channel_id=channel_id, interval=interval)
    await backdate_enabled_at(
        conn, guild_id=guild_id, moment=enabled_since or NOW - timedelta(days=30)
    )
    if with_content:
        await add_fact(conn, created_at=NOW - timedelta(days=1), guild_id=guild_id)
    return gateway.add_channel(channel_id, guild_id)


class TestDueness:
    async def test_a_guild_mid_interval_gets_nothing(self, conn: aiosqlite.Connection) -> None:
        gateway = FakeGateway()
        await ready_guild(conn, gateway, enabled_since=NOW - timedelta(days=2))

        assert await send_due_digests(conn, gateway, now=NOW) == 0
        assert gateway.sent == []
        assert await get_digest_runs(conn, guild_id=GUILD_A, limit=5) == []

    async def test_a_guild_past_its_interval_gets_a_digest(
        self, conn: aiosqlite.Connection
    ) -> None:
        gateway = FakeGateway()
        await ready_guild(conn, gateway)

        assert await send_due_digests(conn, gateway, now=NOW) == 1

        assert len(gateway.sent) == 1
        assert gateway.sent[0].channel_id == CHANNEL_A
        assert "Movie night is on Fridays." in str(gateway.sent[0].embed.fields[0].value)

    async def test_a_second_tick_right_afterwards_posts_nothing(
        self, conn: aiosqlite.Connection
    ) -> None:
        gateway = FakeGateway()
        await ready_guild(conn, gateway)

        await send_due_digests(conn, gateway, now=NOW)
        await send_due_digests(conn, gateway, now=NOW + timedelta(hours=1))
        await send_due_digests(conn, gateway, now=NOW + timedelta(hours=6))

        assert len(gateway.sent) == 1

    async def test_the_next_digest_arrives_one_interval_later(
        self, conn: aiosqlite.Connection
    ) -> None:
        gateway = FakeGateway()
        await ready_guild(conn, gateway)
        await send_due_digests(conn, gateway, now=NOW)

        later = NOW + timedelta(days=7, minutes=1)
        await add_fact(conn, content="Something else.", created_at=later - timedelta(hours=1))

        assert await send_due_digests(conn, gateway, now=later) == 1
        assert len(gateway.sent) == 2

    async def test_a_disabled_guild_is_never_evaluated(
        self, conn: aiosqlite.Connection
    ) -> None:
        gateway = FakeGateway()
        await ready_guild(conn, gateway)
        await configure(conn, enabled=False)

        assert await send_due_digests(conn, gateway, now=NOW) == 0
        assert gateway.resolve_calls == 0

    async def test_an_unconfigured_deployment_does_nothing(
        self, conn: aiosqlite.Connection
    ) -> None:
        gateway = FakeGateway()

        assert await send_due_digests(conn, gateway, now=NOW) == 0

    async def test_a_naive_now_is_refused(self, conn: aiosqlite.Connection) -> None:
        with pytest.raises(ValueError, match="timezone-aware"):
            await send_due_digests(conn, FakeGateway(), now=NOW.replace(tzinfo=None))


class TestTheFirstDigest:
    async def test_a_freshly_enabled_guild_waits_one_interval(
        self, conn: aiosqlite.Connection
    ) -> None:
        gateway = FakeGateway()
        await configure(conn)
        gateway.add_channel(CHANNEL_A, GUILD_A)
        await add_fact(conn, created_at=NOW - timedelta(days=1))

        assert await send_due_digests(conn, gateway, now=NOW) == 0

    async def test_the_first_digest_does_not_repost_the_existing_knowledge_model(
        self, conn: aiosqlite.Connection
    ) -> None:
        # Summarizing everything Aura already knows is the onboarding trigger's
        # job. A "what's new" post opening with two years of facts is neither.
        gateway = FakeGateway()
        await add_fact(conn, content="Known long before.", created_at=NOW - timedelta(days=200))
        await configure(conn)
        await backdate_enabled_at(conn, moment=NOW - timedelta(days=8))
        gateway.add_channel(CHANNEL_A, GUILD_A)

        assert await send_due_digests(conn, gateway, now=NOW) == 0
        assert gateway.sent == []

    async def test_re_enabling_after_a_pause_does_not_replay_the_silent_period(
        self, conn: aiosqlite.Connection
    ) -> None:
        gateway = FakeGateway()
        await ready_guild(conn, gateway)
        await send_due_digests(conn, gateway, now=NOW)

        # Off for a month, during which the server keeps learning things.
        await configure(conn, enabled=False)
        await add_fact(conn, content="Learned while off.", created_at=NOW + timedelta(days=10))
        # Back on a month later, then a full interval passes.
        await configure(conn)
        await backdate_enabled_at(conn, moment=NOW + timedelta(days=30))

        posted = await send_due_digests(conn, gateway, now=NOW + timedelta(days=38))

        assert posted == 0
        assert len(gateway.sent) == 1  # still just the original one


class TestEmptyDigests:
    async def test_a_quiet_period_posts_nothing(self, conn: aiosqlite.Connection) -> None:
        gateway = FakeGateway()
        await ready_guild(conn, gateway, with_content=False)

        assert await send_due_digests(conn, gateway, now=NOW) == 0
        assert gateway.sent == []
        assert gateway.resolve_calls == 0  # not even looked up

    async def test_a_quiet_period_still_advances_the_schedule(
        self, conn: aiosqlite.Connection
    ) -> None:
        # Otherwise the guild stays permanently due, and the next fact to land
        # would trigger a digest within the hour instead of on the cadence.
        gateway = FakeGateway()
        await ready_guild(conn, gateway, with_content=False)
        await send_due_digests(conn, gateway, now=NOW)

        runs = await get_digest_runs(conn, guild_id=GUILD_A, limit=5)
        assert [run.outcome for run in runs] == [DigestRunOutcome.SKIPPED_EMPTY]

        await add_fact(conn, content="Something new.", created_at=NOW + timedelta(hours=1))
        assert await send_due_digests(conn, gateway, now=NOW + timedelta(hours=2)) == 0

    async def test_the_skipped_window_is_covered_by_the_following_digest(
        self, conn: aiosqlite.Connection
    ) -> None:
        gateway = FakeGateway()
        await ready_guild(conn, gateway, with_content=False)
        await send_due_digests(conn, gateway, now=NOW)

        await add_fact(conn, content="Arrived during the quiet week.", created_at=NOW + timedelta(hours=1))

        assert await send_due_digests(conn, gateway, now=NOW + timedelta(days=7, minutes=1)) == 1
        assert "Arrived during the quiet week." in str(gateway.sent[0].embed.fields[0].value)


class TestDowntime:
    """The brief's first attack: a restart across a missed window."""

    async def test_a_missed_window_produces_exactly_one_catch_up(
        self, conn: aiosqlite.Connection
    ) -> None:
        gateway = FakeGateway()
        await ready_guild(conn, gateway)
        await send_due_digests(conn, gateway, now=NOW)

        # The process is down for six weeks. Facts keep being confirmed by
        # moderators using slash commands... which also needs the process, so
        # in practice they arrive right after it returns; either way they fall
        # in the missed window.
        for week in range(1, 6):
            await add_fact(conn, content=f"week {week}", created_at=NOW + timedelta(weeks=week))
        back_up = NOW + timedelta(weeks=6)

        posted = await send_due_digests(conn, gateway, now=back_up)

        assert posted == 1
        assert len(gateway.sent) == 2

    async def test_the_catch_up_is_not_repeated_on_the_following_ticks(
        self, conn: aiosqlite.Connection
    ) -> None:
        gateway = FakeGateway()
        await ready_guild(conn, gateway)
        await send_due_digests(conn, gateway, now=NOW)
        for week in range(1, 6):
            await add_fact(conn, content=f"week {week}", created_at=NOW + timedelta(weeks=week))
        back_up = NOW + timedelta(weeks=6)

        await send_due_digests(conn, gateway, now=back_up)
        for minutes in (1, 30, 90, 600):
            await send_due_digests(conn, gateway, now=back_up + timedelta(minutes=minutes))

        assert len(gateway.sent) == 2
        runs = await get_digest_runs(conn, guild_id=GUILD_A, limit=20)
        assert [run.outcome for run in runs] == [DigestRunOutcome.POSTED] * 2

    async def test_the_catch_up_covers_the_whole_outage_in_one_message(
        self, conn: aiosqlite.Connection
    ) -> None:
        # Not five weekly digests replayed, and not just the last week: one
        # digest containing everything the missed windows would have said.
        gateway = FakeGateway()
        await ready_guild(conn, gateway)
        await send_due_digests(conn, gateway, now=NOW)
        for week in range(1, 6):
            await add_fact(conn, content=f"week {week}", created_at=NOW + timedelta(weeks=week))

        await send_due_digests(conn, gateway, now=NOW + timedelta(weeks=6))

        listed = str(gateway.sent[1].embed.fields[0].value)
        for week in range(1, 6):
            assert f"week {week}" in listed

    async def test_a_missed_window_with_no_content_is_also_caught_up_once(
        self, conn: aiosqlite.Connection
    ) -> None:
        gateway = FakeGateway()
        await ready_guild(conn, gateway, with_content=False)

        await send_due_digests(conn, gateway, now=NOW + timedelta(weeks=6))
        await send_due_digests(conn, gateway, now=NOW + timedelta(weeks=6, hours=1))

        runs = await get_digest_runs(conn, guild_id=GUILD_A, limit=10)
        assert [run.outcome for run in runs] == [DigestRunOutcome.SKIPPED_EMPTY]


class TestTwoGuilds:
    """The brief's second attack: different cadences must not interfere."""

    async def test_two_guilds_on_different_intervals_fire_independently(
        self, conn: aiosqlite.Connection
    ) -> None:
        gateway = FakeGateway()
        await ready_guild(conn, gateway, guild_id=GUILD_A, channel_id=CHANNEL_A, interval=DAY)
        await ready_guild(conn, gateway, guild_id=GUILD_B, channel_id=CHANNEL_B, interval=WEEK)

        # Both start out overdue (enabled 30 days ago), so both fire once...
        assert await send_due_digests(conn, gateway, now=NOW) == 2

        # ...and then the daily one alone comes due the next day.
        await add_fact(conn, content="A day later.", created_at=NOW + timedelta(hours=20), guild_id=GUILD_A)
        await add_fact(conn, content="Also a day later.", created_at=NOW + timedelta(hours=20), guild_id=GUILD_B)
        assert await send_due_digests(conn, gateway, now=NOW + timedelta(days=1, minutes=1)) == 1
        assert gateway.sent[-1].channel_id == CHANNEL_A

    async def test_one_guilds_digest_never_contains_anothers_facts(
        self, conn: aiosqlite.Connection
    ) -> None:
        gateway = FakeGateway()
        await ready_guild(conn, gateway, guild_id=GUILD_A, channel_id=CHANNEL_A)
        await ready_guild(conn, gateway, guild_id=GUILD_B, channel_id=CHANNEL_B)
        await add_fact(conn, content="Guild A secret.", created_at=NOW - timedelta(days=1), guild_id=GUILD_A)
        await add_fact(conn, content="Guild B secret.", created_at=NOW - timedelta(days=1), guild_id=GUILD_B)

        await send_due_digests(conn, gateway, now=NOW)

        by_channel = {message.channel_id: str(message.embed.fields[0].value) for message in gateway.sent}
        assert "Guild A secret." in by_channel[CHANNEL_A]
        assert "Guild B secret." not in by_channel[CHANNEL_A]
        assert "Guild B secret." in by_channel[CHANNEL_B]
        assert "Guild A secret." not in by_channel[CHANNEL_B]

    async def test_one_guild_failing_does_not_stop_the_others(
        self, conn: aiosqlite.Connection, caplog: pytest.LogCaptureFixture
    ) -> None:
        gateway = FakeGateway()
        await ready_guild(conn, gateway, guild_id=GUILD_A, channel_id=CHANNEL_A)
        broken = await ready_guild(conn, gateway, guild_id=GUILD_B, channel_id=CHANNEL_B)
        broken.raises = RuntimeError("Discord is having a day")

        with caplog.at_level(logging.WARNING):
            posted = await send_due_digests(conn, gateway, now=NOW)

        assert posted == 1
        assert [message.channel_id for message in gateway.sent] == [CHANNEL_A]

    async def test_a_corrupt_config_row_does_not_stop_the_others(
        self, conn: aiosqlite.Connection
    ) -> None:
        gateway = FakeGateway()
        await ready_guild(conn, gateway, guild_id=GUILD_A, channel_id=CHANNEL_A)
        await ready_guild(conn, gateway, guild_id=GUILD_B, channel_id=CHANNEL_B)
        await conn.execute(
            "UPDATE digest_config SET interval_seconds = 0 WHERE guild_id = ?", (GUILD_A,)
        )
        await conn.commit()

        assert await send_due_digests(conn, gateway, now=NOW) == 1
        assert [message.channel_id for message in gateway.sent] == [CHANNEL_B]


class TestPostFailures:
    async def test_an_unresolvable_channel_leaves_the_window_open_and_unclaimed(
        self, conn: aiosqlite.Connection, caplog: pytest.LogCaptureFixture
    ) -> None:
        # Resolving happens before the claim, so this common (and often
        # permanent) failure writes no bookkeeping row at all -- rather than
        # writing and un-writing one every hour for as long as it lasts.
        gateway = FakeGateway()
        await ready_guild(conn, gateway)
        del gateway.channels[CHANNEL_A]  # deleted, or Aura's access revoked

        with caplog.at_level(logging.WARNING):
            assert await send_due_digests(conn, gateway, now=NOW) == 0

        assert await get_digest_runs(conn, guild_id=GUILD_A, limit=5) == []
        assert any("unavailable" in record.getMessage() for record in caplog.records)

    async def test_a_released_window_is_retried_on_the_next_tick(
        self, conn: aiosqlite.Connection
    ) -> None:
        gateway = FakeGateway()
        channel = await ready_guild(conn, gateway)
        del gateway.channels[CHANNEL_A]
        await send_due_digests(conn, gateway, now=NOW)

        # A moderator fixes the channel; the next hourly tick must post the same
        # window rather than skipping a week's worth of changes.
        gateway.channels[CHANNEL_A] = channel
        assert await send_due_digests(conn, gateway, now=NOW + timedelta(hours=1)) == 1
        assert len(gateway.sent) == 1
        assert "Movie night is on Fridays." in str(gateway.sent[0].embed.fields[0].value)

    async def test_a_send_that_raises_releases_the_window(
        self, conn: aiosqlite.Connection, caplog: pytest.LogCaptureFixture
    ) -> None:
        gateway = FakeGateway()
        channel = await ready_guild(conn, gateway)
        channel.raises = RuntimeError("Discord returned 503")

        with caplog.at_level(logging.ERROR):
            assert await send_due_digests(conn, gateway, now=NOW) == 0

        runs = await get_digest_runs(conn, guild_id=GUILD_A, limit=5)
        assert [run.outcome for run in runs] == [DigestRunOutcome.POST_FAILED]

        channel.raises = None
        assert await send_due_digests(conn, gateway, now=NOW + timedelta(hours=1)) == 1

    async def test_a_permanently_broken_channel_does_not_accumulate_bookkeeping(
        self, conn: aiosqlite.Connection
    ) -> None:
        # Ten hours of retrying an unresolvable channel must cost ten log lines
        # and nothing else -- no row per tick piling up forever in a guild Aura
        # may no longer even be a member of.
        gateway = FakeGateway()
        await ready_guild(conn, gateway)
        del gateway.channels[CHANNEL_A]

        for hour in range(10):
            await send_due_digests(conn, gateway, now=NOW + timedelta(hours=hour))

        assert gateway.sent == []
        assert await get_digest_runs(conn, guild_id=GUILD_A, limit=50) == []

    async def test_a_channel_that_keeps_rejecting_the_send_retries_without_posting_twice(
        self, conn: aiosqlite.Connection
    ) -> None:
        # The failure that DOES leave a row each tick, because it can only be
        # discovered after the window is claimed. What matters is that every
        # retry re-opens the window and none of them posts.
        gateway = FakeGateway()
        channel = await ready_guild(conn, gateway)
        channel.raises = RuntimeError("send forbidden")

        for hour in range(5):
            await send_due_digests(conn, gateway, now=NOW + timedelta(hours=hour))

        assert gateway.sent == []
        runs = await get_digest_runs(conn, guild_id=GUILD_A, limit=50)
        assert len(runs) == 5
        assert all(run.outcome is DigestRunOutcome.POST_FAILED for run in runs)

        channel.raises = None
        assert await send_due_digests(conn, gateway, now=NOW + timedelta(hours=5)) == 1
        assert len(gateway.sent) == 1

    async def test_a_channel_belonging_to_another_guild_is_refused(
        self, conn: aiosqlite.Connection, caplog: pytest.LogCaptureFixture
    ) -> None:
        # A hand-edited config row pointing at another server's channel would
        # otherwise publish this guild's whole recent knowledge model there.
        gateway = FakeGateway()
        await ready_guild(conn, gateway)
        gateway.channels[CHANNEL_A] = FakeChannel(CHANNEL_A, GUILD_B, gateway.sent)

        with caplog.at_level(logging.ERROR):
            assert await send_due_digests(conn, gateway, now=NOW) == 0

        assert gateway.sent == []
        assert any("belongs to guild" in record.getMessage() for record in caplog.records)
        # Refused before the claim, so the window is untouched: fixing the row
        # and waiting one tick delivers the digest that was never sent.
        assert await get_digest_runs(conn, guild_id=GUILD_A, limit=5) == []


class TestClockSafety:
    async def test_a_clock_jumping_backwards_stands_the_digest_down(
        self, conn: aiosqlite.Connection, caplog: pytest.LogCaptureFixture
    ) -> None:
        gateway = FakeGateway()
        await ready_guild(conn, gateway)
        await send_due_digests(conn, gateway, now=NOW)

        # An NTP correction moves the clock a year back. The window would end
        # before it starts, which must not be written.
        with caplog.at_level(logging.WARNING):
            posted = await send_due_digests(conn, gateway, now=NOW - timedelta(days=365))

        assert posted == 0
        assert any("clock" in record.getMessage() for record in caplog.records)
        assert len(await get_digest_runs(conn, guild_id=GUILD_A, limit=10)) == 1

    async def test_the_schedule_recovers_once_the_clock_is_correct_again(
        self, conn: aiosqlite.Connection
    ) -> None:
        gateway = FakeGateway()
        await ready_guild(conn, gateway)
        await send_due_digests(conn, gateway, now=NOW)
        await send_due_digests(conn, gateway, now=NOW - timedelta(days=365))

        await add_fact(conn, content="After the correction.", created_at=NOW + timedelta(days=8))
        assert await send_due_digests(conn, gateway, now=NOW + timedelta(days=9)) == 1


class TestConcurrentTicks:
    async def test_two_ticks_in_flight_at_once_post_exactly_one_digest(
        self, conn: aiosqlite.Connection
    ) -> None:
        # Reachable when a tick overruns its own interval, when a second
        # process shares the database file, or on a manual re-run. Both
        # evaluations see the same due window and build the same content; only
        # one may claim it, and only the claimer may post.
        gateway = FakeGateway()
        await ready_guild(conn, gateway)

        results = await asyncio.gather(
            *(send_due_digests(conn, gateway, now=NOW) for _ in range(5))
        )

        assert sum(results) == 1
        assert len(gateway.sent) == 1
        assert len(await get_digest_runs(conn, guild_id=GUILD_A, limit=10)) == 1


class TestCorruptBookkeeping:
    async def test_an_unparseable_stored_boundary_stops_that_guild_without_crashing(
        self, conn: aiosqlite.Connection, caplog: pytest.LogCaptureFixture
    ) -> None:
        # Only reachable by editing the database. The digest for that guild must
        # stop with a log line rather than raising through the background task.
        gateway = FakeGateway()
        await ready_guild(conn, gateway)
        await send_due_digests(conn, gateway, now=NOW)
        await conn.execute("UPDATE digest_runs SET covered_until = 'not-a-timestamp'")
        await conn.commit()

        with caplog.at_level(logging.WARNING):
            posted = await send_due_digests(conn, gateway, now=NOW + timedelta(days=30))

        assert posted == 0
        assert len(gateway.sent) == 1
        assert caplog.records

    async def test_a_corrupt_boundary_in_one_guild_does_not_stop_another(
        self, conn: aiosqlite.Connection
    ) -> None:
        gateway = FakeGateway()
        await ready_guild(conn, gateway, guild_id=GUILD_A, channel_id=CHANNEL_A)
        await ready_guild(conn, gateway, guild_id=GUILD_B, channel_id=CHANNEL_B)
        await send_due_digests(conn, gateway, now=NOW)
        await conn.execute(
            "UPDATE digest_runs SET covered_until = 'not-a-timestamp' WHERE guild_id = ?",
            (GUILD_A,),
        )
        await conn.commit()
        await add_fact(conn, content="Later fact.", created_at=NOW + timedelta(days=20), guild_id=GUILD_B)

        posted = await send_due_digests(conn, gateway, now=NOW + timedelta(days=30))

        assert posted == 1
        assert gateway.sent[-1].channel_id == CHANNEL_B


class TestWindowStart:
    def _config(self, enabled_at: datetime) -> DigestConfig:
        return DigestConfig(
            guild_id=GUILD_A,
            channel_id=CHANNEL_A,
            interval_seconds=WEEK,
            digest_enabled=True,
            enabled_at=enabled_at,
            updated_by_id=MODERATOR,
            updated_at=enabled_at,
        )

    def test_with_no_previous_run_the_baseline_is_the_start(self) -> None:
        config = self._config(NOW - timedelta(days=3))

        assert window_start(config, None) == utc_iso(NOW - timedelta(days=3))

    def test_in_steady_state_the_last_window_end_is_the_start(self) -> None:
        config = self._config(NOW - timedelta(days=90))

        assert window_start(config, utc_iso(NOW - timedelta(days=7))) == utc_iso(
            NOW - timedelta(days=7)
        )

    def test_a_baseline_newer_than_the_last_run_wins(self) -> None:
        # What happens after digests are re-enabled: the pause is not replayed.
        config = self._config(NOW - timedelta(days=1))

        assert window_start(config, utc_iso(NOW - timedelta(days=30))) == utc_iso(
            NOW - timedelta(days=1)
        )


class TestTheLoop:
    def _settings(self, interval: float = 1800.0) -> Settings:
        return Settings(_env_file=None, discord_token="fake-token", digest_check_interval_seconds=interval)  # type: ignore[call-arg]

    async def test_a_failing_sweep_does_not_end_the_loop(
        self, conn: aiosqlite.Connection, caplog: pytest.LogCaptureFixture
    ) -> None:
        # A scheduler task that exits silently leaves a bot that looks healthy
        # while its digests simply never arrive again.
        sweeps: list[datetime] = []

        async def sweep(_db, _gateway, *, now: datetime) -> int:
            sweeps.append(now)
            if len(sweeps) == 1:
                raise RuntimeError("the database was briefly locked")
            return 0

        sleeps: list[float] = []

        async def fake_sleep(seconds: float) -> None:
            sleeps.append(seconds)
            if len(sleeps) >= 3:
                raise asyncio.CancelledError

        with patch("aura.digest.scheduler.send_due_digests", sweep):
            with patch("aura.digest.scheduler.asyncio.sleep", fake_sleep):
                with caplog.at_level(logging.ERROR):
                    with pytest.raises(asyncio.CancelledError):
                        await run_digest_scheduler(
                            conn, FakeGateway(), settings=self._settings()
                        )

        assert len(sweeps) == 3  # kept going after the failure
        assert any("continuing" in record.getMessage() for record in caplog.records)

    async def test_the_configured_check_interval_is_what_it_waits(
        self, conn: aiosqlite.Connection
    ) -> None:
        sleeps: list[float] = []

        async def fake_sleep(seconds: float) -> None:
            sleeps.append(seconds)
            raise asyncio.CancelledError

        with patch("aura.digest.scheduler.asyncio.sleep", fake_sleep):
            with pytest.raises(asyncio.CancelledError):
                await run_digest_scheduler(conn, FakeGateway(), settings=self._settings(120.0))

        assert sleeps == [120.0]

    async def test_the_first_sweep_runs_before_the_first_wait(
        self, conn: aiosqlite.Connection
    ) -> None:
        # What makes a restart catch up promptly rather than up to an hour late.
        gateway = FakeGateway()
        await ready_guild(conn, gateway)

        async def fake_sleep(_seconds: float) -> None:
            raise asyncio.CancelledError

        with patch("aura.digest.scheduler.asyncio.sleep", fake_sleep):
            with pytest.raises(asyncio.CancelledError):
                await run_digest_scheduler(conn, gateway, settings=self._settings())

        assert len(gateway.sent) == 1


class TestPostedMessage:
    async def test_mentions_are_suppressed(self, conn: aiosqlite.Connection) -> None:
        # A fact's text is written by a server member. Embeds do not resolve
        # mentions, but a weekly automated post pinging @everyone must be
        # impossible by construction rather than by where the text lands.
        gateway = FakeGateway()
        await ready_guild(conn, gateway, with_content=False)
        await add_fact(conn, content="@everyone must read the rules.", created_at=NOW - timedelta(days=1))

        await send_due_digests(conn, gateway, now=NOW)

        mentions = gateway.sent[0].allowed_mentions
        assert mentions is not None
        assert mentions.everyone is False
        assert mentions.roles is False
        assert mentions.users is False

    async def test_the_digest_is_written_in_the_guilds_own_language(
        self, conn: aiosqlite.Connection
    ) -> None:
        gateway = FakeGateway()
        await configure(conn)
        await backdate_enabled_at(conn, moment=NOW - timedelta(days=30))
        await add_fact(conn, created_at=NOW - timedelta(days=1))
        gateway.add_channel(CHANNEL_A, GUILD_A, locale="de")

        await send_due_digests(conn, gateway, now=NOW)

        assert gateway.sent[0].embed.title == "Was sich auf diesem Server geändert hat"

    async def test_the_recorded_run_matches_what_was_posted(
        self, conn: aiosqlite.Connection
    ) -> None:
        gateway = FakeGateway()
        await ready_guild(conn, gateway, with_content=False)
        await add_fact(conn, content="One.", created_at=NOW - timedelta(days=2))
        await add_fact(conn, content="Two.", created_at=NOW - timedelta(days=1))

        await send_due_digests(conn, gateway, now=NOW)

        run = (await get_digest_runs(conn, guild_id=GUILD_A, limit=1))[0]
        assert run.new_fact_count == 2
        assert run.milestone_count == 0
        assert run.updated_fact_count == 0
        assert run.channel_id == CHANNEL_A
        assert run.covered_until == NOW
