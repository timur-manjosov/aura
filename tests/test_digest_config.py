"""Tests for aura.db.digest_config: the per-guild digest switch, channel and interval.

Mirrors tests/test_extraction_channel_config.py where the shape is the same (an
opt-in switch nobody configured is off, an upsert leaves one row, concurrent
writers do not corrupt it) and diverges where this table genuinely differs: it
is keyed by guild rather than by channel, it carries an interval that can be
hand-edited into nonsense, and it carries the baseline every first digest is
measured from -- which has its own rules about when it may and may not move.

A real in-memory database throughout, never a live gateway connection, per
CLAUDE.md's testing philosophy.
"""
from __future__ import annotations

import asyncio
import logging

import aiosqlite
import pytest

from aura.db.digest_config import (
    MAX_INTERVAL_SECONDS,
    MIN_INTERVAL_SECONDS,
    get_digest_config,
    get_enabled_digest_configs,
    set_digest_config,
)
from aura.db.repository import init_schema
from aura.digest.intervals import DigestInterval

GUILD_A = 100000000000000001
GUILD_B = 200000000000000002
CHANNEL_A = 300000000000000003
CHANNEL_B = 400000000000000004
MODERATOR = 4242


@pytest.fixture
async def conn():
    connection = await aiosqlite.connect(":memory:")
    await init_schema(connection)
    yield connection
    await connection.close()


async def _enable(
    conn: aiosqlite.Connection,
    *,
    guild_id: int = GUILD_A,
    channel_id: int = CHANNEL_A,
    interval: int = int(DigestInterval.WEEKLY),
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


class TestDefaultOff:
    async def test_an_unconfigured_guild_has_no_config(self, conn: aiosqlite.Connection) -> None:
        # The load-bearing default: the digest is opt-in per guild.
        assert await get_digest_config(conn, guild_id=GUILD_A) is None

    async def test_an_empty_database_has_no_enabled_guilds(
        self, conn: aiosqlite.Connection
    ) -> None:
        assert await get_enabled_digest_configs(conn) == []


class TestSetAndGet:
    async def test_enabling_records_every_field(self, conn: aiosqlite.Connection) -> None:
        await _enable(conn, interval=int(DigestInterval.DAILY))

        config = await get_digest_config(conn, guild_id=GUILD_A)

        assert config is not None
        assert config.guild_id == GUILD_A
        assert config.channel_id == CHANNEL_A
        assert config.interval_seconds == int(DigestInterval.DAILY)
        assert config.digest_enabled is True
        assert config.updated_by_id == MODERATOR

    async def test_an_enabled_guild_appears_in_the_schedulers_read(
        self, conn: aiosqlite.Connection
    ) -> None:
        await _enable(conn)

        configs = await get_enabled_digest_configs(conn)

        assert [config.guild_id for config in configs] == [GUILD_A]

    async def test_a_disabled_guild_is_kept_but_never_scheduled(
        self, conn: aiosqlite.Connection
    ) -> None:
        # Disabling must not forget the channel and interval: turning digests
        # back on should not mean choosing them again. This layer writes exactly
        # what it is handed -- carrying an unspecified option forward is the
        # slash command's job (see test_digest_command) -- so what is asserted
        # here is that a disabled row keeps holding its settings at all, rather
        # than being deleted or blanked.
        await _enable(conn, interval=int(DigestInterval.BIWEEKLY))
        await _enable(conn, interval=int(DigestInterval.BIWEEKLY), enabled=False)

        config = await get_digest_config(conn, guild_id=GUILD_A)

        assert config is not None
        assert config.digest_enabled is False
        assert config.channel_id == CHANNEL_A
        assert config.interval_seconds == int(DigestInterval.BIWEEKLY)
        assert await get_enabled_digest_configs(conn) == []

    async def test_toggling_repeatedly_leaves_exactly_one_row(
        self, conn: aiosqlite.Connection
    ) -> None:
        for enabled in (True, False, True, False, True):
            await _enable(conn, enabled=enabled)

        async with conn.execute(
            "SELECT COUNT(*) FROM digest_config WHERE guild_id = ?", (GUILD_A,)
        ) as cursor:
            assert await cursor.fetchone() == (1,)
        config = await get_digest_config(conn, guild_id=GUILD_A)
        assert config is not None and config.digest_enabled is True  # the last write wins

    async def test_changing_the_channel_keeps_the_interval_row_consistent(
        self, conn: aiosqlite.Connection
    ) -> None:
        await _enable(conn, interval=int(DigestInterval.MONTHLY))
        await _enable(conn, channel_id=CHANNEL_B, interval=int(DigestInterval.MONTHLY))

        config = await get_digest_config(conn, guild_id=GUILD_A)

        assert config is not None
        assert config.channel_id == CHANNEL_B
        assert config.interval_seconds == int(DigestInterval.MONTHLY)


class TestTheBaseline:
    """enabled_at decides where a guild's FIRST digest window starts.

    Getting this wrong is not a cosmetic bug in either direction: moved too
    eagerly, a moderator changing the interval mid-week silently discards
    everything that accumulated since the last digest; never moved, a digest
    re-enabled months later opens with all of it.
    """

    async def test_changing_settings_while_enabled_keeps_the_baseline(
        self, conn: aiosqlite.Connection
    ) -> None:
        await _enable(conn)
        original = await get_digest_config(conn, guild_id=GUILD_A)
        assert original is not None

        await _enable(conn, channel_id=CHANNEL_B, interval=int(DigestInterval.DAILY))

        updated = await get_digest_config(conn, guild_id=GUILD_A)
        assert updated is not None
        assert updated.enabled_at == original.enabled_at
        # ...while the diagnostic "last touched" timestamp does move.
        assert updated.updated_at >= original.updated_at

    async def test_re_enabling_after_a_pause_resets_the_baseline(
        self, conn: aiosqlite.Connection
    ) -> None:
        await _enable(conn)
        original = await get_digest_config(conn, guild_id=GUILD_A)
        assert original is not None

        await _enable(conn, enabled=False)
        await asyncio.sleep(0.01)
        await _enable(conn)

        resumed = await get_digest_config(conn, guild_id=GUILD_A)
        assert resumed is not None
        assert resumed.enabled_at > original.enabled_at

    async def test_disabling_also_resets_the_baseline_for_next_time(
        self, conn: aiosqlite.Connection
    ) -> None:
        # Not observable while off, but it is what makes the reset above
        # unconditional rather than dependent on which write came last.
        await _enable(conn)
        first = await get_digest_config(conn, guild_id=GUILD_A)
        assert first is not None

        await asyncio.sleep(0.01)
        await _enable(conn, enabled=False)

        off = await get_digest_config(conn, guild_id=GUILD_A)
        assert off is not None
        assert off.enabled_at > first.enabled_at


class TestIntervalValidation:
    @pytest.mark.parametrize(
        "interval",
        [0, -1, -MAX_INTERVAL_SECONDS, MIN_INTERVAL_SECONDS - 1, MAX_INTERVAL_SECONDS + 1],
    )
    async def test_an_out_of_range_interval_is_refused_at_the_write(
        self, conn: aiosqlite.Connection, interval: int
    ) -> None:
        with pytest.raises(ValueError, match="interval_seconds"):
            await _enable(conn, interval=interval)

        assert await get_digest_config(conn, guild_id=GUILD_A) is None

    @pytest.mark.parametrize("interval", [MIN_INTERVAL_SECONDS, MAX_INTERVAL_SECONDS])
    async def test_the_range_boundaries_themselves_are_accepted(
        self, conn: aiosqlite.Connection, interval: int
    ) -> None:
        await _enable(conn, interval=interval)

        config = await get_digest_config(conn, guild_id=GUILD_A)
        assert config is not None and config.interval_seconds == interval

    @pytest.mark.parametrize("interval", list(DigestInterval))
    async def test_every_offered_choice_is_within_range(
        self, conn: aiosqlite.Connection, interval: DigestInterval
    ) -> None:
        # A cadence the slash command offers but the writer rejects would be a
        # command that fails for one of its own options.
        await _enable(conn, interval=int(interval))

        config = await get_digest_config(conn, guild_id=GUILD_A)
        assert config is not None and config.interval_seconds == int(interval)

    async def test_a_hand_edited_interval_is_skipped_loudly_not_guessed_at(
        self, conn: aiosqlite.Connection, caplog: pytest.LogCaptureFixture
    ) -> None:
        # The one way an impossible interval can exist: someone edited the
        # database. Scheduling anything for it would post on a cadence no
        # moderator chose -- an interval of 0 would post every single tick.
        await _enable(conn)
        await conn.execute("UPDATE digest_config SET interval_seconds = 0")
        await conn.commit()

        with caplog.at_level(logging.WARNING):
            configs = await get_enabled_digest_configs(conn)

        assert configs == []
        assert any(str(GUILD_A) in record.getMessage() for record in caplog.records)

    async def test_one_corrupt_guild_does_not_hide_the_healthy_ones(
        self, conn: aiosqlite.Connection
    ) -> None:
        await _enable(conn, guild_id=GUILD_A)
        await _enable(conn, guild_id=GUILD_B, channel_id=CHANNEL_B)
        await conn.execute(
            "UPDATE digest_config SET interval_seconds = -5 WHERE guild_id = ?", (GUILD_A,)
        )
        await conn.commit()

        configs = await get_enabled_digest_configs(conn)

        assert [config.guild_id for config in configs] == [GUILD_B]


class TestGuildIsolation:
    async def test_two_guilds_keep_entirely_separate_settings(
        self, conn: aiosqlite.Connection
    ) -> None:
        await _enable(conn, guild_id=GUILD_A, channel_id=CHANNEL_A, interval=int(DigestInterval.DAILY))
        await _enable(
            conn, guild_id=GUILD_B, channel_id=CHANNEL_B, interval=int(DigestInterval.MONTHLY)
        )

        first = await get_digest_config(conn, guild_id=GUILD_A)
        second = await get_digest_config(conn, guild_id=GUILD_B)

        assert first is not None and second is not None
        assert (first.channel_id, first.interval_seconds) == (
            CHANNEL_A,
            int(DigestInterval.DAILY),
        )
        assert (second.channel_id, second.interval_seconds) == (
            CHANNEL_B,
            int(DigestInterval.MONTHLY),
        )

    async def test_disabling_one_guild_leaves_the_other_scheduled(
        self, conn: aiosqlite.Connection
    ) -> None:
        await _enable(conn, guild_id=GUILD_A)
        await _enable(conn, guild_id=GUILD_B, channel_id=CHANNEL_B)

        await _enable(conn, guild_id=GUILD_A, enabled=False)

        assert [config.guild_id for config in await get_enabled_digest_configs(conn)] == [GUILD_B]


class TestConcurrency:
    async def test_concurrent_writes_leave_exactly_one_coherent_row(
        self, conn: aiosqlite.Connection
    ) -> None:
        # Two moderators running /aura-digest at the same moment. Whichever
        # lands last wins, but there must be exactly one row and it must hold
        # one writer's values rather than a mix of both.
        await asyncio.gather(
            *(
                _enable(conn, channel_id=channel, interval=int(DigestInterval.DAILY))
                for channel in (CHANNEL_A, CHANNEL_B)
            )
        )

        async with conn.execute("SELECT COUNT(*) FROM digest_config") as cursor:
            assert await cursor.fetchone() == (1,)
        config = await get_digest_config(conn, guild_id=GUILD_A)
        assert config is not None
        assert config.channel_id in (CHANNEL_A, CHANNEL_B)

    async def test_concurrent_writes_across_guilds_do_not_interfere(
        self, conn: aiosqlite.Connection
    ) -> None:
        await asyncio.gather(
            _enable(conn, guild_id=GUILD_A, channel_id=CHANNEL_A),
            _enable(conn, guild_id=GUILD_B, channel_id=CHANNEL_B),
        )

        assert len(await get_enabled_digest_configs(conn)) == 2
