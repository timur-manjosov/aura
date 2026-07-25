"""Tests for aura.db.proactive_channel_config: the per-channel on/off switch.

A real in-memory database throughout, never a live gateway connection, per
CLAUDE.md's testing philosophy. The one invariant that matters most is asserted
first and repeatedly: a channel with no row is OFF.
"""
from __future__ import annotations

import asyncio

import aiosqlite
import pytest

from aura.db.proactive_channel_config import is_channel_enabled, set_channel_enabled
from aura.db.repository import init_schema

GUILD_A = 100000000000000001
GUILD_B = 200000000000000002


@pytest.fixture
async def conn():
    connection = await aiosqlite.connect(":memory:")
    await init_schema(connection)
    yield connection
    await connection.close()


class TestDefaultOff:
    async def test_an_unconfigured_channel_is_disabled(self, conn: aiosqlite.Connection) -> None:
        # The load-bearing default: proactive relief is opt-in per channel.
        assert await is_channel_enabled(conn, channel_id=999) is False

    async def test_an_empty_database_disables_every_channel(
        self, conn: aiosqlite.Connection
    ) -> None:
        for channel_id in (1, 2, 3, 10**18):
            assert await is_channel_enabled(conn, channel_id=channel_id) is False


class TestSetAndGet:
    async def test_enabling_a_channel_turns_it_on(self, conn: aiosqlite.Connection) -> None:
        await set_channel_enabled(
            conn, guild_id=GUILD_A, channel_id=5, enabled=True, updated_by_id=42
        )
        assert await is_channel_enabled(conn, channel_id=5) is True

    async def test_disabling_a_channel_turns_it_off(self, conn: aiosqlite.Connection) -> None:
        await set_channel_enabled(
            conn, guild_id=GUILD_A, channel_id=5, enabled=True, updated_by_id=42
        )
        await set_channel_enabled(
            conn, guild_id=GUILD_A, channel_id=5, enabled=False, updated_by_id=42
        )
        assert await is_channel_enabled(conn, channel_id=5) is False

    async def test_toggling_repeatedly_leaves_exactly_one_row(
        self, conn: aiosqlite.Connection
    ) -> None:
        for enabled in (True, False, True, False, True):
            await set_channel_enabled(
                conn, guild_id=GUILD_A, channel_id=5, enabled=enabled, updated_by_id=1
            )

        async with conn.execute(
            "SELECT COUNT(*) FROM proactive_channel_config WHERE channel_id = 5"
        ) as cursor:
            assert await cursor.fetchone() == (1,)
        assert await is_channel_enabled(conn, channel_id=5) is True  # the last write wins

    async def test_the_editor_and_time_are_recorded_for_audit(
        self, conn: aiosqlite.Connection
    ) -> None:
        await set_channel_enabled(
            conn, guild_id=GUILD_A, channel_id=5, enabled=True, updated_by_id=777
        )
        async with conn.execute(
            "SELECT updated_by_id, updated_at FROM proactive_channel_config WHERE channel_id = 5"
        ) as cursor:
            row = await cursor.fetchone()
        assert row is not None
        assert row[0] == 777
        assert row[1]  # a timestamp string was written

    async def test_a_later_edit_updates_the_editor(self, conn: aiosqlite.Connection) -> None:
        await set_channel_enabled(
            conn, guild_id=GUILD_A, channel_id=5, enabled=True, updated_by_id=111
        )
        await set_channel_enabled(
            conn, guild_id=GUILD_A, channel_id=5, enabled=False, updated_by_id=222
        )
        async with conn.execute(
            "SELECT updated_by_id FROM proactive_channel_config WHERE channel_id = 5"
        ) as cursor:
            assert await cursor.fetchone() == (222,)


class TestChannelIsolation:
    async def test_enabling_one_channel_does_not_enable_another(
        self, conn: aiosqlite.Connection
    ) -> None:
        await set_channel_enabled(
            conn, guild_id=GUILD_A, channel_id=5, enabled=True, updated_by_id=1
        )
        assert await is_channel_enabled(conn, channel_id=5) is True
        assert await is_channel_enabled(conn, channel_id=6) is False

    async def test_channels_in_different_guilds_are_independent(
        self, conn: aiosqlite.Connection
    ) -> None:
        await set_channel_enabled(
            conn, guild_id=GUILD_A, channel_id=5, enabled=True, updated_by_id=1
        )
        await set_channel_enabled(
            conn, guild_id=GUILD_B, channel_id=6, enabled=False, updated_by_id=1
        )
        assert await is_channel_enabled(conn, channel_id=5) is True
        assert await is_channel_enabled(conn, channel_id=6) is False


class TestConcurrency:
    async def test_concurrent_writes_to_one_channel_leave_one_consistent_row(
        self, conn: aiosqlite.Connection
    ) -> None:
        # discord.py dispatches events as separate tasks; two moderators (or one
        # moderator racing themselves) must not corrupt the single row.
        await asyncio.gather(
            *(
                set_channel_enabled(
                    conn,
                    guild_id=GUILD_A,
                    channel_id=5,
                    enabled=bool(i % 2),
                    updated_by_id=i,
                )
                for i in range(50)
            )
        )

        async with conn.execute(
            "SELECT COUNT(*) FROM proactive_channel_config WHERE channel_id = 5"
        ) as cursor:
            assert await cursor.fetchone() == (1,)
        # The value is whichever write landed last; the point is it is a valid
        # 0/1, not a corrupted or duplicated row.
        assert isinstance(await is_channel_enabled(conn, channel_id=5), bool)
