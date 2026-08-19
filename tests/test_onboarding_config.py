"""Tests for aura.db.onboarding_config: the per-guild onboarding switch and channel.

Mirrors tests/test_digest_config.py where the shape is the same (an opt-in
switch nobody configured is off, an upsert leaves one row, concurrent writers
do not corrupt it) and is simpler where onboarding_config genuinely is: no
interval, no baseline timestamp to protect, because onboarding is not
windowed (see aura.onboarding.builder).

A real in-memory database throughout, never a live gateway connection, per
CLAUDE.md's testing philosophy.
"""
from __future__ import annotations

import asyncio

import aiosqlite
import pytest

from aura.db.onboarding_config import get_onboarding_config, set_onboarding_config
from aura.db.repository import init_schema

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
    enabled: bool = True,
) -> None:
    await set_onboarding_config(
        conn, guild_id=guild_id, channel_id=channel_id, enabled=enabled, updated_by_id=MODERATOR
    )


class TestDefaultOff:
    async def test_an_unconfigured_guild_has_no_config(self, conn: aiosqlite.Connection) -> None:
        assert await get_onboarding_config(conn, guild_id=GUILD_A) is None


class TestSetAndGet:
    async def test_enabling_records_every_field(self, conn: aiosqlite.Connection) -> None:
        await _enable(conn)

        config = await get_onboarding_config(conn, guild_id=GUILD_A)

        assert config is not None
        assert config.guild_id == GUILD_A
        assert config.channel_id == CHANNEL_A
        assert config.onboarding_enabled is True
        assert config.updated_by_id == MODERATOR

    async def test_a_disabled_guild_is_kept_but_never_scheduled(
        self, conn: aiosqlite.Connection
    ) -> None:
        # Disabling must not forget the channel: turning onboarding back on
        # should not mean choosing the channel again.
        await _enable(conn)
        await _enable(conn, enabled=False)

        config = await get_onboarding_config(conn, guild_id=GUILD_A)

        assert config is not None
        assert config.onboarding_enabled is False
        assert config.channel_id == CHANNEL_A

    async def test_toggling_repeatedly_leaves_exactly_one_row(
        self, conn: aiosqlite.Connection
    ) -> None:
        for enabled in (True, False, True, False, True):
            await _enable(conn, enabled=enabled)

        async with conn.execute(
            "SELECT COUNT(*) FROM onboarding_config WHERE guild_id = ?", (GUILD_A,)
        ) as cursor:
            assert await cursor.fetchone() == (1,)
        config = await get_onboarding_config(conn, guild_id=GUILD_A)
        assert config is not None and config.onboarding_enabled is True

    async def test_changing_the_channel_overwrites_it(self, conn: aiosqlite.Connection) -> None:
        await _enable(conn, channel_id=CHANNEL_A)
        await _enable(conn, channel_id=CHANNEL_B)

        config = await get_onboarding_config(conn, guild_id=GUILD_A)

        assert config is not None and config.channel_id == CHANNEL_B


class TestGuildIsolation:
    async def test_two_guilds_keep_entirely_separate_settings(
        self, conn: aiosqlite.Connection
    ) -> None:
        await _enable(conn, guild_id=GUILD_A, channel_id=CHANNEL_A)
        await _enable(conn, guild_id=GUILD_B, channel_id=CHANNEL_B)

        first = await get_onboarding_config(conn, guild_id=GUILD_A)
        second = await get_onboarding_config(conn, guild_id=GUILD_B)

        assert first is not None and first.channel_id == CHANNEL_A
        assert second is not None and second.channel_id == CHANNEL_B

    async def test_disabling_one_guild_leaves_the_other_untouched(
        self, conn: aiosqlite.Connection
    ) -> None:
        await _enable(conn, guild_id=GUILD_A)
        await _enable(conn, guild_id=GUILD_B, channel_id=CHANNEL_B)

        await _enable(conn, guild_id=GUILD_A, enabled=False)

        first = await get_onboarding_config(conn, guild_id=GUILD_A)
        second = await get_onboarding_config(conn, guild_id=GUILD_B)
        assert first is not None and first.onboarding_enabled is False
        assert second is not None and second.onboarding_enabled is True


class TestConcurrency:
    async def test_concurrent_writes_leave_exactly_one_coherent_row(
        self, conn: aiosqlite.Connection
    ) -> None:
        await asyncio.gather(
            *(_enable(conn, channel_id=channel) for channel in (CHANNEL_A, CHANNEL_B))
        )

        async with conn.execute("SELECT COUNT(*) FROM onboarding_config") as cursor:
            assert await cursor.fetchone() == (1,)
        config = await get_onboarding_config(conn, guild_id=GUILD_A)
        assert config is not None
        assert config.channel_id in (CHANNEL_A, CHANNEL_B)

    async def test_concurrent_writes_across_guilds_do_not_interfere(
        self, conn: aiosqlite.Connection
    ) -> None:
        await asyncio.gather(
            _enable(conn, guild_id=GUILD_A, channel_id=CHANNEL_A),
            _enable(conn, guild_id=GUILD_B, channel_id=CHANNEL_B),
        )

        first = await get_onboarding_config(conn, guild_id=GUILD_A)
        second = await get_onboarding_config(conn, guild_id=GUILD_B)
        assert first is not None and second is not None
