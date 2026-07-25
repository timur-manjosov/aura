"""Tests for aura.commands.config: /aura-config.

Command callback, permission check and error handler invoked directly against
mocked discord objects and a real in-memory database, matching how the other
moderator-gated commands are tested. No live gateway connection.
"""
from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock

import aiosqlite
import discord
import pytest
from discord import app_commands

from aura.commands.config import _handle_config_command_error, config_command
from aura.db.proactive_channel_config import is_channel_enabled
from aura.db.repository import init_schema
from aura.i18n import SUPPORTED_LOCALES, t

GUILD_A = 100000000000000001


@pytest.fixture
async def conn():
    connection = await aiosqlite.connect(":memory:")
    await init_schema(connection)
    yield connection
    await connection.close()


def _make_interaction(
    *,
    db: aiosqlite.Connection | None,
    locale: str = "en-US",
    guild_id: int = GUILD_A,
    user_id: int = 4242,
) -> MagicMock:
    interaction = MagicMock(spec=discord.Interaction)
    interaction.locale = locale
    interaction.guild_id = guild_id
    interaction.user = MagicMock()
    interaction.user.id = user_id
    interaction.client = MagicMock()
    interaction.client.db = db
    interaction.response = MagicMock()
    interaction.response.send_message = AsyncMock()
    interaction.response.is_done = MagicMock(return_value=False)
    interaction.followup = MagicMock()
    interaction.followup.send = AsyncMock()
    interaction.command = MagicMock()
    interaction.command.name = "aura-config"
    return interaction


def _make_channel(channel_id: int = 555) -> MagicMock:
    channel = MagicMock(spec=discord.TextChannel)
    channel.id = channel_id
    channel.mention = f"<#{channel_id}>"
    return channel


async def _invoke(interaction: discord.Interaction, channel: MagicMock, proactive: bool) -> None:
    await config_command.callback(interaction, channel, proactive)  # pyright: ignore[reportCallIssue, reportArgumentType]


class TestPermissionCheck:
    def test_rejects_a_non_moderator(self) -> None:
        fake = MagicMock(permissions=discord.Permissions(manage_guild=False))
        with pytest.raises(app_commands.MissingPermissions):
            for check in config_command.checks:
                check(fake)

    def test_allows_a_moderator(self) -> None:
        fake = MagicMock(permissions=discord.Permissions(manage_guild=True))
        for check in config_command.checks:
            assert check(fake) is True

    def test_the_command_is_guild_only(self) -> None:
        assert config_command.guild_only is True


class TestErrorHandler:
    async def test_missing_permissions_replies_ephemerally_and_localized(self) -> None:
        interaction = _make_interaction(db=None)

        await _handle_config_command_error(
            interaction, app_commands.MissingPermissions(["manage_guild"])
        )

        interaction.response.send_message.assert_awaited_once()
        args, kwargs = interaction.response.send_message.call_args
        assert "permission" in args[0].lower()
        assert kwargs.get("ephemeral") is True

    async def test_missing_permissions_uses_followup_if_already_responded(self) -> None:
        interaction = _make_interaction(db=None)
        interaction.response.is_done = MagicMock(return_value=True)

        await _handle_config_command_error(
            interaction, app_commands.MissingPermissions(["manage_guild"])
        )

        interaction.followup.send.assert_awaited_once()
        interaction.response.send_message.assert_not_awaited()

    async def test_unexpected_errors_are_logged_and_not_surfaced(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        interaction = _make_interaction(db=None)
        error = app_commands.CommandInvokeError(MagicMock(), ValueError("boom"))

        with caplog.at_level(logging.ERROR):
            await _handle_config_command_error(interaction, error)

        interaction.response.send_message.assert_not_awaited()
        assert any(record.levelno == logging.ERROR for record in caplog.records)


class TestPersistence:
    async def test_enabling_a_channel_persists_and_confirms(
        self, conn: aiosqlite.Connection
    ) -> None:
        interaction = _make_interaction(db=conn)
        channel = _make_channel(555)

        await _invoke(interaction, channel, proactive=True)

        assert await is_channel_enabled(conn, channel_id=555) is True
        args, kwargs = interaction.response.send_message.call_args
        assert kwargs.get("ephemeral") is True
        assert channel.mention in args[0]

    async def test_disabling_a_channel_persists_and_confirms(
        self, conn: aiosqlite.Connection
    ) -> None:
        interaction = _make_interaction(db=conn)
        channel = _make_channel(555)

        await _invoke(interaction, channel, proactive=True)
        await _invoke(interaction, channel, proactive=False)

        assert await is_channel_enabled(conn, channel_id=555) is False

    async def test_the_editor_is_recorded_from_the_invoking_user(
        self, conn: aiosqlite.Connection
    ) -> None:
        interaction = _make_interaction(db=conn, user_id=98765)
        await _invoke(interaction, _make_channel(555), proactive=True)

        async with conn.execute(
            "SELECT updated_by_id FROM proactive_channel_config WHERE channel_id = 555"
        ) as cursor:
            assert await cursor.fetchone() == (98765,)

    async def test_the_on_and_off_confirmations_are_distinct(
        self, conn: aiosqlite.Connection
    ) -> None:
        on = _make_interaction(db=conn)
        await _invoke(on, _make_channel(1), proactive=True)
        off = _make_interaction(db=conn)
        await _invoke(off, _make_channel(2), proactive=False)

        on_text = on.response.send_message.call_args[0][0]
        off_text = off.response.send_message.call_args[0][0]
        assert on_text != off_text


class TestLocalization:
    @pytest.mark.parametrize("locale", sorted(SUPPORTED_LOCALES) + ["xx-INVALID"])
    async def test_any_locale_including_an_unsupported_one_replies_without_crashing(
        self, conn: aiosqlite.Connection, locale: str
    ) -> None:
        interaction = _make_interaction(db=conn, locale=locale)
        await _invoke(interaction, _make_channel(555), proactive=True)

        text = interaction.response.send_message.call_args[0][0]
        assert text  # never blank, never a raw placeholder
        assert not text.startswith("[")

    def test_the_confirmation_keys_exist_in_every_locale(self) -> None:
        for locale in SUPPORTED_LOCALES:
            for key in ("config_proactive_enabled", "config_proactive_disabled", "config_permission_error"):
                assert t(key, locale, channel="#x") != f"[{key}]", f"{key} missing for {locale}"
