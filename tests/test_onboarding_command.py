"""Tests for aura.commands.onboarding: /aura-onboarding.

Mirrors tests/test_digest_command.py's structure exactly, for the sibling
command with the same "channel + on/off, compose, nothing named keeps its
current value" option handling and no interval to worry about.
"""
from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock

import aiosqlite
import discord
import pytest
from discord import app_commands

from aura.commands.onboarding import (
    _handle_onboarding_command_error,
    onboarding_command,
)
from aura.db.onboarding_config import get_onboarding_config, set_onboarding_config
from aura.db.repository import init_schema
from aura.i18n import SUPPORTED_LOCALES, t

GUILD_A = 100000000000000001
CHANNEL_A = 300000000000000003
CHANNEL_B = 400000000000000004
MODERATOR = 4242


@pytest.fixture
async def conn():
    connection = await aiosqlite.connect(":memory:")
    await init_schema(connection)
    yield connection
    await connection.close()


def _make_interaction(
    *, db: aiosqlite.Connection | None, locale: str = "en-US", guild_id: int = GUILD_A
) -> MagicMock:
    interaction = MagicMock(spec=discord.Interaction)
    interaction.locale = locale
    interaction.guild_id = guild_id
    interaction.user = MagicMock()
    interaction.user.id = MODERATOR
    interaction.client = MagicMock()
    interaction.client.db = db
    interaction.response = MagicMock()
    interaction.response.send_message = AsyncMock()
    interaction.response.is_done = MagicMock(return_value=False)
    interaction.followup = MagicMock()
    interaction.followup.send = AsyncMock()
    interaction.command = MagicMock()
    interaction.command.name = "aura-onboarding"
    return interaction


def _make_channel(channel_id: int = CHANNEL_A, *, can_post: bool = True) -> MagicMock:
    channel = MagicMock(spec=discord.TextChannel)
    channel.id = channel_id
    channel.mention = f"<#{channel_id}>"
    channel.guild = MagicMock()
    channel.guild.me = MagicMock()
    channel.permissions_for = MagicMock(
        return_value=discord.Permissions(send_messages=can_post, embed_links=can_post)
    )
    return channel


async def _invoke(
    interaction: discord.Interaction,
    channel: MagicMock | None = None,
    enabled: bool | None = None,
) -> None:
    await onboarding_command.callback(interaction, channel, enabled)  # pyright: ignore[reportCallIssue, reportArgumentType]


def _reply(interaction: MagicMock) -> str:
    interaction.response.send_message.assert_awaited_once()
    return interaction.response.send_message.await_args.args[0]


class TestPermissionCheck:
    def test_rejects_a_non_moderator(self) -> None:
        fake = MagicMock(permissions=discord.Permissions(manage_guild=False))
        with pytest.raises(app_commands.MissingPermissions):
            for check in onboarding_command.checks:
                check(fake)

    def test_allows_a_moderator(self) -> None:
        fake = MagicMock(permissions=discord.Permissions(manage_guild=True))
        for check in onboarding_command.checks:
            assert check(fake) is True

    def test_the_command_is_guild_only(self) -> None:
        assert onboarding_command.guild_only is True


class TestEnabling:
    async def test_naming_a_channel_turns_onboarding_on(
        self, conn: aiosqlite.Connection
    ) -> None:
        interaction = _make_interaction(db=conn)

        await _invoke(interaction, _make_channel())

        config = await get_onboarding_config(conn, guild_id=GUILD_A)
        assert config is not None
        assert config.channel_id == CHANNEL_A
        assert config.onboarding_enabled is True
        assert config.updated_by_id == MODERATOR

    async def test_the_confirmation_names_the_channel(self, conn: aiosqlite.Connection) -> None:
        interaction = _make_interaction(db=conn)

        await _invoke(interaction, _make_channel())

        message = _reply(interaction)
        assert f"<#{CHANNEL_A}>" in message
        assert interaction.response.send_message.await_args.kwargs["ephemeral"] is True

    async def test_an_explicit_enabled_true_also_works(
        self, conn: aiosqlite.Connection
    ) -> None:
        interaction = _make_interaction(db=conn)

        await _invoke(interaction, _make_channel(), True)

        config = await get_onboarding_config(conn, guild_id=GUILD_A)
        assert config is not None and config.onboarding_enabled is True


class TestKeepingWhatIsNotGiven:
    async def test_changing_only_enabled_keeps_the_channel(
        self, conn: aiosqlite.Connection
    ) -> None:
        await _invoke(_make_interaction(db=conn), _make_channel(CHANNEL_B))

        await _invoke(_make_interaction(db=conn), None, False)

        config = await get_onboarding_config(conn, guild_id=GUILD_A)
        assert config is not None
        assert config.channel_id == CHANNEL_B
        assert config.onboarding_enabled is False

    async def test_changing_only_the_channel_keeps_it_enabled(
        self, conn: aiosqlite.Connection
    ) -> None:
        await _invoke(_make_interaction(db=conn), _make_channel(), True)

        await _invoke(_make_interaction(db=conn), _make_channel(CHANNEL_B))

        config = await get_onboarding_config(conn, guild_id=GUILD_A)
        assert config is not None
        assert config.channel_id == CHANNEL_B
        assert config.onboarding_enabled is True

    async def test_re_enabling_keeps_the_previous_channel(
        self, conn: aiosqlite.Connection
    ) -> None:
        await _invoke(_make_interaction(db=conn), _make_channel(CHANNEL_B))
        await _invoke(_make_interaction(db=conn), None, False)

        await _invoke(_make_interaction(db=conn), None, True)

        config = await get_onboarding_config(conn, guild_id=GUILD_A)
        assert config is not None
        assert config.onboarding_enabled is True
        assert config.channel_id == CHANNEL_B

    async def test_the_confirmation_distinguishes_new_from_updated(
        self, conn: aiosqlite.Connection
    ) -> None:
        await _invoke(_make_interaction(db=conn), _make_channel())
        interaction = _make_interaction(db=conn)

        await _invoke(interaction, _make_channel(CHANNEL_B))

        message = _reply(interaction)
        assert message == t("onboarding_updated", "en-US", channel=f"<#{CHANNEL_B}>")


class TestDisabling:
    async def test_disabling_switches_it_off_but_keeps_the_channel(
        self, conn: aiosqlite.Connection
    ) -> None:
        await _invoke(_make_interaction(db=conn), _make_channel())
        interaction = _make_interaction(db=conn)

        await _invoke(interaction, None, False)

        config = await get_onboarding_config(conn, guild_id=GUILD_A)
        assert config is not None
        assert config.onboarding_enabled is False
        assert config.channel_id == CHANNEL_A
        assert "off" in _reply(interaction)

    async def test_disabling_onboarding_that_was_never_set_up_says_so(
        self, conn: aiosqlite.Connection
    ) -> None:
        interaction = _make_interaction(db=conn)

        await _invoke(interaction, None, False)

        assert _reply(interaction) == t("onboarding_already_off", "en-US")
        assert await get_onboarding_config(conn, guild_id=GUILD_A) is None

    async def test_a_channel_given_with_enabled_false_is_stored_but_stays_off(
        self, conn: aiosqlite.Connection
    ) -> None:
        interaction = _make_interaction(db=conn)

        await _invoke(interaction, _make_channel(), False)

        config = await get_onboarding_config(conn, guild_id=GUILD_A)
        assert config is not None
        assert config.channel_id == CHANNEL_A
        assert config.onboarding_enabled is False


class TestRefusals:
    async def test_a_call_with_no_options_changes_nothing(
        self, conn: aiosqlite.Connection
    ) -> None:
        interaction = _make_interaction(db=conn)

        await _invoke(interaction)

        assert _reply(interaction) == t("onboarding_no_options_error", "en-US")
        assert await get_onboarding_config(conn, guild_id=GUILD_A) is None

    async def test_enabling_with_no_channel_and_no_config_asks_for_a_channel(
        self, conn: aiosqlite.Connection
    ) -> None:
        interaction = _make_interaction(db=conn)

        await _invoke(interaction, None, True)

        assert _reply(interaction) == t("onboarding_channel_required", "en-US")
        assert await get_onboarding_config(conn, guild_id=GUILD_A) is None


class TestPermissionWarning:
    async def test_a_channel_aura_cannot_post_in_is_flagged_but_still_saved(
        self, conn: aiosqlite.Connection
    ) -> None:
        interaction = _make_interaction(db=conn)

        await _invoke(interaction, _make_channel(can_post=False))

        message = _reply(interaction)
        assert "⚠️" in message
        config = await get_onboarding_config(conn, guild_id=GUILD_A)
        assert config is not None and config.onboarding_enabled is True

    async def test_a_usable_channel_produces_no_warning(
        self, conn: aiosqlite.Connection
    ) -> None:
        interaction = _make_interaction(db=conn)

        await _invoke(interaction, _make_channel(can_post=True))

        assert "⚠️" not in _reply(interaction)

    async def test_a_permission_check_that_explodes_does_not_break_the_command(
        self, conn: aiosqlite.Connection, caplog: pytest.LogCaptureFixture
    ) -> None:
        channel = _make_channel()
        channel.permissions_for = MagicMock(side_effect=RuntimeError("no cached member"))
        interaction = _make_interaction(db=conn)

        with caplog.at_level(logging.ERROR):
            await _invoke(interaction, channel)

        config = await get_onboarding_config(conn, guild_id=GUILD_A)
        assert config is not None and config.onboarding_enabled is True
        assert "⚠️" not in _reply(interaction)

    async def test_a_guild_with_no_cached_member_produces_no_warning(
        self, conn: aiosqlite.Connection
    ) -> None:
        channel = _make_channel()
        channel.guild.me = None
        interaction = _make_interaction(db=conn)

        await _invoke(interaction, channel)

        assert "⚠️" not in _reply(interaction)


class TestGuildIsolation:
    async def test_configuring_one_guild_leaves_another_untouched(
        self, conn: aiosqlite.Connection
    ) -> None:
        other_guild = 999999999999999999
        await set_onboarding_config(
            conn, guild_id=other_guild, channel_id=CHANNEL_B, enabled=True, updated_by_id=1
        )

        await _invoke(_make_interaction(db=conn), _make_channel())

        untouched = await get_onboarding_config(conn, guild_id=other_guild)
        assert untouched is not None
        assert untouched.channel_id == CHANNEL_B


class TestLocalization:
    @pytest.mark.parametrize("locale", sorted(SUPPORTED_LOCALES))
    async def test_every_reply_resolves_in_every_locale(
        self, conn: aiosqlite.Connection, locale: str
    ) -> None:
        interaction = _make_interaction(db=conn, locale=locale)

        await _invoke(interaction, _make_channel(can_post=False))

        message = _reply(interaction)
        assert "[onboarding_" not in message

    async def test_an_unknown_locale_falls_back_to_english(
        self, conn: aiosqlite.Connection
    ) -> None:
        interaction = _make_interaction(db=conn, locale="xx-XX")

        await _invoke(interaction)

        assert _reply(interaction) == t("onboarding_no_options_error", "en-US")


class TestErrorHandler:
    async def test_a_missing_permission_gets_a_localized_reply(self) -> None:
        interaction = _make_interaction(db=None, locale="de")

        await _handle_onboarding_command_error(
            interaction, app_commands.MissingPermissions(["manage_guild"])
        )

        interaction.response.send_message.assert_awaited_once_with(
            t("onboarding_permission_error", "de"), ephemeral=True
        )

    async def test_a_deferred_interaction_answers_through_the_followup(self) -> None:
        interaction = _make_interaction(db=None)
        interaction.response.is_done = MagicMock(return_value=True)

        await _handle_onboarding_command_error(
            interaction, app_commands.MissingPermissions(["manage_guild"])
        )

        interaction.followup.send.assert_awaited_once()
        interaction.response.send_message.assert_not_awaited()

    async def test_any_other_error_is_logged_rather_than_swallowed(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        interaction = _make_interaction(db=None)

        with caplog.at_level(logging.ERROR):
            await _handle_onboarding_command_error(
                interaction, app_commands.AppCommandError("something else")
            )

        assert any(record.levelno >= logging.ERROR for record in caplog.records)
