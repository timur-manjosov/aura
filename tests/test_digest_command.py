"""Tests for aura.commands.digest: /aura-digest.

Command callback, permission check and error handler invoked directly against
mocked discord objects and a real in-memory database, matching how every other
moderator-gated command is tested. No live gateway connection.

The bulk of the interesting behaviour is combination handling: three optional
options that compose, where "not given" has to mean "keep what is there" rather
than "reset to a default" -- a command that silently moved a server's digest
channel because a moderator changed the cadence would be a quiet, hard-to-notice
kind of wrong.
"""
from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock

import aiosqlite
import discord
import pytest
from discord import app_commands

from aura.commands.digest import _handle_digest_command_error, digest_command
from aura.db.digest_config import get_digest_config, set_digest_config
from aura.db.repository import init_schema
from aura.digest.intervals import DigestInterval
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
    interaction.command.name = "aura-digest"
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


def _choice(interval: DigestInterval) -> app_commands.Choice[int]:
    return app_commands.Choice(name=interval.name, value=int(interval))


async def _invoke(
    interaction: discord.Interaction,
    channel: MagicMock | None = None,
    interval: app_commands.Choice[int] | None = None,
    enabled: bool | None = None,
) -> None:
    await digest_command.callback(interaction, channel, interval, enabled)  # pyright: ignore[reportCallIssue, reportArgumentType]


def _reply(interaction: MagicMock) -> str:
    interaction.response.send_message.assert_awaited_once()
    return interaction.response.send_message.await_args.args[0]


class TestPermissionCheck:
    def test_rejects_a_non_moderator(self) -> None:
        fake = MagicMock(permissions=discord.Permissions(manage_guild=False))
        with pytest.raises(app_commands.MissingPermissions):
            for check in digest_command.checks:
                check(fake)

    def test_allows_a_moderator(self) -> None:
        fake = MagicMock(permissions=discord.Permissions(manage_guild=True))
        for check in digest_command.checks:
            assert check(fake) is True

    def test_the_command_is_guild_only(self) -> None:
        # A digest is a property of a server; there is nothing for it to
        # summarize in a DM.
        assert digest_command.guild_only is True


class TestEnabling:
    async def test_naming_a_channel_turns_the_digest_on_weekly(
        self, conn: aiosqlite.Connection
    ) -> None:
        interaction = _make_interaction(db=conn)

        await _invoke(interaction, _make_channel())

        config = await get_digest_config(conn, guild_id=GUILD_A)
        assert config is not None
        assert config.channel_id == CHANNEL_A
        assert config.digest_enabled is True
        assert config.interval_seconds == int(DigestInterval.WEEKLY)
        assert config.updated_by_id == MODERATOR

    async def test_the_confirmation_names_the_channel_and_cadence(
        self, conn: aiosqlite.Connection
    ) -> None:
        interaction = _make_interaction(db=conn)

        await _invoke(interaction, _make_channel(), _choice(DigestInterval.DAILY))

        message = _reply(interaction)
        assert f"<#{CHANNEL_A}>" in message
        assert "daily" in message
        # The moderator is told the first digest does not replay history.
        assert "from now on" in message
        assert interaction.response.send_message.await_args.kwargs["ephemeral"] is True

    @pytest.mark.parametrize("interval", list(DigestInterval))
    async def test_every_offered_cadence_is_stored(
        self, conn: aiosqlite.Connection, interval: DigestInterval
    ) -> None:
        interaction = _make_interaction(db=conn)

        await _invoke(interaction, _make_channel(), _choice(interval))

        config = await get_digest_config(conn, guild_id=GUILD_A)
        assert config is not None and config.interval_seconds == int(interval)

    async def test_an_explicit_enabled_true_also_works(
        self, conn: aiosqlite.Connection
    ) -> None:
        interaction = _make_interaction(db=conn)

        await _invoke(interaction, _make_channel(), None, True)

        config = await get_digest_config(conn, guild_id=GUILD_A)
        assert config is not None and config.digest_enabled is True


class TestKeepingWhatIsNotGiven:
    async def test_changing_only_the_cadence_keeps_the_channel(
        self, conn: aiosqlite.Connection
    ) -> None:
        await _invoke(_make_interaction(db=conn), _make_channel(CHANNEL_B))

        await _invoke(_make_interaction(db=conn), None, _choice(DigestInterval.MONTHLY))

        config = await get_digest_config(conn, guild_id=GUILD_A)
        assert config is not None
        assert config.channel_id == CHANNEL_B
        assert config.interval_seconds == int(DigestInterval.MONTHLY)

    async def test_changing_only_the_channel_keeps_the_cadence(
        self, conn: aiosqlite.Connection
    ) -> None:
        await _invoke(_make_interaction(db=conn), _make_channel(), _choice(DigestInterval.BIWEEKLY))

        await _invoke(_make_interaction(db=conn), _make_channel(CHANNEL_B))

        config = await get_digest_config(conn, guild_id=GUILD_A)
        assert config is not None
        assert config.channel_id == CHANNEL_B
        assert config.interval_seconds == int(DigestInterval.BIWEEKLY)

    async def test_re_enabling_keeps_both(self, conn: aiosqlite.Connection) -> None:
        await _invoke(_make_interaction(db=conn), _make_channel(CHANNEL_B), _choice(DigestInterval.DAILY))
        await _invoke(_make_interaction(db=conn), None, None, False)

        await _invoke(_make_interaction(db=conn), None, None, True)

        config = await get_digest_config(conn, guild_id=GUILD_A)
        assert config is not None
        assert config.digest_enabled is True
        assert config.channel_id == CHANNEL_B
        assert config.interval_seconds == int(DigestInterval.DAILY)

    async def test_updating_a_running_digest_says_so_rather_than_promising_a_first_one(
        self, conn: aiosqlite.Connection
    ) -> None:
        await _invoke(_make_interaction(db=conn), _make_channel())
        interaction = _make_interaction(db=conn)

        await _invoke(interaction, None, _choice(DigestInterval.DAILY))

        message = _reply(interaction)
        assert "from now on" not in message
        assert "keeps running" in message


class TestDisabling:
    async def test_disabling_switches_it_off_but_keeps_the_settings(
        self, conn: aiosqlite.Connection
    ) -> None:
        await _invoke(_make_interaction(db=conn), _make_channel(), _choice(DigestInterval.DAILY))
        interaction = _make_interaction(db=conn)

        await _invoke(interaction, None, None, False)

        config = await get_digest_config(conn, guild_id=GUILD_A)
        assert config is not None
        assert config.digest_enabled is False
        assert config.channel_id == CHANNEL_A
        assert config.interval_seconds == int(DigestInterval.DAILY)
        assert "off" in _reply(interaction)

    async def test_disabling_a_digest_that_was_never_set_up_says_so(
        self, conn: aiosqlite.Connection
    ) -> None:
        interaction = _make_interaction(db=conn)

        await _invoke(interaction, None, None, False)

        assert _reply(interaction) == t("digest_already_off", "en-US")
        assert await get_digest_config(conn, guild_id=GUILD_A) is None

    async def test_a_channel_given_with_enabled_false_is_stored_but_stays_off(
        self, conn: aiosqlite.Connection
    ) -> None:
        # Explicit beats implicit: `enabled` is the only option that speaks
        # directly to whether digests run.
        interaction = _make_interaction(db=conn)

        await _invoke(interaction, _make_channel(), None, False)

        config = await get_digest_config(conn, guild_id=GUILD_A)
        assert config is not None
        assert config.channel_id == CHANNEL_A
        assert config.digest_enabled is False


class TestRefusals:
    async def test_a_call_with_no_options_changes_nothing(
        self, conn: aiosqlite.Connection
    ) -> None:
        interaction = _make_interaction(db=conn)

        await _invoke(interaction)

        assert _reply(interaction) == t("digest_no_options_error", "en-US")
        assert await get_digest_config(conn, guild_id=GUILD_A) is None

    async def test_a_cadence_with_no_channel_and_no_config_asks_for_a_channel(
        self, conn: aiosqlite.Connection
    ) -> None:
        # Picking a channel for the moderator would mean Aura choosing where it
        # posts unprompted, which is what the opt-in gate exists to prevent.
        interaction = _make_interaction(db=conn)

        await _invoke(interaction, None, _choice(DigestInterval.WEEKLY))

        assert _reply(interaction) == t("digest_channel_required", "en-US")
        assert await get_digest_config(conn, guild_id=GUILD_A) is None

    async def test_enabling_with_no_channel_and_no_config_asks_for_a_channel(
        self, conn: aiosqlite.Connection
    ) -> None:
        interaction = _make_interaction(db=conn)

        await _invoke(interaction, None, None, True)

        assert _reply(interaction) == t("digest_channel_required", "en-US")


class TestPermissionWarning:
    async def test_a_channel_aura_cannot_post_in_is_flagged_but_still_saved(
        self, conn: aiosqlite.Connection
    ) -> None:
        # A warning, never a refusal: permissions can change either way after
        # this call, and refusing here would block a setup that will work.
        interaction = _make_interaction(db=conn)

        await _invoke(interaction, _make_channel(can_post=False))

        message = _reply(interaction)
        assert "⚠️" in message
        config = await get_digest_config(conn, guild_id=GUILD_A)
        assert config is not None and config.digest_enabled is True

    async def test_a_usable_channel_produces_no_warning(
        self, conn: aiosqlite.Connection
    ) -> None:
        interaction = _make_interaction(db=conn)

        await _invoke(interaction, _make_channel(can_post=True))

        assert "⚠️" not in _reply(interaction)

    async def test_a_permission_check_that_explodes_does_not_break_the_command(
        self, conn: aiosqlite.Connection, caplog: pytest.LogCaptureFixture
    ) -> None:
        # Advice must not be able to break the thing it is attached to.
        channel = _make_channel()
        channel.permissions_for = MagicMock(side_effect=RuntimeError("no cached member"))
        interaction = _make_interaction(db=conn)

        with caplog.at_level(logging.ERROR):
            await _invoke(interaction, channel)

        config = await get_digest_config(conn, guild_id=GUILD_A)
        assert config is not None and config.digest_enabled is True
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
        await set_digest_config(
            conn,
            guild_id=other_guild,
            channel_id=CHANNEL_B,
            interval_seconds=int(DigestInterval.MONTHLY),
            enabled=True,
            updated_by_id=1,
        )

        await _invoke(_make_interaction(db=conn), _make_channel(), _choice(DigestInterval.DAILY))

        untouched = await get_digest_config(conn, guild_id=other_guild)
        assert untouched is not None
        assert untouched.channel_id == CHANNEL_B
        assert untouched.interval_seconds == int(DigestInterval.MONTHLY)


class TestLocalization:
    @pytest.mark.parametrize("locale", sorted(SUPPORTED_LOCALES))
    async def test_every_reply_resolves_in_every_locale(
        self, conn: aiosqlite.Connection, locale: str
    ) -> None:
        interaction = _make_interaction(db=conn, locale=locale)

        await _invoke(interaction, _make_channel(can_post=False), _choice(DigestInterval.WEEKLY))

        message = _reply(interaction)
        assert "[digest_" not in message  # no missing translation keys

    async def test_an_unknown_locale_falls_back_to_english(
        self, conn: aiosqlite.Connection
    ) -> None:
        interaction = _make_interaction(db=conn, locale="xx-XX")

        await _invoke(interaction)

        assert _reply(interaction) == t("digest_no_options_error", "en-US")


class TestErrorHandler:
    async def test_a_missing_permission_gets_a_localized_reply(self) -> None:
        interaction = _make_interaction(db=None, locale="de")

        await _handle_digest_command_error(
            interaction, app_commands.MissingPermissions(["manage_guild"])
        )

        interaction.response.send_message.assert_awaited_once_with(
            t("digest_permission_error", "de"), ephemeral=True
        )

    async def test_a_deferred_interaction_answers_through_the_followup(self) -> None:
        interaction = _make_interaction(db=None)
        interaction.response.is_done = MagicMock(return_value=True)

        await _handle_digest_command_error(
            interaction, app_commands.MissingPermissions(["manage_guild"])
        )

        interaction.followup.send.assert_awaited_once()
        interaction.response.send_message.assert_not_awaited()

    async def test_any_other_error_is_logged_rather_than_swallowed(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        # Attaching a local handler stops CommandTree's own logging, so this is
        # the only place an unexpected failure can surface.
        interaction = _make_interaction(db=None)

        with caplog.at_level(logging.ERROR):
            await _handle_digest_command_error(
                interaction, app_commands.AppCommandError("something else")
            )

        assert any(record.levelno >= logging.ERROR for record in caplog.records)
