"""Tests for aura.commands.facts: manual fact entry.

Exercises the command callbacks, the permission checks, the modal, and the
shared error handler directly against mocked discord.Interaction/Message
objects and a real in-memory database -- never a live Discord connection,
per CLAUDE.md's testing philosophy ("a live Discord connection should never
be required to verify this logic is correct").

Checks and error handlers are invoked directly rather than through discord.py's
full dispatch machinery (which would need a real gateway connection to drive),
matching how discord.py itself exposes them: Command/ContextMenu.callback is
the undecorated function, and .checks is the list of predicates a real
dispatch would run before calling it.
"""
from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock

import aiosqlite
import discord
import pytest
from discord import app_commands

from aura.commands.facts import (
    AddFactModal,
    _FIELD_VALUE_DISPLAY_LIMIT,
    _LIST_DISPLAY_LIMIT,
    _TEXT_INPUT_MAX_LENGTH,
    _handle_fact_command_error,
    add_fact_context_menu,
    list_facts_command,
)
from aura.db.repository import get_active_facts, init_schema
from aura.facts_service import add_fact

GUILD_A = 100000000000000001
GUILD_B = 200000000000000002


@pytest.fixture
async def conn():
    connection = await aiosqlite.connect(":memory:")
    await init_schema(connection)
    yield connection
    await connection.close()


def _make_interaction(
    *, db: aiosqlite.Connection | None, locale: str = "en-US", guild_id: int = GUILD_A
) -> MagicMock:
    """A mock Interaction exposing just what the fact commands actually touch."""
    interaction = MagicMock(spec=discord.Interaction)
    interaction.locale = locale
    interaction.guild_id = guild_id
    interaction.client = MagicMock()
    interaction.client.db = db
    interaction.response = MagicMock()
    interaction.response.send_message = AsyncMock()
    interaction.response.send_modal = AsyncMock()
    interaction.response.is_done = MagicMock(return_value=False)
    interaction.followup = MagicMock()
    interaction.followup.send = AsyncMock()
    interaction.command = MagicMock()
    interaction.command.name = "test-command"
    return interaction


def _make_message(
    *, content: str, guild_id: int | None = GUILD_A, channel_id: int = 1, message_id: int = 1
) -> MagicMock:
    """A mock Message exposing just what add_fact_context_menu reads."""
    message = MagicMock(spec=discord.Message)
    message.content = content
    if guild_id is None:
        message.guild = None
    else:
        message.guild = MagicMock()
        message.guild.id = guild_id
    message.channel = MagicMock()
    message.channel.id = channel_id
    message.id = message_id
    return message


def _submit(modal: AddFactModal, value: str) -> None:
    """Simulate what a real modal submission populates: TextInput.value has no
    public setter (discord.py only ever writes to it internally, via the
    payload it receives back from Discord), so this pokes the same private
    attribute discord.py's own _refresh_state does."""
    modal.content_input._value = value  # type: ignore[attr-defined]


async def _invoke_list_facts(interaction: discord.Interaction) -> None:
    """Call list_facts_command's callback directly, bypassing its checks.

    discord.py types Command.callback as a union that also covers
    group/cog-bound callbacks (taking a leading self/group argument before
    interaction). list_facts_command is neither, so at runtime this callback
    only ever takes (interaction,) -- confirmed directly against the
    installed discord.py -- but Pyright can't rule out the other union member
    from the stub alone and flags a bare call site as missing an argument.
    """
    await list_facts_command.callback(interaction)  # pyright: ignore[reportCallIssue]


class TestPermissionChecks:
    """"Permission bypass attempt": both commands must reject a non-moderator."""

    def test_context_menu_rejects_non_moderator(self) -> None:
        fake_interaction = MagicMock(permissions=discord.Permissions(manage_guild=False))
        with pytest.raises(app_commands.MissingPermissions):
            for check in add_fact_context_menu.checks:
                check(fake_interaction)

    def test_context_menu_allows_moderator(self) -> None:
        fake_interaction = MagicMock(permissions=discord.Permissions(manage_guild=True))
        for check in add_fact_context_menu.checks:
            assert check(fake_interaction) is True

    def test_list_command_rejects_non_moderator(self) -> None:
        fake_interaction = MagicMock(permissions=discord.Permissions(manage_guild=False))
        with pytest.raises(app_commands.MissingPermissions):
            for check in list_facts_command.checks:
                check(fake_interaction)

    def test_list_command_allows_moderator(self) -> None:
        fake_interaction = MagicMock(permissions=discord.Permissions(manage_guild=True))
        for check in list_facts_command.checks:
            assert check(fake_interaction) is True


class TestErrorHandler:
    """The MissingPermissions raised above must never surface as an unhandled error."""

    async def test_missing_permissions_replies_ephemerally_and_localized(self) -> None:
        interaction = _make_interaction(db=None)
        error = app_commands.MissingPermissions(["manage_guild"])

        await _handle_fact_command_error(interaction, error)

        interaction.response.send_message.assert_awaited_once()
        args, kwargs = interaction.response.send_message.call_args
        assert "permission" in args[0].lower()
        assert kwargs.get("ephemeral") is True

    async def test_missing_permissions_uses_followup_if_already_responded(self) -> None:
        interaction = _make_interaction(db=None)
        interaction.response.is_done = MagicMock(return_value=True)
        error = app_commands.MissingPermissions(["manage_guild"])

        await _handle_fact_command_error(interaction, error)

        interaction.followup.send.assert_awaited_once()
        interaction.response.send_message.assert_not_awaited()

    async def test_unexpected_errors_are_logged_and_not_surfaced_to_the_user(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        interaction = _make_interaction(db=None)
        error = app_commands.CommandInvokeError(MagicMock(), ValueError("boom"))

        with caplog.at_level(logging.ERROR):
            await _handle_fact_command_error(interaction, error)

        interaction.response.send_message.assert_not_awaited()
        interaction.followup.send.assert_not_awaited()
        assert any(record.levelno == logging.ERROR for record in caplog.records)


class TestAddFactModalConstruction:
    """"No-text source message" / "Oversized source message" construction-time checks."""

    def test_prefill_becomes_the_default(self) -> None:
        modal = AddFactModal(
            db=MagicMock(),
            locale="en-US",
            guild_id=GUILD_A,
            channel_id=1,
            message_id=1,
            prefill_content="hello world",
        )
        assert modal.content_input.default == "hello world"

    def test_empty_prefill_results_in_an_empty_field_not_an_error(self) -> None:
        modal = AddFactModal(
            db=MagicMock(),
            locale="en-US",
            guild_id=GUILD_A,
            channel_id=1,
            message_id=1,
            prefill_content="",
        )
        assert modal.content_input.default is None

    def test_oversized_prefill_is_truncated_to_the_text_inputs_own_max(self) -> None:
        huge = "x" * (_TEXT_INPUT_MAX_LENGTH + 1000)
        modal = AddFactModal(
            db=MagicMock(),
            locale="en-US",
            guild_id=GUILD_A,
            channel_id=1,
            message_id=1,
            prefill_content=huge,
        )
        assert modal.content_input.default is not None
        assert len(modal.content_input.default) == _TEXT_INPUT_MAX_LENGTH
        assert modal.content_input.max_length == _TEXT_INPUT_MAX_LENGTH

    def test_prefill_exactly_at_the_limit_is_not_truncated(self) -> None:
        exact = "y" * _TEXT_INPUT_MAX_LENGTH
        modal = AddFactModal(
            db=MagicMock(),
            locale="en-US",
            guild_id=GUILD_A,
            channel_id=1,
            message_id=1,
            prefill_content=exact,
        )
        assert modal.content_input.default == exact

    def test_title_and_label_are_localized(self) -> None:
        modal = AddFactModal(
            db=MagicMock(),
            locale="en-US",
            guild_id=GUILD_A,
            channel_id=1,
            message_id=1,
            prefill_content="",
        )
        assert modal.title == "Add as Aura Fact"


class TestAddFactContextMenuCallback:
    async def test_opens_a_modal_prefilled_from_the_message(self, conn: aiosqlite.Connection) -> None:
        interaction = _make_interaction(db=conn)
        message = _make_message(content="the pinned announcement", guild_id=GUILD_A)

        await add_fact_context_menu.callback(interaction, message)

        interaction.response.send_modal.assert_awaited_once()
        (modal,), _ = interaction.response.send_modal.call_args
        assert isinstance(modal, AddFactModal)
        assert modal.content_input.default == "the pinned announcement"

    async def test_no_text_message_opens_a_modal_with_an_empty_field(
        self, conn: aiosqlite.Connection
    ) -> None:
        interaction = _make_interaction(db=conn)
        message = _make_message(content="", guild_id=GUILD_A)  # attachment/embed-only message

        await add_fact_context_menu.callback(interaction, message)

        interaction.response.send_modal.assert_awaited_once()
        (modal,), _ = interaction.response.send_modal.call_args
        assert modal.content_input.default is None

    async def test_uses_the_messages_own_guild_not_some_other_source(
        self, conn: aiosqlite.Connection
    ) -> None:
        interaction = _make_interaction(db=conn, guild_id=GUILD_A)
        message = _make_message(content="x", guild_id=GUILD_B, channel_id=7, message_id=42)

        await add_fact_context_menu.callback(interaction, message)

        (modal,), _ = interaction.response.send_modal.call_args
        assert modal._guild_id == GUILD_B
        assert modal._channel_id == 7
        assert modal._message_id == 42


class TestAddFactModalSubmit:
    async def test_empty_submission_creates_no_fact(self, conn: aiosqlite.Connection) -> None:
        modal = AddFactModal(
            db=conn, locale="en-US", guild_id=GUILD_A, channel_id=1, message_id=1,
            prefill_content="",
        )
        _submit(modal, "")
        interaction = _make_interaction(db=conn)

        await modal.on_submit(interaction)

        interaction.response.send_message.assert_awaited_once()
        args, _ = interaction.response.send_message.call_args
        assert "empty" in args[0].lower()
        assert await get_active_facts(conn, GUILD_A) == []

    async def test_whitespace_only_submission_creates_no_fact(
        self, conn: aiosqlite.Connection
    ) -> None:
        modal = AddFactModal(
            db=conn, locale="en-US", guild_id=GUILD_A, channel_id=1, message_id=1,
            prefill_content="",
        )
        _submit(modal, "   \n\t  \n  ")
        interaction = _make_interaction(db=conn)

        await modal.on_submit(interaction)

        assert await get_active_facts(conn, GUILD_A) == []

    async def test_valid_submission_creates_a_fact_and_the_reply_names_its_id(
        self, conn: aiosqlite.Connection
    ) -> None:
        modal = AddFactModal(
            db=conn, locale="en-US", guild_id=GUILD_A, channel_id=5, message_id=9,
            prefill_content="",
        )
        _submit(modal, "  the server was founded in 2020  ")
        interaction = _make_interaction(db=conn)

        await modal.on_submit(interaction)

        [fact] = await get_active_facts(conn, GUILD_A)
        assert fact.content == "the server was founded in 2020"  # surrounding whitespace stripped
        assert fact.channel_id == 5
        assert fact.message_id == 9

        args, kwargs = interaction.response.send_message.call_args
        assert str(fact.id) in args[0]
        assert kwargs.get("ephemeral") is True

    async def test_unicode_content_round_trips(self, conn: aiosqlite.Connection) -> None:
        content = "サーバーのルールは 日本語 でも読めます 🎉 مرحبا بكم"
        modal = AddFactModal(
            db=conn, locale="en-US", guild_id=GUILD_A, channel_id=1, message_id=1,
            prefill_content="",
        )
        _submit(modal, content)
        interaction = _make_interaction(db=conn)

        await modal.on_submit(interaction)

        [fact] = await get_active_facts(conn, GUILD_A)
        assert fact.content == content


class TestListFactsCommand:
    async def test_zero_facts_shows_localized_empty_state(self, conn: aiosqlite.Connection) -> None:
        interaction = _make_interaction(db=conn, guild_id=GUILD_A)

        await _invoke_list_facts(interaction)

        interaction.response.send_message.assert_awaited_once()
        args, kwargs = interaction.response.send_message.call_args
        assert "no active facts" in args[0].lower()
        assert kwargs.get("ephemeral") is True
        assert "embed" not in kwargs

    async def test_guild_isolation_at_the_command_layer(self, conn: aiosqlite.Connection) -> None:
        await add_fact(conn, guild_id=GUILD_A, channel_id=1, message_id=1, content="guild A fact")
        await add_fact(conn, guild_id=GUILD_B, channel_id=1, message_id=2, content="guild B fact")

        interaction = _make_interaction(db=conn, guild_id=GUILD_A)
        await _invoke_list_facts(interaction)

        _, kwargs = interaction.response.send_message.call_args
        embed = kwargs["embed"]
        assert len(embed.fields) == 1
        assert embed.fields[0].value == "guild A fact"

    async def test_over_display_limit_adds_a_correctly_counted_truncation_note(
        self, conn: aiosqlite.Connection
    ) -> None:
        total = _LIST_DISPLAY_LIMIT + 5
        for i in range(total):
            await add_fact(conn, guild_id=GUILD_A, channel_id=1, message_id=i, content=f"fact {i}")

        interaction = _make_interaction(db=conn, guild_id=GUILD_A)
        await _invoke_list_facts(interaction)

        _, kwargs = interaction.response.send_message.call_args
        embed = kwargs["embed"]
        assert len(embed.fields) == _LIST_DISPLAY_LIMIT
        assert embed.footer.text is not None
        assert "5" in embed.footer.text

    async def test_at_exactly_the_display_limit_no_truncation_note_appears(
        self, conn: aiosqlite.Connection
    ) -> None:
        for i in range(_LIST_DISPLAY_LIMIT):
            await add_fact(conn, guild_id=GUILD_A, channel_id=1, message_id=i, content=f"fact {i}")

        interaction = _make_interaction(db=conn, guild_id=GUILD_A)
        await _invoke_list_facts(interaction)

        _, kwargs = interaction.response.send_message.call_args
        embed = kwargs["embed"]
        assert len(embed.fields) == _LIST_DISPLAY_LIMIT
        assert embed.footer.text is None

    async def test_unicode_content_is_displayed_unmangled(self, conn: aiosqlite.Connection) -> None:
        content = "日本語のファクト 🎉 مرحبا"
        await add_fact(conn, guild_id=GUILD_A, channel_id=1, message_id=1, content=content)

        interaction = _make_interaction(db=conn, guild_id=GUILD_A)
        await _invoke_list_facts(interaction)

        _, kwargs = interaction.response.send_message.call_args
        assert kwargs["embed"].fields[0].value == content

    async def test_long_fact_content_is_truncated_for_display_not_left_to_hit_discords_limit(
        self, conn: aiosqlite.Connection
    ) -> None:
        long_content = "z" * 3000  # well within TextInput's 4000 cap, but not for a list row
        await add_fact(conn, guild_id=GUILD_A, channel_id=1, message_id=1, content=long_content)

        interaction = _make_interaction(db=conn, guild_id=GUILD_A)
        await _invoke_list_facts(interaction)

        _, kwargs = interaction.response.send_message.call_args
        field_value = kwargs["embed"].fields[0].value
        assert len(field_value) <= _FIELD_VALUE_DISPLAY_LIMIT
        assert field_value.endswith("…")
