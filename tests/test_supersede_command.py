"""Tests for aura.commands.supersede: the manual /aura-supersede command.

Phase 1c left this command out entirely, so nothing has ever actually been
marked superseded in production despite Phase 1b's supersede_fact existing
and being tested since then. This file exercises the command callback, the
confirmation view's buttons, the permission checks, and the shared error
handler directly against mocked discord.Interaction objects and a real
in-memory database -- never a live Discord connection, matching
test_facts_commands.py's approach and CLAUDE.md's testing philosophy.
"""
from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock

import aiosqlite
import discord
import pytest
from discord import app_commands
from fastembed import TextEmbedding

from aura.commands.facts import list_facts_command
from aura.commands.supersede import (
    SupersedeConfirmView,
    _handle_supersede_command_error,
    supersede_command,
)
from aura.db.repository import get_active_facts, get_fact_by_id, init_schema
from aura.embeddings import find_similar_facts
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
    *,
    db: aiosqlite.Connection | None,
    locale: str = "en-US",
    guild_id: int = GUILD_A,
    user_id: int = 1,
) -> MagicMock:
    """A mock Interaction exposing just what the supersede command actually touches."""
    interaction = MagicMock(spec=discord.Interaction)
    interaction.locale = locale
    interaction.guild_id = guild_id
    interaction.client = MagicMock()
    interaction.client.db = db
    interaction.user = MagicMock()
    interaction.user.id = user_id
    interaction.response = MagicMock()
    interaction.response.send_message = AsyncMock()
    interaction.response.edit_message = AsyncMock()
    interaction.response.is_done = MagicMock(return_value=False)
    interaction.followup = MagicMock()
    interaction.followup.send = AsyncMock()
    interaction.original_response = AsyncMock(return_value=MagicMock())
    interaction.command = MagicMock()
    interaction.command.name = "test-command"
    return interaction


async def _invoke(
    interaction: discord.Interaction, old_fact_id: int, new_fact_id: int
) -> None:
    """Call supersede_command's callback directly, bypassing its checks."""
    await supersede_command.callback(interaction, old_fact_id, new_fact_id)  # pyright: ignore[reportCallIssue, reportArgumentType]


async def _add(
    conn: aiosqlite.Connection,
    model: TextEmbedding,
    *,
    guild_id: int = GUILD_A,
    content: str,
    channel_id: int = 1,
    message_id: int = 1,
):
    return await add_fact(
        conn, model, guild_id=guild_id, channel_id=channel_id, message_id=message_id, content=content
    )


class TestPermissionChecks:
    def test_rejects_non_moderator(self) -> None:
        fake_interaction = MagicMock(permissions=discord.Permissions(manage_guild=False))
        with pytest.raises(app_commands.MissingPermissions):
            for check in supersede_command.checks:
                check(fake_interaction)

    def test_allows_moderator(self) -> None:
        fake_interaction = MagicMock(permissions=discord.Permissions(manage_guild=True))
        for check in supersede_command.checks:
            assert check(fake_interaction) is True


class TestErrorHandler:
    async def test_missing_permissions_replies_ephemerally_and_localized(self) -> None:
        interaction = _make_interaction(db=None)
        error = app_commands.MissingPermissions(["manage_guild"])

        await _handle_supersede_command_error(interaction, error)

        interaction.response.send_message.assert_awaited_once()
        args, kwargs = interaction.response.send_message.call_args
        assert "permission" in args[0].lower()
        assert kwargs.get("ephemeral") is True

    async def test_missing_permissions_uses_followup_if_already_responded(self) -> None:
        interaction = _make_interaction(db=None)
        interaction.response.is_done = MagicMock(return_value=True)
        error = app_commands.MissingPermissions(["manage_guild"])

        await _handle_supersede_command_error(interaction, error)

        interaction.followup.send.assert_awaited_once()
        interaction.response.send_message.assert_not_awaited()

    async def test_unexpected_errors_are_logged_and_not_surfaced_to_the_user(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        interaction = _make_interaction(db=None)
        error = app_commands.CommandInvokeError(MagicMock(), ValueError("boom"))

        with caplog.at_level(logging.ERROR):
            await _handle_supersede_command_error(interaction, error)

        interaction.response.send_message.assert_not_awaited()
        interaction.followup.send.assert_not_awaited()
        assert any(record.levelno == logging.ERROR for record in caplog.records)


class TestSupersedeCommandValidation:
    """Every rejection path the command must handle before ever showing a confirmation."""

    async def test_self_supersession_is_rejected_with_no_db_write(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        fact = await _add(conn, embedding_model, content="a fact")
        interaction = _make_interaction(db=conn)

        await _invoke(interaction, fact.id, fact.id)

        interaction.response.send_message.assert_awaited_once()
        args, kwargs = interaction.response.send_message.call_args
        assert "supersede itself" in args[0].lower()
        assert kwargs.get("ephemeral") is True
        assert "view" not in kwargs
        [still_active] = await get_active_facts(conn, GUILD_A)
        assert still_active.id == fact.id

    async def test_nonexistent_old_fact_is_rejected(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        new = await _add(conn, embedding_model, content="new fact")
        interaction = _make_interaction(db=conn)

        await _invoke(interaction, 999999, new.id)

        args, kwargs = interaction.response.send_message.call_args
        assert "999999" in args[0]
        assert "view" not in kwargs

    async def test_nonexistent_new_fact_is_rejected(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        old = await _add(conn, embedding_model, content="old fact")
        interaction = _make_interaction(db=conn)

        await _invoke(interaction, old.id, 999999)

        args, kwargs = interaction.response.send_message.call_args
        assert "999999" in args[0]
        assert "view" not in kwargs

    async def test_another_guilds_fact_id_is_treated_as_not_found(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        other_guild_fact = await _add(conn, embedding_model, guild_id=GUILD_B, content="not yours")
        new = await _add(conn, embedding_model, content="new fact")
        interaction = _make_interaction(db=conn, guild_id=GUILD_A)

        await _invoke(interaction, other_guild_fact.id, new.id)

        args, kwargs = interaction.response.send_message.call_args
        assert str(other_guild_fact.id) in args[0]
        assert "view" not in kwargs

    async def test_already_superseded_old_fact_is_rejected(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        old = await _add(conn, embedding_model, content="old fact")
        first_new = await _add(conn, embedding_model, content="first replacement")
        interaction = _make_interaction(db=conn)
        await _invoke(interaction, old.id, first_new.id)
        confirm_view = interaction.response.send_message.call_args.kwargs["view"]
        await confirm_view.confirm.callback(_make_interaction(db=conn, user_id=1))

        second_new = await _add(conn, embedding_model, content="second replacement")
        second_interaction = _make_interaction(db=conn)
        await _invoke(second_interaction, old.id, second_new.id)

        args, kwargs = second_interaction.response.send_message.call_args
        assert "already superseded" in args[0].lower()
        assert "view" not in kwargs

    async def test_already_superseded_new_fact_cannot_be_chosen_as_successor(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        original = await _add(conn, embedding_model, content="original")
        first_interaction = _make_interaction(db=conn)
        replacement = await _add(conn, embedding_model, content="replacement")
        await _invoke(first_interaction, original.id, replacement.id)
        view = first_interaction.response.send_message.call_args.kwargs["view"]
        await view.confirm.callback(_make_interaction(db=conn, user_id=1))
        # `original` is now superseded by `replacement`; picking `original` as
        # a *successor* for some unrelated fact must be rejected.
        other_old = await _add(conn, embedding_model, content="unrelated old fact")
        interaction = _make_interaction(db=conn)

        await _invoke(interaction, other_old.id, original.id)

        args, kwargs = interaction.response.send_message.call_args
        assert "isn't currently active" in args[0].lower()
        assert "view" not in kwargs

    async def test_valid_references_show_a_confirmation_view_with_no_db_write_yet(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        old = await _add(conn, embedding_model, content="the old fact content")
        new = await _add(conn, embedding_model, content="the new fact content")
        interaction = _make_interaction(db=conn)

        await _invoke(interaction, old.id, new.id)

        interaction.response.send_message.assert_awaited_once()
        _, kwargs = interaction.response.send_message.call_args
        assert isinstance(kwargs["view"], SupersedeConfirmView)
        embed = kwargs["embed"]
        assert "the old fact content" in "".join(f.value or "" for f in embed.fields)
        assert "the new fact content" in "".join(f.value or "" for f in embed.fields)

        # Nothing committed just by showing the confirmation.
        old_readback = await get_fact_by_id(conn, guild_id=GUILD_A, fact_id=old.id)
        assert old_readback is not None
        assert old_readback.status.value == "active"


class TestSupersedeConfirmView:
    async def test_confirm_commits_the_supersession(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        old = await _add(conn, embedding_model, content="old fact")
        new = await _add(conn, embedding_model, content="new fact")
        view = SupersedeConfirmView(
            db=conn, guild_id=GUILD_A, old_fact=old, new_fact=new, locale="en-US", invoker_id=1
        )
        interaction = _make_interaction(db=conn, user_id=1)

        await view.confirm.callback(interaction)

        interaction.response.edit_message.assert_awaited_once()
        _, kwargs = interaction.response.edit_message.call_args
        assert str(old.id) in kwargs["content"]
        assert str(new.id) in kwargs["content"]
        assert kwargs["view"] is None

        old_readback = await get_fact_by_id(conn, guild_id=GUILD_A, fact_id=old.id)
        assert old_readback is not None
        assert old_readback.status.value == "superseded"
        assert old_readback.superseded_by_id == new.id

    async def test_cancel_commits_nothing(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        old = await _add(conn, embedding_model, content="old fact")
        new = await _add(conn, embedding_model, content="new fact")
        view = SupersedeConfirmView(
            db=conn, guild_id=GUILD_A, old_fact=old, new_fact=new, locale="en-US", invoker_id=1
        )
        interaction = _make_interaction(db=conn, user_id=1)

        await view.cancel.callback(interaction)

        interaction.response.edit_message.assert_awaited_once()
        _, kwargs = interaction.response.edit_message.call_args
        assert "cancelled" in kwargs["content"].lower()

        old_readback = await get_fact_by_id(conn, guild_id=GUILD_A, fact_id=old.id)
        assert old_readback is not None
        assert old_readback.status.value == "active"

    async def test_wrong_user_cannot_confirm(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        old = await _add(conn, embedding_model, content="old fact")
        new = await _add(conn, embedding_model, content="new fact")
        view = SupersedeConfirmView(
            db=conn, guild_id=GUILD_A, old_fact=old, new_fact=new, locale="en-US", invoker_id=1
        )
        stranger_interaction = _make_interaction(db=conn, user_id=999)

        allowed = await view.interaction_check(stranger_interaction)

        assert allowed is False
        stranger_interaction.response.send_message.assert_awaited_once()
        old_readback = await get_fact_by_id(conn, guild_id=GUILD_A, fact_id=old.id)
        assert old_readback is not None
        assert old_readback.status.value == "active"

    async def test_wrong_user_cannot_cancel_either(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        old = await _add(conn, embedding_model, content="old fact")
        new = await _add(conn, embedding_model, content="new fact")
        view = SupersedeConfirmView(
            db=conn, guild_id=GUILD_A, old_fact=old, new_fact=new, locale="en-US", invoker_id=1
        )
        stranger_interaction = _make_interaction(db=conn, user_id=999)

        allowed = await view.interaction_check(stranger_interaction)
        assert allowed is False

    async def test_confirm_race_old_fact_already_superseded_reports_cleanly(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        old = await _add(conn, embedding_model, content="old fact")
        first_new = await _add(conn, embedding_model, content="first replacement")
        second_new = await _add(conn, embedding_model, content="second replacement")

        # Simulate a second mod's concurrent supersession winning the race.
        view = SupersedeConfirmView(
            db=conn, guild_id=GUILD_A, old_fact=old, new_fact=second_new, locale="en-US", invoker_id=1
        )
        from aura.db.repository import supersede_fact_with_existing_successor

        await supersede_fact_with_existing_successor(
            conn, old_fact_id=old.id, new_fact_id=first_new.id, guild_id=GUILD_A
        )

        interaction = _make_interaction(db=conn, user_id=1)
        await view.confirm.callback(interaction)

        _, kwargs = interaction.response.edit_message.call_args
        assert "already superseded" in kwargs["content"].lower()
        old_readback = await get_fact_by_id(conn, guild_id=GUILD_A, fact_id=old.id)
        assert old_readback is not None
        # The winning racer's chain must survive untouched.
        assert old_readback.superseded_by_id == first_new.id

    async def test_confirm_race_new_fact_no_longer_active_reports_cleanly(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        old = await _add(conn, embedding_model, content="old fact")
        chosen_new = await _add(conn, embedding_model, content="chosen replacement")
        even_newer = await _add(conn, embedding_model, content="an even newer fact")

        view = SupersedeConfirmView(
            db=conn, guild_id=GUILD_A, old_fact=old, new_fact=chosen_new, locale="en-US", invoker_id=1
        )
        from aura.db.repository import supersede_fact_with_existing_successor

        # chosen_new gets superseded itself before the confirm click lands.
        await supersede_fact_with_existing_successor(
            conn, old_fact_id=chosen_new.id, new_fact_id=even_newer.id, guild_id=GUILD_A
        )

        interaction = _make_interaction(db=conn, user_id=1)
        await view.confirm.callback(interaction)

        _, kwargs = interaction.response.edit_message.call_args
        assert "no longer active" in kwargs["content"].lower()
        old_readback = await get_fact_by_id(conn, guild_id=GUILD_A, fact_id=old.id)
        assert old_readback is not None
        assert old_readback.status.value == "active"  # untouched by the failed attempt

    async def test_on_timeout_edits_the_stored_message_and_writes_nothing(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        old = await _add(conn, embedding_model, content="old fact")
        new = await _add(conn, embedding_model, content="new fact")
        view = SupersedeConfirmView(
            db=conn, guild_id=GUILD_A, old_fact=old, new_fact=new, locale="en-US", invoker_id=1
        )
        view.message = MagicMock()
        view.message.edit = AsyncMock()

        await view.on_timeout()

        view.message.edit.assert_awaited_once()
        _, kwargs = view.message.edit.call_args
        assert "timed out" in kwargs["content"].lower()
        old_readback = await get_fact_by_id(conn, guild_id=GUILD_A, fact_id=old.id)
        assert old_readback is not None
        assert old_readback.status.value == "active"

    async def test_confirm_then_cancel_double_fire_only_the_first_click_acts(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        # Simulates a moderator's client sending two component interactions
        # (Confirm then Cancel) before the first click's edit_message(view=None)
        # has propagated back to disable the buttons on their end.
        old = await _add(conn, embedding_model, content="old fact")
        new = await _add(conn, embedding_model, content="new fact")
        view = SupersedeConfirmView(
            db=conn, guild_id=GUILD_A, old_fact=old, new_fact=new, locale="en-US", invoker_id=1
        )
        confirm_interaction = _make_interaction(db=conn, user_id=1)
        cancel_interaction = _make_interaction(db=conn, user_id=1)
        cancel_interaction.response.is_done = MagicMock(return_value=False)
        cancel_interaction.response.defer = AsyncMock()

        await view.confirm.callback(confirm_interaction)
        await view.cancel.callback(cancel_interaction)

        # The commit from the first click stands...
        old_readback = await get_fact_by_id(conn, guild_id=GUILD_A, fact_id=old.id)
        assert old_readback is not None
        assert old_readback.status.value == "superseded"
        assert old_readback.superseded_by_id == new.id
        # ...and the second click never touched the displayed message or the DB.
        cancel_interaction.response.edit_message.assert_not_awaited()
        cancel_interaction.response.defer.assert_awaited_once()

    async def test_double_click_of_the_same_confirm_button_only_commits_once(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        old = await _add(conn, embedding_model, content="old fact")
        new = await _add(conn, embedding_model, content="new fact")
        view = SupersedeConfirmView(
            db=conn, guild_id=GUILD_A, old_fact=old, new_fact=new, locale="en-US", invoker_id=1
        )
        first_click = _make_interaction(db=conn, user_id=1)
        second_click = _make_interaction(db=conn, user_id=1)
        second_click.response.is_done = MagicMock(return_value=False)
        second_click.response.defer = AsyncMock()

        await view.confirm.callback(first_click)
        await view.confirm.callback(second_click)

        first_click.response.edit_message.assert_awaited_once()
        second_click.response.edit_message.assert_not_awaited()
        second_click.response.defer.assert_awaited_once()
        old_readback = await get_fact_by_id(conn, guild_id=GUILD_A, fact_id=old.id)
        assert old_readback is not None
        assert old_readback.superseded_by_id == new.id

    async def test_button_labels_are_localized(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        old = await _add(conn, embedding_model, content="old fact")
        new = await _add(conn, embedding_model, content="new fact")
        view = SupersedeConfirmView(
            db=conn, guild_id=GUILD_A, old_fact=old, new_fact=new, locale="en-US", invoker_id=1
        )
        assert view.confirm.label == "Confirm"
        assert view.cancel.label == "Cancel"


class TestEndToEndExclusionFromSimilaritySearch:
    """Deliverable 5: a freshly-superseded fact must vanish from find_similar_facts immediately."""

    async def test_superseded_fact_is_immediately_excluded_with_no_caching_lag(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        old = await _add(conn, embedding_model, content="the event is in room 204")
        new = await _add(conn, embedding_model, content="the event is in room 305")

        before = await find_similar_facts(conn, embedding_model, guild_id=GUILD_A, query="what room is the event in?")
        assert old.id in {fact.id for fact, _ in before}

        view = SupersedeConfirmView(
            db=conn, guild_id=GUILD_A, old_fact=old, new_fact=new, locale="en-US", invoker_id=1
        )
        interaction = _make_interaction(db=conn, user_id=1)
        await view.confirm.callback(interaction)

        after = await find_similar_facts(conn, embedding_model, guild_id=GUILD_A, query="what room is the event in?")
        after_ids = {fact.id for fact, _ in after}
        assert old.id not in after_ids
        assert new.id in after_ids

    async def test_aura_facts_listing_reflects_the_supersession_afterward(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        """Deliverable 6: /aura-facts still shows exactly the active fact, per Phase 1b's
        "nothing is ever deleted, only marked superseded" principle -- the old fact's row
        must still exist (proven by get_fact_by_id elsewhere), just no longer listed as active.
        """
        old = await _add(conn, embedding_model, content="the office is on the 2nd floor")
        new = await _add(conn, embedding_model, content="the office is on the 3rd floor")

        view = SupersedeConfirmView(
            db=conn, guild_id=GUILD_A, old_fact=old, new_fact=new, locale="en-US", invoker_id=1
        )
        await view.confirm.callback(_make_interaction(db=conn, user_id=1))

        list_interaction = _make_interaction(db=conn)
        await list_facts_command.callback(list_interaction, None)  # pyright: ignore[reportCallIssue, reportArgumentType]

        _, kwargs = list_interaction.response.send_message.call_args
        embed = kwargs["embed"]
        listed_ids = {int(field.name.lstrip("#")) for field in embed.fields}
        assert new.id in listed_ids
        assert old.id not in listed_ids

        old_readback = await get_fact_by_id(conn, guild_id=GUILD_A, fact_id=old.id)
        assert old_readback is not None
        assert old_readback.status.value == "superseded"
