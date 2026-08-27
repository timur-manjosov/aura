"""Tests for aura.commands.links: /aura-link and /aura-unlink, the mod-facing surface.

CLAUDE.md's fourth knowledge-model component had a table since Phase 1b and no
way for a human to write to it. These are the two commands that changed that,
exercised against mocked discord.Interaction objects and a real in-memory
database -- never a live Discord connection, matching test_supersede_command.py
and CLAUDE.md's testing philosophy.

The adversarial half is the larger one, deliberately: the interesting inputs
here are a moderator's typo, a second moderator racing them, and someone
guessing another server's fact IDs.
"""
from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock

import aiosqlite
import discord
import pytest
from discord import app_commands

from aura.commands.links import _handle_link_command_error, link_command, unlink_command
from aura.db.models import Fact
from aura.db.repository import (
    create_fact,
    get_linked_facts,
    init_schema,
    link_facts,
    supersede_fact_with_existing_successor,
)

GUILD_A = 100000000000000001
GUILD_B = 200000000000000002

_FAKE_EMBEDDING = bytes(384 * 4)


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
) -> MagicMock:
    """A mock Interaction exposing just what the link commands actually touch."""
    interaction = MagicMock(spec=discord.Interaction)
    interaction.locale = locale
    interaction.guild_id = guild_id
    interaction.client = MagicMock()
    interaction.client.db = db
    interaction.user = MagicMock()
    interaction.user.id = 1
    interaction.response = MagicMock()
    interaction.response.send_message = AsyncMock()
    interaction.response.is_done = MagicMock(return_value=False)
    interaction.followup = MagicMock()
    interaction.followup.send = AsyncMock()
    interaction.command = MagicMock()
    interaction.command.name = "aura-link"
    return interaction


async def _invoke_link(interaction: discord.Interaction, a: int, b: int) -> None:
    """Call link_command's callback directly, bypassing its checks."""
    await link_command.callback(interaction, a, b)  # pyright: ignore[reportCallIssue, reportArgumentType]


async def _invoke_unlink(interaction: discord.Interaction, a: int, b: int) -> None:
    """Call unlink_command's callback directly, bypassing its checks."""
    await unlink_command.callback(interaction, a, b)  # pyright: ignore[reportCallIssue, reportArgumentType]


async def _make_fact(
    conn: aiosqlite.Connection, *, guild_id: int = GUILD_A, content: str = "fact"
) -> Fact:
    return await create_fact(
        conn,
        guild_id=guild_id,
        channel_id=1,
        message_id=1,
        content=content,
        embedding=_FAKE_EMBEDDING,
    )


def _sent_content(interaction: MagicMock) -> str:
    """The text of the single message the command sent."""
    interaction.response.send_message.assert_awaited_once()
    args, kwargs = interaction.response.send_message.call_args
    if args:
        return str(args[0])
    return str(kwargs["content"])


async def _link_row_count(conn: aiosqlite.Connection) -> int:
    async with conn.execute("SELECT COUNT(*) FROM fact_links") as cursor:
        row = await cursor.fetchone()
    assert row is not None
    return row[0]


class TestPermissionChecks:
    @pytest.mark.parametrize("command", [link_command, unlink_command])
    def test_rejects_non_moderator(self, command: app_commands.Command) -> None:
        fake_interaction = MagicMock(permissions=discord.Permissions(manage_guild=False))
        with pytest.raises(app_commands.MissingPermissions):
            for check in command.checks:
                check(fake_interaction)

    @pytest.mark.parametrize("command", [link_command, unlink_command])
    def test_allows_moderator(self, command: app_commands.Command) -> None:
        fake_interaction = MagicMock(permissions=discord.Permissions(manage_guild=True))
        for check in command.checks:
            assert check(fake_interaction) is True

    @pytest.mark.parametrize("command", [link_command, unlink_command])
    def test_is_guild_only(self, command: app_commands.Command) -> None:
        assert command.guild_only is True

    @pytest.mark.parametrize("command", [link_command, unlink_command])
    def test_fact_ids_are_range_bounded_at_the_discord_layer(
        self, command: app_commands.Command
    ) -> None:
        # Declared to Discord so it rejects an out-of-range number before the
        # interaction is dispatched. Without the upper bound, a pasted
        # oversized ID reaches SQLite, which raises a bare OverflowError
        # instead of "no such fact" -- an "interaction failed" for the
        # moderator and a stack trace in the log. Asserted on the registered
        # parameters, not on the annotation, because that is what Discord
        # actually receives.
        for parameter in command.parameters:
            assert parameter.min_value == 1, parameter.name
            assert parameter.max_value == 9007199254740991, parameter.name


class TestErrorHandler:
    async def test_missing_permissions_replies_ephemerally_and_localized(self) -> None:
        interaction = _make_interaction(db=None)
        await _handle_link_command_error(
            interaction, app_commands.MissingPermissions(["manage_guild"])
        )

        interaction.response.send_message.assert_awaited_once()
        _, kwargs = interaction.response.send_message.call_args
        assert kwargs["ephemeral"] is True

    async def test_missing_permissions_uses_followup_once_a_response_exists(self) -> None:
        interaction = _make_interaction(db=None)
        interaction.response.is_done = MagicMock(return_value=True)
        await _handle_link_command_error(
            interaction, app_commands.MissingPermissions(["manage_guild"])
        )

        interaction.followup.send.assert_awaited_once()
        interaction.response.send_message.assert_not_awaited()

    async def test_other_errors_are_logged_not_swallowed(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        interaction = _make_interaction(db=None)
        with caplog.at_level(logging.ERROR, logger="aura.commands.links"):
            await _handle_link_command_error(
                interaction, app_commands.AppCommandError("something broke")
            )
        assert "aura-link" in caplog.text

    async def test_an_error_with_no_command_still_logs(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        # interaction.command is Optional on discord.py's own type; reading
        # .name off None would turn a logged error into a second one.
        interaction = _make_interaction(db=None)
        interaction.command = None
        with caplog.at_level(logging.ERROR, logger="aura.commands.links"):
            await _handle_link_command_error(
                interaction, app_commands.AppCommandError("something broke")
            )
        assert "unknown" in caplog.text


class TestLinkCommand:
    async def test_links_two_active_facts_and_says_so(self, conn: aiosqlite.Connection) -> None:
        a = await _make_fact(conn, content="The tournament starts on Saturday.")
        b = await _make_fact(conn, content="The winner gets a month of Nitro.")
        interaction = _make_interaction(db=conn)

        await _invoke_link(interaction, a.id, b.id)

        assert [f.id for f in await get_linked_facts(conn, guild_id=GUILD_A, fact_id=a.id)] == [
            b.id
        ]
        assert "now linked" in _sent_content(interaction)

    async def test_the_reply_is_ephemeral_and_shows_both_facts(
        self, conn: aiosqlite.Connection
    ) -> None:
        a = await _make_fact(conn, content="alpha content")
        b = await _make_fact(conn, content="beta content")
        interaction = _make_interaction(db=conn)

        await _invoke_link(interaction, a.id, b.id)

        _, kwargs = interaction.response.send_message.call_args
        assert kwargs["ephemeral"] is True
        embed = kwargs["embed"]
        field_values = [field.value for field in embed.fields]
        assert field_values == ["alpha content", "beta content"]

    async def test_argument_order_does_not_matter(self, conn: aiosqlite.Connection) -> None:
        a = await _make_fact(conn, content="a")
        b = await _make_fact(conn, content="b")
        interaction = _make_interaction(db=conn)

        await _invoke_link(interaction, b.id, a.id)

        assert await _link_row_count(conn) == 1

    async def test_linking_the_same_pair_twice_reports_no_change(
        self, conn: aiosqlite.Connection
    ) -> None:
        a = await _make_fact(conn, content="a")
        b = await _make_fact(conn, content="b")
        await link_facts(conn, guild_id=GUILD_A, fact_id_1=a.id, fact_id_2=b.id)
        interaction = _make_interaction(db=conn)

        await _invoke_link(interaction, a.id, b.id)

        assert "already linked" in _sent_content(interaction)
        assert await _link_row_count(conn) == 1

    async def test_self_link_is_refused_before_the_database_is_touched(
        self, conn: aiosqlite.Connection
    ) -> None:
        # The brief's explicit attack: a moderator typing the same ID twice.
        fact = await _make_fact(conn, content="a")
        interaction = _make_interaction(db=conn)

        await _invoke_link(interaction, fact.id, fact.id)

        assert "itself" in _sent_content(interaction)
        assert await _link_row_count(conn) == 0

    async def test_self_link_is_refused_even_for_a_nonexistent_fact(
        self, conn: aiosqlite.Connection
    ) -> None:
        # Ordering matters: the self-check runs first, so this is one message
        # about the real mistake rather than a confusing "no such fact".
        interaction = _make_interaction(db=conn)
        await _invoke_link(interaction, 999999, 999999)
        assert "itself" in _sent_content(interaction)

    async def test_a_nonexistent_fact_is_named_in_the_error(
        self, conn: aiosqlite.Connection
    ) -> None:
        real = await _make_fact(conn, content="a")
        interaction = _make_interaction(db=conn)

        await _invoke_link(interaction, real.id, 999999)

        assert "999999" in _sent_content(interaction)
        assert await _link_row_count(conn) == 0

    async def test_a_superseded_fact_is_refused_and_its_successor_named(
        self, conn: aiosqlite.Connection
    ) -> None:
        # The brief's "sensible error message when one fact is no longer
        # active" -- and the successor is what the moderator almost certainly
        # meant, so the message says which fact to link instead.
        old = await _make_fact(conn, content="old")
        new = await _make_fact(conn, content="new")
        other = await _make_fact(conn, content="other")
        await supersede_fact_with_existing_successor(
            conn, old_fact_id=old.id, new_fact_id=new.id, guild_id=GUILD_A
        )
        interaction = _make_interaction(db=conn)

        await _invoke_link(interaction, old.id, other.id)

        message = _sent_content(interaction)
        assert str(old.id) in message and str(new.id) in message
        assert await _link_row_count(conn) == 0

    async def test_a_superseded_fact_with_no_successor_still_gets_a_sentence(
        self, conn: aiosqlite.Connection
    ) -> None:
        # Broken data (only a hand edit produces it) must not become a
        # KeyError or a message with a blank ID in it.
        broken = await _make_fact(conn, content="broken")
        other = await _make_fact(conn, content="other")
        await conn.execute(
            "UPDATE facts SET status = 'superseded', superseded_at = ? WHERE id = ?",
            ("2026-01-01T00:00:00+00:00", broken.id),
        )
        await conn.commit()
        interaction = _make_interaction(db=conn)

        await _invoke_link(interaction, broken.id, other.id)

        message = _sent_content(interaction)
        assert str(broken.id) in message
        assert "None" not in message
        assert await _link_row_count(conn) == 0

    async def test_another_guilds_fact_reads_exactly_like_a_nonexistent_one(
        self, conn: aiosqlite.Connection
    ) -> None:
        # The isolation property, asserted by comparing the two messages
        # rather than by inspecting either: /aura-link must not be usable as
        # an oracle for which fact IDs exist in other servers.
        mine = await _make_fact(conn, guild_id=GUILD_A, content="mine")
        theirs = await _make_fact(conn, guild_id=GUILD_B, content="theirs")

        foreign_interaction = _make_interaction(db=conn)
        await _invoke_link(foreign_interaction, mine.id, theirs.id)
        foreign_message = _sent_content(foreign_interaction)

        missing_interaction = _make_interaction(db=conn)
        await _invoke_link(missing_interaction, mine.id, theirs.id + 500000)
        missing_message = _sent_content(missing_interaction)

        assert foreign_message.replace(str(theirs.id), "X") == missing_message.replace(
            str(theirs.id + 500000), "X"
        )
        assert await _link_row_count(conn) == 0

    async def test_a_supersession_racing_the_command_is_reported_not_crashed(
        self, conn: aiosqlite.Connection
    ) -> None:
        # Both facts pass the pre-flight checks, then another moderator's
        # /aura-supersede lands before the write. The repository's atomic
        # re-check is what catches it; this asserts the command turns that
        # into a sentence instead of a traceback.
        a = await _make_fact(conn, content="a")
        b = await _make_fact(conn, content="b")
        successor = await _make_fact(conn, content="successor")
        interaction = _make_interaction(db=conn)

        real_get_fact_by_id = __import__(
            "aura.commands.links", fromlist=["get_fact_by_id"]
        ).get_fact_by_id

        async def _supersede_after_lookup(*args: object, **kwargs: object):
            fact = await real_get_fact_by_id(*args, **kwargs)  # type: ignore[arg-type]
            if fact is not None and fact.id == b.id:
                await supersede_fact_with_existing_successor(
                    conn, old_fact_id=b.id, new_fact_id=successor.id, guild_id=GUILD_A
                )
            return fact

        import aura.commands.links as links_module

        links_module.get_fact_by_id = _supersede_after_lookup  # type: ignore[assignment]
        try:
            await _invoke_link(interaction, a.id, b.id)
        finally:
            links_module.get_fact_by_id = real_get_fact_by_id  # type: ignore[assignment]

        # "just now" appears only in the race message, never in the
        # pre-flight one -- so this cannot pass by taking the early exit.
        assert "just now" in _sent_content(interaction)
        assert await _link_row_count(conn) == 0

    async def test_unicode_and_oversized_content_do_not_break_the_embed(
        self, conn: aiosqlite.Connection
    ) -> None:
        # Fact content is capped at 4000 characters by the entry modal, which
        # is four times an embed field's own 1024 limit -- so the truncation
        # has to be real, and it has to survive multibyte text.
        a = await _make_fact(conn, content="🎉 サーバーのルール " * 500)
        b = await _make_fact(conn, content="b")
        interaction = _make_interaction(db=conn)

        await _invoke_link(interaction, a.id, b.id)

        _, kwargs = interaction.response.send_message.call_args
        for field in kwargs["embed"].fields:
            assert len(field.value) <= 1024


class TestUnlinkCommand:
    async def test_removes_an_existing_link(self, conn: aiosqlite.Connection) -> None:
        a = await _make_fact(conn, content="a")
        b = await _make_fact(conn, content="b")
        await link_facts(conn, guild_id=GUILD_A, fact_id_1=a.id, fact_id_2=b.id)
        interaction = _make_interaction(db=conn)

        await _invoke_unlink(interaction, a.id, b.id)

        assert await _link_row_count(conn) == 0
        assert "removed" in _sent_content(interaction)

    async def test_unlinking_what_was_not_linked_says_so_without_erroring(
        self, conn: aiosqlite.Connection
    ) -> None:
        a = await _make_fact(conn, content="a")
        b = await _make_fact(conn, content="b")
        interaction = _make_interaction(db=conn)

        await _invoke_unlink(interaction, a.id, b.id)

        assert "weren't linked" in _sent_content(interaction)

    async def test_a_superseded_fact_can_still_be_unlinked(
        self, conn: aiosqlite.Connection
    ) -> None:
        # The asymmetry with /aura-link, and the reason for it: these are
        # exactly the links a moderator most wants to clean up.
        a = await _make_fact(conn, content="a")
        b = await _make_fact(conn, content="b")
        successor = await _make_fact(conn, content="successor")
        await link_facts(conn, guild_id=GUILD_A, fact_id_1=a.id, fact_id_2=b.id)
        await supersede_fact_with_existing_successor(
            conn, old_fact_id=b.id, new_fact_id=successor.id, guild_id=GUILD_A
        )
        interaction = _make_interaction(db=conn)

        await _invoke_unlink(interaction, a.id, b.id)

        assert await _link_row_count(conn) == 0
        assert "removed" in _sent_content(interaction)

    async def test_self_unlink_is_refused(self, conn: aiosqlite.Connection) -> None:
        fact = await _make_fact(conn, content="a")
        interaction = _make_interaction(db=conn)

        await _invoke_unlink(interaction, fact.id, fact.id)

        assert "itself" in _sent_content(interaction)

    async def test_a_nonexistent_fact_is_named(self, conn: aiosqlite.Connection) -> None:
        real = await _make_fact(conn, content="a")
        interaction = _make_interaction(db=conn)

        await _invoke_unlink(interaction, real.id, 999999)

        assert "999999" in _sent_content(interaction)

    async def test_another_guild_cannot_tear_down_this_guilds_link(
        self, conn: aiosqlite.Connection
    ) -> None:
        # Guild B's moderator guesses guild A's fact IDs. The command must
        # refuse at the fact lookup, and the link must survive.
        a = await _make_fact(conn, guild_id=GUILD_A, content="a")
        b = await _make_fact(conn, guild_id=GUILD_A, content="b")
        await link_facts(conn, guild_id=GUILD_A, fact_id_1=a.id, fact_id_2=b.id)
        interaction = _make_interaction(db=conn, guild_id=GUILD_B)

        await _invoke_unlink(interaction, a.id, b.id)

        assert await _link_row_count(conn) == 1
        assert str(a.id) in _sent_content(interaction)

    async def test_the_reply_is_ephemeral(self, conn: aiosqlite.Connection) -> None:
        a = await _make_fact(conn, content="a")
        b = await _make_fact(conn, content="b")
        interaction = _make_interaction(db=conn)

        await _invoke_unlink(interaction, a.id, b.id)

        _, kwargs = interaction.response.send_message.call_args
        assert kwargs["ephemeral"] is True


class TestLocalization:
    @pytest.mark.parametrize("locale", ["de", "ja", "tr", "ko", "pl", "fr", "es-ES", "pt-BR"])
    async def test_every_locale_gets_a_real_translated_message(
        self, conn: aiosqlite.Connection, locale: str
    ) -> None:
        # Not a bracketed [key] fallback and not the English string: a missing
        # translation must be visible here rather than in a live server.
        a = await _make_fact(conn, content="a")
        b = await _make_fact(conn, content="b")
        interaction = _make_interaction(db=conn, locale=locale)

        await _invoke_link(interaction, a.id, b.id)

        message = _sent_content(interaction)
        assert not message.startswith("[")
        assert "now linked" not in message

    async def test_an_unknown_locale_falls_back_to_english_rather_than_failing(
        self, conn: aiosqlite.Connection
    ) -> None:
        a = await _make_fact(conn, content="a")
        b = await _make_fact(conn, content="b")
        interaction = _make_interaction(db=conn, locale="xx-XX")

        await _invoke_link(interaction, a.id, b.id)

        assert "now linked" in _sent_content(interaction)
