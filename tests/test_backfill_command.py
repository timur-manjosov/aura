"""Tests for aura.commands.backfill: /aura-backfill start|status|pause|cancel.

Command callbacks, permission checks and the group's error handler invoked
directly against mocked discord objects and a real in-memory database, matching
how every other moderator-gated command in this project is tested. No live
gateway connection.

The interesting behaviour here is the refusals, not the happy path. Starting a
backfill is one command with four different reasons to say no, and each of them
answers a question the moderator would otherwise have to work out from a status
reply an hour later: the channel is not opted in, a run is already going, the
paused run's range cannot be changed, or the date is not a date.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import aiosqlite
import discord
import pytest
from discord import app_commands

from aura.commands.backfill import (
    _handle_backfill_error,
    _parse_since,
    _since_snowflake,
    backfill_cancel,
    backfill_group,
    backfill_pause,
    backfill_start,
    backfill_status,
)
from aura.db.backfill_runs import (
    BackfillState,
    advance_cursor,
    get_active_run,
    get_recent_runs,
    set_run_state,
    start_backfill_run,
)
from aura.db.backfill_state import try_acquire_backfill_call_slot
from aura.db.extraction_channel_config import set_extraction_enabled
from aura.db.repository import init_schema
from aura.i18n import SUPPORTED_LOCALES, t

GUILD_A = 100000000000000001
GUILD_B = 200000000000000002
CHANNEL_A = 300000000000000003
CHANNEL_B = 400000000000000004
MODERATOR = 4242

NOW = datetime(2026, 8, 26, 12, 0, 0, tzinfo=timezone.utc)


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
    backfill_daily_cap: int = 30,
) -> MagicMock:
    interaction = MagicMock(spec=discord.Interaction)
    interaction.locale = locale
    interaction.guild_id = guild_id
    interaction.user = MagicMock()
    interaction.user.id = MODERATOR
    interaction.client = MagicMock()
    interaction.client.db = db
    interaction.client.settings = MagicMock()
    interaction.client.settings.backfill_daily_cap = backfill_daily_cap
    interaction.response = MagicMock()
    interaction.response.send_message = AsyncMock()
    interaction.response.is_done = MagicMock(return_value=False)
    interaction.followup = MagicMock()
    interaction.followup.send = AsyncMock()
    interaction.command = MagicMock()
    interaction.command.name = "start"
    return interaction


def _make_channel(channel_id: int = CHANNEL_A, *, guild_id: int = GUILD_A) -> MagicMock:
    channel = MagicMock(spec=discord.TextChannel)
    channel.id = channel_id
    channel.mention = f"<#{channel_id}>"
    channel.guild = MagicMock()
    channel.guild.id = guild_id
    return channel


def _reply(interaction: MagicMock) -> str:
    interaction.response.send_message.assert_awaited_once()
    call = interaction.response.send_message.await_args
    return call.args[0] if call.args else call.kwargs.get("content", "")


def _embed(interaction: MagicMock) -> discord.Embed:
    interaction.response.send_message.assert_awaited_once()
    return interaction.response.send_message.await_args.kwargs["embed"]


async def _enable(conn: aiosqlite.Connection, channel_id: int = CHANNEL_A) -> None:
    await set_extraction_enabled(
        conn, guild_id=GUILD_A, channel_id=channel_id, enabled=True, updated_by_id=MODERATOR
    )


async def _start(interaction, channel, since: str | None = None) -> None:
    await backfill_start.callback(interaction, channel, since)  # pyright: ignore[reportCallIssue, reportArgumentType]


class TestPermissionGate:
    def test_the_group_declares_manage_guild_as_its_default_permission(self) -> None:
        assert backfill_group.default_permissions is not None
        assert backfill_group.default_permissions.manage_guild

    def test_the_group_is_guild_only(self) -> None:
        assert backfill_group.guild_only

    @pytest.mark.parametrize(
        "command", [backfill_start, backfill_status, backfill_pause, backfill_cancel]
    )
    def test_every_subcommand_rejects_a_non_moderator(self, command) -> None:
        """default_permissions is a client-side hint; the check is the real gate."""
        fake = MagicMock(permissions=discord.Permissions(manage_guild=False))
        for check in command.checks:
            with pytest.raises(app_commands.MissingPermissions):
                check(fake)

    @pytest.mark.parametrize(
        "command", [backfill_start, backfill_status, backfill_pause, backfill_cancel]
    )
    def test_every_subcommand_accepts_a_moderator(self, command) -> None:
        fake = MagicMock(permissions=discord.Permissions(manage_guild=True))
        assert all(check(fake) for check in command.checks)

    async def test_a_permission_failure_replies_in_the_users_own_locale(self) -> None:
        interaction = _make_interaction(db=None, locale="de")

        await _handle_backfill_error(
            interaction, app_commands.MissingPermissions(["manage_guild"])
        )

        interaction.response.send_message.assert_awaited_once_with(
            t("backfill_permission_error", "de"), ephemeral=True
        )

    async def test_a_permission_failure_after_a_response_uses_the_followup(self) -> None:
        interaction = _make_interaction(db=None)
        interaction.response.is_done = MagicMock(return_value=True)

        await _handle_backfill_error(
            interaction, app_commands.MissingPermissions(["manage_guild"])
        )

        interaction.followup.send.assert_awaited_once()
        interaction.response.send_message.assert_not_awaited()

    async def test_any_other_error_is_logged_rather_than_disappearing(self, caplog) -> None:
        interaction = _make_interaction(db=None)

        with caplog.at_level(logging.ERROR, logger="aura.commands.backfill"):
            await _handle_backfill_error(
                interaction, app_commands.AppCommandError("something broke")
            )

        assert "Unhandled error in /aura-backfill" in caplog.text
        interaction.response.send_message.assert_not_awaited()

    def test_the_group_has_the_error_handler_attached(self) -> None:
        assert backfill_group.on_error is _handle_backfill_error


class TestSinceParsing:
    def test_a_plain_iso_date_becomes_utc_midnight(self) -> None:
        assert _parse_since("2025-03-14", now=NOW) == datetime(
            2025, 3, 14, tzinfo=timezone.utc
        )

    def test_surrounding_whitespace_is_tolerated(self) -> None:
        assert _parse_since("  2025-03-14 ", now=NOW) is not None

    @pytest.mark.parametrize("raw", ["2025-3-14", "2025-03-4", "2025-3-4"])
    def test_a_missing_leading_zero_is_still_the_same_unambiguous_date(self, raw) -> None:
        """Accepted deliberately: nothing about it can be read as a different day."""
        parsed = _parse_since(raw, now=NOW)
        assert parsed is not None and parsed.year == 2025 and parsed.month == 3

    @pytest.mark.parametrize(
        "raw",
        [
            "14/03/2025",
            "March 14 2025",
            "20250314",
            "2025-13-01",
            "2025-02-30",
            "",
            "   ",
            "yesterday",
            "2025-03-14T12:00:00",
            "​2025-03-14",
            "2025-03-14; DROP TABLE facts",
        ],
    )
    def test_anything_that_is_not_exactly_that_shape_is_refused(self, raw) -> None:
        assert _parse_since(raw, now=NOW) is None

    def test_a_future_date_is_refused(self) -> None:
        assert _parse_since("2027-01-01", now=NOW) is None

    def test_today_is_accepted_because_its_midnight_is_in_the_past(self) -> None:
        assert _parse_since("2026-08-26", now=NOW) is not None


class TestSinceSnowflake:
    """The adversarial pass's one real find: a pre-Discord date is not a snowflake."""

    def test_an_ordinary_date_becomes_the_lowest_snowflake_for_that_instant(self) -> None:
        since = datetime(2025, 3, 14, tzinfo=timezone.utc)

        assert _since_snowflake(since) == discord.utils.time_snowflake(since, high=False)

    @pytest.mark.parametrize("year", [1, 1000, 1969, 1970, 2014])
    def test_a_date_before_discord_existed_clamps_to_the_whole_history(self, year) -> None:
        """time_snowflake is arithmetic: it returns a huge NEGATIVE number here."""
        assert discord.utils.time_snowflake(
            datetime(year, 1, 1, tzinfo=timezone.utc), high=False
        ) < 0
        assert _since_snowflake(datetime(year, 1, 1, tzinfo=timezone.utc)) is None

    def test_discords_own_epoch_clamps_too_because_a_snowflake_is_never_zero(self) -> None:
        assert _since_snowflake(datetime(2015, 1, 1, tzinfo=timezone.utc)) is None

    async def test_a_pre_discord_since_produces_a_whole_history_run_rather_than_a_stuck_one(
        self, conn
    ) -> None:
        await _enable(conn)
        interaction = _make_interaction(db=conn)

        await _start(interaction, _make_channel(), "1970-01-01")

        run = await get_active_run(conn, channel_id=CHANNEL_A)
        assert run is not None
        assert run.after_message_id is None
        assert run.state is BackfillState.RUNNING

    async def test_the_data_layer_refuses_a_negative_snowflake_outright(self, conn) -> None:
        """Belt to the command layer's braces: it must not be storable at all."""
        with pytest.raises(ValueError, match="positive snowflake"):
            await start_backfill_run(
                conn,
                guild_id=GUILD_A,
                channel_id=CHANNEL_A,
                until_message_id=900,
                after_message_id=-5_000_000,
                requested_by_id=MODERATOR,
                now=NOW,
            )


class TestStart:
    async def test_it_refuses_a_channel_extraction_is_not_enabled_for(self, conn) -> None:
        interaction = _make_interaction(db=conn)

        await _start(interaction, _make_channel())

        assert "extraction:true" in _reply(interaction)
        assert await get_active_run(conn, channel_id=CHANNEL_A) is None

    async def test_it_refuses_a_channel_belonging_to_another_guild(self, conn) -> None:
        await _enable(conn)
        interaction = _make_interaction(db=conn)

        await _start(interaction, _make_channel(guild_id=GUILD_B))

        assert _reply(interaction) == t("backfill_wrong_guild_error", "en-US")
        assert await get_active_run(conn, channel_id=CHANNEL_A) is None

    async def test_it_opens_a_run_bounded_by_the_moment_it_was_started(self, conn) -> None:
        await _enable(conn)
        interaction = _make_interaction(db=conn)

        with patch("aura.commands.backfill.utc_now", return_value=NOW):
            await _start(interaction, _make_channel())

        run = await get_active_run(conn, channel_id=CHANNEL_A)
        assert run is not None
        assert run.state is BackfillState.RUNNING
        assert run.after_message_id is None
        assert run.requested_by_id == MODERATOR
        # Exclusive, and the HIGHEST snowflake for that instant: a message
        # written in the same millisecond falls to the live path, not to both.
        assert run.until_message_id == discord.utils.time_snowflake(NOW, high=True)

    async def test_the_since_option_becomes_an_exclusive_lower_bound(self, conn) -> None:
        await _enable(conn)
        interaction = _make_interaction(db=conn)

        with patch("aura.commands.backfill.utc_now", return_value=NOW):
            await _start(interaction, _make_channel(), "2025-03-14")

        run = await get_active_run(conn, channel_id=CHANNEL_A)
        assert run is not None
        assert run.after_message_id == discord.utils.time_snowflake(
            datetime(2025, 3, 14, tzinfo=timezone.utc), high=False
        )

    async def test_an_unusable_since_is_refused_before_anything_is_written(
        self, conn
    ) -> None:
        await _enable(conn)
        interaction = _make_interaction(db=conn)

        await _start(interaction, _make_channel(), "not a date")

        assert _reply(interaction) == t("backfill_since_invalid_error", "en-US")
        assert await get_active_run(conn, channel_id=CHANNEL_A) is None

    async def test_the_confirmation_names_the_daily_cap(self, conn) -> None:
        await _enable(conn)
        interaction = _make_interaction(db=conn, backfill_daily_cap=17)

        await _start(interaction, _make_channel())

        assert "17" in _reply(interaction)

    async def test_the_confirmation_says_a_since_run_starts_from_that_date(
        self, conn
    ) -> None:
        await _enable(conn)
        interaction = _make_interaction(db=conn)

        await _start(interaction, _make_channel(), "2025-03-14")

        assert "2025-03-14" in _reply(interaction)

    async def test_starting_twice_reports_the_run_that_is_already_going(self, conn) -> None:
        await _enable(conn)
        await _start(_make_interaction(db=conn), _make_channel())
        second = _make_interaction(db=conn)

        await _start(second, _make_channel())

        assert t("backfill_already_running", "en-US", channel="<#%d>" % CHANNEL_A) in _reply(
            second
        )
        assert len(await get_recent_runs(conn, guild_id=GUILD_A, limit=10)) == 1

    async def test_the_reply_is_ephemeral(self, conn) -> None:
        await _enable(conn)
        interaction = _make_interaction(db=conn)

        await _start(interaction, _make_channel())

        assert interaction.response.send_message.await_args.kwargs["ephemeral"] is True

    async def test_a_completed_run_does_not_block_a_new_one(self, conn) -> None:
        await _enable(conn)
        first = await start_backfill_run(
            conn,
            guild_id=GUILD_A,
            channel_id=CHANNEL_A,
            until_message_id=900,
            after_message_id=None,
            requested_by_id=MODERATOR,
            now=NOW,
        )
        await set_run_state(
            conn,
            run_id=first.id,
            state=BackfillState.COMPLETED,
            now=NOW,
            from_states=(BackfillState.RUNNING,),
        )
        interaction = _make_interaction(db=conn)

        await _start(interaction, _make_channel())

        active = await get_active_run(conn, channel_id=CHANNEL_A)
        assert active is not None and active.id != first.id


class TestResume:
    async def test_start_resumes_a_paused_run_rather_than_opening_a_new_one(
        self, conn
    ) -> None:
        await _enable(conn)
        run = await start_backfill_run(
            conn,
            guild_id=GUILD_A,
            channel_id=CHANNEL_A,
            until_message_id=900,
            after_message_id=None,
            requested_by_id=MODERATOR,
            now=NOW,
        )
        await advance_cursor(
            conn,
            run_id=run.id,
            cursor_message_id=500,
            cursor_message_at=NOW,
            messages_scanned=42,
            candidates_staged=3,
            calls_spent=1,
            now=NOW,
        )
        await set_run_state(
            conn,
            run_id=run.id,
            state=BackfillState.PAUSED,
            now=NOW,
            from_states=(BackfillState.RUNNING,),
        )
        interaction = _make_interaction(db=conn)

        await _start(interaction, _make_channel())

        resumed = await get_active_run(conn, channel_id=CHANNEL_A)
        assert resumed is not None
        assert resumed.id == run.id, "a new run was opened instead of resuming"
        assert resumed.state is BackfillState.RUNNING
        assert resumed.cursor_message_id == 500, "resuming lost the cursor"
        assert "42" in _reply(interaction), "the reply did not say where it resumed from"

    async def test_resuming_with_a_since_is_refused_rather_than_silently_ignored(
        self, conn
    ) -> None:
        await _enable(conn)
        run = await start_backfill_run(
            conn,
            guild_id=GUILD_A,
            channel_id=CHANNEL_A,
            until_message_id=900,
            after_message_id=None,
            requested_by_id=MODERATOR,
            now=NOW,
        )
        await set_run_state(
            conn,
            run_id=run.id,
            state=BackfillState.PAUSED,
            now=NOW,
            from_states=(BackfillState.RUNNING,),
        )
        interaction = _make_interaction(db=conn)

        await _start(interaction, _make_channel(), "2025-03-14")

        assert _reply(interaction) == t(
            "backfill_resume_since_conflict", "en-US", channel=f"<#{CHANNEL_A}>"
        )
        still = await get_active_run(conn, channel_id=CHANNEL_A)
        assert still is not None and still.state is BackfillState.PAUSED


class TestPauseAndCancel:
    async def _running_run(self, conn, *, channel_id: int = CHANNEL_A):
        await _enable(conn, channel_id)
        return await start_backfill_run(
            conn,
            guild_id=GUILD_A,
            channel_id=channel_id,
            until_message_id=900,
            after_message_id=None,
            requested_by_id=MODERATOR,
            now=NOW,
        )

    async def test_pause_stops_a_running_run_and_keeps_its_place(self, conn) -> None:
        run = await self._running_run(conn)
        await advance_cursor(
            conn,
            run_id=run.id,
            cursor_message_id=500,
            cursor_message_at=NOW,
            messages_scanned=10,
            candidates_staged=1,
            calls_spent=1,
            now=NOW,
        )
        interaction = _make_interaction(db=conn)

        await backfill_pause.callback(interaction, _make_channel())  # pyright: ignore[reportCallIssue, reportArgumentType]

        paused = await get_active_run(conn, channel_id=CHANNEL_A)
        assert paused is not None
        assert paused.state is BackfillState.PAUSED
        assert paused.cursor_message_id == 500

    async def test_pause_on_a_channel_with_no_run_says_so(self, conn) -> None:
        await _enable(conn)
        interaction = _make_interaction(db=conn)

        await backfill_pause.callback(interaction, _make_channel())  # pyright: ignore[reportCallIssue, reportArgumentType]

        assert _reply(interaction) == t(
            "backfill_pause_not_running", "en-US", channel=f"<#{CHANNEL_A}>"
        )

    async def test_pause_on_an_already_paused_run_says_so_rather_than_claiming_success(
        self, conn
    ) -> None:
        run = await self._running_run(conn)
        await set_run_state(
            conn,
            run_id=run.id,
            state=BackfillState.PAUSED,
            now=NOW,
            from_states=(BackfillState.RUNNING,),
        )
        interaction = _make_interaction(db=conn)

        await backfill_pause.callback(interaction, _make_channel())  # pyright: ignore[reportCallIssue, reportArgumentType]

        assert "no running backfill" in _reply(interaction)

    async def test_cancel_ends_a_running_run(self, conn) -> None:
        await self._running_run(conn)
        interaction = _make_interaction(db=conn)

        await backfill_cancel.callback(interaction, _make_channel())  # pyright: ignore[reportCallIssue, reportArgumentType]

        assert await get_active_run(conn, channel_id=CHANNEL_A) is None
        assert (await get_recent_runs(conn, guild_id=GUILD_A, limit=1))[
            0
        ].state is BackfillState.CANCELLED

    async def test_cancel_ends_a_paused_run_too(self, conn) -> None:
        run = await self._running_run(conn)
        await set_run_state(
            conn,
            run_id=run.id,
            state=BackfillState.PAUSED,
            now=NOW,
            from_states=(BackfillState.RUNNING,),
        )
        interaction = _make_interaction(db=conn)

        await backfill_cancel.callback(interaction, _make_channel())  # pyright: ignore[reportCallIssue, reportArgumentType]

        assert (await get_recent_runs(conn, guild_id=GUILD_A, limit=1))[
            0
        ].state is BackfillState.CANCELLED

    async def test_cancel_with_nothing_active_says_so(self, conn) -> None:
        await _enable(conn)
        interaction = _make_interaction(db=conn)

        await backfill_cancel.callback(interaction, _make_channel())  # pyright: ignore[reportCallIssue, reportArgumentType]

        assert _reply(interaction) == t(
            "backfill_cancel_nothing_active", "en-US", channel=f"<#{CHANNEL_A}>"
        )

    async def test_another_guilds_run_cannot_be_stopped_through_a_crafted_channel_id(
        self, conn
    ) -> None:
        """Unreachable through Discord's own option resolution; checked anyway."""
        await set_extraction_enabled(
            conn,
            guild_id=GUILD_B,
            channel_id=CHANNEL_B,
            enabled=True,
            updated_by_id=MODERATOR,
        )
        await start_backfill_run(
            conn,
            guild_id=GUILD_B,
            channel_id=CHANNEL_B,
            until_message_id=900,
            after_message_id=None,
            requested_by_id=MODERATOR,
            now=NOW,
        )
        interaction = _make_interaction(db=conn, guild_id=GUILD_A)

        await backfill_cancel.callback(  # pyright: ignore[reportCallIssue, reportArgumentType]
            interaction, _make_channel(CHANNEL_B, guild_id=GUILD_B)
        )

        theirs = await get_active_run(conn, channel_id=CHANNEL_B)
        assert theirs is not None and theirs.state is BackfillState.RUNNING


class TestStatus:
    async def test_it_reports_todays_budget_even_with_no_runs(self, conn) -> None:
        interaction = _make_interaction(db=conn, backfill_daily_cap=30)

        await backfill_status.callback(interaction, None)  # pyright: ignore[reportCallIssue, reportArgumentType]

        embed = _embed(interaction)
        assert embed.description is not None and "30" in embed.description
        assert any(
            t("backfill_status_none", "en-US") in (field.value or "")
            for field in embed.fields
        )

    async def test_a_run_waiting_on_its_cap_is_distinguishable_from_a_stalled_one(
        self, conn
    ) -> None:
        """'Nothing is happening' and 'the cap is spent' must not look the same."""
        await _enable(conn)
        run = await start_backfill_run(
            conn,
            guild_id=GUILD_A,
            channel_id=CHANNEL_A,
            until_message_id=900,
            after_message_id=None,
            requested_by_id=MODERATOR,
            now=NOW,
        )
        for _ in range(3):
            await try_acquire_backfill_call_slot(
                conn,
                guild_id=GUILD_A,
                run_id=run.id,
                message_count=5,
                daily_cap=3,
                now=NOW,
            )
        interaction = _make_interaction(db=conn, backfill_daily_cap=3)

        with patch("aura.commands.backfill.utc_now", return_value=NOW):
            await backfill_status.callback(interaction, None)  # pyright: ignore[reportCallIssue, reportArgumentType]

        embed = _embed(interaction)
        assert embed.description == t(
            "backfill_status_budget", "en-US", spent=3, cap=3
        )

    async def test_a_run_with_a_cursor_shows_a_position_and_a_permalink(
        self, conn
    ) -> None:
        await _enable(conn)
        run = await start_backfill_run(
            conn,
            guild_id=GUILD_A,
            channel_id=CHANNEL_A,
            until_message_id=900000,
            after_message_id=None,
            requested_by_id=MODERATOR,
            now=NOW,
        )
        await advance_cursor(
            conn,
            run_id=run.id,
            cursor_message_id=800123,
            cursor_message_at=NOW,
            messages_scanned=250,
            candidates_staged=7,
            calls_spent=2,
            now=NOW,
        )
        interaction = _make_interaction(db=conn)

        await backfill_status.callback(interaction, None)  # pyright: ignore[reportCallIssue, reportArgumentType]

        value = _embed(interaction).fields[0].value or ""
        assert f"https://discord.com/channels/{GUILD_A}/{CHANNEL_A}/800123" in value
        assert "250" in value and "7" in value and "2" in value
        assert f"<#{CHANNEL_A}>" in value

    async def test_a_run_that_has_not_read_anything_yet_says_so_rather_than_showing_a_date(
        self, conn
    ) -> None:
        await _enable(conn)
        await start_backfill_run(
            conn,
            guild_id=GUILD_A,
            channel_id=CHANNEL_A,
            until_message_id=900,
            after_message_id=None,
            requested_by_id=MODERATOR,
            now=NOW,
        )
        interaction = _make_interaction(db=conn)

        await backfill_status.callback(interaction, None)  # pyright: ignore[reportCallIssue, reportArgumentType]

        value = _embed(interaction).fields[0].value or ""
        assert t("backfill_progress_not_started", "en-US") in value

    async def test_it_never_shows_a_percentage_it_cannot_know(self, conn) -> None:
        """Aura cannot know a channel's length without reading it, so it says a date."""
        await _enable(conn)
        run = await start_backfill_run(
            conn,
            guild_id=GUILD_A,
            channel_id=CHANNEL_A,
            until_message_id=900000,
            after_message_id=None,
            requested_by_id=MODERATOR,
            now=NOW,
        )
        await advance_cursor(
            conn,
            run_id=run.id,
            cursor_message_id=800123,
            cursor_message_at=NOW,
            messages_scanned=250,
            candidates_staged=7,
            calls_spent=2,
            now=NOW,
        )
        interaction = _make_interaction(db=conn)

        await backfill_status.callback(interaction, None)  # pyright: ignore[reportCallIssue, reportArgumentType]

        assert "%" not in "".join(field.value or "" for field in _embed(interaction).fields)

    async def test_filtering_by_channel_shows_only_that_channel(self, conn) -> None:
        await _enable(conn, CHANNEL_A)
        await _enable(conn, CHANNEL_B)
        await start_backfill_run(
            conn,
            guild_id=GUILD_A,
            channel_id=CHANNEL_A,
            until_message_id=900,
            after_message_id=None,
            requested_by_id=MODERATOR,
            now=NOW,
        )
        await start_backfill_run(
            conn,
            guild_id=GUILD_A,
            channel_id=CHANNEL_B,
            until_message_id=900,
            after_message_id=None,
            requested_by_id=MODERATOR,
            now=NOW,
        )
        interaction = _make_interaction(db=conn)

        await backfill_status.callback(interaction, _make_channel(CHANNEL_B))  # pyright: ignore[reportCallIssue, reportArgumentType]

        values = "".join(field.value or "" for field in _embed(interaction).fields)
        assert f"<#{CHANNEL_B}>" in values
        assert f"<#{CHANNEL_A}>" not in values

    async def test_it_does_not_show_another_guilds_runs(self, conn) -> None:
        await start_backfill_run(
            conn,
            guild_id=GUILD_B,
            channel_id=CHANNEL_B,
            until_message_id=900,
            after_message_id=None,
            requested_by_id=MODERATOR,
            now=NOW,
        )
        interaction = _make_interaction(db=conn, guild_id=GUILD_A)

        await backfill_status.callback(interaction, None)  # pyright: ignore[reportCallIssue, reportArgumentType]

        values = "".join(field.value or "" for field in _embed(interaction).fields)
        assert f"<#{CHANNEL_B}>" not in values

    async def test_every_field_value_stays_inside_discords_hard_cap(self, conn) -> None:
        await _enable(conn)
        run = await start_backfill_run(
            conn,
            guild_id=GUILD_A,
            channel_id=CHANNEL_A,
            until_message_id=900000,
            after_message_id=None,
            requested_by_id=MODERATOR,
            now=NOW,
        )
        await advance_cursor(
            conn,
            run_id=run.id,
            cursor_message_id=800123,
            cursor_message_at=NOW,
            messages_scanned=10**15,
            candidates_staged=10**15,
            calls_spent=10**15,
            now=NOW,
        )
        interaction = _make_interaction(db=conn)

        await backfill_status.callback(interaction, None)  # pyright: ignore[reportCallIssue, reportArgumentType]

        embed = _embed(interaction)
        assert all(len(field.value or "") <= 1024 for field in embed.fields)
        assert len(embed.description or "") <= 4096

    async def test_the_reply_is_ephemeral(self, conn) -> None:
        interaction = _make_interaction(db=conn)

        await backfill_status.callback(interaction, None)  # pyright: ignore[reportCallIssue, reportArgumentType]

        assert interaction.response.send_message.await_args.kwargs["ephemeral"] is True


class TestLocalisation:
    @pytest.mark.parametrize("locale", sorted(SUPPORTED_LOCALES))
    async def test_every_state_label_resolves_in_every_locale(self, locale) -> None:
        for state in BackfillState:
            rendered = t(f"backfill_state_{state.value}", locale)
            assert not rendered.startswith("["), f"{locale} is missing {state.value}"

    @pytest.mark.parametrize("locale", sorted(SUPPORTED_LOCALES))
    async def test_a_start_confirmation_resolves_in_every_locale(self, conn, locale) -> None:
        await _enable(conn)
        interaction = _make_interaction(db=conn, locale=locale)

        await _start(interaction, _make_channel())

        reply = _reply(interaction)
        assert "[" not in reply.split("<#")[0], f"{locale} produced an unresolved key"

    async def test_an_unsupported_locale_falls_back_rather_than_failing(self, conn) -> None:
        await _enable(conn)
        interaction = _make_interaction(db=conn, locale="xx-YY")

        await _start(interaction, _make_channel())

        assert _reply(interaction) == "\n".join(
            [
                t("backfill_started_all", "en-US", channel=f"<#{CHANNEL_A}>", since=""),
                t("backfill_started_explainer", "en-US", cap=30),
            ]
        )
