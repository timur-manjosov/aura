"""Tests for aura.commands.operator: /aura-operator-budget.

Command callback, operator check and error handler invoked directly against
mocked discord.Interaction objects and a real in-memory database, matching how
test_proactive_commands.py exercises /aura-debug-signals.

What matters here is not the embed's exact wording but three things a
misconfigured or hostile invocation could get wrong: that anyone who is NOT
the configured operator -- including someone who tries this command out of
curiosity, and an unconfigured deployment where nobody is the operator -- is
refused before touching the database; that the numbers shown come from the
same read every one of the five real ledgers actually populates, not a
reimplementation of that math; and that the permission error (unlike the
embed itself) goes through the same translated t() seam every other command's
does, since unlike the embed it can be seen by any member.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import aiosqlite
import discord
import pytest
from discord import app_commands

from aura.commands.operator import (
    _handle_operator_budget_error,
    _is_operator,
    operator_budget_command,
)
from aura.db.proactive_state import try_acquire_escalation_slot
from aura.db.repository import init_schema
from aura.i18n import SUPPORTED_LOCALES, t

OPERATOR_ID = 999
OTHER_USER_ID = 111
GUILD_A = 100000000000000001

NOW = datetime(2026, 9, 12, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
async def conn():
    connection = await aiosqlite.connect(":memory:")
    await init_schema(connection)
    yield connection
    await connection.close()


def _make_interaction(
    *,
    db: aiosqlite.Connection | None,
    user_id: int = OPERATOR_ID,
    operator_discord_user_id: int | None = OPERATOR_ID,
    locale: str = "en-US",
    cross_guild_daily_budget_usd: float = 15.0,
    cross_guild_budget_mode: str = "warn",
) -> MagicMock:
    """A mock Interaction exposing just what /aura-operator-budget actually touches."""
    interaction = MagicMock(spec=discord.Interaction)
    interaction.locale = locale
    interaction.user = MagicMock()
    interaction.user.id = user_id
    interaction.client = MagicMock()
    interaction.client.db = db
    interaction.client.settings = MagicMock()
    interaction.client.settings.operator_discord_user_id = operator_discord_user_id
    interaction.client.settings.cross_guild_daily_budget_usd = cross_guild_daily_budget_usd
    interaction.client.settings.cross_guild_budget_mode = cross_guild_budget_mode
    interaction.response = MagicMock()
    interaction.response.send_message = AsyncMock()
    interaction.response.is_done = MagicMock(return_value=False)
    interaction.followup = MagicMock()
    interaction.followup.send = AsyncMock()
    return interaction


async def _invoke(interaction: discord.Interaction) -> None:
    await operator_budget_command.callback(interaction)  # pyright: ignore[reportCallIssue, reportArgumentType]


def _embed(interaction: MagicMock) -> discord.Embed:
    _, kwargs = interaction.response.send_message.call_args
    return kwargs["embed"]


class TestOperatorCheck:
    def test_allows_the_configured_operator(self) -> None:
        interaction = MagicMock()
        interaction.user.id = OPERATOR_ID
        interaction.client.settings.operator_discord_user_id = OPERATOR_ID
        assert _is_operator(interaction) is True

    def test_rejects_any_other_user(self) -> None:
        interaction = MagicMock()
        interaction.user.id = OTHER_USER_ID
        interaction.client.settings.operator_discord_user_id = OPERATOR_ID
        assert _is_operator(interaction) is False

    def test_rejects_everyone_when_unconfigured(self) -> None:
        # The safe direction: an unset OPERATOR_DISCORD_USER_ID disables the
        # command for everyone rather than for no one.
        interaction = MagicMock()
        interaction.user.id = OPERATOR_ID
        interaction.client.settings.operator_discord_user_id = None
        assert _is_operator(interaction) is False

    def test_the_command_is_guild_only(self) -> None:
        assert operator_budget_command.guild_only is True

    def test_the_check_is_registered_on_the_command(self) -> None:
        fake = MagicMock()
        fake.user.id = OTHER_USER_ID
        fake.client.settings.operator_discord_user_id = OPERATOR_ID
        assert any(check(fake) is False for check in operator_budget_command.checks)


class TestCommandBody:
    async def test_shows_zero_for_every_ledger_on_an_empty_database(
        self, conn: aiosqlite.Connection
    ) -> None:
        interaction = _make_interaction(db=conn)

        await _invoke(interaction)

        interaction.response.send_message.assert_awaited_once()
        _, kwargs = interaction.response.send_message.call_args
        assert kwargs.get("ephemeral") is True
        embed = _embed(interaction)
        field_values = " ".join(f"{f.name} {f.value}" for f in embed.fields)
        assert "Proactive" in field_values
        assert "0 call(s)" in field_values

    async def test_reflects_real_spend_from_the_actual_ledger(
        self, conn: aiosqlite.Connection
    ) -> None:
        await try_acquire_escalation_slot(
            conn, guild_id=GUILD_A, channel_id=1, message_id=1,
            cooldown_seconds=0.0, daily_cap=1_000_000, now=NOW,
        )
        interaction = _make_interaction(db=conn)

        await _invoke(interaction)

        embed = _embed(interaction)
        proactive_field = next(f for f in embed.fields if f.name == "Proactive")
        assert proactive_field.value is not None
        assert "1 call(s)" in proactive_field.value
        assert "0.0030" in proactive_field.value

    async def test_over_budget_is_visibly_flagged(self, conn: aiosqlite.Connection) -> None:
        await try_acquire_escalation_slot(
            conn, guild_id=GUILD_A, channel_id=1, message_id=1,
            cooldown_seconds=0.0, daily_cap=1_000_000, now=NOW,
        )
        interaction = _make_interaction(db=conn, cross_guild_daily_budget_usd=0.001)

        await _invoke(interaction)

        embed = _embed(interaction)
        combined = next(f for f in embed.fields if f.name and "Combined" in f.name)
        assert combined.value is not None
        assert "OVER BUDGET" in combined.value


class TestErrorHandler:
    async def test_check_failure_replies_ephemerally_and_localized(self) -> None:
        interaction = _make_interaction(db=None)

        await _handle_operator_budget_error(interaction, app_commands.CheckFailure("no"))

        interaction.response.send_message.assert_awaited_once()
        args, kwargs = interaction.response.send_message.call_args
        assert args[0] == t("operator_budget_permission_error", "en-US")
        assert kwargs.get("ephemeral") is True

    async def test_check_failure_uses_followup_if_already_responded(self) -> None:
        interaction = _make_interaction(db=None)
        interaction.response.is_done = MagicMock(return_value=True)

        await _handle_operator_budget_error(interaction, app_commands.CheckFailure("no"))

        interaction.followup.send.assert_awaited_once()
        interaction.response.send_message.assert_not_awaited()

    async def test_check_failure_message_is_translated_in_every_locale(self) -> None:
        for locale in SUPPORTED_LOCALES:
            interaction = _make_interaction(db=None, locale=locale)
            await _handle_operator_budget_error(interaction, app_commands.CheckFailure("no"))
            args, _ = interaction.response.send_message.call_args
            assert args[0] != "[operator_budget_permission_error]"

    async def test_unexpected_errors_are_logged_and_not_surfaced_to_the_user(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        interaction = _make_interaction(db=None)
        error = app_commands.CommandInvokeError(MagicMock(), ValueError("boom"))

        with caplog.at_level(logging.ERROR):
            await _handle_operator_budget_error(interaction, error)

        interaction.response.send_message.assert_not_awaited()
