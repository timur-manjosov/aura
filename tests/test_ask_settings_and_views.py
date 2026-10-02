"""Tests for the /aura-ask cost-bound settings, their locale strings, and the operator view.

The settings are the operator's only handle on these bounds, so their defaults
and refusals are pinned here; the locale strings are what a capped member
reads; the operator line is where the new ledger's spend becomes visible.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import aiosqlite
import discord
import pytest
from pydantic import ValidationError

from aura.commands.operator import operator_budget_command
from aura.config import Settings
from aura.db.ask_state import try_acquire_ask_call_slot
from aura.db.cross_guild_budget import Ledger
from aura.db.repository import init_schema
from aura.i18n import SUPPORTED_LOCALES, t

NOW = datetime(2026, 10, 2, 12, 0, 0, tzinfo=UTC)
OPERATOR_ID = 999

_LIMIT_KEYS = ("ask_limit_guild_reached", "ask_limit_user_reached")


def _settings(**overrides: object) -> Settings:
    return Settings(_env_file=None, discord_token="fake-token", **overrides)  # type: ignore[arg-type]


class TestDefaults:
    def test_the_shipped_values(self) -> None:
        settings = _settings()
        assert settings.ask_daily_cap_free == 10
        assert settings.ask_daily_cap_pro == 25
        assert settings.ask_user_daily_cap_free == 5
        assert settings.ask_synthesis_max_output_tokens == 700
        assert settings.grounding_max_output_tokens == 300

    def test_values_are_read_from_the_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ASK_DAILY_CAP_FREE", "3")
        monkeypatch.setenv("ASK_DAILY_CAP_PRO", "7")
        monkeypatch.setenv("ASK_USER_DAILY_CAP_FREE", "1")
        monkeypatch.setenv("ASK_SYNTHESIS_MAX_OUTPUT_TOKENS", "512")
        monkeypatch.setenv("GROUNDING_MAX_OUTPUT_TOKENS", "200")
        settings = _settings()
        assert (
            settings.ask_daily_cap_free,
            settings.ask_daily_cap_pro,
            settings.ask_user_daily_cap_free,
            settings.ask_synthesis_max_output_tokens,
            settings.grounding_max_output_tokens,
        ) == (3, 7, 1, 512, 200)


class TestValidation:
    @pytest.mark.parametrize(
        "field", ["ask_daily_cap_free", "ask_daily_cap_pro", "ask_user_daily_cap_free"]
    )
    def test_a_cap_of_zero_is_allowed(self, field: str) -> None:
        assert getattr(_settings(**{field: 0}), field) == 0

    @pytest.mark.parametrize(
        "field", ["ask_daily_cap_free", "ask_daily_cap_pro", "ask_user_daily_cap_free"]
    )
    @pytest.mark.parametrize("value", [-1, 1_000_001, "ten", 2.5])
    def test_a_cap_outside_its_range_is_refused(self, field: str, value: object) -> None:
        with pytest.raises(ValidationError):
            _settings(**{field: value})

    @pytest.mark.parametrize(
        ("field", "low", "high"),
        [
            ("ask_synthesis_max_output_tokens", 256, 8192),
            ("grounding_max_output_tokens", 128, 4096),
        ],
    )
    def test_an_output_ceiling_has_both_bounds(self, field: str, low: int, high: int) -> None:
        assert getattr(_settings(**{field: low}), field) == low
        assert getattr(_settings(**{field: high}), field) == high
        for refused in (0, -1, low - 1, high + 1):
            with pytest.raises(ValidationError):
                _settings(**{field: refused})


class TestLimitMessages:
    @pytest.mark.parametrize("locale", sorted(SUPPORTED_LOCALES))
    @pytest.mark.parametrize("key", _LIMIT_KEYS)
    def test_every_locale_renders_the_reset_time(self, locale: str, key: str) -> None:
        text = t(key, locale, reset="<t:1791072000:R>")
        assert "<t:1791072000:R>" in text
        assert not text.startswith("[")
        assert "{" not in text

    @pytest.mark.parametrize("locale", sorted(SUPPORTED_LOCALES - {"en-US"}))
    @pytest.mark.parametrize("key", _LIMIT_KEYS)
    def test_every_locale_is_translated(self, locale: str, key: str) -> None:
        assert t(key, locale, reset="x") != t(key, "en-US", reset="x")

    def test_the_member_and_guild_notes_differ_in_every_locale(self) -> None:
        for locale in SUPPORTED_LOCALES:
            assert t(_LIMIT_KEYS[0], locale, reset="x") != t(_LIMIT_KEYS[1], locale, reset="x")

    def test_german_speaks_informally(self) -> None:
        for key in _LIMIT_KEYS:
            text = t(key, "de", reset="x")
            assert re.search(r"\b(du|deine?r?)\b", text, re.IGNORECASE)
            assert not re.search(r"\b(Sie|Ihre?)\b", text)


class TestOperatorView:
    async def test_the_ask_ledger_has_its_own_line(self) -> None:
        conn = await aiosqlite.connect(":memory:")
        try:
            await init_schema(conn)
            for user_id in (1, 2, 3):
                attempt = await try_acquire_ask_call_slot(
                    conn,
                    guild_id=100000000000000001,
                    user_id=user_id,
                    guild_cap=25,
                    user_cap=None,
                    now=NOW,
                )
                assert attempt.granted

            interaction = MagicMock(spec=discord.Interaction)
            interaction.locale = "en-US"
            interaction.user = MagicMock(id=OPERATOR_ID)
            interaction.client = MagicMock()
            interaction.client.db = conn
            interaction.client.settings = _settings(
                operator_discord_user_id=OPERATOR_ID, cross_guild_daily_budget_usd=15.0
            )
            interaction.response = MagicMock()
            interaction.response.send_message = AsyncMock()

            with patch("aura.commands.operator.utc_now", return_value=NOW):
                await operator_budget_command.callback(interaction)  # type: ignore[call-arg, arg-type]  # pyright: ignore

            _, kwargs = interaction.response.send_message.call_args
            embed = kwargs["embed"]
            ask = next(f for f in embed.fields if f.name == "Ask")
            assert ask.value == "3 call(s) today · ~$0.0120"
            combined = next(f for f in embed.fields if f.name and "Combined" in f.name)
            assert combined.value is not None
            assert combined.value.startswith("~$0.0120 of $15.00")
        finally:
            await conn.close()

    def test_a_single_ask_does_not_round_to_nothing(self) -> None:
        # The operator view prints four decimals; one ask is $0.0040, not $0.00.
        assert f"{1 * 0.004:.4f}" == "0.0040"
        assert Ledger.ASK.value == "ask"
