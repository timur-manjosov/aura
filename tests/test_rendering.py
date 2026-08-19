"""Tests for aura.rendering: the shared, security-critical fact-line primitives
behind both the digest and onboarding.

This module exists precisely so a fix (the link-hijack escaping in
reports/phase-3e.txt Section 7b) lives in one place. These tests exercise that
place directly, at the unit level, rather than relying on inheriting coverage
through whichever caller's suite happens to exercise it -- a regression here
should fail here, not two callers over.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from aura.db.models import Fact, FactStatus
from aura.i18n import SUPPORTED_LOCALES
from aura.rendering import (
    FIELD_VALUE_LIMIT,
    ITEM_TEXT_LIMIT,
    discord_timestamp,
    fit_lines,
    inline_fact_text,
    source_link,
)

GUILD_A = 100000000000000001
CHANNEL = 300000000000000003
NOW = datetime(2026, 8, 16, 12, 0, 0, tzinfo=timezone.utc)


def fact(fact_id: int = 1, content: str = "x") -> Fact:
    return Fact(
        id=fact_id,
        guild_id=GUILD_A,
        channel_id=CHANNEL,
        message_id=900000000000000000 + fact_id,
        content=content,
        embedding=b"\x00\x00\x00\x00",
        status=FactStatus.ACTIVE,
        created_at=NOW,
    )


class TestInlineFactText:
    def test_ordinary_text_passes_through(self) -> None:
        assert inline_fact_text("Movie night is on Fridays.") == "Movie night is on Fridays."

    def test_whitespace_runs_collapse_to_single_spaces(self) -> None:
        assert inline_fact_text("first\nsecond\n\n\nthird") == "first second third"

    def test_leading_and_trailing_whitespace_is_stripped(self) -> None:
        assert inline_fact_text("  padded  ") == "padded"

    def test_square_brackets_are_escaped(self) -> None:
        assert inline_fact_text("see [here](evil)") == "see \\[here\\]\\(evil\\)".replace(
            "\\(evil\\)", "(evil)"
        )

    def test_a_bracket_link_cannot_be_reconstructed_after_escaping(self) -> None:
        escaped = inline_fact_text("[label](https://evil.example)")
        # No unescaped ] or [ remains that could close a markdown span.
        stripped = escaped.replace("\\[", "").replace("\\]", "")
        assert "[" not in stripped and "]" not in stripped

    def test_a_lone_backslash_is_escaped_so_it_cannot_eat_the_next_char(self) -> None:
        assert inline_fact_text("a\\b") == "a\\\\b"

    def test_a_trailing_backslash_before_truncation_point_is_still_escaped(self) -> None:
        # Truncation happens BEFORE escaping, so a cut landing right before a
        # backslash must not leave that backslash able to eat the closing `]`
        # a caller appends afterward.
        text = "x" * (ITEM_TEXT_LIMIT - 1) + "\\" + "y" * 50
        result = inline_fact_text(text)
        assert len(result) <= ITEM_TEXT_LIMIT + 5  # a little slack for escaping growth
        # Whatever trailing backslash exists must be doubled (escaped), never lone.
        assert not result.rstrip("…").endswith("\\") or result.rstrip("…").endswith("\\\\")

    def test_a_string_of_only_whitespace_becomes_a_placeholder(self) -> None:
        assert inline_fact_text("   \t\n  ") == "…"

    def test_an_empty_string_becomes_a_placeholder(self) -> None:
        assert inline_fact_text("") == "…"

    def test_long_text_is_truncated_with_an_ellipsis(self) -> None:
        result = inline_fact_text("x" * 1000)
        assert len(result) == ITEM_TEXT_LIMIT
        assert result.endswith("…")

    def test_text_at_exactly_the_limit_is_not_truncated(self) -> None:
        text = "x" * ITEM_TEXT_LIMIT
        assert inline_fact_text(text) == text

    def test_unicode_and_emoji_survive(self) -> None:
        text = "서버가 500명을 넘었습니다 🎉 — Grüße"
        assert inline_fact_text(text) == text

    def test_asterisks_are_left_unescaped(self) -> None:
        # Deliberate: matches how /aura-ask and /aura-pending already render
        # fact text. Only [ ] \ are load-bearing for the link-hijack defence.
        assert inline_fact_text("*emphasis* stays") == "*emphasis* stays"

    def test_zero_width_characters_alone_become_a_placeholder(self) -> None:
        assert inline_fact_text("\u200b\u200b\u200b") == "…"


class TestSourceLink:
    def test_the_link_points_at_the_facts_own_message(self) -> None:
        subject = fact(1, "x")
        link = source_link(subject)
        assert link == (
            f"https://discord.com/channels/{GUILD_A}/{CHANNEL}/{subject.message_id}"
        )


class TestDiscordTimestamp:
    def test_renders_a_client_side_day_timestamp(self) -> None:
        assert discord_timestamp(NOW) == f"<t:{int(NOW.timestamp())}:d>"


class TestFitLines:
    def test_all_lines_fit_when_short(self) -> None:
        result = fit_lines(["a", "b", "c"], "en-US", max_items=10, more_items_key="digest_more_items")
        assert result == "a\nb\nc"

    def test_more_than_max_items_is_truncated_with_a_count(self) -> None:
        lines = [f"line {i}" for i in range(15)]
        result = fit_lines(lines, "en-US", max_items=10, more_items_key="digest_more_items")
        rendered_lines = result.split("\n")
        assert len(rendered_lines) == 11  # 10 items + the note
        assert "5 more" in rendered_lines[-1]

    def test_the_character_budget_can_bite_before_the_item_count(self) -> None:
        lines = ["y" * 200 for _ in range(10)]
        result = fit_lines(lines, "en-US", max_items=10, more_items_key="digest_more_items")
        assert len(result) <= FIELD_VALUE_LIMIT
        assert "more" in result

    def test_the_note_accounts_for_every_omitted_item(self) -> None:
        lines = [f"item {i}" for i in range(200)]
        result = fit_lines(lines, "en-US", max_items=10, more_items_key="digest_more_items")
        rendered_lines = result.split("\n")
        listed = len(rendered_lines) - 1
        assert f"and {200 - listed} more" in rendered_lines[-1]

    def test_an_empty_list_produces_an_empty_string(self) -> None:
        assert fit_lines([], "en-US", max_items=10, more_items_key="digest_more_items") == ""

    def test_one_line_too_long_to_fit_alone_still_produces_non_empty_output(self) -> None:
        # Discord rejects an empty field value outright.
        lines = ["z" * 4000 for _ in range(5)]
        result = fit_lines(lines, "en-US", max_items=10, more_items_key="digest_more_items")
        assert result.strip() != ""

    def test_a_custom_more_items_key_is_honoured(self) -> None:
        lines = [f"line {i}" for i in range(15)]
        digest_note = fit_lines(lines, "en-US", max_items=10, more_items_key="digest_more_items")
        onboarding_note = fit_lines(
            lines, "en-US", max_items=10, more_items_key="onboarding_more_items"
        )
        # Both keys currently render identical English text, but the call
        # sites are independently controllable -- verified by each resolving
        # without producing a missing-key marker.
        assert "[digest_more_items]" not in digest_note
        assert "[onboarding_more_items]" not in onboarding_note

    @pytest.mark.parametrize("locale", sorted(SUPPORTED_LOCALES))
    def test_the_field_budget_holds_in_every_locale(self, locale: str) -> None:
        lines = [f"fact number {i} with some more text to pad it out" for i in range(60)]
        result = fit_lines(lines, locale, max_items=10, more_items_key="digest_more_items")
        assert len(result) <= FIELD_VALUE_LIMIT
