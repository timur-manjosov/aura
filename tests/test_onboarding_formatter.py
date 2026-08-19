"""Tests for aura.onboarding.formatter: one assembled onboarding summary as
one Discord embed.

Pure rendering, mirroring tests/test_digest_formatter.py's structure. The
security-critical rendering itself (link-hijack escaping, truncation) is
shared with the digest through aura.rendering and is exercised again here
rather than assumed inherited, because a regression in the shared module
should be caught by every caller's own suite, not just the first one written.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import discord
import pytest

from aura.db.models import Fact, FactStatus
from aura.i18n import SUPPORTED_LOCALES
from aura.onboarding.builder import OnboardingContent
from aura.onboarding.formatter import build_onboarding_embed, onboarding_locale

GUILD_A = 100000000000000001
CHANNEL = 300000000000000003
NOW = datetime(2026, 8, 16, 12, 0, 0, tzinfo=timezone.utc)

EMBED_TOTAL_LIMIT = 6000
FIELD_VALUE_LIMIT = 1024
FIELD_NAME_LIMIT = 256
MAX_FIELDS = 25


def fact(fact_id: int, content: str, *, created_at: datetime | None = None) -> Fact:
    return Fact(
        id=fact_id,
        guild_id=GUILD_A,
        channel_id=CHANNEL,
        message_id=900000000000000000 + fact_id,
        content=content,
        embedding=b"\x00\x00\x00\x00",
        status=FactStatus.ACTIVE,
        created_at=created_at or NOW - timedelta(days=1),
    )


def content(
    *,
    rules: list[Fact] | None = None,
    status_changes: list[Fact] | None = None,
    other: list[Fact] | None = None,
    total_eligible: int | None = None,
) -> OnboardingContent:
    rules = rules or []
    status_changes = status_changes or []
    other = other or []
    return OnboardingContent(
        guild_id=GUILD_A,
        rules=rules,
        status_changes=status_changes,
        other=other,
        total_eligible=(
            total_eligible
            if total_eligible is not None
            else len(rules) + len(status_changes) + len(other)
        ),
    )


def field_named(embed, fragment: str) -> str | None:
    for field in embed.fields:
        if fragment in (field.name or ""):
            return field.value
    return None


class TestLocaleSelection:
    def test_a_guilds_preferred_locale_is_used(self) -> None:
        guild = MagicMock(spec=discord.Guild)
        guild.preferred_locale = "de"
        assert onboarding_locale(guild) == "de"

    def test_no_guild_falls_back_to_default(self) -> None:
        assert onboarding_locale(None) == "en-US"

    def test_a_guild_with_no_preferred_locale_attribute_falls_back(self) -> None:
        guild = MagicMock(spec=discord.Guild)
        guild.preferred_locale = None
        assert onboarding_locale(guild) == "en-US"


class TestSections:
    def test_a_rule_is_listed_with_a_link_to_its_source(self) -> None:
        subject = fact(1, "No spoilers outside #spoilers.")

        embed = build_onboarding_embed(content(rules=[subject]), locale="en-US")

        value = field_named(embed, "Rules")
        assert value is not None
        assert "No spoilers outside #spoilers." in value
        assert f"https://discord.com/channels/{GUILD_A}/{CHANNEL}/{subject.message_id}" in value

    def test_an_empty_section_is_omitted_entirely(self) -> None:
        embed = build_onboarding_embed(content(rules=[fact(1, "Only this.")]), locale="en-US")

        names = [field.name or "" for field in embed.fields]
        assert len(names) == 1
        assert "Rules" in names[0]

    def test_sections_appear_in_priority_order(self) -> None:
        embed = build_onboarding_embed(
            content(
                rules=[fact(1, "a rule")],
                status_changes=[fact(2, "a status")],
                other=[fact(3, "something else")],
            ),
            locale="en-US",
        )

        names = [field.name or "" for field in embed.fields]
        assert "Rules" in names[0]
        assert "Current status" in names[1]
        assert "Also worth knowing" in names[2]

    def test_milestones_have_no_representation_here_at_all(self) -> None:
        # OnboardingContent structurally has no milestones field -- there is no
        # section for build_onboarding_embed to accidentally render one from.
        assert not hasattr(OnboardingContent, "milestones")

    def test_counts_appear_in_the_section_headings(self) -> None:
        embed = build_onboarding_embed(
            content(rules=[fact(index, f"rule {index}") for index in range(3)]),
            locale="en-US",
        )

        assert "(3)" in (embed.fields[0].name or "")

    def test_dates_are_rendered_as_client_side_timestamps(self) -> None:
        subject = fact(1, "Something.", created_at=NOW - timedelta(days=3))

        embed = build_onboarding_embed(content(rules=[subject]), locale="en-US")

        value = field_named(embed, "Rules")
        assert value is not None
        assert f"<t:{int(subject.created_at.timestamp())}:d>" in value

    def test_no_period_line_unlike_the_digest(self) -> None:
        # Onboarding has no window; there is nothing to describe as "since X".
        embed = build_onboarding_embed(content(rules=[fact(1, "x")]), locale="en-US")

        assert embed.description is not None
        assert "<t:" not in embed.description

    def test_the_capped_note_appears_when_facts_were_omitted(self) -> None:
        embed = build_onboarding_embed(
            content(rules=[fact(1, "x")], total_eligible=5), locale="en-US"
        )

        assert embed.footer.text is not None
        assert "4" in embed.footer.text

    def test_no_capped_note_when_nothing_was_omitted(self) -> None:
        embed = build_onboarding_embed(content(rules=[fact(1, "x")]), locale="en-US")

        assert embed.footer.text is not None
        assert "more active fact" not in embed.footer.text


class TestDiscordLimits:
    def test_a_maximum_length_fact_stays_inside_the_field_budget(self) -> None:
        embed = build_onboarding_embed(content(rules=[fact(1, "x" * 4000)]), locale="en-US")

        value = field_named(embed, "Rules")
        assert value is not None and len(value) <= FIELD_VALUE_LIMIT
        assert value.endswith(">")

    def test_a_large_rule_set_is_truncated_with_a_count(self) -> None:
        embed = build_onboarding_embed(
            content(rules=[fact(index, f"rule number {index}") for index in range(200)]),
            locale="en-US",
        )

        value = field_named(embed, "Rules")
        assert value is not None
        assert len(value) <= FIELD_VALUE_LIMIT
        lines = value.split("\n")
        listed = len(lines) - 1
        assert listed <= 10
        assert f"and {200 - listed} more" in lines[-1]
        assert "(200)" in (embed.fields[0].name or "")

    def test_every_section_full_at_once_stays_inside_the_total_embed_limit(self) -> None:
        long_text = "y" * 4000
        embed = build_onboarding_embed(
            content(
                rules=[fact(index, long_text) for index in range(50)],
                status_changes=[fact(100 + index, long_text) for index in range(50)],
                other=[fact(200 + index, long_text) for index in range(50)],
            ),
            locale="en-US",
        )

        assert len(embed) <= EMBED_TOTAL_LIMIT
        assert len(embed.fields) <= MAX_FIELDS
        for field in embed.fields:
            assert field.value is not None and len(field.value) <= FIELD_VALUE_LIMIT
            assert field.name is not None and len(field.name) <= FIELD_NAME_LIMIT

    def test_one_unrenderably_long_item_still_produces_a_non_empty_field(self) -> None:
        embed = build_onboarding_embed(
            content(rules=[fact(index, "z" * 4000) for index in range(30)]), locale="en-US"
        )

        value = field_named(embed, "Rules")
        assert value is not None and value.strip() != ""

    @pytest.mark.parametrize("locale", sorted(SUPPORTED_LOCALES))
    def test_the_limits_hold_in_every_locale(self, locale: str) -> None:
        embed = build_onboarding_embed(
            content(
                rules=[fact(index, "m" * 300) for index in range(40)],
                status_changes=[fact(100 + index, "n" * 300) for index in range(40)],
                other=[fact(200 + index, "o" * 300) for index in range(40)],
                total_eligible=150,
            ),
            locale=locale,
        )

        assert len(embed) <= EMBED_TOTAL_LIMIT
        for field in embed.fields:
            assert field.value is not None and len(field.value) <= FIELD_VALUE_LIMIT


class TestHostileFactText:
    """A fact's text comes from a server member. It is not trusted to be tidy."""

    def test_newlines_are_collapsed_so_one_fact_is_one_line(self) -> None:
        embed = build_onboarding_embed(
            content(rules=[fact(1, "first line\nsecond line\n\n\nthird")]), locale="en-US"
        )

        value = field_named(embed, "Rules")
        assert value is not None
        assert value.count("\n") == 0
        assert "first line second line third" in value

    def test_a_link_in_a_fact_cannot_hijack_the_source_link(self) -> None:
        subject = fact(1, "see [here](https://evil.example) for details")

        embed = build_onboarding_embed(content(rules=[subject]), locale="en-US")

        value = field_named(embed, "Rules")
        assert value is not None
        assert "\\[here\\]" in value
        label, _, target = value.rpartition("](")
        assert target.startswith(f"https://discord.com/channels/{GUILD_A}/")
        assert label.replace("\\[", "").replace("\\]", "").count("]") == 0

    def test_a_trailing_backslash_cannot_escape_the_closing_bracket(self) -> None:
        embed = build_onboarding_embed(
            content(rules=[fact(1, "ends with a backslash \\")]), locale="en-US"
        )

        value = field_named(embed, "Rules")
        assert value is not None
        assert value.endswith(">")
        assert "](https://discord.com/channels/" in value

    def test_a_fact_of_only_whitespace_renders_as_a_placeholder(self) -> None:
        embed = build_onboarding_embed(content(rules=[fact(1, "   \n\t  ")]), locale="en-US")

        value = field_named(embed, "Rules")
        assert value is not None and "[…]" in value

    def test_unicode_survives_intact(self) -> None:
        embed = build_onboarding_embed(
            content(rules=[fact(1, "서버가 500명을 넘었습니다 🎉 — Grüße")]), locale="ko"
        )

        value = embed.fields[0].value
        assert value is not None
        assert "서버가 500명을 넘었습니다 🎉 — Grüße" in value

    def test_at_everyone_in_fact_text_is_present_but_not_specially_rendered(self) -> None:
        # Suppression is the SEND-time responsibility (allowed_mentions=none()
        # in aura.onboarding.listener) -- the embed itself renders the text
        # verbatim, escaped only for the link-hijack concern above.
        embed = build_onboarding_embed(
            content(rules=[fact(1, "@everyone read the rules")]), locale="en-US"
        )

        value = field_named(embed, "Rules")
        assert value is not None and "@everyone" in value


class TestLocalization:
    @pytest.mark.parametrize("locale", sorted(SUPPORTED_LOCALES))
    def test_every_locale_renders_with_no_missing_keys(self, locale: str) -> None:
        embed = build_onboarding_embed(
            content(
                rules=[fact(1, "rule")],
                status_changes=[fact(2, "status")],
                other=[fact(3, "other")],
                total_eligible=10,
            ),
            locale=locale,
        )

        rendered = " ".join(
            [embed.title or "", embed.description or "", embed.footer.text or ""]
            + [f"{field.name} {field.value}" for field in embed.fields]
        )
        assert "[onboarding_" not in rendered

    def test_an_unsupported_locale_falls_back_to_english(self) -> None:
        embed = build_onboarding_embed(content(rules=[fact(1, "x")]), locale="xx-XX")

        assert "Welcome" in (embed.title or "")

    def test_the_translated_heading_is_actually_used(self) -> None:
        embed = build_onboarding_embed(content(rules=[fact(1, "x")]), locale="de")

        assert embed.title is not None and "Willkommen" in embed.title
        rules_field = field_named(embed, "Regeln")
        assert rules_field is not None

    def test_fact_text_is_never_translated(self) -> None:
        german_fact = fact(1, "Die Regeln stehen in #regeln.")

        embed = build_onboarding_embed(content(rules=[german_fact]), locale="ja")

        value = embed.fields[0].value
        assert value is not None and "Die Regeln stehen in #regeln." in value
