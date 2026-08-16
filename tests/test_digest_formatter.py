"""Tests for aura.digest.formatter: one assembled digest as one Discord embed.

Pure rendering, so every case here is built from plain Fact objects with no
database and no gateway involved.

Two groups matter more than the rest. TestDiscordLimits is the one that decides
whether a digest arrives at all: every limit Discord enforces is a limit that,
crossed, turns a working weekly summary into a silent 400 -- and the input that
crosses it (a very long fact, or a very busy week) is ordinary rather than
exotic. TestHostileFactText is the adversarial pass over content Aura does not
control: a fact's text is written by a server member and only ever passed
through a distillation model, so it can contain newlines, markdown, brackets
and mentions.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from aura.db.models import Fact, FactStatus
from aura.digest.builder import DigestChange, DigestContent
from aura.digest.formatter import build_digest_embed
from aura.digest.intervals import DigestInterval
from aura.i18n import SUPPORTED_LOCALES

GUILD_A = 100000000000000001
CHANNEL = 300000000000000003
NOW = datetime(2026, 8, 16, 12, 0, 0, tzinfo=timezone.utc)
WEEK = int(DigestInterval.WEEKLY)

# Discord's own hard caps, restated here rather than imported from the module
# under test: a test that reads the same constant it verifies would pass no
# matter what that constant was changed to.
EMBED_TOTAL_LIMIT = 6000
FIELD_VALUE_LIMIT = 1024
FIELD_NAME_LIMIT = 256
MAX_FIELDS = 25


def fact(
    fact_id: int, content: str, *, created_at: datetime | None = None, guild_id: int = GUILD_A
) -> Fact:
    return Fact(
        id=fact_id,
        guild_id=guild_id,
        channel_id=CHANNEL,
        message_id=900000000000000000 + fact_id,
        content=content,
        embedding=b"\x00\x00\x00\x00",
        status=FactStatus.ACTIVE,
        created_at=created_at or NOW - timedelta(days=1),
    )


def content(
    *,
    new_facts: list[Fact] | None = None,
    milestones: list[Fact] | None = None,
    changes: list[DigestChange] | None = None,
) -> DigestContent:
    return DigestContent(
        guild_id=GUILD_A,
        covered_from=NOW - timedelta(days=7),
        covered_until=NOW,
        new_facts=new_facts or [],
        milestones=milestones or [],
        changes=changes or [],
    )


def change(previous: Fact, current: Fact, *, collapsed: int = 0) -> DigestChange:
    return DigestChange(
        previous=previous,
        current=current,
        changed_at=NOW - timedelta(days=2),
        collapsed_steps=collapsed,
    )


def field_named(embed, fragment: str) -> str | None:
    for field in embed.fields:
        if fragment in (field.name or ""):
            return field.value
    return None


class TestSections:
    def test_a_new_fact_is_listed_with_a_link_to_its_source(self) -> None:
        subject = fact(1, "Movie night is on Fridays.")

        embed = build_digest_embed(content(new_facts=[subject]), locale="en-US", interval_seconds=WEEK)

        value = field_named(embed, "New")
        assert value is not None
        assert "Movie night is on Fridays." in value
        assert (
            f"https://discord.com/channels/{GUILD_A}/{CHANNEL}/{subject.message_id}" in value
        )

    def test_an_empty_section_is_omitted_entirely(self) -> None:
        embed = build_digest_embed(
            content(new_facts=[fact(1, "Only this.")]), locale="en-US", interval_seconds=WEEK
        )

        names = [field.name or "" for field in embed.fields]
        assert len(names) == 1
        assert "New" in names[0]

    def test_milestones_are_their_own_highlighted_section(self) -> None:
        embed = build_digest_embed(
            content(
                milestones=[fact(1, "The server reached 500 members.")],
                new_facts=[fact(2, "Movie night is on Fridays.")],
            ),
            locale="en-US",
            interval_seconds=WEEK,
        )

        names = [field.name or "" for field in embed.fields]
        assert "🏆" in names[0]  # highlighted, and first
        milestone_value = field_named(embed, "Milestones")
        new_value = field_named(embed, "New")
        assert milestone_value is not None and "500 members" in milestone_value
        assert new_value is not None and "500 members" not in new_value

    def test_a_change_shows_the_old_text_and_links_the_new_one(self) -> None:
        old = fact(1, "Meetings are on Monday.")
        new = fact(2, "Meetings are on Thursday.")

        embed = build_digest_embed(
            content(changes=[change(old, new)]), locale="en-US", interval_seconds=WEEK
        )

        value = field_named(embed, "Updated")
        assert value is not None
        assert "Meetings are on Monday." in value
        assert "Meetings are on Thursday." in value
        assert f"/{new.message_id}" in value
        assert f"/{old.message_id}" not in value

    def test_collapsed_steps_are_named_rather_than_hidden(self) -> None:
        embed = build_digest_embed(
            content(changes=[change(fact(1, "A"), fact(2, "D"), collapsed=2)]),
            locale="en-US",
            interval_seconds=WEEK,
        )

        value = field_named(embed, "Updated")
        assert value is not None and "+2" in value

    def test_a_single_step_change_says_nothing_about_intermediates(self) -> None:
        embed = build_digest_embed(
            content(changes=[change(fact(1, "A"), fact(2, "B"))]),
            locale="en-US",
            interval_seconds=WEEK,
        )

        value = field_named(embed, "Updated")
        assert value is not None and "+0" not in value

    def test_counts_appear_in_the_section_headings(self) -> None:
        embed = build_digest_embed(
            content(new_facts=[fact(index, f"fact {index}") for index in range(3)]),
            locale="en-US",
            interval_seconds=WEEK,
        )

        assert "(3)" in (embed.fields[0].name or "")

    def test_dates_are_rendered_as_client_side_timestamps(self) -> None:
        # <t:...> lets each reader see the date in their own locale and
        # timezone; a formatted string would pick one for everybody.
        subject = fact(1, "Something.", created_at=NOW - timedelta(days=3))

        embed = build_digest_embed(
            content(new_facts=[subject]), locale="en-US", interval_seconds=WEEK
        )

        value = field_named(embed, "New")
        assert value is not None
        assert f"<t:{int(subject.created_at.timestamp())}:d>" in value
        assert embed.description is not None
        assert f"<t:{int(NOW.timestamp())}:d>" in embed.description

    def test_the_footer_names_the_configured_cadence(self) -> None:
        embed = build_digest_embed(
            content(new_facts=[fact(1, "x")]),
            locale="en-US",
            interval_seconds=int(DigestInterval.BIWEEKLY),
        )

        assert embed.footer.text is not None
        assert "every two weeks" in embed.footer.text

    def test_a_hand_edited_cadence_is_reported_literally(self) -> None:
        embed = build_digest_embed(
            content(new_facts=[fact(1, "x")]), locale="en-US", interval_seconds=777
        )

        assert embed.footer.text is not None and "777" in embed.footer.text


class TestDiscordLimits:
    """Every bound Discord enforces, checked against inputs that actually reach them."""

    def test_a_maximum_length_fact_stays_inside_the_field_budget(self) -> None:
        # 4000 characters is what the "Add as Aura Fact" modal allows, so this
        # is an ordinary fact rather than a contrived one.
        embed = build_digest_embed(
            content(new_facts=[fact(1, "x" * 4000)]), locale="en-US", interval_seconds=WEEK
        )

        value = field_named(embed, "New")
        assert value is not None and len(value) <= FIELD_VALUE_LIMIT
        assert value.endswith(">")  # truncated inside the link label, not after it

    def test_a_busy_week_is_truncated_with_a_count(self) -> None:
        embed = build_digest_embed(
            content(new_facts=[fact(index, f"fact number {index}") for index in range(200)]),
            locale="en-US",
            interval_seconds=WEEK,
        )

        value = field_named(embed, "New")
        assert value is not None
        assert len(value) <= FIELD_VALUE_LIMIT
        # Whichever of the two bounds bit first -- the item count or the
        # character budget -- the note must account for every item not listed.
        lines = value.split("\n")
        listed = len(lines) - 1
        assert listed <= 10
        assert f"and {200 - listed} more" in lines[-1]
        # The heading still reports the real total, so the truncation is
        # visible rather than silently changing the numbers.
        assert "(200)" in (embed.fields[0].name or "")

    def test_every_section_full_at_once_stays_inside_the_total_embed_limit(self) -> None:
        long_text = "y" * 4000
        embed = build_digest_embed(
            content(
                milestones=[fact(index, long_text) for index in range(50)],
                new_facts=[fact(100 + index, long_text) for index in range(50)],
                changes=[
                    change(fact(200 + index, long_text), fact(300 + index, long_text), collapsed=3)
                    for index in range(50)
                ],
            ),
            locale="en-US",
            interval_seconds=WEEK,
        )

        assert len(embed) <= EMBED_TOTAL_LIMIT
        assert len(embed.fields) <= MAX_FIELDS
        for field in embed.fields:
            assert field.value is not None and len(field.value) <= FIELD_VALUE_LIMIT
            assert field.name is not None and len(field.name) <= FIELD_NAME_LIMIT

    def test_one_unrenderably_long_item_still_produces_a_non_empty_field(self) -> None:
        # Discord rejects an empty field value outright, so "nothing fit" must
        # still say something.
        embed = build_digest_embed(
            content(new_facts=[fact(index, "z" * 4000) for index in range(30)]),
            locale="en-US",
            interval_seconds=WEEK,
        )

        value = field_named(embed, "New")
        assert value is not None and value.strip() != ""

    @pytest.mark.parametrize("locale", sorted(SUPPORTED_LOCALES))
    def test_the_limits_hold_in_every_locale(self, locale: str) -> None:
        # Translations differ in length, and the "and N more" note is reserved
        # for by its own translated length -- a locale whose note is longer
        # must not be the one that overflows.
        embed = build_digest_embed(
            content(
                milestones=[fact(index, "m" * 300) for index in range(40)],
                new_facts=[fact(100 + index, "n" * 300) for index in range(40)],
                changes=[
                    change(fact(200 + index, "o" * 300), fact(300 + index, "p" * 300))
                    for index in range(40)
                ],
            ),
            locale=locale,
            interval_seconds=WEEK,
        )

        assert len(embed) <= EMBED_TOTAL_LIMIT
        for field in embed.fields:
            assert field.value is not None and len(field.value) <= FIELD_VALUE_LIMIT


class TestHostileFactText:
    """A fact's text comes from a server member. It is not trusted to be tidy."""

    def test_newlines_are_collapsed_so_one_fact_is_one_line(self) -> None:
        embed = build_digest_embed(
            content(new_facts=[fact(1, "first line\nsecond line\n\n\nthird")]),
            locale="en-US",
            interval_seconds=WEEK,
        )

        value = field_named(embed, "New")
        assert value is not None
        assert value.count("\n") == 0
        assert "first line second line third" in value

    def test_a_link_in_a_fact_cannot_hijack_the_source_link(self) -> None:
        # The attack: a fact whose text is itself markdown link syntax. An
        # unescaped `]` would end Aura's own label early, leaving the member's
        # URL as the thing the digest line links to. Escaping the brackets makes
        # the whole of the fact text inert label content, so the only link
        # target on the line is still the Discord permalink Aura appended.
        subject = fact(1, "see [here](https://evil.example) for details")

        embed = build_digest_embed(
            content(new_facts=[subject]), locale="en-US", interval_seconds=WEEK
        )

        value = field_named(embed, "New")
        assert value is not None
        assert "\\[here\\]" in value
        label, _, target = value.rpartition("](")
        assert target.startswith(f"https://discord.com/channels/{GUILD_A}/")
        # Nothing in the label can close it: every bracket the fact contributed
        # is escaped, so the only unescaped `]` is the one that opens the target.
        assert label.replace("\\[", "").replace("\\]", "").count("]") == 0

    def test_a_trailing_backslash_cannot_escape_the_closing_bracket(self) -> None:
        embed = build_digest_embed(
            content(new_facts=[fact(1, "ends with a backslash \\")]),
            locale="en-US",
            interval_seconds=WEEK,
        )

        value = field_named(embed, "New")
        assert value is not None
        assert value.endswith(">")
        assert "](https://discord.com/channels/" in value

    def test_a_fact_of_only_whitespace_renders_as_a_placeholder(self) -> None:
        # An empty markdown label renders as a broken link. Unreachable through
        # the modal (it rejects blank input) but cheap to survive.
        embed = build_digest_embed(
            content(new_facts=[fact(1, "   \n\t  ")]), locale="en-US", interval_seconds=WEEK
        )

        value = field_named(embed, "New")
        assert value is not None and "[…]" in value

    def test_unicode_survives_intact(self) -> None:
        embed = build_digest_embed(
            content(new_facts=[fact(1, "서버가 500명을 넘었습니다 🎉 — Grüße")]),
            locale="ko",
            interval_seconds=WEEK,
        )

        value = embed.fields[0].value
        assert value is not None
        assert "서버가 500명을 넘었습니다 🎉 — Grüße" in value


class TestLocalization:
    @pytest.mark.parametrize("locale", sorted(SUPPORTED_LOCALES))
    def test_every_locale_renders_with_no_missing_keys(self, locale: str) -> None:
        # t() returns "[key]" for a key missing everywhere, so a locale file
        # that skipped one of this phase's strings shows up here rather than in
        # a posted digest.
        embed = build_digest_embed(
            content(
                milestones=[fact(1, "milestone")],
                new_facts=[fact(2, "new")],
                changes=[change(fact(3, "old"), fact(4, "new"), collapsed=2)],
            ),
            locale=locale,
            interval_seconds=WEEK,
        )

        rendered = " ".join(
            [embed.title or "", embed.description or "", embed.footer.text or ""]
            + [f"{field.name} {field.value}" for field in embed.fields]
        )
        assert "[digest_" not in rendered

    def test_an_unsupported_locale_falls_back_to_english(self) -> None:
        embed = build_digest_embed(
            content(new_facts=[fact(1, "x")]), locale="xx-XX", interval_seconds=WEEK
        )

        assert embed.title == "What changed in this server"

    def test_the_translated_heading_is_actually_used(self) -> None:
        embed = build_digest_embed(
            content(new_facts=[fact(1, "x")]), locale="de", interval_seconds=WEEK
        )

        assert embed.title == "Was sich auf diesem Server geändert hat"
        assert embed.footer.text is not None and "wöchentlich" in embed.footer.text

    def test_fact_text_is_never_translated(self) -> None:
        # A fact stays in the language it was written in, whatever the reader's
        # locale -- the same rule /aura-pending follows.
        german_fact = fact(1, "Die Regeln stehen in #regeln.")

        embed = build_digest_embed(
            content(new_facts=[german_fact]), locale="ja", interval_seconds=WEEK
        )

        value = embed.fields[0].value
        assert value is not None and "Die Regeln stehen in #regeln." in value
