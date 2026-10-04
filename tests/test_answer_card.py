"""Tests for aura.answer_card, aura.theme and the display helpers of aura.rendering.

Pure rendering: a validated contract answer in, an AnswerCard and its Discord
forms out. Every Discord limit is hit exactly and once over, every untrusted
string is attacked (link hijack, mention spam, invisible characters, huge
input), and the card's checked statements are shown to be exactly what it
displays.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import discord
import pytest

from aura.answer_card import (
    PLAIN_MESSAGE_LIMIT,
    AnswerCard,
    build_answer_card,
    build_fact_list_card,
    build_notice_card,
    card_to_embed,
    card_to_layout_view,
    card_to_plain_text,
    compose_description,
    layout_text_length,
    plain_answer_text,
    question_line,
)
from aura.answer_contract import validate_contract
from aura.db.models import Fact, FactStatus
from aura.i18n import SUPPORTED_LOCALES, t
from aura.rendering import collapse_display_text, escape_display_markdown, link_label, shorten
from aura.theme import (
    ACCENT_COLORS,
    COMPONENTS_V2_COMPONENT_LIMIT,
    COMPONENTS_V2_TEXT_LIMIT,
    DESCRIPTION_MAX_CHARS,
    EMBED_AUTHOR_NAME_LIMIT,
    EMBED_DESCRIPTION_LIMIT,
    EMBED_FIELD_COUNT_LIMIT,
    EMBED_FIELD_VALUE_LIMIT,
    EMBED_TOTAL_LIMIT,
    KIND_SYMBOLS,
    LEAD_MAX_CHARS,
    MAX_POINTS,
    MIN_ACCENT_CONTRAST,
    POINT_MAX_CHARS,
    PROACTIVE_MAX_POINTS,
    QUESTION_DISPLAY_CHARS,
    THEME_BACKGROUNDS,
    MessageKind,
)

GUILD = 123456789012345678
_LOCALES_DIR = Path(__file__).resolve().parent.parent / "src" / "aura" / "i18n" / "locales"


def _fact(fact_id: int, content: str = "The event starts at 18:00.", channel_id: int = 0) -> Fact:
    return Fact(
        id=fact_id,
        guild_id=GUILD,
        channel_id=channel_id or 500 + fact_id,
        message_id=900000000000000000 + fact_id,
        content=content,
        embedding=b"",
        status=FactStatus.ACTIVE,
        created_at=datetime(2026, 8, fact_id % 28 + 1, 12, tzinfo=UTC),
    )


def _reply(n_facts: int, **overrides: Any) -> dict[str, Any]:
    reply: dict[str, Any] = {
        "request_reading": "r",
        "fact_notes": [],
        "relations": [],
        "not_covered_topics": [],
        "tone": "casual",
        "lead": "The event starts at 18:00.",
        "points": [],
        "used_fact_numbers": list(range(1, n_facts + 1)),
        "answers_question": True,
    }
    reply.update(overrides)
    return reply


def _card(
    facts: list[Fact],
    *,
    question: str | None = "When does it start?",
    locale: str = "en-US",
    channel_names: dict[int, str] | None = None,
    proactive: bool = False,
    **overrides: Any,
) -> AnswerCard:
    answer = validate_contract(_reply(len(facts), **overrides), facts)
    return build_answer_card(
        answer,
        facts,
        question=question,
        locale=locale,
        channel_names=channel_names if channel_names is not None else {},
        proactive=proactive,
    )


def _embed_total(embed: discord.Embed) -> int:
    return (
        len(embed.description or "")
        + len(embed.author.name or "")
        + len(embed.footer.text or "")
        + sum(len(field.name or "") + len(field.value or "") for field in embed.fields)
    )


def _layout_texts(view: discord.ui.LayoutView) -> list[str]:
    texts: list[str] = []
    for item in view.walk_children():
        if isinstance(item, discord.ui.TextDisplay):
            texts.append(item.content)
    return texts


def _count_components(view: discord.ui.LayoutView) -> int:
    return sum(1 for _ in view.walk_children())


class TestTheme:
    @staticmethod
    def _luminance(colour: int) -> float:
        def channel(value: int) -> float:
            v = value / 255
            return v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4

        r, g, b = (colour >> 16) & 255, (colour >> 8) & 255, colour & 255
        return 0.2126 * channel(r) + 0.7152 * channel(g) + 0.0722 * channel(b)

    @pytest.mark.parametrize("kind", list(MessageKind))
    @pytest.mark.parametrize("background", list(THEME_BACKGROUNDS))
    def test_every_accent_keeps_three_to_one_contrast_on_every_discord_theme(
        self, kind: MessageKind, background: str
    ) -> None:
        a = self._luminance(ACCENT_COLORS[kind])
        b = self._luminance(THEME_BACKGROUNDS[background])
        ratio = (max(a, b) + 0.05) / (min(a, b) + 0.05)

        assert ratio >= MIN_ACCENT_CONTRAST

    def test_every_kind_has_its_own_colour_and_its_own_symbol(self) -> None:
        assert set(ACCENT_COLORS) == set(MessageKind) == set(KIND_SYMBOLS)
        assert len(set(ACCENT_COLORS.values())) == len(MessageKind)
        assert len(set(KIND_SYMBOLS.values())) == len(MessageKind)

    def test_the_card_bounds_sit_under_discords_limits(self) -> None:
        assert DESCRIPTION_MAX_CHARS < EMBED_DESCRIPTION_LIMIT
        assert QUESTION_DISPLAY_CHARS + 4 <= EMBED_AUTHOR_NAME_LIMIT
        assert LEAD_MAX_CHARS * 2 + 400 < DESCRIPTION_MAX_CHARS


class TestDisplayHelpers:
    def test_collapse_joins_lines_and_drops_invisible_format_characters(self) -> None:
        assert collapse_display_text(" a\n\nb\tc\u200b\u202e\ufeffd ") == "a b cd"

    def test_collapse_keeps_the_two_joiners_emoji_sequences_need(self) -> None:
        family = "\U0001f468\u200d\U0001f469\u200d\U0001f467"
        assert collapse_display_text(family) == family
        assert collapse_display_text("a\u200cb") == "a\u200cb"

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("a*b_c~d`e|f", "a\\*b\\_c\\~d\\`e\\|f"),
            ("[x](https://evil.example)", "\\[x\\](https://evil.example)"),
            ("<@123> <#456> <t:1:R> <:e:1>", "\\<@123\\> \\<#456\\> \\<t:1:R\\> \\<:e:1\\>"),
            ("# heading", "\\# heading"),
            ("-# subtext", "\\-# subtext"),
            ("> quote", "\\> quote"),
            ("1. item", "1\\. item"),
            ("12) item", "12\\) item"),
            ("back\\slash", "back\\\\slash"),
        ],
    )
    def test_escape_makes_markdown_literal(self, text: str, expected: str) -> None:
        assert escape_display_markdown(text) == expected

    def test_escape_defuses_mass_mentions(self) -> None:
        escaped = escape_display_markdown("hey @everyone and @here")

        assert "@everyone" not in escaped
        assert "@here" not in escaped
        assert escaped.replace("\u200b", "") == "hey @everyone and @here"

    def test_escape_inside_a_line_leaves_leading_punctuation_alone(self) -> None:
        assert escape_display_markdown("- x", at_line_start=False) == "- x"

    def test_shorten_cuts_with_an_ellipsis_only_when_needed(self) -> None:
        assert shorten("abc", 3) == "abc"
        assert shorten("abcd", 3) == "ab…"
        assert len(shorten("x" * 5000, 200)) == 200

    def test_link_label_is_never_empty_and_never_breaks_out(self) -> None:
        assert link_label("\u200b \n", 40) == "…"
        assert "](" not in link_label("a](https://evil.example", 40).replace("\\](", "")


class TestTheAnswerCard:
    def test_the_layout_is_question_lead_points_notes_sources_footer(self) -> None:
        facts = [_fact(1), _fact(2, "Sign-up is in #events.")]
        card = _card(
            facts,
            points=[{"text": "You sign up in #events.", "facts": [2]}],
            relations=[{"facts": [1, 2], "kind": "complementary"}],
            not_covered_topics=["prize"],
            channel_names={501: "announcements", 502: "events"},
        )

        assert card.kind is MessageKind.ANSWER
        assert card.top_line == f"{KIND_SYMBOLS[MessageKind.ANSWER]} When does it start?"
        assert card.paragraphs == ("The event starts at 18:00.",)
        assert card.bullets[0].startswith(
            "You sign up in #events. [²](https://discord.com/channels/"
        )
        assert card.notes == ("Not recorded: prize.",)
        assert card.sources[0].startswith("`1` [#announcements](https://discord.com/channels/")
        assert card.sources[0].endswith(f"<t:{int(facts[0].created_at.timestamp())}:d>")
        assert card.footer == t("answer_footer", "en-US")

    def test_citations_are_renumbered_in_display_order(self) -> None:
        facts = [_fact(1), _fact(2), _fact(3)]
        card = _card(
            facts,
            used_fact_numbers=[3, 1],
            points=[{"text": "Detail.", "facts": [1]}],
        )

        assert card.cited_fact_ids == (3, 1)
        assert "[²]" in card.bullets[0]
        assert card.sources[0].startswith("`1`")
        assert str(facts[2].message_id) in card.sources[0]

    def test_a_point_with_two_citations_separates_them(self) -> None:
        facts = [_fact(1), _fact(2)]
        card = _card(facts, points=[{"text": "Both.", "facts": [1, 2]}])

        assert re.search(r"\[¹\]\([^)]+\)\u2009\[²\]\(", card.bullets[0])

    def test_an_unnamed_channel_shows_message_instead_of_a_raw_id(self) -> None:
        fact = _fact(1, channel_id=777)
        card = _card([fact], channel_names={777: "777"}, locale="de")

        assert "[Nachricht](" in card.sources[0]
        assert "777]" not in card.sources[0]

    @pytest.mark.parametrize(
        ("kind", "key"),
        [
            ("same_detail_conflict", "answer_note_conflict"),
            ("unclear_if_same", "answer_note_unclear"),
        ],
    )
    def test_a_caveat_note_comes_from_the_template_before_the_gap_line(
        self, kind: str, key: str
    ) -> None:
        facts = [_fact(1), _fact(2)]
        card = _card(
            facts,
            relations=[{"facts": [1, 2], "kind": kind}],
            not_covered_topics=["location"],
            locale="de",
        )

        assert card.notes == (t(key, "de"), "Nicht vermerkt: location.")

    def test_a_complementary_pair_adds_no_note(self) -> None:
        card = _card([_fact(1), _fact(2)], relations=[{"facts": [1, 2], "kind": "complementary"}])

        assert card.notes == ()

    @pytest.mark.parametrize("locale", sorted(SUPPORTED_LOCALES))
    def test_the_gap_line_renders_in_every_locale(self, locale: str) -> None:
        card = _card([_fact(1)], not_covered_topics=["alpha", "beta"], locale=locale)

        separator = t("answer_list_separator", locale)
        assert card.notes == (t("answer_not_recorded", locale, topics=f"alpha{separator}beta"),)
        assert "{" not in card.notes[0]

    def test_untrusted_text_cannot_become_markup(self) -> None:
        hostile = "Click [here](https://evil.example) <@&1> @everyone **x**"
        card = _card(
            [_fact(1)],
            lead=hostile,
            points=[{"text": hostile, "facts": [1]}],
            not_covered_topics=["[a](evil example)"],
        )
        description = compose_description(card, notes_as_subtext=False)

        assert "](https://evil.example)" not in description.replace("\\](https://evil.example)", "")
        assert "[a](evil example)" not in description
        assert "<@&1>" not in description
        assert "@everyone" not in description
        assert "**x**" not in description

    def test_the_question_is_collapsed_and_cut_to_its_display_bound(self) -> None:
        question = "Wann\n\n ist\u200b " + "🎉" * 300
        line = question_line(question)

        assert line is not None
        assert line.startswith(f"{KIND_SYMBOLS[MessageKind.ANSWER]} Wann ist ")
        assert len(line) <= QUESTION_DISPLAY_CHARS + 2
        assert line.endswith("…")
        assert question_line("   ") is None
        assert question_line(None) is None

    def test_proactive_shows_its_framing_and_at_most_two_points(self) -> None:
        facts = [_fact(1)]
        points = [{"text": f"Detail {i}.", "facts": [1]} for i in range(MAX_POINTS)]
        card = _card(facts, points=points, proactive=True, locale="de")

        assert card.kind is MessageKind.PROACTIVE
        assert card.top_line == collapse_display_text(t("proactive_reply_label", "de"))
        assert len(card.bullets) == PROACTIVE_MAX_POINTS
        assert len(card.checked_points) == PROACTIVE_MAX_POINTS
        assert card.footer == t("proactive_reply_footer", "de")

    def test_an_answer_citing_nothing_is_not_a_card(self) -> None:
        answer = validate_contract(_reply(1, used_fact_numbers=[]), [_fact(1)])

        with pytest.raises(ValueError, match="at least one cited fact"):
            build_answer_card(answer, [_fact(1)], question="q", locale="en-US", channel_names={})

    def test_an_answer_citing_an_unsupplied_fact_is_refused(self) -> None:
        answer = validate_contract(_reply(1), [_fact(1)])

        with pytest.raises(ValueError, match="not supplied"):
            build_answer_card(answer, [_fact(2)], question="q", locale="en-US", channel_names={})


class TestTheCheckedStatements:
    def test_they_are_exactly_the_displayed_lead_and_points(self) -> None:
        facts = [_fact(1), _fact(2)]
        card = _card(
            facts,
            lead="Lead *text*.",
            points=[{"text": "Point [x].", "facts": [2]}],
            not_covered_topics=["topic"],
        )

        assert card.checked_lead == "Lead *text*."
        assert [p.text for p in card.checked_points] == ["Point [x]."]
        assert plain_answer_text(card) == "Lead *text*.\n- Point [x]. [2]"
        assert "topic" not in plain_answer_text(card)

    def test_points_that_overflow_the_description_are_dropped_whole_and_unchecked(self) -> None:
        facts = [_fact(i) for i in range(1, 6)]
        heavy = "*" * POINT_MAX_CHARS  # escaping doubles it
        points = [{"text": heavy, "facts": [1, 2, 3, 4, 5]} for _ in range(MAX_POINTS)]
        card = _card(facts, lead="#" * LEAD_MAX_CHARS, points=points)

        assert len(compose_description(card, notes_as_subtext=False)) <= DESCRIPTION_MAX_CHARS
        assert len(card.bullets) < MAX_POINTS
        assert len(card.checked_points) == len(card.bullets)
        for bullet in card.bullets:
            assert bullet.count("\\*") == POINT_MAX_CHARS


class TestNoModelCards:
    def test_the_fact_list_card_lists_facts_verbatim_with_link_and_date(self) -> None:
        facts = [_fact(1, "Rules are in [#welcome](https://evil.example)."), _fact(2)]
        card = build_fact_list_card(MessageKind.RELATED, "Maybe related:", facts, question="q")

        assert card.paragraphs == (f"{KIND_SYMBOLS[MessageKind.RELATED]} Maybe related:",)
        assert len(card.bullets) == 2
        assert card.bullets[0].startswith("[Rules are in \\[#welcome\\](https://evil.example).](")
        assert card.checked_lead is None
        assert card.sources == ()

    def test_the_fact_list_drops_facts_that_overflow_from_the_end(self) -> None:
        facts = [_fact(i, "x" * 4000) for i in range(1, 30)]
        card = build_fact_list_card(MessageKind.LIMIT, "note", facts, question="q")

        assert len(compose_description(card, notes_as_subtext=False)) <= DESCRIPTION_MAX_CHARS
        assert 0 < len(card.bullets) < 29

    def test_a_notice_card_carries_the_kinds_symbol(self) -> None:
        card = build_notice_card(MessageKind.ERROR, "Something went wrong.", question=None)

        assert card.paragraphs == (f"{KIND_SYMBOLS[MessageKind.ERROR]} Something went wrong.",)
        assert card.top_line is None


class TestTheEmbed:
    def test_it_uses_the_kinds_colour_and_puts_the_question_in_the_author_line(self) -> None:
        card = _card([_fact(1)], question="When **now**?", channel_names={501: "events"})
        embed = card_to_embed(card)

        assert embed.colour is not None
        assert embed.colour.value == ACCENT_COLORS[MessageKind.ANSWER]
        assert embed.author.name == card.top_line
        assert "**now**" in (embed.author.name or "")  # plain-text slot, not escaped
        assert embed.fields[0].name == t("ask_sources_label", "en-US")
        assert (embed.description or "").startswith("The event starts at 18:00.")

    def test_a_maximal_card_fits_every_embed_limit(self) -> None:
        facts = [_fact(i, "y" * 4000) for i in range(1, 11)]
        names = {500 + i: "c" * 200 for i in range(1, 11)}
        points = [
            {"text": "*" * POINT_MAX_CHARS, "facts": list(range(1, 11))} for _ in range(MAX_POINTS)
        ]
        card = _card(
            facts,
            question="q" * 6000,
            lead="_" * LEAD_MAX_CHARS,
            points=points,
            not_covered_topics=["a b c d e f", "g h i j k l", "m n o p q r"],
            relations=[{"facts": [1, 2], "kind": "unclear_if_same"}],
            channel_names=names,
        )
        embed = card_to_embed(card)

        assert len(embed.description or "") <= EMBED_DESCRIPTION_LIMIT
        assert len(embed.fields) <= EMBED_FIELD_COUNT_LIMIT
        assert all(len(field.value or "") <= EMBED_FIELD_VALUE_LIMIT for field in embed.fields)
        assert _embed_total(embed) <= EMBED_TOTAL_LIMIT
        assert len(embed.author.name or "") <= EMBED_AUTHOR_NAME_LIMIT
        assert sum(value.count("\n") + 1 for value in (f.value or "" for f in embed.fields)) == 10

    def test_an_empty_minimal_notice_still_renders(self) -> None:
        embed = card_to_embed(build_notice_card(MessageKind.RELATED, "Nothing.", question=None))

        assert embed.description
        assert not embed.fields
        assert embed.author.name is None


class TestTheContainer:
    def test_it_has_the_accent_colour_subtext_parts_and_a_divider(self) -> None:
        card = _card(
            [_fact(1)],
            not_covered_topics=["prize"],
            channel_names={501: "events"},
        )
        view = card_to_layout_view(card)
        texts = _layout_texts(view)
        container = next(i for i in view.walk_children() if isinstance(i, discord.ui.Container))

        assert container.accent_colour is not None
        assert int(container.accent_colour) == ACCENT_COLORS[MessageKind.ANSWER]
        assert texts[0].startswith("-# ")
        assert "-# Not recorded: prize." in texts[1]
        assert any(isinstance(i, discord.ui.Separator) for i in view.walk_children())
        assert texts[-1] == f"-# {escape_display_markdown(t('answer_footer', 'en-US'))}"

    def test_the_question_is_escaped_where_markdown_renders(self) -> None:
        view = card_to_layout_view(
            _card([_fact(1)], question="[x](https://evil.example) @everyone")
        )

        assert "@everyone" not in _layout_texts(view)[0]
        assert "\\[x\\]" in _layout_texts(view)[0]

    def test_a_maximal_card_fits_the_text_and_component_limits(self) -> None:
        facts = [_fact(i, "y" * 4000) for i in range(1, 11)]
        points = [
            {"text": "*" * POINT_MAX_CHARS, "facts": list(range(1, 11))} for _ in range(MAX_POINTS)
        ]
        card = _card(facts, question="q" * 600, lead="_" * LEAD_MAX_CHARS, points=points)
        view = card_to_layout_view(card, caption="**caption**")

        assert sum(len(text) for text in _layout_texts(view)) <= COMPONENTS_V2_TEXT_LIMIT
        assert _count_components(view) <= COMPONENTS_V2_COMPONENT_LIMIT
        assert layout_text_length(card) <= COMPONENTS_V2_TEXT_LIMIT

    def test_a_caption_sits_above_the_container(self) -> None:
        view = card_to_layout_view(
            build_notice_card(MessageKind.ERROR, "x", question=None), caption="cap"
        )

        assert isinstance(view.children[0], discord.ui.TextDisplay)
        assert view.children[0].content == "cap"
        assert isinstance(view.children[1], discord.ui.Container)


class TestThePlainTextFallback:
    def test_it_holds_the_question_body_and_sources(self) -> None:
        card = _card([_fact(1)], question="When?", channel_names={501: "events"})
        text = card_to_plain_text(card)

        assert text.startswith(f"**{escape_display_markdown(card.top_line or '')}**")
        assert "The event starts at 18:00." in text
        assert t("ask_sources_label", "en-US") in text

    def test_it_never_exceeds_a_plain_messages_limit(self) -> None:
        facts = [_fact(i, "y" * 4000) for i in range(1, 11)]
        names = {500 + i: "c" * 200 for i in range(1, 11)}
        points = [
            {"text": "*" * POINT_MAX_CHARS, "facts": list(range(1, 11))} for _ in range(MAX_POINTS)
        ]
        card = _card(
            facts,
            question="q" * 6000,
            lead="_" * LEAD_MAX_CHARS,
            points=points,
            channel_names=names,
        )

        assert len(card_to_plain_text(card)) <= PLAIN_MESSAGE_LIMIT


def test_every_new_locale_key_has_its_placeholders_in_every_locale() -> None:
    english = json.loads((_LOCALES_DIR / "en-US.json").read_text(encoding="utf-8"))
    for locale in SUPPORTED_LOCALES:
        data = json.loads((_LOCALES_DIR / f"{locale}.json").read_text(encoding="utf-8"))
        assert "{topics}" in data["answer_not_recorded"]
        for key in (
            "answer_footer",
            "answer_source_message",
            "answer_note_conflict",
            "answer_note_unclear",
        ):
            assert data[key].strip()
            assert "{" not in data[key]
        assert set(english) == set(data)


def test_german_texts_use_the_informal_register() -> None:
    data = json.loads((_LOCALES_DIR / "de.json").read_text(encoding="utf-8"))
    for key in ("answer_footer", "answer_note_conflict", "answer_note_unclear"):
        assert not re.search(r"\b(Sie|Ihnen|Ihr)\b", data[key])
