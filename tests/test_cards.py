"""The card looks of the digest, onboarding, /aura-plan and command notices (P5).

What must hold, without Discord or a model:

* every family is one kind with its own colour and symbol, and every accent
  keeps 3:1 contrast on every Discord theme (the theme test covers new kinds);
* member-written text -- fact sentences, channel names, the server's name --
  never becomes markup or a mention, at any length;
* every card fits Discord's limits at maximum load, in both styles and as plain
  text, by dropping lines from the longest section and saying so;
* the digest shows a summary line, the period, and only the sections it has;
  a changed fact shows the old wording struck through; onboarding numbers its
  rules; the plan card lists what Pro adds only on Free and links only when a
  dashboard is configured;
* an answer card without sections renders exactly as before sections existed.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import discord
import pytest

from aura.answer_card import (
    AnswerCard,
    CardItem,
    CardSection,
    card_to_embed,
    card_to_layout_view,
    card_to_plain_text,
    layout_text_length,
    section_lines,
)
from aura.billing import GuildPlan, PlanBasis, PlanTier, Standing, SubscriptionStanding
from aura.cards import (
    build_digest_card,
    build_notice,
    build_onboarding_card,
    build_plan_card,
    build_pro_refusal_card,
)
from aura.commands.plan import standing_lines
from aura.db.models import Fact, FactStatus
from aura.digest.builder import DigestChange, DigestContent
from aura.i18n import SUPPORTED_LOCALES, t
from aura.onboarding.builder import OnboardingContent
from aura.theme import (
    ACCENT_COLORS,
    COMPONENTS_V2_COMPONENT_LIMIT,
    COMPONENTS_V2_TEXT_LIMIT,
    EMBED_FIELD_COUNT_LIMIT,
    EMBED_FIELD_VALUE_LIMIT,
    EMBED_TOTAL_LIMIT,
    KIND_SYMBOLS,
    MAX_SECTION_ITEMS,
    MessageKind,
)

NOW = datetime(2026, 10, 4, 18, 0, tzinfo=UTC)
GUILD = 100000000000000001
NAMES = {11: "events", 12: "regeln"}


def _fact(fact_id: int, content: str, channel_id: int = 11) -> Fact:
    return Fact(
        id=fact_id,
        guild_id=GUILD,
        channel_id=channel_id,
        message_id=1000 + fact_id,
        content=content,
        embedding=b"",
        status=FactStatus.ACTIVE,
        created_at=datetime(2026, 9, 1 + fact_id % 27, 12, 0, tzinfo=UTC),
    )


def _digest(
    new: list[Fact] | None = None,
    milestones: list[Fact] | None = None,
    changes: list[DigestChange] | None = None,
) -> DigestContent:
    return DigestContent(
        guild_id=GUILD,
        covered_from=NOW - timedelta(days=7),
        covered_until=NOW,
        new_facts=new or [],
        milestones=milestones or [],
        changes=changes or [],
    )


def _change(old: str, new: str, steps: int = 0) -> DigestChange:
    return DigestChange(
        previous=_fact(90, old), current=_fact(91, new), changed_at=NOW, collapsed_steps=steps
    )


def _plan(tier: PlanTier, basis: PlanBasis, standing: Standing, granting: int = 1) -> GuildPlan:
    return GuildPlan(
        guild_id=GUILD,
        tier=tier,
        basis=basis,
        standing=SubscriptionStanding(
            standing=standing,
            access_until=NOW + timedelta(days=20),
            paid_through=NOW + timedelta(days=20),
            shown_subscription_id="s",
            granting_subscription_ids=frozenset(f"s{i}" for i in range(granting)),
        ),
    )


def _embed_total(embed: discord.Embed) -> int:
    return len(embed)


HOSTILE = "@everyone [click](https://evil.example) <@&123> **bold** # heading ~~x~~ ‮​"


class TestTheFamilies:
    @pytest.mark.parametrize(
        "kind", [MessageKind.DIGEST, MessageKind.ONBOARDING, MessageKind.PLAN, MessageKind.CONFIRM]
    )
    def test_each_new_kind_has_a_colour_and_a_symbol_of_its_own(self, kind: MessageKind) -> None:
        others = {k: v for k, v in ACCENT_COLORS.items() if k is not kind}
        assert ACCENT_COLORS[kind] not in others.values()
        assert KIND_SYMBOLS[kind] not in {v for k, v in KIND_SYMBOLS.items() if k is not kind}


class TestTheDigestCard:
    def test_summary_period_sections_and_footer(self) -> None:
        card = build_digest_card(
            _digest(
                new=[_fact(1, "Der Kanal #clips ist neu.")],
                milestones=[_fact(2, "2.000 Mitglieder erreicht.")],
                changes=[_change("Bingo um 19 Uhr.", "Bingo um 20 Uhr.", steps=1)],
            ),
            locale="de",
            interval_label="wöchentlich",
            channel_names=NAMES,
        )

        assert card.kind is MessageKind.DIGEST
        assert card.top_line == f"{KIND_SYMBOLS[MessageKind.DIGEST]} Server-Update"
        assert card.paragraphs[0] == "**Neu: 1 · Geändert: 1 · Meilensteine: 1**"
        assert card.paragraphs[1].startswith("Zeitraum: <t:")
        assert [section.heading for section in card.sections] == ["Meilensteine", "Neu", "Geändert"]
        change = card.sections[2].items[0]
        assert change.text == "~~Bingo um 19 Uhr.~~ → Bingo um 20 Uhr."
        assert change.meta is not None and "[#events](" in change.meta
        assert "Zwischenschritt" in change.meta
        assert card.footer == t("digest_card_footer", "de", interval="wöchentlich")

    def test_a_section_with_nothing_in_it_is_left_out(self) -> None:
        card = build_digest_card(
            _digest(new=[_fact(1, "Neu.")]), locale="de", interval_label="täglich", channel_names={}
        )

        assert [section.heading for section in card.sections] == ["Neu"]
        assert card.paragraphs[0] == "**Neu: 1**"

    def test_more_than_the_section_limit_is_counted_not_listed(self) -> None:
        facts = [_fact(i, f"Fakt {i}.") for i in range(1, MAX_SECTION_ITEMS + 5)]
        card = build_digest_card(
            _digest(new=facts), locale="de", interval_label="x", channel_names={}
        )

        section = card.sections[0]
        assert len(section.items) == MAX_SECTION_ITEMS
        assert section.hidden == 4
        assert section_lines(section, subtext=True)[-1] == t("digest_more_items", "de", count=4)

    def test_an_unnamed_channel_shows_message_instead_of_an_id(self) -> None:
        card = build_digest_card(
            _digest(new=[_fact(1, "x", channel_id=77)]),
            locale="en-US",
            interval_label="weekly",
            channel_names={77: "77"},
        )

        meta = card.sections[0].items[0].meta
        assert meta is not None and meta.startswith("[Message](")

    @pytest.mark.parametrize("locale", sorted(SUPPORTED_LOCALES))
    def test_it_renders_in_every_locale_without_a_missing_key(self, locale: str) -> None:
        card = build_digest_card(
            _digest(new=[_fact(1, "x")], changes=[_change("a", "b")]),
            locale=locale,
            interval_label="w",
            channel_names=NAMES,
        )

        text = card_to_plain_text(card)
        assert "[digest_card" not in text and "{count}" not in text


class TestTheOnboardingCard:
    def test_greeting_numbered_rules_then_status_then_the_rest(self) -> None:
        card = build_onboarding_card(
            OnboardingContent(
                guild_id=GUILD,
                rules=[_fact(1, "Keine Werbung."), _fact(2, "PTT in der Lounge.")],
                status_changes=[_fact(3, "#memes ist zu.")],
                other=[_fact(4, "Quiz am 5. Dezember.")],
                total_eligible=7,
            ),
            locale="de",
            server_name="Bastel\nstube",
            channel_names=NAMES,
        )

        assert card.kind is MessageKind.ONBOARDING
        assert (
            card.top_line == f"{KIND_SYMBOLS[MessageKind.ONBOARDING]} Willkommen auf Bastel stube!"
        )
        assert [s.heading for s in card.sections] == ["Regeln", "Gerade aktuell", "Gut zu wissen"]
        assert card.sections[0].numbered is True
        lines = section_lines(card.sections[0], subtext=True)
        assert lines[0].startswith("1. Keine Werbung.\n-# [#events](")
        assert lines[1].startswith("2. ")
        assert card.footer is not None and "3" in card.footer

    def test_a_hostile_server_name_cannot_ping_or_format(self) -> None:
        card = build_onboarding_card(
            OnboardingContent(
                guild_id=GUILD, rules=[_fact(1, "x")], status_changes=[], other=[], total_eligible=1
            ),
            locale="en-US",
            server_name=HOSTILE * 5,
            channel_names={},
        )
        view = card_to_layout_view(card)
        texts = [
            item.content
            for item in view.walk_children()
            if isinstance(item, discord.ui.TextDisplay)
        ]

        assert card.top_line is not None and len(card.top_line) < 100
        assert all("@everyone" not in text.replace("@​everyone", "") for text in texts)
        assert all("[click](" not in text for text in texts)
        assert all("<@&123>" not in text for text in texts)


class TestThePlanCard:
    def test_free_lists_what_pro_adds_and_links_to_the_dashboard(self) -> None:
        plan = _plan(PlanTier.FREE, PlanBasis.SUBSCRIPTION, Standing.NO_SUBSCRIPTION, granting=0)
        card = build_plan_card(
            plan,
            locale="de",
            standing_lines=standing_lines(plan, locale="de"),
            dashboard_url="https://example.com/dashboard",
            ask_caps=(10, 5, 25),
        )

        assert card.kind is MessageKind.PLAN
        assert [s.heading for s in card.sections] == ["Im Free-Plan", "Mit Pro zusätzlich"]
        assert card.sections[0].items[0].text == t(
            "plan_card_feature_ask_free", "de", guild=10, member=5
        )
        assert card.paragraphs[-1] == "[Plan verwalten](<https://example.com/dashboard>)"
        assert card.footer == t("plan_card_footer", "de")

    def test_pro_lists_what_it_includes_and_nothing_to_buy(self) -> None:
        plan = _plan(PlanTier.PRO, PlanBasis.SUBSCRIPTION, Standing.ACTIVE)
        card = build_plan_card(
            plan,
            locale="en-US",
            standing_lines=standing_lines(plan, locale="en-US"),
            dashboard_url=None,
            ask_caps=(10, 5, 25),
        )

        assert [s.heading for s in card.sections] == ["Included in Pro"]
        assert card.sections[0].items[0].text == t("plan_card_feature_ask_pro", "en-US", guild=25)
        assert not any("](" in paragraph for paragraph in card.paragraphs)

    def test_a_guild_paying_twice_gets_the_warning_as_a_note(self) -> None:
        plan = _plan(PlanTier.PRO, PlanBasis.SUBSCRIPTION, Standing.ACTIVE, granting=2)
        card = build_plan_card(
            plan, locale="de", standing_lines=[], dashboard_url=None, ask_caps=(1, 1, 1)
        )

        assert card.notes == (t("plan_multiple_subscriptions", "de", count=2),)

    def test_the_not_enforced_installation_says_included(self) -> None:
        plan = _plan(PlanTier.PRO, PlanBasis.BILLING_NOT_ENFORCED, Standing.NO_SUBSCRIPTION, 0)
        card = build_plan_card(
            plan, locale="de", standing_lines=[], dashboard_url=None, ask_caps=(1, 1, 1)
        )

        assert card.sections[0].heading == "Enthalten"

    def test_the_refusal_keeps_the_sentence_and_links_only_when_configured(self) -> None:
        with_link = build_pro_refusal_card("de", dashboard_url="https://example.com/d")
        without = build_pro_refusal_card("de", dashboard_url=None)

        assert (
            with_link.paragraphs[0]
            == f"{KIND_SYMBOLS[MessageKind.PLAN]} {t('plan_pro_required', 'de')}"
        )
        assert with_link.paragraphs[1] == "[Auf Pro upgraden](<https://example.com/d>)"
        assert without.paragraphs == (with_link.paragraphs[0],)


class TestNotices:
    def test_a_confirmation_keeps_the_text_after_the_symbol(self) -> None:
        card = build_notice(MessageKind.CONFIRM, "Fakt #12 erstellt.\nZweite Zeile.")

        assert card.paragraphs == (
            f"{KIND_SYMBOLS[MessageKind.CONFIRM]} Fakt #12 erstellt.",
            "Zweite Zeile.",
        )
        assert card.top_line is None

    def test_an_empty_text_still_renders(self) -> None:
        embed = card_to_embed(build_notice(MessageKind.CONFIRM, ""))

        assert embed.description is not None


class TestLimitsAtMaximumLoad:
    def _maximal(self) -> AnswerCard:
        long = HOSTILE + "y" * 4000
        names = {11: "c" * 300}
        return build_digest_card(
            _digest(
                new=[_fact(i, long) for i in range(1, 30)],
                milestones=[_fact(i, long) for i in range(30, 60)],
                changes=[_change(long, long, steps=99) for _ in range(30)],
            ),
            locale="de",
            interval_label="w" * 300,
            channel_names=names,
        )

    def test_the_embed_fits_every_limit_and_says_what_it_left_out(self) -> None:
        card = self._maximal()
        embed = card_to_embed(card)

        assert _embed_total(embed) <= EMBED_TOTAL_LIMIT
        assert len(embed.fields) <= EMBED_FIELD_COUNT_LIMIT
        assert all(len(field.value or "") <= EMBED_FIELD_VALUE_LIMIT for field in embed.fields)
        assert any("weitere" in (field.value or "") for field in embed.fields)

    def test_the_container_fits_the_text_and_component_limits(self) -> None:
        view = card_to_layout_view(self._maximal())
        texts = [
            item.content
            for item in view.walk_children()
            if isinstance(item, discord.ui.TextDisplay)
        ]

        assert sum(len(text) for text in texts) <= COMPONENTS_V2_TEXT_LIMIT
        assert len(list(view.walk_children())) <= COMPONENTS_V2_COMPONENT_LIMIT

    def test_the_plain_text_fallback_fits_a_message(self) -> None:
        assert len(card_to_plain_text(self._maximal())) <= 2000

    def test_hostile_fact_text_never_becomes_markup_or_a_ping(self) -> None:
        card = build_digest_card(
            _digest(new=[_fact(1, HOSTILE)]),
            locale="en-US",
            interval_label="weekly",
            channel_names={},
        )
        line = card.sections[0].items[0].text

        assert "[click](" not in line
        assert "<@&123>" not in line
        assert "@everyone" not in line.replace("@​everyone", "")
        assert "‮" not in line and "​" not in line.replace("@​everyone", "")


class TestCardsWithoutSectionsAreUnchanged:
    def test_a_layout_without_sections_has_no_extra_component(self) -> None:
        card = AnswerCard(kind=MessageKind.ANSWER, top_line="t", paragraphs=("Body.",), footer="f")
        view = card_to_layout_view(card)

        assert len(list(view.walk_children())) == 4  # container, top, body, footer
        assert layout_text_length(card) == len("-# t") + len("Body.") + len("-# f")

    def test_an_embed_without_sections_has_no_extra_field(self) -> None:
        card = AnswerCard(kind=MessageKind.ANSWER, top_line="t", paragraphs=("Body.",))

        assert card_to_embed(card).fields == []

    def test_sections_render_as_fields_in_an_embed_and_as_text_in_a_container(self) -> None:
        card = AnswerCard(
            kind=MessageKind.DIGEST,
            top_line="t",
            paragraphs=("Body.",),
            sections=(CardSection("Neu", (CardItem("Eins.", "[#a](https://x) · d"),)),),
        )

        assert [f.name for f in card_to_embed(card).fields] == ["Neu"]
        assert card_to_embed(card).fields[0].value == "• Eins. · [#a](https://x) · d"
        texts = [
            item.content
            for item in card_to_layout_view(card).walk_children()
            if isinstance(item, discord.ui.TextDisplay)
        ]
        assert "**Neu**\n• Eins.\n-# [#a](https://x) · d" in texts
