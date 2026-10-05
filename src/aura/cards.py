"""The card looks of the digest, onboarding, /aura-plan and command notices (P5).

P4 gave Aura's answers a design system (aura.theme) and a renderer
(aura.answer_card). This module builds the other message families out of the
same parts, so every message Aura sends reads as one family: a kind's symbol in
a small top line, a lead, titled sections of lines each with a quiet source
line under it, a quiet footer, and the kind's accent colour.

* **Digest** -- a summary line on top ("Neu: 3 · Geändert: 1"), the period as
  Discord timestamps, then milestones, new facts and changed facts, each with
  its channel and date; a changed fact shows its old wording struck through.
* **Onboarding** -- a greeting with the server's name, the rules numbered first,
  then what currently holds, then the rest.
* **/aura-plan** -- the standing sentence, what this plan includes, and on Free
  a short, factual "with Pro" list; the management link as a masked link. Shown
  only to an admin who asked; nothing here nags anyone else.
* **Notices** -- one line after the kind's symbol: a confirmation, a refusal.

Each family is switched separately (DIGEST_LOOK, ONBOARDING_LOOK, PLAN_LOOK,
NOTICE_LOOK); with the default `classic` the caller never reaches this module.

**No model, ever.** Everything here renders structured data that already
exists -- fact sentences written when the facts were created, their channels,
their dates, a plan's state. CLAUDE.md's "judgment, never knowledge" cuts the
same way in reverse: nothing needs judging, so nothing is generated. Every piece
of member-written text (fact sentences, channel names, the server's name) is
collapsed, cut and escaped before it reaches markup.

Pure: imports no LLM client, database, retrieval or Discord network code.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Final

from aura.answer_card import AnswerCard, CardItem, CardSection, build_notice_card
from aura.billing import GuildPlan, PlanBasis
from aura.db.models import Fact
from aura.i18n import t
from aura.rendering import (
    collapse_display_text,
    discord_timestamp,
    escape_display_markdown,
    link_label,
    shorten,
    source_link,
)
from aura.theme import (
    CHANNEL_LABEL_MAX_CHARS,
    KIND_SYMBOLS,
    MAX_SECTION_ITEMS,
    SECTION_ITEM_MAX_CHARS,
    SOURCE_SEPARATOR,
    MessageKind,
)

if TYPE_CHECKING:
    # Types only: importing the digest or onboarding package at runtime would
    # run its __init__, which imports the scheduler or listener that imports
    # this module.
    from aura.digest.builder import DigestChange, DigestContent
    from aura.onboarding.builder import OnboardingContent

# How much of a server's name the onboarding greeting shows.
_SERVER_NAME_MAX_CHARS: Final = 60

# How much of a superseded sentence a changed-fact line shows: shorter than the
# current one, which is what the reader came for.
_PREVIOUS_ITEM_MAX_CHARS: Final = 90

# The separator of the digest's summary line ("Neu: 3 · Geändert: 1").
_SUMMARY_SEPARATOR: Final = " · "


def _fact_text(text: str, limit: int = SECTION_ITEM_MAX_CHARS) -> str:
    """Return one fact's sentence as a single escaped line of at most `limit` characters."""
    collapsed = collapse_display_text(text) or "…"
    return escape_display_markdown(shorten(collapsed, limit), at_line_start=False)


def _source_meta(fact: Fact, channel_names: Mapping[int, str], locale: str) -> str:
    """Return the quiet source line of one fact: the channel as a link, then the date."""
    name = channel_names.get(fact.channel_id)
    if name and name != str(fact.channel_id):
        label = link_label(f"#{name}", CHANNEL_LABEL_MAX_CHARS)
    else:
        label = link_label(t("answer_source_message", locale), CHANNEL_LABEL_MAX_CHARS)
    return f"[{label}]({source_link(fact)}){SOURCE_SEPARATOR}{discord_timestamp(fact.created_at)}"


def _section(
    heading: str,
    items: Sequence[CardItem],
    *,
    more_template: str,
    numbered: bool = False,
) -> CardSection:
    """Return a section showing at most `MAX_SECTION_ITEMS` lines, the rest counted."""
    shown = tuple(items[:MAX_SECTION_ITEMS])
    return CardSection(
        heading=heading,
        items=shown,
        hidden=len(items) - len(shown),
        more_template=more_template,
        numbered=numbered,
    )


def _change_item(change: DigestChange, channel_names: Mapping[int, str], locale: str) -> CardItem:
    """Return one changed fact: the old wording struck through, then the current one."""
    previous = _fact_text(change.previous.content, _PREVIOUS_ITEM_MAX_CHARS)
    current = _fact_text(change.current.content)
    meta = _source_meta(change.current, channel_names, locale)
    if change.collapsed_steps > 0:
        meta += " " + t("digest_change_collapsed", locale, count=change.collapsed_steps)
    return CardItem(text=f"~~{previous}~~ → {current}", meta=meta)


def build_digest_card(
    content: DigestContent,
    *,
    locale: str,
    interval_label: str,
    channel_names: Mapping[int, str],
) -> AnswerCard:
    """Render an assembled digest as a card.

    Parameters
    ----------
    content
        The assembled digest; never empty (decided before rendering).
    locale
        The guild's language.
    interval_label
        The guild's cadence in words (aura.digest.intervals.describe_interval),
        named in the footer.
    channel_names
        Channel names by channel id, for every fact shown.

    Returns
    -------
    AnswerCard
        Kind DIGEST: the title, a summary line and the period, then the
        milestones, new facts and changed facts as sections. Sections with
        nothing in them are left out.
    """
    counts = [
        (t("digest_card_count_new", locale, count=len(content.new_facts)), content.new_facts),
        (t("digest_card_count_changed", locale, count=len(content.changes)), content.changes),
        (
            t("digest_card_count_milestones", locale, count=len(content.milestones)),
            content.milestones,
        ),
    ]
    summary = _SUMMARY_SEPARATOR.join(label for label, items in counts if items)
    more = t("digest_more_items", locale, count="{count}")
    sections: list[CardSection] = []
    if content.milestones:
        sections.append(
            _section(
                t("digest_card_milestones", locale),
                [
                    CardItem(_fact_text(fact.content), _source_meta(fact, channel_names, locale))
                    for fact in content.milestones
                ],
                more_template=more,
            )
        )
    if content.new_facts:
        sections.append(
            _section(
                t("digest_card_new", locale),
                [
                    CardItem(_fact_text(fact.content), _source_meta(fact, channel_names, locale))
                    for fact in content.new_facts
                ],
                more_template=more,
            )
        )
    if content.changes:
        sections.append(
            _section(
                t("digest_card_changed", locale),
                [_change_item(change, channel_names, locale) for change in content.changes],
                more_template=more,
            )
        )
    return AnswerCard(
        kind=MessageKind.DIGEST,
        top_line=f"{KIND_SYMBOLS[MessageKind.DIGEST]} {t('digest_card_title', locale)}",
        paragraphs=(
            f"**{summary}**",
            t(
                "digest_card_period",
                locale,
                start=discord_timestamp(content.covered_from),
                end=discord_timestamp(content.covered_until),
            ),
        ),
        footer=t("digest_card_footer", locale, interval=interval_label),
        sections=tuple(sections),
    )


def build_onboarding_card(
    content: OnboardingContent,
    *,
    locale: str,
    server_name: str,
    channel_names: Mapping[int, str],
) -> AnswerCard:
    """Render an assembled onboarding summary as a card.

    Parameters
    ----------
    content
        The assembled content; never empty (decided before rendering).
    locale
        The guild's language.
    server_name
        The server's name, member-controlled text: collapsed and cut here, and
        escaped wherever it renders as markup.
    channel_names
        Channel names by channel id, for every fact shown.

    Returns
    -------
    AnswerCard
        Kind ONBOARDING: a greeting with the server's name, a short intro, the
        rules numbered, then what currently holds, then the rest.
    """
    more = t("onboarding_more_items", locale, count="{count}")
    name = shorten(collapse_display_text(server_name) or "Discord", _SERVER_NAME_MAX_CHARS)

    def items(facts: Sequence[Fact]) -> list[CardItem]:
        return [
            CardItem(_fact_text(fact.content), _source_meta(fact, channel_names, locale))
            for fact in facts
        ]

    sections: list[CardSection] = []
    if content.rules:
        sections.append(
            _section(
                t("onboarding_card_rules", locale),
                items(content.rules),
                more_template=more,
                numbered=True,
            )
        )
    if content.status_changes:
        sections.append(
            _section(
                t("onboarding_card_status", locale),
                items(content.status_changes),
                more_template=more,
            )
        )
    if content.other:
        sections.append(
            _section(t("onboarding_card_other", locale), items(content.other), more_template=more)
        )
    footer = t("onboarding_card_footer", locale)
    if content.omitted_count > 0:
        footer += " " + t("onboarding_capped_note", locale, count=content.omitted_count)
    return AnswerCard(
        kind=MessageKind.ONBOARDING,
        top_line=(
            f"{KIND_SYMBOLS[MessageKind.ONBOARDING]} "
            f"{t('onboarding_card_title', locale, server=name)}"
        ),
        paragraphs=(t("onboarding_card_intro", locale),),
        footer=footer,
        sections=tuple(sections),
    )


_PRO_FEATURE_KEYS: Final[tuple[str, ...]] = (
    "plan_card_feature_proactive",
    "plan_card_feature_extraction",
    "plan_card_feature_digest",
    "plan_card_feature_onboarding",
    "plan_card_feature_backfill",
)


def build_plan_card(
    plan: GuildPlan,
    *,
    locale: str,
    standing_lines: Sequence[str],
    dashboard_url: str | None,
    ask_caps: tuple[int, int, int],
) -> AnswerCard:
    """Render /aura-plan as a card.

    Parameters
    ----------
    plan
        The guild's decided plan.
    locale
        The admin's language.
    standing_lines
        The standing sentences /aura-plan already writes (trusted locale text),
        from `aura.commands.plan.standing_lines`.
    dashboard_url
        The operator's billing dashboard, or None.
    ask_caps
        (Free per-guild, Free per-member, Pro per-guild) daily /aura-ask
        answers, from the settings.

    Returns
    -------
    AnswerCard
        Kind PLAN: the standing, what the plan includes, on Free the features
        Pro adds, and the management link when one is configured.

    Notes
    -----
    The Free view lists what Pro adds as plain facts under a neutral heading --
    no exclamation marks, no "upgrade now". It is shown only to an admin who ran
    /aura-plan; the refusal a Pro-only command gives is the only other place a
    plan is mentioned, and only when that admin tried to switch the feature on.
    """
    free_guild, free_member, pro_guild = ask_caps
    manual = CardItem(t("plan_card_feature_manual", locale))
    pro_items = [CardItem(t(key, locale)) for key in _PRO_FEATURE_KEYS]
    sections: list[CardSection]
    if plan.is_pro:
        heading = (
            t("plan_card_included", locale)
            if plan.basis is PlanBasis.BILLING_NOT_ENFORCED
            else t("plan_card_included_pro", locale)
        )
        sections = [
            CardSection(
                heading=heading,
                items=(
                    CardItem(t("plan_card_feature_ask_pro", locale, guild=pro_guild)),
                    manual,
                    *pro_items,
                ),
            )
        ]
    else:
        sections = [
            CardSection(
                heading=t("plan_card_included_free", locale),
                items=(
                    CardItem(
                        t(
                            "plan_card_feature_ask_free",
                            locale,
                            guild=free_guild,
                            member=free_member,
                        )
                    ),
                    manual,
                ),
            ),
            CardSection(heading=t("plan_card_with_pro", locale), items=tuple(pro_items)),
        ]
    paragraphs = list(standing_lines)
    notes: list[str] = []
    granting = len(plan.standing.granting_subscription_ids)
    if granting > 1:
        notes.append(t("plan_multiple_subscriptions", locale, count=granting))
    if dashboard_url:
        paragraphs.append(f"[{link_label(t('plan_card_manage', locale), 80)}](<{dashboard_url}>)")
    return AnswerCard(
        kind=MessageKind.PLAN,
        top_line=f"{KIND_SYMBOLS[MessageKind.PLAN]} {t('plan_card_title', locale)}",
        paragraphs=tuple(paragraphs),
        notes=tuple(notes),
        footer=t("plan_card_footer", locale),
        sections=tuple(sections),
    )


def build_pro_refusal_card(locale: str, *, dashboard_url: str | None) -> AnswerCard:
    """Render the refusal a Pro-only command gives on a Free server, as a PLAN notice.

    Parameters
    ----------
    locale
        The admin's language.
    dashboard_url
        The operator's billing dashboard, or None (then no link).

    Returns
    -------
    AnswerCard
        The plan symbol, the existing refusal sentence (nothing changed, what
        still works), and a masked upgrade link when one is configured.
    """
    card = build_notice_card(MessageKind.PLAN, t("plan_pro_required", locale), question=None)
    if not dashboard_url:
        return card
    link = f"[{link_label(t('plan_card_upgrade', locale), 80)}](<{dashboard_url}>)"
    return AnswerCard(kind=card.kind, top_line=None, paragraphs=(*card.paragraphs, link))


def build_notice(kind: MessageKind, text: str) -> AnswerCard:
    """Render one command reply as a notice card: the kind's symbol, then the text.

    Parameters
    ----------
    kind
        CONFIRM for a confirmation, ERROR for a refusal or failure, PLAN for a
        plan message.
    text
        The reply's localized text, exactly what the classic look sends; it may
        carry markdown and channel mentions the locale file wrote.

    Returns
    -------
    AnswerCard
        A card with no top line, one paragraph per line of `text`, the first
        prefixed with the symbol.
    """
    lines = [line for line in text.split("\n") if line.strip()] or [text]
    first = f"{KIND_SYMBOLS[kind]} {lines[0]}"
    return AnswerCard(kind=kind, top_line=None, paragraphs=(first, *lines[1:]))
