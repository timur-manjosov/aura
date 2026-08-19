"""Rendering an assembled onboarding summary as one Discord embed, in the new
member's language.

Structurally the digest's sibling (aura.digest.formatter) and deliberately so:
the same localization approach (translated headings, untranslated fact text --
see that module's docstring for why), and the same shared, security-critical
line-rendering primitives from aura.rendering, so the digest's link-hijack fix
(reports/phase-3e.txt Section 7b) protects this message too without having
been re-derived or forgotten here.

What differs from the digest, beyond section labels: there is no "period"
line (onboarding has no window, see aura.onboarding.builder), and there is an
extra note for the case the digest never has to handle -- a global item cap
that left facts out entirely rather than a single field's character budget
doing the truncating. See _content_note.
"""
from __future__ import annotations

import discord

from aura.db.models import Fact
from aura.i18n import DEFAULT_LOCALE, t
from aura.onboarding.builder import OnboardingContent
from aura.rendering import discord_timestamp, fit_lines, inline_fact_text, source_link

# How many entries one section lists before collapsing the rest into a count.
# Same value and same reasoning as the digest's _MAX_ITEMS_PER_SECTION: it is
# what fits the 1,024-character field budget at roughly full-length sentences,
# and about as much as anyone reads in a single section. The onboarding
# message's overall size is bounded separately and earlier, by
# onboarding_fact_limit (see aura.onboarding.builder) -- this is the
# per-field safety net underneath that product decision, not a second version
# of it.
_MAX_ITEMS_PER_SECTION = 10

# A colour distinct from every other embed Aura sends (gold is the digest's;
# blurple is proactive relief's; /aura-ask has none), so an onboarding message
# is recognisable as one before a word of it is read.
_ONBOARDING_COLOUR = discord.Colour.green()


def onboarding_locale(guild: discord.Guild | None) -> str:
    """The language one guild's onboarding message is written in.

    Onboarding has no asking user whose interaction.locale could be read --
    the reader is a member who has not typed anything yet -- so it uses the
    guild's own preferred locale, the same signal and fallback
    aura.digest.formatter.digest_locale uses for the same reason. Kept as its
    own function rather than shared with that one because the two are free to
    diverge if either ever grows a per-guild language override.
    """
    preferred = getattr(guild, "preferred_locale", None) if guild is not None else None
    return str(preferred) if preferred else DEFAULT_LOCALE


def _fact_line(fact: Fact) -> str:
    """One fact as a single bulleted line: the sentence, linked, plus its date."""
    return (
        f"• [{inline_fact_text(fact.content)}]({source_link(fact)}) · "
        f"{discord_timestamp(fact.created_at)}"
    )


def _fit_section(facts: list[Fact], locale: str) -> str:
    """Join as many fact lines as fit one onboarding field. See aura.rendering.fit_lines."""
    return fit_lines(
        [_fact_line(fact) for fact in facts],
        locale,
        max_items=_MAX_ITEMS_PER_SECTION,
        more_items_key="onboarding_more_items",
    )


def build_onboarding_embed(content: OnboardingContent, *, locale: str) -> discord.Embed:
    """Render an assembled onboarding summary as the embed that gets posted.

    Sections are ordered by CLAUDE.md's onboarding priority, decided in
    aura.onboarding.builder and simply read off here in the order the content
    already carries it: rules first, current status second, everything else
    last. A section with nothing in it is omitted entirely rather than shown
    as "Rules (0)" -- the same choice build_digest_embed makes, for the same
    reason: this is only ever called for non-empty content (see
    OnboardingContent.is_empty), so an empty section is a real absence.

    Never called for empty content; that case is decided before this function
    is reached, because "post nothing" is not a rendering decision (mirrors
    build_digest_embed exactly).
    """
    embed = discord.Embed(
        title=t("onboarding_title", locale),
        description=t("onboarding_intro", locale),
        colour=_ONBOARDING_COLOUR,
    )

    if content.rules:
        embed.add_field(
            name=t("onboarding_rules_label", locale, count=len(content.rules)),
            value=_fit_section(content.rules, locale),
            inline=False,
        )

    if content.status_changes:
        embed.add_field(
            name=t("onboarding_status_label", locale, count=len(content.status_changes)),
            value=_fit_section(content.status_changes, locale),
            inline=False,
        )

    if content.other:
        embed.add_field(
            name=t("onboarding_other_label", locale, count=len(content.other)),
            value=_fit_section(content.other, locale),
            inline=False,
        )

    footer = t("onboarding_footer", locale)
    if content.omitted_count > 0:
        # The global item cap (aura.onboarding.builder) left facts out
        # entirely, in a section that may not even be shown here -- distinct
        # from a single field's own "…and N more" line, which only ever
        # speaks about the section it is attached to. Named once, at the
        # message level, so the gap is never silent.
        footer = footer + " " + t("onboarding_capped_note", locale, count=content.omitted_count)
    embed.set_footer(text=footer)
    return embed
