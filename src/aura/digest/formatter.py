"""Rendering an assembled digest as one Discord embed, in the reader's language.

Everything localizable here goes through the existing translation-key system --
headings, labels, the period line, the footer. What deliberately does NOT get
translated is the fact sentences themselves: a fact is stored in whatever
language it was written in, and CLAUDE.md's i18n rule is that Aura frames
content in the user's language rather than rewriting the content. The same
choice /aura-pending already makes when it shows a candidate's own words beside
translated labels, for the same reason -- translating a stored fact would mean
generating text, which is exactly what this sub-phase has no LLM call for.

**Every list in here is bounded twice, by item count and by character budget.**
That is not decoration. A guild that confirms two hundred facts in a week, or a
single fact whose text is 4,000 characters (the modal's own limit), would
otherwise produce an embed Discord refuses outright -- and a digest that fails
to send is indistinguishable, from the outside, from a digest that had nothing
to say. Both bounds degrade the same way: the section is truncated and says so.

The actual escaping, link-building and two-tier truncation are shared with
onboarding (aura.onboarding.formatter) through aura.rendering -- see that
module's docstring for why a link-hijack fix must live in exactly one place
rather than being copied.
"""
from __future__ import annotations

import discord

from aura.db.models import Fact
from aura.digest.builder import DigestChange, DigestContent
from aura.digest.intervals import describe_interval
from aura.i18n import DEFAULT_LOCALE, t
from aura.rendering import discord_timestamp, fit_lines, inline_fact_text, source_link

# How many entries one section lists before collapsing the rest into a count.
# Ten is what fits the field budget above at roughly full-length sentences, and
# also about as much as anyone reads in a summary post.
_MAX_ITEMS_PER_SECTION = 10

# A distinct colour from every other embed Aura sends -- /aura-ask has none,
# proactive relief is blurple, a contradiction warning is red -- so a digest is
# recognisable as a digest before a word of it is read.
_DIGEST_COLOUR = discord.Colour.gold()

# The highlight on the milestone section, kept out of the locale files for the
# same reason /aura-pending keeps its contradiction icon out of them: it is not
# text, it must not vary by language, and a translator must not be able to drop
# it by leaving a value empty.
_MILESTONE_ICON = "🏆"


def digest_locale(guild: discord.Guild | None) -> str:
    """The language one guild's digest is written in.

    A digest has no asking user whose interaction.locale could be read, so it
    uses the guild's own preferred locale -- the same signal, and the same
    fallback, that an unprompted proactive answer uses (see
    aura.proactive.responder). Kept as its own function rather than shared with
    that one because the two answers are free to diverge: a per-guild digest
    language is a plausible future setting on digest_config, where forcing the
    same choice onto proactive replies to individual members would be wrong.
    """
    preferred = getattr(guild, "preferred_locale", None) if guild is not None else None
    return str(preferred) if preferred else DEFAULT_LOCALE


def _fact_line(fact: Fact) -> str:
    """One new fact as a single bulleted line: the sentence, linked, plus its date."""
    return (
        f"• [{inline_fact_text(fact.content)}]({source_link(fact)}) · "
        f"{discord_timestamp(fact.created_at)}"
    )


def _change_line(change: DigestChange, locale: str) -> str:
    """One "before -> now" pair, on two lines so the change itself is readable.

    The old sentence is not linked and the new one is: the reader's next
    question about a change is "what does it say now", and the answer is the
    thing worth being one click from its source. A run of collapsed intermediate
    steps is named rather than hidden, so the digest never implies a single tidy
    edit where there was a series of corrections.
    """
    previous = inline_fact_text(change.previous.content)
    current = inline_fact_text(change.current.content)
    line = (
        f"• {previous}\n"
        f"→ [{current}]({source_link(change.current)}) · "
        f"{discord_timestamp(change.changed_at)}"
    )
    if change.collapsed_steps > 0:
        line += " " + t("digest_change_collapsed", locale, count=change.collapsed_steps)
    return line


def _fit_lines(lines: list[str], locale: str) -> str:
    """Join as many rendered lines as fit for one digest field. See aura.rendering.fit_lines."""
    return fit_lines(
        lines, locale, max_items=_MAX_ITEMS_PER_SECTION, more_items_key="digest_more_items"
    )


def build_digest_embed(
    content: DigestContent, *, locale: str, interval_seconds: int
) -> discord.Embed:
    """Render an assembled digest as the embed that gets posted.

    Sections are ordered by how much a reader cares: milestones first because
    they are the reason that category exists as its own thing, then what is new,
    then what changed. A section with nothing in it is omitted entirely rather
    than shown as "New facts (0)" -- a digest is only ever built when at least
    one section has content (see DigestContent.is_empty), so an empty section is
    a real absence, not a nothing-happened digest.

    Never called for an empty digest; that case is decided before this function
    is reached, because "post nothing" is not a rendering decision.
    """
    embed = discord.Embed(
        title=t("digest_title", locale),
        description=t(
            "digest_period",
            locale,
            start=discord_timestamp(content.covered_from),
            end=discord_timestamp(content.covered_until),
        ),
        colour=_DIGEST_COLOUR,
    )

    if content.milestones:
        embed.add_field(
            name=f"{_MILESTONE_ICON} "
            + t("digest_milestones_label", locale, count=len(content.milestones)),
            value=_fit_lines([_fact_line(fact) for fact in content.milestones], locale),
            inline=False,
        )

    if content.new_facts:
        embed.add_field(
            name=t("digest_new_facts_label", locale, count=len(content.new_facts)),
            value=_fit_lines([_fact_line(fact) for fact in content.new_facts], locale),
            inline=False,
        )

    if content.changes:
        embed.add_field(
            name=t("digest_changes_label", locale, count=len(content.changes)),
            value=_fit_lines(
                [_change_line(change, locale) for change in content.changes], locale
            ),
            inline=False,
        )

    embed.set_footer(
        text=t("digest_footer", locale, interval=describe_interval(interval_seconds, locale))
    )
    return embed
