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
"""
from __future__ import annotations

from datetime import datetime

import discord

from aura.db.models import Fact
from aura.digest.builder import DigestChange, DigestContent
from aura.digest.intervals import describe_interval
from aura.i18n import DEFAULT_LOCALE, t

# Discord's own hard cap on an embed field's value, not a choice made here.
# Exceeding it is a 400 from the API, i.e. no digest at all.
_FIELD_VALUE_LIMIT = 1024

# How much of one fact's sentence a digest line shows. A digest is an index of
# what changed, not a replacement for reading the fact: the sentence is a link
# to its own source message, so anyone who wants the whole of a long one is one
# click away from the message it was distilled from.
_ITEM_TEXT_LIMIT = 140

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


def _inline(text: str) -> str:
    """Flatten one fact's sentence into a single, link-safe line.

    Two transformations, both load-bearing rather than cosmetic:

      * Whitespace runs collapse to single spaces. A fact entered through the
        modal can contain newlines, and one multi-line fact in a bulleted list
        turns the whole section into unreadable mush.
      * Backslashes and square brackets are escaped, in that order. The sentence
        becomes the label of a markdown link to its source message, and an
        unescaped `]` inside the label ends the link early, leaving the rest of
        the sentence and the raw URL spilled across the line.

    Nothing else is escaped, matching how /aura-ask and /aura-pending already
    render fact text: a fact containing `*` renders as emphasis, which is
    untidy but harmless, and escaping every markdown character would make
    ordinary punctuation-heavy sentences unreadable to fix a cosmetic problem.

    Truncation happens BEFORE escaping, so a cut can never land inside an escape
    sequence and leave a trailing backslash that would eat the closing bracket.
    """
    collapsed = " ".join(text.split())
    if not collapsed:
        # A fact whose text is nothing but whitespace or zero-width characters.
        # Unreachable through the modal (it rejects blank input) but cheap to
        # survive, and an empty markdown label renders as a broken link.
        return "…"
    if len(collapsed) > _ITEM_TEXT_LIMIT:
        collapsed = collapsed[: _ITEM_TEXT_LIMIT - 1] + "…"
    return collapsed.replace("\\", "\\\\").replace("[", "\\[").replace("]", "\\]")


def _source_link(fact: Fact) -> str:
    """The Discord permalink to the message a fact was distilled from.

    The knowledge model stores this reference instead of a second copy of the
    original text (CLAUDE.md's Fact component), and a digest is the place that
    pays off most: a line of summary the reader can click through to the message
    behind it is a citation, while the same line without one is a claim.
    """
    return f"https://discord.com/channels/{fact.guild_id}/{fact.channel_id}/{fact.message_id}"


def _timestamp(moment: datetime) -> str:
    """A date Discord renders in each reader's own locale and timezone.

    `<t:...:d>` is resolved by the Discord client, not by Aura, which is the
    only way one posted message shows a German reader 16.08.2026 and an American
    one 8/16/2026. Formatting the date here would pick one of those for
    everybody and would need a date format per locale file on top.
    """
    return f"<t:{int(moment.timestamp())}:d>"


def _fact_line(fact: Fact) -> str:
    """One new fact as a single bulleted line: the sentence, linked, plus its date."""
    return f"• [{_inline(fact.content)}]({_source_link(fact)}) · {_timestamp(fact.created_at)}"


def _change_line(change: DigestChange, locale: str) -> str:
    """One "before -> now" pair, on two lines so the change itself is readable.

    The old sentence is not linked and the new one is: the reader's next
    question about a change is "what does it say now", and the answer is the
    thing worth being one click from its source. A run of collapsed intermediate
    steps is named rather than hidden, so the digest never implies a single tidy
    edit where there was a series of corrections.
    """
    previous = _inline(change.previous.content)
    current = _inline(change.current.content)
    line = (
        f"• {previous}\n"
        f"→ [{current}]({_source_link(change.current)}) · {_timestamp(change.changed_at)}"
    )
    if change.collapsed_steps > 0:
        line += " " + t("digest_change_collapsed", locale, count=change.collapsed_steps)
    return line


def _fit_lines(lines: list[str], locale: str) -> str:
    """Join as many rendered lines as fit, and say how many were left out.

    Bounded by both _MAX_ITEMS_PER_SECTION and the field's own character budget,
    because either one alone leaves a hole: ten items of 4,000 characters
    overflow the field, and a hundred one-word facts fit the field but make an
    unreadable wall. Room for the "and N more" note is reserved before the last
    line is accepted, so the note itself can never be what pushes the value over
    the limit.

    Returns the note alone if not even one line fits -- a case only a
    pathological fact can reach, and one where an empty field value would be
    rejected by Discord outright.
    """
    kept: list[str] = []
    used = 0
    for index, line in enumerate(lines[:_MAX_ITEMS_PER_SECTION]):
        would_omit = len(lines) - index - 1
        reserve = (
            len(t("digest_more_items", locale, count=would_omit)) + 1 if would_omit > 0 else 0
        )
        cost = len(line) + (1 if kept else 0)
        if used + cost + reserve > _FIELD_VALUE_LIMIT:
            break
        kept.append(line)
        used += cost

    omitted = len(lines) - len(kept)
    if omitted > 0:
        kept.append(t("digest_more_items", locale, count=omitted))
    return "\n".join(kept)


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
            start=_timestamp(content.covered_from),
            end=_timestamp(content.covered_until),
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
