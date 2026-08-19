"""Shared, security-critical primitives for rendering a Fact as one line of a
public Discord message.

Two features render lists of facts as bulleted, source-linked lines inside
embed fields -- the periodic digest (aura.digest.formatter) and onboarding
(aura.onboarding.formatter) -- and both share the exact same vulnerability
such rendering has to defend against: a fact's own text can contain `[` / `]`
that would otherwise hijack the markdown link built around it (see
reports/phase-3e.txt Section 7b for how that was found and fixed for the
digest). That fix, and the two-tier item-count/character-budget truncation
next to it, lives in exactly ONE place so a future change to either is
automatically applied everywhere a fact list is rendered, instead of relying
on every new caller remembering to copy it correctly.

Everything here is pure and Discord-connection-free, per CLAUDE.md's testing
rule: the interesting behaviour (escaping, truncation, the "and N more" note)
is testable without a gateway, an embed, or even a locale beyond the string it
is handed.
"""
from __future__ import annotations

import unicodedata
from datetime import datetime

from aura.db.models import Fact
from aura.i18n import t

# Discord's own hard cap on an embed field's value, not a choice made here.
# Exceeding it is a 400 from the API, i.e. no message at all.
FIELD_VALUE_LIMIT = 1024

# How much of one fact's sentence a rendered line shows. A rendered line is an
# index of what a fact says, not a replacement for reading it: the sentence is
# a link to its own source message, so anyone who wants the whole of a long
# one is one click away from the message it was distilled from.
ITEM_TEXT_LIMIT = 140


def inline_fact_text(text: str) -> str:
    """Flatten one fact's sentence into a single, link-safe line.

    Two transformations, both load-bearing rather than cosmetic:

      * Whitespace runs collapse to single spaces. A fact entered through the
        modal can contain newlines, and one multi-line fact in a bulleted list
        turns the whole section into unreadable mush.
      * Backslashes and square brackets are escaped, in that order. The
        sentence becomes the label of a markdown link to its source message,
        and an unescaped `]` inside the label ends the link early, leaving the
        rest of the sentence and the raw URL spilled across the line.

    Nothing else is escaped, matching how /aura-ask and /aura-pending already
    render fact text: a fact containing `*` renders as emphasis, which is
    untidy but harmless, and escaping every markdown character would make
    ordinary punctuation-heavy sentences unreadable to fix a cosmetic problem.

    Truncation happens BEFORE escaping, so a cut can never land inside an
    escape sequence and leave a trailing backslash that would eat the closing
    bracket.
    """
    collapsed = " ".join(text.split())
    # str.split() only recognises Unicode whitespace (category Z*), not the
    # zero-width/format characters (category Cf: U+200B ZERO WIDTH SPACE,
    # U+FEFF BOM, and friends) a member can also paste in isolation. Text that
    # collapses to nothing but Cf characters is exactly as blank to a reader
    # as true whitespace, so it gets the same placeholder rather than
    # rendering as an invisible-but-technically-present link label.
    if not collapsed or all(unicodedata.category(ch) == "Cf" for ch in collapsed):
        # A fact whose text is nothing but whitespace or zero-width
        # characters. Unreachable through the modal (it rejects blank input)
        # but cheap to survive, and an empty markdown label renders as a
        # broken link.
        return "…"
    if len(collapsed) > ITEM_TEXT_LIMIT:
        collapsed = collapsed[: ITEM_TEXT_LIMIT - 1] + "…"
    return collapsed.replace("\\", "\\\\").replace("[", "\\[").replace("]", "\\]")


def source_link(fact: Fact) -> str:
    """The Discord permalink to the message a fact was distilled from.

    The knowledge model stores this reference instead of a second copy of the
    original text (CLAUDE.md's Fact component), and every place that renders a
    fact list pays this off the same way: a line the reader can click through
    to the message behind it is a citation, while the same line without one is
    a claim.
    """
    return f"https://discord.com/channels/{fact.guild_id}/{fact.channel_id}/{fact.message_id}"


def discord_timestamp(moment: datetime) -> str:
    """A date Discord renders in each reader's own locale and timezone.

    `<t:...:d>` is resolved by the Discord client, not by Aura, which is the
    only way one posted message shows a German reader 16.08.2026 and an
    American one 8/16/2026. Formatting the date here would pick one of those
    for everybody and would need a date format per locale file on top.
    """
    return f"<t:{int(moment.timestamp())}:d>"


def fit_lines(lines: list[str], locale: str, *, max_items: int, more_items_key: str) -> str:
    """Join as many rendered lines as fit, and say how many were left out.

    Bounded by both `max_items` and `FIELD_VALUE_LIMIT`, because either one
    alone leaves a hole: ten items of 4,000 characters overflow the field, and
    a hundred one-word facts fit the field but make an unreadable wall. Room
    for the "and N more" note is reserved before the last line is accepted, so
    the note itself can never be what pushes the value over the limit.

    `more_items_key` is a translation key taking one `count` argument,
    supplied by the caller rather than hardcoded, so the digest and onboarding
    can each phrase their own overflow note (e.g. "…and {count} more.") without
    this shared module owning either feature's vocabulary.

    Returns the note alone if not even one line fits -- a case only a
    pathological fact can reach, and one where an empty field value would be
    rejected by Discord outright.
    """
    kept: list[str] = []
    used = 0
    for index, line in enumerate(lines[:max_items]):
        would_omit = len(lines) - index - 1
        reserve = (
            len(t(more_items_key, locale, count=would_omit)) + 1 if would_omit > 0 else 0
        )
        cost = len(line) + (1 if kept else 0)
        if used + cost + reserve > FIELD_VALUE_LIMIT:
            break
        kept.append(line)
        used += cost

    omitted = len(lines) - len(kept)
    if omitted > 0:
        kept.append(t(more_items_key, locale, count=omitted))
    return "\n".join(kept)
