"""Shared, security-critical primitives for rendering a Fact as one public line.

Two features render lists of facts as bulleted, source-linked lines inside
embed fields -- the periodic digest (`aura.digest.formatter`) and onboarding
(`aura.onboarding.formatter`) -- and both face the same vulnerability: a fact's
own text can contain `[` or `]` that would otherwise hijack the markdown link
built around it (see reports/phase-3e.txt Section 7b for how that was found and
fixed for the digest). That fix, and the two-tier item-count/character-budget
truncation beside it, lives in exactly ONE place, so a future change to either
reaches every fact-list renderer instead of relying on each new caller to copy
it correctly.

Invariants this module maintains
--------------------------------
* A rendered line is always safe to embed as a markdown link label: escaping
  happens after truncation, so a cut can never land inside an escape sequence.
* A rendered line is never empty. An empty markdown label renders as a broken
  link, so text that collapses to nothing becomes a visible placeholder.
* A joined field value never exceeds `FIELD_VALUE_LIMIT`, including the
  overflow note, which is budgeted before the last line is accepted.

Everything here is pure and Discord-connection-free, per CLAUDE.md's testing
rule: escaping, truncation and the "and N more" note are testable without a
gateway, an embed, or a locale beyond the string handed in. Imports only
`aura.db.models` and `aura.i18n`, never a gateway or a service.
"""

from __future__ import annotations

import unicodedata
from datetime import datetime
from typing import Final

from aura.db.models import Fact
from aura.i18n import t

# Discord's own hard cap on an embed field's value, not a choice made here.
# Exceeding it is a 400 from the API, i.e. no message at all.
FIELD_VALUE_LIMIT: Final = 1024

# How much of one fact's sentence a rendered line shows. A rendered line is an
# index of what a fact says, not a replacement for reading it: the sentence is
# a link to its own source message, so anyone who wants the whole of a long
# one is one click away from the message it was distilled from.
ITEM_TEXT_LIMIT: Final = 140


def inline_fact_text(text: str) -> str:
    """Flatten one fact's sentence into a single, link-safe line.

    Parameters
    ----------
    text
        A fact's stored content, which may contain newlines, arbitrary
        Unicode, and markdown metacharacters.

    Returns
    -------
    str
        A non-empty single-line string, at most `ITEM_TEXT_LIMIT` characters
        before escaping, safe to use as a markdown link label.

    Notes
    -----
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
    """Return the Discord permalink to the message a fact was distilled from.

    Parameters
    ----------
    fact
        The fact whose origin to link. Its guild, channel and message IDs are
        the three components of a Discord permalink.

    Returns
    -------
    str
        An absolute `discord.com/channels/...` URL.

    Notes
    -----
    The knowledge model stores this reference instead of a second copy of the
    original text (CLAUDE.md's Fact component), and every place that renders a
    fact list pays that off the same way: a line the reader can click through
    to the message behind it is a citation, while the same line without one is
    a claim.
    """
    return f"https://discord.com/channels/{fact.guild_id}/{fact.channel_id}/{fact.message_id}"


def discord_timestamp(moment: datetime) -> str:
    """Render a date that Discord localises per reader.

    Parameters
    ----------
    moment
        The instant to show. Only its date part is displayed.

    Returns
    -------
    str
        A `<t:UNIX:d>` token.

    Notes
    -----
    `<t:...:d>` is resolved by the Discord client, not by Aura, which is the
    only way one posted message shows a German reader 16.08.2026 and an
    American one 8/16/2026. Formatting the date here would pick one of those
    for everybody and would need a date format per locale file on top.
    """
    return f"<t:{int(moment.timestamp())}:d>"


def fit_lines(lines: list[str], locale: str, *, max_items: int, more_items_key: str) -> str:
    """Join as many rendered lines as fit, and say how many were left out.

    Parameters
    ----------
    lines
        Already-rendered lines, in the order they should appear.
    locale
        Locale for the overflow note.
    max_items
        Hard ceiling on how many lines may be shown, whatever the budget
        allows.
    more_items_key
        Translation key for the overflow note, taking one `count` argument.
        Supplied by the caller rather than hardcoded, so the digest and
        onboarding can each phrase their own note without this shared module
        owning either feature's vocabulary.

    Returns
    -------
    str
        The kept lines joined by newlines, with the overflow note appended
        when anything was left out. At most `FIELD_VALUE_LIMIT` characters.
        The note alone if not even one line fits.

    Notes
    -----
    Bounded by both `max_items` and `FIELD_VALUE_LIMIT`, because either alone
    leaves a hole: ten items of 4,000 characters overflow the field, and a
    hundred one-word facts fit the field but make an unreadable wall.

    Room for the note is reserved before the last line is accepted, so the note
    itself can never be what pushes the value over the limit. The "note alone"
    case is reachable only by a pathological fact, and is still preferable to
    an empty field value, which Discord rejects outright.
    """
    kept: list[str] = []
    used = 0
    for index, line in enumerate(lines[:max_items]):
        would_omit = len(lines) - index - 1
        reserve = len(t(more_items_key, locale, count=would_omit)) + 1 if would_omit > 0 else 0
        cost = len(line) + (1 if kept else 0)
        if used + cost + reserve > FIELD_VALUE_LIMIT:
            break
        kept.append(line)
        used += cost

    omitted = len(lines) - len(kept)
    if omitted > 0:
        kept.append(t(more_items_key, locale, count=omitted))
    return "\n".join(kept)
