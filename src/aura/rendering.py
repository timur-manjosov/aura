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


def collapse_display_text(text: str) -> str:
    """Return `text` on one line, with invisible format characters removed.

    Parameters
    ----------
    text
        Text written by a model or a member, about to be displayed.

    Returns
    -------
    str
        One line: whitespace runs collapsed to single spaces, leading and
        trailing whitespace removed, and every Unicode format character
        (category Cf) dropped except the zero-width joiner and non-joiner.

    Notes
    -----
    Cf covers zero-width spaces, byte-order marks and the bidirectional
    overrides. An override can make a displayed sentence read differently from
    the text that was checked; a zero-width space can hide a word or make a
    blank label look non-empty. The two joiners are kept because emoji
    sequences and some scripts need them.
    """
    kept = "".join(
        character
        for character in text
        if unicodedata.category(character) != "Cf" or character in _KEPT_FORMAT_CHARACTERS
    )
    return " ".join(kept.split())


def shorten(text: str, limit: int) -> str:
    """Return `text` cut to at most `limit` characters, with an ellipsis when cut.

    Parameters
    ----------
    text
        Any text.
    limit
        The maximum length, at least 1.

    Returns
    -------
    str
        `text` unchanged when it fits; otherwise its first `limit - 1`
        characters, trailing whitespace removed, and "…".
    """
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


# The zero-width non-joiner and joiner: format characters that emoji sequences
# and several scripts need, and so the only ones collapse_display_text keeps.
_KEPT_FORMAT_CHARACTERS: Final = frozenset({"\u200c", "\u200d"})

# Every character Discord's markdown gives a meaning to inside a line: emphasis,
# strikethrough, spoilers, code, link labels, the angle brackets of mention,
# channel, emoji, timestamp and suppressed-link tokens, and block quotes.
_MARKDOWN_SPECIAL: Final = frozenset("\\*_~`|[]<>")

# What turns the START of a line into a block: a heading or subtext (#, -#), a
# list item (-, +, "1."), a block quote (>). Escaped only there -- in the middle
# of a sentence these characters are ordinary punctuation.
_LINE_START_SPECIAL: Final = frozenset("#-+")

# The two mentions that need no ID. A zero-width space after the "@" keeps them
# literal text wherever a message could still notify (a Components V2 text
# display, which pings like plain content); an embed never pings at all.
_MASS_MENTIONS: Final = ("@everyone", "@here")


def escape_display_markdown(text: str, *, at_line_start: bool = True) -> str:
    """Make one line of untrusted text display literally inside Discord markdown.

    Parameters
    ----------
    text
        One line of text written by a model or a member, already collapsed by
        `collapse_display_text` and cut to its display bound.
    at_line_start
        Whether the text may begin a line, where a heading, list or quote
        marker takes effect. False for text that always follows something on
        its line, such as a link label.

    Returns
    -------
    str
        The text with every markdown character backslash-escaped, so it renders
        exactly as written, and with "@everyone" and "@here" defused.

    Notes
    -----
    Escaping, not stripping: a fact or an answer may legitimately contain an
    asterisk or an underscore, and it should display. What must never happen is
    model text becoming markup -- a masked link whose label says one thing and
    whose target is another, a `<t:...>` timestamp or `<#id>` channel token
    naming something no fact says, a spoiler or code block hiding part of an
    answer, a heading or list breaking the card's layout, a mass mention. Bare
    URLs are left as they are (Discord links them, as it did in every legacy
    answer).

    Call it after cutting, never before: a cut after escaping can land between
    a backslash and the character it escapes.
    """
    escaped = "".join(
        "\\" + character if character in _MARKDOWN_SPECIAL else character for character in text
    )
    for mention in _MASS_MENTIONS:
        escaped = escaped.replace(mention, "@\u200b" + mention[1:])
    if not at_line_start:
        return escaped
    if escaped[:1] in _LINE_START_SPECIAL:
        escaped = "\\" + escaped
    else:
        # "1. text" at the start of a line is an ordered list item.
        digits = len(escaped) - len(escaped.lstrip("0123456789"))
        if digits and escaped[digits : digits + 1] in {".", ")"}:
            escaped = escaped[:digits] + "\\" + escaped[digits:]
    return escaped


def link_label(text: str, limit: int) -> str:
    """Return untrusted text as a one-line, escaped masked-link label.

    Parameters
    ----------
    text
        The label, e.g. a channel name. Member-controlled.
    limit
        Display bound before escaping.

    Returns
    -------
    str
        A non-empty label, at most `limit` characters before escaping; "…"
        when the text collapses to nothing.
    """
    collapsed = collapse_display_text(text)
    if not collapsed:
        return "…"
    return escape_display_markdown(shorten(collapsed, limit), at_line_start=False)


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

    A fact whose link was removed at its author's request (P7a: message ID 0)
    links to its server instead: still a working link, no longer to anyone's
    message.
    """
    if fact.message_id == 0:
        return f"https://discord.com/channels/{fact.guild_id}"
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
