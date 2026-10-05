"""Aura's visual design system: one accent colour and one symbol per kind of message,
the hierarchy of an answer card, and the length rules every card is held to.

Constants only. Nothing here formats, sends or decides anything; the answer
card renderer (`aura.answer_card`) and the answer contract (`aura.answer_contract`)
read these values, so a colour, a symbol or a bound changes in exactly one place.

The answer card, top to bottom (P4; the v2 answer format). The digest,
onboarding, /aura-plan and command replies have card looks of their own since
P5 (aura.cards), built from the same parts -- a symbol and a small top line, a
lead, sections of lines each with a quiet source line, a quiet footer -- and
each behind its own setting, so the classic look stays until it is switched:

1. the question, small (the embed's author line, or subtext in a container),
   cut to `QUESTION_DISPLAY_CHARS`, after the kind's symbol;
2. the lead -- the direct answer, the first and most prominent text;
3. up to `MAX_POINTS` details as bullets, each ending in its citations as
   small superscript links to the source messages;
4. the "not recorded" line, rendered by code from the contract's noun phrases,
   in a subdued style (italics in an embed, subtext in a container);
5. the sources, one line per cited fact: its number, the channel as a link to
   the message, and the recording date as a Discord timestamp, which every
   reader sees in their own locale;
6. a quiet footer.

Spacing: one blank line between the lead, the bullet block and the gap line;
bullets are single-spaced. Proactive answers keep their own framing label and
show at most `PROACTIVE_MAX_POINTS` points.

Invariants this module maintains
--------------------------------
* Every message kind has exactly one accent colour and one leading symbol, so a
  kind can be told apart without relying on hue alone: a reader who cannot
  distinguish the colours still sees the symbol.
* Every accent colour has a contrast ratio of at least `MIN_ACCENT_CONTRAST`
  against each Discord background in `THEME_BACKGROUNDS`, light and dark alike
  (asserted by tests/test_theme.py, which computes WCAG 2.x contrast).
* The content bounds below are at or under Discord's own limits, with room for
  the parts code adds around the model's text (citations, separators, labels).

Imports nothing from `aura`; it sits at the bottom of the dependency graph.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Final


class MessageKind(StrEnum):
    """The kinds of message the answer card renderer produces.

    Attributes
    ----------
    ANSWER
        A synthesized /aura-ask answer.
    RELATED
        A no-model reply: nothing answers the question, possibly related facts
        are listed verbatim (or nothing at all is recorded).
    LIMIT
        A no-model reply after a daily cap: the limit note and the best facts.
    PROACTIVE
        An unprompted answer posted by proactive relief.
    ERROR
        Aura could not produce or verify an answer, or a command was refused.
    DIGEST
        The periodic digest: what changed in the knowledge model (P5).
    ONBOARDING
        The summary a new member receives (P5).
    PLAN
        /aura-plan and the one refusal every Pro-only command gives (P5).
    CONFIRM
        A command did what was asked: a fact created, a setting saved (P5).
    """

    ANSWER = "answer"
    RELATED = "related"
    LIMIT = "limit"
    PROACTIVE = "proactive"
    ERROR = "error"
    DIGEST = "digest"
    ONBOARDING = "onboarding"
    PLAN = "plan"
    CONFIRM = "confirm"


# One accent per kind, each tuned to the same luminance band (WCAG relative
# luminance 0.22-0.24) so it keeps at least 3:1 contrast -- the WCAG 2.x bar
# for a non-text graphical element -- against Discord's light theme, its embed
# background, and every dark variant at once. Hue alone separates them only for
# readers who see hue; the symbols below are the cue that does not depend on it.
ACCENT_COLORS: Final[dict[MessageKind, int]] = {
    MessageKind.ANSWER: 0x239586,  # teal
    MessageKind.RELATED: 0x74849A,  # slate
    MessageKind.LIMIT: 0xB37914,  # amber
    MessageKind.PROACTIVE: 0x6C77EF,  # light blurple, the colour proactive relief already used
    MessageKind.ERROR: 0xD85A5A,  # red
    MessageKind.DIGEST: 0x2F8ACA,  # sky blue
    MessageKind.ONBOARDING: 0xC46494,  # rose
    MessageKind.PLAN: 0xAB6CCB,  # violet
    MessageKind.CONFIRM: 0x3D9444,  # green
}

# The leading symbol of each kind. Short, widely supported emoji only, one per
# kind and no others in the chrome: a vocabulary of nine is learnable, a page of
# decoration is noise.
KIND_SYMBOLS: Final[dict[MessageKind, str]] = {
    MessageKind.ANSWER: "❓",
    MessageKind.RELATED: "🔎",
    MessageKind.LIMIT: "⏳",
    MessageKind.PROACTIVE: "💡",
    MessageKind.ERROR: "⚠️",
    MessageKind.DIGEST: "🗞️",
    MessageKind.ONBOARDING: "👋",
    MessageKind.PLAN: "⭐",
    MessageKind.CONFIRM: "✅",
}

# The Discord backgrounds an accent must stay visible on: the light theme and
# its embed fill, and the classic dark, its embed fill, the newer dark and the
# darkest ("onyx") theme.
THEME_BACKGROUNDS: Final[dict[str, int]] = {
    "light": 0xFFFFFF,
    "light_embed": 0xF2F3F5,
    "dark": 0x313338,
    "dark_embed": 0x2B2D31,
    "darker": 0x1A1A1E,
    "onyx": 0x070709,
}

MIN_ACCENT_CONTRAST: Final = 3.0

# --- The answer card's layout rules ------------------------------------------

# The question shown at the top of an answer card, so the channel sees what was
# asked. Cut with an ellipsis; the full question was already in the command.
QUESTION_DISPLAY_CHARS: Final = 200

# The answer contract's own bounds (aura.answer_contract refuses a reply over
# them rather than cutting it: a cut can drop the qualifier that made a sentence
# true). The renderer enforces them again, as a last line only.
LEAD_MAX_CHARS: Final = 300
POINT_MAX_CHARS: Final = 220
MAX_POINTS: Final = 4
PROACTIVE_MAX_POINTS: Final = 2
# A "not recorded" topic is a short noun phrase the reader sees after a code
# template ("Not recorded: start time."), never checked by a model -- so it is
# held to the shape of a label: at most this many words and characters, no
# digits, no sentence punctuation (see aura.answer_contract).
MAX_GAP_TOPICS: Final = 3
GAP_TOPIC_MAX_CHARS: Final = 60
GAP_TOPIC_MAX_WORDS: Final = 6

# The answer card's description: the lead, the points with their citations and
# the gap line. Well under Discord's 4,096 so an answer stays a card, not a page.
DESCRIPTION_MAX_CHARS: Final = 1500

# How much of a channel name a source line shows.
CHANNEL_LABEL_MAX_CHARS: Final = 40

# How many facts a source block lists at most: SYNTHESIS_FACT_LIMIT retrieved
# facts plus at most as many linked ones.
MAX_SOURCES: Final = 10

# --- The section cards (P5: digest, onboarding, /aura-plan) -------------------

# How many lines one section of a card lists before the rest is collapsed into
# a "and N more" line -- the classic digest's and onboarding's own number, so
# the new look never shows less than the old one.
MAX_SECTION_ITEMS: Final = 10

# How much of one fact a section line shows. Shorter than the classic digest's
# 200, because the new look adds a source line under every item.
SECTION_ITEM_MAX_CHARS: Final = 180

# --- Discord's own limits (not choices; exceeding one is a 400) ---------------

EMBED_AUTHOR_NAME_LIMIT: Final = 256
EMBED_DESCRIPTION_LIMIT: Final = 4096
EMBED_FIELD_NAME_LIMIT: Final = 256
EMBED_FIELD_VALUE_LIMIT: Final = 1024
EMBED_FIELD_COUNT_LIMIT: Final = 25
EMBED_FOOTER_LIMIT: Final = 2048
EMBED_TOTAL_LIMIT: Final = 6000
COMPONENTS_V2_TEXT_LIMIT: Final = 4000
COMPONENTS_V2_COMPONENT_LIMIT: Final = 40

# --- Typography ----------------------------------------------------------------

BULLET: Final = "•"
SOURCE_SEPARATOR: Final = " · "
# Between two citation markers, so the links for facts 1 and 2 read as "¹ ²",
# never as "¹²" (twelve). A thin space: visible as a gap, too narrow to read as
# a word break.
CITATION_SEPARATOR: Final = "\u2009"  # a thin space
SUPERSCRIPT_DIGITS: Final = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")
