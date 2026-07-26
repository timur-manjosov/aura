"""Malformed and unicode edge-case inputs, constructed rather than generated.

Phase 2a-1's attack pass hand-tested a handful of these against
`question_likeness`. This is the same idea at corpus scale and, crucially,
through the *whole* pipeline rather than one function: a zero-width-padded
question still has to fail to produce a public post, not merely fail to crash.

Constructed in code, not asked of a model, for three reasons: unicode edge
cases are exactly specifiable and a model would only approximate them; they
cost nothing; and a model cannot accidentally slip unsafe content into a string
of zero-width joiners. They are still run through the deterministic safety
layer along with everything else -- one door, no exceptions -- but they cannot
plausibly trip it.

Each case carries an English `note` describing what it is probing, so the
report can say *why* a case exists rather than printing an unreadable string
and leaving the reader to guess.
"""
from __future__ import annotations

from dataclasses import dataclass

# Discord's own per-message character limit. The "oversized" case sits exactly
# at it, and the "beyond Discord" case deliberately exceeds it -- the second one
# cannot arrive through the gateway, and is included precisely to confirm that
# the pipeline's own 2000-character classification cap (see
# question_detector._MAX_CLASSIFIED_CHARACTERS) is what bounds the work rather
# than Discord's limit being silently relied upon.
DISCORD_MESSAGE_LIMIT = 4000
BEYOND_DISCORD_LENGTH = 100_000

_ZERO_WIDTH_SPACE = "​"
_ZERO_WIDTH_NON_JOINER = "‌"
_ZERO_WIDTH_JOINER = "‍"
_BYTE_ORDER_MARK = "﻿"
_RIGHT_TO_LEFT_OVERRIDE = "‮"
_LEFT_TO_RIGHT_OVERRIDE = "‭"
_COMBINING_MARKS = "̴̧́̈͡"


@dataclass(frozen=True)
class MalformedCase:
    """One constructed edge-case input and what it is probing.

    `is_information_request` is three-valued, and it settles both of the
    questions this category is scored on -- because they are the same question.

    Half of these cases are a *genuine question* wearing zero-width joiners,
    homoglyphs or bidi controls. Stage 1 recognising one as question-like is
    correct behaviour rather than a miss, and Aura going on to answer it is
    correct behaviour too: obfuscating your own question wins you nothing but
    the answer to it. Those carry None and are reported descriptively.

    The other half contain no request at all (emoji, whitespace, punctuation)
    or are a statement. For those, "must not pass Stage 1" and "must never
    produce a post" are both real assertions, and either being violated is a
    genuine defect.
    """

    slug: str
    content: str
    note: str
    is_information_request: bool | None


_ZERO_WIDTH_CYCLE = (_ZERO_WIDTH_SPACE, _ZERO_WIDTH_NON_JOINER, _ZERO_WIDTH_JOINER)


def _zero_width_padded(text: str) -> str:
    """Interleave zero-width characters between every character of `text`.

    The obfuscation a real attacker reaches for first: visually identical to
    the original message, lexically nothing like it. If Stage 1 scored this
    very differently from the clean sentence, a keyword-era assumption would
    have survived into the semantic scorer unnoticed.

    Cycles through three different zero-width code points rather than repeating
    one, so a hypothetical future filter that strips U+200B alone would not
    make this case pass by accident.
    """
    return "".join(
        character + _ZERO_WIDTH_CYCLE[index % len(_ZERO_WIDTH_CYCLE)]
        for index, character in enumerate(text)
    )


def _zalgo(text: str) -> str:
    """Stack combining marks onto every character of `text`.

    Legal unicode, renders as noise, and multiplies the code-point count far
    beyond the grapheme count -- which is what makes it a tokenizer stress case
    rather than merely an ugly one.
    """
    return "".join(character + _COMBINING_MARKS for character in text)


def build_malformed_cases(
    *, sample_question: str, sample_statement: str, count: int
) -> list[MalformedCase]:
    """Return up to `count` edge-case inputs, several derived from real text.

    `sample_question` and `sample_statement` come from the guild's own
    generated messages, so the derived cases are in the guild's language rather
    than in English -- an obfuscated Korean question is a different test from an
    obfuscated English one, and only the first tells us anything about the nine
    locales Aura claims to support.

    The pool is ordered most-interesting-first so a smaller `count` still keeps
    the cases that probe distinct behaviour.
    """
    if count < 0:
        raise ValueError(f"count must be non-negative, got {count}")

    question = sample_question.strip() or "How do I join the next event?"
    statement = sample_statement.strip() or "The event is on Friday."

    pool = [
        MalformedCase(
            slug="zero-width-question",
            content=_zero_width_padded(question),
            note=(
                "a genuine question in this guild's language, padded with U+200B "
                "between every character -- visually identical, lexically destroyed"
            ),
            is_information_request=None,
        ),
        MalformedCase(
            slug="emoji-only",
            content="🎮🎉🔥❓🤔🙃🫠🎲",
            note="no text at all, including a question-mark emoji rather than a '?'",
            is_information_request=False,
        ),
        MalformedCase(
            slug="oversized-at-discord-limit",
            content=(question + " ") * (DISCORD_MESSAGE_LIMIT // max(len(question) + 1, 1) + 1),
            note=(
                f"the same question repeated past Discord's {DISCORD_MESSAGE_LIMIT}-character "
                "message limit"
            ),
            is_information_request=None,
        ),
        MalformedCase(
            slug="mixed-script-homoglyph",
            content=(
                # Cyrillic а/е/о and Greek ο substituted into a Latin sentence:
                # each word looks ordinary and matches nothing lexically.
                "Whеrе аrе thе rulеs fοr this sеrvеr?"
            ),
            note="Latin sentence with Cyrillic and Greek homoglyphs substituted in",
            is_information_request=None,
        ),
        MalformedCase(
            slug="bidi-override",
            content=f"{_RIGHT_TO_LEFT_OVERRIDE}{question}{_LEFT_TO_RIGHT_OVERRIDE}",
            note="wrapped in RTL/LTR override controls, which reverse rendered order",
            is_information_request=None,
        ),
        MalformedCase(
            slug="zalgo-combining-marks",
            content=_zalgo(question[:60]),
            note="every character carrying five stacked combining marks",
            is_information_request=None,
        ),
        MalformedCase(
            slug="control-characters",
            content=f"{_BYTE_ORDER_MARK}\t\r\n\x00\x01\x1b[31m{question}\x07",
            note="BOM, NUL, and ANSI escape sequences wrapped around a real question",
            is_information_request=None,
        ),
        MalformedCase(
            slug="whitespace-only",
            content="  　 \t\n",
            note="non-breaking, em and ideographic spaces only -- nothing was said",
            is_information_request=False,
        ),
        MalformedCase(
            slug="punctuation-only",
            content="???!!!???",
            note="question marks with no question, the naive-baseline worst case",
            is_information_request=False,
        ),
        MalformedCase(
            slug="zero-width-statement",
            content=_zero_width_padded(statement),
            note="the statement counterpart of the padded question, same guild, same language",
            is_information_request=False,
        ),
        MalformedCase(
            slug="beyond-discord-length",
            content=question[:20] * (BEYOND_DISCORD_LENGTH // 20),
            note=(
                f"{BEYOND_DISCORD_LENGTH} characters -- beyond anything Discord can "
                "deliver, so what is being checked is the classifier's own cap"
            ),
            is_information_request=None,
        ),
        MalformedCase(
            slug="single-character",
            content="?",
            note="the shortest input that is not empty",
            is_information_request=False,
        ),
    ]
    return pool[:count]
