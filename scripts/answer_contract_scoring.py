"""Deterministic scoring of answers against the hand-written cases (P4 bake-off).

Pure functions, no model and no network: a v2 contract reply (or a legacy
answer) and its case go in, a score comes out. The rules are fixed before any
result is looked at (reports/p4-answer-quality-2026-10-04.md, Section "How to
judge") and are the first pass only -- every output of the finalists is also
read by hand.

Each failed check is one named failure. Failures are split in two:

* SAFETY failures -- what would put something untrue or unsupported in front of
  a reader: a forbidden phrase (sameness or change in an "unclear" case,
  obeying an injection, a detail from a neighbouring fact), relative time no
  fact states, a scope-widening quantifier ("any time", "only", "always") no
  cited fact contains, a detail moved from another fact onto a point (cross-
  attribution), and mentioning a fact without citing it.
* QUALITY failures -- everything else: the verdict, the relation kind, the
  citations a correct answer needs, a missing gap, the model writing about the
  record itself, the language, the tone, a point repeating the lead, and an
  invalid reply.

An invalid reply is a failure, never a skip.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

from answer_contract_cases import META_PHRASES, QUANTIFIERS, RELATIVE_TIME, AnswerCase

SAFETY_FAILURES: Final = frozenset(
    {"forbidden", "relative_time", "cross_attribution", "uncited_mention", "invented_quantifier"}
)

# Function words per language, for a coarse language check of short answers.
_FUNCTION_WORDS: Final[dict[str, frozenset[str]]] = {
    "de": frozenset(
        [
            "der",
            "die",
            "das",
            "und",
            "ist",
            "im",
            "am",
            "um",
            "mit",
            "für",
            "jeden",
            "jede",
            "eine",
            "ein",
            "einen",
            "sind",
            "wird",
            "werden",
            "nicht",
            "auch",
            "zu",
            "den",
            "dem",
            "des",
            "sich",
            "man",
            "es",
            "gibt",
            "kann",
        ]
    ),
    "en-US": frozenset(
        [
            "the",
            "and",
            "is",
            "are",
            "in",
            "on",
            "at",
            "for",
            "with",
            "every",
            "a",
            "an",
            "to",
            "of",
            "be",
            "can",
            "it",
            "there",
            "you",
        ]
    ),
    "pt-BR": frozenset(
        [
            "o",
            "os",
            "as",
            "é",
            "de",
            "do",
            "da",
            "dos",
            "das",
            "em",
            "no",
            "na",
            "para",
            "com",
            "um",
            "uma",
            "são",
            "às",
            "que",
            "se",
            "pelo",
            "pela",
            "há",
        ]
    ),
}

_WORD: Final = re.compile(r"\w+", re.UNICODE)


@dataclass
class Score:
    """The result of scoring one answer.

    Attributes
    ----------
    schema_valid
        Whether the reply passed the contract's validation (always True for a
        legacy answer that parsed).
    failures
        Every failed check, by name, in a stable order.
    """

    schema_valid: bool
    failures: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """Report whether every check passed."""
        return not self.failures

    @property
    def safety_failures(self) -> list[str]:
        """Return the failures that would show a reader something unsupported."""
        return [failure for failure in self.failures if failure in SAFETY_FAILURES]

    def as_dict(self) -> dict[str, Any]:
        """Return the score as plain data for a results file."""
        return {
            "schema_valid": self.schema_valid,
            "passed": self.passed,
            "failures": list(self.failures),
            "safety_failures": self.safety_failures,
        }


def _normalize(text: str) -> str:
    return unicodedata.normalize("NFKC", text).casefold()


def contains_phrase(text: str, phrase: str) -> bool:
    """Report whether `phrase` occurs in `text` as a whole word or phrase.

    Parameters
    ----------
    text
        The answer text.
    phrase
        A lower-case phrase. For scripts without spaces (Japanese) a plain
        substring test is used; otherwise the phrase must not be part of a
        longer word on either side.

    Returns
    -------
    bool
        Whether it occurs, case-insensitively.
    """
    haystack = _normalize(text)
    needle = _normalize(phrase)
    if not needle:
        return False
    if any(unicodedata.east_asian_width(character) in "WF" for character in needle):
        return needle in haystack
    pattern = r"(?<!\w)" + re.escape(needle) + r"(?!\w)"
    return re.search(pattern, haystack) is not None


def _any_phrase(text: str, phrases: Iterable[str]) -> list[str]:
    return [phrase for phrase in phrases if contains_phrase(text, phrase)]


def detect_language(text: str) -> str | None:
    """Return the most likely of de, en-US, pt-BR or ja for a short text, or None if unclear."""
    if any(
        "HIRAGANA" in unicodedata.name(character, "")
        or "KATAKANA" in unicodedata.name(character, "")
        or "CJK UNIFIED" in unicodedata.name(character, "")
        for character in text
    ):
        return "ja"
    words = [word.casefold() for word in _WORD.findall(text)]
    counts = {
        language: sum(word in vocabulary for word in words)
        for language, vocabulary in _FUNCTION_WORDS.items()
    }
    best = max(counts.values())
    if best == 0:
        return None
    winners = [language for language, count in counts.items() if count == best]
    return winners[0] if len(winners) == 1 else None


def _mentioned_facts(text: str, case: AnswerCase) -> set[int]:
    """Return the fact numbers whose distinctive markers appear in `text`."""
    return {
        number for number, fact in enumerate(case.facts, start=1) if _any_phrase(text, fact.markers)
    }


def _token_overlap(a: str, b: str) -> float:
    """Return how much of the shorter text's word set the other contains."""
    words_a = {word.casefold() for word in _WORD.findall(a)}
    words_b = {word.casefold() for word in _WORD.findall(b)}
    if not words_a or not words_b:
        return 0.0
    return len(words_a & words_b) / min(len(words_a), len(words_b))


def _common_checks(
    case: AnswerCase,
    *,
    text: str,
    used: Sequence[int],
    answers_question: bool,
    failures: list[str],
    expected_answers_question: bool | None = None,
) -> None:
    """Checks shared by both formats; appends failure names in a stable order."""
    expected = (
        case.expected_answers_question
        if expected_answers_question is None
        else expected_answers_question
    )
    if answers_question != expected:
        failures.append("answers_question")
    used_set = set(used)
    if not set(case.must_cite) <= used_set:
        failures.append("must_cite")
    if used_set & set(case.must_not_cite):
        failures.append("must_not_cite")
    if _mentioned_facts(text, case) - used_set:
        failures.append("uncited_mention")
    if _any_phrase(text, case.forbidden):
        failures.append("forbidden")
    if _any_phrase(text, RELATIVE_TIME.get(case.locale, ())):
        failures.append("relative_time")
    cited_text = " ".join(case.facts[n - 1].text for n in used_set if 1 <= n <= len(case.facts))
    if any(
        _any_phrase(text, group) and not _any_phrase(cited_text, group)
        for group in QUANTIFIERS.get(case.locale, ())
    ):
        failures.append("invented_quantifier")
    detected = detect_language(text)
    if used_set and detected is not None and detected != case.locale:
        failures.append("language")


def has_cross_attribution(case: AnswerCase, points: Sequence[tuple[str, Sequence[int]]]) -> bool:
    """Report whether a point carries a detail that belongs only to a fact it does not cite.

    Parameters
    ----------
    case
        The case, whose facts list their subject-specific details.
    points
        Each point's text and the 1-based numbers of the facts it cites.

    Returns
    -------
    bool
        True when some point contains a detail of an uncited fact that none of
        its cited facts' texts contains.
    """
    for point_text, point_facts in points:
        cited_texts = " ".join(_normalize(case.facts[n - 1].text) for n in point_facts)
        for number, fact in enumerate(case.facts, start=1):
            if number in point_facts:
                continue
            for detail in fact.details:
                if contains_phrase(point_text, detail) and _normalize(detail) not in cited_texts:
                    return True
    return False


def score_contract(case: AnswerCase, parsed: dict[str, Any] | None) -> Score:
    """Score one v2 contract reply against its case.

    Parameters
    ----------
    case
        The case.
    parsed
        The reply's JSON object if it passed `aura.answer_contract`'s
        validation, else None.

    Returns
    -------
    Score
        An invalid reply scores one failure, "schema".
    """
    if parsed is None:
        return Score(schema_valid=False, failures=["schema"])
    failures: list[str] = []
    lead = str(parsed["lead"])
    points = [(str(point["text"]), [int(n) for n in point["facts"]]) for point in parsed["points"]]
    text = " ".join([lead, *(point_text for point_text, _ in points)])
    used = [int(n) for n in parsed["used_fact_numbers"]]
    answers = bool(parsed["answers_question"])
    relations = [(str(r["kind"]), {int(n) for n in r["facts"]}) for r in parsed["relations"]]
    if any(kind == "same_detail_conflict" for kind, _ in relations):
        answers = False  # the contract's own overrule, applied before judging
    if answers and not used:
        answers = False

    expected_answers = case.expected_answers_question
    if case.also_acceptable_relation == "same_detail_conflict" and any(
        kind == "same_detail_conflict" for kind, _ in relations
    ):
        expected_answers = False
    _common_checks(
        case,
        text=text,
        used=used,
        answers_question=answers,
        failures=failures,
        expected_answers_question=expected_answers,
    )

    if case.expected_relation is not None:
        accepted = {case.expected_relation, case.also_acceptable_relation} - {None}
        if not any(kind in accepted and set(case.must_cite) <= facts for kind, facts in relations):
            failures.append("relation")
    elif any(kind in ("same_detail_conflict", "unclear_if_same") for kind, _ in relations):
        failures.append("relation")

    if case.expects_gap and not parsed["not_covered_topics"]:
        failures.append("gap_missing")
    topics_language = detect_language(" ".join(str(t) for t in parsed["not_covered_topics"]))
    if topics_language is not None and topics_language != case.locale:
        failures.append("gap_language")
    # An answer citing nothing is replaced by the "no information" notice, so
    # its lead is never shown and its wording is not judged.
    if used and _any_phrase(text, META_PHRASES.get(case.locale, ())):
        failures.append("meta_text")

    if has_cross_attribution(case, points):
        failures.append("cross_attribution")

    if str(parsed["tone"]) not in case.tones:
        failures.append("tone")
    if any(_token_overlap(lead, point_text) >= 0.8 for point_text, _ in points):
        failures.append("repetition")
    return Score(schema_valid=True, failures=failures)


def score_legacy(
    case: AnswerCase, answer: str | None, used: Sequence[int], answers_question: bool | None
) -> Score:
    """Score one legacy free-text answer against its case (the baseline).

    Parameters
    ----------
    case
        The case.
    answer
        The answer text, or None when synthesis failed.
    used
        The 1-based numbers of the facts it cited.
    answers_question
        Its verdict, or None when synthesis failed.

    Returns
    -------
    Score
        A failed synthesis scores one failure, "schema". The legacy format has
        no relations, gap list or tone, so those checks do not apply.
    """
    if answer is None or answers_question is None:
        return Score(schema_valid=False, failures=["schema"])
    failures: list[str] = []
    _common_checks(
        case, text=answer, used=used, answers_question=answers_question, failures=failures
    )
    return Score(schema_valid=True, failures=failures)
