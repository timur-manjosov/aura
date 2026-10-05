"""Deterministic scoring for the P5 evaluation: extraction, supersession, proactive relief.

Pure functions over the cases and over what a model returned. The deterministic
pass flags; it never acquits: every flag, and every sentence a finalist stored,
is then read by hand (the private P5 report lists the confirmed ones). An
invalid reply is a failure, never a skip.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from p5_extraction_cases import ExtractionCase, group_present
from p5_supersession_cases import Pair


@dataclass(frozen=True)
class StoredFact:
    """One candidate a model staged: the 1-based message it names and its sentence."""

    message: int
    content: str


@dataclass
class ExtractionScore:
    """What one run of one batch did.

    Attributes
    ----------
    expected
        Must-store messages in the batch (one per Expected entry's message).
    stored
        Of those, how many have at least one stored fact.
    stored_correct
        Of those, how many have at least one stored fact that is not false by
        the labels (the recall that counts: a false fact is not a recall).
    complete
        Of those, how many have a stored fact with every detail group.
    false_facts
        (message, sentence, reason) per stored fact that is false by the labels.
    optional_stored
        Stored facts from optional messages (reported, not counted).
    failed
        The call returned nothing usable.
    """

    expected: int = 0
    stored: int = 0
    stored_correct: int = 0
    complete: int = 0
    false_facts: list[tuple[int, str, str]] = field(default_factory=list)
    optional_stored: list[tuple[int, str]] = field(default_factory=list)
    failed: bool = False


def score_extraction(case: ExtractionCase, facts: Sequence[StoredFact] | None) -> ExtractionScore:
    """Score one run of one batch against its labels.

    Parameters
    ----------
    case
        The labelled batch.
    facts
        What the model staged; None when the call failed.

    Returns
    -------
    ExtractionScore
        Recall and completeness per must-store message, and every stored fact
        that the labels make false: from a must-not-store message, with a
        condition group missing, or with a forbidden substring.
    """
    expected_messages = sorted({e.message for e in case.expected})
    score = ExtractionScore(expected=len(expected_messages))
    if facts is None:
        score.failed = True
        return score
    by_message: dict[int, list[str]] = {}
    for fact in facts:
        by_message.setdefault(fact.message, []).append(fact.content)

    for message in expected_messages:
        stored = by_message.get(message, [])
        if stored:
            score.stored += 1
        specs = [e for e in case.expected if e.message == message]
        if stored and all(
            any(all(group_present(text, group) for group in spec.details) for text in stored)
            for spec in specs
        ):
            score.complete += 1

    # A message may legitimately be split into several facts (a long
    # announcement with three dates becomes three sentences): a condition is
    # missing only when NO fact from that message carries it, and the flag is
    # then put on the message's first fact.
    union = {message: " \n ".join(texts) for message, texts in by_message.items()}
    first_fact_of = {message: texts[0] for message, texts in by_message.items()}
    for fact in facts:
        reasons: list[str] = []
        if fact.message in case.must_not_store:
            reasons.append("must_not_store")
        for spec in (e for e in case.expected if e.message == fact.message):
            missing = [
                group for group in spec.conditions if not group_present(union[fact.message], group)
            ]
            if missing and fact.content == first_fact_of[fact.message]:
                reasons.append(f"condition_missing:{missing[0][0]}")
            hits = [needle for needle in spec.forbidden if group_present(fact.content, (needle,))]
            if hits:
                reasons.append(f"forbidden:{hits[0]}")
        batch_hits = [needle for needle in case.forbidden if group_present(fact.content, (needle,))]
        if batch_hits:
            reasons.append(f"batch_forbidden:{batch_hits[0]}")
        if fact.message in case.optional and not reasons:
            score.optional_stored.append((fact.message, fact.content))
        if reasons:
            score.false_facts.append((fact.message, fact.content, ";".join(reasons)))
    false_texts = {(message, text) for message, text, _ in score.false_facts}
    score.stored_correct = sum(
        1
        for message in expected_messages
        if any((message, text) not in false_texts for text in by_message.get(message, []))
    )
    return score


def stored_facts_from_row(row: dict[str, Any]) -> list[StoredFact] | None:
    """Return the stored facts of one result line, or None for a failed call."""
    if row.get("call_failed"):
        return None
    return [StoredFact(int(fact["message"]), str(fact["content"])) for fact in row["facts"]]


@dataclass(frozen=True)
class SupersessionScore:
    """One judgement against its label.

    Attributes
    ----------
    correct
        The category equals the label.
    wrong_replacement
        "supersession" where the label is not: the dangerous direction.
    missed_replacement
        The label is "supersession" and the judge said something else.
    failed
        No usable judgement.
    """

    correct: bool
    wrong_replacement: bool
    missed_replacement: bool
    failed: bool


def score_supersession(pair: Pair, category: str | None) -> SupersessionScore:
    """Score one judgement.

    Parameters
    ----------
    pair
        The labelled pair.
    category
        The judged category, or None when the call failed.

    Returns
    -------
    SupersessionScore
        A failure is incorrect and neither kind of replacement error.
    """
    if category is None:
        return SupersessionScore(False, False, pair.category == "supersession", True)
    return SupersessionScore(
        correct=category == pair.category,
        wrong_replacement=category == "supersession" and pair.category != "supersession",
        missed_replacement=pair.category == "supersession" and category != "supersession",
        failed=False,
    )


def proactive_posts(decision: bool, *, human_reply: bool, check_passed: bool | None) -> bool:
    """Return whether production would post, given the model's decision.

    Parameters
    ----------
    decision
        The format's own decision (legacy: answers_question and a citation;
        v2: answers_unprompted).
    human_reply
        A human answered within the grace period: production stands down
        before the model is ever called.
    check_passed
        The answer check's verdict, or None when the check was not run (the
        model-only figure).

    Returns
    -------
    bool
        True only when every gate in production's order would let it through.
    """
    if human_reply or not decision:
        return False
    return check_passed is not False
