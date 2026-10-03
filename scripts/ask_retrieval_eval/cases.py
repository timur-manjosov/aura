"""Labelled questions and the metrics /aura-ask's retrieval is judged by.

The labels and metrics are the quality diagnosis' (2026-10-02, Section 4.1):

* a **positive** question names the facts a good answer should consider;
* a **negative** question should select nothing -- anything it selects is a
  paid call with nothing true to say;
* an **adjacent** question shares a word or topic with some facts (its
  neighbours) but asks about something else.

Imports nothing from aura.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

CaseKind = Literal["positive", "negative", "adjacent"]


@dataclass(frozen=True, slots=True)
class EvalCase:
    """One labelled question.

    Attributes
    ----------
    case_id
        Unique within its set.
    query
        The question, as a member would type it.
    kind
        positive, negative or adjacent.
    register
        How it is phrased (keyword, typo, colloquial, ...), for breakdowns.
    relevant
        Fact IDs a good answer should consider; empty unless positive.
    neighbors
        For an adjacent question, the facts that share its words.
    """

    case_id: str
    query: str
    kind: CaseKind
    register: str
    relevant: frozenset[int]
    neighbors: frozenset[int] = frozenset()


@dataclass(frozen=True, slots=True)
class InventedFact:
    """One invented fact of an evaluation set.

    Attributes
    ----------
    fact_id
        Its ID within the set.
    content
        Its sentence.
    """

    fact_id: int
    content: str


@dataclass(frozen=True, slots=True)
class CaseFile:
    """A set of invented facts and the questions labelled against them.

    Attributes
    ----------
    facts
        The facts; empty for a case file meant for a real database copy.
    cases
        The questions.
    """

    facts: tuple[InventedFact, ...]
    cases: tuple[EvalCase, ...]


def load_case_file(path: Path) -> CaseFile:
    """Read a case file in the quality diagnosis' JSON shape.

    Parameters
    ----------
    path
        JSON with a "cases" list and, optionally, a "facts" list of
        {"id", "content"}.

    Returns
    -------
    CaseFile
        Facts and cases, in file order.
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    return CaseFile(
        facts=tuple(
            InventedFact(fact_id=int(item["id"]), content=str(item["content"]))
            for item in raw.get("facts", [])
        ),
        cases=tuple(
            EvalCase(
                case_id=str(item["id"]),
                query=str(item["query"]),
                kind=item["kind"],
                register=str(item.get("register", "")),
                relevant=frozenset(int(fact_id) for fact_id in item.get("relevant", [])),
                neighbors=frozenset(int(fact_id) for fact_id in item.get("neighbors", [])),
            )
            for item in raw["cases"]
        ),
    )


@dataclass(frozen=True, slots=True)
class Metrics:
    """How well one selection strategy did over one set of questions.

    Attributes
    ----------
    positives
        Positive questions.
    hits
        Positive questions that selected at least one relevant fact.
    full
        Positive questions that selected every relevant fact.
    precision
        Mean share of selected facts that are relevant, over positive
        questions that selected anything (1.0 if none did).
    irrelevant_per_positive
        Mean number of irrelevant facts selected per positive question.
    negatives
        Negative questions.
    negative_false_hits
        Negative questions that selected anything.
    adjacent
        Adjacent questions.
    adjacent_other
        Adjacent questions that selected a fact that is not even a neighbour.
    paid_calls
        Questions of any kind that selected anything, so would reach synthesis.
    """

    positives: int
    hits: int
    full: int
    precision: float
    irrelevant_per_positive: float
    negatives: int
    negative_false_hits: int
    adjacent: int
    adjacent_other: int
    paid_calls: int


def compute_metrics(cases: Sequence[EvalCase], selected: Mapping[str, Sequence[int]]) -> Metrics:
    """Aggregate one strategy's selections.

    Parameters
    ----------
    cases
        The questions to aggregate over.
    selected
        Selected fact IDs per case ID; every case must be present.

    Returns
    -------
    Metrics
        The aggregate.
    """
    precisions: list[float] = []
    irrelevant: list[int] = []
    hits = full = 0
    positives = [case for case in cases if case.kind == "positive"]
    for case in positives:
        chosen = set(selected[case.case_id])
        found = chosen & case.relevant
        hits += bool(found)
        full += found == case.relevant
        irrelevant.append(len(chosen - case.relevant))
        if chosen:
            precisions.append(len(found) / len(chosen))
    negatives = [case for case in cases if case.kind == "negative"]
    adjacent = [case for case in cases if case.kind == "adjacent"]
    return Metrics(
        positives=len(positives),
        hits=hits,
        full=full,
        precision=sum(precisions) / len(precisions) if precisions else 1.0,
        irrelevant_per_positive=sum(irrelevant) / len(irrelevant) if irrelevant else 0.0,
        negatives=len(negatives),
        negative_false_hits=sum(bool(selected[case.case_id]) for case in negatives),
        adjacent=len(adjacent),
        adjacent_other=sum(bool(set(selected[case.case_id]) - case.neighbors) for case in adjacent),
        paid_calls=sum(bool(selected[case.case_id]) for case in cases),
    )


def format_metrics(name: str, metrics: Metrics) -> str:
    """Render one strategy's metrics as one fixed-width line.

    Parameters
    ----------
    name
        The strategy's label.
    metrics
        Its metrics.

    Returns
    -------
    str
        The line.
    """
    return (
        f"{name:28} hit {metrics.hits:4}/{metrics.positives:<4} "
        f"full {metrics.full:4}/{metrics.positives:<4} prec {metrics.precision:5.3f} "
        f"irr/q {metrics.irrelevant_per_positive:4.2f} | "
        f"negFP {metrics.negative_false_hits:3}/{metrics.negatives:<3} | "
        f"adj-other {metrics.adjacent_other}/{metrics.adjacent} | calls {metrics.paid_calls}"
    )
