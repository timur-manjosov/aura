"""Calibrate EXTRACTION_DEDUP_SIMILARITY_THRESHOLD against a hand-written pair corpus.

    .venv/bin/python scripts/calibrate_extraction_dedup_threshold.py

Free and local: embeds every pair in scripts/extraction_dedup_corpus_cases.py
through the real, shipped fastembed model (the same
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 model
aura.embeddings uses in production) and scores each pair by
aura.embeddings.cosine_similarity -- no LLM call anywhere in this path. Then
sweeps candidate thresholds with the same confusion-matrix machinery
scripts/calibrate_extraction_filter.py already uses for the Stage 1 filter,
over a ground truth of "should EXTRACTION_DEDUP_SIMILARITY_THRESHOLD mark
this pair" (DUPLICATE / SUPERSESSION / CONTRADICTION = yes,
INDEPENDENT_RELATED / UNRELATED = no) rather than fact-worthiness.

Reports the full sweep, an F1-optimum and a recall-shifted alternative (per
the calibration brief's explicit ask for both, since this threshold's cost
asymmetry is weaker than Stage 1's and does not automatically justify picking
the sharper of the two), a per-category breakdown, a per-locale breakdown,
and the two named Phase 3a-3 attack cases scored individually.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastembed import TextEmbedding  # noqa: E402

from aura.embeddings import cosine_similarity  # noqa: E402
from extraction_dedup_corpus_cases import (  # noqa: E402
    ALL_CASES,
    NAMED_ATTACK_CASE_NAMES,
    DedupCategory,
    DedupPairCase,
)
from synthetic_corpus.metrics import (  # noqa: E402
    ConfusionCounts,
    confusion_at,
    describe_distribution,
    sweep,
    threshold_range,
)

_EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

_SWEEP_START = -1.0
_SWEEP_STOP = 1.0
_SWEEP_STEP = 0.01

_LOCALES = ("en-US", "es-ES", "pt-BR", "de", "fr", "tr", "pl", "ja", "ko")


def _row(threshold: float, counts: ConfusionCounts) -> str:
    return (
        f"  {threshold:+.2f}  tp={counts.true_positive:>3} fp={counts.false_positive:>3} "
        f"tn={counts.true_negative:>3} fn={counts.false_negative:>3}  "
        f"precision={counts.precision:.3f} recall={counts.recall:.3f} "
        f"specificity={counts.specificity:.3f} f1={counts.f1:.3f}"
    )


async def _score_all(model: TextEmbedding) -> list[tuple[DedupPairCase, float]]:
    """Score every pair with two SEPARATE embed calls -- predecessor and candidate
    are never batched together, matching the real call site: aura.extraction.pipeline
    embeds one freshly distilled candidate and compares it against an already-
    embedded, already-stored active fact. Batching them here would still produce
    the same vectors (fastembed's batching does not attend across documents), but
    scoring them the way the pipeline actually does removes any doubt about that.
    """

    def _run() -> list[tuple[DedupPairCase, float]]:
        predecessors = [case.predecessor for case in ALL_CASES]
        candidates = [case.candidate for case in ALL_CASES]
        pred_vectors = list(model.embed(predecessors))
        cand_vectors = list(model.embed(candidates))
        return [
            (case, cosine_similarity(pred_vec, cand_vec))
            for case, pred_vec, cand_vec in zip(ALL_CASES, pred_vectors, cand_vectors, strict=True)
        ]

    return await asyncio.to_thread(_run)


def _print_sweep(
    label: str, scored: list[tuple[DedupPairCase, float]], thresholds: list[float]
) -> None:
    pairs = [(score, case.should_mark) for case, score in scored]
    positive_scores = [score for score, truth in pairs if truth]
    negative_scores = [score for score, truth in pairs if not truth]

    print(f"\n{label} (n={len(pairs)}, should_mark={len(positive_scores)}, should_not={len(negative_scores)})")
    print(f"  should-mark score distribution:     {describe_distribution(positive_scores)}")
    print(f"  should-not-mark score distribution: {describe_distribution(negative_scores)}")
    for threshold in thresholds:
        print(_row(threshold, confusion_at(pairs, threshold)))


def _best_f1(scored: list[tuple[DedupPairCase, float]], thresholds: list[float]) -> tuple[float, ConfusionCounts]:
    pairs = [(score, case.should_mark) for case, score in scored]
    best_threshold = thresholds[0]
    best_counts = confusion_at(pairs, best_threshold)
    best_f1 = best_counts.f1
    for threshold, counts in sweep(pairs, thresholds):
        if counts.f1 > best_f1:
            best_f1 = counts.f1
            best_threshold = threshold
            best_counts = counts
    return best_threshold, best_counts


def _best_recall_leaning(
    scored: list[tuple[DedupPairCase, float]], thresholds: list[float], *, min_specificity: float
) -> tuple[float, ConfusionCounts] | None:
    """Highest-recall threshold that still keeps specificity at or above min_specificity.

    A recall-shifted alternative to the F1 optimum, not a second F1 optimum --
    the brief asks for a threshold that trades toward recall deliberately, which
    picking a second F1-maximiser would not actually do (F1 already weighs
    precision and recall equally). Walking thresholds from high to low and
    stopping at the first one clearing the specificity floor is the direct way
    to ask "how much recall can we buy before false positives get out of hand".
    """
    pairs = [(score, case.should_mark) for case, score in scored]
    best: tuple[float, ConfusionCounts] | None = None
    for threshold, counts in sweep(pairs, sorted(thresholds, reverse=True)):
        if counts.specificity >= min_specificity:
            if best is None or counts.recall > best[1].recall:
                best = (threshold, counts)
    return best


def _print_per_category(scored: list[tuple[DedupPairCase, float]], threshold: float) -> None:
    print(f"\nPER-CATEGORY at threshold {threshold:+.3f}")
    print("-" * 78)
    for category in DedupCategory:
        subset = [(case, score) for case, score in scored if case.category is category]
        if not subset:
            continue
        marked = sum(1 for _, score in subset if score >= threshold)
        values = [score for _, score in subset]
        expect = "MARK" if category in {
            DedupCategory.DUPLICATE, DedupCategory.SUPERSESSION, DedupCategory.CONTRADICTION
        } else "HOLD BACK"
        print(
            f"  {category.value:22s} n={len(subset):3d} expect={expect:10s} "
            f"marked={marked:3d}/{len(subset):<3d} {describe_distribution(values)}"
        )


def _print_per_locale(scored: list[tuple[DedupPairCase, float]], threshold: float) -> None:
    print(f"\nPER-LOCALE (as predecessor_locale) at threshold {threshold:+.3f}")
    print("-" * 78)
    for locale in _LOCALES:
        subset = [
            (score, case.should_mark) for case, score in scored if case.predecessor_locale == locale
        ]
        if not subset:
            print(f"  {locale:<8} no data")
            continue
        counts = confusion_at(subset, threshold)
        print(f"  {locale:<8} " + _row(threshold, counts).strip())


def _print_named_attack_cases(scored: list[tuple[DedupPairCase, float]], threshold: float) -> None:
    print(f"\nNAMED PHASE 3a-3 ATTACK CASES at threshold {threshold:+.3f}")
    print("-" * 78)
    by_name = {case.name: (case, score) for case, score in scored}
    for name in NAMED_ATTACK_CASE_NAMES:
        case, score = by_name[name]
        marked = "MARKED" if score >= threshold else "held back"
        outcome = "OK (correctly held back)" if marked == "held back" else "FALSE POSITIVE"
        print(f"  {name}")
        print(f"    {case.predecessor_locale} -> {case.candidate_locale}  score={score:+.3f}  {marked}  [{outcome}]")


async def main() -> int:
    print(f"corpus: {len(ALL_CASES)} hand-written pairs, scripts/extraction_dedup_corpus_cases.py")
    model = TextEmbedding(_EMBEDDING_MODEL_NAME)
    scored = await _score_all(model)

    thresholds = threshold_range(_SWEEP_START, _SWEEP_STOP, _SWEEP_STEP)
    coarse = [t for t in thresholds if round(t * 100) % 5 == 0]

    _print_sweep("OVERALL", scored, coarse)

    # The UNRELATED bucket is trivially separable (max score 0.249 in this
    # corpus) and contributes nothing to where the threshold actually has to
    # sit -- it never competes with a should-mark score anywhere in this
    # corpus at any threshold this sweep considers. Excluding it isolates the
    # real precision/recall trade: should-mark vs. the one hard negative
    # category, independent_related, whose scores substantially overlap the
    # should-mark distribution (Section on per-category ranges below). This
    # mirrors scripts/calibrate_extraction_filter.py's own ordinary/hard split
    # for the exact same reason -- an aggregate optimum dominated by an easy
    # majority class can recommend a value that does nothing useful against
    # the case that actually matters.
    hard = [(case, score) for case, score in scored if case.should_mark or case.category is DedupCategory.INDEPENDENT_RELATED]
    _print_sweep("HARD (should-mark vs. independent_related only, UNRELATED excluded)", hard, coarse)

    f1_threshold, f1_counts = _best_f1(scored, thresholds)
    print(f"\nFULL-CORPUS F1-OPTIMUM: threshold={f1_threshold:+.3f}  {_row(f1_threshold, f1_counts).strip()}")
    hard_f1_threshold, hard_f1_counts = _best_f1(hard, thresholds)
    print(f"HARD-ONLY F1-OPTIMUM:   threshold={hard_f1_threshold:+.3f}  {_row(hard_f1_threshold, hard_f1_counts).strip()}")

    # Recall-shifted alternative, computed against the HARD set so the
    # specificity floor is measured against independent_related (the category
    # that actually costs a judgement slot when marked), not diluted by the
    # trivially-separable unrelated pairs.
    recall_leaning = _best_recall_leaning(hard, thresholds, min_specificity=0.40)
    if recall_leaning is not None:
        recall_threshold, recall_counts = recall_leaning
        print(
            f"HARD RECALL-LEANING (specificity>=0.40 vs. independent_related): "
            f"threshold={recall_threshold:+.3f}  {_row(recall_threshold, recall_counts).strip()}"
        )

    print(f"\n=== breakdowns at the CURRENT PLACEHOLDER (+0.700) for comparison ===")
    _print_per_category(scored, 0.70)
    _print_per_locale(scored, 0.70)
    _print_named_attack_cases(scored, 0.70)

    for label, threshold in (
        ("full-corpus F1-optimum", f1_threshold),
        ("hard-only F1-optimum", hard_f1_threshold),
    ):
        print(f"\n=== breakdowns at the {label} ({threshold:+.3f}) ===")
        _print_per_category(scored, threshold)
        _print_per_locale(scored, threshold)
        _print_named_attack_cases(scored, threshold)

    if recall_leaning is not None:
        print(f"\n=== breakdowns at the hard recall-leaning threshold ({recall_leaning[0]:+.3f}) ===")
        _print_per_category(scored, recall_leaning[0])
        _print_per_locale(scored, recall_leaning[0])
        _print_named_attack_cases(scored, recall_leaning[0])

    print("\nFULL PER-PAIR SCORES")
    print("-" * 78)
    for case, score in sorted(scored, key=lambda item: -item[1]):
        mark = "MARK" if case.should_mark else "hold"
        print(f"  {score:+.3f}  {mark:4s}  {case.category.value:20s}  {case.name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
