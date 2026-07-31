"""Calibrate extraction_fact_worthiness_threshold against the Phase 3a-1 corpus.

    .venv/bin/python scripts/calibrate_extraction_filter.py

Free and local: scores every corpus message through the real, shipped
aura.extraction.fact_worthiness detector (fastembed inference only, no LLM
call), then sweeps candidate thresholds with the same confusion-matrix
machinery scripts/synthetic_corpus.metrics already provides Phase 2b's own
calibration. No network access needed once the corpus file exists.

Reports four views of the same sweep, not just an aggregate: overall,
ordinary-negatives-only, hard-negatives-only (the two adversarial near-miss
categories), and per-locale -- because a threshold that looks good in
aggregate can still be hiding a category or a locale where it fails, and the
Phase 3a-1 design brief explicitly asks for that to be reported honestly
rather than averaged away.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastembed import TextEmbedding  # noqa: E402

from aura.extraction.fact_worthiness import create_fact_worthiness_detector  # noqa: E402
from extraction_corpus.corpus_model import (  # noqa: E402
    HARD_NEGATIVE_CATEGORIES,
    LabelAudit,
    ORDINARY_NOT_FACT_WORTHY_CATEGORIES,
    SyntheticMessage,
)
from extraction_corpus.corpus_store import CorpusLoadError, read_corpus  # noqa: E402
from extraction_corpus.scenarios import LOCALES  # noqa: E402
from synthetic_corpus.metrics import (  # noqa: E402
    ConfusionCounts,
    confusion_at,
    describe_distribution,
    sweep,
    threshold_range,
)

DEFAULT_CORPUS_PATH = Path("reports") / "extraction-corpus" / "corpus.json"

# Wide enough to see the sweep's full shape (the useful range in practice sits
# well inside [-1, 1], per the fact_worthiness.py sample and the
# proactive_question_threshold precedent), fine-grained enough that a chosen
# value is not an artefact of a coarse grid.
_SWEEP_START = -1.0
_SWEEP_STOP = 1.0
_SWEEP_STEP = 0.02


def _row(threshold: float, counts: ConfusionCounts) -> str:
    return (
        f"  {threshold:+.2f}  tp={counts.true_positive:>4} fp={counts.false_positive:>4} "
        f"tn={counts.true_negative:>4} fn={counts.false_negative:>4}  "
        f"precision={counts.precision:.3f} recall={counts.recall:.3f} "
        f"specificity={counts.specificity:.3f} f1={counts.f1:.3f} acc={counts.accuracy:.3f}"
    )


async def _score_all(
    detector, messages: list[SyntheticMessage]
) -> list[tuple[SyntheticMessage, float]]:
    scored = []
    for message in messages:
        score = await detector.question_likeness(message.content)
        scored.append((message, score))
    return scored


def _print_sweep(
    label: str, scored: list[tuple[SyntheticMessage, float]], thresholds: list[float]
) -> None:
    pairs = [(score, message.is_fact_worthy) for message, score in scored]
    positive_scores = [score for score, truth in pairs if truth]
    negative_scores = [score for score, truth in pairs if not truth]

    print(f"\n{label} (n={len(pairs)}, positive={len(positive_scores)}, negative={len(negative_scores)})")
    print(f"  positive score distribution: {describe_distribution(positive_scores)}")
    print(f"  negative score distribution: {describe_distribution(negative_scores)}")
    for threshold in thresholds:
        print(_row(threshold, confusion_at(pairs, threshold)))


def _print_per_locale(scored: list[tuple[SyntheticMessage, float]], threshold: float) -> None:
    print(f"\nPER-LOCALE at threshold {threshold:+.2f}")
    print("-" * 78)
    for locale in LOCALES:
        pairs = [(score, message.is_fact_worthy) for message, score in scored if message.locale == locale]
        if not pairs:
            print(f"  {locale:<8} no data")
            continue
        counts = confusion_at(pairs, threshold)
        print(f"  {locale:<8} " + _row(threshold, counts).strip())


def _recommend(scored: list[tuple[SyntheticMessage, float]], thresholds: list[float]) -> float:
    """Pick the threshold maximising F1 over the whole corpus, as a starting point.

    Not the final word -- reports/phase-3a-1.txt records the reasoning for
    whatever value actually ships, the same way config.py's existing
    thresholds document a human judgment call layered on top of the raw
    sweep (see proactive_question_threshold's own comment for the precedent:
    the accuracy/F1 optimum is a data point, not an instruction).
    """
    pairs = [(score, message.is_fact_worthy) for message, score in scored]
    best_threshold = thresholds[0]
    best_f1 = -1.0
    for threshold, counts in sweep(pairs, thresholds):
        if counts.f1 > best_f1:
            best_f1 = counts.f1
            best_threshold = threshold
    return best_threshold


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS_PATH)
    parser.add_argument("--exclude-disputed", action="store_true")
    args = parser.parse_args()

    try:
        corpus = read_corpus(args.corpus)
    except CorpusLoadError as exc:
        print(f"{exc}", file=sys.stderr)
        return 1

    messages = corpus.messages
    if args.exclude_disputed:
        messages = [m for m in messages if m.label_audit is not LabelAudit.DISPUTE]

    print(f"corpus: {args.corpus} ({len(corpus.messages)} messages, generated {corpus.generated_at})")
    audited = [m for m in corpus.messages if m.label_audit is not LabelAudit.NOT_AUDITED]
    disputes = [m for m in corpus.messages if m.label_audit is LabelAudit.DISPUTE]
    if audited:
        print(f"label audit: {len(disputes)}/{len(audited)} disputed ({len(disputes) / len(audited):.1%})")

    model = TextEmbedding("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    detector = await create_fact_worthiness_detector(model)
    scored = await _score_all(detector, messages)

    thresholds = threshold_range(_SWEEP_START, _SWEEP_STOP, _SWEEP_STEP)
    coarse = [t for t in thresholds if round(t * 20) % 5 == 0]  # every 0.25 for the console

    _print_sweep("OVERALL", scored, coarse)

    ordinary = [(m, s) for m, s in scored if m.category in ORDINARY_NOT_FACT_WORTHY_CATEGORIES or m.is_fact_worthy]
    _print_sweep("FACT-WORTHY vs. ORDINARY NEGATIVES ONLY", ordinary, coarse)

    hard = [(m, s) for m, s in scored if m.category in HARD_NEGATIVE_CATEGORIES or m.is_fact_worthy]
    _print_sweep("FACT-WORTHY vs. HARD NEGATIVES ONLY (hedged speculation + adversarial noise)", hard, coarse)

    for category in HARD_NEGATIVE_CATEGORIES:
        subset = [(m, s) for m, s in scored if m.category is category]
        if subset:
            values = [s for _, s in subset]
            print(f"\n  {category.value} score distribution: {describe_distribution(values)}")

    recommended = _recommend(scored, thresholds)
    print(f"\nF1-maximising threshold over the whole corpus: {recommended:+.3f}")
    _print_per_locale(scored, recommended)

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
