"""Confusion counts and threshold sweeps.

Pure arithmetic over already-computed scores -- nothing here embeds, calls, or
reads a database. That separation is what lets the sweep be re-run over a
hundred candidate thresholds in milliseconds after the (slow) scoring pass has
run exactly once.

The positive class is always "Aura acts": Stage 1's positive is "this reads
like someone asking", Stage 2's is "escalate this to paid synthesis". So a
false positive is Aura speaking when it should not have, and a false negative
is Aura staying silent when it could have helped -- which is the asymmetry
every threshold decision in this project has turned on, stated once here rather
than re-derived at each call site.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ConfusionCounts:
    """One threshold's outcome, and the rates derived from it.

    Every rate returns 0.0 rather than raising when its denominator is zero.
    An undefined rate here means "this threshold produced no positives at all",
    which is a legitimate end of a sweep, not an error -- and a sweep that
    crashed at its own extreme would be useless exactly where the interesting
    behaviour is.
    """

    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int

    @property
    def total(self) -> int:
        """How many cases this row was computed over."""
        return (
            self.true_positive
            + self.false_positive
            + self.true_negative
            + self.false_negative
        )

    @property
    def precision(self) -> float:
        """Of everything let through, how much should have been."""
        predicted_positive = self.true_positive + self.false_positive
        return self.true_positive / predicted_positive if predicted_positive else 0.0

    @property
    def recall(self) -> float:
        """Of everything that should have been let through, how much was."""
        actual_positive = self.true_positive + self.false_negative
        return self.true_positive / actual_positive if actual_positive else 0.0

    @property
    def specificity(self) -> float:
        """Of everything that should have been held back, how much was."""
        actual_negative = self.true_negative + self.false_positive
        return self.true_negative / actual_negative if actual_negative else 0.0

    @property
    def accuracy(self) -> float:
        """Share of all cases decided correctly."""
        return (self.true_positive + self.true_negative) / self.total if self.total else 0.0

    @property
    def f1(self) -> float:
        """Harmonic mean of precision and recall."""
        denominator = self.precision + self.recall
        return 2 * self.precision * self.recall / denominator if denominator else 0.0


def confusion_at(scores_and_truth: list[tuple[float, bool]], threshold: float) -> ConfusionCounts:
    """Count outcomes at one threshold, using `score >= threshold` as the rule.

    `>=` and not `>`, matching `aura.proactive.gate.evaluate_message` exactly.
    A sweep that used the other comparison would disagree with the shipped gate
    at precisely the boundary values a threshold decision cares about most.
    """
    true_positive = false_positive = true_negative = false_negative = 0
    for score, is_positive in scores_and_truth:
        passed = score >= threshold
        if is_positive and passed:
            true_positive += 1
        elif is_positive:
            false_negative += 1
        elif passed:
            false_positive += 1
        else:
            true_negative += 1
    return ConfusionCounts(true_positive, false_positive, true_negative, false_negative)


def sweep(
    scores_and_truth: list[tuple[float, bool]], thresholds: list[float]
) -> list[tuple[float, ConfusionCounts]]:
    """Evaluate every candidate threshold against the same scored cases."""
    return [(threshold, confusion_at(scores_and_truth, threshold)) for threshold in thresholds]


def threshold_range(start: float, stop: float, step: float) -> list[float]:
    """Inclusive float range, rounded to avoid binary-float drift in labels.

    Rounding matters for more than looks: an unrounded 0.44999999999999996 in a
    report is the kind of thing a later reader compares against a config value
    of 0.45 and concludes does not match.
    """
    if step <= 0:
        raise ValueError(f"step must be positive, got {step}")
    digits = max(0, -int(round(_log10_of_step(step))) + 2)
    values: list[float] = []
    current = start
    while current <= stop + step / 2:
        values.append(round(current, digits))
        current += step
    return values


def _log10_of_step(step: float) -> float:
    """Order of magnitude of `step`, used to choose rounding precision."""
    from math import log10

    return log10(step)


def describe_distribution(values: list[float]) -> str:
    """Min / p25 / median / p75 / max of a score distribution, as one line.

    Quantiles rather than mean and standard deviation because these
    distributions are not normal and are routinely bimodal -- a mean sitting in
    the empty valley between two clusters would describe a value nothing in the
    data actually takes.
    """
    if not values:
        return "no data"
    ordered = sorted(values)
    last = len(ordered) - 1

    def at(fraction: float) -> float:
        return ordered[int(round(fraction * last))]

    return (
        f"n={len(ordered)}  min={ordered[0]:+.3f}  p25={at(0.25):+.3f}  "
        f"median={at(0.5):+.3f}  p75={at(0.75):+.3f}  max={ordered[-1]:+.3f}"
    )
