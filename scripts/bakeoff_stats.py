"""Statistics for the P4 bake-off: intervals, paired tests, and blind reading.

Pure functions, no model, no network, no randomness except a seeded generator.
The rule they serve (P4 brief, "How to judge"): never write "better" without
support. Two rates whose intervals overlap are reported as "not distinguishable
at this sample size", and a comparison on identical cases is tested paired.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

_Z_95: Final = 1.959963984540054


def wilson_interval(successes: int, trials: int, z: float = _Z_95) -> tuple[float, float]:
    """Return the Wilson score interval for a binomial proportion.

    Parameters
    ----------
    successes
        Number of successes, 0 <= successes <= trials.
    trials
        Number of trials; 0 gives the uninformative interval (0, 1).
    z
        The normal quantile; the default is a two-sided 95 % interval.

    Returns
    -------
    tuple[float, float]
        Lower and upper bound, both within [0, 1].

    Raises
    ------
    ValueError
        If the counts are impossible.
    """
    if trials < 0 or successes < 0 or successes > trials:
        raise ValueError(f"impossible counts: {successes} of {trials}")
    if trials == 0:
        return 0.0, 1.0
    p = successes / trials
    denominator = 1 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denominator
    margin = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denominator
    return max(0.0, centre - margin), min(1.0, centre + margin)


def mcnemar_exact(only_a: int, only_b: int) -> float:
    """Return the two-sided exact McNemar p-value for paired binary outcomes.

    Parameters
    ----------
    only_a
        Cases where only A succeeded (discordant pairs one way).
    only_b
        Cases where only B succeeded (discordant pairs the other way).

    Returns
    -------
    float
        The exact binomial p-value under "both directions equally likely";
        1.0 when there are no discordant pairs.
    """
    if only_a < 0 or only_b < 0:
        raise ValueError("counts must be non-negative")
    n = only_a + only_b
    if n == 0:
        return 1.0
    k = min(only_a, only_b)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def paired_bootstrap_difference(
    a: Sequence[float], b: Sequence[float], *, iterations: int = 10_000, seed: int = 20261004
) -> tuple[float, float, float]:
    """Return the mean difference a - b over paired cases, with a 95 % percentile interval.

    Parameters
    ----------
    a, b
        Per-case scores of two arms on the SAME cases, in the same order.
    iterations
        Bootstrap resamples.
    seed
        Seed of the generator, so a report reproduces exactly.

    Returns
    -------
    tuple[float, float, float]
        Mean difference, lower and upper bound.

    Raises
    ------
    ValueError
        If the two sequences differ in length or are empty.
    """
    if len(a) != len(b) or not a:
        raise ValueError("paired samples must be non-empty and of equal length")
    differences = [x - y for x, y in zip(a, b, strict=True)]
    generator = random.Random(seed)
    n = len(differences)
    means = sorted(
        sum(differences[generator.randrange(n)] for _ in range(n)) / n for _ in range(iterations)
    )
    low = means[int(0.025 * iterations)]
    high = means[min(iterations - 1, int(0.975 * iterations))]
    return sum(differences) / n, low, high


def intervals_overlap(first: tuple[float, float], second: tuple[float, float]) -> bool:
    """Report whether two intervals share any point (then the rates are not distinguishable)."""
    return first[0] <= second[1] and second[0] <= first[1]


def percentile(values: Sequence[float], fraction: float) -> float:
    """Return the nearest-rank percentile of `values` (fraction in [0, 1])."""
    if not values:
        raise ValueError("no values")
    ordered = sorted(values)
    rank = max(1, math.ceil(fraction * len(ordered)))
    return ordered[rank - 1]


@dataclass(frozen=True)
class BlindItem:
    """One output prepared for a blind read.

    Attributes
    ----------
    label
        The neutral label the reader sees ("A", "B", ...).
    text
        The output.
    """

    label: str
    text: str


def blind_shuffle(
    outputs: Sequence[tuple[str, str]], *, seed: int
) -> tuple[list[BlindItem], dict[str, str]]:
    """Hide which arm wrote which output, for a blind read.

    Parameters
    ----------
    outputs
        (arm, text) pairs for one case.
    seed
        Seed for this case's shuffle, so the key can be reproduced.

    Returns
    -------
    tuple[list[BlindItem], dict[str, str]]
        The outputs in shuffled order under the labels A, B, C, ..., and the key
        from label to arm, kept apart from what the reader sees.

    Raises
    ------
    ValueError
        If there are more than 26 outputs or an arm appears twice.
    """
    arms = [arm for arm, _ in outputs]
    if len(set(arms)) != len(arms):
        raise ValueError("each arm may appear once per blind read")
    if len(outputs) > 26:
        raise ValueError("at most 26 outputs per blind read")
    order = list(range(len(outputs)))
    random.Random(seed).shuffle(order)
    items: list[BlindItem] = []
    key: dict[str, str] = {}
    for position, index in enumerate(order):
        label = chr(ord("A") + position)
        arm, text = outputs[index]
        items.append(BlindItem(label, text))
        key[label] = arm
    return items, key
