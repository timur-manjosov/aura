"""Free summaries of P4 bake-off results: per-task tables with intervals and paired tests.

    .venv/bin/python scripts/p4_bakeoff_report.py <task> <results.jsonl ...> [--incumbent ARM]

No model calls, no money. Reads the JSONL files scripts/p4_bakeoff.py writes and
prints one table per task: every arm's quality and safety rates with Wilson 95 %
intervals, validity, latency p50/p95, cost per 1000 calls, and -- against the
named incumbent, on the cases both ran -- an exact McNemar p-value on the
safety metric. Rates whose intervals overlap the incumbent's are marked "~"
("not distinguishable at this sample size").
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Final

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bakeoff_stats import intervals_overlap, mcnemar_exact, percentile, wilson_interval

Row = dict[str, Any]

# The EUR value of one USD for the cost columns; source and date in the report.
EUR_PER_USD: Final = 0.88


def load(paths: Sequence[Path]) -> dict[str, list[Row]]:
    """Return the result rows of every file, grouped by arm."""
    by_arm: dict[str, list[Row]] = defaultdict(list)
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                by_arm[row["arm"]].append(row)
    return dict(by_arm)


def rate(
    rows: Sequence[Row], predicate: Callable[[Row], bool]
) -> tuple[int, int, tuple[float, float]]:
    """Return successes, trials and the Wilson interval of `predicate` over `rows`."""
    successes = sum(1 for row in rows if predicate(row))
    return successes, len(rows), wilson_interval(successes, len(rows))


def fmt_rate(successes: int, trials: int, interval: tuple[float, float], mark: str = "") -> str:
    """Format a rate as 'k/n (lo-hi%)'."""
    return f"{successes}/{trials} ({interval[0] * 100:.0f}-{interval[1] * 100:.0f}%){mark}"


def cost_latency(rows: Sequence[Row]) -> str:
    """Return validity-independent cost and latency columns for a group of rows."""
    called = [row for row in rows if row.get("called")]
    if not called:
        return "no calls"
    usd = [float(row.get("usd") or 0) for row in called]
    seconds = [float(row.get("seconds") or 0) for row in called]
    reasoning = [int(row.get("reasoning_tokens") or 0) for row in called]
    providers = Counter(str(row.get("provider")) for row in called)
    per_thousand = statistics.mean(usd) * 1000
    return (
        f"${per_thousand:.3f}/1k (EUR {per_thousand * EUR_PER_USD:.3f}) · "
        f"p50 {statistics.median(seconds):.2f}s p95 {percentile(seconds, 0.95):.2f}s · "
        f"reasoning tok mean {statistics.mean(reasoning):.0f} · providers {dict(providers)}"
    )


def paired(
    incumbent: Sequence[Row], other: Sequence[Row], ok: Callable[[Row], bool]
) -> tuple[int, int, float]:
    """Return discordant counts and the exact McNemar p on cases both arms ran."""
    key = lambda row: (row["case"], row.get("run", 1))  # noqa: E731
    base = {key(row): ok(row) for row in incumbent}
    only_incumbent = only_other = 0
    for row in other:
        if key(row) not in base:
            continue
        if base[key(row)] and not ok(row):
            only_incumbent += 1
        elif ok(row) and not base[key(row)]:
            only_other += 1
    return only_incumbent, only_other, mcnemar_exact(only_incumbent, only_other)


def _table(
    groups: dict[str, list[Row]],
    incumbent: str | None,
    metrics: list[tuple[str, Callable[[Row], bool], Callable[[Row], bool]]],
    safety_metric: str,
) -> None:
    """Print one row per arm: each metric (with filter) as a rate, then cost and latency."""
    reference: dict[str, tuple[float, float]] = {}
    if incumbent in groups:
        for name, keep, ok in metrics:
            rows = [row for row in groups[incumbent] if keep(row)]
            reference[name] = rate(rows, ok)[2]
    for arm in sorted(groups):
        rows = groups[arm]
        cells = []
        for name, keep, ok in metrics:
            kept = [row for row in rows if keep(row)]
            successes, trials, interval = rate(kept, ok)
            mark = ""
            if arm != incumbent and name in reference:
                mark = " ~" if intervals_overlap(interval, reference[name]) else " !"
            cells.append(f"{name} {fmt_rate(successes, trials, interval, mark)}")
        print(f"{arm:<18} " + " | ".join(cells))
        print(f"{'':<18} {cost_latency(rows)}")
        if incumbent in groups and arm != incumbent:
            keep, ok = next((k, o) for n, k, o in metrics if n == safety_metric)
            a, b, p = paired(
                [r for r in groups[incumbent] if keep(r)], [r for r in rows if keep(r)], ok
            )
            print(
                f"{'':<18} vs {incumbent} on {safety_metric}: only incumbent {a}, only {arm} {b}, McNemar p={p:.3f}"
            )


def report_synth(groups: dict[str, list[Row]], incumbent: str | None) -> None:
    """Print the A table: validity, deterministic pass, safety, by difficulty."""

    def valid(r: Row) -> bool:
        return bool(r["score"]["schema_valid"])

    def passed(r: Row) -> bool:
        return bool(r["score"]["passed"])

    def safe(r: Row) -> bool:
        return valid(r) and not r["score"]["safety_failures"]

    metrics = [
        ("valid", lambda r: True, valid),
        ("pass", lambda r: True, passed),
        ("safe", lambda r: True, safe),
        ("hard-pass", lambda r: r["difficulty"] == "hard", passed),
    ]
    _table(groups, incumbent, metrics, "safe")
    print()
    for arm in sorted(groups):
        failures = Counter(f for r in groups[arm] for f in r["score"]["failures"])
        unstable = _unstable(groups[arm], passed)
        print(f"{arm:<18} failures {dict(failures)} · unstable cases {len(unstable)}")


def _unstable(rows: Sequence[Row], ok: Callable[[Row], bool]) -> list[str]:
    seen: dict[str, set[bool]] = defaultdict(set)
    for row in rows:
        seen[row["case"]].add(ok(row))
    return sorted(case for case, outcomes in seen.items() if len(outcomes) > 1)


def report_checker(groups: dict[str, list[Row]], incumbent: str | None) -> None:
    """Print the B table: controls passed, forgeries refused, per class."""

    def refused(r: Row) -> bool:
        return r["outcome"] != "grounded"

    metrics = [
        ("controls-passed", lambda r: r["kind"] == "control", lambda r: r["outcome"] == "grounded"),
        ("forged-refused", lambda r: r["kind"] == "forged", refused),
        ("check-failed", lambda r: True, lambda r: r["outcome"] == "check_failed"),
    ]
    _table(groups, incumbent, metrics, "forged-refused")
    print()
    for arm in sorted(groups):
        per_class: dict[str, list[bool]] = defaultdict(list)
        for row in groups[arm]:
            if row["kind"] == "forged":
                per_class[row["finding"]].append(refused(row))
        print(
            f"{arm:<18} "
            + ", ".join(f"{name} {sum(v)}/{len(v)}" for name, v in sorted(per_class.items()))
        )


def report_proactive(groups: dict[str, list[Row]], incumbent: str | None) -> None:
    """Print the C table: precision, recall, wrong posts, by category."""
    metrics = [
        ("no-wrong-post", lambda r: not r["should_post"], lambda r: not r["posts"]),
        ("recall", lambda r: r["should_post"], lambda r: bool(r["posts"])),
        ("precision", lambda r: bool(r["posts"]), lambda r: bool(r["should_post"])),
        ("failed", lambda r: True, lambda r: bool(r["failed"])),
    ]
    _table(groups, incumbent, metrics, "no-wrong-post")
    print()
    for arm in sorted(groups):
        wrong = Counter(r["category"] for r in groups[arm] if r["posts"] and not r["should_post"])
        missed = Counter(r["category"] for r in groups[arm] if r["should_post"] and not r["posts"])
        print(f"{arm:<18} wrong posts by category {dict(wrong)} · missed {dict(missed)}")


def _extraction_messages(rows: Sequence[Row]) -> list[Row]:
    flat = []
    for row in rows:
        for message in row["messages"]:
            flat.append(
                {
                    **message,
                    "case": f"{row['case']}#{message['index']}",
                    "run": row["run"],
                    "call_failed": row["call_failed"],
                }
            )
    return flat


def report_extraction(groups: dict[str, list[Row]], incumbent: str | None) -> None:
    """Print the D table: recall, false extractions, failed calls, forbidden hits."""
    flat = {arm: _extraction_messages(rows) for arm, rows in groups.items()}
    metrics = [
        ("recall", lambda r: r["expect_fact"], lambda r: bool(r["extracted"])),
        ("no-false-fact", lambda r: not r["expect_fact"], lambda r: not r["extracted"]),
        ("call-ok", lambda r: True, lambda r: not r["call_failed"]),
    ]
    _table(flat, incumbent, metrics, "no-false-fact")
    for arm in sorted(groups):
        hits = sum(len(row["forbidden_hits"]) for row in groups[arm])
        print(f"{arm:<18} forbidden-substring hits {hits} · {cost_latency(groups[arm])}")


def report_supersession(groups: dict[str, list[Row]], incumbent: str | None) -> None:
    """Print the E table: accuracy overall, on boundary pairs, and failures."""
    metrics = [
        ("correct", lambda r: True, lambda r: bool(r["correct"])),
        ("boundary", lambda r: bool(r["boundary"]), lambda r: bool(r["correct"])),
        ("judged", lambda r: True, lambda r: r["category"] is not None),
    ]
    _table(groups, incumbent, metrics, "correct")
    for arm in sorted(groups):
        confusion = Counter((r["expected"], r["category"]) for r in groups[arm] if not r["correct"])
        print(f"{arm:<18} errors {dict(confusion)}")


def report_throughput(groups: dict[str, list[Row]], incumbent: str | None) -> None:
    """Print validity, error kinds and latency under parallel load."""
    for arm, rows in sorted(groups.items()):
        errors = Counter(str(r.get("finish_reason")) for r in rows if not r.get("valid"))
        wall = [float(r["wall_seconds"]) for r in rows]
        print(
            f"{arm:<18} valid {sum(r['valid'] for r in rows)}/{len(rows)} · not valid by finish "
            f"{dict(errors)} · wall p50 {statistics.median(wall):.2f}s p95 {percentile(wall, 0.95):.2f}s "
            f"max {max(wall):.2f}s"
        )


def report_earlycheck(groups: dict[str, list[Row]], incumbent: str | None) -> None:
    """Print today's check over v2 answers, per source arm and shape."""
    rows = [row for arm_rows in groups.values() for row in arm_rows]
    by_source: dict[str, Counter[str]] = defaultdict(Counter)
    by_shape: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        by_source[row["source_arm"]][row["outcome"]] += 1
        by_shape[row["shape"]][row["outcome"]] += 1
    for source, counts in sorted(by_source.items()):
        print(f"{source:<18} {dict(counts)}")
    for shape, counts in sorted(by_shape.items()):
        print(f"  {shape:<20} {dict(counts)}")


REPORTS: Final[dict[str, Callable[[dict[str, list[Row]], str | None], None]]] = {
    "synth": report_synth,
    "legacy": report_synth,
    "checker": report_checker,
    "proactive": report_proactive,
    "extraction": report_extraction,
    "supersession": report_supersession,
    "throughput": report_throughput,
    "earlycheck": report_earlycheck,
}


def main() -> int:
    """Print the summary for one task; see the module docstring."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=sorted(REPORTS))
    parser.add_argument("paths", type=Path, nargs="+")
    parser.add_argument("--incumbent", default=None)
    args = parser.parse_args()
    REPORTS[args.task](load(args.paths), args.incumbent)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
