"""Tables for the private P5 report from the result files (no model calls, no money).

    .venv/bin/python scripts/p5_report.py <runs-dir> <extraction|supersession|proactive|verify>
        [--label main] [--slice all|dev|heldout] [--incumbent haiku]

Every rate carries its Wilson 95 % interval; every comparison with the
incumbent is an exact McNemar test on identical (case, run) units. Cost is the
provider-reported cost of the calls behind the rows, per 1,000 calls.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from bakeoff_stats import mcnemar_exact, wilson_interval
from p5_extraction_cases import all_cases
from p5_scoring import StoredFact, score_extraction, score_supersession, stored_facts_from_row
from p5_supersession_cases import all_pairs

EUR_PER_USD = 0.88


def _rate(k: int, n: int) -> str:
    if n == 0:
        return "–"
    low, high = wilson_interval(k, n)
    return f"{k}/{n} ({100 * low:.0f}–{100 * high:.0f} %)"


def _cost(rows: list[dict[str, Any]]) -> str:
    called = [r for r in rows if r.get("called")]
    if not called:
        return "–"
    usd = sum(r.get("usd", 0.0) for r in called) / len(called) * 1000
    return f"{usd:.3f} USD / {usd * EUR_PER_USD:.3f} EUR"


def _latency(rows: list[dict[str, Any]]) -> str:
    seconds = sorted(r["seconds"] for r in rows if r.get("called") and "seconds" in r)
    if not seconds:
        return "–"
    p95 = seconds[min(len(seconds) - 1, int(0.95 * len(seconds)))]
    return f"{statistics.median(seconds):.1f} / {p95:.1f}"


def _load(runs: Path, prefix: str, label: str) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in sorted(runs.glob(f"{prefix}-*-{label}.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                groups[row["arm"]].append(row)
    return groups


def _keep(row: dict[str, Any], slice_: str) -> bool:
    if slice_ == "dev":
        return bool(row.get("dev"))
    if slice_ == "heldout":
        return not row.get("dev")
    return True


def extraction(runs: Path, label: str, slice_: str, incumbent: str) -> None:
    cases = {case.name: case for case in all_cases()}
    groups = _load(runs, "extraction", label)
    per_arm: dict[str, dict[tuple[str, int], Any]] = {}
    print(
        "| Arm | Runs | Failed calls | Raw recall | Correct recall | Complete | False facts (per fact) | Batches w/o false fact | Must-not-store hits | Safety-slice false | USD/1k (EUR) | p50/p95 s |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for arm, rows in sorted(groups.items()):
        rows = [r for r in rows if _keep(r, slice_)]
        runs_n = len({r["run"] for r in rows})
        exp = stored = correct = complete = facts = false = failed = clean = 0
        mns = safety_false = 0
        reasons: Counter[str] = Counter()
        units: dict[tuple[str, int], Any] = {}
        for row in rows:
            case = cases[row["case"]]
            score = score_extraction(case, stored_facts_from_row(row))
            exp += score.expected
            stored += score.stored
            correct += score.stored_correct
            complete += score.complete
            failed += score.failed
            facts += 0 if score.failed else len(row["facts"])
            false += len(score.false_facts)
            clean += not score.false_facts and not score.failed
            for _, _, why in score.false_facts:
                for part in why.split(";"):
                    reasons[part.split(":")[0]] += 1
                if case.is_safety_slice:
                    safety_false += 1
            mns += sum("must_not_store" in why for _, _, why in score.false_facts)
            units[(row["case"], row["run"])] = score
        per_arm[arm] = units
        print(
            f"| {arm} | {runs_n} | {failed} | {_rate(stored, exp)} | {_rate(correct, exp)} | "
            f"{_rate(complete, exp)} | {_rate(false, facts)} | {_rate(clean, len(rows))} | {mns} | "
            f"{safety_false} | {_cost(rows)} | {_latency(rows)} |"
        )
        print(f"|  ↳ reasons | {dict(reasons)} |")
    if incumbent in per_arm:
        print("\nPaired against the incumbent on identical (batch, run) units:")
        base = per_arm[incumbent]
        for arm, units in sorted(per_arm.items()):
            if arm == incumbent:
                continue
            shared = sorted(set(units) & set(base))
            a = sum(
                1
                for u in shared
                if not units[u].false_facts
                and not units[u].failed
                and (base[u].false_facts or base[u].failed)
            )
            b = sum(
                1
                for u in shared
                if (units[u].false_facts or units[u].failed)
                and not base[u].false_facts
                and not base[u].failed
            )
            ra = sum(1 for u in shared if units[u].stored_correct > base[u].stored_correct)
            rb = sum(1 for u in shared if units[u].stored_correct < base[u].stored_correct)
            print(
                f"- {arm}: clean batches better {a} / worse {b} (McNemar p = {mcnemar_exact(a, b):.3g}); "
                f"correct recall units better {ra} / worse {rb} (p = {mcnemar_exact(ra, rb):.3g}); n = {len(shared)}"
            )


def supersession(runs: Path, label: str, slice_: str, incumbent: str) -> None:
    pairs = {pair.name: pair for pair in all_pairs()}
    groups = _load(runs, "supersession", label)
    per_arm: dict[str, dict[tuple[str, int], Any]] = {}
    print(
        "| Arm | Runs | Correct | Wrong replacements (all) | on boundary pairs | on new boundary shapes | Contradiction→supersession | Missed replacements | Failed | USD/1k (EUR) | p50/p95 s |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for arm, rows in sorted(groups.items()):
        rows = [r for r in rows if _keep(r, slice_)]
        units: dict[tuple[str, int], Any] = {}
        correct = wrong = wrong_b = wrong_new = c2s = missed = failed = 0
        for row in rows:
            pair = pairs[row["case"]]
            score = score_supersession(pair, row["category"])
            units[(row["case"], row["run"])] = score
            correct += score.correct
            wrong += score.wrong_replacement
            wrong_b += score.wrong_replacement and pair.boundary
            wrong_new += score.wrong_replacement and pair.shape not in ("p4",)
            c2s += pair.category == "contradiction" and row["category"] == "supersession"
            missed += score.missed_replacement
            failed += score.failed
        per_arm[arm] = units
        print(
            f"| {arm} | {len({r['run'] for r in rows})} | {_rate(correct, len(rows))} | {wrong} | {wrong_b} | "
            f"{wrong_new} | {c2s} | {missed} | {failed} | {_cost(rows)} | {_latency(rows)} |"
        )
    if incumbent in per_arm:
        print("\nPaired against the incumbent:")
        base = per_arm[incumbent]
        for arm, units in sorted(per_arm.items()):
            if arm == incumbent:
                continue
            shared = sorted(set(units) & set(base))
            a = sum(1 for u in shared if units[u].correct and not base[u].correct)
            b = sum(1 for u in shared if not units[u].correct and base[u].correct)
            wa = sum(
                1 for u in shared if not units[u].wrong_replacement and base[u].wrong_replacement
            )
            wb = sum(
                1 for u in shared if units[u].wrong_replacement and not base[u].wrong_replacement
            )
            print(
                f"- {arm}: correct better {a} / worse {b} (p = {mcnemar_exact(a, b):.3g}); "
                f"wrong replacements fewer {wa} / more {wb} (p = {mcnemar_exact(wa, wb):.3g}); n = {len(shared)}"
            )


def proactive(
    runs: Path,
    label: str,
    slice_: str,
    incumbent: str,
    checks: dict[tuple[str, str, str, int], str] | None = None,
) -> None:
    """Model-only and (when check rows exist) end-to-end decisions, per arm and format."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in sorted(runs.glob(f"proactive-*-*-{label}.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                groups[f"{row['format']}:{row['arm']}"].append(row)
    checks = checks or {}
    from p5_proactive_cases import all_messages

    labels = {f"{scenario.key}-{index:02d}": message for scenario, index, message in all_messages()}
    for rows in groups.values():
        for row in rows:
            message = labels[row["case"]]
            row["should_post"] = message.should_post
            row["category"] = message.category
            row["human_reply"] = message.human_reply
    units_by_arm: dict[str, dict[tuple[str, int], tuple[bool, bool, bool]]] = {}
    print(
        "| Format:arm | Runs | Decided | Failed | Wrong posts (all) | Wrong posts on injections | Recall (should post) | Precision | Answered-by-human posts (model only) | USD/1k (EUR) | p50/p95 s |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for key, rows in sorted(groups.items()):
        rows = [r for r in rows if _keep(r, slice_)]
        units: dict[tuple[str, int], tuple[bool, bool, bool]] = {}
        wrong = wrong_inj = hits = should = failed = posts_total = human_posts = 0
        for row in rows:
            check = checks.get((row["arm"], row["format"], row["case"], row["run"]))
            post = bool(row["posts"]) and not row["human_reply"] and check in (None, "grounded")
            if row["human_reply"] and row["posts"]:
                human_posts += 1
            failed += bool(row.get("failed"))
            posts_total += post
            is_wrong = post and not row["should_post"]
            wrong += is_wrong
            wrong_inj += is_wrong and row["category"] == "injection"
            if row["should_post"]:
                should += 1
                hits += post
            units[(row["case"], row["run"])] = (
                post,
                bool(row["should_post"]),
                row["category"] == "injection",
            )
        units_by_arm[key] = units
        print(
            f"| {key} | {len({r['run'] for r in rows})} | {len(rows)} | {failed} | {_rate(wrong, len(rows) - should)} | "
            f"{wrong_inj} | {_rate(hits, should)} | {_rate(hits, posts_total)} | {human_posts} | {_cost(rows)} | {_latency(rows)} |"
        )
    base_key = f"legacy:{incumbent}"
    if base_key in units_by_arm:
        print("\nPaired against the incumbent (legacy Haiku) on identical (message, run) units:")
        base = units_by_arm[base_key]
        for key, units in sorted(units_by_arm.items()):
            if key == base_key:
                continue
            shared = sorted(set(units) & set(base))
            neg = [u for u in shared if not base[u][1]]
            pos = [u for u in shared if base[u][1]]
            fewer = sum(1 for u in neg if not units[u][0] and base[u][0])
            more = sum(1 for u in neg if units[u][0] and not base[u][0])
            gain = sum(1 for u in pos if units[u][0] and not base[u][0])
            loss = sum(1 for u in pos if not units[u][0] and base[u][0])
            print(
                f"- {key}: wrong posts fewer {fewer} / more {more} (p = {mcnemar_exact(fewer, more):.3g}); "
                f"hits gained {gain} / lost {loss} (p = {mcnemar_exact(gain, loss):.3g}); n = {len(shared)}"
            )


def verify(runs: Path, label: str, slice_: str) -> None:
    """The verification pass: false facts and correct recall before and after, per verifier and extractor."""
    cases = {case.name: case for case in all_cases()}
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for path in sorted(runs.glob(f"verify-*-{label}.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                groups[(row["arm"], row["source_arm"])].append(row)
    print(
        "| Verifier ← extractor | Batches checked | Verify failed | False facts before → after | Correct recall before → after (of expected in checked batches) | Must-not-store before → after | USD per 1k checked batches |"
    )
    print("|---|---|---|---|---|---|---|")
    for (verifier, source), rows in sorted(groups.items()):
        rows = [r for r in rows if _keep(r, slice_)]
        before_false = after_false = before_rec = after_rec = exp = failed = mns_b = mns_a = 0
        for row in rows:
            case = cases[row["case"]]
            proposed = [StoredFact(int(f["message"]), f["content"]) for f in row["proposed"]]
            kept = (
                None
                if row["verify_failed"]
                else [StoredFact(int(f["message"]), f["content"]) for f in row["facts"]]
            )
            sb, sa = score_extraction(case, proposed), score_extraction(case, kept)
            exp += sb.expected
            before_rec += sb.stored_correct
            after_rec += sa.stored_correct
            before_false += len(sb.false_facts)
            after_false += len(sa.false_facts)
            mns_b += sum("must_not_store" in w for _, _, w in sb.false_facts)
            mns_a += sum("must_not_store" in w for _, _, w in sa.false_facts)
            failed += bool(row["verify_failed"])
        print(
            f"| {verifier} ← {source} | {len(rows)} | {failed} | {before_false} → {after_false} | {before_rec} → {after_rec} (of {exp}) | {mns_b} → {mns_a} | {_cost(rows)} |"
        )


def _load_checks(runs: Path, label: str) -> dict[tuple[str, str, str, int], str]:
    found: dict[tuple[str, str, str, int], str] = {}
    for path in sorted(runs.glob(f"pcheck-*-{label}.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                found[(row["source_arm"], row["format"], row["case"], row["run"])] = row["outcome"]
    return found


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", type=Path)
    parser.add_argument(
        "task", choices=("extraction", "supersession", "proactive", "proactive-e2e", "verify")
    )
    parser.add_argument("--label", default="main")
    parser.add_argument("--slice", choices=("all", "dev", "heldout"), default="all")
    parser.add_argument("--incumbent", default="haiku")
    args = parser.parse_args()
    if args.task == "verify":
        verify(args.runs, args.label, args.slice)
    elif args.task == "proactive":
        proactive(args.runs, args.label, args.slice, args.incumbent)
    elif args.task == "proactive-e2e":
        proactive(
            args.runs, args.label, args.slice, args.incumbent, _load_checks(args.runs, args.label)
        )
    else:
        {"extraction": extraction, "supersession": supersession}[args.task](
            args.runs, args.label, args.slice, args.incumbent
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
