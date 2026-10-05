"""The P5c tables and bars, from the result files of scripts/p5c_supersession.py (free, local).

    .venv/bin/python scripts/p5c_analysis.py --runs-dir <dir> --p5-runs-dir <dir> [--without NAME]

Reads every arm's results, takes each judgement's label from the CURRENT case
set (scripts/p5c_supersession_cases.py, so a documented relabel applies to
every arm alike), and prints the family tables and the bars of the written bar
file (reports/p5c-cleanup-*/00-bars-policy.md) on the held-out slice, plus the
dev slice and every disagreement for the manual read. `--without` drops
judgements by name, for the "every bar also without the relabelled pair" check.

`haiku-old` on the 125 P5 judgements is P5's three runs (same prompt
fingerprint, same code); its fresh run 4 is reported as the drift check only.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Final

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bakeoff_stats import intervals_overlap, mcnemar_exact, wilson_interval
from p5c_supersession_cases import Judgement, all_judgements

SUPERSESSION: Final = "supersession"
FAMILIES: Final = (
    "temporary",
    "temporary-doubt",
    "lasting",
    "temporal-control",
    "temporal-injection",
    "p5",
)
CANDIDATES: Final = ("deepseek-2p-think", "deepseek-2p", "gemini38-vertex", "gpt6luna")
NEW_SHAPES: Final = frozenset(
    {
        "temporary",
        "temporary-series",
        "temporary-doubt",
        "lasting",
        "lasting-after-temporary",
        "temporal-control",
        "temporal-injection",
    }
)


@dataclass(frozen=True)
class Verdict:
    """One judged unit: an arm's answer for one judgement in one run."""

    case: str
    run: int
    category: str | None


def _read(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def load(runs_dir: Path, p5_runs_dir: Path) -> dict[str, list[Verdict]]:
    """Return every arm's verdicts, keyed by arm ("haiku-old", "haiku", candidates)."""
    arms: dict[str, list[Verdict]] = defaultdict(list)
    # The 125 P5 judgements keep P5's shape tags; the 80 new ones have P5c's.
    p5_names = {j.name for j in all_judgements() if j.shape not in NEW_SHAPES}
    for row in _read(p5_runs_dir / "supersession-haiku-main.jsonl"):
        if row["case"] in p5_names:
            arms["haiku-old"].append(Verdict(str(row["case"]), int(row["run"]), row["category"]))  # type: ignore[arg-type]
    for row in _read(runs_dir / "supersession-haiku-old-base.jsonl"):
        arms["haiku-old"].append(Verdict(str(row["case"]), int(row["run"]), row["category"]))  # type: ignore[arg-type]
    for row in _read(runs_dir / "supersession-haiku-old-repro.jsonl"):
        arms["haiku-old-run4"].append(Verdict(str(row["case"]), int(row["run"]), row["category"]))  # type: ignore[arg-type]
    for arm in ("haiku", *CANDIDATES):
        for row in _read(runs_dir / f"supersession-{arm}-eval.jsonl"):
            arms[arm].append(Verdict(str(row["case"]), int(row["run"]), row["category"]))  # type: ignore[arg-type]
    return arms


def _units(
    verdicts: Iterable[Verdict], cases: dict[str, Judgement]
) -> dict[tuple[str, int], Verdict]:
    return {(v.case, v.run): v for v in verdicts if v.case in cases}


def correct(v: Verdict, j: Judgement) -> bool:
    """Report whether a verdict matches the label (an invalid reply never does)."""
    return v.category == j.category


def wrong_replacement(v: Verdict, j: Judgement) -> bool:
    """Report whether a verdict proposes a replacement the label does not."""
    return v.category == SUPERSESSION and j.category != SUPERSESSION


def _rate(hits: int, n: int) -> str:
    lo, hi = wilson_interval(hits, n)
    return f"{hits}/{n} ({100 * lo:.0f}–{100 * hi:.0f} %)"


def table(arms: dict[str, list[Verdict]], cases: dict[str, Judgement], title: str) -> None:
    """Print, per family and arm: correct, wrong replacements, invalid replies."""
    print(f"\n### {title}\n")
    print("| Family | Arm | Correct | Wrong replacements | Invalid |")
    print("|---|---|---|---|---|")
    for family in FAMILIES:
        fam = {n: j for n, j in cases.items() if j.family == family}
        if not fam:
            continue
        for arm in ("haiku-old", "haiku", *CANDIDATES):
            units = _units(arms.get(arm, []), fam)
            if not units:
                continue
            n = len(units)
            ok = sum(correct(v, fam[v.case]) for v in units.values())
            wr = sum(wrong_replacement(v, fam[v.case]) for v in units.values())
            inv = sum(v.category is None for v in units.values())
            print(f"| {family} | {arm} | {_rate(ok, n)} | {wr} | {inv} |")


def paired(
    a: dict[tuple[str, int], Verdict],
    b: dict[tuple[str, int], Verdict],
    cases: dict[str, Judgement],
    success: str,
) -> tuple[int, int, int, float]:
    """Return (units, only-a successes, only-b successes, exact McNemar p) on shared units."""
    keys = sorted(set(a) & set(b))
    test = correct if success == "correct" else (lambda v, j: not wrong_replacement(v, j))
    only_a = sum(test(a[k], cases[k[0]]) and not test(b[k], cases[k[0]]) for k in keys)
    only_b = sum(test(b[k], cases[k[0]]) and not test(a[k], cases[k[0]]) for k in keys)
    return len(keys), only_a, only_b, mcnemar_exact(only_a, only_b)


def _count(units: dict[tuple[str, int], Verdict], cases: dict[str, Judgement], what: str) -> int:
    if what == "wrong":
        return sum(wrong_replacement(v, cases[v.case]) for v in units.values())
    if what == "errors":
        return sum(not correct(v, cases[v.case]) for v in units.values())
    if what == "invalid":
        return sum(v.category is None for v in units.values())
    raise ValueError(what)


def _per_run(units: dict[tuple[str, int], Verdict], cases: dict[str, Judgement]) -> Counter[int]:
    counts: Counter[int] = Counter()
    for (_, run), v in units.items():
        counts[run] += 0
        if wrong_replacement(v, cases[v.case]):
            counts[run] += 1
    return counts


def _not_worse(
    cand: dict[tuple[str, int], Verdict],
    inc: dict[tuple[str, int], Verdict],
    cases: dict[str, Judgement],
) -> tuple[bool, str]:
    """Accuracy not worse beyond the interval: Wilson overlap or higher, McNemar not against."""
    nc, ni = len(cand), len(inc)
    if not nc or not ni:
        return True, "no units"
    okc = sum(correct(v, cases[v.case]) for v in cand.values())
    oki = sum(correct(v, cases[v.case]) for v in inc.values())
    wc, wi = wilson_interval(okc, nc), wilson_interval(oki, ni)
    rate_ok = okc / nc >= oki / ni or intervals_overlap(wc, wi)
    _, only_c, only_i, p = paired(cand, inc, cases, "correct")
    test_ok = not (p < 0.05 and only_i > only_c)
    return rate_ok and test_ok, f"{okc}/{nc} vs {oki}/{ni}, discordant {only_c}/{only_i}, p={p:.3g}"


def bars(arms: dict[str, list[Verdict]], cases: dict[str, Judgement]) -> None:
    """Evaluate F1–F4 and M1–M5 on the held-out judgements in `cases`."""
    held = {n: j for n, j in cases.items() if not j.dev}

    def fam(name: str) -> dict[str, Judgement]:
        return {n: j for n, j in held.items() if j.family == name}

    old = arms["haiku-old"]
    fixed = arms["haiku"]
    for arm_name in ("haiku", *CANDIDATES):
        if arms.get(arm_name):
            fix_bars(arms, held, arm_name)
    _model_bars(arms, held, old, fixed)


def fix_bars(arms: dict[str, list[Verdict]], held: dict[str, Judgement], arm_name: str) -> None:
    """Evaluate F1–F4 for one arm with the fixed prompt against haiku-old (held-out)."""

    def fam(name: str) -> dict[str, Judgement]:
        return {n: j for n, j in held.items() if j.family == name}

    old = arms["haiku-old"]
    fixed = arms[arm_name]
    print(
        f"\n### Bars for shipping the fix: {arm_name} with the fixed prompt vs haiku-old "
        "(held-out)\n"
    )
    temp = fam("temporary")
    n, o_fixed, o_old, p = paired(_units(fixed, temp), _units(old, temp), temp, "no_wrong")
    wr_old = _count(_units(old, temp), temp, "wrong")
    wr_fixed = _count(_units(fixed, temp), temp, "wrong")
    f1 = o_fixed > o_old and p < 0.05
    print(
        f"- F1 temporary wrong replacements: old {wr_old} -> fixed {wr_fixed} of {n} paired "
        f"units; discordant fixed-better {o_fixed} / old-better {o_old}, p={p:.3g}: "
        f"**{'met' if f1 else 'NOT met'}**"
    )
    last = fam("lasting")
    lu_f, lu_o = _units(fixed, last), _units(old, last)
    ok_f = sum(correct(v, last[v.case]) for v in lu_f.values())
    ok_o = sum(correct(v, last[v.case]) for v in lu_o.values())
    nw, ok2 = _not_worse(lu_f, lu_o, last)
    drop = 100 * (ok_o / len(lu_o) - ok_f / len(lu_f))
    f2 = nw and drop <= 5
    print(
        f"- F2 lasting correct: old {_rate(ok_o, len(lu_o))} -> fixed {_rate(ok_f, len(lu_f))} "
        f"({ok2}); drop {drop:.1f} pt: **{'met' if f2 else 'NOT met'}**"
    )
    p5 = fam("p5")
    pu_f, pu_o = _units(fixed, p5), _units(old, p5)
    nw3, d3 = _not_worse(pu_f, pu_o, p5)
    acc_f = sum(correct(v, p5[v.case]) for v in pu_f.values()) / len(pu_f)
    acc_o = sum(correct(v, p5[v.case]) for v in pu_o.values()) / len(pu_o)
    wr_ok = _count(pu_f, p5, "wrong") <= _count(pu_o, p5, "wrong")
    rec = {n: j for n, j in p5.items() if j.shape == "recurring-series"}
    bnd = {n: j for n, j in p5.items() if j.boundary}
    rec_ok = _count(_units(fixed, rec), rec, "errors") <= _count(_units(old, rec), rec, "errors")
    bnd_ok = _count(_units(fixed, bnd), bnd, "errors") <= _count(_units(old, bnd), bnd, "errors")
    f3 = nw3 and (acc_o - acc_f) * 100 <= 5 and wr_ok and rec_ok and bnd_ok
    print(
        f"- F3 p5: {d3}; wrong repl. fixed {_count(pu_f, p5, 'wrong')} vs old "
        f"{_count(pu_o, p5, 'wrong')}; recurring-series errors fixed "
        f"{_count(_units(fixed, rec), rec, 'errors')} vs old {_count(_units(old, rec), rec, 'errors')}; "
        f"boundary errors fixed {_count(_units(fixed, bnd), bnd, 'errors')} vs old "
        f"{_count(_units(old, bnd), bnd, 'errors')}: **{'met' if f3 else 'NOT met'}**"
    )
    all_f, all_o = _units(fixed, held), _units(old, held)
    inv_f = _count(all_f, held, "invalid") / len(all_f)
    inv_o = _count(all_o, held, "invalid") / len(all_o)
    f4 = inv_f <= inv_o + 0.01
    print(
        f"- F4 invalid: fixed {_count(all_f, held, 'invalid')}/{len(all_f)} vs old "
        f"{_count(all_o, held, 'invalid')}/{len(all_o)}: **{'met' if f4 else 'NOT met'}**"
    )


def _model_bars(
    arms: dict[str, list[Verdict]],
    held: dict[str, Judgement],
    old: list[Verdict],
    fixed: list[Verdict],
) -> None:
    """Evaluate M1–M6 for every candidate against Haiku-fixed and haiku-old (held-out)."""

    def fam(name: str) -> dict[str, Judgement]:
        return {n: j for n, j in held.items() if j.family == name}

    temp, p5 = fam("temporary"), fam("p5")
    all_f = _units(fixed, held)
    inv_f = _count(all_f, held, "invalid") / len(all_f)
    print("\n### Bars for switching the model (held-out, fixed prompt)\n")
    fixed_temp_runs = _per_run(_units(fixed, temp), temp)
    best = min(fixed_temp_runs.values())
    for cand in CANDIDATES:
        cu = arms.get(cand, [])
        if not cu:
            continue
        verdicts: list[str] = []
        runs = _per_run(_units(cu, temp), temp)
        m1 = max(runs.values()) <= best
        verdicts.append(
            f"M1 temporary wrong per run {dict(sorted(runs.items()))} vs Haiku-fixed "
            f"best {best}: {'met' if m1 else 'NOT met'}"
        )
        m2 = True
        groups: dict[str, dict[str, Judgement]] = {f: fam(f) for f in FAMILIES if f != "p5"}
        for shape in sorted({j.shape for j in p5.values()}):
            groups[f"p5:{shape}"] = {n: j for n, j in p5.items() if j.shape == shape}
        worse: list[str] = []
        for gname, g in groups.items():
            for inc_name, inc in (("old", old), ("fixed", fixed)):
                ok, detail = _not_worse(_units(cu, g), _units(inc, g), g)
                if not ok:
                    m2 = False
                    worse.append(f"{gname} vs {inc_name}: {detail}")
        verdicts.append(f"M2 not worse anywhere: {'met' if m2 else 'NOT met ' + '; '.join(worse)}")
        series = {
            n: j for n, j in held.items() if j.shape in ("recurring-series", "temporary-series")
        }
        bnd_all = {n: j for n, j in held.items() if j.boundary}
        cu_runs = len({v.run for v in cu}) or 1

        def avg(arm: list[Verdict], g: dict[str, Judgement], what: str) -> float:
            runs_n = len({v.run for v in arm if v.case in g}) or 1
            return _count(_units(arm, g), g, what) / runs_n

        m3 = all(
            avg(cu, series, "errors") <= avg(inc, series, "errors")
            and avg(cu, bnd_all, "wrong") <= avg(inc, bnd_all, "wrong")
            for inc in (old, fixed)
        )
        verdicts.append(
            f"M3 series errors/run {avg(cu, series, 'errors'):.2f} (old {avg(old, series, 'errors'):.2f}, "
            f"fixed {avg(fixed, series, 'errors'):.2f}); boundary wrong/run {avg(cu, bnd_all, 'wrong'):.2f} "
            f"(old {avg(old, bnd_all, 'wrong'):.2f}, fixed {avg(fixed, bnd_all, 'wrong'):.2f}): "
            f"{'met' if m3 else 'NOT met'}"
        )
        doubt = {
            n: j
            for n, j in held.items()
            if j.category == "contradiction" or j.family == "temporary-doubt"
        }
        m4 = avg(cu, doubt, "wrong") <= avg(fixed, doubt, "wrong")
        verdicts.append(
            f"M4 doubt pairs wrong/run {avg(cu, doubt, 'wrong'):.2f} vs fixed "
            f"{avg(fixed, doubt, 'wrong'):.2f}: {'met' if m4 else 'NOT met'}"
        )
        cu_all = _units(cu, held)
        m5 = _count(cu_all, held, "invalid") / len(cu_all) <= inv_f + 0.01
        verdicts.append(
            f"M5 invalid {_count(cu_all, held, 'invalid')}/{len(cu_all)}: "
            f"{'met' if m5 else 'NOT met'}"
        )
        fixed_wrong_cases = {
            v.case for v in fixed if v.case in held and wrong_replacement(v, held[v.case])
        }
        new_wrong = sorted(
            {v.case for v in cu if v.case in held and wrong_replacement(v, held[v.case])}
            - fixed_wrong_cases
        )
        m6 = not new_wrong
        verdicts.append(
            f"M6 wrong replacements where Haiku-fixed has none: {new_wrong or 'none'}: "
            f"{'met' if m6 else 'NOT met'}"
        )
        passed = m1 and m2 and m3 and m4 and m5 and m6
        print(f"- **{cand}** ({cu_runs} runs): {'PASSES' if passed else 'fails'}")
        for line in verdicts:
            print(f"  - {line}")


def disagreements(
    arms: dict[str, list[Verdict]], cases: dict[str, Judgement], a: str, b: str
) -> None:
    """Print every judgement on which arms `a` and `b` answered differently in some run."""
    ua, ub = _units(arms[a], cases), _units(arms[b], cases)
    by_case: dict[str, list[str]] = defaultdict(list)
    for key in sorted(set(ua) & set(ub)):
        va, vb = ua[key], ub[key]
        if va.category != vb.category:
            by_case[key[0]].append(f"run{key[1]}: {a}={va.category} {b}={vb.category}")
    print(f"\n### Disagreements {a} vs {b} ({len(by_case)} judgements)\n")
    for case, lines in sorted(by_case.items()):
        j = cases[case]
        print(
            f"- `{case}` [{j.family}/{j.shape}, label {j.category}, {'dev' if j.dev else 'held'}]: "
            + "; ".join(lines)
        )


def main() -> int:
    """Print the tables, the bars and the disagreements; see the module docstring."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--p5-runs-dir", type=Path, required=True)
    parser.add_argument("--without", default="")
    parser.add_argument("--disagreements", action="store_true")
    args = parser.parse_args()
    dropped = {name for name in args.without.split(",") if name}
    cases = {j.name: j for j in all_judgements() if j.name not in dropped}
    arms = load(args.runs_dir, args.p5_runs_dir)
    held = {n: j for n, j in cases.items() if not j.dev}
    dev = {n: j for n, j in cases.items() if j.dev}
    table(arms, held, "Held-out")
    table(arms, dev, "Dev")
    bars(arms, cases)
    drift = _units(arms["haiku-old-run4"], cases)
    p5_old = _units(arms["haiku-old"], cases)
    changed = sorted(
        {
            c
            for (c, _), v in drift.items()
            if any(p5_old[(c, r)].category != v.category for r in (1, 2, 3) if (c, r) in p5_old)
        }
    )
    print(
        f"\nDrift check (haiku-old run 4 today vs P5 runs 1–3): {len(drift)} units, "
        f"judgements answered differently from any P5 run: {changed or 'none'}"
    )
    if args.disagreements:
        disagreements(arms, cases, "haiku-old", "haiku")
        for cand in CANDIDATES:
            disagreements(arms, cases, "haiku", cand)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
