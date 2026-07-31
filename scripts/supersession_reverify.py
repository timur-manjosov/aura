"""Re-verify the bake-off's findings against the SHIPPED supersession prompt.

    AURA_RUN_REAL_LLM=1 .venv/bin/python scripts/supersession_reverify.py --dry-run
    AURA_RUN_REAL_LLM=1 .venv/bin/python scripts/supersession_reverify.py

THIS SPENDS REAL MONEY. Same structural separation as every other measurement
script here: deliberately not a pytest test, not under tests/, unreachable from
a bare `pytest` (pytest.ini pins `testpaths = tests`), and it refuses to run at
all unless AURA_RUN_REAL_LLM is set -- the one signal tests/conftest.py keys its
hermetic guard off, set by nothing but a human who means it.

WHAT THIS IS, AND HOW IT DIFFERS FROM THE BAKE-OFF. scripts/
supersession_bakeoff.py chose a model, using a prompt written only to be
measured and never shipped (reports/supersession-model-bakeoff.txt Section 7
says so explicitly). This script drives the REAL judge_relationship from
aura.extraction.supersession -- the same function the pipeline calls, with the
same prompt, the same parsing, and the same Rule 1 enforcement. If the shipped
prompt regresses, this catches it; if this script passes, the thing that passed
is the thing that ships.

WHAT IT MEASURES, beyond the category:

  * change_signal, the evidence the model committed to before choosing a
    category. Reported verbatim, plus whether it appears literally inside the
    candidate -- a quoted signal that is not actually in the text is the model
    inventing its own warrant, which is worth knowing even though nothing
    currently rejects it.
  * whether Rule 1's code-level downgrade fired (a "supersession" with no
    signal, escalated to "contradiction"). Counted by capturing the module's
    own warning, so the report can distinguish "the prompt worked" from "the
    net caught it".
  * the reasoning sentence and the script it is written in, since the shipped
    prompt asks for it in the CANDIDATE's language rather than in English --
    a change from the bake-off's prompt, and one that could plausibly have cost
    classification quality. The Hangul/Kana check is automatic; Latin-script
    locales are printed for a human to read.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from supersession_bakeoff_cases import (  # noqa: E402
    ALL_CASES,
    REVERIFICATION_CASE_NAMES,
    SupersessionCase,
)

from aura.config import load_settings  # noqa: E402
from aura.extraction.supersession import (  # noqa: E402
    RelationshipJudgement,
    has_change_signal,
    judge_relationship,
)

RUN_REAL_LLM_ENV = "AURA_RUN_REAL_LLM"

# The model reports/supersession-model-bakeoff.txt chose, and the shipped
# default of SUPERSESSION_MODEL. Not read from settings on purpose: this script
# re-verifies a specific measured claim about a specific model, so a deployment
# that has since pointed SUPERSESSION_MODEL somewhere else must not silently
# change what "re-verified" means.
MODEL = "openrouter/anthropic/claude-haiku-4.5"

# Live OpenRouter pricing per million tokens for MODEL, re-checked 2026-07-31
# rather than assumed -- the standing rule in this project since
# reports/model-bakeoff.txt Section 2 found a carried price already stale.
_PRICE_IN_PER_MTOK = 1.00
_PRICE_OUT_PER_MTOK = 5.00

# Rough token counts for the dry-run estimate only. The shipped prompt is
# larger than the bake-off's (it carries two extra rules), so this is deliberately
# generous rather than carried over.
_PROMPT_TOKENS = 900
_OUTPUT_TOKENS = 90

_DOWNGRADE_MARKER = "no transition wording"


class _DowngradeWatcher(logging.Handler):
    """Counts Rule 1's code-level downgrade by listening for its own warning.

    A downgrade is invisible in the returned judgement -- "contradiction with no
    signal" is also what a correct, uncorrected answer looks like -- so the only
    honest way to tell the two apart from outside the module is to watch the log
    line it emits when it acts.
    """

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.fired = 0

    def emit(self, record: logging.LogRecord) -> None:
        if _DOWNGRADE_MARKER in record.getMessage():
            self.fired += 1


@dataclass
class RunResult:
    """One run of one case through the shipped call."""

    case: SupersessionCase
    run_index: int
    judgement: RelationshipJudgement | None
    downgraded: bool
    latency_seconds: float

    @property
    def correct(self) -> bool:
        return self.judgement is not None and self.judgement.relationship == self.case.category

    @property
    def signal_is_quoted_from_the_candidate(self) -> bool | None:
        """Whether the quoted signal actually appears in Fact B. None when absent."""
        if self.judgement is None or not has_change_signal(self.judgement.change_signal):
            return None
        needle = self.judgement.change_signal.strip().strip("\"'").casefold()
        return needle in self.case.candidate.casefold()


@dataclass
class CaseReport:
    case: SupersessionCase
    runs: list[RunResult] = field(default_factory=list)

    @property
    def all_correct(self) -> bool:
        return bool(self.runs) and all(run.correct for run in self.runs)

    @property
    def consistent(self) -> bool:
        categories = {
            run.judgement.relationship if run.judgement else None for run in self.runs
        }
        return len(categories) == 1


def _reasoning_script(text: str) -> str:
    """A coarse description of which script a reasoning sentence is written in.

    Enough to answer the one question that matters automatically -- did the
    Korean case come back in Korean -- without pretending to be language
    detection. Latin-script output is reported as such and read by a human,
    because "German or English" is not a question a character range can answer.
    """
    scripts: set[str] = set()
    for character in text:
        if not character.isalpha():
            continue
        name = unicodedata.name(character, "")
        for script in ("HANGUL", "KATAKANA", "HIRAGANA", "CJK", "CYRILLIC", "ARABIC"):
            if name.startswith(script):
                scripts.add(script)
                break
        else:
            scripts.add("LATIN")
    return "+".join(sorted(scripts)) if scripts else "none"


def _selected_cases() -> list[SupersessionCase]:
    by_name = {case.name: case for case in ALL_CASES}
    missing = [name for name in REVERIFICATION_CASE_NAMES if name not in by_name]
    if missing:
        raise SystemExit(f"Unknown case name(s) in REVERIFICATION_CASE_NAMES: {missing}")
    return [by_name[name] for name in REVERIFICATION_CASE_NAMES]


def _estimate() -> tuple[int, float]:
    cases = _selected_cases()
    calls = sum(max(case.repeats, 3) for case in cases)
    per_call = (
        _PROMPT_TOKENS / 1_000_000 * _PRICE_IN_PER_MTOK
        + _OUTPUT_TOKENS / 1_000_000 * _PRICE_OUT_PER_MTOK
    )
    return calls, calls * per_call


async def _run_case(case: SupersessionCase, run_index: int) -> RunResult:
    watcher = _DowngradeWatcher()
    logger = logging.getLogger("aura.extraction.supersession")
    logger.addHandler(watcher)
    started = time.monotonic()
    try:
        judgement = await judge_relationship(
            predecessor=case.predecessor, candidate=case.candidate, model=MODEL
        )
    finally:
        logger.removeHandler(watcher)
    return RunResult(
        case=case,
        run_index=run_index,
        judgement=judgement,
        downgraded=watcher.fired > 0,
        latency_seconds=time.monotonic() - started,
    )


def _print_report(reports: list[CaseReport]) -> None:
    print("\n" + "=" * 100)
    print("PER-CASE RESULT (3 runs each, shipped prompt, %s)" % MODEL)
    print("=" * 100)
    for report in reports:
        case = report.case
        verdicts = " ".join(
            (
                "CALL-FAILED"
                if run.judgement is None
                else ("PASS" if run.correct else f"WRONG({run.judgement.relationship.value})")
            )
            for run in sorted(report.runs, key=lambda r: r.run_index)
        )
        print(f"\n{case.name}")
        print(f"  expected      {case.category}")
        print(f"  runs          {verdicts}")
        print(f"  consistent    {'yes' if report.consistent else 'NO -- flip-flopped'}")
        for run in sorted(report.runs, key=lambda r: r.run_index):
            if run.judgement is None:
                print(f"  [{run.run_index}] call failed ({run.latency_seconds:.2f}s)")
                continue
            quoted = run.signal_is_quoted_from_the_candidate
            quoted_note = (
                "n/a (none)"
                if quoted is None
                else ("verbatim in Fact B" if quoted else "NOT IN FACT B")
            )
            print(
                f"  [{run.run_index}] {run.judgement.relationship.value:<14} "
                f"signal={run.judgement.change_signal!r} ({quoted_note})"
                f"{'  RULE-1-DOWNGRADE' if run.downgraded else ''}"
            )
            print(
                f"       reasoning [{_reasoning_script(run.judgement.reasoning)}] "
                f"{run.judgement.reasoning}"
            )

    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    passed = [r for r in reports if r.all_correct]
    print(f"  cases fully correct on all 3 runs   {len(passed)}/{len(reports)}")
    flip = [r for r in reports if not r.consistent]
    print(f"  cases that flip-flopped             {len(flip)}")
    downgrades = sum(1 for r in reports for run in r.runs if run.downgraded)
    print(f"  Rule 1 code-level downgrades fired  {downgrades}")
    failures = [r for r in reports if not r.all_correct]
    if failures:
        print("\n  NOT fully correct:")
        for report in failures:
            print(f"    {report.case.name} (expected {report.case.category})")


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent.parent
        / "reports"
        / "supersession-reverify.json",
    )
    args = parser.parse_args()

    calls, cost = _estimate()
    if args.dry_run:
        print("DRY RUN -- nothing was spent.\n")
        print(f"  cases            {len(_selected_cases())}")
        print(f"  calls (3x each)  {calls}")
        print(f"  model            {MODEL}")
        print(f"  estimated cost   ${cost:.4f}")
        return 0

    if not os.environ.get(RUN_REAL_LLM_ENV):
        print(
            f"Refusing to run: set {RUN_REAL_LLM_ENV}=1 to make real, paid model calls.\n"
            "Use --dry-run to see the cost estimate first.",
            file=sys.stderr,
        )
        return 1

    if not load_settings().llm_api_key:
        print("Refusing to run: no LLM_API_KEY configured.", file=sys.stderr)
        return 1

    print(f"Running {calls} call(s) against {MODEL} (estimated ${cost:.4f})\n", flush=True)

    reports: list[CaseReport] = []
    for case in _selected_cases():
        report = CaseReport(case=case)
        for run_index in range(1, max(case.repeats, 3) + 1):
            result = await _run_case(case, run_index)
            report.runs.append(result)
            status = (
                "FAILED"
                if result.judgement is None
                else ("PASS" if result.correct else "FAIL")
            )
            print(
                f"  {case.name:<52} run {run_index}  {status:<6} "
                f"{result.latency_seconds:5.2f}s",
                flush=True,
            )
        reports.append(report)

    _print_report(reports)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "model": MODEL,
                "prompt": "shipped (aura.extraction.supersession)",
                "estimated_cost_usd": cost,
                "cases": [
                    {
                        "name": report.case.name,
                        "expected": report.case.category,
                        "predecessor": report.case.predecessor,
                        "candidate": report.case.candidate,
                        "all_correct": report.all_correct,
                        "consistent": report.consistent,
                        "runs": [
                            {
                                "run": run.run_index,
                                "category": (
                                    run.judgement.relationship.value
                                    if run.judgement
                                    else None
                                ),
                                "correct": run.correct,
                                "change_signal": (
                                    run.judgement.change_signal if run.judgement else None
                                ),
                                "signal_quoted_from_candidate": (
                                    run.signal_is_quoted_from_the_candidate
                                ),
                                "rule_1_downgrade": run.downgraded,
                                "reasoning": (
                                    run.judgement.reasoning if run.judgement else None
                                ),
                                "reasoning_script": (
                                    _reasoning_script(run.judgement.reasoning)
                                    if run.judgement
                                    else None
                                ),
                                "latency_seconds": round(run.latency_seconds, 3),
                            }
                            for run in sorted(report.runs, key=lambda r: r.run_index)
                        ],
                    }
                    for report in reports
                ],
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\nRaw results written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
