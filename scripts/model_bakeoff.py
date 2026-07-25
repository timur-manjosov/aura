"""Run the PROACTIVE_MODEL / SYNTHESIS_MODEL bake-off against real, paid models.

    AURA_RUN_REAL_LLM=1 .venv/bin/python scripts/model_bakeoff.py

THIS SPENDS REAL MONEY. It is deliberately not a pytest test and does not live
under tests/: pytest.ini pins `testpaths = tests`, so nothing here is reachable
by a bare `pytest` invocation even by accident. On top of that structural
separation it refuses to run at all unless AURA_RUN_REAL_LLM is set -- the same
one signal, set by nothing but a human who means it, that tests/conftest.py
keys its hermetic guard off.

Why a script rather than an opt-in test: a bake-off is not a pass/fail
assertion about Aura's code, it is a measurement of third-party models whose
results are an input to a config decision. Expressing it as a test would mean
either asserting a specific model wins (which bakes today's catalog into the
suite) or asserting nothing (a test that cannot fail). It belongs next to the
data it produces instead.

It drives the real production synthesize_answer, not a copy of its prompt, so
what gets measured is the schema, system prompt and parsing that actually ship.
The failure-mode breakdown comes from capturing aura.synthesis's own log
records, because synthesize_answer deliberately collapses every failure into
None for its callers -- useful in production, not specific enough to choose a
model on.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bakeoff_cases import CASES, BakeOffCase  # noqa: E402

from aura.db.models import Fact, FactStatus  # noqa: E402
from aura.synthesis import synthesize_answer  # noqa: E402

RUN_REAL_LLM_ENV = "AURA_RUN_REAL_LLM"

CANDIDATES = [
    "openrouter/google/gemini-3.1-flash-lite-preview",
    "openrouter/openai/gpt-5.4-mini",
    "openrouter/anthropic/claude-haiku-4.5",
]

# Scripts that reliably identify a language by codepoint alone. Latin-script
# locales are not checkable this way, so they are reported for human reading
# rather than machine-scored -- see `language_ok`.
_SCRIPT_RANGES = {
    "ja": ((0x3040, 0x30FF), (0x4E00, 0x9FFF)),
    "ko": ((0xAC00, 0xD7AF), (0x1100, 0x11FF)),
}


@dataclass
class CaseResult:
    """The measured outcome of one case against one model."""

    case: BakeOffCase
    passed: bool
    answers_question: bool | None
    used_fact_ids: list[int]
    answer: str
    failure: str
    latency_seconds: float
    language_ok: bool | None


def _make_fact(fact_id: int, content: str) -> Fact:
    """Build a Fact carrying only what synthesis reads: its id and its content."""
    return Fact(
        id=fact_id,
        guild_id=1,
        channel_id=1,
        message_id=fact_id,
        content=content,
        embedding=b"",  # synthesis never touches the vector; retrieval already ran
        status=FactStatus.ACTIVE,
        created_at=datetime.now(timezone.utc),
    )


def _language_ok(answer: str, locale: str) -> bool | None:
    """Whether the answer is written in the required script, or None if unscoreable.

    Only meaningful for locales whose script is unambiguous by codepoint. For
    Latin-script locales this returns None and the answer text is printed
    instead, so a human can judge it -- guessing at language ID with a stopword
    heuristic would produce confident wrong numbers, which is worse than
    admitting the check does not apply.
    """
    ranges = _SCRIPT_RANGES.get(locale)
    if ranges is None:
        return None
    return any(any(low <= ord(ch) <= high for low, high in ranges) for ch in answer)


class _FailureCapture(logging.Handler):
    """Collects aura.synthesis's log records so a None can be attributed to a cause."""

    def __init__(self) -> None:
        super().__init__(level=logging.ERROR)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


async def run_case(case: BakeOffCase, model: str, capture: _FailureCapture) -> CaseResult:
    """Run one case against one model and score it. Never raises."""
    facts = [_make_fact(i, content) for i, content in enumerate(case.facts, start=1)]
    capture.messages.clear()

    started = time.monotonic()
    try:
        result = await synthesize_answer(facts, case.message, case.locale, model=model)
    except Exception as exc:  # synthesize_answer is documented never to raise; verify it
        return CaseResult(
            case=case,
            passed=False,
            answers_question=None,
            used_fact_ids=[],
            answer="",
            failure=f"UNEXPECTED RAISE: {type(exc).__name__}: {exc}",
            latency_seconds=time.monotonic() - started,
            language_ok=None,
        )
    latency = time.monotonic() - started

    if result is None:
        return CaseResult(
            case=case,
            passed=False,
            answers_question=None,
            used_fact_ids=[],
            answer="",
            failure="; ".join(capture.messages) or "synthesis returned None",
            latency_seconds=latency,
            language_ok=None,
        )

    verdict_ok = result.answers_question == case.expected_answers_question
    # Citations are scored only when the model was supposed to answer: while
    # declining, what it points at carries no meaning.
    citation_ok = (
        sorted(result.used_fact_ids) == sorted(case.expected_fact_ids)
        if case.expected_answers_question
        else True
    )
    # A claim to answer while citing nothing is self-contradictory, and the
    # production responder already rejects it -- so score it as a miss here too.
    cited_something = bool(result.used_fact_ids) if case.expected_answers_question else True

    failure = ""
    if not verdict_ok:
        failure = (
            f"answers_question={result.answers_question}, "
            f"expected {case.expected_answers_question}"
        )
    elif not cited_something:
        failure = "claimed to answer but cited no fact"
    elif not citation_ok:
        failure = f"cited facts {sorted(result.used_fact_ids)}, expected {case.expected_fact_ids}"

    return CaseResult(
        case=case,
        passed=verdict_ok and citation_ok and cited_something,
        answers_question=result.answers_question,
        used_fact_ids=result.used_fact_ids,
        answer=result.answer,
        failure=failure,
        latency_seconds=latency,
        language_ok=_language_ok(result.answer, case.locale),
    )


async def run_model(model: str, capture: _FailureCapture) -> list[CaseResult]:
    """Run every case against one model, sequentially.

    Sequential on purpose: concurrency here would measure the provider's rate
    limiter as much as the model, and the whole run is a few dozen calls.
    """
    results: list[CaseResult] = []
    for case in CASES:
        result = await run_case(case, model, capture)
        results.append(result)
        mark = "PASS" if result.passed else "FAIL"
        print(f"  [{mark}] {case.name:<26} ({case.locale:<5}) {result.failure}", flush=True)
    return results


def _print_report(all_results: dict[str, list[CaseResult]]) -> None:
    """Print the raw per-case matrix and the per-model summary."""
    models = list(all_results)

    print("\n" + "=" * 100)
    print("RAW PER-CASE RESULTS")
    print("=" * 100)
    header = f"{'case':<26} {'locale':<7} {'expect':<7} " + " ".join(
        f"{m.split('/')[-1][:22]:<23}" for m in models
    )
    print(header)
    print("-" * len(header))
    for index, case in enumerate(CASES):
        row = f"{case.name:<26} {case.locale:<7} {str(case.expected_answers_question):<7} "
        for model in models:
            result = all_results[model][index]
            row += f"{('PASS' if result.passed else 'FAIL'):<23} "
        print(row)

    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    for model in models:
        results = all_results[model]
        passed = sum(r.passed for r in results)
        malformed = sum(1 for r in results if r.answers_question is None)
        latencies = sorted(r.latency_seconds for r in results)
        median = latencies[len(latencies) // 2]
        lang_checked = [r for r in results if r.language_ok is not None]
        lang_ok = sum(1 for r in lang_checked if r.language_ok)
        print(
            f"{model:<52} {passed}/{len(results)} passed | "
            f"{malformed} unusable | median {median:.2f}s | "
            f"max {latencies[-1]:.2f}s | script-check {lang_ok}/{len(lang_checked)}"
        )


def _write_report(all_results: dict[str, list[CaseResult]], path: Path) -> None:
    """Write the full raw results, including every answer, as JSON for later audit."""
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "models": {
            model: [
                {
                    "case": r.case.name,
                    "category": r.case.category,
                    "locale": r.case.locale,
                    "message": r.case.message,
                    "facts": r.case.facts,
                    "expected_answers_question": r.case.expected_answers_question,
                    "expected_fact_ids": r.case.expected_fact_ids,
                    "passed": r.passed,
                    "answers_question": r.answers_question,
                    "used_fact_ids": r.used_fact_ids,
                    "answer": r.answer,
                    "failure": r.failure,
                    "latency_seconds": round(r.latency_seconds, 3),
                    "language_script_ok": r.language_ok,
                }
                for r in results
            ]
            for model, results in all_results.items()
        },
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nRaw results written to {path}")


async def main() -> int:
    """Guard the run, execute every candidate over every case, and report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models", nargs="+", default=CANDIDATES, help="models to evaluate (litellm strings)"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "reports" / "model-bakeoff.json",
    )
    args = parser.parse_args()

    if not os.environ.get(RUN_REAL_LLM_ENV):
        print(
            f"refusing to run: {RUN_REAL_LLM_ENV} is not set. This script makes real, "
            f"paid LLM calls.\nRe-run with: {RUN_REAL_LLM_ENV}=1 "
            f".venv/bin/python scripts/model_bakeoff.py",
            file=sys.stderr,
        )
        return 1

    capture = _FailureCapture()
    logging.getLogger("aura.synthesis").addHandler(capture)
    # litellm is chatty at INFO and would bury the per-case lines.
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)

    print(f"{len(CASES)} cases x {len(args.models)} models = {len(CASES) * len(args.models)} calls")
    all_results: dict[str, list[CaseResult]] = {}
    for model in args.models:
        print(f"\n--- {model} ---", flush=True)
        all_results[model] = await run_model(model, capture)

    _print_report(all_results)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    _write_report(all_results, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
