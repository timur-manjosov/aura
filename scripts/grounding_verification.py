"""Real, paid verification of the independent grounding check.

    AURA_RUN_REAL_LLM=1 .venv/bin/python scripts/grounding_verification.py --dry-run
    AURA_RUN_REAL_LLM=1 .venv/bin/python scripts/grounding_verification.py

THIS SPENDS REAL MONEY. Same structural separation as every other paid harness
in this project (scripts/model_bakeoff.py, scripts/supersession_bakeoff.py,
scripts/variant_verification.py): deliberately not a pytest test, not under
tests/, pytest.ini pins `testpaths = tests` so a bare `pytest` cannot reach it,
and it refuses to run at all unless AURA_RUN_REAL_LLM is set -- the one signal
tests/conftest.py keys its hermetic guard off, set by nothing but a human who
means it.

UNLIKE the bake-offs, this is not a model comparison. GROUNDING_CHECK_MODEL is
a carried-over choice, not a contested one (see aura.config), so what needs
measuring is whether the SHIPPED call actually works -- which is why this drives
the real aura.grounding.verify_answer_grounded rather than a candidate prompt of
its own. Nothing here re-implements the call; if the shipped prompt changes,
this measures the change.

Two things are measured, and they are equally important:

  1. VERDICTS on hand-written cases -- attacks that must be caught, controls
     that must NOT be refused. A check that refuses everything catches every
     attack and makes Aura mute, so counting only attacks caught would report
     half the question as the whole one.

  2. END-TO-END LATENCY on both send paths, before and after, with real calls
     on both sides -- the basis on which the extra wait can actually be judged
     rather than guessed at.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from grounding_verification_cases import ALL_CASES  # noqa: E402

from aura.config import ModelComponent, Settings, load_settings  # noqa: E402
from aura.db.models import Fact, FactStatus  # noqa: E402
from aura.grounding import (  # noqa: E402
    ASK_GROUNDING_TIMEOUT_SECONDS,
    PROACTIVE_GROUNDING_TIMEOUT_SECONDS,
    GroundingOutcome,
    verify_answer_grounded,
)
from aura.synthesis import synthesize_answer  # noqa: E402

RUN_REAL_LLM_ENV = "AURA_RUN_REAL_LLM"

# Live OpenRouter pricing per million tokens, re-checked against the models API
# on 2026-08-15 rather than carried from an earlier report -- the standing rule
# since reports/model-bakeoff.txt Section 2 found a price had gone stale 3.7x.
# No drift found this time.
_PRICE_PER_MTOK = {
    "openrouter/openai/gpt-4o-mini": (0.15, 0.60),
    "openrouter/anthropic/claude-haiku-4.5": (1.00, 5.00),
}

# Rough token counts for the dry-run ESTIMATE only. The real run reports no
# per-call token figures for the same reason reports/supersession-model-bakeoff.txt
# Section 5 gives: instrumenting usage here would mean measuring a one-off script
# more precisely than the production code it drives. OpenRouter's dashboard is
# the authority if the exact billed figure matters.
_GROUNDING_PROMPT_TOKENS = 950
_GROUNDING_OUTPUT_TOKENS = 70
_SYNTHESIS_PROMPT_TOKENS = 500
_SYNTHESIS_OUTPUT_TOKENS = 90

# Repeats per verdict case. temperature is pinned to 0.0 at the call site, but
# pinned is not deterministic across a provider's fleet, and this project has
# been bitten by a run-to-run flip before (reports/model-bakeoff.txt found
# gpt-5.4-mini flip-flopping 4/6 on repeated boundary cases). A verdict that
# changes between runs on a fail-closed gate means a question that answers
# sometimes and goes silent sometimes, so it is measured rather than assumed.
_RUNS_PER_CASE = 3

# Scenarios for the latency half: realistic question/fact shapes, driven through
# the real synthesize_answer and then the real grounding check.
_LATENCY_SCENARIOS = [
    (
        "Where are the server rules?",
        ["The server rules are pinned in #welcome.", "New members should read #welcome first."],
        "en-US",
    ),
    (
        "When is the next maintenance window?",
        [
            "Scheduled maintenance happens on the first Sunday of each month.",
            "During maintenance, #status is the channel to watch for updates.",
        ],
        "en-US",
    ),
    (
        "Wie groß dürfen Uploads sein?",
        ["Uploads in #media are limited to 8 MB."],
        "de",
    ),
    (
        "How do I apply to be a moderator?",
        [
            "Applications for the moderation team go through the form in #mod-applications.",
            "Applicants need to have been a member for at least 30 days.",
        ],
        "en-US",
    ),
    (
        "Where do bug reports go?",
        ["Bug reports go in #bug-reports with a screenshot attached."],
        "en-US",
    ),
]


def _fact(index: int, content: str) -> Fact:
    """A Fact carrying only what the grounding check reads: its text."""
    return Fact(
        id=index,
        guild_id=1,
        channel_id=1,
        message_id=index,
        content=content,
        embedding=b"",
        status=FactStatus.ACTIVE,
        superseded_by_id=None,
        created_at=datetime.now(timezone.utc),
    )


@dataclass
class VerdictResult:
    """One run of one case through the real, shipped grounding check."""

    case_name: str
    run_index: int
    expected_grounded: bool
    outcome: str
    latency_seconds: float

    @property
    def actual_grounded(self) -> bool | None:
        if self.outcome == GroundingOutcome.GROUNDED.value:
            return True
        if self.outcome == GroundingOutcome.UNGROUNDED.value:
            return False
        return None  # the check itself failed; neither a pass nor a catch

    @property
    def correct(self) -> bool:
        return self.actual_grounded is self.expected_grounded


@dataclass
class LatencyResult:
    """One scenario timed through one path, synthesis and check measured apart."""

    scenario: str
    path: str
    synthesis_seconds: float
    grounding_seconds: float
    synthesis_succeeded: bool
    grounding_outcome: str

    @property
    def before_seconds(self) -> float:
        """What the path cost before this feature: synthesis alone."""
        return self.synthesis_seconds

    @property
    def after_seconds(self) -> float:
        return self.synthesis_seconds + self.grounding_seconds


def _grounding_settings(settings: Settings) -> Settings:
    """Settings guaranteed to have the grounding check switched on.

    The repo's own .env predates this field, so an operator running this script
    without adding it would otherwise measure NOT_CONFIGURED against every case
    and report a flawless run in which nothing was ever checked -- the exact
    silent-pass shape the whole feature exists to prevent, arriving through its
    own verification harness.
    """
    if settings.grounding_check_model:
        return settings
    return settings.model_copy(update={"grounding_check_model": "openrouter/openai/gpt-4o-mini"})


async def _run_verdict_cases(settings: Settings) -> list[VerdictResult]:
    results: list[VerdictResult] = []
    for case in ALL_CASES:
        facts = [_fact(index, content) for index, content in enumerate(case.facts, start=1)]
        for run_index in range(1, _RUNS_PER_CASE + 1):
            started = time.perf_counter()
            outcome = await verify_answer_grounded(
                answer=case.answer,
                cited_facts=facts,
                settings=settings,
                timeout_seconds=ASK_GROUNDING_TIMEOUT_SECONDS,
            )
            elapsed = time.perf_counter() - started
            result = VerdictResult(
                case_name=case.name,
                run_index=run_index,
                expected_grounded=case.expected_grounded,
                outcome=outcome.value,
                latency_seconds=elapsed,
            )
            results.append(result)
            mark = "ok " if result.correct else "MISS"
            print(
                f"  [{mark}] {case.name} run {run_index}: {outcome.value} "
                f"(expected grounded={case.expected_grounded}) {elapsed:.2f}s"
            )
    return results


async def _run_latency(settings: Settings) -> list[LatencyResult]:
    results: list[LatencyResult] = []
    paths = [
        ("ask", ModelComponent.SYNTHESIS, ASK_GROUNDING_TIMEOUT_SECONDS),
        ("proactive", ModelComponent.PROACTIVE, PROACTIVE_GROUNDING_TIMEOUT_SECONDS),
    ]
    for question, fact_texts, locale in _LATENCY_SCENARIOS:
        facts = [_fact(index, content) for index, content in enumerate(fact_texts, start=1)]
        for path_name, component, timeout in paths:
            model = settings.resolve_model(component)
            assert model is not None, f"no model configured for {component}"

            started = time.perf_counter()
            synthesis = await synthesize_answer(facts, question, locale, model=model)
            synthesis_seconds = time.perf_counter() - started

            if synthesis is None:
                results.append(
                    LatencyResult(
                        scenario=question,
                        path=path_name,
                        synthesis_seconds=synthesis_seconds,
                        grounding_seconds=0.0,
                        synthesis_succeeded=False,
                        grounding_outcome="not_reached",
                    )
                )
                print(f"  [FAIL] {path_name}: synthesis failed for {question!r}")
                continue

            cited = [fact for fact in facts if fact.id in synthesis.used_fact_ids]
            started = time.perf_counter()
            outcome = await verify_answer_grounded(
                answer=synthesis.answer,
                cited_facts=cited,
                settings=settings,
                timeout_seconds=timeout,
            )
            grounding_seconds = time.perf_counter() - started

            result = LatencyResult(
                scenario=question,
                path=path_name,
                synthesis_seconds=synthesis_seconds,
                grounding_seconds=grounding_seconds,
                synthesis_succeeded=True,
                grounding_outcome=outcome.value,
            )
            results.append(result)
            print(
                f"  {path_name:<10} {question[:38]:<40} synth {synthesis_seconds:5.2f}s "
                f"+ check {grounding_seconds:5.2f}s = {result.after_seconds:5.2f}s "
                f"({outcome.value})"
            )
    return results


def _estimate_cost(settings: Settings) -> float:
    grounding_model = settings.resolve_model(ModelComponent.GROUNDING_CHECK) or ""
    synthesis_model = settings.resolve_model(ModelComponent.SYNTHESIS) or ""

    grounding_calls = len(ALL_CASES) * _RUNS_PER_CASE + len(_LATENCY_SCENARIOS) * 2
    synthesis_calls = len(_LATENCY_SCENARIOS) * 2

    total = 0.0
    for model, calls, prompt_tokens, output_tokens in (
        (grounding_model, grounding_calls, _GROUNDING_PROMPT_TOKENS, _GROUNDING_OUTPUT_TOKENS),
        (synthesis_model, synthesis_calls, _SYNTHESIS_PROMPT_TOKENS, _SYNTHESIS_OUTPUT_TOKENS),
    ):
        prompt_price, output_price = _PRICE_PER_MTOK.get(model, (1.00, 5.00))
        total += calls * (
            prompt_tokens / 1_000_000 * prompt_price + output_tokens / 1_000_000 * output_price
        )
    print(f"  grounding calls: {grounding_calls} on {grounding_model or '<unset>'}")
    print(f"  synthesis calls: {synthesis_calls} on {synthesis_model or '<unset>'}")
    return total


def _summarize(verdicts: list[VerdictResult], latencies: list[LatencyResult]) -> None:
    print("\n=== VERDICTS ===")
    by_case: dict[str, list[VerdictResult]] = {}
    for result in verdicts:
        by_case.setdefault(result.case_name, []).append(result)

    cases_by_name = {case.name: case for case in ALL_CASES}
    attacks_caught = attacks_total = controls_passed = controls_total = 0
    unstable: list[str] = []

    for name, runs in by_case.items():
        case = cases_by_name[name]
        correct = sum(1 for run in runs if run.correct)
        outcomes = {run.outcome for run in runs}
        if len(outcomes) > 1:
            unstable.append(f"{name}: {sorted(outcomes)}")
        if case.expected_grounded:
            controls_total += len(runs)
            controls_passed += correct
        else:
            attacks_total += len(runs)
            attacks_caught += correct
        print(f"  {name:<45} {correct}/{len(runs)} correct  {sorted(outcomes)}")

    print(f"\n  attacks caught:   {attacks_caught}/{attacks_total}")
    print(f"  controls passed:  {controls_passed}/{controls_total}")
    print(f"  unstable cases:   {len(unstable)}")
    for line in unstable:
        print(f"    {line}")

    print("\n=== LATENCY ===")
    for path in ("ask", "proactive"):
        rows = [row for row in latencies if row.path == path and row.synthesis_succeeded]
        if not rows:
            print(f"  {path}: no successful runs")
            continue
        before = [row.before_seconds for row in rows]
        after = [row.after_seconds for row in rows]
        check = [row.grounding_seconds for row in rows]
        print(
            f"  {path:<10} n={len(rows)}  before median {statistics.median(before):.2f}s "
            f"(max {max(before):.2f}s)  ->  after median {statistics.median(after):.2f}s "
            f"(max {max(after):.2f}s)   check alone median {statistics.median(check):.2f}s "
            f"(max {max(check):.2f}s)"
        )


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run", action="store_true", help="estimate cost and make no calls"
    )
    parser.add_argument(
        "--output",
        default="reports/grounding-verification.json",
        help="where to write the raw results",
    )
    args = parser.parse_args()

    if not os.environ.get(RUN_REAL_LLM_ENV):
        print(
            f"Refusing to run: {RUN_REAL_LLM_ENV} is not set. This script makes "
            "real, paid LLM calls."
        )
        return 1

    settings = _grounding_settings(load_settings())
    if settings.llm_api_key is None:
        print("Refusing to run: no LLM_API_KEY configured.")
        return 1

    print("Grounding-check verification")
    print(f"  cases: {len(ALL_CASES)} x {_RUNS_PER_CASE} runs")
    print(f"  latency scenarios: {len(_LATENCY_SCENARIOS)} x 2 paths")
    estimate = _estimate_cost(settings)
    print(f"  estimated cost: ${estimate:.4f}")

    if args.dry_run:
        print("\nDry run: no calls made.")
        return 0

    print("\nRunning verdict cases...")
    verdicts = await _run_verdict_cases(settings)
    print("\nRunning latency measurement...")
    latencies = await _run_latency(settings)

    _summarize(verdicts, latencies)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "grounding_model": settings.resolve_model(ModelComponent.GROUNDING_CHECK),
                "synthesis_model": settings.resolve_model(ModelComponent.SYNTHESIS),
                "proactive_model": settings.resolve_model(ModelComponent.PROACTIVE),
                "runs_per_case": _RUNS_PER_CASE,
                "verdicts": [asdict(result) for result in verdicts],
                "latencies": [asdict(result) for result in latencies],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\nRaw results written to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
