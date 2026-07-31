"""Measure the Phase 3a-2 distillation call against real, paid model calls.

    AURA_RUN_REAL_LLM=1 .venv/bin/python scripts/evaluate_extraction.py --dry-run
    AURA_RUN_REAL_LLM=1 .venv/bin/python scripts/evaluate_extraction.py

THIS SPENDS REAL MONEY. Same structural separation as scripts/model_bakeoff.py:
it is deliberately not a pytest test and does not live under tests/, pytest.ini
pins `testpaths = tests` so a bare `pytest` cannot reach it, and it refuses to
run at all unless AURA_RUN_REAL_LLM is set -- the one signal tests/conftest.py
keys its hermetic guard off, set by nothing but a human who means it.

Why a script rather than an opt-in test, same reasoning as the bake-off: this
is a measurement of a third-party model's behaviour, not a pass/fail assertion
about Aura's code. Expressing it as a test would mean either pinning a specific
model's current quality into the suite or asserting nothing at all.

It drives the real production distill_facts, not a copy of its prompt, so what
gets measured is the schema, system prompt and validation that actually ship.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from extraction_eval_cases import ALL_BATCHES, EvalBatch, section_for  # noqa: E402

from aura.config import ModelComponent, load_settings  # noqa: E402
from aura.db.extraction_queue import QueuedMessage  # noqa: E402
from aura.extraction.distiller import DistilledFact, distill_facts  # noqa: E402

RUN_REAL_LLM_ENV = "AURA_RUN_REAL_LLM"

# Live OpenRouter pricing for the shipped EXTRACTION_MODEL, per million tokens.
# Re-checked at the time of writing rather than carried from Phase 2, since the
# bake-off found one of its own price assumptions had gone stale.
_PRICE_IN_PER_MTOK = 1.00
_PRICE_OUT_PER_MTOK = 5.00

# Rough token counts for the estimate only; the run reports actual usage.
_SYSTEM_PROMPT_TOKENS = 900
_TOKENS_PER_MESSAGE = 40
_OUTPUT_TOKENS_PER_MESSAGE = 35

# A fixed batch timestamp so relative-time resolution ("tomorrow", "at 2") has
# a stable anchor and the report's before/after examples stay reproducible.
_BATCH_TIME = datetime(2026, 7, 30, 11, 0, 0, tzinfo=timezone.utc)


@dataclass
class MessageOutcome:
    """What the model did about one message in one run of one batch."""

    index: int
    text: str
    expect_fact: bool
    note: str
    extracted: list[DistilledFact] = field(default_factory=list)

    @property
    def got_fact(self) -> bool:
        return bool(self.extracted)

    @property
    def correct(self) -> bool:
        return self.got_fact == self.expect_fact


@dataclass
class RunResult:
    """One execution of one batch."""

    batch: EvalBatch
    run_index: int
    outcomes: list[MessageOutcome]
    call_failed: bool
    latency_seconds: float
    forbidden_hits: list[tuple[str, str]] = field(default_factory=list)

    @property
    def correct_count(self) -> int:
        return sum(1 for outcome in self.outcomes if outcome.correct)


def _to_queued(batch: EvalBatch) -> list[QueuedMessage]:
    """Turn an evaluation batch into what the pipeline would hand the distiller."""
    return [
        QueuedMessage(
            channel_id=500,
            # Distinct, stable IDs so the number -> ID mapping is checkable.
            message_id=1000 + index,
            guild_id=100,
            channel_name=batch.channel_name,
            content=message.text,
            # Spread across the batch window, as real messages would be.
            message_created_at=_BATCH_TIME + timedelta(minutes=index),
            enqueued_at=_BATCH_TIME + timedelta(minutes=index),
        )
        for index, message in enumerate(batch.messages)
    ]


async def run_batch(batch: EvalBatch, run_index: int, model: str) -> RunResult:
    """Run one batch once against the real model. Never raises."""
    queued = _to_queued(batch)
    started = time.monotonic()
    distilled = await distill_facts(queued, channel_name=batch.channel_name, model=model)
    latency = time.monotonic() - started

    outcomes = [
        MessageOutcome(
            index=index + 1,
            text=message.text,
            expect_fact=message.expect_fact,
            note=message.note,
        )
        for index, message in enumerate(batch.messages)
    ]

    if distilled is None:
        # A failed or untrusted call. Reported as its own outcome rather than
        # scored as "rejected everything", which would flatter it.
        return RunResult(
            batch=batch,
            run_index=run_index,
            outcomes=outcomes,
            call_failed=True,
            latency_seconds=latency,
        )

    by_message_id = {message.message_id: index for index, message in enumerate(queued)}
    for fact in distilled:
        position = by_message_id.get(fact.message_id)
        if position is not None:
            outcomes[position].extracted.append(fact)

    forbidden_hits = [
        (fact.content, needle)
        for fact in distilled
        for needle in batch.forbidden_substrings
        if needle.lower() in fact.content.lower()
    ]

    return RunResult(
        batch=batch,
        run_index=run_index,
        outcomes=outcomes,
        call_failed=False,
        latency_seconds=latency,
        forbidden_hits=forbidden_hits,
    )


def estimate_cost() -> tuple[int, float]:
    """Return (call count, estimated USD) for a full run, before spending anything."""
    calls = 0
    cost = 0.0
    for batch in ALL_BATCHES:
        message_count = len(batch.messages)
        tokens_in = _SYSTEM_PROMPT_TOKENS + message_count * _TOKENS_PER_MESSAGE
        tokens_out = message_count * _OUTPUT_TOKENS_PER_MESSAGE
        per_call = (
            tokens_in / 1_000_000 * _PRICE_IN_PER_MTOK
            + tokens_out / 1_000_000 * _PRICE_OUT_PER_MTOK
        )
        calls += batch.repeats
        cost += per_call * batch.repeats
    return calls, cost


def _print_dry_run() -> None:
    calls, cost = estimate_cost()
    print("DRY RUN -- nothing was spent.\n")
    print(f"  batches           {len(ALL_BATCHES)}")
    print(f"  calls (w/ repeats){calls:>6}")
    print(f"  messages          {sum(len(b.messages) * b.repeats for b in ALL_BATCHES):>6}")
    print(f"  estimated cost    ${cost:.4f}")
    print("\nPer section:")
    per_section: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for batch in ALL_BATCHES:
        entry = per_section[section_for(batch.name)]
        entry[0] += batch.repeats
        entry[1] += len(batch.messages) * batch.repeats
    for section, (calls_, messages_) in per_section.items():
        print(f"  {section:<28} {calls_:>3} call(s), {messages_:>4} message(s)")


def _summarize(results: list[RunResult]) -> dict[str, dict[str, int]]:
    """Aggregate correctness per report section."""
    summary: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "runs": 0,
            "failed_calls": 0,
            "messages": 0,
            "correct": 0,
            "false_positives": 0,
            "false_negatives": 0,
            "forbidden_hits": 0,
        }
    )
    for result in results:
        entry = summary[section_for(result.batch.name)]
        entry["runs"] += 1
        entry["forbidden_hits"] += len(result.forbidden_hits)
        if result.call_failed:
            entry["failed_calls"] += 1
            continue
        for outcome in result.outcomes:
            entry["messages"] += 1
            if outcome.correct:
                entry["correct"] += 1
            elif outcome.got_fact:
                entry["false_positives"] += 1
            else:
                entry["false_negatives"] += 1
    return dict(summary)


def _print_report(results: list[RunResult]) -> None:
    print("\n" + "=" * 78)
    print("PER-SECTION SUMMARY")
    print("=" * 78)
    summary = _summarize(results)
    for section, entry in summary.items():
        scored = entry["messages"]
        accuracy = entry["correct"] / scored if scored else 0.0
        print(
            f"\n{section}\n"
            f"  runs {entry['runs']}, failed calls {entry['failed_calls']}\n"
            f"  messages {scored}, correct {entry['correct']} ({accuracy:.1%})\n"
            f"  false positives {entry['false_positives']} "
            f"(recorded something it should not have)\n"
            f"  false negatives {entry['false_negatives']} "
            f"(missed a real fact)\n"
            f"  forbidden-substring hits {entry['forbidden_hits']}"
        )

    print("\n" + "=" * 78)
    print("EVERY DISAGREEMENT, IN FULL")
    print("=" * 78)
    any_wrong = False
    for result in results:
        if result.call_failed:
            any_wrong = True
            print(f"\n[{result.batch.name} run {result.run_index}] CALL FAILED")
            continue
        wrong = [outcome for outcome in result.outcomes if not outcome.correct]
        if not wrong and not result.forbidden_hits:
            continue
        any_wrong = True
        print(f"\n[{result.batch.name} run {result.run_index}]")
        for outcome in wrong:
            direction = "FALSE POSITIVE" if outcome.got_fact else "FALSE NEGATIVE"
            print(f"  {direction}: {outcome.text}")
            print(f"    ({outcome.note})")
            for fact in outcome.extracted:
                print(f"    -> [{fact.category.value}] {fact.content}")
        for content, needle in result.forbidden_hits:
            label = "CONTEXT BLEED" if "context-bleed" in result.batch.name else "FORBIDDEN SUBSTRING"
            print(f"  {label}: {needle!r} appeared in: {content}")
    if not any_wrong:
        print("\n  none.")

    print("\n" + "=" * 78)
    print("DISTILLATION QUALITY & QUOTE HANDLING: BEFORE / AFTER")
    print("=" * 78)
    for result in results:
        is_quality = result.batch.name.startswith("distillation-quality")
        is_quote = result.batch.name.startswith("quote-hazard")
        if not (is_quality or is_quote) or result.call_failed:
            continue
        print(f"\n[{result.batch.name}]")
        for outcome in result.outcomes:
            for fact in outcome.extracted:
                verbatim = " <<< VERBATIM COPY" if fact.content.strip() == outcome.text.strip() else ""
                print(f"  before: {outcome.text}")
                print(f"  after : {fact.content}  [{fact.category.value}]{verbatim}")
                print()


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the cost estimate and exit without spending anything",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("reports/phase-3a-2-evaluation.json"),
        help="where to write the raw per-case results",
    )
    parser.add_argument(
        "--only",
        default=None,
        help=(
            "run only batches whose name starts with this prefix. Exists so a "
            "prompt fix can be re-verified against the batches that failed "
            "without re-running (and re-paying for) the whole set."
        ),
    )
    args = parser.parse_args()

    global ALL_BATCHES  # noqa: PLW0603 -- one filter applied before anything runs
    if args.only:
        ALL_BATCHES = tuple(b for b in ALL_BATCHES if b.name.startswith(args.only))
        if not ALL_BATCHES:
            print(f"No batches match {args.only!r}", file=sys.stderr)
            return 1

    if args.dry_run:
        _print_dry_run()
        return 0

    if not os.environ.get(RUN_REAL_LLM_ENV):
        print(
            f"Refusing to run: set {RUN_REAL_LLM_ENV}=1 to make real, paid model calls.\n"
            "Use --dry-run to see the cost estimate first.",
            file=sys.stderr,
        )
        return 1

    settings = load_settings()
    model = settings.resolve_model(ModelComponent.EXTRACTION)
    if not settings.is_llm_configured(ModelComponent.EXTRACTION) or model is None:
        print("Refusing to run: no extraction model or API key configured.", file=sys.stderr)
        return 1

    calls, estimated = estimate_cost()
    print(f"Running {calls} call(s) against {model} (estimated ${estimated:.4f})\n")

    results: list[RunResult] = []
    for batch in ALL_BATCHES:
        for run_index in range(1, batch.repeats + 1):
            result = await run_batch(batch, run_index, model)
            results.append(result)
            status = "FAILED" if result.call_failed else f"{result.correct_count}/{len(result.outcomes)}"
            print(
                f"  {batch.name:<34} run {run_index}  {status:>8}  "
                f"{result.latency_seconds:5.2f}s"
            )

    _print_report(results)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "model": model,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "estimated_cost_usd": estimated,
                "summary": _summarize(results),
                "runs": [
                    {
                        "batch": result.batch.name,
                        "locale": result.batch.locale,
                        "run": result.run_index,
                        "call_failed": result.call_failed,
                        "latency_seconds": result.latency_seconds,
                        "forbidden_hits": result.forbidden_hits,
                        "messages": [
                            {
                                "index": outcome.index,
                                "text": outcome.text,
                                "expect_fact": outcome.expect_fact,
                                "note": outcome.note,
                                "correct": outcome.correct,
                                "extracted": [
                                    {"content": fact.content, "category": fact.category.value}
                                    for fact in outcome.extracted
                                ],
                            }
                            for outcome in result.outcomes
                        ],
                    }
                    for result in results
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
