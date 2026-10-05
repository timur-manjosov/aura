"""The P5 evaluation: real, paid runs of the background functions on invented cases.

    AURA_RUN_REAL_LLM=1 .venv/bin/python scripts/p5_bakeoff.py --ledger <file> \\
        --bucket <dev|main|checks|reserve> --label <label> --out-dir <dir> [--dry-run] \\
        <task> --arm <key> [--runs N] [--cases a,b] [--slice all|dev|heldout] [task options]

THIS SPENDS REAL MONEY, under the rules of every paid harness here: not a pytest
test, refused unless AURA_RUN_REAL_LLM is set, and every call metered by
scripts/llm_metering.py against the bucket's and the total ceiling BEFORE the
request leaves (P5: dev 4, main 14, checks 5, reserve 2, total 25 USD).

It drives the SHIPPED code: fact extraction is
aura.extraction.distiller.distill_facts, the verification
aura.extraction.verifier.verify_distilled_facts, the supersession judge
aura.extraction.supersession.judge_relationship, proactive relief's legacy
decision aura.synthesis.synthesize_answer and its v2 decision
aura.answer_contract.synthesize_contract_answer (with `proactive_posted_at` for
the proactive variant), the checks aura.grounding and aura.answer_check. What an
arm changes is only the model and the request options of scripts/bakeoff_arms.py.

Results are one JSON line per call in <out-dir>/<task>-<arm>-<label>.jsonl; a
stopped run resumes with the same arguments and skips every tag already written.

Tasks:

  extraction    distillation over scripts/p5_extraction_cases.py
  verify        the verification (the arm as verifier) over the stored facts of
                an extraction results file (--source)
  supersession  the judge over scripts/p5_supersession_cases.py
  proactive     the decision over scripts/p5_proactive_cases.py after the shipped
                Stage 1 / Stage 2 scoring (local, free); --format legacy, v2
                (the /aura-ask contract) or v2p (the proactive variant)
  pcheck        the answer check (the arm as checker) over every would-be post of
                a proactive results file (--source): the v2 check for v2/v2p
                rows, the legacy grounding check for legacy rows
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from aura.config import load_settings
from aura.db.models import Fact, FactStatus
from bakeoff_arms import ARMS
from llm_metering import RUN_REAL_LLM_ENV, Ledger
from p4_bakeoff import RunContext, _done_tags, _estimate, _install, _record_fields, run_jobs
from p5_extraction_cases import ExtractionCase, all_cases
from p5_proactive_cases import MESSAGE_TIME, all_messages, all_scenarios
from p5_supersession_cases import all_pairs

BUCKET_CEILINGS: Final[dict[str, float]] = {
    "dev": 4.00,
    "main": 14.00,
    "checks": 5.00,
    "reserve": 2.00,
}
TOTAL_CEILING: Final = 25.00

Job = Callable[[], Awaitable[dict[str, Any]]]

_GUILD_ID: Final = 1
_FACT_TIME: Final = datetime(2026, 8, 1, 12, 0, tzinfo=UTC)


def _selected(name: str, is_dev: bool, opts: argparse.Namespace) -> bool:
    if opts.wanted and name not in opts.wanted:
        return False
    if opts.slice == "dev":
        return is_dev
    if opts.slice == "heldout":
        return not is_dev
    return True


def queued_batch(case: ExtractionCase) -> list[Any]:
    """Return a case's messages as the distiller receives them from the queue."""
    from aura.db.extraction_queue import QueuedMessage

    return [
        QueuedMessage(
            channel_id=500,
            message_id=1000 + index,
            guild_id=100,
            channel_name=case.channel,
            content=chat.text,
            message_created_at=case.timestamp(index),
            enqueued_at=case.timestamp(index),
        )
        for index, chat in enumerate(case.messages, start=1)
    ]


def extraction_jobs(ctx: RunContext, opts: argparse.Namespace) -> list[tuple[str, Job]]:
    """Return one job per batch and run: the shipped distillation."""
    from aura.extraction.distiller import distill_facts

    jobs: list[tuple[str, Job]] = []
    for run_index in range(1, opts.runs + 1):
        for case in all_cases():
            if not _selected(case.name, case.is_dev, opts):
                continue
            tag = f"p5-extraction:{ctx.arm.key}:{case.name}:run{run_index}"

            async def job(
                case: ExtractionCase = case, run_index: int = run_index, tag: str = tag
            ) -> dict[str, Any]:
                batch = queued_batch(case)
                distilled = await distill_facts(
                    batch, channel_name=case.channel, model=ctx.arm.model
                )
                return {
                    "case": case.name,
                    "locale": case.locale,
                    "dev": case.is_dev,
                    "run": run_index,
                    "call_failed": distilled is None,
                    "facts": [
                        {
                            "message": fact.message_id - 1000,
                            "content": fact.content,
                            "category": fact.category.value,
                        }
                        for fact in (distilled or [])
                    ],
                    **_record_fields(ctx.records.get(tag)),
                }

            jobs.append((tag, job))
    return jobs


def verify_jobs(ctx: RunContext, opts: argparse.Namespace) -> list[tuple[str, Job]]:
    """Return one job per non-empty extraction result in --source: the verification."""
    from aura.db.pending_facts import FactCategory
    from aura.extraction.distiller import DistilledFact
    from aura.extraction.verifier import verify_distilled_facts

    cases = {case.name: case for case in all_cases()}
    settings = ctx.settings
    jobs: list[tuple[str, Job]] = []
    source = Path(opts.source)
    for line in source.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("call_failed") or not row.get("facts"):
            continue
        case = cases[row["case"]]
        if not _selected(case.name, case.is_dev, opts):
            continue
        tag = f"p5-verify:{ctx.arm.key}:{row['arm']}:{case.name}:run{row['run']}"

        async def job(
            row: dict[str, Any] = row, case: ExtractionCase = case, tag: str = tag
        ) -> dict[str, Any]:
            batch = queued_batch(case)
            distilled = [
                DistilledFact(
                    message_id=1000 + int(fact["message"]),
                    content=fact["content"],
                    category=FactCategory(fact["category"]),
                )
                for fact in row["facts"]
            ]
            kept = await verify_distilled_facts(
                batch, distilled, channel_name=case.channel, model=ctx.arm.model, settings=settings
            )
            return {
                "case": case.name,
                "source_arm": row["arm"],
                "dev": case.is_dev,
                "run": row["run"],
                "verify_failed": kept is None,
                "proposed": row["facts"],
                "facts": [
                    {
                        "message": fact.message_id - 1000,
                        "content": fact.content,
                        "category": fact.category.value,
                    }
                    for fact in (kept or [])
                ],
                **_record_fields(ctx.records.get(tag)),
            }

        jobs.append((tag, job))
    return jobs


def supersession_jobs(ctx: RunContext, opts: argparse.Namespace) -> list[tuple[str, Job]]:
    """Return one job per pair and run: the shipped judge."""
    from aura.extraction.supersession import judge_relationship

    jobs: list[tuple[str, Job]] = []
    for run_index in range(1, opts.runs + 1):
        for pair in all_pairs():
            import hashlib

            is_dev = int(hashlib.sha256(pair.name.encode()).hexdigest(), 16) % 4 == 0
            if not _selected(pair.name, is_dev, opts):
                continue
            tag = f"p5-supersession:{ctx.arm.key}:{pair.name}:run{run_index}"

            async def job(
                pair: Any = pair, run_index: int = run_index, tag: str = tag, is_dev: bool = is_dev
            ) -> dict[str, Any]:
                judgement = await judge_relationship(
                    predecessor=pair.predecessor, candidate=pair.candidate, model=ctx.arm.model
                )
                return {
                    "case": pair.name,
                    "expected": pair.category,
                    "shape": pair.shape,
                    "boundary": pair.boundary,
                    "dev": is_dev,
                    "run": run_index,
                    "category": judgement.relationship.value if judgement else None,
                    "change_signal": judgement.change_signal if judgement else None,
                    "shared_subject": judgement.shared_subject if judgement else None,
                    "reasoning": judgement.reasoning if judgement else None,
                    **_record_fields(ctx.records.get(tag)),
                }

            jobs.append((tag, job))
    return jobs


def _proactive_prefilter(ctx: RunContext, out_dir: Path) -> dict[str, dict[str, Any]]:
    """Return, per message, the shipped Stage 1 / Stage 2 verdicts and the facts it would get (free)."""
    cache = out_dir / "p5-proactive-prefilter.json"
    if cache.exists():
        loaded: dict[str, dict[str, Any]] = json.loads(cache.read_text(encoding="utf-8"))
        return loaded
    from fastembed import TextEmbedding

    from aura.embeddings import SYNTHESIS_FACT_LIMIT, cosine_similarity, embed_text, embed_texts
    from aura.proactive.question_detector import QuestionDetector

    async def compute() -> dict[str, dict[str, Any]]:
        model = TextEmbedding(ctx.settings.embedding_model)
        detector = await QuestionDetector.create(model)
        vectors: dict[str, list[Any]] = {}
        result: dict[str, dict[str, Any]] = {}
        for scenario, index, message in all_messages():
            if scenario.key not in vectors:
                vectors[scenario.key] = await embed_texts(
                    model, [text for _, text in scenario.facts]
                )
            stage1 = await detector.question_likeness(message.text)
            vector = await embed_text(model, message.text)
            scores = [
                (number, cosine_similarity(vector, fact_vector))
                for number, fact_vector in enumerate(vectors[scenario.key], start=1)
            ]
            ranked = sorted(scores, key=lambda pair: (-pair[1], pair[0]))
            threshold = ctx.settings.proactive_similarity_threshold
            result[f"{scenario.key}-{index:02d}"] = {
                "stage1": stage1,
                "stage1_passed": stage1 >= ctx.settings.proactive_question_threshold,
                "top_similarity": ranked[0][1],
                "stage2_passed": ranked[0][1] >= threshold,
                "facts": [n for n, score in ranked[:SYNTHESIS_FACT_LIMIT] if score >= threshold],
            }
        return result

    computed = asyncio.run(compute())
    out_dir.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(computed, ensure_ascii=False, indent=1), encoding="utf-8")
    return computed


def _scenario_facts(scenario: Any, numbers: list[int]) -> tuple[list[Fact], dict[int, str]]:
    channel_ids = {name: 7000 + i for i, name in enumerate(sorted({c for c, _ in scenario.facts}))}
    facts = [
        Fact(
            id=number,
            guild_id=_GUILD_ID,
            channel_id=channel_ids[scenario.facts[number - 1][0]],
            message_id=1000 + number,
            content=scenario.facts[number - 1][1],
            embedding=b"",
            status=FactStatus.ACTIVE,
            created_at=_FACT_TIME,
        )
        for number in numbers
    ]
    return facts, {cid: name for name, cid in channel_ids.items()}


def proactive_jobs(ctx: RunContext, opts: argparse.Namespace) -> list[tuple[str, Job]]:
    """Return one job per gate-passing message and run: the format's decision."""
    from aura.answer_card import build_answer_card
    from aura.answer_contract import synthesize_contract_answer
    from aura.synthesis import synthesize_answer

    prefilter = _proactive_prefilter(ctx, opts.out_dir)
    jobs: list[tuple[str, Job]] = []
    import hashlib

    for run_index in range(1, opts.runs + 1):
        for scenario, index, message in all_messages():
            key = f"{scenario.key}-{index:02d}"
            gate = prefilter[key]
            is_dev = int(hashlib.sha256(key.encode()).hexdigest(), 16) % 4 == 0
            if not (gate["stage1_passed"] and gate["stage2_passed"]) or not gate["facts"]:
                continue
            if not _selected(key, is_dev, opts):
                continue
            tag = f"p5-proactive-{opts.format}:{ctx.arm.key}:{key}:run{run_index}"

            async def job(
                scenario: Any = scenario,
                message: Any = message,
                key: str = key,
                gate: dict[str, Any] = gate,
                run_index: int = run_index,
                tag: str = tag,
                is_dev: bool = is_dev,
            ) -> dict[str, Any]:
                facts, names = _scenario_facts(scenario, gate["facts"])
                locale = scenario.locale
                detail: dict[str, Any]
                if opts.format in ("v2", "v2p"):
                    answer = await synthesize_contract_answer(
                        facts,
                        message.text,
                        locale,
                        model=ctx.arm.model,
                        settings=ctx.settings,
                        proactive_posted_at=MESSAGE_TIME if opts.format == "v2p" else None,
                    )
                    decision = answer is not None and answer.answers_unprompted
                    detail = {
                        "failed": answer is None,
                        "answers_question": answer.answers_question if answer else None,
                        "message_kind": answer.message_kind.value
                        if answer and answer.message_kind
                        else None,
                        "used": list(answer.used_fact_ids) if answer else [],
                        "relations": [r.kind.value for r in answer.relations] if answer else [],
                        "lead": answer.lead if answer else None,
                        "points": [p.text for p in answer.points] if answer else [],
                    }
                    if answer is not None and answer.used_fact_ids:
                        card = build_answer_card(
                            answer,
                            facts,
                            question=None,
                            locale=locale,
                            channel_names=names,
                            proactive=True,
                        )
                        detail["checked_lead"] = card.checked_lead
                        detail["checked_points"] = [
                            {"text": p.text, "fact_ids": list(p.fact_ids)}
                            for p in card.checked_points
                        ]
                        detail["cited_fact_ids"] = list(card.cited_fact_ids)
                else:
                    result = await synthesize_answer(
                        facts,
                        message.text,
                        locale,
                        model=ctx.arm.model,
                        question_channel_name=scenario.channel,
                        question_asked_at=MESSAGE_TIME,
                        fact_channel_names=names,
                    )
                    decision = (
                        result is not None
                        and result.answers_question
                        and bool(result.used_fact_ids)
                    )
                    detail = {
                        "failed": result is None,
                        "answers_question": result.answers_question if result else None,
                        "used": list(result.used_fact_ids) if result else [],
                        "answer": result.answer if result else None,
                    }
                return {
                    "case": key,
                    "scenario": scenario.key,
                    "locale": scenario.locale,
                    "text": message.text,
                    "category": message.category,
                    "should_post": message.should_post,
                    "human_reply": message.human_reply,
                    "dev": is_dev,
                    "run": run_index,
                    "format": opts.format,
                    "facts_given": gate["facts"],
                    "posts": decision,
                    **detail,
                    **_record_fields(ctx.records.get(tag)),
                }

            jobs.append((tag, job))
    return jobs


def pcheck_jobs(ctx: RunContext, opts: argparse.Namespace) -> list[tuple[str, Job]]:
    """Return one job per would-be post in --source: the check production would run next."""
    from aura.answer_check import build_statements, verify_answer_v2
    from aura.grounding import PROACTIVE_GROUNDING_TIMEOUT_SECONDS, verify_answer_grounded

    scenarios = {scenario.key: scenario for scenario in all_scenarios()}
    v2_settings = ctx.settings.model_copy(update={"answer_v2_check_model": ctx.arm.model})
    legacy_settings = ctx.settings.model_copy(update={"grounding_check_model": ctx.arm.model})
    jobs: list[tuple[str, Job]] = []
    for line in Path(opts.source).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not row.get("posts") or row.get("human_reply"):
            continue
        tag = f"p5-pcheck:{ctx.arm.key}:{row['arm']}:{row['format']}:{row['case']}:run{row['run']}"

        async def job(row: dict[str, Any] = row, tag: str = tag) -> dict[str, Any]:
            scenario = scenarios[row["scenario"]]
            facts, _ = _scenario_facts(scenario, row["facts_given"])
            by_id = {fact.id: fact for fact in facts}
            if row["format"] in ("v2", "v2p"):
                statements = build_statements(
                    row["checked_lead"],
                    [(p["text"], tuple(p["fact_ids"])) for p in row["checked_points"]],
                    tuple(row["cited_fact_ids"]),
                )
                outcome = await verify_answer_v2(
                    statements,
                    [by_id[i] for i in row["cited_fact_ids"]],
                    settings=v2_settings,
                    timeout_seconds=PROACTIVE_GROUNDING_TIMEOUT_SECONDS,
                )
            else:
                outcome = await verify_answer_grounded(
                    answer=row["answer"],
                    cited_facts=[by_id[i] for i in row["used"]],
                    settings=legacy_settings,
                    timeout_seconds=PROACTIVE_GROUNDING_TIMEOUT_SECONDS,
                )
            return {
                "case": row["case"],
                "source_arm": row["arm"],
                "format": row["format"],
                "run": row["run"],
                "outcome": outcome.value,
                **_record_fields(ctx.records.get(tag)),
            }

        jobs.append((tag, job))
    return jobs


TASKS: Final[dict[str, Callable[[RunContext, argparse.Namespace], list[tuple[str, Job]]]]] = {
    "extraction": extraction_jobs,
    "verify": verify_jobs,
    "supersession": supersession_jobs,
    "proactive": proactive_jobs,
    "pcheck": pcheck_jobs,
}

_ESTIMATE_TASK: Final = {
    "extraction": "extraction",
    "verify": "extraction",
    "supersession": "supersession",
    "proactive": "synth",
    "pcheck": "v2check",
}


# Upstream refusals the harness retries (the shipped code never does: in
# production one such refusal drops the batch). Every attempt is metered on its
# own; the count per run is printed so the report can state availability.
_RETRY_DELAYS_SECONDS: Final = (3.0, 8.0, 20.0, 45.0)
RETRIES: dict[str, int] = {}


def _install_retry() -> Callable[[], None]:
    """Wrap the (already metered) litellm.acompletion with retries on upstream 429s."""
    import litellm

    metered = litellm.acompletion

    async def retrying(*args: Any, **kwargs: Any) -> Any:
        from llm_metering import CURRENT_TAG

        for delay in (*_RETRY_DELAYS_SECONDS, None):
            try:
                return await metered(*args, **kwargs)
            except litellm.RateLimitError:
                if delay is None:
                    raise
                tag = CURRENT_TAG.get()
                RETRIES[tag] = RETRIES.get(tag, 0) + 1
                await asyncio.sleep(delay)
        raise AssertionError("unreachable")

    litellm.acompletion = retrying

    def restore() -> None:
        litellm.acompletion = metered

    return restore


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--bucket", choices=sorted(BUCKET_CEILINGS), required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("task", choices=sorted(TASKS))
    parser.add_argument("--arm", choices=sorted(ARMS), required=True)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--cases", default="")
    parser.add_argument("--slice", choices=("all", "dev", "heldout"), default="all")
    parser.add_argument("--format", choices=("legacy", "v2", "v2p"), default="legacy")
    parser.add_argument("--source", default="")
    args = parser.parse_args()
    args.wanted = {name for name in args.cases.split(",") if name}
    return args


def main() -> int:
    """Run one task for one arm; see the module docstring."""
    args = _parse_args()
    if not os.environ.get(RUN_REAL_LLM_ENV):
        print(f"Refusing to spend money: set {RUN_REAL_LLM_ENV}=1 to run this harness.")
        return 2
    arm = ARMS[args.arm]
    ledger = Ledger(args.ledger, BUCKET_CEILINGS, TOTAL_CEILING)
    settings = load_settings()
    ctx = RunContext(arm=arm, settings=settings, records={}, ledger=ledger)
    jobs = TASKS[args.task](ctx, args)
    name = f"{args.task}-{args.format}" if args.task == "proactive" else args.task
    out_path = args.out_dir / f"{name}-{arm.key}-{args.label}.jsonl"
    done = _done_tags(out_path)
    todo = len([tag for tag, _ in jobs if tag not in done])
    state = ledger.snapshot()
    print(
        f"{name} x {arm.key}: {len(jobs)} job(s), {todo} to run, estimate "
        f"${_estimate(arm, _ESTIMATE_TASK[args.task], todo):.3f}; bucket {args.bucket} "
        f"${state.spent(args.bucket):.4f}/{BUCKET_CEILINGS[args.bucket]:.2f}, total "
        f"${state.spent():.4f}/{TOTAL_CEILING:.2f}",
        flush=True,
    )
    if args.dry_run:
        return 0
    restore = _install(ctx, bucket=args.bucket, label=args.label)
    restore_retry = _install_retry()
    try:
        ran, skipped, stopped = asyncio.run(
            run_jobs(jobs, ctx=ctx, out_path=out_path, concurrency=args.concurrency)
        )
    finally:
        restore_retry()
        restore()
    state = ledger.snapshot()
    print(
        f"ran {ran}, skipped {skipped}{', STOPPED BY CEILING' if stopped else ''}; "
        f"upstream 429 retries {sum(RETRIES.values())} on {len(RETRIES)} job(s); bucket "
        f"{args.bucket} ${state.spent(args.bucket):.4f}, total ${state.spent():.4f}",
        flush=True,
    )
    return 3 if stopped else 0


if __name__ == "__main__":
    raise SystemExit(main())
