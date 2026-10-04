"""The P4 model bake-off: real, paid runs of every LLM function on invented cases.

    AURA_RUN_REAL_LLM=1 .venv/bin/python scripts/p4_bakeoff.py --ledger <file> \\
        --bucket bakeoff --label <label> --out-dir <dir> [--dry-run] \\
        <task> --arm <key> [--runs N] [--cases a,b,c] [task options]

THIS SPENDS REAL MONEY, under the rules of every paid harness here: not a pytest
test, refused unless AURA_RUN_REAL_LLM is set, and every call metered by
scripts/llm_metering.py against the bucket's and the total ceiling BEFORE the
request leaves, with one ledger file shared by every invocation of the task.

It drives the SHIPPED code, never a copy of a prompt: the v2 answer is
aura.answer_contract.synthesize_contract_answer, the legacy answer and the
proactive decision aura.synthesis.synthesize_answer, the legacy check
aura.grounding.verify_answer_grounded, the v2 check aura.answer_check, fact
extraction aura.extraction.distiller.distill_facts and the supersession judge
aura.extraction.supersession.judge_relationship. What an arm changes is only the
model and the request options of scripts/bakeoff_arms.py (provider pinning,
reasoning, the output ceiling of a reasoning model); a prompt variant, where a
task offers one, replaces exactly one prompt constant for that run.

Results are one JSON line per call in <out-dir>/<task>-<arm>-<label>.jsonl. A
run that stops -- a ceiling, a crash, Ctrl-C -- can be started again with the
same arguments: every tag already in the file is skipped, so nothing is paid
for twice. A call the ceiling refused leaves no line, and the run stops.

Tasks:

  synth        the v2 contract over scripts/answer_contract_cases.py (A)
  legacy       the legacy synthesis over the same cases, the baseline (A)
  checker      the legacy grounding check over the 134-case corpus of
               scripts/grounding_verification_cases.py, the arm as checker (B)
  proactive    the proactive decision over scripts/proactive_decision_cases.py:
               the shipped Stage 1 and Stage 2 scores are computed locally (free)
               and only the messages they let through reach the model, with the
               facts production would hand it (C); --format v2 measures the v2
               contract's answers_unprompted on the same messages instead
  extraction   fact extraction over scripts/extraction_eval_cases.py and
               scripts/extraction_german_cases.py (D)
  supersession the supersession judge over scripts/supersession_bakeoff_cases.py (E)
  throughput   --calls N v2 answers at --concurrency P, for 429s and latency
  earlycheck   today's grounding check (the arm as checker) over the plain
               text of every v2 answer in --source (C7)
  v2check      the v2 statement check (the arm as checker) over
               scripts/answer_check_cases.py, or over the v2 answers in
               --source (Part F); --check-variant swaps its prompt
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

# These scripts run as `python scripts/<name>.py`; put the repository's import
# roots on sys.path before anything below imports from them.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from answer_contract_cases import CASES, AnswerCase
from answer_contract_scoring import score_contract, score_legacy
from aura.answer_card import build_answer_card, plain_answer_text
from aura.answer_contract import FIELD_ORDER, synthesize_contract_answer, validate_contract
from aura.config import Settings, load_settings
from aura.db.models import Fact, FactStatus
from aura.synthesis import _parse_json_response, synthesize_answer
from bakeoff_arms import ARMS, Arm, prices
from llm_metering import (
    CURRENT_TAG,
    RUN_REAL_LLM_ENV,
    CallRecord,
    CeilingReachedError,
    Ledger,
    install_metering,
)

# The budget buckets of P4 and their ceilings (USD), and the total over all.
BUCKET_CEILINGS: Final[dict[str, float]] = {"dev": 3.00, "bakeoff": 17.00, "checker": 5.00}
TOTAL_CEILING: Final = 25.00

_GUILD_ID: Final = 1
_FACT_TIME: Final = datetime(2026, 8, 1, 12, 0, tzinfo=UTC)


@dataclass
class RunContext:
    """What every job of one run shares."""

    arm: Arm
    settings: Settings
    records: dict[str, CallRecord]
    ledger: Ledger


def case_facts(case: AnswerCase) -> tuple[list[Fact], dict[int, str]]:
    """Build a case's facts as production would hand them, and their channel names."""
    channel_ids: dict[str, int] = {}
    facts: list[Fact] = []
    for number, fact in enumerate(case.facts, start=1):
        channel_id = channel_ids.setdefault(fact.channel, 9000 + len(channel_ids))
        facts.append(
            Fact(
                id=number,
                guild_id=_GUILD_ID,
                channel_id=channel_id,
                message_id=1000 + number,
                content=fact.text,
                embedding=b"",
                status=FactStatus.ACTIVE,
                created_at=_FACT_TIME.replace(day=number),
            )
        )
    return facts, {channel_id: name for name, channel_id in channel_ids.items()}


def _record_fields(record: CallRecord | None) -> dict[str, Any]:
    if record is None:
        return {"called": False}
    return {
        "called": True,
        "raw": record.raw_content,
        "prompt_tokens": record.prompt_tokens,
        "completion_tokens": record.completion_tokens,
        "reasoning_tokens": record.reasoning_tokens,
        "finish_reason": record.finish_reason,
        "provider": record.provider,
        "seconds": round(record.seconds, 3),
        "usd": record.usd,
        "usd_source": record.usd_source,
    }


def _validation_error(
    raw: str | None, facts: list[Fact]
) -> tuple[dict[str, Any] | None, str | None]:
    """Return the parsed reply if it passes the contract, else None and why."""
    if not raw:
        return None, "empty"
    try:
        parsed = _parse_json_response(raw)
    except ValueError as exc:
        return None, f"json: {type(exc).__name__}"
    try:
        validate_contract(parsed, facts)
    except Exception as exc:
        return None, f"{type(exc).__name__}: {str(exc)[:200]}"
    return parsed if isinstance(parsed, dict) else None, None


# --- tasks -------------------------------------------------------------------


def synth_jobs(ctx: RunContext, opts: argparse.Namespace) -> list[tuple[str, Job]]:
    """Return one job per case and run: the v2 contract answer, scored."""
    jobs: list[tuple[str, Job]] = []
    for run_index in range(1, opts.runs + 1):
        for case in _answer_cases(opts):
            tag = f"synth:{ctx.arm.key}:{case.name}:run{run_index}"

            async def job(
                case: AnswerCase = case, run_index: int = run_index, tag: str = tag
            ) -> dict[str, Any]:
                facts, channel_names = case_facts(case)
                started = time.perf_counter()
                answer = await synthesize_contract_answer(
                    facts, case.question, case.locale, model=ctx.arm.model, settings=ctx.settings
                )
                wall = time.perf_counter() - started
                record = ctx.records.get(tag)
                parsed, error = _validation_error(record.raw_content if record else None, facts)
                if answer is None:
                    parsed = None
                    if error is None:
                        error = f"finish_reason={record.finish_reason}" if record else "no response"
                result: dict[str, Any] = {
                    "case": case.name,
                    "shape": case.shape,
                    "register": case.register,
                    "difficulty": case.difficulty,
                    "locale": case.locale,
                    "run": run_index,
                    "wall_seconds": round(wall, 3),
                    **_record_fields(record),
                    "contract": parsed,
                    "invalid_reason": error,
                    "field_order_kept": (
                        list(parsed.keys()) == list(FIELD_ORDER) if parsed else None
                    ),
                    "score": score_contract(case, parsed).as_dict(),
                }
                if answer is not None and answer.used_fact_ids:
                    card = build_answer_card(
                        answer,
                        facts,
                        question=case.question,
                        locale=case.locale,
                        channel_names=channel_names,
                    )
                    result["plain_text"] = plain_answer_text(card)
                    result["checked_lead"] = card.checked_lead
                    result["checked_points"] = [
                        {"text": point.text, "fact_ids": list(point.fact_ids)}
                        for point in card.checked_points
                    ]
                    result["cited_fact_ids"] = list(card.cited_fact_ids)
                    result["answers_unprompted"] = answer.answers_unprompted
                return result

            jobs.append((tag, job))
    return jobs


def legacy_jobs(ctx: RunContext, opts: argparse.Namespace) -> list[tuple[str, Job]]:
    """Return one job per case and run: the legacy answer, scored (the baseline)."""
    jobs: list[tuple[str, Job]] = []
    for run_index in range(1, opts.runs + 1):
        for case in _answer_cases(opts):
            tag = f"legacy:{ctx.arm.key}:{case.name}:run{run_index}"

            async def job(
                case: AnswerCase = case, run_index: int = run_index, tag: str = tag
            ) -> dict[str, Any]:
                facts, channel_names = case_facts(case)
                started = time.perf_counter()
                result_obj = await synthesize_answer(
                    facts,
                    case.question,
                    case.locale,
                    model=ctx.arm.model,
                    question_channel_name="general",
                    question_asked_at=datetime(2026, 10, 4, 15, 0, tzinfo=UTC),
                    fact_channel_names=channel_names,
                )
                wall = time.perf_counter() - started
                record = ctx.records.get(tag)
                used = list(result_obj.used_fact_ids) if result_obj else []
                return {
                    "case": case.name,
                    "shape": case.shape,
                    "register": case.register,
                    "difficulty": case.difficulty,
                    "locale": case.locale,
                    "run": run_index,
                    "wall_seconds": round(wall, 3),
                    **_record_fields(record),
                    "answer": result_obj.answer if result_obj else None,
                    "used": used,
                    "answers_question": result_obj.answers_question if result_obj else None,
                    "score": score_legacy(
                        case,
                        result_obj.answer if result_obj else None,
                        used,
                        result_obj.answers_question if result_obj else None,
                    ).as_dict(),
                }

            jobs.append((tag, job))
    return jobs


Job = Callable[[], Awaitable[dict[str, Any]]]


def _answer_cases(opts: argparse.Namespace) -> list[AnswerCase]:
    return [case for case in CASES if not opts.wanted or case.name in opts.wanted]


def _plain_fact(index: int, content: str, *, channel_id: int = 9000) -> Fact:
    return Fact(
        id=index,
        guild_id=_GUILD_ID,
        channel_id=channel_id,
        message_id=2000 + index,
        content=content,
        embedding=b"",
        status=FactStatus.ACTIVE,
        created_at=_FACT_TIME,
    )


def checker_jobs(ctx: RunContext, opts: argparse.Namespace) -> list[tuple[str, Job]]:
    """Return one job per corpus case and run: the legacy check with the arm as checker."""
    from aura.grounding import ASK_GROUNDING_TIMEOUT_SECONDS, verify_answer_grounded
    from grounding_verification_cases import ALL_CASES

    settings = ctx.settings.model_copy(update={"grounding_check_model": ctx.arm.model})
    jobs: list[tuple[str, Job]] = []
    for run_index in range(1, opts.runs + 1):
        for case in ALL_CASES:
            if opts.wanted and case.name not in opts.wanted:
                continue
            tag = f"checker:{ctx.arm.key}:{case.name}:run{run_index}"

            async def job(
                case: Any = case, run_index: int = run_index, tag: str = tag
            ) -> dict[str, Any]:
                facts = [_plain_fact(i, content) for i, content in enumerate(case.facts, start=1)]
                outcome = await verify_answer_grounded(
                    answer=case.answer,
                    cited_facts=facts,
                    settings=settings,
                    timeout_seconds=ASK_GROUNDING_TIMEOUT_SECONDS,
                )
                passed = outcome.value == "grounded"
                return {
                    "case": case.name,
                    "kind": "forged" if not case.expected_grounded else "control",
                    "finding": case.finding,
                    "guards": case.guards,
                    "language": case.language,
                    "run": run_index,
                    "expected_grounded": case.expected_grounded,
                    "outcome": outcome.value,
                    "correct": passed == case.expected_grounded,
                    **_record_fields(ctx.records.get(tag)),
                }

            jobs.append((tag, job))
    return jobs


def _prefilter(ctx: RunContext, out_dir: Path) -> dict[str, dict[str, Any]]:
    """Return, per proactive message, the shipped Stage 1 and Stage 2 scores (computed once, free).

    Notes
    -----
    Uses the production embedding model, question detector and thresholds:
    Stage 1 is the question-likeness score against PROACTIVE_QUESTION_THRESHOLD,
    Stage 2 the best fact similarity against PROACTIVE_SIMILARITY_THRESHOLD, and
    the facts a passing message would be answered from are the responder's:
    the top SYNTHESIS_FACT_LIMIT facts at or above that threshold. No variants
    exist for these invented facts, so best_similarity is the canonical cosine.
    """
    cache = out_dir / "proactive-prefilter.json"
    if cache.exists():
        loaded: dict[str, dict[str, Any]] = json.loads(cache.read_text(encoding="utf-8"))
        return loaded
    from fastembed import TextEmbedding

    from aura.embeddings import SYNTHESIS_FACT_LIMIT, cosine_similarity, embed_text, embed_texts
    from aura.proactive.question_detector import QuestionDetector
    from proactive_decision_cases import all_messages

    async def compute() -> dict[str, dict[str, Any]]:
        model = TextEmbedding(ctx.settings.embedding_model)
        detector = await QuestionDetector.create(model)
        result: dict[str, dict[str, Any]] = {}
        fact_vectors: dict[str, list[Any]] = {}
        for scenario, index, message in all_messages():
            if scenario.key not in fact_vectors:
                fact_vectors[scenario.key] = await embed_texts(
                    model, [text for _, text in scenario.facts]
                )
            stage1 = await detector.question_likeness(message.text)
            vector = await embed_text(model, message.text)
            scores = [
                (number, cosine_similarity(vector, fact_vector))
                for number, fact_vector in enumerate(fact_vectors[scenario.key], start=1)
            ]
            ranked = sorted(scores, key=lambda pair: (-pair[1], pair[0]))
            threshold = ctx.settings.proactive_similarity_threshold
            chosen = [
                number for number, score in ranked[:SYNTHESIS_FACT_LIMIT] if score >= threshold
            ]
            result[f"{scenario.key}-{index:02d}"] = {
                "stage1": stage1,
                "stage1_passed": stage1 >= ctx.settings.proactive_question_threshold,
                "top_similarity": ranked[0][1],
                "stage2_passed": ranked[0][1] >= threshold,
                "facts": chosen,
                "scores": {str(number): round(score, 4) for number, score in scores},
            }
        return result

    computed = asyncio.run(compute())
    out_dir.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(computed, ensure_ascii=False, indent=1), encoding="utf-8")
    return computed


def proactive_jobs(ctx: RunContext, opts: argparse.Namespace) -> list[tuple[str, Job]]:
    """Return one job per gate-passing proactive message and run: the model's post decision."""
    from proactive_decision_cases import all_messages

    prefilter = _prefilter(ctx, opts.out_dir)
    jobs: list[tuple[str, Job]] = []
    for run_index in range(1, opts.runs + 1):
        for scenario, index, message in all_messages():
            key = f"{scenario.key}-{index:02d}"
            gate = prefilter[key]
            if not (gate["stage1_passed"] and gate["stage2_passed"]) or not gate["facts"]:
                continue
            if opts.wanted and key not in opts.wanted:
                continue
            tag = f"proactive-{opts.format}:{ctx.arm.key}:{key}:run{run_index}"

            async def job(
                scenario: Any = scenario,
                message: Any = message,
                key: str = key,
                gate: dict[str, Any] = gate,
                run_index: int = run_index,
                tag: str = tag,
            ) -> dict[str, Any]:
                channel_ids = {
                    name: 7000 + i for i, name in enumerate(sorted({c for c, _ in scenario.facts}))
                }
                facts = [
                    _plain_fact(
                        number,
                        scenario.facts[number - 1][1],
                        channel_id=channel_ids[scenario.facts[number - 1][0]],
                    )
                    for number in gate["facts"]
                ]
                names = {cid: name for name, cid in channel_ids.items()}
                if opts.format == "v2":
                    answer = await synthesize_contract_answer(
                        facts, message.text, "de", model=ctx.arm.model, settings=ctx.settings
                    )
                    decision = answer is not None and answer.answers_unprompted
                    detail: dict[str, Any] = {
                        "failed": answer is None,
                        "answers_question": answer.answers_question if answer else None,
                        "used": list(answer.used_fact_ids) if answer else [],
                        "relations": [r.kind.value for r in answer.relations] if answer else [],
                        "lead": answer.lead if answer else None,
                    }
                else:
                    result = await synthesize_answer(
                        facts,
                        message.text,
                        "de",
                        model=ctx.arm.model,
                        question_channel_name=scenario.channel,
                        question_asked_at=datetime(2026, 10, 4, 18, 0, tzinfo=UTC),
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
                    "category": message.category,
                    "should_post": message.should_post,
                    "run": run_index,
                    "posts": decision,
                    **detail,
                    **_record_fields(ctx.records.get(tag)),
                }

            jobs.append((tag, job))
    return jobs


def extraction_jobs(ctx: RunContext, opts: argparse.Namespace) -> list[tuple[str, Job]]:
    """Return one job per extraction batch and run: distillation, scored per message."""
    from datetime import timedelta

    from aura.db.extraction_queue import QueuedMessage
    from aura.extraction.distiller import distill_facts
    from extraction_eval_cases import ALL_BATCHES
    from extraction_german_cases import GERMAN_BATCHES

    batch_time = datetime(2026, 7, 30, 11, 0, tzinfo=UTC)
    jobs: list[tuple[str, Job]] = []
    for run_index in range(1, opts.runs + 1):
        for batch in (*ALL_BATCHES, *GERMAN_BATCHES):
            if opts.wanted and batch.name not in opts.wanted:
                continue
            tag = f"extraction:{ctx.arm.key}:{batch.name}:run{run_index}"

            async def job(
                batch: Any = batch, run_index: int = run_index, tag: str = tag
            ) -> dict[str, Any]:
                queued = [
                    QueuedMessage(
                        channel_id=500,
                        message_id=1000 + index,
                        guild_id=100,
                        channel_name=batch.channel_name,
                        content=message.text,
                        message_created_at=batch_time + timedelta(minutes=index),
                        enqueued_at=batch_time + timedelta(minutes=index),
                    )
                    for index, message in enumerate(batch.messages)
                ]
                distilled = await distill_facts(
                    queued, channel_name=batch.channel_name, model=ctx.arm.model
                )
                messages = []
                for index, message in enumerate(batch.messages):
                    extracted = [
                        fact.content
                        for fact in (distilled or [])
                        if fact.message_id == 1000 + index
                    ]
                    messages.append(
                        {
                            "index": index + 1,
                            "text": message.text,
                            "expect_fact": message.expect_fact,
                            "extracted": extracted,
                        }
                    )
                forbidden = [
                    (fact.content, needle)
                    for fact in (distilled or [])
                    for needle in batch.forbidden_substrings
                    if needle.lower() in fact.content.lower()
                ]
                return {
                    "case": batch.name,
                    "locale": batch.locale,
                    "run": run_index,
                    "call_failed": distilled is None,
                    "messages": messages,
                    "forbidden_hits": forbidden,
                    **_record_fields(ctx.records.get(tag)),
                }

            jobs.append((tag, job))
    return jobs


def supersession_jobs(ctx: RunContext, opts: argparse.Namespace) -> list[tuple[str, Job]]:
    """Return one job per supersession pair and run: the judged category against the label."""
    from aura.extraction.supersession import judge_relationship
    from supersession_bakeoff_cases import ALL_CASES as PAIRS

    jobs: list[tuple[str, Job]] = []
    for run_index in range(1, opts.runs + 1):
        for pair in PAIRS:
            if opts.wanted and pair.name not in opts.wanted:
                continue
            tag = f"supersession:{ctx.arm.key}:{pair.name}:run{run_index}"

            async def job(
                pair: Any = pair, run_index: int = run_index, tag: str = tag
            ) -> dict[str, Any]:
                judgement = await judge_relationship(
                    predecessor=pair.predecessor, candidate=pair.candidate, model=ctx.arm.model
                )
                category = str(judgement.relationship.value) if judgement else None
                return {
                    "case": pair.name,
                    "expected": pair.category,
                    "boundary": pair.boundary,
                    "cross_locale": pair.cross_locale,
                    "run": run_index,
                    "category": category,
                    "correct": category == pair.category,
                    **_record_fields(ctx.records.get(tag)),
                }

            jobs.append((tag, job))
    return jobs


def throughput_jobs(ctx: RunContext, opts: argparse.Namespace) -> list[tuple[str, Job]]:
    """Return --calls v2 answers over the cases in turn, for the load test."""
    cases = _answer_cases(opts)
    jobs: list[tuple[str, Job]] = []
    for call_index in range(opts.calls):
        case = cases[call_index % len(cases)]
        tag = f"throughput:{ctx.arm.key}:{call_index:04d}:{case.name}"

        async def job(case: AnswerCase = case, tag: str = tag) -> dict[str, Any]:
            facts, _ = case_facts(case)
            started = time.perf_counter()
            error = None
            try:
                answer = await synthesize_contract_answer(
                    facts, case.question, case.locale, model=ctx.arm.model, settings=ctx.settings
                )
            except Exception as exc:  # the shipped function never raises; belt and braces
                answer, error = None, type(exc).__name__
            return {
                "case": case.name,
                "valid": answer is not None,
                "error": error,
                "wall_seconds": round(time.perf_counter() - started, 3),
                **_record_fields(ctx.records.get(tag)),
            }

        jobs.append((tag, job))
    return jobs


def earlycheck_jobs(ctx: RunContext, opts: argparse.Namespace) -> list[tuple[str, Job]]:
    """Return one job per v2 answer in --source: today's check over its plain text."""
    from aura.grounding import ASK_GROUNDING_TIMEOUT_SECONDS, verify_answer_grounded

    settings = ctx.settings.model_copy(update={"grounding_check_model": ctx.arm.model})
    by_name = {case.name: case for case in CASES}
    jobs: list[tuple[str, Job]] = []
    for line in Path(opts.source).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not row.get("plain_text"):
            continue
        tag = f"earlycheck:{ctx.arm.key}:{row['arm']}:{row['case']}:run{row['run']}"

        async def job(row: dict[str, Any] = row, tag: str = tag) -> dict[str, Any]:
            facts, _ = case_facts(by_name[row["case"]])
            facts_by_id = {fact.id: fact for fact in facts}
            outcome = await verify_answer_grounded(
                answer=row["plain_text"],
                cited_facts=[facts_by_id[i] for i in row["cited_fact_ids"]],
                settings=settings,
                timeout_seconds=ASK_GROUNDING_TIMEOUT_SECONDS,
            )
            return {
                "case": row["case"],
                "source_arm": row["arm"],
                "shape": row["shape"],
                "run": row["run"],
                "outcome": outcome.value,
                "plain_text": row["plain_text"],
                **_record_fields(ctx.records.get(tag)),
            }

        jobs.append((tag, job))
    return jobs


def v2check_jobs(ctx: RunContext, opts: argparse.Namespace) -> list[tuple[str, Job]]:
    """Return one job per case and run: the v2 statement check with the arm as checker.

    With --source, the cases are the v2 answers in that results file (Part C's
    real answers); otherwise the corpus of scripts/answer_check_cases.py.
    """
    from answer_check_cases import all_cases
    from aura.answer_check import build_statements, verify_answer_v2
    from aura.grounding import ASK_GROUNDING_TIMEOUT_SECONDS

    settings = ctx.settings.model_copy(update={"answer_v2_check_model": ctx.arm.model})
    items: list[dict[str, Any]] = []
    if opts.source:
        by_name = {case.name: case for case in CASES}
        for line in Path(opts.source).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("checked_lead") is None:
                continue
            facts, _ = case_facts(by_name[row["case"]])
            items.append(
                {
                    "case": f"{row['arm']}:{row['case']}:r{row['run']}",
                    "facts": facts,
                    "lead": row["checked_lead"],
                    "points": [(p["text"], tuple(p["fact_ids"])) for p in row["checked_points"]],
                    "cited": tuple(row["cited_fact_ids"]),
                    "meta": {"origin": "partC", "source_arm": row["arm"], "shape": row["shape"]},
                }
            )
    else:
        for case in all_cases():
            facts = [_plain_fact(i, text) for i, text in enumerate(case.facts, start=1)]
            items.append(
                {
                    "case": case.name,
                    "facts": facts,
                    "lead": case.lead,
                    "points": list(case.points),
                    "cited": tuple(range(1, len(facts) + 1)),
                    "meta": {
                        "origin": case.origin,
                        "expected_grounded": case.expected_grounded,
                        "klass": case.klass,
                        "language": case.language,
                        "twin": case.twin,
                    },
                }
            )
    jobs: list[tuple[str, Job]] = []
    for run_index in range(1, opts.runs + 1):
        for item in items:
            if opts.wanted and item["case"] not in opts.wanted:
                continue
            tag = f"v2check:{ctx.arm.key}:{item['case']}:run{run_index}"

            async def job(
                item: dict[str, Any] = item, run_index: int = run_index, tag: str = tag
            ) -> dict[str, Any]:
                facts_by_id = {fact.id: fact for fact in item["facts"]}
                if not item["cited"]:
                    return {
                        "case": item["case"],
                        "run": run_index,
                        "outcome": "not_checked",
                        **item["meta"],
                    }
                statements = build_statements(item["lead"], item["points"], item["cited"])
                outcome = await verify_answer_v2(
                    statements,
                    [facts_by_id[i] for i in item["cited"]],
                    settings=settings,
                    timeout_seconds=ASK_GROUNDING_TIMEOUT_SECONDS,
                )
                return {
                    "case": item["case"],
                    "run": run_index,
                    "outcome": outcome.value,
                    "lead": item["lead"],
                    "points": item["points"],
                    **item["meta"],
                    **_record_fields(ctx.records.get(tag)),
                }

            jobs.append((tag, job))
    return jobs


TASKS: Final[dict[str, Callable[[RunContext, argparse.Namespace], list[tuple[str, Job]]]]] = {
    "synth": synth_jobs,
    "legacy": legacy_jobs,
    "checker": checker_jobs,
    "proactive": proactive_jobs,
    "extraction": extraction_jobs,
    "supersession": supersession_jobs,
    "throughput": throughput_jobs,
    "earlycheck": earlycheck_jobs,
    "v2check": v2check_jobs,
}


# --- the runner --------------------------------------------------------------


def _done_tags(path: Path) -> set[str]:
    if not path.exists():
        return set()
    tags = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            tags.add(json.loads(line)["tag"])
    return tags


async def run_jobs(
    jobs: list[tuple[str, Job]], *, ctx: RunContext, out_path: Path, concurrency: int
) -> tuple[int, int, bool]:
    """Run every job not yet in `out_path`, appending one line per finished job.

    Returns
    -------
    tuple[int, int, bool]
        Jobs run now, jobs skipped as already done, and whether the ceiling
        stopped the run.
    """
    done = _done_tags(out_path)
    pending = [(tag, job) for tag, job in jobs if tag not in done]
    semaphore = asyncio.Semaphore(concurrency)
    stopped = asyncio.Event()
    finished = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)

    async def one(tag: str, job: Job) -> None:
        nonlocal finished
        if stopped.is_set():
            return
        async with semaphore:
            if stopped.is_set():
                return
            CURRENT_TAG.set(tag)
            result = await job()
            if tag not in ctx.records and _CEILING_HITS:
                stopped.set()
                return
            with out_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps({"tag": tag, "arm": ctx.arm.key, **result}, ensure_ascii=False)
                    + "\n"
                )
            finished += 1
            if finished % 10 == 0:
                print(f"  {finished}/{len(pending)} done", flush=True)

    await asyncio.gather(*(one(tag, job) for tag, job in pending))
    return finished, len(jobs) - len(pending), stopped.is_set()


# Every refusal the metering made in this process; a job whose call is missing
# while this is non-empty was refused, not failed, and leaves no result line.
_CEILING_HITS: list[str] = []


def _install(ctx: RunContext, *, bucket: str, label: str) -> Callable[[], None]:
    """Install the metering for one arm, noting every ceiling refusal."""
    restore = install_metering(
        ctx.ledger,
        bucket=bucket,
        label=label,
        records=ctx.records,
        prices=prices(),
        routing={ctx.arm.model: ctx.arm.routing},
    )
    import litellm

    metered = litellm.acompletion

    async def noting(*args: Any, **kwargs: Any) -> Any:
        try:
            return await metered(*args, **kwargs)
        except CeilingReachedError as exc:
            _CEILING_HITS.append(str(exc))
            print(f"CEILING: {exc}", flush=True)
            raise

    litellm.acompletion = noting
    return restore


def _estimate(arm: Arm, task: str, calls: int) -> float:
    """Return a rough advance estimate; the ceiling guard uses each call's worst case."""
    typical = {
        "synth": (1800, 190),
        "legacy": (1500, 120),
        "checker": (1500, 80),
        "proactive": (1400, 110),
        "extraction": (2200, 300),
        "supersession": (1700, 120),
        "throughput": (1800, 190),
        "earlycheck": (1500, 80),
        "v2check": (1500, 120),
    }[task]
    prompt, completion = typical
    if arm.routing.max_tokens:
        completion += 300  # reasoning tokens, a rough allowance for the estimate only
    return calls * (prompt * arm.price.input_usd + completion * arm.price.output_usd) / 1_000_000


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
    parser.add_argument("--cases", default="", help="comma-separated case names; default all")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--prompt-variant", default="shipped")
    parser.add_argument("--check-variant", default="shipped")
    parser.add_argument("--format", choices=("legacy", "v2"), default="legacy")
    parser.add_argument("--calls", type=int, default=100)
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
    if args.temperature is not None:
        from dataclasses import replace

        arm = replace(
            arm,
            key=f"{arm.key}-t{args.temperature}",
            routing=replace(arm.routing, temperature=args.temperature),
        )
    if args.check_variant != "shipped":
        from prompt_variants import apply_check_variant

        apply_check_variant(args.check_variant)
        arm = replace_key(arm, f"{arm.key}-{args.check_variant}")
    if args.prompt_variant != "shipped":
        from prompt_variants import apply_variant

        apply_variant(args.prompt_variant)
        arm = replace_key(arm, f"{arm.key}-{args.prompt_variant}")
    ledger = Ledger(args.ledger, BUCKET_CEILINGS, TOTAL_CEILING)
    settings = load_settings()
    ctx = RunContext(arm=arm, settings=settings, records={}, ledger=ledger)
    jobs = TASKS[args.task](ctx, args)
    task_name = f"{args.task}-v2" if args.task == "proactive" and args.format == "v2" else args.task
    out_path = args.out_dir / f"{task_name}-{arm.key}-{args.label}.jsonl"
    done = _done_tags(out_path)
    todo = len([tag for tag, _ in jobs if tag not in done])
    state = ledger.snapshot()
    print(
        f"{args.task} x {arm.key}: {len(jobs)} job(s), {todo} to run, estimate "
        f"${_estimate(arm, args.task, todo):.3f}; bucket {args.bucket} "
        f"${state.spent(args.bucket):.4f}/{BUCKET_CEILINGS[args.bucket]:.2f}, total "
        f"${state.spent():.4f}/{TOTAL_CEILING:.2f}",
        flush=True,
    )
    if args.dry_run:
        return 0
    restore = _install(ctx, bucket=args.bucket, label=args.label)
    try:
        ran, skipped, stopped = asyncio.run(
            run_jobs(jobs, ctx=ctx, out_path=out_path, concurrency=args.concurrency)
        )
    finally:
        restore()
    state = ledger.snapshot()
    print(
        f"ran {ran}, skipped {skipped}{', STOPPED BY CEILING' if stopped else ''}; bucket "
        f"{args.bucket} ${state.spent(args.bucket):.4f}, total ${state.spent():.4f}",
        flush=True,
    )
    return 3 if stopped else 0


def replace_key(arm: Arm, key: str) -> Arm:
    """Return `arm` under another key (a prompt variant's results get their own file)."""
    from dataclasses import replace

    return replace(arm, key=key)


if __name__ == "__main__":
    raise SystemExit(main())
