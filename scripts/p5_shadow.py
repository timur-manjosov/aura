"""The P5 shadow runs: candidate models over REAL data of the operator's own two servers.

    AURA_RUN_REAL_LLM=1 .venv/bin/python scripts/p5_shadow.py --shadow-dir <private dir> \\
        --ledger <file> --label <label> --out-dir <private dir> \\
        <extraction|verify|supersession|proactive> --arm <key> [--runs N] [--format legacy|v2p] [--source F]

Reads a read-only backup of the bot database (`shadow.db`) and the source
messages fetched read-only from Discord (`messages.json`), both in the private,
gitignored reports folder, with the operator's explicit approval (2026-10-04).
Nothing here contains or writes real content outside that folder: results go to
--out-dir, which must be inside reports/.

REAL CONTENT GOES ONLY TO PROVIDERS THE OPERATOR APPROVED for it: DeepInfra and
Together (DeepSeek, GLM), Google Vertex (Gemini, pinned) and the incumbent's own
default providers (Claude Haiku: Anthropic, Amazon Bedrock, Google Vertex,
Azure). Any other arm is refused before a call is built.

Metered like every paid harness here (scripts/llm_metering.py, the P5 ledger,
bucket "checks").
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sqlite3
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
from p4_bakeoff import RunContext, _done_tags, _install, _record_fields, run_jobs
from p5_bakeoff import BUCKET_CEILINGS, TOTAL_CEILING, _install_retry

APPROVED_ARMS: Final = frozenset(
    {"haiku", "deepseek-2p", "deepseek-2p-think", "gemini38-vertex", "glm"}
)

Job = Callable[[], Awaitable[dict[str, Any]]]


def _parse_time(value: str) -> datetime:
    moment = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return moment if moment.tzinfo else moment.replace(tzinfo=UTC)


def _short(value: int) -> str:
    """A non-reversible short key for an id (no full guild, channel or message id is written)."""
    return hashlib.sha256(str(value).encode()).hexdigest()[:8]


class Shadow:
    """The private shadow data, loaded once."""

    def __init__(self, directory: Path) -> None:
        self.messages: list[dict[str, Any]] = json.loads(
            (directory / "messages.json").read_text(encoding="utf-8")
        )["messages"]
        self.db = sqlite3.connect(f"file:{directory / 'shadow.db'}?mode=ro", uri=True)

    def batches(self) -> list[tuple[str, str, list[dict[str, Any]]]]:
        """Return the reconstructed extraction batches: human, non-empty, in time order."""
        groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
        for message in self.messages:
            if message["origin"] not in ("extraction_window", "backfill_history"):
                continue
            if message["author_is_bot"] or not message["content"].strip():
                continue
            groups.setdefault((message["origin"], message["channel_id"]), []).append(message)
        result = []
        for (origin, channel_id), messages in sorted(groups.items(), key=lambda item: item[0][0]):
            messages.sort(key=lambda m: m["timestamp"])
            result.append(
                (f"{origin}-{_short(channel_id)}", f"channel-{_short(channel_id)[:4]}", messages)
            )
        return result

    def facts_at(self, guild_id: int, moment: datetime) -> list[Fact]:
        """Return the facts active in a guild at `moment` (created before, not yet superseded)."""
        facts = []
        for row in self.db.execute(
            "SELECT id, guild_id, channel_id, message_id, content, created_at, superseded_at "
            "FROM facts WHERE guild_id = ? ORDER BY id",
            (guild_id,),
        ):
            created = _parse_time(row[5])
            superseded = _parse_time(row[6]) if row[6] else None
            if created <= moment and (superseded is None or superseded > moment):
                facts.append(
                    Fact(
                        id=row[0],
                        guild_id=row[1],
                        channel_id=row[2],
                        message_id=row[3],
                        content=row[4],
                        embedding=b"",
                        status=FactStatus.ACTIVE,
                        created_at=created,
                    )
                )
        return facts

    def judged_pairs(self) -> list[tuple[str, str, str, str | None]]:
        """Return (key, predecessor, candidate, live judgement) for every judged candidate."""
        rows = self.db.execute(
            "SELECT p.id, f.content, p.content, p.relationship FROM pending_facts p "
            "JOIN facts f ON f.id = p.similar_fact_id WHERE p.similar_fact_id IS NOT NULL ORDER BY p.id"
        )
        return [(f"pair-{row[0]}", row[1], row[2], row[3]) for row in rows]

    def signals(self) -> list[tuple[str, dict[str, Any], int, datetime]]:
        """Return (key, message, guild id, time) for every proactive signal whose message still exists."""
        by_id = {m["message_id"]: m for m in self.messages if m["origin"] == "signal"}
        result = []
        for row in self.db.execute(
            "SELECT id, guild_id, message_id, created_at FROM proactive_signals ORDER BY id"
        ):
            message = by_id.get(row[2])
            if message is not None:
                result.append((f"signal-{row[0]}", message, row[1], _parse_time(row[3])))
        return result


def extraction_jobs(
    ctx: RunContext, shadow: Shadow, opts: argparse.Namespace
) -> list[tuple[str, Job]]:
    """One job per reconstructed batch and run: the shipped distillation."""
    from aura.db.extraction_queue import QueuedMessage
    from aura.extraction.distiller import distill_facts

    jobs: list[tuple[str, Job]] = []
    for run_index in range(1, opts.runs + 1):
        for key, channel, messages in shadow.batches():
            tag = f"shadow-extraction:{ctx.arm.key}:{key}:run{run_index}"

            async def job(
                key: str = key,
                channel: str = channel,
                messages: list[dict[str, Any]] = messages,
                run_index: int = run_index,
                tag: str = tag,
            ) -> dict[str, Any]:
                batch = [
                    QueuedMessage(
                        channel_id=1,
                        message_id=index,
                        guild_id=1,
                        channel_name=channel,
                        content=m["content"],
                        message_created_at=_parse_time(m["timestamp"]),
                        enqueued_at=_parse_time(m["timestamp"]),
                    )
                    for index, m in enumerate(messages, start=1)
                ]
                distilled = await distill_facts(batch, channel_name=channel, model=ctx.arm.model)
                return {
                    "case": key,
                    "run": run_index,
                    "call_failed": distilled is None,
                    "facts": [
                        {
                            "message": f.message_id,
                            "content": f.content,
                            "category": f.category.value,
                        }
                        for f in (distilled or [])
                    ],
                    **_record_fields(ctx.records.get(tag)),
                }

            jobs.append((tag, job))
    return jobs


def verify_jobs(ctx: RunContext, shadow: Shadow, opts: argparse.Namespace) -> list[tuple[str, Job]]:
    """One job per non-empty shadow extraction result in --source: the verification."""
    from aura.db.extraction_queue import QueuedMessage
    from aura.db.pending_facts import FactCategory
    from aura.extraction.distiller import DistilledFact
    from aura.extraction.verifier import verify_distilled_facts

    batches = {key: (channel, messages) for key, channel, messages in shadow.batches()}
    jobs: list[tuple[str, Job]] = []
    for line in Path(opts.source).read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("call_failed") or not row.get("facts"):
            continue
        tag = f"shadow-verify:{ctx.arm.key}:{row['arm']}:{row['case']}:run{row['run']}"

        async def job(row: dict[str, Any] = row, tag: str = tag) -> dict[str, Any]:
            channel, messages = batches[row["case"]]
            batch = [
                QueuedMessage(
                    channel_id=1,
                    message_id=index,
                    guild_id=1,
                    channel_name=channel,
                    content=m["content"],
                    message_created_at=_parse_time(m["timestamp"]),
                    enqueued_at=_parse_time(m["timestamp"]),
                )
                for index, m in enumerate(messages, start=1)
            ]
            distilled = [
                DistilledFact(
                    message_id=f["message"],
                    content=f["content"],
                    category=FactCategory(f["category"]),
                )
                for f in row["facts"]
            ]
            kept = await verify_distilled_facts(
                batch, distilled, channel_name=channel, model=ctx.arm.model, settings=ctx.settings
            )
            return {
                "case": row["case"],
                "source_arm": row["arm"],
                "run": row["run"],
                "verify_failed": kept is None,
                "proposed": row["facts"],
                "facts": [{"message": f.message_id, "content": f.content} for f in (kept or [])],
                **_record_fields(ctx.records.get(tag)),
            }

        jobs.append((tag, job))
    return jobs


def supersession_jobs(
    ctx: RunContext, shadow: Shadow, opts: argparse.Namespace
) -> list[tuple[str, Job]]:
    """One job per real judged pair and run: the shipped judge."""
    from aura.extraction.supersession import judge_relationship

    jobs: list[tuple[str, Job]] = []
    for run_index in range(1, opts.runs + 1):
        for key, predecessor, candidate, live in shadow.judged_pairs():
            tag = f"shadow-supersession:{ctx.arm.key}:{key}:run{run_index}"

            async def job(
                key: str = key,
                predecessor: str = predecessor,
                candidate: str = candidate,
                live: str | None = live,
                run_index: int = run_index,
                tag: str = tag,
            ) -> dict[str, Any]:
                judgement = await judge_relationship(
                    predecessor=predecessor, candidate=candidate, model=ctx.arm.model
                )
                return {
                    "case": key,
                    "run": run_index,
                    "live": live,
                    "category": judgement.relationship.value if judgement else None,
                    **_record_fields(ctx.records.get(tag)),
                }

            jobs.append((tag, job))
    return jobs


def proactive_jobs(
    ctx: RunContext, shadow: Shadow, opts: argparse.Namespace
) -> list[tuple[str, Job]]:
    """One job per real signal (gate recomputed on the facts active then) and run."""
    from fastembed import TextEmbedding

    from aura.answer_contract import synthesize_contract_answer
    from aura.embeddings import SYNTHESIS_FACT_LIMIT, cosine_similarity, embed_text, embed_texts
    from aura.proactive.question_detector import QuestionDetector
    from aura.synthesis import synthesize_answer

    async def gate() -> dict[str, dict[str, Any]]:
        model = TextEmbedding(ctx.settings.embedding_model)
        detector = await QuestionDetector.create(model)
        out: dict[str, dict[str, Any]] = {}
        for key, message, guild_id, moment in shadow.signals():
            facts = shadow.facts_at(guild_id, moment)
            stage1 = await detector.question_likeness(message["content"])
            if not facts:
                out[key] = {"passed": False, "facts": []}
                continue
            vector = await embed_text(model, message["content"])
            vectors = await embed_texts(model, [f.content for f in facts])
            ranked = sorted(
                ((cosine_similarity(vector, v), f) for v, f in zip(vectors, facts, strict=True)),
                key=lambda p: -p[0],
            )
            chosen = [
                f
                for score, f in ranked[:SYNTHESIS_FACT_LIMIT]
                if score >= ctx.settings.proactive_similarity_threshold
            ]
            out[key] = {
                "passed": stage1 >= ctx.settings.proactive_question_threshold and bool(chosen),
                "facts": chosen,
            }
        return out

    verdicts = asyncio.run(gate())
    jobs: list[tuple[str, Job]] = []
    for run_index in range(1, opts.runs + 1):
        for key, message, _guild_id, moment in shadow.signals():
            if not verdicts[key]["passed"]:
                continue
            tag = f"shadow-proactive-{opts.format}:{ctx.arm.key}:{key}:run{run_index}"

            async def job(
                key: str = key,
                message: dict[str, Any] = message,
                moment: datetime = moment,
                run_index: int = run_index,
                tag: str = tag,
            ) -> dict[str, Any]:
                facts = verdicts[key]["facts"]
                if opts.format == "v2p":
                    answer = await synthesize_contract_answer(
                        facts,
                        message["content"],
                        "de",
                        model=ctx.arm.model,
                        settings=ctx.settings,
                        proactive_posted_at=moment,
                    )
                    return {
                        "case": key,
                        "run": run_index,
                        "format": "v2p",
                        "posts": answer is not None and answer.answers_unprompted,
                        "failed": answer is None,
                        "message_kind": answer.message_kind.value
                        if answer and answer.message_kind
                        else None,
                        "relations": [r.kind.value for r in answer.relations] if answer else [],
                        "lead": answer.lead if answer else None,
                        **_record_fields(ctx.records.get(tag)),
                    }
                result = await synthesize_answer(
                    facts,
                    message["content"],
                    "de",
                    model=ctx.arm.model,
                    question_channel_name="chat",
                    question_asked_at=moment,
                )
                return {
                    "case": key,
                    "run": run_index,
                    "format": "legacy",
                    "posts": result is not None
                    and result.answers_question
                    and bool(result.used_fact_ids),
                    "failed": result is None,
                    "answer": result.answer if result else None,
                    **_record_fields(ctx.records.get(tag)),
                }

            jobs.append((tag, job))
    return jobs


TASKS: Final = {
    "extraction": extraction_jobs,
    "verify": verify_jobs,
    "supersession": supersession_jobs,
    "proactive": proactive_jobs,
}


def main() -> int:
    """Run one shadow task for one approved arm."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--shadow-dir", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("task", choices=sorted(TASKS))
    parser.add_argument("--arm", required=True)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--format", choices=("legacy", "v2p"), default="legacy")
    parser.add_argument("--source", default="")
    args = parser.parse_args()
    if not os.environ.get(RUN_REAL_LLM_ENV):
        print(f"Refusing to spend money: set {RUN_REAL_LLM_ENV}=1 to run this harness.")
        return 2
    if args.arm not in APPROVED_ARMS:
        print(
            f"Refusing: real content may only go to the approved providers {sorted(APPROVED_ARMS)}."
        )
        return 2
    for path in (args.shadow_dir, args.out_dir):
        if "reports" not in path.resolve().parts:
            print("Refusing: shadow data and results stay inside reports/.")
            return 2
    shadow = Shadow(args.shadow_dir)
    ledger = Ledger(args.ledger, BUCKET_CEILINGS, TOTAL_CEILING)
    ctx = RunContext(arm=ARMS[args.arm], settings=load_settings(), records={}, ledger=ledger)
    jobs = TASKS[args.task](ctx, shadow, args)
    name = (
        f"shadow-{args.task}-{args.format}" if args.task == "proactive" else f"shadow-{args.task}"
    )
    out_path = args.out_dir / f"{name}-{args.arm}-{args.label}.jsonl"
    print(
        f"{name} x {args.arm}: {len(jobs)} job(s), {len(jobs) - len(_done_tags(out_path))} to run",
        flush=True,
    )
    restore = _install(ctx, bucket="checks", label=args.label)
    restore_retry = _install_retry()
    try:
        ran, skipped, stopped = asyncio.run(
            run_jobs(jobs, ctx=ctx, out_path=out_path, concurrency=args.concurrency)
        )
    finally:
        restore_retry()
        restore()
    os.chmod(out_path, 0o600) if out_path.exists() else None
    state = ledger.snapshot()
    print(
        f"ran {ran}, skipped {skipped}{', STOPPED' if stopped else ''}; checks ${state.spent('checks'):.4f}, total ${state.spent():.4f}",
        flush=True,
    )
    return 3 if stopped else 0


if __name__ == "__main__":
    raise SystemExit(main())
