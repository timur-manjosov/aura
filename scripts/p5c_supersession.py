"""The P5c evaluation: the supersession judge on changes limited in time, real and paid.

    AURA_RUN_REAL_LLM=1 .venv/bin/python scripts/p5c_supersession.py --ledger <file> \\
        --bucket <dev|eval|reserve> --label <label> --out-dir <dir> [--dry-run] \\
        --arm <key> [--old-prompt] [--runs N] [--first-run K] \\
        [--slice all|dev|heldout] [--set all|new|p5] [--cases a,b]

THIS SPENDS REAL MONEY, under the rules of every paid harness here: not a pytest
test, refused unless AURA_RUN_REAL_LLM is set, and every call metered by
scripts/llm_metering.py against the bucket's and the total ceiling BEFORE the
request leaves (P5c: dev 1, eval 4, reserve 1, total 6 USD).

It drives the SHIPPED judge, aura.extraction.supersession.judge_relationship,
over scripts/p5c_supersession_cases.py. What an arm changes is only the model
and the request options of scripts/bakeoff_arms.py. `--old-prompt` replaces the
prompt builder, and nothing else, with the one shipped at commit 6ec27f2 (the
`main` this work started from), read through `git show`: the incumbent before
the fix ("haiku-old" in the report), runnable at any time on the same code.

Results are one JSON line per call in <out-dir>/supersession-<arm>[-old]-<label>.jsonl;
a stopped run resumes with the same arguments and skips every tag already
written.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import os
import subprocess
import sys
import types
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, Final

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from aura.config import load_settings
from bakeoff_arms import ARMS
from llm_metering import RUN_REAL_LLM_ENV, Ledger
from p4_bakeoff import RunContext, _done_tags, _estimate, _install, _record_fields, run_jobs
from p5_bakeoff import _install_retry
from p5c_supersession_cases import Judgement, all_judgements

BUCKET_CEILINGS: Final[dict[str, float]] = {"dev": 1.00, "eval": 4.00, "reserve": 1.00}
TOTAL_CEILING: Final = 6.00

# The commit whose supersession prompt is "the old prompt": main when P5c began.
OLD_PROMPT_COMMIT: Final = "6ec27f2"
_SUPERSESSION_PATH: Final = "src/aura/extraction/supersession.py"

Job = Callable[[], Awaitable[dict[str, Any]]]


def old_prompt_builder() -> Callable[..., list[dict[str, str]]]:
    """Return `_build_messages` exactly as shipped at OLD_PROMPT_COMMIT.

    Returns
    -------
    Callable[..., list[dict[str, str]]]
        The old builder; it reads only its own module's constants.

    Raises
    ------
    subprocess.CalledProcessError
        If the commit or the file cannot be read.
    """
    root = Path(__file__).resolve().parent.parent
    source = subprocess.run(
        ["git", "show", f"{OLD_PROMPT_COMMIT}:{_SUPERSESSION_PATH}"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    module = types.ModuleType("_supersession_at_old_commit")
    exec(compile(source, f"{OLD_PROMPT_COMMIT}:{_SUPERSESSION_PATH}", "exec"), module.__dict__)
    builder: Callable[..., list[dict[str, str]]] = module._build_messages
    return builder


def prompt_fingerprint(builder: Callable[..., list[dict[str, str]]]) -> str:
    """Return the first 16 hex digits of the sha256 of a builder's system prompt."""
    system = builder(predecessor="a", candidate="b")[0]["content"]
    return hashlib.sha256(system.encode("utf-8")).hexdigest()[:16]


def _selected(judgement: Judgement, opts: argparse.Namespace) -> bool:
    if opts.wanted and judgement.name not in opts.wanted:
        return False
    # "p5" = the 125 P5 judgements (their shapes are P5's), "new" = the 80 others.
    if opts.set == "new" and judgement.shape in _P5_SHAPES:
        return False
    if opts.set == "p5" and judgement.shape not in _P5_SHAPES:
        return False
    if opts.slice == "dev":
        return judgement.dev
    if opts.slice == "heldout":
        return not judgement.dev
    return True


_P5_SHAPES: Final = frozenset(
    {
        "different-detail",
        "additional-time",
        "recurring-series",
        "narrowing-vs-replacing",
        "two-things",
        "status-flip",
        "value-change",
        "cross-locale",
        "p4",
    }
)


def supersession_jobs(
    ctx: RunContext, opts: argparse.Namespace, *, arm_label: str, fingerprint: str
) -> list[tuple[str, Job]]:
    """Return one job per selected judgement and run: the shipped judge."""
    from aura.extraction.supersession import judge_relationship

    jobs: list[tuple[str, Job]] = []
    for run_index in range(opts.first_run, opts.first_run + opts.runs):
        for judgement in all_judgements():
            if not _selected(judgement, opts):
                continue
            tag = f"p5c-supersession:{arm_label}:{judgement.name}:run{run_index}"

            async def job(
                judgement: Judgement = judgement, run_index: int = run_index, tag: str = tag
            ) -> dict[str, Any]:
                result = await judge_relationship(
                    predecessor=judgement.predecessor,
                    candidate=judgement.candidate,
                    model=ctx.arm.model,
                )
                return {
                    "case": judgement.name,
                    "expected": judgement.category,
                    "shape": judgement.shape,
                    "family": judgement.family,
                    "boundary": judgement.boundary,
                    "dev": judgement.dev,
                    "run": run_index,
                    "prompt": fingerprint,
                    "category": result.relationship.value if result else None,
                    "change_signal": result.change_signal if result else None,
                    "shared_subject": result.shared_subject if result else None,
                    "reasoning": result.reasoning if result else None,
                    **_record_fields(ctx.records.get(tag)),
                }

            jobs.append((tag, job))
    return jobs


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
    parser.add_argument("--arm", choices=sorted(ARMS), required=True)
    parser.add_argument("--old-prompt", action="store_true")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--first-run", type=int, default=1)
    parser.add_argument("--cases", default="")
    parser.add_argument("--slice", choices=("all", "dev", "heldout"), default="all")
    parser.add_argument("--set", choices=("all", "new", "p5"), default="all")
    args = parser.parse_args()
    args.wanted = {name for name in args.cases.split(",") if name}
    return args


def main() -> int:
    """Run the judge for one arm over the selected judgements; see the module docstring."""
    args = _parse_args()
    if not os.environ.get(RUN_REAL_LLM_ENV):
        print(f"Refusing to spend money: set {RUN_REAL_LLM_ENV}=1 to run this harness.")
        return 2
    import aura.extraction.supersession as supersession

    if args.old_prompt:
        supersession._build_messages = old_prompt_builder()  # type: ignore[assignment]
    fingerprint = prompt_fingerprint(supersession._build_messages)
    arm = ARMS[args.arm]
    arm_label = f"{arm.key}-old" if args.old_prompt else arm.key
    ledger = Ledger(args.ledger, BUCKET_CEILINGS, TOTAL_CEILING)
    settings = load_settings()
    ctx = RunContext(arm=arm, settings=settings, records={}, ledger=ledger)
    jobs = supersession_jobs(ctx, args, arm_label=arm_label, fingerprint=fingerprint)
    out_path = args.out_dir / f"supersession-{arm_label}-{args.label}.jsonl"
    done = _done_tags(out_path)
    todo = len([tag for tag, _ in jobs if tag not in done])
    state = ledger.snapshot()
    print(
        f"supersession x {arm_label} (prompt {fingerprint}): {len(jobs)} job(s), {todo} to "
        f"run, estimate ${_estimate(arm, 'supersession', todo):.3f}; bucket {args.bucket} "
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
        f"ran {ran}, skipped {skipped}{', STOPPED BY CEILING' if stopped else ''}; bucket "
        f"{args.bucket} ${state.spent(args.bucket):.4f}, total ${state.spent():.4f}",
        flush=True,
    )
    return 3 if stopped else 0


if __name__ == "__main__":
    raise SystemExit(main())
