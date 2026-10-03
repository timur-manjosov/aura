"""Evaluate /aura-ask's fact retrieval offline, through the production code path. Free.

Three modes, each comparing hybrid retrieval (aura.retrieval.hybrid, what
/aura-ask does now) with the similarity-only selection /aura-ask used before,
on identical similarity scores:

* ``public``  -- a case file with invented facts (tests/fixtures/
  ask_retrieval_public_set.json), embedded locally.
* ``private`` -- a COPY of a production database, opened read-only, with a
  private case file kept under reports/ (gitignored). Prints counts only: no
  question, no fact text.
* ``scale``   -- invented guilds of 300, 500 and 2,000 facts
  (ask_retrieval_eval.scale_corpus) with quality, latency with and without the
  index cache, the cost of an adversarial question, and memory per 1,000 facts.

Every mode also measures how often a "nothing exact, maybe related" reply would
show facts for a question that selected nothing (see `related_rates`).

Usage (from the repository root)::

    .venv/bin/python scripts/evaluate_ask_retrieval.py public
    .venv/bin/python scripts/evaluate_ask_retrieval.py private \\
        --db <copy>/aura-before.db \\
        --cases reports/quality-diagnosis-2026-10-02/eval-set.private.json
    .venv/bin/python scripts/evaluate_ask_retrieval.py scale --sizes 300 500 2000

Never writes to a database it reads, and makes no network call.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
import tracemalloc
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import aiosqlite
from fastembed import TextEmbedding

REPO_ROOT: Final = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from ask_retrieval_eval.cases import (  # noqa: E402
    EvalCase,
    Metrics,
    compute_metrics,
    format_metrics,
    load_case_file,
)
from ask_retrieval_eval.production_path import (  # noqa: E402
    FactScores,
    QuestionOutcome,
    run_questions,
    seed_invented_guild,
)
from ask_retrieval_eval.scale_corpus import build_scale_corpus  # noqa: E402
from aura.config import Settings  # noqa: E402
from aura.db.repository import init_schema  # noqa: E402
from aura.embeddings import SYNTHESIS_FACT_LIMIT  # noqa: E402
from aura.retrieval.hybrid import HybridRetrievalConfig  # noqa: E402
from aura.retrieval.index_cache import LexicalIndexCache  # noqa: E402
from aura.retrieval.lexical import LexicalIndex  # noqa: E402
from aura.retrieval.stopwords import shipped_stopwords  # noqa: E402

PRODUCTION_EMBEDDING_MODEL: Final = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
PUBLIC_SET: Final = REPO_ROOT / "tests" / "fixtures" / "ask_retrieval_public_set.json"
INVENTED_GUILD_ID: Final = 1


@dataclass(frozen=True, slots=True)
class RelatedRule:
    """A candidate rule for the "nothing exact, maybe related" reply.

    Attributes
    ----------
    name
        Its label in the output.
    shows
        Whether a fact's scores make it shown as "maybe related".
    """

    name: str
    shows: Callable[[FactScores], bool]


RELATED_RULES: Final = (
    RelatedRule(
        "coverage>=0.25 or sim>=0.30", lambda s: s.coverage >= 0.25 or s.similarity >= 0.30
    ),
    RelatedRule(
        "coverage>=0.25 and sim>=0.05", lambda s: s.coverage >= 0.25 and s.similarity >= 0.05
    ),
    RelatedRule("sim>=0.30", lambda s: s.similarity >= 0.30),
    RelatedRule("sim>=0.35", lambda s: s.similarity >= 0.35),
)


def default_config() -> HybridRetrievalConfig:
    """Return the gate's numbers as a deployment without overrides runs them."""
    settings = Settings(_env_file=None, discord_token="unused")  # type: ignore[call-arg]
    return HybridRetrievalConfig.from_settings(settings, fact_limit=SYNTHESIS_FACT_LIMIT)


def metrics_pair(
    cases: Sequence[EvalCase], outcomes: Sequence[QuestionOutcome]
) -> tuple[Metrics, Metrics]:
    """Return (similarity-only, hybrid) metrics over the same outcomes."""
    return (
        compute_metrics(cases, {o.case_id: o.baseline for o in outcomes}),
        compute_metrics(cases, {o.case_id: o.hybrid for o in outcomes}),
    )


def lexical_only_irrelevant(
    cases: Sequence[EvalCase], outcomes: Sequence[QuestionOutcome], config: HybridRetrievalConfig
) -> tuple[int, int]:
    """Count selected facts that qualified only by their words, and how many of them were irrelevant.

    Positive questions only: the part of the precision change hybrid
    retrieval itself causes.
    """
    by_id = {case.case_id: case for case in cases}
    admitted = irrelevant = 0
    for outcome in outcomes:
        case = by_id[outcome.case_id]
        if case.kind != "positive":
            continue
        for fact_id in outcome.hybrid:
            if outcome.scores[fact_id].similarity < config.similarity_threshold:
                admitted += 1
                irrelevant += fact_id not in case.relevant
    return admitted, irrelevant


def related_rates(
    cases: Sequence[EvalCase], outcomes: Sequence[QuestionOutcome]
) -> list[tuple[str, int, int, int, int, int]]:
    """Measure each "maybe related" rule on the questions hybrid retrieval found nothing for.

    Returns
    -------
    list[tuple[str, int, int, int, int, int]]
        Per rule: name, negatives, negatives that would show related facts,
        positives that selected nothing, of those that would show a fact, and
        of those that would show a RELEVANT fact.
    """
    by_id = {case.case_id: case for case in cases}
    rows: list[tuple[str, int, int, int, int, int]] = []
    for rule in RELATED_RULES:
        negatives = shown_negatives = empty_positives = shown_positives = helpful = 0
        for outcome in outcomes:
            case = by_id[outcome.case_id]
            shown = [fid for fid, scores in outcome.scores.items() if rule.shows(scores)]
            if case.kind == "negative":
                negatives += 1
                shown_negatives += bool(not outcome.hybrid and shown)
            elif case.kind == "positive" and not outcome.hybrid:
                empty_positives += 1
                shown_positives += bool(shown)
                helpful += bool(set(shown) & case.relevant)
        rows.append(
            (rule.name, negatives, shown_negatives, empty_positives, shown_positives, helpful)
        )
    return rows


def print_report(
    title: str,
    cases: Sequence[EvalCase],
    outcomes: Sequence[QuestionOutcome],
    config: HybridRetrievalConfig,
    *,
    by_register: bool,
) -> None:
    """Print the metrics, the breakdowns and the related-reply rates. Counts only."""
    baseline, hybrid = metrics_pair(cases, outcomes)
    print(f"== {title}")
    print("  " + format_metrics("similarity only (before)", baseline))
    print("  " + format_metrics("hybrid (now)", hybrid))
    admitted, irrelevant = lexical_only_irrelevant(cases, outcomes, config)
    print(f"  facts admitted by words alone on positives: {admitted}, irrelevant: {irrelevant}")
    if by_register:
        groups: dict[str, list[EvalCase]] = defaultdict(list)
        for case in cases:
            groups[f"{case.kind}:{case.register}"].append(case)
        for group, members in sorted(groups.items()):
            before, after = metrics_pair(
                members, [o for o in outcomes if o.case_id in {m.case_id for m in members}]
            )
            if members[0].kind == "positive":
                print(f"    {group:28} hit {before.hits:3} -> {after.hits:3} of {len(members)}")
            else:
                print(
                    f"    {group:28} selected anything {before.paid_calls:3} -> "
                    f"{after.paid_calls:3} of {len(members)}"
                )
    print("  'maybe related' reply, on questions hybrid found nothing for:")
    for name, negatives, shown_neg, empty_pos, shown_pos, helpful in related_rates(cases, outcomes):
        rate = shown_neg / negatives if negatives else 0.0
        print(
            f"    {name:30} negatives shown {shown_neg:3}/{negatives:<3} ({rate:5.1%}) | "
            f"empty positives shown {shown_pos}/{empty_pos}, with a relevant fact {helpful}"
        )


async def evaluate_invented(
    model: TextEmbedding,
    facts_and_cases: tuple[Sequence[object], Sequence[EvalCase]],
    *,
    config: HybridRetrievalConfig,
    cache: LexicalIndexCache,
) -> tuple[list[QuestionOutcome], aiosqlite.Connection]:
    """Seed an in-memory guild with invented facts and run every question."""
    facts, cases = facts_and_cases
    conn = await aiosqlite.connect(":memory:")
    await init_schema(conn)
    id_map = await seed_invented_guild(
        conn,
        model,
        facts,  # type: ignore[arg-type]
        guild_id=INVENTED_GUILD_ID,
    )
    outcomes = await run_questions(
        conn,
        model,
        cases,
        guild_id=INVENTED_GUILD_ID,
        config=config,
        cache=cache,
        fact_id_map=id_map,
    )
    return outcomes, conn


async def run_public(model: TextEmbedding, path: Path) -> None:
    """Evaluate the shareable set of invented facts."""
    case_file = load_case_file(path)
    config = default_config()
    outcomes, conn = await evaluate_invented(
        model, (case_file.facts, case_file.cases), config=config, cache=LexicalIndexCache()
    )
    await conn.close()
    print_report(f"public set {path.name}", case_file.cases, outcomes, config, by_register=True)


async def run_private(model: TextEmbedding, db_path: Path, cases_path: Path) -> None:
    """Evaluate a read-only copy of a production database. Prints counts only."""
    case_file = load_case_file(cases_path)
    conn = await aiosqlite.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        async with conn.execute(
            "SELECT guild_id, COUNT(*) FROM facts WHERE status = 'active' "
            "GROUP BY guild_id ORDER BY COUNT(*) DESC LIMIT 1"
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            print("no active facts", file=sys.stderr)
            return
        config = default_config()
        outcomes = await run_questions(
            conn,
            model,
            case_file.cases,
            guild_id=int(row[0]),
            config=config,
            cache=LexicalIndexCache(),
        )
        print(f"(guild with the most active facts: {row[1]} active facts)")
        print_report("private set", case_file.cases, outcomes, config, by_register=True)
    finally:
        await conn.close()


def percentile(values: Sequence[float], fraction: float) -> float:
    """Return the value at a fraction of the sorted values (nearest rank)."""
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, round(fraction * len(ordered)) - 1))]


def time_scorer(
    facts: Sequence[tuple[int, str]], questions: Sequence[str]
) -> tuple[float, list[float], list[float], float]:
    """Time the lexical scorer alone: build, cached and uncached questions, memory.

    Returns
    -------
    tuple
        Build time (ms), per-question cached times (ms), per-question uncached
        times (ms, build included), and index memory in bytes.
    """
    stopwords = shipped_stopwords()
    tracemalloc.start()
    started = time.perf_counter()
    index = LexicalIndex.build(facts)
    build_ms = (time.perf_counter() - started) * 1000
    memory = tracemalloc.get_traced_memory()[0]
    tracemalloc.stop()
    cached: list[float] = []
    for question in questions:
        started = time.perf_counter()
        index.coverage(question, stopwords)
        cached.append((time.perf_counter() - started) * 1000)
    uncached: list[float] = []
    for question in questions[:40]:
        cache = LexicalIndexCache()
        started = time.perf_counter()
        cache.index_for(INVENTED_GUILD_ID, facts).coverage(question, stopwords)
        uncached.append((time.perf_counter() - started) * 1000)
    return build_ms, cached, uncached, memory


def adversarial_question(facts: Sequence[tuple[int, str]]) -> str:
    """Return a 32-word question built from the guild's most common long fact words."""
    counts: dict[str, int] = defaultdict(int)
    for _, content in facts:
        for word in content.split():
            word = word.strip(".,!?#@:'\"")
            if len(word) >= 6:
                counts[word] += 1
    common = sorted(counts, key=lambda word: -counts[word])[:32]
    return " ".join(common)


async def run_scale(model: TextEmbedding, sizes: Sequence[int]) -> None:
    """Evaluate invented guilds of the given sizes: quality, latency, memory."""
    config = default_config()
    for size in sizes:
        corpus = build_scale_corpus(size)
        cache = LexicalIndexCache()
        outcomes, conn = await evaluate_invented(
            model, (corpus.facts, corpus.cases), config=config, cache=cache
        )
        await conn.close()
        print_report(f"scale {size} facts", corpus.cases, outcomes, config, by_register=True)
        pairs = [(fact.fact_id, fact.content) for fact in corpus.facts]
        build_ms, cached, uncached, memory = time_scorer(pairs, [c.query for c in corpus.cases])
        end_to_end = [o.retrieval_ms for o in outcomes[1:]]
        print(
            f"  latency: index build {build_ms:6.1f} ms | word scoring cached p50 "
            f"{percentile(cached, 0.5):5.2f} p95 {percentile(cached, 0.95):5.2f} max "
            f"{max(cached):5.2f} ms | uncached (build included) p50 "
            f"{percentile(uncached, 0.5):6.1f} p95 {percentile(uncached, 0.95):6.1f} ms"
        )
        print(
            f"  retrieve_for_question end to end (thread hop, gate, ranking; cache warm) "
            f"p50 {percentile(end_to_end, 0.5):5.2f} p95 {percentile(end_to_end, 0.95):5.2f} ms"
        )
        attack = adversarial_question(pairs)
        index = LexicalIndex.build(pairs)
        started = time.perf_counter()
        for _ in range(5):
            index.coverage(attack, shipped_stopwords())
        attack_ms = (time.perf_counter() - started) * 1000 / 5
        characters = sum(len(content) for _, content in pairs)
        print(
            f"  adversarial 32-common-word question: {attack_ms:5.2f} ms | index memory "
            f"{memory / 1e6:5.2f} MB = {memory / size * 1000 / 1e6:4.2f} MB per 1,000 facts "
            f"({characters / size:5.1f} characters per fact)"
        )


def main(argv: Sequence[str] | None = None) -> int:
    """Run one evaluation mode.

    Parameters
    ----------
    argv
        Command-line arguments; `sys.argv[1:]` when None.

    Returns
    -------
    int
        Process exit code.
    """
    parser = argparse.ArgumentParser(description="Offline /aura-ask retrieval evaluation (free).")
    modes = parser.add_subparsers(dest="mode", required=True)
    public = modes.add_parser("public")
    public.add_argument("--cases", type=Path, default=PUBLIC_SET)
    private = modes.add_parser("private")
    private.add_argument("--db", type=Path, required=True)
    private.add_argument("--cases", type=Path, required=True)
    scale = modes.add_parser("scale")
    scale.add_argument("--sizes", type=int, nargs="+", default=[300, 500, 2000])
    args = parser.parse_args(argv)

    model = TextEmbedding(PRODUCTION_EMBEDDING_MODEL)
    if args.mode == "public":
        asyncio.run(run_public(model, args.cases))
    elif args.mode == "private":
        asyncio.run(run_private(model, args.db, args.cases))
    else:
        asyncio.run(run_scale(model, args.sizes))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
