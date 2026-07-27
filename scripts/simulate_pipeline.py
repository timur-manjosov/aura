"""Run Aura's real Stage 1/2/3 logic against the synthetic corpus and report.

    .venv/bin/python scripts/simulate_pipeline.py

Stages 1 and 2 are FREE -- local embeddings and numpy, no network, no provider
-- so this is the part meant to be re-run at real scale as often as a threshold
question comes up. It needs no API key and no AURA_RUN_REAL_LLM, and it makes
no LLM calls whatsoever.

The two Stage 3 passes are opt-in, separately capped, and cost money:

    AURA_RUN_REAL_LLM=1 .venv/bin/python scripts/simulate_pipeline.py \\
        --stage3-adversarial --stage3-calibration

  --stage3-adversarial   puts every adversarial case straight to the live model,
                         bypassing Stages 1 and 2, to measure the last line of
                         defence on its own. Capped at MAX_STAGE3_ADVERSARIAL_CALLS.
  --stage3-calibration   a stratified sample of the accuracy-focused categories
                         through the live model. Capped at
                         MAX_STAGE3_CALIBRATION_CALLS -- see that constant for why
                         that number.

Nothing here touches Discord, and nothing here writes to a production database:
the corpus lives in an isolated scratch database whose guard refuses to open
anything that is not one (see synthetic_corpus.scratch_db).

This tool does not choose a threshold. It prints the whole sweep with the
current value marked, and Phase 2b-3 decides from it.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastembed import TextEmbedding  # noqa: E402

from aura.config import ModelComponent, load_settings  # noqa: E402
from aura.proactive.gate import ProactiveGateConfig  # noqa: E402
from aura.proactive.question_detector import QuestionDetector  # noqa: E402
from synthetic_corpus import report as report_sections  # noqa: E402
from synthetic_corpus.budget import BudgetExceededError, CallBudget  # noqa: E402
from synthetic_corpus.corpus_model import (  # noqa: E402
    CALIBRATION_CATEGORIES,
    LabelAudit,
    MessageCategory,
    Stage1Truth,
    effective_stage1_truth,
)
from synthetic_corpus.corpus_store import (  # noqa: E402
    assert_corpus_matches_database,
    assert_corpus_matches_scenario_grid,
    read_corpus,
    read_fact_key_map,
)
from synthetic_corpus.leakage import LeakageChecker  # noqa: E402
from synthetic_corpus.llm import RUN_REAL_LLM_ENV  # noqa: E402
from synthetic_corpus.pricing import PricingUnavailableError, fetch_model_prices  # noqa: E402
from synthetic_corpus.scenarios import describe_grid  # noqa: E402
from synthetic_corpus.scratch_db import DEFAULT_SCRATCH_PATH, open_scratch_database  # noqa: E402
from synthetic_corpus.simulation import (  # noqa: E402
    ScoredCase,
    Stage3Outcome,
    run_stage3,
    score_corpus,
    verify_gate_agreement,
)

logger = logging.getLogger("simulate_pipeline")

DEFAULT_CORPUS_PATH = Path("reports") / "synthetic-corpus" / "corpus.json"
DEFAULT_REPORT_PATH = Path("reports") / "phase-2b-2.txt"
DEFAULT_RESULTS_PATH = Path("reports") / "phase-2b-2-simulation.json"

# Hard ceiling on the adversarial Stage 3 pass, enforced by CallBudget rather
# than merely documented. Sized at the corpus's own adversarial case count plus
# a small margin: every adversarial case is meant to reach the model in this
# pass, and anything beyond that is a bug, not a bigger sample.
MAX_STAGE3_ADVERSARIAL_CALLS = 140

# Hard ceiling on the optional calibration sample.
#
# 90 = 9 locales x 10 cases, and the number is chosen rather than inherited.
# The Phase 2a-3 bake-off used 12 cases total, which gave each of the two
# decisive categories -- a fact that only partially answers, and two
# contradictory active facts -- exactly ONE case each; a single coin flip moved
# the whole verdict, which is why those two had to be re-run three times by
# hand. At 90 the stratification below gives each of those categories 27 cases
# spread over all nine locales, so one flip moves a category's rate by under
# four points and a per-locale weakness has three cases to show up in rather
# than one. It is still deliberately small: at Haiku's measured ~$0.0007 per
# call this is about six cents, and the free Stage 1/2 sweep -- not this -- is
# the evidence this phase is really built on.
MAX_STAGE3_CALIBRATION_CALLS = 90

# How the calibration sample is split across categories. Weighted toward the
# two hard ones for the same reason the corpus itself is.
_CALIBRATION_SAMPLE_SHAPE: dict[MessageCategory, int] = {
    MessageCategory.ANSWERED_QUESTION: 18,
    MessageCategory.UNANSWERED_QUESTION: 9,
    MessageCategory.PARTIAL_ANSWER: 27,
    MessageCategory.CONTRADICTORY_FACTS: 27,
    MessageCategory.OFF_TOPIC_CHATTER: 9,
}

# A ceiling in dollars as well as in calls, for the same reason the generator
# has both: a call count cannot catch a call that is individually far more
# expensive than expected.
DEFAULT_STAGE3_MAX_SPEND_USD = 0.50

# Fixed so the same corpus always yields the same calibration sample; a sample
# that reshuffled between runs would make two reports incomparable for no gain.
_SAMPLE_SEED = 20260725


def _stratified_sample(cases: list[ScoredCase]) -> list[ScoredCase]:
    """Pick the calibration sample: fixed shape, fixed seed, locale-balanced.

    Draws round-robin across locales within each category, so a category's
    quota is spread over all nine locales instead of being filled from whichever
    guilds happened to come first.
    """
    rng = random.Random(_SAMPLE_SEED)
    chosen: list[ScoredCase] = []

    for category, quota in _CALIBRATION_SAMPLE_SHAPE.items():
        by_locale: dict[str, list[ScoredCase]] = defaultdict(list)
        for case in cases:
            if case.message.category is category and not case.stage1_error:
                by_locale[case.message.locale].append(case)
        for pool in by_locale.values():
            rng.shuffle(pool)

        locales = sorted(by_locale)
        taken = 0
        while taken < quota and any(by_locale[locale] for locale in locales):
            for locale in locales:
                if taken >= quota:
                    break
                if by_locale[locale]:
                    chosen.append(by_locale[locale].pop())
                    taken += 1

    return chosen[:MAX_STAGE3_CALIBRATION_CALLS]


async def _run_stage3_pass(
    cases: list[ScoredCase],
    *,
    model: str,
    similarity_threshold: float,
    force: bool,
    budget: CallBudget,
    price,
    label: str,
) -> list[Stage3Outcome]:
    """Run one bounded Stage 3 pass, stopping the moment a ceiling is reached.

    The cap is enforced here, in code, by authorizing every call against the
    budget before it goes out -- not by trusting the sample size to stay small.
    A partial pass is reported as partial rather than discarded.
    """
    outcomes: list[Stage3Outcome] = []
    for index, case in enumerate(cases, start=1):
        try:
            budget.authorize(model)
        except BudgetExceededError as exc:
            logger.warning("%s stopped at %d/%d: %s", label, index - 1, len(cases), exc)
            break

        outcome = await run_stage3(
            case,
            model=model,
            similarity_threshold=similarity_threshold,
            force=force,
        )
        outcomes.append(outcome)

        # The real spend is booked from litellm's own accounting of the call in
        # `aura.synthesis`; since that function does not surface usage, the
        # budget here is charged the bake-off's measured per-call figure so the
        # dollar ceiling still bites. Deliberately an over-estimate rather than
        # an under-estimate.
        budget.record(price, input_tokens=520, output_tokens=90)

        if index % 10 == 0:
            print(f"  {label}: {index}/{len(cases)} ({budget.summary()})", flush=True)

    return outcomes


def _load_stage3(
    path: Path, scored: list[ScoredCase]
) -> tuple[dict[str, Stage3Outcome], list[Stage3Outcome], str, float]:
    """Rebuild Stage 3 outcomes from a previous run's results JSON.

    Only outcomes for cases still present in the current corpus are restored,
    so a reused file from a different corpus contributes nothing rather than
    silently mixing two runs' results.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    metadata = payload.get("metadata", {})
    reused_model = str(metadata.get("stage3_model") or "unknown model")
    # Carry the original run's spend forward rather than reporting zero:
    # the calls were genuinely paid for, just not by this invocation.
    reused_spend = float(metadata.get("stage3_spend_usd") or 0.0)
    category_by_key = {case.message.key: case.message.category for case in scored}
    locale_by_key = {case.message.key: case.message.locale for case in scored}

    outcomes: dict[str, Stage3Outcome] = {}
    calibration: list[Stage3Outcome] = []
    for row in payload.get("cases", []):
        stage3 = row.get("stage3")
        key = row.get("key")
        if not stage3 or key not in category_by_key:
            continue
        outcome = Stage3Outcome(
            case_key=key,
            category=category_by_key[key],
            locale=locale_by_key[key],
            called=bool(stage3.get("called")),
            answers_question=stage3.get("answers_question"),
            cited_fact_ids=[],
            answer_excerpt=stage3.get("answer_excerpt", ""),
            would_post=bool(stage3.get("would_post")),
            failure=stage3.get("failure", ""),
            latency_seconds=float(stage3.get("latency_seconds") or 0.0),
        )
        outcomes[key] = outcome
        if outcome.category in CALIBRATION_CATEGORIES:
            calibration.append(outcome)
    return outcomes, calibration, reused_model, reused_spend


def _write_results_json(
    path: Path,
    scored: list[ScoredCase],
    stage3: dict[str, Stage3Outcome],
    metadata: dict[str, object],
) -> None:
    """Persist every raw number, so the report can be re-derived without re-running."""
    payload = {
        "metadata": metadata,
        "cases": [
            {
                "key": case.message.key,
                "guild": case.message.guild_key,
                "category": case.message.category.value,
                "locale": case.message.locale,
                "adversarial_kind": (
                    case.message.adversarial_kind.value if case.message.adversarial_kind else None
                ),
                "stage1_truth": effective_stage1_truth(case.message).value,
                "label_audit": case.message.label_audit.value,
                "stage1_score": case.stage1_score,
                "stage1_error": case.stage1_error,
                "stage2_top_score": case.stage2_top_score,
                "stage2_runner_up_score": case.stage2_runner_up_score,
                "stage2_gap": case.stage2_gap,
                "stage2_top_fact_key": case.stage2_top_fact_key,
                "target_fact_keys": case.message.target_fact_keys,
                "target_rank": case.target_rank,
                "target_score": case.target_score,
                "stage3": (
                    {
                        "called": stage3[case.message.key].called,
                        "answers_question": stage3[case.message.key].answers_question,
                        "would_post": stage3[case.message.key].would_post,
                        "failure": stage3[case.message.key].failure,
                        "latency_seconds": round(stage3[case.message.key].latency_seconds, 3),
                        "answer_excerpt": stage3[case.message.key].answer_excerpt,
                    }
                    if case.message.key in stage3
                    else None
                ),
            }
            for case in scored
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


async def main() -> int:
    """Score the corpus, sweep the thresholds, optionally call Stage 3, and write it up."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS_PATH)
    parser.add_argument("--db", type=Path, default=DEFAULT_SCRATCH_PATH)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS_PATH)
    parser.add_argument("--stage3-adversarial", action="store_true")
    parser.add_argument("--stage3-calibration", action="store_true")
    parser.add_argument(
        "--stage3-max-spend-usd", type=float, default=DEFAULT_STAGE3_MAX_SPEND_USD
    )
    parser.add_argument(
        "--reuse-stage3",
        type=Path,
        default=None,
        help=(
            "load Stage 3 outcomes from a previous run's results JSON instead of "
            "calling the model again. Free, and the reason the raw results are "
            "persisted: re-deriving the report after fixing how it is presented "
            "should not cost another two hundred paid calls."
        ),
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    settings = load_settings()
    gate_config = ProactiveGateConfig.from_settings(settings)

    # Every refusal that can be decided without doing work is decided here,
    # before the corpus is read or the embedding model is loaded. A guard that
    # fires after thirty seconds of setup teaches an operator to run the thing
    # and see what happens, which is the opposite of what a spend guard is for.
    wants_stage3 = args.stage3_adversarial or args.stage3_calibration
    stage3_model = settings.resolve_model(ModelComponent.PROACTIVE)
    stage3_price = None
    if wants_stage3:
        if not os.environ.get(RUN_REAL_LLM_ENV):
            print(
                f"refusing to run a Stage 3 pass: {RUN_REAL_LLM_ENV} is not set. "
                "Stages 1 and 2 need no opt-in and cost nothing; re-run without the "
                "--stage3-* flags for those.",
                file=sys.stderr,
            )
            return 1
        if not stage3_model or not settings.is_llm_configured(ModelComponent.PROACTIVE):
            print("refusing to run a Stage 3 pass: no model is configured", file=sys.stderr)
            return 1
        try:
            stage3_price = fetch_model_prices([stage3_model])[stage3_model]
        except PricingUnavailableError as exc:
            print(f"{exc}", file=sys.stderr)
            return 1

    corpus = read_corpus(args.corpus)
    assert_corpus_matches_scenario_grid(corpus)
    print(f"corpus: {len(corpus.messages)} messages across {len(corpus.guilds)} guilds")
    model = TextEmbedding(settings.embedding_model)
    detector = await QuestionDetector.create(model)

    async with open_scratch_database(args.db) as conn:
        fact_key_by_id = await read_fact_key_map(conn)
        if not fact_key_by_id:
            print(
                f"{args.db} holds no synthetic facts; run generate_synthetic_corpus.py first",
                file=sys.stderr,
            )
            return 1
        assert_corpus_matches_database(corpus, fact_key_by_id)

        print("scoring Stage 1 and Stage 2 over the whole corpus (free, local)...", flush=True)
        scored = await score_corpus(conn, model, detector, corpus, fact_key_by_id)

        print("cross-checking the sweep against the real gate...", flush=True)
        # Cooldown and cap neutralised so the budget gate cannot mask a Stage
        # 1/2 disagreement -- this check is about the two threshold comparisons,
        # not about the ledger, which has its own tests.
        agreement_config = ProactiveGateConfig(
            question_threshold=gate_config.question_threshold,
            similarity_threshold=gate_config.similarity_threshold,
            minimum_confidence_gap=gate_config.minimum_confidence_gap,
            cooldown_seconds=0.0,
            daily_cap=1_000_000,
        )
        agreements, disagreement_count, disagreements = await verify_gate_agreement(
            conn,
            model,
            detector,
            corpus,
            scored,
            agreement_config,
            datetime.now(timezone.utc),
        )

        print("re-running the leakage check against the shipped exemplars...", flush=True)
        checker = await LeakageChecker.create(model)
        leakage_candidates = [
            case.message.content
            for case in scored
            if case.message.category is not MessageCategory.ADVERSARIAL_MALFORMED
        ]
        leakage_findings = await checker.check(model, leakage_candidates)
        leakage_similarities = await checker.max_similarity(model, leakage_candidates)

    stage3: dict[str, Stage3Outcome] = {}
    calibration_outcomes: list[Stage3Outcome] = []
    reused_from = ""
    stage3_budget = CallBudget(
        max_calls=MAX_STAGE3_ADVERSARIAL_CALLS + MAX_STAGE3_CALIBRATION_CALLS,
        max_spend_usd=args.stage3_max_spend_usd,
    )

    if args.reuse_stage3 is not None:
        stage3, calibration_outcomes, reused_model, reused_spend = _load_stage3(
            args.reuse_stage3, scored
        )
        stage3_model = stage3_model or reused_model
        # The reused calls were genuinely paid for, just not by this run.
        # Carrying the figure forward keeps the report honest about what the
        # numbers in it cost, rather than showing a fresh $0.00.
        stage3_budget.spent_usd = reused_spend
        reused_from = (
            f"{len(stage3)} outcomes reused from {args.reuse_stage3.name}, "
            f"originally ${reused_spend:.4f}"
        )
        print(f"\n{reused_from}; this run made no calls of its own")

    if args.stage3_adversarial and stage3_model and stage3_price:
        adversarial = [case for case in scored if case.message.adversarial_kind is not None]
        adversarial = adversarial[:MAX_STAGE3_ADVERSARIAL_CALLS]
        print(f"\nStage 3 adversarial pass: {len(adversarial)} live calls", flush=True)
        outcomes = await _run_stage3_pass(
            adversarial,
            model=stage3_model,
            similarity_threshold=settings.similarity_threshold,
            force=True,
            budget=stage3_budget,
            price=stage3_price,
            label="adversarial",
        )
        for outcome in outcomes:
            stage3[outcome.case_key] = outcome

    if args.stage3_calibration and stage3_model and stage3_price:
        sample = _stratified_sample(scored)
        print(f"\nStage 3 calibration pass: {len(sample)} live calls", flush=True)
        calibration_outcomes = await _run_stage3_pass(
            sample,
            model=stage3_model,
            similarity_threshold=settings.similarity_threshold,
            force=False,
            budget=stage3_budget,
            price=stage3_price,
            label="calibration",
        )
        for outcome in calibration_outcomes:
            stage3.setdefault(outcome.case_key, outcome)

    lines = _build_report(
        corpus=corpus,
        scored=scored,
        stage3=stage3,
        calibration_outcomes=calibration_outcomes,
        settings=settings,
        agreements=agreements,
        disagreement_count=disagreement_count,
        disagreements=disagreements,
        leakage_findings=leakage_findings,
        leakage_similarities=leakage_similarities,
        stage3_model=(
            stage3_model
            if (args.stage3_adversarial or args.stage3_calibration or reused_from)
            else None
        ),
        stage3_budget=stage3_budget,
        stage3_reused=reused_from,
    )

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _write_results_json(
        args.results,
        scored,
        stage3,
        {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "corpus_generated_at": corpus.generated_at.isoformat(),
            "generator_model": corpus.generator_model,
            "reviewer_model": corpus.reviewer_model,
            "stage3_model": stage3_model,
            "stage3_calls": stage3_budget.calls or len(stage3),
            "stage3_spend_usd": round(stage3_budget.spent_usd, 6),
            "current_thresholds": {
                "PROACTIVE_QUESTION_THRESHOLD": settings.proactive_question_threshold,
                "PROACTIVE_SIMILARITY_THRESHOLD": settings.proactive_similarity_threshold,
                "PROACTIVE_CONFIDENCE_GAP": settings.proactive_confidence_gap,
            },
        },
    )

    print("\n".join(lines[-40:]))
    print(f"\nfull report written to {args.report}")
    print(f"raw per-case results written to {args.results}")
    if stage3_budget.calls:
        print(
            f"Stage 3 spend: ~${stage3_budget.spent_usd:.4f} over "
            f"{stage3_budget.calls} live calls (estimated from the bake-off's "
            "measured per-call token usage)"
        )
    return 0


def _build_report(
    *,
    corpus,
    scored: list[ScoredCase],
    stage3: dict[str, Stage3Outcome],
    calibration_outcomes: list[Stage3Outcome],
    settings,
    agreements: int,
    disagreement_count: int,
    disagreements: list[str],
    leakage_findings,
    leakage_similarities: list[float],
    stage3_model: str | None,
    stage3_budget: CallBudget,
    stage3_reused: str = "",
) -> list[str]:
    """Assemble the whole written report from the scored cases."""
    calibration = [case for case in scored if case.message.category in CALIBRATION_CATEGORIES]
    undisputed = [
        case
        for case in calibration
        if case.message.label_audit is not LabelAudit.DISPUTE
    ]
    adversarial_scoreable = [
        case
        for case in scored
        if case.message.adversarial_kind is not None
        and effective_stage1_truth(case.message) is not Stage1Truth.NOT_SCORED
    ]

    lines: list[str] = [
        "PHASE 2b-2 -- SYNTHETIC DISCORD-SCENARIO CORPUS & PIPELINE SIMULATION",
        f"Generated: {datetime.now(timezone.utc).date().isoformat()}",
        "",
        "This report produces evidence. It does not choose a threshold -- that is",
        "Phase 2b-3's decision, and every sweep below prints its whole range with the",
        "currently-configured value marked in place rather than a recommendation.",
        "",
        f"Corpus:       {len(corpus.messages)} messages, "
        f"{sum(len(guild.facts) for guild in corpus.guilds)} facts, "
        f"{len(corpus.guilds)} guilds, 9 locales",
        f"Generator:    {corpus.generator_model}",
        f"Reviewer:     {corpus.reviewer_model} (safety filter + label audit)",
        f"Generation:   ${corpus.generation_cost_usd:.4f} over {corpus.generation_calls} calls",
        f"Stage 1/2:    free -- local embeddings only, {len(scored)} cases scored",
        f"Stage 3:      {stage3_model or 'not run'}"
        + (f", {stage3_budget.calls} live calls" if stage3_budget.calls else "")
        + (f" ({stage3_reused})" if stage3_reused else ""),
        "",
        "Current configuration under test:",
        f"  PROACTIVE_QUESTION_THRESHOLD  = {settings.proactive_question_threshold}",
        f"  PROACTIVE_SIMILARITY_THRESHOLD= {settings.proactive_similarity_threshold}",
        f"  PROACTIVE_CONFIDENCE_GAP      = {settings.proactive_confidence_gap}",
    ]

    lines += section("0. HOW TO READ THIS REPORT")
    lines += report_sections.reading_guide(calibration)

    lines += section("1. THE SCENARIO GRID")
    lines += [describe_grid()]

    lines += section("2. HARNESS INTEGRITY -- does the sweep describe the real gate?")
    lines += [
        "The sweep applies thresholds to raw scores itself, because it applies a",
        "hundred of them. That is a reimplementation of the gate's two comparisons, so",
        "a sample of cases was ALSO run through the genuine",
        "aura.proactive.gate.evaluate_message at the configured values and the two",
        "conclusions compared.",
        "",
        f"  agreements:    {agreements}",
        f"  disagreements: {disagreement_count}",
    ]
    lines += [f"    {line}" for line in disagreements[:10]]
    if disagreement_count == 0:
        lines += ["", "  The numbers below therefore describe the gate that actually ships."]

    lines += section("3. TRAIN/TEST LEAKAGE CHECK")
    lines += [
        "Every generated message compared against the 19 question and 19 statement",
        "exemplars Stage 1 is built around (read live from",
        "aura.proactive.question_detector, not copied), semantically and lexically.",
        "",
        f"  messages checked:      {len(leakage_similarities)}",
        f"  flagged as too close:  {len(leakage_findings)}",
    ]
    if leakage_similarities:
        ranked = sorted(leakage_similarities, reverse=True)
        lines += [
            f"  closest similarity:    {ranked[0]:.3f} (flag threshold 0.85)",
            f"  median similarity:     {ranked[len(ranked) // 2]:.3f}",
            f"  90th percentile:       {ranked[int(len(ranked) * 0.1)]:.3f}",
        ]
    for finding in leakage_findings[:10]:
        lines.append("  " + finding.describe())
    if not leakage_findings:
        lines += [
            "",
            "  Nothing in the corpus is a paraphrase of a sentence the classifier was",
            "  built around, so the numbers below are not the detector grading its own",
            "  reference material.",
        ]

    lines += section("4. LABEL AUDIT -- are the ground-truth labels actually true?")
    lines += report_sections.label_audit_summary(corpus)

    lines += section("5. STAGE 1 -- PROACTIVE_QUESTION_THRESHOLD SWEEP")
    lines += report_sections.stage1_sweep_table(
        calibration,
        settings.proactive_question_threshold,
        label="5a. All calibration cases",
    )
    lines += [""]
    lines += report_sections.stage1_sweep_table(
        undisputed,
        settings.proactive_question_threshold,
        label="5b. Same sweep, excluding cases the independent label audit disputed",
    )
    lines += [""]
    lines += report_sections.stage1_sweep_table(
        [*calibration, *adversarial_scoreable],
        settings.proactive_question_threshold,
        label="5c. Same sweep, with toxic/contentless adversarial cases added as negatives",
    )

    lines += section("6. STAGE 1 -- SCORE DISTRIBUTIONS")
    lines += report_sections.stage1_distributions(scored)

    lines += section("7. STAGE 2 -- SIMILARITY x CONFIDENCE-GAP SWEEP")
    lines += report_sections.stage2_sweep_table(
        calibration,
        settings.proactive_similarity_threshold,
        settings.proactive_confidence_gap,
    )

    lines += section("8. STAGE 2 -- RETRIEVAL QUALITY AND SCORE DISTRIBUTIONS")
    lines += report_sections.stage2_retrieval_quality(calibration)

    lines += section("9. THE PARTIAL-ANSWER TRADE-OFF (reported, not scored)")
    lines += report_sections.partial_answer_tradeoff(
        calibration,
        settings.proactive_similarity_threshold,
        settings.proactive_confidence_gap,
    )

    lines += section("10. ADVERSARIAL & MALFORMED INPUT -- END-TO-END")
    lines += report_sections.adversarial_report(
        scored,
        stage3,
        settings.proactive_question_threshold,
        settings.proactive_similarity_threshold,
        settings.proactive_confidence_gap,
    )

    lines += section("11. STAGE 3 CALIBRATION SAMPLE (optional pass)")
    lines += report_sections.stage3_calibration_summary(calibration_outcomes)
    if stage3_budget.calls or stage3_reused:
        lines += [
            "",
            f"Stage 3 spend: ~${stage3_budget.spent_usd:.4f} over "
            f"{stage3_budget.calls or len(stage3)} calls, against an in-code ceiling of "
            f"{stage3_budget.max_calls} calls / ${stage3_budget.max_spend_usd:.2f}.",
        ]
    if stage3_reused:
        lines += [
            f"({stage3_reused} -- this invocation made no calls of its own.)",
        ]

    lines += section("12. WHAT THIS CORPUS FOUND (derived, not recommended)")
    lines += report_sections.key_observations(
        calibration,
        settings.proactive_question_threshold,
        settings.proactive_similarity_threshold,
        settings.proactive_confidence_gap,
    )
    lines += [
        "",
        "No threshold is proposed here. Phase 2b-3 owns that decision and has the",
        "sweeps above, the per-locale distributions, and the raw per-case JSON to",
        "make it from.",
    ]

    return lines


def section(title: str) -> list[str]:
    """Delegate to the report module's section header, so the style lives in one place."""
    return report_sections.section(title)


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
