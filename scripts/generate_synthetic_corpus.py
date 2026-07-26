"""Generate the Phase 2b-2 synthetic Discord-scenario corpus.

    AURA_RUN_REAL_LLM=1 .venv/bin/python scripts/generate_synthetic_corpus.py

THIS SPENDS REAL MONEY, and it is a one-off tooling cost, not a recurring one:
the corpus it produces is a file, and every later question about thresholds is
answered by re-running the (free) simulator against that file. Like
scripts/model_bakeoff.py it deliberately does not live under tests/ --
pytest.ini pins `testpaths = tests`, so a bare `pytest` cannot reach it -- and
it refuses to run at all unless AURA_RUN_REAL_LLM is set.

Run it with --dry-run first. That costs nothing, prints the full cost estimate
against live OpenRouter pricing, and is the number to look at before deciding
to spend anything.

Generation model selection (CLAUDE.md's LLM Usage & Model Selection applied to
a tooling task rather than a bot component): this is bulk content generation,
not a judgment call, so the reasoning-depth and calibration arguments that
decided PROACTIVE_MODEL do not apply and a cheaper model is correct. What the
task *does* need is genuine multilingual fluency -- a corpus whose Japanese and
Korean are stilted machine translation would poison the very locale axis this
phase exists to measure -- and reliable JSON, since every batch comes back as a
structured list. Prices are re-checked live at run time rather than trusted
from a note in this repository; the model bake-off already found one of this
project's recorded prices had gone stale by 3.7x.

The safety reviewer is deliberately a model from a *different vendor* than the
generator. A generator grading its own adversarial output is not an independent
review -- it shares whatever blind spot produced the text.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastembed import TextEmbedding  # noqa: E402

from synthetic_corpus.budget import BudgetExceededError, CallBudget, ModelPrice  # noqa: E402
from synthetic_corpus.corpus_model import (  # noqa: E402
    ADVERSARIAL_CATEGORIES,
    MessageCategory,
    RejectedCase,
    SyntheticCorpus,
)
from synthetic_corpus.corpus_store import store_corpus, write_corpus  # noqa: E402
from synthetic_corpus.generator import (  # noqa: E402
    GenerationContext,
    audit_labels,
    generate_guild,
    review_adversarial,
)
from synthetic_corpus.leakage import LeakageChecker  # noqa: E402
from synthetic_corpus.llm import GenerationError, require_real_llm_optin, resolve_api_key  # noqa: E402
from synthetic_corpus.pricing import PricingUnavailableError, fetch_model_prices  # noqa: E402
from synthetic_corpus.safety import (  # noqa: E402
    SAFETY_PROBES,
    SafetyLayer,
    deterministic_verdict,
)
from synthetic_corpus.scenarios import (  # noqa: E402
    MESSAGES_PER_CATEGORY,
    SCENARIOS,
    describe_grid,
)
from synthetic_corpus.scratch_db import (  # noqa: E402
    DEFAULT_SCRATCH_PATH,
    ScratchDatabaseSafetyError,
    assert_scratch_destination_usable,
    open_scratch_database,
)

logger = logging.getLogger("generate_synthetic_corpus")

# Bulk generation: cheap, strongly multilingual, native structured output. See
# this module's docstring for why those three and not raw price alone.
DEFAULT_GENERATOR_MODEL = "openrouter/google/gemini-3.1-flash-lite"

# Independent safety review and label audit, deliberately a different vendor
# from the generator. Also deliberately a NON-REASONING model: the first run of
# this tooling used qwen3.5-flash, which spent ~13,700 output tokens of hidden
# reasoning on each two-field safety verdict -- 82% of the whole run's cost, for
# a yes/no answer. The per-model token breakdown in the report exists so that
# never goes unnoticed again.
DEFAULT_REVIEWER_MODEL = "openrouter/openai/gpt-4o-mini"

DEFAULT_CORPUS_PATH = Path("reports") / "synthetic-corpus" / "corpus.json"

# Ceilings, not expectations. The estimate below lands near 170 calls; this
# leaves room for the retry path without leaving room for a runaway loop.
DEFAULT_MAX_CALLS = 400
DEFAULT_MAX_SPEND_USD = 1.00

# Rough characters-per-token for mixed-script text. Used only for the pre-run
# estimate -- actual spend is booked from the provider's own reported usage, so
# a bad guess here misleads the operator for one screen and cannot affect the
# cap.
_CHARACTERS_PER_TOKEN = 3.2


def _estimate_calls_and_tokens(guild_count: int) -> tuple[int, int, int, int]:
    """Estimate (generator calls, reviewer calls, input tokens, output tokens).

    Per guild: one call each for base facts, contradiction facts, the five
    calibration categories and the two adversarial categories -- nine generator
    calls -- plus one reviewer call per adversarial case and one per batch of
    the label audit.

    The per-call token figures are measured from the first real run of this
    tooling rather than guessed from prompt length, which understated them by a
    factor of three. They remain an estimate; the ceiling that actually bounds
    the run is `CallBudget`, booked from the provider's own reported usage.
    """
    adversarial_cases = (
        MESSAGES_PER_CATEGORY["adversarial_injection"]
        + MESSAGES_PER_CATEGORY["adversarial_toxic"]
    )
    calibration_cases = sum(
        count
        for name, count in MESSAGES_PER_CATEGORY.items()
        if not name.startswith("adversarial")
    )
    generator_calls = 9 * guild_count
    audit_batches = -(-(calibration_cases + adversarial_cases) // 8) * guild_count
    reviewer_calls = adversarial_cases * guild_count + audit_batches

    measured_input_per_call = 1100
    measured_output_per_call = 1500
    input_tokens = generator_calls * measured_input_per_call
    output_tokens = generator_calls * measured_output_per_call
    return generator_calls, reviewer_calls, input_tokens, output_tokens


def _print_cost_estimate(
    guild_count: int, generator_price: ModelPrice, reviewer_price: ModelPrice
) -> float:
    """Print the pre-run cost estimate and return the estimated total in USD."""
    generator_calls, reviewer_calls, input_tokens, output_tokens = _estimate_calls_and_tokens(
        guild_count
    )
    generator_cost = generator_price.cost(input_tokens=input_tokens, output_tokens=output_tokens)
    # Reviewer prompts are a fixed rubric plus either one short message (safety
    # review) or eight (label audit); responses are small JSON objects.
    reviewer_input = int(reviewer_calls * 2600 / _CHARACTERS_PER_TOKEN)
    reviewer_output = reviewer_calls * 60
    reviewer_cost = reviewer_price.cost(
        input_tokens=reviewer_input, output_tokens=reviewer_output
    )
    total = generator_cost + reviewer_cost

    print("\nPRE-RUN COST ESTIMATE (live OpenRouter pricing, fetched just now)")
    print("-" * 78)
    print(
        f"  generator {generator_price.model}\n"
        f"    ${generator_price.usd_per_million_input:.3f} in / "
        f"${generator_price.usd_per_million_output:.3f} out per Mtok\n"
        f"    {generator_calls} calls, ~{input_tokens:,} in / ~{output_tokens:,} out "
        f"-> ${generator_cost:.4f}"
    )
    print(
        f"  reviewer  {reviewer_price.model}\n"
        f"    ${reviewer_price.usd_per_million_input:.3f} in / "
        f"${reviewer_price.usd_per_million_output:.3f} out per Mtok\n"
        f"    {reviewer_calls} calls, ~{reviewer_input:,} in / ~{reviewer_output:,} out "
        f"-> ${reviewer_cost:.4f}"
    )
    print(f"  ESTIMATED TOTAL: ${total:.4f} for {guild_count} guilds")
    print("-" * 78)
    return total


async def _apply_leakage_filter(
    model: TextEmbedding, corpus: SyntheticCorpus
) -> tuple[int, list[float]]:
    """Drop every message too close to a Stage 1 reference exemplar.

    Returns (dropped count, the whole corpus's max-similarity distribution).
    Dropped, not warned about: a message that re-tests the classifier against a
    paraphrase of its own exemplars makes the evaluation look better than the
    system is, and a corpus that reports a flattering threshold is worse than
    one that reports none.

    Malformed cases are exempt. They are constructed, not generated -- several
    are *derived* from the guild's own question by design -- so "close to an
    exemplar" is not evidence of leakage for them, and their whole purpose is
    to be degenerate text no similarity number describes usefully.
    """
    checker = await LeakageChecker.create(model)
    candidates = [
        message
        for message in corpus.messages
        if message.category is not MessageCategory.ADVERSARIAL_MALFORMED
    ]
    if not candidates:
        return 0, []

    findings = await checker.check(model, [message.content for message in candidates])
    flagged = {finding.text for finding in findings}
    finding_by_text = {finding.text: finding for finding in findings}

    if flagged:
        print(f"\nLEAKAGE: {len(flagged)} generated message(s) flagged and removed")
        for finding in findings[:10]:
            print("  " + finding.describe())

    kept: list = []
    for message in corpus.messages:
        if message.content in flagged:
            finding = finding_by_text[message.content]
            corpus.rejected.append(
                RejectedCase(
                    category=message.category,
                    locale=message.locale,
                    reason=(
                        f"near-duplicate of a Stage 1 exemplar "
                        f"({finding.triggered_by}: cos={finding.cosine:.3f}, "
                        f"lex={finding.lexical:.3f})"
                    ),
                    layer="leakage",
                    content=message.content,
                )
            )
            continue
        kept.append(message)

    corpus.messages = kept
    distribution = await checker.max_similarity(
        model, [message.content for message in candidates]
    )
    return len(flagged), distribution


def _print_composition(corpus: SyntheticCorpus) -> None:
    """Print what the corpus actually ended up containing, by category and locale."""
    print("\nCORPUS COMPOSITION")
    print("-" * 78)
    by_category: dict[str, int] = {}
    by_locale: dict[str, int] = {}
    for message in corpus.messages:
        by_category[message.category.value] = by_category.get(message.category.value, 0) + 1
        by_locale[message.locale] = by_locale.get(message.locale, 0) + 1

    for category, count in sorted(by_category.items(), key=lambda pair: -pair[1]):
        print(f"  {category:<26} {count:>4}")
    print(f"  {'TOTAL':<26} {len(corpus.messages):>4}")
    print("\n  by locale: " + ", ".join(f"{locale}={count}" for locale, count in sorted(by_locale.items())))
    print(f"  facts: {sum(len(guild.facts) for guild in corpus.guilds)} across {len(corpus.guilds)} guilds")

    if corpus.rejected:
        print(f"\n  rejected before entering the corpus: {len(corpus.rejected)}")
        by_layer: dict[str, int] = {}
        for rejection in corpus.rejected:
            by_layer[rejection.layer] = by_layer.get(rejection.layer, 0) + 1
        for layer, count in sorted(by_layer.items()):
            print(f"    {layer:<20} {count:>4}")


async def _verify_safety_filter(*, reviewer_model: str, reviewer_price: ModelPrice) -> int:
    """Prove the two-layer safety filter still rejects, and say which layer did it.

    A run that reports "0 cases refused" is ambiguous on its own: it means
    either that the generator produced nothing unsafe, or that the filter
    stopped working. This resolves that ambiguity by feeding it cases that MUST
    be rejected, including two written specifically so only the model-review
    layer can catch them.
    """
    budget = CallBudget(max_calls=len(SAFETY_PROBES) * 3, max_spend_usd=0.05)
    context = GenerationContext(
        generator_model=reviewer_model,
        reviewer_model=reviewer_model,
        generator_price=reviewer_price,
        reviewer_price=reviewer_price,
        budget=budget,
        api_key=resolve_api_key(),
    )

    print("SAFETY FILTER VERIFICATION")
    print("-" * 78)
    print("Every probe below MUST be rejected. Probes marked 'layer 2 only' use")
    print("languages and wording the keyword scan cannot see, so they verify the")
    print("independent model review rather than the term list.\n")

    failures = 0
    for name, text, expect_deterministic in SAFETY_PROBES:
        deterministic = deterministic_verdict(text)
        decision = await review_adversarial(context, text)
        aim = "layer 1" if expect_deterministic else "layer 2 only"

        caught_by_expected_layer = (
            deterministic.accepted is not expect_deterministic
            if expect_deterministic
            else deterministic.accepted and not decision.accepted
        )
        ok = not decision.accepted and caught_by_expected_layer
        failures += 0 if ok else 1
        print(
            f"  [{'PASS' if ok else 'FAIL'}] {name:<28} ({aim:<12}) "
            f"rejected={not decision.accepted} by={decision.layer or '-'}: {decision.reason}"
        )

    print(f"\n{budget.summary()}")
    if failures:
        print(f"\n{failures} probe(s) were NOT rejected as expected.", file=sys.stderr)
        return 1
    print("\nAll probes rejected by the layer they were aimed at.")
    return 0


async def main() -> int:
    """Estimate, confirm, generate, screen, and store the corpus."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generator-model", default=DEFAULT_GENERATOR_MODEL)
    parser.add_argument("--reviewer-model", default=DEFAULT_REVIEWER_MODEL)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS_PATH)
    parser.add_argument("--db", type=Path, default=DEFAULT_SCRATCH_PATH)
    parser.add_argument("--max-calls", type=int, default=DEFAULT_MAX_CALLS)
    parser.add_argument("--max-spend-usd", type=float, default=DEFAULT_MAX_SPEND_USD)
    parser.add_argument(
        "--limit", type=int, default=0, help="generate only the first N guilds (0 = all)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the cost estimate and the scenario grid, then exit without calling anything",
    )
    parser.add_argument(
        "--yes", action="store_true", help="skip the interactive spend confirmation"
    )
    parser.add_argument(
        "--verify-safety-filter",
        action="store_true",
        help=(
            "run the safety probes through the real two-layer filter and exit. "
            "Costs a few hundredths of a cent and proves the filter still rejects."
        ),
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    # Validate the destination before anything else, including before the cost
    # estimate. The scratch-database guard would catch a bad --db path anyway,
    # but it lives at the *end* of the run, where catching it means an operator
    # has already spent the whole generation budget to be told the output has
    # nowhere to go. A guard that only fires after the money is gone is a guard
    # in name.
    try:
        await assert_scratch_destination_usable(args.db)
    except ScratchDatabaseSafetyError as exc:
        print(f"refusing to run: {exc}", file=sys.stderr)
        return 1

    scenarios = SCENARIOS[: args.limit] if args.limit > 0 else SCENARIOS

    if args.verify_safety_filter:
        require_real_llm_optin("generate_synthetic_corpus.py --verify-safety-filter")
        try:
            prices = fetch_model_prices([args.reviewer_model])
        except PricingUnavailableError as exc:
            print(f"{exc}", file=sys.stderr)
            return 1
        return await _verify_safety_filter(
            reviewer_model=args.reviewer_model, reviewer_price=prices[args.reviewer_model]
        )

    print("SCENARIO GRID")
    print("-" * 78)
    print(describe_grid())

    try:
        prices = fetch_model_prices([args.generator_model, args.reviewer_model])
    except PricingUnavailableError as exc:
        print(f"\n{exc}", file=sys.stderr)
        return 1

    estimate = _print_cost_estimate(
        len(scenarios), prices[args.generator_model], prices[args.reviewer_model]
    )

    if args.dry_run:
        print("\n--dry-run: nothing was called and nothing was spent.")
        return 0

    require_real_llm_optin("generate_synthetic_corpus.py")

    if estimate > args.max_spend_usd:
        print(
            f"\nrefusing to start: the estimate (${estimate:.4f}) already exceeds the "
            f"--max-spend-usd ceiling (${args.max_spend_usd:.2f}).",
            file=sys.stderr,
        )
        return 1

    if not args.yes:
        answer = input(f"\nProceed and spend up to ${args.max_spend_usd:.2f}? [y/N] ")
        if answer.strip().lower() not in {"y", "yes"}:
            print("aborted; nothing was spent.")
            return 0

    budget = CallBudget(max_calls=args.max_calls, max_spend_usd=args.max_spend_usd)
    context = GenerationContext(
        generator_model=args.generator_model,
        reviewer_model=args.reviewer_model,
        generator_price=prices[args.generator_model],
        reviewer_price=prices[args.reviewer_model],
        budget=budget,
        api_key=resolve_api_key(),
    )

    corpus = SyntheticCorpus(
        generated_at=datetime.now(timezone.utc),
        generator_model=args.generator_model,
        reviewer_model=args.reviewer_model,
        guilds=[],
        messages=[],
    )

    stopped_early = ""
    for scenario in scenarios:
        print(f"\n--- {scenario.key} ({scenario.locale}, {scenario.size}) ---", flush=True)
        try:
            generated = await generate_guild(context, scenario)
        except BudgetExceededError as exc:
            stopped_early = str(exc)
            print(f"\nSTOPPED: {exc}", file=sys.stderr)
            break
        except GenerationError as exc:
            print(f"  generation failed for {scenario.key}: {exc}", file=sys.stderr)
            continue

        corpus.guilds.append(generated.guild)
        corpus.messages.extend(generated.messages)
        corpus.rejected.extend(generated.rejected)
        print(
            f"  {len(generated.guild.facts)} facts, {len(generated.messages)} messages, "
            f"{len(generated.rejected)} rejected | {budget.summary()}",
            flush=True,
        )

    if corpus.messages and not stopped_early:
        print("\n--- label audit (independent model, not the generator) ---", flush=True)
        try:
            agreements, disputes = await audit_labels(context, corpus.messages)
            total = agreements + disputes
            rate = disputes / total if total else 0.0
            print(
                f"  {agreements} agree, {disputes} dispute "
                f"({rate:.1%} of {total} audited) | {budget.summary()}"
            )
        except BudgetExceededError as exc:
            stopped_early = str(exc)
            print(f"  STOPPED during the label audit: {exc}", file=sys.stderr)

    corpus.generation_cost_usd = round(budget.spent_usd, 6)
    corpus.generation_calls = budget.calls

    if not corpus.guilds:
        print("\nnothing was generated; not writing a corpus.", file=sys.stderr)
        print(f"actual spend: ${budget.spent_usd:.4f}")
        return 1

    embedding_model = TextEmbedding(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    dropped, distribution = await _apply_leakage_filter(embedding_model, corpus)
    if distribution:
        ranked = sorted(distribution, reverse=True)
        print(
            f"\nleakage check: {dropped} dropped; closest surviving similarity to any "
            f"Stage 1 exemplar = {ranked[min(dropped, len(ranked) - 1)]:.3f}, "
            f"median = {ranked[len(ranked) // 2]:.3f}"
        )

    problems = corpus.check_referential_integrity()
    if problems:
        print(f"\nCORPUS INTEGRITY: {len(problems)} problem(s)", file=sys.stderr)
        for problem in problems[:20]:
            print(f"  {problem}", file=sys.stderr)
        return 1

    write_corpus(corpus, args.corpus)
    async with open_scratch_database(args.db, reset=True) as conn:
        await store_corpus(conn, embedding_model, corpus)

    _print_composition(corpus)
    print(f"\ncorpus written to {args.corpus}")
    print(f"scratch database written to {args.db}")
    print(f"\nACTUAL SPEND: ${budget.spent_usd:.4f} over {budget.calls} calls")
    print(f"  (estimated beforehand: ${estimate:.4f})")
    for line in budget.per_model_summary():
        print("  " + line)

    adversarial_kept = sum(
        1 for message in corpus.messages if message.category in ADVERSARIAL_CATEGORIES
    )
    safety_rejections = sum(
        1
        for rejection in corpus.rejected
        if rejection.layer in {SafetyLayer.DETERMINISTIC, SafetyLayer.MODEL_REVIEW}
    )
    print(
        f"  adversarial cases kept: {adversarial_kept}; refused by the safety "
        f"filter: {safety_rejections}"
    )

    if stopped_early:
        print(f"\nNOTE: the run stopped early -- {stopped_early}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
