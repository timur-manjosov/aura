"""Generate the Phase 3a-1 fact-worthiness synthetic corpus.

    AURA_RUN_REAL_LLM=1 .venv/bin/python scripts/generate_extraction_corpus.py

THIS SPENDS REAL MONEY, and it is a one-off tooling cost, not a recurring one
-- same framing as scripts/generate_synthetic_corpus.py, which this script is
a deliberate sibling of (see scripts/extraction_corpus/ for why the corpus
shape itself is simpler: one filter to calibrate, not three pipeline stages).

Run it with --dry-run first. That costs nothing and prints the full cost
estimate against live OpenRouter pricing.

Model selection reasoning is identical to generate_synthetic_corpus.py's own
(bulk content generation, not judgment; cheap; genuinely multilingual;
reliable JSON) and reuses the exact same two models for that reason -- there
is no argument for a different generator or reviewer here that wasn't already
made there.
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

from aura.extraction.fact_worthiness import (  # noqa: E402
    FACT_WORTHY_EXEMPLARS,
    NOT_FACT_WORTHY_EXEMPLARS,
)
from extraction_corpus.corpus_model import SyntheticCorpus  # noqa: E402
from extraction_corpus.corpus_store import write_corpus  # noqa: E402
from extraction_corpus.generator import GenerationContext, audit_labels, generate_locale  # noqa: E402
from extraction_corpus.scenarios import SCENARIOS, TOTAL_PER_LOCALE, describe_grid  # noqa: E402
from synthetic_corpus.budget import BudgetExceededError, CallBudget, ModelPrice  # noqa: E402
from synthetic_corpus.leakage import LeakageChecker  # noqa: E402
from synthetic_corpus.llm import GenerationError, require_real_llm_optin, resolve_api_key  # noqa: E402
from synthetic_corpus.pricing import PricingUnavailableError, fetch_model_prices  # noqa: E402

logger = logging.getLogger("generate_extraction_corpus")

DEFAULT_GENERATOR_MODEL = "openrouter/google/gemini-3.1-flash-lite"
DEFAULT_REVIEWER_MODEL = "openrouter/openai/gpt-4o-mini"

DEFAULT_CORPUS_PATH = Path("reports") / "extraction-corpus" / "corpus.json"

DEFAULT_MAX_CALLS = 340
DEFAULT_MAX_SPEND_USD = 0.75

_CHARACTERS_PER_TOKEN = 3.2

# Measured-shape estimate, same method as generate_synthetic_corpus.py's own:
# four generator calls per locale (fact-worthy, ordinary, hedged, adversarial)
# rather than nine, and label-audit batches of 10 rather than 8.
#
# Input tokens per call are held constant: the prompt text (instructions,
# register rules, the requested `count`) barely changes size with `count`
# itself. Output tokens do NOT scale that way -- a call asked for 150 items
# returns roughly 5x the JSON of a call asked for 30 -- so output is modelled
# per requested item rather than per call. The per-item rate below is backed
# out from Phase 3a-1's own actual run (reports/phase-3a-1.txt): 900 measured
# output tokens per call at an average of 459 requested items / 36 generator
# calls = 12.75 items/call, i.e. ~70.6 tokens/item. Phase 3a-1b scales the
# per-locale item counts 5x (scripts/extraction_corpus/scenarios.py), so this
# fix matters here specifically -- the old fixed-per-call estimate would have
# understated the real output volume by roughly 5x.
_MEASURED_INPUT_PER_CALL = 700
_MEASURED_OUTPUT_TOKENS_PER_ITEM = 900 / (459 / 36)


def _estimate_calls_and_tokens(locale_count: int) -> tuple[int, int, int, int]:
    generator_calls = 4 * locale_count
    total_messages = TOTAL_PER_LOCALE * locale_count
    audit_batches = -(-total_messages // 10)
    reviewer_calls = audit_batches
    input_tokens = generator_calls * _MEASURED_INPUT_PER_CALL
    output_tokens = round(total_messages * _MEASURED_OUTPUT_TOKENS_PER_ITEM)
    return generator_calls, reviewer_calls, input_tokens, output_tokens


def _print_cost_estimate(
    locale_count: int, generator_price: ModelPrice, reviewer_price: ModelPrice
) -> float:
    generator_calls, reviewer_calls, input_tokens, output_tokens = _estimate_calls_and_tokens(
        locale_count
    )
    generator_cost = generator_price.cost(input_tokens=input_tokens, output_tokens=output_tokens)
    reviewer_input = int(reviewer_calls * 2000 / _CHARACTERS_PER_TOKEN)
    reviewer_output = reviewer_calls * 100
    reviewer_cost = reviewer_price.cost(input_tokens=reviewer_input, output_tokens=reviewer_output)
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
    print(f"  ESTIMATED TOTAL: ${total:.4f} for {locale_count} locales")
    print("-" * 78)
    return total


async def _apply_leakage_filter(model: TextEmbedding, corpus: SyntheticCorpus) -> tuple[int, list[float]]:
    """Drop every message too close to a fact-worthiness reference exemplar.

    Same reasoning as generate_synthetic_corpus.py's own leakage filter,
    retargeted at this filter's exemplar pair via LeakageChecker's
    `exemplars` override -- built for exactly this reuse (see leakage.py).
    """
    exemplars = (*FACT_WORTHY_EXEMPLARS, *NOT_FACT_WORTHY_EXEMPLARS)
    checker = await LeakageChecker.create(model, exemplars=exemplars)
    contents = [message.content for message in corpus.messages]
    if not contents:
        return 0, []

    findings = await checker.check(model, contents)
    flagged = {finding.text for finding in findings}
    finding_by_text = {finding.text: finding for finding in findings}

    if flagged:
        print(f"\nLEAKAGE: {len(flagged)} generated message(s) flagged and removed")
        for finding in findings[:10]:
            print("  " + finding.describe())

    from extraction_corpus.corpus_model import RejectedCase

    kept = []
    for message in corpus.messages:
        if message.content in flagged:
            finding = finding_by_text[message.content]
            corpus.rejected.append(
                RejectedCase(
                    category=message.category,
                    locale=message.locale,
                    reason=(
                        f"near-duplicate of a fact-worthiness exemplar "
                        f"({finding.triggered_by}: cos={finding.cosine:.3f}, "
                        f"lex={finding.lexical:.3f})"
                    ),
                    layer="leakage",
                )
            )
            continue
        kept.append(message)

    corpus.messages = kept
    distribution = await checker.max_similarity(model, contents)
    return len(flagged), distribution


def _print_composition(corpus: SyntheticCorpus) -> None:
    print("\nCORPUS COMPOSITION")
    print("-" * 78)
    by_category: dict[str, int] = {}
    by_locale: dict[str, int] = {}
    for message in corpus.messages:
        by_category[message.category.value] = by_category.get(message.category.value, 0) + 1
        by_locale[message.locale] = by_locale.get(message.locale, 0) + 1

    for category, count in sorted(by_category.items(), key=lambda pair: -pair[1]):
        print(f"  {category:<28} {count:>4}")
    print(f"  {'TOTAL':<28} {len(corpus.messages):>4}")
    print("\n  by locale: " + ", ".join(f"{loc}={count}" for loc, count in sorted(by_locale.items())))

    positive = sum(1 for m in corpus.messages if m.is_fact_worthy)
    total = len(corpus.messages) or 1
    print(f"\n  fact-worthy: {positive}/{len(corpus.messages)} ({positive / total:.1%})")

    if corpus.rejected:
        print(f"\n  rejected before entering the corpus: {len(corpus.rejected)}")
        by_layer: dict[str, int] = {}
        for rejection in corpus.rejected:
            by_layer[rejection.layer] = by_layer.get(rejection.layer, 0) + 1
        for layer, count in sorted(by_layer.items()):
            print(f"    {layer:<20} {count:>4}")


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generator-model", default=DEFAULT_GENERATOR_MODEL)
    parser.add_argument("--reviewer-model", default=DEFAULT_REVIEWER_MODEL)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS_PATH)
    parser.add_argument("--max-calls", type=int, default=DEFAULT_MAX_CALLS)
    parser.add_argument("--max-spend-usd", type=float, default=DEFAULT_MAX_SPEND_USD)
    parser.add_argument(
        "--limit", type=int, default=0, help="generate only the first N locales (0 = all)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the cost estimate and the scenario grid, then exit without calling anything",
    )
    parser.add_argument(
        "--yes", action="store_true", help="skip the interactive spend confirmation"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    scenarios = SCENARIOS[: args.limit] if args.limit > 0 else SCENARIOS

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

    require_real_llm_optin("generate_extraction_corpus.py")

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
        messages=[],
    )

    stopped_early = ""
    for scenario in scenarios:
        print(f"\n--- {scenario.locale} ---", flush=True)
        try:
            generated = await generate_locale(context, scenario)
        except BudgetExceededError as exc:
            stopped_early = str(exc)
            print(f"\nSTOPPED: {exc}", file=sys.stderr)
            break
        except GenerationError as exc:
            print(f"  generation failed for {scenario.locale}: {exc}", file=sys.stderr)
            continue

        corpus.messages.extend(generated.messages)
        corpus.rejected.extend(generated.rejected)
        print(
            f"  {len(generated.messages)} messages, {len(generated.rejected)} rejected | "
            f"{budget.summary()}",
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

    if not corpus.messages:
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
            f"exemplar = {ranked[min(dropped, len(ranked) - 1)]:.3f}, "
            f"median = {ranked[len(ranked) // 2]:.3f}"
        )

    problems = corpus.check_referential_integrity()
    if problems:
        print(f"\nCORPUS INTEGRITY: {len(problems)} problem(s)", file=sys.stderr)
        for problem in problems[:20]:
            print(f"  {problem}", file=sys.stderr)
        return 1

    write_corpus(corpus, args.corpus)
    _print_composition(corpus)
    print(f"\ncorpus written to {args.corpus}")
    print(f"\nACTUAL SPEND: ${budget.spent_usd:.4f} over {budget.calls} calls")
    print(f"  (estimated beforehand: ${estimate:.4f})")
    for line in budget.per_model_summary():
        print("  " + line)

    if stopped_early:
        print(f"\nNOTE: the run stopped early -- {stopped_early}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
