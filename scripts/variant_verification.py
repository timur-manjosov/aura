"""Real, paid verification of Multi-Representation Indexing Part 1's pipeline.

    AURA_RUN_REAL_LLM=1 .venv/bin/python scripts/variant_verification.py --dry-run
    AURA_RUN_REAL_LLM=1 .venv/bin/python scripts/variant_verification.py

THIS SPENDS REAL MONEY. Same structural separation as every other measurement
script here: deliberately not a pytest test, not under tests/, unreachable from
a bare `pytest` (pytest.ini pins `testpaths = tests`), and it refuses to run at
all unless AURA_RUN_REAL_LLM is set -- the one signal tests/conftest.py keys its
hermetic guard off, set by nothing but a human who means it.

WHAT THIS IS. Drives the REAL aura.variants_service functions -- the same
_generate_variants and _audit_variants the production pipeline calls, with the
same prompts, the same parsing, the same fail-closed rules -- against nine
hand-written canonical facts: three fact shapes (a channel-specific rule with
an explicit exception, a different channel-specific rule with no exception at
all -- the over-generalisation risk case, and a plain event fact as a
baseline), each written in three locales (en-US, de, ja).

WHAT IT MEASURES:

  * whether the exception/qualifier survives into every generated variant, and
    whether the independent audit model actually catches it when it does not
    (rather than the generation model simply never dropping it, which would
    prove nothing about the audit).
  * whether the channel-specific case's variants stay scoped to the one named
    channel, and again whether the audit catches an over-generalisation if the
    generator produces one.
  * diversity among the STORED (post-audit) variants for each case: mean and
    max pairwise cosine similarity, computed with the real, shipped embedding
    model -- not the audit model's opinion, a separate, measurable signal.
  * how many variants get rejected in practice, and why (the audit's own
    reasoning sentence, printed verbatim).
"""
from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np  # noqa: E402
from fastembed import TextEmbedding  # noqa: E402

from aura.config import load_settings  # noqa: E402
from aura.embeddings import cosine_similarity, embed_texts  # noqa: E402
from aura.variants_service import (  # noqa: E402
    _audit_variants,
    _generate_variants,
)

RUN_REAL_LLM_ENV = "AURA_RUN_REAL_LLM"

# The shipped defaults (see .env.example): Haiku 4.5 generates, a different
# vendor (GPT-4o-mini) audits. Not read from settings on purpose -- this
# script verifies a specific measured claim about these two specific models,
# so a deployment that has since repointed either config value must not
# silently change what "verified" means here.
GENERATION_MODEL = "openrouter/anthropic/claude-haiku-4.5"
AUDIT_MODEL = "openrouter/openai/gpt-4o-mini"

VARIANT_COUNT = 6

# Live OpenRouter pricing per million tokens, re-checked 2026-07-31 rather than
# assumed -- the standing rule in this project since reports/model-bakeoff.txt
# Section 2 found a carried price already stale.
_HAIKU_PRICE_IN_PER_MTOK = 1.00
_HAIKU_PRICE_OUT_PER_MTOK = 5.00
_GPT4O_MINI_PRICE_IN_PER_MTOK = 0.15
_GPT4O_MINI_PRICE_OUT_PER_MTOK = 0.60

# Rough token counts for the dry-run estimate only.
_GENERATION_PROMPT_TOKENS = 500
_GENERATION_OUTPUT_TOKENS = 400
_AUDIT_PROMPT_TOKENS = 700
_AUDIT_OUTPUT_TOKENS = 300


@dataclass
class VerificationCase:
    """One canonical fact this script generates and audits variants for."""

    name: str
    locale: str
    canonical: str
    # Substrings whose presence in a variant is the exception/qualifier this
    # case is checking for. Empty for cases with no exception to preserve.
    required_substrings: tuple[str, ...] = ()
    # Substring identifying the specific, narrow scope (a channel or role
    # name) that must not be generalised away. Empty for cases with no
    # narrow-scope risk.
    scope_substring: str = ""


CASES: list[VerificationCase] = [
    # --- Case A: explicit exception/qualifier, three locales ---------------
    VerificationCase(
        name="exception-en-US",
        locale="en-US",
        canonical=(
            "Uploads in #trading are capped at 5MB, except on Saturdays "
            "when the limit is lifted."
        ),
        required_substrings=("saturday",),
    ),
    VerificationCase(
        name="exception-de",
        locale="de",
        canonical=(
            "Uploads in #handel sind auf 5MB begrenzt, außer samstags, "
            "wenn die Grenze aufgehoben ist."
        ),
        required_substrings=("samstag",),
    ),
    VerificationCase(
        name="exception-ja",
        locale="ja",
        canonical=(
            "#取引チャンネルでのアップロードは5MBに制限されていますが、"
            "土曜日はこの制限が解除されます。"
        ),
        required_substrings=("土曜",),
    ),
    # --- Case B: channel-specific rule, over-generalisation risk ------------
    VerificationCase(
        name="scope-en-US",
        locale="en-US",
        canonical=(
            "Voice channel #late-night-lounge is reserved for members with "
            "the Night Owl role only."
        ),
        scope_substring="late-night-lounge",
    ),
    VerificationCase(
        name="scope-de",
        locale="de",
        canonical=(
            "Der Sprachkanal #nachtschwaermer ist ausschließlich Mitgliedern "
            "mit der Rolle Nachteule vorbehalten."
        ),
        scope_substring="nachtschwaermer",
    ),
    VerificationCase(
        name="scope-ja",
        locale="ja",
        canonical=(
            "ボイスチャンネル #夜更かしラウンジ は「夜型」ロールを"
            "持つメンバー専用です。"
        ),
        scope_substring="夜更かしラウンジ",
    ),
    # --- Case C: plain baseline, no special risk ----------------------------
    VerificationCase(
        name="plain-en-US",
        locale="en-US",
        canonical="The winter tournament starts Saturday at 6pm.",
    ),
    VerificationCase(
        name="plain-de",
        locale="de",
        canonical="Das Winterturnier beginnt am Samstag um 18 Uhr.",
    ),
    VerificationCase(
        name="plain-ja",
        locale="ja",
        canonical="冬季トーナメントは土曜日の午後6時に始まります。",
    ),
]


@dataclass
class VariantOutcome:
    content: str
    faithful: bool
    reasoning: str
    has_required_substrings: bool
    keeps_scope: bool


@dataclass
class CaseResult:
    case: VerificationCase
    generated: list[str] = field(default_factory=list)
    outcomes: list[VariantOutcome] = field(default_factory=list)
    generation_latency: float = 0.0
    audit_latency: float = 0.0
    audit_failed: bool = False
    mean_pairwise_similarity: float | None = None
    max_pairwise_similarity: float | None = None

    @property
    def stored(self) -> list[VariantOutcome]:
        return [o for o in self.outcomes if o.faithful]

    @property
    def dropped_qualifier_undetected(self) -> bool:
        """A variant missing the required exception that the audit still approved."""
        return any(
            o.faithful and not o.has_required_substrings for o in self.outcomes
        )

    @property
    def over_generalised_undetected(self) -> bool:
        """A variant losing the narrow scope that the audit still approved."""
        return any(o.faithful and not o.keeps_scope for o in self.outcomes)


def _has_required_substrings(text: str, case: VerificationCase) -> bool:
    if not case.required_substrings:
        return True
    lowered = text.casefold()
    return all(substring.casefold() in lowered for substring in case.required_substrings)


def _keeps_scope(text: str, case: VerificationCase) -> bool:
    if not case.scope_substring:
        return True
    return case.scope_substring.casefold() in text.casefold()


def _estimate() -> float:
    per_generation_call = (
        _GENERATION_PROMPT_TOKENS / 1_000_000 * _HAIKU_PRICE_IN_PER_MTOK
        + _GENERATION_OUTPUT_TOKENS / 1_000_000 * _HAIKU_PRICE_OUT_PER_MTOK
    )
    per_audit_call = (
        _AUDIT_PROMPT_TOKENS / 1_000_000 * _GPT4O_MINI_PRICE_IN_PER_MTOK
        + _AUDIT_OUTPUT_TOKENS / 1_000_000 * _GPT4O_MINI_PRICE_OUT_PER_MTOK
    )
    return len(CASES) * (per_generation_call + per_audit_call)


async def _run_case(case: VerificationCase, embedding_model: TextEmbedding) -> CaseResult:
    result = CaseResult(case=case)

    started = time.monotonic()
    generated = await _generate_variants(case.canonical, count=VARIANT_COUNT, model=GENERATION_MODEL)
    result.generation_latency = time.monotonic() - started
    if not generated:
        return result
    result.generated = generated

    started = time.monotonic()
    verdicts = await _audit_variants(canonical=case.canonical, variants=generated, model=AUDIT_MODEL)
    result.audit_latency = time.monotonic() - started
    if verdicts is None:
        result.audit_failed = True
        return result

    for content, verdict in zip(generated, verdicts, strict=True):
        result.outcomes.append(
            VariantOutcome(
                content=content,
                faithful=verdict.faithful,
                reasoning=verdict.reasoning,
                has_required_substrings=_has_required_substrings(content, case),
                keeps_scope=_keeps_scope(content, case),
            )
        )

    stored_contents = [o.content for o in result.outcomes if o.faithful]
    if len(stored_contents) >= 2:
        embeddings = await embed_texts(embedding_model, stored_contents)
        pairs = [
            cosine_similarity(a, b) for a, b in itertools.combinations(embeddings, 2)
        ]
        result.mean_pairwise_similarity = float(np.mean(pairs))
        result.max_pairwise_similarity = float(np.max(pairs))

    return result


def _print_report(results: list[CaseResult]) -> None:
    print("\n" + "=" * 100)
    print(f"PER-CASE RESULT (generation={GENERATION_MODEL}, audit={AUDIT_MODEL})")
    print("=" * 100)
    for result in results:
        case = result.case
        print(f"\n{case.name}  [{case.locale}]")
        print(f"  canonical: {case.canonical}")
        if not result.generated:
            print("  GENERATION FAILED -- no variants produced")
            continue
        if result.audit_failed:
            print(f"  generated {len(result.generated)} variant(s), AUDIT FAILED -- nothing stored")
            continue
        print(
            f"  generated {len(result.generated)}, stored {len(result.stored)}, "
            f"rejected {len(result.generated) - len(result.stored)}"
        )
        for outcome in result.outcomes:
            mark = "KEEP" if outcome.faithful else "DROP"
            flags = []
            if case.required_substrings and not outcome.has_required_substrings:
                flags.append("MISSING-EXCEPTION")
            if case.scope_substring and not outcome.keeps_scope:
                flags.append("SCOPE-CHANGED")
            flag_str = f" [{', '.join(flags)}]" if flags else ""
            print(f"    {mark}{flag_str}: {outcome.content}")
            if not outcome.faithful:
                print(f"         audit reasoning: {outcome.reasoning}")
        if result.mean_pairwise_similarity is not None:
            print(
                f"  diversity: mean pairwise sim={result.mean_pairwise_similarity:.3f}, "
                f"max={result.max_pairwise_similarity:.3f}"
            )
        if result.dropped_qualifier_undetected:
            print("  *** WARNING: a variant dropped the exception AND the audit approved it ***")
        if result.over_generalised_undetected:
            print("  *** WARNING: a variant lost the narrow scope AND the audit approved it ***")

    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    generation_failures = [r for r in results if not r.generated]
    audit_failures = [r for r in results if r.generated and r.audit_failed]
    print(f"  cases with a failed generation call   {len(generation_failures)}/{len(results)}")
    print(f"  cases with a failed audit call         {len(audit_failures)}/{len(results)}")
    total_generated = sum(len(r.generated) for r in results)
    total_stored = sum(len(r.stored) for r in results)
    print(f"  total variants generated               {total_generated}")
    print(f"  total variants stored (post-audit)      {total_stored}")
    print(f"  total variants rejected by audit        {total_generated - total_stored}")
    undetected_qualifier = [r for r in results if r.dropped_qualifier_undetected]
    undetected_scope = [r for r in results if r.over_generalised_undetected]
    print(f"  cases with an undetected dropped exception   {len(undetected_qualifier)}")
    print(f"  cases with an undetected over-generalisation {len(undetected_scope)}")


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "reports" / "variant-indexing-part1.json",
    )
    args = parser.parse_args()

    cost = _estimate()
    if args.dry_run:
        print("DRY RUN -- nothing was spent.\n")
        print(f"  cases              {len(CASES)}")
        print(f"  calls              {len(CASES) * 2} ({len(CASES)} generation + {len(CASES)} audit)")
        print(f"  generation model   {GENERATION_MODEL}")
        print(f"  audit model        {AUDIT_MODEL}")
        print(f"  estimated cost     ${cost:.4f}")
        return 0

    if not os.environ.get(RUN_REAL_LLM_ENV):
        print(
            f"Refusing to run: set {RUN_REAL_LLM_ENV}=1 to make real, paid model calls.\n"
            "Use --dry-run to see the cost estimate first.",
            file=sys.stderr,
        )
        return 1

    if not load_settings().llm_api_key:
        print("Refusing to run: no LLM_API_KEY configured.", file=sys.stderr)
        return 1

    print(f"Running {len(CASES) * 2} call(s) (estimated ${cost:.4f})\n", flush=True)

    embedding_model = TextEmbedding(load_settings().embedding_model)

    results: list[CaseResult] = []
    for case in CASES:
        result = await _run_case(case, embedding_model)
        status = (
            "GEN-FAILED" if not result.generated
            else "AUDIT-FAILED" if result.audit_failed
            else f"{len(result.stored)}/{len(result.generated)} stored"
        )
        print(f"  {case.name:<16} {status}", flush=True)
        results.append(result)

    _print_report(results)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "generation_model": GENERATION_MODEL,
                "audit_model": AUDIT_MODEL,
                "variant_count_requested": VARIANT_COUNT,
                "estimated_cost_usd": cost,
                "cases": [
                    {
                        "name": r.case.name,
                        "locale": r.case.locale,
                        "canonical": r.case.canonical,
                        "generated": r.generated,
                        "audit_failed": r.audit_failed,
                        "outcomes": [
                            {
                                "content": o.content,
                                "faithful": o.faithful,
                                "reasoning": o.reasoning,
                                "has_required_substrings": o.has_required_substrings,
                                "keeps_scope": o.keeps_scope,
                            }
                            for o in r.outcomes
                        ],
                        "mean_pairwise_similarity": r.mean_pairwise_similarity,
                        "max_pairwise_similarity": r.max_pairwise_similarity,
                        "dropped_qualifier_undetected": r.dropped_qualifier_undetected,
                        "over_generalised_undetected": r.over_generalised_undetected,
                    }
                    for r in results
                ],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
