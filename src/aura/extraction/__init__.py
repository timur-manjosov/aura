"""Automatic fact extraction (CLAUDE.md's Phase 3a): populating the knowledge model
without waiting for a moderator to run the "Add as Aura Fact" context menu.

Phase 3a-1 built only the free, local first filter
(aura.extraction.fact_worthiness) and the channel-scoping gate it would need
(aura.db.extraction_channel_config), both deliberately unwired. Phase 3a-2
closes the loop: aura.extraction.pipeline runs the free gates on every message,
batches survivors per channel in a restart-durable queue, and hands each closed
batch to aura.extraction.distiller -- the paid call that turns raw chat into
distilled sentences and, just as importantly, refuses the hedges, jokes and
hypotheticals the local filter is measurably bad at rejecting on its own.

Phase 3a-3 adds aura.extraction.supersession on top: when the dedup check flags
a candidate as a possible restatement of an existing fact, a second, separate
call judges what that similarity actually MEANS -- successor, complement,
conflict, or a false positive -- so the moderator reads a proposal with a reason
rather than a bare similarity score.

**What this package still does not do: create a fact, or retire one.** Every
candidate it produces lands in the staging table (aura.db.pending_facts) and
becomes a real, citable fact only when a moderator confirms it through
/aura-pending. A "supersession" judgment is likewise a proposal and nothing
more: no code path here calls supersede_fact, and retiring a fact remains
something only /aura-supersede does, run by a human. See reports/phase-3a-2.txt
and reports/phase-3a-3.txt for what each call was measured to actually do,
including where each is still weak.
"""
from aura.extraction.distiller import DistilledFact, distill_facts
from aura.extraction.fact_worthiness import (
    FACT_WORTHY_EXEMPLARS,
    NOT_FACT_WORTHY_EXEMPLARS,
    create_fact_worthiness_detector,
)
from aura.extraction.pipeline import (
    flush_due_batches,
    handle_extraction_message,
    run_extraction_sweeper,
    should_extract,
    sweep_interval_seconds,
    withdraw_message,
)
from aura.extraction.supersession import RelationshipJudgement, judge_relationship

__all__ = [
    "FACT_WORTHY_EXEMPLARS",
    "NOT_FACT_WORTHY_EXEMPLARS",
    "DistilledFact",
    "RelationshipJudgement",
    "create_fact_worthiness_detector",
    "distill_facts",
    "flush_due_batches",
    "judge_relationship",
    "handle_extraction_message",
    "run_extraction_sweeper",
    "should_extract",
    "sweep_interval_seconds",
    "withdraw_message",
]
