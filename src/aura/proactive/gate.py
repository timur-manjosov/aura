"""The staged gate between "someone said something" and "spend money answering it".

Three gates, cheapest first, each one only reached if the previous one passed:

1. **Question-likeness** (free, local). Does this even read like someone asking
   for information? See aura.proactive.question_detector.
2. **Fact confidence** (free, local). Is there at least one fact similar enough
   to be worth paying to reason about? See _score_best_two_facts, and the
   "Why there is no confidence-gap gate" note below for what this stage
   deliberately does NOT decide.
3. **Budget** (durable, atomic). Is this channel off cooldown, and does this
   guild have any of today's escalations left? See aura.db.proactive_state.

**Why there is no confidence-gap gate.** Through Phase 2b-3, Stage 2 also
required the best fact to beat the runner-up by PROACTIVE_CONFIDENCE_GAP, on
the theory that two facts scoring almost equally meant a near-duplicate or an
unsuperseded contradiction, and that answering from whichever ranked first was
a coin flip. Phase 2b-4 removed that requirement, because a similarity gap
cannot tell apart the two situations it was being asked to distinguish:

  * two facts compete because one is stale and the other replaced it -- Aura
    must not answer from either without saying so; and
  * two facts compete because both are genuinely relevant and COMPLEMENTARY --
    the case CLAUDE.md's "Link" component exists for, where the right answer
    combines them and cites both.

Both produce the same small number here. Phase 2b-2's corpus already showed it
(median gaps of 0.075 / 0.069 / 0.075 across answered, partial and
contradictory cases -- indistinguishable), and Aura's first live day confirmed
it from the other direction: two genuinely complementary maintenance facts were
refused escalation three times at gaps of 0.047, 0.012 and 0.014, while
/aura-ask -- which has no such gate -- combined the very same pair correctly in
two different languages, citing both sources.

Telling those two situations apart is a judgement about MEANING, not about
geometry, so it is made where judgement lives: the synthesis prompt asks the
model to check whether the facts it would cite genuinely conflict on the same
specific detail and to refuse (answers_question=false) when they do. That is
Stage 3's job, proven on real contradictions, and this stage no longer
pre-empts it. The measured gap is still recorded on the trail -- it remains
useful diagnostic reading -- it simply decides nothing.

Nothing in this module calls an LLM or posts anything, in this sub-phase or
by accident later: the gate's entire output is a DecisionTrail describing what
it *would* do. Phase 2a-3 attaches synthesis behind the ELIGIBLE verdict.

**Why the budget gate is third and not first.** It is the only gate that
mutates anything, and it must be engaged before the expensive step it guards
-- not after a successful answer. The economic attack this defends against is
someone who knows which facts exist crafting messages designed to sail through
Stages 1 and 2 on purpose: if the cooldown were only recorded after a
successful post, every one of those messages would reach synthesis and each
one would cost real money. Claiming the slot the moment a message becomes
eligible caps that at one paid attempt per channel per cooldown window and, on
top of that, at the guild's daily cap -- whether the attempt then succeeds,
fails, or crashes.

Putting it third rather than first costs nothing that matters: Stages 1 and 2
are local CPU work with no per-call price, and evaluating them even for
messages the budget would refuse is what makes the debug trail show *why* a
message was held back rather than just *that* it was.
"""
from __future__ import annotations

import logging
from datetime import datetime
from math import isfinite
from typing import TYPE_CHECKING

import aiosqlite
from fastembed import TextEmbedding
from pydantic import BaseModel, ConfigDict, Field

from aura.db.proactive_signals import DecisionTrail, GateVerdict
from aura.db.proactive_state import (
    MAX_COOLDOWN_SECONDS,
    MAX_DAILY_CAP,
    EscalationOutcome,
    try_acquire_escalation_slot,
)
from aura.embeddings import find_similar_facts
from aura.proactive.question_detector import QuestionDetector

if TYPE_CHECKING:
    from aura.config import Settings

logger = logging.getLogger(__name__)

# Two, still, but for a different reason than before Phase 2b-4. The top score
# alone decides this stage now -- if it clears the threshold then at least one
# fact qualifies, and if it does not then none can. The runner-up is fetched
# solely so the trail can keep reporting the gap between them as diagnostic
# context. Ranking deeper here would be work whose result is thrown away: the
# responder does its own retrieval when it builds the synthesis context, and
# THAT is where the number of facts actually matters (see
# aura.embeddings.SYNTHESIS_FACT_LIMIT).
_STAGE2_TOP_K = 2

_ESCALATION_VERDICTS: dict[EscalationOutcome, GateVerdict] = {
    EscalationOutcome.GRANTED: GateVerdict.ELIGIBLE,
    EscalationOutcome.COOLDOWN_ACTIVE: GateVerdict.COOLDOWN_ACTIVE,
    EscalationOutcome.DAILY_CAP_REACHED: GateVerdict.DAILY_CAP_REACHED,
    EscalationOutcome.ALREADY_ESCALATED: GateVerdict.DUPLICATE_DELIVERY,
}


class ProactiveGateConfig(BaseModel):
    """The four numbers the gate decides with, validated once instead of per message.

    A separate model rather than passing Settings straight through, for two
    reasons: the gate stays independently testable with explicit values (no
    environment, no Discord token required to construct one), and a
    nonsensical threshold fails at startup where an operator sees it rather
    than silently disabling proactive relief in production.

    Four, not five: Phase 2b-4 removed minimum_confidence_gap from this model
    entirely rather than keeping it here unused, so nothing can read it back
    and assume it still gates something. See this module's docstring for why
    the gap stopped being a gate; PROACTIVE_CONFIDENCE_GAP survives in Settings
    only so an existing .env carrying it does not fail to start.

    Bounds are the mathematical limits of what each number is compared
    against, not taste: a contrastive score cannot leave [-2, 2] (a
    difference of two cosine similarities) and a raw similarity cannot leave
    [-1, 1].
    """

    # Rejects any field this model does not define, which is what makes the
    # removal above enforceable rather than cosmetic: a caller still passing
    # minimum_confidence_gap now fails loudly at construction instead of having
    # it silently dropped and believing a gap is still being applied. The same
    # protection catches a plain misspelling of any other threshold.
    model_config = ConfigDict(extra="forbid")

    # Strictly greater than the floor of the contrastive range, because the
    # floor is reserved: question_likeness returns it for text it cannot score
    # at all, and a threshold of exactly -2.0 would let that sentinel pass.
    question_threshold: float = Field(gt=-2.0, le=2.0, allow_inf_nan=False)
    similarity_threshold: float = Field(ge=-1.0, le=1.0, allow_inf_nan=False)
    # Upper-bounded, not merely non-negative: see MAX_COOLDOWN_SECONDS and
    # MAX_DAILY_CAP for the arithmetic that overflows past them.
    cooldown_seconds: float = Field(ge=0.0, le=MAX_COOLDOWN_SECONDS, allow_inf_nan=False)
    daily_cap: int = Field(ge=0, le=MAX_DAILY_CAP)

    @classmethod
    def from_settings(cls, settings: Settings) -> ProactiveGateConfig:
        """Build the gate's configuration from application settings.

        The single mapping from environment variables to gate behaviour, so
        no call site has to know which setting drives which stage.
        """
        return cls(
            question_threshold=settings.proactive_question_threshold,
            similarity_threshold=settings.proactive_similarity_threshold,
            cooldown_seconds=settings.proactive_cooldown_seconds,
            daily_cap=settings.proactive_daily_cap,
        )


async def evaluate_message(
    conn: aiosqlite.Connection,
    model: TextEmbedding,
    detector: QuestionDetector,
    *,
    guild_id: int,
    channel_id: int,
    message_id: int,
    content: str,
    config: ProactiveGateConfig,
    now: datetime,
) -> DecisionTrail:
    """Decide whether content is eligible to proceed to paid synthesis, and why.

    Claims an escalation slot as a side effect when, and only when, the
    returned verdict is ELIGIBLE -- see this module's docstring for why that
    happens here and not later. Every other verdict leaves the database
    untouched apart from the caller's own diagnostic row.

    Never calls an LLM and never writes a fact. Requires a timezone-aware
    `now`, passed in rather than read here, so the cooldown and the daily
    boundary are testable at the exact moments they matter.
    """
    stage1_score = await detector.question_likeness(content)
    stage1_passed = stage1_score >= config.question_threshold
    if not stage1_passed:
        return DecisionTrail(
            verdict=GateVerdict.STAGE1_REJECTED,
            stage1_score=stage1_score,
            stage1_passed=False,
        )

    top_score, runner_up_score = await _score_best_two_facts(
        conn, model, guild_id=guild_id, query=content
    )
    gap = None if top_score is None or runner_up_score is None else top_score - runner_up_score

    # The whole of Stage 2 since Phase 2b-4: does any fact clear the bar? The
    # top score answers that on its own, since nothing can clear a threshold
    # the best candidate misses. `gap` is computed and recorded beside it as
    # diagnostic context and is deliberately never compared against anything --
    # see this module's docstring for why a gap cannot decide what it was
    # previously being asked to decide.
    if top_score is None or top_score < config.similarity_threshold:
        return DecisionTrail(
            verdict=GateVerdict.NO_MATCHING_FACT,
            stage1_score=stage1_score,
            stage1_passed=True,
            stage2_top_score=top_score,
            stage2_runner_up_score=runner_up_score,
            stage2_gap=gap,
            stage2_passed=False,
        )

    attempt = await try_acquire_escalation_slot(
        conn,
        guild_id=guild_id,
        channel_id=channel_id,
        message_id=message_id,
        cooldown_seconds=config.cooldown_seconds,
        daily_cap=config.daily_cap,
        now=now,
    )

    return DecisionTrail(
        verdict=_ESCALATION_VERDICTS[attempt.outcome],
        stage1_score=stage1_score,
        stage1_passed=True,
        stage2_top_score=top_score,
        stage2_runner_up_score=runner_up_score,
        stage2_gap=gap,
        stage2_passed=True,
        cooldown_seconds_remaining=attempt.cooldown_seconds_remaining,
        daily_count=attempt.daily_count,
        daily_cap=attempt.daily_cap,
    )


async def _score_best_two_facts(
    conn: aiosqlite.Connection, model: TextEmbedding, *, guild_id: int, query: str
) -> tuple[float | None, float | None]:
    """Return the best and second-best active-fact similarity for query, or None where absent.

    Reuses Phase 1d's find_similar_facts rather than reimplementing ranking,
    which does mean this embeds the message a second time (Stage 1 already
    embedded it internally). That duplicate inference is accepted knowingly:
    it only happens for the minority of messages that pass Stage 1, it is
    local CPU work offloaded to a worker thread, and the alternative is
    widening two established APIs -- question_likeness would have to return
    its embedding alongside its score, and find_similar_facts would have to
    accept a precomputed one -- to save a few milliseconds on a path that
    spends nothing.

    Drops any result whose similarity is not finite. A fact whose stored
    embedding is degenerate cannot be meaningfully compared to anything, and
    letting a NaN through would silently lose every comparison against the
    threshold while also making the whole trail unrecordable (see
    DecisionTrail's allow_inf_nan note).
    """
    results = await find_similar_facts(
        conn, model, guild_id=guild_id, query=query, top_k=_STAGE2_TOP_K
    )

    scores: list[float] = []
    for fact, score in results:
        if isfinite(score):
            scores.append(score)
        else:
            logger.warning(
                "Fact %s in guild %s scored non-finite similarity (%r); excluding it from "
                "proactive matching, as its stored embedding is unusable",
                fact.id,
                guild_id,
                score,
            )

    top_score = scores[0] if scores else None
    runner_up_score = scores[1] if len(scores) > 1 else None
    return top_score, runner_up_score
