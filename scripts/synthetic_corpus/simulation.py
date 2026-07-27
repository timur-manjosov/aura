"""Running the real pipeline against the corpus, at real scale, without Discord.

The functions called here are the shipped ones -- `QuestionDetector`,
`find_similar_facts`, `evaluate_message`, `synthesize_answer` -- not
reimplementations. That is not a stylistic preference: the whole value of this
phase's output is that Phase 2b-3 can trust a threshold picked from it, and a
number produced by a parallel copy of the gate would only be trustworthy until
the copy drifted.

Discord never appears. It does not need to: Stage 1 and Stage 2 are already
pure functions of text and a database, per CLAUDE.md's own testing principle,
and Stage 3 takes a list of facts, a string and a locale. The one place the
real pipeline reaches for a `discord.Message` is the responder's *posting*
step, which this harness deliberately stops short of -- what it measures is
whether a post would have happened, which is the decision, not the network call.

Scoring runs once and produces raw numbers; every threshold question is then
answered from those numbers (see `metrics`). So a hundred candidate thresholds
cost one embedding pass between them, not a hundred.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime

import aiosqlite
from fastembed import TextEmbedding

from aura.db.connection import connection_lock
from aura.db.models import Fact
from aura.embeddings import find_similar_facts
from aura.proactive.gate import ProactiveGateConfig, evaluate_message
from aura.proactive.question_detector import QuestionDetector
from aura.synthesis import SynthesisResult, synthesize_answer
from synthetic_corpus.corpus_model import (
    MessageCategory,
    SyntheticCorpus,
    SyntheticMessage,
)
from synthetic_corpus.corpus_store import message_discord_id

logger = logging.getLogger(__name__)

# Stage 2 ranks the top two facts in production (aura.proactive.gate), because
# only the best and its runner-up matter. The simulator asks for more so it can
# also report *where* the ground-truth fact ranked when it was not first --
# "second by a hair" and "eleventh" call for different threshold decisions and
# a top-2 view cannot tell them apart.
_SIMULATION_TOP_K = 10


@dataclass
class ScoredCase:
    """Every raw number one corpus message produced, before any threshold.

    Deliberately stores scores rather than verdicts. A verdict is a score plus
    a threshold, and this phase's entire job is to leave the threshold open.
    """

    message: SyntheticMessage
    guild_id: int
    stage1_score: float
    stage1_error: str = ""
    stage2_top_score: float | None = None
    stage2_runner_up_score: float | None = None
    stage2_gap: float | None = None
    stage2_top_fact_key: str | None = None
    target_rank: int | None = None
    target_score: float | None = None
    retrieved: list[tuple[str, float]] = field(default_factory=list)
    top_facts: list[Fact] = field(default_factory=list)

    @property
    def target_is_top(self) -> bool:
        """Whether the fact this case was generated against ranked first."""
        return self.target_rank == 1


@dataclass
class Stage3Outcome:
    """One live synthesis call's result, reduced to what the policy reads."""

    case_key: str
    category: MessageCategory
    locale: str
    called: bool
    answers_question: bool | None
    cited_fact_ids: list[int]
    answer_excerpt: str
    would_post: bool
    failure: str = ""
    latency_seconds: float = 0.0


async def score_corpus(
    conn: aiosqlite.Connection,
    model: TextEmbedding,
    detector: QuestionDetector,
    corpus: SyntheticCorpus,
    fact_key_by_id: dict[int, tuple[str, str]],
    *,
    progress_every: int = 50,
) -> list[ScoredCase]:
    """Run Stage 1 and Stage 2 over every message in the corpus.

    Costs nothing beyond local CPU: `question_likeness` and `find_similar_facts`
    are both embedding-and-numpy only. That is what makes running this over
    hundreds of cases, rather than a hand-picked dozen, the default rather than
    an indulgence.

    A message whose Stage 1 scoring *raises* is recorded with its exception
    instead of aborting the run. Some malformed cases exist specifically to
    provoke that, and "which inputs raise" is a finding to report, not a reason
    to lose the other five hundred results.
    """
    scored: list[ScoredCase] = []

    for index, message in enumerate(corpus.messages, start=1):
        guild = corpus.guild_by_key(message.guild_key)

        try:
            stage1_score = await detector.question_likeness(message.content)
            stage1_error = ""
        except Exception as exc:
            stage1_score = float("-inf")
            stage1_error = f"{type(exc).__name__}: {exc}"
            logger.error("Stage 1 raised on %s: %s", message.key, stage1_error)

        case = ScoredCase(
            message=message,
            guild_id=guild.guild_id,
            stage1_score=stage1_score,
            stage1_error=stage1_error,
        )

        if not stage1_error:
            try:
                results = await find_similar_facts(
                    conn,
                    model,
                    guild_id=guild.guild_id,
                    query=message.content,
                    top_k=_SIMULATION_TOP_K,
                )
            except Exception as exc:
                case.stage1_error = f"stage2 {type(exc).__name__}: {exc}"
                logger.error("Stage 2 raised on %s: %s", message.key, case.stage1_error)
                results = []

            case.top_facts = [fact for fact, _ in results[:5]]
            case.retrieved = [
                (fact_key_by_id.get(fact.id, ("?", "?"))[1], score) for fact, score in results
            ]
            if case.retrieved:
                case.stage2_top_score = case.retrieved[0][1]
                case.stage2_top_fact_key = case.retrieved[0][0]
            if len(case.retrieved) > 1:
                case.stage2_runner_up_score = case.retrieved[1][1]
                case.stage2_gap = case.retrieved[0][1] - case.retrieved[1][1]

            targets = set(message.target_fact_keys)
            if targets:
                for rank, (fact_key, score) in enumerate(case.retrieved, start=1):
                    if fact_key in targets:
                        case.target_rank = rank
                        case.target_score = score
                        break

        scored.append(case)
        if progress_every and index % progress_every == 0:
            logger.info("scored %d/%d messages", index, len(corpus.messages))

    return scored


async def verify_gate_agreement(
    conn: aiosqlite.Connection,
    model: TextEmbedding,
    detector: QuestionDetector,
    corpus: SyntheticCorpus,
    scored: list[ScoredCase],
    config: ProactiveGateConfig,
    now: datetime,
    *,
    sample: int = 40,
) -> tuple[int, int, list[str]]:
    """Cross-check the sweep's own decision rule against the real gate.

    The sweep applies thresholds to raw scores itself, because it has to apply
    a hundred of them. That is a reimplementation of two comparisons, and a
    reimplementation that silently drifts from `evaluate_message` would make
    every number in this report describe a gate that does not exist.

    So a sample of cases is additionally run through the genuine
    `evaluate_message` at the *current* configured thresholds, and its verdict
    compared with what the sweep concludes at those same values. Returns
    (agreements, disagreements, descriptions of the disagreements).

    Runs against the same connection, which means it writes escalation rows for
    the cases it finds eligible -- harmless in a scratch database, and the
    cooldown is set to zero by the caller so those rows cannot suppress each
    other.
    """
    # Clear the ledger this check itself writes into. Without this, a second
    # run over the same scratch database finds its own rows from the first and
    # every sampled message comes back ALREADY_ESCALATED -- which does not
    # change the two fields being compared, but does make a re-run behave
    # differently from a first run for reasons that have nothing to do with
    # what is being verified.
    async with connection_lock(conn):
        await conn.execute("DELETE FROM proactive_escalations")
        await conn.commit()

    agreements = 0
    disagreements: list[str] = []
    by_key = {case.message.key: case for case in scored}

    step = max(1, len(corpus.messages) // sample)
    for position, message in enumerate(corpus.messages[::step]):
        case = by_key.get(message.key)
        if case is None or case.stage1_error:
            continue
        guild = corpus.guild_by_key(message.guild_key)

        trail = await evaluate_message(
            conn,
            model,
            detector,
            guild_id=guild.guild_id,
            channel_id=guild.guild_id + 1,
            message_id=message_discord_id(guild.index, position),
            content=message.content,
            config=config,
            now=now,
        )

        sweep_stage1_passed = case.stage1_score >= config.question_threshold
        # One comparison, matching the gate since Phase 2b-4: the confidence
        # gap is still measured onto every case, and still swept in the report,
        # but it is no longer part of the escalate/hold decision here either --
        # if it were, this integrity check would agree with a gate that no
        # longer exists.
        sweep_stage2_passed = (
            sweep_stage1_passed
            and case.stage2_top_score is not None
            and case.stage2_top_score >= config.similarity_threshold
        )

        if trail.stage1_passed == sweep_stage1_passed and bool(
            trail.stage2_passed
        ) == sweep_stage2_passed:
            agreements += 1
        else:
            disagreements.append(
                f"{message.key}: gate(stage1={trail.stage1_passed}, "
                f"stage2={trail.stage2_passed}) vs sweep(stage1={sweep_stage1_passed}, "
                f"stage2={sweep_stage2_passed}); verdict={trail.verdict}"
            )

    return agreements, len(disagreements), disagreements


async def run_stage3(
    case: ScoredCase,
    *,
    model: str,
    similarity_threshold: float,
    force: bool,
) -> Stage3Outcome:
    """Call the real `synthesize_answer` for one case and apply Trigger 2's policy.

    Mirrors `aura.proactive.responder` exactly on the two things that decide
    whether anything is posted: the facts handed to synthesis are the retrieved
    ones scoring at or above `similarity_threshold`, and a post happens only if
    the model both says `answers_question` and cites at least one fact. What is
    deliberately NOT mirrored is the `channel.send` -- the decision is the
    measurement, and the network call would only prove Discord works.

    `force=True` supplies the single best fact even when nothing clears the
    threshold. That mode exists for the adversarial pass: the point there is to
    test Stage 3's own resistance to manipulation *assuming the earlier gates
    were somehow bypassed*, which cannot be measured by a case that never
    reaches Stage 3 at all.
    """
    relevant = [
        fact
        for fact, (_, score) in zip(case.top_facts, case.retrieved, strict=False)
        if score >= similarity_threshold
    ]
    if not relevant and force:
        relevant = case.top_facts[:1]

    if not relevant:
        return Stage3Outcome(
            case_key=case.message.key,
            category=case.message.category,
            locale=case.message.locale,
            called=False,
            answers_question=None,
            cited_fact_ids=[],
            answer_excerpt="",
            would_post=False,
            failure="no fact cleared the retrieval threshold; synthesis never runs",
        )

    started = time.monotonic()
    try:
        result: SynthesisResult | None = await synthesize_answer(
            relevant, case.message.content, case.message.locale, model=model
        )
    except Exception as exc:  # synthesize_answer is documented never to raise
        return Stage3Outcome(
            case_key=case.message.key,
            category=case.message.category,
            locale=case.message.locale,
            called=True,
            answers_question=None,
            cited_fact_ids=[],
            answer_excerpt="",
            would_post=False,
            failure=f"UNEXPECTED RAISE: {type(exc).__name__}: {exc}",
            latency_seconds=time.monotonic() - started,
        )
    latency = time.monotonic() - started

    if result is None:
        return Stage3Outcome(
            case_key=case.message.key,
            category=case.message.category,
            locale=case.message.locale,
            called=True,
            answers_question=None,
            cited_fact_ids=[],
            answer_excerpt="",
            would_post=False,
            failure="synthesis returned None",
            latency_seconds=latency,
        )

    would_post = bool(result.answers_question and result.used_fact_ids)
    return Stage3Outcome(
        case_key=case.message.key,
        category=case.message.category,
        locale=case.message.locale,
        called=True,
        answers_question=result.answers_question,
        cited_fact_ids=result.used_fact_ids,
        answer_excerpt=result.answer[:300],
        would_post=would_post,
        latency_seconds=latency,
    )
