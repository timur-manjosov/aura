"""Tests for aura.proactive.gate: the staged decision between silence and spending.

Two kinds of test live here, deliberately mixed:

* Stage ordering and short-circuiting, with a stubbed detector and controlled
  fact scores, because what is under test is the gate's logic and not the
  model's judgement -- and a stub makes "Stage 2 was never evaluated" provable
  rather than inferred.
* Real semantic behaviour against the real embedding model, because the claim
  that a genuine repeat question beats the runner-up fact by a clear margin is
  a claim about embeddings, and a mock cannot exercise it.

No Discord anywhere in this file, per CLAUDE.md's testing principle: the gate
takes IDs and a string, not a Message.
"""
from __future__ import annotations

import asyncio
import logging
import math
import re
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import aiosqlite
import numpy as np
import pytest
from fastembed import TextEmbedding
from pydantic import ValidationError

from aura.config import Settings
from aura.db.proactive_signals import GateVerdict
from aura.db.proactive_state import EscalationOutcome, count_escalations_on
from aura.db.repository import init_schema
from aura.embeddings import EMBEDDING_DTYPE
from aura.facts_service import add_fact
from aura.proactive.gate import (
    _ESCALATION_VERDICTS,
    ProactiveGateConfig,
    evaluate_message,
)
from aura.proactive.question_detector import QuestionDetector

GUILD_A = 100000000000000001
GUILD_B = 200000000000000002
CHANNEL_1 = 5551
CHANNEL_2 = 5552

NOON = datetime(2026, 7, 24, 12, 0, 0, tzinfo=timezone.utc)

# A config with round numbers, so a test's intent is readable from its inputs
# rather than from the production defaults (which are placeholders and will
# move in Phase 2b -- a test pinned to them would break on retuning).
CONFIG = ProactiveGateConfig(
    question_threshold=0.0,
    similarity_threshold=0.5,
    cooldown_seconds=900.0,
    daily_cap=5,
)


@pytest.fixture
async def conn():
    connection = await aiosqlite.connect(":memory:")
    await init_schema(connection)
    yield connection
    await connection.close()


def _stub_detector(score: float) -> MagicMock:
    detector = MagicMock(spec=QuestionDetector)
    detector.question_likeness = AsyncMock(return_value=score)
    return detector


class _FixedSimilarityModel:
    """An embedding model whose vectors have chosen cosine similarities to the query.

    Stage 2's threshold and gap arithmetic has to be testable at exact values,
    which real sentences cannot be steered to. Each fact is placed at a chosen
    angle from the query direction: with the query at e0, fact i is
    s_i * e0 + sqrt(1 - s_i^2) * e_(i+1), a unit vector whose dot product with
    the query is exactly s_i.

    Every fact leans on its own extra dimension so the similarities are
    independent of each other. Laying the facts along plain basis vectors
    instead would make them mutually orthogonal, which silently caps what the
    set can express -- two facts cannot both sit at 0.9 from one query if they
    are at 90 degrees from each other -- and that is precisely the
    near-duplicate case the confidence gap exists to catch.
    """

    def __init__(self, similarities: list[float]) -> None:
        self._similarities = similarities

    def embed(self, documents: list[str], **_kwargs: object):
        dimension = len(self._similarities) + 1
        for document in documents:
            vector = np.zeros(dimension, dtype=np.float32)
            if document.startswith("fact:"):
                similarity = self._similarities[int(document.split(":")[1])]
                vector[0] = similarity
                vector[int(document.split(":")[1]) + 1] = math.sqrt(
                    max(0.0, 1.0 - similarity**2)
                )
            else:
                vector[0] = 1.0
            yield vector


async def _seed_scored_facts(
    conn: aiosqlite.Connection, similarities: list[float], *, guild_id: int = GUILD_A
) -> _FixedSimilarityModel:
    """Seed one fact per requested similarity and return a model that reproduces them."""
    model = _FixedSimilarityModel(similarities)
    for index in range(len(similarities)):
        await add_fact(
            conn,
            model,  # type: ignore[arg-type]
            guild_id=guild_id,
            channel_id=CHANNEL_1,
            message_id=1000 + index,
            content=f"fact:{index}",
        )
    return model


async def _evaluate(
    conn: aiosqlite.Connection,
    model: object,
    detector: object,
    *,
    content: str = "where are the rules?",
    guild_id: int = GUILD_A,
    channel_id: int = CHANNEL_1,
    message_id: int = 1,
    config: ProactiveGateConfig = CONFIG,
    now: datetime = NOON,
):
    return await evaluate_message(
        conn,
        model,  # type: ignore[arg-type]
        detector,  # type: ignore[arg-type]
        guild_id=guild_id,
        channel_id=channel_id,
        message_id=message_id,
        content=content,
        config=config,
        now=now,
    )


class TestStageOneGate:
    """The first phase where the question-likeness score decides anything."""

    async def test_a_score_below_the_threshold_is_rejected(
        self, conn: aiosqlite.Connection
    ) -> None:
        model = await _seed_scored_facts(conn, [0.99])

        decision = await _evaluate(conn, model, _stub_detector(-0.5))

        assert decision.verdict is GateVerdict.STAGE1_REJECTED
        assert decision.stage1_passed is False
        assert decision.would_escalate is False

    async def test_a_score_exactly_at_the_threshold_passes(
        self, conn: aiosqlite.Connection
    ) -> None:
        # Inclusive on purpose: the threshold is the lowest acceptable score,
        # and an off-by-one here would silently move the gate a whole step.
        model = await _seed_scored_facts(conn, [0.99])

        decision = await _evaluate(conn, model, _stub_detector(CONFIG.question_threshold))

        assert decision.stage1_passed is True
        assert decision.verdict is GateVerdict.ELIGIBLE

    async def test_a_rejected_message_never_reaches_stage_two(
        self, conn: aiosqlite.Connection
    ) -> None:
        # Proven by the absence of the numbers rather than by inspecting calls:
        # None means "never evaluated", and a fact scoring 0.99 sits right
        # there waiting to be found if the short-circuit ever breaks.
        model = await _seed_scored_facts(conn, [0.99])

        decision = await _evaluate(conn, model, _stub_detector(-1.0))

        assert decision.stage2_top_score is None
        assert decision.stage2_passed is None
        assert decision.stage2_gap is None

    async def test_a_rejected_message_never_touches_the_budget(
        self, conn: aiosqlite.Connection
    ) -> None:
        model = await _seed_scored_facts(conn, [0.99])

        decision = await _evaluate(conn, model, _stub_detector(-1.0))

        assert decision.daily_count is None
        assert decision.cooldown_seconds_remaining is None
        assert await count_escalations_on(conn, guild_id=GUILD_A, day="2026-07-24") == 0

    async def test_the_unscoreable_sentinel_can_never_pass_the_gate(
        self, conn: aiosqlite.Connection
    ) -> None:
        # question_likeness returns -2.0 for text it cannot score (empty,
        # degenerate embedding). Under the previous one-sided score that
        # sentinel was 0.0, which the calibrated negative threshold would now
        # have let through -- the exact regression this asserts against.
        model = await _seed_scored_facts(conn, [0.99])
        negative_threshold = CONFIG.model_copy(update={"question_threshold": -1.9})

        decision = await _evaluate(
            conn, model, _stub_detector(-2.0), config=negative_threshold
        )

        assert decision.verdict is GateVerdict.STAGE1_REJECTED


class TestStageTwoSimilarity:
    async def test_a_confident_unopposed_match_passes(
        self, conn: aiosqlite.Connection
    ) -> None:
        model = await _seed_scored_facts(conn, [0.9, 0.1])

        decision = await _evaluate(conn, model, _stub_detector(0.5))

        assert decision.stage2_top_score == pytest.approx(0.9)
        assert decision.stage2_runner_up_score == pytest.approx(0.1)
        assert decision.stage2_gap == pytest.approx(0.8)
        assert decision.stage2_passed is True
        assert decision.verdict is GateVerdict.ELIGIBLE

    async def test_a_match_below_the_threshold_is_refused(
        self, conn: aiosqlite.Connection
    ) -> None:
        model = await _seed_scored_facts(conn, [0.49, 0.0])

        decision = await _evaluate(conn, model, _stub_detector(0.5))

        assert decision.verdict is GateVerdict.NO_MATCHING_FACT
        assert decision.stage2_passed is False
        assert decision.stage2_top_score == pytest.approx(0.49)

    async def test_a_match_exactly_at_the_threshold_passes(
        self, conn: aiosqlite.Connection
    ) -> None:
        # Inclusive, and proven exactly: the threshold is set to the score the
        # gate actually computed, rather than to the value the vectors were
        # built from. An embedding round-trips through float32, so a literal
        # 0.5 here would compare against 0.49999997 and test float precision
        # instead of the boundary rule.
        model = await _seed_scored_facts(conn, [0.5, 0.0])
        probe = await _evaluate(conn, model, _stub_detector(0.5), message_id=1)
        assert probe.stage2_top_score is not None

        at_the_boundary = CONFIG.model_copy(
            update={"similarity_threshold": probe.stage2_top_score}
        )
        # A different channel and message, so neither the cooldown nor the
        # duplicate guard can decide this instead of Stage 2.
        decision = await _evaluate(
            conn,
            model,
            _stub_detector(0.5),
            channel_id=CHANNEL_2,
            message_id=2,
            config=at_the_boundary,
        )

        assert decision.verdict is GateVerdict.ELIGIBLE

    async def test_a_guild_with_no_facts_at_all_is_refused_without_crashing(
        self, conn: aiosqlite.Connection
    ) -> None:
        # The state every new server starts in. It must be an ordinary "no",
        # not an IndexError on an empty result list.
        model = _FixedSimilarityModel([0.9])

        decision = await _evaluate(conn, model, _stub_detector(0.5))

        assert decision.verdict is GateVerdict.NO_MATCHING_FACT
        assert decision.stage2_top_score is None
        assert decision.stage2_passed is False

    async def test_a_guild_with_exactly_one_fact_has_no_runner_up_to_beat(
        self, conn: aiosqlite.Connection
    ) -> None:
        # Nothing competes, so there is no ambiguity to guard against and the
        # gap check must not invent one by treating a missing runner-up as 0.
        model = await _seed_scored_facts(conn, [0.9])

        decision = await _evaluate(conn, model, _stub_detector(0.5))

        assert decision.stage2_runner_up_score is None
        assert decision.stage2_gap is None
        assert decision.verdict is GateVerdict.ELIGIBLE

    async def test_only_this_guilds_facts_are_considered(
        self, conn: aiosqlite.Connection
    ) -> None:
        # A near-perfect match in another server must not make this server's
        # message answerable.
        model = await _seed_scored_facts(conn, [0.99], guild_id=GUILD_B)

        decision = await _evaluate(conn, model, _stub_detector(0.5), guild_id=GUILD_A)

        assert decision.verdict is GateVerdict.NO_MATCHING_FACT
        assert decision.stage2_top_score is None


class TestTheConfidenceGapNoLongerGates:
    """Phase 2b-4: competing facts escalate instead of being silenced.

    The inverse of what this class asserted through Phase 2b-3. The gap it used
    to enforce could not tell "one of these is stale" apart from "both of these
    are relevant and complementary" -- they produce the same number -- so the
    distinction moved to Stage 3, which is asked about it directly. These tests
    pin the removal so it cannot be reintroduced by accident: a near-tie is now
    a reason to let the model look, never a reason to stay silent.
    """

    async def test_two_similarly_scored_facts_now_escalate(
        self, conn: aiosqlite.Connection
    ) -> None:
        # The live case in miniature: two facts both clearly about the question,
        # 0.02 apart. Through Phase 2b-3 this was AMBIGUOUS_FACTS and Aura said
        # nothing -- while /aura-ask, given the same pair, combined them.
        model = await _seed_scored_facts(conn, [0.92, 0.9])

        decision = await _evaluate(conn, model, _stub_detector(0.5))

        assert decision.verdict is GateVerdict.ELIGIBLE
        assert decision.stage2_passed is True
        # Still measured and still recorded -- as diagnostics, deciding nothing.
        assert decision.stage2_gap == pytest.approx(0.02, abs=1e-6)

    async def test_two_identically_scored_facts_now_escalate(
        self, conn: aiosqlite.Connection
    ) -> None:
        # An exact tie is the strongest form of the old block, and the case
        # where a numeric rule is least able to say anything useful.
        model = await _seed_scored_facts(conn, [0.9, 0.9])

        decision = await _evaluate(conn, model, _stub_detector(0.5))

        assert decision.verdict is GateVerdict.ELIGIBLE
        assert decision.stage2_gap == pytest.approx(0.0, abs=1e-6)

    async def test_a_vanishing_gap_between_strong_facts_still_escalates(
        self, conn: aiosqlite.Connection
    ) -> None:
        # Smaller than any gap observed live (0.012 was the tightest), so the
        # removal is pinned past the real evidence rather than at it.
        model = await _seed_scored_facts(conn, [0.9001, 0.9])

        decision = await _evaluate(conn, model, _stub_detector(0.5))

        assert decision.verdict is GateVerdict.ELIGIBLE

    async def test_the_runner_up_no_longer_competes_at_all(
        self, conn: aiosqlite.Connection
    ) -> None:
        # 0.48 does not clear the bar on its own. Through Phase 2b-3 it could
        # still veto the 0.50 fact above it by sitting close to it; now a fact
        # that does not qualify as an answer cannot silence one that does.
        model = await _seed_scored_facts(conn, [0.5, 0.48])

        decision = await _evaluate(conn, model, _stub_detector(0.5))

        assert decision.verdict is GateVerdict.ELIGIBLE
        assert decision.stage2_runner_up_score == pytest.approx(0.48, abs=1e-6)

    async def test_the_gate_config_no_longer_accepts_a_gap_at_all(self) -> None:
        # Removed from the model rather than left in and ignored, so nothing can
        # set it and quietly believe it still gates something.
        assert "minimum_confidence_gap" not in ProactiveGateConfig.model_fields
        with pytest.raises(ValidationError):
            ProactiveGateConfig(
                question_threshold=0.0,
                similarity_threshold=0.5,
                minimum_confidence_gap=0.1,  # type: ignore[call-arg]
                cooldown_seconds=900.0,
                daily_cap=5,
            )

    async def test_a_below_threshold_top_score_is_still_refused(
        self, conn: aiosqlite.Connection
    ) -> None:
        # The similarity bar is the one Stage 2 check left; removing the gap
        # must not have removed it too.
        model = await _seed_scored_facts(conn, [0.49, 0.48])

        decision = await _evaluate(conn, model, _stub_detector(0.5))

        assert decision.verdict is GateVerdict.NO_MATCHING_FACT
        assert decision.stage2_passed is False

    async def test_only_the_top_two_facts_are_scored_for_the_trail(
        self, conn: aiosqlite.Connection
    ) -> None:
        # The gate still ranks only two: the top decides, the runner-up is
        # recorded. A crowd further down changes neither.
        model = await _seed_scored_facts(conn, [0.9, 0.2, 0.19, 0.18, 0.17, 0.16])

        decision = await _evaluate(conn, model, _stub_detector(0.5))

        assert decision.stage2_runner_up_score == pytest.approx(0.2)
        assert decision.verdict is GateVerdict.ELIGIBLE

    async def test_a_non_finite_similarity_is_excluded_rather_than_poisoning_the_decision(
        self, conn: aiosqlite.Connection, caplog: pytest.LogCaptureFixture
    ) -> None:
        # A fact whose stored embedding is degenerate cannot be compared to
        # anything. Left in, its NaN would lose every threshold comparison
        # silently AND make the whole trail unrecordable (NaN is rejected by
        # DecisionTrail), so one bad row would blind the gate for a whole guild.
        #
        # One healthy fact, so the outcome does not depend on where a NaN
        # happens to sort: NaN compares false against everything, so its
        # position in a sorted list is not something a test should rely on.
        model = await _seed_scored_facts(conn, [0.9])
        await add_fact(
            conn,
            _NanEmbeddingModel(dimension=2),  # type: ignore[arg-type]
            guild_id=GUILD_A,
            channel_id=CHANNEL_1,
            message_id=9999,
            content="fact with a broken embedding",
        )

        with caplog.at_level(logging.WARNING):
            decision = await _evaluate(conn, model, _stub_detector(0.5))

        assert decision.stage2_top_score == pytest.approx(0.9, abs=1e-6)
        assert decision.verdict is GateVerdict.ELIGIBLE
        assert any(record.levelno >= logging.WARNING for record in caplog.records)


class _NanEmbeddingModel:
    """Produces an embedding that cannot be compared to anything.

    Matches the dimension of the facts it sits beside, so the comparison
    reaches the NaN rather than failing earlier on a shape mismatch.
    """

    def __init__(self, *, dimension: int) -> None:
        self._dimension = dimension

    def embed(self, documents: list[str], **_kwargs: object):
        for _ in documents:
            yield np.full(self._dimension, np.nan, dtype=EMBEDDING_DTYPE)


class TestBudgetIsClaimedAtTheRightMoment:
    async def test_an_eligible_message_claims_a_slot(
        self, conn: aiosqlite.Connection
    ) -> None:
        model = await _seed_scored_facts(conn, [0.9, 0.1])

        decision = await _evaluate(conn, model, _stub_detector(0.5))

        assert decision.verdict is GateVerdict.ELIGIBLE
        assert decision.daily_count == 1
        assert decision.daily_cap == 5
        assert await count_escalations_on(conn, guild_id=GUILD_A, day="2026-07-24") == 1

    async def test_the_slot_is_claimed_before_anything_downstream_could_run(
        self, conn: aiosqlite.Connection
    ) -> None:
        # The economic-attack lesson: the ledger row exists the instant the
        # verdict is ELIGIBLE, so a caller that then crashes, times out, or is
        # cancelled has already spent the slot. If the row were written after
        # a successful answer instead, every failed attempt would be free and
        # a crafted message could be retried without limit.
        model = await _seed_scored_facts(conn, [0.9, 0.1])

        await _evaluate(conn, model, _stub_detector(0.5))

        async with conn.execute(
            "SELECT COUNT(*) FROM proactive_escalations WHERE channel_id = ?", (CHANNEL_1,)
        ) as cursor:
            assert await cursor.fetchone() == (1,)

    async def test_a_stage_two_failure_claims_nothing(
        self, conn: aiosqlite.Connection
    ) -> None:
        # Being unable to answer must not burn a slot -- otherwise ordinary
        # chatter in a server with few facts would exhaust the daily budget
        # without a single answer being possible.
        model = await _seed_scored_facts(conn, [0.2, 0.1])

        for message_id in range(10):
            await _evaluate(conn, model, _stub_detector(0.5), message_id=message_id)

        assert await count_escalations_on(conn, guild_id=GUILD_A, day="2026-07-24") == 0

    async def test_a_second_eligible_message_in_the_channel_is_held_by_the_cooldown(
        self, conn: aiosqlite.Connection
    ) -> None:
        model = await _seed_scored_facts(conn, [0.9, 0.1])

        first = await _evaluate(conn, model, _stub_detector(0.5), message_id=1)
        second = await _evaluate(
            conn, model, _stub_detector(0.5), message_id=2, now=NOON + timedelta(seconds=30)
        )

        assert first.verdict is GateVerdict.ELIGIBLE
        assert second.verdict is GateVerdict.COOLDOWN_ACTIVE
        assert second.stage2_passed is True  # it earned an answer; the budget said no
        assert second.cooldown_seconds_remaining == pytest.approx(870.0)

    async def test_the_daily_cap_shows_up_as_its_own_verdict(
        self, conn: aiosqlite.Connection
    ) -> None:
        model = await _seed_scored_facts(conn, [0.9, 0.1])

        for index in range(CONFIG.daily_cap):
            decision = await _evaluate(
                conn, model, _stub_detector(0.5), channel_id=7000 + index, message_id=index
            )
            assert decision.verdict is GateVerdict.ELIGIBLE

        capped = await _evaluate(
            conn, model, _stub_detector(0.5), channel_id=8000, message_id=800
        )

        assert capped.verdict is GateVerdict.DAILY_CAP_REACHED
        assert capped.daily_count == CONFIG.daily_cap
        assert capped.stage2_passed is True

    async def test_a_redelivered_eligible_message_is_reported_as_a_duplicate(
        self, conn: aiosqlite.Connection
    ) -> None:
        model = await _seed_scored_facts(conn, [0.9, 0.1])

        first = await _evaluate(conn, model, _stub_detector(0.5), message_id=77)
        again = await _evaluate(conn, model, _stub_detector(0.5), message_id=77)

        assert first.verdict is GateVerdict.ELIGIBLE
        assert again.verdict is GateVerdict.DUPLICATE_DELIVERY
        assert await count_escalations_on(conn, guild_id=GUILD_A, day="2026-07-24") == 1

    async def test_every_escalation_outcome_maps_to_a_verdict(self) -> None:
        # A missing entry would raise KeyError inside the gate, which the
        # listener would swallow into a log line -- so a new outcome added
        # later must fail here instead of degrading quietly in production.
        assert set(_ESCALATION_VERDICTS) == set(EscalationOutcome)
        assert len(set(_ESCALATION_VERDICTS.values())) == len(EscalationOutcome)


class TestConcurrentEligibleMessages:
    async def test_a_burst_in_one_channel_yields_exactly_one_eligible_verdict(
        self, conn: aiosqlite.Connection
    ) -> None:
        # End to end through the gate, not just the ledger: fifty messages
        # that all deserve an answer, arriving together, must produce one
        # escalation and forty-nine held-back verdicts.
        model = await _seed_scored_facts(conn, [0.9, 0.1])

        decisions = await asyncio.gather(
            *(
                _evaluate(conn, model, _stub_detector(0.5), message_id=index)
                for index in range(50)
            )
        )

        verdicts = [decision.verdict for decision in decisions]
        assert verdicts.count(GateVerdict.ELIGIBLE) == 1
        assert verdicts.count(GateVerdict.COOLDOWN_ACTIVE) == 49
        assert await count_escalations_on(conn, guild_id=GUILD_A, day="2026-07-24") == 1

    async def test_a_burst_across_channels_of_one_guild_cannot_exceed_the_cap(
        self, conn: aiosqlite.Connection
    ) -> None:
        model = await _seed_scored_facts(conn, [0.9, 0.1])

        decisions = await asyncio.gather(
            *(
                _evaluate(
                    conn,
                    model,
                    _stub_detector(0.5),
                    channel_id=7000 + index,
                    message_id=index,
                )
                for index in range(40)
            )
        )

        eligible = [d for d in decisions if d.would_escalate]
        assert len(eligible) == CONFIG.daily_cap
        assert await count_escalations_on(conn, guild_id=GUILD_A, day="2026-07-24") == CONFIG.daily_cap


class TestConfigValidation:
    def test_settings_defaults_map_onto_the_gate(self) -> None:
        settings = Settings(_env_file=None, discord_token="token")  # type: ignore[call-arg]

        config = ProactiveGateConfig.from_settings(settings)

        assert config.question_threshold == settings.proactive_question_threshold
        assert config.similarity_threshold == settings.proactive_similarity_threshold
        assert config.cooldown_seconds == settings.proactive_cooldown_seconds
        assert config.daily_cap == settings.proactive_daily_cap

    @pytest.mark.parametrize(
        "overrides",
        [
            {"question_threshold": -2.0},  # the unscoreable sentinel would pass
            {"question_threshold": 2.5},  # outside the contrastive range
            {"question_threshold": float("nan")},
            {"similarity_threshold": 1.5},  # outside the cosine range
            {"similarity_threshold": float("inf")},
            {"cooldown_seconds": -1.0},
            {"cooldown_seconds": float("nan")},
            {"daily_cap": -1},
        ],
    )
    def test_a_nonsensical_value_is_refused_at_construction(
        self, overrides: dict[str, float]
    ) -> None:
        # Caught once, at startup, instead of silently disabling a gate for
        # every message afterwards.
        with pytest.raises(ValidationError):
            ProactiveGateConfig(**{**CONFIG.model_dump(), **overrides})

    def test_a_zero_cap_is_valid_because_it_is_a_deliberate_off_switch(self) -> None:
        assert ProactiveGateConfig(**{**CONFIG.model_dump(), "daily_cap": 0}).daily_cap == 0


class TestWithTheRealModel:
    """The semantic claims Stage 2 rests on, against real embeddings."""

    async def test_a_real_repeat_question_beats_its_runner_up_by_a_wide_margin(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        for message_id, content in enumerate(
            [
                "The server rules are in the welcome channel.",
                "The weekly community call happens on Thursdays at 19:00 UTC.",
                "Bug reports belong in the bug-reports channel.",
            ]
        ):
            await add_fact(
                conn,
                embedding_model,
                guild_id=GUILD_A,
                channel_id=CHANNEL_1,
                message_id=message_id,
                content=content,
            )

        detector = await QuestionDetector.create(embedding_model)
        # A low similarity bar, because the measured reality is that a
        # question and the fact answering it are not paraphrases: real repeats
        # scored 0.53-0.78 against their own fact (see config.py).
        lenient = CONFIG.model_copy(update={"similarity_threshold": 0.4})

        decision = await _evaluate(
            conn,
            embedding_model,
            detector,
            content="when is the weekly community call again?",
            config=lenient,
        )

        assert decision.stage1_passed is True
        # The gap is still measured and still recorded; since Phase 2b-4 it is
        # asserted as a recorded diagnostic rather than as a gate -- a genuine
        # repeat question does pull clearly ahead of unrelated facts, and that
        # remains worth knowing even though nothing is decided from it.
        assert decision.stage2_gap is not None
        assert decision.stage2_gap > 0.0
        assert decision.verdict is GateVerdict.ELIGIBLE

    async def test_real_chatter_with_facts_present_stays_silent(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        await add_fact(
            conn,
            embedding_model,
            guild_id=GUILD_A,
            channel_id=CHANNEL_1,
            message_id=1,
            content="The server rules are in the welcome channel.",
        )
        detector = await QuestionDetector.create(embedding_model)

        decisions = [
            await _evaluate(
                conn,
                embedding_model,
                detector,
                content=content,
                message_id=index,
            )
            for index, content in enumerate(
                [
                    "I just finished my coffee and it was really good.",
                    "congrats on the release, that looked like a lot of work",
                    "Ich habe gestern einen ziemlich guten Film gesehen.",
                    "昨日は新しいラーメン屋に行きました。",
                ]
            )
        ]

        assert all(decision.would_escalate is False for decision in decisions)
        assert await count_escalations_on(conn, guild_id=GUILD_A, day="2026-07-24") == 0

    @pytest.mark.parametrize(
        "hostile",
        [
            "🎉🎊🥳",
            "​‌‍",  # zero-width characters only
            "'; DROP TABLE proactive_escalations; --",
            "a\x00b",
            "x" * 5000,
            "‮reversed override text‬",
            "e" + "́" * 200,
        ],
    )
    async def test_hostile_input_produces_a_verdict_instead_of_an_exception(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding, hostile: str
    ) -> None:
        await add_fact(
            conn,
            embedding_model,
            guild_id=GUILD_A,
            channel_id=CHANNEL_1,
            message_id=1,
            content="The server rules are in the welcome channel.",
        )
        detector = await QuestionDetector.create(embedding_model)

        decision = await _evaluate(conn, embedding_model, detector, content=hostile)

        assert decision.verdict in set(GateVerdict)

    async def test_a_sql_injection_attempt_in_a_message_changes_nothing(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        detector = await QuestionDetector.create(embedding_model)

        await _evaluate(
            conn,
            embedding_model,
            detector,
            content="'; DROP TABLE proactive_escalations; DROP TABLE facts; --",
        )

        # Both tables still answer queries, which they could not if the text
        # had been interpolated rather than bound.
        async with conn.execute("SELECT COUNT(*) FROM proactive_escalations") as cursor:
            assert await cursor.fetchone() == (0,)
        async with conn.execute("SELECT COUNT(*) FROM facts") as cursor:
            assert await cursor.fetchone() == (0,)


class TestBlankAndDegenerateContent:
    async def test_empty_content_is_rejected_without_a_database_write(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        detector = await QuestionDetector.create(embedding_model)

        decision = await _evaluate(conn, embedding_model, detector, content="")

        assert decision.verdict is GateVerdict.STAGE1_REJECTED
        assert await count_escalations_on(conn, guild_id=GUILD_A, day="2026-07-24") == 0

    async def test_a_detector_returning_a_nan_score_fails_closed(
        self, conn: aiosqlite.Connection
    ) -> None:
        # The detector guards against this itself; this proves the gate does
        # not quietly escalate when it somehow does not. A NaN loses every
        # comparison, so it cannot pass Stage 1 -- and the resulting trail is
        # refused by DecisionTrail rather than written as a NULL score.
        model = await _seed_scored_facts(conn, [0.9, 0.1])

        with pytest.raises(ValidationError):
            await _evaluate(conn, model, _stub_detector(float("nan")))

        assert await count_escalations_on(conn, guild_id=GUILD_A, day="2026-07-24") == 0

    async def test_a_detector_failure_propagates_instead_of_escalating(
        self, conn: aiosqlite.Connection
    ) -> None:
        model = await _seed_scored_facts(conn, [0.9, 0.1])
        detector = MagicMock(spec=QuestionDetector)
        detector.question_likeness = AsyncMock(side_effect=RuntimeError("ONNX failed"))

        with pytest.raises(RuntimeError):
            await _evaluate(conn, model, detector)

        assert await count_escalations_on(conn, guild_id=GUILD_A, day="2026-07-24") == 0

    async def test_a_fact_embedded_at_another_dimension_fails_closed(
        self, conn: aiosqlite.Connection
    ) -> None:
        # Reachable by changing EMBEDDING_MODEL on a database that already has
        # facts: the stored vectors no longer line up with what the new model
        # produces, and the comparison raises instead of scoring. That is a
        # Phase 1d-wide problem (/aura-ask hits it too), so what matters here
        # is only the direction of the failure -- no escalation, nothing
        # spent -- and that the listener above turns it into a log line.
        model = await _seed_scored_facts(conn, [0.9])
        await add_fact(
            conn,
            _FixedSimilarityModel([0.5, 0.5, 0.5]),  # type: ignore[arg-type]
            guild_id=GUILD_A,
            channel_id=CHANNEL_1,
            message_id=4321,
            content="fact:0",
        )

        with pytest.raises(ValueError):
            await _evaluate(conn, model, _stub_detector(0.5))

        assert await count_escalations_on(conn, guild_id=GUILD_A, day="2026-07-24") == 0

    async def test_cancellation_mid_evaluation_claims_no_slot(
        self, conn: aiosqlite.Connection
    ) -> None:
        model = await _seed_scored_facts(conn, [0.9, 0.1])
        detector = MagicMock(spec=QuestionDetector)
        detector.question_likeness = AsyncMock(side_effect=asyncio.CancelledError())

        with pytest.raises(asyncio.CancelledError):
            await _evaluate(conn, model, detector)

        assert await count_escalations_on(conn, guild_id=GUILD_A, day="2026-07-24") == 0


class TestKnowledgeModelIsReadOnly:
    async def test_evaluation_never_creates_updates_or_deletes_a_fact(
        self, conn: aiosqlite.Connection
    ) -> None:
        model = await _seed_scored_facts(conn, [0.9, 0.1])

        async with conn.execute("SELECT id, content, status FROM facts ORDER BY id") as cursor:
            before = await cursor.fetchall()

        for message_id in range(10):
            await _evaluate(conn, model, _stub_detector(0.5), message_id=message_id)

        async with conn.execute("SELECT id, content, status FROM facts ORDER BY id") as cursor:
            assert await cursor.fetchall() == before

    async def test_the_only_sql_write_is_to_the_escalation_ledger(
        self, conn: aiosqlite.Connection
    ) -> None:
        model = await _seed_scored_facts(conn, [0.9, 0.1])
        statements: list[str] = []

        await conn.set_trace_callback(statements.append)
        try:
            await _evaluate(conn, model, _stub_detector(0.5))
        finally:
            await conn.set_trace_callback(None)  # pyright: ignore[reportArgumentType]

        assert statements, "no SQL was captured; the trace callback is not working"
        # Word boundaries, not a substring search: the fact columns include
        # "created_at" and "superseded_at", so matching "CREATE" loosely would
        # classify every ordinary SELECT as a write.
        mutating = re.compile(r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|REPLACE)\b", re.IGNORECASE)
        writes = [statement for statement in statements if mutating.search(statement)]
        assert writes, "the eligible path must have written its escalation"
        for statement in writes:
            assert "proactive_escalations" in statement, statement
