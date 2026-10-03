"""aura.retrieval.hybrid: the gate, the ranking, the related-facts rule, and the fail-safe."""

from __future__ import annotations

import asyncio
import logging
import random
import threading
from datetime import UTC, datetime
from typing import ClassVar
from unittest.mock import patch

import pytest

import aura.retrieval.hybrid as hybrid
from aura.config import Settings
from aura.db.models import Fact, FactStatus
from aura.embeddings import SYNTHESIS_FACT_LIMIT
from aura.retrieval.hybrid import (
    RELATED_COVERAGE_THRESHOLD,
    RELATED_FACT_LIMIT,
    HybridRetrievalConfig,
    ScoredFact,
    assemble_retrieval,
    is_possibly_related,
    qualifies,
    rank_score,
    retrieve_for_question,
    select_facts,
)
from aura.retrieval.index_cache import LexicalIndexCache
from aura.retrieval.lexical import LexicalIndex
from aura.retrieval.stopwords import StopwordLoadError

GUILD = 100000000000000001
CONFIG = HybridRetrievalConfig(
    similarity_threshold=0.40,
    coverage_threshold=0.5,
    similarity_floor=0.05,
    ranking_weight=0.5,
    fact_limit=SYNTHESIS_FACT_LIMIT,
)


def make_fact(fact_id: int, content: str = "irrelevant") -> Fact:
    return Fact(
        id=fact_id,
        guild_id=GUILD,
        channel_id=1,
        message_id=fact_id,
        content=content,
        embedding=b"\x00" * 4,
        status=FactStatus.ACTIVE,
        created_at=datetime(2026, 10, 1, tzinfo=UTC),
    )


def scored(fact_id: int, similarity: float, coverage: float) -> ScoredFact:
    return ScoredFact(fact=make_fact(fact_id), similarity=similarity, coverage=coverage)


def old_selection(results: list[tuple[Fact, float]], threshold: float) -> list[int]:
    """What /aura-ask selected before hybrid retrieval: top five by similarity, then the bar."""
    ranked = sorted(results, key=lambda pair: (-pair[1], pair[0].id))
    return [fact.id for fact, score in ranked[:SYNTHESIS_FACT_LIMIT] if score >= threshold]


class TestTheGate:
    @pytest.mark.parametrize(
        ("similarity", "coverage", "expected"),
        [
            (0.40, 0.0, True),  # the similarity bar, unchanged
            (0.3999, 0.0, False),
            (0.05, 0.5, True),  # both lexical bounds exactly
            (0.0499, 1.0, False),  # under the floor: words alone are not enough
            (0.39, 0.4999, False),  # under the coverage bar
            (-0.03, 0.9, False),  # the measured "werde"/"werden" case
            (0.2, 0.9, True),  # "Mentoriate": noise-level similarity, plain words
        ],
    )
    def test_boundaries(self, similarity: float, coverage: float, expected: bool) -> None:
        assert qualifies(scored(1, similarity, coverage), CONFIG) is expected

    @pytest.mark.parametrize("similarity", [float("nan"), float("inf"), float("-inf")])
    def test_an_unusable_vector_never_qualifies(self, similarity: float) -> None:
        assert not qualifies(scored(1, similarity, 1.0), CONFIG)

    def test_the_numbers_come_from_settings(self) -> None:
        overrides: dict[str, object] = {
            "discord_token": "x",
            "similarity_threshold": 0.33,
            "ask_lexical_coverage_threshold": 0.6,
            "ask_lexical_similarity_floor": 0.1,
            "ask_lexical_ranking_weight": 0.25,
        }
        settings = Settings(_env_file=None, **overrides)  # type: ignore[arg-type]
        assert HybridRetrievalConfig.from_settings(settings, fact_limit=4) == HybridRetrievalConfig(
            similarity_threshold=0.33,
            coverage_threshold=0.6,
            similarity_floor=0.1,
            ranking_weight=0.25,
            fact_limit=4,
        )


class TestTheRanking:
    def test_coverage_lifts_a_fact_that_contains_the_asked_word(self) -> None:
        # The diagnosis' "Wann findet die Wartung statt?": the embedding put
        # a different topic's fact first, on sentence shape alone.
        shape_only = scored(10, similarity=0.472, coverage=0.0)
        right_topic = scored(7, similarity=0.445, coverage=0.9)
        assert [s.fact.id for s in select_facts([shape_only, right_topic], CONFIG)] == [7, 10]
        assert rank_score(right_topic, CONFIG) == pytest.approx(0.445 + 0.45)

    def test_ties_break_by_ascending_fact_id(self) -> None:
        facts = [scored(fact_id, 0.5, 0.0) for fact_id in (9, 3, 7)]
        assert [s.fact.id for s in select_facts(facts, CONFIG)] == [3, 7, 9]

    def test_at_most_the_limit_is_kept(self) -> None:
        facts = [scored(fact_id, 0.9 - fact_id / 100, 0.0) for fact_id in range(1, 12)]
        assert [s.fact.id for s in select_facts(facts, CONFIG)] == [1, 2, 3, 4, 5]

    def test_a_limit_of_zero_keeps_nothing(self) -> None:
        config = HybridRetrievalConfig(0.4, 0.5, 0.05, 0.5, fact_limit=0)
        assert select_facts([scored(1, 0.9, 1.0)], config) == []

    def test_with_no_word_matches_it_is_exactly_the_old_selection(self) -> None:
        rng = random.Random(4)
        for _ in range(500):
            results = [
                (make_fact(fact_id), round(rng.uniform(-0.2, 0.9), rng.choice([1, 2, 4])))
                for fact_id in rng.sample(range(1, 60), rng.randint(0, 15))
            ]
            retrieval = assemble_retrieval(results, {}, CONFIG, lexical_available=False)
            assert [fact.id for fact in retrieval.facts] == old_selection(results, 0.40)


class TestTheOperatorSwitch:
    def test_floor_at_the_bar_and_no_weight_restore_the_previous_selection_exactly(
        self,
    ) -> None:
        # DEPLOYMENT.md's documented way back without a rollback:
        # ASK_LEXICAL_SIMILARITY_FLOOR=<SIMILARITY_THRESHOLD>, ASK_LEXICAL_RANKING_WEIGHT=0.
        switched_off = HybridRetrievalConfig(0.40, 0.5, 0.40, 0.0, SYNTHESIS_FACT_LIMIT)
        rng = random.Random(9)
        for _ in range(500):
            results = [
                (make_fact(fact_id), round(rng.uniform(-0.2, 0.9), 3))
                for fact_id in rng.sample(range(1, 60), rng.randint(0, 15))
            ]
            coverage = {fact.id: rng.choice([0.0, 0.3, 0.6, 1.0]) for fact, _ in results}
            retrieval = assemble_retrieval(results, coverage, switched_off, lexical_available=True)
            assert [fact.id for fact in retrieval.facts] == old_selection(results, 0.40)
            assert retrieval.related == []


class TestRelatedFacts:
    def test_only_when_nothing_qualified(self) -> None:
        results = [(make_fact(1), 0.5), (make_fact(2), 0.1)]
        retrieval = assemble_retrieval(results, {2: 0.3}, CONFIG, lexical_available=True)
        assert retrieval.facts == [results[0][0]]
        assert retrieval.related == []

    @pytest.mark.parametrize(
        ("similarity", "coverage", "expected"),
        [
            (0.05, RELATED_COVERAGE_THRESHOLD, True),
            (0.05, RELATED_COVERAGE_THRESHOLD - 0.001, False),
            (0.0499, 0.4, False),
            (0.39, 0.0, False),  # similarity alone never makes a fact "related"
            (float("nan"), 0.4, False),
        ],
    )
    def test_boundaries(self, similarity: float, coverage: float, expected: bool) -> None:
        assert is_possibly_related(scored(1, similarity, coverage), CONFIG) is expected

    def test_at_most_three_best_first(self) -> None:
        results = [(make_fact(fact_id), 0.1 + fact_id / 100) for fact_id in range(1, 7)]
        coverage = dict.fromkeys(range(1, 7), 0.3)
        retrieval = assemble_retrieval(results, coverage, CONFIG, lexical_available=True)
        assert len(retrieval.related) == RELATED_FACT_LIMIT
        assert [fact.id for fact in retrieval.related_facts] == [6, 5, 4]
        assert retrieval.facts == []

    def test_nothing_is_related_when_word_matching_failed(self) -> None:
        retrieval = assemble_retrieval([(make_fact(1), 0.35)], {}, CONFIG, lexical_available=False)
        assert retrieval.related == [] and retrieval.facts == []


class TestCounting:
    def test_found_by_words_only_counts_selected_facts_below_the_bar(self) -> None:
        results = [(make_fact(1), 0.6), (make_fact(2), 0.2), (make_fact(3), 0.1)]
        retrieval = assemble_retrieval(
            results, {1: 0.0, 2: 0.9, 3: 0.4}, CONFIG, lexical_available=True
        )
        # 0.2 + 0.5 x 0.9 outranks 0.6 + 0: the fact containing the words leads.
        assert [fact.id for fact in retrieval.facts] == [2, 1]
        assert retrieval.found_by_words_only == 1
        assert retrieval.active_fact_count == 3


class TestRetrieveForQuestion:
    FACTS: ClassVar[list[Fact]] = [
        make_fact(1, "Das Mentoriat findet jeden Dienstag um 16 Uhr statt."),
        make_fact(2, "Ein zweites Mentoriat gibt es donnerstags um 10 Uhr."),
        make_fact(3, "Die Serverwartung ist jeden Donnerstag."),
    ]

    def results(self, *similarities: float) -> list[tuple[Fact, float]]:
        return list(zip(self.FACTS, similarities, strict=True))

    async def test_a_keyword_finds_every_fact_that_contains_it(self) -> None:
        retrieval = await retrieve_for_question(
            self.results(0.2, 0.19, 0.22),
            question="Mentoriate",
            guild_id=GUILD,
            config=CONFIG,
            cache=LexicalIndexCache(),
        )
        assert [fact.id for fact in retrieval.facts] == [1, 2]
        assert retrieval.found_by_words_only == 2
        assert retrieval.lexical_available

    async def test_no_facts_means_nothing_and_no_thread(self) -> None:
        with patch("asyncio.to_thread", side_effect=AssertionError):
            retrieval = await retrieve_for_question(
                [], question="x", guild_id=GUILD, config=CONFIG, cache=LexicalIndexCache()
            )
        assert retrieval.facts == [] and retrieval.related == []
        assert retrieval.active_fact_count == 0

    async def test_the_word_scoring_runs_off_the_event_loop_thread(self) -> None:
        loop_thread = threading.get_ident()
        seen: list[int] = []
        original = LexicalIndex.coverage

        def recording(self: LexicalIndex, question: str, stopwords: frozenset[str]):
            seen.append(threading.get_ident())
            return original(self, question, stopwords)

        with patch.object(LexicalIndex, "coverage", recording):
            await retrieve_for_question(
                self.results(0.2, 0.2, 0.2),
                question="Mentoriate",
                guild_id=GUILD,
                config=CONFIG,
                cache=LexicalIndexCache(),
            )
        assert seen and all(thread != loop_thread for thread in seen)

    async def test_the_event_loop_keeps_running_while_a_large_guild_is_scored(self) -> None:
        facts = [
            make_fact(fact_id, f"Fakt Nummer {fact_id} über das Thema{fact_id % 97} im Kanal")
            for fact_id in range(1, 3001)
        ]
        ticks = 0
        stop = asyncio.Event()

        async def ticker() -> None:
            nonlocal ticks
            while not stop.is_set():
                ticks += 1
                await asyncio.sleep(0)

        task = asyncio.create_task(ticker())
        await retrieve_for_question(
            [(fact, 0.1) for fact in facts],
            question="Thema12 Kanal",
            guild_id=GUILD,
            config=CONFIG,
            cache=LexicalIndexCache(),
        )
        stop.set()
        await task
        assert ticks > 1

    @pytest.mark.parametrize(
        "failure",
        [RuntimeError("scorer bug"), MemoryError(), StopwordLoadError("files missing")],
        ids=["bug", "memory", "stopwords"],
    )
    async def test_a_failing_word_matcher_falls_back_to_similarity_and_warns_once(
        self, failure: BaseException, caplog: pytest.LogCaptureFixture
    ) -> None:
        with (
            patch.object(hybrid, "shipped_stopwords", side_effect=failure),
            caplog.at_level(logging.DEBUG, logger="aura.retrieval.hybrid"),
        ):
            retrieval = await retrieve_for_question(
                self.results(0.45, 0.2, 0.41),
                question="Mentoriate",
                guild_id=GUILD,
                config=CONFIG,
                cache=LexicalIndexCache(),
            )
        assert [fact.id for fact in retrieval.facts] == [1, 3]
        assert not retrieval.lexical_available
        assert retrieval.related == []
        records = [record for record in caplog.records if record.name == "aura.retrieval.hybrid"]
        assert len(records) == 1 and records[0].levelno == logging.WARNING
        assert "Mentoriat" not in records[0].getMessage()

    async def test_a_failing_index_build_falls_back_too(self) -> None:
        with patch.object(LexicalIndex, "build", side_effect=ValueError("broken")):
            retrieval = await retrieve_for_question(
                self.results(0.45, 0.2, 0.1),
                question="Mentoriate",
                guild_id=GUILD,
                config=CONFIG,
                cache=LexicalIndexCache(),
            )
        assert [fact.id for fact in retrieval.facts] == [1]
        assert not retrieval.lexical_available

    async def test_a_thread_that_cannot_start_falls_back_on_the_loop(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with (
            patch.object(
                hybrid.asyncio,
                "to_thread",
                side_effect=RuntimeError("cannot schedule new futures after shutdown"),
            ),
            caplog.at_level(logging.WARNING, logger="aura.retrieval.hybrid"),
        ):
            retrieval = await retrieve_for_question(
                self.results(0.45, 0.2, 0.41),
                question="Mentoriate",
                guild_id=GUILD,
                config=CONFIG,
                cache=LexicalIndexCache(),
            )
        assert [fact.id for fact in retrieval.facts] == [1, 3]
        assert len(caplog.records) == 1

    async def test_concurrent_questions_during_a_fact_change_each_see_their_own_facts(
        self,
    ) -> None:
        cache = LexicalIndexCache()
        before = [(fact, 0.1) for fact in self.FACTS[:2]]
        added = make_fact(4, "Das Sommerfest ist am 1. August.")
        after = [*before, (added, 0.1)]

        async def ask(results: list[tuple[Fact, float]], question: str) -> list[int]:
            retrieval = await retrieve_for_question(
                results, question=question, guild_id=GUILD, config=CONFIG, cache=cache
            )
            return [fact.id for fact in retrieval.facts]

        outcomes = await asyncio.gather(
            *(ask(before if index % 2 else after, "Sommerfest") for index in range(40))
        )
        for index, outcome in enumerate(outcomes):
            assert outcome == ([] if index % 2 else [4])
        assert await ask(after, "Sommerfest") == [4]
