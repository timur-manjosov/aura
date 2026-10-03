"""aura.retrieval.index_cache: the index always matches the facts asked about, within its bounds."""

from __future__ import annotations

import gc
import logging
import random
import string
import threading
import tracemalloc

import pytest

from aura.retrieval.index_cache import LexicalIndexCache, fact_set_digest
from aura.retrieval.lexical import LexicalIndex
from aura.retrieval.stopwords import shipped_stopwords

GUILD_A = 100000000000000001
GUILD_B = 200000000000000002

FACTS = [
    (1, "Die Serverwartung ist jeden Donnerstag um 5:00 MEZ."),
    (2, "Das Mentoriat findet jeden Dienstag statt."),
]


def covers(index: LexicalIndex, question: str) -> dict[int, float]:
    return index.coverage(question, shipped_stopwords())


class TestDigest:
    def test_the_order_of_the_facts_does_not_matter(self) -> None:
        assert fact_set_digest(FACTS) == fact_set_digest(list(reversed(FACTS)))

    @pytest.mark.parametrize(
        "changed",
        [
            [*FACTS, (3, "Neu.")],  # added
            FACTS[:1],  # superseded or deleted
            [(1, FACTS[0][1]), (2, "Das Mentoriat findet jeden Mittwoch statt.")],  # edited
            [(1, FACTS[0][1]), (3, FACTS[1][1])],  # same text under another ID
            [(1, FACTS[0][1] + " "), (2, FACTS[1][1])],  # one character more
        ],
        ids=["added", "removed", "edited", "re-identified", "one-character"],
    )
    def test_any_change_changes_the_digest(self, changed: list[tuple[int, str]]) -> None:
        assert fact_set_digest(changed) != fact_set_digest(FACTS)

    def test_content_boundaries_cannot_be_shifted(self) -> None:
        assert fact_set_digest([(1, "ab"), (2, "c")]) != fact_set_digest([(1, "a"), (2, "bc")])

    def test_unusual_text_hashes(self) -> None:
        assert fact_set_digest([(1, "\ud800 lone surrogate"), (2, "\x00"), (3, "")])


class TestFreshness:
    def test_the_same_facts_reuse_the_same_index(self) -> None:
        cache = LexicalIndexCache()
        first = cache.index_for(GUILD_A, FACTS)
        assert cache.index_for(GUILD_A, list(reversed(FACTS))) is first
        assert cache.statistics().hits == 1 and cache.statistics().builds == 1

    def test_a_new_fact_is_found_by_the_next_question(self) -> None:
        cache = LexicalIndexCache()
        assert covers(cache.index_for(GUILD_A, FACTS), "Sommerfest") == {1: 0.0, 2: 0.0}
        added = [*FACTS, (3, "Das Sommerfest ist am 1. August.")]
        assert covers(cache.index_for(GUILD_A, added), "Sommerfest")[3] == 1.0

    def test_an_edited_fact_is_scored_by_its_new_text(self) -> None:
        cache = LexicalIndexCache()
        assert covers(cache.index_for(GUILD_A, FACTS), "Mentoriat")[2] == 1.0
        edited = [FACTS[0], (2, "Das Tutorium findet jeden Dienstag statt.")]
        coverage = covers(cache.index_for(GUILD_A, edited), "Mentoriat")
        assert coverage[2] == 0.0

    def test_a_superseded_or_deleted_fact_is_no_longer_scored(self) -> None:
        cache = LexicalIndexCache()
        cache.index_for(GUILD_A, FACTS)
        assert covers(cache.index_for(GUILD_A, FACTS[:1]), "Mentoriat") == {1: 0.0}

    def test_guilds_never_share_an_index(self) -> None:
        cache = LexicalIndexCache()
        index_a = cache.index_for(GUILD_A, FACTS)
        index_b = cache.index_for(GUILD_B, [(1, "Das Turnier ist am Samstag.")])
        assert covers(cache.index_for(GUILD_A, FACTS), "Turnier") == {1: 0.0, 2: 0.0}
        assert covers(index_b, "Turnier") == {1: 1.0}
        assert cache.index_for(GUILD_A, FACTS) is index_a

    def test_identical_facts_in_two_guilds_are_two_entries(self) -> None:
        cache = LexicalIndexCache()
        cache.index_for(GUILD_A, FACTS)
        cache.index_for(GUILD_B, FACTS)
        assert cache.statistics().guilds == 2

    def test_a_change_and_back_rebuilds_both_times(self) -> None:
        cache = LexicalIndexCache()
        cache.index_for(GUILD_A, FACTS)
        cache.index_for(GUILD_A, FACTS[:1])
        cache.index_for(GUILD_A, FACTS)
        assert cache.statistics().builds == 3


class TestBounds:
    def test_bounds_below_one_are_refused(self) -> None:
        with pytest.raises(ValueError):
            LexicalIndexCache(max_guilds=0)
        with pytest.raises(ValueError):
            LexicalIndexCache(max_bytes=0)

    def test_the_least_recently_used_guild_is_evicted_first(self) -> None:
        cache = LexicalIndexCache(max_guilds=2)
        cache.index_for(1, FACTS)
        cache.index_for(2, FACTS)
        cache.index_for(1, FACTS)  # guild 1 is now the most recent
        cache.index_for(3, FACTS)
        statistics = cache.statistics()
        assert statistics.guilds == 2 and statistics.evictions == 1
        builds = statistics.builds
        cache.index_for(1, FACTS)
        assert cache.statistics().builds == builds  # kept
        cache.index_for(2, FACTS)
        assert cache.statistics().builds == builds + 1  # evicted, rebuilt

    def test_the_byte_budget_evicts_until_it_fits(self) -> None:
        single = LexicalIndex.build(FACTS).estimated_bytes
        cache = LexicalIndexCache(max_bytes=single * 2 + single // 2)
        for guild_id in range(1, 6):
            cache.index_for(guild_id, FACTS)
        statistics = cache.statistics()
        assert statistics.guilds == 2
        assert statistics.cached_bytes <= single * 2 + single // 2
        assert statistics.evictions == 3

    def test_a_guild_larger_than_the_budget_is_scored_but_not_kept(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        cache = LexicalIndexCache(max_bytes=10)
        with caplog.at_level(logging.WARNING, logger="aura.retrieval.index_cache"):
            first = cache.index_for(GUILD_A, FACTS)
            second = cache.index_for(GUILD_A, FACTS)
        assert covers(first, "Mentoriat")[2] == 1.0
        assert first is not second
        assert cache.statistics().guilds == 0 and cache.statistics().cached_bytes == 0
        warnings = [record for record in caplog.records if record.levelno == logging.WARNING]
        assert len(warnings) == 1
        assert str(GUILD_A) not in warnings[0].getMessage()

    def test_the_byte_count_returns_to_zero_when_cleared(self) -> None:
        cache = LexicalIndexCache()
        cache.index_for(GUILD_A, FACTS)
        assert cache.statistics().cached_bytes > 0
        cache.clear()
        assert cache.statistics().cached_bytes == 0 and cache.statistics().guilds == 0

    @pytest.mark.parametrize("fact_count", [50, 500, 2000])
    def test_the_memory_estimate_is_close_to_and_above_the_real_size(self, fact_count: int) -> None:
        rng = random.Random(fact_count)
        vocabulary = [
            "".join(rng.choice(string.ascii_lowercase + "äöü") for _ in range(rng.randint(3, 12)))
            for _ in range(5000)
        ]
        facts = [
            (fact_id, " ".join(rng.choice(vocabulary) for _ in range(10)) + ".")
            for fact_id in range(1, fact_count + 1)
        ]
        LexicalIndex.build([(1, "warm-up äöü")])
        gc.collect()
        tracemalloc.start()
        baseline = tracemalloc.get_traced_memory()[0]
        index = LexicalIndex.build(facts)
        gc.collect()
        measured = tracemalloc.get_traced_memory()[0] - baseline
        tracemalloc.stop()
        assert measured * 0.95 <= index.estimated_bytes <= measured * 1.5


class TestConcurrency:
    def test_concurrent_requests_across_a_change_each_get_their_own_facts(self) -> None:
        cache = LexicalIndexCache()
        versions = [FACTS, [*FACTS, (3, "Das Sommerfest ist am 1. August.")], FACTS[:1]]
        failures: list[str] = []
        barrier = threading.Barrier(12)

        def ask(worker: int) -> None:
            barrier.wait()
            for round_number in range(60):
                facts = versions[(worker + round_number) % len(versions)]
                index = cache.index_for(GUILD_A, facts)
                if index.fact_ids != tuple(sorted(fact_id for fact_id, _ in facts)):
                    failures.append(f"worker {worker} got {index.fact_ids}")
                summer = covers(index, "Sommerfest").get(3, 0.0)
                if (summer == 1.0) != any(fact_id == 3 for fact_id, _ in facts):
                    failures.append(f"worker {worker} scored a stale index")

        threads = [threading.Thread(target=ask, args=(worker,)) for worker in range(12)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert failures == []
        statistics = cache.statistics()
        assert statistics.guilds == 1
        held = sum(entry.index.estimated_bytes for entry in cache._entries.values())
        assert statistics.cached_bytes == held

    def test_the_byte_count_never_drifts_under_concurrent_eviction(self) -> None:
        cache = LexicalIndexCache(max_guilds=3)
        barrier = threading.Barrier(8)

        def churn(worker: int) -> None:
            barrier.wait()
            for round_number in range(100):
                guild = (worker * 7 + round_number) % 10
                cache.index_for(guild, FACTS if round_number % 2 else FACTS[:1])

        threads = [threading.Thread(target=churn, args=(worker,)) for worker in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        statistics = cache.statistics()
        assert statistics.guilds <= 3
        # Recount from what is actually held: the running total must agree.
        held = sum(entry.index.estimated_bytes for entry in cache._entries.values())
        assert statistics.cached_bytes == held
