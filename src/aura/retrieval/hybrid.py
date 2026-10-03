"""Which facts /aura-ask hands to synthesis: embedding similarity OR the question's own words.

/aura-ask used to keep a fact only if its embedding similarity to the question
reached SIMILARITY_THRESHOLD (0.40). For a single keyword, an inflected form, a
compound or a casual phrasing the embedding model scores near its noise floor
(about 0.20 against any fact), so a fact that plainly contains the word never
reached the model and the asker was told nothing was recorded.

**The gate.** A fact qualifies when

* its similarity is at least SIMILARITY_THRESHOLD, unchanged; or
* its lexical coverage (aura.retrieval.lexical) is at least
  ASK_LEXICAL_COVERAGE_THRESHOLD **and** its similarity is at least
  ASK_LEXICAL_SIMILARITY_FLOOR.

**The ranking.** Qualifying facts are ordered by
similarity + ASK_LEXICAL_RANKING_WEIGHT x coverage, ties by ascending fact ID,
and the first SYNTHESIS_FACT_LIMIT are kept -- the same bound as before.

The three numbers come from the quality diagnosis of 2026-10-02 (Section 4),
measured on 77 hand-written questions over a real guild's facts: 74 instead of
47 found their facts, with the same 6 of 24 unrelated questions selecting
anything and precision 0.90 instead of 0.83. Their reasons are at each setting
in aura.config.

**When nothing qualifies,** up to `RELATED_FACT_LIMIT` facts that contain part
of the question's subject are returned as "possibly related" -- shown verbatim,
never handed to a model (see aura.commands.ask).

**For /aura-ask only.** Proactive relief and extraction dedup keep calling
aura.embeddings.find_similar_facts and best_similarity exactly as before, with
their own calibrated thresholds; a structural test keeps them from importing
this package.

**Fail-safe.** If word matching cannot run -- the stopword files are missing,
or anything in the scorer raises -- the question is answered from similarity
alone, which is exactly the selection before this module existed, and one
WARNING is logged. It never turns into an error reply.

Imports aura.config (settings only), aura.db.models (the Fact type) and the
rest of this package; no Discord, no database connection, no LLM.
"""

from __future__ import annotations

import asyncio
import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

from aura.config import Settings
from aura.db.models import Fact
from aura.retrieval.index_cache import LexicalIndexCache
from aura.retrieval.stopwords import shipped_stopwords

logger = logging.getLogger(__name__)

# The one process-wide cache /aura-ask uses. Process-wide on purpose, unlike
# the database connection and the embedding model, which live on the client:
# it holds only data derived from facts the caller passes in, every entry is
# keyed by the exact facts it was built from, and losing it costs one rebuild.
SHARED_INDEX_CACHE: Final = LexicalIndexCache()

# A fact is "possibly related" to a question nothing qualified for when its
# coverage reaches this -- a quarter of the question's weighted subject -- and
# its similarity reaches the gate's floor. Measured on every invented question
# that should select nothing (greetings, off-topic questions, sound-alikes,
# questions about topics that do not exist): 5 of 235 got a related list, at
# most 1 of 10 in any one set; on the real guild's 24 such questions, 1. The
# quality diagnosis also proposed "or similarity >= 0.30"; that half showed a
# list for 17 to 37 percent of the same questions, so it is not used.
RELATED_COVERAGE_THRESHOLD: Final = 0.25

# How many possibly related facts are listed: a pointer, not an answer.
RELATED_FACT_LIMIT: Final = 3


@dataclass(frozen=True, slots=True)
class HybridRetrievalConfig:
    """The numbers the gate and the ranking use.

    Attributes
    ----------
    similarity_threshold
        A fact at or above this similarity qualifies on similarity alone.
    coverage_threshold
        A fact at or above this lexical coverage qualifies by its words...
    similarity_floor
        ...provided its similarity is at least this.
    ranking_weight
        How much coverage adds to similarity when qualifying facts are ranked.
    fact_limit
        How many qualifying facts are kept.
    """

    similarity_threshold: float
    coverage_threshold: float
    similarity_floor: float
    ranking_weight: float
    fact_limit: int

    @classmethod
    def from_settings(cls, settings: Settings, *, fact_limit: int) -> HybridRetrievalConfig:
        """Read the gate's numbers from the loaded settings.

        Parameters
        ----------
        settings
            Loaded configuration.
        fact_limit
            How many facts may reach synthesis (SYNTHESIS_FACT_LIMIT).

        Returns
        -------
        HybridRetrievalConfig
            The configuration.
        """
        return cls(
            similarity_threshold=settings.similarity_threshold,
            coverage_threshold=settings.ask_lexical_coverage_threshold,
            similarity_floor=settings.ask_lexical_similarity_floor,
            ranking_weight=settings.ask_lexical_ranking_weight,
            fact_limit=fact_limit,
        )


@dataclass(frozen=True, slots=True)
class ScoredFact:
    """One active fact with both of its scores for one question.

    Attributes
    ----------
    fact
        The fact.
    similarity
        Its embedding similarity (aura.embeddings.best_similarity); NaN for an
        unusable stored vector.
    coverage
        Its lexical coverage, in [0, 1]; 0 when word matching is unavailable.
    """

    fact: Fact
    similarity: float
    coverage: float


def qualifies(scored: ScoredFact, config: HybridRetrievalConfig) -> bool:
    """Report whether a fact may reach synthesis for this question.

    Parameters
    ----------
    scored
        The fact and its two scores.
    config
        The gate's numbers.

    Returns
    -------
    bool
        True when its similarity reaches the threshold, or its coverage
        reaches the coverage threshold while its similarity reaches the
        floor. Always False for a non-finite similarity: a fact whose vector
        is unusable can lose a score, never gain one.

    Notes
    -----
    The floor exists because coverage alone admitted measured false hits:
    "Wie werde ich Mentor?" matched an unrelated fact through "werde"/"werden"
    at an embedding similarity of -0.03. A floor of 0.05 removed it and lost
    nothing measured; higher floors started losing correct matches.
    """
    if not math.isfinite(scored.similarity):
        return False
    if scored.similarity >= config.similarity_threshold:
        return True
    return (
        scored.coverage >= config.coverage_threshold
        and scored.similarity >= config.similarity_floor
    )


def is_possibly_related(scored: ScoredFact, config: HybridRetrievalConfig) -> bool:
    """Report whether a fact may be listed as possibly related when nothing qualified.

    Parameters
    ----------
    scored
        The fact and its two scores.
    config
        The gate's floor.

    Returns
    -------
    bool
        True when its coverage reaches `RELATED_COVERAGE_THRESHOLD` and its
        similarity, finite, reaches the gate's floor.
    """
    return (
        math.isfinite(scored.similarity)
        and scored.coverage >= RELATED_COVERAGE_THRESHOLD
        and scored.similarity >= config.similarity_floor
    )


def rank_score(scored: ScoredFact, config: HybridRetrievalConfig) -> float:
    """Return the number qualifying facts are ordered by, highest first.

    Parameters
    ----------
    scored
        A fact and its two scores.
    config
        The ranking weight.

    Returns
    -------
    float
        similarity + ranking_weight x coverage.
    """
    return scored.similarity + config.ranking_weight * scored.coverage


def _ranked(scored_facts: Sequence[ScoredFact], config: HybridRetrievalConfig) -> list[ScoredFact]:
    """Order facts by descending `rank_score`, ties by ascending fact ID."""
    return sorted(scored_facts, key=lambda scored: (-rank_score(scored, config), scored.fact.id))


def select_facts(
    scored_facts: Sequence[ScoredFact], config: HybridRetrievalConfig
) -> list[ScoredFact]:
    """Apply the gate and the ranking.

    Parameters
    ----------
    scored_facts
        Every active fact of the guild with its scores.
    config
        The gate's and the ranking's numbers.

    Returns
    -------
    list[ScoredFact]
        At most `config.fact_limit` qualifying facts, by descending
        `rank_score`, ties broken by ascending fact ID so the same question
        over the same facts always selects the same facts in the same order.
    """
    qualifying = [scored for scored in scored_facts if qualifies(scored, config)]
    return _ranked(qualifying, config)[: max(config.fact_limit, 0)]


@dataclass(frozen=True, slots=True)
class AskRetrieval:
    """What retrieval found for one /aura-ask question.

    Attributes
    ----------
    selected
        The facts for synthesis, best first, with their scores.
    related
        When `selected` is empty: up to `RELATED_FACT_LIMIT` possibly related
        facts, best first. Always empty otherwise.
    active_fact_count
        How many active facts the guild had.
    found_by_words_only
        How many selected facts qualified through their words alone, below
        the similarity threshold.
    lexical_available
        False when word matching failed and only similarity was used.
    """

    selected: list[ScoredFact]
    related: list[ScoredFact]
    active_fact_count: int
    found_by_words_only: int
    lexical_available: bool

    @property
    def facts(self) -> list[Fact]:
        """The selected facts alone, best first."""
        return [scored.fact for scored in self.selected]

    @property
    def related_facts(self) -> list[Fact]:
        """The possibly related facts alone, best first."""
        return [scored.fact for scored in self.related]


def assemble_retrieval(
    similarity_results: Sequence[tuple[Fact, float]],
    coverage: Mapping[int, float],
    config: HybridRetrievalConfig,
    *,
    lexical_available: bool,
) -> AskRetrieval:
    """Combine both scores and apply the gate, the ranking and the related-facts rule.

    Parameters
    ----------
    similarity_results
        Every active fact with its embedding similarity.
    coverage
        Lexical coverage per fact ID; a missing fact counts as 0.
    config
        The gate's and the ranking's numbers.
    lexical_available
        Whether `coverage` came from a working word matcher.

    Returns
    -------
    AskRetrieval
        The selection, and the related facts when nothing was selected.
    """
    scored = [
        ScoredFact(fact=fact, similarity=similarity, coverage=coverage.get(fact.id, 0.0))
        for fact, similarity in similarity_results
    ]
    selected = select_facts(scored, config)
    related = (
        []
        if selected
        else _ranked([s for s in scored if is_possibly_related(s, config)], config)[
            :RELATED_FACT_LIMIT
        ]
    )
    return AskRetrieval(
        selected=selected,
        related=related,
        active_fact_count=len(scored),
        found_by_words_only=sum(
            1 for s in selected if not s.similarity >= config.similarity_threshold
        ),
        lexical_available=lexical_available,
    )


def _retrieve_blocking(
    similarity_results: Sequence[tuple[Fact, float]],
    question: str,
    guild_id: int,
    config: HybridRetrievalConfig,
    cache: LexicalIndexCache,
) -> AskRetrieval:
    """Score the words, then assemble. All of it CPU work, run in a worker thread."""
    try:
        index = cache.index_for(
            guild_id, [(fact.id, fact.content) for fact, _ in similarity_results]
        )
        coverage = index.coverage(question, shipped_stopwords())
    except Exception as error:
        logger.warning(
            "/aura-ask word matching failed (%s); answering from embedding similarity only",
            type(error).__name__,
            exc_info=True,
        )
        return assemble_retrieval(similarity_results, {}, config, lexical_available=False)
    return assemble_retrieval(similarity_results, coverage, config, lexical_available=True)


async def retrieve_for_question(
    similarity_results: Sequence[tuple[Fact, float]],
    *,
    question: str,
    guild_id: int,
    config: HybridRetrievalConfig,
    cache: LexicalIndexCache = SHARED_INDEX_CACHE,
) -> AskRetrieval:
    """Choose the facts one /aura-ask question is answered from.

    Parameters
    ----------
    similarity_results
        EVERY active fact of the guild with its embedding similarity, as
        aura.embeddings.find_similar_facts returns them when asked for all.
    question
        The question, as it was embedded.
    guild_id
        The guild asked in; keys the index cache.
    config
        The gate's and the ranking's numbers.
    cache
        Where the guild's lexical index is kept between questions.

    Returns
    -------
    AskRetrieval
        The selected facts and how they were found. Never raises because of
        word matching: when it fails, coverage is 0 for every fact, the
        selection is exactly the similarity-only selection, and
        `lexical_available` is False.

    Notes
    -----
    Everything here is CPU work over every active fact -- hashing the facts to
    find their index, building it when they changed, scoring, sorting -- so
    all of it runs in a worker thread, like every embedding call, per
    CLAUDE.md's Performance rule; the event loop only waits. If the thread
    itself cannot be started (the executor is shutting down), the selection
    falls back to similarity alone, in place, which is cheap.
    """
    if not similarity_results:
        return assemble_retrieval([], {}, config, lexical_available=True)
    try:
        return await asyncio.to_thread(
            _retrieve_blocking, similarity_results, question, guild_id, config, cache
        )
    except RuntimeError as error:
        logger.warning(
            "/aura-ask word matching could not be scheduled (%s); answering from embedding "
            "similarity only",
            type(error).__name__,
        )
        return assemble_retrieval(similarity_results, {}, config, lexical_available=False)
