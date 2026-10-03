"""Run labelled questions through /aura-ask's retrieval exactly as the command does. Free.

For every question this calls aura.embeddings.find_similar_facts for every
active fact and aura.retrieval.hybrid.retrieve_for_question on the result --
the two calls aura.commands.ask makes -- and, on the same similarity results,
the selection /aura-ask used before hybrid retrieval (the five most similar
facts at or above SIMILARITY_THRESHOLD), so both are measured on identical
inputs.

Imports aura (read paths only, plus create_fact for an in-memory guild of
invented facts) and the sibling `cases` module.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass

import aiosqlite
from fastembed import TextEmbedding

from ask_retrieval_eval.cases import EvalCase, InventedFact
from aura.db.repository import create_fact
from aura.embeddings import EMBEDDING_DTYPE, SYNTHESIS_FACT_LIMIT, embed_texts, find_similar_facts
from aura.retrieval.hybrid import HybridRetrievalConfig, retrieve_for_question
from aura.retrieval.index_cache import LexicalIndexCache
from aura.retrieval.stopwords import shipped_stopwords

# One batch per call to the embedding model when seeding a guild: large enough
# to amortise the per-call cost, small enough to keep memory flat at thousands.
_EMBEDDING_BATCH: int = 256


@dataclass(frozen=True, slots=True)
class FactScores:
    """One fact's two scores for one question.

    Attributes
    ----------
    similarity
        Embedding similarity, as find_similar_facts computed it.
    coverage
        Lexical coverage, as the hybrid gate saw it.
    """

    similarity: float
    coverage: float


@dataclass(frozen=True, slots=True)
class QuestionOutcome:
    """What both selections chose for one question.

    Attributes
    ----------
    case_id
        The question's case ID.
    hybrid
        Fact IDs the hybrid gate selected, in order.
    baseline
        Fact IDs the similarity-only selection selected, in order.
    scores
        Both scores for every active fact.
    retrieval_ms
        Wall time of retrieve_for_question alone (word scoring, the gate and
        the ranking -- what hybrid retrieval adds to a question).
    """

    case_id: str
    hybrid: tuple[int, ...]
    baseline: tuple[int, ...]
    scores: dict[int, FactScores]
    retrieval_ms: float


async def seed_invented_guild(
    conn: aiosqlite.Connection,
    model: TextEmbedding,
    facts: Sequence[InventedFact],
    *,
    guild_id: int,
) -> dict[int, int]:
    """Store invented facts as one guild's active facts, with real embeddings.

    Parameters
    ----------
    conn
        An open database with the schema initialised.
    model
        The production embedding model.
    facts
        The facts to store.
    guild_id
        The guild to store them under.

    Returns
    -------
    dict[int, int]
        Database fact ID per invented fact ID.
    """
    stored: dict[int, int] = {}
    for start in range(0, len(facts), _EMBEDDING_BATCH):
        batch = facts[start : start + _EMBEDDING_BATCH]
        vectors = await embed_texts(model, [fact.content for fact in batch])
        for fact, vector in zip(batch, vectors, strict=True):
            row = await create_fact(
                conn,
                guild_id=guild_id,
                channel_id=1,
                message_id=fact.fact_id,
                content=fact.content,
                embedding=vector.astype(EMBEDDING_DTYPE, copy=False).tobytes(),
            )
            stored[fact.fact_id] = row.id
    return stored


async def run_questions(
    conn: aiosqlite.Connection,
    model: TextEmbedding,
    cases: Sequence[EvalCase],
    *,
    guild_id: int,
    config: HybridRetrievalConfig,
    cache: LexicalIndexCache,
    fact_id_map: dict[int, int] | None = None,
) -> list[QuestionOutcome]:
    """Answer every question's retrieval both ways.

    Parameters
    ----------
    conn
        The database to read the guild's facts from.
    model
        The production embedding model.
    cases
        The questions.
    guild_id
        The guild to search.
    config
        The hybrid gate's numbers; its similarity threshold is also the
        similarity-only selection's bar.
    cache
        The lexical index cache to use.
    fact_id_map
        Database fact ID per label fact ID, when the labels use other IDs
        (invented sets); None when they are database IDs already.

    Returns
    -------
    list[QuestionOutcome]
        One outcome per case, in order, with fact IDs in the labels' terms.
    """
    to_label = (
        {database_id: label_id for label_id, database_id in fact_id_map.items()}
        if fact_id_map is not None
        else None
    )

    def label(fact_id: int) -> int:
        return to_label[fact_id] if to_label is not None else fact_id

    outcomes: list[QuestionOutcome] = []
    for case in cases:
        results = await find_similar_facts(
            conn, model, guild_id=guild_id, query=case.query, top_k=sys.maxsize
        )
        baseline = tuple(
            label(fact.id)
            for fact, similarity in results[:SYNTHESIS_FACT_LIMIT]
            if similarity >= config.similarity_threshold
        )
        started = time.perf_counter()
        retrieval = await retrieve_for_question(
            results, question=case.query, guild_id=guild_id, config=config, cache=cache
        )
        elapsed_ms = (time.perf_counter() - started) * 1000
        coverage = cache.index_for(
            guild_id, [(fact.id, fact.content) for fact, _ in results]
        ).coverage(case.query, shipped_stopwords())
        outcomes.append(
            QuestionOutcome(
                case_id=case.case_id,
                hybrid=tuple(label(fact.id) for fact in retrieval.facts),
                baseline=baseline,
                scores={
                    label(fact.id): FactScores(similarity=similarity, coverage=coverage[fact.id])
                    for fact, similarity in results
                },
                retrieval_ms=elapsed_ms,
            )
        )
    return outcomes
