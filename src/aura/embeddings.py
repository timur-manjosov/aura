"""Semantic embedding and similarity search over a guild's active facts.

Ranking only -- this module decides how similar two pieces of text are, not
whether a result is similar *enough* to act on. That threshold belongs to
each caller (Phase 1e's direct query, Phase 2's much stricter proactive-relief
bar), so it's deliberately absent here.
"""
from __future__ import annotations

import asyncio
from collections.abc import Sequence

import aiosqlite
import numpy as np
from fastembed import TextEmbedding

from aura.db.fact_variants import FactVariant, get_active_fact_variants
from aura.db.models import Fact
from aura.db.repository import get_active_facts

# Fixed everywhere an embedding is produced or read, so serialization
# (ndarray.tobytes()) and deserialization (np.frombuffer(..., dtype=...))
# can never silently disagree about how many bytes represent one float --
# see Fact.embedding's docstring, the other place this same dtype matters.
EMBEDDING_DTYPE = np.float32

# How many facts either answering trigger may carry into one synthesis call.
#
# A bound, not a preference, which is why it is a named constant rather than a
# bare default buried in a signature: it is the only thing standing between a
# guild with hundreds of facts on one topic and an unbounded prompt. Both
# triggers observe it -- /aura-ask by taking this default, proactive relief by
# passing it explicitly, since for an unprompted call it is a cost ceiling and
# not a convenience.
#
# Five, for two independent reasons that agree. Semantically, CLAUDE.md's
# "Link" component asks for thematically related facts to be pulled into ONE
# synthesized answer with multiple citations; a handful is what a human would
# weigh at once, and past that an answer stops reading as an answer. And in
# cost terms it stays negligible: fact content is capped at 4000 characters by
# the entry modal, so five is a hard worst case of ~20k characters (~6k tokens)
# per call, while real facts are one distilled sentence each and land nearer
# ~500 tokens for the whole set. Raising this would widen the worst case
# faster than it would improve any answer.
SYNTHESIS_FACT_LIMIT = 5


async def embed_text(model: TextEmbedding, text: str) -> np.ndarray:
    """Compute one text's embedding vector as a fixed-dtype numpy array.

    fastembed's inference is blocking, CPU-bound work (an ONNX Runtime
    session call); running it directly on the event loop would stall every
    other coroutine -- including unrelated Discord events -- for its
    duration. asyncio.to_thread offloads it to a worker thread instead, per
    CLAUDE.md's Performance section.

    Takes text wrapped in a single-element list rather than passing the bare
    string to model.embed(): TextEmbedding.embed's signature accepts a bare
    str as one document (verified: it does not iterate the string
    character-by-character), but a list is unambiguous regardless, and this
    is the one call site every other embedding call in this module goes
    through.
    """

    def _run() -> np.ndarray:
        (raw_embedding,) = list(model.embed([text]))
        return np.asarray(raw_embedding, dtype=EMBEDDING_DTYPE)

    return await asyncio.to_thread(_run)


async def embed_texts(model: TextEmbedding, texts: Sequence[str]) -> list[np.ndarray]:
    """Compute one embedding vector per text, in one batched inference call.

    One call into fastembed for the whole sequence rather than a loop over
    embed_text, per CLAUDE.md's Performance section: the fixed per-call cost
    of an ONNX Runtime invocation dominates at these batch sizes, so N
    separate calls cost far more than one call with N documents. Results
    come back in input order, which callers rely on to line a vector up with
    the text it came from.

    Rejects an empty sequence rather than returning an empty list: every
    caller embeds a known-non-empty set, so an empty batch means a bug
    upstream, and failing here is far cheaper to diagnose than the
    mis-shaped, silently-empty comparison it would otherwise produce.
    """
    if not texts:
        raise ValueError("embed_texts requires at least one text")

    def _run() -> list[np.ndarray]:
        return [
            np.asarray(raw_embedding, dtype=EMBEDDING_DTYPE)
            for raw_embedding in model.embed(list(texts))
        ]

    embeddings = await asyncio.to_thread(_run)
    if len(embeddings) != len(texts):
        raise ValueError(
            f"embedding model returned {len(embeddings)} vectors for {len(texts)} texts; "
            "results can no longer be matched to their input"
        )
    return embeddings


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors, normalizing both first.

    Normalizes unconditionally instead of trusting the embedding model to
    already return unit vectors: verified empirically that
    paraphrase-multilingual-MiniLM-L12-v2 does not (raw norms observed
    around 3.6-5.8, not 1.0). Normalizing costs nothing when a vector
    already happens to be unit length, and removes a whole class of silent
    scoring bug if that assumption is ever wrong -- for this model, or any
    other EMBEDDING_MODEL a deployment might swap in later.

    Returns 0.0 -- not NaN, not a raised exception -- if either vector is
    all-zero, since 0/0 similarity to a degenerate vector is undefined, not
    "very similar."
    """
    a64 = np.asarray(a, dtype=np.float64)
    b64 = np.asarray(b, dtype=np.float64)

    norm_a = float(np.linalg.norm(a64))
    norm_b = float(np.linalg.norm(b64))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return float(np.dot(a64 / norm_a, b64 / norm_b))


def group_variants_by_fact(variants: Sequence[FactVariant]) -> dict[int, list[FactVariant]]:
    """Group a flat list of variants by the fact_id each one paraphrases.

    A plain lookup table, built once per search rather than once per fact
    scored: every caller here fetches all of a guild's active variants in one
    query (see aura.db.fact_variants.get_active_fact_variants) and then needs
    O(1) access to "this fact's variants" per fact in the scoring loop, not a
    fresh scan of the flat list for every one of them.
    """
    grouped: dict[int, list[FactVariant]] = {}
    for variant in variants:
        grouped.setdefault(variant.fact_id, []).append(variant)
    return grouped


def best_similarity(
    query_embedding: np.ndarray,
    fact: Fact,
    variants_by_fact: dict[int, list[FactVariant]] | None = None,
) -> float:
    """Max cosine similarity between query_embedding and fact's canonical sentence OR any of its variants.

    Multi-Representation Indexing Part 2 (see aura.variants_service and
    reports/variant-indexing-part1.txt): a fact is findable through any of its
    audited, meaning-preserving paraphrasings, not only through its one
    canonical vector. This is the one place that "maximum over candidate
    vectors" logic lives -- every similarity search in this project (direct
    query, proactive gate/responder, extraction dedup) calls through here
    rather than each re-implementing its own max, which is exactly the kind of
    drift CLAUDE.md's PROACTIVE_SIMILARITY_THRESHOLD/SIMILARITY_THRESHOLD split
    already showed this project one duplicated constant can cause.

    Which vector wins never changes what a caller shows or cites: every caller
    still reads `fact.content`, the canonical sentence, off the returned Fact
    -- a variant is only ever the reason a fact was found, never the text
    displayed.

    variants_by_fact is optional so a caller that has not fetched any variants
    (or a fact with none stored) degrades cleanly to comparing against the
    canonical vector alone, the exact pre-Part-2 behaviour.

    A vector whose stored bytes decode to a non-finite score (a corrupted
    embedding) is skipped rather than allowed to win or silently poison the
    max -- one bad variant must not sink a fact that is otherwise perfectly
    findable through its canonical sentence or its other variants. Returns
    NaN, matching cosine_similarity's non-finite passthrough, only if every
    candidate vector for this fact -- canonical included -- is unusable;
    callers that already guard against a non-finite score (e.g.
    aura.extraction.pipeline._best_matching_fact) see the same signal they did
    before this helper existed.
    """
    candidates = [np.frombuffer(fact.embedding, dtype=EMBEDDING_DTYPE)]
    if variants_by_fact:
        candidates.extend(
            np.frombuffer(variant.embedding, dtype=EMBEDDING_DTYPE)
            for variant in variants_by_fact.get(fact.id, ())
        )

    scores = [cosine_similarity(query_embedding, candidate) for candidate in candidates]
    finite_scores = [score for score in scores if np.isfinite(score)]
    if not finite_scores:
        return float("nan")
    return max(finite_scores)


async def find_similar_facts(
    conn: aiosqlite.Connection,
    model: TextEmbedding,
    *,
    guild_id: int,
    query: str,
    top_k: int = SYNTHESIS_FACT_LIMIT,
) -> list[tuple[Fact, float]]:
    """Rank guild_id's active facts by similarity to query, most similar first.

    A linear scan over every active fact in the guild is the correct design
    at this project's data volume, not a placeholder for a future vector
    database -- CLAUDE.md's Performance section already rules that out as
    infrastructure sized for a scale problem Aura doesn't have. Each fact now
    scores against 1 (canonical only) to ~7 (canonical + up to
    VARIANT_COUNT audited variants) vectors instead of 1 -- see
    reports/variant-indexing-part2.txt Section on performance for why that
    stays negligible at this project's realistically-anticipated fact counts
    (hundreds, not thousands) and is worth revisiting only if that changes.

    Never errors on sparse data: an empty guild, or a top_k larger than the
    number of active facts, both just return however many results actually
    exist (possibly zero).

    Ties are broken by fact id, oldest first, so the ranking is fully
    determined by the data rather than by the order SQLite happened to return
    rows in. This is not cosmetic. Since Phase 2b-4 the top_k cut decides which
    facts are shown to a paid model, and near-duplicate facts -- the case that
    produces ties -- are precisely what a guild accumulates on a topic it has
    documented more than once. Without a tiebreak, the same question could draw
    a different set of facts across two calls, which would make an inconsistent
    answer impossible to reproduce and therefore impossible to diagnose. Oldest
    first is the meaningful direction of the two: where a fact really has been
    superseded but nobody has run /aura-supersede yet, the original is the one
    whose Discord permalink a moderator needs in order to fix it.
    """
    query_embedding = await embed_text(model, query)
    facts = await get_active_facts(conn, guild_id)
    variants_by_fact = group_variants_by_fact(await get_active_fact_variants(conn, guild_id))

    scored = [
        (fact, best_similarity(query_embedding, fact, variants_by_fact)) for fact in facts
    ]
    scored.sort(key=lambda pair: (-pair[1], pair[0].id))
    return scored[:top_k]
