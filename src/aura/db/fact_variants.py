"""Storage for Multi-Representation Indexing Part 1's generated fact variants.

See aura.variants_service for how a variant is generated and audited before it
ever reaches this module, and schema.sql's fact_variants table comment for why
this is an internal indexing aid over `facts.content` -- the same role
`facts.embedding` plays -- rather than a fifth knowledge-model component.

This module deliberately stores variants; it does not read them into any
existing similarity search. Wiring the read side into /aura-ask, proactive
relief, or the extraction dedup check is Part 2 (out of scope here) -- see
CLAUDE.md's Multi-Representation Indexing note and the phase brief this module
was built against.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime

import aiosqlite
from pydantic import BaseModel

from aura.db.connection import connection_lock, utc_now_iso

_VARIANT_COLUMNS = "id, fact_id, content, embedding, created_at"


class FactVariant(BaseModel):
    """One stored, audited paraphrase of an active fact's canonical sentence.

    Attributes
    ----------
    id
        Database primary key.
    fact_id
        The fact this paraphrases.
    content
        The paraphrase itself, audited for fidelity before it was stored.
    embedding
        `content`'s vector, same dtype as a fact's (see
        `aura.embeddings.EMBEDDING_DTYPE`).
    created_at
        When it was generated.

    Notes
    -----
    Carries no status of its own -- see schema.sql's fact_variants comment for
    why a variant's lifecycle is entirely derived from the fact it paraphrases
    rather than tracked independently.
    """

    id: int
    fact_id: int
    content: str
    embedding: bytes
    created_at: datetime


def _row_to_variant(row: sqlite3.Row) -> FactVariant:
    id_, fact_id, content, embedding, created_at = row
    return FactVariant(
        id=id_,
        fact_id=fact_id,
        content=content,
        embedding=embedding,
        created_at=datetime.fromisoformat(created_at),
    )


async def store_fact_variants(
    conn: aiosqlite.Connection,
    *,
    fact_id: int,
    contents: list[str],
    embeddings: list[bytes],
) -> list[FactVariant]:
    """Insert every (content, embedding) pair for one fact, in one transaction.

    Parameters
    ----------
    conn
        Open database connection.
    fact_id
        The fact being paraphrased. Must exist: the foreign key is enforced.
    contents
        The audited paraphrases. Must be non-empty.
    embeddings
        One vector per paraphrase, in the same order. Must be the same length
        as `contents`.

    Returns
    -------
    list[FactVariant]
        The stored rows, in insertion order.

    Raises
    ------
    ValueError
        If `contents` is empty or the two lists differ in length.

    Notes
    -----
    All-or-nothing: either every variant this call was given lands, or -- on any
    failure, the foreign key check included -- none of them do. A partial write
    would mean some of one generation episode's variants exist and some do not,
    with nothing in the data to explain why, which is worse than the caller
    simply not calling this at all.

    An empty or mismatched pair of lists is a bug upstream, not a legitimate
    "store nothing" request: a caller that means "nothing survived the audit"
    does not call this function at all.
    """
    if not contents:
        raise ValueError("store_fact_variants requires at least one variant")
    if len(contents) != len(embeddings):
        raise ValueError(
            f"contents and embeddings must be the same length, got "
            f"{len(contents)} and {len(embeddings)}"
        )

    created_at = utc_now_iso()
    async with connection_lock(conn):
        try:
            stored: list[FactVariant] = []
            for content, embedding in zip(contents, embeddings, strict=True):
                cursor = await conn.execute(
                    "INSERT INTO fact_variants (fact_id, content, embedding, created_at) "
                    "VALUES (?, ?, ?, ?)",
                    (fact_id, content, embedding, created_at),
                )
                variant_id = cursor.lastrowid
                assert variant_id is not None  # guaranteed by sqlite after a successful INSERT
                stored.append(
                    FactVariant(
                        id=variant_id,
                        fact_id=fact_id,
                        content=content,
                        embedding=embedding,
                        created_at=datetime.fromisoformat(created_at),
                    )
                )
            await conn.commit()
        except BaseException:
            await conn.rollback()
            raise

    return stored


async def get_variants_for_fact(conn: aiosqlite.Connection, fact_id: int) -> list[FactVariant]:
    """Return every stored variant for one fact, whatever that fact's status.

    Parameters
    ----------
    conn
        Open database connection.
    fact_id
        The fact whose variants to read.

    Returns
    -------
    list[FactVariant]
        Every stored variant, oldest first. Empty if the fact has none.

    Notes
    -----
    Deliberately unfiltered by the fact's status -- this is the raw storage
    view used by tests and by any future admin/debug surface that wants to see
    everything that was ever generated for one specific fact. Callers that need
    "only variants of currently-active facts" want get_active_fact_variants
    instead, which is where that filter belongs (see its own docstring).
    """
    async with connection_lock(conn):
        async with conn.execute(
            f"SELECT {_VARIANT_COLUMNS} FROM fact_variants WHERE fact_id = ? ORDER BY id ASC",
            (fact_id,),
        ) as cursor:
            rows = await cursor.fetchall()
    return [_row_to_variant(row) for row in rows]


async def get_active_fact_variants(conn: aiosqlite.Connection, guild_id: int) -> list[FactVariant]:
    """Return every stored variant whose fact is still active, for one guild.

    Parameters
    ----------
    conn
        Open database connection.
    guild_id
        Guild to read. Passed explicitly so no call site can search across
        guilds by accident.

    Returns
    -------
    list[FactVariant]
        Variants of that guild's ACTIVE facts only.

    Notes
    -----
    The join this whole schema decision was built around (see schema.sql's
    fact_variants comment): a fact retired via /aura-supersede drops its
    variants out of this result the instant its own `status` flips, with no
    second write ever required against fact_variants itself. Not called from
    any similarity search yet -- that wiring is Part 2 -- but this is the
    exact query shape that read path will use, and it is exercised here so the
    join property is proven rather than only asserted in a comment.
    """
    async with connection_lock(conn):
        async with conn.execute(
            """
            SELECT fv.id, fv.fact_id, fv.content, fv.embedding, fv.created_at
            FROM fact_variants fv
            JOIN facts f ON f.id = fv.fact_id
            WHERE f.guild_id = ? AND f.status = 'active'
            ORDER BY fv.id ASC
            """,
            (guild_id,),
        ) as cursor:
            rows = await cursor.fetchall()
    return [_row_to_variant(row) for row in rows]
