"""Pydantic models mirroring the knowledge model schema: fact, timestamp, status, link."""
from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel


class FactStatus(StrEnum):
    """Whether a fact currently reflects reality, or has been superseded by a newer one.

    A plain string subclass (not just an Enum) so a member can be bound
    directly as a SQLite query parameter and compared directly against the
    TEXT values read back from the `status` column.
    """

    ACTIVE = "active"
    SUPERSEDED = "superseded"


class Fact(BaseModel):
    """One distilled, sourced statement about a server, per CLAUDE.md's knowledge model.

    `superseded_by_id` chains to the fact that replaced this one when
    `status` is SUPERSEDED; both it and `superseded_at` are None while a
    fact is still active.

    `embedding` is `content`'s vector representation, float32 always (see
    aura.embeddings.EMBEDDING_DTYPE), stored raw via ndarray.tobytes() and
    read back via np.frombuffer(fact.embedding, dtype=EMBEDDING_DTYPE) --
    never re-derived here, since deserializing needs the dtype declared once
    and shared, not guessed independently at every read site.
    """

    id: int
    guild_id: int
    channel_id: int
    message_id: int
    content: str
    embedding: bytes
    status: FactStatus
    superseded_by_id: int | None = None
    created_at: datetime
    superseded_at: datetime | None = None


class FactLink(BaseModel):
    """An undirected thematic relationship between two facts.

    `fact_a_id` is always the smaller of the two IDs; see the CHECK
    constraint on `fact_links` in schema.sql for why.
    """

    fact_a_id: int
    fact_b_id: int
    created_at: datetime
