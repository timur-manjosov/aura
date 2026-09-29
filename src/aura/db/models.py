"""Pydantic models mirroring the knowledge model schema.

One class per component of CLAUDE.md's four-part knowledge model: `Fact`
carries the distilled sentence, its origin reference and its timestamp,
`FactStatus` carries the active/superseded status, and `FactLink` carries the
thematic relationship between two facts.

These are the shapes every layer above the database speaks in. They hold no
behaviour and open no connection; `aura.db.repository` and its siblings own the
SQL that produces and consumes them. Imports nothing from `aura`, so any module
may depend on it.
"""

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
    """One distilled, sourced statement about a server.

    Attributes
    ----------
    id
        Database primary key.
    guild_id, channel_id, message_id
        The Discord permalink components of the message this was distilled
        from. The knowledge model stores this reference instead of a second
        copy of the original text (CLAUDE.md, Fact).
    content
        The distilled sentence itself -- one sentence, not the raw message.
    embedding
        `content`'s vector representation, float32 always (see
        `aura.embeddings.EMBEDDING_DTYPE`), stored raw via `ndarray.tobytes`
        and read back via `np.frombuffer(..., dtype=EMBEDDING_DTYPE)`. Never
        re-derived here: deserializing needs the dtype declared once and
        shared, not guessed independently at every read site.
    status
        Whether this fact currently reflects reality.
    superseded_by_id, superseded_at
        The fact that replaced this one, and when. Both are None exactly while
        `status` is ACTIVE.
    created_at
        The knowledge model's Timestamp component.

    Notes
    -----
    A superseded fact is never deleted, so the history of what used to be true
    stays intact; retrieval filters on `status` instead.
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

    Attributes
    ----------
    fact_a_id, fact_b_id
        The linked facts. `fact_a_id` is always the smaller of the two, so one
        relationship has exactly one row whichever order it was created in;
        see the CHECK constraint on `fact_links` in schema.sql.
    created_at
        When the link was recorded.

    Notes
    -----
    This is CLAUDE.md's Link component: what lets thematically related facts,
    spread across time and channels, be pulled into one synthesized answer
    with multiple citations rather than returned as isolated fragments.
    """

    fact_a_id: int
    fact_b_id: int
    created_at: datetime
