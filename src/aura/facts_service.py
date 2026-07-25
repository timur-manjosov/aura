"""Service layer between Discord-facing commands and the fact repository.

This exists as a seam, not for its own sake: a fact's embedding has to be
computed at the exact moment it's created, and that has to happen in exactly
one place. Commands must call add_fact() here, never
aura.db.repository.create_fact() directly, so that hook only ever needs to be
added in this one function.
"""
from __future__ import annotations

import aiosqlite
from fastembed import TextEmbedding

from aura.db.models import Fact
from aura.db.repository import create_fact
from aura.embeddings import EMBEDDING_DTYPE, embed_text


async def add_fact(
    conn: aiosqlite.Connection,
    model: TextEmbedding,
    *,
    guild_id: int,
    channel_id: int,
    message_id: int,
    content: str,
) -> Fact:
    """Create a new active fact from a Discord message, embedding included.

    Embeds content before touching the database, then makes exactly one
    repository call that writes content and embedding together in the same
    row. Two separate writes -- insert, then a follow-up update with the
    embedding -- would leave a window where a fact exists without one, the
    same atomicity reasoning supersede_fact already applies to its own
    two-statement transaction.
    """
    embedding = await embed_text(model, content)
    return await create_fact(
        conn,
        guild_id=guild_id,
        channel_id=channel_id,
        message_id=message_id,
        content=content,
        embedding=embedding.astype(EMBEDDING_DTYPE, copy=False).tobytes(),
    )
