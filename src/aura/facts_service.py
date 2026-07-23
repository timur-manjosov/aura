"""Service layer between Discord-facing commands and the fact repository.

This exists as a seam, not for its own sake: a later phase needs to compute
and store an embedding at the exact moment a fact is created, and that has to
happen in exactly one place. Commands must call add_fact() here, never
aura.db.repository.create_fact() directly, so that hook only ever needs to be
added in this one function.
"""
from __future__ import annotations

import aiosqlite

from aura.db.models import Fact
from aura.db.repository import create_fact


async def add_fact(
    conn: aiosqlite.Connection,
    *,
    guild_id: int,
    channel_id: int,
    message_id: int,
    content: str,
) -> Fact:
    """Create a new active fact from a Discord message."""
    return await create_fact(
        conn, guild_id=guild_id, channel_id=channel_id, message_id=message_id, content=content
    )
