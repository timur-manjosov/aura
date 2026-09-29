"""Per-channel on/off state for automatic fact extraction (Phase 3a), set by a
moderator.

A deliberate sibling of aura.db.proactive_channel_config, not a reuse of it:
see the extraction_channel_config table comment in schema.sql for why
extraction needs its own independent gate rather than piggy-backing on
proactive relief's. This module owns exactly one table and imports nothing
from aura.db.repository, for the same isolation reason
proactive_channel_config.py gives.

**A channel with no row is OFF.** Same invariant, same reasoning: extraction
is opt-in per channel, not opt-out, and is_extraction_enabled returns False
for an unconfigured channel rather than assuming a default.
"""

from __future__ import annotations

import aiosqlite

from aura.db.connection import connection_lock, utc_now_iso


async def set_extraction_enabled(
    conn: aiosqlite.Connection,
    *,
    guild_id: int,
    channel_id: int,
    enabled: bool,
    updated_by_id: int,
) -> None:
    """Turn automatic fact extraction on or off for one channel.

    Parameters
    ----------
    conn
        Open database connection.
    guild_id
        Guild the channel belongs to, stored so a guild-wide read needs no
        Discord lookup.
    channel_id
        Channel to configure. The table's primary key.
    enabled
        The new state.
    updated_by_id
        The moderator who made the change, recorded for accountability.

    Returns
    -------
    None

    Notes
    -----
    Idempotent per channel. An upsert keyed on channel_id, identical shape to
    proactive_channel_config.set_channel_enabled: toggling the same channel
    repeatedly leaves exactly one row, always reflecting the most recent
    decision. Writes through the shared per-connection lock like every other
    writer (see aura.db.connection).
    """
    async with connection_lock(conn):
        await conn.execute(
            """
            INSERT INTO extraction_channel_config
                (channel_id, guild_id, extraction_enabled, updated_by_id, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT (channel_id) DO UPDATE SET
                extraction_enabled = excluded.extraction_enabled,
                updated_by_id = excluded.updated_by_id,
                updated_at = excluded.updated_at
            """,
            (channel_id, guild_id, int(enabled), updated_by_id, utc_now_iso()),
        )
        await conn.commit()


async def is_extraction_enabled(conn: aiosqlite.Connection, *, channel_id: int) -> bool:
    """Report whether automatic fact extraction is enabled for one channel.

    Parameters
    ----------
    conn
        Open database connection.
    channel_id
        Channel to check.

    Returns
    -------
    bool
        The configured state, or False when the channel has no row. Extraction
        is opt-in: an unconfigured channel is OFF, never a default-on.

    Notes
    -----
    Not called from any live path yet in Phase 3a-1 -- the filter this gate
    will guard is built and tested but unwired (see aura.extraction) -- but it
    is written to the same cheap, single-indexed-lookup standard
    is_channel_enabled already holds, since this is exactly the query a future
    extraction pipeline would run first, per message, before doing anything
    else.
    """
    async with connection_lock(conn):
        async with conn.execute(
            "SELECT extraction_enabled FROM extraction_channel_config WHERE channel_id = ?",
            (channel_id,),
        ) as cursor:
            row = await cursor.fetchone()
    return bool(row[0]) if row is not None else False
