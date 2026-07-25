"""Per-channel on/off state for proactive relief, set by a moderator.

NOT part of the knowledge model, for the same reason aura.db.proactive_signals
and aura.db.proactive_state aren't: a channel's on/off switch is none of the
four things CLAUDE.md admits into it. This module owns one table
(proactive_channel_config, see schema.sql) and imports nothing from
aura.db.repository in either direction, so the whole proactive mechanism stays
separable from the facts it protects.

The one invariant worth stating outright, because the whole pipeline hangs off
it: **a channel with no row is OFF.** Proactive relief is opt-in per channel,
not opt-out -- see the table comment in schema.sql for why (first money, first
public posts, still-placeholder thresholds, CLAUDE.md's "deliberately
conservative" mandate). is_channel_enabled returns False for an unconfigured
channel rather than assuming a default, and the pipeline reads it as the very
first, cheapest gate so a disabled channel incurs zero further computation.
"""
from __future__ import annotations

import aiosqlite

from aura.db.connection import connection_lock, utc_now_iso


async def set_channel_enabled(
    conn: aiosqlite.Connection,
    *,
    guild_id: int,
    channel_id: int,
    enabled: bool,
    updated_by_id: int,
) -> None:
    """Turn proactive relief on or off for one channel, recording who did it.

    An upsert keyed on channel_id: toggling the same channel repeatedly leaves
    exactly one row, always reflecting the most recent decision. Writes through
    the shared per-connection lock like every other writer (see
    aura.db.connection), since an unsynchronized COMMIT here would end another
    operation's in-flight transaction early.
    """
    async with connection_lock(conn):
        await conn.execute(
            """
            INSERT INTO proactive_channel_config
                (channel_id, guild_id, proactive_enabled, updated_by_id, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT (channel_id) DO UPDATE SET
                proactive_enabled = excluded.proactive_enabled,
                updated_by_id = excluded.updated_by_id,
                updated_at = excluded.updated_at
            """,
            (channel_id, guild_id, int(enabled), updated_by_id, utc_now_iso()),
        )
        await conn.commit()


async def is_channel_enabled(conn: aiosqlite.Connection, *, channel_id: int) -> bool:
    """Whether proactive relief is enabled for channel_id. False if never configured.

    The default is OFF, not ON: a channel with no row has never been opted in,
    and Aura must stay silent there. This is read once per incoming message as
    the pipeline's first gate, so it is a single indexed primary-key lookup and
    nothing more -- the whole point is that a disabled channel is cheap.

    Read under the connection lock, consistent with every other access on this
    shared connection; a lone SELECT still has to be serialized against a
    concurrent multi-statement writer's transaction boundary.
    """
    async with connection_lock(conn):
        async with conn.execute(
            "SELECT proactive_enabled FROM proactive_channel_config WHERE channel_id = ?",
            (channel_id,),
        ) as cursor:
            row = await cursor.fetchone()
    return bool(row[0]) if row is not None else False
