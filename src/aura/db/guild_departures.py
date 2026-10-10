"""Which servers Aura has left and when their data becomes due for purging (P7a, R3).

A departure is marked when Discord reports that Aura was removed from a
server, or when the reconciliation at start-up finds a server with data that
Aura is no longer in (removed while the bot was offline). It is cleared when
Aura comes back to that server before the period ends -- nothing is lost then.
The purge itself is `aura.db.deletion.purge_guild`, run by the purge job.

The period is stored with the mark (`purge_after`), never recomputed: a later,
shorter setting must not shorten a period that already started.

Imports only `aura.db.connection`.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta

import aiosqlite

from aura.db.connection import connection_lock, utc_iso


@dataclass(frozen=True)
class GuildDeparture:
    """One server Aura has left, with the moment its data may be purged.

    Attributes
    ----------
    guild_id
        The server.
    left_at, purge_after
        Fixed-width UTC ISO-8601 strings.
    """

    guild_id: int
    left_at: str
    purge_after: str


async def mark_departed(
    conn: aiosqlite.Connection, *, guild_id: int, now: datetime, grace_days: int
) -> bool:
    """Record that Aura left a server, unless that is already recorded.

    Parameters
    ----------
    conn
        Open database connection.
    guild_id
        The server.
    now
        When the departure was noticed.
    grace_days
        The period before its data may be purged (at least 1).

    Returns
    -------
    bool
        True if a new mark was written; False if one already existed (the
        earlier mark, and its period, are kept).

    Raises
    ------
    ValueError
        If `grace_days` is below 1.
    """
    if grace_days < 1:
        raise ValueError("grace_days must be at least 1")
    async with connection_lock(conn):
        cursor = await conn.execute(
            """
            INSERT INTO guild_departures (guild_id, left_at, purge_after)
            VALUES (?, ?, ?)
            ON CONFLICT (guild_id) DO NOTHING
            """,
            (guild_id, utc_iso(now), utc_iso(now + timedelta(days=grace_days))),
        )
        await conn.commit()
    return cursor.rowcount == 1


async def clear_departure(conn: aiosqlite.Connection, *, guild_id: int) -> bool:
    """Forget a departure because Aura is back in that server.

    Parameters
    ----------
    conn
        Open database connection.
    guild_id
        The server.

    Returns
    -------
    bool
        True if a mark was removed.
    """
    async with connection_lock(conn):
        cursor = await conn.execute("DELETE FROM guild_departures WHERE guild_id = ?", (guild_id,))
        await conn.commit()
    return cursor.rowcount == 1


async def get_departures(conn: aiosqlite.Connection) -> list[GuildDeparture]:
    """Return every marked departure, oldest period end first.

    Parameters
    ----------
    conn
        Open database connection.

    Returns
    -------
    list[GuildDeparture]
        All marks.
    """
    async with (
        connection_lock(conn),
        conn.execute(
            "SELECT guild_id, left_at, purge_after FROM guild_departures ORDER BY purge_after, guild_id"
        ) as cursor,
    ):
        rows = await cursor.fetchall()
    return [GuildDeparture(guild_id=row[0], left_at=row[1], purge_after=row[2]) for row in rows]


async def due_departures(conn: aiosqlite.Connection, *, now: datetime) -> list[GuildDeparture]:
    """Return the departures whose period has ended.

    Parameters
    ----------
    conn
        Open database connection.
    now
        The current moment.

    Returns
    -------
    list[GuildDeparture]
        Marks with `purge_after` at or before `now`.
    """
    moment = utc_iso(now)
    return [
        departure for departure in await get_departures(conn) if departure.purge_after <= moment
    ]


async def reconcile_departures(
    conn: aiosqlite.Connection,
    *,
    guilds_with_data: Iterable[int],
    present_guild_ids: Iterable[int],
    now: datetime,
    grace_days: int,
) -> tuple[int, int]:
    """Bring the marks in line with the servers Aura is actually in.

    Parameters
    ----------
    conn
        Open database connection.
    guilds_with_data
        Servers that have any non-billing row.
    present_guild_ids
        Servers Discord says Aura is in right now (including temporarily
        unavailable ones).
    now
        When this runs.
    grace_days
        The period for newly found departures.

    Returns
    -------
    tuple[int, int]
        (marks written, marks cleared).

    Notes
    -----
    A server Aura left while offline gets its period starting now, not at the
    unknown moment it actually left -- the later, conservative end.
    """
    present = set(present_guild_ids)
    marked = 0
    for guild_id in sorted(set(guilds_with_data) - present):
        if await mark_departed(conn, guild_id=guild_id, now=now, grace_days=grace_days):
            marked += 1
    cleared = 0
    for departure in await get_departures(conn):
        if departure.guild_id in present and await clear_departure(
            conn, guild_id=departure.guild_id
        ):
            cleared += 1
    return marked, cleared
