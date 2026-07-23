"""Async data access for the knowledge model: facts, supersession, and links.

All operations against a given aiosqlite.Connection are serialized through a
lock scoped to that connection (see _lock_for). This isn't about SQLite
throughput (aiosqlite already funnels every statement through one worker
thread) -- it's about transaction integrity. A bare aiosqlite.Connection has
exactly one transaction at a time; without this lock, two concurrently-running
multi-statement operations (e.g. two supersede_fact calls) interleave their
statements into what SQLite sees as a *single* transaction, so one call's
rollback can silently discard the other call's already-committed work.
Serializing every operation's DB work behind one lock is the simplest correct
fix at the data volume this project targets.

The lock is deliberately *per-connection*, not a single module-level
asyncio.Lock. asyncio's lock only binds to a specific event loop lazily, the
first time it's actually contended -- a module-level lock that's never
contended (e.g. in a test that never runs two ops concurrently) will happily
survive being reused from a later, different event loop, but the moment it
*is* contended under one loop, it permanently binds to that loop and raises
RuntimeError if a different loop ever contends it again. In production
there's exactly one connection and one long-lived event loop for the
process's whole life, so this never surfaces -- but a test suite naturally
creates a fresh event loop per test, so a shared module-level lock would
break the second test that exercises real contention. Keying the lock by
connection identity means each connection (and, in tests, each fresh
in-memory database) gets its own lock tied to whichever loop actually uses it.
"""
from __future__ import annotations

import asyncio
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from weakref import WeakKeyDictionary

import aiosqlite

from aura.db.models import Fact, FactStatus

_SCHEMA_PATH = Path(__file__).parent / "schema.sql"

_FACT_COLUMNS = (
    "id, guild_id, channel_id, message_id, content, status, "
    "superseded_by_id, created_at, superseded_at"
)

_connection_locks: WeakKeyDictionary[aiosqlite.Connection, asyncio.Lock] = WeakKeyDictionary()


def _lock_for(conn: aiosqlite.Connection) -> asyncio.Lock:
    """Return this connection's write/read serialization lock, creating it on first use."""
    lock = _connection_locks.get(conn)
    if lock is None:
        lock = asyncio.Lock()
        _connection_locks[conn] = lock
    return lock


class RepositoryError(Exception):
    """Base class for knowledge-model data-access errors."""


class FactAlreadySupersededError(RepositoryError):
    """Raised by supersede_fact when old_fact_id can't be superseded.

    Covers both cases identically and on purpose: the fact never existed, or
    it did but is no longer active (including having just lost a concurrent
    race to another supersede_fact call). The caller cannot tell these apart,
    and shouldn't need to -- either way nothing is superseded and nothing new
    is left dangling.
    """


class SelfLinkError(RepositoryError):
    """Raised when link_facts is called with the same fact ID twice."""


class FactNotFoundError(RepositoryError):
    """Raised when link_facts is given a fact ID that doesn't exist."""


class CrossGuildLinkError(RepositoryError):
    """Raised when link_facts is asked to link facts from two different guilds.

    fact_links has no guild_id column of its own -- guild isolation for links
    is enforced here, at write time, so every later read (get_linked_facts)
    can trust that anything reachable from a fact never crosses a guild
    boundary, without having to re-check on every query.
    """


def _now_iso() -> str:
    """Current UTC time as a fixed-width ISO-8601 string (always 6 fractional digits).

    datetime.isoformat() omits the microsecond field entirely when it's
    zero, which makes plain lexicographic string comparison of timestamps
    unreliable. strftime's %f is always zero-padded to 6 digits, so these
    strings sort correctly as text -- relied on by the periodic digest in a
    later phase, and handy for `sqlite3 data/aura.db "select * from facts
    order by created_at"` in the meantime.
    """
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f+00:00")


def _row_to_fact(row: sqlite3.Row) -> Fact:
    (
        id_,
        guild_id,
        channel_id,
        message_id,
        content,
        status,
        superseded_by_id,
        created_at,
        superseded_at,
    ) = row
    return Fact(
        id=id_,
        guild_id=guild_id,
        channel_id=channel_id,
        message_id=message_id,
        content=content,
        status=FactStatus(status),
        superseded_by_id=superseded_by_id,
        created_at=created_at,
        superseded_at=superseded_at,
    )


async def init_schema(conn: aiosqlite.Connection) -> None:
    """Enable required PRAGMAs and create the knowledge-model tables if they don't exist.

    Must run exactly once per connection, before any other function in this
    module is called: PRAGMA foreign_keys is a per-connection setting SQLite
    never infers or persists on its own, so a connection it hasn't run
    against would silently accept orphaned fact_links rows.
    """
    await conn.execute("PRAGMA foreign_keys = ON")
    await conn.execute("PRAGMA journal_mode = WAL")
    schema_sql = await asyncio.to_thread(_SCHEMA_PATH.read_text, encoding="utf-8")
    await conn.executescript(schema_sql)
    await conn.commit()


async def create_fact(
    conn: aiosqlite.Connection,
    *,
    guild_id: int,
    channel_id: int,
    message_id: int,
    content: str,
) -> Fact:
    """Insert a new active fact and return it."""
    created_at = _now_iso()
    async with _lock_for(conn):
        cursor = await conn.execute(
            """
            INSERT INTO facts (guild_id, channel_id, message_id, content, status, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (guild_id, channel_id, message_id, content, FactStatus.ACTIVE, created_at),
        )
        await conn.commit()
        fact_id = cursor.lastrowid
        assert fact_id is not None  # guaranteed by sqlite after a successful INSERT

    return Fact(
        id=fact_id,
        guild_id=guild_id,
        channel_id=channel_id,
        message_id=message_id,
        content=content,
        status=FactStatus.ACTIVE,
        superseded_by_id=None,
        created_at=datetime.fromisoformat(created_at),
        superseded_at=None,
    )


async def supersede_fact(
    conn: aiosqlite.Connection,
    *,
    old_fact_id: int,
    guild_id: int,
    channel_id: int,
    message_id: int,
    content: str,
) -> Fact:
    """Insert a new active fact and mark old_fact_id superseded by it, atomically.

    Raises FactAlreadySupersededError, with the whole transaction rolled
    back (including the new insert), if old_fact_id doesn't exist, is
    already superseded, or belongs to a different guild than guild_id.
    """
    now = _now_iso()
    async with _lock_for(conn):
        try:
            cursor = await conn.execute(
                """
                INSERT INTO facts (guild_id, channel_id, message_id, content, status, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (guild_id, channel_id, message_id, content, FactStatus.ACTIVE, now),
            )
            new_fact_id = cursor.lastrowid
            assert new_fact_id is not None

            update_cursor = await conn.execute(
                """
                UPDATE facts
                SET status = ?, superseded_by_id = ?, superseded_at = ?
                WHERE id = ? AND status = ? AND guild_id = ?
                """,
                (FactStatus.SUPERSEDED, new_fact_id, now, old_fact_id, FactStatus.ACTIVE, guild_id),
            )

            if update_cursor.rowcount != 1:
                raise FactAlreadySupersededError(
                    f"Fact {old_fact_id} in guild {guild_id} was not superseded: it does "
                    "not exist, does not belong to that guild, or is already superseded "
                    "(possibly by a concurrent call)."
                )
        except BaseException:
            await conn.rollback()
            raise

        await conn.commit()

    return Fact(
        id=new_fact_id,
        guild_id=guild_id,
        channel_id=channel_id,
        message_id=message_id,
        content=content,
        status=FactStatus.ACTIVE,
        superseded_by_id=None,
        created_at=datetime.fromisoformat(now),
        superseded_at=None,
    )


async def link_facts(conn: aiosqlite.Connection, fact_id_1: int, fact_id_2: int) -> None:
    """Create an undirected link between two facts, no-op if it already exists.

    Raises SelfLinkError if the two IDs are equal, FactNotFoundError if
    either fact doesn't exist, or CrossGuildLinkError if they belong to
    different guilds.
    """
    if fact_id_1 == fact_id_2:
        raise SelfLinkError(f"Cannot link fact {fact_id_1} to itself.")

    fact_a_id, fact_b_id = sorted((fact_id_1, fact_id_2))
    now = _now_iso()

    async with _lock_for(conn):
        async with conn.execute(
            "SELECT id, guild_id FROM facts WHERE id IN (?, ?)", (fact_a_id, fact_b_id)
        ) as cursor:
            rows = await cursor.fetchall()

        guild_by_id = {row[0]: row[1] for row in rows}
        for fact_id in (fact_a_id, fact_b_id):
            if fact_id not in guild_by_id:
                raise FactNotFoundError(f"Fact {fact_id} does not exist.")

        if guild_by_id[fact_a_id] != guild_by_id[fact_b_id]:
            raise CrossGuildLinkError(
                f"Cannot link fact {fact_a_id} (guild {guild_by_id[fact_a_id]}) to fact "
                f"{fact_b_id} (guild {guild_by_id[fact_b_id]}): they belong to different guilds."
            )

        await conn.execute(
            "INSERT OR IGNORE INTO fact_links (fact_a_id, fact_b_id, created_at) VALUES (?, ?, ?)",
            (fact_a_id, fact_b_id, now),
        )
        await conn.commit()


async def get_active_facts(conn: aiosqlite.Connection, guild_id: int) -> list[Fact]:
    """Return every currently-active fact for a guild."""
    async with _lock_for(conn):
        async with conn.execute(
            f"SELECT {_FACT_COLUMNS} FROM facts WHERE guild_id = ? AND status = ?",
            (guild_id, FactStatus.ACTIVE),
        ) as cursor:
            rows = await cursor.fetchall()
    return [_row_to_fact(row) for row in rows]


async def get_linked_facts(conn: aiosqlite.Connection, fact_id: int) -> list[Fact]:
    """Return every fact linked to fact_id, checking both sides of the undirected link."""
    async with _lock_for(conn):
        async with conn.execute(
            f"""
            SELECT {_FACT_COLUMNS} FROM facts
            WHERE id IN (
                SELECT fact_b_id FROM fact_links WHERE fact_a_id = ?
                UNION
                SELECT fact_a_id FROM fact_links WHERE fact_b_id = ?
            )
            """,
            (fact_id, fact_id),
        ) as cursor:
            rows = await cursor.fetchall()
    return [_row_to_fact(row) for row in rows]
