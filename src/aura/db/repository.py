"""Async data access for the knowledge model: facts, supersession, and links.

Every operation here runs under the shared per-connection lock from
aura.db.connection, which is also where the reasoning for that lock lives --
it is a rule for all writers on the connection, not just this module's.
"""
from __future__ import annotations

import asyncio
import sqlite3
from datetime import datetime
from pathlib import Path

import aiosqlite

from aura.db.connection import connection_lock, utc_now_iso
from aura.db.models import Fact, FactStatus

_SCHEMA_PATH = Path(__file__).parent / "schema.sql"

_FACT_COLUMNS = (
    "id, guild_id, channel_id, message_id, content, embedding, status, "
    "superseded_by_id, created_at, superseded_at"
)

_INSERT_FACT_SQL = """
INSERT INTO facts (guild_id, channel_id, message_id, content, embedding, status, created_at)
VALUES (?, ?, ?, ?, ?, ?, ?)
"""


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


class SelfSupersessionError(RepositoryError):
    """Raised when supersede_fact_with_existing_successor is asked to make a fact its own successor."""


class SuccessorNotActiveError(RepositoryError):
    """Raised by supersede_fact_with_existing_successor when new_fact_id can't be a successor.

    Covers three cases identically, for the same reason FactAlreadySupersededError
    does for old_fact_id: the successor doesn't exist, belongs to a different
    guild, or is itself already superseded. A fact that isn't currently true
    can't be the fact that retires something else -- allowing it would let a
    supersession chain point at a dead end instead of at whatever's actually
    current.
    """


def _row_to_fact(row: sqlite3.Row) -> Fact:
    (
        id_,
        guild_id,
        channel_id,
        message_id,
        content,
        embedding,
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
        embedding=embedding,
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


async def insert_fact_within_transaction(
    conn: aiosqlite.Connection,
    *,
    guild_id: int,
    channel_id: int,
    message_id: int,
    content: str,
    embedding: bytes,
    created_at: str,
) -> Fact:
    """Insert one active fact. THE CALLER MUST ALREADY HOLD THE CONNECTION LOCK.

    The single statement that brings a fact into existence, factored out so
    there is exactly one of it in the codebase rather than one per operation
    that has to compose a fact insert into a larger transaction. Three callers
    need that today -- create_fact, supersede_fact, and confirming a staged
    extraction candidate (aura.db.pending_facts) -- and a fourth would otherwise
    mean a fourth chance for one copy to drift out of step with the schema.

    Neither locks nor commits, on purpose: every caller is mid-transaction and
    owns both. Calling it without the lock held is a transaction-integrity bug
    of exactly the kind aura.db.connection's docstring describes, and calling it
    from a coroutine that already holds the lock through connection_lock is
    fine -- it simply does not take it again.

    Takes created_at rather than reading the clock, so a caller writing several
    rows in one transaction can timestamp them from one instant.
    """
    cursor = await conn.execute(
        _INSERT_FACT_SQL,
        (guild_id, channel_id, message_id, content, embedding, FactStatus.ACTIVE, created_at),
    )
    fact_id = cursor.lastrowid
    assert fact_id is not None  # guaranteed by sqlite after a successful INSERT

    return Fact(
        id=fact_id,
        guild_id=guild_id,
        channel_id=channel_id,
        message_id=message_id,
        content=content,
        embedding=embedding,
        status=FactStatus.ACTIVE,
        superseded_by_id=None,
        created_at=datetime.fromisoformat(created_at),
        superseded_at=None,
    )


async def create_fact(
    conn: aiosqlite.Connection,
    *,
    guild_id: int,
    channel_id: int,
    message_id: int,
    content: str,
    embedding: bytes,
) -> Fact:
    """Insert a new active fact in its own transaction and return it.

    embedding is required, not optional: every fact this schema can produce
    must carry one from the moment it's written, or find_similar_facts (see
    aura.embeddings) has a silent invariant violation waiting to happen the
    first time it scans a fact with none. Callers must compute it before
    calling this function -- see aura.facts_service.add_fact for why the
    computation itself belongs there, one call site, not here.
    """
    async with connection_lock(conn):
        fact = await insert_fact_within_transaction(
            conn,
            guild_id=guild_id,
            channel_id=channel_id,
            message_id=message_id,
            content=content,
            embedding=embedding,
            created_at=utc_now_iso(),
        )
        await conn.commit()
    return fact


async def supersede_fact(
    conn: aiosqlite.Connection,
    *,
    old_fact_id: int,
    guild_id: int,
    channel_id: int,
    message_id: int,
    content: str,
    embedding: bytes,
) -> Fact:
    """Insert a new active fact and mark old_fact_id superseded by it, atomically.

    embedding is required for the same reason it's required on create_fact:
    the new fact's content is different text than the one it replaces, so it
    needs its own vector, computed by the caller before this is called. Not
    wired to any command yet (see Phase 1d's scope), but every row this
    function can produce must already satisfy find_similar_facts's
    every-active-fact-has-an-embedding invariant.

    Raises FactAlreadySupersededError, with the whole transaction rolled
    back (including the new insert), if old_fact_id doesn't exist, is
    already superseded, or belongs to a different guild than guild_id.
    """
    now = utc_now_iso()
    async with connection_lock(conn):
        try:
            new_fact = await insert_fact_within_transaction(
                conn,
                guild_id=guild_id,
                channel_id=channel_id,
                message_id=message_id,
                content=content,
                embedding=embedding,
                created_at=now,
            )
            new_fact_id = new_fact.id

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

    return new_fact


async def supersede_fact_with_existing_successor(
    conn: aiosqlite.Connection,
    *,
    old_fact_id: int,
    new_fact_id: int,
    guild_id: int,
) -> None:
    """Mark old_fact_id superseded by the already-existing new_fact_id, atomically.

    This is the manual /aura-supersede command's operation, not a variant of
    supersede_fact above: supersede_fact creates a brand-new fact row and
    supersedes the old one with it in the same transaction, for Phase 3a's
    future automatic detection (where the "new" content only exists at the
    moment of detection). Here both facts already exist -- the mod picked an
    old fact and an already-created replacement (e.g. via the "Add as Aura
    Fact" context menu) -- so the only thing left to do is link the chain.
    Reusing supersede_fact by copying the successor's content across would
    insert a second, duplicate active fact and leave the mod's actual chosen
    successor an unlinked orphan: a real correctness bug, not just an unused
    code path. Hence a separate, minimal sibling function instead of reusing
    supersede_fact -- which stays completely untouched.

    Raises SelfSupersessionError if old_fact_id == new_fact_id, checked
    before touching the database. Raises FactAlreadySupersededError if
    old_fact_id doesn't exist, belongs to a different guild than guild_id, or
    is already superseded. Raises SuccessorNotActiveError if new_fact_id
    doesn't exist, belongs to a different guild, or is not itself active --
    checked under the same lock and transaction as the update, so a
    concurrent change to either fact can't slip in between the check and the
    commit.
    """
    if old_fact_id == new_fact_id:
        raise SelfSupersessionError(f"Fact {old_fact_id} cannot supersede itself.")

    now = utc_now_iso()
    async with connection_lock(conn):
        try:
            async with conn.execute(
                "SELECT status FROM facts WHERE id = ? AND guild_id = ?",
                (new_fact_id, guild_id),
            ) as cursor:
                successor_row = await cursor.fetchone()

            if successor_row is None or FactStatus(successor_row[0]) != FactStatus.ACTIVE:
                raise SuccessorNotActiveError(
                    f"Fact {new_fact_id} in guild {guild_id} cannot be a successor: it does "
                    "not exist, does not belong to that guild, or is not currently active."
                )

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


async def get_fact_by_id(conn: aiosqlite.Connection, *, guild_id: int, fact_id: int) -> Fact | None:
    """Return the fact with fact_id in guild_id, or None if no such fact exists there.

    Scoped by guild_id, not just fact_id, so a moderator in one guild can
    never reference (or learn anything about) another guild's fact by
    guessing its numeric ID -- the same isolation get_active_facts and
    get_linked_facts already give every other read path.
    """
    async with connection_lock(conn):
        async with conn.execute(
            f"SELECT {_FACT_COLUMNS} FROM facts WHERE id = ? AND guild_id = ?",
            (fact_id, guild_id),
        ) as cursor:
            row = await cursor.fetchone()
    if row is None:
        return None
    return _row_to_fact(row)


async def link_facts(conn: aiosqlite.Connection, fact_id_1: int, fact_id_2: int) -> None:
    """Create an undirected link between two facts, no-op if it already exists.

    Raises SelfLinkError if the two IDs are equal, FactNotFoundError if
    either fact doesn't exist, or CrossGuildLinkError if they belong to
    different guilds.
    """
    if fact_id_1 == fact_id_2:
        raise SelfLinkError(f"Cannot link fact {fact_id_1} to itself.")

    fact_a_id, fact_b_id = sorted((fact_id_1, fact_id_2))
    now = utc_now_iso()

    async with connection_lock(conn):
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
    async with connection_lock(conn):
        async with conn.execute(
            f"SELECT {_FACT_COLUMNS} FROM facts WHERE guild_id = ? AND status = ?",
            (guild_id, FactStatus.ACTIVE),
        ) as cursor:
            rows = await cursor.fetchall()
    return [_row_to_fact(row) for row in rows]


async def get_linked_facts(conn: aiosqlite.Connection, fact_id: int) -> list[Fact]:
    """Return every fact linked to fact_id, checking both sides of the undirected link."""
    async with connection_lock(conn):
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
