"""P7a's additive columns and indexes, migrated in place on a database that predates them.

`CREATE TABLE IF NOT EXISTS` cannot reshape a table that already exists, so a
database created before P7a keeps its older shape until this runs. Every
change here is additive -- three nullable columns, one nullable timestamp, and
indexes -- so it is migrated in place at start-up rather than refused, the same
stance `aura.db.pending_facts.verify_pending_facts_schema` takes.

The indexes on the new columns live here and NOT in schema.sql: `init_schema`
runs schema.sql before this module, and an index on a column an old database
does not have yet would fail there. Creating them here, after the columns
exist, gives a fresh database and a migrated one the same indexes.

Imports only `aura.db.connection`.
"""

from __future__ import annotations

from typing import Final

import aiosqlite

from aura.db.connection import connection_lock

# (table, column, declared type) -- in the order schema.sql declares them.
ADDED_COLUMNS: Final[tuple[tuple[str, str, str], ...]] = (
    ("facts", "source_author_id", "INTEGER"),
    ("pending_facts", "source_author_id", "INTEGER"),
    ("extraction_queue", "author_id", "INTEGER"),
    ("extraction_channel_config", "privacy_notice_posted_at", "TEXT"),
)

ADDED_INDEXES: Final[tuple[str, ...]] = (
    "CREATE INDEX IF NOT EXISTS idx_facts_source_author ON facts(source_author_id)",
    "CREATE INDEX IF NOT EXISTS idx_pending_facts_source_author ON pending_facts(source_author_id)",
    "CREATE INDEX IF NOT EXISTS idx_ask_calls_user ON ask_calls(user_id)",
)


async def verify_data_obligations_schema(conn: aiosqlite.Connection) -> list[str]:
    """Add P7a's columns and indexes where they are missing.

    Parameters
    ----------
    conn
        Open database connection, after `init_schema`.

    Returns
    -------
    list[str]
        The `table.column` names that were added; empty when the database was
        already current.

    Notes
    -----
    Idempotent, and complete after a crash between two statements: each
    column is checked on its own. A table that does not exist at all is left
    alone (`init_schema` creates every table, so that only happens in a test
    that builds a partial database on purpose).
    """
    added: list[str] = []
    async with connection_lock(conn):
        for table, column, declared_type in ADDED_COLUMNS:
            async with conn.execute(f"PRAGMA table_info({table})") as cursor:
                columns = {row[1] for row in await cursor.fetchall()}
            if not columns or column in columns:
                continue
            await conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {declared_type}")
            added.append(f"{table}.{column}")
        for statement in ADDED_INDEXES:
            await conn.execute(statement)
        await conn.commit()
    return added
