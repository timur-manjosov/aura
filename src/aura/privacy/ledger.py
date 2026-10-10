"""The deletion ledger: a content-free record of every executed deletion, in its own file.

Why a separate file: a database backup restored after a deletion would bring
the deleted rows back. The ledger is never part of a database backup and never
restored with one; at every start (and every purge tick) each entry's rule is
run again against the live database, bounded to data from before the entry's
moment, so whatever a restored backup brought back is removed again and
nothing that arrived later is touched.

What an entry holds: the kind of deletion (member, server, one fact), why
(member request, admin request, moderator, operator, Aura left the server), the
server it applied to, for a member their Discord user ID and the mode they
chose, for one fact its ID and timestamp, when it was requested and completed,
and how many rows it removed per table. No fact text, no message text, no
names. The member ID is kept because without it the deletion could not be
re-applied after a restore -- this is stated in the privacy facts sheet.

Writing order, and why it makes a crash harmless: the entry is written first,
then the deletion runs, then the entry is marked complete. A crash in between
leaves an entry whose rule the next start runs anyway.

Imports `aura.db.connection`, `aura.db.deletion` and `aura.db.encryption`.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Final

import aiosqlite

from aura.db.connection import connection_lock, utc_iso
from aura.db.deletion import DeletionCounts, MemberDeletionMode
from aura.db.encryption import DatabaseOpenError, connect_database, create_encrypted_database

_LEDGER_SCHEMA: Final = """
CREATE TABLE IF NOT EXISTS deletions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT NOT NULL CHECK (kind IN ('member', 'guild', 'fact')),
    reason TEXT NOT NULL CHECK (reason IN (
        'member_request', 'admin_request', 'moderator_request', 'operator_request',
        'left_server'
    )),
    guild_id INTEGER,
    user_id INTEGER,
    mode TEXT CHECK (mode IS NULL OR mode IN ('delete_facts', 'unlink')),
    fact_id INTEGER,
    fact_created_at TEXT,
    requested_at TEXT NOT NULL,
    completed_at TEXT,
    counts TEXT,
    CHECK (kind != 'member' OR (user_id > 0 AND mode IS NOT NULL)),
    CHECK (kind != 'guild' OR guild_id IS NOT NULL),
    CHECK (kind != 'fact' OR (guild_id IS NOT NULL AND fact_id IS NOT NULL
                              AND fact_created_at IS NOT NULL))
);
"""


class DeletionKind(StrEnum):
    """What one ledger entry deleted."""

    MEMBER = "member"
    GUILD = "guild"
    FACT = "fact"


class DeletionReason(StrEnum):
    """Why it was deleted."""

    MEMBER_REQUEST = "member_request"
    ADMIN_REQUEST = "admin_request"
    MODERATOR_REQUEST = "moderator_request"
    OPERATOR_REQUEST = "operator_request"
    LEFT_SERVER = "left_server"


@dataclass(frozen=True)
class LedgerEntry:
    """One executed (or started) deletion.

    Attributes
    ----------
    kind, reason
        What and why.
    requested_at
        The bound of the rule: data from after this moment is never touched.
    guild_id
        The server; None for a member's request across all servers.
    user_id, mode
        A member entry's subject and choice.
    fact_id, fact_created_at
        A fact entry's fact.
    entry_id
        The ledger's own row ID; None before it is written.
    completed_at
        None while the deletion has not finished (a crash, or still running).
    """

    kind: DeletionKind
    reason: DeletionReason
    requested_at: datetime
    guild_id: int | None = None
    user_id: int | None = None
    mode: MemberDeletionMode | None = None
    fact_id: int | None = None
    fact_created_at: str | None = None
    entry_id: int | None = None
    completed_at: str | None = None


class DeletionLedger:
    """The open ledger file.

    Notes
    -----
    Holds its own aiosqlite connection (encrypted with the database key when
    one is configured) and serializes every write through that connection's
    lock, like the main database.
    """

    def __init__(self, conn: aiosqlite.Connection) -> None:
        self._conn = conn

    @classmethod
    async def open(cls, path: str, key_hex: str | None) -> DeletionLedger:
        """Open (or, the first time, create) the ledger file.

        Parameters
        ----------
        path
            The ledger file.
        key_hex
            The database key, or None.

        Returns
        -------
        DeletionLedger
            Ready to use.

        Raises
        ------
        aura.db.encryption.DatabaseOpenError
            If an existing file cannot be read with the key (or without one).

        Notes
        -----
        Unlike the main database, a missing ledger is created even with a key
        set: an empty ledger is the correct state of a deployment that has
        never deleted anything, and the file holds no knowledge to lose.
        """
        if key_hex is not None and not await asyncio.to_thread(Path(path).exists):
            await asyncio.to_thread(create_encrypted_database, path, key_hex)
        conn = await connect_database(path, key_hex)
        try:
            await conn.execute("PRAGMA secure_delete = ON")
            await conn.execute("PRAGMA journal_mode = WAL")
            await conn.executescript(_LEDGER_SCHEMA)
            await conn.commit()
        except Exception as exc:
            # A file that opened but cannot be used (damaged, or a key that the
            # first read did not catch): close it rather than leave a worker
            # thread behind, and report it like any other unreadable file.
            await conn.close()
            raise DatabaseOpenError(f"{path}: the deletion ledger cannot be used.") from exc
        return cls(conn)

    async def close(self) -> None:
        """Close the ledger file."""
        await self._conn.close()

    async def record(self, entry: LedgerEntry) -> int:
        """Write an entry before its deletion runs.

        Parameters
        ----------
        entry
            The deletion about to be executed.

        Returns
        -------
        int
            The entry's ID, for `complete`.
        """
        async with connection_lock(self._conn):
            cursor = await self._conn.execute(
                """
                INSERT INTO deletions
                    (kind, reason, guild_id, user_id, mode, fact_id, fact_created_at,
                     requested_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    entry.kind.value,
                    entry.reason.value,
                    entry.guild_id,
                    entry.user_id,
                    entry.mode.value if entry.mode is not None else None,
                    entry.fact_id,
                    entry.fact_created_at,
                    utc_iso(entry.requested_at),
                ),
            )
            await self._conn.commit()
        entry_id = cursor.lastrowid
        assert entry_id is not None  # guaranteed by sqlite after a successful INSERT
        return entry_id

    async def complete(self, entry_id: int, counts: DeletionCounts, *, at: datetime) -> None:
        """Mark an entry's deletion as finished, with its row counts.

        Parameters
        ----------
        entry_id
            From `record`.
        counts
            What the deletion removed.
        at
            When it finished.

        Returns
        -------
        None
        """
        async with connection_lock(self._conn):
            await self._conn.execute(
                "UPDATE deletions SET completed_at = ?, counts = ? WHERE id = ?",
                (utc_iso(at), json.dumps(dict(sorted(counts.rows.items()))), entry_id),
            )
            await self._conn.commit()

    async def entries(self) -> list[LedgerEntry]:
        """Return every entry, oldest first.

        Returns
        -------
        list[LedgerEntry]
            All entries, finished or not.
        """
        async with (
            connection_lock(self._conn),
            self._conn.execute(
                """
                SELECT id, kind, reason, guild_id, user_id, mode, fact_id, fact_created_at,
                       requested_at, completed_at
                FROM deletions ORDER BY id
                """
            ) as cursor,
        ):
            rows = await cursor.fetchall()
        return [
            LedgerEntry(
                entry_id=row[0],
                kind=DeletionKind(row[1]),
                reason=DeletionReason(row[2]),
                guild_id=row[3],
                user_id=row[4],
                mode=MemberDeletionMode(row[5]) if row[5] is not None else None,
                fact_id=row[6],
                fact_created_at=row[7],
                requested_at=datetime.fromisoformat(row[8]),
                completed_at=row[9],
            )
            for row in rows
        ]

    async def count(self) -> int:
        """Return the number of entries.

        Returns
        -------
        int
            Entries in the ledger.
        """
        async with (
            connection_lock(self._conn),
            self._conn.execute("SELECT COUNT(*) FROM deletions") as cursor,
        ):
            row = await cursor.fetchone()
        assert row is not None
        return int(row[0])
