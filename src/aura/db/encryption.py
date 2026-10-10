"""Opening Aura's SQLite files, plaintext or encrypted at rest with SQLCipher (P7a, R1).

With `DATABASE_ENCRYPTION_KEY` unset, every connection is exactly the
`aiosqlite.connect` of before. With it set, the database file (and its WAL,
and every backup made through this module) is encrypted by SQLCipher, an
SQLite build that encrypts each page before it reaches the disk. Reads see
plain rows, so retrieval -- local embeddings, word matching -- is unchanged.

What this protects, stated so nobody reads more into it: the files and every
copy of them (backups, a file moved off the server, a disk the host replaces).
It does not protect against someone logged into the running server, who can
read the key from `.env` like the bot does. The key is a raw 256-bit key given
as 64 hex characters, so no key-derivation step is involved and the PRAGMA that
sets it can be built from a strictly validated string only.

Invariants:

- A key that does not open the file, or a plaintext file opened with a key, or
  an encrypted file opened without one, raises `DatabaseOpenError` before any
  statement runs. Nothing is ever created or overwritten in that case.
- With a key set, a MISSING file is refused too: the encrypted database is
  created once by `export_encrypted`, never silently as an empty file, so a
  wrong path or a lost file can never look like a fresh, empty server.
- Nothing here logs or prints content; reports hold counts, a schema hash and
  integrity results.

`sqlcipher3` is imported only when a key is in use, so a plaintext deployment
does not depend on it. Imports nothing from `aura` but this package's
connection helpers.
"""

from __future__ import annotations

import hashlib
import os
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Final, Protocol

import aiosqlite

# 32 bytes as lower- or upper-case hex. Anything else is refused before it can
# reach a PRAGMA, which takes no bound parameters.
DATABASE_KEY_PATTERN: Final = re.compile(r"\A[0-9a-fA-F]{64}\Z")

# The setting the maintenance commands read the key from, and the one a key
# rotation reads the new key from.
DATABASE_KEY_ENV: Final = "DATABASE_ENCRYPTION_KEY"
NEW_DATABASE_KEY_ENV: Final = "DATABASE_ENCRYPTION_NEW_KEY"

_AIOSQLITE_CHUNK_SIZE: Final = 64


class DatabaseOpenError(Exception):
    """A database file could not be opened as configured; nothing was changed."""


class _DbApiConnection(Protocol):
    """The DB-API surface both sqlite3 and sqlcipher3 connections share and this module uses."""

    def execute(self, sql: str, parameters: Any = ..., /) -> Any: ...

    def commit(self) -> None: ...

    def close(self) -> None: ...

    def backup(self, target: Any, /) -> None: ...


def validate_database_key(raw: str) -> str:
    """Return a configured key in canonical lower-case hex, or raise.

    Parameters
    ----------
    raw
        The setting's value, surrounding whitespace allowed.

    Returns
    -------
    str
        64 lower-case hex characters.

    Raises
    ------
    ValueError
        If the value is not exactly 64 hex characters. The message never
        contains the value.
    """
    stripped = raw.strip()
    if not DATABASE_KEY_PATTERN.match(stripped):
        raise ValueError(
            f"{DATABASE_KEY_ENV} must be exactly 64 hex characters (generate one with "
            '`python -c "import secrets; print(secrets.token_hex(32))"`).'
        )
    return stripped.lower()


def _sqlcipher() -> ModuleType:
    """Import the SQLCipher binding, or explain what is missing."""
    try:
        from sqlcipher3 import dbapi2
    except ImportError as exc:  # pragma: no cover - the image always ships it
        raise DatabaseOpenError(
            "DATABASE_ENCRYPTION_KEY is set but the sqlcipher3 package is not installed."
        ) from exc
    return dbapi2


def _key_pragma(key_hex: str) -> str:
    """Return the PRAGMA value for a raw key; `key_hex` must already be validated."""
    assert DATABASE_KEY_PATTERN.match(key_hex)
    return f"\"x'{key_hex}'\""


def _verify_readable(conn: _DbApiConnection, path: str, *, encrypted: bool) -> None:
    """Read the schema once so a wrong key or a wrong file type fails here, not later."""
    try:
        conn.execute("SELECT count(*) FROM sqlite_master").fetchone()
    except Exception as exc:
        conn.close()
        if encrypted:
            raise DatabaseOpenError(
                f"{path}: the configured key does not open this file (wrong key, or the "
                "file is not encrypted). Nothing was changed."
            ) from exc
        raise DatabaseOpenError(
            f"{path}: not readable as a plain SQLite database (it may be encrypted -- set "
            "DATABASE_ENCRYPTION_KEY -- or damaged). Nothing was changed."
        ) from exc


def open_sync(
    path: str | Path,
    key_hex: str | None,
    *,
    must_exist: bool = True,
    check_same_thread: bool = True,
) -> _DbApiConnection:
    """Open one database file synchronously, verified readable.

    Parameters
    ----------
    path
        The file.
    key_hex
        A validated key (see `validate_database_key`), or None for plaintext.
    must_exist
        Refuse a missing file instead of creating an empty one.
    check_same_thread
        Passed to the driver; False for a connection aiosqlite's worker thread
        opens (it is used from that thread only).

    Returns
    -------
    sqlite3.Connection or sqlcipher3 connection
        Open and verified.

    Raises
    ------
    DatabaseOpenError
        If the file is missing while `must_exist`, or cannot be read with the
        given key (or without one).
    """
    location = str(path)
    if must_exist and not Path(location).exists():
        raise DatabaseOpenError(f"{location}: no such database file. Nothing was created.")
    if key_hex is None:
        plain = sqlite3.connect(location, check_same_thread=check_same_thread)
        _verify_readable(plain, location, encrypted=False)
        return plain
    dbapi = _sqlcipher()
    encrypted = dbapi.connect(location, check_same_thread=check_same_thread)
    encrypted.execute(f"PRAGMA key = {_key_pragma(key_hex)}")
    _verify_readable(encrypted, location, encrypted=True)
    return encrypted  # type: ignore[no-any-return]


def connect_database(path: str, key_hex: str | None) -> aiosqlite.Connection:
    """Return an awaitable aiosqlite connection to Aura's database, encrypted or not.

    Parameters
    ----------
    path
        The database file.
    key_hex
        A validated key, or None for the plaintext behaviour of before.

    Returns
    -------
    aiosqlite.Connection
        Not yet started: `await` it, exactly like `aiosqlite.connect(...)`.

    Raises
    ------
    DatabaseOpenError
        When awaited, under the conditions `open_sync` names. Without a key a
        missing file is created, as before P7a; with a key it is refused.

    Notes
    -----
    aiosqlite runs every statement on one worker thread; the connector below
    runs on that thread, so the connection never crosses threads.
    """
    if key_hex is None:

        def plain_connector() -> sqlite3.Connection:
            return open_sync(path, None, must_exist=False)  # type: ignore[return-value]

        return aiosqlite.Connection(plain_connector, _AIOSQLITE_CHUNK_SIZE)

    def encrypted_connector() -> sqlite3.Connection:
        return open_sync(path, key_hex, must_exist=True)  # type: ignore[return-value]

    return aiosqlite.Connection(encrypted_connector, _AIOSQLITE_CHUNK_SIZE)


def create_encrypted_database(path: str | Path, key_hex: str) -> None:
    """Create a new, empty encrypted database file.

    Parameters
    ----------
    path
        The new file; it must not exist.
    key_hex
        A validated key.

    Returns
    -------
    None

    Raises
    ------
    DatabaseOpenError
        If the file exists, or cannot be read back with the key.

    Notes
    -----
    For files that may legitimately start empty, such as the deletion ledger.
    The main database is never created this way (see the module docstring).
    """
    _refuse_existing(path)
    conn: _DbApiConnection = _sqlcipher().connect(str(path))
    try:
        conn.execute(f"PRAGMA key = {_key_pragma(key_hex)}")
        # Writing the header is what makes the file exist with this key.
        conn.execute("PRAGMA user_version = 0")
        conn.execute("CREATE TABLE IF NOT EXISTS _created (id INTEGER)")
        conn.execute("DROP TABLE _created")
        conn.commit()
    finally:
        conn.close()
    _restrict_permissions(path)
    open_sync(path, key_hex, must_exist=True).close()


def is_integrity_error(error: BaseException) -> bool:
    """Report whether an exception is a constraint violation, from either driver.

    Parameters
    ----------
    error
        Any exception raised by a statement.

    Returns
    -------
    bool
        True for sqlite3's IntegrityError and for SQLCipher's, which is a
        different class with the same meaning.
    """
    if isinstance(error, sqlite3.IntegrityError):
        return True
    return type(error).__name__ == "IntegrityError" and type(error).__module__.startswith(
        "sqlcipher3"
    )


@dataclass(frozen=True)
class DatabaseReport:
    """What a maintenance step checked about one database file, without any content.

    Attributes
    ----------
    integrity
        `PRAGMA integrity_check`'s first line ("ok" when sound).
    cipher_integrity
        For an encrypted file, "ok" when every page's authentication code
        verified; "plaintext" for an unencrypted file.
    schema_hash
        First 16 hex characters of the SHA-256 of every schema statement,
        ordered -- the same fingerprint the deploy reports use.
    table_counts
        Rows per table, by name.
    """

    integrity: str
    cipher_integrity: str
    schema_hash: str
    table_counts: dict[str, int]


def _report(conn: _DbApiConnection, *, encrypted: bool) -> DatabaseReport:
    integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
    if encrypted:
        problems = conn.execute("PRAGMA cipher_integrity_check").fetchall()
        cipher_integrity = "ok" if not problems else f"{len(problems)} problem(s)"
    else:
        cipher_integrity = "plaintext"
    schema = "\n".join(
        row[0]
        for row in conn.execute(
            "SELECT sql FROM sqlite_master WHERE sql IS NOT NULL ORDER BY type, name"
        ).fetchall()
    )
    tables = [
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name"
        ).fetchall()
    ]
    counts = {
        table: conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0] for table in tables
    }
    return DatabaseReport(
        integrity=integrity,
        cipher_integrity=cipher_integrity,
        schema_hash=hashlib.sha256(schema.encode("utf-8")).hexdigest()[:16],
        table_counts=counts,
    )


def inspect_database(path: str | Path, key_hex: str | None) -> DatabaseReport:
    """Open a file read-only in effect and report its integrity, schema and row counts.

    Parameters
    ----------
    path
        The file; must exist.
    key_hex
        Its key, or None for a plaintext file.

    Returns
    -------
    DatabaseReport
        Counts and checks only.

    Raises
    ------
    DatabaseOpenError
        If the file is missing or the key does not fit.
    """
    conn = open_sync(path, key_hex, must_exist=True)
    try:
        return _report(conn, encrypted=key_hex is not None)
    finally:
        conn.close()


def _refuse_existing(target: str | Path) -> None:
    for suffix in ("", "-wal", "-shm", "-journal"):
        if Path(f"{target}{suffix}").exists():
            raise DatabaseOpenError(f"{target}{suffix} already exists; refusing to overwrite it.")


def _restrict_permissions(target: str | Path) -> None:
    """Make a written database file readable by its owner only."""
    os.chmod(target, 0o600)


def _export(
    source: str | Path,
    target: str | Path,
    *,
    source_key: str | None,
    target_key: str | None,
) -> DatabaseReport:
    """Copy every table, index and row of `source` into a new file with another key."""
    _refuse_existing(target)
    if not Path(source).exists():
        raise DatabaseOpenError(f"{source}: no such database file. Nothing was created.")
    # sqlcipher_export runs on a SQLCipher connection even for a plaintext
    # source: SQLCipher reads a plaintext file when no key is set.
    conn: _DbApiConnection = _sqlcipher().connect(str(source))
    if source_key is not None:
        conn.execute(f"PRAGMA key = {_key_pragma(source_key)}")
    _verify_readable(conn, str(source), encrypted=source_key is not None)
    before = _report(conn, encrypted=source_key is not None)
    target_pragma = _key_pragma(target_key) if target_key is not None else "''"
    try:
        # The attached file's name is a bound parameter; its key cannot be,
        # and is a validated hex string or the empty literal.
        conn.execute(f"ATTACH DATABASE ? AS export_target KEY {target_pragma}", (str(target),))
        conn.execute("SELECT sqlcipher_export('export_target')")
        conn.execute("DETACH DATABASE export_target")
    finally:
        conn.close()
    _restrict_permissions(target)
    after = inspect_database(target, target_key)
    if after.table_counts != before.table_counts or after.schema_hash != before.schema_hash:
        raise DatabaseOpenError(
            f"{target}: the copy does not match its source (schema or row counts differ); "
            "the source is unchanged, delete the copy."
        )
    if after.integrity != "ok" or after.cipher_integrity not in {"ok", "plaintext"}:
        raise DatabaseOpenError(f"{target}: the copy failed its integrity check.")
    return after


def export_encrypted(source: str | Path, target: str | Path, key_hex: str) -> DatabaseReport:
    """Write an encrypted copy of a plaintext database and verify it against the source.

    Parameters
    ----------
    source
        The plaintext file. Opened, never changed.
    target
        The new encrypted file. Must not exist (nor its -wal/-shm/-journal).
    key_hex
        The validated key for the copy.

    Returns
    -------
    DatabaseReport
        The copy's report; its schema hash and row counts equal the source's.

    Raises
    ------
    DatabaseOpenError
        If the target exists, the source is unreadable, or the copy does not
        match the source.
    """
    return _export(source, target, source_key=None, target_key=key_hex)


def export_plaintext(source: str | Path, target: str | Path, key_hex: str) -> DatabaseReport:
    """Write a plaintext copy of an encrypted database -- the rollback direction.

    Parameters
    ----------
    source
        The encrypted file. Opened, never changed.
    target
        The new plaintext file; must not exist.
    key_hex
        The source's key.

    Returns
    -------
    DatabaseReport
        The copy's report.

    Raises
    ------
    DatabaseOpenError
        As for `export_encrypted`.
    """
    return _export(source, target, source_key=key_hex, target_key=None)


def backup_database(source: str | Path, target: str | Path, key_hex: str | None) -> DatabaseReport:
    """Take a consistent online backup of a live database, encrypted with the same key.

    Parameters
    ----------
    source
        The live file; may be in use by the bot (SQLite's backup API copies a
        consistent snapshot).
    target
        The new backup file; must not exist.
    key_hex
        The database's key, or None for plaintext. The backup gets the same.

    Returns
    -------
    DatabaseReport
        The backup's report, read back from the written file -- a backup that
        cannot be read back with the key raises instead of returning.

    Raises
    ------
    DatabaseOpenError
        If the target exists or the backup cannot be read back.
    """
    _refuse_existing(target)
    source_conn = open_sync(source, key_hex, must_exist=True)
    try:
        if key_hex is None:
            target_conn: _DbApiConnection = sqlite3.connect(str(target))
        else:
            target_conn = _sqlcipher().connect(str(target))
            target_conn.execute(f"PRAGMA key = {_key_pragma(key_hex)}")
        try:
            source_conn.backup(target_conn)
        finally:
            target_conn.close()
    finally:
        source_conn.close()
    _restrict_permissions(target)
    return inspect_database(target, key_hex)


def rekey_database(path: str | Path, old_key_hex: str, new_key_hex: str) -> DatabaseReport:
    """Re-encrypt a database file under a new key, in place; the bot must be stopped.

    Parameters
    ----------
    path
        The encrypted file.
    old_key_hex, new_key_hex
        Its current key and the new one; both validated, and different.

    Returns
    -------
    DatabaseReport
        The file's report, read back with the NEW key.

    Raises
    ------
    DatabaseOpenError
        If the old key does not open the file, the keys are equal, or the file
        cannot be read back with the new key.
    """
    if old_key_hex == new_key_hex:
        raise DatabaseOpenError("The new key equals the old one; nothing to rotate.")
    conn = open_sync(path, old_key_hex, must_exist=True)
    try:
        # rekey cannot run with a WAL that holds pages; fold it in first.
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        conn.execute("PRAGMA journal_mode = DELETE")
        conn.execute(f"PRAGMA rekey = {_key_pragma(new_key_hex)}")
    finally:
        conn.close()
    report = inspect_database(path, new_key_hex)
    reopened = open_sync(path, new_key_hex, must_exist=True)
    try:
        reopened.execute("PRAGMA journal_mode = WAL")
    finally:
        reopened.close()
    return report
