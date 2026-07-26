"""The isolated scratch database, and the guard that makes it impossible to
confuse with production data.

Synthetic guilds, synthetic facts and synthetic embeddings must never touch
`data/aura.db`. "Must never" is not a convention here -- a convention is a
comment someone can typo past. Four independent layers have to agree before a
single byte is written, and each one alone is sufficient to stop the accident:

1. **The path must name itself.** The database's own filename has to contain
   the token "synthetic". A mistyped `--db data/aura.db` fails on the filename
   before anything opens.
2. **Known production paths are refused outright.** The packaged default, the
   `DATABASE_PATH` environment variable, and whatever `.env` sets are all
   resolved and rejected, as is anything inside their directory.
3. **An existing file must carry the marker table.** A scratch database
   announces itself in its own schema (`synthetic_corpus_marker`). Production's
   schema does not create that table and never will, so pointing this tool at a
   real database -- under any filename, through any symlink, from any working
   directory -- fails on open.
4. **Writes go through this module only.** Nothing else in the tooling opens a
   connection, so there is one door and the guard is on it.

Layer 3 is the load-bearing one, because it is the only layer that inspects the
*file* rather than the path leading to it. Layers 1 and 2 exist so the failure
happens early and legibly, in the common case where someone simply passed the
wrong path.

The schema itself is the real `aura.db.repository.init_schema`, not a copy: the
whole point of this corpus is to run the shipped Stage 2 retrieval against it,
and retrieval reads through `get_active_facts`, which reads the production
tables. A hand-written stand-in schema here would mean measuring something
subtly different from what ships.
"""
from __future__ import annotations

import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

import aiosqlite
from dotenv import dotenv_values

from aura.config import Settings
from aura.db.repository import init_schema

# The filename token every scratch database must carry. Checked against the
# file's own name, not the directory, so a "synthetic" directory holding
# `aura.db` still fails.
REQUIRED_PATH_TOKEN = "synthetic"

# The table whose presence marks a database as scratch. Production's schema.sql
# does not contain it, so this is a property of the file rather than of the
# path that reached it.
MARKER_TABLE = "synthetic_corpus_marker"

_MARKER_SQL = f"""
CREATE TABLE IF NOT EXISTS {MARKER_TABLE} (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    created_at TEXT NOT NULL,
    note TEXT NOT NULL
)
"""

# Tables that exist ONLY in a scratch database. Two things live here that the
# production schema has no place for: the corpus's own messages (Aura stores no
# message text at all -- a fact references its origin by ID, per CLAUDE.md's
# knowledge model), and the mapping from a generated fact's stable corpus key
# back to the database row it became, which is what lets the simulator score a
# Stage 2 result against the fact it was generated for.
#
# Their presence is a second, independent reason a production database could
# never be mistaken for one of these, and vice versa.
_SCRATCH_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS synthetic_fact_keys (
    fact_id INTEGER PRIMARY KEY,
    guild_key TEXT NOT NULL,
    fact_key TEXT NOT NULL,
    contradicts_key TEXT,
    UNIQUE (guild_key, fact_key)
);

CREATE TABLE IF NOT EXISTS synthetic_messages (
    key TEXT PRIMARY KEY,
    guild_key TEXT NOT NULL,
    category TEXT NOT NULL,
    locale TEXT NOT NULL,
    content TEXT NOT NULL,
    target_fact_keys TEXT NOT NULL,
    adversarial_kind TEXT,
    rationale TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_synthetic_messages_guild
    ON synthetic_messages(guild_key, category);
"""

_MARKER_NOTE = (
    "Phase 2b-2 synthetic scenario corpus. Every guild, fact and message in this "
    "file is machine-generated evaluation data and must never be treated as real "
    "server knowledge. See scripts/synthetic_corpus/."
)

DEFAULT_SCRATCH_PATH = Path("reports") / "synthetic-corpus" / "synthetic-corpus.db"


class ScratchDatabaseSafetyError(RuntimeError):
    """Raised when a database path or file could be, or is, production data.

    Its own exception type rather than ValueError so a caller can never catch
    this by accident while meaning to catch something else -- and so the tests
    that verify the guard are asserting on the guard, not on a message string.
    """


def _production_database_paths() -> set[Path]:
    """Every path this deployment might mean by "the real database".

    Three independent sources, because a deployment can point the bot at its
    database through any of them and this guard has to refuse all of them, not
    just the one that happens to be set right now:

    * the packaged default in `Settings` (read from the field definition rather
      than by constructing Settings, which would demand a Discord token this
      tool has no business needing),
    * the `DATABASE_PATH` process environment variable,
    * `DATABASE_PATH` in the repository's `.env`.
    """
    candidates: list[str] = []

    field = Settings.model_fields.get("database_path")
    if field is not None and isinstance(field.default, str):
        candidates.append(field.default)

    from_environment = os.environ.get("DATABASE_PATH")
    if from_environment:
        candidates.append(from_environment)

    try:
        from_dotenv = dotenv_values(".env").get("DATABASE_PATH")
    except OSError:
        from_dotenv = None
    if from_dotenv:
        candidates.append(from_dotenv)

    return {Path(candidate).expanduser().resolve() for candidate in candidates if candidate}


def assert_safe_scratch_path(path: Path) -> Path:
    """Resolve `path` and refuse it if it could be production data.

    Returns the resolved path. Raises ScratchDatabaseSafetyError -- loudly,
    never a warning and never a silent fallback to a safe default -- if the
    filename does not identify itself as synthetic, or if it resolves to (or
    sits beside) any path this deployment might mean by "the real database".

    Deliberately checks the *resolved* path, so `../data/aura.db`, a symlink
    into `data/`, and an absolute path all collapse to the same comparison.
    """
    resolved = Path(path).expanduser().resolve()

    if REQUIRED_PATH_TOKEN not in resolved.name.lower():
        raise ScratchDatabaseSafetyError(
            f"refusing to use {resolved} as a scratch database: its filename must "
            f"contain {REQUIRED_PATH_TOKEN!r} so it can never be mistaken for real "
            "data at a glance"
        )

    for production in _production_database_paths():
        if resolved == production:
            raise ScratchDatabaseSafetyError(
                f"refusing to use {resolved}: it is this deployment's production database"
            )
        if resolved.parent == production.parent:
            raise ScratchDatabaseSafetyError(
                f"refusing to use {resolved}: it sits in {production.parent}, the "
                "directory holding this deployment's production database"
            )

    return resolved


async def _table_exists(conn: aiosqlite.Connection, table: str) -> bool:
    """Whether `table` exists in the connected database."""
    async with conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (table,)
    ) as cursor:
        return await cursor.fetchone() is not None


async def _assert_marked_or_mark(conn: aiosqlite.Connection, *, file_existed: bool) -> None:
    """Ensure the open database is a scratch database, marking a fresh one.

    An already-existing file must already carry the marker. This is the layer
    that catches a production database reached under a filename that happens to
    satisfy every path check -- a copy, a symlink, a rename -- because it asks
    the file itself rather than the path.
    """
    if file_existed and not await _table_exists(conn, MARKER_TABLE):
        raise ScratchDatabaseSafetyError(
            "refusing to write to an existing database that carries no "
            f"{MARKER_TABLE!r} table. Only databases created by this tool are "
            "safe to write synthetic data into; this one was created by something "
            "else, which means it may be real data."
        )

    await conn.execute(_MARKER_SQL)
    await conn.execute(
        f"INSERT OR IGNORE INTO {MARKER_TABLE} (id, created_at, note) "
        "VALUES (1, datetime('now'), ?)",
        (_MARKER_NOTE,),
    )
    await conn.executescript(_SCRATCH_TABLES_SQL)
    await conn.commit()


async def assert_scratch_destination_usable(path: Path) -> Path:
    """Run every safety layer against `path` without creating or writing anything.

    Exists so a caller that is about to spend money can find out FIRST whether
    its output has somewhere legitimate to go. `open_scratch_database` performs
    the same checks, but it runs at the end of a generation run -- and a guard
    that fires only after the budget is gone tells an operator something they
    can no longer act on.

    Creates nothing: an absent file is fine (it will be created later, marked),
    and an existing one is opened read-only for the marker check.
    """
    resolved = assert_safe_scratch_path(path)
    if not resolved.exists():
        return resolved

    async with aiosqlite.connect(f"file:{resolved}?mode=ro", uri=True) as probe:
        if not await _table_exists(probe, MARKER_TABLE):
            raise ScratchDatabaseSafetyError(
                f"refusing to use {resolved}: it exists but carries no "
                f"{MARKER_TABLE!r} table, so it was not created by this tool and "
                "may be real data."
            )
    return resolved


@asynccontextmanager
async def open_scratch_database(
    path: Path = DEFAULT_SCRATCH_PATH, *, reset: bool = False
) -> AsyncGenerator[aiosqlite.Connection]:
    """Open (creating if needed) the isolated scratch database at `path`.

    Every layer of the guard runs before the production schema is initialised,
    so a refused path never gets so much as an empty file created for it -- with
    one deliberate exception noted below.

    `reset=True` deletes an existing scratch database first, so a regenerated
    corpus never inherits rows from an older one. It only ever deletes a file
    that has already passed every safety layer, which is why the marker is
    checked (by opening the existing file) before the delete rather than after.
    """
    resolved = assert_safe_scratch_path(path)

    if resolved.exists():
        # Verify the existing file is ours BEFORE deciding to delete or write to
        # it. Opening for this check creates nothing: the file already exists.
        async with aiosqlite.connect(resolved) as probe:
            if not await _table_exists(probe, MARKER_TABLE):
                raise ScratchDatabaseSafetyError(
                    f"refusing to touch {resolved}: it exists but carries no "
                    f"{MARKER_TABLE!r} table, so it was not created by this tool "
                    "and may be real data."
                )
        if reset:
            resolved.unlink()
            # WAL sidecars would otherwise survive the delete and resurrect
            # pages of the old corpus into the new one.
            for suffix in ("-wal", "-shm"):
                sidecar = resolved.with_name(resolved.name + suffix)
                if sidecar.exists():
                    sidecar.unlink()

    file_existed = resolved.exists()
    resolved.parent.mkdir(parents=True, exist_ok=True)

    async with aiosqlite.connect(resolved) as conn:
        await _assert_marked_or_mark(conn, file_existed=file_existed)
        await init_schema(conn)
        yield conn
