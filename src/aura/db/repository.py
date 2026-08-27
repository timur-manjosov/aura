"""Async data access for the knowledge model: facts, supersession, and links.

Every operation here runs under the shared per-connection lock from
aura.db.connection, which is also where the reasoning for that lock lives --
it is a rule for all writers on the connection, not just this module's.

**Links stopped being dormant scaffolding here.** fact_links has existed since
Phase 1b with zero call sites; the link phase wired it to a mod command and to
both answering triggers' retrieval. Three things about the link functions
changed in that wiring, and each was a real defect rather than a style
preference -- see each function's own docstring for the reasoning:

  * every link operation is now scoped by guild_id like every other read here,
    instead of deriving the guild from the facts it was handed;
  * linking requires both facts to be currently ACTIVE, so a link can never be
    created into a fact that is already retired;
  * the read side gained a batched, many-facts-at-once neighbour lookup and a
    supersession-chain resolver, because retrieval asks about several facts at
    once and must follow a link to whatever is current rather than to whatever
    the moderator happened to point at months ago.
"""
from __future__ import annotations

import asyncio
import logging
import sqlite3
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path

import aiosqlite

from aura.db.connection import connection_lock, utc_now_iso
from aura.db.models import Fact, FactStatus

logger = logging.getLogger(__name__)

_SCHEMA_PATH = Path(__file__).parent / "schema.sql"

# How many IDs one `WHERE id IN (...)` may carry. Well under SQLite's own
# SQLITE_MAX_VARIABLE_NUMBER (999 on builds predating 3.32, far higher since),
# chosen against the lower bound rather than the installed one so the limit
# cannot depend on which SQLite a deployment happens to link against.
_ID_QUERY_CHUNK_SIZE = 500

# The same bound for a statement that binds each ID TWICE -- get_linked_fact_ids
# has to ask about both ends of an undirected link in one query, so a chunk of
# _ID_QUERY_CHUNK_SIZE would bind 1000 parameters plus the guild and blow the
# very limit that constant was chosen against. Derived from it rather than
# written as a second literal, so the two cannot drift apart.
_LINK_QUERY_CHUNK_SIZE = _ID_QUERY_CHUNK_SIZE // 2

# How far resolve_active_successors will follow a supersession chain before
# giving up on it. Not a limit any real chain can reach: a fact corrected once
# a week for two years is a chain of ~100, and every real chain observed is 1-2
# links long. It exists because the walk is driven by data in a table a human
# with database access can hand-edit, and an unbounded walk over hand-editable
# data is an unbounded number of queries. A chain that exceeds this resolves to
# nothing (the link is dropped, loudly) rather than to a guess -- fail closed,
# the same direction every other refusal in this project takes.
_MAX_SUPERSESSION_HOPS = 100

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
    """Raised when link_facts or unlink_facts is called with the same fact ID twice.

    A fact is trivially related to itself, so a self-link carries no
    information a reader could ever use -- and the schema agrees: fact_links'
    `CHECK (fact_a_id < fact_b_id)` makes the row unrepresentable. Rejected
    here, before the database is touched, so the caller gets a sentence
    instead of an IntegrityError. unlink_facts raises it too rather than
    quietly reporting "nothing was unlinked": a self-unlink is the same
    caller mistake and deserves the same answer.
    """


class FactNotFoundError(RepositoryError):
    """Raised when a link operation names a fact that doesn't exist IN THAT GUILD.

    One error for "no such fact" and for "that fact belongs to another guild",
    deliberately -- the same isolation rule get_fact_by_id already states: a
    moderator in one guild must not be able to learn anything about another
    guild's facts by guessing numeric IDs, and two distinguishable errors
    would turn /aura-link into an ID oracle across guild boundaries.

    This replaces the CrossGuildLinkError that link_facts raised through Phase
    1b. That error was only reachable because link_facts derived the guild
    from the two facts it was handed instead of being told which guild the
    caller was acting in; scoping the operation by guild_id, like every other
    read in this module, makes a cross-guild pair indistinguishable from a
    nonexistent one -- which is the property that was wanted all along.
    """


class FactNotActiveError(RepositoryError):
    """Raised when link_facts names a fact that exists in the guild but is superseded.

    Linking is a statement about what is currently true together, so both
    ends must currently be true. A link INTO an already-retired fact would be
    dead the moment it was written -- retrieval would resolve it forward to
    whatever replaced it (see resolve_active_successors), which is exactly the
    fact the moderator should have named in the first place.

    Distinct from FactNotFoundError on purpose, and this leaks nothing: by the
    time this can be raised the fact is already known to be in the caller's own
    guild, so the extra detail is about a fact they can already read. It buys a
    genuinely useful error message -- /aura-link names the successor and tells
    the moderator to link that instead.

    Deliberately NOT enforced by unlink_facts: removing a link must keep
    working after either end has been retired, or a link created before a
    supersession could never be cleaned up.
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


async def link_facts(
    conn: aiosqlite.Connection, *, guild_id: int, fact_id_1: int, fact_id_2: int
) -> bool:
    """Link two active facts of one guild thematically. Returns whether a row was created.

    CLAUDE.md's fourth knowledge-model component, written: a moderator has
    decided these two DIFFERENT facts belong in one answer together, which is
    the one relationship no similarity search can derive on its own -- it is
    not "the same fact worded differently" (that is a variant) and not "this
    replaced that" (that is supersession).

    Undirected, and normalized to make that true in the data rather than only
    in the reader's head: the pair is sorted so the smaller ID is always
    fact_a_id, which is what fact_links' own `CHECK (fact_a_id < fact_b_id)`
    plus its two-column primary key turn into a hard one-row-per-pair
    guarantee. Calling this with the arguments swapped therefore hits the same
    row, and cannot produce a second one.

    Returns True if this call created the link and False if it already
    existed. The database operation is idempotent either way (INSERT OR
    IGNORE); the distinction is returned rather than swallowed because the
    caller is a moderator who typed two IDs and deserves to be told whether
    anything actually changed.

    Both facts must exist in `guild_id` and both must be ACTIVE. Raises
    SelfLinkError if the IDs are equal (before touching the database),
    FactNotFoundError if either is not a fact of this guild, and
    FactNotActiveError if either is superseded -- each checked under the same
    lock and in the same transaction as the insert, so a fact superseded
    concurrently cannot slip in between the check and the commit.
    """
    if fact_id_1 == fact_id_2:
        raise SelfLinkError(f"Cannot link fact {fact_id_1} to itself.")

    fact_a_id, fact_b_id = sorted((fact_id_1, fact_id_2))
    now = utc_now_iso()

    async with connection_lock(conn):
        try:
            async with conn.execute(
                "SELECT id, status FROM facts WHERE guild_id = ? AND id IN (?, ?)",
                (guild_id, fact_a_id, fact_b_id),
            ) as cursor:
                rows = await cursor.fetchall()

            status_by_id = {row[0]: FactStatus(row[1]) for row in rows}
            # Existence is settled for BOTH facts before activity is looked at
            # for either. Interleaving the two checks would make the error a
            # caller gets depend on which of their two IDs happens to be
            # numerically smaller -- so naming a superseded fact of their own
            # alongside a nonexistent one would report the supersession and
            # hide the typo. It also keeps this in step with /aura-link's
            # pre-flight checks, which run in the same two passes.
            for fact_id in (fact_a_id, fact_b_id):
                if fact_id not in status_by_id:
                    raise FactNotFoundError(
                        f"Fact {fact_id} does not exist in guild {guild_id}."
                    )
            for fact_id in (fact_a_id, fact_b_id):
                if status_by_id[fact_id] is not FactStatus.ACTIVE:
                    raise FactNotActiveError(
                        f"Fact {fact_id} in guild {guild_id} is superseded and cannot be linked."
                    )

            cursor = await conn.execute(
                "INSERT OR IGNORE INTO fact_links (fact_a_id, fact_b_id, created_at) "
                "VALUES (?, ?, ?)",
                (fact_a_id, fact_b_id, now),
            )
            created = cursor.rowcount == 1
        except BaseException:
            await conn.rollback()
            raise

        await conn.commit()

    return created


async def unlink_facts(
    conn: aiosqlite.Connection, *, guild_id: int, fact_id_1: int, fact_id_2: int
) -> bool:
    """Remove the link between two facts of one guild. Returns whether a row was deleted.

    The exact inverse of link_facts, including the same argument-order
    indifference (the pair is sorted the same way), and the same guild
    scoping: the EXISTS guards mean a moderator cannot delete another guild's
    link by naming its IDs, even though fact_links itself carries no guild
    column.

    Deliberately does NOT require either fact to still be active. A link is
    typically created between two active facts and then outlives one of them;
    refusing to remove it at that point would leave exactly the links a
    moderator most wants to clean up permanently unremovable. Removing a link
    can never make Aura state something false, only make it cite less, so the
    conservative direction here is the permissive one.

    Returns False -- not an error -- when there was no such link. "Unlink two
    facts that were not linked" is already the state the caller asked for, and
    the command surface says so plainly rather than treating it as a failure.
    Raises SelfLinkError if the two IDs are equal; see that class for why this
    is an error rather than a False.
    """
    if fact_id_1 == fact_id_2:
        raise SelfLinkError(f"Cannot unlink fact {fact_id_1} from itself.")

    fact_a_id, fact_b_id = sorted((fact_id_1, fact_id_2))

    async with connection_lock(conn):
        try:
            cursor = await conn.execute(
                """
                DELETE FROM fact_links
                WHERE fact_a_id = ? AND fact_b_id = ?
                  AND EXISTS (SELECT 1 FROM facts WHERE id = ? AND guild_id = ?)
                  AND EXISTS (SELECT 1 FROM facts WHERE id = ? AND guild_id = ?)
                """,
                (fact_a_id, fact_b_id, fact_a_id, guild_id, fact_b_id, guild_id),
            )
            removed = cursor.rowcount == 1
        except BaseException:
            await conn.rollback()
            raise

        await conn.commit()

    return removed


async def get_active_facts(conn: aiosqlite.Connection, guild_id: int) -> list[Fact]:
    """Return every currently-active fact for a guild."""
    async with connection_lock(conn):
        async with conn.execute(
            f"SELECT {_FACT_COLUMNS} FROM facts WHERE guild_id = ? AND status = ?",
            (guild_id, FactStatus.ACTIVE),
        ) as cursor:
            rows = await cursor.fetchall()
    return [_row_to_fact(row) for row in rows]


async def get_facts_created_between(
    conn: aiosqlite.Connection, *, guild_id: int, since: str, until: str
) -> list[Fact]:
    """Return the facts a guild gained in the half-open window (since, until], oldest first.

    ACTIVE facts only, and that filter carries a decision rather than being a
    copy of get_active_facts' habit: a fact created inside the window and
    already superseded before the window closed is not something the server
    "gained", it is a correction that happened too fast to be worth reporting as
    news. Leaving it out is what keeps the periodic digest a statement about
    what is true now (see aura.digest.builder, which makes the same choice on
    the supersession side for the same reason).

    HALF-OPEN, deliberately: `since` is the previous window's `until`, so a fact
    created at exactly that instant belongs to the window that already reported
    it. Closed at the far end so the caller's single `now` bounds the window and
    a fact written between the query and the run being recorded lands in the
    next digest instead of falling between the two.

    Both bounds are fixed-width UTC ISO-8601 strings (see
    aura.db.connection.utc_iso) and are compared as text in SQL, exactly as the
    proactive cooldown compares its own: that formatting exists precisely so
    lexicographic order is chronological order, and it keeps a hand-edited
    timestamp from turning a digest into a parse error.
    """
    async with connection_lock(conn):
        async with conn.execute(
            f"""
            SELECT {_FACT_COLUMNS} FROM facts
            WHERE guild_id = ? AND status = ? AND created_at > ? AND created_at <= ?
            ORDER BY created_at ASC, id ASC
            """,
            (guild_id, FactStatus.ACTIVE, since, until),
        ) as cursor:
            rows = await cursor.fetchall()
    return [_row_to_fact(row) for row in rows]


async def get_facts_superseded_between(
    conn: aiosqlite.Connection, *, guild_id: int, since: str, until: str
) -> list[Fact]:
    """Return the facts a guild retired in the half-open window (since, until], oldest first.

    The other half of "what changed", read off the supersession chain the
    knowledge model has carried since Phase 1b: a fact whose `superseded_at`
    falls in the window stopped being true during it, and `superseded_by_id`
    says what replaced it. Nothing is ever deleted, which is exactly why this
    question is answerable at all.

    Same half-open window and same string comparison as
    get_facts_created_between, for the same reasons.
    """
    async with connection_lock(conn):
        async with conn.execute(
            f"""
            SELECT {_FACT_COLUMNS} FROM facts
            WHERE guild_id = ? AND status = ? AND superseded_at > ? AND superseded_at <= ?
            ORDER BY superseded_at ASC, id ASC
            """,
            (guild_id, FactStatus.SUPERSEDED, since, until),
        ) as cursor:
            rows = await cursor.fetchall()
    return [_row_to_fact(row) for row in rows]


async def get_facts_by_ids(
    conn: aiosqlite.Connection, *, guild_id: int, fact_ids: Iterable[int]
) -> dict[int, Fact]:
    """Return the requested facts of one guild, keyed by ID; missing IDs are absent.

    Returns a mapping rather than a list because every caller so far is
    following references (a supersession chain's next link) and asks about one
    ID at a time after fetching -- handing back a list would make each of them
    rebuild this dict.

    Batched in chunks rather than issued one query per ID, per CLAUDE.md's
    Performance section, and chunked rather than sent as one enormous IN clause
    because SQLite caps how many bound parameters a statement may carry
    (SQLITE_MAX_VARIABLE_NUMBER, 999 on older builds). A caller walking a long
    chain must not turn into a query the database refuses to prepare.

    Guild-scoped, like every other read here: a chain that somehow pointed at
    another guild's fact yields nothing rather than crossing the boundary.
    """
    unique_ids = list(dict.fromkeys(fact_ids))
    facts: dict[int, Fact] = {}
    for start in range(0, len(unique_ids), _ID_QUERY_CHUNK_SIZE):
        chunk = unique_ids[start : start + _ID_QUERY_CHUNK_SIZE]
        placeholders = ", ".join("?" for _ in chunk)
        async with connection_lock(conn):
            async with conn.execute(
                f"SELECT {_FACT_COLUMNS} FROM facts "
                f"WHERE guild_id = ? AND id IN ({placeholders})",
                (guild_id, *chunk),
            ) as cursor:
                rows = await cursor.fetchall()
        for row in rows:
            fact = _row_to_fact(row)
            facts[fact.id] = fact
    return facts


async def get_linked_facts(
    conn: aiosqlite.Connection, *, guild_id: int, fact_id: int
) -> list[Fact]:
    """Return every fact of guild_id linked to fact_id, whatever its status, oldest ID first.

    Checks both sides of the undirected link, which is what makes "linked" a
    symmetric question rather than one that depends on which ID a moderator
    happened to type first.

    Deliberately unfiltered by status, exactly like get_variants_for_fact: this
    is the raw view of what a moderator actually linked, used by the command
    surface and by tests. Retrieval wants the very different question "what
    should be cited alongside this fact, resolved through any supersession
    that has happened since" -- that is expand_with_linked_facts, built on
    get_linked_fact_ids and resolve_active_successors below.

    Guild-scoped, and that is defense in depth rather than the only guard:
    link_facts already refuses to create a link whose ends sit in different
    guilds, so this filter can only ever matter if a link is written by
    something other than link_facts (a hand-edited database, a future writer).
    It costs one WHERE clause and removes the possibility entirely, matching
    the isolation every other read in this module already provides.
    """
    async with connection_lock(conn):
        async with conn.execute(
            f"""
            SELECT {_FACT_COLUMNS} FROM facts
            WHERE guild_id = ? AND id IN (
                SELECT fact_b_id FROM fact_links WHERE fact_a_id = ?
                UNION
                SELECT fact_a_id FROM fact_links WHERE fact_b_id = ?
            )
            ORDER BY id ASC
            """,
            (guild_id, fact_id, fact_id),
        ) as cursor:
            rows = await cursor.fetchall()
    return [_row_to_fact(row) for row in rows]


async def get_linked_fact_ids(
    conn: aiosqlite.Connection, *, guild_id: int, fact_ids: Iterable[int]
) -> dict[int, list[int]]:
    """Return each requested fact's linked neighbour IDs, keyed by the fact asked about.

    The batched counterpart to get_linked_facts, and the reason it exists is
    CLAUDE.md's Performance rule rather than taste: retrieval asks this about
    every citation candidate at once (up to SYNTHESIS_FACT_LIMIT of them), and
    a loop of single-fact queries would be N round trips on a path a user is
    waiting on. One query per chunk instead.

    Returns IDs, not Facts, because every caller immediately hands them to
    resolve_active_successors -- a link may point at a fact that has since been
    superseded, so the row this query finds is frequently NOT the row that
    should be cited, and materializing it would be work thrown away.

    Neighbours are returned whatever their status, for the same reason: it is
    resolution's job to decide what a link means now, not this query's. A fact
    with no links is absent from the mapping rather than present with an empty
    list, matching get_facts_by_ids' "missing IDs are absent" contract; each
    list is sorted ascending so a caller's ordering is decided by the data
    rather than by SQLite's row order.

    Guild-scoped on BOTH ends of every link, so a link that somehow spanned
    two guilds yields nothing rather than leaking one guild's fact into the
    other's answer.
    """
    unique_ids = list(dict.fromkeys(fact_ids))
    neighbours: dict[int, set[int]] = {}

    for start in range(0, len(unique_ids), _LINK_QUERY_CHUNK_SIZE):
        chunk = unique_ids[start : start + _LINK_QUERY_CHUNK_SIZE]
        placeholders = ", ".join("?" for _ in chunk)
        async with connection_lock(conn):
            async with conn.execute(
                f"""
                SELECT link.fact_a_id, link.fact_b_id
                FROM fact_links link
                JOIN facts fact_a ON fact_a.id = link.fact_a_id
                JOIN facts fact_b ON fact_b.id = link.fact_b_id
                WHERE fact_a.guild_id = ? AND fact_b.guild_id = ?
                  AND (
                    link.fact_a_id IN ({placeholders})
                    OR link.fact_b_id IN ({placeholders})
                  )
                """,
                (guild_id, guild_id, *chunk, *chunk),
            ) as cursor:
                rows = await cursor.fetchall()

        requested = set(chunk)
        for fact_a_id, fact_b_id in rows:
            # Both directions are checked rather than one: a single row can
            # match this chunk through either end, or through both at once
            # when two requested facts are linked to each other.
            if fact_a_id in requested:
                neighbours.setdefault(fact_a_id, set()).add(fact_b_id)
            if fact_b_id in requested:
                neighbours.setdefault(fact_b_id, set()).add(fact_a_id)

    return {fact_id: sorted(neighbour_ids) for fact_id, neighbour_ids in neighbours.items()}


async def resolve_active_successors(
    conn: aiosqlite.Connection, *, guild_id: int, fact_ids: Iterable[int]
) -> dict[int, Fact]:
    """Map each requested fact ID to the ACTIVE fact it resolves to, following supersession.

    A fact that is still active resolves to itself. A superseded one resolves
    to whatever `superseded_by_id` chains to, transitively, however many times
    it has been replaced -- so a link a moderator drew at a fact months ago
    still lands on what is true today rather than on a fact retrieval is
    forbidden to cite. This is the read-side counterpart of the `status` +
    successor chaining the knowledge model has carried since Phase 1b, and the
    reason a link never has to be rewritten when either end is superseded.

    An ID is ABSENT from the result -- never guessed at -- if it does not
    exist in guild_id, if its chain runs off the end (a superseded fact with
    no successor, or one whose successor belongs to another guild), if the
    chain cycles, or if it is longer than _MAX_SUPERSESSION_HOPS. Every one of
    those is broken data rather than a normal state, and the two that can only
    come from outside this module's own writes are logged. Fail closed: a link
    that resolves to nothing simply contributes no citation candidate, which
    costs an answer some context; resolving it to a fact that is not current
    would cost the answer its correctness.

    Batched breadth-first rather than one walk per ID: several links commonly
    point into the same chain (or at the same fact), so the chains are walked
    together, one query per hop for the whole set, and each fact is fetched at
    most once.
    """
    unique_ids = list(dict.fromkeys(fact_ids))
    if not unique_ids:
        return {}

    known: dict[int, Fact] = {}
    frontier = unique_ids
    for _ in range(_MAX_SUPERSESSION_HOPS):
        # Already-known IDs are dropped rather than re-fetched: each fact has
        # exactly one successor, so a fact reached twice was already expanded
        # the first time. This is also what makes a cycle terminate here
        # instead of looping until the hop limit.
        missing = [fact_id for fact_id in frontier if fact_id not in known]
        if not missing:
            break
        fetched = await get_facts_by_ids(conn, guild_id=guild_id, fact_ids=missing)
        known.update(fetched)
        frontier = [
            fact.superseded_by_id
            for fact in fetched.values()
            if fact.status is FactStatus.SUPERSEDED and fact.superseded_by_id is not None
        ]
        if not frontier:
            break
    else:
        logger.warning(
            "Supersession chains in guild %s exceed %d hops; the links pointing into "
            "them resolve to nothing rather than to an intermediate state",
            guild_id,
            _MAX_SUPERSESSION_HOPS,
        )

    resolved: dict[int, Fact] = {}
    for origin_id in unique_ids:
        visited: set[int] = set()
        current = known.get(origin_id)
        while current is not None and current.status is not FactStatus.ACTIVE:
            if current.id in visited:
                logger.warning(
                    "Supersession chain from fact %s in guild %s cycles at fact %s; "
                    "resolving it to nothing",
                    origin_id,
                    guild_id,
                    current.id,
                )
                current = None
                break
            visited.add(current.id)
            successor_id = current.superseded_by_id
            current = known.get(successor_id) if successor_id is not None else None
        if current is not None:
            resolved[origin_id] = current

    return resolved
