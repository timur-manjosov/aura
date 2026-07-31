"""Phase 3a-2's staging area: fact candidates the distillation model wrote, before
any human has confirmed them.

NOT part of the knowledge model, and structured so that stays obviously true --
the same standing this module's Phase 2 siblings have. A sentence a model
proposed is not one of CLAUDE.md's four components; it becomes one only when a
moderator says so, at which point it is written into `facts` through the
ordinary insert path and this table keeps only a pointer to what it became.

**Why a separate table rather than a third `facts.status` value** is argued at
length in schema.sql, above the `pending_facts` definition, because the choice
is a schema decision before it is a code one. The short version: every read
path over `facts` filters on status = 'active' and nothing else, so a
'pending' status would turn each of them into a place where one forgotten
predicate leaks an unconfirmed, machine-written sentence into a cited public
answer. A separate table cannot be read by accident.

**The confirmation race is the interesting part of this module.** Two
moderators pressing Confirm on the same candidate at the same moment -- or one
confirming while another discards -- must produce exactly one outcome and at
most one fact. That is enforced the same way supersession is
(aura.db.repository): a guarded `UPDATE ... WHERE status = 'pending'` whose
rowcount decides whether the caller won, inside the same transaction as the
fact insert, so a loser's insert is rolled back rather than left behind as a
duplicate active fact.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime
from enum import StrEnum

import aiosqlite
from pydantic import BaseModel

from aura.db.connection import connection_lock, utc_now_iso
from aura.db.models import Fact
from aura.db.repository import RepositoryError, insert_fact_within_transaction

_PENDING_COLUMNS = (
    "id, guild_id, channel_id, message_id, content, embedding, category, status, "
    "similar_fact_id, similar_fact_score, relationship, relationship_reasoning, "
    "confirmed_fact_id, created_at, resolved_at, resolved_by_id"
)

# Phase 3a-3's two additive columns, named here so the migration below and the
# CREATE TABLE in schema.sql cannot drift apart silently.
_RELATIONSHIP_COLUMN = "relationship"
_RELATIONSHIP_REASONING_COLUMN = "relationship_reasoning"


class FactCategory(StrEnum):
    """What kind of thing a distilled candidate asserts, as judged by the model.

    The closed vocabulary the distillation prompt is given (see
    aura.extraction.distiller) and the CHECK constraint on `pending_facts`
    enforces, defined here rather than in the extraction package for the same
    reason GateVerdict lives in aura.db.proactive_signals rather than in
    aura.proactive: the layer that persists a value owns its closed set, and
    the layer that produces it imports that.

    MILESTONE is the newest and the one that needs a definition rather than
    just a name: an achievement or threshold the server actually reached, and
    only when the message carries a concrete, checkable component -- a number,
    a named participant, a specific event. "We just passed 500 members" is a
    milestone; "congrats!!" beside it is not a fact at all and must not be
    extracted. reports/phase-3a-1b.txt Section 3 found that distinction to be
    the largest single source of label disagreement in the calibration corpus,
    and identified it as one embedding geometry cannot make -- which is why it
    is made here, in a prompt read by a model, instead of by a threshold.
    """

    ANNOUNCEMENT = "announcement"
    RULE = "rule"
    DECISION = "decision"
    EVENT = "event"
    STATUS_CHANGE = "status_change"
    MILESTONE = "milestone"


class SupersessionRelationship(StrEnum):
    """What a dedup hit between a candidate and an existing active fact MEANS.

    The closed vocabulary of Phase 3a-3's judgment call (see
    aura.extraction.supersession), living here beside FactCategory for the same
    reason that one does: the layer that persists a value owns its closed set,
    and the layer that produces it imports that.

    Every value is a PROPOSAL shown to a moderator, and none of them causes
    anything to happen on its own -- SUPERSESSION in particular does not
    supersede. The only code path that retires a fact is /aura-supersede, run by
    a human, and that stays true no matter what this field says.

      SUPERSESSION   the candidate is a genuine, later successor on the same
                     specific detail, and the moderator may want to retire the
                     predecessor after confirming it.
      COMPLEMENTARY  both facts are true at once about the same subject; no
                     replacement is called for, whether the candidate refines
                     the predecessor or simply states something else about it.
      CONTRADICTION  the two cannot both hold, and nothing in either wording
                     says which is current. The one value that must never be
                     resolved automatically -- it exists to escalate, and the
                     bake-off that chose the model for this call chose it
                     precisely on its willingness to answer this instead of
                     guessing (reports/supersession-model-bakeoff.txt).
      INDEPENDENT    the embedding hit was a false positive: similar wording,
                     different subject. Nothing should happen to either fact.
    """

    SUPERSESSION = "supersession"
    COMPLEMENTARY = "complementary"
    CONTRADICTION = "contradiction"
    INDEPENDENT = "independent"


class PendingFactStatus(StrEnum):
    """Where a candidate stands in review.

    Resolved candidates are kept rather than deleted, the same way a superseded
    fact is kept: "the model proposed this and a moderator rejected it" is the
    only evidence a later phase has about whether extraction earns its cost,
    and it is unrecoverable once thrown away.
    """

    PENDING = "pending"
    CONFIRMED = "confirmed"
    DISCARDED = "discarded"


class PendingFact(BaseModel):
    """One staged candidate, as read back from the database.

    Carries the distilled sentence rather than the source message: the source
    is referenced by guild/channel/message ID, which resolves to a Discord
    permalink exactly as a real Fact's origin does, so raw text is never
    duplicated into a durable row.

    similar_fact_id/similar_fact_score are the advisory dedup hint (see
    aura.extraction.pipeline) and decide nothing on their own.
    relationship/relationship_reasoning are Phase 3a-3's judgment of what that
    hit means, and decide nothing either -- both are shown to the reviewing
    moderator, and both are None whenever the judgment never ran (no dedup hit,
    the daily cap refused the call, or the call failed).
    confirmed_fact_id is the real fact a confirmation produced, so the trail
    from an automatically created fact back to the candidate and the message
    behind it stays walkable.
    """

    id: int
    guild_id: int
    channel_id: int
    message_id: int
    content: str
    embedding: bytes
    category: FactCategory
    status: PendingFactStatus
    similar_fact_id: int | None = None
    similar_fact_score: float | None = None
    relationship: SupersessionRelationship | None = None
    relationship_reasoning: str | None = None
    confirmed_fact_id: int | None = None
    created_at: datetime
    resolved_at: datetime | None = None
    resolved_by_id: int | None = None


class PendingFactNotFoundError(RepositoryError):
    """Raised when a candidate ID does not exist in the given guild.

    Guild-scoped like every other read in this project, so a moderator in one
    server can neither resolve nor learn anything about another server's
    candidate by guessing a numeric ID.
    """


class PendingFactAlreadyResolvedError(RepositoryError):
    """Raised when a candidate was confirmed or discarded by someone else first.

    Covers "already confirmed" and "already discarded" identically and on
    purpose: from the loser's point of view both mean the same thing -- this
    candidate's outcome was decided by another call, nothing this call did was
    written, and re-deciding it is exactly what must not happen.
    """


def _row_to_pending_fact(row: sqlite3.Row) -> PendingFact:
    return PendingFact(
        id=row[0],
        guild_id=row[1],
        channel_id=row[2],
        message_id=row[3],
        content=row[4],
        embedding=row[5],
        category=FactCategory(row[6]),
        status=PendingFactStatus(row[7]),
        similar_fact_id=row[8],
        similar_fact_score=row[9],
        relationship=None if row[10] is None else SupersessionRelationship(row[10]),
        relationship_reasoning=row[11],
        confirmed_fact_id=row[12],
        created_at=row[13],
        resolved_at=row[14],
        resolved_by_id=row[15],
    )


async def verify_pending_facts_schema(conn: aiosqlite.Connection) -> None:
    """Add Phase 3a-3's judgment columns to a pending_facts table that predates them.

    Called once at startup, after init_schema, for the same reason
    verify_signal_schema is: `CREATE TABLE IF NOT EXISTS` cannot reshape a table
    that already exists, so a database created by Phase 3a-2 would keep its
    older shape and every staged candidate would fail at write time -- one
    logged exception per batch, forever, with the bot otherwise looking healthy.

    Purely additive and therefore migrated in place rather than refused: no
    existing column's meaning changes, both new columns are NULL for every
    pre-existing row, and NULL is already the ordinary "never judged" state, so
    old candidates simply keep rendering with Phase 3a-2's plain similarity
    hint. Nothing an operator has to decide, so nothing to stop the boot for.

    Idempotent: a table already at the current shape adds nothing, and a
    partially migrated one (a crash between the two ALTERs) is completed in one
    pass. A database with no such table at all passes untouched.
    """
    async with conn.execute("PRAGMA table_info(pending_facts)") as cursor:
        columns = {row[1] for row in await cursor.fetchall()}

    if not columns:
        return

    async with connection_lock(conn):
        migrated = False
        if _RELATIONSHIP_COLUMN not in columns:
            values = ", ".join(
                f"'{relationship.value}'" for relationship in SupersessionRelationship
            )
            await conn.execute(
                f"ALTER TABLE pending_facts ADD COLUMN {_RELATIONSHIP_COLUMN} TEXT "
                f"CHECK ({_RELATIONSHIP_COLUMN} IN ({values}))"
            )
            migrated = True
        if _RELATIONSHIP_REASONING_COLUMN not in columns:
            await conn.execute(
                f"ALTER TABLE pending_facts ADD COLUMN {_RELATIONSHIP_REASONING_COLUMN} TEXT"
            )
            migrated = True
        if migrated:
            await conn.commit()


async def stage_pending_fact(
    conn: aiosqlite.Connection,
    *,
    guild_id: int,
    channel_id: int,
    message_id: int,
    content: str,
    embedding: bytes,
    category: FactCategory,
    similar_fact_id: int | None = None,
    similar_fact_score: float | None = None,
) -> PendingFact | None:
    """Stage one distilled candidate for review, or return None if it is already staged.

    Idempotent by (channel_id, message_id, content), absorbed by name through
    the UNIQUE constraint rather than by a blanket INSERT OR IGNORE -- the same
    distinction record_signal documents: OR IGNORE would also swallow a CHECK
    or NOT NULL violation, turning a corrupt candidate into no candidate with
    no error and no log line.

    Returning None rather than raising for a duplicate is what makes a
    re-distilled batch safe (see aura.extraction.pipeline): a crash between
    claiming a spend slot and clearing the queue re-runs the same messages,
    which produce the same sentences, which must land as the same candidates
    rather than as a second set for a moderator to reject one by one.
    """
    created_at = utc_now_iso()
    async with connection_lock(conn):
        cursor = await conn.execute(
            """
            INSERT INTO pending_facts
                (guild_id, channel_id, message_id, content, embedding, category, status,
                 similar_fact_id, similar_fact_score, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (channel_id, message_id, content) DO NOTHING
            """,
            (
                guild_id,
                channel_id,
                message_id,
                content,
                embedding,
                category,
                PendingFactStatus.PENDING,
                similar_fact_id,
                similar_fact_score,
                created_at,
            ),
        )
        await conn.commit()
        if cursor.rowcount != 1:
            return None
        pending_id = cursor.lastrowid
        assert pending_id is not None  # guaranteed by sqlite after a successful INSERT

    return PendingFact(
        id=pending_id,
        guild_id=guild_id,
        channel_id=channel_id,
        message_id=message_id,
        content=content,
        embedding=embedding,
        category=category,
        status=PendingFactStatus.PENDING,
        similar_fact_id=similar_fact_id,
        similar_fact_score=similar_fact_score,
        confirmed_fact_id=None,
        created_at=datetime.fromisoformat(created_at),
    )


async def record_relationship_judgement(
    conn: aiosqlite.Connection,
    *,
    guild_id: int,
    pending_id: int,
    relationship: SupersessionRelationship,
    reasoning: str,
) -> bool:
    """Attach Phase 3a-3's judgment to a staged candidate. Returns whether it applied.

    Deliberately a separate write from stage_pending_fact rather than two more
    arguments to it, because the judgment is paid for and staging is not: the
    candidate must land in the review queue whether or not the judgment call
    happens, succeeds, or is refused by the daily cap. Splitting them is what
    makes "the judgment failed" degrade into "the candidate carries Phase
    3a-2's plain hint" instead of into a lost candidate.

    Guarded on TWO conditions, both structural rather than defensive:

    * `status = 'pending'`. A candidate a moderator already confirmed or
      discarded must not be annotated afterwards with a judgment they never
      saw -- the stored row would then claim they decided with information in
      front of them that arrived later.
    * `similar_fact_id IS NOT NULL`. This judgment only exists to explain a
      dedup hit, and a candidate with no hit has nothing for it to explain. The
      call site is what keeps the paid call rare (see aura.extraction.pipeline);
      this makes the same invariant true of the DATA regardless of what any
      future call site does.

    Returns False for either refusal rather than raising: both are ordinary
    races, not errors, and the caller has nothing to undo -- the money was
    already spent before this write.
    """
    async with connection_lock(conn):
        cursor = await conn.execute(
            """
            UPDATE pending_facts
            SET relationship = ?, relationship_reasoning = ?
            WHERE id = ? AND guild_id = ? AND status = ? AND similar_fact_id IS NOT NULL
            """,
            (relationship, reasoning, pending_id, guild_id, PendingFactStatus.PENDING),
        )
        await conn.commit()
    return cursor.rowcount == 1


async def get_pending_facts(
    conn: aiosqlite.Connection, *, guild_id: int, limit: int
) -> list[PendingFact]:
    """Return guild_id's unresolved candidates, oldest first.

    Oldest first, unlike the newest-first diagnostic views elsewhere: this is a
    work queue, and the correct order to review a queue in is the order it
    arrived, so nothing sits at the bottom forever while newer candidates keep
    landing on top of it.

    Rejects a negative limit rather than passing it to SQL, where LIMIT -1
    means no limit at all -- the same trap get_recent_signals documents.
    """
    if limit < 0:
        raise ValueError(f"limit must not be negative, got {limit}")

    async with connection_lock(conn):
        async with conn.execute(
            f"""
            SELECT {_PENDING_COLUMNS} FROM pending_facts
            WHERE guild_id = ? AND status = ?
            ORDER BY id ASC
            LIMIT ?
            """,
            (guild_id, PendingFactStatus.PENDING, limit),
        ) as cursor:
            rows = await cursor.fetchall()
    return [_row_to_pending_fact(row) for row in rows]


async def count_pending_facts(conn: aiosqlite.Connection, *, guild_id: int) -> int:
    """Return how many candidates are still awaiting review in guild_id."""
    async with connection_lock(conn):
        async with conn.execute(
            "SELECT COUNT(*) FROM pending_facts WHERE guild_id = ? AND status = ?",
            (guild_id, PendingFactStatus.PENDING),
        ) as cursor:
            row = await cursor.fetchone()
    return int(row[0]) if row else 0


async def get_pending_fact(
    conn: aiosqlite.Connection, *, guild_id: int, pending_id: int
) -> PendingFact | None:
    """Return one candidate by ID within guild_id, or None if it isn't there.

    Guild-scoped, not just ID-scoped, for the same isolation reason
    get_fact_by_id is.
    """
    async with connection_lock(conn):
        async with conn.execute(
            f"SELECT {_PENDING_COLUMNS} FROM pending_facts WHERE id = ? AND guild_id = ?",
            (pending_id, guild_id),
        ) as cursor:
            row = await cursor.fetchone()
    return None if row is None else _row_to_pending_fact(row)


async def confirm_pending_fact(
    conn: aiosqlite.Connection, *, guild_id: int, pending_id: int, resolved_by_id: int
) -> Fact:
    """Turn one staged candidate into a real active fact, atomically.

    The whole operation -- claiming the candidate, inserting the fact, and
    linking the two -- happens in one transaction under one held lock, and the
    claim comes FIRST. That ordering is what makes the two-moderator race
    deterministic: the guarded `UPDATE ... WHERE status = 'pending'` can only
    succeed once, and a caller whose rowcount is 0 raises before any fact
    exists. A loser therefore cannot leave a duplicate active fact behind, and
    a crash between the insert and the link rolls both back rather than
    stranding a fact whose candidate still reads as unreviewed.

    The fact is written through insert_fact_within_transaction, the same single
    statement /aura-facts' manual entry and supersession use, so an
    automatically extracted fact is in every respect an ordinary fact once
    confirmed -- there is no second kind of row and nothing downstream has to
    know where it came from.

    Raises PendingFactNotFoundError if pending_id doesn't exist in guild_id,
    and PendingFactAlreadyResolvedError if someone else confirmed or discarded
    it first (including a concurrent caller that won this race).
    """
    now = utc_now_iso()
    async with connection_lock(conn):
        try:
            async with conn.execute(
                f"SELECT {_PENDING_COLUMNS} FROM pending_facts WHERE id = ? AND guild_id = ?",
                (pending_id, guild_id),
            ) as cursor:
                row = await cursor.fetchone()

            if row is None:
                raise PendingFactNotFoundError(
                    f"Candidate {pending_id} does not exist in guild {guild_id}."
                )
            candidate = _row_to_pending_fact(row)

            claim_cursor = await conn.execute(
                """
                UPDATE pending_facts
                SET status = ?, resolved_at = ?, resolved_by_id = ?
                WHERE id = ? AND guild_id = ? AND status = ?
                """,
                (
                    PendingFactStatus.CONFIRMED,
                    now,
                    resolved_by_id,
                    pending_id,
                    guild_id,
                    PendingFactStatus.PENDING,
                ),
            )
            if claim_cursor.rowcount != 1:
                raise PendingFactAlreadyResolvedError(
                    f"Candidate {pending_id} in guild {guild_id} was already confirmed or "
                    "discarded (possibly by a concurrent call); nothing was changed."
                )

            fact = await insert_fact_within_transaction(
                conn,
                guild_id=candidate.guild_id,
                channel_id=candidate.channel_id,
                message_id=candidate.message_id,
                content=candidate.content,
                embedding=candidate.embedding,
                created_at=now,
            )

            await conn.execute(
                "UPDATE pending_facts SET confirmed_fact_id = ? WHERE id = ?",
                (fact.id, pending_id),
            )
        except BaseException:
            await conn.rollback()
            raise

        await conn.commit()

    return fact


async def discard_pending_fact(
    conn: aiosqlite.Connection, *, guild_id: int, pending_id: int, resolved_by_id: int
) -> None:
    """Reject one staged candidate, atomically and without writing any fact.

    The same guarded claim confirm_pending_fact makes, which is what settles
    the mixed race -- one moderator confirming while another discards. Whoever
    reaches the UPDATE first decides the candidate's outcome for good; the
    other raises PendingFactAlreadyResolvedError and writes nothing. The
    outcome depends on which call arrived first and on nothing else, and no
    interleaving produces both a fact and a discard.

    Raises PendingFactNotFoundError if pending_id doesn't exist in guild_id.
    """
    now = utc_now_iso()
    async with connection_lock(conn):
        try:
            async with conn.execute(
                "SELECT status FROM pending_facts WHERE id = ? AND guild_id = ?",
                (pending_id, guild_id),
            ) as cursor:
                row = await cursor.fetchone()

            if row is None:
                raise PendingFactNotFoundError(
                    f"Candidate {pending_id} does not exist in guild {guild_id}."
                )

            claim_cursor = await conn.execute(
                """
                UPDATE pending_facts
                SET status = ?, resolved_at = ?, resolved_by_id = ?
                WHERE id = ? AND guild_id = ? AND status = ?
                """,
                (
                    PendingFactStatus.DISCARDED,
                    now,
                    resolved_by_id,
                    pending_id,
                    guild_id,
                    PendingFactStatus.PENDING,
                ),
            )
            if claim_cursor.rowcount != 1:
                raise PendingFactAlreadyResolvedError(
                    f"Candidate {pending_id} in guild {guild_id} was already confirmed or "
                    "discarded (possibly by a concurrent call); nothing was changed."
                )
        except BaseException:
            await conn.rollback()
            raise

        await conn.commit()
