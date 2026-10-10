"""Executing a deletion: ledger entry first, then the rule, then the entry marked complete.

Every deletion path -- a member's button, a server admin's command, a
moderator's single fact, the operator, and the purge after Aura left a server
-- goes through one of the three functions here, so each is recorded in the
ledger the same way and re-applied the same way (`reapply_ledger`).

Imports `aura.db.deletion` and `aura.privacy.ledger`.
"""

from __future__ import annotations

import logging
from datetime import datetime

import aiosqlite

from aura.db.connection import utc_iso, utc_now
from aura.db.deletion import (
    DeletionCounts,
    MemberDeletionMode,
    forget_fact,
    forget_member,
    purge_guild,
)
from aura.privacy.ledger import DeletionKind, DeletionLedger, DeletionReason, LedgerEntry

logger = logging.getLogger(__name__)


async def execute_member_deletion(
    db: aiosqlite.Connection,
    ledger: DeletionLedger,
    *,
    user_id: int,
    guild_id: int | None,
    mode: MemberDeletionMode,
    reason: DeletionReason,
    now: datetime,
) -> DeletionCounts:
    """Delete a member's data (one server, or all) and record it in the ledger.

    Parameters
    ----------
    db
        The main database.
    ledger
        The deletion ledger.
    user_id
        The member (> 0).
    guild_id
        One server, or None for all.
    mode
        What happens to facts from their messages.
    reason
        MEMBER_REQUEST or OPERATOR_REQUEST.
    now
        The request's moment: the rule's bound.

    Returns
    -------
    DeletionCounts
        What was removed.

    Raises
    ------
    ValueError
        If `user_id` is not positive (nothing is recorded or deleted).
    """
    if user_id <= 0:
        raise ValueError("user_id must be a real Discord ID")
    entry_id = await ledger.record(
        LedgerEntry(
            kind=DeletionKind.MEMBER,
            reason=reason,
            requested_at=now,
            guild_id=guild_id,
            user_id=user_id,
            mode=mode,
        )
    )
    counts = await forget_member(db, user_id=user_id, guild_id=guild_id, mode=mode, before=now)
    await ledger.complete(entry_id, counts, at=utc_now())
    logger.info(
        "Deletion %d (member, %s, %s, %s): %s",
        entry_id,
        reason.value,
        "one server" if guild_id is not None else "all servers",
        mode.value,
        counts.summary(),
    )
    return counts


async def execute_guild_purge(
    db: aiosqlite.Connection,
    ledger: DeletionLedger,
    *,
    guild_id: int,
    reason: DeletionReason,
    now: datetime,
) -> DeletionCounts:
    """Delete a server's data (billing records excepted) and record it in the ledger.

    Parameters
    ----------
    db
        The main database.
    ledger
        The deletion ledger.
    guild_id
        The server.
    reason
        ADMIN_REQUEST, OPERATOR_REQUEST or LEFT_SERVER.
    now
        The bound.

    Returns
    -------
    DeletionCounts
        What was removed.
    """
    entry_id = await ledger.record(
        LedgerEntry(kind=DeletionKind.GUILD, reason=reason, requested_at=now, guild_id=guild_id)
    )
    counts = await purge_guild(db, guild_id=guild_id, before=now)
    await ledger.complete(entry_id, counts, at=utc_now())
    logger.info("Deletion %d (server, %s): %s", entry_id, reason.value, counts.summary())
    return counts


async def execute_fact_deletion(
    db: aiosqlite.Connection,
    ledger: DeletionLedger,
    *,
    guild_id: int,
    fact_id: int,
    fact_created_at: datetime,
    reason: DeletionReason,
    now: datetime,
) -> DeletionCounts:
    """Delete one fact for good and record it in the ledger.

    Parameters
    ----------
    db
        The main database.
    ledger
        The deletion ledger.
    guild_id
        The fact's server.
    fact_id
        The fact.
    fact_created_at
        Its timestamp, read just before; recorded so re-application can never
        hit a different fact that reuses the ID.
    reason
        MODERATOR_REQUEST or OPERATOR_REQUEST.
    now
        When it was requested.

    Returns
    -------
    DeletionCounts
        What was removed; empty if the fact no longer exists.
    """
    created = utc_iso(fact_created_at)
    entry_id = await ledger.record(
        LedgerEntry(
            kind=DeletionKind.FACT,
            reason=reason,
            requested_at=now,
            guild_id=guild_id,
            fact_id=fact_id,
            fact_created_at=created,
        )
    )
    counts = await forget_fact(db, guild_id=guild_id, fact_id=fact_id, created_at=created)
    await ledger.complete(entry_id, counts, at=utc_now())
    logger.info("Deletion %d (one fact, %s): %s", entry_id, reason.value, counts.summary())
    return counts


async def reapply_entry(db: aiosqlite.Connection, entry: LedgerEntry) -> DeletionCounts:
    """Run one ledger entry's rule again, bounded to its original moment.

    Parameters
    ----------
    db
        The main database.
    entry
        A ledger entry.

    Returns
    -------
    DeletionCounts
        What had come back and was removed again; empty in the normal case.
    """
    if entry.kind is DeletionKind.MEMBER:
        assert entry.user_id is not None and entry.mode is not None  # ledger CHECK
        return await forget_member(
            db,
            user_id=entry.user_id,
            guild_id=entry.guild_id,
            mode=entry.mode,
            before=entry.requested_at,
        )
    if entry.kind is DeletionKind.GUILD:
        assert entry.guild_id is not None  # ledger CHECK
        return await purge_guild(db, guild_id=entry.guild_id, before=entry.requested_at)
    assert entry.guild_id is not None and entry.fact_id is not None  # ledger CHECK
    return await forget_fact(
        db, guild_id=entry.guild_id, fact_id=entry.fact_id, created_at=entry.fact_created_at
    )


async def reapply_ledger(db: aiosqlite.Connection, ledger: DeletionLedger) -> DeletionCounts:
    """Re-apply every ledger entry; finish any that never completed.

    Parameters
    ----------
    db
        The main database.
    ledger
        The deletion ledger.

    Returns
    -------
    DeletionCounts
        Everything removed again, summed. Empty unless a backup was restored or
        a deletion was interrupted.

    Notes
    -----
    Idempotent: on a database where every entry already applied, each rule
    finds nothing. An entry that never completed (a crash between recording it
    and finishing the deletion) is completed here.
    """
    total = DeletionCounts()
    for entry in await ledger.entries():
        counts = await reapply_entry(db, entry)
        if entry.completed_at is None and entry.entry_id is not None:
            await ledger.complete(entry.entry_id, counts, at=utc_now())
        total = total.merged(counts)
    if total.total:
        logger.warning(
            "Deletion ledger re-applied: rows that had been deleted were present again "
            "(restored backup or interrupted deletion) and were removed: %s",
            total.summary(),
        )
    return total
