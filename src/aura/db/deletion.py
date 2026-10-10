"""Every rule by which Aura deletes data (P7a): what a member, a server, one fact or age removes.

This module is the ONLY place that deletes knowledge-model rows. CLAUDE.md's
"old facts are never deleted, only superseded" still describes normal
operation; the rules here are the deliberate exceptions the law and Discord's
Developer Terms require -- a deletion request, the clean-up after Aura leaves a
server, and rows kept no longer than needed.

Invariants every public function keeps:

- **One transaction, all or nothing.** Each runs under the connection lock in
  a single transaction; a failure rolls everything back, so no server or
  member is ever left half-deleted. A crash mid-way is the same as never
  having started, and the caller simply runs it again.
- **Dry run = the same statements, rolled back.** `dry_run=True` executes the
  identical code and returns the identical counts, then rolls back. A dry
  run's numbers are therefore exactly what the real run will remove, by
  construction rather than by a second, parallel counting query.
- **Bounded in time.** Every rule takes `before`: only data that existed then
  (or, for rows taken from a message, came from a message written by then) is
  touched. That is what lets the deletion ledger re-apply a rule at every
  start without ever deleting what arrived later -- a member's new messages
  after their request, a server's new facts after it invited Aura back.
- **Never billing.** `guild_subscriptions` and `stripe_processed_events` are not
  touched by any rule (whether and how long they must stay is a question for
  the tax advisor).
- **Counts only.** Nothing here logs or returns content.
- After a real (not dry) run the WAL is checkpointed and truncated, so the old
  page images leave the WAL file too; with `secure_delete` on, freed pages in
  the main file are zeroed.

Imports only `aura.db.connection` and `aura.db.repository`'s errors.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Final

import aiosqlite

from aura.db.connection import connection_lock, utc_iso

# Discord's epoch (2015-01-01T00:00:00Z) in milliseconds: a snowflake's top 42
# bits count milliseconds since then.
DISCORD_EPOCH_MS: Final = 1_420_070_400_000
_SNOWFLAKE_LOW_BITS: Final = 22

# The value a removed member or moderator ID is replaced with. Discord never
# issues 0 as an ID, so it cannot collide with a real one.
REMOVED_ID: Final = 0

# Tables whose rows carry a guild_id and belong to that server's data, with the
# column that says when each row came to exist and whether it points at a
# Discord message (message_id), in an order that respects every foreign key.
# Facts, candidates, backfill runs and their dependants are handled first, by
# their own helpers.
_GUILD_TABLES: Final[tuple[tuple[str, str, bool], ...]] = (
    ("proactive_signals", "created_at", True),
    ("proactive_escalations", "escalated_at", True),
    ("extraction_queue", "enqueued_at", True),
    ("extraction_calls", "called_at", False),
    ("supersession_calls", "called_at", False),
    ("variant_calls", "called_at", False),
    ("ask_calls", "created_at", False),
    ("digest_runs", "ran_at", False),
    ("digest_config", "updated_at", False),
    ("onboarding_config", "updated_at", False),
    ("onboarding_sends", "sent_at", False),
    ("proactive_channel_config", "updated_at", False),
    ("extraction_channel_config", "updated_at", False),
    ("guild_departures", "left_at", False),
)

# Configuration tables that record which moderator last changed them.
_MODERATOR_COLUMNS: Final[tuple[tuple[str, str, str], ...]] = (
    ("proactive_channel_config", "updated_by_id", "updated_at"),
    ("extraction_channel_config", "updated_by_id", "updated_at"),
    ("digest_config", "updated_by_id", "updated_at"),
    ("onboarding_config", "updated_by_id", "updated_at"),
    ("backfill_runs", "requested_by_id", "started_at"),
    ("pending_facts", "resolved_by_id", "resolved_at"),
)

# Every table that marks a server as having data with Aura (billing excluded).
DATA_TABLES_WITH_GUILD: Final[tuple[str, ...]] = (
    "facts",
    "pending_facts",
    "backfill_runs",
    "backfill_calls",
    *(table for table, _, _ in _GUILD_TABLES if table != "guild_departures"),
)


class MemberDeletionMode(StrEnum):
    """What a member's deletion request does to the facts taken from their messages.

    DELETE_FACTS: those facts are deleted (the pre-selected choice; it matches
    Discord's wording most strictly). UNLINK: the facts stay as server
    knowledge, but their link to the member's message and the stored author are
    removed.
    """

    DELETE_FACTS = "delete_facts"
    UNLINK = "unlink"


@dataclass(frozen=True)
class DeletionCounts:
    """How many rows a rule removed or changed, per table and action.

    Attributes
    ----------
    rows
        Keys like "facts" (deleted) or "ask_calls.anonymized" (changed); only
        non-zero entries are kept.
    """

    rows: Mapping[str, int] = field(default_factory=dict)

    @property
    def total(self) -> int:
        """Return the number of rows touched, all tables together."""
        return sum(self.rows.values())

    def merged(self, other: DeletionCounts) -> DeletionCounts:
        """Return the sum of two counts.

        Parameters
        ----------
        other
            Counts to add.

        Returns
        -------
        DeletionCounts
            A new value; neither operand changes.
        """
        combined: Counter[str] = Counter(self.rows)
        combined.update(other.rows)
        return DeletionCounts(rows={key: value for key, value in combined.items() if value})

    def summary(self) -> str:
        """Return "table=count, ..." sorted by name, or "nothing" when empty.

        Returns
        -------
        str
            Safe to log: names of tables and numbers only.
        """
        if not self.rows:
            return "nothing"
        return ", ".join(f"{key}={value}" for key, value in sorted(self.rows.items()))


@dataclass(frozen=True)
class RetentionPolicy:
    """How long rows that exist only for a short-lived purpose are kept.

    Attributes
    ----------
    proactive_signal_days
        Diagnostic gate rows and escalation ledger rows (both point at a
        member's message).
    ask_member_id_days
        After this, an /aura-ask ledger row keeps its server and day but loses
        the member's ID.
    onboarding_send_days
        Welcome records (they name the member who joined).
    """

    proactive_signal_days: int
    ask_member_id_days: int
    onboarding_send_days: int

    def __post_init__(self) -> None:
        """Refuse a period below one day: today's caps and cooldowns read today's rows."""
        for name in ("proactive_signal_days", "ask_member_id_days", "onboarding_send_days"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be at least 1 day")


def latest_snowflake_at(moment: datetime) -> int:
    """Return the largest Discord snowflake that can have been created at or before a moment.

    Parameters
    ----------
    moment
        A timezone-aware instant.

    Returns
    -------
    int
        Every message ID at or below this was written at or before `moment`;
        0 for a moment before Discord's epoch.

    Raises
    ------
    ValueError
        If `moment` is naive.
    """
    if moment.tzinfo is None:
        raise ValueError("latest_snowflake_at requires a timezone-aware datetime")
    milliseconds = int(moment.timestamp() * 1000) - DISCORD_EPOCH_MS
    if milliseconds < 0:
        return 0
    return (milliseconds << _SNOWFLAKE_LOW_BITS) | ((1 << _SNOWFLAKE_LOW_BITS) - 1)


async def _fill_temp_ids(conn: aiosqlite.Connection, table: str, ids: Iterable[int]) -> None:
    """(Re)fill a TEMP table of IDs; part of the caller's transaction."""
    await conn.execute(f"CREATE TEMP TABLE IF NOT EXISTS {table} (id INTEGER PRIMARY KEY)")
    await conn.execute(f"DELETE FROM {table}")
    await conn.executemany(
        f"INSERT OR IGNORE INTO {table} (id) VALUES (?)", [(item,) for item in ids]
    )


async def _ids(conn: aiosqlite.Connection, sql: str, parameters: tuple[object, ...]) -> list[int]:
    async with conn.execute(sql, parameters) as cursor:
        return [row[0] for row in await cursor.fetchall()]


async def _run(
    conn: aiosqlite.Connection,
    counts: Counter[str],
    key: str,
    sql: str,
    parameters: tuple[object, ...] = (),
) -> None:
    cursor = await conn.execute(sql, parameters)
    if cursor.rowcount > 0:
        counts[key] += cursor.rowcount


async def _delete_candidates(
    conn: aiosqlite.Connection, candidate_ids: list[int], counts: Counter[str]
) -> None:
    """Delete extraction candidates and the spend rows that point at them."""
    if not candidate_ids:
        return
    await _fill_temp_ids(conn, "doomed_candidates", candidate_ids)
    await _run(
        conn,
        counts,
        "supersession_calls",
        "DELETE FROM supersession_calls WHERE pending_fact_id IN (SELECT id FROM doomed_candidates)",
    )
    await _run(
        conn,
        counts,
        "pending_facts",
        "DELETE FROM pending_facts WHERE id IN (SELECT id FROM doomed_candidates)",
    )


async def _delete_facts(
    conn: aiosqlite.Connection, fact_ids: list[int], counts: Counter[str]
) -> None:
    """Delete facts and repair everything that pointed at them.

    Notes
    -----
    A predecessor that pointed at a deleted fact is re-pointed at the first
    surviving fact further along the chain; when there is none it stays
    superseded with no successor -- it was replaced, and the replacement being
    deleted on request must not bring the outdated fact back into answers.
    Links and variants of a deleted fact go with it; a candidate that proposed
    it as "possibly the same" loses that hint (and the model's reasoning,
    which may quote it); the candidate a deleted fact was confirmed from holds
    the same sentence and is deleted.
    """
    if not fact_ids:
        return
    doomed = set(fact_ids)
    await _fill_temp_ids(conn, "doomed_facts", doomed)
    async with conn.execute(
        "SELECT id, superseded_by_id FROM facts WHERE id IN (SELECT id FROM doomed_facts)"
    ) as cursor:
        successor_of: dict[int, int | None] = {row[0]: row[1] for row in await cursor.fetchall()}
    async with conn.execute(
        """
        SELECT id, superseded_by_id FROM facts
        WHERE superseded_by_id IN (SELECT id FROM doomed_facts)
          AND id NOT IN (SELECT id FROM doomed_facts)
        """
    ) as cursor:
        predecessors = await cursor.fetchall()
    for predecessor_id, successor_id in predecessors:
        target: int | None = successor_id
        seen: set[int] = set()
        while target is not None and target in doomed and target not in seen:
            seen.add(target)
            target = successor_of.get(target)
        if target is not None and target in doomed:
            target = None
        await conn.execute(
            "UPDATE facts SET superseded_by_id = ? WHERE id = ?", (target, predecessor_id)
        )
        counts["facts.successor_repointed"] += 1

    await _run(
        conn,
        counts,
        "fact_links",
        """
        DELETE FROM fact_links
        WHERE fact_a_id IN (SELECT id FROM doomed_facts)
           OR fact_b_id IN (SELECT id FROM doomed_facts)
        """,
    )
    await _run(
        conn,
        counts,
        "variant_calls",
        "DELETE FROM variant_calls WHERE fact_id IN (SELECT id FROM doomed_facts)",
    )
    await _run(
        conn,
        counts,
        "fact_variants",
        "DELETE FROM fact_variants WHERE fact_id IN (SELECT id FROM doomed_facts)",
    )
    origin_candidates = await _ids(
        conn,
        "SELECT id FROM pending_facts WHERE confirmed_fact_id IN (SELECT id FROM doomed_facts)",
        (),
    )
    await _delete_candidates(conn, origin_candidates, counts)
    await _run(
        conn,
        counts,
        "pending_facts.hint_cleared",
        """
        UPDATE pending_facts
        SET similar_fact_id = NULL, similar_fact_score = NULL,
            relationship = NULL, relationship_reasoning = NULL
        WHERE similar_fact_id IN (SELECT id FROM doomed_facts)
        """,
    )
    await _run(conn, counts, "facts", "DELETE FROM facts WHERE id IN (SELECT id FROM doomed_facts)")


async def _finish(conn: aiosqlite.Connection, *, dry_run: bool, changed: bool) -> None:
    """Commit and clear the WAL when something changed, or roll back a dry run."""
    if dry_run:
        await conn.rollback()
        return
    await conn.commit()
    if changed:
        await conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")


async def forget_member(
    conn: aiosqlite.Connection,
    *,
    user_id: int,
    guild_id: int | None,
    mode: MemberDeletionMode,
    before: datetime,
    dry_run: bool = False,
) -> DeletionCounts:
    """Remove what Aura holds about one Discord user, in one server or in all of them.

    Parameters
    ----------
    conn
        Open database connection.
    user_id
        The member. Must be a real ID (> 0).
    guild_id
        The one server to act in, or None for every server.
    mode
        What happens to facts taken from their messages.
    before
        Only rows from before this moment (for message-derived rows: from a
        message written by then) are touched.
    dry_run
        Count without changing anything.

    Returns
    -------
    DeletionCounts
        What was (or, in a dry run, would be) deleted or changed.

    Raises
    ------
    ValueError
        If `user_id` is not positive.

    Notes
    -----
    Always, whatever the mode: candidates from their messages and their
    messages still waiting in the extraction queue are deleted; their ID on
    /aura-ask counters becomes 0 (the server's counts stay); their welcome
    records are deleted; where they appear as the moderator who changed a
    setting, ran a backfill or resolved a candidate, the ID becomes 0.
    Facts whose author was never looked up (NULL) or could not be (0) are not
    matched -- run the author lookup before enabling deletion.
    """
    if user_id <= 0:
        raise ValueError("user_id must be a real Discord ID")
    scope = "" if guild_id is None else " AND guild_id = ?"
    scope_params: tuple[object, ...] = () if guild_id is None else (guild_id,)
    newest_message = latest_snowflake_at(before)
    moment = utc_iso(before)
    counts: Counter[str] = Counter()

    async with connection_lock(conn):
        try:
            fact_ids = await _ids(
                conn,
                "SELECT id FROM facts WHERE source_author_id = ? AND message_id > 0 "
                f"AND message_id <= ?{scope}",
                (user_id, newest_message, *scope_params),
            )
            if mode is MemberDeletionMode.DELETE_FACTS:
                await _delete_facts(conn, fact_ids, counts)
            elif fact_ids:
                await _fill_temp_ids(conn, "doomed_facts", fact_ids)
                await _run(
                    conn,
                    counts,
                    "facts.unlinked",
                    """
                    UPDATE facts SET channel_id = 0, message_id = 0, source_author_id = NULL
                    WHERE id IN (SELECT id FROM doomed_facts)
                    """,
                )
            candidate_ids = await _ids(
                conn,
                "SELECT id FROM pending_facts WHERE source_author_id = ? AND message_id > 0 "
                f"AND message_id <= ?{scope}",
                (user_id, newest_message, *scope_params),
            )
            await _delete_candidates(conn, candidate_ids, counts)
            await _run(
                conn,
                counts,
                "extraction_queue",
                f"DELETE FROM extraction_queue WHERE author_id = ? AND message_id <= ?{scope}",
                (user_id, newest_message, *scope_params),
            )
            await _run(
                conn,
                counts,
                "ask_calls.anonymized",
                f"UPDATE ask_calls SET user_id = {REMOVED_ID} "
                f"WHERE user_id = ? AND created_at <= ?{scope}",
                (user_id, moment, *scope_params),
            )
            await _run(
                conn,
                counts,
                "onboarding_sends",
                f"DELETE FROM onboarding_sends WHERE user_id = ? AND sent_at <= ?{scope}",
                (user_id, moment, *scope_params),
            )
            for table, id_column, time_column in _MODERATOR_COLUMNS:
                await _run(
                    conn,
                    counts,
                    f"{table}.{id_column}_removed",
                    f"UPDATE {table} SET {id_column} = {REMOVED_ID} "
                    f"WHERE {id_column} = ? AND {time_column} <= ?{scope}",
                    (user_id, moment, *scope_params),
                )
            await _finish(conn, dry_run=dry_run, changed=bool(counts))
        except BaseException:
            await conn.rollback()
            raise
    return DeletionCounts(rows=dict(counts))


async def purge_guild(
    conn: aiosqlite.Connection,
    *,
    guild_id: int,
    before: datetime,
    dry_run: bool = False,
) -> DeletionCounts:
    """Delete everything Aura holds for one server, except its billing records.

    Parameters
    ----------
    conn
        Open database connection.
    guild_id
        The server.
    before
        Only rows that existed by then -- or, for rows taken from a message,
        whose message was written by then -- are deleted.
    dry_run
        Count without changing anything.

    Returns
    -------
    DeletionCounts
        What was (or would be) deleted.

    Notes
    -----
    Used for the clean-up after Aura left a server, for a server admin's own
    request, and for the operator. The guild's departure mark (if any, and if
    it predates `before`) goes too, so a purged server is no longer "pending".
    """
    newest_message = latest_snowflake_at(before)
    moment = utc_iso(before)
    counts: Counter[str] = Counter()

    async with connection_lock(conn):
        try:
            fact_ids = await _ids(
                conn,
                "SELECT id FROM facts WHERE guild_id = ? "
                "AND (created_at <= ? OR (message_id > 0 AND message_id <= ?))",
                (guild_id, moment, newest_message),
            )
            await _delete_facts(conn, fact_ids, counts)
            candidate_ids = await _ids(
                conn,
                "SELECT id FROM pending_facts WHERE guild_id = ? "
                "AND (created_at <= ? OR (message_id > 0 AND message_id <= ?))",
                (guild_id, moment, newest_message),
            )
            await _delete_candidates(conn, candidate_ids, counts)
            run_ids = await _ids(
                conn,
                "SELECT id FROM backfill_runs WHERE guild_id = ? AND started_at <= ?",
                (guild_id, moment),
            )
            await _fill_temp_ids(conn, "doomed_runs", run_ids)
            await _run(
                conn,
                counts,
                "backfill_calls",
                """
                DELETE FROM backfill_calls
                WHERE run_id IN (SELECT id FROM doomed_runs)
                   OR (guild_id = ? AND called_at <= ?)
                """,
                (guild_id, moment),
            )
            await _run(
                conn,
                counts,
                "backfill_runs",
                "DELETE FROM backfill_runs WHERE id IN (SELECT id FROM doomed_runs)",
            )
            for table, time_column, has_message in _GUILD_TABLES:
                if has_message:
                    condition = (
                        f"guild_id = ? AND ({time_column} <= ? "
                        "OR (message_id > 0 AND message_id <= ?))"
                    )
                    parameters: tuple[object, ...] = (guild_id, moment, newest_message)
                else:
                    condition = f"guild_id = ? AND {time_column} <= ?"
                    parameters = (guild_id, moment)
                await _run(
                    conn, counts, table, f"DELETE FROM {table} WHERE {condition}", parameters
                )
            await _finish(conn, dry_run=dry_run, changed=bool(counts))
        except BaseException:
            await conn.rollback()
            raise
    return DeletionCounts(rows=dict(counts))


async def forget_fact(
    conn: aiosqlite.Connection,
    *,
    guild_id: int,
    fact_id: int,
    created_at: str | None = None,
    dry_run: bool = False,
) -> DeletionCounts:
    """Delete one fact for good, repairing its chain and links.

    Parameters
    ----------
    conn
        Open database connection.
    guild_id
        The server the fact must belong to; another server's fact is never
        matched.
    fact_id
        The fact.
    created_at
        When given, the fact must also carry exactly this timestamp -- how the
        ledger re-applies the deletion without ever hitting a different fact
        that later reused the ID in a restored backup.
    dry_run
        Count without changing anything.

    Returns
    -------
    DeletionCounts
        Empty when no such fact exists (already deleted, wrong server).
    """
    counts: Counter[str] = Counter()
    sql = "SELECT id FROM facts WHERE id = ? AND guild_id = ?"
    parameters: tuple[object, ...] = (fact_id, guild_id)
    if created_at is not None:
        sql += " AND created_at = ?"
        parameters = (*parameters, created_at)
    async with connection_lock(conn):
        try:
            await _delete_facts(conn, await _ids(conn, sql, parameters), counts)
            await _finish(conn, dry_run=dry_run, changed=bool(counts))
        except BaseException:
            await conn.rollback()
            raise
    return DeletionCounts(rows=dict(counts))


async def apply_retention(
    conn: aiosqlite.Connection,
    *,
    now: datetime,
    policy: RetentionPolicy,
    dry_run: bool = False,
) -> DeletionCounts:
    """Remove rows whose purpose has passed.

    Parameters
    ----------
    conn
        Open database connection.
    now
        The current moment (timezone-aware).
    policy
        The periods.
    dry_run
        Count without changing anything.

    Returns
    -------
    DeletionCounts
        What was (or would be) removed or anonymized.

    Notes
    -----
    The /aura-ask rows keep their server and day after losing the member ID:
    the counts per server are what later plan limits and cost reviews need,
    and the member's share only ever matters for today's per-member cap.
    """
    signals_before = utc_iso(now - timedelta(days=policy.proactive_signal_days))
    ask_before = utc_iso(now - timedelta(days=policy.ask_member_id_days))
    onboarding_before = utc_iso(now - timedelta(days=policy.onboarding_send_days))
    counts: Counter[str] = Counter()
    async with connection_lock(conn):
        try:
            await _run(
                conn,
                counts,
                "proactive_signals",
                "DELETE FROM proactive_signals WHERE created_at < ?",
                (signals_before,),
            )
            await _run(
                conn,
                counts,
                "proactive_escalations",
                "DELETE FROM proactive_escalations WHERE escalated_at < ?",
                (signals_before,),
            )
            await _run(
                conn,
                counts,
                "ask_calls.anonymized",
                f"UPDATE ask_calls SET user_id = {REMOVED_ID} "
                f"WHERE created_at < ? AND user_id != {REMOVED_ID}",
                (ask_before,),
            )
            await _run(
                conn,
                counts,
                "onboarding_sends",
                "DELETE FROM onboarding_sends WHERE sent_at < ?",
                (onboarding_before,),
            )
            await _finish(conn, dry_run=dry_run, changed=bool(counts))
        except BaseException:
            await conn.rollback()
            raise
    return DeletionCounts(rows=dict(counts))


async def guilds_with_data(conn: aiosqlite.Connection) -> set[int]:
    """Return every server that has any non-billing row.

    Parameters
    ----------
    conn
        Open database connection.

    Returns
    -------
    set[int]
        Guild IDs. A server known only from a subscription row is not in it.
    """
    union = " UNION ".join(f"SELECT guild_id FROM {table}" for table in DATA_TABLES_WITH_GUILD)
    async with connection_lock(conn), conn.execute(union) as cursor:
        return {row[0] for row in await cursor.fetchall()}
