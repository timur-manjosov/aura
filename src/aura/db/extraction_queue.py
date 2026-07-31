"""The durable batch collector: candidate messages waiting for their channel's
distillation window to close.

**Durable by construction, not by a recovery step.** The phase brief holds this
to the same bar as the proactive spend ledger: a batch that was still filling
when the container restarted must not be lost. That rules out the obvious
implementation -- a dict of channel_id to a list, with an asyncio timer per
channel -- because every part of it dies with the process. So there is no
in-memory batch state anywhere in this pipeline. A message is enqueued by
writing one row; a batch is "due" if the oldest row in its channel is older
than the window; a restart simply reads the table and continues, because the
table was the only state there ever was.

That also removes a whole class of race the timer design would have had. There
are no timers to cancel, no channel whose timer fired while its list was being
appended to, and no window where a message is in a list but not yet in the
database. `due_channels` and `claim_batch` are ordinary queries over rows that
are either committed or not.

**Why this table holds raw message text** -- the one place in Aura that does --
is argued above the table definition in schema.sql. In short: it is a buffer
with a lifetime of one batch window, not a record, and the alternative is
re-fetching every message from Discord at flush time.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta

import aiosqlite
from pydantic import BaseModel

from aura.db.connection import connection_lock, utc_iso

# Upper bound on the batch window, mirroring MAX_COOLDOWN_SECONDS in
# aura.db.proactive_state and existing for the same reason: the cutoff is
# computed as `now - timedelta(seconds=...)`, which raises rather than
# saturating, so a value past this would turn a misconfiguration into an
# exception on every sweep instead of a refusal an operator can see. A batch
# window longer than a day is not batching.
MAX_BATCH_WINDOW_SECONDS = 24 * 60 * 60.0

_QUEUED_COLUMNS = (
    "channel_id, message_id, guild_id, channel_name, content, "
    "message_created_at, enqueued_at"
)


class QueuedMessage(BaseModel):
    """One candidate message waiting to be distilled.

    channel_name and message_created_at are carried rather than re-derived at
    flush time because they are what the distillation model is actually shown
    as context (the phase brief's "channel context" decision), and both can
    have changed -- or become unfetchable -- by the time the batch closes. The
    context a fact was distilled under should be the context its message
    arrived in.
    """

    channel_id: int
    message_id: int
    guild_id: int
    channel_name: str
    content: str
    message_created_at: datetime
    enqueued_at: datetime


def _row_to_queued_message(row: sqlite3.Row) -> QueuedMessage:
    return QueuedMessage(
        channel_id=row[0],
        message_id=row[1],
        guild_id=row[2],
        channel_name=row[3],
        content=row[4],
        message_created_at=row[5],
        enqueued_at=row[6],
    )


async def enqueue_message(
    conn: aiosqlite.Connection,
    *,
    guild_id: int,
    channel_id: int,
    message_id: int,
    channel_name: str,
    content: str,
    message_created_at: datetime,
    now: datetime,
) -> bool:
    """Add one message to its channel's pending batch. Returns False if already queued.

    Idempotent against Discord redelivering the same message event after a
    resumed session -- the same hazard proactive_escalations documents at
    length, and the same answer: absorbed by naming the conflicting constraint
    rather than by a blanket INSERT OR IGNORE, so a genuine NOT NULL or CHECK
    violation still surfaces instead of silently becoming a missing row.

    Requires timezone-aware datetimes, rejected rather than assumed to be UTC,
    for the reason utc_iso gives: on a host in a non-UTC zone the assumption is
    wrong by whole hours, which here would mean flushing a batch early or late.
    """
    if message_created_at.tzinfo is None:
        raise ValueError(
            f"message_created_at must be timezone-aware, got {message_created_at!r}"
        )
    if now.tzinfo is None:
        raise ValueError(f"now must be a timezone-aware datetime, got {now!r}")

    async with connection_lock(conn):
        cursor = await conn.execute(
            """
            INSERT INTO extraction_queue
                (channel_id, message_id, guild_id, channel_name, content,
                 message_created_at, enqueued_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (channel_id, message_id) DO NOTHING
            """,
            (
                channel_id,
                message_id,
                guild_id,
                channel_name,
                content,
                utc_iso(message_created_at),
                utc_iso(now),
            ),
        )
        await conn.commit()
    return cursor.rowcount == 1


async def remove_queued_message(
    conn: aiosqlite.Connection, *, channel_id: int, message_id: int
) -> bool:
    """Withdraw one message from its pending batch. Returns whether a row was removed.

    This is the edit/delete abort the phase brief asks for, and its whole scope:
    a message withdrawn before its batch closes is never distilled and never
    costs anything. A message already distilled is past this function's reach --
    retracting an already-staged candidate when its source is edited days later
    is explicitly out of scope for this sub-phase, and the batch window is the
    part of that problem worth solving now because it covers the common case
    (a typo fixed, a message thought better of) at essentially no cost.

    Returning whether anything was removed rather than silently succeeding
    lets the caller log the difference between "an edit withdrew a real
    candidate" and the overwhelmingly common "an edit to a message that was
    never queued at all", which are worth being able to tell apart when
    reading why an expected fact never appeared.
    """
    async with connection_lock(conn):
        cursor = await conn.execute(
            "DELETE FROM extraction_queue WHERE channel_id = ? AND message_id = ?",
            (channel_id, message_id),
        )
        await conn.commit()
    return cursor.rowcount > 0


async def due_channels(
    conn: aiosqlite.Connection, *, window_seconds: float, now: datetime
) -> list[int]:
    """Return the channels whose oldest queued message is past the batch window.

    "Oldest message decides" rather than "newest message decides", which is the
    difference between a window and a debounce: a channel that receives one
    candidate every four minutes would, under a debounce, never flush at all
    while the conversation continued. The window guarantees a bounded wait for
    every message regardless of what arrives after it.

    Read-only. Nothing here claims anything, so two sweeps overlapping is
    harmless -- claim_batch below is what serializes them.
    """
    if not 0 <= window_seconds <= MAX_BATCH_WINDOW_SECONDS:
        raise ValueError(
            f"window_seconds must be between 0 and {MAX_BATCH_WINDOW_SECONDS}, "
            f"got {window_seconds!r}"
        )
    if now.tzinfo is None:
        raise ValueError(f"now must be a timezone-aware datetime, got {now!r}")

    cutoff = utc_iso(now - timedelta(seconds=window_seconds))
    async with connection_lock(conn):
        async with conn.execute(
            """
            SELECT channel_id FROM extraction_queue
            GROUP BY channel_id
            HAVING MIN(enqueued_at) <= ?
            ORDER BY MIN(enqueued_at) ASC
            """,
            (cutoff,),
        ) as cursor:
            rows = await cursor.fetchall()
    return [int(row[0]) for row in rows]


async def read_batch(
    conn: aiosqlite.Connection, *, channel_id: int, limit: int
) -> list[QueuedMessage]:
    """Read up to limit queued messages for one channel, oldest first.

    Reads without deleting, deliberately. The rows are cleared only after the
    batch has been distilled and its candidates staged (see
    aura.extraction.pipeline), so a crash anywhere in between leaves the batch
    intact to be retried rather than losing it. Retrying costs one more slot
    from the daily cap and re-stages the same candidates idempotently, which is
    the cheap direction to fail in; deleting first would make a crashed batch
    unrecoverable, which is not.
    """
    if limit < 0:
        raise ValueError(f"limit must not be negative, got {limit}")

    async with connection_lock(conn):
        async with conn.execute(
            f"""
            SELECT {_QUEUED_COLUMNS} FROM extraction_queue
            WHERE channel_id = ?
            ORDER BY enqueued_at ASC, message_id ASC
            LIMIT ?
            """,
            (channel_id, limit),
        ) as cursor:
            rows = await cursor.fetchall()
    return [_row_to_queued_message(row) for row in rows]


async def clear_batch(
    conn: aiosqlite.Connection, *, channel_id: int, message_ids: list[int]
) -> int:
    """Delete exactly the messages a finished batch consumed, and return how many.

    Scoped to the specific message IDs rather than "everything in this channel"
    on purpose: messages that arrived while the batch was being distilled are
    already queued for the NEXT batch, and deleting the channel wholesale would
    silently drop them. This is the one place a plausible-looking simplification
    would lose data with no error.
    """
    if not message_ids:
        return 0

    placeholders = ",".join("?" for _ in message_ids)
    async with connection_lock(conn):
        cursor = await conn.execute(
            f"DELETE FROM extraction_queue WHERE channel_id = ? AND message_id IN ({placeholders})",
            (channel_id, *message_ids),
        )
        await conn.commit()
    return cursor.rowcount


async def count_queued(conn: aiosqlite.Connection, *, channel_id: int | None = None) -> int:
    """Return how many messages are waiting, in one channel or across all of them.

    Read-only; used by the tests that prove a restart loses nothing and by
    operational logging, never by a decision.
    """
    async with connection_lock(conn):
        if channel_id is None:
            query, params = "SELECT COUNT(*) FROM extraction_queue", ()
        else:
            query, params = (
                "SELECT COUNT(*) FROM extraction_queue WHERE channel_id = ?",
                (channel_id,),
            )
        async with conn.execute(query, params) as cursor:
            row = await cursor.fetchone()
    return int(row[0]) if row else 0
