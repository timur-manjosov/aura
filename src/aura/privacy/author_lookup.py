"""The one-time author lookup for facts and candidates stored before P7a.

From P7a on, every new fact and candidate stores who wrote its source message.
Rows from before have NULL there, and a member's deletion request cannot find
them. This lookup asks Discord, once per source message, and stores the answer:
the author's ID, or 0 when the message is gone or unreadable (looked up,
unknown). A transient failure (Discord error) leaves the row NULL so the next
run tries again.

How it treats Discord and the bot:

- **Paced.** One fetch at a time with `LOOKUP_PAUSE_SECONDS` between them, on
  top of discord.py's own per-route rate limiter -- a few requests a second at
  most, so answers and posts keep their share of Discord's limits.
- **Backs off on a rate limit.** A 429 that reaches this code (discord.py
  already waited and retried) ends the batch at once; `run_author_lookup`
  waits `RATE_LIMIT_BACKOFF_SECONDS` before the next batch.
- **Resumable.** Every answer is stored as soon as it arrives, and only NULL
  rows are ever asked about, so a run cut off at any point -- a restart, a
  cancellation -- continues where it stopped.
- **Never blocks the bot.** It runs as a background task; the database lock is
  held for one small UPDATE per message, never across a Discord request.

Started once the gateway is ready when DATA_DELETION_ENABLED is on, and on
demand by the operator (`/aura-operator-privacy lookup-authors`).

Nothing here stores or logs message text: one HTTP fetch per message, of which
only the author ID is kept.

Imports `aura.db.connection`.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import Enum
from typing import Final, Protocol

import aiosqlite

from aura.db.connection import connection_lock
from aura.db.deletion import REMOVED_ID

logger = logging.getLogger(__name__)

# One run's ceiling, so a large backlog is worked off over several runs rather
# than holding the operator's interaction (and Discord's rate limit) for long.
MAX_LOOKUPS_PER_RUN: Final = 500

# The pause between two fetches: at most four lookups a second from this task.
LOOKUP_PAUSE_SECONDS: Final = 0.25

# How long the background run waits after Discord answered with a rate limit.
RATE_LIMIT_BACKOFF_SECONDS: Final = 60.0

# The background run's ceiling on batches, so a source that keeps failing
# transiently cannot keep a task alive forever.
MAX_ROUNDS: Final = 100


class AuthorUnknown(Enum):
    """The message no longer exists or Aura can no longer read it."""

    UNKNOWN = "unknown"


class LookupFailed(Enum):
    """A transient failure; try again later."""

    FAILED = "failed"


class LookupRateLimited(Enum):
    """Discord refused with a rate limit; stop and wait before trying again."""

    RATE_LIMITED = "rate_limited"


LookupAnswer = int | AuthorUnknown | LookupFailed | LookupRateLimited


class MessageAuthorSource(Protocol):
    """Where the lookup asks for a message's author."""

    async def author_of(self, channel_id: int, message_id: int) -> LookupAnswer:
        """Return the author's ID, UNKNOWN, FAILED, or RATE_LIMITED."""
        ...


@dataclass(frozen=True)
class AuthorLookupResult:
    """Counts of one lookup run.

    Attributes
    ----------
    resolved
        Source messages whose author was found and stored.
    unknown
        Source messages that are gone or unreadable (stored as 0).
    failed
        Transient failures, left for the next run.
    remaining
        Source messages still without an author after this run.
    rate_limited
        True when Discord's rate limit ended the run early.
    """

    resolved: int
    unknown: int
    failed: int
    remaining: int
    rate_limited: bool = False


async def _missing_sources(conn: aiosqlite.Connection) -> list[tuple[int, int]]:
    async with (
        connection_lock(conn),
        conn.execute(
            """
            SELECT channel_id, message_id FROM facts
            WHERE source_author_id IS NULL AND message_id > 0
            UNION
            SELECT channel_id, message_id FROM pending_facts
            WHERE source_author_id IS NULL AND message_id > 0
            ORDER BY 1, 2
            """
        ) as cursor,
    ):
        return [(row[0], row[1]) for row in await cursor.fetchall()]


async def _store_author(
    conn: aiosqlite.Connection, *, channel_id: int, message_id: int, author_id: int
) -> None:
    async with connection_lock(conn):
        for table in ("facts", "pending_facts"):
            await conn.execute(
                f"UPDATE {table} SET source_author_id = ? "
                "WHERE channel_id = ? AND message_id = ? AND source_author_id IS NULL",
                (author_id, channel_id, message_id),
            )
        await conn.commit()


async def lookup_missing_authors(
    conn: aiosqlite.Connection,
    source: MessageAuthorSource,
    *,
    limit: int = MAX_LOOKUPS_PER_RUN,
    pause_seconds: float = LOOKUP_PAUSE_SECONDS,
    sleep: Callable[[float], Awaitable[object]] = asyncio.sleep,
) -> AuthorLookupResult:
    """Look up the authors of source messages stored without one (one batch).

    Parameters
    ----------
    conn
        The main database.
    source
        Asks Discord.
    limit
        At most this many messages this run.
    pause_seconds
        The pause between two fetches.
    sleep
        How to pause (replaced in tests).

    Returns
    -------
    AuthorLookupResult
        Counts only.

    Notes
    -----
    Only fills NULLs (`WHERE source_author_id IS NULL`), so a value a newer
    write stored meanwhile is never overwritten, and a second run is cheap.
    """
    missing = await _missing_sources(conn)
    resolved = unknown = failed = 0
    rate_limited = False
    for index, (channel_id, message_id) in enumerate(missing[:limit]):
        if index:
            await sleep(pause_seconds)
        answer = await source.author_of(channel_id, message_id)
        if isinstance(answer, LookupRateLimited):
            rate_limited = True
            break
        if isinstance(answer, LookupFailed):
            failed += 1
            continue
        author_id = REMOVED_ID if isinstance(answer, AuthorUnknown) else answer
        if author_id == REMOVED_ID:
            unknown += 1
        else:
            resolved += 1
        await _store_author(conn, channel_id=channel_id, message_id=message_id, author_id=author_id)
    remaining = len(await _missing_sources(conn))
    logger.info(
        "Author lookup: %d resolved, %d unknown (message gone or unreadable), %d failed, "
        "%d still missing%s",
        resolved,
        unknown,
        failed,
        remaining,
        "; stopped by Discord's rate limit" if rate_limited else "",
    )
    return AuthorLookupResult(
        resolved=resolved,
        unknown=unknown,
        failed=failed,
        remaining=remaining,
        rate_limited=rate_limited,
    )


async def run_author_lookup(
    conn: aiosqlite.Connection,
    source: MessageAuthorSource,
    *,
    sleep: Callable[[float], Awaitable[object]] = asyncio.sleep,
    max_rounds: int = MAX_ROUNDS,
) -> int:
    """Run batches until no author is missing or no batch makes progress.

    Parameters
    ----------
    conn
        The main database.
    source
        Asks Discord.
    sleep
        How to pause (replaced in tests).
    max_rounds
        At most this many batches.

    Returns
    -------
    int
        The number of sources still without an author when it stopped.

    Notes
    -----
    A batch ended by a rate limit is followed by a pause of
    RATE_LIMIT_BACKOFF_SECONDS and another batch; a batch in which every lookup
    failed transiently ends the run (the operator can start it again).
    """
    remaining = 0
    for _ in range(max_rounds):
        result = await lookup_missing_authors(conn, source, sleep=sleep)
        remaining = result.remaining
        if remaining == 0:
            break
        if result.rate_limited:
            await sleep(RATE_LIMIT_BACKOFF_SECONDS)
            continue
        if result.resolved + result.unknown == 0:
            break
    return remaining
