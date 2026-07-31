"""Invariants shared by every module that writes to Aura's SQLite connection.

These live here, not in aura.db.repository, because they are properties of
*how this project uses one aiosqlite connection* rather than properties of
the knowledge model. Keeping them separate lets a non-knowledge-model writer
(Phase 2's proactive_signals diagnostics) honour the same transaction rules
without importing the fact repository at all.

All operations against a given aiosqlite.Connection are serialized through a
lock scoped to that connection (see connection_lock). This isn't about SQLite
throughput (aiosqlite already funnels every statement through one worker
thread) -- it's about transaction integrity. A bare aiosqlite.Connection has
exactly one transaction at a time; without this lock, two concurrently-running
multi-statement operations (e.g. two supersede_fact calls) interleave their
statements into what SQLite sees as a *single* transaction, so one call's
rollback can silently discard the other call's already-committed work.
Serializing every operation's DB work behind one lock is the simplest correct
fix at the data volume this project targets.

The lock is deliberately *per-connection*, not a single module-level
asyncio.Lock. asyncio's lock only binds to a specific event loop lazily, the
first time it's actually contended -- a module-level lock that's never
contended (e.g. in a test that never runs two ops concurrently) will happily
survive being reused from a later, different event loop, but the moment it
*is* contended under one loop, it permanently binds to that loop and raises
RuntimeError if a different loop ever contends it again. In production
there's exactly one connection and one long-lived event loop for the
process's whole life, so this never surfaces -- but a test suite naturally
creates a fresh event loop per test, so a shared module-level lock would
break the second test that exercises real contention. Keying the lock by
connection identity means each connection (and, in tests, each fresh
in-memory database) gets its own lock tied to whichever loop actually uses it.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from weakref import WeakKeyDictionary

import aiosqlite

_connection_locks: WeakKeyDictionary[aiosqlite.Connection, asyncio.Lock] = WeakKeyDictionary()


def connection_lock(conn: aiosqlite.Connection) -> asyncio.Lock:
    """Return this connection's operation-serialization lock, creating it on first use.

    Every coroutine that issues statements against a shared connection must
    hold this lock for the whole of its logical operation -- not just its
    writes, and not just its multi-statement operations. A single unguarded
    ``INSERT`` + ``commit()`` is enough to end another coroutine's in-flight
    transaction early, because SQLite has no notion of which caller a
    ``COMMIT`` belongs to. See this module's docstring for the full reasoning.
    """
    lock = _connection_locks.get(conn)
    if lock is None:
        lock = asyncio.Lock()
        _connection_locks[conn] = lock
    return lock


def utc_iso(moment: datetime) -> str:
    """Format a moment as a fixed-width UTC ISO-8601 string (always 6 fractional digits).

    datetime.isoformat() omits the microsecond field entirely when it's
    zero, which makes plain lexicographic string comparison of timestamps
    unreliable. strftime's %f is always zero-padded to 6 digits, so these
    strings sort correctly as text -- relied on by the periodic digest in a
    later phase, by proactive_signals' newest-first ordering, by the
    proactive cooldown's timestamp comparison (aura.db.proactive_state,
    which compares these as strings in SQL rather than parsing them), and
    handy for `sqlite3 data/aura.db "select * from facts order by
    created_at"` in the meantime.

    Converts to UTC rather than trusting the offset it was handed, so a
    moment expressed in any timezone still sorts against every other
    timestamp in the database. A naive datetime is rejected instead of being
    assumed to be UTC: on a host set to a non-UTC zone that assumption is
    silently wrong by whole hours, which for the cooldown means expiring it
    early.
    """
    if moment.tzinfo is None:
        raise ValueError(f"utc_iso requires a timezone-aware datetime, got {moment!r}")
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f+00:00")


def utc_day(moment: datetime) -> str:
    """Return the UTC calendar day (YYYY-MM-DD) a timezone-aware moment falls in.

    Converts to UTC first rather than reading the date off whatever offset it
    arrived with: a moment written as 01:30+05:30 is still the previous UTC
    day, and taking .date() without converting would file it under the wrong
    day and hand a guild a second daily budget.

    Rejects a naive datetime instead of assuming UTC, because that assumption
    is wrong exactly when it matters -- datetime.now() on a host set to
    Europe/Berlin is two hours ahead of UTC in summer, which would move the
    reset boundary and, for two hours a day, file rows under tomorrow.

    Lives here beside utc_iso rather than in the module that first needed it
    (aura.db.proactive_state), because both daily spend ledgers key on it now
    -- the proactive one and the extraction one -- and a day boundary defined
    twice is a day boundary that can disagree with itself. The full reasoning
    for why the boundary is UTC at all, rather than any local zone, is in
    aura.db.proactive_state's module docstring, where the trade-off it makes
    is felt.
    """
    if moment.tzinfo is None:
        raise ValueError(f"utc_day requires a timezone-aware datetime, got {moment!r}")
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%d")


def utc_now() -> datetime:
    """Current time as a timezone-aware UTC datetime.

    The one clock read for the whole application, so a caller that needs
    both a timestamp and something derived from it (the proactive cap's day
    key, for instance) can take one reading and pass it down rather than
    reading the clock twice and straddling a boundary between the two.
    """
    return datetime.now(timezone.utc)


def utc_now_iso() -> str:
    """Current UTC time as a fixed-width ISO-8601 string; see utc_iso."""
    return utc_iso(utc_now())
