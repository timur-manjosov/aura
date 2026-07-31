"""Durable budget protection: the per-channel cooldown and per-guild daily cap.

NOT part of the knowledge model, for the same reason aura.db.proactive_signals
isn't: a spend limit is none of the four things CLAUDE.md admits into it. This
module owns one table (proactive_escalations, see schema.sql) and imports
nothing from aura.db.repository in either direction, so the whole proactive
budget mechanism stays separable from the facts it protects.

Two properties this module exists to guarantee, both of them the kind of thing
that is cheap to get right here and expensive to discover in production:

**Durable, not in-memory.** Aura runs in a container with a restart policy. A
cooldown or a daily counter held in a Python dict silently resets every time
the container restarts, so a crash loop would hand out unlimited escalations
while every log line still claimed the cap was being enforced. The ledger is
the only state there is; a restart reads it back and resumes mid-cooldown and
mid-day with no recovery step (see the restart tests).

**Race-condition safe, not check-then-set.** Two messages arriving in the same
channel milliseconds apart are two concurrent tasks (discord.py dispatches
each event separately), so "read the count, decide, then write" is a textbook
lost-update race: both read 19 of 20, both decide there's room, both write.
Acquisition is therefore ONE guarded INSERT whose WHERE clause re-checks the
cooldown and the cap at write time -- SQLite evaluates it in a single implicit
transaction, so the check cannot be separated from the write even by a second
process sharing the database file. The per-connection lock from
aura.db.connection is held on top of that, as every writer on this connection
must, but correctness does not depend on it.

**What "per channel" includes.** Discord gives a thread its own channel ID, so
a thread gets its own cooldown rather than sharing its parent's. That is the
intended reading -- a thread is a separate conversation, and one answered
question in it should not silence the channel it hangs off -- but it does mean
the cooldown alone cannot bound spending: someone creating threads could earn
one escalation per thread. The per-guild daily cap is what closes that, which
is precisely why the two protections are independent rather than one.

**Why UTC for the daily boundary, not local time.** Stated explicitly because
it is a real decision, not a default that fell out of the implementation:

1. There is no "local" to use. A guild has no timezone -- Discord removed the
   server-region concept, and a guild's members are spread across timezones,
   so any local zone Aura picked would be arbitrary for most of the server.
2. DST would silently change the budget twice a year. In a zone with DST, one
   local day is 23 hours long and another is 25, so a "daily" cap would cover
   an hour less on one day and an hour more on another -- a budget guarantee
   that quietly varies is not a guarantee. UTC has no DST and every day is
   exactly 24 hours.
3. "Local" in a container means "whatever TZ the host happens to have". The
   image sets no TZ, so the boundary would be an accident of the deployment
   host rather than a documented property of the bot.
4. Every other timestamp Aura writes is already UTC (aura.db.connection).

The cost is that the reset lands mid-evening in the Americas rather than at a
tidy local midnight. That is the right trade for a spend limit, where a
predictable, uniform window matters and the exact wall-clock moment does not.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from enum import StrEnum
from math import isfinite

import aiosqlite
from pydantic import BaseModel

# utc_day moved to aura.db.connection once a second daily ledger (extraction's)
# needed the same day boundary; it is re-exported here because this module is
# where it was first defined and where the reasoning for choosing UTC lives, so
# existing importers keep working and keep landing next to that reasoning.
from aura.db.connection import connection_lock, utc_day, utc_iso

logger = logging.getLogger(__name__)

# One statement that checks the cooldown, checks the cap, and writes the row.
# The three guards are deliberately not three round trips:
#   * WHERE NOT EXISTS (...)  -- this channel has no escalation newer than the
#     cooldown cutoff.
#   * (SELECT COUNT(*) ...) < ?  -- this guild has capacity left today.
#   * ON CONFLICT DO NOTHING  -- this exact message has not escalated before
#     (Discord redelivering an event must not consume a second slot).
#
# ON CONFLICT names the constraint rather than using a blanket INSERT OR
# IGNORE, for the reason record_signal documents at length: OR IGNORE would
# also swallow a NOT NULL or CHECK violation, turning a corrupted write into a
# missing row with no error and no log line. Here that would mean silently
# failing to record a spend, which is the one thing this table exists to do.
_ACQUIRE_SLOT_SQL = """
INSERT INTO proactive_escalations
    (guild_id, channel_id, message_id, escalated_at, escalation_day)
SELECT ?, ?, ?, ?, ?
WHERE NOT EXISTS (
        SELECT 1 FROM proactive_escalations
        WHERE channel_id = ? AND escalated_at > ?
    )
  AND (
        SELECT COUNT(*) FROM proactive_escalations
        WHERE guild_id = ? AND escalation_day = ?
    ) < ?
ON CONFLICT (channel_id, message_id) DO NOTHING
"""


# Upper bounds on the two configurable numbers, because both are used in
# arithmetic that raises rather than saturating, and a value past these turns a
# misconfiguration into an exception on *every* message forever:
#
#   * cooldown_seconds becomes `now - timedelta(seconds=...)`, which raises
#     OverflowError once the result would fall outside datetime's range (about
#     2000 years back) and again, differently, once timedelta itself overflows.
#   * daily_cap is bound into SQL, and sqlite3 refuses a Python int that does
#     not fit a signed 64-bit integer.
#
# The gate fails closed in that state, so nothing would be overspent -- but a
# per-message traceback on a bot that otherwise looks healthy is precisely the
# grey area CLAUDE.md rules out, so the values are refused where an operator
# can still see why. Both limits are far past anything useful: a per-channel
# cooldown longer than a month is the daily cap's job, and a cap of a million
# escalations a day is not a budget.
MAX_COOLDOWN_SECONDS = 30 * 24 * 60 * 60.0
MAX_DAILY_CAP = 1_000_000


class EscalationOutcome(StrEnum):
    """Why a message did or didn't get a slot in the escalation budget."""

    GRANTED = "granted"
    COOLDOWN_ACTIVE = "cooldown_active"
    DAILY_CAP_REACHED = "daily_cap_reached"
    ALREADY_ESCALATED = "already_escalated"


class _LedgerState(BaseModel):
    """The three ledger readings every acquisition decision is made from."""

    already_escalated: bool
    last_escalated_at: str | None
    daily_count: int


class EscalationAttempt(BaseModel):
    """The result of one attempt to acquire an escalation slot, with the state behind it.

    Carries the numbers the decision was made on, not just the verdict, so
    the debug trail can show *why* a message was held back without
    re-deriving state that has since moved on.
    """

    outcome: EscalationOutcome
    # Cooldown remaining as observed *before* this attempt, so a granted
    # escalation reports the state it walked into (typically 0.0) rather than
    # the cooldown it just started.
    cooldown_seconds_remaining: float
    # Slots spent on this UTC day *including* this attempt when it was
    # granted, so the trail reads as "3 of 20 of today's budget is gone".
    daily_count: int
    daily_cap: int

    @property
    def granted(self) -> bool:
        """Whether this attempt actually took a slot from the budget."""
        return self.outcome is EscalationOutcome.GRANTED


async def count_escalations_on(conn: aiosqlite.Connection, *, guild_id: int, day: str) -> int:
    """Return how many escalations guild_id has already spent on a given UTC day.

    Read-only; used by the debug command to show live cap usage. Takes the
    day as a string produced by utc_day so the caller's clock, not this
    function's, defines "today".
    """
    async with connection_lock(conn):
        async with conn.execute(
            "SELECT COUNT(*) FROM proactive_escalations WHERE guild_id = ? AND escalation_day = ?",
            (guild_id, day),
        ) as cursor:
            row = await cursor.fetchone()
    return int(row[0]) if row else 0


async def is_still_freshest_escalation(
    conn: aiosqlite.Connection, *, channel_id: int, message_id: int
) -> bool:
    """Whether message_id's escalation is still the most recent one granted in its channel.

    Read-only. Used by Phase 2b-1's wake-time freshness recheck (see
    aura.proactive.listener), immediately before the paid synthesis call a
    granted slot authorizes. Under the default configuration this can never
    return False for a message that just finished its own grace period: the
    cooldown a grant starts blocks a second grant in the same channel for
    proactive_cooldown_seconds, far longer than proactive_grace_period_seconds
    is meant to be. It exists as a safety net against an operator configuring
    the two closer together, which would otherwise let a second message in
    the same channel earn its own grant and reach synthesis before -- or
    instead of -- the first one's grace period even ends. Standing down is
    the safe direction: "someone newer already has this channel's turn."

    Returns False, not True, for a message_id with no escalation row at all.
    Unreachable in production -- a granted slot is never deleted -- but the
    honest failure direction for a row this function cannot find, matching
    the rest of this module's bias toward silence over a guess.
    """
    async with connection_lock(conn):
        async with conn.execute(
            "SELECT escalated_at FROM proactive_escalations WHERE channel_id = ? AND message_id = ?",
            (channel_id, message_id),
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            return False

        async with conn.execute(
            "SELECT EXISTS (SELECT 1 FROM proactive_escalations WHERE channel_id = ? AND escalated_at > ?)",
            (channel_id, row[0]),
        ) as cursor:
            newer = await cursor.fetchone()

    assert newer is not None  # a scalar SELECT with no FROM always returns one row
    return not bool(newer[0])


async def try_acquire_escalation_slot(
    conn: aiosqlite.Connection,
    *,
    guild_id: int,
    channel_id: int,
    message_id: int,
    cooldown_seconds: float,
    daily_cap: int,
    now: datetime,
) -> EscalationAttempt:
    """Atomically take one slot from the channel cooldown and the guild's daily cap.

    Call this the moment a message is known to be worth escalating and
    *before* any expensive work runs on its behalf -- that ordering is the
    whole point (see aura.proactive.gate). A slot is recorded when it is
    claimed, not when the work it authorizes succeeds, so a crash or an API
    failure downstream spends the slot instead of quietly refunding it. For a
    mechanism whose job is bounding spend, erring toward "already spent" is
    the only safe direction.

    Returns an EscalationAttempt describing whether the slot was granted and
    the cooldown/cap state behind that answer. Never raises on a normal
    refusal -- being on cooldown or out of budget is an expected outcome, not
    an error.

    A daily_cap of 0 is valid and means "never escalate", which is a useful
    off switch rather than a misconfiguration.

    Requires a timezone-aware `now`, injected rather than read from the clock
    here, so the daily boundary and cooldown expiry are testable at the exact
    moment they matter.
    """
    if not isfinite(cooldown_seconds) or not 0 <= cooldown_seconds <= MAX_COOLDOWN_SECONDS:
        raise ValueError(
            f"cooldown_seconds must be a finite number between 0 and {MAX_COOLDOWN_SECONDS}, "
            f"got {cooldown_seconds!r}"
        )
    if not 0 <= daily_cap <= MAX_DAILY_CAP:
        raise ValueError(f"daily_cap must be between 0 and {MAX_DAILY_CAP}, got {daily_cap}")
    if now.tzinfo is None:
        raise ValueError(f"now must be a timezone-aware datetime, got {now!r}")

    day = utc_day(now)
    escalated_at = utc_iso(now)
    # One `now` produces both the timestamp and the day key, so they can never
    # disagree -- two separate clock reads straddling midnight would file a row
    # under one day with a timestamp from another.
    cooldown_cutoff = utc_iso(now - timedelta(seconds=cooldown_seconds))

    async with connection_lock(conn):
        try:
            state = await _read_state(
                conn, guild_id=guild_id, channel_id=channel_id, message_id=message_id, day=day
            )
            blocked = _classify_block(state, cooldown_cutoff=cooldown_cutoff, daily_cap=daily_cap)
            if blocked is not None:
                return _refusal(
                    blocked,
                    state,
                    now=now,
                    cooldown_seconds=cooldown_seconds,
                    daily_cap=daily_cap,
                )

            cursor = await conn.execute(
                _ACQUIRE_SLOT_SQL,
                (
                    guild_id,
                    channel_id,
                    message_id,
                    escalated_at,
                    day,
                    channel_id,
                    cooldown_cutoff,
                    guild_id,
                    day,
                    daily_cap,
                ),
            )

            if cursor.rowcount == 1:
                await conn.commit()
                return EscalationAttempt(
                    outcome=EscalationOutcome.GRANTED,
                    cooldown_seconds_remaining=_remaining(
                        state.last_escalated_at, now=now, cooldown_seconds=cooldown_seconds
                    ),
                    daily_count=state.daily_count + 1,
                    daily_cap=daily_cap,
                )

            # Unreachable from a single process: the read above happened under
            # this connection's lock, so nothing local could have taken the
            # slot in between. Reachable if a second process shares the
            # database file, which is exactly why the guards live inside the
            # INSERT and not in the Python above it.
            await conn.rollback()
            logger.warning(
                "Escalation slot for message %s/%s was taken between check and write; "
                "another process is writing to this database",
                channel_id,
                message_id,
            )
            recheck = await _read_state(
                conn, guild_id=guild_id, channel_id=channel_id, message_id=message_id, day=day
            )
            lost_race = (
                _classify_block(recheck, cooldown_cutoff=cooldown_cutoff, daily_cap=daily_cap)
                or EscalationOutcome.COOLDOWN_ACTIVE
            )
            return _refusal(
                lost_race,
                recheck,
                now=now,
                cooldown_seconds=cooldown_seconds,
                daily_cap=daily_cap,
            )
        except BaseException:
            await conn.rollback()
            raise


async def _read_state(
    conn: aiosqlite.Connection, *, guild_id: int, channel_id: int, message_id: int, day: str
) -> _LedgerState:
    """Read the ledger's current answer to all three guard questions at once.

    Caller must already hold the connection lock. Used to explain a refusal;
    the authority on whether a slot is granted is _ACQUIRE_SLOT_SQL itself.
    """
    async with conn.execute(
        """
        SELECT
            EXISTS (
                SELECT 1 FROM proactive_escalations
                WHERE channel_id = ? AND message_id = ?
            ),
            (
                SELECT MAX(escalated_at) FROM proactive_escalations
                WHERE channel_id = ?
            ),
            (
                SELECT COUNT(*) FROM proactive_escalations
                WHERE guild_id = ? AND escalation_day = ?
            )
        """,
        (channel_id, message_id, channel_id, guild_id, day),
    ) as cursor:
        row = await cursor.fetchone()

    assert row is not None  # a scalar SELECT with no FROM always returns one row
    return _LedgerState(
        already_escalated=bool(row[0]),
        last_escalated_at=row[1],
        daily_count=int(row[2]),
    )


def _classify_block(
    state: _LedgerState, *, cooldown_cutoff: str, daily_cap: int
) -> EscalationOutcome | None:
    """Return the reason this message can't escalate, or None if nothing blocks it.

    Ordered so the reason is deterministic when several apply at once, and
    identity beats budget: a redelivered message is not a cooldown violation
    or a spend, it is the same event arriving twice, and reporting it as
    either would misattribute a Discord retry to the person who wrote it.

    Compares timestamps as strings rather than parsing them, matching the SQL
    guard exactly. utc_iso pads every field to a fixed width precisely so
    lexicographic order is chronological order (see aura.db.connection), which
    keeps a corrupted or hand-edited timestamp from turning a refusal into a
    parse error.
    """
    if state.already_escalated:
        return EscalationOutcome.ALREADY_ESCALATED
    if state.last_escalated_at is not None and state.last_escalated_at > cooldown_cutoff:
        return EscalationOutcome.COOLDOWN_ACTIVE
    if state.daily_count >= daily_cap:
        return EscalationOutcome.DAILY_CAP_REACHED
    return None


def _refusal(
    outcome: EscalationOutcome,
    state: _LedgerState,
    *,
    now: datetime,
    cooldown_seconds: float,
    daily_cap: int,
) -> EscalationAttempt:
    """Package a refusal together with the state that caused it."""
    return EscalationAttempt(
        outcome=outcome,
        cooldown_seconds_remaining=_remaining(
            state.last_escalated_at, now=now, cooldown_seconds=cooldown_seconds
        ),
        daily_count=state.daily_count,
        daily_cap=daily_cap,
    )


def _remaining(last_escalated_at: str | None, *, now: datetime, cooldown_seconds: float) -> float:
    """Seconds of cooldown left on a channel, for display in the debug trail only.

    Best-effort by design. The decision to grant or refuse a slot is made by
    string comparison in SQL and never depends on this number, so an
    unparseable timestamp (only reachable by editing the database by hand)
    degrades one diagnostic figure instead of blocking every message in the
    channel forever.

    Clamped at zero rather than reporting a negative remainder, and equally
    happy with a timestamp from the future: a clock corrected backwards by NTP
    leaves rows dated ahead of `now`, and reporting the longer wait matches
    what the SQL guard will actually do -- fail closed, never spend early.
    """
    if last_escalated_at is None:
        return 0.0
    try:
        last = datetime.fromisoformat(last_escalated_at)
    except ValueError:
        logger.warning(
            "Unparseable escalation timestamp %r; reporting the full cooldown as remaining",
            last_escalated_at,
        )
        return cooldown_seconds
    if last.tzinfo is None:
        return cooldown_seconds
    return max(0.0, cooldown_seconds - (now - last).total_seconds())
