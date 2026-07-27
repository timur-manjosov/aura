"""Phase 2 diagnostic scaffolding: the proactive gate's decision trail per message.

NOT part of the knowledge model, and structured so that stays obviously
true. CLAUDE.md admits exactly four things into that model -- fact,
timestamp, status, link -- and a gate decision is none of them. It gets
its own table (see schema.sql), its own row type below rather than a place
in aura.db.models, and no import of aura.db.repository in either direction,
so this whole module can be deleted once proactive relief is tuned without
touching a line of knowledge-model code.

One row per human message Aura can see, recording every stage the gate
evaluated and the state it evaluated against. It is deliberately a trail and
not a score: "this message was rejected" is nearly useless for tuning, while
"it passed Stage 1 at +0.11, matched a fact at 0.79 with a 0.22 gap, and was
then held back because the channel had 240s of cooldown left" says exactly
which number to move. Phase 2b recalibrates the thresholds against this table,
which is the reason it records the losing numbers as carefully as the winning
ones.

Writes go through the same per-connection lock every other writer uses (see
aura.db.connection): these rows are throwaway, but they are written on the
same aiosqlite connection facts are, and an unsynchronized commit here would
end another operation's in-flight transaction early.
"""
from __future__ import annotations

from datetime import datetime
from enum import StrEnum

import aiosqlite
from pydantic import BaseModel, Field

from aura.db.connection import connection_lock, utc_now_iso

_SIGNAL_COLUMNS = (
    "id, guild_id, channel_id, message_id, stage1_score, stage1_passed, "
    "stage2_top_score, stage2_runner_up_score, stage2_gap, stage2_passed, "
    "cooldown_seconds_remaining, daily_count, daily_cap, "
    "synthesis_answers_question, synthesis_posted, grace_period_outcome, verdict, created_at"
)

# The column that proves a table is the current shape rather than Phase 2a-1's:
# 2a-1's table had no `verdict`, and its old `score` column held a different
# measurement, so it must be dropped (see verify_signal_schema) rather than
# migrated.
_TRAIL_MARKER_COLUMN = "verdict"

# The Phase 2a-3 columns. Purely additive over Phase 2a-2's shape -- no existing
# column changes meaning -- so a 2a-2 table is migrated in place with ALTER
# TABLE ADD COLUMN rather than dropped, preserving the tuning data it holds.
_SYNTHESIS_COLUMNS = ("synthesis_answers_question", "synthesis_posted")

# The Phase 2b-1 column: additive over Phase 2a-3's shape for the same reason
# -- no existing column's meaning changes -- so it is migrated in place too,
# just with its own ALTER (it is a text enum, not a 0/1 boolean).
_GRACE_COLUMN = "grace_period_outcome"


class OutdatedDiagnosticTableError(Exception):
    """Raised at startup when proactive_signals still has Phase 2a-1's shape.

    Deliberately fatal and deliberately not a silent migration. Phase 2a-1's
    table carried a single `score` column holding a one-sided similarity;
    this phase's `stage1_score` holds a contrastive one. The two are not the
    same measurement and must never share a column, because Phase 2b's
    recalibration reads this table as one homogeneous sample -- silently
    mixing both scales would produce a confidently wrong threshold.

    Dropping the old rows automatically would be the convenient choice, and
    is rejected on purpose: this code cannot see how much observation data a
    given deployment has collected, so that call belongs to whoever can. The
    table is disposable scaffolding and the fix is one command, which the
    message below spells out.
    """


class GateVerdict(StrEnum):
    """The single outcome of one message's trip through the proactive gate.

    Exhaustive and mutually exclusive: every classified message ends on
    exactly one of these, which is what makes the debug trail readable at a
    glance and the values countable for tuning. Stored as text rather than an
    integer so `sqlite3 data/aura.db "select verdict, count(*) from
    proactive_signals group by verdict"` is directly legible.
    """

    STAGE1_REJECTED = "stage1_rejected"
    NO_MATCHING_FACT = "no_matching_fact"
    # HISTORICAL as of Phase 2b-4: the gate can no longer produce this verdict,
    # because the confidence-gap check that produced it is gone (see
    # aura.proactive.gate). It stays in the enum, and stays mapped in the debug
    # command, because rows carrying it already exist in deployed databases --
    # three of them in the live guild whose evidence motivated the removal --
    # and get_recent_signals parses this column straight back through
    # GateVerdict(...). Dropping the member would turn every historical row
    # into a ValueError and take /aura-debug-signals down for the whole page it
    # appears on, which is a strictly worse outcome than one member that is
    # only ever read.
    AMBIGUOUS_FACTS = "ambiguous_facts"
    COOLDOWN_ACTIVE = "cooldown_active"
    DAILY_CAP_REACHED = "daily_cap_reached"
    DUPLICATE_DELIVERY = "duplicate_delivery"
    ELIGIBLE = "eligible"


class GracePeriodOutcome(StrEnum):
    """How Phase 2b-1's wait between ELIGIBLE and paid synthesis was resolved.

    Only ever set for a message whose gate verdict is ELIGIBLE -- every other
    verdict never enters a grace period, and its trail row keeps this column
    NULL. Exhaustive and mutually exclusive, same reasoning as GateVerdict:
    countable for tuning, and readable by a moderator at a glance.
    """

    # The wait is still in flight. Written the instant the wait begins (see
    # aura.proactive.listener), so a moderator inspecting the trail mid-wait
    # sees an honest "still deciding" rather than a stale gap.
    PENDING = "pending"
    # A different human posted in the same channel before the timer expired.
    CANCELLED_BY_HUMAN = "cancelled_by_human"
    # The timer expired with nobody else answering, the wake-time freshness
    # recheck agreed, and synthesis was attempted (see synthesis_answers_question
    # and synthesis_posted for what it decided).
    EXPIRED_AND_PROCEEDED = "expired_and_proceeded"
    # The timer expired, but the wake-time recheck found the channel had been
    # disabled mid-wait, or that a newer escalation had already superseded
    # this one in the same channel -- so synthesis was never even attempted.
    STOOD_DOWN_ON_RECHECK = "stood_down_on_recheck"


class DecisionTrail(BaseModel):
    """Everything the gate decided about one message, and the numbers behind it.

    Produced by aura.proactive.gate and persisted by record_signal below.
    Every field past Stage 1 is optional because the gate short-circuits: a
    message that fails Stage 1 has no Stage 2 score, and None means "never
    evaluated" where 0.0 would falsely claim it was evaluated and scored
    zero.

    allow_inf_nan=False on every float is the second line of defence behind
    question_likeness's own guard, and it is not decorative: SQLite stores a
    NaN REAL as NULL and accepts +/-inf verbatim, so a non-finite score would
    become either a NOT NULL violation or a row that silently poisons every
    later comparison and average taken from this table.
    """

    verdict: GateVerdict
    stage1_score: float = Field(allow_inf_nan=False)
    stage1_passed: bool
    stage2_top_score: float | None = Field(default=None, allow_inf_nan=False)
    stage2_runner_up_score: float | None = Field(default=None, allow_inf_nan=False)
    stage2_gap: float | None = Field(default=None, allow_inf_nan=False)
    stage2_passed: bool | None = None
    cooldown_seconds_remaining: float | None = Field(default=None, allow_inf_nan=False)
    daily_count: int | None = None
    daily_cap: int | None = None

    @property
    def would_escalate(self) -> bool:
        """Whether this message would proceed to paid synthesis once Phase 2a-3 exists.

        The one place that reads the verdict as a yes/no, so no call site has
        to re-derive "eligible" from the individual stage flags and get it
        subtly wrong.
        """
        return self.verdict is GateVerdict.ELIGIBLE


class ProactiveSignal(DecisionTrail):
    """One persisted decision trail, as read back from the database.

    Carries no message content on purpose: guild_id, channel_id and
    message_id already resolve to a Discord permalink, so the original text
    stays where it was written instead of being copied into Aura's database
    -- the same reasoning CLAUDE.md applies to a Fact's origin reference.

    The two synthesis fields are populated by update_synthesis_outcome after
    the gate trail is first recorded, so they are None on a fresh row and stay
    None for any message that never reached synthesis (verdict != ELIGIBLE) or
    whose synthesis produced no result. The debug view reads that None as
    "synthesis did not run / produced nothing" rather than a false zero.

    grace_period_outcome is populated the same way, by update_grace_outcome,
    and follows the same split-write pattern one step earlier in the pipeline
    (see GracePeriodOutcome).
    """

    id: int
    guild_id: int
    channel_id: int
    message_id: int
    synthesis_answers_question: bool | None = None
    synthesis_posted: bool | None = None
    grace_period_outcome: GracePeriodOutcome | None = None
    created_at: datetime


async def verify_signal_schema(conn: aiosqlite.Connection) -> None:
    """Reconcile an existing proactive_signals table with the current shape.

    Called once at startup, after init_schema. `CREATE TABLE IF NOT EXISTS`
    cannot reshape a table that already exists, so an INSERT against a table
    from an older phase would fail at runtime -- one logged exception per
    message, forever, with the bot otherwise looking healthy. That is the exact
    class of grey area CLAUDE.md rules out, so the mismatch is reconciled once,
    loudly, here.

    The two older shapes are handled deliberately differently, because the
    changes differ in kind:

    * **Phase 2a-1 (no `verdict` column).** Rejected outright, not migrated.
      Its old `score` column held a one-sided similarity where `stage1_score`
      now holds a contrastive one; the two are not the same measurement, and
      silently carrying the rows forward would poison Phase 2b's recalibration
      set. Dropping them is a call for whoever can see how much data a given
      deployment has, so this fails with an actionable message instead.

    * **Phase 2a-2 (has `verdict`, lacks the synthesis columns) and Phase
      2a-3 (has the synthesis columns, lacks `grace_period_outcome`).** Both
      migrated in place with ALTER TABLE ADD COLUMN. Both changes are purely
      additive -- no existing column's meaning changes -- so the collected
      data stays valid and there is nothing to force an operator to decide.

    A database with no such table (a fresh deployment, or one init_schema just
    created with the current shape) passes untouched.
    """
    async with conn.execute("PRAGMA table_info(proactive_signals)") as cursor:
        columns = {row[1] for row in await cursor.fetchall()}

    if not columns:
        return

    if _TRAIL_MARKER_COLUMN not in columns:
        raise OutdatedDiagnosticTableError(
            "The proactive_signals table in this database has Phase 2a-1's shape "
            f"(no '{_TRAIL_MARKER_COLUMN}' column) and cannot hold a decision trail. It is "
            "disposable diagnostic scaffolding, and its old 'score' column held a different "
            "measurement than the current 'stage1_score', so the rows cannot be carried "
            "forward. Drop it and restart: "
            'sqlite3 data/aura.db "DROP TABLE proactive_signals;"'
        )

    # Additive Phase 2a-3 and 2b-1 migrations. Each ADD COLUMN is guarded by
    # the membership check above, so this is idempotent: a table already at
    # the current shape adds nothing. A partially-migrated table (some but not
    # all columns present, e.g. a crash between ALTERs, or a 2a-3 table that
    # never saw 2b-1) is completed in one pass.
    async with connection_lock(conn):
        migrated = False
        for column in _SYNTHESIS_COLUMNS:
            if column not in columns:
                await conn.execute(
                    f"ALTER TABLE proactive_signals ADD COLUMN {column} INTEGER "
                    f"CHECK ({column} IN (0, 1))"
                )
                migrated = True
        if _GRACE_COLUMN not in columns:
            outcomes = ", ".join(f"'{outcome.value}'" for outcome in GracePeriodOutcome)
            await conn.execute(
                f"ALTER TABLE proactive_signals ADD COLUMN {_GRACE_COLUMN} TEXT "
                f"CHECK ({_GRACE_COLUMN} IN ({outcomes}))"
            )
            migrated = True
        if migrated:
            await conn.commit()


async def record_signal(
    conn: aiosqlite.Connection,
    *,
    guild_id: int,
    channel_id: int,
    message_id: int,
    decision: DecisionTrail,
) -> None:
    """Record one message's decision trail, ignoring a message already recorded.

    Duplicates are absorbed by name -- ON CONFLICT on the
    (channel_id, message_id) constraint -- and not with a blanket
    INSERT OR IGNORE, which suppresses *every* constraint violation on the
    row, NOT NULL and CHECK included. That difference is not academic: a
    non-finite score is written by SQLite as NULL, so under OR IGNORE such a
    row would disappear with no row, no error and no log line, which is the
    same silently-skewed table the uniqueness constraint exists to prevent,
    arriving through a different door. A redelivered gateway event is the
    one conflict worth swallowing here; anything else must surface.

    Keeping the first row rather than overwriting it is what makes the trail
    honest about a redelivery: the first evaluation is the one that decided
    something (and possibly spent a slot), while the second sees a ledger
    that already contains itself and can only conclude DUPLICATE_DELIVERY.
    Recording that second look would replace a real decision with an artefact
    of Discord's retry behaviour.

    Non-finite scores are rejected by DecisionTrail's own field constraints
    before they can reach this function; see that model's docstring.
    """
    async with connection_lock(conn):
        await conn.execute(
            """
            INSERT INTO proactive_signals
                (guild_id, channel_id, message_id, stage1_score, stage1_passed,
                 stage2_top_score, stage2_runner_up_score, stage2_gap, stage2_passed,
                 cooldown_seconds_remaining, daily_count, daily_cap, verdict, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (channel_id, message_id) DO NOTHING
            """,
            (
                guild_id,
                channel_id,
                message_id,
                decision.stage1_score,
                decision.stage1_passed,
                decision.stage2_top_score,
                decision.stage2_runner_up_score,
                decision.stage2_gap,
                decision.stage2_passed,
                decision.cooldown_seconds_remaining,
                decision.daily_count,
                decision.daily_cap,
                decision.verdict,
                utc_now_iso(),
            ),
        )
        await conn.commit()


async def update_synthesis_outcome(
    conn: aiosqlite.Connection,
    *,
    channel_id: int,
    message_id: int,
    answers_question: bool | None,
    posted: bool,
) -> None:
    """Record what synthesis decided for an already-recorded eligible message.

    Split from record_signal on purpose. The gate trail is written the instant
    the gate decides (record_signal), before the seconds-long synthesis call
    runs, so that a concurrent redelivery cannot win the (channel_id,
    message_id) ON CONFLICT and replace a real ELIGIBLE trail with a
    DUPLICATE_DELIVERY artefact while this message is still waiting on the LLM.
    This UPDATE then fills the synthesis fields onto that same row once the call
    returns.

    answers_question is None when synthesis produced no result at all (the LLM
    call failed, returned malformed output, or is not configured); it is a real
    bool when the model answered. posted is always definite -- True only if a
    message was actually sent. A row this UPDATE never reaches (synthesis
    crashed the whole handler, or the message was a concurrent duplicate whose
    ELIGIBLE trail another task recorded) simply keeps its NULL synthesis
    columns, which the debug view renders honestly as "no synthesis outcome".
    """
    async with connection_lock(conn):
        await conn.execute(
            """
            UPDATE proactive_signals
            SET synthesis_answers_question = ?, synthesis_posted = ?
            WHERE channel_id = ? AND message_id = ?
            """,
            (
                None if answers_question is None else int(answers_question),
                int(posted),
                channel_id,
                message_id,
            ),
        )
        await conn.commit()


async def update_grace_outcome(
    conn: aiosqlite.Connection,
    *,
    channel_id: int,
    message_id: int,
    outcome: GracePeriodOutcome,
) -> None:
    """Record Phase 2b-1's grace-period outcome for an already-recorded eligible message.

    Split from record_signal for the same reason update_synthesis_outcome is:
    the ELIGIBLE trail is written before the (now much longer) wait begins, so
    a concurrent redelivery cannot win the row and replace a real decision
    with a DUPLICATE_DELIVERY artefact while the wait is still running.

    Called up to twice per message: once with PENDING the instant the wait
    starts, and once more with whichever terminal outcome ends it. Called only
    for a message whose gate verdict is ELIGIBLE -- see GracePeriodOutcome.
    """
    async with connection_lock(conn):
        await conn.execute(
            "UPDATE proactive_signals SET grace_period_outcome = ? "
            "WHERE channel_id = ? AND message_id = ?",
            (outcome.value, channel_id, message_id),
        )
        await conn.commit()


async def get_recent_signals(
    conn: aiosqlite.Connection, *, guild_id: int, limit: int
) -> list[ProactiveSignal]:
    """Return guild_id's most recently recorded decision trails, newest first.

    Ordered by id rather than created_at: insertion order is what "most
    recent" means here, and two messages classified within the same
    microsecond would otherwise tie on their timestamp and come back in an
    arbitrary order.

    Rejects a negative limit instead of passing it to SQL, where LIMIT -1
    means *no limit at all* -- a caller that arrived at -1 by arithmetic
    would silently get every row in the table rather than the small page it
    asked for.
    """
    if limit < 0:
        raise ValueError(f"limit must not be negative, got {limit}")

    async with connection_lock(conn):
        async with conn.execute(
            f"""
            SELECT {_SIGNAL_COLUMNS} FROM proactive_signals
            WHERE guild_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (guild_id, limit),
        ) as cursor:
            rows = await cursor.fetchall()

    return [
        ProactiveSignal(
            id=row[0],
            guild_id=row[1],
            channel_id=row[2],
            message_id=row[3],
            stage1_score=row[4],
            stage1_passed=row[5],
            stage2_top_score=row[6],
            stage2_runner_up_score=row[7],
            stage2_gap=row[8],
            stage2_passed=row[9],
            cooldown_seconds_remaining=row[10],
            daily_count=row[11],
            daily_cap=row[12],
            synthesis_answers_question=None if row[13] is None else bool(row[13]),
            synthesis_posted=None if row[14] is None else bool(row[14]),
            grace_period_outcome=None if row[15] is None else GracePeriodOutcome(row[15]),
            verdict=GateVerdict(row[16]),
            created_at=row[17],
        )
        for row in rows
    ]
