"""The operator-wide brake across all five daily-cap ledgers (Phase 4a-2).

CLAUDE.md's own Open Items section named this gap before any code existed for
it: PROACTIVE_DAILY_CAP (and, since Phase 3a/3b/variant-indexing, its four
siblings -- EXTRACTION_DAILY_CAP, SUPERSESSION_DAILY_CAP, VARIANT_DAILY_CAP,
BACKFILL_DAILY_CAP) each bound one guild's worst-case daily spend, which is
the correct and complete answer when that guild brings its own OpenRouter key.
It stops being complete the moment many guilds share ONE operator-funded key --
which is exactly the shape Timur's flat-subscription plan needs -- because five
independent per-guild ceilings say nothing about their SUM.

Timur chose Option A from reports/phase-4a-multitenancy-audit.txt Section 4: a
flat, per-guild subscription price, not metered billing. That decision is what
shapes this module. A flat-fee operator does not need to know what any one
guild spent -- there is nothing to bill them individually for -- they need to
know whether TOTAL spend across every subscribed guild is still inside what
the subscription revenue actually covers. That is one number, not five, which
is why enforce_cross_guild_budget below combines all five ledgers into a
single estimate compared against a single budget, rather than giving each
ledger its own cross-guild ceiling the way the per-guild caps in config.py do.

**No new table, no new per-call cost measurement.** Every one of the five
ledgers already carries guild_id and a UTC day column (see
aura.db.{proactive,extraction,supersession,variant,backfill}_state) -- the
only new query this module needs is the SAME guarded COUNT those modules
already run, with the `WHERE guild_id = ?` clause dropped. Converting that
count into a dollar figure reuses the worst-case per-call cost each ledger's
own daily-cap field already documents in config.py, copied here as fixed
constants rather than measured per call -- a deliberately rough, conservative
brake for a single self-funded operator, not a billing system. See
_COST_PER_CALL_USD below for where each number comes from.

**Two modes, WARN by default.** A crossed budget is logged loudly either way.
Only HARD mode actually refuses new calls, and only once the combined
estimate is already at or over budget -- checked before the call that would
add to it, so the call that would push the total over is the one refused.
WARN is the default for the same reason Timur gave for keeping the loosened
Phase 2b-3/2b-4 proactive thresholds rather than tightening them back down
(see CLAUDE.md's "Proactive Relief: Visibly Active by Design"): for a single
operator subsidizing a handful of guilds, an accidental total outage across
every one of them is a worse failure than an observed, correctable cost
overrun. See CrossGuildBudgetMode's own docstring.
"""
from __future__ import annotations

import logging
from enum import StrEnum
from math import isfinite

import aiosqlite
from pydantic import BaseModel

from aura.config import CrossGuildBudgetMode
from aura.db.connection import connection_lock

logger = logging.getLogger(__name__)


class Ledger(StrEnum):
    """The five existing per-guild daily-cap ledgers, named the way call sites already do."""

    PROACTIVE = "proactive"
    EXTRACTION = "extraction"
    SUPERSESSION = "supersession"
    VARIANT = "variant"
    BACKFILL = "backfill"


# (table, UTC-day column) for each ledger. Copied from the INSERT statements in
# aura.db.{proactive,extraction,supersession,variant,backfill}_state rather
# than imported from them, because those modules expose typed acquire/count
# functions, not their raw table names -- and a raw table name is all a
# cross-guild COUNT needs.
_LEDGER_TABLES: dict[Ledger, tuple[str, str]] = {
    Ledger.PROACTIVE: ("proactive_escalations", "escalation_day"),
    Ledger.EXTRACTION: ("extraction_calls", "call_day"),
    Ledger.SUPERSESSION: ("supersession_calls", "call_day"),
    Ledger.VARIANT: ("variant_calls", "call_day"),
    Ledger.BACKFILL: ("backfill_calls", "call_day"),
}

# Rough, conservative worst-case USD cost per row -- copied as numbers from the
# reasoning already written out beside each ledger's own daily-cap field in
# config.py, not re-measured here. Kept in this one dict, beside the table
# mapping above, specifically so the two age together rather than drifting the
# way EXTRACTION_DEDUP_SIMILARITY_THRESHOLD's comment once silently did (it
# hardcoded "0.70" and went stale when the field was recalibrated to 0.60
# elsewhere -- see that field's own comment in config.py for the incident).
_COST_PER_CALL_USD: dict[Ledger, float] = {
    # PROACTIVE_DAILY_CAP's comment: "~$0.001-0.003 per Stage 3 call" at the
    # shipped model's pricing. The conservative (higher) end is used here,
    # since this module exists to be a brake, not an average.
    Ledger.PROACTIVE: 0.003,
    # EXTRACTION_DAILY_CAP's comment: "about $0.011" for a full
    # EXTRACTION_BATCH_MAX_MESSAGES batch at the shipped model's pricing.
    Ledger.EXTRACTION: 0.011,
    # SUPERSESSION_DAILY_CAP's comment: "about $0.001" per judgment call.
    Ledger.SUPERSESSION: 0.001,
    # VARIANT_DAILY_CAP's own comment calls the cost "negligible" without
    # giving a figure (variant generation runs at human click-speed, not
    # message speed, so the realistic per-day total is small regardless of
    # this constant). One episode is a generation call plus an independent
    # audit call, both short prompts structurally closer to supersession's
    # single judgment call than to extraction's full batch -- priced here as
    # 2x supersession's per-call cost rather than as zero, so this module's
    # worst-case math never silently drops a ledger to nothing.
    Ledger.VARIANT: 0.002,
    # BACKFILL_DAILY_CAP's comment: backfill spends the SAME distillation call
    # extraction does, at the same per-call cost.
    Ledger.BACKFILL: 0.011,
}


class LedgerSpend(BaseModel):
    """One ledger's cross-guild call count and rough estimated spend for one UTC day."""

    ledger: Ledger
    call_count: int
    estimated_usd: float


class CrossGuildBudgetStatus(BaseModel):
    """The operator-wide picture for one UTC day: every ledger's spend, and the combined total.

    Produced by get_cross_guild_status below and consumed both by
    enforce_cross_guild_budget (to decide whether a new call may proceed) and
    by /aura-operator-budget -- the same numbers either way, so what
    enforcement acts on and what the operator is shown are never two different
    computations that could quietly disagree.
    """

    day: str
    ledgers: list[LedgerSpend]
    total_estimated_usd: float
    budget_usd: float
    mode: CrossGuildBudgetMode

    @property
    def over_budget(self) -> bool:
        """Whether today's combined rough estimate has already cleared the operator's budget."""
        return self.total_estimated_usd > self.budget_usd


async def get_cross_guild_status(
    conn: aiosqlite.Connection,
    *,
    day: str,
    budget_usd: float,
    mode: CrossGuildBudgetMode,
) -> CrossGuildBudgetStatus:
    """Read-only: today's cross-guild call count and rough estimated spend, per ledger and combined.

    Five cheap COUNT(*) queries, one per ledger, each with no guild_id filter --
    the one structural difference from the per-guild counters each ledger
    module already exposes (count_escalations_on and its four siblings), which
    all filter on a specific guild_id. day is taken as a string produced by
    aura.db.connection.utc_day, matching every other ledger's own convention,
    so the caller's clock defines "today" rather than this function's.
    """
    if not isfinite(budget_usd) or budget_usd < 0:
        raise ValueError(f"budget_usd must be a finite number >= 0, got {budget_usd!r}")

    ledgers: list[LedgerSpend] = []
    async with connection_lock(conn):
        for ledger, (table, day_column) in _LEDGER_TABLES.items():
            # table and day_column come only from the fixed, hardcoded mapping
            # above -- never from a caller or from user input -- so building
            # this SQL string is safe despite not being parameterized itself;
            # `day` is still bound as a real parameter.
            query = f"SELECT COUNT(*) FROM {table} WHERE {day_column} = ?"
            async with conn.execute(query, (day,)) as cursor:
                row = await cursor.fetchone()
            count = int(row[0]) if row else 0
            ledgers.append(
                LedgerSpend(
                    ledger=ledger,
                    call_count=count,
                    estimated_usd=count * _COST_PER_CALL_USD[ledger],
                )
            )

    total = sum(entry.estimated_usd for entry in ledgers)
    return CrossGuildBudgetStatus(
        day=day,
        ledgers=ledgers,
        total_estimated_usd=total,
        budget_usd=budget_usd,
        mode=mode,
    )


async def enforce_cross_guild_budget(
    conn: aiosqlite.Connection,
    *,
    day: str,
    budget_usd: float,
    mode: CrossGuildBudgetMode,
) -> bool:
    """Whether a new call, at any of the five ledgers, may proceed right now.

    Call this BEFORE the ledger's own per-guild try_acquire_*_slot -- the same
    "claim before spending" ordering every ledger already uses internally, one
    layer up. Always returns True in WARN mode (the default): a crossed budget
    is logged loudly, at WARNING level, every time this is called while over
    budget, but nothing is refused, per this module's own docstring on why
    WARN is the safer default for a single self-funded operator.

    In HARD mode, returns False once today's combined estimate -- BEFORE this
    call -- is already at or above budget_usd, so the specific call that would
    tip the running total over the line is the one that gets refused, not one
    after it. The caller is responsible for treating False exactly like its
    own ledger's DAILY_CAP_REACHED refusal (drop, pause, or skip, per that
    call site's own existing behavior) -- this function never writes anything
    itself and never claims a slot on any ledger's behalf.

    Cheap by construction: five indexed COUNT(*) queries, no joins, run only
    at the point a message, batch, candidate, fact, or backfill tick has
    already cleared every earlier, free gate and is about to claim a real
    per-guild spend slot -- never on Aura's hot path for traffic that would
    have been rejected anyway.

    KNOWN LIMITATION, deliberate and documented rather than silently accepted:
    this is a plain read, separate in time from the per-guild acquire it
    guards, NOT one atomic operation the way each ledger's own guarded INSERT
    is internally. Two calls for two different guilds racing at the exact
    same instant can both read the same "before" total and both be granted,
    each then claiming its own per-guild slot -- HARD mode can be overshot by
    genuinely concurrent traffic across guilds, proven in
    test_cross_guild_budget.py's own concurrency test. Closing this
    completely would mean holding one lock across every one of the five
    ledgers' acquire calls for the whole life of this check, which does not
    fit this module's own design constraint (reuse the five existing ledgers
    exactly as they are, no shared table, no rework of their internals) for a
    brake whose own spec calls for a rough, conservative estimate rather than
    cent-exact enforcement. The overshoot this can produce is bounded by how
    many guilds genuinely race within one instant, not unbounded -- at the
    "handful of test and community servers" scale CLAUDE.md names as Aura's
    realistic near-term footprint, that number is small. Revisit if real
    guild counts make this bound worth tightening.
    """
    status = await get_cross_guild_status(conn, day=day, budget_usd=budget_usd, mode=mode)
    if status.over_budget:
        logger.warning(
            "Cross-guild operator budget exceeded for UTC day %s: estimated $%.2f across "
            "all guilds against a $%.2f budget (mode=%s). Per ledger: %s",
            day,
            status.total_estimated_usd,
            budget_usd,
            mode,
            ", ".join(f"{entry.ledger}={entry.call_count}" for entry in status.ledgers),
        )
        if mode is CrossGuildBudgetMode.HARD:
            return False
    return True
