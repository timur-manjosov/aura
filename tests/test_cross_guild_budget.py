"""Tests for aura.db.cross_guild_budget: Phase 4a-2's operator-wide brake.

Populates each of the five real ledgers through their OWN acquire functions
(never raw INSERTs) so this file is honest against the real schema and the
real per-guild guards those five modules already enforce -- a raw INSERT could
silently drift from what production actually writes. Real SQLite throughout,
matching every other ledger test in this project (see test_proactive_state.py's
own docstring for why): the guarantees under test are the database's.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

import aiosqlite
import pytest

from aura.config import CrossGuildBudgetMode
from aura.db.backfill_runs import start_backfill_run
from aura.db.backfill_state import try_acquire_backfill_call_slot
from aura.db.connection import utc_day
from aura.db.cross_guild_budget import (
    Ledger,
    enforce_cross_guild_budget,
    get_cross_guild_status,
)
from aura.db.extraction_state import try_acquire_extraction_call_slot
from aura.db.pending_facts import FactCategory, stage_pending_fact
from aura.db.proactive_state import try_acquire_escalation_slot
from aura.db.repository import init_schema
from aura.db.supersession_state import try_acquire_supersession_call_slot
from aura.db.variant_state import try_acquire_variant_call_slot
from aura.facts_service import add_fact

GUILD_A = 100000000000000001
GUILD_B = 200000000000000002
GUILD_C = 300000000000000003
MODERATOR = 4242
UNTIL = 900000000000000000

NOON = datetime(2026, 9, 12, 12, 0, 0, tzinfo=timezone.utc)
DAY = utc_day(NOON)
TOMORROW = datetime(2026, 9, 13, 12, 0, 0, tzinfo=timezone.utc)

EMBEDDING = b"\x00" * 16


@pytest.fixture
async def conn():
    connection = await aiosqlite.connect(":memory:")
    await init_schema(connection)
    yield connection
    await connection.close()


async def _spend_proactive(conn: aiosqlite.Connection, *, guild_id: int, message_id: int) -> None:
    attempt = await try_acquire_escalation_slot(
        conn,
        guild_id=guild_id,
        channel_id=1,
        message_id=message_id,
        cooldown_seconds=0.0,
        daily_cap=1_000_000,
        now=NOON,
    )
    assert attempt.granted


async def _spend_extraction(conn: aiosqlite.Connection, *, guild_id: int, channel_id: int) -> None:
    attempt = await try_acquire_extraction_call_slot(
        conn, guild_id=guild_id, channel_id=channel_id, message_count=5,
        daily_cap=1_000_000, now=NOON,
    )
    assert attempt.granted


async def _spend_supersession(
    conn: aiosqlite.Connection, *, guild_id: int, message_id: int = 1
) -> None:
    """Stage a real candidate first -- supersession_calls.pending_fact_id is a
    REFERENCES constraint, and PRAGMA foreign_keys is ON (see init_schema)."""
    staged = await stage_pending_fact(
        conn,
        guild_id=guild_id,
        channel_id=1,
        message_id=message_id,
        content=f"A candidate sentence #{message_id}.",
        embedding=EMBEDDING,
        category=FactCategory.RULE,
    )
    assert staged is not None
    attempt = await try_acquire_supersession_call_slot(
        conn, guild_id=guild_id, pending_fact_id=staged.id,
        daily_cap=1_000_000, now=NOON,
    )
    assert attempt.granted


async def _spend_variant(
    conn: aiosqlite.Connection, embedding_model, *, guild_id: int, message_id: int = 1
) -> None:
    """Create a real active fact first -- variant_calls.fact_id is a REFERENCES
    constraint, the same reasoning _spend_supersession's own comment gives."""
    fact = await add_fact(
        conn, embedding_model, guild_id=guild_id, channel_id=1,
        message_id=message_id, content=f"fact number {message_id}",
    )
    attempt = await try_acquire_variant_call_slot(
        conn, guild_id=guild_id, fact_id=fact.id, daily_cap=1_000_000, now=NOON,
    )
    assert attempt.granted


async def _spend_backfill(
    conn: aiosqlite.Connection, *, guild_id: int, channel_id: int = 1
) -> None:
    """Start a real run first -- backfill_calls.run_id is a REFERENCES constraint,
    the same reasoning _spend_supersession's own comment gives."""
    run = await start_backfill_run(
        conn, guild_id=guild_id, channel_id=channel_id, until_message_id=UNTIL,
        after_message_id=None, requested_by_id=MODERATOR, now=NOON,
    )
    attempt = await try_acquire_backfill_call_slot(
        conn, guild_id=guild_id, run_id=run.id, message_count=5,
        daily_cap=1_000_000, now=NOON,
    )
    assert attempt.granted


class TestGetCrossGuildStatusOnAnEmptyDatabase:
    async def test_every_ledger_reads_zero(self, conn: aiosqlite.Connection) -> None:
        status = await get_cross_guild_status(
            conn, day=DAY, budget_usd=10.0, mode=CrossGuildBudgetMode.WARN
        )
        assert status.total_estimated_usd == 0.0
        assert {entry.ledger: entry.call_count for entry in status.ledgers} == {
            Ledger.PROACTIVE: 0,
            Ledger.EXTRACTION: 0,
            Ledger.SUPERSESSION: 0,
            Ledger.VARIANT: 0,
            Ledger.BACKFILL: 0,
        }
        assert not status.over_budget

    async def test_enforce_allows_in_both_modes_when_nothing_has_spent(
        self, conn: aiosqlite.Connection
    ) -> None:
        for mode in (CrossGuildBudgetMode.WARN, CrossGuildBudgetMode.HARD):
            assert await enforce_cross_guild_budget(
                conn, day=DAY, budget_usd=0.0, mode=mode
            )


class TestCrossGuildSummingIsActuallyCrossGuild:
    """The one property this whole module exists for: the count is NOT filtered by guild_id."""

    async def test_spend_from_three_different_guilds_on_one_ledger_all_counts(
        self, conn: aiosqlite.Connection
    ) -> None:
        await _spend_proactive(conn, guild_id=GUILD_A, message_id=1)
        await _spend_proactive(conn, guild_id=GUILD_B, message_id=2)
        await _spend_proactive(conn, guild_id=GUILD_C, message_id=3)

        status = await get_cross_guild_status(
            conn, day=DAY, budget_usd=1000.0, mode=CrossGuildBudgetMode.WARN
        )
        proactive = next(e for e in status.ledgers if e.ledger is Ledger.PROACTIVE)
        assert proactive.call_count == 3

    async def test_a_single_guild_saturating_its_own_cap_still_counts_toward_the_total(
        self, conn: aiosqlite.Connection
    ) -> None:
        # One guild alone can trip the cross-guild total; nothing here requires
        # more than one guild to matter.
        for message_id in range(1, 6):
            await _spend_proactive(conn, guild_id=GUILD_A, message_id=message_id)

        status = await get_cross_guild_status(
            conn, day=DAY, budget_usd=1000.0, mode=CrossGuildBudgetMode.WARN
        )
        proactive = next(e for e in status.ledgers if e.ledger is Ledger.PROACTIVE)
        assert proactive.call_count == 5

    async def test_all_five_ledgers_combine_into_one_total(
        self, conn: aiosqlite.Connection, embedding_model
    ) -> None:
        await _spend_proactive(conn, guild_id=GUILD_A, message_id=1)
        await _spend_extraction(conn, guild_id=GUILD_A, channel_id=1)
        await _spend_supersession(conn, guild_id=GUILD_B)
        await _spend_variant(conn, embedding_model, guild_id=GUILD_B)
        await _spend_backfill(conn, guild_id=GUILD_C)

        status = await get_cross_guild_status(
            conn, day=DAY, budget_usd=1000.0, mode=CrossGuildBudgetMode.WARN
        )
        counts = {entry.ledger: entry.call_count for entry in status.ledgers}
        assert counts == {
            Ledger.PROACTIVE: 1,
            Ledger.EXTRACTION: 1,
            Ledger.SUPERSESSION: 1,
            Ledger.VARIANT: 1,
            Ledger.BACKFILL: 1,
        }
        # 0.003 + 0.011 + 0.001 + 0.002 + 0.011 -- see _COST_PER_CALL_USD.
        assert status.total_estimated_usd == pytest.approx(0.028)

    async def test_a_different_utc_day_does_not_contribute(
        self, conn: aiosqlite.Connection
    ) -> None:
        await _spend_proactive(conn, guild_id=GUILD_A, message_id=1)

        status_tomorrow = await get_cross_guild_status(
            conn, day=utc_day(TOMORROW), budget_usd=1000.0, mode=CrossGuildBudgetMode.WARN
        )
        assert status_tomorrow.total_estimated_usd == 0.0


class TestWarnModeNeverRefuses:
    async def test_enforce_returns_true_even_wildly_over_budget(
        self, conn: aiosqlite.Connection
    ) -> None:
        for message_id in range(1, 21):
            await _spend_proactive(conn, guild_id=GUILD_A, message_id=message_id)

        assert await enforce_cross_guild_budget(
            conn, day=DAY, budget_usd=0.0, mode=CrossGuildBudgetMode.WARN
        )

    async def test_being_over_budget_in_warn_mode_still_logs_a_warning(
        self, conn: aiosqlite.Connection, caplog: pytest.LogCaptureFixture
    ) -> None:
        await _spend_proactive(conn, guild_id=GUILD_A, message_id=1)

        with caplog.at_level(logging.WARNING, logger="aura.db.cross_guild_budget"):
            await enforce_cross_guild_budget(
                conn, day=DAY, budget_usd=0.0, mode=CrossGuildBudgetMode.WARN
            )
        assert any("Cross-guild operator budget exceeded" in record.message for record in caplog.records)

    async def test_being_under_budget_logs_nothing(
        self, conn: aiosqlite.Connection, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING, logger="aura.db.cross_guild_budget"):
            await enforce_cross_guild_budget(
                conn, day=DAY, budget_usd=1000.0, mode=CrossGuildBudgetMode.WARN
            )
        assert caplog.records == []


class TestHardModeActuallyRefuses:
    async def test_enforce_returns_false_once_over_budget(
        self, conn: aiosqlite.Connection
    ) -> None:
        await _spend_proactive(conn, guild_id=GUILD_A, message_id=1)  # $0.003

        assert not await enforce_cross_guild_budget(
            conn, day=DAY, budget_usd=0.002, mode=CrossGuildBudgetMode.HARD
        )

    async def test_exactly_at_budget_is_still_allowed_strictly_over_is_not(
        self, conn: aiosqlite.Connection
    ) -> None:
        await _spend_proactive(conn, guild_id=GUILD_A, message_id=1)  # $0.003 exactly

        assert await enforce_cross_guild_budget(
            conn, day=DAY, budget_usd=0.003, mode=CrossGuildBudgetMode.HARD
        )
        assert not await enforce_cross_guild_budget(
            conn, day=DAY, budget_usd=0.0029999, mode=CrossGuildBudgetMode.HARD
        )

    async def test_a_zero_budget_with_nothing_spent_yet_is_still_allowed(
        self, conn: aiosqlite.Connection
    ) -> None:
        # over_budget is strictly-greater-than, so $0.00 spent against a $0.00
        # budget is not yet "over" -- the FIRST call of the day is never
        # refused by this brake alone (only a call that would make the running
        # total exceed budget is).
        assert await enforce_cross_guild_budget(
            conn, day=DAY, budget_usd=0.0, mode=CrossGuildBudgetMode.HARD
        )

    async def test_the_per_guild_cap_still_applies_independently_of_this_brake(
        self, conn: aiosqlite.Connection
    ) -> None:
        # This module never touches any ledger's own per-guild guard. A guild
        # comfortably under the cross-guild budget can still be refused by its
        # own daily_cap, unrelated to anything here.
        attempt = await try_acquire_escalation_slot(
            conn, guild_id=GUILD_A, channel_id=1, message_id=1,
            cooldown_seconds=0.0, daily_cap=0, now=NOON,
        )
        assert not attempt.granted

        assert await enforce_cross_guild_budget(
            conn, day=DAY, budget_usd=1000.0, mode=CrossGuildBudgetMode.HARD
        )

    async def test_a_different_utc_day_resets_the_hard_refusal(
        self, conn: aiosqlite.Connection
    ) -> None:
        await _spend_proactive(conn, guild_id=GUILD_A, message_id=1)
        assert not await enforce_cross_guild_budget(
            conn, day=DAY, budget_usd=0.0, mode=CrossGuildBudgetMode.HARD
        )
        assert await enforce_cross_guild_budget(
            conn, day=utc_day(TOMORROW), budget_usd=0.0, mode=CrossGuildBudgetMode.HARD
        )


class TestAdversarialInput:
    async def test_negative_budget_is_rejected(self, conn: aiosqlite.Connection) -> None:
        with pytest.raises(ValueError, match="budget_usd"):
            await get_cross_guild_status(
                conn, day=DAY, budget_usd=-1.0, mode=CrossGuildBudgetMode.WARN
            )

    async def test_nan_budget_is_rejected(self, conn: aiosqlite.Connection) -> None:
        with pytest.raises(ValueError, match="budget_usd"):
            await get_cross_guild_status(
                conn, day=DAY, budget_usd=float("nan"), mode=CrossGuildBudgetMode.WARN
            )

    async def test_infinite_budget_is_rejected(self, conn: aiosqlite.Connection) -> None:
        with pytest.raises(ValueError, match="budget_usd"):
            await get_cross_guild_status(
                conn, day=DAY, budget_usd=float("inf"), mode=CrossGuildBudgetMode.WARN
            )

    async def test_an_empty_day_string_is_simply_never_matched(
        self, conn: aiosqlite.Connection
    ) -> None:
        # Not a crash: an unrecognized day key just matches nothing, the same
        # honest-zero behavior every ledger's own day-keyed COUNT already has.
        await _spend_proactive(conn, guild_id=GUILD_A, message_id=1)
        status = await get_cross_guild_status(
            conn, day="", budget_usd=1000.0, mode=CrossGuildBudgetMode.WARN
        )
        assert status.total_estimated_usd == 0.0


class TestConcurrentAccessAcrossGuildsAndLedgers:
    """Real asyncio.gather, matching every other ledger test's own convention.

    This module itself never writes -- there is nothing here for two
    concurrent callers to race INTO -- so what matters is that a burst of
    concurrent spends across every ledger and several guilds, interleaved with
    concurrent reads of the combined status, never raises and always reports a
    total consistent with what was actually granted.
    """

    async def test_concurrent_spends_across_guilds_and_ledgers_sum_correctly(
        self, conn: aiosqlite.Connection
    ) -> None:
        tasks = []
        for i in range(10):
            tasks.append(_spend_proactive(conn, guild_id=GUILD_A, message_id=i))
        for i in range(10, 20):
            tasks.append(_spend_extraction(conn, guild_id=GUILD_B, channel_id=i))
        for i in range(20, 30):
            tasks.append(_spend_backfill(conn, guild_id=GUILD_C, channel_id=i))

        await asyncio.gather(*tasks)

        status = await get_cross_guild_status(
            conn, day=DAY, budget_usd=1000.0, mode=CrossGuildBudgetMode.WARN
        )
        counts = {entry.ledger: entry.call_count for entry in status.ledgers}
        assert counts[Ledger.PROACTIVE] == 10
        assert counts[Ledger.EXTRACTION] == 10
        assert counts[Ledger.BACKFILL] == 10

    async def test_the_hard_gate_can_be_overshot_by_genuinely_simultaneous_guilds(
        self, conn: aiosqlite.Connection
    ) -> None:
        """Documents a real, accepted limitation rather than hiding it.

        enforce_cross_guild_budget is a plain read, separate in time from the
        per-guild acquire it guards -- not one atomic operation the way each
        ledger's own guarded INSERT is. Many different guilds checking and
        then spending at the exact same instant can all read the same
        "before" total and all be granted, overshooting a HARD budget that
        would have refused them one at a time. See the module docstring's own
        "KNOWN LIMITATION" note for why this is accepted rather than fixed:
        closing it would mean holding one lock across all five ledgers' own
        acquire calls, which this module's design deliberately does not do.
        """
        budget = 0.0025  # under even a single $0.003 proactive escalation

        async def _race_one_guild(guild_id: int, message_id: int) -> bool:
            allowed = await enforce_cross_guild_budget(
                conn, day=DAY, budget_usd=budget, mode=CrossGuildBudgetMode.HARD
            )
            if allowed:
                await _spend_proactive(conn, guild_id=guild_id, message_id=message_id)
            return allowed

        results = await asyncio.gather(
            *(_race_one_guild(1_000_000 + i, i) for i in range(10))
        )

        # The bug this test exists to prove: more than one guild raced past a
        # budget only one spend should have cleared.
        assert sum(results) > 1

        # Not unbounded, though: a later, sequential call still sees the real
        # total and is refused, exactly as designed once the race is over.
        assert not await enforce_cross_guild_budget(
            conn, day=DAY, budget_usd=budget, mode=CrossGuildBudgetMode.HARD
        )

    async def test_concurrent_enforce_calls_never_raise_and_agree_on_over_budget(
        self, conn: aiosqlite.Connection
    ) -> None:
        for message_id in range(1, 11):
            await _spend_proactive(conn, guild_id=GUILD_A, message_id=message_id)

        results = await asyncio.gather(
            *(
                enforce_cross_guild_budget(
                    conn, day=DAY, budget_usd=0.01, mode=CrossGuildBudgetMode.HARD
                )
                for _ in range(20)
            )
        )
        # Nothing here claims a slot, so every concurrent read sees the same
        # already-committed state and must agree with itself.
        assert all(result is False for result in results)
