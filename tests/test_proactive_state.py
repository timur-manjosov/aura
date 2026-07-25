"""Tests for aura.db.proactive_state: the durable cooldown and daily cap.

This is the file the whole sub-phase exists for. The mechanism it covers is
the one that will stand between an incoming message and a metered API call, so
it is tested the way something with money behind it should be: real SQLite
(never a mock -- the guarantees under test are the database's), genuinely
concurrent callers via asyncio.gather rather than sequential calls, real
file-backed databases reopened from scratch to simulate a container restart,
and injected clocks to land exactly on the boundaries that only happen once a
day.

No Discord anywhere in this file, and no embedding model: the budget logic is
pure data access and is verified as such.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import aiosqlite
import pytest

from aura.config import Settings
from aura.db.proactive_state import (
    MAX_COOLDOWN_SECONDS,
    MAX_DAILY_CAP,
    EscalationOutcome,
    _LedgerState,
    count_escalations_on,
    is_still_freshest_escalation,
    try_acquire_escalation_slot,
    utc_day,
)
from aura.db.repository import get_active_facts, init_schema

GUILD_A = 100000000000000001
GUILD_B = 200000000000000002
CHANNEL_1 = 5551
CHANNEL_2 = 5552

NOON = datetime(2026, 7, 24, 12, 0, 0, tzinfo=timezone.utc)
COOLDOWN = 900.0
CAP = 20


@pytest.fixture
async def conn():
    connection = await aiosqlite.connect(":memory:")
    await init_schema(connection)
    yield connection
    await connection.close()


async def _acquire(
    conn: aiosqlite.Connection,
    *,
    guild_id: int = GUILD_A,
    channel_id: int = CHANNEL_1,
    message_id: int = 1,
    cooldown_seconds: float = COOLDOWN,
    daily_cap: int = CAP,
    now: datetime = NOON,
):
    return await try_acquire_escalation_slot(
        conn,
        guild_id=guild_id,
        channel_id=channel_id,
        message_id=message_id,
        cooldown_seconds=cooldown_seconds,
        daily_cap=daily_cap,
        now=now,
    )


def _settings_upper_bound(field_name: str) -> float:
    """Return the `le=` constraint pydantic recorded for a Settings field.

    Searched rather than indexed: the constraint objects arrive in whatever
    order pydantic collected them, so metadata[0] is whichever of ge/le it
    happened to see first.
    """
    for constraint in Settings.model_fields[field_name].metadata:
        if hasattr(constraint, "le"):
            return constraint.le
    raise AssertionError(f"{field_name} declares no upper bound at all")


class TestUtcDay:
    def test_a_utc_moment_maps_to_its_own_date(self) -> None:
        assert utc_day(NOON) == "2026-07-24"

    def test_an_offset_moment_is_converted_before_the_date_is_read(self) -> None:
        # 01:30+05:30 is still 20:00 the previous day in UTC. Reading .date()
        # off the local value would file this under the wrong day and hand the
        # guild a second daily budget.
        moment = datetime(2026, 7, 25, 1, 30, tzinfo=ZoneInfo("Asia/Kolkata"))
        assert utc_day(moment) == "2026-07-24"

    def test_a_naive_datetime_is_rejected_rather_than_assumed_to_be_utc(self) -> None:
        with pytest.raises(ValueError):
            utc_day(datetime(2026, 7, 24, 12, 0))

    @pytest.mark.parametrize(
        "zone",
        ["Europe/Berlin", "America/New_York", "Australia/Lord_Howe", "Pacific/Kiritimati"],
    )
    def test_the_day_boundary_is_the_same_instant_in_every_timezone(self, zone: str) -> None:
        # The point of choosing UTC: every deployment, on any host, agrees on
        # when the cap resets. Lord Howe has a 30-minute DST shift and
        # Kiritimati is UTC+14, both of which would move a local boundary.
        boundary = datetime(2026, 7, 25, 0, 0, tzinfo=timezone.utc)
        local = boundary.astimezone(ZoneInfo(zone))
        assert utc_day(local) == "2026-07-25"
        assert utc_day(local - timedelta(microseconds=1)) == "2026-07-24"

    def test_dst_transitions_cannot_shorten_or_lengthen_a_capped_day(self) -> None:
        berlin = ZoneInfo("Europe/Berlin")

        # The hazard itself, demonstrated rather than asserted about: the local
        # calendar day of a spring-forward is only 23 hours long, so a cap
        # keyed to local days would cover an hour less on this day and an hour
        # more in autumn -- a spend limit that quietly varies twice a year.
        # Converted to UTC before subtracting, because Python ignores the
        # offset entirely when both operands share one tzinfo object -- the
        # difference of the two local midnights reads as a tidy 1 day until
        # you ask what actually elapsed.
        local_day = datetime(2026, 3, 30, tzinfo=berlin).astimezone(timezone.utc) - datetime(
            2026, 3, 29, tzinfo=berlin
        ).astimezone(timezone.utc)
        assert local_day == timedelta(hours=23)

        # Under UTC the same calendar day is exactly 24 hours, with the
        # boundary in one unambiguous place.
        midnight = datetime(2026, 3, 29, tzinfo=timezone.utc)
        assert utc_day(midnight) == "2026-03-29"
        assert utc_day(midnight + timedelta(hours=24, microseconds=-1)) == "2026-03-29"
        assert utc_day(midnight + timedelta(hours=24)) == "2026-03-30"

        # And the two sides of the local clock jump land in that same UTC day,
        # so nothing about the transition moves a guild's window.
        assert (
            utc_day(datetime(2026, 3, 29, 1, 30, tzinfo=berlin))
            == utc_day(datetime(2026, 3, 29, 3, 30, tzinfo=berlin))
            == "2026-03-29"
        )


class TestFirstAcquisition:
    async def test_the_first_eligible_message_gets_a_slot(
        self, conn: aiosqlite.Connection
    ) -> None:
        attempt = await _acquire(conn)

        assert attempt.outcome is EscalationOutcome.GRANTED
        assert attempt.granted is True
        assert attempt.daily_count == 1
        assert attempt.daily_cap == CAP
        assert attempt.cooldown_seconds_remaining == 0.0

    async def test_a_granted_slot_is_written_to_the_ledger(
        self, conn: aiosqlite.Connection
    ) -> None:
        await _acquire(conn, message_id=42)

        async with conn.execute(
            "SELECT guild_id, channel_id, message_id, escalation_day FROM proactive_escalations"
        ) as cursor:
            assert await cursor.fetchall() == [(GUILD_A, CHANNEL_1, 42, "2026-07-24")]

    async def test_a_refused_slot_writes_nothing(self, conn: aiosqlite.Connection) -> None:
        await _acquire(conn, message_id=1)
        await _acquire(conn, message_id=2)  # blocked by cooldown

        async with conn.execute("SELECT COUNT(*) FROM proactive_escalations") as cursor:
            assert await cursor.fetchone() == (1,)


class TestCooldown:
    async def test_a_second_message_in_the_same_channel_is_refused(
        self, conn: aiosqlite.Connection
    ) -> None:
        await _acquire(conn, message_id=1)
        attempt = await _acquire(conn, message_id=2, now=NOON + timedelta(seconds=60))

        assert attempt.outcome is EscalationOutcome.COOLDOWN_ACTIVE
        assert attempt.granted is False
        assert attempt.cooldown_seconds_remaining == pytest.approx(840.0)

    async def test_a_different_channel_is_not_affected(self, conn: aiosqlite.Connection) -> None:
        # The cooldown is about not spamming one conversation, not about
        # muting the whole server.
        await _acquire(conn, channel_id=CHANNEL_1, message_id=1)
        attempt = await _acquire(conn, channel_id=CHANNEL_2, message_id=2)

        assert attempt.outcome is EscalationOutcome.GRANTED

    async def test_the_same_channel_id_in_another_guild_is_still_the_same_channel(
        self, conn: aiosqlite.Connection
    ) -> None:
        # Channel IDs are globally unique snowflakes, so this can only happen
        # in a test -- asserted so the cooldown's scoping is deliberate rather
        # than incidental.
        await _acquire(conn, guild_id=GUILD_A, channel_id=CHANNEL_1, message_id=1)
        attempt = await _acquire(conn, guild_id=GUILD_B, channel_id=CHANNEL_1, message_id=2)

        assert attempt.outcome is EscalationOutcome.COOLDOWN_ACTIVE

    async def test_a_message_exactly_at_the_cooldown_expiry_is_allowed(
        self, conn: aiosqlite.Connection
    ) -> None:
        # The boundary is inclusive: strictly-greater-than in SQL means a row
        # exactly at the cutoff no longer blocks.
        await _acquire(conn, message_id=1)
        attempt = await _acquire(conn, message_id=2, now=NOON + timedelta(seconds=COOLDOWN))

        assert attempt.outcome is EscalationOutcome.GRANTED

    async def test_a_message_one_microsecond_before_expiry_is_refused(
        self, conn: aiosqlite.Connection
    ) -> None:
        await _acquire(conn, message_id=1)
        attempt = await _acquire(
            conn, message_id=2, now=NOON + timedelta(seconds=COOLDOWN, microseconds=-1)
        )

        assert attempt.outcome is EscalationOutcome.COOLDOWN_ACTIVE

    async def test_a_zero_cooldown_never_blocks(self, conn: aiosqlite.Connection) -> None:
        await _acquire(conn, message_id=1, cooldown_seconds=0.0)
        attempt = await _acquire(conn, message_id=2, cooldown_seconds=0.0)

        assert attempt.outcome is EscalationOutcome.GRANTED

    async def test_a_clock_that_jumped_backwards_fails_closed(
        self, conn: aiosqlite.Connection
    ) -> None:
        # NTP correcting a fast clock leaves ledger rows dated in the future.
        # The safe reading is "still on cooldown" -- never "spend again now".
        await _acquire(conn, message_id=1, now=NOON + timedelta(hours=1))
        attempt = await _acquire(conn, message_id=2, now=NOON)

        assert attempt.outcome is EscalationOutcome.COOLDOWN_ACTIVE
        # Reported as more than the whole cooldown, matching how long it will
        # actually be enforced, rather than a negative or clamped-to-zero lie.
        assert attempt.cooldown_seconds_remaining > COOLDOWN


class TestThreadScoping:
    async def test_a_thread_gets_its_own_cooldown_because_discord_gives_it_its_own_id(
        self, conn: aiosqlite.Connection
    ) -> None:
        # Intended, not incidental: a thread is a separate conversation. The
        # consequence is that the cooldown alone cannot bound spending, which
        # is what the guild-wide cap is for (asserted immediately below).
        parent_channel = CHANNEL_1
        thread = 777777

        await _acquire(conn, channel_id=parent_channel, message_id=1)
        in_thread = await _acquire(conn, channel_id=thread, message_id=2)

        assert in_thread.outcome is EscalationOutcome.GRANTED

    async def test_many_threads_still_cannot_exceed_the_guilds_daily_cap(
        self, conn: aiosqlite.Connection
    ) -> None:
        # The attack the per-thread cooldown would otherwise open: spin up a
        # thread per question and every one of them is off cooldown.
        results = await asyncio.gather(
            *(
                _acquire(conn, channel_id=700000 + index, message_id=index, daily_cap=4)
                for index in range(50)
            )
        )

        assert sum(attempt.granted for attempt in results) == 4


class TestDailyCap:
    async def test_the_cap_halts_escalation_once_reached(
        self, conn: aiosqlite.Connection
    ) -> None:
        # Deliberately engineered to exhaust the budget as fast as possible:
        # a distinct channel per message so the cooldown never interferes, and
        # nothing between them.
        for index in range(CAP):
            attempt = await _acquire(conn, channel_id=9000 + index, message_id=index)
            assert attempt.granted is True

        blocked = await _acquire(conn, channel_id=9999, message_id=999)

        assert blocked.outcome is EscalationOutcome.DAILY_CAP_REACHED
        assert blocked.daily_count == CAP
        assert blocked.daily_cap == CAP

    async def test_the_cap_keeps_holding_on_every_later_attempt(
        self, conn: aiosqlite.Connection
    ) -> None:
        # A cap that only refuses once, then leaks, is worse than no cap: the
        # logs would show it working.
        for index in range(CAP):
            await _acquire(conn, channel_id=9000 + index, message_id=index)

        for index in range(10):
            blocked = await _acquire(conn, channel_id=8000 + index, message_id=500 + index)
            assert blocked.outcome is EscalationOutcome.DAILY_CAP_REACHED

        assert await count_escalations_on(conn, guild_id=GUILD_A, day="2026-07-24") == CAP

    async def test_a_cap_of_zero_refuses_everything(self, conn: aiosqlite.Connection) -> None:
        attempt = await _acquire(conn, daily_cap=0)

        assert attempt.outcome is EscalationOutcome.DAILY_CAP_REACHED
        assert await count_escalations_on(conn, guild_id=GUILD_A, day="2026-07-24") == 0

    async def test_a_cap_of_one_allows_exactly_one(self, conn: aiosqlite.Connection) -> None:
        assert (await _acquire(conn, channel_id=CHANNEL_1, message_id=1, daily_cap=1)).granted
        second = await _acquire(conn, channel_id=CHANNEL_2, message_id=2, daily_cap=1)

        assert second.outcome is EscalationOutcome.DAILY_CAP_REACHED

    async def test_the_cap_is_scoped_per_guild_not_globally(
        self, conn: aiosqlite.Connection
    ) -> None:
        for index in range(CAP):
            await _acquire(conn, guild_id=GUILD_A, channel_id=9000 + index, message_id=index)

        other_guild = await _acquire(conn, guild_id=GUILD_B, channel_id=7777, message_id=777)

        assert other_guild.outcome is EscalationOutcome.GRANTED

    async def test_the_cap_is_shared_across_all_channels_of_one_guild(
        self, conn: aiosqlite.Connection
    ) -> None:
        # The specific mistake this guards: a cap accidentally keyed by
        # channel would give a 30-channel server 30x its budget.
        for index in range(CAP):
            await _acquire(conn, guild_id=GUILD_A, channel_id=9000 + index, message_id=index)

        assert await count_escalations_on(conn, guild_id=GUILD_A, day="2026-07-24") == CAP
        blocked = await _acquire(conn, guild_id=GUILD_A, channel_id=12345, message_id=4242)
        assert blocked.outcome is EscalationOutcome.DAILY_CAP_REACHED

    async def test_near_simultaneous_messages_across_many_channels_of_one_guild_share_the_cap(
        self, conn: aiosqlite.Connection
    ) -> None:
        # The attack: one user firing eligible messages into 40 different
        # channels at once, so the per-channel cooldown never engages and only
        # the guild-wide cap stands between them and 40 paid calls.
        results = await asyncio.gather(
            *(
                _acquire(conn, guild_id=GUILD_A, channel_id=9000 + index, message_id=index, daily_cap=5)
                for index in range(40)
            )
        )

        granted = [attempt for attempt in results if attempt.granted]
        assert len(granted) == 5
        assert await count_escalations_on(conn, guild_id=GUILD_A, day="2026-07-24") == 5

    async def test_two_guilds_hammering_at_once_each_get_their_own_budget(
        self, conn: aiosqlite.Connection
    ) -> None:
        results = await asyncio.gather(
            *(
                _acquire(
                    conn,
                    guild_id=GUILD_A if index % 2 == 0 else GUILD_B,
                    channel_id=9000 + index,
                    message_id=index,
                    daily_cap=3,
                )
                for index in range(40)
            )
        )

        assert sum(attempt.granted for attempt in results) == 6  # 3 per guild
        assert await count_escalations_on(conn, guild_id=GUILD_A, day="2026-07-24") == 3
        assert await count_escalations_on(conn, guild_id=GUILD_B, day="2026-07-24") == 3


class TestDailyReset:
    async def test_the_cap_resets_on_the_next_utc_day(self, conn: aiosqlite.Connection) -> None:
        for index in range(CAP):
            await _acquire(conn, channel_id=9000 + index, message_id=index)
        assert (await _acquire(conn, channel_id=1, message_id=900)).granted is False

        tomorrow = NOON + timedelta(days=1)
        attempt = await _acquire(conn, channel_id=1, message_id=901, now=tomorrow)

        assert attempt.outcome is EscalationOutcome.GRANTED
        assert attempt.daily_count == 1  # a fresh day, not 21

    async def test_the_reset_happens_exactly_at_midnight_utc(
        self, conn: aiosqlite.Connection
    ) -> None:
        # The reset moment itself, from both sides, one microsecond apart.
        last_moment = datetime(2026, 7, 24, 23, 59, 59, 999999, tzinfo=timezone.utc)
        first_moment = datetime(2026, 7, 25, 0, 0, 0, tzinfo=timezone.utc)

        for index in range(CAP):
            await _acquire(conn, channel_id=9000 + index, message_id=index, now=last_moment)

        still_capped = await _acquire(conn, channel_id=1, message_id=800, now=last_moment)
        assert still_capped.outcome is EscalationOutcome.DAILY_CAP_REACHED

        # A different channel, so only the cap (not the cooldown) is in play.
        after_reset = await _acquire(conn, channel_id=2, message_id=801, now=first_moment)
        assert after_reset.outcome is EscalationOutcome.GRANTED
        assert after_reset.daily_count == 1

    async def test_the_cooldown_does_not_reset_with_the_day(
        self, conn: aiosqlite.Connection
    ) -> None:
        # Two independent protections: rolling over into a new day must not
        # hand a channel permission to interrupt again seconds later.
        before_midnight = datetime(2026, 7, 24, 23, 59, 30, tzinfo=timezone.utc)
        after_midnight = datetime(2026, 7, 25, 0, 0, 30, tzinfo=timezone.utc)

        await _acquire(conn, message_id=1, now=before_midnight)
        attempt = await _acquire(conn, message_id=2, now=after_midnight)

        assert attempt.outcome is EscalationOutcome.COOLDOWN_ACTIVE

    async def test_a_non_utc_now_is_filed_under_the_correct_utc_day(
        self, conn: aiosqlite.Connection
    ) -> None:
        # The timestamp and the day key are derived from one `now`, and both
        # convert to UTC first. If only one of them did, a caller in a
        # non-UTC offset would write a row whose day key disagreed with its own
        # timestamp -- and the cap would be counting a different day than the
        # cooldown was measuring.
        kolkata = datetime(2026, 7, 25, 1, 30, tzinfo=ZoneInfo("Asia/Kolkata"))

        await _acquire(conn, now=kolkata)

        async with conn.execute(
            "SELECT escalated_at, escalation_day FROM proactive_escalations"
        ) as cursor:
            row = await cursor.fetchone()
        assert row is not None
        escalated_at, escalation_day = row
        assert escalation_day == "2026-07-24"  # 20:00 UTC the previous day
        assert escalated_at.startswith("2026-07-24T20:00:00")
        assert await count_escalations_on(conn, guild_id=GUILD_A, day=escalation_day) == 1

    async def test_yesterdays_rows_do_not_count_toward_today(
        self, conn: aiosqlite.Connection
    ) -> None:
        for index in range(CAP):
            await _acquire(conn, channel_id=9000 + index, message_id=index, now=NOON)

        assert await count_escalations_on(conn, guild_id=GUILD_A, day="2026-07-24") == CAP
        assert await count_escalations_on(conn, guild_id=GUILD_A, day="2026-07-25") == 0


class TestDuplicateDelivery:
    """Discord's gateway redelivers events after a resumed session."""

    async def test_a_redelivered_message_does_not_consume_a_second_slot(
        self, conn: aiosqlite.Connection
    ) -> None:
        first = await _acquire(conn, message_id=77)
        again = await _acquire(conn, message_id=77)

        assert first.outcome is EscalationOutcome.GRANTED
        assert again.outcome is EscalationOutcome.ALREADY_ESCALATED
        assert await count_escalations_on(conn, guild_id=GUILD_A, day="2026-07-24") == 1

    async def test_a_redelivery_is_reported_as_a_duplicate_not_as_a_cooldown_hit(
        self, conn: aiosqlite.Connection
    ) -> None:
        # The distinction matters for the debug trail: a cooldown verdict
        # would blame the member for a retry Discord performed.
        await _acquire(conn, message_id=77)
        again = await _acquire(conn, message_id=77, now=NOON + timedelta(seconds=1))

        assert again.outcome is EscalationOutcome.ALREADY_ESCALATED

    async def test_a_redelivery_after_the_cooldown_expired_is_still_a_duplicate(
        self, conn: aiosqlite.Connection
    ) -> None:
        # Without the uniqueness constraint this is the case that leaks: the
        # cooldown no longer blocks it, so the old message would escalate a
        # second time and spend a second slot.
        await _acquire(conn, message_id=77)
        much_later = NOON + timedelta(hours=5)

        again = await _acquire(conn, message_id=77, now=much_later)

        assert again.outcome is EscalationOutcome.ALREADY_ESCALATED
        assert await count_escalations_on(conn, guild_id=GUILD_A, day="2026-07-24") == 1

    async def test_a_redelivery_on_a_later_day_still_does_not_double_count(
        self, conn: aiosqlite.Connection
    ) -> None:
        await _acquire(conn, message_id=77, now=NOON)
        again = await _acquire(conn, message_id=77, now=NOON + timedelta(days=2))

        assert again.outcome is EscalationOutcome.ALREADY_ESCALATED
        assert await count_escalations_on(conn, guild_id=GUILD_A, day="2026-07-26") == 0

    async def test_twenty_concurrent_redeliveries_collapse_to_one_slot(
        self, conn: aiosqlite.Connection
    ) -> None:
        results = await asyncio.gather(*(_acquire(conn, message_id=77) for _ in range(20)))

        assert sum(attempt.granted for attempt in results) == 1
        assert await count_escalations_on(conn, guild_id=GUILD_A, day="2026-07-24") == 1

    async def test_the_same_message_id_in_two_channels_is_two_messages(
        self, conn: aiosqlite.Connection
    ) -> None:
        await _acquire(conn, channel_id=CHANNEL_1, message_id=77)
        other = await _acquire(conn, channel_id=CHANNEL_2, message_id=77)

        assert other.outcome is EscalationOutcome.GRANTED


class TestConcurrency:
    """Genuinely simultaneous callers, not the same call twice in a row."""

    async def test_only_one_of_fifty_concurrent_messages_in_a_channel_acquires_the_lock(
        self, conn: aiosqlite.Connection
    ) -> None:
        # The core race: discord.py dispatches every message event as its own
        # task, so a burst in one channel really does arrive as N coroutines
        # interleaving. A check-then-set implementation passes a sequential
        # test and fails this one.
        results = await asyncio.gather(
            *(_acquire(conn, message_id=index) for index in range(50))
        )

        granted = [attempt for attempt in results if attempt.granted]
        assert len(granted) == 1
        assert [attempt.outcome for attempt in results].count(
            EscalationOutcome.COOLDOWN_ACTIVE
        ) == 49
        async with conn.execute("SELECT COUNT(*) FROM proactive_escalations") as cursor:
            assert await cursor.fetchone() == (1,)

    async def test_the_cap_is_never_overshot_by_concurrent_callers(
        self, conn: aiosqlite.Connection
    ) -> None:
        # Every message in its own channel, so the cooldown never helps and
        # the cap alone has to hold under 100 simultaneous claims.
        results = await asyncio.gather(
            *(
                _acquire(conn, channel_id=9000 + index, message_id=index, daily_cap=7)
                for index in range(100)
            )
        )

        assert sum(attempt.granted for attempt in results) == 7
        async with conn.execute("SELECT COUNT(*) FROM proactive_escalations") as cursor:
            assert await cursor.fetchone() == (7,)

    async def test_no_granted_attempt_reports_a_count_above_the_cap(
        self, conn: aiosqlite.Connection
    ) -> None:
        results = await asyncio.gather(
            *(
                _acquire(conn, channel_id=9000 + index, message_id=index, daily_cap=5)
                for index in range(60)
            )
        )

        granted_counts = sorted(a.daily_count for a in results if a.granted)
        assert granted_counts == [1, 2, 3, 4, 5]  # each saw a distinct slot number

    async def test_a_burst_leaves_the_connection_with_no_open_transaction(
        self, conn: aiosqlite.Connection
    ) -> None:
        # A refused attempt must not leave a transaction dangling: the next
        # writer on this shared connection would then be inside it, and a
        # later rollback anywhere would discard that writer's committed work.
        await asyncio.gather(*(_acquire(conn, message_id=index) for index in range(30)))

        assert conn.in_transaction is False

    async def test_concurrent_bursts_in_different_guilds_do_not_interfere(
        self, conn: aiosqlite.Connection
    ) -> None:
        results = await asyncio.gather(
            *(
                _acquire(
                    conn,
                    guild_id=GUILD_A if index % 2 == 0 else GUILD_B,
                    channel_id=CHANNEL_1 if index % 2 == 0 else CHANNEL_2,
                    message_id=index,
                )
                for index in range(40)
            )
        )

        assert sum(attempt.granted for attempt in results) == 2  # one per channel


class TestTheGuardLivesInTheWrite:
    """The check must be inseparable from the write, not merely near it.

    The tests above prove the observable property under real concurrency, but
    they cannot show *which* mechanism provides it: the per-connection lock
    alone would be enough to pass them, and a lock is not enough in general --
    it does not exist between two processes sharing the database file, and it
    would not have existed at all if someone later moved the pre-read outside
    it. These tests remove the Python-level decision entirely, by making the
    pre-read report a stale "nothing blocks this" every time, so a refusal can
    only come from the WHERE clause of the INSERT itself.
    """

    @staticmethod
    def _blind_precheck(monkeypatch: pytest.MonkeyPatch) -> None:
        clear_reading = _LedgerState(
            already_escalated=False, last_escalated_at=None, daily_count=0
        )

        async def always_clear(*_args: object, **_kwargs: object) -> _LedgerState:
            return clear_reading

        monkeypatch.setattr("aura.db.proactive_state._read_state", always_clear)

    async def test_the_cooldown_is_enforced_by_the_insert_itself(
        self, conn: aiosqlite.Connection, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._blind_precheck(monkeypatch)

        first = await _acquire(conn, message_id=1)
        second = await _acquire(conn, message_id=2, now=NOON + timedelta(seconds=1))

        assert first.granted is True
        assert second.granted is False  # refused by SQL, not by the pre-read
        async with conn.execute("SELECT COUNT(*) FROM proactive_escalations") as cursor:
            assert await cursor.fetchone() == (1,)

    async def test_the_cap_is_enforced_by_the_insert_itself(
        self, conn: aiosqlite.Connection, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._blind_precheck(monkeypatch)

        granted = [
            (await _acquire(conn, channel_id=9000 + index, message_id=index, daily_cap=3)).granted
            for index in range(10)
        ]

        assert sum(granted) == 3
        async with conn.execute("SELECT COUNT(*) FROM proactive_escalations") as cursor:
            assert await cursor.fetchone() == (3,)

    async def test_a_duplicate_is_rejected_by_the_constraint_itself(
        self, conn: aiosqlite.Connection, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._blind_precheck(monkeypatch)

        await _acquire(conn, message_id=77)
        # Far past the cooldown, so only the uniqueness constraint can refuse.
        again = await _acquire(conn, message_id=77, now=NOON + timedelta(days=1))

        assert again.granted is False
        async with conn.execute("SELECT COUNT(*) FROM proactive_escalations") as cursor:
            assert await cursor.fetchone() == (1,)

    async def test_a_lost_race_is_logged_rather_than_passing_silently(
        self,
        conn: aiosqlite.Connection,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        # Reaching this branch means something outside this process wrote to
        # the ledger. Refusing is correct, but doing it silently would hide a
        # second writer on the database indefinitely.
        self._blind_precheck(monkeypatch)
        await _acquire(conn, message_id=1)

        with caplog.at_level(logging.WARNING):
            refused = await _acquire(conn, message_id=2, now=NOON + timedelta(seconds=1))

        assert refused.granted is False
        assert any(record.levelno >= logging.WARNING for record in caplog.records)

    async def test_a_lost_race_leaves_no_open_transaction(
        self, conn: aiosqlite.Connection, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The refused INSERT still opens a transaction; leaving it open would
        # put the next writer on this connection inside it.
        self._blind_precheck(monkeypatch)
        await _acquire(conn, message_id=1)
        await _acquire(conn, message_id=2, now=NOON + timedelta(seconds=1))

        assert conn.in_transaction is False


class TestRestartDurability:
    """A container restart must resume the protections, not reset them."""

    @staticmethod
    async def _open(path: Path) -> aiosqlite.Connection:
        connection = await aiosqlite.connect(path)
        await init_schema(connection)
        return connection

    async def test_a_cooldown_survives_a_restart_mid_window(self, tmp_path: Path) -> None:
        database = tmp_path / "aura.db"

        first = await self._open(database)
        try:
            assert (await _acquire(first, message_id=1)).granted is True
        finally:
            await first.close()

        # A brand-new connection with brand-new in-memory state, exactly as a
        # restarted container has.
        second = await self._open(database)
        try:
            attempt = await _acquire(second, message_id=2, now=NOON + timedelta(seconds=60))
            assert attempt.outcome is EscalationOutcome.COOLDOWN_ACTIVE
            assert attempt.cooldown_seconds_remaining == pytest.approx(840.0)
        finally:
            await second.close()

    async def test_the_daily_count_survives_a_restart_mid_day(self, tmp_path: Path) -> None:
        database = tmp_path / "aura.db"

        first = await self._open(database)
        try:
            for index in range(CAP - 1):
                await _acquire(first, channel_id=9000 + index, message_id=index)
        finally:
            await first.close()

        second = await self._open(database)
        try:
            last = await _acquire(second, channel_id=8000, message_id=800)
            assert last.granted is True
            assert last.daily_count == CAP  # resumed at 19, not at 0

            blocked = await _acquire(second, channel_id=8001, message_id=801)
            assert blocked.outcome is EscalationOutcome.DAILY_CAP_REACHED
        finally:
            await second.close()

    async def test_a_restart_loop_cannot_hand_out_extra_escalations(
        self, tmp_path: Path
    ) -> None:
        # The failure this whole design exists to prevent: with in-memory
        # state, a crash-looping container would grant one escalation per
        # restart while every log line still claimed the cap was enforced.
        database = tmp_path / "aura.db"

        for restart in range(10):
            connection = await self._open(database)
            try:
                await _acquire(
                    connection, channel_id=CHANNEL_1, message_id=restart, daily_cap=3
                )
            finally:
                await connection.close()

        connection = await self._open(database)
        try:
            assert await count_escalations_on(connection, guild_id=GUILD_A, day="2026-07-24") == 1
        finally:
            await connection.close()

    async def test_a_restart_after_the_cap_is_reached_stays_capped(
        self, tmp_path: Path
    ) -> None:
        database = tmp_path / "aura.db"

        first = await self._open(database)
        try:
            for index in range(3):
                await _acquire(first, channel_id=9000 + index, message_id=index, daily_cap=3)
        finally:
            await first.close()

        second = await self._open(database)
        try:
            blocked = await _acquire(second, channel_id=7000, message_id=700, daily_cap=3)
            assert blocked.outcome is EscalationOutcome.DAILY_CAP_REACHED
        finally:
            await second.close()

    async def test_a_restart_after_midnight_starts_the_new_day_clean(
        self, tmp_path: Path
    ) -> None:
        database = tmp_path / "aura.db"

        first = await self._open(database)
        try:
            for index in range(3):
                await _acquire(
                    first, channel_id=9000 + index, message_id=index, daily_cap=3, now=NOON
                )
        finally:
            await first.close()

        second = await self._open(database)
        try:
            attempt = await _acquire(
                second,
                channel_id=7000,
                message_id=700,
                daily_cap=3,
                now=NOON + timedelta(days=1),
            )
            assert attempt.outcome is EscalationOutcome.GRANTED
            assert attempt.daily_count == 1
        finally:
            await second.close()


class TestInputValidation:
    @pytest.mark.parametrize("cooldown", [-1.0, -0.001, float("nan"), float("inf")])
    async def test_a_nonsensical_cooldown_is_rejected(
        self, conn: aiosqlite.Connection, cooldown: float
    ) -> None:
        # A NaN cooldown would make every timestamp comparison false and
        # silently disable the cooldown entirely; an infinite one would format
        # into an unparseable cutoff string.
        with pytest.raises(ValueError):
            await _acquire(conn, cooldown_seconds=cooldown)

    async def test_a_negative_cap_is_rejected_rather_than_treated_as_zero(
        self, conn: aiosqlite.Connection
    ) -> None:
        with pytest.raises(ValueError):
            await _acquire(conn, daily_cap=-1)

    @pytest.mark.parametrize(
        "cooldown",
        [
            MAX_COOLDOWN_SECONDS + 1,
            1e11,  # far enough back that datetime itself has no such date
            1e15,  # far enough that timedelta overflows first, differently
        ],
    )
    async def test_an_absurdly_long_cooldown_is_refused_instead_of_overflowing(
        self, conn: aiosqlite.Connection, cooldown: float
    ) -> None:
        # `now - timedelta(seconds=cooldown)` raises rather than saturating, so
        # without this bound a single mistyped environment variable would raise
        # OverflowError on every message forever -- fail-closed, but as a
        # permanent traceback on a bot that otherwise looks healthy.
        with pytest.raises(ValueError, match="cooldown_seconds"):
            await _acquire(conn, cooldown_seconds=cooldown)

    @pytest.mark.parametrize("cap", [MAX_DAILY_CAP + 1, 2**63, 10**30])
    async def test_a_cap_beyond_sqlites_integer_range_is_refused(
        self, conn: aiosqlite.Connection, cap: int
    ) -> None:
        # sqlite3 refuses to bind a Python int that does not fit a signed
        # 64-bit integer, which would otherwise be an OverflowError per message.
        with pytest.raises(ValueError, match="daily_cap"):
            await _acquire(conn, daily_cap=cap)

    async def test_the_bounds_themselves_are_accepted(
        self, conn: aiosqlite.Connection
    ) -> None:
        # An inclusive bound that rejects its own limit would be an off-by-one
        # in the direction nobody tests.
        attempt = await _acquire(
            conn, cooldown_seconds=MAX_COOLDOWN_SECONDS, daily_cap=MAX_DAILY_CAP
        )
        assert attempt.granted is True

    def test_the_settings_bounds_agree_with_the_limits_enforced_here(self) -> None:
        # config.py states the same two limits as literals rather than
        # importing them (it must not depend on the data layer). This is what
        # keeps the two copies honest: a loosened bound in Settings would
        # otherwise let a value through to fail here, per message, instead.
        assert _settings_upper_bound("proactive_cooldown_seconds") == MAX_COOLDOWN_SECONDS
        assert _settings_upper_bound("proactive_daily_cap") == MAX_DAILY_CAP

    async def test_a_naive_now_is_rejected(self, conn: aiosqlite.Connection) -> None:
        with pytest.raises(ValueError):
            await _acquire(conn, now=datetime(2026, 7, 24, 12, 0))

    async def test_a_rejected_call_writes_nothing(self, conn: aiosqlite.Connection) -> None:
        for bad in ({"cooldown_seconds": -1.0}, {"daily_cap": -1}):
            with pytest.raises(ValueError):
                await _acquire(conn, **bad)  # type: ignore[arg-type]

        async with conn.execute("SELECT COUNT(*) FROM proactive_escalations") as cursor:
            assert await cursor.fetchone() == (0,)


class TestCountEscalations:
    async def test_an_empty_ledger_counts_zero(self, conn: aiosqlite.Connection) -> None:
        assert await count_escalations_on(conn, guild_id=GUILD_A, day="2026-07-24") == 0

    async def test_a_day_with_no_rows_counts_zero(self, conn: aiosqlite.Connection) -> None:
        await _acquire(conn)
        assert await count_escalations_on(conn, guild_id=GUILD_A, day="2026-01-01") == 0

    async def test_counting_is_scoped_to_one_guild(self, conn: aiosqlite.Connection) -> None:
        await _acquire(conn, guild_id=GUILD_A, channel_id=CHANNEL_1, message_id=1)
        await _acquire(conn, guild_id=GUILD_B, channel_id=CHANNEL_2, message_id=2)

        assert await count_escalations_on(conn, guild_id=GUILD_A, day="2026-07-24") == 1

    async def test_a_malformed_day_string_counts_zero_rather_than_matching_everything(
        self, conn: aiosqlite.Connection
    ) -> None:
        await _acquire(conn)
        assert await count_escalations_on(conn, guild_id=GUILD_A, day="not-a-day") == 0
        assert await count_escalations_on(conn, guild_id=GUILD_A, day="%") == 0


class TestKnowledgeModelIsolation:
    async def test_acquiring_slots_never_touches_the_knowledge_model(
        self, conn: aiosqlite.Connection
    ) -> None:
        for index in range(5):
            await _acquire(conn, channel_id=9000 + index, message_id=index)

        assert await get_active_facts(conn, GUILD_A) == []
        async with conn.execute("SELECT COUNT(*) FROM facts") as cursor:
            assert await cursor.fetchone() == (0,)

    async def test_the_ledger_has_no_foreign_key_into_the_knowledge_model(
        self, conn: aiosqlite.Connection
    ) -> None:
        async with conn.execute("PRAGMA foreign_key_list(proactive_escalations)") as cursor:
            assert await cursor.fetchall() == []

    async def test_the_ledger_stores_no_message_content(
        self, conn: aiosqlite.Connection
    ) -> None:
        # Same rule as the signals table: origin is referenced by ID, raw text
        # is never duplicated into Aura's database.
        async with conn.execute("PRAGMA table_info(proactive_escalations)") as cursor:
            columns = {row[1] for row in await cursor.fetchall()}

        assert columns == {
            "id",
            "guild_id",
            "channel_id",
            "message_id",
            "escalated_at",
            "escalation_day",
        }


class TestIsStillFreshestEscalation:
    """Phase 2b-1's wake-time recheck: has a newer grant superseded this one in its channel?"""

    async def test_the_only_escalation_in_its_channel_is_its_own_freshest(
        self, conn: aiosqlite.Connection
    ) -> None:
        await _acquire(conn, channel_id=CHANNEL_1, message_id=1)

        assert await is_still_freshest_escalation(
            conn, channel_id=CHANNEL_1, message_id=1
        ) is True

    async def test_a_later_grant_in_the_same_channel_supersedes_the_earlier_one(
        self, conn: aiosqlite.Connection
    ) -> None:
        # Only reachable with cooldown_seconds shorter than would normally be
        # configured relative to the grace period -- exactly the
        # misconfiguration this recheck exists to guard against (see
        # aura.proactive.listener._still_fresh_enough_for_synthesis).
        await _acquire(conn, channel_id=CHANNEL_1, message_id=1, now=NOON)
        await _acquire(
            conn,
            channel_id=CHANNEL_1,
            message_id=2,
            cooldown_seconds=0.0,
            now=NOON + timedelta(seconds=1),
        )

        assert await is_still_freshest_escalation(
            conn, channel_id=CHANNEL_1, message_id=1
        ) is False
        assert await is_still_freshest_escalation(
            conn, channel_id=CHANNEL_1, message_id=2
        ) is True

    async def test_a_later_grant_in_a_different_channel_does_not_affect_this_one(
        self, conn: aiosqlite.Connection
    ) -> None:
        await _acquire(conn, channel_id=CHANNEL_1, message_id=1, now=NOON)
        await _acquire(conn, channel_id=CHANNEL_2, message_id=2, now=NOON + timedelta(seconds=1))

        assert await is_still_freshest_escalation(
            conn, channel_id=CHANNEL_1, message_id=1
        ) is True

    async def test_a_later_grant_in_a_different_guild_does_not_affect_this_one(
        self, conn: aiosqlite.Connection
    ) -> None:
        # Same channel_id reused across guilds would be a Discord impossibility
        # (channel IDs are globally unique), but the query keys on channel_id
        # alone, so this is worth proving directly rather than assumed.
        await _acquire(conn, guild_id=GUILD_A, channel_id=CHANNEL_1, message_id=1, now=NOON)
        await _acquire(
            conn,
            guild_id=GUILD_B,
            channel_id=CHANNEL_2,
            message_id=2,
            now=NOON + timedelta(seconds=1),
        )

        assert await is_still_freshest_escalation(
            conn, channel_id=CHANNEL_1, message_id=1
        ) is True

    async def test_a_message_with_no_escalation_row_at_all_is_not_fresh(
        self, conn: aiosqlite.Connection
    ) -> None:
        # Unreachable in production (a granted slot is never deleted), but the
        # honest failure direction for a row this function cannot find.
        assert await is_still_freshest_escalation(
            conn, channel_id=CHANNEL_1, message_id=999
        ) is False
