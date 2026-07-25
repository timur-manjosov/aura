"""Tests for aura.proactive.grace: Phase 2b-1's in-memory grace-period tracker.

Exercises GraceRegistry directly, independent of the listener and of Discord,
per CLAUDE.md's testing principle -- what is under test here is the
cancellation and timing logic itself, not how the listener wires it up (see
test_proactive_listener.py for that).

Real, short asyncio.sleep durations are used throughout rather than mocking
the clock: this module's entire job is racing two coroutines against each
other, and a mocked clock would test the mock's model of that race, not the
real one.
"""
from __future__ import annotations

import asyncio

import pytest

from aura.proactive.grace import GraceRegistry, GraceWaitOutcome, _PendingGrace

CHANNEL_A = 111
CHANNEL_B = 222
ASKER = 1001
OTHER_HUMAN = 2002


class TestNoticeHumanMessage:
    """Cancellation is opt-in: only a genuinely different human, in the right channel, counts."""

    def test_a_miss_on_a_channel_with_nothing_pending_is_silently_a_no_op(self) -> None:
        registry = GraceRegistry()
        # Must not raise, and there is nothing to assert on besides that.
        registry.notice_human_message(channel_id=CHANNEL_A, author_id=OTHER_HUMAN, message_id=1)

    async def test_the_original_askers_own_message_does_not_cancel(self) -> None:
        registry = GraceRegistry()
        wait_task = asyncio.create_task(
            registry.wait(channel_id=CHANNEL_A, asker_id=ASKER, message_id=1, seconds=0.2)
        )
        await asyncio.sleep(0.05)

        registry.notice_human_message(channel_id=CHANNEL_A, author_id=ASKER, message_id=2)

        outcome = await wait_task
        assert outcome is GraceWaitOutcome.EXPIRED

    async def test_a_redelivery_of_the_original_message_does_not_cancel(self) -> None:
        registry = GraceRegistry()
        wait_task = asyncio.create_task(
            registry.wait(channel_id=CHANNEL_A, asker_id=ASKER, message_id=1, seconds=0.2)
        )
        await asyncio.sleep(0.05)

        # Same message_id as the one being waited on, even though the author
        # argument claims to be someone else -- this is what a gateway replay
        # of the original event looks like, not a reply to it.
        registry.notice_human_message(channel_id=CHANNEL_A, author_id=OTHER_HUMAN, message_id=1)

        outcome = await wait_task
        assert outcome is GraceWaitOutcome.EXPIRED

    async def test_a_genuinely_different_human_cancels(self) -> None:
        registry = GraceRegistry()
        wait_task = asyncio.create_task(
            registry.wait(channel_id=CHANNEL_A, asker_id=ASKER, message_id=1, seconds=5.0)
        )
        await asyncio.sleep(0.05)

        registry.notice_human_message(channel_id=CHANNEL_A, author_id=OTHER_HUMAN, message_id=2)

        outcome = await asyncio.wait_for(wait_task, timeout=1.0)
        assert outcome is GraceWaitOutcome.CANCELLED_BY_HUMAN

    async def test_a_different_human_in_a_different_channel_does_not_cancel(self) -> None:
        registry = GraceRegistry()
        wait_task = asyncio.create_task(
            registry.wait(channel_id=CHANNEL_A, asker_id=ASKER, message_id=1, seconds=0.2)
        )
        await asyncio.sleep(0.05)

        registry.notice_human_message(channel_id=CHANNEL_B, author_id=OTHER_HUMAN, message_id=99)

        outcome = await wait_task
        assert outcome is GraceWaitOutcome.EXPIRED


class TestNoticeMessageGone:
    """A deleted or edited message stands its own grace period down, regardless of author."""

    async def test_deleting_the_watched_message_cancels_even_though_no_author_is_given(
        self,
    ) -> None:
        registry = GraceRegistry()
        wait_task = asyncio.create_task(
            registry.wait(channel_id=CHANNEL_A, asker_id=ASKER, message_id=1, seconds=5.0)
        )
        await asyncio.sleep(0.05)

        registry.notice_message_gone(channel_id=CHANNEL_A, message_id=1)

        outcome = await asyncio.wait_for(wait_task, timeout=1.0)
        assert outcome is GraceWaitOutcome.CANCELLED_BY_HUMAN

    async def test_a_different_messages_deletion_does_not_cancel(self) -> None:
        registry = GraceRegistry()
        wait_task = asyncio.create_task(
            registry.wait(channel_id=CHANNEL_A, asker_id=ASKER, message_id=1, seconds=0.2)
        )
        await asyncio.sleep(0.05)

        registry.notice_message_gone(channel_id=CHANNEL_A, message_id=999)

        outcome = await wait_task
        assert outcome is GraceWaitOutcome.EXPIRED

    def test_nothing_pending_is_a_silent_no_op(self) -> None:
        registry = GraceRegistry()
        registry.notice_message_gone(channel_id=CHANNEL_A, message_id=1)


class TestExpiry:
    async def test_an_unbothered_wait_expires_normally(self) -> None:
        registry = GraceRegistry()

        outcome = await registry.wait(
            channel_id=CHANNEL_A, asker_id=ASKER, message_id=1, seconds=0.05
        )

        assert outcome is GraceWaitOutcome.EXPIRED

    async def test_a_tie_at_expiry_favors_cancellation_over_proceeding(self) -> None:
        # The exact race the phase's attack plan calls out: a human message
        # arriving in the same instant the timer would otherwise fire. Built
        # directly rather than timed, so it is deterministic rather than
        # "usually passes": the cancel event is already set before the race
        # even starts, guaranteeing both the sleep and the cancellation are
        # ready on the very first check asyncio.wait makes.
        pending = _PendingGrace(asker_id=ASKER, message_id=1)
        pending.cancel_event.set()

        outcome = await GraceRegistry._race_sleep_against_cancellation(pending, 0.0)

        assert outcome is GraceWaitOutcome.CANCELLED_BY_HUMAN


class TestSupersession:
    """At most one grace period is ever pending per channel."""

    async def test_a_second_registration_in_the_same_channel_cancels_the_first(self) -> None:
        registry = GraceRegistry()
        first = asyncio.create_task(
            registry.wait(channel_id=CHANNEL_A, asker_id=ASKER, message_id=1, seconds=5.0)
        )
        await asyncio.sleep(0.05)

        second = asyncio.create_task(
            registry.wait(channel_id=CHANNEL_A, asker_id=2, message_id=2, seconds=0.05)
        )

        first_outcome = await asyncio.wait_for(first, timeout=1.0)
        second_outcome = await second
        assert first_outcome is GraceWaitOutcome.CANCELLED_BY_HUMAN
        assert second_outcome is GraceWaitOutcome.EXPIRED

    async def test_superseding_does_not_leak_the_first_waits_dict_entry(self) -> None:
        registry = GraceRegistry()
        first = asyncio.create_task(
            registry.wait(channel_id=CHANNEL_A, asker_id=ASKER, message_id=1, seconds=5.0)
        )
        await asyncio.sleep(0.05)

        await registry.wait(channel_id=CHANNEL_A, asker_id=2, message_id=2, seconds=0.05)
        await asyncio.wait_for(first, timeout=1.0)

        assert registry._pending == {}


class TestNoLeaksUnderConcurrency:
    """Many simultaneous grace periods across channels and guilds must not interfere or leak."""

    async def test_many_channels_at_once_resolve_independently_with_no_leftover_state(
        self,
    ) -> None:
        registry = GraceRegistry()
        channels = range(1000, 1050)

        outcomes = await asyncio.gather(
            *(
                registry.wait(channel_id=channel, asker_id=ASKER, message_id=channel, seconds=0.05)
                for channel in channels
            )
        )

        assert all(outcome is GraceWaitOutcome.EXPIRED for outcome in outcomes)
        assert registry._pending == {}

    async def test_cancellations_only_affect_their_own_channel(self) -> None:
        registry = GraceRegistry()
        tasks = {
            channel: asyncio.create_task(
                registry.wait(
                    channel_id=channel, asker_id=ASKER, message_id=channel, seconds=5.0
                )
            )
            for channel in (CHANNEL_A, CHANNEL_B)
        }
        await asyncio.sleep(0.05)

        registry.notice_human_message(
            channel_id=CHANNEL_A, author_id=OTHER_HUMAN, message_id=99999
        )

        cancelled = await asyncio.wait_for(tasks[CHANNEL_A], timeout=1.0)
        assert cancelled is GraceWaitOutcome.CANCELLED_BY_HUMAN
        assert not tasks[CHANNEL_B].done()

        tasks[CHANNEL_B].cancel()
        with pytest.raises(asyncio.CancelledError):
            await tasks[CHANNEL_B]

    async def test_no_task_is_left_pending_after_a_burst_of_mixed_outcomes(self) -> None:
        # Half the channels get a human answer, half just time out -- proving
        # the "no orphaned task" guarantee holds for both exit paths at once,
        # not just whichever one happens to be exercised alone above.
        registry = GraceRegistry()
        tasks = {
            channel: asyncio.create_task(
                registry.wait(
                    channel_id=channel, asker_id=ASKER, message_id=channel, seconds=0.3
                )
            )
            for channel in range(2000, 2020)
        }
        await asyncio.sleep(0.05)

        for channel in range(2000, 2010):
            registry.notice_human_message(
                channel_id=channel, author_id=OTHER_HUMAN, message_id=99999
            )

        results = await asyncio.gather(*tasks.values())
        expected_cancelled = sum(1 for r in results if r is GraceWaitOutcome.CANCELLED_BY_HUMAN)
        expected_expired = sum(1 for r in results if r is GraceWaitOutcome.EXPIRED)
        assert expected_cancelled == 10
        assert expected_expired == 10
        assert registry._pending == {}

        # Every task genuinely finished (not just gathered) -- so nothing is
        # left running in the background after this test returns.
        assert all(task.done() and not task.cancelled() for task in tasks.values())


class TestRestartIsARegistryWithNoState:
    """A process restart is modelled correctly by simply constructing a fresh registry."""

    def test_a_fresh_registry_has_nothing_pending_and_needs_no_recovery_step(self) -> None:
        registry = GraceRegistry()
        assert registry._pending == {}

    async def test_a_fresh_registry_can_immediately_wait_without_any_prior_state(self) -> None:
        # There is nothing that would need to be "resumed": a message whose
        # grace period was in flight when the process died before this phase
        # existed simply never gets a proactive evaluation. Asserted here as
        # "this works with zero setup", which is the whole point.
        outcome = await GraceRegistry().wait(
            channel_id=CHANNEL_A, asker_id=ASKER, message_id=1, seconds=0.01
        )
        assert outcome is GraceWaitOutcome.EXPIRED
