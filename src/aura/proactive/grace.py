"""Phase 2b-1: the grace period between gate ELIGIBLE and paid synthesis.

Aura is deliberately conservative about volunteering answers (CLAUDE.md's
Trigger 2), but Phase 2a-3 shipped one piece of that conservatism still
missing: nothing gave a human the chance to answer first. This module is that
piece -- a bounded wait, inserted between "the gate says ELIGIBLE" (an
escalation slot already spent, see aura.proactive.gate) and "call the paid
synthesis model" (see aura.proactive.responder), which stands down the moment
a genuinely different human posts in the same channel.

**In-memory, not durable -- an explicit, documented trade-off.** Unlike the
escalation ledger (aura.db.proactive_state), which must survive a restart
because bypassing the budget it enforces has a real cost, a grace-period timer
lost on restart simply means that one message never gets a proactive
evaluation: no bypass, no trust consequence. So GraceRegistry is a plain dict
on the client, built fresh once in AuraClient.setup_hook exactly like the
question detector, with no recovery step and no startup error over an empty
one. A message whose grace period was in flight when the process died is
simply never resumed -- there is nothing to resume, and nothing wrong with
that.

**Cancellation is a plain asyncio.Event, not asyncio.Task.cancel().** A real
Task.cancel() would raise asyncio.CancelledError inside the waiting
coroutine -- indistinguishable, from inside that coroutine, from the
CancelledError a shutdown delivers to the very same task, which
aura.proactive.listener's caller must let propagate untouched rather than
treat as "a human answered". Racing a sleep against an Event, and cancelling
whichever loses, keeps "someone else answered" a plain return value instead of
hijacking asyncio's own shutdown-cancellation channel.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum, auto

logger = logging.getLogger(__name__)


class GraceWaitOutcome(Enum):
    """How one grace-period wait ended, as far as this module itself can know.

    Deliberately smaller than aura.db.proactive_signals.GracePeriodOutcome:
    this module has no idea whether the wake-time freshness recheck that
    follows an EXPIRED wait will actually reach synthesis, or whether it will
    stand down instead -- that decision, and the full persisted trail, belong
    to aura.proactive.listener.
    """

    EXPIRED = auto()
    CANCELLED_BY_HUMAN = auto()


@dataclass
class _PendingGrace:
    """One channel's in-flight grace period: who it's waiting on, and how to cancel it."""

    asker_id: int
    message_id: int
    cancel_event: asyncio.Event = field(default_factory=asyncio.Event)


class GraceRegistry:
    """Tracks at most one in-flight grace period per channel, in memory only.

    Built once per process (see AuraClient.setup_hook) and shared across every
    message handled afterwards, the same way the question detector and
    embedding model are -- it is exactly the kind of cross-message state a
    fresh instance per call would silently defeat.
    """

    def __init__(self) -> None:
        self._pending: dict[int, _PendingGrace] = {}

    def notice_human_message(self, *, channel_id: int, author_id: int, message_id: int) -> None:
        """Cancel channel_id's pending grace period if message_id is a genuinely different human.

        Three ways this is a deliberate no-op:
          * no grace period is pending in this channel at all -- the common
            case, and the only work done for it is one dict lookup;
          * message_id is the very message the grace period is waiting on
            (its own arrival, or a gateway redelivery of it) -- not a reply;
          * author_id is the same person who asked -- their own follow-up
            message is not "someone else answered", per this phase's design.

        Safe, and intended, to be called unconditionally for every
        classifiable message Aura sees, before that message's own gate
        evaluation even begins: cancellation must not depend on whatever this
        message's own pipeline decides.
        """
        pending = self._pending.get(channel_id)
        if pending is None:
            return
        if pending.message_id == message_id:
            return
        if pending.asker_id == author_id:
            return
        pending.cancel_event.set()

    def notice_message_gone(self, *, channel_id: int, message_id: int) -> None:
        """Cancel channel_id's pending grace period if it was waiting on message_id.

        Called from both on_message_delete and on_message_edit (see
        aura.main): an edit may change the question into something the
        original gate scores no longer describe, and re-validating edited
        content against Stage 1/2 is out of this phase's scope entirely. The
        conservative move for either event is the same -- stand down rather
        than risk answering a question that isn't the one that was asked, or
        that no longer exists at all.
        """
        pending = self._pending.get(channel_id)
        if pending is not None and pending.message_id == message_id:
            pending.cancel_event.set()

    async def wait(
        self, *, channel_id: int, asker_id: int, message_id: int, seconds: float
    ) -> GraceWaitOutcome:
        """Wait out the grace period for one eligible message, or return early if cancelled.

        Registers this message as channel_id's pending grace period for the
        duration of the wait and removes it again on every exit path --
        expiry, human cancellation, or this coroutine's own task being
        cancelled by a shutdown -- so nothing is ever left pointing at a wait
        that has already ended.

        If a grace period is already pending in this channel when this one
        starts, the older one is cancelled immediately rather than silently
        orphaned. That is only reachable through a misconfiguration where
        PROACTIVE_GRACE_PERIOD_SECONDS is not comfortably below
        PROACTIVE_COOLDOWN_SECONDS (normally the channel's own cooldown
        already rules out a second grant while the first is still waiting),
        logged loudly since it means an operator's thresholds need attention.
        "The newer message supersedes the still-waiting older one" is the same
        "never stack two answers in one channel" principle the wake-time
        freshness recheck (aura.db.proactive_state.is_still_freshest_escalation)
        enforces from the other direction.
        """
        superseded = self._pending.get(channel_id)
        if superseded is not None:
            logger.warning(
                "A new grace period for channel %s (message %s) is superseding one "
                "already pending there (message %s) -- check that "
                "PROACTIVE_GRACE_PERIOD_SECONDS is comfortably below "
                "PROACTIVE_COOLDOWN_SECONDS",
                channel_id,
                message_id,
                superseded.message_id,
            )
            superseded.cancel_event.set()

        pending = _PendingGrace(asker_id=asker_id, message_id=message_id)
        self._pending[channel_id] = pending
        try:
            return await self._race_sleep_against_cancellation(pending, seconds)
        finally:
            # Only clear the slot if it is still ours: a supersede above (or,
            # in principle, a second registration this instance never issued)
            # may already have replaced it with a newer pending grace, which
            # this wait must not delete out from under.
            current = self._pending.get(channel_id)
            if current is pending:
                del self._pending[channel_id]

    @staticmethod
    async def _race_sleep_against_cancellation(
        pending: _PendingGrace, seconds: float
    ) -> GraceWaitOutcome:
        """Run the timer and the cancellation signal concurrently; report whichever wins.

        Both helper tasks are always cancelled and awaited in the finally
        block, whether this coroutine's own task is cancelled from outside
        (a shutdown, propagating through asyncio.wait) or returns normally --
        so no orphaned task and no "Task was destroyed but it is pending"
        warning can survive one call to this function.
        """
        sleep_task = asyncio.ensure_future(asyncio.sleep(seconds))
        cancel_task = asyncio.ensure_future(pending.cancel_event.wait())
        try:
            done, _ = await asyncio.wait(
                {sleep_task, cancel_task}, return_when=asyncio.FIRST_COMPLETED
            )
            # A human answering is the conservative outcome, so a tie (both
            # tasks completing in the same loop iteration) resolves toward
            # staying silent rather than toward proceeding.
            if cancel_task in done:
                return GraceWaitOutcome.CANCELLED_BY_HUMAN
            return GraceWaitOutcome.EXPIRED
        finally:
            for task in (sleep_task, cancel_task):
                if not task.done():
                    task.cancel()
            await asyncio.gather(sleep_task, cancel_task, return_exceptions=True)
