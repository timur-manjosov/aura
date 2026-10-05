"""When a batch whose distillation or verification call failed may be tried again.

A model call that did not complete for a reason outside the batch -- a
timeout, a provider outage, a refused key; aura.llm_failures decides -- is not
the model judging a batch. That holds for the verification call (P5,
aura.extraction.verifier returning VERIFICATION_UNAVAILABLE) and for the
distillation call itself (P5c, aura.extraction.distiller returning
DISTILLATION_UNAVAILABLE). Clearing the batch at that point -- the path an
unusable reply takes -- would lose every fact in it to an outage that says
nothing about the messages. (An unusable REPLY, and a call refused because of
the request itself, still take that path, for the reason
aura.extraction.verifier gives.) So both extraction paths hold such a batch and
try it again later, with a growing pause, a bounded number of times:

* the live path leaves the batch queued (aura.extraction.pipeline);
* backfill leaves its cursor where it is (aura.backfill.worker).

Each attempt re-runs the distillation as well -- a live batch may have grown in
the meantime -- and claims its own daily slot, so a provider that keeps failing
costs at most EXTRACTION_VERIFY_MAX_ATTEMPTS slots per batch, never a slot per
sweep. A failed distillation and a failed verification of the same batch count
against the same attempts. (The setting names predate P5c and are kept so no
deployed `.env` breaks; they bound both calls.) After the last attempt the batch takes the old path (cleared, or the
cursor moves past it) with an ERROR log line naming how many attempts failed,
so an operator sees it.

The state lives in memory, deliberately: a restart forgets the counts, and
since the batch (or the cursor) is still where it was, the restarted bot simply
tries it again -- an extra attempt, never a lost batch. One entry per channel
or backfill run, bounded by MAX_TRACKED_KEYS.

Imports nothing from Discord, the database or any model-calling module.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Final

# More keys than a deployment has channels with a failing batch at once by
# orders of magnitude; the bound only stops a pathological case from growing
# without limit. Evicting the oldest entry forgets its count, which means one
# more attempt for that batch -- the safe direction.
MAX_TRACKED_KEYS: Final = 10_000


@dataclass(frozen=True)
class _Failures:
    count: int
    next_attempt_at: datetime


class VerificationRetries:
    """Per-key failure counts and the earliest moment of the next attempt.

    A key is ``("live", channel_id)`` or ``("backfill", run_id)``.

    Notes
    -----
    Not thread-safe, and need not be: both extraction paths run on the bot's
    one event loop, and none of these methods awaits.
    """

    def __init__(self) -> None:
        self._entries: OrderedDict[tuple[str, int], _Failures] = OrderedDict()

    def is_waiting(self, key: tuple[str, int], now: datetime) -> bool:
        """Report whether `key` failed recently and its pause has not ended.

        Parameters
        ----------
        key
            The batch's key.
        now
            Timezone-aware current moment.

        Returns
        -------
        bool
            True while the batch must not be attempted yet.
        """
        entry = self._entries.get(key)
        return entry is not None and now < entry.next_attempt_at

    def record_failure(
        self,
        key: tuple[str, int],
        now: datetime,
        *,
        max_attempts: int,
        base_delay_seconds: float,
    ) -> int | None:
        """Count one failed distillation or verification call of `key` and schedule the next attempt.

        Parameters
        ----------
        key
            The batch's key.
        now
            Timezone-aware moment of the failure.
        max_attempts
            How many attempts a batch gets in total.
        base_delay_seconds
            The pause after the first failure; it doubles after each further one.

        Returns
        -------
        int or None
            The number of failed attempts so far when another attempt is
            allowed; None when this was the last one -- the entry is then
            removed and the caller gives the batch up.
        """
        previous = self._entries.pop(key, None)
        count = (previous.count if previous else 0) + 1
        if count >= max_attempts:
            return None
        delay = timedelta(seconds=base_delay_seconds * 2 ** (count - 1))
        self._entries[key] = _Failures(count=count, next_attempt_at=now + delay)
        while len(self._entries) > MAX_TRACKED_KEYS:
            self._entries.popitem(last=False)
        return count

    def failures(self, key: tuple[str, int]) -> int:
        """Return how many attempts of `key` have failed and are still counted."""
        entry = self._entries.get(key)
        return entry.count if entry else 0

    def clear(self, key: tuple[str, int]) -> None:
        """Forget `key`: its batch succeeded, was given up, or no longer exists."""
        self._entries.pop(key, None)

    def reset(self) -> None:
        """Forget every key (tests)."""
        self._entries.clear()


# The one instance both extraction paths share for the process's life.
VERIFICATION_RETRIES: Final = VerificationRetries()


def live_key(channel_id: int) -> tuple[str, int]:
    """Return the retry key of a live extraction batch."""
    return ("live", channel_id)


def backfill_key(run_id: int) -> tuple[str, int]:
    """Return the retry key of a backfill run's current batch."""
    return ("backfill", run_id)
