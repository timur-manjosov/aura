"""Moving a subscription's state from Stripe to the bot, correctly under reordering and retries.

One function does it, and both the webhook route and the periodic reconciler
call it, so there is exactly one path by which a plan can change:

  1. Ask the bot whether this event was already applied, and which version of
     the subscription it holds. An applied event stops here.
  2. Fetch the subscription from Stripe -- its state NOW, not the event's copy.
  3. Ask the bot to store that snapshot only if its version is still the one
     read in step 1.

Why this order is correct rather than merely careful. A stored snapshot always
came from a fetch that started after the previous stored snapshot was
committed: step 3's compare-and-swap refuses any write whose step 1 happened
before another write landed, and step 1 precedes step 2. So the stored state is
always the most recently fetched one, whatever order Stripe delivered events in,
however concurrent webhook deliveries interleave, and even if an earlier request
to the bot times out here and completes there later -- that late write carries a
stale version and is refused. No clock is compared anywhere.

A refused write (version conflict) is retried from step 1 a bounded number of
times, then reported as a failure, so Stripe redelivers the event later rather
than this service looping.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from enum import StrEnum

from aura_web.bot_billing import ApplyOutcome, BotBillingClient, BotBillingError
from aura_web.stripe_api import StripeAPIError, StripeClient

logger = logging.getLogger(__name__)

# Three reads-then-writes. A conflict means another sync for the same
# subscription committed in between; two in a row is already unusual, three is
# contention worth giving back to Stripe's own retry schedule.
MAX_SYNC_ATTEMPTS = 3

# How long after startup the first reconciliation runs. Soon enough that a
# restart after an outage heals missed events promptly; not immediately, so a
# crash-looping container does not hit Stripe's list endpoint on every restart.
FIRST_RECONCILIATION_DELAY_SECONDS = 60.0


class SyncOutcome(StrEnum):
    """What syncing one subscription did."""

    APPLIED = "applied"
    DUPLICATE = "duplicate"
    NOT_AURA = "not_aura"


class SyncConflictError(Exception):
    """Every attempt lost a compare-and-swap to a concurrent sync."""


class LivemodeMismatchError(Exception):
    """A subscription from the other Stripe mode than this service is configured for."""


async def sync_subscription(
    *,
    stripe: StripeClient,
    bot: BotBillingClient,
    subscription_id: str,
    event_id: str | None,
    event_type: str | None,
    live_mode: bool,
) -> SyncOutcome:
    """Bring the bot's copy of one subscription up to Stripe's current state.

    Parameters
    ----------
    stripe
        Reads the subscription's current state.
    bot
        Holds the stored copy and applies the write.
    subscription_id
        The subscription to bring up to date.
    event_id, event_type
        The Stripe event driving this sync, or None for reconciliation.
    live_mode
        Whether this deployment accepts live events.

    Returns
    -------
    SyncOutcome
        APPLIED when the bot stored the snapshot, DUPLICATE when this event was
        already applied, NOT_AURA when the subscription carries no Aura guild
        metadata (nothing is sent to the bot).

    Raises
    ------
    StripeAPIError
        When Stripe cannot be reached or answers with something unusable.
    BotBillingError
        When the bot's internal API cannot be reached or refuses the call.
    SyncConflictError
        After MAX_SYNC_ATTEMPTS compare-and-swaps lost to concurrent syncs.
    LivemodeMismatchError
        For a subscription from the other Stripe mode.

    Notes
    -----
    Every exception above leaves the bot's state exactly as it was, and each is
    a failure to answer now rather than an answer: the webhook route replies
    with a status Stripe retries, and the reconciler tries again on its next
    pass.

    A subscription not on the Pro price (or at quantity 0) is still pushed,
    flagged `on_pro_price=False`, so the bot records it as granting nothing.
    Skipping it instead would leave any copy the bot already holds granting
    Pro until that copy's own window ran out.
    """
    for attempt in range(1, MAX_SYNC_ATTEMPTS + 1):
        state = await bot.get_sync_state(subscription_id=subscription_id, event_id=event_id)
        if state.event_processed:
            return SyncOutcome.DUPLICATE

        snapshot = await stripe.retrieve_subscription(subscription_id)
        if snapshot.livemode != live_mode:
            raise LivemodeMismatchError(
                f"subscription {subscription_id} is livemode={snapshot.livemode}, "
                f"this service is configured for livemode={live_mode}"
            )
        if snapshot.guild_id is None:
            if state.version > 0:
                # The bot holds this subscription, so it carried Aura's
                # metadata once -- and only the operator can remove it. Its
                # stored copy now never changes again and keeps granting inside
                # its own window: someone has to look (Phase 4c audit, F-22).
                logger.error(
                    "Stripe subscription %s no longer carries Aura guild metadata, but the bot "
                    "holds version %d of it; that copy is frozen and still decides the guild's "
                    "plan until it expires. Restore the metadata in Stripe or cancel the "
                    "subscription.",
                    subscription_id,
                    state.version,
                )
            else:
                logger.info(
                    "Stripe subscription %s carries no Aura guild metadata; not an Aura "
                    "subscription",
                    subscription_id,
                )
            return SyncOutcome.NOT_AURA
        if not snapshot.on_pro_price:
            logger.warning(
                "Stripe subscription %s for guild %s is not on the Pro price at a quantity of at "
                "least one; pushing it to the bot as granting nothing",
                subscription_id,
                snapshot.guild_id,
            )

        result = await bot.apply_snapshot(
            event_id=event_id,
            event_type=event_type,
            expected_version=state.version,
            snapshot=snapshot,
        )
        if result.outcome is ApplyOutcome.APPLIED:
            return SyncOutcome.APPLIED
        if result.outcome is ApplyOutcome.DUPLICATE:
            return SyncOutcome.DUPLICATE
        logger.info(
            "Sync of subscription %s lost a race to a concurrent sync (attempt %d of %d); retrying",
            subscription_id,
            attempt,
            MAX_SYNC_ATTEMPTS,
        )
    raise SyncConflictError(
        f"subscription {subscription_id} kept changing during {MAX_SYNC_ATTEMPTS} sync attempts"
    )


@dataclass
class ReconciliationReport:
    """What one reconciliation pass did, for its log line and its tests."""

    examined: int = 0
    applied: int = 0
    failed: list[str] = field(default_factory=list)


async def reconcile_subscriptions(
    *, stripe: StripeClient, bot: BotBillingClient, live_mode: bool
) -> ReconciliationReport:
    """Re-sync every Aura subscription in the Stripe account, one at a time.

    Parameters
    ----------
    stripe
        Lists and reads the account's Aura subscriptions.
    bot
        Receives each snapshot.
    live_mode
        Whether this deployment accepts live events.

    Returns
    -------
    ReconciliationReport
        Counts per outcome. One subscription at a time, so a failure on one
        does not abandon the rest.

    Notes
    -----
    Recovers what a webhook cannot: an event Stripe stopped retrying while this
    service or the bot was down. A failure on one subscription is logged and
    the pass continues, so one broken subscription cannot starve the others --
    the same per-item isolation the bot's own sweeps use.
    """
    report = ReconciliationReport()
    for subscription_id in await stripe.list_aura_subscription_ids():
        report.examined += 1
        try:
            outcome = await sync_subscription(
                stripe=stripe,
                bot=bot,
                subscription_id=subscription_id,
                event_id=None,
                event_type=None,
                live_mode=live_mode,
            )
        except (StripeAPIError, BotBillingError, SyncConflictError, LivemodeMismatchError) as exc:
            logger.warning(
                "Reconciliation could not sync subscription %s: %s", subscription_id, exc
            )
            report.failed.append(subscription_id)
            continue
        if outcome is SyncOutcome.APPLIED:
            report.applied += 1
    return report


async def run_reconciler(
    *,
    stripe: StripeClient,
    bot: BotBillingClient,
    live_mode: bool,
    interval_seconds: float,
    first_delay_seconds: float = FIRST_RECONCILIATION_DELAY_SECONDS,
) -> None:
    """Reconcile periodically for the process's life. Never dies of a failure it can survive.

    Parameters
    ----------
    stripe, bot
        Passed through to each reconciliation pass.
    live_mode
        Whether this deployment accepts live events.
    interval_seconds
        How long to wait between passes.
    first_delay_seconds
        How long to wait before the first pass, so startup is not slowed by it.

    Returns
    -------
    None
        Runs until cancelled. Every failure a pass can survive is logged and
        the loop continues, because a reconciler that dies is a reconciler
        nobody notices is gone.
    """
    await asyncio.sleep(first_delay_seconds)
    while True:
        try:
            report = await reconcile_subscriptions(stripe=stripe, bot=bot, live_mode=live_mode)
            logger.info(
                "Stripe reconciliation: %d Aura subscription(s) examined, %d re-synced, %d failed",
                report.examined,
                report.applied,
                len(report.failed),
            )
        except (StripeAPIError, BotBillingError) as exc:
            logger.warning("Stripe reconciliation could not run: %s", exc)
        except Exception:
            logger.exception("Stripe reconciliation failed; continuing")
        await asyncio.sleep(interval_seconds)
