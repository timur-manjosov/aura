"""Store a subscription snapshot and move the runtime plan gate with it, as one step.

The database write and the in-memory view update belong together: a snapshot
that committed but never reached the gate would leave a paying guild on Free
until the process restarted. Keeping both in this one function, with no await
between the commit returning and the view updating, is what makes that
impossible to get wrong at a call site -- the internal API calls this and never
the two halves separately.
"""

from __future__ import annotations

import logging
from datetime import datetime

import aiosqlite

from aura.billing.plan_gate import PlanGate
from aura.db.subscriptions import (
    ApplyOutcome,
    ApplyResult,
    SubscriptionSnapshot,
    apply_subscription_snapshot,
)

logger = logging.getLogger(__name__)


async def apply_snapshot(
    conn: aiosqlite.Connection,
    gate: PlanGate,
    *,
    snapshot: SubscriptionSnapshot,
    event_id: str | None,
    event_type: str | None,
    expected_version: int,
    now: datetime,
) -> ApplyResult:
    """Apply one snapshot through the compare-and-swap, then update the gate.

    Parameters
    ----------
    conn
        Open database connection.
    gate
        The in-memory plan gate to write through to.
    snapshot
        What Stripe says about the subscription.
    event_id, event_type
        The Stripe event driving this write, or None for a reconciliation write.
    expected_version
        The version the caller read before fetching from Stripe.
    now
        When the write is made.

    Returns
    -------
    ApplyResult
        Exactly what `apply_subscription_snapshot` returned.

    Notes
    -----
    The gate is updated only when the write actually committed, and in the same
    step -- which is what makes the gate's in-memory view unable to drift from the
    table while the process runs. A duplicate or conflicting event changes
    neither.
    """
    result = await apply_subscription_snapshot(
        conn,
        snapshot=snapshot,
        event_id=event_id,
        event_type=event_type,
        expected_version=expected_version,
        now=now,
    )
    source = f"event {event_id} ({event_type})" if event_id is not None else "reconciliation"

    if result.outcome is ApplyOutcome.APPLIED and result.record is not None:
        gate.record_applied(result.record)
        logger.info(
            "Applied Stripe subscription %s for guild %s from %s: status=%s, "
            "cancel_at_period_end=%s, version %d",
            snapshot.subscription_id,
            snapshot.guild_id,
            source,
            snapshot.status.value,
            snapshot.cancel_at_period_end,
            result.version,
        )
        if result.previous_guild_id is not None:
            logger.warning(
                "Stripe subscription %s moved from guild %s to guild %s: its metadata was "
                "changed in Stripe, which only the operator can do",
                snapshot.subscription_id,
                result.previous_guild_id,
                snapshot.guild_id,
            )
        granting = gate.plan_for(snapshot.guild_id).standing.granting_subscription_ids
        if len(granting) > 1:
            logger.warning(
                "Guild %s now has %d subscriptions granting Pro (%s): the guild is paying more "
                "than once. The payer of the extra subscription can cancel it from the billing "
                "portal; /aura-plan shows the count to the guild's admins.",
                snapshot.guild_id,
                len(granting),
                ", ".join(sorted(granting)),
            )
    elif result.outcome is ApplyOutcome.DUPLICATE_EVENT:
        logger.info(
            "Ignored %s for subscription %s: already applied", source, snapshot.subscription_id
        )
    else:
        logger.info(
            "Refused a stale snapshot of subscription %s from %s: version is now %d",
            snapshot.subscription_id,
            source,
            result.version,
        )
    return result
