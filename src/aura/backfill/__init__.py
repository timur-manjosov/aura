"""Backfill (Phase 3b): the extraction chain, applied backwards over a channel's
existing history.

**This package adds no fifth mechanism and no new judgement.** CLAUDE.md admits
one mechanism with four triggers, and backfill is not a fifth one -- it is the
automatic-extraction path (Phase 3a) pointed at messages that were written
before Aura was watching. Every decision about whether a message contains a fact
is made by code that already shipped and was already measured: the local
fact-worthiness filter, the distillation call and its prompt, the dedup
comparison, the supersession proposal. Nothing here re-implements any of it, and
nothing here contains a threshold, a prompt or a model name.

What this package owns is the four things that only exist when you read history
rather than react to it:

  * history  -- getting pages out of Discord in strict old-to-new order,
                enforced rather than assumed, and without hammering the API.
  * gateway  -- the single seam to a live client, so everything else is testable
                with no Discord connection at all.
  * worker   -- where to start, how much to do at once, when to stop, and the
                restart-safe cursor that makes a multi-day run survive a deploy.

**Nothing here creates a fact, and nothing here retires one.** Backfill's output
is exactly what live extraction's is: candidates in aura.db.pending_facts, each
becoming a real, citable fact only when a moderator confirms it, and each
supersession verdict remaining a proposal that only /aura-supersede can act on.
The volume is the one thing that genuinely differs -- a year of history can
produce hundreds of candidates where live traffic produces a trickle -- and the
bundled review surface for that volume is Phase 3c, deliberately not here.
"""
from aura.backfill.gateway import BackfillGateway, ClientBackfillGateway
from aura.backfill.history import (
    ChannelUnreadable,
    fetch_history_page,
    is_strictly_increasing,
    ordered_page,
)
from aura.backfill.worker import advance_due_backfills, run_backfill_worker

__all__ = [
    "BackfillGateway",
    "ChannelUnreadable",
    "ClientBackfillGateway",
    "advance_due_backfills",
    "fetch_history_page",
    "is_strictly_increasing",
    "ordered_page",
    "run_backfill_worker",
]
