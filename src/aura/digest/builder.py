"""Assembling one guild's digest: what changed in the knowledge model, from the
knowledge model itself.

**No LLM call happens anywhere in this module, and that is a design decision
rather than a phase boundary.** Everything a digest says is already structured
before this code runs -- a fact's distilled sentence was written when the fact
was created, its category was judged when the candidate was staged, its
timestamp and its supersession chain are columns. There is nothing here for a
model to reason about, so paying one to re-render structured data as prose would
buy latency, cost and a chance of invention in exchange for nothing. CLAUDE.md's
rule that "the model is paid for judgment, never for knowledge" cuts the same
way in reverse: no judgment is required, so no model is called. That is also why
this sub-phase needs no grounding check -- there is no generated text that could
fail to be grounded.

**The digest reports the net change over its window, not a replay of every event
in it.** That single principle settles the three cases that would otherwise each
need their own rule, and it is what a reader of a weekly summary actually wants:
they were away, and they want to know what is true now that was not true (or was
different) when they last looked.

  * A fact created and superseded inside the same window is not "new" -- it is
    not even true any more -- so it is absent from the new-facts section (see
    get_facts_created_between, which filters on ACTIVE for exactly this).
  * A fact superseded several times inside one window is reported once, as its
    state before the window against its state now: "A -> D", not "A -> B -> C ->
    D". The intermediate steps are real history, kept forever in the chain and
    reachable through the database, but a summary that lists them is a worse
    summary.
  * A supersession whose PREDECESSOR was itself created inside this window is
    not reported as a change at all. The reader never saw the predecessor, so
    "X replaced Y" would be telling them about the retirement of something they
    were never told existed; the net effect for them is simply that the
    successor is new, which the new-facts section already says.
"""
from __future__ import annotations

import logging
from datetime import datetime

import aiosqlite
from pydantic import BaseModel

from aura.db.connection import utc_iso
from aura.db.models import Fact, FactStatus
from aura.db.pending_facts import get_milestone_fact_ids
from aura.db.repository import (
    get_facts_by_ids,
    get_facts_created_between,
    get_facts_superseded_between,
)

logger = logging.getLogger(__name__)

# How far a supersession chain is followed before the walk gives up and reports
# what it has reached so far.
#
# Not reachable by any code path in this project: supersede_fact_with_existing_successor
# refuses a successor that is not itself active, so a chain can only ever grow at
# its head and can never close into a cycle. The bound exists because "cannot
# happen" and "cannot hang the scheduler" are different guarantees, and only the
# second one survives a hand-edited database. A digest that renders a slightly
# wrong pair is a cosmetic problem; a background task spinning forever inside a
# while loop is an outage.
_MAX_CHAIN_DEPTH = 25


class DigestChange(BaseModel):
    """One "this used to be true, now this is" pair, collapsed from a chain.

    `previous` is the state the reader could have known before the window;
    `current` is where that chain now ends. `collapsed_steps` counts the
    intermediate supersessions that happened in between and are deliberately not
    listed -- shown as a small note rather than hidden entirely, so a digest
    never implies a single tidy edit where there was a run of corrections.
    """

    previous: Fact
    current: Fact
    changed_at: datetime
    collapsed_steps: int


class DigestContent(BaseModel):
    """Everything one digest has to say about one window, before any formatting.

    Pure data with no Discord types in it, so the whole question of "is this
    digest right" is answerable without a gateway, an embed or a locale -- which
    is what lets the interesting cases (chains, milestones, an empty window) be
    tested as units, per CLAUDE.md's testing rule.

    Milestones are held apart from `new_facts` rather than flagged inside it:
    they are rendered as their own highlighted section, and a fact belongs to
    exactly one of the two.
    """

    guild_id: int
    covered_from: datetime
    covered_until: datetime
    new_facts: list[Fact]
    milestones: list[Fact]
    changes: list[DigestChange]

    @property
    def is_empty(self) -> bool:
        """Whether this window produced nothing worth posting.

        The digest's one silence rule, and the reason it is a property here
        rather than a check at the call site: an empty digest must be skipped
        rather than posted (CLAUDE.md's "deliberately conservative" stance --
        no unprompted interruption without content), and that decision should
        read off the content itself instead of being re-derived from three
        separate length checks wherever it is needed.
        """
        return not (self.new_facts or self.milestones or self.changes)

    @property
    def total_items(self) -> int:
        """How many individual entries this digest would render."""
        return len(self.new_facts) + len(self.milestones) + len(self.changes)


async def build_digest(
    conn: aiosqlite.Connection, *, guild_id: int, since: str, until: str
) -> DigestContent:
    """Assemble one guild's digest for the half-open window (since, until].

    Both bounds are fixed-width UTC ISO-8601 strings (aura.db.connection.utc_iso),
    not datetimes, because that is what the ledger stores and what SQL compares
    against; converting them here and back at every query would add a parsing
    failure mode to a path that has none.

    Read-only from start to finish. Nothing in a digest writes to the knowledge
    model, marks anything as reported, or has any effect on a fact whatsoever --
    the bookkeeping that records the window as covered belongs to the scheduler
    and happens outside this function, so building a digest twice (for a retry,
    or in a test) is free of consequences.
    """
    created = await get_facts_created_between(conn, guild_id=guild_id, since=since, until=until)
    milestone_ids = await get_milestone_fact_ids(conn, guild_id=guild_id)
    retired = await get_facts_superseded_between(conn, guild_id=guild_id, since=since, until=until)

    changes = await _collapse_chains(conn, guild_id=guild_id, retired=retired, since=since)

    return DigestContent(
        guild_id=guild_id,
        covered_from=datetime.fromisoformat(since),
        covered_until=datetime.fromisoformat(until),
        new_facts=[fact for fact in created if fact.id not in milestone_ids],
        milestones=[fact for fact in created if fact.id in milestone_ids],
        changes=changes,
    )


async def _collapse_chains(
    conn: aiosqlite.Connection, *, guild_id: int, retired: list[Fact], since: str
) -> list[DigestChange]:
    """Turn the window's retired facts into one "before -> now" pair per chain.

    Three filters decide what becomes a reported change, applied in this order
    because each is cheaper and more decisive than the next:

      1. A retired fact that is itself the successor of another retired fact in
         this same window is an INTERMEDIATE step, not the start of a chain.
         Dropping these is what turns A->B->C into one pair instead of two.
      2. A chain whose starting fact was created inside the window is not a
         change the reader can recognise (see this module's docstring).
      3. A chain with no successor at all -- a row marked superseded with a NULL
         superseded_by_id, which no code path in this project can produce -- is
         skipped with a warning rather than rendered as "X -> X".

    The successor lookups are batched round by round rather than issued per
    chain, per CLAUDE.md's Performance rule: in practice one round resolves
    every head, because the overwhelmingly common chain is a single hop.
    """
    retired_by_id = {fact.id: fact for fact in retired}
    successor_of_retired = {
        fact.superseded_by_id for fact in retired if fact.superseded_by_id is not None
    }
    roots = [
        fact
        for fact in retired
        if fact.id not in successor_of_retired
        # utc_iso rather than parsing `since`: the same fixed-width text
        # comparison SQL just used to select these rows, reused here so the two
        # can never disagree about which side of the boundary a fact falls on.
        and utc_iso(fact.created_at) <= since
    ]
    if not roots:
        return []

    known = await _resolve_chain_facts(conn, guild_id=guild_id, seeds=retired_by_id)

    changes: list[DigestChange] = []
    for root in roots:
        head, steps = _walk_to_head(root, known)
        if head.id == root.id:
            logger.warning(
                "Fact %s in guild %s is marked superseded but points at no successor; "
                "leaving it out of the digest",
                root.id,
                guild_id,
            )
            continue
        assert root.superseded_at is not None  # every superseded row carries one
        changes.append(
            DigestChange(
                previous=root,
                current=head,
                changed_at=root.superseded_at,
                collapsed_steps=steps - 1,
            )
        )
    return changes


async def _resolve_chain_facts(
    conn: aiosqlite.Connection, *, guild_id: int, seeds: dict[int, Fact]
) -> dict[int, Fact]:
    """Fetch every fact reachable by following supersession pointers out of `seeds`.

    Breadth-first in batched rounds -- one query per round, never one per link --
    so a window containing fifty retired facts costs one extra query rather than
    fifty. Bounded by the same depth limit the walk uses, for the same reason:
    a pointer loop that cannot exist must still not be able to spin here.
    """
    known = dict(seeds)
    for _ in range(_MAX_CHAIN_DEPTH):
        missing = {
            fact.superseded_by_id
            for fact in known.values()
            if fact.superseded_by_id is not None and fact.superseded_by_id not in known
        }
        if not missing:
            break
        fetched = await get_facts_by_ids(conn, guild_id=guild_id, fact_ids=missing)
        if not fetched:
            # Every remaining pointer is dangling or points outside this guild.
            # _walk_to_head reports what it can reach; there is nothing left to
            # fetch, so another round would query the same missing IDs forever.
            break
        known.update(fetched)
    return known


def _walk_to_head(root: Fact, known: dict[int, Fact]) -> tuple[Fact, int]:
    """Follow root's supersession chain to its end. Returns (end, steps taken).

    Pure, so the interesting behaviour -- a multi-step chain, a dangling
    pointer, the depth cap -- is testable without a database. Stops at the first
    fact that is still active, has no successor, or whose successor could not be
    resolved, and reports what it reached rather than raising: a digest that
    shows a slightly stale "now" is worth more than a digest that fails to
    render at all.
    """
    current = root
    steps = 0
    seen = {root.id}
    for _ in range(_MAX_CHAIN_DEPTH):
        if current.status is FactStatus.ACTIVE or current.superseded_by_id is None:
            return current, steps
        successor = known.get(current.superseded_by_id)
        if successor is None:
            logger.warning(
                "Fact %s points at successor %s, which is not readable in guild %s; "
                "reporting the chain as ending here",
                current.id,
                current.superseded_by_id,
                current.guild_id,
            )
            return current, steps
        if successor.id in seen:
            logger.error(
                "Supersession chain from fact %s loops back to fact %s; reporting the "
                "chain as ending here",
                root.id,
                successor.id,
            )
            return current, steps
        seen.add(successor.id)
        current = successor
        steps += 1

    logger.warning(
        "Supersession chain from fact %s is longer than %d steps; reporting the first "
        "%d and stopping",
        root.id,
        _MAX_CHAIN_DEPTH,
        _MAX_CHAIN_DEPTH,
    )
    return current, steps
