"""Assembling one member's onboarding summary: the currently active knowledge
model, filtered and capped down to what a brand-new person actually needs.

**No LLM call in the base case, for the same reason the digest has none (see
aura.digest.builder).** Everything an onboarding message says was already
written and structured before this code runs -- a fact's distilled sentence,
its category, its timestamp. There is nothing here for a model to reason
about, so nothing is paid for. The one exception this project's brief asks to
be *checked* rather than ruled out is a server whose number of active facts
grows large enough that plain category filtering can no longer pick a sensible
subset -- see reports/phase-3d.txt for where that line is judged to sit, and
why the current, real fact counts in this project are nowhere near it.

**Unlike the digest, onboarding is not windowed.** The digest reports the net
change since the last one; onboarding reports the full active state, because a
brand-new member has no "last time" to measure from -- summarizing everything
Aura currently knows is exactly this trigger's job (see CLAUDE.md and
reports/phase-3e.txt Section 9, which names this trigger explicitly as the
one that owns that responsibility).

**Category priority is a product decision, not a side effect of query order.**
Rules and policies come first -- the thing a new member is most likely to need
before their first message. Status changes (what recently changed about how
the server currently runs) come second. Everything else CLAUDE.md's knowledge
model can produce -- announcements, decisions, events, and any fact with no
category at all (manually added through "Add as Aura Fact") -- is grouped
last, under a general heading. MILESTONE is excluded entirely, not merely
deprioritized: reports/phase-3d.txt records this as a deliberate choice --
milestones are retrospectively interesting to existing members, not
actionable for someone who has no context for what changed or why, which is
what CLAUDE.md's onboarding trigger exists to give.
"""
from __future__ import annotations

import aiosqlite
from pydantic import BaseModel

from aura.db.models import Fact
from aura.db.pending_facts import FactCategory, get_confirmed_fact_categories
from aura.db.repository import get_active_facts

# The categories shown ahead of the general "other" bucket, in the order they
# are shown. MILESTONE is deliberately absent from this whole module -- see
# this module's docstring -- rather than being routed into "other", so a
# milestone fact never appears in an onboarding message under any heading.
_PRIORITY_CATEGORIES = (FactCategory.RULE, FactCategory.STATUS_CHANGE)


class OnboardingContent(BaseModel):
    """Everything one member's onboarding message has to say, before any formatting.

    Pure data with no Discord types in it, matching aura.digest.builder's
    DigestContent for the same reason: the interesting behaviour (which facts
    are eligible, how they are bucketed, what the cap left out) is testable as
    a unit, per CLAUDE.md's testing rule.

    `total_eligible` counts every active, non-milestone fact BEFORE the cap
    was applied -- so `omitted_count` (total_eligible minus what is actually
    shown) is knowable even though the individual omitted facts themselves are
    not kept around.
    """

    guild_id: int
    rules: list[Fact]
    status_changes: list[Fact]
    other: list[Fact]
    total_eligible: int

    @property
    def shown_count(self) -> int:
        """How many facts this message actually lists, across all sections."""
        return len(self.rules) + len(self.status_changes) + len(self.other)

    @property
    def omitted_count(self) -> int:
        """How many eligible facts the global cap left out entirely.

        Zero whenever every eligible fact fit under the cap -- the common case
        on a small server, and the only case a brand-new server can be in.
        """
        return max(0, self.total_eligible - self.shown_count)

    @property
    def is_empty(self) -> bool:
        """Whether this guild currently has nothing worth onboarding a member with.

        The same "deliberately conservative" stance the digest's is_empty and
        Trigger 2 both take: a guild with zero eligible active facts (a brand
        new server, or one where every fact happens to be a milestone) gets no
        message rather than an empty shell of headings.
        """
        return not (self.rules or self.status_changes or self.other)


def _take(facts: list[Fact], remaining: int) -> tuple[list[Fact], int]:
    """Take up to `remaining` facts off the front of `facts`. Returns (taken, still-remaining)."""
    taken = facts[:remaining]
    return taken, remaining - len(taken)


async def build_onboarding_content(
    conn: aiosqlite.Connection, *, guild_id: int, limit: int
) -> OnboardingContent:
    """Assemble one guild's onboarding content: its active facts, bucketed and capped.

    Read-only from start to finish, exactly like aura.digest.build_digest:
    nothing here writes to the knowledge model or records that onboarding was
    shown to anyone -- that bookkeeping is aura.db.onboarding_state's job, and
    it happens outside this function, so building this content twice (a race
    between two concurrent join handlers, or a test) is free of consequences.

    Facts within each bucket are ordered newest-first: unlike the digest's
    chronological "what happened, in order" framing, onboarding is a snapshot
    of current state, and if the cap has to cut a bucket short, the most
    recently established rule or the most recently changed status is a better
    thing to keep than the oldest one.

    `limit` bounds the TOTAL number of facts shown across all three sections
    combined, spent in priority order (rules, then status changes, then
    everything else) -- see aura.config.Settings.onboarding_fact_limit for why
    a single total is the right shape rather than a per-section cap. Rejects a
    negative limit rather than silently taking zero, the same defensive
    posture get_pending_facts and get_recent_signals both use for the same
    kind of parameter.
    """
    if limit < 0:
        raise ValueError(f"limit must not be negative, got {limit}")

    active = await get_active_facts(conn, guild_id)
    categories = await get_confirmed_fact_categories(conn, guild_id=guild_id)

    eligible = [f for f in active if categories.get(f.id) != FactCategory.MILESTONE]

    def _bucket(category: FactCategory) -> list[Fact]:
        return sorted(
            (f for f in eligible if categories.get(f.id) == category),
            key=lambda f: f.created_at,
            reverse=True,
        )

    rules_all = _bucket(FactCategory.RULE)
    status_all = _bucket(FactCategory.STATUS_CHANGE)
    other_all = sorted(
        (f for f in eligible if categories.get(f.id) not in _PRIORITY_CATEGORIES),
        key=lambda f: f.created_at,
        reverse=True,
    )

    remaining = limit
    rules, remaining = _take(rules_all, remaining)
    status_changes, remaining = _take(status_all, remaining)
    other, remaining = _take(other_all, remaining)

    return OnboardingContent(
        guild_id=guild_id,
        rules=rules,
        status_changes=status_changes,
        other=other,
        total_eligible=len(eligible),
    )
