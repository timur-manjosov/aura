"""The shape of the Phase 3a-1 fact-worthiness corpus, and its ground truth.

Deliberately a much simpler shape than scripts/synthetic_corpus's Stage 1/
Stage 2/Stage 3 corpus. That corpus exists to simulate proactive relief's
whole pipeline -- guilds, facts, retrieval, escalation -- and needed guild
structure and fact/message linkage to do it. This one measures a single,
free, local, message-only classifier (aura.extraction.fact_worthiness): does
this text read as an authoritative, checkable statement about the server, or
as ordinary chat? There is no retrieval, no guild-scoped fact set, and no
second stage to simulate, so there is no guild model, no fact model, and no
target_fact_keys -- a message's category IS its ground truth, exactly as in
the Stage 1 corpus, and nothing here needs anything more than that.

Every label exists by construction, same principle as
synthetic_corpus.corpus_model: a message is FACT_WORTHY_ANNOUNCEMENT because
the announcement prompt produced it, not because something read it afterwards
and guessed.
"""
from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class MessageCategory(StrEnum):
    """What a message was generated to be, spanning the two truth classes.

    Five fact-worthy subcategories and eight not-fact-worthy ones, per the
    Phase 3a-1 design brief's category list plus two hard-negative categories
    the brief's "Attack It" section calls for explicitly:

    * HEDGED_SPECULATION -- "I think the event might be Saturday" as opposed
      to "The event is Saturday": the brief's own example of a near-miss pair
      that looks almost identical to its fact-worthy counterpart.
    * ADVERSARIAL_NOISE -- text shaped like a rule, quote or hypothetical
      that is NOT an actual current statement about the server ("would be
      funny if there were a rule that...", a sarcastic misquote of a rule
      someone else stated, "what if we required..."). The brief's own example
      of content designed to fool a filter that pattern-matches on
      rule-shaped phrasing rather than on whether something is actually true
      right now. Phase 3a-1's label audit found a fifth intended technique --
      "a rule that used to apply but no longer does" -- was not actually a
      decoy (the retraction itself is a genuine status-change fact); Phase
      3a-1b dropped it from generator.py's prompt rather than keeping it in
      this category. See reports/phase-3a-1.txt Section 4 and
      reports/phase-3a-1b.txt.
    """

    FACT_WORTHY_ANNOUNCEMENT = "fact_worthy_announcement"
    FACT_WORTHY_RULE_POLICY = "fact_worthy_rule_policy"
    FACT_WORTHY_DECISION = "fact_worthy_decision"
    FACT_WORTHY_EVENT_SCHEDULE = "fact_worthy_event_schedule"
    FACT_WORTHY_STATUS_CHANGE = "fact_worthy_status_change"
    GREETING = "greeting"
    SMALL_TALK = "small_talk"
    OPINION = "opinion"
    QUESTION = "question"
    PERSONAL = "personal"
    REACTION = "reaction"
    HEDGED_SPECULATION = "hedged_speculation"
    ADVERSARIAL_NOISE = "adversarial_noise"


FACT_WORTHY_CATEGORIES: tuple[MessageCategory, ...] = (
    MessageCategory.FACT_WORTHY_ANNOUNCEMENT,
    MessageCategory.FACT_WORTHY_RULE_POLICY,
    MessageCategory.FACT_WORTHY_DECISION,
    MessageCategory.FACT_WORTHY_EVENT_SCHEDULE,
    MessageCategory.FACT_WORTHY_STATUS_CHANGE,
)

# The six subcategories from the design brief's own "not fact-worthy" list.
ORDINARY_NOT_FACT_WORTHY_CATEGORIES: tuple[MessageCategory, ...] = (
    MessageCategory.GREETING,
    MessageCategory.SMALL_TALK,
    MessageCategory.OPINION,
    MessageCategory.QUESTION,
    MessageCategory.PERSONAL,
    MessageCategory.REACTION,
)

# The two hard-negative categories the brief's "Attack It" section asks for.
# Scored and reported separately from the ordinary negatives in the sweep, per
# the same reasoning synthetic_corpus.corpus_model keeps partial_answer and
# contradictory_facts as their own categories rather than folding them into a
# generic "should not escalate" bucket: a threshold that only looks good on
# easy negatives has not been tested on the case that actually matters.
HARD_NEGATIVE_CATEGORIES: tuple[MessageCategory, ...] = (
    MessageCategory.HEDGED_SPECULATION,
    MessageCategory.ADVERSARIAL_NOISE,
)

NOT_FACT_WORTHY_CATEGORIES: tuple[MessageCategory, ...] = (
    *ORDINARY_NOT_FACT_WORTHY_CATEGORIES,
    *HARD_NEGATIVE_CATEGORIES,
)


def is_fact_worthy(category: MessageCategory) -> bool:
    """The ground-truth label a category implies: is Aura's knowledge model better for having it?"""
    return category in FACT_WORTHY_CATEGORIES


class LabelAudit(StrEnum):
    """Whether an independent model agreed with a message's own fact-worthiness label."""

    NOT_AUDITED = "not_audited"
    AGREE = "agree"
    DISPUTE = "dispute"
    UNAVAILABLE = "unavailable"


class SyntheticMessage(BaseModel):
    """One labelled message: what was said, in which locale, and what it was generated to be."""

    key: str
    category: MessageCategory
    locale: str
    content: str
    # Free-text note from the generator on why this case is what it claims to
    # be. Never used for scoring; a spot-check aid only, same role
    # synthetic_corpus.corpus_model.SyntheticMessage.rationale plays.
    rationale: str = ""
    label_audit: LabelAudit = LabelAudit.NOT_AUDITED

    @property
    def is_fact_worthy(self) -> bool:
        """This message's ground-truth label."""
        return is_fact_worthy(self.category)


class RejectedCase(BaseModel):
    """A generated case refused before entering the corpus. Recorded, never dropped silently."""

    category: MessageCategory
    locale: str
    reason: str
    layer: str


class SyntheticCorpus(BaseModel):
    """The whole generated fact-worthiness corpus, plus its provenance."""

    generated_at: datetime
    generator_model: str
    reviewer_model: str
    messages: list[SyntheticMessage] = Field(default_factory=list)
    rejected: list[RejectedCase] = Field(default_factory=list)
    generation_cost_usd: float = 0.0
    generation_calls: int = 0

    def check_referential_integrity(self) -> list[str]:
        """Return every broken invariant, empty when the corpus is sound.

        There is no cross-referencing to check here (no guilds, no facts) --
        this checks the one invariant that still matters: every message key is
        unique, so a later lookup by key can never silently resolve to the
        wrong message.
        """
        problems: list[str] = []
        seen: set[str] = set()
        for message in self.messages:
            if message.key in seen:
                problems.append(f"duplicate message key {message.key}")
            seen.add(message.key)
        return problems
