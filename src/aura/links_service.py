"""Retrieval's link expansion: a found fact brings its linked facts along.

CLAUDE.md names four knowledge-model components, and this module is where the
fourth one finally reaches an answer. Similarity search finds a fact because
the question and the fact use nearby language. A LINK is the case that search
structurally cannot reach: a moderator decided two DIFFERENT facts belong in
one answer -- "the tournament starts Saturday" and "the winner gets a month of
Nitro" -- which are about one topic to a human and nowhere near each other to
an embedding model. Nothing about the phrasing of either sentence would ever
pull the second one into a search for the first; only the link does.

**What this adds is candidates, never citations.** Every fact this module
appends goes into the synthesis prompt alongside the similarity hits, and the
synthesis model decides on relevance which of them -- if any -- to actually
cite. That division is deliberate and it is the same one CLAUDE.md draws for
competing facts: a link says "a human thought these belong together", which is
useful evidence and not a verdict on whether it answers THIS question. Making
a linked fact automatically cited would let a stale link put a sentence into
an answer nobody vouched for; making it merely available cannot, because the
model still has to find it relevant, and the grounding check still reads the
finished answer against exactly the facts it claims to have used.

**One hop, never a transitive closure.** A link from a citation candidate is
followed; a link from that neighbour is not. A moderator linking A-B and B-C
asserted two relationships, not the third one, and treating link-of-link as a
link would let an answer about A quietly inherit facts three topics away --
CLAUDE.md's "linked facts", not "everything eventually reachable". This is
also what keeps a long chain A-B-C-D... from flooding synthesis: the chain
contributes only the direct neighbours of what similarity already found.

**A link resolves to what is true now.** A link points at a fact ID, and that
fact may have been superseded since -- possibly several times. Retrieval
follows the supersession chain to the current active fact (see
aura.db.repository.resolve_active_successors) instead of at the dead row the
moderator originally named, which is what lets a link outlive both of the
facts it was drawn between without ever needing to be rewritten. A link that
resolves to nothing contributes nothing.
"""
from __future__ import annotations

import aiosqlite

from aura.db.models import Fact
from aura.db.repository import get_linked_fact_ids, resolve_active_successors

# How many linked facts may join one synthesis call, on top of the similarity
# hits that pulled them in.
#
# A backstop against one specific shape, and worth naming because it is NOT the
# same risk aura.embeddings.SYNTHESIS_FACT_LIMIT guards. That limit stands
# between a guild's hundreds of active facts and an unbounded prompt: its input
# grows with the server. This one's input grows only with deliberate moderator
# effort, since every link is typed by hand -- so the realistic worst case is
# not a flood at all, it is a legitimate hub fact ("the tournament") a
# moderator has linked to its date, its prize, its rules and its signup
# thread. Five keeps that whole hub, which is exactly the answer the link
# feature exists to make possible.
#
# What it actually bounds is the pathological case: one fact linked to fifty,
# by a mistake or by a moderator acting in bad faith, silently doubling every
# synthesis prompt in the guild. Five caps the combined worst case at ten facts
# (~40k characters, since fact content is capped at 4000 by the entry modal),
# and the realistic case at nowhere near it -- a real fact is one distilled
# sentence.
LINKED_FACT_LIMIT = 5


async def expand_with_linked_facts(
    conn: aiosqlite.Connection,
    *,
    guild_id: int,
    facts: list[Fact],
    limit: int = LINKED_FACT_LIMIT,
) -> list[Fact]:
    """Return `facts` followed by up to `limit` active facts linked to them.

    `facts` are the citation candidates similarity search already selected, in
    its ranking order, and they are returned first and unchanged -- this only
    ever appends. Never raises for ordinary sparse data: no facts, no links, or
    links that all resolve to nothing each just give back what was passed in.

    The appended facts are ordered deterministically, and that matters for the
    same reason find_similar_facts breaks its ties explicitly: this list
    decides what a paid model sees, so two identical calls must produce two
    identical prompts or an odd answer cannot be reproduced, let alone
    diagnosed. The order is "neighbours of the best-ranked candidate first,
    then by fact ID" -- the highest-ranked fact's links are the ones most
    likely to matter, and ID ascending settles the rest without appealing to
    SQLite's row order.

    Duplicates are impossible among the APPENDED facts: one already among
    `facts` is never appended (the common case for two candidates linked to
    each other), and two links that resolve through supersession onto the same
    successor contribute it once. `facts` itself is echoed back exactly as
    given, duplicates included -- deduplicating a caller's own ranking would be
    a surprising thing for an expansion to do, and neither caller can produce
    one anyway, since both build the list from find_similar_facts.
    """
    if not facts or limit <= 0:
        return list(facts)

    candidate_ids = {fact.id for fact in facts}
    neighbours_by_fact = await get_linked_fact_ids(
        conn, guild_id=guild_id, fact_ids=[fact.id for fact in facts]
    )

    ordered_neighbour_ids: list[int] = []
    seen_neighbour_ids: set[int] = set()
    for fact in facts:
        for neighbour_id in neighbours_by_fact.get(fact.id, ()):
            if neighbour_id in candidate_ids or neighbour_id in seen_neighbour_ids:
                continue
            seen_neighbour_ids.add(neighbour_id)
            ordered_neighbour_ids.append(neighbour_id)

    if not ordered_neighbour_ids:
        return list(facts)

    resolved = await resolve_active_successors(
        conn, guild_id=guild_id, fact_ids=ordered_neighbour_ids
    )

    expanded = list(facts)
    for neighbour_id in ordered_neighbour_ids:
        if len(expanded) - len(facts) >= limit:
            break
        # Resolution is what decides which row a link means today; an
        # unresolvable link (superseded into nothing, cross-guild, cyclic)
        # is simply skipped -- see resolve_active_successors for why that
        # direction is the safe one.
        linked_fact = resolved.get(neighbour_id)
        if linked_fact is None or linked_fact.id in candidate_ids:
            continue
        candidate_ids.add(linked_fact.id)
        expanded.append(linked_fact)

    return expanded
