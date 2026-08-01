"""Service layer between Discord-facing commands and the fact repository.

This exists as a seam, not for its own sake: a fact's embedding has to be
computed at the exact moment it's created, and that has to happen in exactly
one place. Commands must call add_fact() (the manual "Add as Aura Fact" path)
or confirm_fact() (the automatic /aura-pending confirmation path) here, never
aura.db.repository.create_fact() or aura.db.pending_facts.confirm_pending_fact()
directly, so that a hook that must run whenever ANY fact becomes active only
ever needs to be added in this one module.

**Multi-Representation Indexing, Part 1's trigger point lives here.** Both
functions below end by scheduling aura.variants_service.generate_variants_for_fact
as a background task -- never awaited inline -- because both of their callers
are on a live Discord interaction's response path (a modal submission, a
button press), and generating variants costs two sequential LLM round trips
nothing in that path is waiting on. See _schedule_variant_generation for the
fire-and-forget mechanics and why they are safe.
"""
from __future__ import annotations

import asyncio
import logging

import aiosqlite
from fastembed import TextEmbedding

from aura.db.fact_variants import FactVariant
from aura.db.models import Fact
from aura.db.pending_facts import confirm_pending_fact
from aura.db.repository import create_fact
from aura.embeddings import EMBEDDING_DTYPE, embed_text
from aura.variants_service import generate_variants_for_fact

logger = logging.getLogger(__name__)

# Holds a strong reference to every in-flight variant-generation task so the
# event loop cannot garbage-collect it mid-run -- a well-known asyncio trap:
# asyncio.create_task's return value is the ONLY strong reference by default,
# and nothing else in this module keeps one, so a task with nothing else
# referencing it can be collected before it finishes, silently cancelling it
# partway through. Each task removes itself via add_done_callback the moment
# it completes, so this set's steady-state size is the number of variant
# generations genuinely in flight, not an ever-growing leak.
_background_tasks: set[asyncio.Task[list[FactVariant]]] = set()


def _schedule_variant_generation(
    conn: aiosqlite.Connection, model: TextEmbedding, fact: Fact
) -> None:
    """Fire-and-forget aura.variants_service.generate_variants_for_fact for fact.

    Deliberately not awaited: both callers below (add_fact, confirm_fact) are
    invoked from a Discord interaction that is about to send its own response
    (a modal's on_submit, a button's callback), and generation costs two
    sequential LLM round trips nothing on that path should wait on -- the fact
    itself is already committed and fully citable without a single variant.
    generate_variants_for_fact never raises (see its own docstring), so
    nothing here needs a done-callback beyond the GC-safety one above; a
    genuinely unexpected exception would still be logged by asyncio's default
    unhandled-task-exception handler rather than disappearing silently.
    """
    task = asyncio.create_task(generate_variants_for_fact(conn, model, fact))
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)


async def add_fact(
    conn: aiosqlite.Connection,
    model: TextEmbedding,
    *,
    guild_id: int,
    channel_id: int,
    message_id: int,
    content: str,
) -> Fact:
    """Create a new active fact from a Discord message, embedding included.

    Embeds content before touching the database, then makes exactly one
    repository call that writes content and embedding together in the same
    row. Two separate writes -- insert, then a follow-up update with the
    embedding -- would leave a window where a fact exists without one, the
    same atomicity reasoning supersede_fact already applies to its own
    two-statement transaction.

    Schedules variant generation as background enrichment once the fact
    exists (see _schedule_variant_generation) -- the manual "Add as Aura
    Fact" half of Multi-Representation Indexing Part 1's single trigger
    point.
    """
    embedding = await embed_text(model, content)
    fact = await create_fact(
        conn,
        guild_id=guild_id,
        channel_id=channel_id,
        message_id=message_id,
        content=content,
        embedding=embedding.astype(EMBEDDING_DTYPE, copy=False).tobytes(),
    )
    _schedule_variant_generation(conn, model, fact)
    return fact


async def confirm_fact(
    conn: aiosqlite.Connection,
    model: TextEmbedding,
    *,
    guild_id: int,
    pending_id: int,
    resolved_by_id: int,
) -> Fact:
    """Confirm a staged extraction candidate into a real active fact.

    A thin wrapper around aura.db.pending_facts.confirm_pending_fact that adds
    exactly one thing: scheduling variant generation once the fact exists,
    the automatic /aura-pending half of the same single trigger point
    add_fact serves for the manual path (see this module's docstring). The
    embedding itself is unaffected -- confirm_pending_fact reuses the
    embedding computed back when the candidate was staged (see
    aura.extraction.pipeline), exactly as it did before this wrapper existed.

    Raises whatever confirm_pending_fact raises (PendingFactNotFoundError,
    PendingFactAlreadyResolvedError), unchanged and without catching them --
    a candidate that was never confirmed has no fact for variant generation to
    run against, so those paths never reach the scheduling call below.
    """
    fact = await confirm_pending_fact(
        conn, guild_id=guild_id, pending_id=pending_id, resolved_by_id=resolved_by_id
    )
    _schedule_variant_generation(conn, model, fact)
    return fact
