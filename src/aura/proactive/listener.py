"""The passive on_message path: gate it, record it, and -- when confident -- answer.

This is where Phase 2a-3 closed the loop, and where Phase 2b-1 inserts one more
step into it. Each qualifying message in a proactive-enabled channel goes
through the full gate (see aura.proactive.gate), its decision trail is written
to the diagnostic table, and a message the gate finds ELIGIBLE now waits out a
grace period (see aura.proactive.grace) before being handed to the responder
(see aura.proactive.responder), which is the only thing here that spends money
or posts -- and only when its own hard code-gate agrees.

The pipeline order is deliberate:
  0. should_classify -- is this human guild text at all? (pure, free)
  0.5. grace-period cancellation notice -- could this message be a different
     human answering someone else's still-pending question in this channel?
     An in-memory dict lookup, unconditional and free, so it runs before any
     other gate and regardless of what this message's own pipeline decides.
  1. channel-enabled -- has a moderator opted this channel in? THE cheapest
     gate among the ones that cost a DB read, so a disabled channel incurs
     zero further computation: no embedding inference, no fact scan, no
     diagnostic write.
  2. the staged gate -- question-likeness, fact confidence, budget (free/local
     until the slot claim), recorded as a trail.
  2.5. the grace period -- behind the ELIGIBLE verdict, with the escalation
     slot already spent: wait for a human to answer first, then recheck
     freshness before ever reaching Stage 3.
  3. the responder -- paid synthesis + self-assessment + public post.

The filtering half is a pure predicate (should_classify) rather than inline
conditions, so every exclusion is independently testable without a Discord
connection, per CLAUDE.md's testing principle.
"""
from __future__ import annotations

import logging
import unicodedata

import aiosqlite
import discord
from fastembed import TextEmbedding

from aura.config import Settings
from aura.db.connection import utc_now
from aura.db.proactive_channel_config import is_channel_enabled
from aura.db.proactive_signals import (
    GracePeriodOutcome,
    record_signal,
    update_grace_outcome,
    update_synthesis_outcome,
)
from aura.db.proactive_state import is_still_freshest_escalation
from aura.proactive.gate import ProactiveGateConfig, evaluate_message
from aura.proactive.grace import GraceRegistry, GraceWaitOutcome
from aura.proactive.question_detector import QuestionDetector
from aura.proactive.responder import ProactiveResponseOutcome, respond_with_synthesis

logger = logging.getLogger(__name__)

# Everything else Discord sends through this event is text Discord itself
# wrote -- join notices, pin notifications, boost announcements, thread
# creation, call updates -- not a member asking anything.
_CLASSIFIABLE_MESSAGE_TYPES = frozenset(
    {discord.MessageType.default, discord.MessageType.reply}
)

# Unicode general categories that render as nothing. Cf is the one that
# matters in practice (zero-width space/joiner, bidi marks, BOM); Cc and the
# three separator categories are included because they are equally invisible
# and equally cheap to exclude. Kept as a duplicate of
# aura.extraction.pipeline._INVISIBLE_CATEGORIES rather than a shared
# constant -- see should_classify's docstring for why the two paths do not
# share code.
_INVISIBLE_CATEGORIES = frozenset({"Cf", "Cc", "Zs", "Zl", "Zp"})


def _has_visible_content(content: str) -> bool:
    """Whether content contains at least one character a human could actually read.

    str.strip() is not enough: it removes whitespace as Python defines it,
    which does NOT include the Unicode format characters (category Cf) a
    Discord message can be composed entirely of -- zero-width space,
    zero-width joiner, the bidi marks, the byte-order mark.
    `"\\u200b\\u200b".strip()` is still truthy, so a message nobody can see
    would otherwise reach the question detector.

    Identical in behaviour and reasoning to
    aura.extraction.pipeline._has_visible_content, found and fixed there in
    Phase 3a-2's adversarial pass (reports/phase-3a-2.txt Section 10), which
    flagged this exact gap here as real but out of that sub-phase's scope.
    Duplicated rather than imported for the same reason should_classify as a
    whole is a duplicate of should_extract -- see should_classify's
    docstring.
    """
    return any(
        not character.isspace() and unicodedata.category(character) not in _INVISIBLE_CATEGORIES
        for character in content
    )


def should_classify(message: discord.Message) -> bool:
    """Whether message is human-written guild text worth scoring at all.

    Pure and side-effect free, and deliberately independent of any client
    state: each exclusion below is a rule about the message itself, so each
    can be verified on its own.
    """
    if message.guild is None:
        # A DM. Proactive relief is a server-scoped idea -- there is no
        # shared channel to relieve and no guild to attribute the signal to.
        return False

    if message.author.bot or message.webhook_id is not None:
        # Other bots, webhook integrations, and Aura itself (its own author
        # is a bot user, so this covers the bot's own messages and its edits
        # of them without needing to know its own ID here).
        return False

    if message.interaction_metadata is not None:
        # A response to a slash command or context menu. That path already
        # has its own explicit, user-invoked trigger; scoring it would count
        # Aura's own answers as questions asked.
        return False

    if message.type not in _CLASSIFIABLE_MESSAGE_TYPES:
        return False

    # Attachment-only, sticker-only, or embed-only messages carry no text to
    # score. Whitespace-only content is the same case wearing a disguise --
    # and so, per _has_visible_content, is content built entirely from
    # Unicode format characters that render as nothing.
    return _has_visible_content(message.content)


def _log_reference(message: discord.Message) -> str:
    """Best-effort "channel/message" identifier for a log line, safe on a broken message.

    Called only from the failure path below, where raising would defeat the
    very guarantee that path exists to provide. On the installed discord.py
    every attribute it reads is a plain slot that cannot raise -- but a
    "never raises" contract that depends on a third-party library's internal
    storage choice is not one this project gets to claim, so it is made true
    here instead of assumed.
    """
    try:
        return f"{message.channel.id}/{message.id}"
    except Exception:
        return "<unidentifiable message>"


async def handle_message(
    message: discord.Message,
    *,
    db: aiosqlite.Connection,
    detector: QuestionDetector,
    model: TextEmbedding,
    config: ProactiveGateConfig,
    settings: Settings,
    grace_registry: GraceRegistry,
) -> None:
    """Run message through the full proactive pipeline, recording and (if confident) answering.

    Catches every exception on purpose, and catches it around the filtering
    and the channel-enabled gate as well as the evaluation and the response.
    This runs on every message in every channel Aura can see, so one malformed
    message, one embedding failure, one busy database or one failed post must
    degrade into a log line -- never into an exception travelling back up
    through the gateway's event dispatch. discord.py would log an unhandled
    exception here rather than drop the connection, but "the framework probably
    survives it" is exactly the grey area CLAUDE.md rules out.

    The failure direction matters and is deliberate: an exception anywhere in
    here means the gate fails closed -- a broken evaluation can never authorize
    spending, and a broken response never posts. The escalation slot is claimed
    atomically inside the gate and is never refunded, so a crash after it is
    claimed spends the slot rather than reopening it (the conservative
    direction for a spend limit).

    asyncio.CancelledError inherits from BaseException, not Exception, so a
    shutdown cancelling this task still propagates as it should.
    """
    try:
        if not should_classify(message):
            return

        assert message.guild is not None  # guaranteed by should_classify

        # Phase 2b-1: could this message be a different human answering
        # someone else's still-pending question in this same channel? Checked
        # unconditionally, before any other gate, and independent of whatever
        # THIS message's own pipeline goes on to decide -- a message that
        # itself fails Stage 1, or arrives in a channel that isn't proactive-
        # enabled, can still be the human answer that cancels another
        # message's pending grace period. Free: a miss is one dict lookup.
        grace_registry.notice_human_message(
            channel_id=message.channel.id,
            author_id=message.author.id,
            message_id=message.id,
        )

        # The cheapest gate that costs a DB read: a channel no moderator has
        # opted in stops here, before any embedding inference, fact scan or
        # diagnostic write. A disabled channel must incur literally zero of
        # that work, not merely skip the post -- so this single indexed
        # lookup is all it costs.
        if not await is_channel_enabled(db, channel_id=message.channel.id):
            return

        decision = await evaluate_message(
            db,
            model,
            detector,
            guild_id=message.guild.id,
            channel_id=message.channel.id,
            message_id=message.id,
            content=message.content,
            config=config,
            # One clock read for the whole decision, taken here rather than
            # inside the gate so the cooldown cutoff and the daily-cap day key
            # are derived from the same instant -- two reads could straddle
            # midnight and file a row under one day using another day's window.
            now=utc_now(),
        )

        # Record the gate trail immediately, BEFORE the seconds-long synthesis
        # call. If synthesis later fails or a redelivered duplicate arrives
        # while it runs, the ELIGIBLE trail is already on the row and cannot be
        # overwritten by a DUPLICATE_DELIVERY artefact (see
        # update_synthesis_outcome).
        await record_signal(
            db,
            guild_id=message.guild.id,
            channel_id=message.channel.id,
            message_id=message.id,
            decision=decision,
        )

        if decision.would_escalate:
            # Behind the ELIGIBLE verdict only, and behind a slot that has
            # already been spent: wait out the grace period, then hand off to
            # the responder if nothing stood it down, then record what it
            # decided onto the same trail row so the full decision -- gate,
            # grace period AND synthesis -- is visible to a moderator.
            outcome = await _wait_then_respond(
                message, db=db, model=model, settings=settings, grace_registry=grace_registry
            )
            await update_synthesis_outcome(
                db,
                channel_id=message.channel.id,
                message_id=message.id,
                answers_question=outcome.answers_question,
                posted=outcome.posted,
            )
            if outcome.posted:
                logger.info("Aura posted a proactive answer to message %s", _log_reference(message))
    except Exception:
        logger.exception(
            "Proactive pipeline failed for message %s", _log_reference(message)
        )


async def _wait_then_respond(
    message: discord.Message,
    *,
    db: aiosqlite.Connection,
    model: TextEmbedding,
    settings: Settings,
    grace_registry: GraceRegistry,
) -> ProactiveResponseOutcome:
    """Phase 2b-1's policy: wait for a human first, recheck freshness, then respond.

    Called only behind the gate's ELIGIBLE verdict, with an escalation slot
    already spent (see aura.proactive.gate) -- this function's whole job is
    deciding whether that already-paid-for chance is actually spent on a
    synthesis call or given up to a human instead. Never raises for any of
    its own outcomes (cancellation, a stale recheck); the caller's single
    record-and-log path always runs regardless of which one occurs.

    grace_period_outcome is written PENDING the instant the wait starts and
    exactly once more when it ends, mirroring the split-write pattern
    update_synthesis_outcome already uses one stage later -- see
    aura.db.proactive_signals.GracePeriodOutcome.
    """
    assert message.guild is not None  # guaranteed by should_classify, upstream
    channel_id = message.channel.id

    await update_grace_outcome(
        db, channel_id=channel_id, message_id=message.id, outcome=GracePeriodOutcome.PENDING
    )

    wait_outcome = await grace_registry.wait(
        channel_id=channel_id,
        asker_id=message.author.id,
        message_id=message.id,
        seconds=settings.proactive_grace_period_seconds,
    )

    if wait_outcome is GraceWaitOutcome.CANCELLED_BY_HUMAN:
        await update_grace_outcome(
            db,
            channel_id=channel_id,
            message_id=message.id,
            outcome=GracePeriodOutcome.CANCELLED_BY_HUMAN,
        )
        return ProactiveResponseOutcome(answers_question=None, posted=False)

    # The timer expired with nobody else answering. Recheck freshness before
    # spending the paid call at all -- one step earlier than
    # aura.proactive.responder's own pre-post recheck -- so a channel disabled
    # mid-wait, or a channel a newer grant has since superseded, never even
    # reaches synthesis.
    if not await _still_fresh_enough_for_synthesis(
        db, channel_id=channel_id, message_id=message.id
    ):
        await update_grace_outcome(
            db,
            channel_id=channel_id,
            message_id=message.id,
            outcome=GracePeriodOutcome.STOOD_DOWN_ON_RECHECK,
        )
        return ProactiveResponseOutcome(answers_question=None, posted=False)

    await update_grace_outcome(
        db,
        channel_id=channel_id,
        message_id=message.id,
        outcome=GracePeriodOutcome.EXPIRED_AND_PROCEEDED,
    )
    return await respond_with_synthesis(message, db=db, model=model, settings=settings)


async def _still_fresh_enough_for_synthesis(
    conn: aiosqlite.Connection, *, channel_id: int, message_id: int
) -> bool:
    """The wake-time freshness recheck, run once the grace period expires normally.

    Generalizes the same "recheck immediately before acting on delayed state"
    principle Phase 2a-3 already applied to a mid-flight /aura-config change
    (see aura.proactive.responder's own pre-post recheck) -- applied one stage
    earlier here, before the paid call rather than only before the post it
    might produce.

    Checks channel-enabled state and per-channel cooldown freshness.
    Deliberately does NOT re-check the daily cap: this message's own
    escalation slot was already granted atomically before the wait began (see
    aura.proactive.gate), the ledger it was granted against only ever grows,
    and a cap that can never be exceeded cannot retroactively un-grant a slot
    that already exists -- there is no stale state for a cap recheck to catch
    that the original atomic grant did not already rule out. What CAN go
    stale is the channel switch (hot-toggleable via /aura-config at any time)
    and, only under a misconfigured PROACTIVE_GRACE_PERIOD_SECONDS that isn't
    comfortably below PROACTIVE_COOLDOWN_SECONDS, whether a second message in
    the same channel has since earned a newer grant; both are checked here.
    """
    if not await is_channel_enabled(conn, channel_id=channel_id):
        return False
    return await is_still_freshest_escalation(conn, channel_id=channel_id, message_id=message_id)
