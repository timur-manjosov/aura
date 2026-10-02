"""/aura-ask: anyone can ask Aura a question, answered by synthesizing the server's facts.

The direct-query trigger from CLAUDE.md's knowledge model -- the one this
whole project exists to serve. Unlike every other command so far, this has
no permission gate; it's explicitly open to any member.

**Every paid answer is bounded.** A question that matches at least one fact
claims a slot in the guild's daily /aura-ask ledger (aura.db.ask_state) right
before synthesis, after the operator-wide brake (aura.db.cross_guild_budget)
has agreed. Which caps apply is the plan gate's answer at that moment. When a
cap or the brake says no, the asker is not refused: they get, visible only to
them, a note that today's AI answers are used up and up to three of the facts
retrieval already found -- no model call, no slot, no cost.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, time, timedelta
from typing import TYPE_CHECKING, Final

import aiosqlite
import discord
from discord import app_commands

from aura.billing import PlanGate
from aura.config import ModelComponent, Settings
from aura.db.ask_state import AskCallOutcome, try_acquire_ask_call_slot
from aura.db.connection import utc_day, utc_now
from aura.db.cross_guild_budget import enforce_cross_guild_budget
from aura.db.models import Fact
from aura.discord_context import channel_display_name, fact_channel_names
from aura.embeddings import find_similar_facts
from aura.grounding import (
    ASK_GROUNDING_TIMEOUT_SECONDS,
    GroundingOutcome,
    verify_answer_grounded,
)
from aura.i18n import t
from aura.links_service import expand_with_linked_facts
from aura.rendering import discord_timestamp, inline_fact_text, source_link
from aura.synthesis import synthesize_answer

if TYPE_CHECKING:
    from aura.main import AuraClient

logger = logging.getLogger(__name__)

# The moment a command can trigger a real, metered API call, an unthrottled
# command is a live cost-exposure problem, not a hypothetical one. This is
# a safety margin, not a tunable feature -- a plain constant, not a config
# value (the OpenRouter account's own spending cap is the outer safety net;
# this is the inner one).
_COOLDOWN_USES = 1
_COOLDOWN_SECONDS = 30.0

_ANSWER_DISPLAY_LIMIT = 4096  # Discord's own embed description hard cap

# A question is cut to this many characters before it is embedded and sent to
# the model. Discord lets a slash-command option carry 6,000, which made the
# question the one input of a paid call with no bound of its own; no genuine
# question about a server needs more than this.
_MAX_QUESTION_CHARS: Final = 1000

# How many retrieved facts the free, no-model answer lists. Retrieval returns
# them best match first, and three keeps the reply a pointer to the sources
# rather than a wall of text standing in for the answer it could not write.
_FREE_ANSWER_FACT_LIMIT: Final = 3

# How much of a guild ID a log line may carry: enough to tell a handful of
# guilds apart while reading the log, not enough to identify one.
_LOGGED_GUILD_ID_DIGITS: Final = 4


def _truncate(content: str, limit: int) -> str:
    """Truncate content to limit characters, appending an ellipsis if it was cut."""
    if len(content) <= limit:
        return content
    return content[: limit - 1] + "…"


def _guild_log_label(guild_id: int) -> str:
    """Return the shortened guild ID a log line may carry."""
    return f"{str(guild_id)[:_LOGGED_GUILD_ID_DIGITS]}…"


def _next_utc_midnight(now: datetime) -> datetime:
    """Return the start of the UTC day after `now` -- when every daily cap resets."""
    today = now.astimezone(UTC).date()
    return datetime.combine(today + timedelta(days=1), time(0), tzinfo=UTC)


async def _claim_paid_answer(
    db: aiosqlite.Connection,
    *,
    settings: Settings,
    plan_gate: PlanGate,
    guild_id: int,
    user_id: int,
    now: datetime,
) -> AskCallOutcome:
    """Decide whether this question may spend a paid answer, claiming the slot if so.

    Parameters
    ----------
    db
        Open database connection.
    settings
        Loaded configuration: the three caps and the operator-wide budget.
    plan_gate
        Decides whether the guild is on Pro at this moment.
    guild_id, user_id
        Who is asking, and where.
    now
        One clock read for the brake and the slot, so both see the same UTC day.

    Returns
    -------
    AskCallOutcome
        GRANTED when a slot was claimed and synthesis may run. Otherwise the
        reason it may not; a HARD-mode operator budget reports as
        GUILD_CAP_REACHED, since the asker sees the same thing either way.

    Notes
    -----
    The brake comes first and claims nothing, the same order as at every other
    ledger's call site (see aura.db.cross_guild_budget). The plan is read here,
    at the moment of the call, so a plan change takes effect on the next
    question while the answers already spent today keep counting. An
    unenforced gate (BILLING_MODE=disabled) and a complimentary guild are Pro,
    exactly as the gate says.
    """
    if not await enforce_cross_guild_budget(
        db,
        day=utc_day(now),
        budget_usd=settings.cross_guild_daily_budget_usd,
        mode=settings.cross_guild_budget_mode,
    ):
        logger.warning(
            "/aura-ask in guild %s answered without a model call: the operator's "
            "cross-guild daily budget is exceeded (mode=hard)",
            _guild_log_label(guild_id),
        )
        return AskCallOutcome.GUILD_CAP_REACHED

    # /aura-ask is a Free feature: the plan only picks which cap applies, it
    # never decides whether the question is answered. The gate is an in-memory
    # view that is not supposed to fail, but if it ever did, this command must
    # keep answering (the Phase 4c audit's invariant for Free features), so a
    # failure means the smaller Free caps -- the cheaper direction -- and never
    # an error reply.
    try:
        is_pro = plan_gate.allows_pro(guild_id)
    except Exception:
        logger.exception(
            "/aura-ask in guild %s: the plan gate failed; using the Free caps",
            _guild_log_label(guild_id),
        )
        is_pro = False

    if is_pro:
        guild_cap, user_cap = settings.ask_daily_cap_pro, None
    else:
        guild_cap, user_cap = settings.ask_daily_cap_free, settings.ask_user_daily_cap_free

    attempt = await try_acquire_ask_call_slot(
        db, guild_id=guild_id, user_id=user_id, guild_cap=guild_cap, user_cap=user_cap, now=now
    )
    logger.info(
        "/aura-ask in guild %s: %s (guild %d of %d today, member %d of %s)",
        _guild_log_label(guild_id),
        attempt.outcome.value,
        attempt.guild_count,
        attempt.guild_cap,
        attempt.user_count,
        "no cap" if attempt.user_cap is None else attempt.user_cap,
    )
    return attempt.outcome


def _free_answer_embed(
    facts: list[Fact], outcome: AskCallOutcome, locale: str, *, now: datetime
) -> discord.Embed:
    """Build the no-model reply: the limit note and the best-matching facts, with sources.

    Parameters
    ----------
    facts
        Retrieved facts, best match first, at least one. Only the first
        `_FREE_ANSWER_FACT_LIMIT` are shown.
    outcome
        Which ceiling refused the paid answer; picks the member or guild note.
    locale
        The asker's locale.
    now
        The moment of the question, for when the caps reset.

    Returns
    -------
    discord.Embed
        The note, then one line per fact: its sentence as a link to the source
        message, and the date it was recorded.

    Notes
    -----
    Each fact goes through aura.rendering, the same escaping the digest and
    onboarding use: a fact containing `]` cannot break out of its link label,
    and whitespace or invisible characters cannot hide one. The fact text is
    shown, never rephrased -- the answer is true by construction because it is
    only what was recorded.
    """
    key = (
        "ask_limit_user_reached"
        if outcome is AskCallOutcome.USER_CAP_REACHED
        else "ask_limit_guild_reached"
    )
    reset = int(_next_utc_midnight(now).timestamp())
    note = t(key, locale, reset=f"<t:{reset}:R>")
    lines = [
        f"• [{inline_fact_text(fact.content)}]({source_link(fact)}) · "
        f"{discord_timestamp(fact.created_at)}"
        for fact in facts[:_FREE_ANSWER_FACT_LIMIT]
    ]
    return discord.Embed(
        description=_truncate(note + "\n\n" + "\n".join(lines), _ANSWER_DISPLAY_LIMIT)
    )


async def _send_free_answer(
    interaction: discord.Interaction[AuraClient], embed: discord.Embed
) -> None:
    """Send the no-model reply so that only the asker sees it.

    Parameters
    ----------
    interaction
        The deferred interaction.
    embed
        The reply built by `_free_answer_embed`.

    Returns
    -------
    None

    Notes
    -----
    The interaction was deferred publicly, before anyone knew a cap would
    bind, and Discord turns the first followup after a defer into the deferred
    message itself -- public, whatever flag the followup carries. Deleting the
    deferred message first makes the followup a message of its own, which
    honours the ephemeral flag. If the delete fails the reply is still sent:
    an answer the channel can see beats none, and its content is only facts
    the channel's members could already ask about.
    """
    try:
        await interaction.delete_original_response()
    except discord.HTTPException:
        logger.warning(
            "/aura-ask could not remove its deferred message before a free answer; "
            "the reply may be visible to the channel"
        )
    await interaction.followup.send(embed=embed, ephemeral=True)


async def _handle_ask_command_error(
    interaction: discord.Interaction[AuraClient], error: app_commands.AppCommandError
) -> None:
    """The cooldown is the only check /aura-ask has (no permission gate) -- handle it cleanly.

    Attaching this via .error() stops CommandTree's default logging for
    this command (it only logs when a command has no local handler), so
    anything other than the cooldown is logged here instead of silently
    disappearing.
    """
    if isinstance(error, app_commands.CommandOnCooldown):
        locale = str(interaction.locale)
        message = t("ask_cooldown", locale, seconds=round(error.retry_after))
        if interaction.response.is_done():
            await interaction.followup.send(message, ephemeral=True)
        else:
            await interaction.response.send_message(message, ephemeral=True)
        return

    logger.error("Unhandled error in /aura-ask", exc_info=error)


@app_commands.command(name="aura-ask", description="Ask Aura a question about this server.")
@app_commands.describe(question="What do you want to know?")
@app_commands.guild_only()
@app_commands.checks.cooldown(_COOLDOWN_USES, _COOLDOWN_SECONDS)
async def ask_command(interaction: discord.Interaction[AuraClient], question: str) -> None:
    """Answer question by synthesizing across the guild's relevant active facts, with sources.

    Parameters
    ----------
    interaction
        The command invocation. Carries the invoker's locale, the guild it was
        run in, and the client the database, models and plan gate hang off.
    question
        What the user asked, taken verbatim and fenced as untrusted input in the
        synthesis prompt.

    Returns
    -------
    None
    """
    assert interaction.guild_id is not None  # guaranteed by guild_only()
    assert interaction.channel_id is not None  # guaranteed by guild_only(): always a real channel
    locale = str(interaction.locale)

    # Discord requires an initial response within 3 seconds; an LLM call
    # will essentially never be that fast. Deferring first, before any of
    # the slow work below starts, is not an edge case to catch later --
    # get this wrong and the feature fails on every single use, not
    # occasionally.
    await interaction.response.defer()

    client = interaction.client
    settings = client.settings

    if not settings.is_llm_configured(ModelComponent.SYNTHESIS):
        await interaction.followup.send(t("ask_not_configured", locale))
        return

    db = client.db
    assert db is not None  # setup_hook always finishes before commands go live
    model = client.embedding_model
    assert model is not None  # setup_hook always finishes before commands go live

    # Cut before the embedding as well as before synthesis, so the question
    # retrieval matched is the question the model reads.
    question = question[:_MAX_QUESTION_CHARS]

    results = await find_similar_facts(db, model, guild_id=interaction.guild_id, query=question)
    relevant_facts = [fact for fact, score in results if score >= settings.similarity_threshold]

    if not relevant_facts:
        # A normal outcome, not an error -- Aura simply doesn't have
        # anything relevant yet. No LLM call: this both saves cost and
        # avoids handing the model irrelevant facts and having it try to
        # answer anyway.
        await interaction.followup.send(t("ask_no_info", locale))
        return

    # CLAUDE.md's fourth knowledge-model component, on the read path: a fact a
    # moderator deliberately linked to one of these becomes available to cite
    # too, resolved through any supersession that has happened since. Only the
    # candidate set widens -- what is actually cited stays the synthesis model's
    # decision, and the threshold above is untouched, so this can never turn a
    # question Aura has nothing for into one it answers anyway (an empty
    # relevant_facts already returned, above).
    synthesis_facts = await expand_with_linked_facts(
        db, guild_id=interaction.guild_id, facts=relevant_facts
    )

    # The slot is claimed here and nowhere earlier: only now is a paid call
    # certain, since a question with no matching fact returned above without
    # one. It is never refunded -- a failed or rejected answer still spent it.
    plan_gate = client.plan_gate
    assert plan_gate is not None  # setup_hook always finishes before commands go live
    now = utc_now()
    outcome = await _claim_paid_answer(
        db,
        settings=settings,
        plan_gate=plan_gate,
        guild_id=interaction.guild_id,
        user_id=interaction.user.id,
        now=now,
    )
    if outcome is not AskCallOutcome.GRANTED:
        await _send_free_answer(
            interaction, _free_answer_embed(relevant_facts, outcome, locale, now=now)
        )
        return

    # Resolve the model through the one seam every trigger uses; is_llm_configured
    # above already guaranteed this component resolves to a non-empty model.
    model_name = settings.resolve_model(ModelComponent.SYNTHESIS)
    assert model_name is not None  # guaranteed by is_llm_configured() above
    result = await synthesize_answer(
        synthesis_facts,
        question,
        locale,
        model=model_name,
        question_channel_name=channel_display_name(interaction.channel, interaction.channel_id),
        question_asked_at=interaction.created_at,
        fact_channel_names=fact_channel_names(
            interaction.guild, {fact.channel_id for fact in synthesis_facts}
        ),
    )

    if result is None:
        await interaction.followup.send(t("ask_error", locale))
        return

    # Filtered against synthesis_facts, not relevant_facts: a fact the model
    # cited only because a link made it available must still reach the
    # grounding check below and the source list further down. Narrowing this
    # back to the similarity hits would hide exactly the citations this phase
    # exists to produce -- from the check that verifies them, and from the
    # reader who has to be able to follow them.
    cited_facts = [fact for fact in synthesis_facts if fact.id in result.used_fact_ids]

    # The independent grounding check, and the last thing that runs before this
    # command speaks. It reads the answer synthesis just wrote against the facts
    # that answer says it drew from, and returns a verdict only -- nothing it
    # produces can reach the text below, which is why the embed is built after
    # it rather than handed to it (see aura.grounding).
    #
    # Both refusal branches reply rather than falling silent: someone explicitly
    # asked, and "I have nothing to say and will not tell you why" is a worse
    # answer than an honest one. That is the opposite of Trigger 2's policy for
    # the same two outcomes, deliberately -- see the proactive responder.
    grounding = await verify_answer_grounded(
        answer=result.answer,
        cited_facts=cited_facts,
        settings=settings,
        timeout_seconds=ASK_GROUNDING_TIMEOUT_SECONDS,
    )
    if grounding is GroundingOutcome.UNGROUNDED:
        await interaction.followup.send(t("ask_grounding_rejected", locale))
        return
    if grounding is GroundingOutcome.CHECK_FAILED:
        await interaction.followup.send(t("ask_grounding_unverified", locale))
        return

    embed = discord.Embed(description=_truncate(result.answer, _ANSWER_DISPLAY_LIMIT))
    if cited_facts:
        links = "\n".join(
            f"https://discord.com/channels/{fact.guild_id}/{fact.channel_id}/{fact.message_id}"
            for fact in cited_facts
        )
        embed.add_field(name=t("ask_sources_label", locale), value=links, inline=False)

    # A normal, visible message -- not ephemeral. Unlike the moderator
    # debug tools, a good answer has value to everyone who can see the
    # channel it was asked in.
    await interaction.followup.send(embed=embed)


ask_command.error(_handle_ask_command_error)


def register_ask_command(tree: app_commands.CommandTree) -> None:
    """Register /aura-ask onto tree.

    Parameters
    ----------
    tree
        The command tree to register into.

    Returns
    -------
    None
    """
    tree.add_command(ask_command)
