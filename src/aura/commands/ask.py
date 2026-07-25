"""/aura-ask: anyone can ask Aura a question, answered by synthesizing the server's facts.

The direct-query trigger from CLAUDE.md's knowledge model -- the one this
whole project exists to serve. Unlike every other command so far, this has
no permission gate; it's explicitly open to any member.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import discord
from discord import app_commands

from aura.config import ModelComponent
from aura.embeddings import find_similar_facts
from aura.i18n import t
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


def _truncate(content: str, limit: int) -> str:
    """Truncate content to limit characters, appending an ellipsis if it was cut."""
    if len(content) <= limit:
        return content
    return content[: limit - 1] + "…"


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
    """Answer question by synthesizing across the guild's relevant active facts, with sources."""
    assert interaction.guild_id is not None  # guaranteed by guild_only()
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

    results = await find_similar_facts(db, model, guild_id=interaction.guild_id, query=question)
    relevant_facts = [fact for fact, score in results if score >= settings.similarity_threshold]

    if not relevant_facts:
        # A normal outcome, not an error -- Aura simply doesn't have
        # anything relevant yet. No LLM call: this both saves cost and
        # avoids handing the model irrelevant facts and having it try to
        # answer anyway.
        await interaction.followup.send(t("ask_no_info", locale))
        return

    # Resolve the model through the one seam every trigger uses; is_llm_configured
    # above already guaranteed this component resolves to a non-empty model.
    model_name = settings.resolve_model(ModelComponent.SYNTHESIS)
    assert model_name is not None  # guaranteed by is_llm_configured() above
    result = await synthesize_answer(relevant_facts, question, locale, model=model_name)

    if result is None:
        await interaction.followup.send(t("ask_error", locale))
        return

    embed = discord.Embed(description=_truncate(result.answer, _ANSWER_DISPLAY_LIMIT))
    cited_facts = [fact for fact in relevant_facts if fact.id in result.used_fact_ids]
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
    """Register /aura-ask onto tree."""
    tree.add_command(ask_command)
