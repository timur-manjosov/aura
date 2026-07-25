"""Manual fact entry: a message context menu to create facts, and a list command."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import aiosqlite
import discord
from discord import app_commands
from fastembed import TextEmbedding

from aura.db.repository import get_active_facts
from aura.embeddings import find_similar_facts
from aura.facts_service import add_fact
from aura.i18n import t

if TYPE_CHECKING:
    from aura.main import AuraClient

logger = logging.getLogger(__name__)

# Discord's own hard cap for a modal TextInput's default/max_length, not a
# choice made here -- see discord.ui.TextInput's docs ("Can only be up to
# 4000 characters"). Truncate to this, never guess or hardcode a different one.
_TEXT_INPUT_MAX_LENGTH = 4000

_LIST_DISPLAY_LIMIT = 20
# Comfortably under an embed field value's 1024-character hard cap, and
# facts are meant to be distilled sentences (see CLAUDE.md) -- this is a
# safety margin for a moderator pasting something far longer, not the
# expected case.
_FIELD_VALUE_DISPLAY_LIMIT = 200

# Discord's slash command STRING options default to a 6000-character max --
# comfortably enough for a full-text query, but far too long to ever put
# raw into an embed title (hard-capped by Discord at 256 characters). This
# is the display-only budget for a query embedded in fact_search_results_header;
# the full, untruncated query text is still what gets embedded and searched.
_QUERY_DISPLAY_LIMIT = 100


def _truncate(content: str, limit: int) -> str:
    """Truncate content to limit characters, appending an ellipsis if it was cut."""
    if len(content) <= limit:
        return content
    return content[: limit - 1] + "…"


class AddFactModal(discord.ui.Modal):
    """Collects and distills a fact's content from a moderator, pre-filled from a message.

    Takes the DB connection and embedding model directly rather than
    reaching into interaction.client.db / .embedding_model from on_submit:
    Modal.on_submit/on_error are generic over the client type in the base
    class, so narrowing their interaction parameter to Interaction[AuraClient]
    here would be an incompatible override. Accepting both as constructor
    arguments (from callers that *aren't* overriding a generic base method,
    and so can type interaction.client precisely) sidesteps that entirely.
    """

    def __init__(
        self,
        *,
        db: aiosqlite.Connection,
        model: TextEmbedding,
        locale: str,
        guild_id: int,
        channel_id: int,
        message_id: int,
        prefill_content: str,
    ) -> None:
        super().__init__(title=_truncate(t("fact_add_modal_title", locale), 45))
        self._db = db
        self._model = model
        self._guild_id = guild_id
        self._channel_id = channel_id
        self._message_id = message_id

        default = _truncate(prefill_content, _TEXT_INPUT_MAX_LENGTH) if prefill_content else None
        self.content_input = discord.ui.TextInput(
            style=discord.TextStyle.paragraph,
            default=default,
            max_length=_TEXT_INPUT_MAX_LENGTH,
            required=True,
        )
        label_text = _truncate(t("fact_add_modal_label", locale), 45)
        self.add_item(discord.ui.Label(text=label_text, component=self.content_input))

    async def on_submit(self, interaction: discord.Interaction) -> None:
        """Validate and store the submitted content, replying ephemerally either way."""
        locale = str(interaction.locale)
        content = (self.content_input.value or "").strip()

        if not content:
            await interaction.response.send_message(
                t("fact_add_empty_error", locale), ephemeral=True
            )
            return

        fact = await add_fact(
            self._db,
            self._model,
            guild_id=self._guild_id,
            channel_id=self._channel_id,
            message_id=self._message_id,
            content=content,
        )
        await interaction.response.send_message(
            t("fact_add_success", locale, fact_id=fact.id), ephemeral=True
        )

    async def on_error(self, _interaction: discord.Interaction, error: Exception) -> None:
        """Log unexpected failures through Aura's own logger.

        Attaching a local error handler (this method, or the command-level
        ones below) suppresses discord.py's own default logging for that
        interaction, so this replaces it rather than adding to it.
        """
        logger.error("Unhandled error submitting AddFactModal", exc_info=error)


async def _handle_fact_command_error(
    interaction: discord.Interaction[AuraClient], error: app_commands.AppCommandError
) -> None:
    """Shared error handler for both fact commands.

    Attaching this via .error() stops CommandTree's default logging for
    these two commands (it only logs when a command has no local handler),
    so anything other than the permission-check failure is logged here
    instead, rather than silently disappearing.
    """
    if isinstance(error, app_commands.MissingPermissions):
        locale = str(interaction.locale)
        message = t("fact_permission_error", locale)
        if interaction.response.is_done():
            await interaction.followup.send(message, ephemeral=True)
        else:
            await interaction.response.send_message(message, ephemeral=True)
        return

    logger.error(
        "Unhandled error in fact command %r",
        interaction.command.name if interaction.command else "<unknown>",
        exc_info=error,
    )


@app_commands.context_menu(name="Add as Aura Fact")
@app_commands.guild_only()
@app_commands.checks.has_permissions(manage_guild=True)
async def add_fact_context_menu(
    interaction: discord.Interaction[AuraClient], message: discord.Message
) -> None:
    """Open a pre-filled modal to turn the right-clicked message into a fact."""
    assert message.guild is not None  # guaranteed by guild_only()
    db = interaction.client.db
    assert db is not None  # setup_hook always finishes before commands go live
    model = interaction.client.embedding_model
    assert model is not None  # setup_hook always finishes before commands go live

    locale = str(interaction.locale)
    modal = AddFactModal(
        db=db,
        model=model,
        locale=locale,
        guild_id=message.guild.id,
        channel_id=message.channel.id,
        message_id=message.id,
        prefill_content=message.content,
    )
    await interaction.response.send_modal(modal)


@app_commands.command(name="aura-facts", description="List Aura's current active facts for this server.")
@app_commands.describe(
    query="Optional: find facts similar to this text instead of listing all of them."
)
@app_commands.guild_only()
@app_commands.checks.has_permissions(manage_guild=True)
async def list_facts_command(
    interaction: discord.Interaction[AuraClient], query: str | None = None
) -> None:
    """List the guild's active facts, or -- if query is given -- rank them by similarity to it.

    query omitted or blank: unchanged Phase 1c behavior, newest-first,
    capped at _LIST_DISPLAY_LIMIT. query given: a raw debug view over
    find_similar_facts's ranking (fact content alongside its similarity
    score) -- not the synthesized, cited answer a later phase builds on top
    of this, on purpose.
    """
    assert interaction.guild_id is not None  # guaranteed by guild_only()
    locale = str(interaction.locale)

    db = interaction.client.db
    assert db is not None  # setup_hook always finishes before commands go live

    stripped_query = query.strip() if query else ""
    if stripped_query:
        model = interaction.client.embedding_model
        assert model is not None  # setup_hook always finishes before commands go live
        await _respond_with_search_results(
            interaction, db, model, guild_id=interaction.guild_id, query=stripped_query, locale=locale
        )
        return

    facts = await get_active_facts(db, interaction.guild_id)

    if not facts:
        await interaction.response.send_message(t("fact_list_empty", locale), ephemeral=True)
        return

    ordered = sorted(facts, key=lambda fact: fact.created_at, reverse=True)
    shown = ordered[:_LIST_DISPLAY_LIMIT]
    remaining = len(ordered) - len(shown)

    embed = discord.Embed(title=t("fact_list_title", locale))
    for fact in shown:
        embed.add_field(
            name=f"#{fact.id}",
            value=_truncate(fact.content, _FIELD_VALUE_DISPLAY_LIMIT),
            inline=False,
        )
    if remaining > 0:
        embed.set_footer(text=t("fact_list_truncated_note", locale, count=remaining))

    await interaction.response.send_message(embed=embed, ephemeral=True)


async def _respond_with_search_results(
    interaction: discord.Interaction[AuraClient],
    db: aiosqlite.Connection,
    model: TextEmbedding,
    *,
    guild_id: int,
    query: str,
    locale: str,
) -> None:
    """Run find_similar_facts and render a raw debug view: content plus score, per result."""
    results = await find_similar_facts(db, model, guild_id=guild_id, query=query)
    display_query = _truncate(query, _QUERY_DISPLAY_LIMIT)

    if not results:
        await interaction.response.send_message(
            t("fact_search_no_matches", locale, query=display_query), ephemeral=True
        )
        return

    embed = discord.Embed(title=t("fact_search_results_header", locale, query=display_query))
    for fact, score in results:
        embed.add_field(
            name=f"#{fact.id} ({score:.3f})",
            value=_truncate(fact.content, _FIELD_VALUE_DISPLAY_LIMIT),
            inline=False,
        )

    await interaction.response.send_message(embed=embed, ephemeral=True)


add_fact_context_menu.error(_handle_fact_command_error)
list_facts_command.error(_handle_fact_command_error)


def register_fact_commands(tree: app_commands.CommandTree) -> None:
    """Register the fact-related context menu and slash command onto tree."""
    tree.add_command(add_fact_context_menu)
    tree.add_command(list_facts_command)
