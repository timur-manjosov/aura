"""/aura-config: a moderator's on/off switch for proactive relief, per channel.

This is the real mod control for CLAUDE.md's second trigger -- the one that
decides which channels Aura may volunteer answers in. It configures the
existing Trigger 2 mechanism; it is not a fifth mechanism, and it has no
moderation authority of any kind, consistent with CLAUDE.md's non-goals.

Mod-gated on manage_guild, the same permission the fact commands and the debug
command use, so "who may configure Aura" is one consistent answer across the
bot. A channel with no setting is OFF (proactive relief is opt-in; see
aura.db.proactive_channel_config), so this command is how a moderator turns it
on -- and off again.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import discord
from discord import app_commands

from aura.db.proactive_channel_config import set_channel_enabled
from aura.i18n import t

if TYPE_CHECKING:
    from aura.main import AuraClient

logger = logging.getLogger(__name__)


async def _handle_config_command_error(
    interaction: discord.Interaction[AuraClient], error: app_commands.AppCommandError
) -> None:
    """Turn the permission check's failure into a clean localized reply, log everything else.

    Attaching this via .error() stops CommandTree's default logging for this
    command (it only logs when a command has no local handler), so anything
    other than the permission failure is logged here rather than silently
    disappearing.
    """
    if isinstance(error, app_commands.MissingPermissions):
        locale = str(interaction.locale)
        message = t("config_permission_error", locale)
        if interaction.response.is_done():
            await interaction.followup.send(message, ephemeral=True)
        else:
            await interaction.response.send_message(message, ephemeral=True)
        return

    logger.error("Unhandled error in /aura-config", exc_info=error)


@app_commands.command(
    name="aura-config",
    description="Configure Aura for this server (moderators only).",
)
@app_commands.describe(
    channel="The channel to configure.",
    proactive="Whether Aura may volunteer answers in that channel (on) or must stay silent (off).",
)
@app_commands.guild_only()
@app_commands.checks.has_permissions(manage_guild=True)
async def config_command(
    interaction: discord.Interaction[AuraClient],
    channel: discord.TextChannel,
    proactive: bool,
) -> None:
    """Turn proactive relief on or off for one channel, persisting the choice."""
    assert interaction.guild_id is not None  # guaranteed by guild_only()
    locale = str(interaction.locale)

    db = interaction.client.db
    assert db is not None  # setup_hook always finishes before commands go live

    await set_channel_enabled(
        db,
        guild_id=interaction.guild_id,
        channel_id=channel.id,
        enabled=proactive,
        updated_by_id=interaction.user.id,
    )

    key = "config_proactive_enabled" if proactive else "config_proactive_disabled"
    # Ephemeral: a configuration confirmation is for the moderator who ran it,
    # not an announcement to the channel.
    await interaction.response.send_message(
        t(key, locale, channel=channel.mention), ephemeral=True
    )


config_command.error(_handle_config_command_error)


def register_config_command(tree: app_commands.CommandTree) -> None:
    """Register /aura-config onto tree."""
    tree.add_command(config_command)
