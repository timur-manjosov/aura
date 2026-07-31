"""/aura-config: a moderator's on/off switches for proactive relief and
automatic fact extraction, per channel.

This is the real mod control for CLAUDE.md's second trigger -- the one that
decides which channels Aura may volunteer answers in -- and, since Phase 3a-1,
for the independent per-channel gate Phase 3a's automatic fact extraction will
read once it is wired up (aura.extraction, aura.db.extraction_channel_config).
It configures two existing mechanisms; it is not a new one, and it has no
moderation authority of any kind, consistent with CLAUDE.md's non-goals.

Both options are optional and independent -- a moderator can flip either one,
or both in the same call -- because reports/phase-3-pre-analysis.md Section 1c
found a real risk in coupling extraction's channel scoping to proactive
relief's: a channel opted into answering questions is not necessarily one a
moderator wants read for facts, and the reverse holds just as much. At least
one must be given; a call with neither is rejected before touching the
database, since there is nothing to persist and no confirmation to give.

Mod-gated on manage_guild, the same permission the fact commands and the debug
command use, so "who may configure Aura" is one consistent answer across the
bot. A channel with no setting is OFF for both switches (see
aura.db.proactive_channel_config and aura.db.extraction_channel_config), so
this command is how a moderator turns either on -- and off again.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import discord
from discord import app_commands

from aura.db.extraction_channel_config import set_extraction_enabled
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
    extraction=(
        "Whether Aura may automatically read that channel's messages for facts "
        "(on) or must not (off)."
    ),
)
@app_commands.guild_only()
@app_commands.checks.has_permissions(manage_guild=True)
async def config_command(
    interaction: discord.Interaction[AuraClient],
    channel: discord.TextChannel,
    proactive: bool | None = None,
    extraction: bool | None = None,
) -> None:
    """Turn proactive relief and/or automatic fact extraction on or off for one channel.

    Both switches are independent (see this module's docstring); each is only
    written, and only mentioned in the confirmation, if the moderator actually
    passed a value for it. A call with neither is rejected before either
    switch is touched.
    """
    assert interaction.guild_id is not None  # guaranteed by guild_only()
    locale = str(interaction.locale)

    if proactive is None and extraction is None:
        await interaction.response.send_message(
            t("config_no_options_error", locale), ephemeral=True
        )
        return

    db = interaction.client.db
    assert db is not None  # setup_hook always finishes before commands go live

    confirmation_keys: list[str] = []

    if proactive is not None:
        await set_channel_enabled(
            db,
            guild_id=interaction.guild_id,
            channel_id=channel.id,
            enabled=proactive,
            updated_by_id=interaction.user.id,
        )
        confirmation_keys.append(
            "config_proactive_enabled" if proactive else "config_proactive_disabled"
        )

    if extraction is not None:
        await set_extraction_enabled(
            db,
            guild_id=interaction.guild_id,
            channel_id=channel.id,
            enabled=extraction,
            updated_by_id=interaction.user.id,
        )
        confirmation_keys.append(
            "config_extraction_enabled" if extraction else "config_extraction_disabled"
        )

    message = "\n".join(t(key, locale, channel=channel.mention) for key in confirmation_keys)
    # Ephemeral: a configuration confirmation is for the moderator who ran it,
    # not an announcement to the channel.
    await interaction.response.send_message(message, ephemeral=True)


config_command.error(_handle_config_command_error)


def register_config_command(tree: app_commands.CommandTree) -> None:
    """Register /aura-config onto tree."""
    tree.add_command(config_command)
