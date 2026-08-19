"""/aura-onboarding: a moderator's switch for onboarding messages -- where they
post, and whether they run at all.

The mod control for CLAUDE.md's third trigger, a deliberate sibling of
/aura-digest for exactly the same reason /aura-digest is a sibling of
/aura-config rather than a fourth option on it: onboarding is one guild-wide
setting with exactly one channel (see the onboarding_config table comment in
schema.sql), not a per-channel switch a guild sensibly has many of at once.

Simpler than /aura-digest in one respect: there is no interval to configure,
because onboarding is not scheduled -- it fires once per member join (see
aura.onboarding.listener), so the only choices are "where" and "on or off".

Mod-gated on manage_guild, the same permission every other Aura configuration
and fact command uses.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import discord
from discord import app_commands

from aura.db.onboarding_config import get_onboarding_config, set_onboarding_config
from aura.i18n import t

if TYPE_CHECKING:
    from aura.main import AuraClient

logger = logging.getLogger(__name__)


async def _handle_onboarding_command_error(
    interaction: discord.Interaction[AuraClient], error: app_commands.AppCommandError
) -> None:
    """Turn the permission check's failure into a clean localized reply, log everything else.

    Mirrors aura.commands.digest._handle_digest_command_error exactly, for the
    same reason: attaching this via .error() stops CommandTree's default
    logging for this command, so anything other than the permission failure
    must be logged here or it disappears silently.
    """
    if isinstance(error, app_commands.MissingPermissions):
        locale = str(interaction.locale)
        message = t("onboarding_permission_error", locale)
        if interaction.response.is_done():
            await interaction.followup.send(message, ephemeral=True)
        else:
            await interaction.response.send_message(message, ephemeral=True)
        return

    logger.error("Unhandled error in /aura-onboarding", exc_info=error)


def _cannot_post_in(channel: discord.TextChannel) -> bool:
    """Whether Aura currently lacks what it needs to post an onboarding message in channel.

    Identical logic and identical reasoning to aura.commands.digest.
    _cannot_post_in: checked at configuration time so a moderator learns now
    rather than wondering why the next new member got no welcome, and it is a
    warning appended to the confirmation, never a refusal -- permissions can
    change either way after this call.
    """
    try:
        me = channel.guild.me
        if me is None:
            return False
        permissions = channel.permissions_for(me)
        return not (permissions.send_messages and permissions.embed_links)
    except Exception:
        logger.exception("Could not check onboarding permissions for channel %s", channel.id)
        return False


@app_commands.command(
    name="aura-onboarding",
    description="Configure Aura's onboarding message for new members (moderators only).",
)
@app_commands.describe(
    channel="The channel a new member's summary is posted in.",
    enabled="Whether onboarding messages are sent at all.",
)
@app_commands.guild_only()
@app_commands.checks.has_permissions(manage_guild=True)
async def onboarding_command(
    interaction: discord.Interaction[AuraClient],
    channel: discord.TextChannel | None = None,
    enabled: bool | None = None,
) -> None:
    """Set onboarding's channel, or switch it off, mirroring /aura-digest's option handling.

    The same two rules /aura-digest documents apply here, for the same
    reasons: naming a channel turns onboarding ON unless `enabled` says
    otherwise, and anything not named keeps its current value. A call with no
    options at all is rejected before touching the database.
    """
    assert interaction.guild_id is not None  # guaranteed by guild_only()
    locale = str(interaction.locale)

    if channel is None and enabled is None:
        await interaction.response.send_message(
            t("onboarding_no_options_error", locale), ephemeral=True
        )
        return

    db = interaction.client.db
    assert db is not None  # setup_hook always finishes before commands go live

    existing = await get_onboarding_config(db, guild_id=interaction.guild_id)
    if channel is not None:
        target_channel_id: int | None = channel.id
    elif existing is not None:
        target_channel_id = existing.channel_id
    else:
        target_channel_id = None

    if target_channel_id is None:
        message = "onboarding_already_off" if enabled is False else "onboarding_channel_required"
        await interaction.response.send_message(t(message, locale), ephemeral=True)
        return

    target_enabled = enabled if enabled is not None else True
    was_enabled = existing is not None and existing.onboarding_enabled

    await set_onboarding_config(
        db,
        guild_id=interaction.guild_id,
        channel_id=target_channel_id,
        enabled=target_enabled,
        updated_by_id=interaction.user.id,
    )

    if not target_enabled:
        await interaction.response.send_message(t("onboarding_disabled", locale), ephemeral=True)
        return

    channel_mention = channel.mention if channel is not None else f"<#{target_channel_id}>"
    lines = [
        t(
            "onboarding_updated" if was_enabled else "onboarding_enabled",
            locale,
            channel=channel_mention,
        )
    ]
    if channel is not None and _cannot_post_in(channel):
        lines.append(t("onboarding_permission_warning", locale, channel=channel_mention))

    # Ephemeral: a configuration confirmation is for the moderator who ran it,
    # not an announcement to the channel.
    await interaction.response.send_message("\n".join(lines), ephemeral=True)


onboarding_command.error(_handle_onboarding_command_error)


def register_onboarding_command(tree: app_commands.CommandTree) -> None:
    """Register /aura-onboarding onto tree."""
    tree.add_command(onboarding_command)
