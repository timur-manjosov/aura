"""/aura-digest: a moderator's switch for the periodic digest -- where it posts,
how often, and whether it runs at all.

The mod control for CLAUDE.md's fourth trigger, and a deliberate sibling of
/aura-config rather than a fourth option on it. Two reasons, and the second is
the one that decided it:

  * /aura-config configures per-CHANNEL switches, and a guild sensibly has many
    channels enabled for each. A digest is one guild-wide setting with exactly
    one channel (see the digest_config table comment in schema.sql), so folding
    it in would put a guild-scoped option on a command whose every other option
    is scoped to the channel argument beside it.
  * The digest also carries an interval, which no per-channel switch has. On one
    command, "channel" would mean two different things depending on which of the
    other options were given.

Mod-gated on manage_guild, the same permission every other Aura configuration
and fact command uses, so "who may configure Aura" keeps one consistent answer.
A guild with no setting has digests OFF (see aura.db.digest_config), so this
command is how a moderator turns them on -- and off again.

**Turning digests on never retro-posts.** The first digest covers what changes
from the moment they are switched on, not the server's whole history: summarizing
everything Aura already knows is the onboarding trigger's job, and a "what's
new" post that opens with two years of accumulated facts is neither new nor
readable. The confirmation says so, because a moderator who expected a summary
of the existing knowledge model should find that out from the reply rather than
from a week of waiting.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import discord
from discord import app_commands

from aura.db.digest_config import get_digest_config, set_digest_config
from aura.digest.intervals import DigestInterval, describe_interval
from aura.i18n import t

if TYPE_CHECKING:
    from aura.main import AuraClient

logger = logging.getLogger(__name__)

# What a moderator gets when they name a channel but no cadence. Weekly is the
# cadence CLAUDE.md's fourth trigger is described in terms of, and the one that
# suits a knowledge model that changes a handful of times a week: daily would
# often post a two-line digest, monthly would summarize changes nobody remembers
# the context for.
_DEFAULT_INTERVAL = DigestInterval.WEEKLY

# Choice labels are English, like every other slash-command name and description
# in this project. Localizing command METADATA needs discord.py's own
# app_commands.Translator machinery, which is a different mechanism from the
# t()-based lookup used for every reply below, and it is deferred project-wide
# (see the comment in aura.main.create_client). The digest itself -- the thing
# server members actually read -- is fully localized.
_INTERVAL_CHOICES = [
    app_commands.Choice(name="Daily", value=int(DigestInterval.DAILY)),
    app_commands.Choice(name="Weekly", value=int(DigestInterval.WEEKLY)),
    app_commands.Choice(name="Every two weeks", value=int(DigestInterval.BIWEEKLY)),
    app_commands.Choice(name="Every 30 days", value=int(DigestInterval.MONTHLY)),
]


async def _handle_digest_command_error(
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
        message = t("digest_permission_error", locale)
        if interaction.response.is_done():
            await interaction.followup.send(message, ephemeral=True)
        else:
            await interaction.response.send_message(message, ephemeral=True)
        return

    logger.error("Unhandled error in /aura-digest", exc_info=error)


def _cannot_post_in(channel: discord.TextChannel) -> bool:
    """Whether Aura currently lacks what it needs to post a digest in channel.

    Checked at configuration time purely so the moderator finds out now instead
    of wondering next week why nothing arrived -- it is a WARNING appended to
    the confirmation, never a refusal. Permissions can change after this call
    either way, so the scheduler still has to handle a failed send (it does; the
    window is retried), and refusing here would only trade a working setup for a
    blocked one whenever this check is wrong.

    Both permissions matter and both are checked: without send_messages nothing
    is posted at all, and without embed_links the message is sent and silently
    stripped to nothing, which is the more confusing of the two failures.

    Never raises. A channel whose guild has no cached `me` member yields no
    warning rather than a crash -- the check is advice, and advice must not be
    able to break the command it is attached to.
    """
    try:
        me = channel.guild.me
        if me is None:
            return False
        permissions = channel.permissions_for(me)
        return not (permissions.send_messages and permissions.embed_links)
    except Exception:
        logger.exception("Could not check digest permissions for channel %s", channel.id)
        return False


@app_commands.command(
    name="aura-digest",
    description="Configure Aura's periodic digest for this server (moderators only).",
)
@app_commands.describe(
    channel="The channel the digest is posted in.",
    interval="How often the digest is posted.",
    enabled="Whether the digest runs at all. Turning it off keeps the channel and interval.",
)
@app_commands.choices(interval=_INTERVAL_CHOICES)
@app_commands.guild_only()
@app_commands.checks.has_permissions(manage_guild=True)
async def digest_command(
    interaction: discord.Interaction[AuraClient],
    channel: discord.TextChannel | None = None,
    interval: app_commands.Choice[int] | None = None,
    enabled: bool | None = None,
) -> None:
    """Set the digest's channel and/or cadence, or switch it off.

    Every option is optional and they compose, which keeps the common edits to
    one word each: name a channel to start, pass an interval to change the
    cadence, pass enabled:false to stop. Two rules resolve the combinations, and
    both are written down here because they are conventions rather than
    deductions:

      * Naming a channel or an interval turns digests ON unless `enabled` says
        otherwise. Configuring something you did not want to happen is not a
        thing anyone means to do.
      * Everything not named keeps its current value, so changing the cadence
        never silently moves the channel and vice versa.

    A call with no options at all is rejected before touching the database:
    there is nothing to persist and no confirmation to give.
    """
    assert interaction.guild_id is not None  # guaranteed by guild_only()
    locale = str(interaction.locale)

    if channel is None and interval is None and enabled is None:
        await interaction.response.send_message(
            t("digest_no_options_error", locale), ephemeral=True
        )
        return

    db = interaction.client.db
    assert db is not None  # setup_hook always finishes before commands go live

    existing = await get_digest_config(db, guild_id=interaction.guild_id)
    if channel is not None:
        target_channel_id: int | None = channel.id
    elif existing is not None:
        target_channel_id = existing.channel_id
    else:
        target_channel_id = None

    if target_channel_id is None:
        # Nothing configured and no channel named: there is no default worth
        # guessing. Picking one for the moderator would mean Aura choosing where
        # it posts unprompted, which is the whole thing the opt-in gate exists
        # to prevent.
        message = "digest_already_off" if enabled is False else "digest_channel_required"
        await interaction.response.send_message(t(message, locale), ephemeral=True)
        return

    target_interval = (
        interval.value
        if interval is not None
        else (existing.interval_seconds if existing is not None else int(_DEFAULT_INTERVAL))
    )
    target_enabled = enabled if enabled is not None else True
    was_enabled = existing is not None and existing.digest_enabled

    await set_digest_config(
        db,
        guild_id=interaction.guild_id,
        channel_id=target_channel_id,
        interval_seconds=target_interval,
        enabled=target_enabled,
        updated_by_id=interaction.user.id,
    )

    if not target_enabled:
        await interaction.response.send_message(t("digest_disabled", locale), ephemeral=True)
        return

    interval_label = describe_interval(target_interval, locale)
    channel_mention = channel.mention if channel is not None else f"<#{target_channel_id}>"
    # Two different confirmations, because they answer two different questions.
    # A moderator switching digests on needs to know when the first one lands
    # and that it will not retro-post; one adjusting a running digest does not,
    # and telling them "the first one arrives in a week" would be wrong -- their
    # existing schedule and baseline are untouched.
    lines = [
        t(
            "digest_updated" if was_enabled else "digest_enabled",
            locale,
            channel=channel_mention,
            interval=interval_label,
        )
    ]
    if channel is not None and _cannot_post_in(channel):
        lines.append(t("digest_permission_warning", locale, channel=channel_mention))

    # Ephemeral: a configuration confirmation is for the moderator who ran it,
    # not an announcement to the channel.
    await interaction.response.send_message("\n".join(lines), ephemeral=True)


digest_command.error(_handle_digest_command_error)


def register_digest_command(tree: app_commands.CommandTree) -> None:
    """Register /aura-digest onto tree."""
    tree.add_command(digest_command)
