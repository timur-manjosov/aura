"""/aura-backfill: a moderator points the extraction chain at a channel's past.

The mod control for Phase 3b, and the FIRST command in this project built as a
subcommand group rather than one command with optional arguments. That is a
deliberate departure from /aura-config and /aura-digest, and the reason is that
those two configure a setting while this one drives a JOB. `start`, `status`,
`pause` and `cancel` are four different verbs against a thing that is running,
not four optional fields of one description -- folding them into one command
would produce exactly the ambiguity /aura-digest's own docstring rejects, where
"channel" means something different depending on which other options were given.

**Backfill is a decision, never a side effect.** Turning extraction on for a
channel with /aura-config affects messages from that moment on and nothing
before it. Reading two years of history is a separate choice with a separate
cost, so it needs a separate, explicit command -- and this one refuses to run
against a channel extraction is not already enabled for, because backfilling a
channel a moderator has not agreed to have read at all would be the same
decision made twice by one command.

**There is no `resume` subcommand, on purpose.** `start` on a paused run resumes
it exactly where its cursor stands. A fifth verb that meant "start, but only if
it already exists" would be a distinction moderators have to remember for no
benefit; the confirmation says which of the two happened.

Mod-gated on manage_guild, the same permission every other Aura configuration
and fact command uses, so "who may decide what Aura reads" keeps one consistent
answer across the bot. Every reply is ephemeral: starting a backfill is an
operational decision for the moderator who made it, not an announcement to the
channel being read.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import discord
from discord import app_commands

from aura.db.backfill_runs import (
    BackfillAlreadyActiveError,
    BackfillRun,
    BackfillState,
    get_active_run,
    get_recent_runs,
    set_run_state,
    start_backfill_run,
)
from aura.db.backfill_state import count_backfill_calls_on
from aura.db.connection import utc_day, utc_now
from aura.db.extraction_channel_config import is_extraction_enabled
from aura.i18n import t

if TYPE_CHECKING:
    from aura.main import AuraClient

logger = logging.getLogger(__name__)

# The `since:` option's one accepted shape. A plain ISO date rather than a
# datetime, and rather than one of the several formats a date parser would
# accept: Discord has no date option type, so this is a free-text string either
# way, and a single unambiguous format is the only one that means the same thing
# to a moderator in every locale. "03/04/2026" does not.
_SINCE_FORMAT = "%Y-%m-%d"

# How many finished runs /aura-backfill status lists beside the live one. Enough
# to see that a channel has been backfilled before and roughly when, short
# enough to stay inside one embed without paging.
_RECENT_RUN_LIMIT = 5

# Comfortably under an embed field value's 1024-character hard cap, matching the
# display budget every other Aura embed uses.
_FIELD_VALUE_DISPLAY_LIMIT = 900


def _parse_since(raw: str, *, now: datetime) -> datetime | None:
    """Parse a `since:` argument into a UTC midnight, or None if it is unusable.

    Accepts year-month-day and nothing else. A month or day written without its
    leading zero ("2025-3-14") is accepted, because it is the same unambiguous
    date; a format where the day and the month could swap places ("03/04/2026")
    is not, because it means two different things to two moderators.

    Returns None for a date at or after `now` -- a backfill from the future has
    nothing to read, and accepting it would produce a run that completes
    instantly and reads as a failure to the moderator who asked for it.

    Interpreted as UTC midnight rather than in the moderator's own timezone,
    which is a real (if small) surprise and is stated in the command's reply:
    every other boundary in this project is UTC (spend ledgers, digest windows,
    fact timestamps), and a `since:` that quietly meant something different from
    the timestamps it is compared against would be worse than one that is
    consistently a few hours off from what someone expected.
    """
    try:
        parsed = datetime.strptime(raw.strip(), _SINCE_FORMAT)
    except ValueError:
        return None
    since = parsed.replace(tzinfo=timezone.utc)
    return since if since < now else None


def _since_snowflake(since: datetime) -> int | None:
    """The exclusive lower-bound snowflake for a `since:` date, or None for "everything".

    Found by this sub-phase's adversarial pass, and worth stating because the
    failure it prevents is silent rather than loud: discord.utils.time_snowflake
    is plain arithmetic against Discord's own epoch, so it happily returns a
    large NEGATIVE number for any date before 2015 ("0001-01-01" gives
    -266571789159628800000). Passed on as `after`, that is not a message id;
    Discord rejects the request, this layer retries it four times with backoff,
    and the run then sits at zero progress with nothing in the moderator's
    status reply to explain why.

    Clamped to None rather than refused, because None is exactly what such a
    date MEANS: "start from the beginning of the channel". A moderator who typed
    1970 wanted everything, and giving them everything is both what they asked
    for and what the command's own default already does.

    high=False gives the LOWEST snowflake for that instant, and `after` is
    exclusive, so the run includes everything written from that midnight onward
    apart from a message landing in the exact same millisecond -- a boundary a
    moderator picking a date has no opinion about.
    """
    snowflake = discord.utils.time_snowflake(since, high=False)
    return snowflake if snowflake > 0 else None


def _truncate(content: str, limit: int) -> str:
    """Truncate content to limit characters, appending an ellipsis if it was cut.

    The same helper, with the same reasoning, aura.commands.links and
    aura.commands.pending both carry: an embed field value has a 1,024-character
    hard cap that Discord enforces by rejecting the whole message, so a bound
    that is checked here is one that cannot turn a status reply into an error a
    moderator has no way to interpret.
    """
    if len(content) <= limit:
        return content
    return content[: limit - 1] + "\u2026"


def _permalink(guild_id: int, channel_id: int, message_id: int) -> str:
    """A Discord permalink to one message.

    The same reference CLAUDE.md's Fact component describes: channel id plus
    message id, resolvable by Discord itself, so a cursor position can be shown
    as somewhere a moderator can click rather than as a snowflake.
    """
    return f"https://discord.com/channels/{guild_id}/{channel_id}/{message_id}"


def _describe_progress(run: BackfillRun, locale: str) -> str:
    """One line saying how far a run has got, in terms a moderator can act on.

    Deliberately reports a POSITION rather than a percentage. Aura does not know
    how many messages a channel holds without reading all of them, so any
    percentage would be invented -- and an invented progress bar on a job that
    may take days is exactly the kind of confident wrongness this project keeps
    out of its answers. "Processed up to 14 March 2025" is smaller and true.
    """
    if run.cursor_message_at is None:
        return t("backfill_progress_not_started", locale)
    return t(
        "backfill_progress",
        locale,
        position=discord.utils.format_dt(run.cursor_message_at, style="D"),
        scanned=run.messages_scanned,
        staged=run.candidates_staged,
        calls=run.calls_spent,
    )


def _state_label(state: BackfillState, locale: str) -> str:
    """The localized name of a run state.

    A lookup rather than a formatted enum value, so the five states read as
    sentences a moderator understands ("paused -- resume it with
    /aura-backfill start") instead of as internal vocabulary.
    """
    return t(f"backfill_state_{state.value}", locale)


async def _handle_backfill_error(
    interaction: discord.Interaction, error: app_commands.AppCommandError
) -> None:
    """Turn the permission check's failure into a clean localized reply, log everything else.

    Attaching this via .error() stops CommandTree's default logging for this
    group (it only logs when a command has no local handler), so anything other
    than the permission failure is logged here rather than silently
    disappearing.

    Takes an unparameterized Interaction, unlike every sibling command's own
    handler: Group.error's callback type is invariant in the client parameter,
    so a handler declared over Interaction[AuraClient] does not satisfy it. That
    costs nothing here -- this function reaches for the locale, the response and
    the command name, never for the client -- and it is a real signature
    difference rather than a type-checker workaround to silence.
    """
    if isinstance(error, app_commands.MissingPermissions):
        locale = str(interaction.locale)
        message = t("backfill_permission_error", locale)
        if interaction.response.is_done():
            await interaction.followup.send(message, ephemeral=True)
        else:
            await interaction.response.send_message(message, ephemeral=True)
        return

    command_name = interaction.command.name if interaction.command else "<unknown>"
    logger.error("Unhandled error in /aura-backfill %s", command_name, exc_info=error)


backfill_group = app_commands.Group(
    name="aura-backfill",
    description="Read a channel's existing history for facts (moderators only).",
    guild_only=True,
    default_permissions=discord.Permissions(manage_guild=True),
)


@backfill_group.command(
    name="start",
    description="Start (or resume) reading a channel's existing history for facts.",
)
@app_commands.describe(
    channel="The channel to read. Automatic extraction must already be enabled for it.",
    since="Optional earliest date to read from, as YYYY-MM-DD (UTC). Default: all history.",
)
@app_commands.checks.has_permissions(manage_guild=True)
async def backfill_start(
    interaction: discord.Interaction[AuraClient],
    channel: discord.TextChannel,
    since: str | None = None,
) -> None:
    """Open a run over channel's history, or resume the paused one it already has.

    Four refusals, each answering a different question a moderator would
    otherwise have to ask afterwards:

      * The channel belongs to another guild -- unreachable through Discord's
        own option resolution, which only offers this guild's channels, and
        checked anyway because a backfill is a read of message content and
        cross-guild reads are not something to leave depending on the front
        door.
      * Extraction is not enabled for the channel. Backfilling a channel nobody
        agreed to have read would make one command carry two decisions.
      * A run is already running there. Told plainly, with its progress, rather
        than silently starting a second cursor over the same history.
      * `since` is unusable, or given alongside a paused run whose bounds are
        already fixed. Resuming with different bounds is not a thing a cursor
        can mean, so it is refused with the two commands that do work.
    """
    assert interaction.guild_id is not None  # guaranteed by the group's guild_only
    locale = str(interaction.locale)
    db = interaction.client.db
    assert db is not None  # setup_hook always finishes before commands go live

    if channel.guild.id != interaction.guild_id:
        await interaction.response.send_message(
            t("backfill_wrong_guild_error", locale), ephemeral=True
        )
        return

    if not await is_extraction_enabled(db, channel_id=channel.id):
        await interaction.response.send_message(
            t("backfill_extraction_disabled_error", locale, channel=channel.mention),
            ephemeral=True,
        )
        return

    existing = await get_active_run(db, channel_id=channel.id)
    if existing is not None and existing.state is BackfillState.RUNNING:
        await interaction.response.send_message(
            t("backfill_already_running", locale, channel=channel.mention)
            + "\n"
            + _describe_progress(existing, locale),
            ephemeral=True,
        )
        return

    if existing is not None:
        # A paused run. Resuming keeps its cursor and therefore its original
        # bounds, so a new `since` cannot be honoured and must not be silently
        # ignored either.
        if since is not None:
            await interaction.response.send_message(
                t("backfill_resume_since_conflict", locale, channel=channel.mention),
                ephemeral=True,
            )
            return
        resumed = await set_run_state(
            db,
            run_id=existing.id,
            state=BackfillState.RUNNING,
            now=utc_now(),
            from_states=(BackfillState.PAUSED,),
        )
        message_key = "backfill_resumed" if resumed else "backfill_resume_raced"
        await interaction.response.send_message(
            t(message_key, locale, channel=channel.mention)
            + "\n"
            + _describe_progress(existing, locale),
            ephemeral=True,
        )
        return

    now = utc_now()
    after_message_id: int | None = None
    if since is not None:
        parsed = _parse_since(since, now=now)
        if parsed is None:
            await interaction.response.send_message(
                t("backfill_since_invalid_error", locale), ephemeral=True
            )
            return
        after_message_id = _since_snowflake(parsed)

    # THE BOUNDARY WITH LIVE EXTRACTION, fixed here and never recomputed: every
    # message from this instant on belongs to the live path, which was already
    # watching this channel before the command was run. high=True gives the
    # HIGHEST snowflake for this instant and the bound is exclusive, so a
    # message written in this very millisecond falls to the live path rather
    # than to both.
    until_message_id = discord.utils.time_snowflake(now, high=True)

    try:
        run = await start_backfill_run(
            db,
            guild_id=interaction.guild_id,
            channel_id=channel.id,
            until_message_id=until_message_id,
            after_message_id=after_message_id,
            requested_by_id=interaction.user.id,
            now=now,
        )
    except BackfillAlreadyActiveError as exc:
        # Another moderator started a run on this channel between the read above
        # and this write. The partial unique index refused the second insert,
        # which is the whole point of it being a database constraint rather than
        # a prior read.
        await interaction.response.send_message(
            t("backfill_already_running", locale, channel=channel.mention)
            + "\n"
            + _describe_progress(exc.existing, locale),
            ephemeral=True,
        )
        return
    except ValueError:
        # start_backfill_run refuses a range with nothing in it. Reachable only
        # through a `since` at or past this instant, which _parse_since already
        # rejects -- kept because a ValueError escaping into the generic error
        # handler would tell the moderator nothing.
        await interaction.response.send_message(
            t("backfill_since_invalid_error", locale), ephemeral=True
        )
        return

    logger.info(
        "Backfill run %s started for channel %s in guild %s by user %s (since=%s)",
        run.id,
        channel.id,
        interaction.guild_id,
        interaction.user.id,
        since or "the beginning of the channel",
    )

    lines = [
        t(
            "backfill_started_all" if since is None else "backfill_started_since",
            locale,
            channel=channel.mention,
            since=since or "",
        ),
        t("backfill_started_explainer", locale, cap=interaction.client.settings.backfill_daily_cap),
    ]
    await interaction.response.send_message("\n".join(lines), ephemeral=True)


@backfill_group.command(
    name="status", description="Show what Aura's history backfills are doing in this server."
)
@app_commands.describe(channel="Only show this channel's backfill. Default: the whole server.")
@app_commands.checks.has_permissions(manage_guild=True)
async def backfill_status(
    interaction: discord.Interaction[AuraClient],
    channel: discord.TextChannel | None = None,
) -> None:
    """Show the live run (if any) and the last few finished ones.

    Always reports today's budget alongside, because "nothing is happening" and
    "the daily cap is spent and it resumes after midnight UTC" look identical
    from the outside and mean completely different things. A run that is waiting
    on its cap is the expected way a large channel spends its second day, not a
    fault, and a moderator should be able to see that without reading a
    container log.
    """
    assert interaction.guild_id is not None  # guaranteed by the group's guild_only
    locale = str(interaction.locale)
    db = interaction.client.db
    assert db is not None  # setup_hook always finishes before commands go live

    settings = interaction.client.settings
    spent = await count_backfill_calls_on(
        db, guild_id=interaction.guild_id, day=utc_day(utc_now())
    )

    if channel is not None:
        active = await get_active_run(db, channel_id=channel.id)
        runs = [active] if active is not None else []
        runs += [
            run
            for run in await get_recent_runs(
                db, guild_id=interaction.guild_id, limit=_RECENT_RUN_LIMIT
            )
            if run.channel_id == channel.id and (active is None or run.id != active.id)
        ]
    else:
        runs = await get_recent_runs(
            db, guild_id=interaction.guild_id, limit=_RECENT_RUN_LIMIT
        )

    embed = discord.Embed(title=t("backfill_status_title", locale))
    embed.description = t(
        "backfill_status_budget",
        locale,
        spent=spent,
        cap=settings.backfill_daily_cap,
    )

    if not runs:
        embed.add_field(
            name=t("backfill_status_none_label", locale),
            value=t("backfill_status_none", locale),
            inline=False,
        )
    for run in runs[:_RECENT_RUN_LIMIT]:
        # The channel goes in the field VALUE, not its name: Discord renders a
        # <#id> mention inside an embed field's value and prints it literally in
        # the name, so a name-side mention would show a raw snowflake to every
        # moderator who ran this command.
        lines = [
            f"<#{run.channel_id}> - {_state_label(run.state, locale)}",
            _describe_progress(run, locale),
        ]
        if run.cursor_message_id is not None:
            lines.append(_permalink(run.guild_id, run.channel_id, run.cursor_message_id))
        embed.add_field(
            name=t("backfill_status_run_label", locale, run_id=run.id),
            value=_truncate("\n".join(lines), _FIELD_VALUE_DISPLAY_LIMIT),
            inline=False,
        )

    await interaction.response.send_message(embed=embed, ephemeral=True)


@backfill_group.command(
    name="pause", description="Pause a channel's backfill, keeping its place."
)
@app_commands.describe(channel="The channel whose backfill should pause.")
@app_commands.checks.has_permissions(manage_guild=True)
async def backfill_pause(
    interaction: discord.Interaction[AuraClient], channel: discord.TextChannel
) -> None:
    """Stop advancing a run without losing where it got to.

    The worker may be mid-batch when this lands, and that is handled where it
    matters rather than here: the cursor advance is guarded on the run still
    being 'running', so a batch that finishes after this command does not move
    the cursor. Candidates it had already staged stay staged and reviewable --
    discarding them would be a write that throws away work the moderator asked
    for before they changed their mind about the rest -- and resuming re-reads
    that one page, which re-stages the same candidates idempotently.
    """
    await _transition(
        interaction,
        channel=channel,
        target=BackfillState.PAUSED,
        from_states=(BackfillState.RUNNING,),
        success_key="backfill_paused",
        failure_key="backfill_pause_not_running",
    )


@backfill_group.command(
    name="cancel", description="End a channel's backfill for good."
)
@app_commands.describe(channel="The channel whose backfill should end.")
@app_commands.checks.has_permissions(manage_guild=True)
async def backfill_cancel(
    interaction: discord.Interaction[AuraClient], channel: discord.TextChannel
) -> None:
    """End a run permanently. Its cursor is kept as a record, never resumed.

    Cancel does NOT touch anything the run already produced. Candidates it
    staged remain in the review queue exactly as they are, because they are
    proposals a moderator asked for and a later change of mind about reading
    MORE history says nothing about the ones already read. Discarding them is
    what /aura-pending is for, one at a time, which is the only place in this
    project where a candidate is ever thrown away.
    """
    await _transition(
        interaction,
        channel=channel,
        target=BackfillState.CANCELLED,
        from_states=(BackfillState.RUNNING, BackfillState.PAUSED),
        success_key="backfill_cancelled",
        failure_key="backfill_cancel_nothing_active",
    )


async def _transition(
    interaction: discord.Interaction[AuraClient],
    *,
    channel: discord.TextChannel,
    target: BackfillState,
    from_states: tuple[BackfillState, ...],
    success_key: str,
    failure_key: str,
) -> None:
    """Shared body for pause and cancel: find the live run, move it, report.

    Both commands are the same three steps against different states, so they are
    written once. The guarded UPDATE is what actually decides -- the read above
    it only exists to produce a specific message when there is nothing to move,
    and a run that changes state between the two produces the same "nothing to
    do" reply rather than a false success.
    """
    assert interaction.guild_id is not None  # guaranteed by the group's guild_only
    locale = str(interaction.locale)
    db = interaction.client.db
    assert db is not None  # setup_hook always finishes before commands go live

    run = await get_active_run(db, channel_id=channel.id)
    if run is None or run.guild_id != interaction.guild_id or run.state not in from_states:
        # The guild check is not failure handling: a run belongs to the guild
        # whose channel it reads, and letting one server's moderator stop
        # another server's job through a hand-crafted channel id is not
        # something to leave depending on Discord's option resolution.
        await interaction.response.send_message(
            t(failure_key, locale, channel=channel.mention), ephemeral=True
        )
        return

    moved = await set_run_state(
        db, run_id=run.id, state=target, now=utc_now(), from_states=from_states
    )
    if not moved:
        await interaction.response.send_message(
            t(failure_key, locale, channel=channel.mention), ephemeral=True
        )
        return

    logger.info(
        "Backfill run %s (channel %s) moved to %s by user %s",
        run.id,
        channel.id,
        target.value,
        interaction.user.id,
    )
    await interaction.response.send_message(
        t(success_key, locale, channel=channel.mention)
        + "\n"
        + _describe_progress(run, locale),
        ephemeral=True,
    )


backfill_group.error(_handle_backfill_error)


def register_backfill_command(tree: app_commands.CommandTree) -> None:
    """Register the /aura-backfill group onto tree."""
    tree.add_command(backfill_group)
