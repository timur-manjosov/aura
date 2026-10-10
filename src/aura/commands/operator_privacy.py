"""/aura-operator-privacy: the operator's side of P7a's deletion paths.

For requests that arrive at PRIVACY_CONTACT -- from a member who left a server
and cannot use /aura-privacy any more, from a server's owner, or from Discord
itself -- the operator runs the same deletion rules, recorded in the same
ledger:

- `forget-member user_id [server_id] [mode]`: a member's data, in one server or
  all (the default), facts deleted (the default) or unlinked.
- `forget-server server_id`: everything of one server, billing records
  excepted -- also for a server Aura is no longer in.
- `lookup-authors`: the one-time lookup of authors for facts stored before P7a.
- `status`: counts only -- ledger entries, servers marked as left, sources
  still without an author.

Gated on OPERATOR_DISCORD_USER_ID like the other operator commands, English
like them (see aura.commands.operator), every reply ephemeral. IDs are taken as
text because Discord's integer options cannot carry a 64-bit snowflake.
Registered only when DATA_DELETION_ENABLED is on.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Final, cast

import discord
from discord import app_commands

from aura.commands.operator import _handle_operator_budget_error, _is_operator
from aura.config import MAX_SQLITE_INTEGER
from aura.db.connection import connection_lock, utc_iso, utc_now
from aura.db.deletion import MemberDeletionMode
from aura.db.guild_departures import get_departures
from aura.privacy.author_lookup import lookup_missing_authors
from aura.privacy.gateway import ClientMessageAuthorSource
from aura.privacy.ledger import DeletionReason
from aura.privacy.requests import execute_guild_purge, execute_member_deletion

if TYPE_CHECKING:
    from aura.main import AuraClient

logger = logging.getLogger(__name__)

_MODE_CHOICES: Final = [
    app_commands.Choice(name="delete the facts from their messages", value="delete_facts"),
    app_commands.Choice(name="keep the facts, remove the link", value="unlink"),
]


def parse_discord_id(raw: str | None) -> int | None:
    """Return a Discord ID typed as text, or None when it is not one.

    Parameters
    ----------
    raw
        The option's text.

    Returns
    -------
    int or None
        A positive integer that fits SQLite's INTEGER; None for anything else
        (signs, spaces inside, letters, zero, too large).
    """
    if raw is None:
        return None
    stripped = raw.strip()
    if not stripped.isascii() or not stripped.isdigit():
        return None
    value = int(stripped)
    return value if 0 < value <= MAX_SQLITE_INTEGER else None


privacy_group = app_commands.Group(
    name="aura-operator-privacy",
    description="Operator-only: deletion requests that arrived by contact, and their status.",
    guild_only=True,
)


@privacy_group.command(name="forget-member", description="Delete a member's data.")
@app_commands.describe(
    user_id="The member's Discord user ID.",
    server_id="Only this server (default: every server).",
    mode="What happens to facts taken from their messages (default: delete them).",
)
@app_commands.choices(mode=_MODE_CHOICES)
@app_commands.check(_is_operator)
async def forget_member_command(
    interaction: discord.Interaction[AuraClient],
    user_id: str,
    server_id: str | None = None,
    mode: app_commands.Choice[str] | None = None,
) -> None:
    """Run a member's deletion for a request that came by contact.

    Parameters
    ----------
    interaction
        The operator's invocation.
    user_id
        The member.
    server_id
        One server, or None for all.
    mode
        delete_facts (default) or unlink.

    Returns
    -------
    None
    """
    member = parse_discord_id(user_id)
    guild_id = parse_discord_id(server_id) if server_id else None
    if member is None or (server_id and guild_id is None):
        await interaction.response.send_message(
            "Not a Discord ID; nothing was deleted.", ephemeral=True
        )
        return
    chosen = MemberDeletionMode(mode.value) if mode is not None else MemberDeletionMode.DELETE_FACTS
    db, ledger = interaction.client.db, interaction.client.deletion_ledger
    assert db is not None and ledger is not None
    await interaction.response.defer(ephemeral=True, thinking=True)
    counts = await execute_member_deletion(
        db,
        ledger,
        user_id=member,
        guild_id=guild_id,
        mode=chosen,
        reason=DeletionReason.OPERATOR_REQUEST,
        now=utc_now(),
    )
    await interaction.followup.send(
        f"Done ({'one server' if guild_id else 'all servers'}, {chosen.value}): {counts.summary()}",
        ephemeral=True,
    )


@privacy_group.command(name="forget-server", description="Delete a server's data (not billing).")
@app_commands.describe(server_id="The server's Discord ID.")
@app_commands.check(_is_operator)
async def forget_server_command(
    interaction: discord.Interaction[AuraClient], server_id: str
) -> None:
    """Purge one server's data for a request that came by contact.

    Parameters
    ----------
    interaction
        The operator's invocation.
    server_id
        The server.

    Returns
    -------
    None
    """
    guild_id = parse_discord_id(server_id)
    if guild_id is None:
        await interaction.response.send_message(
            "Not a Discord ID; nothing was deleted.", ephemeral=True
        )
        return
    db, ledger = interaction.client.db, interaction.client.deletion_ledger
    assert db is not None and ledger is not None
    await interaction.response.defer(ephemeral=True, thinking=True)
    counts = await execute_guild_purge(
        db, ledger, guild_id=guild_id, reason=DeletionReason.OPERATOR_REQUEST, now=utc_now()
    )
    await interaction.followup.send(f"Done: {counts.summary()}", ephemeral=True)


@privacy_group.command(
    name="lookup-authors", description="Look up authors of facts stored before P7a."
)
@app_commands.check(_is_operator)
async def lookup_authors_command(interaction: discord.Interaction[AuraClient]) -> None:
    """Run one batch of the author lookup and report its counts.

    Parameters
    ----------
    interaction
        The operator's invocation.

    Returns
    -------
    None
    """
    db = interaction.client.db
    assert db is not None
    await interaction.response.defer(ephemeral=True, thinking=True)
    result = await lookup_missing_authors(db, ClientMessageAuthorSource(interaction.client))
    await interaction.followup.send(
        f"Author lookup: {result.resolved} found, {result.unknown} unknown (message gone or "
        f"unreadable), {result.failed} failed (retry later), {result.remaining} still missing"
        + (
            " -- stopped by Discord's rate limit, run it again in a minute."
            if result.rate_limited
            else "."
        ),
        ephemeral=True,
    )


@privacy_group.command(name="status", description="Counts only: ledger, departures, authors.")
@app_commands.check(_is_operator)
async def status_command(interaction: discord.Interaction[AuraClient]) -> None:
    """Show the data obligations' state as counts.

    Parameters
    ----------
    interaction
        The operator's invocation.

    Returns
    -------
    None
    """
    db, ledger = interaction.client.db, interaction.client.deletion_ledger
    assert db is not None and ledger is not None
    settings = interaction.client.settings
    departures = await get_departures(db)
    now = utc_iso(utc_now())
    due = sum(1 for departure in departures if departure.purge_after <= now)
    async with (
        connection_lock(db),
        db.execute(
            "SELECT (SELECT COUNT(*) FROM facts WHERE source_author_id IS NULL AND message_id > 0)"
            " + (SELECT COUNT(*) FROM pending_facts WHERE source_author_id IS NULL"
            " AND message_id > 0)"
        ) as cursor,
    ):
        row = await cursor.fetchone()
    missing = int(row[0]) if row else 0
    await interaction.response.send_message(
        f"Deletion ledger: {await ledger.count()} entr(ies). Servers marked as left: "
        f"{len(departures)} ({due} past their {settings.guild_purge_grace_days}-day period). "
        f"Purge mode: {settings.data_purge_mode.value}. Facts/candidates without a looked-up "
        f"author: {missing}.",
        ephemeral=True,
    )


@privacy_group.error
async def _privacy_group_error(
    interaction: discord.Interaction, error: app_commands.AppCommandError
) -> None:
    if isinstance(error, app_commands.CheckFailure):
        await _handle_operator_budget_error(
            cast("discord.Interaction[AuraClient]", interaction), error
        )
        return
    logger.error("Unhandled error in /aura-operator-privacy", exc_info=error)


def register_operator_privacy_commands(tree: app_commands.CommandTree) -> None:
    """Register /aura-operator-privacy onto tree.

    Parameters
    ----------
    tree
        The command tree to register into.

    Returns
    -------
    None
    """
    tree.add_command(privacy_group)
