"""Server admins' data commands (P7a): delete one fact, delete the whole server's data, export.

- `/aura-forget fact_id` (DATA_DELETION_ENABLED): a moderator deletes one fact
  for good, after seeing it and confirming. Its replacement chain and links are
  repaired (`aura.db.deletion.forget_fact`). For a fact that names someone, or
  any fact a member asks to have removed.
- `/aura-delete-server-data` (DATA_DELETION_ENABLED): deletes everything Aura
  holds for this server, billing records excepted. Confirmed by typing the
  server's name (case and spacing do not matter); the permission is checked
  again when the name is submitted.
- `/aura-export` (DATA_EXPORT_ENABLED): every fact of the server as CSV and
  Markdown (`aura.privacy.export`), attached to a reply only the admin sees; at
  most one export per server per EXPORT_COOLDOWN_SECONDS.

All three need "Manage Server", the permission every other moderator command
uses, and act on the server they are run in only -- no option names another
server. Every executed deletion is recorded in the deletion ledger.
"""

from __future__ import annotations

import contextlib
import io
import logging
import re
import time
import unicodedata
from typing import TYPE_CHECKING, Final

import discord
from discord import app_commands

from aura.db.connection import utc_now
from aura.db.repository import get_fact_by_id, get_guild_facts, get_guild_links
from aura.i18n import t
from aura.privacy.export import ExportTooLargeError, build_export
from aura.privacy.ledger import DeletionReason
from aura.privacy.requests import execute_fact_deletion, execute_guild_purge
from aura.rendering import collapse_display_text, escape_display_markdown

if TYPE_CHECKING:
    from aura.main import AuraClient

logger = logging.getLogger(__name__)

_CONFIRMATION_TIMEOUT_SECONDS: Final = 60.0
_FACT_DISPLAY_LIMIT: Final = 1000
_NAME_INPUT_MAX_LENGTH: Final = 100

# Guild ID -> monotonic time of its last export, this process.
_last_export: dict[int, float] = {}


def normalise_server_name(name: str) -> str:
    """Return a server name in the form the typed confirmation is compared in.

    Parameters
    ----------
    name
        A server name or what an admin typed.

    Returns
    -------
    str
        NFKC-normalised, case-folded, invisible format characters (zero-width
        spaces, bidirectional marks) removed, whitespace runs collapsed,
        trimmed -- so a name an admin cannot type exactly can still be
        confirmed, and nothing invisible decides the match.
    """
    folded = unicodedata.normalize("NFKC", name).casefold()
    visible = "".join(character for character in folded if unicodedata.category(character) != "Cf")
    return re.sub(r"\s+", " ", visible).strip()


def names_match(typed: str, server_name: str) -> bool:
    """Report whether a typed confirmation names this server.

    Parameters
    ----------
    typed
        What the admin typed.
    server_name
        The server's current name.

    Returns
    -------
    bool
        True only for a non-empty match after normalisation.
    """
    expected = normalise_server_name(server_name)
    return bool(expected) and normalise_server_name(typed) == expected


async def _permission_error(
    interaction: discord.Interaction[AuraClient], error: app_commands.AppCommandError, name: str
) -> None:
    if isinstance(error, app_commands.MissingPermissions):
        message = t("data_admin_permission_error", str(interaction.locale))
        if interaction.response.is_done():
            await interaction.followup.send(message, ephemeral=True)
        else:
            await interaction.response.send_message(message, ephemeral=True)
        return
    logger.error("Unhandled error in /%s", name, exc_info=error)


class ForgetFactView(discord.ui.View):
    """Confirm or cancel deleting one fact; only the invoking moderator may press."""

    def __init__(self, *, locale: str, invoker_id: int, guild_id: int, fact_id: int) -> None:
        super().__init__(timeout=_CONFIRMATION_TIMEOUT_SECONDS)
        self._locale = locale
        self._invoker_id = invoker_id
        self._guild_id = guild_id
        self._fact_id = fact_id
        self._resolved = False
        self.message: discord.Message | None = None
        self.confirm.label = t("forget_confirm_button", locale)
        self.cancel.label = t("privacy_cancel_button", locale)

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        """Let only the moderator who ran the command press the buttons.

        Parameters
        ----------
        interaction
            The button press.

        Returns
        -------
        bool
            True only for the invoking moderator.
        """
        if interaction.user.id != self._invoker_id:
            await interaction.response.send_message(
                t("privacy_wrong_user", self._locale), ephemeral=True
            )
            return False
        return True

    @discord.ui.button(style=discord.ButtonStyle.danger)
    async def confirm(self, interaction: discord.Interaction, _button: discord.ui.Button) -> None:
        """Delete the fact, re-reading it so a race fails cleanly."""
        if self._resolved:
            if not interaction.response.is_done():
                await interaction.response.defer()
            return
        self._resolved = True
        self.stop()
        client = interaction.client
        db = getattr(client, "db", None)
        ledger = getattr(client, "deletion_ledger", None)
        assert db is not None and ledger is not None  # set up before commands go live
        fact = await get_fact_by_id(db, guild_id=self._guild_id, fact_id=self._fact_id)
        if fact is None:
            await interaction.response.edit_message(
                content=t("forget_not_found", self._locale, fact_id=self._fact_id),
                embed=None,
                view=None,
            )
            return
        await execute_fact_deletion(
            db,
            ledger,
            guild_id=self._guild_id,
            fact_id=fact.id,
            fact_created_at=fact.created_at,
            reason=DeletionReason.MODERATOR_REQUEST,
            now=utc_now(),
        )
        await interaction.response.edit_message(
            content=t("forget_done", self._locale, fact_id=self._fact_id), embed=None, view=None
        )

    @discord.ui.button(style=discord.ButtonStyle.secondary)
    async def cancel(self, interaction: discord.Interaction, _button: discord.ui.Button) -> None:
        """Back out; nothing is deleted."""
        if self._resolved:
            if not interaction.response.is_done():
                await interaction.response.defer()
            return
        self._resolved = True
        self.stop()
        await interaction.response.edit_message(
            content=t("privacy_cancelled", self._locale), embed=None, view=None
        )

    async def on_timeout(self) -> None:
        """An unanswered confirmation deletes nothing."""
        if self._resolved:
            return
        self._resolved = True
        if self.message is not None:
            with contextlib.suppress(discord.HTTPException):
                await self.message.edit(
                    content=t("privacy_expired", self._locale), embed=None, view=None
                )


@app_commands.command(
    name="aura-forget",
    description="Delete one fact for good (moderators only).",
)
@app_commands.describe(fact_id="The fact's number (#N in /aura-facts).")
@app_commands.guild_only()
@app_commands.checks.has_permissions(manage_guild=True)
async def forget_command(interaction: discord.Interaction[AuraClient], fact_id: int) -> None:
    """Show one fact of this server and ask to confirm deleting it.

    Parameters
    ----------
    interaction
        The command invocation.
    fact_id
        The fact to delete; only this server's facts are found.

    Returns
    -------
    None
    """
    assert interaction.guild_id is not None  # guaranteed by guild_only()
    locale = str(interaction.locale)
    db = interaction.client.db
    assert db is not None
    fact = await get_fact_by_id(db, guild_id=interaction.guild_id, fact_id=fact_id)
    if fact is None:
        await interaction.response.send_message(
            t("forget_not_found", locale, fact_id=fact_id), ephemeral=True
        )
        return
    content = fact.content
    if len(content) > _FACT_DISPLAY_LIMIT:
        content = content[: _FACT_DISPLAY_LIMIT - 1] + "…"
    embed = discord.Embed(
        title=t("forget_confirm_title", locale, fact_id=fact.id),
        description=escape_display_markdown(collapse_display_text(content))
        + "\n\n"
        + t("forget_confirm_text", locale),
    )
    view = ForgetFactView(
        locale=locale,
        invoker_id=interaction.user.id,
        guild_id=interaction.guild_id,
        fact_id=fact.id,
    )
    await interaction.response.send_message(embed=embed, view=view, ephemeral=True)
    view.message = await interaction.original_response()


@forget_command.error
async def _forget_error(
    interaction: discord.Interaction[AuraClient], error: app_commands.AppCommandError
) -> None:
    await _permission_error(interaction, error, "aura-forget")


class DeleteServerDataModal(discord.ui.Modal):
    """Asks for the server's name before deleting all of its data."""

    def __init__(self, *, locale: str, guild_id: int) -> None:
        super().__init__(title=t("server_delete_modal_title", locale)[:45])
        self._locale = locale
        self._guild_id = guild_id
        self.name_input: discord.ui.TextInput[discord.ui.Modal] = discord.ui.TextInput(
            label=t("server_delete_modal_label", locale)[:45],
            max_length=_NAME_INPUT_MAX_LENGTH,
            required=True,
        )
        self.add_item(self.name_input)

    async def on_submit(self, interaction: discord.Interaction) -> None:
        """Delete the server's data if the name matches and the permission still holds.

        Parameters
        ----------
        interaction
            The modal submission.

        Returns
        -------
        None
        """
        guild = interaction.guild
        if guild is None or guild.id != self._guild_id or not interaction.permissions.manage_guild:
            await interaction.response.send_message(
                t("data_admin_permission_error", self._locale), ephemeral=True
            )
            return
        if not names_match(self.name_input.value or "", guild.name):
            await interaction.response.send_message(
                t("server_delete_name_mismatch", self._locale), ephemeral=True
            )
            return
        client = interaction.client
        db = getattr(client, "db", None)
        ledger = getattr(client, "deletion_ledger", None)
        assert db is not None and ledger is not None  # set up before commands go live
        await interaction.response.defer(ephemeral=True, thinking=True)
        counts = await execute_guild_purge(
            db, ledger, guild_id=guild.id, reason=DeletionReason.ADMIN_REQUEST, now=utc_now()
        )
        await interaction.followup.send(
            t("server_delete_done", self._locale, count=counts.total), ephemeral=True
        )

    async def on_error(  # type: ignore[override]
        self, _interaction: discord.Interaction, error: Exception
    ) -> None:
        """Log an unexpected failure; the purge is one transaction, so nothing is half-done."""
        logger.error("Unhandled error deleting a server's data", exc_info=error)


@app_commands.command(
    name="aura-delete-server-data",
    description="Delete everything Aura stores for this server (server managers only).",
)
@app_commands.guild_only()
@app_commands.checks.has_permissions(manage_guild=True)
async def delete_server_data_command(interaction: discord.Interaction[AuraClient]) -> None:
    """Open the name confirmation for deleting this server's data.

    Parameters
    ----------
    interaction
        The command invocation.

    Returns
    -------
    None
    """
    assert interaction.guild_id is not None  # guaranteed by guild_only()
    await interaction.response.send_modal(
        DeleteServerDataModal(locale=str(interaction.locale), guild_id=interaction.guild_id)
    )


@delete_server_data_command.error
async def _delete_server_error(
    interaction: discord.Interaction[AuraClient], error: app_commands.AppCommandError
) -> None:
    await _permission_error(interaction, error, "aura-delete-server-data")


@app_commands.command(
    name="aura-export",
    description="Export this server's facts as CSV and Markdown (server managers only).",
)
@app_commands.guild_only()
@app_commands.checks.has_permissions(manage_guild=True)
async def export_command(interaction: discord.Interaction[AuraClient]) -> None:
    """Send this server's facts as two files, visible only to the admin who asked.

    Parameters
    ----------
    interaction
        The command invocation.

    Returns
    -------
    None

    Notes
    -----
    The cooldown is claimed before the first await, so two exports started
    together cannot both pass it.
    """
    assert interaction.guild_id is not None  # guaranteed by guild_only()
    locale = str(interaction.locale)
    guild_id = interaction.guild_id
    cooldown = interaction.client.settings.export_cooldown_seconds
    now_monotonic = time.monotonic()
    last = _last_export.get(guild_id)
    if last is not None and now_monotonic - last < cooldown:
        minutes = max(1, int((cooldown - (now_monotonic - last) + 59) // 60))
        await interaction.response.send_message(
            t("export_too_soon", locale, minutes=minutes), ephemeral=True
        )
        return
    _last_export[guild_id] = now_monotonic
    await interaction.response.defer(ephemeral=True, thinking=True)
    db = interaction.client.db
    assert db is not None
    facts = await get_guild_facts(db, guild_id)
    if not facts:
        await interaction.followup.send(t("export_empty", locale), ephemeral=True)
        return
    try:
        files = build_export(
            facts, await get_guild_links(db, guild_id), locale=locale, exported_at=utc_now()
        )
    except ExportTooLargeError:
        await interaction.followup.send(t("export_too_large", locale), ephemeral=True)
        return
    await interaction.followup.send(
        t("export_ready", locale, count=len(facts)),
        files=[discord.File(io.BytesIO(file.data), filename=file.filename) for file in files],
        ephemeral=True,
    )


@export_command.error
async def _export_error(
    interaction: discord.Interaction[AuraClient], error: app_commands.AppCommandError
) -> None:
    await _permission_error(interaction, error, "aura-export")


def register_data_admin_commands(
    tree: app_commands.CommandTree, *, deletion: bool, export: bool
) -> None:
    """Register the switched-on data commands onto tree.

    Parameters
    ----------
    tree
        The command tree to register into.
    deletion
        DATA_DELETION_ENABLED: /aura-forget and /aura-delete-server-data.
    export
        DATA_EXPORT_ENABLED: /aura-export.

    Returns
    -------
    None
    """
    if deletion:
        tree.add_command(forget_command)
        tree.add_command(delete_server_data_command)
    if export:
        tree.add_command(export_command)
