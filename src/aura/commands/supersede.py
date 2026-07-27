"""/aura-supersede: a moderator's command to retire an old fact in favor of a new one.

Phase 1b built supersede_fact -- atomic, race-condition-safe -- back when
Trigger 2 posted rarely enough that the gap barely mattered. Phase 1c
deliberately left the mod-facing command out, so nothing has ever actually
been marked superseded in production. That gap matters more now: Phase 2b-3
made Aura visibly active, and its synthesis prompt actively detects
contradictions between an old and a replacing fact and refuses to answer
rather than risk citing something outdated. Every one of those refusals is a
case a mod could have prevented outright by retiring the old fact when the
new one was added. This is purely the command surface on top of the existing
data layer -- see aura.db.repository.supersede_fact_with_existing_successor's
docstring for why the data layer needed one new, minimal function rather than
reusing supersede_fact verbatim.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import aiosqlite
import discord
from discord import app_commands

from aura.db.models import Fact, FactStatus
from aura.db.repository import (
    FactAlreadySupersededError,
    SuccessorNotActiveError,
    get_fact_by_id,
    supersede_fact_with_existing_successor,
)
from aura.i18n import t

if TYPE_CHECKING:
    from aura.main import AuraClient

logger = logging.getLogger(__name__)

# Long enough for a moderator to read both facts and decide, short enough
# that a forgotten confirmation prompt doesn't sit around indefinitely.
_CONFIRMATION_TIMEOUT_SECONDS = 60.0

# Comfortably under an embed field value's 1024-character hard cap, matching
# the same display budget aura.commands.facts uses for fact content.
_FIELD_VALUE_DISPLAY_LIMIT = 200


def _truncate(content: str, limit: int) -> str:
    """Truncate content to limit characters, appending an ellipsis if it was cut."""
    if len(content) <= limit:
        return content
    return content[: limit - 1] + "…"


class SupersedeConfirmView(discord.ui.View):
    """The explicit confirm/cancel step before a supersession is committed.

    interaction_check restricts both buttons to the moderator who invoked the
    command. Discord ephemeral replies are already only visible to that
    person, but visibility isn't the property being enforced here -- without
    this check, anyone who somehow obtained a reference to the interaction's
    message (a compromised token, a client bug) could confirm on the
    moderator's behalf. Belt and suspenders around a change that can't be
    undone by this command.
    """

    def __init__(
        self,
        *,
        db: aiosqlite.Connection,
        guild_id: int,
        old_fact: Fact,
        new_fact: Fact,
        locale: str,
        invoker_id: int,
    ) -> None:
        super().__init__(timeout=_CONFIRMATION_TIMEOUT_SECONDS)
        self._db = db
        self._guild_id = guild_id
        self._old_fact = old_fact
        self._new_fact = new_fact
        self._locale = locale
        self._invoker_id = invoker_id
        # Set post-super().__init__(): only once _init_children() has run does
        # self.confirm/self.cancel refer to the real Button instances rather
        # than the plain decorated functions, so labels can be localized here.
        self.confirm.label = t("supersede_confirm_button", locale)
        self.cancel.label = t("supersede_cancel_button", locale)
        # Populated by the caller right after the confirmation message is
        # sent, so on_timeout has something to edit -- send_message's return
        # value isn't the message itself, only interaction.original_response() is.
        self.message: discord.Message | None = None
        # Guards against a genuine double-fire race: a moderator's client can
        # send two component interactions (e.g. Confirm then Cancel, or the
        # same button twice) before this message's first edit_message(view=None)
        # has actually propagated back to their client. Checked and set
        # synchronously, with no `await` in between, so two concurrently
        # dispatched callbacks on this same view instance cannot both pass
        # the check -- asyncio only switches tasks at an await point.
        self._resolved = False

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        """Reject a button press from anyone but the moderator who ran the command."""
        if interaction.user.id != self._invoker_id:
            await interaction.response.send_message(
                t("supersede_wrong_user_error", self._locale), ephemeral=True
            )
            return False
        return True

    @discord.ui.button(style=discord.ButtonStyle.danger)
    async def confirm(self, interaction: discord.Interaction, _button: discord.ui.Button) -> None:
        """Commit the supersession, re-validating atomically so a race fails cleanly, not silently."""
        if self._resolved:
            # A second, near-simultaneous click on this same view (see
            # _resolved's docstring) -- the first click already decided this
            # view's outcome, so this one must not act again, but Discord
            # still requires *some* response within its own timeout.
            if not interaction.response.is_done():
                await interaction.response.defer()
            return
        self._resolved = True

        try:
            await supersede_fact_with_existing_successor(
                self._db,
                old_fact_id=self._old_fact.id,
                new_fact_id=self._new_fact.id,
                guild_id=self._guild_id,
            )
        except FactAlreadySupersededError:
            # Someone else superseded old_fact_id (or it stopped existing)
            # between showing this confirmation and this button press.
            await interaction.response.edit_message(
                content=t("supersede_old_race_error", self._locale, fact_id=self._old_fact.id),
                embed=None,
                view=None,
            )
            self.stop()
            return
        except SuccessorNotActiveError:
            # new_fact_id stopped being active in that same window -- e.g. it
            # was itself superseded by a third fact in the meantime.
            await interaction.response.edit_message(
                content=t("supersede_new_race_error", self._locale, fact_id=self._new_fact.id),
                embed=None,
                view=None,
            )
            self.stop()
            return

        await interaction.response.edit_message(
            content=t(
                "supersede_success",
                self._locale,
                old_fact_id=self._old_fact.id,
                new_fact_id=self._new_fact.id,
            ),
            embed=None,
            view=None,
        )
        self.stop()

    @discord.ui.button(style=discord.ButtonStyle.secondary)
    async def cancel(self, interaction: discord.Interaction, _button: discord.ui.Button) -> None:
        """Back out of the confirmation: nothing is ever written to the database."""
        if self._resolved:
            if not interaction.response.is_done():
                await interaction.response.defer()
            return
        self._resolved = True

        await interaction.response.edit_message(
            content=t("supersede_cancelled", self._locale), embed=None, view=None
        )
        self.stop()

    async def on_timeout(self) -> None:
        """A moderator who never answers leaves nothing committed either."""
        if self._resolved:
            # self.stop() (called by confirm/cancel) cancels discord.py's own
            # timeout task, so this shouldn't normally fire after either
            # button already ran -- checked anyway so a scheduling edge case
            # can never double-edit a message a button callback already
            # finalized.
            return
        self._resolved = True
        if self.message is not None:
            await self.message.edit(content=t("supersede_expired", self._locale), embed=None, view=None)


async def _handle_supersede_command_error(
    interaction: discord.Interaction[AuraClient], error: app_commands.AppCommandError
) -> None:
    """Shared error handler for /aura-supersede, matching every other mod-gated command's pattern.

    Attaching this via .error() stops CommandTree's default logging for this
    command (it only logs when a command has no local handler), so anything
    other than the permission-check failure is logged here instead of
    silently disappearing.
    """
    if isinstance(error, app_commands.MissingPermissions):
        locale = str(interaction.locale)
        message = t("supersede_permission_error", locale)
        if interaction.response.is_done():
            await interaction.followup.send(message, ephemeral=True)
        else:
            await interaction.response.send_message(message, ephemeral=True)
        return

    logger.error("Unhandled error in /aura-supersede", exc_info=error)


@app_commands.command(
    name="aura-supersede",
    description="Mark an existing fact as superseded by another (moderators only).",
)
@app_commands.describe(
    old_fact_id="The ID (the #N shown by /aura-facts) of the fact that's no longer true.",
    new_fact_id="The ID (the #N shown by /aura-facts) of the fact that replaces it.",
)
@app_commands.guild_only()
@app_commands.checks.has_permissions(manage_guild=True)
async def supersede_command(
    interaction: discord.Interaction[AuraClient], old_fact_id: int, new_fact_id: int
) -> None:
    """Validate both fact references, then ask for confirmation before superseding old_fact_id.

    Every rejection path here is a friendly, pre-flight check for a common
    mistake, distinct from the atomic re-validation supersede_fact_with_existing_successor
    performs at commit time -- both exist because time passes between this
    validation and a moderator actually pressing Confirm.
    """
    assert interaction.guild_id is not None  # guaranteed by guild_only()
    locale = str(interaction.locale)
    guild_id = interaction.guild_id

    db = interaction.client.db
    assert db is not None  # setup_hook always finishes before commands go live

    if old_fact_id == new_fact_id:
        await interaction.response.send_message(t("supersede_self_error", locale), ephemeral=True)
        return

    old_fact = await get_fact_by_id(db, guild_id=guild_id, fact_id=old_fact_id)
    if old_fact is None:
        await interaction.response.send_message(
            t("supersede_fact_not_found", locale, fact_id=old_fact_id), ephemeral=True
        )
        return

    new_fact = await get_fact_by_id(db, guild_id=guild_id, fact_id=new_fact_id)
    if new_fact is None:
        await interaction.response.send_message(
            t("supersede_fact_not_found", locale, fact_id=new_fact_id), ephemeral=True
        )
        return

    if old_fact.status != FactStatus.ACTIVE:
        await interaction.response.send_message(
            t("supersede_old_already_superseded", locale, fact_id=old_fact_id), ephemeral=True
        )
        return

    if new_fact.status != FactStatus.ACTIVE:
        await interaction.response.send_message(
            t("supersede_new_not_active", locale, fact_id=new_fact_id), ephemeral=True
        )
        return

    embed = discord.Embed(title=t("supersede_confirm_title", locale))
    embed.add_field(
        name=t("supersede_confirm_old_label", locale, fact_id=old_fact.id),
        value=_truncate(old_fact.content, _FIELD_VALUE_DISPLAY_LIMIT),
        inline=False,
    )
    embed.add_field(
        name=t("supersede_confirm_new_label", locale, fact_id=new_fact.id),
        value=_truncate(new_fact.content, _FIELD_VALUE_DISPLAY_LIMIT),
        inline=False,
    )

    view = SupersedeConfirmView(
        db=db,
        guild_id=guild_id,
        old_fact=old_fact,
        new_fact=new_fact,
        locale=locale,
        invoker_id=interaction.user.id,
    )
    await interaction.response.send_message(embed=embed, view=view, ephemeral=True)
    view.message = await interaction.original_response()


supersede_command.error(_handle_supersede_command_error)


def register_supersede_command(tree: app_commands.CommandTree) -> None:
    """Register /aura-supersede onto tree."""
    tree.add_command(supersede_command)
