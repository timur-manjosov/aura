"""/aura-privacy: what Aura stores and how to get it deleted, for every member (P7a, R7 + R2).

Visible only to the member who runs it. It says, in a few lines, what Aura
keeps, where text goes to be processed, how long a server's data stays after
Aura leaves, and how to have one's own data deleted; it links the full privacy
policy (PRIVACY_POLICY_URL) and names the contact (PRIVACY_CONTACT). The texts
are placeholders in the locale files, so the final legal wording replaces them
without a code change.

With DATA_DELETION_ENABLED, it carries the member's own deletion path: a button,
then a confirmation where the member picks the scope (all servers, the
default, or this one) and what happens to facts taken from their messages
(deleted, the default, or kept without the link to them). Only the member who
opened it can press anything, and only their own Discord ID is ever used --
there is no way to name someone else.

Registered only when PRIVACY_INFO_ENABLED is on.
"""

from __future__ import annotations

import contextlib
import logging
import time
from typing import TYPE_CHECKING, Final

import discord
from discord import app_commands

from aura.db.connection import utc_now
from aura.db.deletion import DeletionCounts, MemberDeletionMode
from aura.i18n import t
from aura.privacy.ledger import DeletionReason
from aura.privacy.requests import execute_member_deletion

if TYPE_CHECKING:
    from aura.config import Settings
    from aura.main import AuraClient

logger = logging.getLogger(__name__)

_CONFIRMATION_TIMEOUT_SECONDS: Final = 120.0

# One executed request per member per this many seconds: a request is
# idempotent, but every one is a ledger entry re-applied at each start.
REQUEST_COOLDOWN_SECONDS: Final = 300.0

_SCOPE_ALL: Final = "all"
_SCOPE_THIS: Final = "this"

# Member ID -> monotonic time of their last executed request, this process.
_last_request: dict[int, float] = {}


def privacy_text(settings: Settings, locale: str) -> str:
    """Return the member-facing privacy summary.

    Parameters
    ----------
    settings
        Loaded configuration (contact, period, whether deletion is on).
    locale
        The reader's locale.

    Returns
    -------
    str
        The summary's lines, joined.
    """
    contact = settings.privacy_contact or ""
    lines = [
        t("privacy_what", locale),
        t("privacy_ai", locale),
        t("privacy_retention", locale, days=settings.guild_purge_grace_days),
        t(
            "privacy_delete_hint" if settings.data_deletion_enabled else "privacy_contact_hint",
            locale,
            contact=contact,
        ),
    ]
    return "\n\n".join(lines)


def result_text(counts: DeletionCounts, mode: MemberDeletionMode, locale: str) -> str:
    """Return the localized result of a member's deletion, in counts only.

    Parameters
    ----------
    counts
        What was removed.
    mode
        The mode the member chose.
    locale
        The member's locale.

    Returns
    -------
    str
        One sentence.
    """
    if mode is MemberDeletionMode.DELETE_FACTS:
        facts = counts.rows.get("facts", 0)
        return t("privacy_delete_done", locale, facts=facts, other=counts.total - facts)
    unlinked = counts.rows.get("facts.unlinked", 0)
    return t("privacy_unlink_done", locale, facts=unlinked, other=counts.total - unlinked)


def _too_soon(user_id: int, now: float) -> bool:
    last = _last_request.get(user_id)
    return last is not None and now - last < REQUEST_COOLDOWN_SECONDS


class DeletionConfirmView(discord.ui.View):
    """The member's choice of scope and mode, and the confirm/cancel step."""

    def __init__(self, *, locale: str, invoker_id: int, guild_id: int, contact: str) -> None:
        super().__init__(timeout=_CONFIRMATION_TIMEOUT_SECONDS)
        self._locale = locale
        self._invoker_id = invoker_id
        self._guild_id = guild_id
        self._contact = contact
        self.scope = _SCOPE_ALL
        self.mode = MemberDeletionMode.DELETE_FACTS
        self._resolved = False
        self.message: discord.Message | None = None
        self._scope_select: discord.ui.Select[DeletionConfirmView] = discord.ui.Select(
            options=[
                discord.SelectOption(
                    label=t("privacy_scope_all", locale), value=_SCOPE_ALL, default=True
                ),
                discord.SelectOption(label=t("privacy_scope_this", locale), value=_SCOPE_THIS),
            ],
            row=0,
        )
        self._scope_select.callback = self._choose_scope  # type: ignore[method-assign]
        self.add_item(self._scope_select)
        self._mode_select: discord.ui.Select[DeletionConfirmView] = discord.ui.Select(
            options=[
                discord.SelectOption(
                    label=t("privacy_mode_delete", locale),
                    value=MemberDeletionMode.DELETE_FACTS.value,
                    default=True,
                ),
                discord.SelectOption(
                    label=t("privacy_mode_unlink", locale), value=MemberDeletionMode.UNLINK.value
                ),
            ],
            row=1,
        )
        self._mode_select.callback = self._choose_mode  # type: ignore[method-assign]
        self.add_item(self._mode_select)
        self.confirm.label = t("privacy_confirm_button", locale)
        self.cancel.label = t("privacy_cancel_button", locale)

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        """Let only the member who opened it use it.

        Parameters
        ----------
        interaction
            The component interaction.

        Returns
        -------
        bool
            True only for the invoking member.
        """
        if interaction.user.id != self._invoker_id:
            await interaction.response.send_message(
                t("privacy_wrong_user", self._locale), ephemeral=True
            )
            return False
        return True

    async def _choose_scope(self, interaction: discord.Interaction) -> None:
        """Remember the chosen scope."""
        values = self._scope_select.values
        self.scope = _SCOPE_THIS if values and values[0] == _SCOPE_THIS else _SCOPE_ALL
        await interaction.response.defer()

    async def _choose_mode(self, interaction: discord.Interaction) -> None:
        """Remember the chosen mode."""
        values = self._mode_select.values
        self.mode = (
            MemberDeletionMode.UNLINK
            if values and values[0] == MemberDeletionMode.UNLINK.value
            else MemberDeletionMode.DELETE_FACTS
        )
        await interaction.response.defer()

    @discord.ui.button(style=discord.ButtonStyle.danger, row=2)
    async def confirm(self, interaction: discord.Interaction, _button: discord.ui.Button) -> None:
        """Execute the deletion for the invoking member only."""
        if self._resolved:
            if not interaction.response.is_done():
                await interaction.response.defer()
            return
        self._resolved = True
        self.stop()
        now_monotonic = time.monotonic()
        if _too_soon(self._invoker_id, now_monotonic):
            await interaction.response.edit_message(
                content=t("privacy_too_soon", self._locale), embed=None, view=None
            )
            return
        _last_request[self._invoker_id] = now_monotonic
        await interaction.response.defer()
        client = interaction.client
        db = getattr(client, "db", None)
        ledger = getattr(client, "deletion_ledger", None)
        try:
            if db is None or ledger is None:
                raise RuntimeError("database or ledger not ready")
            counts = await execute_member_deletion(
                db,
                ledger,
                user_id=self._invoker_id,
                guild_id=self._guild_id if self.scope == _SCOPE_THIS else None,
                mode=self.mode,
                reason=DeletionReason.MEMBER_REQUEST,
                now=utc_now(),
            )
        except Exception:
            logger.exception("A member's deletion request failed; nothing was committed")
            _last_request.pop(self._invoker_id, None)
            await interaction.edit_original_response(
                content=t("privacy_delete_failed", self._locale, contact=self._contact),
                embed=None,
                view=None,
            )
            return
        await interaction.edit_original_response(
            content=result_text(counts, self.mode, self._locale), embed=None, view=None
        )

    @discord.ui.button(style=discord.ButtonStyle.secondary, row=2)
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


class PrivacyView(discord.ui.View):
    """The privacy summary's buttons: the policy link and, when enabled, "delete my data"."""

    def __init__(self, *, settings: Settings, locale: str, invoker_id: int, guild_id: int) -> None:
        super().__init__(timeout=_CONFIRMATION_TIMEOUT_SECONDS)
        self._settings = settings
        self._locale = locale
        self._invoker_id = invoker_id
        self._guild_id = guild_id
        if settings.privacy_policy_url is not None:
            self.add_item(
                discord.ui.Button(
                    style=discord.ButtonStyle.link,
                    label=t("privacy_policy_button", locale),
                    url=settings.privacy_policy_url,
                )
            )
        if settings.data_deletion_enabled:
            delete: discord.ui.Button[PrivacyView] = discord.ui.Button(
                style=discord.ButtonStyle.danger, label=t("privacy_delete_button", locale)
            )
            delete.callback = self._open_confirmation  # type: ignore[method-assign]
            self.add_item(delete)

    async def _open_confirmation(self, interaction: discord.Interaction) -> None:
        if interaction.user.id != self._invoker_id:
            await interaction.response.send_message(
                t("privacy_wrong_user", self._locale), ephemeral=True
            )
            return
        view = DeletionConfirmView(
            locale=self._locale,
            invoker_id=self._invoker_id,
            guild_id=self._guild_id,
            contact=self._settings.privacy_contact or "",
        )
        await interaction.response.send_message(
            t("privacy_delete_confirm_text", self._locale), view=view, ephemeral=True
        )
        view.message = await interaction.original_response()


@app_commands.command(
    name="aura-privacy",
    description="What Aura stores about you, and how to have it deleted.",
)
@app_commands.guild_only()
async def privacy_command(interaction: discord.Interaction[AuraClient]) -> None:
    """Show the privacy summary, visible only to the member who asked.

    Parameters
    ----------
    interaction
        The command invocation.

    Returns
    -------
    None
    """
    assert interaction.guild_id is not None  # guaranteed by guild_only()
    locale = str(interaction.locale)
    settings = interaction.client.settings
    embed = discord.Embed(
        title=t("privacy_title", locale), description=privacy_text(settings, locale)
    )
    view = PrivacyView(
        settings=settings,
        locale=locale,
        invoker_id=interaction.user.id,
        guild_id=interaction.guild_id,
    )
    await interaction.response.send_message(embed=embed, view=view, ephemeral=True)


async def _handle_privacy_error(
    interaction: discord.Interaction[AuraClient], error: app_commands.AppCommandError
) -> None:
    """Log an unexpected failure; there is no permission to fail."""
    logger.error("Unhandled error in /aura-privacy", exc_info=error)


privacy_command.error(_handle_privacy_error)


def register_privacy_command(tree: app_commands.CommandTree) -> None:
    """Register /aura-privacy onto tree.

    Parameters
    ----------
    tree
        The command tree to register into.

    Returns
    -------
    None
    """
    tree.add_command(privacy_command)
