"""Command confirmations in the configured look (P5).

A confirmation -- a setting saved, a fact created, a candidate confirmed or
discarded -- is either the classic plain line (NOTICE_LOOK=classic, the
default: exactly the call every command made before P5) or the notice card of
aura.cards (NOTICE_LOOK=card): the confirmation symbol and the same text, in
ANSWER_CARD_STYLE, mentions disabled.
"""

from __future__ import annotations

from typing import Any

import discord

from aura.card_delivery import edit_with_card, reply_with_card
from aura.cards import build_notice
from aura.config import CardStyle, MessageLook
from aura.theme import MessageKind


def notice_cards_on(interaction: discord.Interaction[Any]) -> bool:
    """Report whether command notices are drawn as cards (NOTICE_LOOK=card).

    Parameters
    ----------
    interaction
        The command invocation; its client carries the settings.

    Returns
    -------
    bool
        False when the client has no settings (a test double), so the classic
        reply is the fallback.
    """
    settings = getattr(interaction.client, "settings", None)
    return getattr(settings, "notice_look", MessageLook.CLASSIC) is MessageLook.CARD


def _card_style(interaction: discord.Interaction[Any]) -> CardStyle:
    """Return the client's ANSWER_CARD_STYLE, or the embed when it has no settings."""
    settings = getattr(interaction.client, "settings", None)
    style = getattr(settings, "answer_card_style", CardStyle.EMBED)
    return style if isinstance(style, CardStyle) else CardStyle.EMBED


async def send_confirmation(interaction: discord.Interaction[Any], text: str) -> None:
    """Send a command's confirmation as the first response.

    Parameters
    ----------
    interaction
        The command invocation; not yet responded to.
    text
        The localized confirmation, exactly what the classic look sends.

    Returns
    -------
    None
    """
    if notice_cards_on(interaction):
        await reply_with_card(
            interaction,
            build_notice(MessageKind.CONFIRM, text),
            style=_card_style(interaction),
        )
        return
    await interaction.response.send_message(text, ephemeral=True)


async def edit_into_confirmation(interaction: discord.Interaction[Any], text: str) -> None:
    """Replace a component's message with a confirmation (the /aura-pending buttons).

    Parameters
    ----------
    interaction
        The button press.
    text
        The localized confirmation, exactly what the classic look shows.

    Returns
    -------
    None
    """
    if notice_cards_on(interaction):
        await edit_with_card(interaction, build_notice(MessageKind.CONFIRM, text))
        return
    await interaction.response.edit_message(content=text, embed=None, view=None)
