"""Sending a card (aura.answer_card / aura.cards) to Discord, in the configured style (P5).

Three ways a card reaches a reader, each in ANSWER_CARD_STYLE: posted in a
channel (the digest, onboarding), as a reply to a command, and as the edit of a
command's earlier message (the /aura-pending buttons). Every send disables
mentions: a card can carry a fact's or a server's text, and "@everyone" in one
must never notify anyone, whichever style draws it.

When the container style is configured but cannot be used -- the installed
discord.py has no Components V2, or Discord refuses the container with a 400 --
the same card goes out as a classic embed instead, once. An edit always uses
the embed: a message cannot gain the Components V2 flag after it was sent.

Imports Discord and the renderer only: no model, no database.
"""

from __future__ import annotations

import logging
from typing import Any

import discord

from aura.answer_card import (
    AnswerCard,
    card_to_embed,
    card_to_layout_view,
    components_v2_available,
)
from aura.config import CardStyle

logger = logging.getLogger(__name__)

_BAD_REQUEST = 400


def _use_container(style: CardStyle) -> bool:
    return style is CardStyle.CONTAINER and components_v2_available()


async def send_card_to_channel(
    channel: discord.abc.Messageable, card: AnswerCard, *, style: CardStyle
) -> None:
    """Post `card` in `channel`, mentions disabled.

    Parameters
    ----------
    channel
        Where to post.
    card
        The card.
    style
        ANSWER_CARD_STYLE.

    Returns
    -------
    None

    Raises
    ------
    discord.HTTPException
        When the send fails for any reason other than a refused container (the
        caller's own failure path handles it, exactly as for the classic embed).
    """
    mentions = discord.AllowedMentions.none()
    if _use_container(style):
        try:
            await channel.send(view=card_to_layout_view(card), allowed_mentions=mentions)
            return
        except discord.HTTPException as exc:
            if exc.status != _BAD_REQUEST:
                raise
            logger.warning(
                "Discord refused a %s card as a container; sending it as an embed", card.kind.value
            )
    await channel.send(embed=card_to_embed(card), allowed_mentions=mentions)


async def reply_with_card(
    interaction: discord.Interaction[Any],
    card: AnswerCard,
    *,
    style: CardStyle,
    ephemeral: bool = True,
) -> None:
    """Answer a command with `card`: the first response, or a followup when one was sent.

    Parameters
    ----------
    interaction
        The command invocation.
    card
        The card.
    style
        ANSWER_CARD_STYLE.
    ephemeral
        Only the invoker sees it (every command reply that has a card is).

    Returns
    -------
    None
    """
    mentions = discord.AllowedMentions.none()

    async def send(**kwargs: Any) -> None:
        if interaction.response.is_done():
            await interaction.followup.send(
                ephemeral=ephemeral, allowed_mentions=mentions, **kwargs
            )
        else:
            await interaction.response.send_message(
                ephemeral=ephemeral, allowed_mentions=mentions, **kwargs
            )

    if _use_container(style):
        try:
            await send(view=card_to_layout_view(card))
            return
        except discord.HTTPException as exc:
            if exc.status != _BAD_REQUEST:
                raise
            logger.warning(
                "Discord refused a %s card as a container; sending it as an embed", card.kind.value
            )
    await send(embed=card_to_embed(card))


async def edit_with_card(interaction: discord.Interaction[Any], card: AnswerCard) -> None:
    """Replace the message a component belongs to with `card`, drawn as an embed.

    Parameters
    ----------
    interaction
        A button press on the message to replace.
    card
        The card.

    Returns
    -------
    None
    """
    await interaction.response.edit_message(content=None, embed=card_to_embed(card), view=None)
