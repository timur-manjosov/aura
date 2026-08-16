"""The one seam between the digest scheduler and a live Discord connection.

The scheduler needs exactly one thing from the gateway -- turn a stored channel
ID into something it can post into -- and everything else it does is database
reads, pure assembly and one send. Naming that one thing as a Protocol is what
keeps the whole of the scheduling, bookkeeping and content logic testable
without a gateway connection, per CLAUDE.md's rule that this project's logic
must never require a live Discord client to verify.

It is also where the two questions a stored channel ID can fail on are answered
once, rather than at the send: the channel may no longer exist or be visible,
and it may not be a text channel at all any more.
"""
from __future__ import annotations

import logging
from typing import Protocol

import discord

logger = logging.getLogger(__name__)


class DigestGateway(Protocol):
    """Resolves a digest's target channel. Implemented against the real client below."""

    async def resolve_channel(self, channel_id: int) -> discord.TextChannel | None:
        """Return the text channel to post into, or None if it cannot be used."""
        ...


class ClientDigestGateway:
    """Resolves digest channels through a live discord.py client.

    Cache first, HTTP second: `get_channel` is a free dictionary lookup that
    succeeds for every channel the bot can see, and `fetch_channel` is only
    reached for one that has fallen out of (or never entered) the cache. A
    scheduler tick over a handful of guilds therefore normally makes no HTTP
    requests at all.
    """

    def __init__(self, client: discord.Client) -> None:
        self._client = client

    async def resolve_channel(self, channel_id: int) -> discord.TextChannel | None:
        """Return channel_id as a postable text channel, or None with a logged reason.

        Returns None rather than raising for every way this can fail, because
        none of them is exceptional from the scheduler's point of view: a
        moderator can delete the digest channel, revoke Aura's access to it, or
        convert it into a type Aura cannot post an embed into, and all three are
        "no digest this time, tell the log why" rather than a crash in a
        background task.

        The isinstance check is not defensive typing. The slash command only
        accepts a TextChannel, but nothing stops that channel being deleted and
        its ID reused, or the API returning a type this code cannot treat as a
        guild text channel -- and the digest's guild check downstream depends on
        the object actually having a `.guild`.
        """
        channel = self._client.get_channel(channel_id)
        if channel is None:
            try:
                channel = await self._client.fetch_channel(channel_id)
            except discord.HTTPException as exc:
                # NotFound (deleted), Forbidden (access revoked) and every
                # transient API error land here; all mean "not postable now".
                logger.warning("Digest channel %s could not be fetched: %s", channel_id, exc)
                return None

        if not isinstance(channel, discord.TextChannel):
            logger.warning(
                "Digest channel %s is a %s, not a text channel; no digest can be posted "
                "there until a moderator picks another with /aura-digest",
                channel_id,
                type(channel).__name__,
            )
            return None
        return channel
