"""The discord.py side of the P7a purge job and author lookup: thin adapters over the client.

Kept separate so `aura.privacy.sweeper` and `aura.privacy.author_lookup` stay
testable without a gateway connection.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import discord

from aura.privacy.author_lookup import AuthorUnknown, LookupAnswer, LookupFailed, LookupRateLimited

if TYPE_CHECKING:
    from aura.main import AuraClient

logger = logging.getLogger(__name__)


class ClientGuildPresence:
    """Answers "is Aura in this server right now" from the live client."""

    def __init__(self, client: AuraClient) -> None:
        self._client = client

    def is_ready(self) -> bool:
        """Report whether the gateway is connected and its guild list complete.

        Returns
        -------
        bool
            discord.py's own readiness flag, false while (re)connecting.
        """
        return self._client.is_ready() and not self._client.is_closed()

    def is_member_of(self, guild_id: int) -> bool:
        """Report whether Aura is in a server, counting temporarily unavailable ones.

        Parameters
        ----------
        guild_id
            The server.

        Returns
        -------
        bool
            True when the client knows the guild at all.
        """
        return self._client.get_guild(guild_id) is not None


class ClientMessageAuthorSource:
    """Looks up a message's author through the bot's own token."""

    def __init__(self, client: AuraClient) -> None:
        self._client = client

    async def author_of(self, channel_id: int, message_id: int) -> LookupAnswer:
        """Return the author ID of one message.

        Parameters
        ----------
        channel_id, message_id
            The message.

        Returns
        -------
        int or AuthorUnknown or LookupFailed or LookupRateLimited
            The author's ID; UNKNOWN when the channel or message is gone or no
            longer readable by Aura; RATE_LIMITED for a 429 that discord.py's
            own retries did not absorb; FAILED for any other error (retried
            later).
        """
        try:
            channel = self._client.get_channel(channel_id) or await self._client.fetch_channel(
                channel_id
            )
            if not isinstance(channel, discord.abc.Messageable):
                return AuthorUnknown.UNKNOWN
            message = await channel.fetch_message(message_id)
        except (discord.NotFound, discord.Forbidden):
            return AuthorUnknown.UNKNOWN
        except discord.HTTPException as exc:
            if exc.status == 429:
                return LookupRateLimited.RATE_LIMITED
            logger.warning("Author lookup: Discord answered HTTP %s; retried later", exc.status)
            return LookupFailed.FAILED
        return message.author.id
