"""The one seam between the backfill worker and a live Discord connection.

Exactly the shape aura.digest.gateway established, and deliberately so: the
worker needs one thing from the gateway -- turn a stored channel id into
something it can read history from -- and everything else it does is database
work, local filtering and one LLM call. Naming that one thing as a Protocol is
what keeps the cursor logic, the ordering enforcement, the boundary against live
extraction and the spend accounting testable with no gateway connection at all,
per CLAUDE.md's rule that this project's logic must never require a live Discord
client to verify.

The difference from the digest's gateway is what the resolved channel is FOR.
The digest resolves a channel to post into and treats every failure as "not this
tick"; backfill resolves a channel to READ, and two of its failures are
permanent decisions someone made -- the channel was deleted, or Read Message
History was revoked. Those end a run rather than deferring it (see
aura.backfill.history.ChannelUnreadable), so this gateway distinguishes them
instead of flattening everything into None.
"""
from __future__ import annotations

import logging
from typing import Protocol

import discord

from aura.backfill.history import ChannelUnreadable

logger = logging.getLogger(__name__)


class BackfillGateway(Protocol):
    """Resolves a backfill run's source channel. Implemented against the real client below."""

    async def resolve_channel(self, channel_id: int) -> discord.TextChannel:
        """Return the text channel to read history from.

        Raises ChannelUnreadable when the channel is permanently unusable.
        Returns normally, or raises, and never returns None -- a backfill run
        with no channel has nothing to defer to.
        """
        ...


class ClientBackfillGateway:
    """Resolves backfill source channels through a live discord.py client.

    Cache first, HTTP second, exactly as ClientDigestGateway does: `get_channel`
    is a free dictionary lookup that succeeds for every channel the bot can see,
    and `fetch_channel` is only reached for one that has fallen out of (or never
    entered) the cache. A worker tick over a handful of runs therefore normally
    makes no HTTP request for resolution at all.
    """

    def __init__(self, client: discord.Client) -> None:
        self._client = client

    async def resolve_channel(self, channel_id: int) -> discord.TextChannel:
        """Return channel_id as a readable text channel, or raise ChannelUnreadable.

        Every failure here is treated as permanent, and that is a deliberate
        difference from the digest gateway rather than an oversight. A digest
        whose channel is temporarily unresolvable simply arrives next hour; a
        backfill run whose channel cannot be resolved has nothing to do for as
        long as that lasts, and retrying it forever would fill the log with one
        line per tick while a moderator waits for a status that never changes.
        Ending the run with FAILED puts the situation in front of the person who
        can fix it, and starting a new one afterwards costs one command.

        The isinstance check is not defensive typing. The slash command only
        accepts a TextChannel, but nothing stops that channel being deleted and
        its id reused by a channel type with no readable message history, and
        `history()` on such an object is not a failure worth discovering inside
        a paid pipeline.
        """
        channel = self._client.get_channel(channel_id)
        if channel is None:
            try:
                channel = await self._client.fetch_channel(channel_id)
            except discord.HTTPException as exc:
                raise ChannelUnreadable(
                    f"channel {channel_id} could not be fetched: {exc}"
                ) from exc

        if not isinstance(channel, discord.TextChannel):
            raise ChannelUnreadable(
                f"channel {channel_id} is a {type(channel).__name__}, not a text channel"
            )
        return channel
