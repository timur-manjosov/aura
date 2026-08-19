"""The one seam between the onboarding listener and a live Discord connection.

Structurally identical to aura.digest.gateway -- same Protocol, same
cache-then-fetch resolution, same two failure modes answered once (a channel
that no longer exists or is no longer visible, and one that is not a text
channel at all). Deliberately a SEPARATE small adapter rather than a shared
one: this ~40-line class is not the security-critical logic this sub-phase
was told to reuse rather than reinvent (that is aura.rendering, and it is
shared); it is a thin, feature-specific wrapper whose only content is which
log line names which feature. Sharing it would mean either generic log
messages that no longer say "digest" or "onboarding" plainly, or a
parameterised "feature name" threaded through a class that exists to resolve
one channel ID -- more machinery than either caller needs. Duplicating this
much, once, is the cheaper and clearer trade.
"""
from __future__ import annotations

import logging
from typing import Protocol

import discord

logger = logging.getLogger(__name__)


class OnboardingGateway(Protocol):
    """Resolves onboarding's target channel. Implemented against the real client below."""

    async def resolve_channel(self, channel_id: int) -> discord.TextChannel | None:
        """Return the text channel to post into, or None if it cannot be used."""
        ...


class ClientOnboardingGateway:
    """Resolves onboarding channels through a live discord.py client.

    Cache first, HTTP second, exactly as ClientDigestGateway does: a join
    handler that fires for every new member should not cost an HTTP round
    trip for every one of them when the channel is already cached, which it
    almost always is.
    """

    def __init__(self, client: discord.Client) -> None:
        self._client = client

    async def resolve_channel(self, channel_id: int) -> discord.TextChannel | None:
        """Return channel_id as a postable text channel, or None with a logged reason.

        Returns None rather than raising for every way this can fail: a
        moderator can delete the onboarding channel, revoke Aura's access to
        it, or convert it into a type Aura cannot post an embed into, and all
        three are "no onboarding message this time, tell the log why" rather
        than a crash in a gateway event handler.
        """
        channel = self._client.get_channel(channel_id)
        if channel is None:
            try:
                channel = await self._client.fetch_channel(channel_id)
            except discord.HTTPException as exc:
                logger.warning(
                    "Onboarding channel %s could not be fetched: %s", channel_id, exc
                )
                return None

        if not isinstance(channel, discord.TextChannel):
            logger.warning(
                "Onboarding channel %s is a %s, not a text channel; no onboarding message "
                "can be posted there until a moderator picks another with /aura-onboarding",
                channel_id,
                type(channel).__name__,
            )
            return None
        return channel
