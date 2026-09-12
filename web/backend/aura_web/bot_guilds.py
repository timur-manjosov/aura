"""A short-lived cache of the guilds Aura is actually a member of.

Two problems make a bare call to DiscordClient.fetch_bot_guild_ids the wrong
thing to do per request.

The first is the rate limit. /users/@me/guilds is the route Discord throttles
hardest for dashboard-shaped traffic, and the answer changes when a human
invites or removes a bot -- on the order of days, not seconds. Caching it for
a minute removes one Discord round trip from every page load and costs
nothing anyone can perceive.

The second is the thundering herd the first fix would otherwise create: when
the entry expires, every in-flight request would refresh simultaneously and
turn one expensive route into N. A single refresh lock means the first caller
does the work and the rest await its result.

A refresh that fails does not immediately break the page. A cached list
inside the stale tolerance is served with a warning, because a membership
list five minutes old is right in every case that matters here and an error
page is right in none of them. Past that tolerance the cache is dropped and
the error propagates: this service fails closed. It never falls back to "show
the user every guild they can manage", which would list servers Aura is not
in -- the exact filtering failure this sub-phase's brief calls out.
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable

from aura_web.discord_api import DiscordAPIError, DiscordClient

logger = logging.getLogger(__name__)


class BotGuildCache:
    """TTL cache over the bot's guild membership, with bounded stale tolerance.

    Construct it inside the running event loop (the application lifespan does,
    and tests build one per test). The refresh lock binds to whichever loop
    first contends it, so one cache object must not outlive its loop -- the
    same constraint, for the same reason, that aura.db.connection documents
    for its per-connection lock.
    """

    def __init__(
        self,
        client: DiscordClient,
        *,
        ttl_seconds: float,
        stale_tolerance_seconds: float,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        if stale_tolerance_seconds < 0:
            raise ValueError("stale_tolerance_seconds must not be negative")
        self._client = client
        self._ttl = ttl_seconds
        self._stale_tolerance = stale_tolerance_seconds
        # A monotonic clock, not a wall clock: a TTL measured against
        # datetime.now() inverts when the host's clock steps backwards over an
        # NTP correction, and a cache that believes its entry is from the
        # future never refreshes again.
        self._monotonic = monotonic
        self._lock = asyncio.Lock()
        self._guild_ids: frozenset[str] | None = None
        self._fetched_at: float | None = None

    async def get(self) -> frozenset[str]:
        """Return the bot's guild IDs, refreshing if the cached entry has expired.

        Raises whatever DiscordClient raised if there is no cached value new
        enough to stand in -- callers must surface that rather than degrade to
        an unfiltered list.
        """
        if self._is_fresh():
            assert self._guild_ids is not None  # guaranteed by _is_fresh
            return self._guild_ids

        async with self._lock:
            # Re-checked inside the lock: while this caller waited, the holder
            # may have refreshed already, and refreshing twice would defeat
            # the point of holding the lock at all.
            if self._is_fresh():
                assert self._guild_ids is not None
                return self._guild_ids

            try:
                guild_ids = await self._client.fetch_bot_guild_ids()
            except DiscordAPIError as exc:
                return self._fall_back_to_stale(exc)

            self._guild_ids = guild_ids
            self._fetched_at = self._monotonic()
            return guild_ids

    def invalidate(self) -> None:
        """Drop the cached entry, forcing the next get() to refetch."""
        self._guild_ids = None
        self._fetched_at = None

    def _is_fresh(self) -> bool:
        if self._guild_ids is None or self._fetched_at is None:
            return False
        return (self._monotonic() - self._fetched_at) < self._ttl

    def _fall_back_to_stale(self, exc: DiscordAPIError) -> frozenset[str]:
        if self._guild_ids is None or self._fetched_at is None:
            raise exc
        age = self._monotonic() - self._fetched_at
        if age > self._ttl + self._stale_tolerance:
            # Dropped rather than kept: a list this old failing to refresh is
            # more likely to be wrong than useful, and keeping it would let a
            # long Discord outage serve an ever-more-wrong answer silently.
            self.invalidate()
            raise exc
        logger.warning(
            "Could not refresh Aura's guild membership from Discord (%s); "
            "serving a cached list %.0fs old",
            exc,
            age,
        )
        return self._guild_ids
