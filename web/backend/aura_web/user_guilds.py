"""A short-lived, per-user cache of the guilds a signed-in user belongs to.

One page load asks for the same user's guild list more than once: the guild
list and the billing view each need it, and the frontend requests both at the
same moment. Discord's per-user bucket on ``/users/@me/guilds`` allows far
less than that -- in production the second call of one page load, 400 ms after
the first, was answered 429, and the billing view failed with it. No test
could see this, because the Discord double never rate-limited.

So the answer is reused for a short time, and concurrent requests for the same
user share one call (a single flight per user): N simultaneous requests cost
one Discord round trip, not N.

Invariants:

* Keyed by a SHA-256 digest of the access token, never the token itself, so
  this structure holds no credential. A refreshed or newly issued token is a
  new key; the old entry simply ages out.
* Bounded: at most ``max_entries`` users are remembered, least recently used
  forgotten first, because every signed-in user adds one.
* A credential failure (DiscordAuthError) drops the user's entry and is never
  masked by a cached answer: a revoked authorization ends the session.
* Other failures may be bridged by an entry within the stale tolerance, the
  same rule aura_web.bot_guilds applies to Aura's own membership; past it, the
  failure propagates and the caller fails closed.

Imports only the Discord client; knows nothing about sessions or routes.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass

from aura_web.discord_api import DiscordAPIError, DiscordAuthError, DiscordClient, PartialGuild

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _Entry:
    guilds: tuple[PartialGuild, ...]
    fetched_at: float


def _cache_key(access_token: str) -> str:
    return hashlib.sha256(access_token.encode("utf-8")).hexdigest()


class UserGuildCache:
    """TTL cache over each user's ``/users/@me/guilds``, with one fetch in flight per user.

    Construct it inside the running event loop (the application lifespan does,
    and tests build one per test): in-flight fetches are tasks on that loop.

    Parameters
    ----------
    client
        The Discord client the guild lists are fetched with.
    ttl_seconds
        How long a fetched list is served without asking Discord again.
    stale_tolerance_seconds
        How far past the TTL a list may still be served when a refresh fails
        for any reason other than a rejected credential.
    max_entries
        How many users' lists are remembered at most.
    monotonic
        The clock; injectable so tests need no sleeps.

    Raises
    ------
    ValueError
        If ``ttl_seconds`` or ``max_entries`` is not positive, or
        ``stale_tolerance_seconds`` is negative.
    """

    def __init__(
        self,
        client: DiscordClient,
        *,
        ttl_seconds: float,
        stale_tolerance_seconds: float,
        max_entries: int,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        if stale_tolerance_seconds < 0:
            raise ValueError("stale_tolerance_seconds must not be negative")
        if max_entries <= 0:
            raise ValueError("max_entries must be positive")
        self._client = client
        self._ttl = ttl_seconds
        self._stale_tolerance = stale_tolerance_seconds
        self._max_entries = max_entries
        self._monotonic = monotonic
        self._entries: OrderedDict[str, _Entry] = OrderedDict()
        self._in_flight: dict[str, asyncio.Task[tuple[PartialGuild, ...]]] = {}

    def __len__(self) -> int:
        return len(self._entries)

    async def get(self, access_token: str) -> list[PartialGuild]:
        """Return the user's guilds, from cache while fresh, otherwise from Discord.

        Parameters
        ----------
        access_token
            The user's access token, carrying the ``guilds`` scope.

        Returns
        -------
        list[PartialGuild]
            A new list on every call; callers may not affect one another.

        Raises
        ------
        DiscordAuthError
            When Discord rejects the token. The user's entry is dropped first.
        DiscordAPIError
            When a refresh is due, fails, and no entry within the stale
            tolerance remains.

        Notes
        -----
        The fetch runs as its own task and each caller awaits it shielded: a
        caller whose client disconnects is cancelled alone, and the fetch
        still completes for every other caller waiting on it.
        """
        key = _cache_key(access_token)
        entry = self._entries.get(key)
        if entry is not None and self._age(entry) < self._ttl:
            self._entries.move_to_end(key)
            return list(entry.guilds)

        task = self._in_flight.get(key)
        if task is None:
            task = asyncio.ensure_future(self._refresh(key, access_token))
            self._in_flight[key] = task
            task.add_done_callback(lambda done: self._finish(key, done))
        return list(await asyncio.shield(task))

    def forget(self, access_token: str) -> None:
        """Drop the user's cached list, so the next get() asks Discord.

        Parameters
        ----------
        access_token
            The access token the list was cached under.
        """
        self._entries.pop(_cache_key(access_token), None)

    async def _refresh(self, key: str, access_token: str) -> tuple[PartialGuild, ...]:
        try:
            guilds = tuple(await self._client.fetch_user_guilds(access_token))
        except DiscordAuthError:
            self._entries.pop(key, None)
            raise
        except DiscordAPIError as exc:
            return self._fall_back_to_stale(key, exc)
        self._entries[key] = _Entry(guilds, self._monotonic())
        self._entries.move_to_end(key)
        while len(self._entries) > self._max_entries:
            self._entries.popitem(last=False)
        return guilds

    def _finish(self, key: str, task: asyncio.Task[tuple[PartialGuild, ...]]) -> None:
        if self._in_flight.get(key) is task:
            del self._in_flight[key]
        # Marks a failure as retrieved even when every waiter was cancelled
        # first, so asyncio does not report it as never retrieved.
        if not task.cancelled():
            task.exception()

    def _age(self, entry: _Entry) -> float:
        return self._monotonic() - entry.fetched_at

    def _fall_back_to_stale(self, key: str, exc: DiscordAPIError) -> tuple[PartialGuild, ...]:
        entry = self._entries.get(key)
        if entry is None:
            raise exc
        age = self._age(entry)
        if age > self._ttl + self._stale_tolerance:
            del self._entries[key]
            raise exc
        logger.warning(
            "Could not refresh a user's guild list from Discord (%s); serving a cached list %.0fs old",
            exc,
            age,
        )
        return entry.guilds
