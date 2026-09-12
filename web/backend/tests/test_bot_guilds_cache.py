"""The bot-membership cache: freshness, the refresh lock, and failing closed.

The behaviour under failure is the part that matters. A cache that quietly
returns an empty set when Discord is unreachable would make every user's
guild list empty; one that returned "everything" would list servers Aura is
not in. Neither is acceptable, so the failure path raises -- and these tests
pin that down rather than the happy path.
"""
from __future__ import annotations

import asyncio

import pytest

from aura_web.bot_guilds import BotGuildCache
from aura_web.discord_api import DiscordUnavailableError


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class RecordingClient:
    """A stand-in for DiscordClient that counts calls and can be made to fail."""

    def __init__(self, guild_ids: set[str]) -> None:
        self.guild_ids = guild_ids
        self.calls = 0
        self.failure: Exception | None = None
        self.delay = 0.0

    async def fetch_bot_guild_ids(self) -> frozenset[str]:
        self.calls += 1
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.failure is not None:
            raise self.failure
        return frozenset(self.guild_ids)


class TestFreshness:
    async def test_the_first_call_fetches(self) -> None:
        client = RecordingClient({"1", "2"})
        cache = BotGuildCache(client, ttl_seconds=60, stale_tolerance_seconds=0)  # type: ignore[arg-type]

        assert await cache.get() == frozenset({"1", "2"})
        assert client.calls == 1

    async def test_a_second_call_inside_the_ttl_does_not_refetch(self) -> None:
        clock = FakeClock()
        client = RecordingClient({"1"})
        cache = BotGuildCache(client, ttl_seconds=60, stale_tolerance_seconds=0, monotonic=clock)  # type: ignore[arg-type]

        await cache.get()
        clock.advance(59)
        await cache.get()

        assert client.calls == 1

    async def test_a_call_past_the_ttl_refetches(self) -> None:
        clock = FakeClock()
        client = RecordingClient({"1"})
        cache = BotGuildCache(client, ttl_seconds=60, stale_tolerance_seconds=0, monotonic=clock)  # type: ignore[arg-type]

        await cache.get()
        clock.advance(61)
        client.guild_ids = {"1", "2"}
        assert await cache.get() == frozenset({"1", "2"})
        assert client.calls == 2

    async def test_invalidate_forces_a_refetch(self) -> None:
        client = RecordingClient({"1"})
        cache = BotGuildCache(client, ttl_seconds=600, stale_tolerance_seconds=0)  # type: ignore[arg-type]

        await cache.get()
        cache.invalidate()
        await cache.get()

        assert client.calls == 2

    async def test_an_empty_membership_list_is_cached_like_any_other(self) -> None:
        """Aura in no guilds is a real answer, not a cache miss to retry forever."""
        client = RecordingClient(set())
        cache = BotGuildCache(client, ttl_seconds=600, stale_tolerance_seconds=0)  # type: ignore[arg-type]

        assert await cache.get() == frozenset()
        await cache.get()
        assert client.calls == 1


class TestFailureHandling:
    async def test_a_cold_failure_propagates_rather_than_returning_empty(self) -> None:
        """Returning an empty set here would silently empty every user's dashboard."""
        client = RecordingClient({"1"})
        client.failure = DiscordUnavailableError("down")
        cache = BotGuildCache(client, ttl_seconds=60, stale_tolerance_seconds=0)  # type: ignore[arg-type]

        with pytest.raises(DiscordUnavailableError):
            await cache.get()

    async def test_a_warm_failure_inside_the_tolerance_serves_the_cached_list(self) -> None:
        clock = FakeClock()
        client = RecordingClient({"1"})
        cache = BotGuildCache(client, ttl_seconds=60, stale_tolerance_seconds=300, monotonic=clock)  # type: ignore[arg-type]
        await cache.get()

        clock.advance(100)
        client.failure = DiscordUnavailableError("down")

        assert await cache.get() == frozenset({"1"})

    async def test_a_warm_failure_past_the_tolerance_propagates(self) -> None:
        clock = FakeClock()
        client = RecordingClient({"1"})
        cache = BotGuildCache(client, ttl_seconds=60, stale_tolerance_seconds=300, monotonic=clock)  # type: ignore[arg-type]
        await cache.get()

        clock.advance(400)
        client.failure = DiscordUnavailableError("down")

        with pytest.raises(DiscordUnavailableError):
            await cache.get()

    async def test_a_list_too_stale_to_serve_is_dropped_not_kept(self) -> None:
        """Otherwise a long outage serves an ever-more-wrong answer, silently."""
        clock = FakeClock()
        client = RecordingClient({"1"})
        cache = BotGuildCache(client, ttl_seconds=60, stale_tolerance_seconds=100, monotonic=clock)  # type: ignore[arg-type]
        await cache.get()
        clock.advance(500)
        client.failure = DiscordUnavailableError("down")
        with pytest.raises(DiscordUnavailableError):
            await cache.get()

        client.failure = None
        client.guild_ids = {"9"}
        assert await cache.get() == frozenset({"9"})

    async def test_recovery_after_an_outage_refetches_cleanly(self) -> None:
        clock = FakeClock()
        client = RecordingClient({"1"})
        cache = BotGuildCache(client, ttl_seconds=60, stale_tolerance_seconds=300, monotonic=clock)  # type: ignore[arg-type]
        await cache.get()
        clock.advance(100)
        client.failure = DiscordUnavailableError("down")
        await cache.get()

        client.failure = None
        client.guild_ids = {"1", "2"}
        clock.advance(100)

        assert await cache.get() == frozenset({"1", "2"})


class TestRefreshLock:
    async def test_concurrent_cold_reads_cost_exactly_one_fetch(self) -> None:
        """Without the lock this is N requests against Discord's tightest route."""
        client = RecordingClient({"1"})
        client.delay = 0.02
        cache = BotGuildCache(client, ttl_seconds=60, stale_tolerance_seconds=0)  # type: ignore[arg-type]

        results = await asyncio.gather(*(cache.get() for _ in range(20)))

        assert client.calls == 1
        assert all(result == frozenset({"1"}) for result in results)

    async def test_concurrent_reads_after_expiry_also_cost_one_fetch(self) -> None:
        clock = FakeClock()
        client = RecordingClient({"1"})
        cache = BotGuildCache(client, ttl_seconds=60, stale_tolerance_seconds=0, monotonic=clock)  # type: ignore[arg-type]
        await cache.get()
        clock.advance(61)
        client.delay = 0.02

        await asyncio.gather(*(cache.get() for _ in range(10)))

        assert client.calls == 2

    async def test_a_failing_concurrent_refresh_fails_every_waiter_identically(self) -> None:
        client = RecordingClient({"1"})
        client.failure = DiscordUnavailableError("down")
        client.delay = 0.01
        cache = BotGuildCache(client, ttl_seconds=60, stale_tolerance_seconds=0)  # type: ignore[arg-type]

        results = await asyncio.gather(
            *(cache.get() for _ in range(5)), return_exceptions=True
        )

        assert all(isinstance(result, DiscordUnavailableError) for result in results)


class TestConstruction:
    @pytest.mark.parametrize(("ttl", "tolerance"), [(0, 0), (-1, 0), (60, -1)])
    def test_nonsense_bounds_are_rejected(self, ttl: float, tolerance: float) -> None:
        client = RecordingClient(set())
        with pytest.raises(ValueError):
            BotGuildCache(client, ttl_seconds=ttl, stale_tolerance_seconds=tolerance)  # type: ignore[arg-type]
