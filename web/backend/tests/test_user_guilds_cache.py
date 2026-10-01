"""The per-user guild cache: freshness, one fetch per user, and what a failure may and may not hide.

The failure rules are the point. A rejected token must never be papered over
by a cached answer (the session has to end), a cold failure must never become
an empty list, and a stale list must stop being served once it is old enough
to be wrong.
"""

from __future__ import annotations

import asyncio
import gc
import logging

import pytest

from aura_web.discord_api import DiscordAuthError, DiscordUnavailableError, PartialGuild
from aura_web.user_guilds import UserGuildCache

MANAGE_GUILD = str(1 << 5)


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def guild(guild_id: str) -> PartialGuild:
    return PartialGuild(id=guild_id, name=f"Guild {guild_id}", icon=None, permissions=MANAGE_GUILD)


class RecordingClient:
    """A stand-in for DiscordClient: a guild list per token, a call log, and failure switches."""

    def __init__(self) -> None:
        self.guilds: dict[str, list[PartialGuild]] = {}
        self.calls: list[str] = []
        self.failure: Exception | None = None
        self.release: asyncio.Event | None = None

    async def fetch_user_guilds(self, access_token: str) -> list[PartialGuild]:
        self.calls.append(access_token)
        if self.release is not None:
            await self.release.wait()
        if self.failure is not None:
            raise self.failure
        return list(self.guilds.get(access_token, []))


def make_cache(
    client: RecordingClient,
    clock: FakeClock,
    *,
    ttl: float = 30,
    tolerance: float = 60,
    max_entries: int = 100,
) -> UserGuildCache:
    return UserGuildCache(
        client,  # type: ignore[arg-type]
        ttl_seconds=ttl,
        stale_tolerance_seconds=tolerance,
        max_entries=max_entries,
        monotonic=clock,
    )


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def client() -> RecordingClient:
    recording = RecordingClient()
    recording.guilds = {"alice-token": [guild("1")], "bob-token": [guild("2")]}
    return recording


class TestFreshness:
    async def test_a_second_read_inside_the_ttl_does_not_ask_discord(
        self, client: RecordingClient, clock: FakeClock
    ) -> None:
        cache = make_cache(client, clock)
        await cache.get("alice-token")
        clock.advance(29.9)
        assert await cache.get("alice-token") == [guild("1")]
        assert client.calls == ["alice-token"]

    async def test_a_read_at_the_ttl_asks_again_and_sees_the_change(
        self, client: RecordingClient, clock: FakeClock
    ) -> None:
        cache = make_cache(client, clock)
        await cache.get("alice-token")
        client.guilds["alice-token"] = []
        clock.advance(30)
        assert await cache.get("alice-token") == []
        assert len(client.calls) == 2

    async def test_users_never_see_each_other_s_guilds(
        self, client: RecordingClient, clock: FakeClock
    ) -> None:
        cache = make_cache(client, clock)
        assert await cache.get("alice-token") == [guild("1")]
        assert await cache.get("bob-token") == [guild("2")]
        assert await cache.get("alice-token") == [guild("1")]

    async def test_a_caller_mutating_its_list_does_not_touch_the_cache(
        self, client: RecordingClient, clock: FakeClock
    ) -> None:
        cache = make_cache(client, clock)
        first = await cache.get("alice-token")
        first.append(guild("999"))
        first.clear()
        assert await cache.get("alice-token") == [guild("1")]

    async def test_forget_makes_the_next_read_ask_discord(
        self, client: RecordingClient, clock: FakeClock
    ) -> None:
        cache = make_cache(client, clock)
        await cache.get("alice-token")
        cache.forget("alice-token")
        cache.forget("never-seen-token")
        await cache.get("alice-token")
        assert client.calls == ["alice-token", "alice-token"]

    async def test_no_token_is_held_in_the_cache(
        self, client: RecordingClient, clock: FakeClock
    ) -> None:
        cache = make_cache(client, clock)
        await cache.get("alice-token")
        assert "alice-token" not in repr(vars(cache)).replace("RecordingClient", "")
        assert all("alice-token" not in key for key in cache._entries)


class TestOneFetchPerUser:
    async def test_simultaneous_reads_for_one_user_cost_one_call(
        self, client: RecordingClient, clock: FakeClock
    ) -> None:
        cache = make_cache(client, clock)
        client.release = asyncio.Event()
        readers = [asyncio.create_task(cache.get("alice-token")) for _ in range(20)]
        await asyncio.sleep(0)
        client.release.set()
        results = await asyncio.gather(*readers)
        assert client.calls == ["alice-token"]
        assert all(result == [guild("1")] for result in results)

    async def test_different_users_are_fetched_independently(
        self, client: RecordingClient, clock: FakeClock
    ) -> None:
        cache = make_cache(client, clock)
        results = await asyncio.gather(cache.get("alice-token"), cache.get("bob-token"))
        assert results == [[guild("1")], [guild("2")]]
        assert sorted(client.calls) == ["alice-token", "bob-token"]

    async def test_a_cancelled_waiter_does_not_cancel_the_fetch_for_the_others(
        self, client: RecordingClient, clock: FakeClock
    ) -> None:
        cache = make_cache(client, clock)
        client.release = asyncio.Event()
        first = asyncio.create_task(cache.get("alice-token"))
        second = asyncio.create_task(cache.get("alice-token"))
        await asyncio.sleep(0)
        first.cancel()
        client.release.set()
        assert await second == [guild("1")]
        with pytest.raises(asyncio.CancelledError):
            await first
        assert client.calls == ["alice-token"]

    async def test_a_failed_fetch_fails_every_waiter_and_the_next_read_retries(
        self, client: RecordingClient, clock: FakeClock
    ) -> None:
        cache = make_cache(client, clock)
        client.release = asyncio.Event()
        client.failure = DiscordUnavailableError("Discord rate-limited (retry after 0.4s)")
        readers = [asyncio.create_task(cache.get("alice-token")) for _ in range(5)]
        await asyncio.sleep(0)
        client.release.set()
        outcomes = await asyncio.gather(*readers, return_exceptions=True)
        assert all(isinstance(outcome, DiscordUnavailableError) for outcome in outcomes)
        client.failure = None
        client.release = None
        assert await cache.get("alice-token") == [guild("1")]
        assert len(client.calls) == 2

    async def test_a_failure_nobody_waited_for_is_not_reported_as_unretrieved(
        self, client: RecordingClient, clock: FakeClock
    ) -> None:
        loop = asyncio.get_running_loop()
        reported: list[dict[str, object]] = []
        previous_handler = loop.get_exception_handler()
        loop.set_exception_handler(lambda _loop, context: reported.append(context))
        try:
            cache = make_cache(client, clock)
            client.release = asyncio.Event()
            client.failure = DiscordUnavailableError("down")
            reader = asyncio.create_task(cache.get("alice-token"))
            await asyncio.sleep(0)
            reader.cancel()
            client.release.set()
            for _ in range(5):
                await asyncio.sleep(0)
            del reader
            gc.collect()
        finally:
            loop.set_exception_handler(previous_handler)
        assert reported == []


class TestFailures:
    async def test_a_cold_failure_propagates_instead_of_an_empty_list(
        self, client: RecordingClient, clock: FakeClock
    ) -> None:
        cache = make_cache(client, clock)
        client.failure = DiscordUnavailableError("down")
        with pytest.raises(DiscordUnavailableError):
            await cache.get("alice-token")
        assert len(cache) == 0

    async def test_a_rate_limited_refresh_inside_the_tolerance_serves_the_last_list(
        self,
        client: RecordingClient,
        clock: FakeClock,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        cache = make_cache(client, clock)
        await cache.get("alice-token")
        clock.advance(89)
        client.failure = DiscordUnavailableError("Discord rate-limited (retry after 1s)")
        with caplog.at_level(logging.WARNING):
            assert await cache.get("alice-token") == [guild("1")]
        assert "alice-token" not in caplog.text

    async def test_a_failed_refresh_past_the_tolerance_propagates_and_drops_the_entry(
        self, client: RecordingClient, clock: FakeClock
    ) -> None:
        cache = make_cache(client, clock)
        await cache.get("alice-token")
        clock.advance(91)
        client.failure = DiscordUnavailableError("down")
        with pytest.raises(DiscordUnavailableError):
            await cache.get("alice-token")
        assert len(cache) == 0

    async def test_a_rejected_token_is_never_bridged_by_the_cached_list(
        self, client: RecordingClient, clock: FakeClock
    ) -> None:
        cache = make_cache(client, clock)
        await cache.get("alice-token")
        clock.advance(31)
        client.failure = DiscordAuthError("401")
        with pytest.raises(DiscordAuthError):
            await cache.get("alice-token")
        assert len(cache) == 0
        client.failure = DiscordUnavailableError("down")
        with pytest.raises(DiscordUnavailableError):
            await cache.get("alice-token")


class TestBoundedMemory:
    async def test_never_more_than_max_entries_and_the_least_recent_goes_first(
        self, client: RecordingClient, clock: FakeClock
    ) -> None:
        cache = make_cache(client, clock, max_entries=2)
        await cache.get("alice-token")
        await cache.get("bob-token")
        await cache.get("alice-token")
        await cache.get("carol-token")
        assert len(cache) == 2
        await cache.get("alice-token")
        assert client.calls.count("alice-token") == 1
        await cache.get("bob-token")
        assert client.calls.count("bob-token") == 2

    async def test_a_refreshed_entry_counts_as_recently_used(
        self, client: RecordingClient, clock: FakeClock
    ) -> None:
        cache = make_cache(client, clock, max_entries=2)
        await cache.get("alice-token")
        await cache.get("bob-token")
        clock.advance(31)
        await cache.get("alice-token")
        await cache.get("carol-token")
        clock.advance(1)
        await cache.get("alice-token")
        assert client.calls.count("alice-token") == 2

    async def test_a_flood_of_distinct_tokens_stays_bounded(
        self, client: RecordingClient, clock: FakeClock
    ) -> None:
        cache = make_cache(client, clock, max_entries=10)
        for index in range(1_000):
            await cache.get(f"token-{index}")
        assert len(cache) == 10


class TestConstruction:
    @pytest.mark.parametrize(
        ("ttl", "tolerance", "max_entries"),
        [(0, 0, 1), (-1, 0, 1), (1, -1, 1), (1, 0, 0), (1, 0, -5)],
    )
    def test_nonsense_bounds_are_refused(
        self, client: RecordingClient, ttl: float, tolerance: float, max_entries: int
    ) -> None:
        with pytest.raises(ValueError):
            UserGuildCache(
                client,  # type: ignore[arg-type]
                ttl_seconds=ttl,
                stale_tolerance_seconds=tolerance,
                max_entries=max_entries,
            )
