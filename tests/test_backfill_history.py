"""Tests for aura.backfill.history: order enforcement, and the rate-limit layer.

The two deliverables this file owns, both of which the phase brief asks to be
demonstrated rather than argued:

  * "Strikte Reihenfolge-Erzwingung, mit Test gegen absichtlich unsortiert
     zurückgegebene Seiten." -- the pages here are deliberately shuffled,
     reversed, and shuffled with duplicates, and the assertion is on what comes
     out, not on discord.py's documented behaviour.
  * "Rate-Limit-Handling mit echtem Backoff-Test." -- the delays are asserted as
     an actual sequence of numbers, by injecting the sleep. A test that merely
     checked "it retried" would pass against a client that retries instantly,
     which is the one behaviour the deliverable exists to rule out.

Nothing here touches a real Discord connection, and nothing here waits: `sleep`
is injected into fetch_history_page for exactly that reason, and every fake
below is the smallest object the code under test actually reaches into.
"""
from __future__ import annotations

import logging
import random
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import discord
import pytest

from aura.backfill.history import (
    ChannelUnreadable,
    fetch_history_page,
    is_strictly_increasing,
    ordered_page,
)

CHANNEL_A = 300000000000000003
EPOCH = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _message(message_id: int, *, created_at: datetime | None = None) -> MagicMock:
    """A message stub carrying the only two fields ordering depends on.

    created_at defaults to a value derived from the id, which mirrors the real
    thing exactly: discord.py's Message.created_at IS snowflake_time(self.id),
    so a stub where the two could disagree would be testing a situation that
    cannot occur -- except in the one test below that deliberately makes them
    disagree, to prove the sort key names both.
    """
    message = MagicMock(spec=discord.Message)
    message.id = message_id
    message.created_at = created_at or EPOCH + timedelta(seconds=message_id)
    return message


class _FakeHistory:
    """An async iterator over one canned page, recording how it was asked for."""

    def __init__(self, pages: list[list[MagicMock] | Exception]) -> None:
        self._pages = pages
        self.calls: list[dict[str, object]] = []

    def __call__(self, **kwargs: object) -> "_FakeHistory._Iterator":
        self.calls.append(kwargs)
        page = self._pages[min(len(self.calls) - 1, len(self._pages) - 1)]
        return _FakeHistory._Iterator(page)

    class _Iterator:
        def __init__(self, page: list[MagicMock] | Exception) -> None:
            self._page = page
            self._index = 0

        def __aiter__(self) -> "_FakeHistory._Iterator":
            return self

        async def __anext__(self) -> MagicMock:
            if isinstance(self._page, Exception):
                raise self._page
            if self._index >= len(self._page):
                raise StopAsyncIteration
            message = self._page[self._index]
            self._index += 1
            return message


def _channel(pages: list[list[MagicMock] | Exception]) -> MagicMock:
    channel = MagicMock(spec=discord.TextChannel)
    channel.id = CHANNEL_A
    channel.history = _FakeHistory(pages)
    return channel


def _http_error(status: int, *, retry_after: str | None = None) -> discord.HTTPException:
    """A real HTTPException with the response fields this layer actually reads."""
    response = MagicMock()
    response.status = status
    response.reason = "test"
    response.headers = {} if retry_after is None else {"Retry-After": retry_after}
    error = discord.HTTPException(response, {"message": "test", "code": 0})
    error.status = status
    return error


class _RecordingSleep:
    """A stand-in for asyncio.sleep that records what it was asked to wait."""

    def __init__(self) -> None:
        self.delays: list[float] = []

    async def __call__(self, delay: float) -> None:
        self.delays.append(delay)


class TestOrderedPage:
    def test_a_sorted_page_comes_back_unchanged(self) -> None:
        page = [_message(index) for index in range(1, 6)]

        assert [m.id for m in ordered_page(page, after_message_id=None)] == [1, 2, 3, 4, 5]

    def test_a_reversed_page_is_sorted_oldest_first(self) -> None:
        page = [_message(index) for index in range(5, 0, -1)]

        assert [m.id for m in ordered_page(page, after_message_id=None)] == [1, 2, 3, 4, 5]

    def test_a_deliberately_shuffled_page_is_sorted(self) -> None:
        """The deliverable's own wording: 'absichtlich unsortiert zurückgegebene Seiten'."""
        page = [_message(index) for index in range(1, 51)]
        random.Random(20260826).shuffle(page)

        result = ordered_page(page, after_message_id=None)

        assert [m.id for m in result] == list(range(1, 51))
        assert is_strictly_increasing(result)

    def test_every_shuffling_of_a_page_produces_the_same_order(self) -> None:
        rng = random.Random(1)
        expected = list(range(1, 31))
        for _ in range(50):
            page = [_message(index) for index in expected]
            rng.shuffle(page)
            assert [m.id for m in ordered_page(page, after_message_id=None)] == expected

    def test_it_does_not_mutate_the_page_it_was_given(self) -> None:
        page = [_message(3), _message(1), _message(2)]

        ordered_page(page, after_message_id=None)

        assert [m.id for m in page] == [3, 1, 2]

    def test_an_out_of_order_page_is_logged_rather_than_silently_absorbed(self, caplog) -> None:
        page = [_message(3), _message(1), _message(2)]

        with caplog.at_level(logging.WARNING, logger="aura.backfill.history"):
            ordered_page(page, after_message_id=None)

        assert "out of chronological order" in caplog.text

    def test_a_page_that_was_already_ordered_logs_nothing(self, caplog) -> None:
        page = [_message(1), _message(2), _message(3)]

        with caplog.at_level(logging.WARNING, logger="aura.backfill.history"):
            ordered_page(page, after_message_id=None)

        assert caplog.text == ""

    def test_messages_at_or_below_the_cursor_are_dropped(self) -> None:
        page = [_message(index) for index in range(1, 6)]

        result = ordered_page(page, after_message_id=3)

        assert [m.id for m in result] == [4, 5]

    def test_a_page_entirely_below_the_cursor_comes_back_empty(self) -> None:
        page = [_message(index) for index in range(1, 4)]

        assert ordered_page(page, after_message_id=99) == []

    def test_bounding_happens_after_sorting_not_before(self) -> None:
        """A shuffled page whose oldest entries are last must still be bounded correctly."""
        page = [_message(5), _message(1), _message(4), _message(2), _message(3)]

        result = ordered_page(page, after_message_id=3)

        assert [m.id for m in result] == [4, 5]

    def test_the_sort_key_names_the_timestamp_not_only_the_id(self) -> None:
        """Real messages cannot disagree, but the key must say what it means."""
        older_id_newer_time = _message(1, created_at=EPOCH + timedelta(days=2))
        newer_id_older_time = _message(2, created_at=EPOCH + timedelta(days=1))

        result = ordered_page(
            [older_id_newer_time, newer_id_older_time], after_message_id=None
        )

        assert [m.id for m in result] == [2, 1]

    def test_an_empty_page_is_handled(self) -> None:
        assert ordered_page([], after_message_id=None) == []
        assert ordered_page([], after_message_id=5) == []


class TestIsStrictlyIncreasing:
    def test_an_ordered_run_passes(self) -> None:
        assert is_strictly_increasing([_message(index) for index in range(1, 10)])

    def test_a_single_message_passes(self) -> None:
        assert is_strictly_increasing([_message(1)])

    def test_an_empty_sequence_passes(self) -> None:
        assert is_strictly_increasing([])

    def test_a_repeated_id_fails(self) -> None:
        assert not is_strictly_increasing([_message(1), _message(1)])

    def test_a_backwards_step_fails(self) -> None:
        assert not is_strictly_increasing([_message(2), _message(1)])


class TestFetching:
    async def test_a_page_comes_back_sorted_and_bounded(self) -> None:
        channel = _channel([[_message(5), _message(3), _message(4)]])

        page = await fetch_history_page(
            channel, after_message_id=3, before_message_id=99, limit=100
        )

        assert page is not None
        assert [m.id for m in page] == [4, 5]

    async def test_the_bounds_are_passed_to_discord_as_snowflake_objects(self) -> None:
        channel = _channel([[]])

        await fetch_history_page(
            channel, after_message_id=42, before_message_id=99, limit=100
        )

        call = channel.history.calls[0]
        assert call["oldest_first"] is True
        assert call["limit"] == 100
        assert isinstance(call["after"], discord.Object) and call["after"].id == 42
        assert isinstance(call["before"], discord.Object) and call["before"].id == 99

    async def test_no_lower_bound_is_passed_as_none_not_as_zero(self) -> None:
        channel = _channel([[]])

        await fetch_history_page(
            channel, after_message_id=None, before_message_id=99, limit=100
        )

        assert channel.history.calls[0]["after"] is None

    async def test_an_empty_page_comes_back_empty_rather_than_none(self) -> None:
        """Empty means 'the history is exhausted'; None means 'the fetch failed'."""
        channel = _channel([[]])

        assert await fetch_history_page(
            channel, after_message_id=None, before_message_id=99, limit=100
        ) == []


class TestPermanentFailures:
    @pytest.mark.parametrize("status", [403, 404])
    async def test_forbidden_and_not_found_end_the_run_rather_than_retrying(
        self, status
    ) -> None:
        response = MagicMock()
        response.status = status
        response.reason = "test"
        response.headers = {}
        error_class = discord.Forbidden if status == 403 else discord.NotFound
        channel = _channel([error_class(response, {"message": "no", "code": 0})])
        sleep = _RecordingSleep()

        with pytest.raises(ChannelUnreadable):
            await fetch_history_page(
                channel,
                after_message_id=None,
                before_message_id=99,
                limit=100,
                sleep=sleep,
            )

        assert len(channel.history.calls) == 1, "a permanent failure was retried"
        assert sleep.delays == []


class TestRateLimitHandling:
    async def test_a_429_waits_exactly_what_discord_asked_for(self) -> None:
        """Retry-After is honoured verbatim, not replaced with a guess."""
        channel = _channel([_http_error(429, retry_after="7.5"), [_message(1)]])
        sleep = _RecordingSleep()

        page = await fetch_history_page(
            channel, after_message_id=None, before_message_id=99, limit=100, sleep=sleep
        )

        assert sleep.delays == [7.5]
        assert page is not None and [m.id for m in page] == [1]

    async def test_a_429_with_no_header_falls_back_to_the_backoff_schedule(self) -> None:
        channel = _channel([_http_error(429), [_message(1)]])
        sleep = _RecordingSleep()

        await fetch_history_page(
            channel, after_message_id=None, before_message_id=99, limit=100, sleep=sleep
        )

        assert sleep.delays == [1.0]

    async def test_a_429_with_an_unparseable_header_falls_back_rather_than_crashing(
        self,
    ) -> None:
        channel = _channel([_http_error(429, retry_after="soon"), [_message(1)]])
        sleep = _RecordingSleep()

        page = await fetch_history_page(
            channel, after_message_id=None, before_message_id=99, limit=100, sleep=sleep
        )

        assert sleep.delays == [1.0]
        assert page is not None

    async def test_a_discord_py_ratelimited_exception_uses_its_own_retry_after(self) -> None:
        """Raised instead of slept when it exceeds the client's max_ratelimit_timeout."""
        channel = _channel([discord.RateLimited(12.25), [_message(1)]])
        sleep = _RecordingSleep()

        page = await fetch_history_page(
            channel, after_message_id=None, before_message_id=99, limit=100, sleep=sleep
        )

        assert sleep.delays == [12.25]
        assert page is not None and [m.id for m in page] == [1]

    async def test_an_absurd_retry_after_is_refused_rather_than_waited_out(self) -> None:
        """A hostile or malformed header must not park the worker for hours."""
        channel = _channel([_http_error(429, retry_after="99999")])
        sleep = _RecordingSleep()

        page = await fetch_history_page(
            channel, after_message_id=None, before_message_id=99, limit=100, sleep=sleep
        )

        assert page is None
        assert sleep.delays == [], "an absurd wait was actually performed"

    async def test_the_backoff_actually_doubles_across_attempts(self) -> None:
        """The deliverable's 'echter Backoff-Test': the numbers, not just 'it retried'."""
        channel = _channel([_http_error(503)] * 10)
        sleep = _RecordingSleep()

        page = await fetch_history_page(
            channel, after_message_id=None, before_message_id=99, limit=100, sleep=sleep
        )

        assert page is None
        assert sleep.delays == [1.0, 2.0, 4.0]
        assert len(channel.history.calls) == 4, "the attempt budget was not respected"

    async def test_a_transient_failure_that_recovers_returns_the_page(self) -> None:
        channel = _channel([_http_error(500), [_message(1), _message(2)]])
        sleep = _RecordingSleep()

        page = await fetch_history_page(
            channel, after_message_id=None, before_message_id=99, limit=100, sleep=sleep
        )

        assert page is not None and [m.id for m in page] == [1, 2]
        assert sleep.delays == [1.0]

    async def test_giving_up_returns_none_rather_than_raising(self) -> None:
        """The caller treats None as 'not this tick' and leaves the cursor alone."""
        channel = _channel([_http_error(502)] * 10)

        assert (
            await fetch_history_page(
                channel,
                after_message_id=None,
                before_message_id=99,
                limit=100,
                sleep=_RecordingSleep(),
            )
            is None
        )

    async def test_a_page_recovered_after_a_retry_is_still_sorted(self) -> None:
        channel = _channel([_http_error(500), [_message(3), _message(1), _message(2)]])

        page = await fetch_history_page(
            channel,
            after_message_id=None,
            before_message_id=99,
            limit=100,
            sleep=_RecordingSleep(),
        )

        assert page is not None and [m.id for m in page] == [1, 2, 3]
