"""Tests for aura.db.extraction_queue: the durable batch collector.

The headline requirement from the phase brief is durability -- "the collected,
not-yet-processed batch must not be lost across a container restart" -- and the
brief is explicit that it wants this demonstrated rather than asserted. So
TestRestartDurability uses a real file-backed database, closes the connection
the way a dying process would, opens a fresh one, and checks what survived. An
in-memory database cannot show this: closing it destroys the data regardless of
whether the design was durable.

A real database throughout, never a live gateway connection.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path

import aiosqlite
import pytest

from aura.db.extraction_queue import (
    MAX_BATCH_WINDOW_SECONDS,
    clear_batch,
    count_queued,
    due_channels,
    enqueue_message,
    read_batch,
    remove_queued_message,
)
from aura.db.repository import init_schema

GUILD_A = 100000000000000001
CHANNEL_A = 500000000000000005
CHANNEL_B = 600000000000000006

NOW = datetime(2026, 7, 30, 12, 0, 0, tzinfo=timezone.utc)
WINDOW = 300.0


@pytest.fixture
async def conn():
    connection = await aiosqlite.connect(":memory:")
    await init_schema(connection)
    yield connection
    await connection.close()


async def _enqueue(
    conn: aiosqlite.Connection,
    *,
    channel_id: int = CHANNEL_A,
    message_id: int = 1,
    content: str = "Maintenance today at 14:00 UTC.",
    now: datetime = NOW,
) -> bool:
    return await enqueue_message(
        conn,
        guild_id=GUILD_A,
        channel_id=channel_id,
        message_id=message_id,
        channel_name="announcements",
        content=content,
        message_created_at=now,
        now=now,
    )


class TestEnqueue:
    async def test_a_message_is_queued_and_readable(self, conn: aiosqlite.Connection) -> None:
        assert await _enqueue(conn) is True
        batch = await read_batch(conn, channel_id=CHANNEL_A, limit=10)
        assert len(batch) == 1
        assert batch[0].content == "Maintenance today at 14:00 UTC."
        assert batch[0].channel_name == "announcements"
        assert batch[0].guild_id == GUILD_A

    async def test_a_redelivered_message_does_not_queue_twice(
        self, conn: aiosqlite.Connection
    ) -> None:
        # Discord replays events after a resumed session; a duplicate must not
        # end up in the batch twice and be paid for twice.
        assert await _enqueue(conn) is True
        assert await _enqueue(conn) is False
        assert await count_queued(conn, channel_id=CHANNEL_A) == 1

    async def test_concurrent_redeliveries_of_one_message_queue_it_once(
        self, conn: aiosqlite.Connection
    ) -> None:
        results = await asyncio.gather(*(_enqueue(conn) for _ in range(20)))
        assert results.count(True) == 1
        assert await count_queued(conn, channel_id=CHANNEL_A) == 1

    async def test_a_naive_timestamp_is_rejected_rather_than_assumed_to_be_utc(
        self, conn: aiosqlite.Connection
    ) -> None:
        # Assuming UTC is wrong by whole hours on a non-UTC host, which here
        # means flushing a batch early or late.
        with pytest.raises(ValueError):
            await enqueue_message(
                conn,
                guild_id=GUILD_A,
                channel_id=CHANNEL_A,
                message_id=1,
                channel_name="general",
                content="text",
                message_created_at=datetime(2026, 7, 30, 12, 0),
                now=NOW,
            )
        with pytest.raises(ValueError):
            await enqueue_message(
                conn,
                guild_id=GUILD_A,
                channel_id=CHANNEL_A,
                message_id=1,
                channel_name="general",
                content="text",
                message_created_at=NOW,
                now=datetime(2026, 7, 30, 12, 0),
            )

    async def test_unicode_and_oversized_content_survive_a_round_trip(
        self, conn: aiosqlite.Connection
    ) -> None:
        # Nine locales, emoji, RTL, zero-width joiners, and a message at
        # Discord's own upper end. The queue stores text verbatim; truncation
        # is the distiller's prompt-building concern, not this layer's.
        hostile = (
            "\U0001F1F0\U0001F1F7 \uacf5\uc9c0\u200d mixed \u200f\u0639\u0631\u0628\u064a\u200f "
            "\x00 embedded NUL \U0001F600 " + "\u00fc" * 5000
        )
        await _enqueue(conn, content=hostile)
        batch = await read_batch(conn, channel_id=CHANNEL_A, limit=10)
        assert batch[0].content == hostile


class TestWithdrawal:
    async def test_removing_a_queued_message_reports_that_it_removed_one(
        self, conn: aiosqlite.Connection
    ) -> None:
        await _enqueue(conn)
        assert await remove_queued_message(conn, channel_id=CHANNEL_A, message_id=1) is True
        assert await count_queued(conn, channel_id=CHANNEL_A) == 0

    async def test_removing_a_message_that_was_never_queued_is_a_clean_no_op(
        self, conn: aiosqlite.Connection
    ) -> None:
        # The overwhelmingly common case: an edit to any of the ~99% of
        # messages that never cleared the fact-worthiness filter.
        assert await remove_queued_message(conn, channel_id=CHANNEL_A, message_id=42) is False

    async def test_removing_twice_is_idempotent(self, conn: aiosqlite.Connection) -> None:
        # on_message_delete and on_raw_message_delete both fire for one cached
        # deletion, so this happens on every ordinary delete.
        await _enqueue(conn)
        assert await remove_queued_message(conn, channel_id=CHANNEL_A, message_id=1) is True
        assert await remove_queued_message(conn, channel_id=CHANNEL_A, message_id=1) is False

    async def test_withdrawal_only_touches_its_own_message(
        self, conn: aiosqlite.Connection
    ) -> None:
        await _enqueue(conn, message_id=1)
        await _enqueue(conn, message_id=2)
        await remove_queued_message(conn, channel_id=CHANNEL_A, message_id=1)
        remaining = await read_batch(conn, channel_id=CHANNEL_A, limit=10)
        assert [message.message_id for message in remaining] == [2]

    async def test_a_message_id_reused_in_another_channel_is_unaffected(
        self, conn: aiosqlite.Connection
    ) -> None:
        await _enqueue(conn, channel_id=CHANNEL_A, message_id=1)
        await _enqueue(conn, channel_id=CHANNEL_B, message_id=1)
        await remove_queued_message(conn, channel_id=CHANNEL_A, message_id=1)
        assert await count_queued(conn, channel_id=CHANNEL_B) == 1


class TestDueChannels:
    async def test_a_fresh_batch_is_not_yet_due(self, conn: aiosqlite.Connection) -> None:
        await _enqueue(conn, now=NOW)
        assert await due_channels(conn, window_seconds=WINDOW, now=NOW) == []

    async def test_a_batch_becomes_due_once_its_oldest_message_ages_out(
        self, conn: aiosqlite.Connection
    ) -> None:
        await _enqueue(conn, now=NOW)
        later = NOW + timedelta(seconds=WINDOW)
        assert await due_channels(conn, window_seconds=WINDOW, now=later) == [CHANNEL_A]

    async def test_the_window_is_not_a_debounce(self, conn: aiosqlite.Connection) -> None:
        # A steady trickle of new candidates must not postpone the batch
        # forever: the OLDEST message decides, so every message has a bounded
        # wait regardless of what arrives after it.
        await _enqueue(conn, message_id=1, now=NOW)
        await _enqueue(conn, message_id=2, now=NOW + timedelta(seconds=240))
        await _enqueue(conn, message_id=3, now=NOW + timedelta(seconds=290))

        due = await due_channels(
            conn, window_seconds=WINDOW, now=NOW + timedelta(seconds=WINDOW)
        )
        assert due == [CHANNEL_A]

    async def test_channels_become_due_independently(
        self, conn: aiosqlite.Connection
    ) -> None:
        await _enqueue(conn, channel_id=CHANNEL_A, message_id=1, now=NOW)
        await _enqueue(
            conn, channel_id=CHANNEL_B, message_id=2, now=NOW + timedelta(seconds=200)
        )
        due = await due_channels(
            conn, window_seconds=WINDOW, now=NOW + timedelta(seconds=WINDOW)
        )
        assert due == [CHANNEL_A]

    async def test_an_empty_queue_has_nothing_due(self, conn: aiosqlite.Connection) -> None:
        assert await due_channels(conn, window_seconds=WINDOW, now=NOW) == []

    async def test_a_zero_window_makes_everything_immediately_due(
        self, conn: aiosqlite.Connection
    ) -> None:
        # Used by the tests below, and a legitimate (if pointless) production
        # setting: it turns batching off rather than breaking.
        await _enqueue(conn, now=NOW)
        assert await due_channels(conn, window_seconds=0.0, now=NOW) == [CHANNEL_A]

    async def test_an_absurd_window_is_refused_where_an_operator_can_see_it(
        self, conn: aiosqlite.Connection
    ) -> None:
        # The cutoff is `now - timedelta(seconds=...)`, which raises rather
        # than saturating, so an unbounded value would throw on every sweep.
        with pytest.raises(ValueError):
            await due_channels(
                conn, window_seconds=MAX_BATCH_WINDOW_SECONDS + 1, now=NOW
            )
        with pytest.raises(ValueError):
            await due_channels(conn, window_seconds=-1.0, now=NOW)


class TestBatchReadAndClear:
    async def test_a_batch_reads_oldest_first(self, conn: aiosqlite.Connection) -> None:
        for offset, message_id in enumerate((3, 1, 2)):
            await _enqueue(
                conn, message_id=message_id, now=NOW + timedelta(seconds=offset)
            )
        batch = await read_batch(conn, channel_id=CHANNEL_A, limit=10)
        assert [message.message_id for message in batch] == [3, 1, 2]

    async def test_reading_does_not_remove(self, conn: aiosqlite.Connection) -> None:
        # Load-bearing: the rows are cleared only after their candidates are
        # staged, so a crash in between leaves the batch to be retried rather
        # than losing it.
        await _enqueue(conn)
        await read_batch(conn, channel_id=CHANNEL_A, limit=10)
        assert await count_queued(conn, channel_id=CHANNEL_A) == 1

    async def test_the_limit_caps_one_batch_and_leaves_the_rest_queued(
        self, conn: aiosqlite.Connection
    ) -> None:
        for message_id in range(1, 26):
            await _enqueue(
                conn, message_id=message_id, now=NOW + timedelta(seconds=message_id)
            )
        batch = await read_batch(conn, channel_id=CHANNEL_A, limit=20)
        assert len(batch) == 20
        assert await count_queued(conn, channel_id=CHANNEL_A) == 25

    async def test_clearing_removes_only_the_messages_the_batch_consumed(
        self, conn: aiosqlite.Connection
    ) -> None:
        # The plausible simplification here -- "delete everything in this
        # channel" -- would silently drop messages that arrived while the batch
        # was being distilled. This is the test that rules it out.
        await _enqueue(conn, message_id=1, now=NOW)
        await _enqueue(conn, message_id=2, now=NOW)
        batch = await read_batch(conn, channel_id=CHANNEL_A, limit=10)

        await _enqueue(conn, message_id=3, now=NOW + timedelta(seconds=1))

        removed = await clear_batch(
            conn,
            channel_id=CHANNEL_A,
            message_ids=[message.message_id for message in batch],
        )
        assert removed == 2

        remaining = await read_batch(conn, channel_id=CHANNEL_A, limit=10)
        assert [message.message_id for message in remaining] == [3]

    async def test_clearing_nothing_is_a_no_op(self, conn: aiosqlite.Connection) -> None:
        await _enqueue(conn)
        assert await clear_batch(conn, channel_id=CHANNEL_A, message_ids=[]) == 0
        assert await count_queued(conn, channel_id=CHANNEL_A) == 1

    async def test_clearing_is_scoped_to_its_channel(
        self, conn: aiosqlite.Connection
    ) -> None:
        await _enqueue(conn, channel_id=CHANNEL_A, message_id=1)
        await _enqueue(conn, channel_id=CHANNEL_B, message_id=1)
        await clear_batch(conn, channel_id=CHANNEL_A, message_ids=[1])
        assert await count_queued(conn, channel_id=CHANNEL_B) == 1

    async def test_a_negative_limit_is_rejected(self, conn: aiosqlite.Connection) -> None:
        with pytest.raises(ValueError):
            await read_batch(conn, channel_id=CHANNEL_A, limit=-1)


class TestRestartDurability:
    """The phase brief's explicit ask: simulate a restart mid-window, verify nothing is lost.

    A real file on disk and a genuinely new connection, because that is the only
    way this can be demonstrated rather than asserted -- an in-memory database
    would lose the data on close whether or not the design was durable, and a
    reused connection would prove nothing about a new process.
    """

    async def test_a_half_filled_batch_survives_a_restart(self, tmp_path: Path) -> None:
        database = tmp_path / "aura.db"

        # --- process 1: messages arrive, the window has not closed yet -------
        first = await aiosqlite.connect(database)
        await init_schema(first)
        for message_id in (1, 2, 3):
            await _enqueue(first, message_id=message_id, now=NOW)
        assert await due_channels(first, window_seconds=WINDOW, now=NOW) == []
        # The container dies here: no flush, no cleanup, no shutdown hook.
        await first.close()

        # --- process 2: a cold start with no recovery step -------------------
        second = await aiosqlite.connect(database)
        await init_schema(second)
        try:
            assert await count_queued(second) == 3

            # The window closed while the process was down. The new process
            # picks the batch up on its first sweep, without being told a
            # restart happened.
            later = NOW + timedelta(seconds=WINDOW)
            assert await due_channels(second, window_seconds=WINDOW, now=later) == [CHANNEL_A]

            batch = await read_batch(second, channel_id=CHANNEL_A, limit=20)
            assert [message.message_id for message in batch] == [1, 2, 3]
            # And the content itself survived, not just the row count -- the
            # batch has to be distillable, not merely present.
            assert all(message.content for message in batch)
            assert all(message.channel_name == "announcements" for message in batch)
        finally:
            await second.close()

    async def test_a_restart_between_reading_a_batch_and_clearing_it_loses_nothing(
        self, tmp_path: Path
    ) -> None:
        # The crash window that actually matters: the batch was read and
        # possibly distilled, but the rows were not cleared. It must be
        # retryable, which means still present.
        database = tmp_path / "aura.db"

        first = await aiosqlite.connect(database)
        await init_schema(first)
        await _enqueue(first, message_id=1, now=NOW)
        batch = await read_batch(first, channel_id=CHANNEL_A, limit=20)
        assert len(batch) == 1
        await first.close()  # crash before clear_batch

        second = await aiosqlite.connect(database)
        await init_schema(second)
        try:
            assert await count_queued(second, channel_id=CHANNEL_A) == 1
            retry = await read_batch(second, channel_id=CHANNEL_A, limit=20)
            assert [message.message_id for message in retry] == [1]
        finally:
            await second.close()

    async def test_a_withdrawal_before_the_restart_stays_withdrawn(
        self, tmp_path: Path
    ) -> None:
        # The opposite direction: durability must not resurrect a message the
        # author deleted.
        database = tmp_path / "aura.db"

        first = await aiosqlite.connect(database)
        await init_schema(first)
        await _enqueue(first, message_id=1, now=NOW)
        await _enqueue(first, message_id=2, now=NOW)
        await remove_queued_message(first, channel_id=CHANNEL_A, message_id=1)
        await first.close()

        second = await aiosqlite.connect(database)
        await init_schema(second)
        try:
            batch = await read_batch(second, channel_id=CHANNEL_A, limit=20)
            assert [message.message_id for message in batch] == [2]
        finally:
            await second.close()
