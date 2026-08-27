"""Tests for aura.backfill.worker: the cursor, the order, the boundary, the cap.

End to end over a real in-memory database, the real embedding model and a fake
channel that paginates a fixed corpus exactly the way Discord does. Only the two
paid calls are mocked (conftest's autouse guard fails the run if anything reaches
a real one); everything the sub-phase actually builds -- which messages get
picked, in what order, what the cursor does, what a restart costs, what the caps
do -- runs for real.

The fake channel is the one stand-in that matters, and it is deliberately
adversarial by default in one specific way: `shuffle_pages` makes it return every
page in scrambled order, which is what the ordering deliverable is tested
against. A run over a shuffled channel and a run over a sorted one must produce
byte-identical results, and several tests below assert exactly that.

THE FOUR "ATTACK IT" ITEMS FROM THE PHASE BRIEF each have their own class:
TestRestartMidRun, TestSupersessionChainOverHistory, TestCapIndependence and
TestLargeHistory.
"""
from __future__ import annotations

import asyncio
import random
import time
from datetime import datetime, timedelta, timezone
from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch

import aiosqlite
import discord
import pytest
from fastembed import TextEmbedding

from aura.backfill.history import ChannelUnreadable, is_strictly_increasing
from aura.backfill.worker import advance_due_backfills, run_backfill_worker
from aura.config import Settings
from aura.db.backfill_runs import (
    BackfillState,
    advance_cursor,
    get_active_run,
    get_recent_runs,
    set_run_state,
    start_backfill_run,
)
from aura.db.backfill_state import count_backfill_calls_on
from aura.db.connection import utc_day
from aura.db.extraction_queue import count_queued, enqueue_message
from aura.db.extraction_state import count_extraction_calls_on
from aura.db.models import FactStatus
from aura.db.pending_facts import (
    FactCategory,
    SupersessionRelationship,
    confirm_pending_fact,
    get_pending_facts,
)
from aura.db.repository import get_active_facts, get_fact_by_id, init_schema
from aura.db.supersession_state import count_supersession_calls_on
from aura.embeddings import embed_text
from aura.extraction.distiller import DistilledFact
from aura.extraction.supersession import RelationshipJudgement
from aura.facts_service import add_fact

GUILD_A = 100000000000000001
GUILD_B = 200000000000000002
CHANNEL_A = 300000000000000003
CHANNEL_B = 400000000000000004
MODERATOR = 4242

NOW = datetime(2026, 8, 26, 12, 0, 0, tzinfo=timezone.utc)
# The run's upper bound. Every corpus message below sits under it, so the bound
# itself is only exercised by the tests that deliberately put a message above it.
FIRST_ID = 800000000000000000
UNTIL_ID = FIRST_ID + 1_000_000
EPOCH = datetime(2025, 1, 1, tzinfo=timezone.utc)


@pytest.fixture
async def conn():
    connection = await aiosqlite.connect(":memory:")
    await init_schema(connection)
    yield connection
    await connection.close()


def _settings(**overrides) -> Settings:
    values = {
        "discord_token": "fake-token",
        "llm_api_key": "test-key",
        "extraction_model": "openrouter/anthropic/claude-haiku-4.5",
        "supersession_model": "openrouter/anthropic/claude-haiku-4.5",
        "extraction_batch_max_messages": 5,
        "extraction_fact_worthiness_threshold": -0.02,
        "backfill_daily_cap": 30,
        # No waiting anywhere in the suite: the pause is a live-traffic courtesy
        # and its own behaviour is asserted separately, by reading the value.
        "backfill_page_pause_seconds": 0.0,
        "backfill_check_interval_seconds": 1.0,
        **overrides,
    }
    return Settings(_env_file=None, **values)  # type: ignore[call-arg]


def _message(
    message_id: int,
    *,
    content: str = "The server is down for maintenance today at 14:00 UTC.",
    channel_id: int = CHANNEL_A,
    guild_id: int | None = GUILD_A,
    bot: bool = False,
    webhook_id: int | None = None,
    message_type: discord.MessageType = discord.MessageType.default,
) -> MagicMock:
    """A message stub carrying exactly what should_extract and the batch read."""
    message = MagicMock(spec=discord.Message)
    message.id = message_id
    message.content = content
    if guild_id is None:
        message.guild = None
    else:
        message.guild = MagicMock()
        message.guild.id = guild_id
    message.channel = MagicMock()
    message.channel.id = channel_id
    message.channel.name = "announcements"
    message.author = MagicMock()
    message.author.bot = bot
    message.webhook_id = webhook_id
    message.interaction_metadata = None
    message.type = message_type
    # Mirrors the real thing: discord.py derives created_at FROM the snowflake,
    # so the two can never disagree, and neither can they here.
    message.created_at = EPOCH + timedelta(seconds=message_id - FIRST_ID)
    return message


class FakeChannel:
    """A text channel paginating a fixed corpus the way Discord's API does.

    Honours `after`, `before`, `limit` and `oldest_first` against a sorted
    corpus, records every call, and can be told to scramble each page before
    returning it -- which is what the ordering deliverable is tested against.
    """

    def __init__(
        self,
        corpus: list[MagicMock],
        *,
        channel_id: int = CHANNEL_A,
        shuffle_pages: bool = False,
        seed: int = 20260826,
        raises: Exception | None = None,
    ) -> None:
        self.id = channel_id
        # Deliberately loose: two tests replace these with None to exercise the
        # channel-name fallback and the cross-guild refusal, both of which are
        # states a real (partial or reused) channel object can genuinely be in.
        self.name: str | None = "announcements"
        self.guild: MagicMock | None = MagicMock()
        self.guild.id = GUILD_A
        self._corpus = sorted(corpus, key=lambda message: message.id)
        self._shuffle = shuffle_pages
        self._rng = random.Random(seed)
        self.raises = raises
        self.calls: list[dict[str, object]] = []

    def history(self, **kwargs: object):
        self.calls.append(kwargs)
        if self.raises is not None:
            return _RaisingIterator(self.raises)

        after = kwargs.get("after")
        before = kwargs.get("before")
        limit = cast("int", kwargs.get("limit") or 100)
        after_id = after.id if isinstance(after, discord.Object) else 0
        before_id = (
            before.id if isinstance(before, discord.Object) else 1 << 63
        )

        page = [
            message
            for message in self._corpus
            if after_id < message.id < before_id
        ][: int(limit)]
        if self._shuffle:
            page = list(page)
            self._rng.shuffle(page)
        return _ListIterator(page)

    @property
    def page_requests(self) -> int:
        return len(self.calls)


class _ListIterator:
    def __init__(self, page: list[MagicMock]) -> None:
        self._page = page
        self._index = 0

    def __aiter__(self) -> "_ListIterator":
        return self

    async def __anext__(self) -> MagicMock:
        if self._index >= len(self._page):
            raise StopAsyncIteration
        message = self._page[self._index]
        self._index += 1
        return message


class _RaisingIterator:
    def __init__(self, error: Exception) -> None:
        self._error = error

    def __aiter__(self) -> "_RaisingIterator":
        return self

    async def __anext__(self):
        raise self._error


class FakeGateway:
    """A BackfillGateway resolving from a dict, with an explicit failure mode."""

    def __init__(self, channels: dict[int, FakeChannel] | None = None) -> None:
        self.channels = channels or {}
        self.unreadable: set[int] = set()

    def add(self, channel: FakeChannel) -> FakeChannel:
        self.channels[channel.id] = channel
        return channel

    async def resolve_channel(self, channel_id: int) -> discord.TextChannel:
        if channel_id in self.unreadable or channel_id not in self.channels:
            raise ChannelUnreadable(f"channel {channel_id} is unreadable")
        # A real TextChannel cannot be constructed without a gateway connection,
        # which is the whole point of the seam being a Protocol -- the same cast,
        # for the same reason, test_digest_scheduler.py's fake gateway makes.
        return cast("discord.TextChannel", self.channels[channel_id])


class RecordingDistiller:
    """Stands in for the paid distillation call, recording every batch it saw.

    Turns each candidate message into exactly one distilled fact by default, so
    "which messages reached the model, in what order" is directly readable off
    `self.batches` -- which is the property most of this file is about.
    """

    def __init__(self, *, transform=None, returns=None) -> None:
        self.batches: list[list[int]] = []
        self.contents: list[str] = []
        self.channel_names: list[str] = []
        self.calls = 0
        self._transform = transform
        self._returns = returns

    async def __call__(self, candidates, *, channel_name: str, model: str):
        self.calls += 1
        self.batches.append([queued.message_id for queued in candidates])
        self.contents.extend(queued.content for queued in candidates)
        self.channel_names.append(channel_name)
        if self._returns is not None:
            return self._returns
        return [
            DistilledFact(
                message_id=queued.message_id,
                content=(
                    self._transform(queued)
                    if self._transform is not None
                    else f"Distilled: {queued.content}"
                ),
                category=FactCategory.ANNOUNCEMENT,
            )
            for queued in candidates
        ]

    @property
    def seen_message_ids(self) -> list[int]:
        """Every message id handed to the model, in the order it was handed over."""
        return [message_id for batch in self.batches for message_id in batch]


def _detector(score: float = 1.0) -> MagicMock:
    """A fact-worthiness detector returning a controlled score.

    Mocked for the same reason tests/test_extraction_pipeline.py mocks it: the
    filter's calibration is reports/phase-3a-1b.txt's subject, and what these
    tests isolate is the worker's own threshold comparison and ordering, not the
    exemplar geometry behind the number.
    """
    detector = MagicMock()
    detector.question_likeness = AsyncMock(return_value=score)
    return detector


def _scored_detector(scores: dict[str, float], default: float = -1.0) -> MagicMock:
    """A detector that scores by message content, for mixed fact-worthy corpora."""
    detector = MagicMock()

    async def score(content: str) -> float:
        return scores.get(content, default)

    detector.question_likeness = AsyncMock(side_effect=score)
    return detector


async def _start(
    conn: aiosqlite.Connection,
    *,
    guild_id: int = GUILD_A,
    channel_id: int = CHANNEL_A,
    after_message_id: int | None = None,
    until_message_id: int = UNTIL_ID,
):
    return await start_backfill_run(
        conn,
        guild_id=guild_id,
        channel_id=channel_id,
        until_message_id=until_message_id,
        after_message_id=after_message_id,
        requested_by_id=MODERATOR,
        now=NOW,
    )


async def _drain(
    conn: aiosqlite.Connection,
    model: TextEmbedding,
    gateway: FakeGateway,
    detector,
    *,
    settings: Settings,
    now: datetime = NOW,
    max_ticks: int = 200,
) -> int:
    """Run ticks until nothing advances. Returns how many ticks did work.

    Bounded so a bug that never terminates fails as a test rather than as a
    hung suite.
    """
    ticks = 0
    for _ in range(max_ticks):
        advanced = await advance_due_backfills(
            conn, model, gateway, detector, settings=settings, now=now
        )
        if not advanced:
            return ticks
        ticks += 1
    raise AssertionError(f"backfill did not settle within {max_ticks} ticks")


class TestChronologicalOrder:
    """Deliverable 5: strict old-to-new order, enforced rather than assumed."""

    async def test_a_whole_run_reaches_the_model_oldest_first(
        self, conn, embedding_model
    ) -> None:
        corpus = [_message(FIRST_ID + index) for index in range(12)]
        gateway = FakeGateway()
        gateway.add(FakeChannel(corpus))
        distiller = RecordingDistiller()
        await _start(conn)

        with patch("aura.backfill.worker.distill_facts", distiller):
            await _drain(conn, embedding_model, gateway, _detector(), settings=_settings())

        assert distiller.seen_message_ids == [message.id for message in corpus]

    async def test_deliberately_unsorted_pages_still_reach_the_model_in_order(
        self, conn, embedding_model
    ) -> None:
        """The deliverable's own attack: pages returned scrambled, every time."""
        corpus = [_message(FIRST_ID + index) for index in range(12)]
        gateway = FakeGateway()
        gateway.add(FakeChannel(corpus, shuffle_pages=True))
        distiller = RecordingDistiller()
        await _start(conn)

        with patch("aura.backfill.worker.distill_facts", distiller):
            await _drain(conn, embedding_model, gateway, _detector(), settings=_settings())

        assert distiller.seen_message_ids == [message.id for message in corpus]
        for batch in distiller.batches:
            assert batch == sorted(batch), "a single batch reached the model out of order"

    async def test_a_shuffled_channel_produces_the_same_result_as_a_sorted_one(
        self, conn, embedding_model
    ) -> None:
        """If order enforcement worked, the two runs are indistinguishable."""
        corpus = [_message(FIRST_ID + index) for index in range(17)]

        results = []
        for shuffle in (False, True):
            connection = await aiosqlite.connect(":memory:")
            await init_schema(connection)
            gateway = FakeGateway()
            gateway.add(FakeChannel(corpus, shuffle_pages=shuffle))
            distiller = RecordingDistiller()
            await _start(connection)
            with patch("aura.backfill.worker.distill_facts", distiller):
                await _drain(
                    connection, embedding_model, gateway, _detector(), settings=_settings()
                )
            staged = await get_pending_facts(connection, guild_id=GUILD_A, limit=100)
            results.append(
                (distiller.batches, [(fact.message_id, fact.content) for fact in staged])
            )
            await connection.close()

        assert results[0] == results[1]

    async def test_the_cursor_only_ever_moves_forward_across_a_whole_run(
        self, conn, embedding_model
    ) -> None:
        corpus = [_message(FIRST_ID + index) for index in range(23)]
        gateway = FakeGateway()
        gateway.add(FakeChannel(corpus, shuffle_pages=True))
        settings = _settings()
        await _start(conn)

        positions: list[int] = []
        with patch("aura.backfill.worker.distill_facts", RecordingDistiller()):
            for _ in range(50):
                advanced = await advance_due_backfills(
                    conn, embedding_model, gateway, _detector(), settings=settings, now=NOW
                )
                run = (await get_recent_runs(conn, guild_id=GUILD_A, limit=1))[0]
                if run.cursor_message_id is not None:
                    positions.append(run.cursor_message_id)
                if not advanced:
                    break

        assert positions == sorted(positions)
        assert len(set(positions)) == len(positions) or positions[-1] == positions[-2]

    async def test_pages_are_requested_with_oldest_first(self, conn, embedding_model) -> None:
        corpus = [_message(FIRST_ID + index) for index in range(3)]
        gateway = FakeGateway()
        channel = gateway.add(FakeChannel(corpus))
        await _start(conn)

        with patch("aura.backfill.worker.distill_facts", RecordingDistiller()):
            await advance_due_backfills(
                conn, embedding_model, gateway, _detector(), settings=_settings(), now=NOW
            )

        assert all(call["oldest_first"] is True for call in channel.calls)


class TestRestartMidRun:
    """Attack It #1: 'exakt keine Nachricht doppelt, keine übersprungen'."""

    async def test_a_restart_mid_run_neither_repeats_nor_skips_a_candidate(
        self, tmp_path, embedding_model
    ) -> None:
        path = tmp_path / "aura.db"
        corpus = [_message(FIRST_ID + index) for index in range(37)]
        settings = _settings()

        first = await aiosqlite.connect(path)
        await init_schema(first)
        await _start(first)
        gateway_before = FakeGateway()
        gateway_before.add(FakeChannel(corpus))
        distiller_before = RecordingDistiller()
        with patch("aura.backfill.worker.distill_facts", distiller_before):
            # Exactly three batches, then the process dies.
            for _ in range(3):
                await advance_due_backfills(
                    first, embedding_model, gateway_before, _detector(), settings=settings, now=NOW
                )
        seen_before = list(distiller_before.seen_message_ids)
        staged_before = {
            fact.message_id
            for fact in await get_pending_facts(first, guild_id=GUILD_A, limit=100)
        }
        await first.close()  # how a dying container ends

        assert seen_before, "the first process did no work, so the test proves nothing"
        assert len(seen_before) < len(corpus), "the first process finished the whole corpus"

        second = await aiosqlite.connect(path)
        await init_schema(second)
        try:
            gateway_after = FakeGateway()
            gateway_after.add(FakeChannel(corpus))
            distiller_after = RecordingDistiller()
            with patch("aura.backfill.worker.distill_facts", distiller_after):
                await _drain(
                    second, embedding_model, gateway_after, _detector(), settings=settings
                )
            seen_after = distiller_after.seen_message_ids

            # NOTHING SKIPPED: every message in the corpus reached the model.
            assert sorted(seen_before + seen_after) == [message.id for message in corpus]
            # NOTHING REPEATED: the two halves do not overlap at all.
            assert set(seen_before).isdisjoint(seen_after)
            # And the second half continued in order from where the first stopped.
            assert min(seen_after) > max(seen_before)
            assert seen_after == sorted(seen_after)

            staged = await get_pending_facts(second, guild_id=GUILD_A, limit=100)
            assert len(staged) == len(corpus)
            assert len({fact.message_id for fact in staged}) == len(corpus)
            assert staged_before <= {fact.message_id for fact in staged}
        finally:
            await second.close()

    async def test_a_crash_after_staging_but_before_the_cursor_costs_nothing_at_all(
        self, conn, embedding_model
    ) -> None:
        """The measured behaviour, which is better than the design promised.

        The cursor is advanced LAST on purpose, so a crash in that window
        re-fetches the page. The design's promise was only that the repeat costs
        a slot and produces no duplicate candidate (pending_facts' UNIQUE
        constraint). What actually happens is stronger, and it falls out of the
        live-path boundary check doing double duty: staged_message_ids finds the
        candidates the crashed tick had already written, so the retry SKIPS
        those messages entirely and carries on with the next ones. No duplicate,
        no re-distillation of what was already distilled, and no message
        skipped -- see reports/phase-3b.txt.
        """
        corpus = [_message(FIRST_ID + index) for index in range(6)]
        gateway = FakeGateway()
        gateway.add(FakeChannel(corpus))
        # Two per batch, so the first tick leaves the run mid-history rather
        # than finishing it -- a crash after the last batch is a different (and
        # less interesting) case.
        settings = _settings(extraction_batch_max_messages=2)
        run = await _start(conn)
        distiller = RecordingDistiller()

        with patch("aura.backfill.worker.distill_facts", distiller):
            await advance_due_backfills(
                conn, embedding_model, gateway, _detector(), settings=settings, now=NOW
            )
            # "Crash" between the staging and the cursor advance.
            await conn.execute(
                "UPDATE backfill_runs SET cursor_message_id = NULL, cursor_message_at = NULL "
                "WHERE id = ?",
                (run.id,),
            )
            await conn.commit()

            await advance_due_backfills(
                conn, embedding_model, gateway, _detector(), settings=settings, now=NOW
            )

        assert distiller.batches == [
            [FIRST_ID + 0, FIRST_ID + 1],
            [FIRST_ID + 2, FIRST_ID + 3],
        ], "the retry re-distilled messages it had already staged candidates for"
        staged = await get_pending_facts(conn, guild_id=GUILD_A, limit=100)
        assert len(staged) == 4
        assert len({fact.message_id for fact in staged}) == 4

    async def test_a_crash_before_staging_re_does_the_identical_batch(
        self, conn, embedding_model
    ) -> None:
        """The other half of the same window, where nothing was written yet.

        Here the retry genuinely re-does the batch, because there is no record
        anywhere that it ever ran. That costs one extra slot from the daily cap
        -- deliberately never refunded, the same conservative direction every
        ledger in this project takes -- and produces the same candidates once.
        """
        corpus = [_message(FIRST_ID + index) for index in range(6)]
        gateway = FakeGateway()
        gateway.add(FakeChannel(corpus))
        settings = _settings(extraction_batch_max_messages=2)
        distiller = RecordingDistiller()
        await _start(conn)

        crashing = AsyncMock(side_effect=RuntimeError("died before staging"))
        with patch("aura.backfill.worker.stage_distilled_candidates", crashing):
            with patch("aura.backfill.worker.distill_facts", distiller):
                await advance_due_backfills(
                    conn, embedding_model, gateway, _detector(), settings=settings, now=NOW
                )
        assert await get_pending_facts(conn, guild_id=GUILD_A, limit=10) == []

        with patch("aura.backfill.worker.distill_facts", distiller):
            await advance_due_backfills(
                conn, embedding_model, gateway, _detector(), settings=settings, now=NOW
            )

        assert distiller.batches == [
            [FIRST_ID + 0, FIRST_ID + 1],
            [FIRST_ID + 0, FIRST_ID + 1],
        ], "the retry did not re-do the batch the crash lost"
        staged = await get_pending_facts(conn, guild_id=GUILD_A, limit=100)
        assert len(staged) == 2
        # The one accepted cost: the crashed attempt's slot is not refunded.
        assert await count_backfill_calls_on(conn, guild_id=GUILD_A, day=utc_day(NOW)) == 2

    async def test_a_cancel_while_a_batch_is_in_flight_does_not_move_the_cursor(
        self, conn, embedding_model
    ) -> None:
        corpus = [_message(FIRST_ID + index) for index in range(4)]
        gateway = FakeGateway()
        gateway.add(FakeChannel(corpus))
        run = await _start(conn)

        async def cancel_mid_call(candidates, *, channel_name: str, model: str):
            await set_run_state(
                conn,
                run_id=run.id,
                state=BackfillState.CANCELLED,
                now=NOW,
                from_states=(BackfillState.RUNNING,),
            )
            return [
                DistilledFact(
                    message_id=queued.message_id,
                    content=f"Distilled: {queued.content}",
                    category=FactCategory.ANNOUNCEMENT,
                )
                for queued in candidates
            ]

        with patch("aura.backfill.worker.distill_facts", cancel_mid_call):
            await advance_due_backfills(
                conn, embedding_model, gateway, _detector(), settings=_settings(), now=NOW
            )

        after = (await get_recent_runs(conn, guild_id=GUILD_A, limit=1))[0]
        assert after.state is BackfillState.CANCELLED
        assert after.cursor_message_id is None, "a cancelled run's cursor moved anyway"
        # The candidates it had already staged are kept: they are proposals the
        # moderator asked for, and cancelling means "read no MORE", not "undo".
        assert len(await get_pending_facts(conn, guild_id=GUILD_A, limit=100)) == 4

    async def test_a_paused_run_is_not_advanced_and_resumes_exactly_where_it_stopped(
        self, conn, embedding_model
    ) -> None:
        corpus = [_message(FIRST_ID + index) for index in range(16)]
        gateway = FakeGateway()
        gateway.add(FakeChannel(corpus))
        settings = _settings()
        run = await _start(conn)
        distiller = RecordingDistiller()

        with patch("aura.backfill.worker.distill_facts", distiller):
            await advance_due_backfills(
                conn, embedding_model, gateway, _detector(), settings=settings, now=NOW
            )
            await set_run_state(
                conn,
                run_id=run.id,
                state=BackfillState.PAUSED,
                now=NOW,
                from_states=(BackfillState.RUNNING,),
            )
            calls_while_paused_before = distiller.calls
            for _ in range(3):
                await advance_due_backfills(
                    conn, embedding_model, gateway, _detector(), settings=settings, now=NOW
                )
            assert distiller.calls == calls_while_paused_before, "a paused run advanced"

            seen_before_pause = list(distiller.seen_message_ids)
            await set_run_state(
                conn,
                run_id=run.id,
                state=BackfillState.RUNNING,
                now=NOW,
                from_states=(BackfillState.PAUSED,),
            )
            await _drain(conn, embedding_model, gateway, _detector(), settings=settings)

        seen_after = distiller.seen_message_ids[len(seen_before_pause) :]
        assert set(seen_before_pause).isdisjoint(seen_after)
        assert sorted(seen_before_pause + seen_after) == [m.id for m in corpus]


class TestLiveExtractionBoundary:
    """Deliverable 6: no message is ever processed by both paths."""

    async def test_messages_at_or_after_the_run_start_are_never_fetched(
        self, conn, embedding_model
    ) -> None:
        past = [_message(FIRST_ID + index) for index in range(4)]
        future = [_message(UNTIL_ID + index) for index in range(3)]
        gateway = FakeGateway()
        gateway.add(FakeChannel(past + future))
        distiller = RecordingDistiller()
        await _start(conn, until_message_id=UNTIL_ID)

        with patch("aura.backfill.worker.distill_facts", distiller):
            await _drain(conn, embedding_model, gateway, _detector(), settings=_settings())

        assert distiller.seen_message_ids == [message.id for message in past]

    async def test_a_message_the_live_path_is_holding_right_now_is_skipped(
        self, conn, embedding_model
    ) -> None:
        """The four-minutes-before-start case: queued for a live batch, not backfill's."""
        corpus = [_message(FIRST_ID + index) for index in range(5)]
        contested = corpus[2]
        await enqueue_message(
            conn,
            guild_id=GUILD_A,
            channel_id=CHANNEL_A,
            message_id=contested.id,
            channel_name="announcements",
            content=contested.content,
            message_created_at=contested.created_at,
            now=NOW,
        )
        gateway = FakeGateway()
        gateway.add(FakeChannel(corpus))
        distiller = RecordingDistiller()
        await _start(conn)

        with patch("aura.backfill.worker.distill_facts", distiller):
            await _drain(conn, embedding_model, gateway, _detector(), settings=_settings())

        assert contested.id not in distiller.seen_message_ids
        assert sorted(distiller.seen_message_ids) == [
            message.id for message in corpus if message.id != contested.id
        ]
        # And backfill did not steal it out of the live queue either.
        assert await count_queued(conn, channel_id=CHANNEL_A) == 1

    async def test_a_message_the_live_path_already_staged_is_skipped(
        self, conn, embedding_model
    ) -> None:
        corpus = [_message(FIRST_ID + index) for index in range(5)]
        already = corpus[1]
        from aura.db.pending_facts import stage_pending_fact

        embedding = await embed_text(embedding_model, already.content)
        await stage_pending_fact(
            conn,
            guild_id=GUILD_A,
            channel_id=CHANNEL_A,
            message_id=already.id,
            content="Already extracted live.",
            embedding=embedding.tobytes(),
            category=FactCategory.ANNOUNCEMENT,
        )
        gateway = FakeGateway()
        gateway.add(FakeChannel(corpus))
        distiller = RecordingDistiller()
        await _start(conn)

        with patch("aura.backfill.worker.distill_facts", distiller):
            await _drain(conn, embedding_model, gateway, _detector(), settings=_settings())

        assert already.id not in distiller.seen_message_ids

    @pytest.mark.parametrize("status", ["confirmed", "discarded"])
    async def test_a_candidate_a_moderator_already_resolved_is_not_re_proposed(
        self, conn, embedding_model, status
    ) -> None:
        """The one that would actively annoy: work someone already decided on."""
        corpus = [_message(FIRST_ID + index) for index in range(4)]
        resolved = corpus[0]
        from aura.db.pending_facts import (
            discard_pending_fact,
            stage_pending_fact,
        )

        embedding = await embed_text(embedding_model, resolved.content)
        staged = await stage_pending_fact(
            conn,
            guild_id=GUILD_A,
            channel_id=CHANNEL_A,
            message_id=resolved.id,
            content="Decided on already.",
            embedding=embedding.tobytes(),
            category=FactCategory.ANNOUNCEMENT,
        )
        assert staged is not None
        if status == "confirmed":
            await confirm_pending_fact(
                conn, guild_id=GUILD_A, pending_id=staged.id, resolved_by_id=MODERATOR
            )
        else:
            await discard_pending_fact(
                conn, guild_id=GUILD_A, pending_id=staged.id, resolved_by_id=MODERATOR
            )

        gateway = FakeGateway()
        gateway.add(FakeChannel(corpus))
        distiller = RecordingDistiller()
        await _start(conn)

        with patch("aura.backfill.worker.distill_facts", distiller):
            await _drain(conn, embedding_model, gateway, _detector(), settings=_settings())

        assert resolved.id not in distiller.seen_message_ids

    async def test_backfill_never_writes_to_the_live_extraction_queue_or_its_ledger(
        self, conn, embedding_model
    ) -> None:
        corpus = [_message(FIRST_ID + index) for index in range(9)]
        gateway = FakeGateway()
        gateway.add(FakeChannel(corpus))
        distiller = RecordingDistiller()
        await _start(conn)

        with patch("aura.backfill.worker.distill_facts", distiller):
            await _drain(conn, embedding_model, gateway, _detector(), settings=_settings())

        assert distiller.calls > 0, "the run did nothing, so the check is vacuous"
        assert await count_queued(conn) == 0
        assert await count_extraction_calls_on(conn, guild_id=GUILD_A, day=utc_day(NOW)) == 0

    async def test_the_run_starts_after_an_optional_since_bound(
        self, conn, embedding_model
    ) -> None:
        corpus = [_message(FIRST_ID + index) for index in range(10)]
        gateway = FakeGateway()
        gateway.add(FakeChannel(corpus))
        distiller = RecordingDistiller()
        await _start(conn, after_message_id=corpus[5].id)

        with patch("aura.backfill.worker.distill_facts", distiller):
            await _drain(conn, embedding_model, gateway, _detector(), settings=_settings())

        assert distiller.seen_message_ids == [message.id for message in corpus[6:]]


class TestGatesAreTheLivePaths:
    async def test_bots_webhooks_and_system_messages_never_reach_the_model(
        self, conn, embedding_model
    ) -> None:
        corpus = [
            _message(FIRST_ID + 0),
            _message(FIRST_ID + 1, bot=True),
            _message(FIRST_ID + 2, webhook_id=999),
            _message(FIRST_ID + 3, message_type=discord.MessageType.pins_add),
            _message(FIRST_ID + 4, content="​​"),
            _message(FIRST_ID + 5),
        ]
        gateway = FakeGateway()
        gateway.add(FakeChannel(corpus))
        distiller = RecordingDistiller()
        await _start(conn)

        with patch("aura.backfill.worker.distill_facts", distiller):
            await _drain(conn, embedding_model, gateway, _detector(), settings=_settings())

        assert distiller.seen_message_ids == [FIRST_ID + 0, FIRST_ID + 5]

    async def test_the_fact_worthiness_threshold_is_the_same_one_live_extraction_uses(
        self, conn, embedding_model
    ) -> None:
        worthy = "The winter tournament starts Saturday at 6 PM."
        chatter = "lmaooo that clip was amazing"
        corpus = [
            _message(FIRST_ID + 0, content=worthy),
            _message(FIRST_ID + 1, content=chatter),
            _message(FIRST_ID + 2, content=worthy),
        ]
        gateway = FakeGateway()
        gateway.add(FakeChannel(corpus))
        distiller = RecordingDistiller()
        await _start(conn)
        detector = _scored_detector({worthy: 0.5, chatter: -0.9})

        with patch("aura.backfill.worker.distill_facts", distiller):
            await _drain(
                conn,
                embedding_model,
                gateway,
                detector,
                settings=_settings(extraction_fact_worthiness_threshold=-0.02),
            )

        assert distiller.seen_message_ids == [FIRST_ID + 0, FIRST_ID + 2]

    async def test_a_stretch_of_history_with_nothing_fact_worthy_costs_no_calls(
        self, conn, embedding_model
    ) -> None:
        chatter = "lmaooo that clip was amazing"
        corpus = [_message(FIRST_ID + index, content=chatter) for index in range(40)]
        gateway = FakeGateway()
        gateway.add(FakeChannel(corpus))
        distiller = RecordingDistiller()
        await _start(conn)

        with patch("aura.backfill.worker.distill_facts", distiller):
            await _drain(
                conn,
                embedding_model,
                gateway,
                _scored_detector({chatter: -0.9}),
                settings=_settings(),
            )

        assert distiller.calls == 0
        assert await count_backfill_calls_on(conn, guild_id=GUILD_A, day=utc_day(NOW)) == 0
        run = (await get_recent_runs(conn, guild_id=GUILD_A, limit=1))[0]
        assert run.state is BackfillState.COMPLETED
        assert run.messages_scanned == 40

    async def test_the_batch_never_exceeds_the_configured_maximum(
        self, conn, embedding_model
    ) -> None:
        corpus = [_message(FIRST_ID + index) for index in range(23)]
        gateway = FakeGateway()
        gateway.add(FakeChannel(corpus))
        distiller = RecordingDistiller()
        await _start(conn)

        with patch("aura.backfill.worker.distill_facts", distiller):
            await _drain(
                conn,
                embedding_model,
                gateway,
                _detector(),
                settings=_settings(extraction_batch_max_messages=5),
            )

        assert all(len(batch) <= 5 for batch in distiller.batches)
        assert distiller.seen_message_ids == [message.id for message in corpus]

    async def test_the_channel_name_reaches_the_distiller_as_context(
        self, conn, embedding_model
    ) -> None:
        gateway = FakeGateway()
        gateway.add(FakeChannel([_message(FIRST_ID)]))
        distiller = RecordingDistiller()
        await _start(conn)

        with patch("aura.backfill.worker.distill_facts", distiller):
            await _drain(conn, embedding_model, gateway, _detector(), settings=_settings())

        assert distiller.channel_names == ["announcements"]


class TestCompletion:
    async def test_a_run_that_reaches_the_end_is_marked_completed(
        self, conn, embedding_model
    ) -> None:
        gateway = FakeGateway()
        gateway.add(FakeChannel([_message(FIRST_ID + index) for index in range(3)]))
        await _start(conn)

        with patch("aura.backfill.worker.distill_facts", RecordingDistiller()):
            await _drain(conn, embedding_model, gateway, _detector(), settings=_settings())

        run = (await get_recent_runs(conn, guild_id=GUILD_A, limit=1))[0]
        assert run.state is BackfillState.COMPLETED
        assert run.finished_at is not None

    async def test_a_completed_run_stops_costing_requests(self, conn, embedding_model) -> None:
        gateway = FakeGateway()
        channel = gateway.add(FakeChannel([_message(FIRST_ID)]))
        await _start(conn)

        with patch("aura.backfill.worker.distill_facts", RecordingDistiller()):
            await _drain(conn, embedding_model, gateway, _detector(), settings=_settings())
            requests_when_done = channel.page_requests
            for _ in range(5):
                await advance_due_backfills(
                    conn, embedding_model, gateway, _detector(), settings=_settings(), now=NOW
                )

        assert channel.page_requests == requests_when_done

    async def test_an_empty_channel_completes_immediately_without_spending_anything(
        self, conn, embedding_model
    ) -> None:
        gateway = FakeGateway()
        gateway.add(FakeChannel([]))
        distiller = RecordingDistiller()
        await _start(conn)

        with patch("aura.backfill.worker.distill_facts", distiller):
            await _drain(conn, embedding_model, gateway, _detector(), settings=_settings())

        run = (await get_recent_runs(conn, guild_id=GUILD_A, limit=1))[0]
        assert run.state is BackfillState.COMPLETED
        assert distiller.calls == 0
        assert run.messages_scanned == 0

    async def test_an_unreadable_channel_ends_the_run_as_failed(
        self, conn, embedding_model
    ) -> None:
        gateway = FakeGateway()
        gateway.add(FakeChannel([_message(FIRST_ID)]))
        gateway.unreadable.add(CHANNEL_A)
        await _start(conn)

        with patch("aura.backfill.worker.distill_facts", RecordingDistiller()):
            await advance_due_backfills(
                conn, embedding_model, gateway, _detector(), settings=_settings(), now=NOW
            )

        run = (await get_recent_runs(conn, guild_id=GUILD_A, limit=1))[0]
        assert run.state is BackfillState.FAILED
        assert run.finished_at is not None

    async def test_a_revoked_permission_mid_run_ends_the_run_and_keeps_the_cursor(
        self, conn, embedding_model
    ) -> None:
        corpus = [_message(FIRST_ID + index) for index in range(12)]
        gateway = FakeGateway()
        gateway.add(FakeChannel(corpus))
        settings = _settings()
        await _start(conn)

        with patch("aura.backfill.worker.distill_facts", RecordingDistiller()):
            await advance_due_backfills(
                conn, embedding_model, gateway, _detector(), settings=settings, now=NOW
            )
            mid_cursor = (await get_recent_runs(conn, guild_id=GUILD_A, limit=1))[
                0
            ].cursor_message_id
            gateway.unreadable.add(CHANNEL_A)
            await advance_due_backfills(
                conn, embedding_model, gateway, _detector(), settings=settings, now=NOW
            )

        run = (await get_recent_runs(conn, guild_id=GUILD_A, limit=1))[0]
        assert run.state is BackfillState.FAILED
        assert run.cursor_message_id == mid_cursor


class TestCapIndependence:
    """Attack It #3: the two caps must not touch each other, in either direction."""

    async def test_a_run_stops_at_its_cap_and_keeps_its_cursor(
        self, conn, embedding_model
    ) -> None:
        corpus = [_message(FIRST_ID + index) for index in range(40)]
        gateway = FakeGateway()
        gateway.add(FakeChannel(corpus))
        distiller = RecordingDistiller()
        settings = _settings(backfill_daily_cap=3, extraction_batch_max_messages=5)
        await _start(conn)

        with patch("aura.backfill.worker.distill_facts", distiller):
            await _drain(conn, embedding_model, gateway, _detector(), settings=settings)

        assert distiller.calls == 3
        run = (await get_recent_runs(conn, guild_id=GUILD_A, limit=1))[0]
        assert run.state is BackfillState.RUNNING, "a capped run was ended rather than paused"
        assert run.cursor_message_id is not None

    async def test_a_capped_run_resumes_on_the_next_utc_day_with_nothing_lost(
        self, conn, embedding_model
    ) -> None:
        """The multi-day run the phase brief describes, in one test."""
        corpus = [_message(FIRST_ID + index) for index in range(30)]
        gateway = FakeGateway()
        gateway.add(FakeChannel(corpus))
        distiller = RecordingDistiller()
        settings = _settings(backfill_daily_cap=2, extraction_batch_max_messages=5)
        await _start(conn)

        with patch("aura.backfill.worker.distill_facts", distiller):
            await _drain(conn, embedding_model, gateway, _detector(), settings=settings, now=NOW)
            day_one = list(distiller.seen_message_ids)
            await _drain(
                conn,
                embedding_model,
                gateway,
                _detector(),
                settings=settings,
                now=NOW + timedelta(days=1),
            )
            day_two = distiller.seen_message_ids[len(day_one) :]
            await _drain(
                conn,
                embedding_model,
                gateway,
                _detector(),
                settings=settings,
                now=NOW + timedelta(days=2),
            )
            day_three = distiller.seen_message_ids[len(day_one) + len(day_two) :]

        assert len(day_one) == 10 and len(day_two) == 10 and len(day_three) == 10
        assert set(day_one).isdisjoint(day_two)
        assert set(day_one + day_two).isdisjoint(day_three)
        assert day_one + day_two + day_three == [message.id for message in corpus]

        # The history is read, but the run has not yet SEEN that it is: its last
        # batch exhausted day three's budget, so the empty page that ends it
        # falls to day four. Still running, cursor at the end, nothing lost.
        run = (await get_recent_runs(conn, guild_id=GUILD_A, limit=1))[0]
        assert run.state is BackfillState.RUNNING
        assert run.cursor_message_id == corpus[-1].id

        with patch("aura.backfill.worker.distill_facts", distiller):
            await _drain(
                conn,
                embedding_model,
                gateway,
                _detector(),
                settings=settings,
                now=NOW + timedelta(days=3),
            )
        assert distiller.seen_message_ids == [message.id for message in corpus]
        run = (await get_recent_runs(conn, guild_id=GUILD_A, limit=1))[0]
        assert run.state is BackfillState.COMPLETED

    async def test_a_capped_run_costs_no_history_requests_at_all(
        self, conn, embedding_model
    ) -> None:
        """The cheap pre-check: a run with no budget must not still fetch ten pages."""
        corpus = [_message(FIRST_ID + index) for index in range(40)]
        gateway = FakeGateway()
        channel = gateway.add(FakeChannel(corpus))
        settings = _settings(backfill_daily_cap=1, extraction_batch_max_messages=5)
        await _start(conn)

        with patch("aura.backfill.worker.distill_facts", RecordingDistiller()):
            await _drain(conn, embedding_model, gateway, _detector(), settings=settings)
            requests_after_cap = channel.page_requests
            for _ in range(5):
                await advance_due_backfills(
                    conn, embedding_model, gateway, _detector(), settings=settings, now=NOW
                )

        assert channel.page_requests == requests_after_cap

    async def test_a_zero_cap_stops_backfill_without_touching_live_extraction(
        self, conn, embedding_model
    ) -> None:
        corpus = [_message(FIRST_ID + index) for index in range(10)]
        gateway = FakeGateway()
        gateway.add(FakeChannel(corpus))
        distiller = RecordingDistiller()
        await _start(conn)

        with patch("aura.backfill.worker.distill_facts", distiller):
            await _drain(
                conn,
                embedding_model,
                gateway,
                _detector(),
                settings=_settings(backfill_daily_cap=0),
            )

        assert distiller.calls == 0
        assert await count_extraction_calls_on(conn, guild_id=GUILD_A, day=utc_day(NOW)) == 0

    async def test_a_live_candidate_arriving_during_a_backfill_is_processed_normally(
        self, conn, embedding_model
    ) -> None:
        """The brief's exact wording, tested end to end through both paths.

        The backfill exhausts its own cap first, and only then is a live batch
        flushed -- so if the two budgets shared a number, the live batch would be
        the thing that got refused.
        """
        from aura.db.extraction_channel_config import set_extraction_enabled
        from aura.extraction.pipeline import flush_due_batches

        await set_extraction_enabled(
            conn, guild_id=GUILD_A, channel_id=CHANNEL_B, enabled=True, updated_by_id=MODERATOR
        )
        corpus = [_message(FIRST_ID + index) for index in range(40)]
        gateway = FakeGateway()
        gateway.add(FakeChannel(corpus))
        settings = _settings(backfill_daily_cap=2, extraction_batch_max_messages=5)
        await _start(conn)

        live_message = _message(UNTIL_ID + 1, channel_id=CHANNEL_B)
        await enqueue_message(
            conn,
            guild_id=GUILD_A,
            channel_id=CHANNEL_B,
            message_id=live_message.id,
            channel_name="general",
            content="The tournament starts Saturday at 6 PM.",
            message_created_at=live_message.created_at,
            now=NOW,
        )

        live_distiller = RecordingDistiller()
        with patch("aura.backfill.worker.distill_facts", RecordingDistiller()):
            await _drain(conn, embedding_model, gateway, _detector(), settings=settings)
        assert await count_backfill_calls_on(conn, guild_id=GUILD_A, day=utc_day(NOW)) == 2

        with patch("aura.extraction.pipeline.distill_facts", live_distiller):
            flushed = await flush_due_batches(
                conn,
                embedding_model,
                settings=_settings(extraction_batch_window_seconds=0.0),
                now=NOW,
            )

        assert flushed == 1, "a live batch was refused because a backfill spent its own cap"
        assert live_distiller.seen_message_ids == [live_message.id]
        assert await count_extraction_calls_on(conn, guild_id=GUILD_A, day=utc_day(NOW)) == 1


class TestSupersessionChainOverHistory:
    """Attack It #2: a rule changed three times must land on the LAST version.

    The chain itself is a moderator's work -- backfill proposes and never
    supersedes, exactly as live extraction does -- so this test plays the
    moderator: it confirms each candidate in the order backfill offered it and
    supersedes the predecessor the judgement pointed at. What is under test is
    whether the ORDER backfill offers them in makes that sequence land on the
    truly latest version, which is the whole reason chronological processing is
    enforced rather than assumed.
    """

    RULE_VERSIONS = [
        "Members may claim up to 3 pet roles.",
        "From now on, members may claim up to 5 pet roles.",
        "From now on, members may claim up to 8 pet roles.",
        "From now on, members may claim up to 2 pet roles.",
    ]

    async def _run_chain(self, conn, embedding_model, *, shuffle_pages: bool) -> list[str]:
        corpus = [
            _message(FIRST_ID + index * 10, content=version)
            for index, version in enumerate(self.RULE_VERSIONS)
        ]
        gateway = FakeGateway()
        gateway.add(FakeChannel(corpus, shuffle_pages=shuffle_pages))
        distiller = RecordingDistiller(transform=lambda queued: queued.content)
        # One version per batch, so each is staged against whatever is active at
        # that moment -- which is what makes the chain observable at all.
        settings = _settings(extraction_batch_max_messages=1)
        await _start(conn)

        judgement = RelationshipJudgement(
            relationship=SupersessionRelationship.SUPERSESSION,
            reasoning="Fact B changes the limit Fact A states.",
            change_signal="From now on",
            shared_subject="the pet role limit",
        )
        offered: list[str] = []

        with (
            patch("aura.backfill.worker.distill_facts", distiller),
            patch(
                "aura.extraction.pipeline.judge_relationship",
                AsyncMock(return_value=judgement),
            ),
        ):
            for _ in range(20):
                advanced = await advance_due_backfills(
                    conn, embedding_model, gateway, _detector(), settings=settings, now=NOW
                )
                # Play the moderator: confirm whatever is newly pending, and
                # retire the predecessor the proposal named.
                for candidate in await get_pending_facts(conn, guild_id=GUILD_A, limit=50):
                    offered.append(candidate.content)
                    predecessor_id = candidate.similar_fact_id
                    relationship = candidate.relationship
                    fact = await confirm_pending_fact(
                        conn,
                        guild_id=GUILD_A,
                        pending_id=candidate.id,
                        resolved_by_id=MODERATOR,
                    )
                    if (
                        predecessor_id is not None
                        and relationship is SupersessionRelationship.SUPERSESSION
                    ):
                        from aura.db.repository import supersede_fact_with_existing_successor

                        await supersede_fact_with_existing_successor(
                            conn,
                            old_fact_id=predecessor_id,
                            new_fact_id=fact.id,
                            guild_id=GUILD_A,
                        )
                if not advanced:
                    break

        return offered

    async def test_the_chain_lands_on_the_last_version_in_the_history(
        self, conn, embedding_model
    ) -> None:
        offered = await self._run_chain(conn, embedding_model, shuffle_pages=False)

        assert offered == self.RULE_VERSIONS, "the versions were not proposed in order"
        active = await get_active_facts(conn, GUILD_A)
        assert [fact.content for fact in active] == [self.RULE_VERSIONS[-1]]

    async def test_it_lands_on_the_last_version_even_with_scrambled_pages(
        self, conn, embedding_model
    ) -> None:
        """Without order enforcement this is where the chain would end on '3 pet roles'."""
        offered = await self._run_chain(conn, embedding_model, shuffle_pages=True)

        assert offered == self.RULE_VERSIONS
        active = await get_active_facts(conn, GUILD_A)
        assert [fact.content for fact in active] == [self.RULE_VERSIONS[-1]]

    async def test_the_whole_chain_stays_walkable_backwards(
        self, conn, embedding_model
    ) -> None:
        """Old facts are never deleted, only chained -- CLAUDE.md's Status component."""
        await self._run_chain(conn, embedding_model, shuffle_pages=False)

        active = await get_active_facts(conn, GUILD_A)
        assert len(active) == 1
        contents = [active[0].content]
        # Walk backwards: every superseded fact points at its successor, so the
        # predecessor of each link is findable by scanning for it.
        async with conn.execute(
            "SELECT content FROM facts WHERE status = ? ORDER BY id ASC",
            (FactStatus.SUPERSEDED,),
        ) as cursor:
            superseded = [row[0] for row in await cursor.fetchall()]
        assert superseded + contents == self.RULE_VERSIONS

    async def test_backfill_itself_never_supersedes_anything(
        self, conn, embedding_model
    ) -> None:
        """The judgement is a proposal. Without a moderator, nothing is retired."""
        corpus = [
            _message(FIRST_ID + index * 10, content=version)
            for index, version in enumerate(self.RULE_VERSIONS)
        ]
        await add_fact(
            conn,
            embedding_model,
            guild_id=GUILD_A,
            channel_id=CHANNEL_A,
            message_id=FIRST_ID - 1,
            content=self.RULE_VERSIONS[0],
        )
        gateway = FakeGateway()
        gateway.add(FakeChannel(corpus))
        judgement = RelationshipJudgement(
            relationship=SupersessionRelationship.SUPERSESSION,
            reasoning="Fact B changes the limit Fact A states.",
            change_signal="From now on",
            shared_subject="the pet role limit",
        )
        await _start(conn)

        with (
            patch(
                "aura.backfill.worker.distill_facts",
                RecordingDistiller(transform=lambda queued: queued.content),
            ),
            patch(
                "aura.extraction.pipeline.judge_relationship",
                AsyncMock(return_value=judgement),
            ),
        ):
            await _drain(
                conn,
                embedding_model,
                gateway,
                _detector(),
                settings=_settings(extraction_batch_max_messages=1),
            )

        active = await get_active_facts(conn, GUILD_A)
        assert [fact.content for fact in active] == [self.RULE_VERSIONS[0]]
        assert active[0].superseded_by_id is None
        judged = [
            candidate
            for candidate in await get_pending_facts(conn, guild_id=GUILD_A, limit=50)
            if candidate.relationship is SupersessionRelationship.SUPERSESSION
        ]
        assert judged, "the proposals themselves never arrived, so the check is vacuous"

    async def test_the_supersession_cap_binding_never_costs_a_candidate(
        self, conn, embedding_model
    ) -> None:
        """Backfill shares SUPERSESSION_DAILY_CAP with live extraction, and that is safe.

        A refused judgement degrades to Phase 3a-2's plain similarity hint --
        the candidate is still staged and still reviewed, exactly as it would be
        if the judgement call did not exist.
        """
        await add_fact(
            conn,
            embedding_model,
            guild_id=GUILD_A,
            channel_id=CHANNEL_A,
            message_id=FIRST_ID - 1,
            content=self.RULE_VERSIONS[0],
        )
        corpus = [
            _message(FIRST_ID + index * 10, content=version)
            for index, version in enumerate(self.RULE_VERSIONS[1:])
        ]
        gateway = FakeGateway()
        gateway.add(FakeChannel(corpus))
        await _start(conn)

        with (
            patch(
                "aura.backfill.worker.distill_facts",
                RecordingDistiller(transform=lambda queued: queued.content),
            ),
            patch("aura.extraction.pipeline.judge_relationship", AsyncMock()) as judge,
        ):
            await _drain(
                conn,
                embedding_model,
                gateway,
                _detector(),
                settings=_settings(
                    extraction_batch_max_messages=1, supersession_daily_cap=0
                ),
            )

        judge.assert_not_awaited()
        assert await count_supersession_calls_on(conn, guild_id=GUILD_A, day=utc_day(NOW)) == 0
        staged = await get_pending_facts(conn, guild_id=GUILD_A, limit=50)
        assert len(staged) == len(corpus), "a refused judgement cost a candidate"
        assert all(candidate.relationship is None for candidate in staged)


class TestLargeHistory:
    """Attack It #4: several thousand messages, performance and correctness."""

    async def test_three_thousand_messages_are_processed_in_order_without_losing_any(
        self, conn, embedding_model
    ) -> None:
        chatter = "lmaooo that clip was amazing"
        worthy = "The winter tournament starts Saturday at 6 PM."
        # ~10% fact-worthy by construction, matching the ratio
        # reports/phase-3a-1b.txt's corpus was built to.
        corpus = [
            _message(
                FIRST_ID + index,
                content=worthy if index % 10 == 0 else chatter,
            )
            for index in range(3000)
        ]
        expected = [message.id for message in corpus if message.id % 10 == FIRST_ID % 10]
        gateway = FakeGateway()
        channel = gateway.add(FakeChannel(corpus))
        distiller = RecordingDistiller()
        settings = _settings(
            backfill_daily_cap=1000, extraction_batch_max_messages=20
        )
        await _start(conn)

        started = time.perf_counter()
        with patch("aura.backfill.worker.distill_facts", distiller):
            await _drain(
                conn,
                embedding_model,
                gateway,
                _scored_detector({worthy: 0.5, chatter: -0.9}),
                settings=settings,
                max_ticks=500,
            )
        elapsed = time.perf_counter() - started

        assert distiller.seen_message_ids == expected
        assert is_strictly_increasing(
            [MagicMock(id=i, created_at=EPOCH + timedelta(seconds=i - FIRST_ID)) for i in expected]
        )
        run = (await get_recent_runs(conn, guild_id=GUILD_A, limit=1))[0]
        assert run.state is BackfillState.COMPLETED
        assert run.messages_scanned == 3000
        # 3,000 messages is 31 page requests (100 per page plus the empty one
        # that ends it) and 15 distillation calls at 20 candidates each. A run
        # that needed materially more of either would mean the cursor or the
        # batching was re-reading history.
        assert channel.page_requests <= 40, channel.page_requests
        assert distiller.calls == 15
        # Generous: the assertion worth making is "this is not quadratic", not a
        # wall-clock budget that fails on a loaded CI box.
        assert elapsed < 60.0, f"3,000 messages took {elapsed:.1f}s"

    async def test_a_large_run_survives_a_restart_at_an_arbitrary_point(
        self, tmp_path, embedding_model
    ) -> None:
        chatter = "lmaooo that clip was amazing"
        worthy = "The winter tournament starts Saturday at 6 PM."
        corpus = [
            _message(FIRST_ID + index, content=worthy if index % 10 == 0 else chatter)
            for index in range(1200)
        ]
        expected = [message.id for message in corpus if message.id % 10 == FIRST_ID % 10]
        settings = _settings(backfill_daily_cap=1000, extraction_batch_max_messages=20)
        detector = _scored_detector({worthy: 0.5, chatter: -0.9})
        path = tmp_path / "aura.db"

        seen: list[int] = []
        first = await aiosqlite.connect(path)
        await init_schema(first)
        await _start(first)
        gateway = FakeGateway()
        gateway.add(FakeChannel(corpus))
        distiller = RecordingDistiller()
        with patch("aura.backfill.worker.distill_facts", distiller):
            for _ in range(2):
                await advance_due_backfills(
                    first, embedding_model, gateway, detector, settings=settings, now=NOW
                )
        seen += distiller.seen_message_ids
        await first.close()

        second = await aiosqlite.connect(path)
        await init_schema(second)
        try:
            gateway = FakeGateway()
            gateway.add(FakeChannel(corpus))
            distiller = RecordingDistiller()
            with patch("aura.backfill.worker.distill_facts", distiller):
                await _drain(
                    second, embedding_model, gateway, detector, settings=settings, max_ticks=500
                )
            seen += distiller.seen_message_ids

            assert seen == expected
            assert len(set(seen)) == len(seen)
        finally:
            await second.close()


class TestFailureIsolation:
    async def test_one_broken_run_does_not_starve_another(self, conn, embedding_model) -> None:
        gateway = FakeGateway()
        gateway.add(FakeChannel([_message(FIRST_ID + index) for index in range(3)]))
        gateway.add(
            FakeChannel(
                [_message(FIRST_ID + index, channel_id=CHANNEL_B) for index in range(3)],
                channel_id=CHANNEL_B,
            )
        )
        gateway.unreadable.add(CHANNEL_A)
        await _start(conn, channel_id=CHANNEL_A)
        await _start(conn, channel_id=CHANNEL_B)
        distiller = RecordingDistiller()

        with patch("aura.backfill.worker.distill_facts", distiller):
            await _drain(conn, embedding_model, gateway, _detector(), settings=_settings())

        assert distiller.calls > 0, "the healthy run was starved by the broken one"
        runs = {run.channel_id: run for run in await get_recent_runs(conn, guild_id=GUILD_A, limit=5)}
        assert runs[CHANNEL_A].state is BackfillState.FAILED
        assert runs[CHANNEL_B].state is BackfillState.COMPLETED

    async def test_an_unexpected_exception_in_one_run_leaves_its_cursor_alone(
        self, conn, embedding_model
    ) -> None:
        gateway = FakeGateway()
        gateway.add(FakeChannel([_message(FIRST_ID + index) for index in range(3)]))
        await _start(conn)

        with patch(
            "aura.backfill.worker.distill_facts",
            AsyncMock(side_effect=RuntimeError("boom")),
        ):
            advanced = await advance_due_backfills(
                conn, embedding_model, gateway, _detector(), settings=_settings(), now=NOW
            )

        assert advanced == 0
        run = (await get_recent_runs(conn, guild_id=GUILD_A, limit=1))[0]
        assert run.state is BackfillState.RUNNING
        assert run.cursor_message_id is None

    async def test_a_failed_distillation_still_moves_past_the_batch(
        self, conn, embedding_model
    ) -> None:
        """A batch the model cannot handle must not loop forever spending a slot a tick."""
        gateway = FakeGateway()
        gateway.add(FakeChannel([_message(FIRST_ID + index) for index in range(3)]))
        await _start(conn)

        with patch(
            "aura.backfill.worker.distill_facts", AsyncMock(return_value=None)
        ) as distiller:
            await _drain(conn, embedding_model, gateway, _detector(), settings=_settings())

        assert distiller.await_count == 1
        run = (await get_recent_runs(conn, guild_id=GUILD_A, limit=1))[0]
        assert run.state is BackfillState.COMPLETED
        assert run.calls_spent == 1
        assert await get_pending_facts(conn, guild_id=GUILD_A, limit=10) == []

    async def test_an_empty_distillation_result_is_not_a_failure(
        self, conn, embedding_model
    ) -> None:
        gateway = FakeGateway()
        gateway.add(FakeChannel([_message(FIRST_ID + index) for index in range(3)]))
        await _start(conn)

        with patch("aura.backfill.worker.distill_facts", AsyncMock(return_value=[])):
            await _drain(conn, embedding_model, gateway, _detector(), settings=_settings())

        run = (await get_recent_runs(conn, guild_id=GUILD_A, limit=1))[0]
        assert run.state is BackfillState.COMPLETED
        assert run.candidates_staged == 0

    async def test_a_transient_fetch_failure_leaves_the_cursor_and_retries_next_tick(
        self, conn, embedding_model
    ) -> None:
        corpus = [_message(FIRST_ID + index) for index in range(3)]
        response = MagicMock()
        response.status = 503
        response.reason = "test"
        response.headers = {}
        broken = discord.HTTPException(response, {"message": "nope", "code": 0})
        broken.status = 503

        gateway = FakeGateway()
        channel = gateway.add(FakeChannel(corpus, raises=broken))
        await _start(conn)

        with (
            patch("aura.backfill.worker.distill_facts", RecordingDistiller()),
            patch("aura.backfill.history.asyncio.sleep", AsyncMock()),
        ):
            advanced = await advance_due_backfills(
                conn, embedding_model, gateway, _detector(), settings=_settings(), now=NOW
            )

        assert advanced == 0
        run = (await get_recent_runs(conn, guild_id=GUILD_A, limit=1))[0]
        assert run.state is BackfillState.RUNNING
        assert run.cursor_message_id is None

        channel.raises = None
        with patch("aura.backfill.worker.distill_facts", RecordingDistiller()) as distiller:
            await _drain(conn, embedding_model, gateway, _detector(), settings=_settings())
        assert distiller.seen_message_ids == [message.id for message in corpus]

    async def test_no_extraction_model_configured_means_the_run_simply_waits(
        self, conn, embedding_model
    ) -> None:
        gateway = FakeGateway()
        channel = gateway.add(FakeChannel([_message(FIRST_ID)]))
        await _start(conn)

        with patch("aura.backfill.worker.distill_facts", RecordingDistiller()):
            advanced = await advance_due_backfills(
                conn,
                embedding_model,
                gateway,
                _detector(),
                settings=_settings(llm_api_key=None),
                now=NOW,
            )

        assert advanced == 0
        assert channel.page_requests == 0
        run = (await get_recent_runs(conn, guild_id=GUILD_A, limit=1))[0]
        assert run.state is BackfillState.RUNNING


class TestGuildIsolation:
    async def test_two_guilds_runs_do_not_see_each_others_facts_or_candidates(
        self, conn, embedding_model
    ) -> None:
        await add_fact(
            conn,
            embedding_model,
            guild_id=GUILD_B,
            channel_id=CHANNEL_B,
            message_id=1,
            content="Members may claim up to 3 pet roles.",
        )
        corpus = [
            _message(FIRST_ID, content="From now on, members may claim up to 5 pet roles.")
        ]
        gateway = FakeGateway()
        gateway.add(FakeChannel(corpus))
        await _start(conn, guild_id=GUILD_A)

        with (
            patch(
                "aura.backfill.worker.distill_facts",
                RecordingDistiller(transform=lambda queued: queued.content),
            ),
            patch("aura.extraction.pipeline.judge_relationship", AsyncMock()) as judge,
        ):
            await _drain(conn, embedding_model, gateway, _detector(), settings=_settings())

        judge.assert_not_awaited()
        staged = await get_pending_facts(conn, guild_id=GUILD_A, limit=10)
        assert len(staged) == 1
        assert staged[0].similar_fact_id is None, "another guild's fact was used as a predecessor"
        assert await get_pending_facts(conn, guild_id=GUILD_B, limit=10) == []


class TestWorkerLoop:
    async def test_the_loop_survives_a_failing_sweep_and_keeps_going(self) -> None:
        connection = await aiosqlite.connect(":memory:")
        await init_schema(connection)
        calls = 0

        async def flaky(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("boom")
            return 0

        # Captured BEFORE the patch below: `asyncio` is one shared module object,
        # so a replacement that called asyncio.sleep would call itself.
        real_sleep = asyncio.sleep

        async def instant(_delay: float) -> None:
            # The loop's own sleep, made free: what is under test is that the
            # loop comes back after an exception, not how long it waits first.
            await real_sleep(0)

        try:
            with (
                patch("aura.backfill.worker.advance_due_backfills", flaky),
                patch("aura.backfill.worker.asyncio.sleep", instant),
            ):
                task = asyncio.create_task(
                    run_backfill_worker(
                        connection,
                        MagicMock(),
                        FakeGateway(),
                        _detector(),
                        settings=_settings(),
                    )
                )
                for _ in range(200):
                    await asyncio.sleep(0)
                    if calls >= 2:
                        break
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
        finally:
            await connection.close()

        assert calls >= 2, "the loop died on the first failing sweep"

    async def test_cancelling_the_worker_propagates_rather_than_being_swallowed(self) -> None:
        connection = await aiosqlite.connect(":memory:")
        await init_schema(connection)
        try:
            task = asyncio.create_task(
                run_backfill_worker(
                    connection,
                    MagicMock(),
                    FakeGateway(),
                    _detector(),
                    settings=_settings(),
                )
            )
            await asyncio.sleep(0)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        finally:
            await connection.close()


class TestInjectedClock:
    async def test_a_naive_now_is_rejected(self, conn, embedding_model) -> None:
        with pytest.raises(ValueError, match="timezone-aware"):
            await advance_due_backfills(
                conn,
                embedding_model,
                FakeGateway(),
                _detector(),
                settings=_settings(),
                now=datetime(2026, 8, 26, 12, 0, 0),
            )


class TestCrossGuildIsolation:
    """The adversarial pass's second real find, and the one with teeth.

    A run reads MESSAGE CONTENT. A stored channel id that now resolves into a
    different server -- a deleted channel whose id Discord reused, or a
    hand-edited row -- would have Aura reading that server's conversations and
    staging candidates under this guild's id. That is a data leak, not a
    rendering bug, so it is checked in the worker rather than left to the slash
    command's own guild scoping.
    """

    async def test_a_channel_that_now_belongs_to_another_guild_ends_the_run(
        self, conn, embedding_model
    ) -> None:
        corpus = [_message(FIRST_ID + index) for index in range(4)]
        channel = FakeChannel(corpus)
        assert channel.guild is not None
        channel.guild.id = GUILD_B  # the id was reused by another server
        gateway = FakeGateway()
        gateway.add(channel)
        distiller = RecordingDistiller()
        await _start(conn, guild_id=GUILD_A)

        with patch("aura.backfill.worker.distill_facts", distiller):
            await advance_due_backfills(
                conn, embedding_model, gateway, _detector(), settings=_settings(), now=NOW
            )

        assert distiller.calls == 0, "another server's history was read"
        assert await get_pending_facts(conn, guild_id=GUILD_A, limit=10) == []
        assert await get_pending_facts(conn, guild_id=GUILD_B, limit=10) == []
        run = (await get_recent_runs(conn, guild_id=GUILD_A, limit=1))[0]
        assert run.state is BackfillState.FAILED

    async def test_a_channel_with_no_guild_at_all_ends_the_run(
        self, conn, embedding_model
    ) -> None:
        channel = FakeChannel([_message(FIRST_ID)])
        channel.guild = None
        gateway = FakeGateway()
        gateway.add(channel)
        distiller = RecordingDistiller()
        await _start(conn)

        with patch("aura.backfill.worker.distill_facts", distiller):
            await advance_due_backfills(
                conn, embedding_model, gateway, _detector(), settings=_settings(), now=NOW
            )

        assert distiller.calls == 0
        assert (
            await get_recent_runs(conn, guild_id=GUILD_A, limit=1)
        )[0].state is BackfillState.FAILED

    async def test_a_moderators_cancel_is_not_overwritten_by_a_channel_failure(
        self, conn, embedding_model
    ) -> None:
        """'Ended by a moderator' and 'the channel broke' mean different things."""
        run = await _start(conn)
        await set_run_state(
            conn,
            run_id=run.id,
            state=BackfillState.CANCELLED,
            now=NOW,
            from_states=(BackfillState.RUNNING,),
        )
        gateway = FakeGateway()
        gateway.unreadable.add(CHANNEL_A)

        with patch("aura.backfill.worker.distill_facts", RecordingDistiller()):
            await advance_due_backfills(
                conn, embedding_model, gateway, _detector(), settings=_settings(), now=NOW
            )

        assert (
            await get_recent_runs(conn, guild_id=GUILD_A, limit=1)
        )[0].state is BackfillState.CANCELLED


class TestConcurrentTicks:
    async def test_two_ticks_racing_one_run_stage_each_candidate_exactly_once(
        self, conn, embedding_model
    ) -> None:
        """Two overlapping sweeps -- a tick that outran its own interval.

        The prize for getting this wrong is a duplicate candidate a moderator
        has to reject twice, and a cursor that skipped a page because two
        advances landed out of order.
        """
        corpus = [_message(FIRST_ID + index) for index in range(6)]
        gateway = FakeGateway()
        gateway.add(FakeChannel(corpus))
        settings = _settings(extraction_batch_max_messages=2)
        await _start(conn)
        distiller = RecordingDistiller()

        with patch("aura.backfill.worker.distill_facts", distiller):
            await asyncio.gather(
                *(
                    advance_due_backfills(
                        conn,
                        embedding_model,
                        gateway,
                        _detector(),
                        settings=settings,
                        now=NOW,
                    )
                    for _ in range(4)
                )
            )

        staged = await get_pending_facts(conn, guild_id=GUILD_A, limit=100)
        assert len(staged) == len({fact.message_id for fact in staged}), "a duplicate landed"
        run = (await get_recent_runs(conn, guild_id=GUILD_A, limit=1))[0]
        assert run.cursor_message_id is not None

        # And the rest of the history is still reachable: nothing was skipped.
        with patch("aura.backfill.worker.distill_facts", distiller):
            await _drain(conn, embedding_model, gateway, _detector(), settings=settings)
        final = await get_pending_facts(conn, guild_id=GUILD_A, limit=100)
        assert {fact.message_id for fact in final} == {message.id for message in corpus}

    async def test_cancellation_propagates_rather_than_being_logged_as_a_failure(
        self, conn, embedding_model
    ) -> None:
        """CancelledError is a BaseException, so shutdown must not be swallowed."""
        gateway = FakeGateway()
        gateway.add(FakeChannel([_message(FIRST_ID + index) for index in range(3)]))
        await _start(conn)

        with patch(
            "aura.backfill.worker.distill_facts",
            AsyncMock(side_effect=asyncio.CancelledError()),
        ):
            with pytest.raises(asyncio.CancelledError):
                await advance_due_backfills(
                    conn, embedding_model, gateway, _detector(), settings=_settings(), now=NOW
                )


class TestDegenerateInput:
    async def test_unicode_survives_the_whole_path_byte_for_byte(
        self, conn, embedding_model
    ) -> None:
        contents = [
            "サーバーは本日14時からメンテナンスのため停止します。",
            "Ab sofort gilt die neue Regel im Handelskanal 🎉",
            "قواعد الخادم‏تغيرت",
            "오늘부터 새 멤버는 이메일 인증을 해야 합니다",
        ]
        corpus = [
            _message(FIRST_ID + index, content=content)
            for index, content in enumerate(contents)
        ]
        gateway = FakeGateway()
        gateway.add(FakeChannel(corpus))
        distiller = RecordingDistiller(transform=lambda queued: queued.content)
        await _start(conn)

        with patch("aura.backfill.worker.distill_facts", distiller):
            await _drain(conn, embedding_model, gateway, _detector(), settings=_settings())

        staged = await get_pending_facts(conn, guild_id=GUILD_A, limit=10)
        assert sorted(fact.content for fact in staged) == sorted(contents)

    async def test_a_five_thousand_character_message_is_handled(
        self, conn, embedding_model
    ) -> None:
        corpus = [_message(FIRST_ID, content="a" * 5000)]
        gateway = FakeGateway()
        gateway.add(FakeChannel(corpus))
        distiller = RecordingDistiller()
        await _start(conn)

        with patch("aura.backfill.worker.distill_facts", distiller):
            await _drain(conn, embedding_model, gateway, _detector(), settings=_settings())

        assert distiller.batches == [[FIRST_ID]]
        assert distiller.contents == ["a" * 5000], "the message was truncated on the way in"

    async def test_a_channel_whose_name_is_missing_falls_back_to_its_id(
        self, conn, embedding_model
    ) -> None:
        """A name is prompt context, never an identifier -- it must not break a run."""
        channel = FakeChannel([_message(FIRST_ID)])
        channel.name = None
        gateway = FakeGateway()
        gateway.add(channel)
        distiller = RecordingDistiller()
        await _start(conn)

        with patch("aura.backfill.worker.distill_facts", distiller):
            await _drain(conn, embedding_model, gateway, _detector(), settings=_settings())

        assert distiller.channel_names == [str(CHANNEL_A)]

    async def test_a_page_containing_the_same_message_twice_stages_it_once(
        self, conn, embedding_model
    ) -> None:
        """Not reachable through discord.py today; absorbed regardless."""
        duplicated = _message(FIRST_ID)
        gateway = FakeGateway()
        channel = FakeChannel([duplicated, _message(FIRST_ID + 1)])
        channel._corpus = [duplicated, duplicated, _message(FIRST_ID + 1)]  # type: ignore[attr-defined]
        gateway.add(channel)
        await _start(conn)

        with patch(
            "aura.backfill.worker.distill_facts",
            RecordingDistiller(transform=lambda queued: queued.content),
        ):
            await _drain(conn, embedding_model, gateway, _detector(), settings=_settings())

        staged = await get_pending_facts(conn, guild_id=GUILD_A, limit=10)
        assert len({fact.message_id for fact in staged}) == len(staged)


class TestTheConnectionLockIsNotHeldAcrossThePaidCall:
    """The one that would be easy to get wrong and invisible in production.

    A 60-second distillation request made while holding the per-connection lock
    would stall every command in every guild behind one backfill batch. Phase
    3a-3 added this test for the judgement call (reports/phase-3a-3.txt Section
    10); backfill makes the same call from a different place, so it needs its
    own.
    """

    async def test_an_unrelated_read_completes_while_a_batch_is_in_flight(
        self, conn, embedding_model
    ) -> None:
        gateway = FakeGateway()
        gateway.add(FakeChannel([_message(FIRST_ID + index) for index in range(3)]))
        await _start(conn)
        in_flight = asyncio.Event()
        release = asyncio.Event()

        async def blocking_distill(candidates, *, channel_name: str, model: str):
            in_flight.set()
            await release.wait()
            return [
                DistilledFact(
                    message_id=queued.message_id,
                    content=f"Distilled: {queued.content}",
                    category=FactCategory.ANNOUNCEMENT,
                )
                for queued in candidates
            ]

        with patch("aura.backfill.worker.distill_facts", blocking_distill):
            sweep = asyncio.create_task(
                advance_due_backfills(
                    conn, embedding_model, gateway, _detector(), settings=_settings(), now=NOW
                )
            )
            await asyncio.wait_for(in_flight.wait(), timeout=5.0)

            # The assertion, and the reason it is a wait_for rather than a plain
            # await: if the lock were held across the call above, this read
            # would block until `release` is set and the timeout would fire.
            facts = await asyncio.wait_for(get_active_facts(conn, GUILD_A), timeout=2.0)
            assert facts == []

            release.set()
            await sweep

    async def test_the_assertion_above_is_not_vacuous(
        self, conn, embedding_model
    ) -> None:
        """A genuinely held lock DOES block that read, so the test can actually fail."""
        from aura.db.connection import connection_lock

        async with connection_lock(conn):
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(get_active_facts(conn, GUILD_A), timeout=0.2)
