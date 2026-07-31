"""Tests for aura.extraction.pipeline: the gates, the flush, and the isolation.

The distillation call is mocked throughout (conftest's autouse guard fails the
run if anything reaches a real one); what is exercised here is everything
around it -- which messages get in, what the daily cap does when it bites, what
happens when the model returns nothing or returns garbage, and that none of it
is entangled with Trigger 2.

The embedding model is real, because the dedup step's whole job is a semantic
comparison a mock cannot meaningfully stand in for.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import aiosqlite
import discord
import pytest
from fastembed import TextEmbedding

from aura.config import ModelComponent, Settings
from aura.db.extraction_channel_config import set_extraction_enabled
from aura.db.extraction_queue import count_queued, enqueue_message
from aura.db.extraction_state import count_extraction_calls_on
from aura.db.connection import utc_day, utc_now
from aura.db.models import FactStatus
from aura.db.pending_facts import (
    FactCategory,
    PendingFactStatus,
    SupersessionRelationship,
    get_pending_facts,
)
from aura.db.proactive_channel_config import set_channel_enabled
from aura.db.repository import get_active_facts, init_schema
from aura.db.supersession_state import count_supersession_calls_on
from aura.extraction.distiller import DistilledFact
from aura.extraction.supersession import RelationshipJudgement
from aura.extraction.pipeline import (
    flush_due_batches,
    handle_extraction_message,
    should_extract,
    sweep_interval_seconds,
    withdraw_message,
)
from aura.facts_service import add_fact

GUILD_A = 100000000000000001
CHANNEL_A = 500000000000000005
NOW = datetime(2026, 7, 30, 12, 0, 0, tzinfo=timezone.utc)


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
        # Zero window: every queued message is immediately due, so the flush
        # tests do not have to wait or fake a clock.
        "extraction_batch_window_seconds": 0.0,
        "extraction_daily_cap": 50,
        "extraction_batch_max_messages": 20,
        "extraction_fact_worthiness_threshold": -0.02,
        **overrides,
    }
    return Settings(_env_file=None, **values)  # type: ignore[call-arg]


def _message(
    *,
    content: str = "The server is down for maintenance today at 14:00 UTC.",
    message_id: int = 1,
    channel_id: int = CHANNEL_A,
    bot: bool = False,
    webhook_id: int | None = None,
    interaction_metadata: object | None = None,
    message_type: discord.MessageType = discord.MessageType.default,
    guild: bool = True,
) -> MagicMock:
    message = MagicMock(spec=discord.Message)
    message.content = content
    message.id = message_id
    message.guild = MagicMock() if guild else None
    if guild:
        message.guild.id = GUILD_A
    message.channel = MagicMock()
    message.channel.id = channel_id
    message.channel.name = "announcements"
    message.author = MagicMock()
    message.author.bot = bot
    message.webhook_id = webhook_id
    message.interaction_metadata = interaction_metadata
    message.type = message_type
    message.created_at = NOW
    return message


def _detector(score: float = 1.0) -> MagicMock:
    """A fact-worthiness detector returning a controlled score.

    Mocked deliberately: the filter's calibration is reports/phase-3a-1b.txt's
    subject, and what these tests need to isolate is the pipeline's own
    threshold comparison, not the exemplar geometry behind the number.
    """
    detector = MagicMock()
    detector.question_likeness = AsyncMock(return_value=score)
    return detector


class TestShouldExtract:
    def test_ordinary_guild_text_qualifies(self) -> None:
        assert should_extract(_message()) is True

    def test_a_dm_is_rejected(self) -> None:
        assert should_extract(_message(guild=False)) is False

    def test_a_bot_message_is_rejected(self) -> None:
        # Including Aura's own posts: extracting from its own proactive answers
        # would let it launder synthesis back into the knowledge model as a new
        # fact, a loop that would eventually cite itself.
        assert should_extract(_message(bot=True)) is False

    def test_a_webhook_message_is_rejected(self) -> None:
        assert should_extract(_message(webhook_id=123)) is False

    def test_a_slash_command_response_is_rejected(self) -> None:
        assert should_extract(_message(interaction_metadata=MagicMock())) is False

    @pytest.mark.parametrize(
        "message_type",
        [
            discord.MessageType.pins_add,
            discord.MessageType.new_member,
            discord.MessageType.premium_guild_subscription,
            discord.MessageType.thread_created,
        ],
    )
    def test_system_message_types_are_rejected(
        self, message_type: discord.MessageType
    ) -> None:
        assert should_extract(_message(message_type=message_type)) is False

    def test_a_reply_qualifies(self) -> None:
        assert should_extract(_message(message_type=discord.MessageType.reply)) is True

    @pytest.mark.parametrize("content", ["", "   ", "\n\t", "\xa0\xa0"])
    def test_empty_and_whitespace_only_content_is_rejected(self, content: str) -> None:
        assert should_extract(_message(content=content)) is False

    @pytest.mark.parametrize(
        "content",
        [
            pytest.param("​​​", id="zero-width-space"),
            pytest.param("‍‏﻿", id="joiner-bidi-bom"),
            pytest.param("⁠" * 20, id="word-joiner"),
            pytest.param("​ \n‍\t", id="mixed-with-real-whitespace"),
        ],
    )
    def test_a_message_of_only_invisible_characters_is_rejected(self, content: str) -> None:
        # Found by this phase's adversarial pass. str.strip() does not remove
        # Unicode format characters, so "​​".strip() is still truthy
        # and a message nobody can see would otherwise ride into a paid batch.
        # The shipped threshold already scores these near -0.49, far under the
        # -0.02 bar, so nothing was getting through -- but that is a calibrated
        # placeholder, and "no visible characters" is a structural property
        # that should not depend on where it happens to sit.
        assert should_extract(_message(content=content)) is False

    @pytest.mark.parametrize(
        "content",
        [
            pytest.param("​The event is Saturday.​", id="wrapped-in-zero-width"),
            pytest.param("\U0001F1F0\U0001F1F7 공지", id="emoji-and-hangul"),
            pytest.param("‏عربي‏", id="rtl-marked-arabic"),
        ],
    )
    def test_invisible_characters_around_real_text_still_qualify(self, content: str) -> None:
        # The check must reject only genuinely empty messages, never strip a
        # locale's real content along the way -- nine languages and four
        # scripts make that a live risk rather than a theoretical one.
        assert should_extract(_message(content=content)) is True


class TestIntakeGates:
    async def test_a_qualifying_message_in_an_enabled_channel_is_queued(
        self, conn: aiosqlite.Connection
    ) -> None:
        await set_extraction_enabled(
            conn, guild_id=GUILD_A, channel_id=CHANNEL_A, enabled=True, updated_by_id=1
        )
        await handle_extraction_message(
            _message(), db=conn, detector=_detector(0.5), settings=_settings()
        )
        assert await count_queued(conn, channel_id=CHANNEL_A) == 1

    async def test_a_channel_nobody_enabled_queues_nothing(
        self, conn: aiosqlite.Connection
    ) -> None:
        # Opt-in per channel, exactly like proactive relief -- and the default
        # for a channel with no row is OFF.
        await handle_extraction_message(
            _message(), db=conn, detector=_detector(0.5), settings=_settings()
        )
        assert await count_queued(conn) == 0

    async def test_a_disabled_channel_does_not_even_score_the_message(
        self, conn: aiosqlite.Connection
    ) -> None:
        # The channel gate must come before the embedding inference, not merely
        # before the write: a disabled channel should cost one indexed lookup
        # and nothing else.
        detector = _detector(0.5)
        await handle_extraction_message(
            _message(), db=conn, detector=detector, settings=_settings()
        )
        detector.question_likeness.assert_not_awaited()

    async def test_a_message_below_the_threshold_is_not_queued(
        self, conn: aiosqlite.Connection
    ) -> None:
        await set_extraction_enabled(
            conn, guild_id=GUILD_A, channel_id=CHANNEL_A, enabled=True, updated_by_id=1
        )
        await handle_extraction_message(
            _message(), db=conn, detector=_detector(-0.9), settings=_settings()
        )
        assert await count_queued(conn) == 0

    async def test_a_message_exactly_at_the_threshold_is_queued(
        self, conn: aiosqlite.Connection
    ) -> None:
        # The comparison is >=, so the boundary belongs to the passing side.
        await set_extraction_enabled(
            conn, guild_id=GUILD_A, channel_id=CHANNEL_A, enabled=True, updated_by_id=1
        )
        await handle_extraction_message(
            _message(), db=conn, detector=_detector(-0.02), settings=_settings()
        )
        assert await count_queued(conn) == 1

    async def test_nothing_is_queued_when_no_extraction_llm_is_configured(
        self, conn: aiosqlite.Connection
    ) -> None:
        # With no model there is nothing that could ever drain the queue, so
        # enqueueing would accumulate raw message text forever for a pipeline
        # that cannot run.
        await set_extraction_enabled(
            conn, guild_id=GUILD_A, channel_id=CHANNEL_A, enabled=True, updated_by_id=1
        )
        settings = _settings(llm_api_key=None, extraction_model=None, synthesis_model=None)
        detector = _detector(0.9)
        await handle_extraction_message(
            _message(), db=conn, detector=detector, settings=settings
        )
        assert await count_queued(conn) == 0
        detector.question_likeness.assert_not_awaited()

    async def test_a_failure_anywhere_degrades_to_a_log_line(
        self, conn: aiosqlite.Connection, caplog: pytest.LogCaptureFixture
    ) -> None:
        # This runs on every message Aura can see, and shares on_message with
        # the proactive path; an exception escaping here would travel back into
        # the gateway's dispatch and take both paths down.
        await set_extraction_enabled(
            conn, guild_id=GUILD_A, channel_id=CHANNEL_A, enabled=True, updated_by_id=1
        )
        detector = _detector()
        detector.question_likeness = AsyncMock(side_effect=RuntimeError("model exploded"))

        with caplog.at_level("ERROR"):
            await handle_extraction_message(
                _message(), db=conn, detector=detector, settings=_settings()
            )

        assert any(record.levelname == "ERROR" for record in caplog.records)
        assert await count_queued(conn) == 0  # failed closed

    async def test_a_broken_message_object_does_not_raise(
        self, conn: aiosqlite.Connection
    ) -> None:
        broken = MagicMock(spec=discord.Message)
        type(broken).guild = property(lambda _self: (_ for _ in ()).throw(AttributeError()))
        await handle_extraction_message(
            broken, db=conn, detector=_detector(), settings=_settings()
        )


class TestIndependenceFromTriggerTwo:
    """Both passive paths see the same message; neither may gate the other.

    reports/phase-3-pre-analysis.md Section 1c found this to be a real
    collision risk rather than a hypothetical one, so it is tested from both
    directions rather than argued from the fact that the tables are separate.
    """

    async def test_extraction_runs_in_a_channel_where_proactive_relief_is_off(
        self, conn: aiosqlite.Connection
    ) -> None:
        await set_extraction_enabled(
            conn, guild_id=GUILD_A, channel_id=CHANNEL_A, enabled=True, updated_by_id=1
        )
        # proactive_channel_config deliberately left empty -> Trigger 2 is off.
        await handle_extraction_message(
            _message(), db=conn, detector=_detector(0.5), settings=_settings()
        )
        assert await count_queued(conn, channel_id=CHANNEL_A) == 1

    async def test_enabling_proactive_relief_does_not_enable_extraction(
        self, conn: aiosqlite.Connection
    ) -> None:
        await set_channel_enabled(
            conn, guild_id=GUILD_A, channel_id=CHANNEL_A, enabled=True, updated_by_id=1
        )
        await handle_extraction_message(
            _message(), db=conn, detector=_detector(0.5), settings=_settings()
        )
        assert await count_queued(conn) == 0

    async def test_extraction_never_touches_the_proactive_spend_ledger(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        # The two budgets are independent: a guild that has exhausted its
        # extraction cap must still be able to answer questions, and vice
        # versa.
        await set_extraction_enabled(
            conn, guild_id=GUILD_A, channel_id=CHANNEL_A, enabled=True, updated_by_id=1
        )
        await handle_extraction_message(
            _message(), db=conn, detector=_detector(0.5), settings=_settings()
        )
        with patch(
            "aura.extraction.pipeline.distill_facts", AsyncMock(return_value=[])
        ) as distiller:
            await flush_due_batches(
                conn, embedding_model, settings=_settings(), now=utc_now()
            )

        # The flush genuinely ran -- otherwise the assertions below would pass
        # vacuously, which is exactly how a broken isolation test looks.
        distiller.assert_awaited_once()

        async with conn.execute("SELECT COUNT(*) FROM proactive_escalations") as cursor:
            assert await cursor.fetchone() == (0,)
        async with conn.execute("SELECT COUNT(*) FROM proactive_signals") as cursor:
            assert await cursor.fetchone() == (0,)


class TestWithdrawal:
    async def test_an_edited_message_leaves_the_pending_batch(
        self, conn: aiosqlite.Connection
    ) -> None:
        await set_extraction_enabled(
            conn, guild_id=GUILD_A, channel_id=CHANNEL_A, enabled=True, updated_by_id=1
        )
        await handle_extraction_message(
            _message(), db=conn, detector=_detector(0.5), settings=_settings()
        )
        assert await count_queued(conn, channel_id=CHANNEL_A) == 1

        await withdraw_message(conn, channel_id=CHANNEL_A, message_id=1)
        assert await count_queued(conn, channel_id=CHANNEL_A) == 0

    async def test_a_withdrawn_message_is_never_distilled(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        # The point of the whole edit/delete hook: withdrawal inside the batch
        # window means the message never reaches the paid call at all.
        await set_extraction_enabled(
            conn, guild_id=GUILD_A, channel_id=CHANNEL_A, enabled=True, updated_by_id=1
        )
        await handle_extraction_message(
            _message(message_id=1), db=conn, detector=_detector(0.5), settings=_settings()
        )
        await handle_extraction_message(
            _message(message_id=2, content="The tournament starts Saturday at 18:00."),
            db=conn,
            detector=_detector(0.5),
            settings=_settings(),
        )
        await withdraw_message(conn, channel_id=CHANNEL_A, message_id=1)

        distiller = AsyncMock(return_value=[])
        with patch("aura.extraction.pipeline.distill_facts", distiller):
            await flush_due_batches(conn, embedding_model, settings=_settings(), now=utc_now())

        sent_batch = distiller.call_args.args[0]
        assert [message.message_id for message in sent_batch] == [2]

    async def test_withdrawing_everything_spends_no_slot(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        await set_extraction_enabled(
            conn, guild_id=GUILD_A, channel_id=CHANNEL_A, enabled=True, updated_by_id=1
        )
        await handle_extraction_message(
            _message(), db=conn, detector=_detector(0.5), settings=_settings()
        )
        await withdraw_message(conn, channel_id=CHANNEL_A, message_id=1)

        with patch("aura.extraction.pipeline.distill_facts", AsyncMock()) as distiller:
            assert (
                await flush_due_batches(
                    conn, embedding_model, settings=_settings(), now=utc_now()
                )
                == 0
            )
        distiller.assert_not_awaited()
        assert (
            await count_extraction_calls_on(conn, guild_id=GUILD_A, day=utc_day(NOW)) == 0
        )

    async def test_withdrawal_never_raises_on_a_database_failure(
        self, conn: aiosqlite.Connection, caplog: pytest.LogCaptureFixture
    ) -> None:
        with patch(
            "aura.extraction.pipeline.remove_queued_message",
            AsyncMock(side_effect=RuntimeError("db is gone")),
        ):
            with caplog.at_level("ERROR"):
                await withdraw_message(conn, channel_id=CHANNEL_A, message_id=1)
        assert any(record.levelname == "ERROR" for record in caplog.records)


async def _queue(conn: aiosqlite.Connection, *, message_id: int, content: str) -> None:
    await enqueue_message(
        conn,
        guild_id=GUILD_A,
        channel_id=CHANNEL_A,
        message_id=message_id,
        channel_name="announcements",
        content=content,
        message_created_at=NOW,
        now=NOW,
    )


class TestFlush:
    async def test_a_distilled_batch_is_staged_and_cleared(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        await _queue(conn, message_id=1, content="server down at 2 today")

        distilled = [
            DistilledFact(
                message_id=1,
                content="The server is down for maintenance on 2026-07-30 at 14:00 UTC.",
                category=FactCategory.STATUS_CHANGE,
            )
        ]
        with patch(
            "aura.extraction.pipeline.distill_facts", AsyncMock(return_value=distilled)
        ):
            assert await flush_due_batches(conn, embedding_model, settings=_settings(), now=NOW) == 1

        staged = await get_pending_facts(conn, guild_id=GUILD_A, limit=10)
        assert len(staged) == 1
        assert staged[0].content == distilled[0].content
        assert staged[0].category is FactCategory.STATUS_CHANGE
        assert staged[0].status is PendingFactStatus.PENDING
        assert staged[0].message_id == 1
        assert await count_queued(conn) == 0

    async def test_staging_never_creates_an_active_fact(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        # The single most important property of this whole sub-phase.
        await _queue(conn, message_id=1, content="server down at 2")
        distilled = [
            DistilledFact(
                message_id=1, content="Maintenance is today.", category=FactCategory.EVENT
            )
        ]
        with patch(
            "aura.extraction.pipeline.distill_facts", AsyncMock(return_value=distilled)
        ):
            await flush_due_batches(conn, embedding_model, settings=_settings(), now=NOW)

        assert await get_active_facts(conn, GUILD_A) == []

    async def test_an_empty_result_clears_the_batch_without_staging(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        # The common case: the pre-filter let a few things through and the
        # model correctly judged none of them fact-worthy.
        await _queue(conn, message_id=1, content="lol nice")
        with patch(
            "aura.extraction.pipeline.distill_facts", AsyncMock(return_value=[])
        ):
            await flush_due_batches(conn, embedding_model, settings=_settings(), now=NOW)

        assert await get_pending_facts(conn, guild_id=GUILD_A, limit=10) == []
        assert await count_queued(conn) == 0

    async def test_a_failed_call_clears_the_batch_and_still_spends_the_slot(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        # Both halves are deliberate: the slot is never refunded (that is what
        # bounds a reliably-failing model), and the batch is not retried
        # forever (which would spend a slot per sweep on a batch the model
        # cannot handle).
        await _queue(conn, message_id=1, content="something")
        with patch(
            "aura.extraction.pipeline.distill_facts", AsyncMock(return_value=None)
        ):
            await flush_due_batches(conn, embedding_model, settings=_settings(), now=NOW)

        assert await count_queued(conn) == 0
        assert await get_pending_facts(conn, guild_id=GUILD_A, limit=10) == []
        assert (
            await count_extraction_calls_on(conn, guild_id=GUILD_A, day=utc_day(NOW)) == 1
        )

    async def test_the_batch_size_limit_caps_one_call_and_leaves_the_rest(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        for message_id in range(1, 8):
            await _queue(conn, message_id=message_id, content=f"announcement {message_id}")

        distiller = AsyncMock(return_value=[])
        with patch("aura.extraction.pipeline.distill_facts", distiller):
            await flush_due_batches(
                conn, embedding_model, settings=_settings(extraction_batch_max_messages=3), now=NOW
            )

        assert len(distiller.call_args.args[0]) == 3
        # Overflow waits for the next sweep; it is not dropped.
        assert await count_queued(conn, channel_id=CHANNEL_A) == 4

    async def test_the_channel_name_reaches_the_distiller(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        await _queue(conn, message_id=1, content="x")
        distiller = AsyncMock(return_value=[])
        with patch("aura.extraction.pipeline.distill_facts", distiller):
            await flush_due_batches(conn, embedding_model, settings=_settings(), now=NOW)
        assert distiller.call_args.kwargs["channel_name"] == "announcements"

    async def test_a_batch_still_inside_its_window_is_not_touched(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        # The clock is injected precisely so this is testable at the moment it
        # matters rather than depending on the wall clock at test time.
        await _queue(conn, message_id=1, content="x")
        settings = _settings(extraction_batch_window_seconds=300.0)

        with patch("aura.extraction.pipeline.distill_facts", AsyncMock()) as distiller:
            flushed = await flush_due_batches(
                conn, embedding_model, settings=settings, now=NOW + timedelta(seconds=299)
            )

        assert flushed == 0
        distiller.assert_not_awaited()
        assert await count_queued(conn) == 1

    async def test_the_same_batch_flushes_once_its_window_closes(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        await _queue(conn, message_id=1, content="x")
        settings = _settings(extraction_batch_window_seconds=300.0)

        with patch(
            "aura.extraction.pipeline.distill_facts", AsyncMock(return_value=[])
        ) as distiller:
            flushed = await flush_due_batches(
                conn, embedding_model, settings=settings, now=NOW + timedelta(seconds=300)
            )

        assert flushed == 1
        distiller.assert_awaited_once()
        assert await count_queued(conn) == 0

    async def test_one_channels_failure_does_not_starve_another(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        await _queue(conn, message_id=1, content="channel A")
        await enqueue_message(
            conn,
            guild_id=GUILD_A,
            channel_id=999,
            message_id=2,
            channel_name="other",
            content="channel B",
            message_created_at=NOW,
            now=NOW,
        )

        async def _explode_for_the_first_channel(batch, **kwargs):
            if batch[0].channel_id == CHANNEL_A:
                raise RuntimeError("boom")
            return []

        with patch(
            "aura.extraction.pipeline.distill_facts",
            AsyncMock(side_effect=_explode_for_the_first_channel),
        ):
            await flush_due_batches(conn, embedding_model, settings=_settings(), now=NOW)

        # The exploding channel keeps its batch; the healthy one is cleared.
        assert await count_queued(conn, channel_id=CHANNEL_A) == 1
        assert await count_queued(conn, channel_id=999) == 0


class TestDailyCapBehaviour:
    async def test_a_batch_is_dropped_rather_than_held_once_the_cap_is_reached(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        # Holding it would accumulate raw message text for the rest of the UTC
        # day and then release a flood of calls at midnight. "No more spend
        # today" has to mean today's batches are gone, not merely postponed.
        await _queue(conn, message_id=1, content="something")
        with patch("aura.extraction.pipeline.distill_facts", AsyncMock()) as distiller:
            await flush_due_batches(
                conn, embedding_model, settings=_settings(extraction_daily_cap=0), now=NOW
            )

        distiller.assert_not_awaited()
        assert await count_queued(conn) == 0
        assert await get_pending_facts(conn, guild_id=GUILD_A, limit=10) == []

    async def test_the_cap_bounds_the_number_of_calls_in_one_day(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        settings = _settings(extraction_daily_cap=2)
        distiller = AsyncMock(return_value=[])

        with patch("aura.extraction.pipeline.distill_facts", distiller):
            for channel_id in range(900, 905):
                await enqueue_message(
                    conn,
                    guild_id=GUILD_A,
                    channel_id=channel_id,
                    message_id=channel_id,
                    channel_name="c",
                    content="x",
                    message_created_at=NOW,
                    now=NOW,
                )
            await flush_due_batches(conn, embedding_model, settings=settings, now=NOW)

        assert distiller.await_count == 2
        assert (
            await count_extraction_calls_on(conn, guild_id=GUILD_A, day=utc_day(NOW)) == 2
        )


class TestDedupHint:
    async def test_a_near_duplicate_candidate_is_flagged_with_its_predecessor(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        existing = await add_fact(
            conn,
            embedding_model,
            guild_id=GUILD_A,
            channel_id=CHANNEL_A,
            message_id=900,
            content="Der Server wird um 14 Uhr einer Wartung ausgesetzt.",
        )
        await _queue(conn, message_id=1, content="wartung heute")

        distilled = [
            DistilledFact(
                message_id=1,
                content="Der Server wird heute um 14 Uhr gewartet.",
                category=FactCategory.STATUS_CHANGE,
            )
        ]
        with patch(
            "aura.extraction.pipeline.distill_facts", AsyncMock(return_value=distilled)
        ):
            await flush_due_batches(conn, embedding_model, settings=_settings(), now=NOW)

        staged = await get_pending_facts(conn, guild_id=GUILD_A, limit=10)
        assert len(staged) == 1
        assert staged[0].similar_fact_id == existing.id
        assert staged[0].similar_fact_score is not None
        assert staged[0].similar_fact_score >= 0.70

    async def test_an_unrelated_candidate_is_not_flagged(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        await add_fact(
            conn,
            embedding_model,
            guild_id=GUILD_A,
            channel_id=CHANNEL_A,
            message_id=900,
            content="Der Server wird um 14 Uhr einer Wartung ausgesetzt.",
        )
        await _queue(conn, message_id=1, content="turnier")

        distilled = [
            DistilledFact(
                message_id=1,
                content="The winter tournament starts on Saturday at 18:00 in #events.",
                category=FactCategory.EVENT,
            )
        ]
        with patch(
            "aura.extraction.pipeline.distill_facts", AsyncMock(return_value=distilled)
        ):
            await flush_due_batches(conn, embedding_model, settings=_settings(), now=NOW)

        staged = await get_pending_facts(conn, guild_id=GUILD_A, limit=10)
        assert staged[0].similar_fact_id is None
        assert staged[0].similar_fact_score is None

    async def test_a_flagged_candidate_does_not_supersede_anything_automatically(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        # Explicitly out of scope for this sub-phase: the hint is advisory and
        # the replacement decision stays with the moderator until Phase 3a-3.
        existing = await add_fact(
            conn,
            embedding_model,
            guild_id=GUILD_A,
            channel_id=CHANNEL_A,
            message_id=900,
            content="Der Server wird um 14 Uhr einer Wartung ausgesetzt.",
        )
        await _queue(conn, message_id=1, content="wartung")
        distilled = [
            DistilledFact(
                message_id=1,
                content="Der Server wird heute um 14 Uhr gewartet.",
                category=FactCategory.STATUS_CHANGE,
            )
        ]
        with patch(
            "aura.extraction.pipeline.distill_facts", AsyncMock(return_value=distilled)
        ):
            await flush_due_batches(conn, embedding_model, settings=_settings(), now=NOW)

        active = await get_active_facts(conn, GUILD_A)
        assert [fact.id for fact in active] == [existing.id]
        assert active[0].superseded_by_id is None

    async def test_an_empty_knowledge_model_flags_nothing(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        await _queue(conn, message_id=1, content="x")
        distilled = [
            DistilledFact(
                message_id=1, content="A brand new fact.", category=FactCategory.ANNOUNCEMENT
            )
        ]
        with patch(
            "aura.extraction.pipeline.distill_facts", AsyncMock(return_value=distilled)
        ):
            await flush_due_batches(conn, embedding_model, settings=_settings(), now=NOW)

        staged = await get_pending_facts(conn, guild_id=GUILD_A, limit=10)
        assert staged[0].similar_fact_id is None


class TestSupersessionJudgement:
    """Phase 3a-3's paid call, and the two properties that keep it bounded.

    The judgement itself is mocked here (its own prompt and parsing live in
    tests/test_supersession_judge.py, and its behaviour against a real model in
    scripts/supersession_reverify.py). What this class checks is the wiring: WHO
    gets judged, WHO does not, and what happens when the answer never arrives.
    """

    @staticmethod
    async def _maintenance_fact(conn: aiosqlite.Connection, model: TextEmbedding):
        return await add_fact(
            conn,
            model,
            guild_id=GUILD_A,
            channel_id=CHANNEL_A,
            message_id=900,
            content="Der Server wird um 14 Uhr einer Wartung ausgesetzt.",
        )

    @staticmethod
    def _judgement(
        relationship: SupersessionRelationship = SupersessionRelationship.SUPERSESSION,
        reasoning: str = "Beide nennen dieselbe Wartung zur selben Uhrzeit.",
    ) -> RelationshipJudgement:
        return RelationshipJudgement(
            relationship=relationship,
            reasoning=reasoning,
            change_signal="wurde verschoben",
        )

    async def test_a_flagged_candidate_is_judged_and_the_answer_is_stored(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        await self._maintenance_fact(conn, embedding_model)
        await _queue(conn, message_id=1, content="wartung heute")
        distilled = [
            DistilledFact(
                message_id=1,
                content="Der Server wird heute um 14 Uhr gewartet.",
                category=FactCategory.STATUS_CHANGE,
            )
        ]
        judge = AsyncMock(return_value=self._judgement())
        with (
            patch("aura.extraction.pipeline.distill_facts", AsyncMock(return_value=distilled)),
            patch("aura.extraction.pipeline.judge_relationship", judge),
        ):
            await flush_due_batches(conn, embedding_model, settings=_settings(), now=NOW)

        judge.assert_awaited_once()
        staged = await get_pending_facts(conn, guild_id=GUILD_A, limit=10)
        assert staged[0].relationship is SupersessionRelationship.SUPERSESSION
        assert staged[0].relationship_reasoning == (
            "Beide nennen dieselbe Wartung zur selben Uhrzeit."
        )
        assert await count_supersession_calls_on(
            conn, guild_id=GUILD_A, day=utc_day(NOW)
        ) == 1

    async def test_the_self_consistency_fields_are_never_stored(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        # change_signal and shared_subject exist to make the model commit to its
        # evidence before choosing a category. Storing them would imply Aura
        # relies on them afterwards; nothing does, and the whole row is checked
        # rather than the two columns that exist today.
        await self._maintenance_fact(conn, embedding_model)
        await _queue(conn, message_id=1, content="wartung heute")
        distilled = [
            DistilledFact(
                message_id=1,
                content="Der Server wird heute um 14 Uhr gewartet.",
                category=FactCategory.STATUS_CHANGE,
            )
        ]
        judgement = RelationshipJudgement(
            relationship=SupersessionRelationship.SUPERSESSION,
            reasoning="Der Wartungstag hat sich geändert.",
            change_signal="wurde verschoben auf",
            shared_subject="die Wartung des Servers",
        )
        with (
            patch("aura.extraction.pipeline.distill_facts", AsyncMock(return_value=distilled)),
            patch(
                "aura.extraction.pipeline.judge_relationship",
                AsyncMock(return_value=judgement),
            ),
        ):
            await flush_due_batches(conn, embedding_model, settings=_settings(), now=NOW)

        async with conn.execute("SELECT * FROM pending_facts") as cursor:
            row = await cursor.fetchone()
        assert row is not None
        stored = " ".join(str(value) for value in row)
        assert "wurde verschoben auf" not in stored
        assert "die Wartung des Servers" not in stored
        assert "Der Wartungstag hat sich geändert." in stored

    async def test_the_judge_sees_exactly_the_two_sentences(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        # "Judgment, never knowledge": the pair, and nothing else about the
        # guild, reaches the call.
        predecessor = await self._maintenance_fact(conn, embedding_model)
        await _queue(conn, message_id=1, content="wartung heute")
        distilled = [
            DistilledFact(
                message_id=1,
                content="Der Server wird heute um 14 Uhr gewartet.",
                category=FactCategory.STATUS_CHANGE,
            )
        ]
        judge = AsyncMock(return_value=self._judgement())
        with (
            patch("aura.extraction.pipeline.distill_facts", AsyncMock(return_value=distilled)),
            patch("aura.extraction.pipeline.judge_relationship", judge),
        ):
            await flush_due_batches(conn, embedding_model, settings=_settings(), now=NOW)

        kwargs = judge.call_args.kwargs
        assert kwargs["predecessor"] == predecessor.content
        assert kwargs["candidate"] == distilled[0].content
        assert set(kwargs) == {"predecessor", "candidate", "model"}

    async def test_an_unflagged_candidate_is_never_judged(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        # The cost leak the phase brief asks to be verified by test rather than
        # by reading: everything below the dedup threshold must cost nothing.
        await self._maintenance_fact(conn, embedding_model)
        await _queue(conn, message_id=1, content="turnier")
        distilled = [
            DistilledFact(
                message_id=1,
                content="The winter tournament starts on Saturday at 18:00 in #events.",
                category=FactCategory.EVENT,
            )
        ]
        judge = AsyncMock()
        with (
            patch("aura.extraction.pipeline.distill_facts", AsyncMock(return_value=distilled)),
            patch("aura.extraction.pipeline.judge_relationship", judge),
        ):
            await flush_due_batches(conn, embedding_model, settings=_settings(), now=NOW)

        judge.assert_not_awaited()
        staged = await get_pending_facts(conn, guild_id=GUILD_A, limit=10)
        assert staged[0].relationship is None
        assert await count_supersession_calls_on(
            conn, guild_id=GUILD_A, day=utc_day(NOW)
        ) == 0

    async def test_a_guild_with_no_facts_yet_never_judges_anything(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        await _queue(conn, message_id=1, content="x")
        distilled = [
            DistilledFact(
                message_id=1, content="A brand new fact.", category=FactCategory.ANNOUNCEMENT
            )
        ]
        judge = AsyncMock()
        with (
            patch("aura.extraction.pipeline.distill_facts", AsyncMock(return_value=distilled)),
            patch("aura.extraction.pipeline.judge_relationship", judge),
        ):
            await flush_due_batches(conn, embedding_model, settings=_settings(), now=NOW)

        judge.assert_not_awaited()

    async def test_only_the_flagged_candidate_of_a_mixed_batch_is_judged(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        # One call per FLAGGED candidate, not one per candidate: the second and
        # third sentences here are about entirely different things and must not
        # each buy a judgement.
        await self._maintenance_fact(conn, embedding_model)
        for message_id in (1, 2, 3):
            await _queue(conn, message_id=message_id, content=f"m{message_id}")
        distilled = [
            DistilledFact(
                message_id=1,
                content="Der Server wird heute um 14 Uhr gewartet.",
                category=FactCategory.STATUS_CHANGE,
            ),
            DistilledFact(
                message_id=2,
                content="The art contest submissions close on August 30.",
                category=FactCategory.EVENT,
            ),
            DistilledFact(
                message_id=3,
                content="New members must react to the rules message.",
                category=FactCategory.RULE,
            ),
        ]
        judge = AsyncMock(return_value=self._judgement())
        with (
            patch("aura.extraction.pipeline.distill_facts", AsyncMock(return_value=distilled)),
            patch("aura.extraction.pipeline.judge_relationship", judge),
        ):
            await flush_due_batches(conn, embedding_model, settings=_settings(), now=NOW)

        assert judge.await_count == 1
        assert await count_supersession_calls_on(
            conn, guild_id=GUILD_A, day=utc_day(NOW)
        ) == 1
        staged = await get_pending_facts(conn, guild_id=GUILD_A, limit=10)
        judged = [c for c in staged if c.relationship is not None]
        assert len(judged) == 1
        assert judged[0].similar_fact_id is not None

    async def test_a_re_distilled_batch_does_not_pay_for_the_same_judgement_twice(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        # The crash-retry path spends a second EXTRACTION slot deliberately, but
        # the candidate it re-produces is absorbed by the UNIQUE constraint --
        # and an already-staged candidate must not buy a second judgement.
        await self._maintenance_fact(conn, embedding_model)
        await _queue(conn, message_id=1, content="wartung heute")
        distilled = [
            DistilledFact(
                message_id=1,
                content="Der Server wird heute um 14 Uhr gewartet.",
                category=FactCategory.STATUS_CHANGE,
            )
        ]
        judge = AsyncMock(return_value=self._judgement())
        with (
            patch("aura.extraction.pipeline.distill_facts", AsyncMock(return_value=distilled)),
            patch("aura.extraction.pipeline.judge_relationship", judge),
        ):
            await flush_due_batches(conn, embedding_model, settings=_settings(), now=NOW)
            await _queue(conn, message_id=1, content="wartung heute")
            await flush_due_batches(conn, embedding_model, settings=_settings(), now=NOW)

        assert judge.await_count == 1
        assert await count_extraction_calls_on(
            conn, guild_id=GUILD_A, day=utc_day(NOW)
        ) == 2
        assert await count_supersession_calls_on(
            conn, guild_id=GUILD_A, day=utc_day(NOW)
        ) == 1

    async def test_the_daily_cap_stops_judging_without_stopping_extraction(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        # The softer failure this cap has, compared with extraction's: when it
        # binds, the candidate is still staged and still reviewable -- it simply
        # carries the plain similarity hint.
        await self._maintenance_fact(conn, embedding_model)
        await _queue(conn, message_id=1, content="wartung heute")
        distilled = [
            DistilledFact(
                message_id=1,
                content="Der Server wird heute um 14 Uhr gewartet.",
                category=FactCategory.STATUS_CHANGE,
            )
        ]
        judge = AsyncMock(return_value=self._judgement())
        with (
            patch("aura.extraction.pipeline.distill_facts", AsyncMock(return_value=distilled)),
            patch("aura.extraction.pipeline.judge_relationship", judge),
        ):
            await flush_due_batches(
                conn,
                embedding_model,
                settings=_settings(supersession_daily_cap=0),
                now=NOW,
            )

        judge.assert_not_awaited()
        staged = await get_pending_facts(conn, guild_id=GUILD_A, limit=10)
        assert len(staged) == 1
        assert staged[0].relationship is None
        assert staged[0].similar_fact_id is not None

    async def test_the_cap_bounds_judgements_across_several_flushes(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        await self._maintenance_fact(conn, embedding_model)
        judge = AsyncMock(return_value=self._judgement())
        with (
            patch("aura.extraction.pipeline.judge_relationship", judge),
        ):
            for message_id in (1, 2, 3, 4):
                await _queue(conn, message_id=message_id, content=f"wartung {message_id}")
                distilled = [
                    DistilledFact(
                        message_id=message_id,
                        content=f"Der Server wird heute um 14 Uhr gewartet ({message_id}).",
                        category=FactCategory.STATUS_CHANGE,
                    )
                ]
                with patch(
                    "aura.extraction.pipeline.distill_facts",
                    AsyncMock(return_value=distilled),
                ):
                    await flush_due_batches(
                        conn,
                        embedding_model,
                        settings=_settings(supersession_daily_cap=2),
                        now=NOW,
                    )

        assert judge.await_count == 2
        assert await count_supersession_calls_on(
            conn, guild_id=GUILD_A, day=utc_day(NOW)
        ) == 2
        # All four candidates are still staged and reviewable.
        assert len(await get_pending_facts(conn, guild_id=GUILD_A, limit=10)) == 4

    async def test_no_supersession_model_configured_means_no_call_and_no_spend(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        await self._maintenance_fact(conn, embedding_model)
        await _queue(conn, message_id=1, content="wartung heute")
        distilled = [
            DistilledFact(
                message_id=1,
                content="Der Server wird heute um 14 Uhr gewartet.",
                category=FactCategory.STATUS_CHANGE,
            )
        ]
        judge = AsyncMock()
        # Extraction has its own model, so the batch is still distilled; only
        # the judgement has nothing to resolve.
        # synthesis_model is pinned to None explicitly rather than left out:
        # this repository's real .env is loaded into the process environment the
        # moment litellm is imported, and SUPERSESSION falls back to synthesis,
        # so an omitted value here would silently BE configured.
        settings = Settings(  # type: ignore[call-arg]
            _env_file=None,
            discord_token="fake-token",
            llm_api_key="test-key",
            synthesis_model=None,
            extraction_model="openrouter/anthropic/claude-haiku-4.5",
            extraction_batch_window_seconds=0.0,
        )
        assert settings.resolve_model(ModelComponent.SUPERSESSION) is None
        with (
            patch("aura.extraction.pipeline.distill_facts", AsyncMock(return_value=distilled)),
            patch("aura.extraction.pipeline.judge_relationship", judge),
        ):
            await flush_due_batches(conn, embedding_model, settings=settings, now=NOW)

        judge.assert_not_awaited()
        staged = await get_pending_facts(conn, guild_id=GUILD_A, limit=10)
        assert len(staged) == 1
        assert staged[0].relationship is None
        assert await count_supersession_calls_on(
            conn, guild_id=GUILD_A, day=utc_day(NOW)
        ) == 0

    async def test_a_failed_judgement_leaves_the_candidate_intact(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        await self._maintenance_fact(conn, embedding_model)
        await _queue(conn, message_id=1, content="wartung heute")
        distilled = [
            DistilledFact(
                message_id=1,
                content="Der Server wird heute um 14 Uhr gewartet.",
                category=FactCategory.STATUS_CHANGE,
            )
        ]
        with (
            patch("aura.extraction.pipeline.distill_facts", AsyncMock(return_value=distilled)),
            patch(
                "aura.extraction.pipeline.judge_relationship", AsyncMock(return_value=None)
            ),
        ):
            await flush_due_batches(conn, embedding_model, settings=_settings(), now=NOW)

        staged = await get_pending_facts(conn, guild_id=GUILD_A, limit=10)
        assert len(staged) == 1
        assert staged[0].relationship is None
        assert staged[0].similar_fact_id is not None
        assert await count_queued(conn) == 0

    async def test_an_exploding_judgement_does_not_lose_the_batch(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding, caplog
    ) -> None:
        # judge_relationship is documented never to raise, so this is the
        # unanticipated case -- a bug in it, or in the write after it. The batch
        # and its candidates must survive regardless.
        await self._maintenance_fact(conn, embedding_model)
        await _queue(conn, message_id=1, content="wartung heute")
        distilled = [
            DistilledFact(
                message_id=1,
                content="Der Server wird heute um 14 Uhr gewartet.",
                category=FactCategory.STATUS_CHANGE,
            )
        ]
        with (
            patch("aura.extraction.pipeline.distill_facts", AsyncMock(return_value=distilled)),
            patch(
                "aura.extraction.pipeline.judge_relationship",
                AsyncMock(side_effect=RuntimeError("boom")),
            ),
            caplog.at_level("ERROR"),
        ):
            await flush_due_batches(conn, embedding_model, settings=_settings(), now=NOW)

        staged = await get_pending_facts(conn, guild_id=GUILD_A, limit=10)
        assert len(staged) == 1
        assert staged[0].relationship is None
        assert await count_queued(conn) == 0
        assert any(record.levelname == "ERROR" for record in caplog.records)

    async def test_a_failed_judgement_still_spends_its_slot(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        # The conservative direction, and the same one both other ledgers chose:
        # a slot is claimed before the call it authorizes, so a reliably-failing
        # model cannot earn unlimited retries out of the day's budget.
        await self._maintenance_fact(conn, embedding_model)
        await _queue(conn, message_id=1, content="wartung heute")
        distilled = [
            DistilledFact(
                message_id=1,
                content="Der Server wird heute um 14 Uhr gewartet.",
                category=FactCategory.STATUS_CHANGE,
            )
        ]
        with (
            patch("aura.extraction.pipeline.distill_facts", AsyncMock(return_value=distilled)),
            patch(
                "aura.extraction.pipeline.judge_relationship", AsyncMock(return_value=None)
            ),
        ):
            await flush_due_batches(conn, embedding_model, settings=_settings(), now=NOW)

        assert await count_supersession_calls_on(
            conn, guild_id=GUILD_A, day=utc_day(NOW)
        ) == 1

    async def test_two_sweeps_racing_on_one_flagged_batch_judge_it_once(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        # Not reachable from the single sweeper task, but a second process
        # sharing the database file always is. The staging table's UNIQUE
        # constraint is what has to hold -- and because the judgement fires only
        # for a candidate this call actually staged, the loser pays for nothing.
        await self._maintenance_fact(conn, embedding_model)
        await _queue(conn, message_id=1, content="wartung heute")
        distilled = [
            DistilledFact(
                message_id=1,
                content="Der Server wird heute um 14 Uhr gewartet.",
                category=FactCategory.STATUS_CHANGE,
            )
        ]
        judge = AsyncMock(return_value=self._judgement())
        with (
            patch("aura.extraction.pipeline.distill_facts", AsyncMock(return_value=distilled)),
            patch("aura.extraction.pipeline.judge_relationship", judge),
        ):
            await asyncio.gather(
                flush_due_batches(conn, embedding_model, settings=_settings(), now=NOW),
                flush_due_batches(conn, embedding_model, settings=_settings(), now=NOW),
            )

        assert judge.await_count == 1
        assert await count_supersession_calls_on(
            conn, guild_id=GUILD_A, day=utc_day(NOW)
        ) == 1
        assert len(await get_pending_facts(conn, guild_id=GUILD_A, limit=10)) == 1

    async def test_the_database_lock_is_not_held_across_the_paid_call(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        # CLAUDE.md's performance rule, checked where it is easiest to get
        # wrong: this call can take a minute, and holding the connection lock
        # for its duration would stall every command in every guild behind one
        # advisory judgement. The test blocks inside the judge and requires an
        # unrelated database read to complete anyway.
        await self._maintenance_fact(conn, embedding_model)
        await _queue(conn, message_id=1, content="wartung heute")
        distilled = [
            DistilledFact(
                message_id=1,
                content="Der Server wird heute um 14 Uhr gewartet.",
                category=FactCategory.STATUS_CHANGE,
            )
        ]
        in_flight = asyncio.Event()
        release = asyncio.Event()

        async def _slow_judge(**_kwargs):
            in_flight.set()
            await release.wait()
            return self._judgement()

        with (
            patch("aura.extraction.pipeline.distill_facts", AsyncMock(return_value=distilled)),
            patch("aura.extraction.pipeline.judge_relationship", _slow_judge),
        ):
            flush = asyncio.create_task(
                flush_due_batches(conn, embedding_model, settings=_settings(), now=NOW)
            )
            await asyncio.wait_for(in_flight.wait(), timeout=5)
            # The judgement is mid-call. An ordinary read must not be queued
            # behind it.
            assert await asyncio.wait_for(get_active_facts(conn, GUILD_A), timeout=5)
            release.set()
            await asyncio.wait_for(flush, timeout=5)

    async def test_cancellation_during_a_judgement_still_propagates(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        # The judgement's own error handling catches Exception, not
        # BaseException, so a shutdown cancelling the sweeper must travel
        # through it rather than being logged as "the judgement failed" and
        # leaving the task alive.
        await self._maintenance_fact(conn, embedding_model)
        await _queue(conn, message_id=1, content="wartung heute")
        distilled = [
            DistilledFact(
                message_id=1,
                content="Der Server wird heute um 14 Uhr gewartet.",
                category=FactCategory.STATUS_CHANGE,
            )
        ]
        with (
            patch("aura.extraction.pipeline.distill_facts", AsyncMock(return_value=distilled)),
            patch(
                "aura.extraction.pipeline.judge_relationship",
                AsyncMock(side_effect=asyncio.CancelledError()),
            ),
        ):
            with pytest.raises(asyncio.CancelledError):
                await flush_due_batches(
                    conn, embedding_model, settings=_settings(), now=NOW
                )

    async def test_a_judgement_never_supersedes_a_fact(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        # The single most important property of this sub-phase, and the reason
        # the pipeline may propose at all: even the strongest possible verdict
        # writes nothing to `facts`.
        existing = await self._maintenance_fact(conn, embedding_model)
        await _queue(conn, message_id=1, content="wartung heute")
        distilled = [
            DistilledFact(
                message_id=1,
                content="Der Server wird heute um 14 Uhr gewartet.",
                category=FactCategory.STATUS_CHANGE,
            )
        ]
        with (
            patch("aura.extraction.pipeline.distill_facts", AsyncMock(return_value=distilled)),
            patch(
                "aura.extraction.pipeline.judge_relationship",
                AsyncMock(return_value=self._judgement()),
            ),
        ):
            await flush_due_batches(conn, embedding_model, settings=_settings(), now=NOW)

        active = await get_active_facts(conn, GUILD_A)
        assert [fact.id for fact in active] == [existing.id]
        assert active[0].superseded_by_id is None
        assert active[0].status is FactStatus.ACTIVE


class TestRetrySafety:
    async def test_re_distilling_the_same_batch_does_not_duplicate_candidates(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        # The crash-retry path: a process that died between claiming a slot and
        # clearing the queue re-does the batch on the next sweep. The same
        # messages produce the same sentences, and those must land as the same
        # candidate rather than as a second one to reject by hand.
        await _queue(conn, message_id=1, content="server down at 2")
        distilled = [
            DistilledFact(
                message_id=1,
                content="Maintenance runs today at 14:00 UTC.",
                category=FactCategory.STATUS_CHANGE,
            )
        ]

        with patch(
            "aura.extraction.pipeline.distill_facts", AsyncMock(return_value=distilled)
        ):
            await flush_due_batches(conn, embedding_model, settings=_settings(), now=NOW)
            # Simulate the retry: the same messages arrive back in the queue.
            await _queue(conn, message_id=1, content="server down at 2")
            await flush_due_batches(conn, embedding_model, settings=_settings(), now=NOW)

        assert len(await get_pending_facts(conn, guild_id=GUILD_A, limit=10)) == 1
        # The retry did spend a second slot -- deliberate, and the conservative
        # direction for a spend limit.
        assert (
            await count_extraction_calls_on(conn, guild_id=GUILD_A, day=utc_day(NOW)) == 2
        )

    async def test_a_candidate_citing_a_message_outside_its_batch_is_skipped_not_fatal(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        # Unreachable through the real distiller (validation maps every number
        # back through the batch), but the alternative to checking is a
        # KeyError that loses the batch's other, good candidates.
        await _queue(conn, message_id=1, content="a")
        distilled = [
            DistilledFact(
                message_id=9999, content="From nowhere.", category=FactCategory.RULE
            ),
            DistilledFact(
                message_id=1, content="A genuine fact.", category=FactCategory.RULE
            ),
        ]
        with patch(
            "aura.extraction.pipeline.distill_facts", AsyncMock(return_value=distilled)
        ):
            await flush_due_batches(conn, embedding_model, settings=_settings(), now=NOW)

        staged = await get_pending_facts(conn, guild_id=GUILD_A, limit=10)
        assert [candidate.content for candidate in staged] == ["A genuine fact."]


class TestSweepInterval:
    @pytest.mark.parametrize(
        "window,expected",
        [(0.0, 1.0), (2.0, 1.0), (60.0, 30.0), (300.0, 30.0), (86400.0, 30.0)],
    )
    def test_the_interval_stays_inside_its_bounds(
        self, window: float, expected: float
    ) -> None:
        # Bounded below so a tiny window cannot spin the loop, and above so a
        # long window does not make the sweeper effectively idle.
        assert sweep_interval_seconds(window) == expected


class TestConcurrentFlushes:
    async def test_two_sweeps_racing_on_one_batch_do_not_duplicate_candidates(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        # Not reachable from the single sweeper task, but a second process
        # sharing the database file always is -- and the staging table's
        # UNIQUE constraint is what has to hold, not the loop's cardinality.
        await _queue(conn, message_id=1, content="server down at 2")
        distilled = [
            DistilledFact(
                message_id=1,
                content="Maintenance runs today at 14:00 UTC.",
                category=FactCategory.STATUS_CHANGE,
            )
        ]
        with patch(
            "aura.extraction.pipeline.distill_facts", AsyncMock(return_value=distilled)
        ):
            await asyncio.gather(
                flush_due_batches(conn, embedding_model, settings=_settings(), now=NOW),
                flush_due_batches(conn, embedding_model, settings=_settings(), now=NOW),
            )

        assert len(await get_pending_facts(conn, guild_id=GUILD_A, limit=10)) == 1
        assert await get_active_facts(conn, GUILD_A) == []
