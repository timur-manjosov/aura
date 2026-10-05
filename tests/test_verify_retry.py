"""A batch whose extraction verification failed is tried again later, never silently lost.

Covers aura.extraction.verify_retry and both paths that use it: the live
extraction pipeline (the batch stays queued) and backfill (the cursor stays).
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import aiosqlite
import pytest

from aura.billing import PlanGate
from aura.db.connection import utc_day
from aura.db.extraction_queue import count_queued
from aura.db.extraction_state import count_extraction_calls_on
from aura.db.pending_facts import FactCategory, get_pending_facts
from aura.db.repository import init_schema
from aura.extraction import verify_retry
from aura.extraction.distiller import DistilledFact
from aura.extraction.pipeline import flush_due_batches, withdraw_message
from aura.extraction.verify_retry import (
    VERIFICATION_RETRIES,
    VerificationRetries,
    backfill_key,
    live_key,
)
from tests.test_extraction_pipeline import CHANNEL_A, GUILD_A, _queue
from tests.test_extraction_pipeline import NOW as PIPELINE_NOW
from tests.test_extraction_pipeline import _settings as pipeline_settings
from tests.test_extraction_verifier import _check, _response

VERIFY_MODEL = "openrouter/z-ai/glm-5.3-flash"
T0 = datetime(2026, 10, 6, 12, 0, tzinfo=UTC)
TIMEOUT = TimeoutError("provider unavailable")


@pytest.fixture
async def conn():
    connection = await aiosqlite.connect(":memory:")
    await init_schema(connection)
    yield connection
    await connection.close()


class TestTheTracker:
    def test_a_first_failure_waits_the_base_delay(self) -> None:
        retries = VerificationRetries()

        assert retries.record_failure(("live", 1), T0, max_attempts=4, base_delay_seconds=600) == 1

        assert retries.is_waiting(("live", 1), T0 + timedelta(seconds=599))
        assert not retries.is_waiting(("live", 1), T0 + timedelta(seconds=600))

    def test_the_pause_doubles_after_each_failure(self) -> None:
        retries = VerificationRetries()
        retries.record_failure(("live", 1), T0, max_attempts=4, base_delay_seconds=600)
        later = T0 + timedelta(seconds=600)

        assert (
            retries.record_failure(("live", 1), later, max_attempts=4, base_delay_seconds=600) == 2
        )

        assert retries.is_waiting(("live", 1), later + timedelta(seconds=1199))
        assert not retries.is_waiting(("live", 1), later + timedelta(seconds=1200))

    def test_the_last_attempt_gives_up_and_forgets_the_key(self) -> None:
        retries = VerificationRetries()
        for expected in (1, 2, 3):
            assert (
                retries.record_failure(("live", 1), T0, max_attempts=4, base_delay_seconds=1)
                == expected
            )

        assert retries.record_failure(("live", 1), T0, max_attempts=4, base_delay_seconds=1) is None
        assert retries.failures(("live", 1)) == 0
        assert not retries.is_waiting(("live", 1), T0)

    def test_one_attempt_means_no_retry(self) -> None:
        retries = VerificationRetries()

        assert retries.record_failure(("live", 1), T0, max_attempts=1, base_delay_seconds=1) is None

    def test_live_and_backfill_keys_with_the_same_number_are_independent(self) -> None:
        retries = VerificationRetries()
        retries.record_failure(live_key(7), T0, max_attempts=4, base_delay_seconds=60)

        assert retries.is_waiting(live_key(7), T0)
        assert not retries.is_waiting(backfill_key(7), T0)

    def test_an_unknown_key_is_never_waiting(self) -> None:
        assert not VerificationRetries().is_waiting(("live", 99), T0)

    def test_the_number_of_keys_is_bounded_and_the_oldest_is_forgotten_first(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(verify_retry, "MAX_TRACKED_KEYS", 3)
        retries = VerificationRetries()
        for channel in range(5):
            retries.record_failure(live_key(channel), T0, max_attempts=4, base_delay_seconds=60)

        assert [retries.failures(live_key(channel)) for channel in range(5)] == [0, 0, 1, 1, 1]

    def test_the_largest_pause_stays_representable(self) -> None:
        retries = VerificationRetries()
        for _ in range(9):
            retries.record_failure(("live", 1), T0, max_attempts=10, base_delay_seconds=86400)

        assert retries.is_waiting(("live", 1), T0 + timedelta(days=255))


def _distilled(batch: list[Any]) -> list[DistilledFact]:
    return [
        DistilledFact(
            message_id=queued.message_id,
            content="Der Spieleabend ist am 31. Juli 2026 um 20 Uhr.",
            category=FactCategory.EVENT,
        )
        for queued in batch
    ]


class _Live:
    """Run sweeps of the live path with a scripted verifier."""

    def __init__(
        self, conn: aiosqlite.Connection, embedding_model: Any, **overrides: object
    ) -> None:
        self.conn = conn
        self.embedding_model = embedding_model
        self.settings = pipeline_settings(extraction_verify_model=VERIFY_MODEL, **overrides)
        self.distill = AsyncMock(side_effect=lambda batch, **_kwargs: _distilled(batch))
        self.verify = AsyncMock()

    async def sweep(self, now: datetime, *verify_replies: object) -> int:
        self.verify.side_effect = list(verify_replies) or None
        with (
            patch("aura.extraction.pipeline.distill_facts", self.distill),
            patch("aura.extraction.verifier.litellm.acompletion", self.verify),
        ):
            return await flush_due_batches(
                self.conn,
                self.embedding_model,
                settings=self.settings,
                now=now,
                plan_gate=PlanGate.unenforced(),
            )

    async def slots(self, now: datetime) -> int:
        return await count_extraction_calls_on(self.conn, guild_id=GUILD_A, day=utc_day(now))


def _keep_all() -> MagicMock:
    return _response({"checks": [_check(1)]})


class TestTheLivePath:
    async def test_a_held_batch_is_not_attempted_during_its_pause(
        self, conn: aiosqlite.Connection, embedding_model: Any
    ) -> None:
        live = _Live(conn, embedding_model)
        await _queue(conn, message_id=1, content="Morgen um 20 Uhr ist Spieleabend.")
        await live.sweep(PIPELINE_NOW, TIMEOUT)

        ran = await live.sweep(PIPELINE_NOW + timedelta(seconds=599))

        assert ran == 0
        assert live.distill.await_count == 1
        assert await live.slots(PIPELINE_NOW) == 1
        assert await count_queued(conn) == 1

    async def test_after_the_pause_the_batch_is_tried_again_and_staged(
        self, conn: aiosqlite.Connection, embedding_model: Any
    ) -> None:
        live = _Live(conn, embedding_model)
        await _queue(conn, message_id=1, content="Morgen um 20 Uhr ist Spieleabend.")
        await live.sweep(PIPELINE_NOW, TIMEOUT)

        ran = await live.sweep(PIPELINE_NOW + timedelta(seconds=600), _keep_all())

        assert ran == 1
        assert [
            fact.message_id for fact in await get_pending_facts(conn, guild_id=GUILD_A, limit=5)
        ] == [1]
        assert await count_queued(conn) == 0
        assert VERIFICATION_RETRIES.failures(live_key(CHANNEL_A)) == 0

    async def test_a_verification_that_keeps_failing_is_given_up_loudly_after_every_attempt(
        self, conn: aiosqlite.Connection, embedding_model: Any, caplog: pytest.LogCaptureFixture
    ) -> None:
        live = _Live(conn, embedding_model)
        await _queue(conn, message_id=1, content="Morgen um 20 Uhr ist Spieleabend.")
        now = PIPELINE_NOW
        with caplog.at_level(logging.WARNING):
            for _ in range(4):
                await live.sweep(now, TIMEOUT)
                now += timedelta(hours=2)

        assert live.distill.await_count == 4
        assert await live.slots(PIPELINE_NOW) == 4
        assert await count_queued(conn) == 0
        assert await get_pending_facts(conn, guild_id=GUILD_A, limit=5) == []
        errors = [
            record
            for record in caplog.records
            if record.levelno == logging.ERROR and record.name == "aura.extraction.pipeline"
        ]
        assert len(errors) == 1
        assert "failed on all 4 attempt(s)" in errors[0].getMessage()

    @pytest.mark.parametrize(
        "reply",
        [
            _response(None, raw="not json at all {"),
            _response({"checks": []}),
            _response({"checks": [_check(1)]}, finish_reason="length"),
        ],
        ids=["unparsable", "missing-check", "cut-off"],
    )
    async def test_an_unusable_reply_is_not_retried_so_a_crafted_batch_costs_one_slot(
        self, conn: aiosqlite.Connection, embedding_model: Any, reply: MagicMock
    ) -> None:
        live = _Live(conn, embedding_model)
        await _queue(conn, message_id=1, content="Morgen um 20 Uhr ist Spieleabend.")

        await live.sweep(PIPELINE_NOW, reply)

        assert await count_queued(conn) == 0
        assert await live.slots(PIPELINE_NOW) == 1
        assert VERIFICATION_RETRIES.failures(live_key(CHANNEL_A)) == 0

    async def test_a_failed_distillation_is_still_cleared_at_once(
        self, conn: aiosqlite.Connection, embedding_model: Any
    ) -> None:
        live = _Live(conn, embedding_model)
        live.distill.side_effect = lambda *_args, **_kwargs: None
        await _queue(conn, message_id=1, content="Morgen um 20 Uhr ist Spieleabend.")

        await live.sweep(PIPELINE_NOW)

        assert await count_queued(conn) == 0
        assert live.verify.await_count == 0

    async def test_a_batch_the_model_judges_empty_is_cleared_without_verification(
        self, conn: aiosqlite.Connection, embedding_model: Any
    ) -> None:
        live = _Live(conn, embedding_model)
        live.distill.side_effect = lambda *_args, **_kwargs: []
        await _queue(conn, message_id=1, content="lol")

        await live.sweep(PIPELINE_NOW)

        assert await count_queued(conn) == 0
        assert live.verify.await_count == 0

    async def test_a_retry_refused_by_the_daily_cap_is_dropped_and_forgotten(
        self, conn: aiosqlite.Connection, embedding_model: Any
    ) -> None:
        live = _Live(conn, embedding_model, extraction_daily_cap=1)
        await _queue(conn, message_id=1, content="Morgen um 20 Uhr ist Spieleabend.")
        await live.sweep(PIPELINE_NOW, TIMEOUT)

        await live.sweep(PIPELINE_NOW + timedelta(seconds=600))

        assert live.distill.await_count == 1
        assert await count_queued(conn) == 0
        assert VERIFICATION_RETRIES.failures(live_key(CHANNEL_A)) == 0

    async def test_a_held_batch_whose_messages_are_withdrawn_leaves_nothing_behind(
        self, conn: aiosqlite.Connection, embedding_model: Any
    ) -> None:
        live = _Live(conn, embedding_model)
        await _queue(conn, message_id=1, content="Morgen um 20 Uhr ist Spieleabend.")
        await live.sweep(PIPELINE_NOW, TIMEOUT)
        await withdraw_message(conn, channel_id=CHANNEL_A, message_id=1)

        await live.sweep(PIPELINE_NOW + timedelta(seconds=600))

        assert live.distill.await_count == 1
        assert VERIFICATION_RETRIES.failures(live_key(CHANNEL_A)) == 0

    async def test_a_restart_during_the_pause_tries_the_batch_again_rather_than_losing_it(
        self, conn: aiosqlite.Connection, embedding_model: Any
    ) -> None:
        live = _Live(conn, embedding_model)
        await _queue(conn, message_id=1, content="Morgen um 20 Uhr ist Spieleabend.")
        await live.sweep(PIPELINE_NOW, TIMEOUT)
        VERIFICATION_RETRIES.reset()

        await live.sweep(PIPELINE_NOW + timedelta(seconds=1), _keep_all())

        assert [
            fact.message_id for fact in await get_pending_facts(conn, guild_id=GUILD_A, limit=5)
        ] == [1]

    async def test_a_success_starts_the_count_again_for_the_next_batch(
        self, conn: aiosqlite.Connection, embedding_model: Any
    ) -> None:
        live = _Live(conn, embedding_model)
        await _queue(conn, message_id=1, content="Morgen um 20 Uhr ist Spieleabend.")
        await live.sweep(PIPELINE_NOW, TIMEOUT)
        await live.sweep(PIPELINE_NOW + timedelta(seconds=600), _keep_all())
        await _queue(conn, message_id=2, content="Ab morgen gilt die neue Regel.")

        await live.sweep(PIPELINE_NOW + timedelta(seconds=601), TIMEOUT)

        assert VERIFICATION_RETRIES.failures(live_key(CHANNEL_A)) == 1
        assert await count_queued(conn) == 1

    async def test_an_exception_in_the_verifier_itself_never_loses_the_batch(
        self, conn: aiosqlite.Connection, embedding_model: Any
    ) -> None:
        live = _Live(conn, embedding_model)
        await _queue(conn, message_id=1, content="Morgen um 20 Uhr ist Spieleabend.")

        with patch(
            "aura.extraction.pipeline.verify_if_configured",
            AsyncMock(side_effect=RuntimeError("bug")),
        ):
            await live.sweep(PIPELINE_NOW)

        assert await count_queued(conn) == 1

    async def test_without_a_verification_model_nothing_is_ever_held(
        self, conn: aiosqlite.Connection, embedding_model: Any
    ) -> None:
        live = _Live(conn, embedding_model)
        live.settings = pipeline_settings()
        await _queue(conn, message_id=1, content="Morgen um 20 Uhr ist Spieleabend.")

        await live.sweep(PIPELINE_NOW)

        assert await count_queued(conn) == 0
        assert live.verify.await_count == 0


class TestTheBackfillPath:
    async def _setup(self, conn: aiosqlite.Connection) -> tuple[Any, Any]:
        from tests.test_backfill_worker import FIRST_ID, FakeChannel, FakeGateway, _message, _start

        gateway = FakeGateway()
        channel = gateway.add(
            FakeChannel(
                [
                    _message(FIRST_ID + index, content=f"Message number {index}.")
                    for index in range(3)
                ]
            )
        )
        await _start(conn)
        return gateway, channel

    async def _tick(
        self,
        conn: aiosqlite.Connection,
        embedding_model: Any,
        gateway: Any,
        now: datetime,
        verify: AsyncMock,
        **overrides: object,
    ) -> int:
        from aura.backfill.worker import advance_due_backfills
        from tests.test_backfill_worker import RecordingDistiller, _detector
        from tests.test_backfill_worker import _settings as backfill_settings

        with (
            patch("aura.backfill.worker.distill_facts", RecordingDistiller()),
            patch("aura.extraction.verifier.litellm.acompletion", verify),
        ):
            return await advance_due_backfills(
                conn,
                embedding_model,
                gateway,
                _detector(),
                settings=backfill_settings(extraction_verify_model=VERIFY_MODEL, **overrides),
                now=now,
                plan_gate=PlanGate.unenforced(),
            )

    async def test_a_failed_verification_keeps_the_cursor_and_reads_nothing_during_the_pause(
        self, conn: aiosqlite.Connection, embedding_model: Any
    ) -> None:
        from aura.db.backfill_runs import get_recent_runs
        from tests.test_backfill_worker import NOW

        gateway, channel = await self._setup(conn)
        assert (
            await self._tick(conn, embedding_model, gateway, NOW, AsyncMock(side_effect=TIMEOUT))
            == 0
        )
        requests_after_failure = channel.page_requests

        await self._tick(
            conn,
            embedding_model,
            gateway,
            NOW + timedelta(seconds=599),
            AsyncMock(side_effect=TIMEOUT),
        )

        run = (await get_recent_runs(conn, guild_id=GUILD_A, limit=1))[0]
        assert run.cursor_message_id is None
        assert channel.page_requests == requests_after_failure

    async def test_the_retried_batch_is_staged_and_counts_both_calls(
        self, conn: aiosqlite.Connection, embedding_model: Any
    ) -> None:
        from aura.db.backfill_runs import get_recent_runs
        from tests.test_backfill_worker import NOW

        gateway, _ = await self._setup(conn)
        await self._tick(conn, embedding_model, gateway, NOW, AsyncMock(side_effect=TIMEOUT))
        keep = _response({"checks": [_check(1), _check(2), _check(3)]})

        assert (
            await self._tick(
                conn,
                embedding_model,
                gateway,
                NOW + timedelta(seconds=600),
                AsyncMock(return_value=keep),
            )
            == 1
        )

        run = (await get_recent_runs(conn, guild_id=GUILD_A, limit=1))[0]
        assert len(await get_pending_facts(conn, guild_id=GUILD_A, limit=10)) == 3
        assert run.calls_spent == 2
        assert VERIFICATION_RETRIES.failures(backfill_key(run.id)) == 0

    async def test_a_verification_that_keeps_failing_moves_past_after_every_attempt_loudly(
        self, conn: aiosqlite.Connection, embedding_model: Any, caplog: pytest.LogCaptureFixture
    ) -> None:
        from aura.db.backfill_runs import get_recent_runs
        from tests.test_backfill_worker import NOW

        gateway, _ = await self._setup(conn)
        now = NOW
        with caplog.at_level(logging.WARNING):
            for _ in range(4):
                await self._tick(
                    conn, embedding_model, gateway, now, AsyncMock(side_effect=TIMEOUT)
                )
                now += timedelta(hours=2)

        run = (await get_recent_runs(conn, guild_id=GUILD_A, limit=1))[0]
        assert run.cursor_message_id is not None
        assert run.calls_spent == 4
        assert await get_pending_facts(conn, guild_id=GUILD_A, limit=10) == []
        assert any(
            record.levelno == logging.ERROR and "failed on all 4 attempt(s)" in record.getMessage()
            for record in caplog.records
        )

    async def test_an_unusable_reply_moves_past_without_a_retry(
        self, conn: aiosqlite.Connection, embedding_model: Any
    ) -> None:
        from aura.db.backfill_runs import get_recent_runs
        from tests.test_backfill_worker import NOW

        gateway, _ = await self._setup(conn)

        moved = await self._tick(
            conn,
            embedding_model,
            gateway,
            NOW,
            AsyncMock(return_value=_response(None, raw="{broken")),
        )

        run = (await get_recent_runs(conn, guild_id=GUILD_A, limit=1))[0]
        assert moved == 1
        assert run.calls_spent == 1
        assert VERIFICATION_RETRIES.failures(backfill_key(run.id)) == 0
