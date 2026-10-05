"""A batch whose DISTILLATION call failed for a reason outside the batch is held, never lost (P5c).

The P5 hold-and-retry of a failed verification call (tests/test_verify_retry.py)
now covers the extraction call itself, with the same backoff, the same attempt
limit -- shared with the verification -- and the same slot accounting; an
unusable reply and a request the provider refuses are still not retried.
"""

from __future__ import annotations

import logging
from datetime import timedelta
from typing import Any
from unittest.mock import AsyncMock, patch

import aiosqlite
import pytest

from aura.db.extraction_queue import count_queued
from aura.db.pending_facts import get_pending_facts
from aura.db.repository import init_schema
from aura.extraction.distiller import DISTILLATION_UNAVAILABLE, distill_facts
from aura.extraction.verify_retry import VERIFICATION_RETRIES, backfill_key, live_key
from tests.test_extraction_pipeline import CHANNEL_A, GUILD_A, _queue
from tests.test_extraction_pipeline import NOW as PIPELINE_NOW
from tests.test_verify_retry import TIMEOUT, _distilled, _keep_all, _Live


@pytest.fixture
async def conn():
    connection = await aiosqlite.connect(":memory:")
    await init_schema(connection)
    yield connection
    await connection.close()


def _unavailable_then(*replies: object) -> Any:
    """A distiller stand-in: unavailable for each `None` in `replies`, else distilled."""
    script = list(replies)

    def distill(batch: list[Any], **_kwargs: object) -> object:
        step = script.pop(0) if script else "ok"
        return DISTILLATION_UNAVAILABLE if step is None else _distilled(batch)

    return distill


class TestTheLivePath:
    async def test_a_failed_extraction_call_keeps_the_batch_queued(
        self, conn: aiosqlite.Connection, embedding_model: Any, caplog: pytest.LogCaptureFixture
    ) -> None:
        live = _Live(conn, embedding_model)
        live.distill.side_effect = _unavailable_then(None)
        await _queue(conn, message_id=1, content="Morgen um 20 Uhr ist Spieleabend.")

        with caplog.at_level(logging.WARNING):
            ran = await live.sweep(PIPELINE_NOW)

        assert ran == 1
        assert await count_queued(conn) == 1
        assert VERIFICATION_RETRIES.failures(live_key(CHANNEL_A)) == 1
        assert live.verify.await_count == 0
        assert "distillation call failed" in caplog.text
        assert "attempt 1 of 4" in caplog.text

    async def test_nothing_is_read_or_spent_during_the_pause(
        self, conn: aiosqlite.Connection, embedding_model: Any
    ) -> None:
        live = _Live(conn, embedding_model)
        live.distill.side_effect = _unavailable_then(None)
        await _queue(conn, message_id=1, content="Morgen um 20 Uhr ist Spieleabend.")
        await live.sweep(PIPELINE_NOW)

        ran = await live.sweep(PIPELINE_NOW + timedelta(seconds=599))

        assert ran == 0
        assert live.distill.await_count == 1
        assert await live.slots(PIPELINE_NOW) == 1

    async def test_after_the_pause_the_batch_is_distilled_again_and_staged(
        self, conn: aiosqlite.Connection, embedding_model: Any
    ) -> None:
        live = _Live(conn, embedding_model)
        live.distill.side_effect = _unavailable_then(None)
        await _queue(conn, message_id=1, content="Morgen um 20 Uhr ist Spieleabend.")
        await live.sweep(PIPELINE_NOW)

        ran = await live.sweep(PIPELINE_NOW + timedelta(seconds=600), _keep_all())

        assert ran == 1
        assert [f.message_id for f in await get_pending_facts(conn, guild_id=GUILD_A, limit=5)] == [
            1
        ]
        assert await count_queued(conn) == 0
        assert await live.slots(PIPELINE_NOW) == 2
        assert VERIFICATION_RETRIES.failures(live_key(CHANNEL_A)) == 0

    async def test_an_outage_that_lasts_is_given_up_with_one_error_after_every_attempt(
        self, conn: aiosqlite.Connection, embedding_model: Any, caplog: pytest.LogCaptureFixture
    ) -> None:
        live = _Live(conn, embedding_model)
        live.distill.side_effect = _unavailable_then(None, None, None, None, None)
        await _queue(conn, message_id=1, content="Morgen um 20 Uhr ist Spieleabend.")
        now = PIPELINE_NOW
        with caplog.at_level(logging.WARNING):
            for _ in range(5):
                await live.sweep(now)
                now += timedelta(hours=2)

        assert live.distill.await_count == 4
        assert await live.slots(PIPELINE_NOW) == 4
        assert await count_queued(conn) == 0
        errors = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert len(errors) == 1
        assert "distillation call failed" in errors[0].getMessage()
        assert "failed on all 4 attempt(s)" in errors[0].getMessage()

    async def test_a_failed_extraction_and_a_failed_verification_share_the_attempts(
        self, conn: aiosqlite.Connection, embedding_model: Any
    ) -> None:
        live = _Live(conn, embedding_model, extraction_verify_max_attempts=2)
        live.distill.side_effect = _unavailable_then(None)
        await _queue(conn, message_id=1, content="Morgen um 20 Uhr ist Spieleabend.")
        await live.sweep(PIPELINE_NOW)

        await live.sweep(PIPELINE_NOW + timedelta(seconds=600), TIMEOUT)

        assert await count_queued(conn) == 0
        assert VERIFICATION_RETRIES.failures(live_key(CHANNEL_A)) == 0
        assert await live.slots(PIPELINE_NOW) == 2

    async def test_an_unusable_reply_is_cleared_at_once_and_costs_one_slot(
        self, conn: aiosqlite.Connection, embedding_model: Any
    ) -> None:
        live = _Live(conn, embedding_model)
        live.distill.side_effect = lambda *_args, **_kwargs: None
        await _queue(conn, message_id=1, content="Morgen um 20 Uhr ist Spieleabend.")

        await live.sweep(PIPELINE_NOW)
        await live.sweep(PIPELINE_NOW + timedelta(hours=3))

        assert await count_queued(conn) == 0
        assert await live.slots(PIPELINE_NOW) == 1
        assert VERIFICATION_RETRIES.failures(live_key(CHANNEL_A)) == 0

    async def test_a_retry_refused_by_the_daily_cap_is_dropped_not_held_forever(
        self, conn: aiosqlite.Connection, embedding_model: Any
    ) -> None:
        live = _Live(conn, embedding_model, extraction_daily_cap=1)
        live.distill.side_effect = _unavailable_then(None)
        await _queue(conn, message_id=1, content="Morgen um 20 Uhr ist Spieleabend.")
        await live.sweep(PIPELINE_NOW)

        await live.sweep(PIPELINE_NOW + timedelta(seconds=600))

        assert live.distill.await_count == 1
        assert await count_queued(conn) == 0
        assert VERIFICATION_RETRIES.failures(live_key(CHANNEL_A)) == 0

    async def test_a_restart_during_the_pause_tries_again_instead_of_losing_the_batch(
        self, conn: aiosqlite.Connection, embedding_model: Any
    ) -> None:
        live = _Live(conn, embedding_model)
        live.distill.side_effect = _unavailable_then(None)
        await _queue(conn, message_id=1, content="Morgen um 20 Uhr ist Spieleabend.")
        await live.sweep(PIPELINE_NOW)
        VERIFICATION_RETRIES.reset()

        await live.sweep(PIPELINE_NOW + timedelta(seconds=1), _keep_all())

        assert len(await get_pending_facts(conn, guild_id=GUILD_A, limit=5)) == 1

    async def test_without_a_verification_model_a_failed_call_is_still_held(
        self, conn: aiosqlite.Connection, embedding_model: Any
    ) -> None:
        from tests.test_extraction_pipeline import _settings as pipeline_settings

        live = _Live(conn, embedding_model)
        live.settings = pipeline_settings()
        live.distill.side_effect = _unavailable_then(None)
        await _queue(conn, message_id=1, content="Morgen um 20 Uhr ist Spieleabend.")

        await live.sweep(PIPELINE_NOW)

        assert await count_queued(conn) == 1


class TestTheBackfillPath:
    async def _tick(
        self,
        conn: aiosqlite.Connection,
        embedding_model: Any,
        gateway: Any,
        now: Any,
        distiller: Any,
        **overrides: object,
    ) -> int:
        from aura.backfill.worker import advance_due_backfills
        from aura.billing import PlanGate
        from tests.test_backfill_worker import _detector
        from tests.test_backfill_worker import _settings as backfill_settings

        with patch("aura.backfill.worker.distill_facts", distiller):
            return await advance_due_backfills(
                conn,
                embedding_model,
                gateway,
                _detector(),
                settings=backfill_settings(**overrides),
                now=now,
                plan_gate=PlanGate.unenforced(),
            )

    async def _setup(self, conn: aiosqlite.Connection) -> tuple[Any, Any]:
        from tests.test_backfill_worker import FIRST_ID, FakeChannel, FakeGateway, _message, _start

        gateway = FakeGateway()
        channel = gateway.add(
            FakeChannel([_message(FIRST_ID + i, content=f"Message number {i}.") for i in range(3)])
        )
        await _start(conn)
        return gateway, channel

    async def test_a_failed_extraction_call_keeps_the_cursor_and_reads_nothing_while_waiting(
        self, conn: aiosqlite.Connection, embedding_model: Any
    ) -> None:
        from aura.db.backfill_runs import get_recent_runs
        from tests.test_backfill_worker import NOW

        gateway, channel = await self._setup(conn)
        unavailable = AsyncMock(return_value=DISTILLATION_UNAVAILABLE)
        await self._tick(conn, embedding_model, gateway, NOW, unavailable)
        requests_after_failure = channel.page_requests

        await self._tick(conn, embedding_model, gateway, NOW + timedelta(seconds=599), unavailable)

        run = (await get_recent_runs(conn, guild_id=GUILD_A, limit=1))[0]
        assert run.cursor_message_id is None
        assert channel.page_requests == requests_after_failure
        assert unavailable.await_count == 1
        assert VERIFICATION_RETRIES.failures(backfill_key(run.id)) == 1

    async def test_the_retried_batch_is_staged_and_both_calls_are_counted(
        self, conn: aiosqlite.Connection, embedding_model: Any
    ) -> None:
        from aura.db.backfill_runs import get_recent_runs
        from tests.test_backfill_worker import NOW, RecordingDistiller

        gateway, _ = await self._setup(conn)
        await self._tick(
            conn, embedding_model, gateway, NOW, AsyncMock(return_value=DISTILLATION_UNAVAILABLE)
        )

        await self._tick(
            conn, embedding_model, gateway, NOW + timedelta(seconds=600), RecordingDistiller()
        )

        run = (await get_recent_runs(conn, guild_id=GUILD_A, limit=1))[0]
        assert len(await get_pending_facts(conn, guild_id=GUILD_A, limit=10)) == 3
        assert run.calls_spent == 2

    async def test_an_outage_that_lasts_moves_past_the_batch_loudly_and_counts_every_call(
        self, conn: aiosqlite.Connection, embedding_model: Any, caplog: pytest.LogCaptureFixture
    ) -> None:
        from aura.db.backfill_runs import get_recent_runs
        from tests.test_backfill_worker import NOW

        gateway, _ = await self._setup(conn)
        unavailable = AsyncMock(return_value=DISTILLATION_UNAVAILABLE)
        now = NOW
        with caplog.at_level(logging.WARNING):
            for _ in range(4):
                await self._tick(conn, embedding_model, gateway, now, unavailable)
                now += timedelta(hours=2)

        run = (await get_recent_runs(conn, guild_id=GUILD_A, limit=1))[0]
        assert run.cursor_message_id is not None
        assert run.calls_spent == 4
        assert any(
            r.levelno == logging.ERROR and "distillation call failed" in r.getMessage()
            for r in caplog.records
        )

    async def test_an_unusable_reply_moves_past_at_once(
        self, conn: aiosqlite.Connection, embedding_model: Any
    ) -> None:
        from aura.db.backfill_runs import get_recent_runs
        from tests.test_backfill_worker import NOW

        gateway, _ = await self._setup(conn)

        await self._tick(conn, embedding_model, gateway, NOW, AsyncMock(return_value=None))

        run = (await get_recent_runs(conn, guild_id=GUILD_A, limit=1))[0]
        assert run.cursor_message_id is not None
        assert run.calls_spent == 1
        assert VERIFICATION_RETRIES.failures(backfill_key(run.id)) == 0


class _StatusError(Exception):
    def __init__(self, status_code: int, message: str = "provider error") -> None:
        super().__init__(message)
        self.status_code = status_code


class TestTheDistillerClassifiesItsOwnFailures:
    """Which failures of the extraction call are held (unavailable) and which are not (None)."""

    @pytest.fixture(autouse=True)
    def _llm_settings(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DISCORD_TOKEN", "fake-token")
        monkeypatch.setenv("LLM_API_KEY", "test-key")

    @pytest.mark.parametrize(
        "error",
        [
            TimeoutError(),
            OSError("connection reset"),
            _StatusError(429),
            _StatusError(500),
            _StatusError(503),
            _StatusError(401),
            _StatusError(402),
            _StatusError(
                403, '{"error":{"message":"Key limit exceeded (total limit)","code":403}}'
            ),
        ],
        ids=["timeout", "network", "429", "500", "503", "401", "402", "403-key-limit"],
    )
    async def test_a_call_that_never_completed_is_reported_unavailable(
        self, error: Exception
    ) -> None:
        from tests.test_distiller import MODEL, _queued

        with patch("litellm.acompletion", AsyncMock(side_effect=error)):
            result = await distill_facts([_queued(1, "x")], channel_name="g", model=MODEL)

        assert result is DISTILLATION_UNAVAILABLE

    @pytest.mark.parametrize(
        "error",
        [
            _StatusError(400),
            _StatusError(403, '{"error":{"message":"Your input was flagged","code":403}}'),
            _StatusError(404),
            _StatusError(422),
            RuntimeError("bug"),
        ],
        ids=["400", "403-moderation", "404", "422", "unknown"],
    )
    async def test_a_request_the_provider_refuses_is_not_retried(self, error: Exception) -> None:
        from tests.test_distiller import MODEL, _queued

        with patch("litellm.acompletion", AsyncMock(side_effect=error)):
            result = await distill_facts([_queued(1, "x")], channel_name="g", model=MODEL)

        assert result is None
