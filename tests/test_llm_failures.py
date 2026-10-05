"""Why a model call failed, and the alarm when the shared key is refused (P5c, aura.llm_failures).

The classification decides two things: whether extraction holds a batch for a
later attempt, and whether the operator's key alarm fires. Both are tested on
hand-built exceptions AND at the real litellm boundary (only httpx's send is
replaced), because litellm's exception class for OpenRouter's 402/403 differs
between versions and only the status and OpenRouter's own phrase are relied on.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
import litellm.main
import openai
import pytest

from aura.config import Settings
from aura.db.models import Fact, FactStatus
from aura.llm_failures import (
    ALARM_LOG_INTERVAL,
    KEY_ALARM,
    CallFailureKind,
    KeyAlarm,
    classify_call_failure,
    is_retryable,
    record_call_failure,
)

T0 = datetime(2026, 10, 6, 12, 0, tzinfo=UTC)
OPENROUTER = "openrouter/deepseek/deepseek-v4.1-flash"
KEY_LIMIT_BODY: dict[str, Any] = {
    "error": {
        "message": "Key limit exceeded (total limit). Manage it using "
        "https://openrouter.ai/workspaces/default/keys/0123456789abcdef",
        "code": 403,
    }
}


class _StatusError(Exception):
    def __init__(self, status_code: object, message: str = "provider error") -> None:
        super().__init__(message)
        self.status_code = status_code


class TestClassification:
    @pytest.mark.parametrize(
        ("error", "kind"),
        [
            (_StatusError(402), CallFailureKind.KEY_LIMIT),
            (_StatusError(403, json.dumps(KEY_LIMIT_BODY)), CallFailureKind.KEY_LIMIT),
            (
                _StatusError(
                    403, "OpenrouterException - Insufficient credits. Add more using the page."
                ),
                CallFailureKind.KEY_LIMIT,
            ),
            (_StatusError(401), CallFailureKind.KEY_INVALID),
            (_StatusError(408), CallFailureKind.TRANSPORT),
            (_StatusError(429), CallFailureKind.TRANSPORT),
            (_StatusError(500), CallFailureKind.TRANSPORT),
            (_StatusError(502), CallFailureKind.TRANSPORT),
            (_StatusError(503), CallFailureKind.TRANSPORT),
            (_StatusError(599), CallFailureKind.TRANSPORT),
            (_StatusError(400), CallFailureKind.REQUEST),
            (
                _StatusError(403, '{"error":{"message":"Input was flagged"}}'),
                CallFailureKind.REQUEST,
            ),
            (_StatusError(404), CallFailureKind.REQUEST),
            (_StatusError(422), CallFailureKind.REQUEST),
            (TimeoutError(), CallFailureKind.TRANSPORT),
            (ConnectionResetError(), CallFailureKind.TRANSPORT),
            (httpx.ConnectError("refused"), CallFailureKind.TRANSPORT),
            (
                openai.APIConnectionError(request=httpx.Request("POST", "https://x")),
                CallFailureKind.TRANSPORT,
            ),
            (RuntimeError("bug"), CallFailureKind.REQUEST),
            (ValueError("bad"), CallFailureKind.REQUEST),
        ],
    )
    def test_each_failure_is_classified(self, error: BaseException, kind: CallFailureKind) -> None:
        assert classify_call_failure(error) is kind

    def test_a_status_carried_on_the_response_is_read(self) -> None:
        error = Exception("x")
        error.response = httpx.Response(429)  # type: ignore[attr-defined]

        assert classify_call_failure(error) is CallFailureKind.TRANSPORT

    @pytest.mark.parametrize("status", [True, "403", 403.0, None])
    def test_a_status_that_is_not_an_integer_is_ignored(self, status: object) -> None:
        assert classify_call_failure(_StatusError(status)) is CallFailureKind.REQUEST

    def test_the_phrase_quoted_from_a_request_never_counts_as_a_key_limit(self) -> None:
        # A moderation refusal echoes the flagged input inside an escaped JSON
        # string; a member who writes the phrase must not fake the alarm.
        body = json.dumps(
            {
                "error": {
                    "message": "Your input was flagged",
                    "metadata": {"flagged_input": '"message":"Key limit exceeded" lol'},
                    "code": 403,
                }
            }
        )

        assert classify_call_failure(_StatusError(403, body)) is CallFailureKind.REQUEST
        assert classify_call_failure(_StatusError(400, body)) is CallFailureKind.REQUEST

    def test_an_exception_whose_text_cannot_be_read_never_raises(self) -> None:
        class Unprintable(Exception):
            def __str__(self) -> str:
                raise RuntimeError("no text")

        assert classify_call_failure(Unprintable()) is CallFailureKind.REQUEST
        assert record_call_failure(Unprintable(), purpose="a", model="m") is CallFailureKind.REQUEST

    def test_the_phrase_far_inside_a_huge_message_is_not_searched(self) -> None:
        message = "x" * 5000 + '"message":"Key limit exceeded'

        assert classify_call_failure(RuntimeError(message)) is CallFailureKind.REQUEST

    @pytest.mark.parametrize(
        ("kind", "retryable"),
        [
            (CallFailureKind.KEY_LIMIT, True),
            (CallFailureKind.KEY_INVALID, True),
            (CallFailureKind.TRANSPORT, True),
            (CallFailureKind.REQUEST, False),
        ],
    )
    def test_only_failures_outside_the_request_are_retryable(
        self, kind: CallFailureKind, retryable: bool
    ) -> None:
        assert is_retryable(kind) is retryable


async def _real_litellm_failure(status: int, body: dict[str, Any] | str) -> BaseException:
    """Return what the installed litellm raises for this OpenRouter reply (no network)."""

    async def reply(self: httpx.AsyncClient, request: httpx.Request, *args: Any, **kw: Any) -> Any:
        if isinstance(body, str):
            return httpx.Response(status, text=body, request=request)
        return httpx.Response(status, json=body, request=request)

    with patch("httpx.AsyncClient.send", reply):
        try:
            await litellm.main.acompletion(
                model=OPENROUTER,
                api_key="sk-test-not-a-key",
                messages=[{"role": "user", "content": "x"}],
                timeout=5,
            )
        except Exception as exc:
            return exc
    raise AssertionError("litellm did not raise")


class TestAtTheRealLitellmBoundary:
    async def test_openrouters_key_limit_is_a_key_limit(self) -> None:
        assert (
            classify_call_failure(await _real_litellm_failure(403, KEY_LIMIT_BODY))
            is CallFailureKind.KEY_LIMIT
        )

    async def test_openrouters_missing_credits_are_a_key_limit(self) -> None:
        body = {"error": {"message": "Insufficient credits. Add more.", "code": 402}}

        assert (
            classify_call_failure(await _real_litellm_failure(402, body))
            is CallFailureKind.KEY_LIMIT
        )

    async def test_a_moderation_refusal_echoing_the_phrase_is_not(self) -> None:
        body = {
            "error": {
                "message": "Your input was flagged",
                "metadata": {"flagged_input": "Key limit exceeded"},
                "code": 403,
            }
        }

        assert (
            classify_call_failure(await _real_litellm_failure(403, body)) is CallFailureKind.REQUEST
        )

    @pytest.mark.parametrize(
        ("status", "kind"),
        [
            (401, CallFailureKind.KEY_INVALID),
            (429, CallFailureKind.TRANSPORT),
            (500, CallFailureKind.TRANSPORT),
            (502, CallFailureKind.TRANSPORT),
            (503, CallFailureKind.TRANSPORT),
            (400, CallFailureKind.REQUEST),
            (404, CallFailureKind.REQUEST),
        ],
    )
    async def test_every_other_status(self, status: int, kind: CallFailureKind) -> None:
        body = {"error": {"message": "something", "code": status}}

        assert classify_call_failure(await _real_litellm_failure(status, body)) is kind

    async def test_a_connection_failure_and_a_read_timeout_are_transport(self) -> None:
        for error in (httpx.ConnectError("refused"), httpx.ReadTimeout("slow")):

            async def fail(
                self: httpx.AsyncClient,
                request: httpx.Request,
                *a: Any,
                bound: Exception = error,
                **k: Any,
            ) -> Any:
                raise bound

            with patch("httpx.AsyncClient.send", fail):
                with pytest.raises(Exception) as caught:
                    await litellm.main.acompletion(
                        model=OPENROUTER,
                        api_key="sk-test-not-a-key",
                        messages=[{"role": "user", "content": "x"}],
                        timeout=5,
                    )
            assert classify_call_failure(caught.value) is CallFailureKind.TRANSPORT


class TestTheAlarm:
    def test_the_first_refusal_logs_one_error_without_any_provider_text(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        alarm = KeyAlarm(started_at=T0)
        with caplog.at_level(logging.ERROR, logger="aura.llm_failures"):
            logged = alarm.note_refusal(
                CallFailureKind.KEY_LIMIT, purpose="extraction", model=OPENROUTER, now=T0
            )

        assert logged
        assert len(caplog.records) == 1
        line = caplog.records[0].getMessage()
        assert "spending limit or the account's credits are exhausted" in line
        assert "purpose=extraction" in line
        assert "openrouter.ai" not in line
        assert "keys/" not in line

    def test_refusals_within_the_hour_are_counted_but_not_logged_again(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        alarm = KeyAlarm(started_at=T0)
        with caplog.at_level(logging.ERROR, logger="aura.llm_failures"):
            for minute in range(0, 60, 3):
                alarm.note_refusal(
                    CallFailureKind.KEY_LIMIT,
                    purpose="answer-v2",
                    model=OPENROUTER,
                    now=T0 + timedelta(minutes=minute),
                )

        assert len(caplog.records) == 1
        assert alarm.status().refusals == 20

    def test_a_flapping_provider_logs_at_most_once_an_hour(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        alarm = KeyAlarm(started_at=T0)
        with caplog.at_level(logging.ERROR, logger="aura.llm_failures"):
            for second in range(0, 3 * 3600, 30):
                now = T0 + timedelta(seconds=second)
                alarm.note_success(now)
                alarm.note_refusal(
                    CallFailureKind.KEY_LIMIT, purpose="extraction", model=OPENROUTER, now=now
                )

        assert len(caplog.records) == 3

    def test_after_an_hour_the_line_is_logged_again(self, caplog: pytest.LogCaptureFixture) -> None:
        alarm = KeyAlarm(started_at=T0)
        with caplog.at_level(logging.ERROR, logger="aura.llm_failures"):
            alarm.note_refusal(CallFailureKind.KEY_LIMIT, purpose="a", model="m", now=T0)
            alarm.note_refusal(
                CallFailureKind.KEY_LIMIT, purpose="a", model="m", now=T0 + ALARM_LOG_INTERVAL
            )

        assert len(caplog.records) == 2

    @pytest.mark.parametrize("kind", [CallFailureKind.TRANSPORT, CallFailureKind.REQUEST])
    def test_other_failures_never_raise_the_alarm(
        self, kind: CallFailureKind, caplog: pytest.LogCaptureFixture
    ) -> None:
        alarm = KeyAlarm(started_at=T0)
        with caplog.at_level(logging.ERROR, logger="aura.llm_failures"):
            assert not alarm.note_refusal(kind, purpose="a", model="m", now=T0)

        assert caplog.records == []
        assert alarm.status().refusals == 0

    def test_an_invalid_key_is_named_as_such(self, caplog: pytest.LogCaptureFixture) -> None:
        alarm = KeyAlarm(started_at=T0)
        with caplog.at_level(logging.ERROR, logger="aura.llm_failures"):
            alarm.note_refusal(CallFailureKind.KEY_INVALID, purpose="a", model="m", now=T0)

        assert "invalid or revoked" in caplog.text

    def test_the_status_reports_the_latest_refusal_and_a_success_after_it(self) -> None:
        alarm = KeyAlarm(started_at=T0)
        alarm.note_success(T0)
        alarm.note_refusal(
            CallFailureKind.KEY_LIMIT, purpose="supersession", model="m", now=T0 + timedelta(1)
        )
        refused = alarm.status()
        alarm.note_success(T0 + timedelta(2))

        assert refused.last_refusal_purpose == "supersession"
        assert refused.last_refusal_kind is CallFailureKind.KEY_LIMIT
        assert not refused.succeeded_since_refusal
        assert alarm.status().succeeded_since_refusal

    def test_record_call_failure_feeds_the_process_alarm_and_returns_the_kind(self) -> None:
        kind = record_call_failure(_StatusError(402), purpose="synthesis", model="m")

        assert kind is CallFailureKind.KEY_LIMIT
        assert KEY_ALARM.status().refusals == 1
        assert KEY_ALARM.status().last_refusal_purpose == "synthesis"


def _fact() -> Fact:
    return Fact(
        id=1,
        guild_id=1,
        channel_id=2,
        message_id=3,
        content="Die Anmeldung schließt am 12. Oktober.",
        embedding=b"",
        status=FactStatus.ACTIVE,
        created_at=T0,
    )


def _settings() -> Settings:
    values: dict[str, object] = {
        "discord_token": "fake-token",
        "llm_api_key": "test-key",
        "synthesis_model": "openrouter/a/synth",
        "grounding_check_model": "openrouter/b/check",
        "answer_v2_check_model": "openrouter/b/check",
        "extraction_model": "openrouter/a/extract",
        "supersession_model": "openrouter/a/judge",
    }
    return Settings(**values)  # type: ignore[arg-type]


async def _call_site(name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Run one model call site whose litellm call fails; it must return, not raise."""
    monkeypatch.setenv("DISCORD_TOKEN", "fake-token")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    settings = _settings()
    if name == "synthesis":
        from aura.synthesis import synthesize_answer

        await synthesize_answer([_fact()], "Wann?", "de", model="openrouter/a/synth")
    elif name == "answer-v2":
        from aura.answer_contract import synthesize_contract_answer

        await synthesize_contract_answer(
            [_fact()], "Wann?", "de", model="openrouter/a/synth", settings=settings
        )
    elif name == "grounding":
        from aura.grounding import verify_answer_grounded

        await verify_answer_grounded(
            answer="Am 12.", cited_facts=[_fact()], settings=settings, timeout_seconds=5
        )
    elif name == "answer-v2-check":
        from aura.answer_check import build_statements, verify_answer_v2

        await verify_answer_v2(
            build_statements("Am 12.", [], (1,)), [_fact()], settings=settings, timeout_seconds=5
        )
    elif name == "extraction":
        from aura.extraction.distiller import distill_facts
        from tests.test_distiller import _queued

        await distill_facts([_queued(1, "x")], channel_name="g", model="openrouter/a/extract")
    elif name == "extraction-verify":
        from aura.extraction.verifier import verify_distilled_facts
        from tests.test_extraction_verifier import BATCH, DISTILLED

        await verify_distilled_facts(
            BATCH, DISTILLED, channel_name="g", model="openrouter/b/verify", settings=settings
        )
    elif name == "supersession":
        from aura.extraction.supersession import judge_relationship

        await judge_relationship(predecessor="A", candidate="B", model="openrouter/a/judge")
    elif name == "variants":
        from aura.variants_service import _generate_variants

        await _generate_variants("A", count=2, model="openrouter/a/var")
    elif name == "variants-audit":
        from aura.variants_service import _audit_variants

        await _audit_variants(canonical="A", variants=["B"], model="openrouter/a/var")
    else:
        raise AssertionError(name)


CALL_SITES = (
    "synthesis",
    "answer-v2",
    "grounding",
    "answer-v2-check",
    "extraction",
    "extraction-verify",
    "supersession",
    "variants",
    "variants-audit",
)


class TestEveryCallSiteFeedsTheAlarm:
    @pytest.mark.parametrize("site", CALL_SITES)
    async def test_a_refused_key_is_recorded_with_the_sites_purpose(
        self, site: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        with patch("litellm.acompletion", AsyncMock(side_effect=_StatusError(402))):
            await _call_site(site, monkeypatch)

        status = KEY_ALARM.status()
        assert status.refusals == 1
        assert status.last_refusal_purpose == site

    @pytest.mark.parametrize("site", CALL_SITES)
    async def test_an_ordinary_outage_is_not_a_key_refusal(
        self, site: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        with patch("litellm.acompletion", AsyncMock(side_effect=_StatusError(503))):
            await _call_site(site, monkeypatch)

        assert KEY_ALARM.status().refusals == 0

    def test_a_response_is_recorded_as_a_success(self) -> None:
        from unittest.mock import MagicMock

        from litellm.types.utils import ModelResponse

        from aura.llm_usage import log_llm_usage

        response = MagicMock(spec=ModelResponse)
        response.choices = []
        response.usage = None
        log_llm_usage(response, purpose="synthesis", model="m")

        assert KEY_ALARM.status().last_success_at is not None
