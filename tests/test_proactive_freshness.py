"""A proactive answer never arrives late into a conversation that has moved on (P5c).

PROACTIVE_REQUEST_TIMEOUT_SECONDS: unset is exactly the call of before; set, it
is the client timeout and a hard deadline around the answer call, in both
formats, and /aura-ask never reads it. The grace period's watch continues
while the answer is written and checked (aura.proactive.grace.AnswerWatch), and
an answer past `answer_deadline_seconds` after the grace period is not posted.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import aiosqlite
import pytest
from pydantic import ValidationError

from aura import answer_contract, synthesis
from aura.billing import PlanGate
from aura.db.repository import init_schema
from aura.grounding import PROACTIVE_GROUNDING_TIMEOUT_SECONDS, GroundingOutcome
from aura.proactive.grace import GraceRegistry
from aura.proactive.listener import _wait_then_respond
from aura.proactive.responder import (
    DEFAULT_ANSWER_TIMEOUT_SECONDS,
    answer_deadline_seconds,
    respond_with_synthesis,
)
from aura.synthesis import SynthesisResult
from tests.test_answer_v2_paths import (
    GUILD,
    _contract_for,
    _MatchingModel,
    _proactive_setup,
    _settings,
    _tripwire,
)

# Patching responder.asyncio.wait_for patches the module everyone shares; the
# shrunk stand-in must call the real one.
_REAL_WAIT_FOR = asyncio.wait_for

ASKER = 7
OTHER_MEMBER = 8
CHANNEL = 555


@pytest.fixture
async def conn():
    connection = await aiosqlite.connect(":memory:")
    await init_schema(connection)
    yield connection
    await connection.close()


class _Clock:
    """A monotonic clock the test moves by hand."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class TestTheSetting:
    def test_unset_by_default(self) -> None:
        assert _settings().proactive_request_timeout_seconds is None

    @pytest.mark.parametrize("value", [4.9, 300.1, float("inf"), float("nan"), -1.0])
    def test_out_of_range_values_refuse_to_start(self, value: float) -> None:
        with pytest.raises(ValidationError):
            _settings(proactive_request_timeout_seconds=value)

    def test_read_from_the_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PROACTIVE_REQUEST_TIMEOUT_SECONDS", "60")

        assert _settings().proactive_request_timeout_seconds == 60.0

    def test_the_stated_default_equals_both_answer_calls_own(self) -> None:
        assert DEFAULT_ANSWER_TIMEOUT_SECONDS == synthesis._REQUEST_TIMEOUT_SECONDS
        assert DEFAULT_ANSWER_TIMEOUT_SECONDS == answer_contract._REQUEST_TIMEOUT_SECONDS

    @pytest.mark.parametrize(("timeout", "deadline"), [(None, 75.0), (60.0, 105.0)])
    def test_the_posting_deadline(self, timeout: float | None, deadline: float) -> None:
        settings = _settings(proactive_request_timeout_seconds=timeout)

        assert answer_deadline_seconds(settings) == deadline
        assert PROACTIVE_GROUNDING_TIMEOUT_SECONDS == 30.0


class TestTheWatch:
    def test_a_fresh_answer_may_be_posted(self) -> None:
        registry = GraceRegistry()
        with registry.watch_answer(channel_id=CHANNEL, asker_id=ASKER, message_id=1) as watch:
            assert watch.stale_reason(deadline_seconds=60) is None

    def test_another_member_writing_makes_it_stale(self) -> None:
        registry = GraceRegistry()
        with registry.watch_answer(channel_id=CHANNEL, asker_id=ASKER, message_id=1) as watch:
            registry.notice_human_message(channel_id=CHANNEL, author_id=OTHER_MEMBER, message_id=2)

            assert watch.stale_reason(deadline_seconds=60) == "the conversation moved on"

    def test_the_askers_own_follow_up_does_not(self) -> None:
        registry = GraceRegistry()
        with registry.watch_answer(channel_id=CHANNEL, asker_id=ASKER, message_id=1) as watch:
            registry.notice_human_message(channel_id=CHANNEL, author_id=ASKER, message_id=2)
            registry.notice_human_message(channel_id=CHANNEL, author_id=OTHER_MEMBER, message_id=1)

            assert watch.stale_reason(deadline_seconds=60) is None

    def test_a_message_in_another_channel_does_not(self) -> None:
        registry = GraceRegistry()
        with registry.watch_answer(channel_id=CHANNEL, asker_id=ASKER, message_id=1) as watch:
            registry.notice_human_message(channel_id=999, author_id=OTHER_MEMBER, message_id=2)

            assert watch.stale_reason(deadline_seconds=60) is None

    def test_an_edited_or_deleted_question_makes_it_stale(self) -> None:
        registry = GraceRegistry()
        with registry.watch_answer(channel_id=CHANNEL, asker_id=ASKER, message_id=1) as watch:
            registry.notice_message_gone(channel_id=CHANNEL, message_id=1)

            assert watch.stale_reason(deadline_seconds=60) == "the conversation moved on"

    def test_past_the_deadline_it_is_too_late(self) -> None:
        clock = _Clock()
        registry = GraceRegistry()
        with registry.watch_answer(
            channel_id=CHANNEL, asker_id=ASKER, message_id=1, clock=clock
        ) as watch:
            clock.now += 60.0
            assert watch.stale_reason(deadline_seconds=60) is None
            clock.now += 0.001
            assert watch.stale_reason(deadline_seconds=60) == "too late"

    def test_the_registration_is_removed_on_exit_even_after_an_error(self) -> None:
        registry = GraceRegistry()
        with (
            pytest.raises(RuntimeError),
            registry.watch_answer(channel_id=CHANNEL, asker_id=ASKER, message_id=1),
        ):
            raise RuntimeError("boom")

        assert CHANNEL not in registry._pending

    def test_an_occupied_channel_starts_stale_and_keeps_the_newer_registration(self) -> None:
        registry = GraceRegistry()
        with registry.watch_answer(channel_id=CHANNEL, asker_id=ASKER, message_id=1):
            newer = registry._pending[CHANNEL]
            with registry.watch_answer(channel_id=CHANNEL, asker_id=ASKER, message_id=2) as second:
                assert second.stale_reason(deadline_seconds=60) == "the conversation moved on"
            assert registry._pending[CHANNEL] is newer
            assert not newer.cancel_event.is_set()

    async def test_a_new_grace_period_in_the_channel_supersedes_the_watch(self) -> None:
        registry = GraceRegistry()
        with registry.watch_answer(channel_id=CHANNEL, asker_id=ASKER, message_id=1) as watch:
            waiting = asyncio.create_task(
                registry.wait(channel_id=CHANNEL, asker_id=OTHER_MEMBER, message_id=2, seconds=0.01)
            )
            await waiting

            assert watch.stale_reason(deadline_seconds=60) == "the conversation moved on"
        assert CHANNEL not in registry._pending


class _Stale:
    def __init__(self, reason: str | None) -> None:
        self.reason = reason
        self.deadlines: list[float] = []

    def stale_reason(self, *, deadline_seconds: float) -> str | None:
        self.deadlines.append(deadline_seconds)
        return self.reason


async def _respond(conn: Any, message: Any, settings: Any, freshness: Any) -> Any:
    return await respond_with_synthesis(
        message,
        db=conn,
        model=_MatchingModel(),  # type: ignore[arg-type]
        settings=settings,
        freshness=freshness,
    )


def _legacy_patches(fact: Any, synth: Any | None = None) -> Any:
    legacy = SynthesisResult(answer="In #welcome.", used_fact_ids=[fact.id], answers_question=True)
    return (
        patch(
            "aura.proactive.responder.synthesize_answer",
            synth or AsyncMock(return_value=legacy),
        ),
        patch(
            "aura.proactive.responder.verify_answer_grounded",
            AsyncMock(return_value=GroundingOutcome.GROUNDED),
        ),
    )


def _v2_patches(fact: Any, synth: Any | None = None) -> Any:
    return (
        patch(
            "aura.proactive.responder.synthesize_contract_answer",
            synth or AsyncMock(return_value=_contract_for(fact, lead="In #welcome.")),
        ),
        patch(
            "aura.proactive.responder.verify_answer_v2",
            AsyncMock(return_value=GroundingOutcome.GROUNDED),
        ),
    )


FORMATS = {
    "legacy": ({}, _legacy_patches),
    "v2": ({"proactive_answer_format": "v2", "proactive_model": "pro/active"}, _v2_patches),
}


@pytest.mark.parametrize("fmt", sorted(FORMATS))
class TestTheResponder:
    async def test_a_fresh_answer_is_posted_and_asked_with_the_posting_deadline(
        self, conn: aiosqlite.Connection, fmt: str
    ) -> None:
        overrides, patches = FORMATS[fmt]
        fact, message = await _proactive_setup(conn)
        fresh = _Stale(None)
        first, second = patches(fact)
        with first, second:
            outcome = await _respond(conn, message, _settings(**overrides), fresh)

        assert outcome.posted is True
        assert fresh.deadlines == [75.0]

    @pytest.mark.parametrize("reason", ["the conversation moved on", "too late"])
    async def test_a_stale_answer_is_silence_and_an_info_line(
        self, conn: aiosqlite.Connection, fmt: str, reason: str, caplog: pytest.LogCaptureFixture
    ) -> None:
        overrides, patches = FORMATS[fmt]
        fact, message = await _proactive_setup(conn)
        first, second = patches(fact)
        with first, second, caplog.at_level(logging.INFO, logger="aura.proactive.responder"):
            outcome = await _respond(conn, message, _settings(**overrides), _Stale(reason))

        assert outcome.posted is False
        assert outcome.answers_question is True
        message.channel.send.assert_not_awaited()
        assert f"Proactive answer withheld in channel {CHANNEL}: {reason}" in caplog.text
        assert "where are the rules" not in caplog.text

    async def test_the_setting_is_passed_as_the_calls_own_timeout(
        self, conn: aiosqlite.Connection, fmt: str
    ) -> None:
        overrides, patches = FORMATS[fmt]
        fact, message = await _proactive_setup(conn)
        first, second = patches(fact)
        with first as synth, second:
            await _respond(
                conn,
                message,
                _settings(proactive_request_timeout_seconds=60, **overrides),
                None,
            )

        assert synth.await_args.kwargs["timeout_seconds"] == 60

    async def test_unset_the_call_gets_no_timeout_of_its_own_and_no_deadline(
        self, conn: aiosqlite.Connection, fmt: str
    ) -> None:
        overrides, patches = FORMATS[fmt]
        fact, message = await _proactive_setup(conn)
        first, second = patches(fact)
        with (
            first as synth,
            second,
            patch("aura.proactive.responder.asyncio.wait_for", _tripwire("a deadline")),
        ):
            outcome = await _respond(conn, message, _settings(**overrides), None)

        assert outcome.posted is True
        assert synth.await_args.kwargs["timeout_seconds"] is None

    async def test_set_a_call_past_its_deadline_is_silence_and_never_reaches_the_check(
        self, conn: aiosqlite.Connection, fmt: str, caplog: pytest.LogCaptureFixture
    ) -> None:
        overrides, patches = FORMATS[fmt]
        fact, message = await _proactive_setup(conn)

        async def never_answers(*_args: object, **_kwargs: object) -> None:
            await asyncio.sleep(3600)

        hung = AsyncMock(side_effect=never_answers)
        first, _ = patches(fact, hung)
        settings = _settings(proactive_request_timeout_seconds=5, **overrides)
        with (
            first,
            patch("aura.proactive.responder.verify_answer_grounded", _tripwire("legacy check")),
            patch("aura.proactive.responder.verify_answer_v2", _tripwire("v2 check")),
            patch("aura.proactive.responder.asyncio.wait_for", _fast_wait_for),
            caplog.at_level(logging.WARNING, logger="aura.proactive.responder"),
        ):
            # Bounded here too, so a responder that ignored its deadline fails
            # this test in seconds instead of sleeping an hour.
            outcome = await _REAL_WAIT_FOR(_respond(conn, message, settings, None), timeout=5)

        assert outcome.posted is False
        assert outcome.answers_question is None
        message.channel.send.assert_not_awaited()
        assert "ran past its 5-second deadline" in caplog.text


async def _fast_wait_for(awaitable: Any, timeout: float) -> Any:  # noqa: ASYNC109
    """asyncio.wait_for with the deadline shrunk, so the test does not wait `timeout` seconds."""
    assert timeout == 5
    return await _REAL_WAIT_FOR(awaitable, timeout=0.01)


class TestAskIsUntouched:
    @pytest.mark.parametrize("answer_format", ["legacy", "v2"])
    async def test_ask_keeps_its_30_seconds_whatever_proactive_says(
        self, conn: aiosqlite.Connection, answer_format: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from tests.test_answer_v2_paths import _add, _ask, _interaction

        await _add(conn, _MatchingModel(), "The event starts on Saturday at 18:00.")
        settings = _settings(
            answer_format=answer_format,
            proactive_request_timeout_seconds=120,
            answer_v2_check_model="openrouter/fake/check",
        )
        monkeypatch.setenv("DISCORD_TOKEN", "fake-token")
        monkeypatch.setenv("LLM_API_KEY", "fake-key")
        monkeypatch.setenv("SYNTHESIS_MODEL", "openrouter/fake/synth")
        seen: list[object] = []

        async def capture(*_args: object, **kwargs: object) -> Any:
            seen.append(kwargs.get("timeout"))
            raise TimeoutError

        with patch("litellm.acompletion", capture):
            await _ask(_interaction(conn, _MatchingModel(), settings), "When does the event start?")

        assert seen
        assert seen[0] == 30


class TestTheListenerKeepsWatching:
    async def _run(
        self, conn: aiosqlite.Connection, message: Any, settings: Any, during: Any
    ) -> Any:
        registry = GraceRegistry()
        message.author = MagicMock()
        message.author.id = ASKER
        message.id = 1
        self.synthesized = 0

        async def synth(*_args: object, **_kwargs: object) -> Any:
            self.synthesized += 1
            during(registry)
            fact = (await _proactive_fact(conn))[0]
            return SynthesisResult(
                answer="In #welcome.", used_fact_ids=[fact.id], answers_question=True
            )

        with (
            patch("aura.proactive.responder.synthesize_answer", synth),
            patch(
                "aura.proactive.responder.verify_answer_grounded",
                AsyncMock(return_value=GroundingOutcome.GROUNDED),
            ),
            patch("aura.proactive.listener.update_grace_outcome", AsyncMock()),
            # The wake-time recheck (an escalation row, the plan) is not what
            # these tests are about; it passes so the answer is really written.
            patch(
                "aura.proactive.listener._still_fresh_enough_for_synthesis",
                AsyncMock(return_value=True),
            ),
        ):
            outcome = await _wait_then_respond(
                message,
                db=conn,
                model=_MatchingModel(),  # type: ignore[arg-type]
                settings=settings,
                grace_registry=registry,
                plan_gate=PlanGate.unenforced(),
            )
        assert CHANNEL not in registry._pending
        assert self.synthesized == 1
        return outcome

    async def test_a_member_answering_while_the_answer_is_written_stops_the_post(
        self, conn: aiosqlite.Connection
    ) -> None:
        _, message = await _proactive_setup(conn)

        outcome = await self._run(
            conn,
            message,
            _settings(proactive_grace_period_seconds=0),
            lambda registry: registry.notice_human_message(
                channel_id=CHANNEL, author_id=OTHER_MEMBER, message_id=2
            ),
        )

        assert outcome.posted is False
        message.channel.send.assert_not_awaited()

    async def test_the_question_deleted_while_the_answer_is_written_stops_the_post(
        self, conn: aiosqlite.Connection
    ) -> None:
        _, message = await _proactive_setup(conn)

        outcome = await self._run(
            conn,
            message,
            _settings(proactive_grace_period_seconds=0),
            lambda registry: registry.notice_message_gone(channel_id=CHANNEL, message_id=1),
        )

        assert outcome.posted is False

    async def test_a_quiet_channel_still_gets_its_answer(self, conn: aiosqlite.Connection) -> None:
        _, message = await _proactive_setup(conn)

        outcome = await self._run(
            conn,
            message,
            _settings(proactive_grace_period_seconds=0),
            lambda registry: registry.notice_human_message(
                channel_id=CHANNEL, author_id=ASKER, message_id=3
            ),
        )

        assert outcome.posted is True


async def _proactive_fact(conn: aiosqlite.Connection) -> list[Any]:
    from aura.db.repository import get_active_facts

    return await get_active_facts(conn, guild_id=GUILD)
