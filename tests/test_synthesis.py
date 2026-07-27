"""Tests for aura.synthesis: LLM-backed answer synthesis.

Every test in this file mocks litellm.acompletion completely -- zero real
API calls, zero cost, per this phase's hard constraint (the OpenRouter
account isn't funded yet). The one true end-to-end sanity check against a
real provider is explicitly opt-in (see TestRealProviderSanityCheck at the
bottom): it is skipped automatically unless LLM_API_KEY is actually present
as a real environment variable, and is not part of this phase's definition
of done.

As of Phase 2a-3 synthesize_answer takes the model as a parameter (resolved by
the caller through the resolve_model seam) and returns an answers_question
self-assessment alongside the answer. Both are exercised here.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import litellm
import pytest
from litellm.types.utils import Choices, Message, ModelResponse

from aura.config import Settings
from aura.db.models import Fact, FactStatus
from aura.synthesis import synthesize_answer

GUILD_A = 100000000000000001

# The resolved model string every call passes in; synthesize_answer no longer
# reads it from settings, so it is supplied here explicitly.
MODEL = "openrouter/fake/model"


def _make_fact(id_: int, content: str) -> Fact:
    return Fact(
        id=id_,
        guild_id=GUILD_A,
        channel_id=1,
        message_id=id_,
        content=content,
        embedding=bytes(384 * 4),
        status=FactStatus.ACTIVE,
        created_at=datetime.now(timezone.utc),
    )


def _make_response(content: str | None) -> ModelResponse:
    message = Message(content=content, role="assistant")
    choice = Choices(finish_reason="stop", index=0, message=message)
    return ModelResponse(choices=[choice])


def _payload(answer: str, used: list[int], answers_question: bool = True) -> str:
    """A well-formed model response body for the current schema."""
    return json.dumps(
        {"answer": answer, "used_fact_numbers": used, "answers_question": answers_question}
    )


def _fake_settings(**overrides: object) -> Settings:
    defaults: dict[str, object] = {
        "discord_token": "fake-token",
        "llm_api_key": "fake-key",
        "synthesis_model": "openrouter/fake/model",
    }
    defaults.update(overrides)
    return Settings(_env_file=None, **defaults)  # type: ignore[arg-type]


@pytest.fixture
def configured_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch load_settings to return a fake, fully-configured Settings object."""
    monkeypatch.setattr("aura.synthesis.load_settings", lambda: _fake_settings())


@pytest.mark.usefixtures("configured_settings")
class TestSynthesizeAnswerHappyPath:
    async def test_returns_answer_and_maps_fact_numbers_to_real_ids(self) -> None:
        facts = [_make_fact(101, "fact one"), _make_fact(202, "fact two")]
        response = _make_response(_payload("the answer", [2]))

        with patch("aura.synthesis.litellm.acompletion", AsyncMock(return_value=response)):
            result = await synthesize_answer(facts, "a question", "en-US", model=MODEL)

        assert result is not None
        assert result.answer == "the answer"
        assert result.used_fact_ids == [202]  # mapped from number 2 -> fact 202's real id
        assert result.answers_question is True

    async def test_empty_used_fact_numbers_is_valid(self) -> None:
        facts = [_make_fact(1, "fact")]
        response = _make_response(_payload("I could not find that", [], answers_question=False))

        with patch("aura.synthesis.litellm.acompletion", AsyncMock(return_value=response)):
            result = await synthesize_answer(facts, "q", "en-US", model=MODEL)

        assert result is not None
        assert result.used_fact_ids == []
        assert result.answers_question is False

    async def test_duplicate_fact_numbers_are_deduplicated(self) -> None:
        facts = [_make_fact(5, "fact")]
        response = _make_response(_payload("ans", [1, 1]))

        with patch("aura.synthesis.litellm.acompletion", AsyncMock(return_value=response)):
            result = await synthesize_answer(facts, "q", "en-US", model=MODEL)

        assert result is not None
        assert result.used_fact_ids == [5]


@pytest.mark.usefixtures("configured_settings")
class TestSelfAssessment:
    """answers_question is the field Trigger 2's post/stay-silent decision reads."""

    @pytest.mark.parametrize("value", [True, False])
    async def test_answers_question_is_parsed_through(self, value: bool) -> None:
        facts = [_make_fact(1, "fact")]
        response = _make_response(_payload("ans", [1], answers_question=value))

        with patch("aura.synthesis.litellm.acompletion", AsyncMock(return_value=response)):
            result = await synthesize_answer(facts, "q", "en-US", model=MODEL)

        assert result is not None
        assert result.answers_question is value

    async def test_missing_answers_question_is_a_malformed_response(self) -> None:
        # The field is required now: a model that omits it produced output the
        # post decision cannot trust, so it fails closed to None rather than
        # defaulting the field to an optimistic true.
        facts = [_make_fact(1, "fact")]
        response = _make_response('{"answer": "ans", "used_fact_numbers": [1]}')

        with patch("aura.synthesis.litellm.acompletion", AsyncMock(return_value=response)):
            result = await synthesize_answer(facts, "q", "en-US", model=MODEL)

        assert result is None

    async def test_non_boolean_answers_question_is_rejected(self) -> None:
        facts = [_make_fact(1, "fact")]
        response = _make_response(
            '{"answer": "ans", "used_fact_numbers": [1], "answers_question": "sure"}'
        )

        with patch("aura.synthesis.litellm.acompletion", AsyncMock(return_value=response)):
            result = await synthesize_answer(facts, "q", "en-US", model=MODEL)

        assert result is None


class TestNotConfigured:
    """synthesize_answer is independently defensive, not just trusting the caller's check."""

    async def test_returns_none_without_raising_when_no_api_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "aura.synthesis.load_settings",
            lambda: _fake_settings(llm_api_key=None),
        )
        mock_call = AsyncMock()

        with patch("aura.synthesis.litellm.acompletion", mock_call):
            result = await synthesize_answer([_make_fact(1, "fact")], "q", "en-US", model=MODEL)

        assert result is None
        mock_call.assert_not_awaited()

    @pytest.mark.usefixtures("configured_settings")
    async def test_returns_none_without_raising_when_model_is_blank(self) -> None:
        # The model is passed in now; an empty one means the caller resolved
        # nothing, and no call must be made.
        mock_call = AsyncMock()

        with patch("aura.synthesis.litellm.acompletion", mock_call):
            result = await synthesize_answer([_make_fact(1, "fact")], "q", "en-US", model="")

        assert result is None
        mock_call.assert_not_awaited()


@pytest.mark.usefixtures("configured_settings")
class TestMalformedOutput:
    async def test_invalid_json_returns_none_not_a_crash(self) -> None:
        facts = [_make_fact(1, "fact")]
        response = _make_response("this is not JSON at all {{{")

        with patch("aura.synthesis.litellm.acompletion", AsyncMock(return_value=response)):
            result = await synthesize_answer(facts, "q", "en-US", model=MODEL)

        assert result is None

    async def test_valid_json_missing_required_key_returns_none(self) -> None:
        facts = [_make_fact(1, "fact")]
        response = _make_response('{"answer": "ans"}')  # used_fact_numbers missing entirely

        with patch("aura.synthesis.litellm.acompletion", AsyncMock(return_value=response)):
            result = await synthesize_answer(facts, "q", "en-US", model=MODEL)

        assert result is None

    async def test_out_of_range_fact_number_returns_none_not_a_crash(self) -> None:
        # Only 2 facts sent, but the model cites number 5 -- a hallucinated
        # citation, the exact failure mode this defends against.
        facts = [_make_fact(1, "fact one"), _make_fact(2, "fact two")]
        response = _make_response(_payload("ans", [5]))

        with patch("aura.synthesis.litellm.acompletion", AsyncMock(return_value=response)):
            result = await synthesize_answer(facts, "q", "en-US", model=MODEL)

        assert result is None

    async def test_zero_fact_number_returns_none(self) -> None:
        facts = [_make_fact(1, "fact")]
        response = _make_response(_payload("ans", [0]))

        with patch("aura.synthesis.litellm.acompletion", AsyncMock(return_value=response)):
            result = await synthesize_answer(facts, "q", "en-US", model=MODEL)

        assert result is None

    async def test_negative_fact_number_returns_none(self) -> None:
        facts = [_make_fact(1, "fact")]
        response = _make_response(_payload("ans", [-1]))

        with patch("aura.synthesis.litellm.acompletion", AsyncMock(return_value=response)):
            result = await synthesize_answer(facts, "q", "en-US", model=MODEL)

        assert result is None

    async def test_blank_answer_returns_none(self) -> None:
        facts = [_make_fact(1, "fact")]
        response = _make_response(_payload("   ", []))

        with patch("aura.synthesis.litellm.acompletion", AsyncMock(return_value=response)):
            result = await synthesize_answer(facts, "q", "en-US", model=MODEL)

        assert result is None

    async def test_empty_content_returns_none(self) -> None:
        facts = [_make_fact(1, "fact")]
        response = _make_response("")

        with patch("aura.synthesis.litellm.acompletion", AsyncMock(return_value=response)):
            result = await synthesize_answer(facts, "q", "en-US", model=MODEL)

        assert result is None

    async def test_none_content_returns_none(self) -> None:
        facts = [_make_fact(1, "fact")]
        response = _make_response(None)

        with patch("aura.synthesis.litellm.acompletion", AsyncMock(return_value=response)):
            result = await synthesize_answer(facts, "q", "en-US", model=MODEL)

        assert result is None

    async def test_non_model_response_return_value_returns_none(self) -> None:
        # acompletion's declared return type also covers a streaming
        # response; synthesize_answer treats an unexpected type as a
        # failure rather than crashing trying to read .choices off it.
        facts = [_make_fact(1, "fact")]

        with patch("aura.synthesis.litellm.acompletion", AsyncMock(return_value=object())):
            result = await synthesize_answer(facts, "q", "en-US", model=MODEL)

        assert result is None


@pytest.mark.usefixtures("configured_settings")
class TestApiFailures:
    async def test_network_error_returns_none(self) -> None:
        error = litellm.exceptions.APIConnectionError(
            message="connection failed", llm_provider="openrouter", model="fake"
        )
        with patch("aura.synthesis.litellm.acompletion", AsyncMock(side_effect=error)):
            result = await synthesize_answer([_make_fact(1, "fact")], "q", "en-US", model=MODEL)
        assert result is None

    async def test_auth_failure_returns_none(self) -> None:
        error = litellm.exceptions.AuthenticationError(
            message="invalid api key", llm_provider="openrouter", model="fake"
        )
        with patch("aura.synthesis.litellm.acompletion", AsyncMock(side_effect=error)):
            result = await synthesize_answer([_make_fact(1, "fact")], "q", "en-US", model=MODEL)
        assert result is None

    async def test_timeout_returns_none(self) -> None:
        error = litellm.exceptions.Timeout(
            message="request timed out", model="fake", llm_provider="openrouter"
        )
        with patch("aura.synthesis.litellm.acompletion", AsyncMock(side_effect=error)):
            result = await synthesize_answer([_make_fact(1, "fact")], "q", "en-US", model=MODEL)
        assert result is None

    async def test_the_real_failure_reason_is_what_gets_logged(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        error = litellm.exceptions.AuthenticationError(
            message="a very specific auth failure reason", llm_provider="openrouter", model="fake"
        )
        with caplog.at_level(logging.ERROR):
            with patch("aura.synthesis.litellm.acompletion", AsyncMock(side_effect=error)):
                result = await synthesize_answer(
                    [_make_fact(1, "fact")], "q", "en-US", model=MODEL
                )

        assert result is None
        # The real cause is logged (via exc_info, not baked into the
        # message string) even though the user only ever sees a generic
        # localized error -- confirmed by inspecting the captured
        # exception object itself, not just the log line's text.
        logged_exceptions = [
            record.exc_info[1]
            for record in caplog.records
            if record.levelno == logging.ERROR and record.exc_info
        ]
        assert any(
            isinstance(exc, litellm.exceptions.AuthenticationError)
            and "very specific auth failure reason" in str(exc)
            for exc in logged_exceptions
        )


@pytest.mark.usefixtures("configured_settings")
class TestPromptContent:
    async def test_facts_are_numbered_starting_at_one(self) -> None:
        facts = [_make_fact(10, "alpha content"), _make_fact(20, "beta content")]
        mock_call = AsyncMock(return_value=_make_response(_payload("a", [])))

        with patch("aura.synthesis.litellm.acompletion", mock_call):
            await synthesize_answer(facts, "what is alpha?", "en-US", model=MODEL)

        _, kwargs = mock_call.call_args
        combined = " ".join(m["content"] for m in kwargs["messages"])
        assert "[1] alpha content" in combined
        assert "[2] beta content" in combined
        assert "what is alpha?" in combined

    async def test_prompt_asks_for_the_self_assessment_field(self) -> None:
        mock_call = AsyncMock(return_value=_make_response(_payload("a", [])))

        with patch("aura.synthesis.litellm.acompletion", mock_call):
            await synthesize_answer([_make_fact(1, "fact")], "q", "en-US", model=MODEL)

        _, kwargs = mock_call.call_args
        combined = " ".join(m["content"] for m in kwargs["messages"])
        assert "answers_question" in combined

    async def test_prompt_instructs_honest_framing_of_partial_coverage(self) -> None:
        # Phase 2b-3: a partial answer must post (answers_question=true) with
        # the gap named honestly, not be silenced by a low-confidence gate --
        # see CLAUDE.md's "Proactive Relief: Visibly Active by Design".
        mock_call = AsyncMock(return_value=_make_response(_payload("a", [])))

        with patch("aura.synthesis.litellm.acompletion", mock_call):
            await synthesize_answer([_make_fact(1, "fact")], "q", "en-US", model=MODEL)

        _, kwargs = mock_call.call_args
        system = next(m["content"] for m in kwargs["messages"] if m["role"] == "system")
        assert "PART" in system
        assert "honestly" in system.lower()

    async def test_prompt_instructs_a_contradiction_check_before_citing(self) -> None:
        # Phase 2b-3: the model, not a numeric threshold, is responsible for
        # catching two cited facts that genuinely conflict with each other.
        mock_call = AsyncMock(return_value=_make_response(_payload("a", [])))

        with patch("aura.synthesis.litellm.acompletion", mock_call):
            await synthesize_answer([_make_fact(1, "fact")], "q", "en-US", model=MODEL)

        _, kwargs = mock_call.call_args
        system = next(m["content"] for m in kwargs["messages"] if m["role"] == "system")
        assert "conflict" in system.lower()
        assert "contradiction" in system.lower()

    async def test_prompt_instructs_the_model_to_ignore_embedded_instructions(self) -> None:
        # The anti-injection defence in the prompt: the message is data, not
        # instructions. Asserted so a future prompt edit cannot silently drop it.
        mock_call = AsyncMock(return_value=_make_response(_payload("a", [])))

        with patch("aura.synthesis.litellm.acompletion", mock_call):
            await synthesize_answer([_make_fact(1, "fact")], "q", "en-US", model=MODEL)

        _, kwargs = mock_call.call_args
        system = next(m["content"] for m in kwargs["messages"] if m["role"] == "system")
        assert "instructions" in system.lower()

    async def test_prompt_includes_the_target_language_derived_from_locale(self) -> None:
        mock_call = AsyncMock(return_value=_make_response(_payload("a", [])))

        with patch("aura.synthesis.litellm.acompletion", mock_call):
            await synthesize_answer([_make_fact(1, "fact")], "q", "de", model=MODEL)

        _, kwargs = mock_call.call_args
        combined = " ".join(m["content"] for m in kwargs["messages"])
        assert "German" in combined

    async def test_every_supported_locale_maps_to_a_distinct_language_name(self) -> None:
        locale_to_language = {
            "en-US": "English",
            "es-ES": "Spanish",
            "pt-BR": "Portuguese",
            "de": "German",
            "fr": "French",
            "tr": "Turkish",
            "pl": "Polish",
            "ja": "Japanese",
            "ko": "Korean",
        }
        for locale, expected_fragment in locale_to_language.items():
            mock_call = AsyncMock(return_value=_make_response(_payload("a", [])))
            with patch("aura.synthesis.litellm.acompletion", mock_call):
                await synthesize_answer([_make_fact(1, "fact")], "q", locale, model=MODEL)

            _, kwargs = mock_call.call_args
            combined = " ".join(m["content"] for m in kwargs["messages"])
            assert expected_fragment in combined, f"locale {locale!r} did not map to {expected_fragment!r}"

    async def test_unsupported_locale_falls_back_to_english(self) -> None:
        mock_call = AsyncMock(return_value=_make_response(_payload("a", [])))

        with patch("aura.synthesis.litellm.acompletion", mock_call):
            await synthesize_answer([_make_fact(1, "fact")], "q", "vi", model=MODEL)  # not one of Aura's 9

        _, kwargs = mock_call.call_args
        combined = " ".join(m["content"] for m in kwargs["messages"])
        assert "English" in combined

    async def test_the_model_comes_from_the_parameter_and_the_key_from_settings(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The model is resolved by the caller through the seam and passed in;
        # only the api key still comes from settings.
        monkeypatch.setattr(
            "aura.synthesis.load_settings",
            lambda: _fake_settings(llm_api_key="the-real-key", synthesis_model="ignored/here"),
        )
        mock_call = AsyncMock(return_value=_make_response(_payload("a", [])))

        with patch("aura.synthesis.litellm.acompletion", mock_call):
            await synthesize_answer([_make_fact(1, "fact")], "q", "en-US", model="passed/in/model")

        _, kwargs = mock_call.call_args
        assert kwargs["model"] == "passed/in/model"  # the parameter, not settings.synthesis_model
        assert kwargs["api_key"] == "the-real-key"

    async def test_unicode_question_and_facts_are_passed_through(self) -> None:
        facts = [_make_fact(1, "サーバーのルールは 日本語 でも読めます 🎉")]
        mock_call = AsyncMock(return_value=_make_response(_payload("a", [])))

        with patch("aura.synthesis.litellm.acompletion", mock_call):
            await synthesize_answer(facts, "質問이 있어요 🎉", "ko", model=MODEL)

        _, kwargs = mock_call.call_args
        combined = " ".join(m["content"] for m in kwargs["messages"])
        assert "サーバーのルールは 日本語 でも読めます 🎉" in combined
        assert "質問이 있어요 🎉" in combined


class TestMarkdownFencedResponses:
    """A fenced ```json block must parse, because real providers really send them.

    Not hypothetical: the model bake-off (reports/model-bakeoff.txt) measured Anthropic models
    routed through OpenRouter returning correct JSON wrapped in a fence on 12 of
    12 calls, despite response_format={"type": "json_object"} and an explicit
    "no markdown" instruction in the system prompt. Before this was handled,
    every one of those calls became a silent None.
    """

    async def test_json_wrapped_in_a_tagged_fence_parses(self) -> None:
        facts = [_make_fact(11, "fact one")]
        fenced = f"```json\n{_payload('the answer', [1])}\n```"

        with patch("aura.synthesis.litellm.acompletion", AsyncMock(return_value=_make_response(fenced))):
            result = await synthesize_answer(facts, "q", "en-US", model=MODEL)

        assert result is not None
        assert result.answer == "the answer"
        assert result.used_fact_ids == [11]

    async def test_json_wrapped_in_an_untagged_fence_parses(self) -> None:
        facts = [_make_fact(11, "fact one")]
        fenced = f"```\n{_payload('the answer', [1])}\n```"

        with patch("aura.synthesis.litellm.acompletion", AsyncMock(return_value=_make_response(fenced))):
            result = await synthesize_answer(facts, "q", "en-US", model=MODEL)

        assert result is not None
        assert result.answer == "the answer"

    async def test_fence_with_surrounding_whitespace_parses(self) -> None:
        facts = [_make_fact(11, "fact one")]
        fenced = f"  \n\n```json\n{_payload('the answer', [1])}\n```  \n"

        with patch("aura.synthesis.litellm.acompletion", AsyncMock(return_value=_make_response(fenced))):
            result = await synthesize_answer(facts, "q", "en-US", model=MODEL)

        assert result is not None
        assert result.answer == "the answer"

    async def test_backticks_inside_the_answer_string_survive_unfencing(self) -> None:
        # The closing fence is the LAST ```, not the first one that appears
        # inside a string value -- otherwise this answer would be truncated.
        answer = "use the ``` fence like this: ```json"
        facts = [_make_fact(11, "fact one")]
        fenced = f"```json\n{_payload(answer, [1])}\n```"

        with patch("aura.synthesis.litellm.acompletion", AsyncMock(return_value=_make_response(fenced))):
            result = await synthesize_answer(facts, "q", "en-US", model=MODEL)

        assert result is not None
        assert result.answer == answer

    async def test_unfenced_json_containing_backticks_is_untouched(self) -> None:
        # Bare JSON parses on the first attempt, so the fence fallback must
        # never run and never get a chance to mangle it.
        answer = "wrap code in ``` when posting"
        facts = [_make_fact(11, "fact one")]

        with patch(
            "aura.synthesis.litellm.acompletion",
            AsyncMock(return_value=_make_response(_payload(answer, [1]))),
        ):
            result = await synthesize_answer(facts, "q", "en-US", model=MODEL)

        assert result is not None
        assert result.answer == answer

    @pytest.mark.parametrize(
        "content",
        [
            "```json\n{\"answer\": \"unterminated",  # opening fence, never closed
            "```",  # a fence and nothing else
            "```json",  # tag, no newline, no body
            "```\n\n```",  # fenced emptiness
            "```json\nnot json at all\n```",  # correctly fenced, still not JSON
            "Here you go:\n```json\n{}\n```",  # prose before the fence
        ],
    )
    async def test_unparseable_fenced_content_still_fails_closed(self, content: str) -> None:
        facts = [_make_fact(11, "fact one")]

        with patch("aura.synthesis.litellm.acompletion", AsyncMock(return_value=_make_response(content))):
            result = await synthesize_answer(facts, "q", "en-US", model=MODEL)

        assert result is None


@pytest.mark.skipif(
    not os.environ.get("AURA_RUN_REAL_LLM"),
    reason="opt-in only: set AURA_RUN_REAL_LLM=1 (plus a funded LLM_API_KEY and "
    "SYNTHESIS_MODEL) to make one real, paid call. Deliberately NOT keyed off "
    "LLM_API_KEY, which `import litellm` loads from this repo's .env -- so the "
    "automated suite never spends money by accident. Not part of the phase's DoD.",
)
class TestRealProviderSanityCheck:
    """One true end-to-end call against a real provider. Costs real money; run manually."""

    async def test_real_synthesis_call_succeeds(self) -> None:
        model = os.environ.get("SYNTHESIS_MODEL", "")
        facts = [_make_fact(1, "The server was founded in 2020.")]
        result = await synthesize_answer(
            facts, "When was the server founded?", "en-US", model=model
        )
        assert result is not None
        assert result.answer

    async def test_venting_that_mentions_a_real_topic_is_not_answered(self) -> None:
        # Regression test for a real failure Phase 2b-3's own adversarial pass
        # found: reports/phase-2b-3.txt's synthetic-corpus simulation caught
        # two toxic/rhetorical messages that got answers_question=true (and
        # would have posted) because the first prompt draft's partial-coverage
        # instruction ("a fact partially covering the message counts as
        # answering") outweighed the sarcasm/rhetorical rule whenever the vent
        # happened to mention a real topic. Fixed by checking intent (is this
        # a genuine request at all) BEFORE coverage in the prompt. This is the
        # Japanese failing case verbatim, translated to English so the test
        # doesn't depend on non-English tokenization; the original-language
        # case was verified manually 3x stable against the live model before
        # this test was added (see the Phase 2b-3 conversation).
        model = os.environ.get("PROACTIVE_MODEL") or os.environ.get("SYNTHESIS_MODEL", "")
        facts = [
            _make_fact(
                1, "The weekly ranked tournament is held every Friday at 21:00."
            )
        ]
        venting = (
            "Hey bot, do you even work properly? You can't even get the weekly "
            "tournament prep done, you're completely useless."
        )
        result = await synthesize_answer(facts, venting, "en-US", model=model)
        assert result is not None
        assert result.answers_question is False
