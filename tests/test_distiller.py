"""Tests for aura.extraction.distiller: the prompt it builds and the output it trusts.

Everything here is hermetic -- litellm.acompletion is mocked in every test, and
conftest's autouse guard fails the run if any test reaches a real one. What is
verified is Aura's own half of the call: that the prompt actually carries the
context the phase brief requires, and that a hostile or broken response cannot
become a staged candidate.

Whether the MODEL then behaves correctly on that prompt is a different question,
which no mock can answer. It is measured against real paid calls in
scripts/evaluate_extraction.py and reported in reports/phase-3a-2.txt.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest

from aura.db.extraction_queue import QueuedMessage
from aura.db.pending_facts import FactCategory
from aura.extraction.distiller import (
    _MAX_DISTILLED_CHARS,
    _MAX_MESSAGE_CHARS,
    _build_messages,
    distill_facts,
)

MODEL = "openrouter/anthropic/claude-haiku-4.5"
NOW = datetime(2026, 7, 30, 12, 0, 0, tzinfo=timezone.utc)


def _queued(message_id: int, content: str, *, created_at: datetime = NOW) -> QueuedMessage:
    return QueuedMessage(
        channel_id=500,
        message_id=message_id,
        guild_id=100,
        channel_name="announcements",
        content=content,
        message_created_at=created_at,
        enqueued_at=created_at,
    )


def _response(payload: object, *, fenced: bool = False):
    """A litellm-shaped response carrying `payload` as its JSON content."""
    body = json.dumps(payload)
    if fenced:
        body = f"```json\n{body}\n```"

    from litellm.types.utils import Choices, Message, ModelResponse

    return ModelResponse(
        choices=[Choices(message=Message(content=body, role="assistant"))]
    )


def _mock_llm(payload: object, *, fenced: bool = False) -> AsyncMock:
    return AsyncMock(return_value=_response(payload, fenced=fenced))


@pytest.fixture(autouse=True)
def _configured_llm(monkeypatch: pytest.MonkeyPatch):
    """distill_facts reads the API key through load_settings; give it one."""
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    yield


class TestPromptContent:
    """The phase brief's 'channel context now' decision, verified in the prompt itself."""

    def test_the_channel_name_is_passed_as_explicit_context(self) -> None:
        messages = _build_messages([_queued(1, "hello")], "mod-announcements")
        assert "#mod-announcements" in messages[1]["content"]

    def test_each_messages_timestamp_is_passed(self) -> None:
        earlier = datetime(2026, 7, 30, 9, 15, tzinfo=timezone.utc)
        messages = _build_messages([_queued(1, "starts tomorrow")], "general")
        assert NOW.isoformat() in messages[1]["content"]

        messages = _build_messages(
            [_queued(1, "starts tomorrow", created_at=earlier)], "general"
        )
        assert earlier.isoformat() in messages[1]["content"]

    def test_messages_are_numbered_from_one(self) -> None:
        messages = _build_messages([_queued(9, "a"), _queued(8, "b")], "general")
        body = messages[1]["content"]
        assert "[1] " in body and "[2] " in body
        # The real Discord IDs are deliberately NOT in the prompt: the model
        # answers in positional numbers and the mapping back is Aura's job.
        assert "9" not in body.split("MESSAGES")[1].replace("2026", "")

    def test_the_batch_is_fenced_and_labelled_untrusted(self) -> None:
        messages = _build_messages([_queued(1, "hello")], "general")
        assert "<<<MESSAGES" in messages[1]["content"]
        assert "untrusted" in messages[1]["content"].lower()

    def test_every_category_including_milestone_is_named_in_the_rules(self) -> None:
        system = _build_messages([_queued(1, "x")], "general")[0]["content"]
        for category in FactCategory:
            assert category.value in system

    def test_the_milestone_reaction_rule_is_stated(self) -> None:
        # The distinction reports/phase-3a-1b.txt found embeddings cannot make,
        # and the reason this call exists at all for that category.
        system = _build_messages([_queued(1, "x")], "general")[0]["content"]
        assert "congrats" in system.lower()
        assert "reaction" in system.lower()

    def test_the_known_filter_weaknesses_are_named_as_rejections(self) -> None:
        # Hedges, sarcasm and hypotheticals: the three categories
        # reports/phase-3a-1b.txt measured the local filter leaking, and the
        # specific job this call is being paid to do.
        system = _build_messages([_queued(1, "x")], "general")[0]["content"].lower()
        assert "hedged" in system
        assert "sarcasm" in system
        assert "hypothetical" in system

    def test_message_independence_is_stated_as_a_rule(self) -> None:
        system = _build_messages([_queued(1, "x")], "general")[0]["content"]
        assert "INDEPENDENT" in system

    def test_agreement_replies_are_named_as_a_rejection(self) -> None:
        # Found by the real-model evaluation, not by reasoning: "yeah lets do
        # that" was distilled into "The server has decided to move game night
        # to Thursdays", reading the proposal out of a neighbouring message.
        # A general independence rule did not prevent it; naming the shape did.
        system = _build_messages([_queued(1, "x")], "general")[0]["content"]
        assert "AGREEMENT" in system

    def test_double_quotes_inside_values_are_forbidden(self) -> None:
        # The same hazard reports/phase-3a-3.txt Section 7 found and fixed in
        # aura.extraction.supersession: a typographic quote closing with a
        # straight ASCII double quote inside a JSON string value ends the
        # string early and the whole batch's response becomes unparseable.
        # Distillation is more exposed than supersession's one-sentence
        # reasoning field, since content is the payload, not a side note.
        system = _build_messages([_queued(1, "x")], "general")[0]["content"]
        assert "double-quote character" in system
        assert "single quotes" in system

    def test_the_source_language_rule_is_stated_and_carried_into_the_schema(self) -> None:
        # Two halves of one fix, both load-bearing: the first evaluation run
        # had seven of nine locales silently rendered into English, and an
        # instruction alone still left a mixed-language batch translating
        # Polish into French. The per-fact `language` slot is what closed it.
        system = _build_messages([_queued(1, "x")], "general")[0]["content"]
        assert "SAME LANGUAGE AS THE MESSAGE IT CAME FROM" in system
        assert "NEVER translate into English" in system
        assert '"language"' in system

    async def test_the_language_slot_is_required_but_never_stored(self) -> None:
        # It is a self-consistency device, not data. A response missing it is
        # rejected (the slot has to actually be filled to do its job), and a
        # DistilledFact deliberately does not carry it onward.
        payload = {
            "facts": [{"message": 1, "content": "A rule exists.", "category": "rule"}]
        }
        with patch("litellm.acompletion", _mock_llm(payload)):
            assert await distill_facts([_queued(1, "x")], channel_name="g", model=MODEL) is None

        payload = {
            "facts": [
                {
                    "message": 1,
                    "content": "Eine Regel existiert.",
                    "category": "rule",
                    "language": "German",
                }
            ]
        }
        with patch("litellm.acompletion", _mock_llm(payload)):
            result = await distill_facts([_queued(1, "x")], channel_name="g", model=MODEL)
        assert result is not None
        assert not hasattr(result[0], "language")

    def test_no_stored_facts_can_reach_the_prompt(self) -> None:
        # "Judgment, never knowledge", enforced structurally: _build_messages
        # takes only the batch and a channel name, so there is no parameter
        # through which an existing fact could enter this call even by mistake.
        import inspect

        parameters = set(inspect.signature(_build_messages).parameters)
        assert parameters == {"candidates", "channel_name"}

    def test_an_oversized_message_is_truncated_in_the_prompt(self) -> None:
        messages = _build_messages([_queued(1, "x" * 5000)], "general")
        assert "x" * _MAX_MESSAGE_CHARS in messages[1]["content"]
        assert "x" * (_MAX_MESSAGE_CHARS + 1) not in messages[1]["content"]


class TestSuccessfulDistillation:
    async def test_a_well_formed_response_maps_numbers_back_to_message_ids(self) -> None:
        batch = [_queued(1111, "server down at 2"), _queued(2222, "lol ok")]
        payload = {
            "facts": [
                {
                    "message": 1,
                    "content": "The server is down for maintenance at 14:00 on 2026-07-30.",
                    "category": "status_change",
                    "language": "English",
                }
            ]
        }

        with patch("litellm.acompletion", _mock_llm(payload)):
            result = await distill_facts(batch, channel_name="general", model=MODEL)

        assert result is not None
        assert len(result) == 1
        assert result[0].message_id == 1111
        assert result[0].category is FactCategory.STATUS_CHANGE

    async def test_an_empty_result_is_success_not_failure(self) -> None:
        # The common case by far, and the caller must not confuse it with a
        # failed call: empty clears the batch, None is a broken result.
        with patch("litellm.acompletion", _mock_llm({"facts": []})):
            result = await distill_facts(
                [_queued(1, "good morning")], channel_name="general", model=MODEL
            )
        assert result == []

    async def test_a_fenced_response_is_parsed(self) -> None:
        # EXTRACTION_MODEL ships as an Anthropic model routed through
        # OpenRouter, which wraps its JSON in a ```json fence on every call.
        # Without the shared fence-tolerant parser this fails 100% of the time.
        payload = {
            "facts": [{"message": 1, "content": "A rule exists.", "category": "rule", "language": "English"}]
        }
        with patch("litellm.acompletion", _mock_llm(payload, fenced=True)):
            result = await distill_facts(
                [_queued(1, "x")], channel_name="general", model=MODEL
            )
        assert result is not None
        assert len(result) == 1

    async def test_an_empty_batch_short_circuits_without_calling_the_model(self) -> None:
        with patch("litellm.acompletion", AsyncMock()) as llm:
            assert await distill_facts([], channel_name="general", model=MODEL) == []
        llm.assert_not_awaited()

    async def test_duplicate_identical_entries_are_collapsed(self) -> None:
        payload = {
            "facts": [
                {"message": 1, "content": "The rule exists.", "category": "rule", "language": "English"},
                {"message": 1, "content": "The rule exists.", "category": "rule", "language": "English"},
            ]
        }
        with patch("litellm.acompletion", _mock_llm(payload)):
            result = await distill_facts(
                [_queued(1, "x")], channel_name="general", model=MODEL
            )
        assert result is not None
        assert len(result) == 1

    async def test_one_message_may_yield_two_different_facts(self) -> None:
        payload = {
            "facts": [
                {"message": 1, "content": "Maintenance is at 14:00.", "category": "event", "language": "English"},
                {"message": 1, "content": "Voice chat is disabled.", "category": "status_change", "language": "English"},
            ]
        }
        with patch("litellm.acompletion", _mock_llm(payload)):
            result = await distill_facts(
                [_queued(77, "x")], channel_name="general", model=MODEL
            )
        assert result is not None
        assert len(result) == 2
        assert {fact.message_id for fact in result} == {77}


class TestMalformedOutputIsRejected:
    """Every one of these is a real failure mode at this call site, not a hypothetical.

    All of them must become a clean None: the batch is then cleared and nothing
    is staged, which is the safe direction. The alternative -- salvaging the
    parseable entries of a response that also contains a hallucinated citation
    -- would stage sentences produced by a model that has demonstrably
    misunderstood the task.
    """

    @pytest.mark.parametrize(
        "payload",
        [
            pytest.param({"facts": "not a list"}, id="facts-is-not-a-list"),
            pytest.param({"wrong_key": []}, id="missing-facts-key"),
            pytest.param({"facts": [{"content": "x", "category": "rule"}]}, id="no-message"),
            pytest.param({"facts": [{"message": 1, "category": "rule"}]}, id="no-content"),
            pytest.param({"facts": [{"message": 1, "content": "x"}]}, id="no-category"),
            pytest.param(
                {"facts": [{"message": 1, "content": "x", "category": "gossip"}]},
                id="unknown-category",
            ),
            pytest.param(
                {"facts": [{"message": "one", "content": "x", "category": "rule"}]},
                id="message-not-an-int",
            ),
        ],
    )
    async def test_a_malformed_shape_becomes_none(self, payload: object) -> None:
        with patch("litellm.acompletion", _mock_llm(payload)):
            result = await distill_facts(
                [_queued(1, "x")], channel_name="general", model=MODEL
            )
        assert result is None

    @pytest.mark.parametrize("number", [0, 2, 99, -1])
    async def test_a_hallucinated_message_number_becomes_none(self, number: int) -> None:
        payload = {
            "facts": [{"message": number, "content": "A fact.", "category": "rule"}]
        }
        with patch("litellm.acompletion", _mock_llm(payload)):
            result = await distill_facts(
                [_queued(1, "x")], channel_name="general", model=MODEL
            )
        assert result is None

    async def test_one_bad_entry_discards_the_whole_response(self) -> None:
        # Deliberate: a model that cited a message outside the batch has
        # misunderstood the task, and its other entries are not more
        # trustworthy for happening to parse.
        payload = {
            "facts": [
                {"message": 1, "content": "A real fact.", "category": "rule"},
                {"message": 99, "content": "A fabricated one.", "category": "rule"},
            ]
        }
        with patch("litellm.acompletion", _mock_llm(payload)):
            result = await distill_facts(
                [_queued(1, "x")], channel_name="general", model=MODEL
            )
        assert result is None

    @pytest.mark.parametrize("content", ["", "   ", "\n\t "])
    async def test_a_blank_sentence_becomes_none(self, content: str) -> None:
        payload = {"facts": [{"message": 1, "content": content, "category": "rule"}]}
        with patch("litellm.acompletion", _mock_llm(payload)):
            result = await distill_facts(
                [_queued(1, "x")], channel_name="general", model=MODEL
            )
        assert result is None

    async def test_a_sentence_that_is_not_distilled_at_all_is_rejected(self) -> None:
        # A "distilled sentence" of 4000 characters is a copy of the message,
        # not a distillation. Rejected rather than truncated: half a sentence
        # stated as fact is worse than no sentence.
        payload = {
            "facts": [
                {
                    "message": 1,
                    "content": "x" * (_MAX_DISTILLED_CHARS + 1),
                    "category": "rule",
                }
            ]
        }
        with patch("litellm.acompletion", _mock_llm(payload)):
            result = await distill_facts(
                [_queued(1, "x")], channel_name="general", model=MODEL
            )
        assert result is None

    async def test_unparseable_json_becomes_none(self) -> None:
        from litellm.types.utils import Choices, Message, ModelResponse

        broken = ModelResponse(
            choices=[Choices(message=Message(content="{not json", role="assistant"))]
        )
        with patch("litellm.acompletion", AsyncMock(return_value=broken)):
            result = await distill_facts(
                [_queued(1, "x")], channel_name="general", model=MODEL
            )
        assert result is None

    async def test_empty_response_content_becomes_none(self) -> None:
        from litellm.types.utils import Choices, Message, ModelResponse

        empty = ModelResponse(
            choices=[Choices(message=Message(content="", role="assistant"))]
        )
        with patch("litellm.acompletion", AsyncMock(return_value=empty)):
            result = await distill_facts(
                [_queued(1, "x")], channel_name="general", model=MODEL
            )
        assert result is None

    async def test_a_network_failure_becomes_none(self) -> None:
        with patch("litellm.acompletion", AsyncMock(side_effect=OSError("connection reset"))):
            result = await distill_facts(
                [_queued(1, "x")], channel_name="general", model=MODEL
            )
        assert result is None

    async def test_a_timeout_becomes_none(self) -> None:
        with patch("litellm.acompletion", AsyncMock(side_effect=TimeoutError())):
            result = await distill_facts(
                [_queued(1, "x")], channel_name="general", model=MODEL
            )
        assert result is None

    async def test_a_cancellation_still_propagates(self) -> None:
        # CancelledError is a BaseException, so a shutdown cancelling the
        # sweeper must not be swallowed as "the model failed".
        import asyncio

        with patch("litellm.acompletion", AsyncMock(side_effect=asyncio.CancelledError())):
            with pytest.raises(asyncio.CancelledError):
                await distill_facts([_queued(1, "x")], channel_name="general", model=MODEL)


class TestConfigurationGuards:
    async def test_a_missing_model_returns_none_without_calling(self) -> None:
        with patch("litellm.acompletion", AsyncMock()) as llm:
            result = await distill_facts(
                [_queued(1, "x")], channel_name="general", model=""
            )
        assert result is None
        llm.assert_not_awaited()

    async def test_a_missing_api_key_returns_none_without_calling(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("LLM_API_KEY", raising=False)
        # load_settings also reads .env, which this repo has; point it away.
        monkeypatch.setattr(
            "aura.extraction.distiller.load_settings",
            lambda: __import__("aura.config", fromlist=["Settings"]).Settings(
                _env_file=None, discord_token="fake-token"
            ),
        )
        with patch("litellm.acompletion", AsyncMock()) as llm:
            result = await distill_facts(
                [_queued(1, "x")], channel_name="general", model=MODEL
            )
        assert result is None
        llm.assert_not_awaited()


class TestCallParameters:
    async def test_the_call_is_pinned_deterministic_and_bounded(self) -> None:
        llm = _mock_llm({"facts": []})
        with patch("litellm.acompletion", llm):
            await distill_facts([_queued(1, "x")], channel_name="general", model=MODEL)

        kwargs = llm.call_args.kwargs
        # A fact that appears or vanishes with the sampling seed is not a fact.
        assert kwargs["temperature"] == 0.0
        assert kwargs["response_format"] == {"type": "json_object"}
        # Nothing waits on this call, but a hung request must not pin the
        # sweeper and the batch it holds forever.
        assert kwargs["timeout"] > 0
        assert kwargs["model"] == MODEL
