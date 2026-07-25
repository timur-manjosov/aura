"""Tests for aura.proactive.question_detector: contrastive question-likeness scoring.

Uses the real embedding model (the session-scoped embedding_model fixture in
conftest.py) wherever real semantic behaviour is what's under test -- that
questions in nine different languages outscore statements is the entire
point of this module, and a mock cannot exercise it. Fakes appear only where
the test is about the plumbing rather than the meaning: proving inference is
offloaded off the event loop, and forcing failure modes the real model will
not produce on demand.

The scoring here is contrastive (question-exemplar similarity minus
statement-exemplar similarity), so the assertions differ from Phase 2a-1's in
two ways worth stating up front: the range is [-2, 2] rather than [-1, 1], and
a score of 0.0 is the neutral middle of the scale rather than its floor --
which is why unscoreable text now returns the floor instead of zero.

No Discord anywhere in this file, per CLAUDE.md's testing principle.
"""
from __future__ import annotations

import asyncio
import threading
from collections.abc import Iterable

import numpy as np
import pytest
from fastembed import TextEmbedding

from aura.config import Settings
from aura.embeddings import cosine_similarity, embed_text
from aura.proactive.question_detector import (
    _MAX_CLASSIFIED_CHARACTERS,
    _NO_QUESTION_EVIDENCE,
    QUESTION_EXEMPLARS,
    STATEMENT_EXEMPLARS,
    QuestionDetector,
)

_MODEL_DIM = 384


class _FakeEmbeddingModel:
    """Minimal stand-in for TextEmbedding, exposing only the embed() call used here."""

    def __init__(self, vector: np.ndarray | None = None) -> None:
        self.vector = np.ones(4, dtype=np.float32) if vector is None else vector
        self.calls: list[list[str]] = []
        self.thread_idents: list[int] = []

    def embed(self, documents: list[str], **_kwargs: object) -> Iterable[np.ndarray]:
        self.calls.append(list(documents))
        self.thread_idents.append(threading.get_ident())
        return [self.vector for _ in documents]


class _ExplodingEmbeddingModel:
    """Stands in for the embedding model failing mid-inference."""

    def embed(self, _documents: list[str], **_kwargs: object) -> Iterable[np.ndarray]:
        raise RuntimeError("ONNX Runtime session failed")


async def _fake_detector(model: _FakeEmbeddingModel) -> QuestionDetector:
    """A detector over one trivial exemplar per side, for plumbing-only tests."""
    return await QuestionDetector.create(
        model,  # type: ignore[arg-type]
        question_exemplars=("q",),
        statement_exemplars=("s",),
    )


@pytest.fixture(scope="session")
async def detector(embedding_model: TextEmbedding) -> QuestionDetector:
    """One real detector for the whole session -- exemplar embedding is the expensive part."""
    return await QuestionDetector.create(embedding_model)


class TestExemplarSets:
    def test_question_set_size_stays_in_the_intended_range(self) -> None:
        assert 15 <= len(QUESTION_EXEMPLARS) <= 20

    def test_the_statement_set_is_the_same_size_as_the_question_set(self) -> None:
        # The contrastive score subtracts a maximum over one set from a
        # maximum over the other. A lopsided pair would bias every score in
        # the direction of whichever set had more chances to match, which
        # would look like a threshold problem rather than a set-size problem.
        assert len(STATEMENT_EXEMPLARS) == len(QUESTION_EXEMPLARS)

    @pytest.mark.parametrize("exemplars", [QUESTION_EXEMPLARS, STATEMENT_EXEMPLARS])
    def test_no_exemplar_is_blank(self, exemplars: tuple[str, ...]) -> None:
        assert all(exemplar.strip() for exemplar in exemplars)

    @pytest.mark.parametrize("exemplars", [QUESTION_EXEMPLARS, STATEMENT_EXEMPLARS])
    def test_exemplars_are_unique(self, exemplars: tuple[str, ...]) -> None:
        assert len(set(exemplars)) == len(exemplars)

    def test_the_two_sets_share_no_sentence(self) -> None:
        # A sentence in both sets contributes equally to each side and
        # cancels itself out, which is a quiet way of shrinking the set.
        assert not set(QUESTION_EXEMPLARS) & set(STATEMENT_EXEMPLARS)

    @pytest.mark.parametrize("exemplars", [QUESTION_EXEMPLARS, STATEMENT_EXEMPLARS])
    def test_each_set_is_genuinely_multilingual_not_english_with_decoration(
        self, exemplars: tuple[str, ...]
    ) -> None:
        # The whole design rests on one shared set covering nine locales. An
        # exemplar set that quietly drifted back to English-only would still
        # pass every scoring test below on English input while degrading for
        # everyone else, so the multilingual property is asserted directly.
        non_ascii_exemplars = [e for e in exemplars if not e.isascii()]
        assert len(non_ascii_exemplars) >= 8

    def test_both_sets_carry_the_same_language_mix(self) -> None:
        # Mirrored language-for-language, not merely both-multilingual: if the
        # statement side were missing a language the question side has, then
        # for messages in that language the subtraction would have nothing
        # same-language to cancel against, and the topical bias the
        # contrastive score exists to remove would come straight back.
        assert sum(e.isascii() for e in QUESTION_EXEMPLARS) == sum(
            e.isascii() for e in STATEMENT_EXEMPLARS
        )

    def test_the_statement_set_is_not_interrogative(self) -> None:
        # Not a proxy for the semantics (the model does not read punctuation),
        # but a guard against the set drifting into questions over time, which
        # would make the subtraction cancel the very signal it measures.
        assert not any("?" in e or "？" in e for e in STATEMENT_EXEMPLARS)


class TestCreate:
    async def test_embeds_every_exemplar_from_both_sets_exactly_once(self) -> None:
        fake = _FakeEmbeddingModel()
        await QuestionDetector.create(
            fake,  # type: ignore[arg-type]
            question_exemplars=("a", "b"),
            statement_exemplars=("c", "d", "e"),
        )

        # One batched call for all five, not one call per set and not five
        # calls: the fixed per-invocation cost dominates at these sizes.
        assert len(fake.calls) == 1
        assert fake.calls[0] == ["a", "b", "c", "d", "e"]

    async def test_the_two_sets_are_kept_apart_after_the_shared_batch(self) -> None:
        # The batch is split by length. Getting that boundary wrong would mix
        # statement vectors into the question side, where the score would
        # still look plausible -- close to zero -- while measuring nothing.
        fake = _FakeEmbeddingModel()
        detector = await QuestionDetector.create(
            fake,  # type: ignore[arg-type]
            question_exemplars=("a", "b"),
            statement_exemplars=("c", "d", "e"),
        )

        assert len(detector._question_embeddings) == 2
        assert len(detector._statement_embeddings) == 3

    @pytest.mark.parametrize(
        ("questions", "statements"),
        [((), ("s",)), (("q",), ()), ((), ())],
        ids=["no-questions", "no-statements", "neither"],
    )
    async def test_an_empty_exemplar_set_is_rejected_at_construction(
        self, questions: tuple[str, ...], statements: tuple[str, ...]
    ) -> None:
        # An empty side makes max() raise on every single message, i.e. a
        # startup misconfiguration that only surfaces once traffic arrives.
        with pytest.raises(ValueError):
            await QuestionDetector.create(
                _FakeEmbeddingModel(),  # type: ignore[arg-type]
                question_exemplars=questions,
                statement_exemplars=statements,
            )

    async def test_exemplars_are_embedded_off_the_event_loop(self) -> None:
        fake = _FakeEmbeddingModel()
        await _fake_detector(fake)

        assert fake.thread_idents
        assert all(ident != threading.get_ident() for ident in fake.thread_idents)

    async def test_the_exemplars_are_never_re_embedded_per_message(self) -> None:
        # The caching discipline the whole design depends on: adding the
        # statement side must cost one extra batch at startup and nothing per
        # message. Counting calls proves it rather than assuming it.
        fake = _FakeEmbeddingModel()
        detector = await _fake_detector(fake)
        calls_after_construction = len(fake.calls)

        for index in range(5):
            await detector.question_likeness(f"message number {index}")

        # Exactly one embedding call per message, for the message itself.
        assert len(fake.calls) == calls_after_construction + 5
        assert all(len(call) == 1 for call in fake.calls[calls_after_construction:])


class TestContrastiveSemantics:
    """The behaviour the rescoring exists to produce."""

    async def test_the_score_is_question_similarity_minus_statement_similarity(
        self, detector: QuestionDetector, embedding_model: TextEmbedding
    ) -> None:
        text = "Weiß jemand, wo die Serverregeln stehen?"
        embedding = await embed_text(embedding_model, text)
        question_side = max(
            cosine_similarity(e, embedding) for e in detector._question_embeddings
        )
        statement_side = max(
            cosine_similarity(e, embedding) for e in detector._statement_embeddings
        )

        score = await detector.question_likeness(text)

        assert score == pytest.approx(question_side - statement_side)

    async def test_each_side_is_a_maximum_not_a_mean(
        self, detector: QuestionDetector, embedding_model: TextEmbedding
    ) -> None:
        text = "Weiß jemand, wo die Serverregeln stehen?"
        embedding = await embed_text(embedding_model, text)
        per_question = [cosine_similarity(e, embedding) for e in detector._question_embeddings]
        per_statement = [
            cosine_similarity(e, embedding) for e in detector._statement_embeddings
        ]

        score = await detector.question_likeness(text)

        assert score == pytest.approx(max(per_question) - max(per_statement))
        mean_difference = sum(per_question) / len(per_question) - sum(per_statement) / len(
            per_statement
        )
        assert score > mean_difference

    async def test_a_question_exemplar_scores_positively_against_itself(
        self, detector: QuestionDetector
    ) -> None:
        # It matches its own side at 1.0, so the score is 1.0 minus whatever
        # the nearest statement scores -- clearly positive, but no longer
        # exactly 1.0 the way the one-sided score was.
        score = await detector.question_likeness(QUESTION_EXEMPLARS[0])
        assert 0.0 < score < 1.0

    async def test_a_statement_exemplar_scores_negatively_against_itself(
        self, detector: QuestionDetector
    ) -> None:
        score = await detector.question_likeness(STATEMENT_EXEMPLARS[0])
        assert -1.0 < score < 0.0

    async def test_a_paired_question_outscores_its_own_statement_twin(
        self, detector: QuestionDetector
    ) -> None:
        # The two sets are mirrored topic-for-topic, so this pair differs
        # almost entirely in interrogative form -- the hardest possible test
        # of whether form is what is being measured.
        for question, statement in zip(QUESTION_EXEMPLARS, STATEMENT_EXEMPLARS):
            assert await detector.question_likeness(question) > await detector.question_likeness(
                statement
            ), f"{question!r} did not outscore {statement!r}"

    async def test_a_same_topic_statement_does_not_outscore_an_unrelated_question(
        self, detector: QuestionDetector
    ) -> None:
        # The exact failure the one-sided score had: this embedding space
        # clusters by topic, so a statement *about the rules* used to beat a
        # question about something else entirely. Subtracting the statement
        # side is what fixes it, and this is the regression test for that fix.
        same_topic_statement = "The server rules are pinned at the top of this channel."
        unrelated_question = "does anyone know if the tournament sign-up is still open"

        assert await detector.question_likeness(
            unrelated_question
        ) > await detector.question_likeness(same_topic_statement)

    async def test_a_same_topic_statement_in_another_language_also_loses(
        self, detector: QuestionDetector
    ) -> None:
        # The cross-language form of the same failure, which is why both
        # exemplar sets carry the same language mix.
        assert await detector.question_likeness(
            "anyone know where the sign-up form went"
        ) > await detector.question_likeness("Die Serverregeln stehen ganz oben im Kanal.")


class TestSemanticBehaviour:
    @pytest.mark.parametrize(
        "question",
        [
            "How do I change my nickname here?",
            "Wo finde ich die Regeln für diesen Server?",
            "¿Alguien me puede decir cuándo empieza el evento?",
            "Quelqu'un sait où sont les règles du serveur ?",
            "この設定はどこで変更できますか？",
            "누가 이거 설정하는 방법 알려줄 수 있나요?",
            "Gdzie znajdę informacje o tym serwerze?",
            "Bu kanala nasıl erişebilirim?",
            "Alguém sabe onde ficam as regras?",
        ],
    )
    async def test_questions_in_every_supported_language_outscore_statements(
        self, detector: QuestionDetector, question: str
    ) -> None:
        # Statements in several languages, so a question is never merely
        # beating a *foreign-language* statement by language alone.
        statements = [
            "I just finished my coffee and it was really good.",
            "Ich habe gestern einen ziemlich guten Film gesehen.",
            "El clima ha estado muy cálido esta semana.",
            "昨日は新しいラーメン屋に行きました。",
        ]
        question_score = await detector.question_likeness(question)
        statement_scores = [await detector.question_likeness(s) for s in statements]

        assert question_score > max(statement_scores)

    @pytest.mark.parametrize(
        "question",
        [
            "How do I change my nickname here?",
            "Wo finde ich die Regeln für diesen Server?",
            "¿Alguien me puede decir cuándo empieza el evento?",
            "この設定はどこで変更できますか？",
            "Alguém sabe onde ficam as regras?",
        ],
    )
    async def test_a_real_question_clears_the_shipped_stage_one_threshold(
        self, detector: QuestionDetector, question: str
    ) -> None:
        # Relative ordering is not enough once the score gates something: the
        # shipped default has to actually admit ordinary questions. Read from
        # Settings rather than hardcoded, so retuning the threshold in Phase
        # 2b updates this expectation with it.
        from aura.config import Settings

        threshold = Settings(  # type: ignore[call-arg]
            _env_file=None, discord_token="token"
        ).proactive_question_threshold

        assert await detector.question_likeness(question) >= threshold

    async def test_an_informal_question_with_no_question_mark_still_scores_as_one(
        self, detector: QuestionDetector
    ) -> None:
        # The specific case a regex over "?" and interrogative words cannot
        # catch, and the reason this detector is semantic at all.
        informal = "does anybody happen to know where the onboarding doc lives"
        statement = "I finally finished rearranging my desk this weekend."

        assert await detector.question_likeness(informal) > await detector.question_likeness(
            statement
        )

    async def test_scoring_is_deterministic_for_the_same_text(
        self, detector: QuestionDetector
    ) -> None:
        text = "Where can I read the rules?"
        first = await detector.question_likeness(text)
        second = await detector.question_likeness(text)
        assert first == second


class TestBlankInput:
    async def test_empty_string_scores_the_floor_without_running_inference(self) -> None:
        fake = _FakeEmbeddingModel()
        detector = await _fake_detector(fake)
        fake.calls.clear()

        assert await detector.question_likeness("") == _NO_QUESTION_EVIDENCE
        assert fake.calls == []  # no inference at all, not merely a discarded result

    @pytest.mark.parametrize("blank", ["   ", "\n\n", "\t \r\n ", "  ", "　"])
    async def test_whitespace_only_text_scores_the_floor(
        self, detector: QuestionDetector, blank: str
    ) -> None:
        assert await detector.question_likeness(blank) == _NO_QUESTION_EVIDENCE

    def test_the_unscoreable_sentinel_is_the_floor_and_not_the_neutral_midpoint(self) -> None:
        # The trap the rescoring introduced: 0.0 was the bottom of the old
        # one-sided scale but is the *middle* of this one, and the calibrated
        # Stage 1 threshold is negative -- so returning 0.0 for text that
        # could not be scored would have let empty and degenerate input
        # through the gate. ProactiveGateConfig forbids a threshold at or
        # below this value, which is what makes the floor unpassable.
        assert _NO_QUESTION_EVIDENCE == -2.0


class TestHostileInput:
    """Deliberate attempts to break the classifier with real-world garbage."""

    @pytest.mark.parametrize(
        "text",
        [
            "🎉",
            "🎉🎊🥳🎈🎂🍰🎁",
            "👨‍👩‍👧‍👦",  # ZWJ emoji sequence
            "🇩🇪🇯🇵🇰🇷",  # regional indicator pairs
            "Hello 世界 Привет مرحبا שלום 🎉",  # five scripts, one string
            "​‌‍",  # zero-width characters only
            "test​with​zero​width",
            "مرحبا بكم في الخادم، أين يمكنني أن أجد القواعد؟",  # RTL Arabic
            "שלום, איפה אפשר למצוא את הכללים?",  # RTL Hebrew
            "‮reversed override text‬",  # RTL override control chars
            "a\x00b",  # embedded null byte
            "\x01\x02\x03\x1b[31m",  # control characters and an ANSI escape
            "e" + "́" * 200,  # 200 stacked combining accents
            "𝕬𝖚𝖗𝖆 𝓺𝓾𝓮𝓼𝓽𝓲𝓸𝓷",  # mathematical alphanumeric symbols
            "{not_a_format_field} {{escaped}} %s %d",
            "'; DROP TABLE proactive_signals; --",
            "\\n\\t\\\\",
            "ﷺ" * 50,  # ligature that expands heavily under normalization
            "क्षि" * 100,  # Devanagari conjuncts
            "0",
            "?",
            "??????????",
        ],
    )
    async def test_hostile_text_returns_a_finite_score_and_never_raises(
        self, detector: QuestionDetector, text: str
    ) -> None:
        score = await detector.question_likeness(text)
        assert isinstance(score, float)
        assert -2.0 <= score <= 2.0

    async def test_a_full_log_dump_is_scored_without_crashing(
        self, detector: QuestionDetector
    ) -> None:
        log_dump = "2026-07-24T12:00:00Z ERROR something went wrong in module x\n" * 2000
        score = await detector.question_likeness(log_dump)
        assert -2.0 <= score <= 2.0

    async def test_text_beyond_the_cap_scores_identically_to_its_truncated_form(
        self, detector: QuestionDetector
    ) -> None:
        base = "where can I find the server rules " * 500
        assert len(base) > _MAX_CLASSIFIED_CHARACTERS

        full = await detector.question_likeness(base)
        truncated = await detector.question_likeness(base[:_MAX_CLASSIFIED_CHARACTERS])
        assert full == truncated

    async def test_a_message_of_a_million_characters_still_completes(
        self, detector: QuestionDetector
    ) -> None:
        # Far larger than Discord permits (4000 characters), so this is only
        # reachable through a bug elsewhere -- which is exactly when a hang
        # would be hardest to diagnose.
        score = await detector.question_likeness("x" * 1_000_000)
        assert -2.0 <= score <= 2.0


class TestDegenerateEmbeddings:
    async def test_a_nan_embedding_scores_the_floor_instead_of_propagating_nan(self) -> None:
        # A NaN score would lose every comparison against the Stage 1
        # threshold silently, and SQLite stores a NaN REAL as NULL -- which the
        # NOT NULL column would then reject at write time, turning a bad
        # vector into a lost row and a confusing traceback.
        nan_model = _FakeEmbeddingModel(np.full(4, np.nan, dtype=np.float32))
        detector = await _fake_detector(nan_model)

        assert await detector.question_likeness("anything") == _NO_QUESTION_EVIDENCE

    async def test_an_all_zero_embedding_scores_zero_difference_not_nan(self) -> None:
        # cosine_similarity returns 0.0 for a degenerate vector, so both sides
        # are 0.0 and the contrastive score is a legitimate 0.0 -- not NaN.
        # Neutral rather than the floor, because the subtraction genuinely
        # completed; the Stage 2 bar is what stops such a message going on.
        zero_model = _FakeEmbeddingModel(np.zeros(4, dtype=np.float32))
        detector = await _fake_detector(zero_model)

        assert await detector.question_likeness("anything") == 0.0

    async def test_a_failing_embedding_call_propagates_for_the_caller_to_handle(self) -> None:
        # question_likeness deliberately does not swallow this: the listener
        # is the layer that decides a failed classification is survivable
        # (see aura.proactive.listener.handle_message), and a detector that
        # silently returned a score on an inference failure would be
        # indistinguishable from one that scored a real non-question.
        detector = QuestionDetector(
            _ExplodingEmbeddingModel(),  # type: ignore[arg-type]
            [np.ones(4, dtype=np.float32)],
            [np.ones(4, dtype=np.float32)],
        )

        with pytest.raises(RuntimeError):
            await detector.question_likeness("anything")


class TestConcurrency:
    async def test_inference_runs_off_the_event_loop(self) -> None:
        # The Performance principle in CLAUDE.md, asserted rather than
        # assumed: a busy channel scoring messages must never be doing
        # ONNX inference on the loop thread.
        fake = _FakeEmbeddingModel()
        detector = await _fake_detector(fake)
        fake.thread_idents.clear()

        await detector.question_likeness("some message")

        assert fake.thread_idents
        assert all(ident != threading.get_ident() for ident in fake.thread_idents)

    async def test_a_burst_of_concurrent_messages_matches_a_sequential_baseline(
        self, detector: QuestionDetector
    ) -> None:
        # Simulates a busy channel: many messages scored at once must
        # produce exactly the results they would have produced one at a time.
        texts = [f"how do I do thing number {i} on this server?" for i in range(40)]

        sequential = [await detector.question_likeness(text) for text in texts]
        concurrent = await asyncio.gather(*(detector.question_likeness(t) for t in texts))

        assert concurrent == sequential

    async def test_concurrent_scoring_of_hostile_and_normal_text_interleaved(
        self, detector: QuestionDetector
    ) -> None:
        texts = [
            "where are the rules?",
            "🎉",
            "",
            "​",
            "x" * 5000,
            "サーバーのルールはどこですか？",
            "   ",
            "a\x00b",
        ] * 5

        scores = await asyncio.gather(*(detector.question_likeness(t) for t in texts))

        assert len(scores) == len(texts)
        assert all(isinstance(score, float) and -2.0 <= score <= 2.0 for score in scores)


class TestVectorShape:
    async def test_the_real_model_produces_the_expected_dimension_for_both_sets(
        self, detector: QuestionDetector
    ) -> None:
        # Guards the assumption every cosine comparison here rests on: both
        # exemplar sets and the incoming message must live in the same space.
        vectors = detector._question_embeddings + detector._statement_embeddings
        assert vectors
        assert all(vector.shape == (_MODEL_DIM,) for vector in vectors)
