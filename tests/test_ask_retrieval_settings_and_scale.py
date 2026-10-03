"""The three ASK_LEXICAL_* settings, and hybrid retrieval on an invented server of hundreds of facts."""

from __future__ import annotations

import aiosqlite
import pytest
from fastembed import TextEmbedding
from pydantic import ValidationError

from ask_retrieval_eval.cases import compute_metrics
from ask_retrieval_eval.production_path import QuestionOutcome, run_questions, seed_invented_guild
from ask_retrieval_eval.scale_corpus import ScaleCorpus, build_scale_corpus
from aura.config import Settings
from aura.db.repository import init_schema
from aura.embeddings import SYNTHESIS_FACT_LIMIT
from aura.i18n import t
from aura.i18n.translator import SUPPORTED_LOCALES
from aura.retrieval.hybrid import HybridRetrievalConfig
from aura.retrieval.index_cache import LexicalIndexCache
from aura.retrieval.lexical import LexicalIndex
from aura.retrieval.stopwords import shipped_stopwords

SCALE_GUILD = 1


def _settings(**overrides: object) -> Settings:
    return Settings(_env_file=None, discord_token="x", **overrides)  # type: ignore[arg-type]


class TestSettings:
    def test_defaults_are_the_measured_values(self) -> None:
        settings = _settings()
        assert settings.ask_lexical_coverage_threshold == 0.5
        assert settings.ask_lexical_similarity_floor == 0.05
        assert settings.ask_lexical_ranking_weight == 0.5
        assert settings.similarity_threshold == 0.4

    def test_they_are_read_from_the_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ASK_LEXICAL_COVERAGE_THRESHOLD", "0.6")
        monkeypatch.setenv("ASK_LEXICAL_SIMILARITY_FLOOR", "0.1")
        monkeypatch.setenv("ASK_LEXICAL_RANKING_WEIGHT", "0")
        settings = _settings()
        assert (
            settings.ask_lexical_coverage_threshold,
            settings.ask_lexical_similarity_floor,
            settings.ask_lexical_ranking_weight,
        ) == (0.6, 0.1, 0.0)

    @pytest.mark.parametrize(
        ("name", "value"),
        [
            ("ask_lexical_coverage_threshold", 0.0),  # would admit every fact above the floor
            ("ask_lexical_coverage_threshold", -0.1),
            ("ask_lexical_coverage_threshold", 1.01),
            ("ask_lexical_coverage_threshold", float("nan")),
            ("ask_lexical_similarity_floor", -1.01),
            ("ask_lexical_similarity_floor", 1.01),
            ("ask_lexical_similarity_floor", float("inf")),
            ("ask_lexical_ranking_weight", -0.01),
            ("ask_lexical_ranking_weight", 10.01),
            ("ask_lexical_ranking_weight", float("nan")),
        ],
    )
    def test_out_of_range_values_are_refused(self, name: str, value: float) -> None:
        with pytest.raises(ValidationError):
            _settings(**{name: value})

    @pytest.mark.parametrize(
        ("name", "value"),
        [
            ("ask_lexical_coverage_threshold", 1.0),
            ("ask_lexical_coverage_threshold", 0.01),
            ("ask_lexical_similarity_floor", -1.0),
            ("ask_lexical_similarity_floor", 1.0),
            ("ask_lexical_ranking_weight", 0.0),
            ("ask_lexical_ranking_weight", 10.0),
        ],
    )
    def test_the_bounds_themselves_are_accepted(self, name: str, value: float) -> None:
        assert getattr(_settings(**{name: value}), name) == value


class TestTheRelatedReplyText:
    @pytest.mark.parametrize("locale", sorted(SUPPORTED_LOCALES))
    def test_every_locale_has_its_own_text(self, locale: str) -> None:
        text = t("ask_no_info_related", locale)
        assert text and not text.startswith("[")
        if locale != "en-US":
            assert text != t("ask_no_info_related", "en-US")

    def test_german_reads_as_designed(self) -> None:
        assert t("ask_no_info_related", "de") == (
            "Dazu ist nichts Genaues gespeichert. Vielleicht verwandt:"
        )


class TestTheScaleCorpus:
    def test_it_is_deterministic_and_exact_in_size(self) -> None:
        first, second = build_scale_corpus(300), build_scale_corpus(300)
        assert first == second
        assert len(first.facts) == 300
        assert [fact.fact_id for fact in first.facts] == list(range(1, 301))

    def test_a_smaller_corpus_is_a_prefix_of_a_larger_one(self) -> None:
        small, large = build_scale_corpus(300), build_scale_corpus(2000)
        assert large.facts[:299] == small.facts[:299]

    def test_labels_point_at_existing_facts(self) -> None:
        corpus = build_scale_corpus(500)
        fact_ids = {fact.fact_id for fact in corpus.facts}
        for case in corpus.cases:
            if case.kind == "positive":
                assert case.relevant and case.relevant <= fact_ids
            else:
                assert not case.relevant

    def test_it_spans_all_nine_languages_and_the_hard_negatives(self) -> None:
        corpus = build_scale_corpus(2000)
        assert set(corpus.languages) == {"de", "en", "es", "fr", "pt", "tr", "pl", "ja", "ko"}
        registers = {case.register for case in corpus.cases if case.kind == "negative"}
        assert {"greeting", "off-topic", "sound-alike", "held-out"} <= registers

    @pytest.mark.parametrize("size", [7, 100_000])
    def test_impossible_sizes_are_refused(self, size: int) -> None:
        with pytest.raises(ValueError):
            build_scale_corpus(size)


@pytest.fixture(scope="session")
async def scale_outcomes(
    embedding_model: TextEmbedding,
) -> tuple[ScaleCorpus, list[QuestionOutcome]]:
    corpus = build_scale_corpus(500)
    connection = await aiosqlite.connect(":memory:")
    try:
        await init_schema(connection)
        id_map = await seed_invented_guild(
            connection, embedding_model, corpus.facts, guild_id=SCALE_GUILD
        )
        outcomes = await run_questions(
            connection,
            embedding_model,
            corpus.cases,
            guild_id=SCALE_GUILD,
            config=HybridRetrievalConfig.from_settings(
                _settings(), fact_limit=SYNTHESIS_FACT_LIMIT
            ),
            cache=LexicalIndexCache(),
            fact_id_map=id_map,
        )
    finally:
        await connection.close()
    return corpus, outcomes


class TestHybridRetrievalAtScale:
    """500 invented facts, 200 positive and 75 negative questions (measured: 197/200, 59/75)."""

    def test_it_finds_far_more_and_loses_nothing_in_precision(
        self, scale_outcomes: tuple[ScaleCorpus, list[QuestionOutcome]]
    ) -> None:
        corpus, outcomes = scale_outcomes
        cases = corpus.cases
        before = compute_metrics(cases, {o.case_id: o.baseline for o in outcomes})
        after = compute_metrics(cases, {o.case_id: o.hybrid for o in outcomes})
        assert after.hits >= 195 and after.hits >= before.hits + 25
        assert after.full >= 188
        assert after.precision >= before.precision
        assert after.negative_false_hits <= before.negative_false_hits + 1

    def test_no_question_the_previous_selection_answered_is_lost(
        self, scale_outcomes: tuple[ScaleCorpus, list[QuestionOutcome]]
    ) -> None:
        corpus, outcomes = scale_outcomes
        relevant = {case.case_id: case.relevant for case in corpus.cases}
        lost = [
            o.case_id
            for o in outcomes
            if set(o.baseline) & relevant[o.case_id] and not set(o.hybrid) & relevant[o.case_id]
        ]
        assert lost == []

    def test_every_fact_asked_by_its_own_text_comes_first(
        self, scale_outcomes: tuple[ScaleCorpus, list[QuestionOutcome]]
    ) -> None:
        # Exercised on the stored scores of the 500-fact guild: a question that
        # is a fact's full text scores that fact 1.0 on similarity and on words.
        corpus, _ = scale_outcomes
        facts = [(fact.fact_id, fact.content) for fact in corpus.facts]
        index = LexicalIndex.build(facts)
        for fact_id, content in facts[::7]:
            coverage = index.coverage(content, shipped_stopwords())
            assert coverage[fact_id] == pytest.approx(1.0)
            assert coverage[fact_id] == max(coverage.values())
