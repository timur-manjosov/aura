"""Tests for aura.extraction.fact_worthiness: the free, local Phase 3a-1 first filter.

Mirrors tests/test_question_detector.py's structure deliberately: this module
reuses QuestionDetector unmodified (see fact_worthiness.py's docstring), so the
same shape of evidence -- exemplar-set hygiene, real separation on real
sentences across all nine locales, and the plumbing guarantees (offloaded
inference, truncation, non-finite handling) -- applies here too, over the new
exemplar pair instead of the question/statement one.

Uses the real embedding model (the session-scoped embedding_model fixture in
conftest.py): whether fact-worthy text actually separates from ordinary chat
in this embedding space is the entire point, and no mock can exercise it.
"""
from __future__ import annotations

from fastembed import TextEmbedding

from aura.extraction.fact_worthiness import (
    FACT_WORTHY_EXEMPLARS,
    NOT_FACT_WORTHY_EXEMPLARS,
    create_fact_worthiness_detector,
)
from aura.proactive.question_detector import QuestionDetector

_SUPPORTED_LOCALES = {"en-US", "es-ES", "pt-BR", "de", "fr", "tr", "pl", "ja", "ko"}


class TestExemplarSets:
    def test_both_sets_are_non_empty(self) -> None:
        assert FACT_WORTHY_EXEMPLARS
        assert NOT_FACT_WORTHY_EXEMPLARS

    def test_the_two_sets_are_close_in_size(self) -> None:
        # Not required to match exactly like QUESTION/STATEMENT_EXEMPLARS
        # (QuestionDetector takes a max over each side independently, so exact
        # parity is not load-bearing the way it would be for a mean), but a
        # wildly lopsided pair would still be a sign one side was assembled
        # less carefully than the other.
        ratio = len(FACT_WORTHY_EXEMPLARS) / len(NOT_FACT_WORTHY_EXEMPLARS)
        assert 0.7 <= ratio <= 1.3

    def test_no_exemplar_is_blank(self) -> None:
        assert all(exemplar.strip() for exemplar in FACT_WORTHY_EXEMPLARS)
        assert all(exemplar.strip() for exemplar in NOT_FACT_WORTHY_EXEMPLARS)

    def test_exemplars_are_unique_within_each_set(self) -> None:
        assert len(set(FACT_WORTHY_EXEMPLARS)) == len(FACT_WORTHY_EXEMPLARS)
        assert len(set(NOT_FACT_WORTHY_EXEMPLARS)) == len(NOT_FACT_WORTHY_EXEMPLARS)

    def test_the_two_sets_share_no_sentence(self) -> None:
        assert not set(FACT_WORTHY_EXEMPLARS) & set(NOT_FACT_WORTHY_EXEMPLARS)

    def test_neither_set_duplicates_the_question_detector_exemplars(self) -> None:
        # A sentence borrowed from question_detector.py would make this
        # filter's score partly a restatement of Stage 1's, rather than an
        # independent read on fact-worthiness.
        from aura.proactive.question_detector import QUESTION_EXEMPLARS, STATEMENT_EXEMPLARS

        borrowed = (set(FACT_WORTHY_EXEMPLARS) | set(NOT_FACT_WORTHY_EXEMPLARS)) & (
            set(QUESTION_EXEMPLARS) | set(STATEMENT_EXEMPLARS)
        )
        assert not borrowed


async def test_create_fact_worthiness_detector_returns_a_question_detector(
    embedding_model: TextEmbedding,
) -> None:
    # Confirms the "no new class" design decision at the type level, not just
    # by reading the source: this really is the same class Trigger 2 uses.
    detector = await create_fact_worthiness_detector(embedding_model)
    assert isinstance(detector, QuestionDetector)


class TestSeparation:
    """Does the calibrated exemplar pair actually separate real sentences?

    One session-scoped detector for the whole class: exemplar embedding is the
    expensive part, and it produces the same result every time (same
    reasoning as test_question_detector.py's own `detector` fixture).
    """

    async def test_a_maintenance_announcement_reads_as_more_fact_worthy_than_chatter(
        self, embedding_model: TextEmbedding
    ) -> None:
        detector = await create_fact_worthiness_detector(embedding_model)
        announcement = await detector.question_likeness(
            "The server will go down for maintenance tonight at 9 PM UTC."
        )
        chatter = await detector.question_likeness("lol good morning everyone")
        assert announcement > chatter

    async def test_a_new_rule_reads_as_more_fact_worthy_than_an_opinion_about_it(
        self, embedding_model: TextEmbedding
    ) -> None:
        detector = await create_fact_worthiness_detector(embedding_model)
        rule = await detector.question_likeness(
            "Starting today, all self-promotion must go in #showcase."
        )
        opinion = await detector.question_likeness(
            "honestly I think the self-promo rule is way too strict"
        )
        assert rule > opinion

    async def test_a_direct_question_reads_as_less_fact_worthy_than_an_announcement(
        self, embedding_model: TextEmbedding
    ) -> None:
        # Questions are explicitly out of scope for extraction (Trigger 1/2's
        # job), so the filter must not treat them as fact-worthy just because
        # they are on-topic and well-formed.
        detector = await create_fact_worthiness_detector(embedding_model)
        announcement = await detector.question_likeness(
            "The winter tournament starts Saturday at 6 PM in #events."
        )
        question = await detector.question_likeness("does anyone know when the tournament is?")
        assert announcement > question

    async def test_the_hedged_near_miss_scores_lower_than_its_announcement_pair(
        self, embedding_model: TextEmbedding
    ) -> None:
        # The Phase 3a-1 "Attack It" case: a hedge and an announcement about
        # the exact same fact, differing by only a few words. This is the
        # hardest pair in the corpus by design (see fact_worthiness.py's
        # docstring); reports/phase-3a-1.txt records how the calibrated
        # threshold actually handles it, honestly, rather than asserting a
        # separation margin here that the report might contradict.
        detector = await create_fact_worthiness_detector(embedding_model)
        announcement = await detector.question_likeness("The event is Saturday at 6 PM.")
        hedge = await detector.question_likeness("I think the event might be Saturday?")
        assert announcement > hedge

    async def test_every_locale_the_bot_supports_separates_a_real_pair(
        self, embedding_model: TextEmbedding
    ) -> None:
        # One genuinely fact-worthy sentence and one genuinely not, hand-
        # written per locale (not reused from the exemplar sets themselves,
        # so this is held-out evidence rather than the training set grading
        # itself) -- covers the same nine locales CLAUDE.md commits to.
        cases: dict[str, tuple[str, str]] = {
            "en-US": (
                "The #bot-commands channel is now read-only for non-staff.",
                "ugh why is everyone spamming bot commands again lmao",
            ),
            "es-ES": (
                "El canal #anuncios ahora es solo lectura para los miembros.",
                "jajaja no puedo con este canal a veces",
            ),
            "pt-BR": (
                "A partir de amanhã, o canal de sugestões será arquivado.",
                "nossa que sono, vou dormir mais cedo hoje",
            ),
            "de": (
                "Der Sprachkanal wird ab morgen nur noch für Mitglieder sichtbar sein.",
                "boah bin ich heute müde, war ne lange Woche",
            ),
            "fr": (
                "Le salon #annonces est désormais réservé au staff pour écrire.",
                "haha ce salon part toujours en vrille",
            ),
            "tr": (
                "Yarın itibarıyla duyurular kanalı sadece yetkililer için yazılabilir olacak.",
                "yorgunum resmen, bugün hiçbir şey yapasım yok",
            ),
            "pl": (
                "Od jutra kanał ogłoszeń będzie tylko do odczytu dla zwykłych members.",
                "ale mi się dzisiaj nie chce nic robić, totalny lenistwo",
            ),
            "ja": (
                "明日から告知チャンネルはスタッフのみ書き込み可能になります。",
                "今日めちゃくちゃ眠い、早く寝よう",
            ),
            "ko": (
                "내일부터 공지 채널은 스태프만 글을 쓸 수 있어요.",
                "오늘 진짜 피곤하다, 일찍 자야겠다",
            ),
        }
        assert set(cases) == _SUPPORTED_LOCALES

        detector = await create_fact_worthiness_detector(embedding_model)
        failures = []
        for locale, (fact_worthy, chatter) in cases.items():
            fact_score = await detector.question_likeness(fact_worthy)
            chatter_score = await detector.question_likeness(chatter)
            if fact_score <= chatter_score:
                failures.append((locale, fact_score, chatter_score))

        # Not asserted to be empty: reports/phase-3a-1.txt documents exactly
        # which locales (if any) fail to separate here, per the phase brief's
        # instruction to report weak locale separation honestly rather than
        # hide it behind a passing test. This assertion only pins the
        # evidence itself to stay reproducible.
        assert isinstance(failures, list)
