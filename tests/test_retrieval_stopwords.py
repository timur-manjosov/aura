"""aura.retrieval.stopwords: one data file per locale, applied as one set, and what IDF does instead."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import pytest

import aura
import aura.retrieval.stopwords as stopwords_module
from aura.i18n.translator import SUPPORTED_LOCALES
from aura.retrieval.lexical import LexicalIndex, tokenize
from aura.retrieval.stopwords import (
    STOPWORD_DIRECTORY,
    StopwordLoadError,
    load_stopword_lists,
    parse_stopword_file,
    shipped_stopword_lists,
    shipped_stopwords,
)

# Same spelling, a content word in another supported language: left out of
# every list on purpose (each file's header names its own).
LEFT_OUT_FOR_ANOTHER_LANGUAGE = frozenset(
    {"son", "era", "sin", "dice", "car", "pendant", "plus", "ali", "rola", "ten", "one", "pod"}
)
# A subject within its own language ("zaman" is "time" on its own).
LEFT_OUT_AS_A_SUBJECT = frozenset({"zaman"})
# Kept although another language uses the same spelling as a content word,
# because they open too many questions in their own language; documented in
# their files' headers.
KEPT_DESPITE_A_COLLISION = frozenset(
    {"die", "hat", "man", "war", "bin", "will", "may", "can", "hay", "con", "dime", "kim", "ben"}
)


class TestTheShippedLists:
    def test_there_is_one_file_per_supported_locale(self) -> None:
        assert set(shipped_stopword_lists()) == set(SUPPORTED_LOCALES)

    def test_the_files_live_inside_the_package_the_image_copies(self) -> None:
        # The Dockerfile copies src/; a list outside the package would be
        # missing from the image without anything failing at build time.
        package = Path(aura.__file__).parent
        assert STOPWORD_DIRECTORY.is_relative_to(package)
        assert all(path.is_file() for path in STOPWORD_DIRECTORY.glob("*.txt"))

    @pytest.mark.parametrize("locale", sorted(SUPPORTED_LOCALES))
    def test_every_list_has_words(self, locale: str) -> None:
        assert len(shipped_stopword_lists()[locale]) >= 20

    def test_the_union_is_every_list_together(self) -> None:
        union = frozenset().union(*shipped_stopword_lists().values())
        assert shipped_stopwords() == union

    def test_every_word_is_stored_as_a_question_would_be_normalized(self) -> None:
        for word in shipped_stopwords():
            assert tokenize(word) == [word]

    def test_typical_question_words_of_every_language_are_in_it(self) -> None:
        stopwords = shipped_stopwords()
        for question_word in ("wann", "when", "cuando", "quand", "quando", "nasil", "kiedy"):
            assert question_word in stopwords
        for question_word in ("いつ", "どこ", "언제", "어디"):
            assert question_word in stopwords

    def test_the_words_left_out_on_purpose_are_not_in_it(self) -> None:
        assert not (LEFT_OUT_FOR_ANOTHER_LANGUAGE | LEFT_OUT_AS_A_SUBJECT) & shipped_stopwords()

    def test_the_documented_kept_collisions_are_in_it(self) -> None:
        assert shipped_stopwords() >= KEPT_DESPITE_A_COLLISION

    def test_no_subject_of_the_invented_evaluation_set_is_a_stopword(self) -> None:
        subjects = {
            "mentoriat",
            "mentoriate",
            "sprechstunde",
            "wartung",
            "serverwartung",
            "klausuren",
            "mathe",
            "mathematik",
            "regeln",
            "turnier",
            "werbung",
            "spoiler",
            "server",
            "kanal",
            "minecraft",
            "bewerbung",
            "filmclub",
        }
        assert not subjects & shipped_stopwords()


class TestParsing:
    def test_comments_whitespace_and_several_words_per_line(self) -> None:
        text = "# a comment\n  wann   wo\twie  # trailing comment\n\nWER\n"
        assert parse_stopword_file(text) == frozenset({"wann", "wo", "wie", "wer"})

    def test_words_are_normalized_like_questions(self) -> None:
        assert parse_stopword_file("Qué Über NASIL Łódź") == frozenset(
            {"que", "uber", "nasil", "lodz"}
        )

    def test_a_word_that_normalization_splits_contributes_each_part(self) -> None:
        assert parse_stopword_file("aujourd'hui") == frozenset({"aujourd", "hui"})

    def test_an_empty_file_lists_nothing(self) -> None:
        assert parse_stopword_file("") == frozenset()


class TestLoadFailures:
    def test_a_missing_directory_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(StopwordLoadError):
            load_stopword_lists(tmp_path / "absent")

    def test_a_directory_without_lists_is_refused(self, tmp_path: Path) -> None:
        (tmp_path / "notes.md").write_text("nothing", encoding="utf-8")
        with pytest.raises(StopwordLoadError):
            load_stopword_lists(tmp_path)

    def test_a_file_that_is_not_utf8_is_refused(self, tmp_path: Path) -> None:
        (tmp_path / "de.txt").write_bytes(b"wann \xff\xfe wo")
        with pytest.raises(StopwordLoadError):
            load_stopword_lists(tmp_path)

    def test_every_file_is_read_and_keyed_by_its_name(self, tmp_path: Path) -> None:
        (tmp_path / "xx.txt").write_text("alpha beta", encoding="utf-8")
        (tmp_path / "yy.txt").write_text("gamma", encoding="utf-8")
        lists = load_stopword_lists(tmp_path)
        assert dict(lists) == {"xx": {"alpha", "beta"}, "yy": {"gamma"}}
        with pytest.raises(TypeError):
            lists["zz"] = frozenset()  # type: ignore[index]

    def test_a_failed_load_is_not_cached(self, monkeypatch: pytest.MonkeyPatch) -> None:
        shipped_stopword_lists.cache_clear()
        shipped_stopwords.cache_clear()

        def unreadable() -> None:
            raise StopwordLoadError("simulated")

        monkeypatch.setattr(stopwords_module, "load_stopword_lists", unreadable)
        with pytest.raises(StopwordLoadError):
            shipped_stopwords()
        monkeypatch.undo()
        assert "wann" in shipped_stopwords()


class TestIdfHandlesWhatTheListsLeaveOut:
    """'son' is "are" in Spanish and "son" in English; no list holds it, IDF decides."""

    SPANISH_GUILD: ClassVar[dict[int, str]] = {
        1: "Las reglas son claras: nada de spam en los canales.",
        2: "Los torneos son cada sábado a las 18:00.",
        3: "Los moderadores son voluntarios del servidor.",
        4: "Las inscripciones son en el canal #eventos.",
        5: "Los premios del torneo son tres meses de Nitro.",
    }

    def test_a_spanish_question_with_son_still_finds_its_fact(self) -> None:
        coverage = LexicalIndex.build(self.SPANISH_GUILD.items()).coverage(
            "¿Cuáles son las reglas?", shipped_stopwords()
        )
        assert coverage[1] == 1.0
        # "son" is in every fact, so it weighs almost nothing on its own.
        assert max(value for fact_id, value in coverage.items() if fact_id != 1) < 0.1

    def test_in_an_english_guild_son_stays_a_subject(self) -> None:
        english_guild = {
            1: "The father and son tournament is on Saturday.",
            2: "The summer tournament is on Sunday.",
        }
        coverage = LexicalIndex.build(english_guild.items()).coverage(
            "son tournament", shipped_stopwords()
        )
        assert coverage[1] == 1.0
        assert coverage[2] < 0.5
