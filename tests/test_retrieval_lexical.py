"""aura.retrieval.lexical: every rule of the word-matching scorer, its boundaries, and attacks on it."""

from __future__ import annotations

import math
import random
import time
import unicodedata
from typing import ClassVar

import pytest

from aura.retrieval.lexical import (
    MAX_QUERY_TOKENS,
    MAX_TOKEN_LENGTH,
    MIN_TYPO_LENGTH,
    TYPO_MATCH_SCORE,
    WORD_MATCH_THRESHOLD,
    LexicalIndex,
    char_ngrams,
    inverse_document_frequency,
    is_spaceless_script,
    normalize,
    query_tokens,
    tokenize,
    within_one_edit,
    word_match,
)
from aura.retrieval.stopwords import shipped_stopwords

STOPWORDS = shipped_stopwords()
NO_STOPWORDS: frozenset[str] = frozenset()


def coverage_of(facts: dict[int, str], question: str, stopwords: frozenset[str] = STOPWORDS):
    return LexicalIndex.build(facts.items()).coverage(question, stopwords)


def reference_normalize(text: str) -> str:
    """The diagnosis prototype's normalization, character by character, plus the stroke letters."""
    visible = "".join(ch for ch in text if unicodedata.category(ch) != "Cf")
    folded = unicodedata.normalize("NFKC", visible).casefold()
    folded = folded.translate(
        str.maketrans(
            {"ı": "i", "ł": "l", "đ": "d", "ø": "o", "ħ": "h", "ŧ": "t", "œ": "oe", "æ": "ae"}
        )
    )
    kept: list[str] = []
    for character in unicodedata.normalize("NFD", folded):
        if (
            unicodedata.category(character) == "Mn"
            and kept
            and unicodedata.name(kept[-1], "").startswith("LATIN")
        ):
            continue
        kept.append(character)
    return unicodedata.normalize("NFC", "".join(kept))


def reference_coverage(
    facts: dict[int, str], question: str, stopwords: frozenset[str]
) -> dict[int, float]:
    """Coverage computed pair by pair from word_match -- the definition the index must equal."""
    tokens = query_tokens(question, stopwords)
    fact_words = {fact_id: set(tokenize(content)) for fact_id, content in facts.items()}
    if not tokens or not facts:
        return dict.fromkeys(facts, 0.0)
    best = {
        token: {
            fact_id: max((word_match(token, word) for word in words), default=0.0)
            for fact_id, words in fact_words.items()
        }
        for token in tokens
    }
    weights = {
        token: inverse_document_frequency(
            sum(1 for score in best[token].values() if score >= WORD_MATCH_THRESHOLD), len(facts)
        )
        for token in tokens
    }
    total = sum(weights.values())
    return {
        fact_id: sum(
            weights[token] * best[token][fact_id]
            for token in tokens
            if best[token][fact_id] >= WORD_MATCH_THRESHOLD
        )
        / total
        for fact_id in facts
    }


class TestNormalize:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("Mentoriate", "mentoriate"),
            ("MATHEKLAUSUREN", "matheklausuren"),
            ("Prüfung", "prufung"),
            ("Straße", "strasse"),
            ("Ärger ÖL Übung", "arger ol ubung"),
            ("İstanbul", "istanbul"),
            ("ISTANBUL", "istanbul"),
            ("ıstanbul", "istanbul"),
            ("NASIL", "nasil"),
            ("nasıl", "nasil"),
            ("Łódź", "lodz"),
            ("Œuvre", "oeuvre"),
            ("ΟΔΟΣ", "οδοσ"),
            ("οδός", "οδόσ"),
            ("ＭＥＮＴＯＲＩＡＴ", "mentoriat"),
            ("Mentoriát", "mentoriat"),
            ("Mentor​iat", "mentoriat"),
            ("Mentor­iat", "mentoriat"),
            ("Men‍⁠tor﻿iat", "mentoriat"),
            ("が", "が"),
            ("が", "が"),
            ("مرحبا", "مرحبا"),
            ("", ""),
        ],
    )
    def test_known_forms(self, raw: str, expected: str) -> None:
        assert normalize(raw) == expected

    def test_a_mark_after_a_non_latin_letter_is_kept(self) -> None:
        # A Hebrew point changes the letter; only Latin diacritics are folded.
        assert normalize("בּ") == unicodedata.normalize("NFC", "בּ")

    def test_it_equals_the_character_by_character_reference_on_random_text(self) -> None:
        rng = random.Random(20261002)
        pool = [chr(code) for code in range(32, 0x250)] + [
            chr(code)
            for code in (
                *range(0x300, 0x370),
                0x200B,
                0x200D,
                0xFEFF,
                0xAD,
                0x2060,
                0x130,
                0x131,
                0x3C2,
                0x3099,
                0x304B,
                0xAC00,
                0x4E00,
                0x627,
                0x64E,
                0xFF21,
                0x1DC0,
                0x20D0,
                0x5BC,
            )
        ]
        for _ in range(20_000):
            text = "".join(rng.choice(pool) for _ in range(rng.randint(0, 14)))
            assert normalize(text) == reference_normalize(text), repr(text)


class TestTokenize:
    def test_words_split_on_anything_that_is_not_a_word_character(self) -> None:
        assert tokenize("Wann finden Mentoriate statt?!") == [
            "wann",
            "finden",
            "mentoriate",
            "statt",
        ]

    def test_a_hyphenated_compound_is_two_tokens(self) -> None:
        assert tokenize("Mathe-Klausuren") == ["mathe", "klausuren"]

    def test_a_run_switching_scripts_splits_at_the_switch(self) -> None:
        assert tokenize("毎週月曜日14時") == ["毎週月曜日", "14", "時"]
        assert tokenize("rulesチャンネル") == ["rules", "チャンネル"]

    @pytest.mark.parametrize("text", ["", "   \n\t ", "🎉🎉🎉", "​​", "?!.,;:", "\x00\x00"])
    def test_input_without_words_has_no_tokens(self, text: str) -> None:
        assert tokenize(text) == []

    def test_a_nul_byte_separates_words(self) -> None:
        assert tokenize("Mentor\x00iat") == ["mentor", "iat"]


class TestQueryTokens:
    def test_stopwords_are_removed_and_order_is_kept(self) -> None:
        assert query_tokens("Was ist mit den Mentoriaten?", STOPWORDS) == ["mentoriaten"]
        assert query_tokens("Wann sind die Mathe-Klausuren?", STOPWORDS) == ["mathe", "klausuren"]

    def test_duplicates_count_once(self) -> None:
        assert query_tokens("Wartung wartung WARTUNG", NO_STOPWORDS) == ["wartung"]

    @pytest.mark.parametrize(("token", "kept"), [("ip", False), ("abc", True), ("ab", False)])
    def test_latin_tokens_need_three_characters(self, token: str, kept: bool) -> None:
        assert (query_tokens(token, NO_STOPWORDS) == [token]) is kept

    @pytest.mark.parametrize(("token", "kept"), [("7", False), ("14", True), ("2026", True)])
    def test_digit_tokens_need_two_characters(self, token: str, kept: bool) -> None:
        assert (query_tokens(token, NO_STOPWORDS) == [token]) is kept

    def test_a_single_spaceless_character_is_a_token(self) -> None:
        assert query_tokens("夏", NO_STOPWORDS) == ["夏"]

    def test_the_limits_are_the_measured_ones(self) -> None:
        assert (MAX_TOKEN_LENGTH, MAX_QUERY_TOKENS) == (40, 32)

    def test_a_token_at_the_length_limit_is_kept_and_one_past_it_dropped(self) -> None:
        assert query_tokens("a" * 40, NO_STOPWORDS) == ["a" * 40]
        assert query_tokens("a" * 41, NO_STOPWORDS) == []

    def test_at_most_32_tokens_are_used(self) -> None:
        words = [f"wort{index:03d}" for index in range(100)]
        assert query_tokens(" ".join(words), NO_STOPWORDS) == words[:32]

    def test_dropped_tokens_do_not_count_toward_the_cap(self) -> None:
        words = [f"wort{index:03d}" for index in range(32)]
        noise = " ".join(["der", "x" * 50, "ab"] * 20)
        assert query_tokens(noise + " " + " ".join(words), STOPWORDS) == words

    @pytest.mark.parametrize("text", ["", "   ", "der die das", "🎉", "​", "a b c"])
    def test_a_question_without_content_has_no_tokens(self, text: str) -> None:
        assert query_tokens(text, STOPWORDS) == []


class TestCharNgrams:
    def test_latin_words_use_padded_trigrams(self) -> None:
        assert char_ngrams("abcd") == frozenset({"#ab", "abc", "bcd", "cd#"})

    def test_spaceless_runs_use_bigrams(self) -> None:
        assert char_ngrams("ルール") == frozenset({"ルー", "ール"})
        assert is_spaceless_script("ルール")
        assert not is_spaceless_script("rules")

    def test_a_single_spaceless_character_is_its_own_gram(self) -> None:
        assert char_ngrams("夏") == frozenset({"夏"})

    def test_the_empty_token_has_no_grams(self) -> None:
        assert char_ngrams("") == frozenset()


class TestWithinOneEdit:
    @pytest.mark.parametrize(
        ("first", "second", "expected"),
        [
            ("wartung", "wartung", True),
            ("wartung", "wartnug", True),  # neighbours swapped
            ("wartung", "wartng", True),  # one deleted
            ("wartung", "warttung", True),  # one inserted
            ("wartung", "wertung", True),  # one substituted
            ("wartung", "wratnug", False),  # two swaps
            ("wartung", "wartungen", False),  # two inserted
            ("wartung", "watrnug", False),  # not neighbours
            ("", "a", True),
            ("", "", True),
        ],
    )
    def test_edit_shapes(self, first: str, second: str, expected: bool) -> None:
        assert within_one_edit(first, second) is expected
        assert within_one_edit(second, first) is expected


class TestWordMatch:
    def test_identical_words_match_fully(self) -> None:
        assert word_match("mentoriat", "mentoriat") == 1.0

    @pytest.mark.parametrize(
        ("query", "fact"),
        [
            ("mentoriate", "mentoriat"),
            ("mentoriaten", "mentoriat"),
            ("mentoriats", "mentoriat"),
            ("klausuren", "klausur"),
            ("wartung", "serverwartung"),
            ("mathe", "mathematik"),
        ],
    )
    def test_inflected_and_contained_forms_match(self, query: str, fact: str) -> None:
        assert word_match(query, fact) >= WORD_MATCH_THRESHOLD

    def test_a_fact_word_inside_a_compound_of_the_question_matches(self) -> None:
        assert word_match("matheklausuren", "klausuren") >= WORD_MATCH_THRESHOLD

    def test_the_reverse_direction_needs_a_fact_word_of_five_characters(self) -> None:
        # "liga" (4) sits inside "ligatur", but four letters are too unspecific.
        assert word_match("ligatur", "liga") < WORD_MATCH_THRESHOLD
        assert word_match("ligatur", "ligat") >= WORD_MATCH_THRESHOLD

    @pytest.mark.parametrize(
        ("query", "fact"),
        [
            ("regenl", "regeln"),  # 6: swap
            ("regln", "regeln"),  # 5/6: deletion
            ("kegeln", "regeln"),  # 6: substitution
            ("kegel", "regel"),  # the measured real-word collision
            ("turner", "turnier"),  # 6/7: the other one
        ],
    )
    def test_no_typo_is_forgiven_below_seven_characters(self, query: str, fact: str) -> None:
        assert word_match(query, fact) < WORD_MATCH_THRESHOLD

    @pytest.mark.parametrize(
        ("query", "fact"),
        [
            ("wartnug", "wartung"),  # 7: swap
            ("wertung", "wartung"),  # 7: substitution
            ("turneire", "turniere"),  # 8: swap
            ("turnere", "turniere"),  # 7/8: deletion
            ("sprechstnde", "sprechstunde"),
        ],
    )
    def test_a_typo_from_seven_characters_on_is_forgiven(self, query: str, fact: str) -> None:
        assert word_match(query, fact) >= TYPO_MATCH_SCORE

    def test_the_typo_rule_never_applies_to_spaceless_scripts(self) -> None:
        first, second = "サーバールール", "サーバーノール"
        assert len(first) >= MIN_TYPO_LENGTH and within_one_edit(first, second)
        assert word_match(first, second) < TYPO_MATCH_SCORE

    @pytest.mark.parametrize("pair", [("mentos", "mentoriat"), ("monitor", "mentoriat")])
    def test_sound_alikes_do_not_match(self, pair: tuple[str, str]) -> None:
        assert word_match(*pair) < WORD_MATCH_THRESHOLD

    def test_one_typing_error_between_long_words_always_leaves_a_shared_trigram(self) -> None:
        # The index only compares words that share an n-gram; this is the fact
        # that makes that shortcut exact for the typo rule.
        rng = random.Random(7)
        for _ in range(3000):
            word = "".join(rng.choice("abcde") for _ in range(rng.randint(7, 12)))
            position = rng.randrange(len(word))
            edits = [
                word[:position] + word[position + 1 :],
                word[:position] + "x" + word[position:],
                word[:position] + "x" + word[position + 1 :],
            ]
            if position < len(word) - 1:
                edits.append(
                    word[:position] + word[position + 1] + word[position] + word[position + 2 :]
                )
            for edited in edits:
                if min(len(word), len(edited)) >= MIN_TYPO_LENGTH:
                    assert char_ngrams(word) & char_ngrams(edited), (word, edited)


class TestCoverage:
    FACTS: ClassVar[dict[int, str]] = {
        1: "Jeden Dienstag um 18 Uhr gibt es eine Sprechstunde für neue Mitglieder.",
        2: "Eine Sprechstunde findet jeden zweiten Samstag statt.",
        3: "Die Serverwartung ist jeden Donnerstag um 5:00 MEZ.",
        4: "Werbung für andere Server ist verboten.",
    }

    def test_a_keyword_covers_every_fact_that_contains_it(self) -> None:
        coverage = coverage_of(self.FACTS, "Sprechstunden")
        assert coverage[1] >= 0.5 and coverage[2] >= 0.5
        assert coverage[3] == 0.0 and coverage[4] == 0.0

    def test_a_word_no_fact_contains_still_counts_against_coverage(self) -> None:
        # "Owner" is in no fact, at full weight; "Server" matches one.
        coverage = coverage_of(self.FACTS, "Wer ist der Owner vom Server?")
        assert 0.0 < coverage[4] < 0.5

    def test_a_rare_word_weighs_more_than_a_common_one(self) -> None:
        facts = {index: f"Kanal {index} ist für Thema {index}." for index in range(1, 30)}
        facts[99] = "Der Kanal memes ist für Bilder."
        coverage = coverage_of(facts, "Kanal memes")
        assert coverage[99] > 0.9
        assert max(value for fact_id, value in coverage.items() if fact_id != 99) < 0.2

    @pytest.mark.parametrize("question", ["", "   ", "der die das", "🎉🎉", "​", "?"])
    def test_no_content_word_means_no_coverage(self, question: str) -> None:
        assert set(coverage_of(self.FACTS, question).values()) == {0.0}

    def test_an_empty_index_scores_nothing(self) -> None:
        assert LexicalIndex.build([]).coverage("Sprechstunde", STOPWORDS) == {}

    def test_the_order_facts_are_given_in_does_not_matter(self) -> None:
        forward = LexicalIndex.build(self.FACTS.items())
        backward = LexicalIndex.build(reversed(list(self.FACTS.items())))
        assert forward.fact_ids == backward.fact_ids == (1, 2, 3, 4)
        assert forward.coverage("Wartung Server", STOPWORDS) == backward.coverage(
            "Wartung Server", STOPWORDS
        )

    def test_a_fact_id_given_twice_keeps_its_last_content(self) -> None:
        index = LexicalIndex.build([(1, "Sprechstunde"), (1, "Wartung")])
        assert index.coverage("Wartung", STOPWORDS) == {1: 1.0}

    def test_a_question_containing_a_facts_full_text_covers_it_fully_and_best(self) -> None:
        for fact_id, content in self.FACTS.items():
            coverage = coverage_of(self.FACTS, content)
            assert coverage[fact_id] == pytest.approx(1.0)
            assert coverage[fact_id] == max(coverage.values())

    def test_the_index_equals_the_pair_by_pair_definition(self) -> None:
        rng = random.Random(1)
        alphabet = "abcdefgh"
        for _ in range(150):
            vocabulary = [
                "".join(rng.choice(alphabet) for _ in range(rng.randint(2, 11))) for _ in range(30)
            ]
            facts = {
                fact_id: " ".join(rng.choice(vocabulary) for _ in range(rng.randint(0, 8)))
                for fact_id in rng.sample(range(1, 500), rng.randint(1, 25))
            }
            question = " ".join(
                rng.choice(vocabulary)
                if rng.random() < 0.7
                else "".join(rng.choice(alphabet) for _ in range(rng.randint(1, 12)))
                for _ in range(rng.randint(0, 6))
            )
            expected = reference_coverage(facts, question, NO_STOPWORDS)
            actual = coverage_of(facts, question, NO_STOPWORDS)
            # Equal up to the last bit of rounding: the reference sums with
            # Python's compensated sum(), the index accumulates one token at a
            # time.
            assert actual.keys() == expected.keys()
            for fact_id, value in expected.items():
                assert actual[fact_id] == pytest.approx(value, rel=1e-12, abs=1e-15)


class TestScripts:
    def test_turkish_dotted_capital_i_matches_lowercase(self) -> None:
        coverage = coverage_of({1: "İstanbul turnuvası cumartesi."}, "istanbul")
        assert coverage[1] == 1.0

    def test_greek_final_sigma_matches_capitals(self) -> None:
        assert coverage_of({1: "Ο ΚΟΣΜΟΣ είναι εδώ."}, "κοσμος")[1] == 1.0

    def test_arabic_matches_itself(self) -> None:
        assert coverage_of({1: "البطولة يوم السبت"}, "البطولة")[1] == 1.0

    def test_japanese_finds_a_word_inside_a_run(self) -> None:
        facts = {1: "サーバーのルールは #rules チャンネルにあります。", 2: "大会は土曜日です。"}
        coverage = coverage_of(facts, "ルール")
        assert coverage[1] == 1.0 and coverage[2] == 0.0

    def test_full_width_latin_matches_ascii(self) -> None:
        assert coverage_of({1: "Das Mentoriat ist montags."}, "ＭＥＮＴＯＲＩＡＴ")[1] == 1.0

    def test_a_combining_accent_matches_the_plain_letter(self) -> None:
        assert coverage_of({1: "Das Mentoriat ist montags."}, "Mentoriát")[1] == 1.0

    def test_a_zero_width_space_inside_a_word_does_not_split_it(self) -> None:
        coverage = coverage_of({1: "Das Mentoriat ist montags."}, "Mentor​iate")
        assert coverage[1] >= 0.5


class TestAttacks:
    FACTS: ClassVar[dict[int, str]] = {
        1: "Die Serverwartung ist jeden Donnerstag um 5:00 MEZ.",
        2: "Das Mentoriat für Neulinge findet jeden Dienstag statt.",
    }

    def test_a_6000_character_token_is_ignored(self) -> None:
        assert set(coverage_of(self.FACTS, "x" * 6000).values()) == {0.0}

    def test_a_9000_character_blob_of_a_fact_word_matches_nothing(self) -> None:
        started = time.perf_counter()
        coverage = coverage_of(self.FACTS, "wartung" * 1300)
        assert set(coverage.values()) == {0.0}
        assert time.perf_counter() - started < 0.5

    def test_nul_bytes_and_sql_shaped_text_are_just_text(self) -> None:
        for question in ["'; DROP TABLE facts; --", "Wartung\x00\x00", "1 OR 1=1 /* */"]:
            values = coverage_of(self.FACTS, question).values()
            assert all(0.0 <= value <= 1.0 for value in values)
        assert coverage_of(self.FACTS, "Wartung\x00")[1] > 0.5

    def test_random_multiscript_input_never_raises_and_stays_in_bounds(self) -> None:
        rng = random.Random(3)
        pool = (
            [chr(code) for code in range(32, 0x2FF)]
            + list("ルールサーバー大会土曜日규칙서버عربيΟΔΟΣ🎉👍​‍﻿́\x00 \n\t")
            + ["Mentoriat", "Wartung", "Server", "wartung"]
        )
        index = LexicalIndex.build(self.FACTS.items())
        slowest = 0.0
        for _ in range(3000):
            question = "".join(rng.choice(pool) for _ in range(rng.randint(0, 60)))
            started = time.perf_counter()
            values = index.coverage(question, STOPWORDS).values()
            slowest = max(slowest, time.perf_counter() - started)
            assert all(0.0 <= value <= 1.0 and not math.isnan(value) for value in values)
        assert slowest < 0.05

    def test_adversarial_facts_index_without_raising(self) -> None:
        facts = {
            1: "",
            2: "   ",
            3: "🎉" * 500,
            4: "​" * 50,
            5: "a" * 4000,
            6: "日" * 4000,
            7: "\x00",
            8: "[link](https://example.com) `code` **bold**",
        }
        index = LexicalIndex.build(facts.items())
        values = index.coverage("aaa 日日 link", STOPWORDS).values()
        assert all(0.0 <= value <= 1.0 for value in values)
