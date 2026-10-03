"""Lexical coverage: how much of what a question is about appears, word for word, in a fact.

The second signal /aura-ask retrieves with, next to embedding similarity. The
embedding model scores a single keyword, an inflected form or a German compound
near its noise floor (about 0.20 against ANY fact), so "Mentoriate" or
"Matheklausuren" never reached a fact that plainly contains the word. Comparing
the words themselves closes that gap for free, locally, in milliseconds (see
aura.retrieval.hybrid for how the two signals combine).

The score is IDF-weighted fuzzy coverage: the share of a question's content
words, each weighted by how rare it is among the guild's active facts, that one
fact contains. Two words count as the same word by character n-gram
containment, which matches inflection and compounds ("Mentoriate" and
"Mentoriat", "Klausuren" inside "Matheklausuren") without a stemmer per
language, or by a single typing error in a long word.

Invariants
----------
* Pure. Imports the standard library and numpy only: no Discord, no database,
  no model, no global state. The stopword set is a parameter, never read here.
* Total. Never raises on any string -- empty, whitespace-only, emoji-only,
  invisible-only, NUL bytes, any script. Such input simply has no content
  tokens and covers nothing.
* Bounded. A question contributes at most `MAX_QUERY_TOKENS` tokens of at most
  `MAX_TOKEN_LENGTH` characters, so a pasted blob costs no more than a short
  question with many words.
* Every coverage is in [0, 1].
* `LexicalIndex.coverage` returns what `word_match` applied fact by fact
  would give, up to floating-point rounding in the last bit; the index only
  skips word pairs that cannot match.
"""

from __future__ import annotations

import math
import re
import sys
import unicodedata
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Final

import numpy as np

# A token shorter than this carries too little to match on safely ("ab", "ip").
MIN_TOKEN_LENGTH: Final = 3

# Digits may be shorter, because a day of the month or an hour ("14") is a real
# anchor in a question about a schedule.
MIN_DIGIT_TOKEN_LENGTH: Final = 2

# Containment at or above this makes two words the same word. The diagnosis
# swept 0.6 / 0.7 / 0.8: 0.6 admitted two sound-alike false hits (4-of-6 trigram
# overlaps such as "Mentos"), 0.8 changed nothing it measured.
WORD_MATCH_THRESHOLD: Final = 0.7

# A single typing error (one insertion, deletion, substitution, or swap of two
# neighbours) is forgiven only when both words have at least this many
# characters. Shorter real words one edit apart are common ("Regel"/"Kegel",
# "Turner"/"Turnier"), and the embedding cannot veto such a pair reliably.
MIN_TYPO_LENGTH: Final = 7
TYPO_MATCH_SCORE: Final = 0.9

# The reverse direction -- a fact's word found inside a longer word of the
# question, as in a compound the asker wrote ("Matheklausuren" contains
# "Klausuren") -- only for fact words long enough to be specific. "Mathe" (5)
# is the shortest it was measured to need.
MIN_REVERSE_CONTAINMENT_LENGTH: Final = 5

# How many content tokens of one question are scored. A genuine question has a
# handful; the cap is what bounds the work an adversarial one can cause.
MAX_QUERY_TOKENS: Final = 32

# A "word" longer than this is a pasted blob, not a word. Found by the
# diagnosis' attack pass: one fact word repeated a thousand times contained
# every n-gram of that word, matched it with coverage 1.0, and took 66 ms.
MAX_TOKEN_LENGTH: Final = 40

# Scripts written without spaces between words. A run of them is one token, so
# it is compared by character bigrams instead of padded trigrams. Hangul is
# written with spaces but its syllables are dense enough that bigrams match
# inflected forms better than trigrams over two- and three-syllable words.
_SPACELESS_CLASS: Final = "぀-ヿ㐀-䶿一-鿿가-힯"

# Two alternatives, so a run that switches between a spaceless script and any
# other word character ("毎週月曜日14時", "rulesチャンネル") splits at the switch:
# the Latin or digit part stays comparable with Latin and digit words.
_TOKEN_PATTERN: Final = re.compile(f"[{_SPACELESS_CLASS}]+|[^\\W{_SPACELESS_CLASS}]+")
_SPACELESS_CHARACTER: Final = re.compile(f"[{_SPACELESS_CLASS}]")
_NON_ASCII_RUN: Final = re.compile(r"[^\x00-\x7f]+")

# What one Python int object costs, for the memory estimate: the values of the
# n-gram dictionary and the fact IDs are ints too large to be shared singletons.
_INT_OBJECT_BYTES: Final = sys.getsizeof(2**40)

# Latin letters with a stroke or a ligature have no canonical decomposition, so
# removing combining marks cannot fold them. Folded explicitly, for the same
# reason umlauts are: people type without them ("nasil" for "nasıl", "lodz" for
# "łódź", "oeuvre" for "œuvre").
_UNDECOMPOSABLE_LATIN: Final[Mapping[int, str]] = str.maketrans(
    {"ı": "i", "ł": "l", "đ": "d", "ø": "o", "ħ": "h", "ŧ": "t", "œ": "oe", "æ": "ae"}
)


def normalize(text: str) -> str:
    """Return text in the one form every comparison here uses.

    Parameters
    ----------
    text
        Any string.

    Returns
    -------
    str
        `text` with invisible format characters (category Cf: zero-width
        space, joiners, soft hyphen, BOM) removed, NFKC-normalized (full-width
        Latin becomes ASCII), case-folded (Greek final sigma becomes sigma,
        "ß" becomes "ss"), and with diacritics removed from Latin letters only
        ("Ü" -> "u", "İ" -> "i", "ł" -> "l").

    Notes
    -----
    Removing Cf characters keeps "Mentor<ZWSP>iat" one word; without it the
    zero-width space split the word and the match dropped from 0.89 to 0.26
    (the diagnosis' attack pass).

    Marks are removed only after a Latin letter. Elsewhere a mark can carry
    meaning, and after NFC the common ones are part of their letter anyway.

    No ASCII character is a format character or a combining mark, so both
    character-by-character steps look only at runs of non-ASCII characters;
    German text with a few umlauts is then nearly as cheap as plain ASCII.
    """
    if text.isascii():
        # Nothing below could change an ASCII string except its case.
        return text.casefold()
    visible = _NON_ASCII_RUN.sub(_without_format_characters, text)
    folded = unicodedata.normalize("NFKC", visible).casefold().translate(_UNDECOMPOSABLE_LATIN)
    decomposed = unicodedata.normalize("NFD", folded)
    pieces: list[str] = []
    copied_up_to = 0
    for run in _NON_ASCII_RUN.finditer(decomposed):
        start, end = run.span()
        pieces.append(decomposed[copied_up_to:start])
        # The character a mark would belong to: the last one kept. Before the
        # run that is an ASCII character, and ASCII is never removed.
        previous = decomposed[start - 1] if start else ""
        for character in run.group():
            if unicodedata.category(character) == "Mn" and previous and _is_latin(previous):
                continue
            pieces.append(character)
            previous = character
        copied_up_to = end
    pieces.append(decomposed[copied_up_to:])
    return unicodedata.normalize("NFC", "".join(pieces))


def _without_format_characters(run: re.Match[str]) -> str:
    """Return one matched run of characters without its category-Cf characters."""
    return "".join(
        character for character in run.group() if unicodedata.category(character) != "Cf"
    )


def _is_latin(character: str) -> bool:
    """Report whether a character is a Latin letter."""
    return unicodedata.name(character, "").startswith("LATIN")


def is_spaceless_script(token: str) -> bool:
    """Report whether a token is written in a script without spaces between words.

    Parameters
    ----------
    token
        One token, as `tokenize` returns it.

    Returns
    -------
    bool
        True for a run of Kana, CJK ideographs or Hangul syllables.
    """
    return _SPACELESS_CHARACTER.search(token) is not None


def tokenize(text: str) -> list[str]:
    """Split text into normalized word tokens.

    Parameters
    ----------
    text
        Any string.

    Returns
    -------
    list[str]
        Every run of word characters of one kind (spaceless script, or any
        other word character), in order, normalized. Possibly empty.
    """
    return _TOKEN_PATTERN.findall(normalize(text))


def query_tokens(text: str, stopwords: frozenset[str]) -> list[str]:
    """Return the tokens of a question that can carry its subject.

    Parameters
    ----------
    text
        The question.
    stopwords
        Normalized tokens that never carry a subject.

    Returns
    -------
    list[str]
        Distinct tokens, in first-seen order, without stopwords, without
        tokens longer than `MAX_TOKEN_LENGTH` or too short to match on
        (`MIN_TOKEN_LENGTH`, or `MIN_DIGIT_TOKEN_LENGTH` for digits; a token of
        a spaceless script may be a single character), at most
        `MAX_QUERY_TOKENS` of them.
    """
    kept: dict[str, None] = {}
    for token in tokenize(text):
        if token in kept or token in stopwords or len(token) > MAX_TOKEN_LENGTH:
            continue
        if is_spaceless_script(token):
            minimum = 1
        elif token.isdigit():
            minimum = MIN_DIGIT_TOKEN_LENGTH
        else:
            minimum = MIN_TOKEN_LENGTH
        if len(token) < minimum:
            continue
        kept[token] = None
        if len(kept) == MAX_QUERY_TOKENS:
            break
    return list(kept)


def char_ngrams(token: str) -> frozenset[str]:
    """Return a token's character n-grams.

    Parameters
    ----------
    token
        One normalized token.

    Returns
    -------
    frozenset[str]
        Trigrams of the token padded with "#" on both sides, so the first and
        last letters carry weight; for a spaceless-script token, its unpadded
        bigrams, or the token itself if it is one character long. Empty only
        for an empty token.
    """
    if is_spaceless_script(token):
        if len(token) < 2:
            return frozenset({token}) if token else frozenset()
        return frozenset(token[index : index + 2] for index in range(len(token) - 1))
    padded = f"#{token}#"
    return frozenset(padded[index : index + 3] for index in range(len(padded) - 2))


def within_one_edit(first: str, second: str) -> bool:
    """Report whether two strings differ by at most one typing error.

    Parameters
    ----------
    first, second
        Two strings.

    Returns
    -------
    bool
        True when they are equal, or differ by one inserted or deleted
        character, one substituted character, or one swap of two neighbouring
        characters.
    """
    if first == second:
        return True
    if abs(len(first) - len(second)) > 1:
        return False
    if len(first) == len(second):
        differences = [
            index for index, (a, b) in enumerate(zip(first, second, strict=True)) if a != b
        ]
        if len(differences) == 1:
            return True
        return (
            len(differences) == 2
            and differences[1] == differences[0] + 1
            and first[differences[0]] == second[differences[1]]
            and first[differences[1]] == second[differences[0]]
        )
    shorter, longer = (first, second) if len(first) < len(second) else (second, first)
    return any(longer[:index] + longer[index + 1 :] == shorter for index in range(len(longer)))


def _typo_applies(query_token: str, fact_token: str) -> bool:
    """Report whether the single-edit rule may match these two tokens at all."""
    return (
        min(len(query_token), len(fact_token)) >= MIN_TYPO_LENGTH
        and not is_spaceless_script(query_token)
        and not is_spaceless_script(fact_token)
    )


def word_match(query_token: str, fact_token: str) -> float:
    """Score how strongly one question word refers to one fact word.

    Parameters
    ----------
    query_token, fact_token
        Two normalized tokens.

    Returns
    -------
    float
        In [0, 1]: 1.0 for identical tokens; otherwise the larger of the share
        of the question word's n-grams found in the fact word and -- for a fact
        word of at least `MIN_REVERSE_CONTAINMENT_LENGTH` characters -- the
        share of the fact word's n-grams found in the question word; raised to
        `TYPO_MATCH_SCORE` when both words have at least `MIN_TYPO_LENGTH`
        characters, neither is in a spaceless script, and they are one typing
        error apart.

    Notes
    -----
    The reference definition. `LexicalIndex` computes the same number for many
    fact words at once and is tested against this function.
    """
    if query_token == fact_token:
        return 1.0
    query_grams = char_ngrams(query_token)
    fact_grams = char_ngrams(fact_token)
    shared = len(query_grams & fact_grams)
    score = shared / len(query_grams) if query_grams else 0.0
    if len(fact_token) >= MIN_REVERSE_CONTAINMENT_LENGTH and fact_grams:
        score = max(score, shared / len(fact_grams))
    if (
        score < TYPO_MATCH_SCORE
        and _typo_applies(query_token, fact_token)
        and within_one_edit(query_token, fact_token)
    ):
        score = TYPO_MATCH_SCORE
    return score


def inverse_document_frequency(matching_facts: int, total_facts: int) -> float:
    """Weight a question word by how few of the guild's facts contain it.

    Parameters
    ----------
    matching_facts
        How many facts contain the word (at or above `WORD_MATCH_THRESHOLD`).
    total_facts
        How many active facts the guild has.

    Returns
    -------
    float
        The BM25 form, log(1 + (N - n + 0.5) / (n + 0.5)): always positive, so
        a sum of weights is never zero, and highest for a word no fact
        contains.

    Notes
    -----
    A word no fact contains still counts, at full weight, in the denominator of
    a coverage: "Owner vom Server" covers a fact about the server only partly,
    because nothing recorded mentions an owner.
    """
    return math.log(1 + (total_facts - matching_facts + 0.5) / (matching_facts + 0.5))


@dataclass(frozen=True, slots=True, eq=False)
class LexicalIndex:
    """One guild's active facts, prepared once for scoring any number of questions.

    Built with `LexicalIndex.build`; immutable afterwards, so one instance may
    serve concurrent questions from several threads.

    Attributes
    ----------
    fact_ids
        The indexed facts, ascending. `coverage` is keyed by these.
    estimated_bytes
        An estimate of the memory the index holds, what
        aura.retrieval.index_cache bounds itself by. Measured against
        tracemalloc in the tests.

    Notes
    -----
    Every distinct word of every fact is stored once, and every distinct
    n-gram once, with an inverted index from n-gram to the words containing it
    (two flat arrays, not one array per n-gram). A question word is then
    compared only with the words that share at least one n-gram with it: any
    other pair scores 0 by containment, and no single typing error between two
    words of `MIN_TYPO_LENGTH` or more characters removes every shared trigram
    (an edit changes at most four of a word's at least seven padded trigrams).
    The shared-n-gram counts come from one `numpy.bincount`, so a guild with
    thousands of facts costs well under a millisecond per question instead of
    the ~77 ms the per-pair prototype took at 500.
    """

    fact_ids: tuple[int, ...]
    estimated_bytes: int
    _words: tuple[str, ...]
    _gram_counts: np.ndarray
    _word_lengths: np.ndarray
    _typo_eligible: np.ndarray
    _gram_ids: Mapping[str, int]
    _posting_offsets: np.ndarray
    _posting_words: np.ndarray
    _fact_offsets: np.ndarray
    _fact_positions: np.ndarray

    @classmethod
    def build(cls, facts: Iterable[tuple[int, str]]) -> LexicalIndex:
        """Tokenize every fact once and build the inverted index.

        Parameters
        ----------
        facts
            (fact ID, content) pairs. A fact ID given twice keeps its last
            content.

        Returns
        -------
        LexicalIndex
            The index, with facts ordered by ascending ID.
        """
        contents = dict(facts)
        fact_ids = tuple(sorted(contents))
        word_ids: dict[str, int] = {}
        facts_of_word: list[list[int]] = []
        for position, fact_id in enumerate(fact_ids):
            for word in dict.fromkeys(tokenize(contents[fact_id])):
                word_id = word_ids.setdefault(word, len(word_ids))
                if word_id == len(facts_of_word):
                    facts_of_word.append([])
                facts_of_word[word_id].append(position)

        words = tuple(word_ids)
        gram_counts = np.empty(len(words), dtype=np.int32)
        gram_ids: dict[str, int] = {}
        pair_grams: list[int] = []
        pair_words: list[int] = []
        for word_id, word in enumerate(words):
            grams = char_ngrams(word)
            gram_counts[word_id] = len(grams)
            pair_grams.extend(gram_ids.setdefault(gram, len(gram_ids)) for gram in grams)
            pair_words.extend([word_id] * len(grams))
        gram_of_pair = np.asarray(pair_grams, dtype=np.int32)
        posting_words = np.asarray(pair_words, dtype=np.int32)[
            np.argsort(gram_of_pair, kind="stable")
        ]
        posting_offsets = np.zeros(len(gram_ids) + 1, dtype=np.int64)
        np.cumsum(np.bincount(gram_of_pair, minlength=len(gram_ids)), out=posting_offsets[1:])

        word_lengths = np.fromiter((len(word) for word in words), dtype=np.int32, count=len(words))
        typo_eligible = np.fromiter(
            (len(word) >= MIN_TYPO_LENGTH and not is_spaceless_script(word) for word in words),
            dtype=bool,
            count=len(words),
        )
        fact_offsets = np.zeros(len(words) + 1, dtype=np.int64)
        np.cumsum([len(positions) for positions in facts_of_word], out=fact_offsets[1:])
        fact_positions = np.fromiter(
            (position for positions in facts_of_word for position in positions),
            dtype=np.int32,
            count=int(fact_offsets[-1]),
        )
        arrays = (
            gram_counts,
            word_lengths,
            typo_eligible,
            posting_offsets,
            posting_words,
            fact_offsets,
            fact_positions,
        )
        return cls(
            fact_ids=fact_ids,
            estimated_bytes=(
                sys.getsizeof(fact_ids)
                + _INT_OBJECT_BYTES * len(fact_ids)
                + sys.getsizeof(words)
                + sum(sys.getsizeof(word) for word in words)
                + sys.getsizeof(gram_ids)
                + sum(sys.getsizeof(gram) + _INT_OBJECT_BYTES for gram in gram_ids)
                + sum(sys.getsizeof(array) for array in arrays)
            ),
            _words=words,
            _gram_counts=gram_counts,
            _word_lengths=word_lengths,
            _typo_eligible=typo_eligible,
            _gram_ids=gram_ids,
            _posting_offsets=posting_offsets,
            _posting_words=posting_words,
            _fact_offsets=fact_offsets,
            _fact_positions=fact_positions,
        )

    def coverage(self, question: str, stopwords: frozenset[str]) -> dict[int, float]:
        """Score every indexed fact by the share of the question's subject it contains.

        Parameters
        ----------
        question
            The question, as asked.
        stopwords
            Normalized tokens to ignore (see aura.retrieval.stopwords).

        Returns
        -------
        dict[int, float]
            Coverage in [0, 1] for every indexed fact ID. All zero when the
            question has no content token or nothing is indexed.

        Notes
        -----
        coverage(fact) = sum over question words w of idf(w) * match(w, fact),
        divided by the sum of idf(w), where match is the best `word_match`
        between w and any word of the fact, counted only at or above
        `WORD_MATCH_THRESHOLD`.
        """
        fact_count = len(self.fact_ids)
        tokens = query_tokens(question, stopwords)
        if not tokens or fact_count == 0:
            return dict.fromkeys(self.fact_ids, 0.0)

        weighted = np.zeros(fact_count, dtype=np.float64)
        weight_sum = 0.0
        for token in tokens:
            best = self._best_match_per_fact(token)
            weight = inverse_document_frequency(int(np.count_nonzero(best)), fact_count)
            weighted += weight * best
            weight_sum += weight
        return dict(zip(self.fact_ids, (weighted / weight_sum).tolist(), strict=True))

    def _best_match_per_fact(self, token: str) -> np.ndarray:
        """Return each fact's best `word_match` for one question word, zero below the threshold."""
        best = np.zeros(len(self.fact_ids), dtype=np.float64)
        grams = char_ngrams(token)
        postings = [
            self._posting_words[self._posting_offsets[gram_id] : self._posting_offsets[gram_id + 1]]
            for gram_id in (self._gram_ids.get(gram) for gram in grams)
            if gram_id is not None
        ]
        if not postings:
            return best
        shared_per_word = np.bincount(np.concatenate(postings), minlength=len(self._words))
        candidates = np.flatnonzero(shared_per_word)
        shared = shared_per_word[candidates]

        scores = shared / len(grams)
        reverse = self._word_lengths[candidates] >= MIN_REVERSE_CONTAINMENT_LENGTH
        scores[reverse] = np.maximum(
            scores[reverse], shared[reverse] / self._gram_counts[candidates[reverse]]
        )
        if len(token) >= MIN_TYPO_LENGTH and not is_spaceless_script(token):
            maybe_typo = np.flatnonzero(
                (scores < TYPO_MATCH_SCORE)
                & self._typo_eligible[candidates]
                & (np.abs(self._word_lengths[candidates] - len(token)) <= 1)
            )
            for index in maybe_typo.tolist():
                if within_one_edit(token, self._words[candidates[index]]):
                    scores[index] = TYPO_MATCH_SCORE

        matched = scores >= WORD_MATCH_THRESHOLD
        matched_words = candidates[matched]
        if matched_words.size == 0:
            return best
        starts = self._fact_offsets[matched_words]
        counts = self._fact_offsets[matched_words + 1] - starts
        positions = self._fact_positions[
            np.repeat(starts - np.cumsum(counts) + counts, counts) + np.arange(int(counts.sum()))
        ]
        np.maximum.at(best, positions, np.repeat(scores[matched], counts))
        return best
