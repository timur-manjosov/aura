"""Train/test leakage: does any generated message re-test the classifier against
a paraphrase of its own reference exemplars?

Stage 1 scores a message by comparing it to the fixed QUESTION_EXEMPLARS and
STATEMENT_EXEMPLARS sets in `aura.proactive.question_detector`. If a generated
"question" is a light rewording of one of those exemplars, it will score
beautifully -- and that score measures nothing except that the sentence is
close to a sentence the detector was built around. A corpus with that in it
would report a better threshold than the real one, which is worse than
reporting no threshold at all.

Two independent detectors, because each misses a case the other catches:

* **Semantic** -- cosine similarity against the exemplars, using the very
  embedding model Stage 1 uses. This is the one that matters, because it
  measures closeness in the space the decision is actually made in.
* **Lexical** -- token and character-n-gram overlap. Cheap insurance against
  the pathological case where the embedding model happens to place a near-copy
  further away than its own wording suggests, and against a copy dressed up
  with punctuation or casing changes the embedding would normalise away
  anyway.

Neither threshold is a tuned number: both sit well below "obviously the same
sentence" and well above "same topic, different sentence", and the checker's
job is to hand back a ranked list for a human to look at, not to be the final
word. Anything it flags is excluded from the corpus, not warned about -- see
`generate_synthetic_corpus.py`.
"""
from __future__ import annotations

import unicodedata
from dataclasses import dataclass

import numpy as np
from fastembed import TextEmbedding

from aura.embeddings import cosine_similarity, embed_texts
from aura.proactive.question_detector import QUESTION_EXEMPLARS, STATEMENT_EXEMPLARS

# Above this cosine similarity to any exemplar, a generated sentence is treated
# as a near-duplicate. Calibrated by measurement, not taste: in this embedding
# space genuinely different sentences on the same topic sit far below it, while
# a reworded copy of an exemplar sits above it. The verification test pins both
# ends -- a deliberate near-duplicate must be caught, and an ordinary on-topic
# sentence must not be.
LEAKAGE_COSINE_THRESHOLD = 0.85

# Overlap of normalised tokens (or character trigrams for scripts that do not
# space their words) above which two sentences are treated as the same sentence
# wearing different punctuation.
LEAKAGE_LEXICAL_THRESHOLD = 0.60

_CHARACTER_NGRAM_SIZE = 3


@dataclass(frozen=True)
class LeakageFinding:
    """One generated text that sits too close to a reference exemplar."""

    text: str
    exemplar: str
    cosine: float
    lexical: float
    triggered_by: str

    def describe(self) -> str:
        """One-line human-readable form for the report."""
        return (
            f"[{self.triggered_by}] cos={self.cosine:.3f} lex={self.lexical:.3f}\n"
            f"    generated: {self.text[:110]}\n"
            f"    exemplar : {self.exemplar[:110]}"
        )


def _normalise(text: str) -> str:
    """Casefold, NFKC-normalise and strip punctuation, keeping word boundaries.

    NFKC first so full-width and half-width forms of the same character (a real
    possibility in the Japanese and Korean parts of this corpus) do not read as
    different tokens.
    """
    folded = unicodedata.normalize("NFKC", text).casefold()
    return "".join(
        " " if unicodedata.category(char).startswith(("P", "S", "Z", "C")) else char
        for char in folded
    )


def _token_set(text: str) -> set[str]:
    """Whitespace tokens of `text`, or character n-grams where it has none.

    Japanese and Korean do not separate words with spaces, so a whitespace
    tokenizer would reduce a whole sentence to one token and make every
    Jaccard comparison meaningless. Falling back to character n-grams keeps the
    lexical check honest in all nine locales instead of silently working in
    six of them.
    """
    normalised = _normalise(text)
    tokens = {token for token in normalised.split() if token}
    if len(tokens) >= 2:
        return tokens

    compact = "".join(normalised.split())
    if len(compact) <= _CHARACTER_NGRAM_SIZE:
        return {compact} if compact else set()
    return {
        compact[index : index + _CHARACTER_NGRAM_SIZE]
        for index in range(len(compact) - _CHARACTER_NGRAM_SIZE + 1)
    }


def lexical_overlap(left: str, right: str) -> float:
    """Jaccard overlap of two texts after normalisation, in [0, 1].

    Returns 0.0 when either side normalises to nothing, rather than raising or
    returning 1.0 for two empty strings: an empty generated message is a
    generation bug for the corpus validator to catch, not a leak.
    """
    left_tokens = _token_set(left)
    right_tokens = _token_set(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def all_exemplars() -> tuple[str, ...]:
    """Every reference sentence Stage 1 is built around, both sides of the contrast."""
    return (*QUESTION_EXEMPLARS, *STATEMENT_EXEMPLARS)


class LeakageChecker:
    """Flags generated text that is too close to Stage 1's reference exemplars.

    Holds the exemplar embeddings so they are computed once rather than per
    candidate -- the same reasoning QuestionDetector applies to the same set.
    Build one with `create`.
    """

    def __init__(self, exemplars: tuple[str, ...], embeddings: list[np.ndarray]) -> None:
        """Bind exemplars to their already-computed embeddings. Prefer create()."""
        if len(exemplars) != len(embeddings):
            raise ValueError("exemplars and embeddings must line up one-to-one")
        self._exemplars = exemplars
        self._embeddings = embeddings

    @classmethod
    async def create(
        cls, model: TextEmbedding, exemplars: tuple[str, ...] | None = None
    ) -> LeakageChecker:
        """Embed the exemplar set once and return a ready checker.

        The `exemplars` parameter exists so the verification tests can pin a
        known set; production callers pass nothing and get the real sets Stage 1
        actually uses, read live from `question_detector` rather than copied
        here -- a copy would keep passing after someone edited the real one.
        """
        chosen = all_exemplars() if exemplars is None else exemplars
        if not chosen:
            raise ValueError("LeakageChecker needs at least one exemplar")
        return cls(chosen, await embed_texts(model, chosen))

    async def check(self, model: TextEmbedding, texts: list[str]) -> list[LeakageFinding]:
        """Return a finding for every text too close to any exemplar.

        Embeds all candidates in one batched call rather than one per text (see
        aura.embeddings.embed_texts), and returns findings sorted worst-first so
        a human reviewing the report reads the most suspicious case first.
        """
        scoreable = [text for text in texts if text.strip()]
        if not scoreable:
            return []

        embeddings = await embed_texts(model, scoreable)
        findings: list[LeakageFinding] = []

        for text, embedding in zip(scoreable, embeddings, strict=True):
            best_cosine = -1.0
            best_cosine_exemplar = ""
            best_lexical = 0.0
            best_lexical_exemplar = ""

            for exemplar, exemplar_embedding in zip(self._exemplars, self._embeddings, strict=True):
                cosine = cosine_similarity(exemplar_embedding, embedding)
                if cosine > best_cosine:
                    best_cosine, best_cosine_exemplar = cosine, exemplar
                lexical = lexical_overlap(text, exemplar)
                if lexical > best_lexical:
                    best_lexical, best_lexical_exemplar = lexical, exemplar

            semantic_hit = best_cosine >= LEAKAGE_COSINE_THRESHOLD
            lexical_hit = best_lexical >= LEAKAGE_LEXICAL_THRESHOLD
            if not (semantic_hit or lexical_hit):
                continue

            triggered_by = (
                "semantic+lexical"
                if semantic_hit and lexical_hit
                else ("semantic" if semantic_hit else "lexical")
            )
            findings.append(
                LeakageFinding(
                    text=text,
                    exemplar=best_cosine_exemplar if semantic_hit else best_lexical_exemplar,
                    cosine=best_cosine,
                    lexical=best_lexical,
                    triggered_by=triggered_by,
                )
            )

        findings.sort(key=lambda finding: max(finding.cosine, finding.lexical), reverse=True)
        return findings

    async def max_similarity(self, model: TextEmbedding, texts: list[str]) -> list[float]:
        """Return each text's highest cosine similarity to any exemplar.

        Separate from `check` so the report can show the whole distribution --
        "nothing was flagged" is far more convincing next to "and the closest
        thing in the corpus scored 0.61" than on its own.
        """
        if not texts:
            return []
        embeddings = await embed_texts(model, texts)
        return [
            max(cosine_similarity(exemplar, embedding) for exemplar in self._embeddings)
            for embedding in embeddings
        ]
