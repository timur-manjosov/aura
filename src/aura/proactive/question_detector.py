"""How question-like is a piece of text? Semantic scoring, nothing else.

Scaffolding for CLAUDE.md's second trigger, proactive relief. This module
scores only. It never reads a fact, never calls an LLM, never touches
Discord -- it answers exactly one question, "does this text look like
someone asking for information?", and hands back a number.

Detection is semantic rather than keyword-based, deliberately. Aura supports
nine languages, so a regex over interrogative markers ("?", "how", "wie",
"どうやって", ...) would need per-locale maintenance forever and would still
miss the most common real phrasing: "not sure where the setup guide is" is
someone asking for information with no question mark and no interrogative
word anywhere in it. The multilingual embedding model Phase 1d established
(paraphrase-multilingual-MiniLM-L12-v2) places a German question and its
English translation close together in one shared vector space, which is
precisely what makes a single small mixed-language exemplar set cover all
nine locales at once -- there is no per-locale exemplar file here, and adding
a tenth language must not create one.

The score is *contrastive*: similarity to the closest question exemplar minus
similarity to the closest statement exemplar. Phase 2a-1 scored one-sidedly
against questions alone, and its own attack pass measured that mechanism at
66% accuracy on a 29-sentence held-out set -- worse than a naive "contains a
question mark" baseline (76%), while the contrastive form below reached 79%.
The diagnosis: this embedding space clusters by topic and paraphrase, not by
interrogative form, so "the rules are pinned in #welcome" lands close to
"where are the rules?" on topic alone. Subtracting the statement side cancels
that shared topical component out, which is why the statement exemplars
mirror the question exemplars topic-for-topic and language-for-language
rather than being generic filler.

The returned score is a raw ranking signal, not a calibrated probability.
This module still does not decide what counts as question-like *enough* --
that threshold lives in configuration and is applied by the gate (see
aura.proactive.gate). Same split aura.embeddings already draws: rank here,
threshold at the call site.
"""
from __future__ import annotations

import logging
import math

import numpy as np
from fastembed import TextEmbedding

from aura.embeddings import cosine_similarity, embed_text, embed_texts

logger = logging.getLogger(__name__)

# Sentences that are unambiguously someone asking for information, spread
# across several of Aura's supported locales. Mixed languages on purpose: the
# multilingual embedding space is what makes one shared set work everywhere,
# and proving that with a set that is itself multilingual keeps the property
# from quietly regressing into "English exemplars that happen to work."
#
# Several entries carry no interrogative marker at all -- no question mark, no
# question word -- because that phrasing is extremely common in real chat and
# is exactly what a keyword approach cannot see.
QUESTION_EXEMPLARS: tuple[str, ...] = (
    # English -- explicit interrogatives
    "How do I get access to this channel?",
    "Where can I find the server rules?",
    "Does anyone know when the next event is?",
    "Can someone explain how this works?",
    # English -- requests for information with no interrogative marker
    "anyone know if the meeting is still happening today",
    "not sure where the setup guide is, could use a pointer",
    "I'm looking for the instructions to get started",
    # German
    "Wie kann ich hier mitmachen?",
    "Weiß jemand, wo die Serverregeln stehen?",
    "Ich suche die Anleitung für den Einstieg.",
    # Spanish
    "¿Alguien sabe cómo funciona esto?",
    "¿Dónde puedo encontrar la información del servidor?",
    # Portuguese (Brazil)
    "Alguém pode explicar como isso funciona?",
    # French
    "Comment est-ce que ça marche exactement ?",
    # Turkish
    "Bunu nasıl yapabilirim, bilen var mı?",
    # Polish
    "Gdzie mogę znaleźć zasady serwera?",
    # Japanese
    "これはどうやって設定すればいいですか？",
    "サーバーのルールはどこで見られますか？",
    # Korean
    "이거 어떻게 하는지 아시는 분 계신가요?",
)

# The mirror of QUESTION_EXEMPLARS: the same topics, in the same languages, in
# the same order, differing only in being answers and remarks rather than
# requests for information. That one-to-one pairing is the whole mechanism --
# the contrastive score subtracts these from the questions above, so anything
# the two sets share (server vocabulary, "rules", "setup guide", the language
# a message happens to be written in) cancels out, and what survives is the
# interrogative form the raw one-sided score could not see.
#
# Generic filler ("I like pizza") would not do this job: it shares no topic
# with the question set, so subtracting it would leave the topical component
# almost untouched and reproduce the exact failure this replaces.
STATEMENT_EXEMPLARS: tuple[str, ...] = (
    # English -- plain declaratives, mirroring the four explicit interrogatives
    "You already have access to this channel.",
    "The server rules are pinned in the welcome channel.",
    "The next event starts on Friday evening.",
    "This works the same way it did before the update.",
    # English -- the informal register of the unmarked questions above
    "the meeting is still happening today as far as I know",
    "the setup guide is in the pinned messages, third link down",
    "I finished the instructions and got everything working",
    # German
    "Du kannst hier jederzeit mitmachen.",
    "Die Serverregeln hängen im Willkommenskanal.",
    "Ich habe die Anleitung für den Einstieg schon gelesen.",
    # Spanish
    "Aquí esto funciona igual que en el servidor anterior.",
    "La información del servidor está en el primer mensaje fijado.",
    # Portuguese (Brazil)
    "Isso funciona bem depois da última atualização.",
    # French
    "Ça marche très bien depuis la mise à jour.",
    # Turkish
    "Bunu dün akşam hallettim, şimdi çalışıyor.",
    # Polish
    "Zasady serwera są przypięte w kanale powitalnym.",
    # Japanese
    "これは設定パネルから変更しました。",
    "サーバーのルールは最初のメッセージに書いてあります。",
    # Korean
    "이거 어제 다 설정해 놨어요.",
)

# The model's tokenizer truncates at 128 tokens (measured against the
# installed fastembed build, not assumed): 500 characters of English already
# fill 80 of them and 1000 characters saturate the window entirely. So text
# beyond this cap cannot change any realistic message's score -- the cap
# exists to bound tokenizer work on a pasted log dump (Discord permits up to
# 4000 characters in one message), not to alter what a message scores.
_MAX_CLASSIFIED_CHARACTERS = 2000

# What to return for text that cannot be scored at all. The floor of the
# contrastive range, NOT its neutral midpoint: 0.0 means "equally question-like
# and statement-like", which on a contrastive scale is an ordinary score that
# real messages routinely beat -- and the calibrated Stage 1 threshold is
# itself slightly negative (see Settings.proactive_question_threshold), so 0.0
# would have *passed* the gate. The floor is the only value that cannot,
# guaranteed by ProactiveGateConfig requiring a threshold strictly above it.
#
# This is the one place the one-sided-to-contrastive rescoring changed the
# meaning of an existing constant rather than just its magnitude: under Phase
# 2a-1's [0, 1]-ish similarity, 0.0 genuinely was the bottom of the scale.
_NO_QUESTION_EVIDENCE = -2.0


class QuestionDetector:
    """Scores text contrastively against fixed multilingual question and statement exemplars.

    Holds both exemplar sets' embeddings so they are computed once for the
    process rather than per message. Construct it with
    QuestionDetector.create, which does that embedding for you.
    """

    def __init__(
        self,
        model: TextEmbedding,
        question_embeddings: list[np.ndarray],
        statement_embeddings: list[np.ndarray],
    ) -> None:
        """Bind two already-embedded exemplar sets. Prefer create() over calling this directly."""
        self._model = model
        self._question_embeddings = question_embeddings
        self._statement_embeddings = statement_embeddings

    @classmethod
    async def create(
        cls,
        model: TextEmbedding,
        question_exemplars: tuple[str, ...] = QUESTION_EXEMPLARS,
        statement_exemplars: tuple[str, ...] = STATEMENT_EXEMPLARS,
    ) -> QuestionDetector:
        """Embed both exemplar sets once, up front, and return a ready detector.

        Called once at startup (see AuraClient.setup_hook), never per
        message: re-embedding these fixed sentences for every incoming
        message would multiply the per-message inference cost by the size of
        both sets, for an identical result every time. Adding the statement
        side therefore costs one extra batch at startup and nothing at all
        per message.

        Both sets go into a *single* batched inference call, not one call
        each: at these batch sizes the fixed per-call cost of an ONNX
        Runtime invocation dominates (see aura.embeddings.embed_texts).

        The exemplar parameters exist so tests can substitute known sets;
        production always uses the module-level sets.
        """
        if not question_exemplars:
            raise ValueError("QuestionDetector needs at least one question exemplar")
        if not statement_exemplars:
            raise ValueError("QuestionDetector needs at least one statement exemplar")

        embeddings = await embed_texts(model, (*question_exemplars, *statement_exemplars))
        split = len(question_exemplars)
        return cls(model, embeddings[:split], embeddings[split:])

    async def question_likeness(self, text: str) -> float:
        """Return how much more question-like than statement-like text reads.

        The score is a difference of two cosine similarities, so it spans
        [-2.0, 2.0] in principle: positive means the text sits closer to the
        question exemplars than to the statement exemplars, negative the
        other way round, and zero says neither. In practice real messages
        land in a much narrower band around zero, because both sides share
        the same topical signal -- which is exactly what the subtraction is
        there to remove.

        Each side is a maximum, not a mean, across its set: a message only
        has to look like *one* kind of question to be one, and averaging
        would drown that single strong match in eighteen unrelated ones. The
        same argument applies unchanged to the statement side.

        Returns _NO_QUESTION_EVIDENCE (the floor, -2.0) rather than raising
        for the two inputs where a score would be meaningless: text that is
        empty or whitespace-only (nothing was asked), and the degenerate case
        of a non-finite similarity, which would otherwise propagate a NaN
        into comparisons and into SQLite, where a NaN REAL silently becomes
        NULL. The floor rather than zero specifically so an unscoreable
        message cannot pass a gate; see that constant.

        Never truncates in a way that changes a realistic result; see
        _MAX_CLASSIFIED_CHARACTERS.
        """
        stripped = text.strip()
        if not stripped:
            return _NO_QUESTION_EVIDENCE

        embedding = await embed_text(self._model, stripped[:_MAX_CLASSIFIED_CHARACTERS])
        question_similarity = max(
            cosine_similarity(exemplar, embedding) for exemplar in self._question_embeddings
        )
        statement_similarity = max(
            cosine_similarity(exemplar, embedding) for exemplar in self._statement_embeddings
        )
        score = question_similarity - statement_similarity

        if not math.isfinite(score):
            logger.warning("Question-likeness scored non-finite (%r); treating as not a question", score)
            return _NO_QUESTION_EVIDENCE

        return score
