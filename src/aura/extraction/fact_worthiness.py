"""How fact-worthy is a piece of text? The free, local first filter for Phase 3a.

Trigger 2 (aura.proactive.question_detector) only ever scores a message once a
question detector has already let it through -- a grammatical/semantic filter
that answers "does this look like someone asking for information?" Automatic
extraction has no such head start: it must look at *every* message in an
enabled channel and decide, on content alone, whether it is the kind of thing
CLAUDE.md's knowledge model wants at all (an announcement, a rule, a decision,
an event or schedule, a status/availability change) as opposed to the
overwhelming majority of ordinary chat that is not (greetings, small talk,
opinions, questions -- Trigger 1/2's job, not this one -- personal messages,
and reactions with no content of their own).

This module reuses aura.proactive.question_detector.QuestionDetector exactly
as it stands, unmodified: that class already takes two arbitrary exemplar
sets and computes a contrastive score generically (see
reports/phase-3-pre-analysis.md Section 8, which verified this directly
before this module was written). A second instance, constructed here with
FACT_WORTHY_EXEMPLARS and NOT_FACT_WORTHY_EXEMPLARS instead of the question/
statement pair, is the whole mechanism -- no new detector class, no change to
question_detector.py. The method name `question_likeness` stays literally
what it says on this instance too (a fact-worthiness detector answering
"how fact-worthy is this text" through a method called question_likeness
reads oddly) -- accepted deliberately rather than forking or renaming the
class, because the alternative was renaming a class and a method that eighty
call sites across src/ and tests/ already depend on (aura.proactive.gate,
aura.proactive.listener, main.py, and every proactive test module) for a
purely cosmetic gain. See reports/phase-3a-1.txt for the full reasoning.

Like QUESTION_EXEMPLARS/STATEMENT_EXEMPLARS, both sets below are topically
paired -- a fact-worthy exemplar and its nearest NOT_FACT_WORTHY_EXEMPLARS
neighbour talk about the same subject in the same language, differing only in
register: authoritative/distilled vs. hedged, personal, opinionated, or
conversational. That pairing is what makes the subtraction isolate "is this
authoritative and checkable" rather than "does this mention server topics" --
the same reasoning question_detector.py's own docstring gives for why its
statement exemplars mirror its question exemplars rather than being generic
filler.

This is deliberately the hardest register distinction in the whole project.
"I think the event is Saturday" and "The event is Saturday" are near-identical
in embedding space -- both are on-topic, declarative, and even share most of
their words -- and differing only in a hedge the embedding model may or may
not weight heavily. reports/phase-3a-1.txt documents that near-miss pair
explicitly and reports the calibrated threshold's actual behaviour on it
rather than assuming separation that was never measured.
"""
from __future__ import annotations

from fastembed import TextEmbedding

from aura.proactive.question_detector import QuestionDetector

# Authoritative, checkable statements about how the server works right now --
# CLAUDE.md's "one distilled sentence" register, the same register the two
# real hand-entered facts in data/aura.db happen to be written in (see
# reports/phase-3-pre-analysis.md Section 4). Five subcategories per the
# Phase 3a-1 design brief: announcements, rules/policies, decisions,
# events/schedules, and status/availability changes. Every entry could,
# almost unedited, become a Fact.content value.
FACT_WORTHY_EXEMPLARS: tuple[str, ...] = (
    # English -- status/availability change
    "The server will be down for maintenance today at 2 PM UTC.",
    "Voice chat is temporarily disabled while we move to a new host.",
    # English -- rule/policy
    "New members must verify their email before they can post in any channel.",
    "Self-promotion is only allowed in the #showcase channel, once per week.",
    # English -- decision
    "The community vote decided the next game night will be Stardew Valley.",
    "The mod team decided to keep the NSFW channel closed permanently.",
    # English -- event/schedule
    "The winter tournament starts Saturday at 6 PM in the #events channel.",
    "Study sessions now run every Tuesday and Thursday at 7 PM server time.",
    # English -- announcement
    "The giveaway winner will be announced next Friday in #announcements.",
    # German -- status change / rule / event
    "Der Server wird um 14 Uhr einer Wartung ausgesetzt.",
    "Ab sofort müssen neue Mitglieder ihre E-Mail bestätigen, bevor sie schreiben können.",
    "In 2 Wochen findet ein Event in Minecraft statt.",
    # Spanish -- rule / decision
    "A partir de hoy, la autopromoción solo está permitida en el canal #proyectos.",
    "El equipo de moderación decidió cerrar el canal de sugerencias antiguas.",
    # Portuguese (Brazil) -- event / status change
    "O servidor vai passar por manutenção às 14h de hoje.",
    "A partir da próxima semana, as reuniões de estudo serão às terças-feiras.",
    # French -- rule / announcement
    "Les nouveaux membres doivent valider leur e-mail avant de pouvoir écrire.",
    "Le gagnant du concours sera annoncé vendredi prochain.",
    # Turkish -- status change / event
    "Sunucu bugün saat 14:00'te bakım nedeniyle kapatılacak.",
    "Turnuva bu cumartesi saat 18:00'de #etkinlikler kanalında başlıyor.",
    # Polish -- rule / decision
    "Od dzisiaj nowi członkowie muszą potwierdzić swój e-mail przed pisaniem.",
    "Zespół moderacji zdecydował o zamknięciu starego kanału sugestii.",
    # Japanese -- status change / event
    "サーバーは本日14時からメンテナンスのため停止します。",
    "冬季トーナメントは土曜日18時に#eventsチャンネルで始まります。",
    # Korean -- rule / decision
    "오늘부터 새 멤버는 글을 쓰기 전에 이메일 인증을 해야 합니다.",
    "커뮤니티 투표 결과 다음 게임의 밤은 스타듀밸리로 결정되었습니다.",
)

# The mirror of FACT_WORTHY_EXEMPLARS: the same subjects, in the same
# languages, differing only in NOT being an authoritative, checkable
# statement. Spans the six not-fact-worthy subcategories from the design
# brief: greetings, small talk, opinions without factual content, questions
# (Trigger 1/2's job, never extraction's), personal/private remarks, and
# reactions with no content of their own.
#
# Several entries are deliberately near-misses of their fact-worthy
# counterpart rather than obviously unrelated chatter -- a hedge ("I think"),
# a wish, or a reaction to the same event -- because generic filler would not
# test the register distinction this filter actually has to make. See this
# module's docstring for why that is the hard case, and
# reports/phase-3a-1.txt for how the calibrated threshold handles it.
NOT_FACT_WORTHY_EXEMPLARS: tuple[str, ...] = (
    # English -- small talk/personal, same topic as the status-change pair
    "ugh my wifi keeps dropping, hope it's back before the maintenance thing",
    "voice chat being down again? classic, this always happens to me",
    # English -- opinion, same topic as the rule pair
    "honestly I don't think email verification stops anyone determined enough",
    "the self-promo rule is way too strict if you ask me",
    # English -- hedge/question, same topic as the decision pair
    "I heard we might be doing Stardew again? not sure if that's final",
    "wait did they actually decide to keep NSFW closed or is that still up for debate",
    # English -- hedge, same topic as the event pair (the Attack It near-miss)
    "I think the tournament might be around this weekend?",
    "pretty sure study sessions are sometime this week, could be wrong though",
    # English -- reaction, same topic as the announcement pair
    "omg I hope I win the giveaway this time lol",
    # English -- greeting / personal / question / reaction, generic register
    "good morning everyone, hope you all slept well",
    "just got back from the gym, absolutely dead right now",
    "does anyone know when the next event actually is",
    "lmaooo that clip was amazing",
    # German -- hedge / opinion / greeting
    "ich glaube der Server geht heute um 14 Uhr offline, oder war das gestern",
    "ehrlich gesagt nervt mich diese E-Mail-Bestätigung total",
    "guten Morgen zusammen, hoffe ihr hattet alle einen guten Start",
    # Spanish -- opinion / reaction
    "no sé, a mí la regla de autopromoción me parece un poco exagerada",
    "uy qué pena que cerraron el canal de sugerencias, lo usaba mucho",
    # Portuguese (Brazil) -- hedge / personal
    "acho que o servidor vai cair mais tarde, não tenho certeza da hora",
    "cheguei agora do trabalho, que dia cansativo",
    # French -- opinion / reaction
    "je trouve que c'est un peu long comme vérification franchement",
    "ah trop bien pour le gagnant, j'espère que c'est moi la prochaine fois",
    # Turkish -- hedge / small talk
    "sanırım sunucu bugün bir ara kapanacaktı ama emin değilim",
    "turnuva bu hafta sonu falan mıydı, hatırlamıyorum",
    # Polish -- opinion / greeting
    "moim zdaniem ta weryfikacja e-mail to strata czasu",
    "dzień dobry wszystkim, miłego dnia",
    # Japanese -- hedge / reaction
    "メンテナンスって今日だったかな、ちょっと自信ない",
    "トーナメント楽しみすぎる、勝てるといいな",
    # Korean -- hedge / small talk
    "메인테넌스 오늘 맞나? 확실친 않은데",
    "오늘 완전 피곤하다, 방금 집에 왔어",
)


async def create_fact_worthiness_detector(model: TextEmbedding) -> QuestionDetector:
    """Build a fact-worthiness detector: a second QuestionDetector instance.

    Deliberately a thin wrapper around QuestionDetector.create rather than a
    second class -- see this module's docstring. Callers that want the raw
    class for typing or mocking should still import QuestionDetector directly;
    this function exists so a call site never has to spell out
    FACT_WORTHY_EXEMPLARS/NOT_FACT_WORTHY_EXEMPLARS itself, the same
    convenience main.py's setup_hook gets from calling QuestionDetector.create
    with its module-level defaults.
    """
    return await QuestionDetector.create(
        model,
        question_exemplars=FACT_WORTHY_EXEMPLARS,
        statement_exemplars=NOT_FACT_WORTHY_EXEMPLARS,
    )
