"""The hand-picked evaluation set for the supersession-judgment call.

WHO USES THIS, and it is now two scripts rather than one. scripts/
supersession_bakeoff.py ran the 32 cases below against three candidate models
with its own throwaway prompt, to choose a model (reports/
supersession-model-bakeoff.txt). scripts/supersession_reverify.py re-runs a
named subset -- REVERIFICATION_CASE_NAMES at the bottom of this file -- through
the SHIPPED prompt in aura.extraction.supersession, to measure whether the two
rules that prompt added actually fix the four pairs the bake-off measured going
wrong. The three PHASE_3A3_ATTACK_CASES exist only for the second of those and
were written after the first had finished.


Kept separate from the runner (scripts/supersession_bakeoff.py) so the cases can
be reviewed, extended and diffed as data, without reading the harness around
them -- the same split scripts/bakeoff_cases.py and
scripts/extraction_eval_cases.py already use.

WHAT THIS MEASURES. Phase 3a-2 (reports/phase-3a-2.txt Section 6) flags a
candidate fact as a possible successor to an existing active fact whenever
their embeddings score above EXTRACTION_DEDUP_SIMILARITY_THRESHOLD (0.70), but
never acts on the flag -- a moderator sees "this may replace fact #12" and
decides by hand. Phase 3a-3 wants to replace that bare flag with a real
judgement: is the candidate actually a successor (SUPERSESSION), does it sit
alongside the old fact without conflict (COMPLEMENTARY), does it conflict with
no way to tell which is current (CONTRADICTION), or is the embedding hit a
false positive with no real relationship at all (INDEPENDENT)? This bake-off
picks the model for that judgement; it does not build the judgement itself.

WHY HAND-WRITTEN, NOT GENERATED (same reasoning as phase-3a-2.txt Section 5,
which quotes phase-3a-1b.txt Section 9 on the same point): the boundary this
evaluation exists to probe -- supersession vs. complementary refinement,
contradiction vs. thematic false positive -- is exactly the boundary a
generator model would blur the same way the model under test might. A
generated corpus of "pairs that are ambiguous between two categories" would
be only as good at being ambiguous as the model asked to generate it.

FOUR MATCHED BOUNDARY PAIRS, each repeated 3x (see `repeats`), because Phase
2's bake-off (reports/model-bakeoff.txt Section 4) found the decisive failure
mode is not "produces bad JSON" but "flip-flops on exactly the cases that
matter" -- gpt-5.4-mini scored 4/6 on identical repeated calls. Two pairs here
share a predecessor fact word-for-word and diverge only in whether the
candidate is a refinement or a replacement (SUPERSESSION-1 / COMPLEMENTARY-1,
mirroring CLAUDE.md's own "tournament starts Saturday" / "starts Saturday at
18:00" example). The other two share a predecessor and diverge only in
whether the candidate changes the same fact's value (CONTRADICTION-1) or
changes the subject entirely while reusing the wording (INDEPENDENT-1) -- the
"Attack It" case the phase brief specifically asked for.
"""
from __future__ import annotations

from dataclasses import dataclass

# Expected-category labels. English regardless of a case's own locale, per
# CLAUDE.md ("development... operates in English") -- these are evaluation
# labels, not user-facing text.
SUPERSESSION = "supersession"
COMPLEMENTARY = "complementary"
CONTRADICTION = "contradiction"
INDEPENDENT = "independent"

ALL_CATEGORIES = (SUPERSESSION, COMPLEMENTARY, CONTRADICTION, INDEPENDENT)


@dataclass(frozen=True)
class SupersessionCase:
    """One evaluation pair: an existing (predecessor) fact and a candidate fact.

    Mirrors what Phase 3a-2's dedup check actually flags: an active fact
    already in the database, and a new distilled candidate that scored above
    the similarity threshold against it. `predecessor_locale` /
    `candidate_locale` differ for the cross-locale cases, since facts are
    stored in whatever language they were originally written in.

    `boundary` marks the four matched pairs designed to separate a calibrated
    model from an optimistic one; `repeats` is 3 for those and 1 otherwise
    (Phase 2's bake-off methodology -- see module docstring).
    """

    name: str
    category: str
    predecessor_locale: str
    predecessor: str
    candidate_locale: str
    candidate: str
    rationale: str
    boundary: bool = False
    repeats: int = 1

    @property
    def cross_locale(self) -> bool:
        return self.predecessor_locale != self.candidate_locale


# =============================================================================
# 1. SUPERSESSION -- candidate is the clear, later successor; predecessor
#    should be superseded.
# =============================================================================
SUPERSESSION_CASES: list[SupersessionCase] = [
    SupersessionCase(
        name="supersession-tournament-day-change",
        category=SUPERSESSION,
        predecessor_locale="en-US",
        predecessor="The winter tournament starts Saturday.",
        candidate_locale="en-US",
        candidate="The winter tournament originally set for Saturday has been "
        "moved to Sunday.",
        rationale="Explicit reschedule language ('originally set for ... has been "
        "moved to') names the old value and replaces it outright -- not a "
        "refinement, a correction. Matched against COMPLEMENTARY-1, which shares "
        "the identical predecessor sentence and adds detail instead of changing "
        "the day, to test whether the model tells refinement from replacement "
        "rather than defaulting to one.",
        boundary=True,
        repeats=3,
    ),
    SupersessionCase(
        name="supersession-meeting-time-moved",
        category=SUPERSESSION,
        predecessor_locale="en-US",
        predecessor="The weekly team meeting is held every Wednesday at 21:00 UTC.",
        candidate_locale="en-US",
        candidate="Starting this week, the weekly team meeting has been moved to "
        "Wednesdays at 19:00 UTC.",
        rationale="Same event, same weekday, one changed value (time), with "
        "explicit change language ('starting this week ... moved to'). Textbook "
        "supersession: the new value replaces the old one entirely.",
    ),
    SupersessionCase(
        name="supersession-channel-renamed",
        category=SUPERSESSION,
        predecessor_locale="de",
        predecessor="Der Ankündigungskanal heißt #news.",
        candidate_locale="de",
        candidate="Der Ankündigungskanal #news wurde in #ankündigungen umbenannt.",
        rationale="The candidate names the old channel and states it was renamed "
        "-- the predecessor's claim (the channel is called #news) is no longer "
        "true and should be marked superseded.",
    ),
    SupersessionCase(
        name="supersession-level-requirement-raised-cross",
        category=SUPERSESSION,
        predecessor_locale="en-US",
        predecessor="Members must be level 5 to access the trading channel.",
        candidate_locale="de",
        candidate="Ab sofort müssen Mitglieder Level 10 erreichen, um auf den "
        "Handelskanal zuzugreifen.",
        rationale="Same rule (the level gate on the trading channel), same "
        "specific detail (the required level), one changed value, with explicit "
        "'ab sofort' (effective immediately) signalling a replacement rather "
        "than an addition. Cross-locale: predecessor in English, candidate in "
        "German, as would happen if a German-speaking mod restated an "
        "English-origin rule.",
    ),
    SupersessionCase(
        name="supersession-tournament-rescheduled-cross",
        category=SUPERSESSION,
        predecessor_locale="ja",
        predecessor="トーナメントは8月15日に開催される。",
        candidate_locale="en-US",
        candidate="The tournament originally set for August 15 has been "
        "rescheduled to August 22.",
        rationale="The candidate explicitly names the predecessor's date (August "
        "15) and states it moved to August 22 -- an unambiguous, cross-locale "
        "supersession of the same event.",
    ),
    SupersessionCase(
        name="supersession-moderator-role-transfer",
        category=SUPERSESSION,
        predecessor_locale="fr",
        predecessor="Le modérateur en chef du serveur est Marc.",
        candidate_locale="fr",
        candidate="Marc a quitté son poste de modérateur en chef ; c'est "
        "désormais Julie qui occupe ce rôle.",
        rationale="The candidate states the predecessor's subject (who holds the "
        "head-moderator role) explicitly changed hands, naming both the old "
        "holder and the departure. The predecessor's specific claim is now false.",
    ),
    SupersessionCase(
        name="supersession-upload-limit-increased-cross",
        category=SUPERSESSION,
        predecessor_locale="tr",
        predecessor="Sunucudaki dosya yükleme limiti 8 MB'dir.",
        candidate_locale="pl",
        candidate="Limit przesyłania plików na serwerze został zwiększony do 25 MB.",
        rationale="Same fact (the server-wide upload limit), one changed value, "
        "with explicit 'został zwiększony' (was increased) signalling a "
        "replacement of the old number, not a second, coexisting limit. "
        "Cross-locale: Turkish predecessor, Polish candidate.",
    ),
    SupersessionCase(
        name="supersession-channel-closed",
        category=SUPERSESSION,
        predecessor_locale="pt-BR",
        predecessor="O canal #sugestões está aberto para todos os membros "
        "enviarem ideias.",
        candidate_locale="pt-BR",
        candidate="O canal #sugestões foi fechado permanentemente e não aceita "
        "mais sugestões.",
        rationale="A status_change from open to closed on the same channel. "
        "CLAUDE.md names retirement of a rule/feature as itself a current, "
        "checkable fact that replaces the prior state -- the predecessor's "
        "'is open' claim is now false.",
    ),
]

# =============================================================================
# 2. COMPLEMENTARY -- both facts are independently valid; no replacement needed.
# =============================================================================
COMPLEMENTARY_CASES: list[SupersessionCase] = [
    SupersessionCase(
        name="complementary-tournament-time-detail",
        category=COMPLEMENTARY,
        predecessor_locale="en-US",
        predecessor="The winter tournament starts Saturday.",
        candidate_locale="en-US",
        candidate="The winter tournament starts Saturday at 18:00 UTC in the "
        "#events channel.",
        rationale="CLAUDE.md's own worked example: a refinement that adds detail "
        "(exact time, channel) without changing or contradicting the original "
        "claim (the day). Matched against SUPERSESSION-1, which shares this "
        "exact predecessor sentence but changes the day instead of adding "
        "detail to it, to test whether the model can tell the two apart rather "
        "than defaulting to one.",
        boundary=True,
        repeats=3,
    ),
    SupersessionCase(
        name="complementary-trading-channel-two-rules",
        category=COMPLEMENTARY,
        predecessor_locale="de",
        predecessor="Im Handelskanal sind nur Textnachrichten erlaubt, keine Bilder.",
        candidate_locale="de",
        candidate="Im Handelskanal muss jeder Handel vom Käufer und Verkäufer "
        "per Reaktion bestätigt werden.",
        rationale="Two independent rules about the same channel (a format "
        "restriction and a confirmation procedure) that do not touch the same "
        "specific detail -- both stay true at once.",
    ),
    SupersessionCase(
        name="complementary-art-contest-two-aspects-cross",
        category=COMPLEMENTARY,
        predecessor_locale="en-US",
        predecessor="The art contest submissions close on August 30.",
        candidate_locale="ja",
        candidate="アートコンテストの優勝者には特別ロールが付与される。",
        rationale="Same event (the art contest), different non-overlapping "
        "aspects -- the submission deadline and the winner's prize. Neither "
        "statement bears on the other. Cross-locale: English predecessor, "
        "Japanese candidate.",
    ),
    SupersessionCase(
        name="complementary-boost-level-consequence",
        category=COMPLEMENTARY,
        predecessor_locale="es-ES",
        predecessor="El servidor alcanzó el nivel 2 de boost en julio.",
        candidate_locale="es-ES",
        candidate="Gracias al nivel 2 de boost, ahora hay 100 emojis "
        "personalizados disponibles.",
        rationale="The candidate is a consequence of the predecessor, not a "
        "replacement of it -- both remain simultaneously true and a synthesis "
        "answer benefits from citing both together.",
    ),
    SupersessionCase(
        name="complementary-mod-application-two-aspects-cross",
        category=COMPLEMENTARY,
        predecessor_locale="fr",
        predecessor="Les candidatures de modérateur sont ouvertes deux fois par "
        "an, en mai et novembre.",
        candidate_locale="de",
        candidate="Bewerbungen für Moderatoren müssen über das Formular im "
        "Kanal #bewerbungen eingereicht werden.",
        rationale="Same topic (moderator applications), non-overlapping "
        "specifics -- timing versus submission mechanism. Cross-locale: French "
        "predecessor, German candidate, plausible if the two facts were "
        "extracted from different channels' conversations.",
    ),
    SupersessionCase(
        name="complementary-game-night-day-vs-game",
        category=COMPLEMENTARY,
        predecessor_locale="tr",
        predecessor="Oyun gecesi her Cuma akşamı düzenlenir.",
        candidate_locale="tr",
        candidate="Oyun gecesinde bu ay Among Us oynanacak.",
        rationale="A recurring schedule fact and a this-month specific detail "
        "about the same recurring event -- neither claim conflicts with or "
        "replaces the other.",
    ),
    SupersessionCase(
        name="complementary-onboarding-two-steps-cross",
        category=COMPLEMENTARY,
        predecessor_locale="pt-BR",
        predecessor="Novos membros recebem um cargo temporário 'Visitante' ao "
        "entrar.",
        candidate_locale="en-US",
        candidate="New members must react to the rules message to receive the "
        "full member role.",
        rationale="Two sequential steps of the same onboarding flow (temporary "
        "role on join, then a separate action to earn the full role) -- both "
        "true, neither replaces the other. Cross-locale: Portuguese "
        "predecessor, English candidate.",
    ),
    SupersessionCase(
        name="complementary-voice-capacity-vs-recommendation",
        category=COMPLEMENTARY,
        predecessor_locale="ko",
        predecessor="음성 채널은 최대 20명까지 입장할 수 있다.",
        candidate_locale="ko",
        candidate="음성 채널 사용 시 헤드셋 마이크 사용을 권장한다.",
        rationale="A hard capacity limit and a soft usage recommendation about "
        "the same channel -- independent facts about different properties of "
        "the same subject.",
    ),
]

# =============================================================================
# 3. CONTRADICTION, DIRECTION UNCLEAR -- both cannot be true at once, but
#    nothing indicates which one is current. Escalate to a moderator, never
#    auto-resolve.
# =============================================================================
CONTRADICTION_CASES: list[SupersessionCase] = [
    SupersessionCase(
        name="contradiction-upload-limit-same-channel",
        category=CONTRADICTION,
        predecessor_locale="en-US",
        predecessor="The upload limit in the #resources channel is 25 MB.",
        candidate_locale="en-US",
        candidate="The upload limit in the #resources channel is 10 MB.",
        rationale="Same channel, same specific detail (the numeric limit), two "
        "different values, no change language ('increased to', 'now', 'moved') "
        "indicating an update -- both simply assert a value. Matched against "
        "INDEPENDENT-1, which shares this exact predecessor but changes the "
        "channel along with the number, to test whether the model requires the "
        "SAME specific detail to call it a genuine conflict.",
        boundary=True,
        repeats=3,
    ),
    SupersessionCase(
        name="contradiction-pet-role-limit",
        category=CONTRADICTION,
        predecessor_locale="de",
        predecessor="Mitglieder dürfen maximal 3 Haustier-Rollen gleichzeitig haben.",
        candidate_locale="de",
        candidate="Mitglieder dürfen maximal 5 Haustier-Rollen gleichzeitig haben.",
        rationale="Same specific rule and detail (the pet-role cap), two "
        "different numbers, neither sentence signals a change -- genuinely "
        "unclear which is current.",
    ),
    SupersessionCase(
        name="contradiction-announcement-posting-rights-cross",
        category=CONTRADICTION,
        predecessor_locale="ja",
        predecessor="イベント告知チャンネルは現在誰でも投稿できる。",
        candidate_locale="tr",
        candidate="Etkinlik duyuru kanalına sadece moderatörler mesaj atabilir.",
        rationale="Same channel and same specific question (who may post), "
        "mutually exclusive answers ('anyone' vs. 'moderators only'), no cue "
        "for which is current. Cross-locale: Japanese predecessor, Turkish "
        "candidate.",
    ),
    SupersessionCase(
        name="contradiction-server-creation-year",
        category=CONTRADICTION,
        predecessor_locale="es-ES",
        predecessor="El servidor fue creado en 2019.",
        candidate_locale="es-ES",
        candidate="El servidor fue creado en 2021.",
        rationale="A fact that cannot legitimately change over time (a founding "
        "date) stated two different ways -- not a case where 'the newer one "
        "wins' applies, since founding dates do not get superseded. Genuinely "
        "unclear which is a data-entry error.",
    ),
    SupersessionCase(
        name="contradiction-voice-capacity-cross",
        category=CONTRADICTION,
        predecessor_locale="fr",
        predecessor="Le canal vocal principal peut accueillir jusqu'à 30 personnes.",
        candidate_locale="en-US",
        candidate="The main voice channel can hold up to 15 people.",
        rationale="Same channel, same specific detail (capacity), two "
        "irreconcilable numbers, no change language. Cross-locale: French "
        "predecessor, English candidate.",
    ),
    SupersessionCase(
        name="contradiction-art-contest-winner",
        category=CONTRADICTION,
        predecessor_locale="pt-BR",
        predecessor="O vencedor do concurso de arte foi @Ana.",
        candidate_locale="pt-BR",
        candidate="O vencedor do concurso de arte foi @Marcos.",
        rationale="Same contest, same specific question (who won), two "
        "different, mutually exclusive names, with no correction language "
        "('actually', 'the real winner is') to indicate one supersedes the "
        "other -- exactly the kind of pair that must escalate rather than "
        "guess.",
    ),
    SupersessionCase(
        name="contradiction-vip-level-requirement-cross",
        category=CONTRADICTION,
        predecessor_locale="pl",
        predecessor="Aby dołączyć do kanału głosowego VIP, trzeba mieć poziom 20.",
        candidate_locale="ko",
        candidate="VIP 음성 채널에 참여하려면 레벨 15가 필요하다.",
        rationale="Same specific gate (the VIP voice channel's level "
        "requirement), two different thresholds, no change language. "
        "Cross-locale: Polish predecessor, Korean candidate.",
    ),
    SupersessionCase(
        name="contradiction-boost-tier",
        category=CONTRADICTION,
        predecessor_locale="ko",
        predecessor="서버는 현재 부스트 3단계이다.",
        candidate_locale="ko",
        candidate="서버는 현재 부스트 2단계이다.",
        rationale="Both claim to state the server's CURRENT boost tier as of "
        "now, with different values and no timestamp cue distinguishing which "
        "is more recent -- unlike a genuine supersession, neither sentence "
        "references the other's value.",
    ),
]

# =============================================================================
# 4. INDEPENDENT / FALSE POSITIVE -- only thematically similar; embeddings
#    landed above the dedup threshold, but nothing here belongs in the other
#    three categories.
# =============================================================================
INDEPENDENT_CASES: list[SupersessionCase] = [
    SupersessionCase(
        name="independent-upload-limit-different-channel",
        category=INDEPENDENT,
        predecessor_locale="en-US",
        predecessor="The upload limit in the #resources channel is 25 MB.",
        candidate_locale="en-US",
        candidate="The upload limit in the #screenshots channel is 10 MB.",
        rationale="Near-identical wording to CONTRADICTION-1's predecessor, but "
        "the changed word is the CHANNEL, not the number -- these are two "
        "separate rules about two separate channels, coincidentally similar "
        "enough in phrasing to clear an embedding threshold. The exact 'Attack "
        "It' case the phase brief asked for: does the model resist proposing a "
        "conflict from topic proximity alone?",
        boundary=True,
        repeats=3,
    ),
    SupersessionCase(
        name="independent-cheating-different-competitions",
        category=INDEPENDENT,
        predecessor_locale="de",
        predecessor="Betrug wird im Valorant-Turnier nicht toleriert und führt "
        "zur Disqualifikation.",
        candidate_locale="de",
        candidate="Betrug wird im Baucontest in Minecraft nicht toleriert und "
        "führt zur Disqualifikation.",
        rationale="Identical template sentence applied to two entirely "
        "different competitions -- high lexical/embedding overlap, zero actual "
        "relationship between the two rules.",
    ),
    SupersessionCase(
        name="independent-new-member-rule-different-topic-cross",
        category=INDEPENDENT,
        predecessor_locale="en-US",
        predecessor="New members must read the #rules channel before posting.",
        candidate_locale="de",
        candidate="Neue Mitglieder müssen ihre Discord-E-Mail bestätigen, bevor "
        "sie eine eigene Rolle wählen können.",
        rationale="Shared 'new members must X before Y' template, but X and Y "
        "name completely different actions (reading rules vs. verifying email; "
        "posting vs. choosing a role) -- topically adjacent, substantively "
        "unrelated. Cross-locale: English predecessor, German candidate.",
    ),
    SupersessionCase(
        name="independent-two-tournaments-same-weekend",
        category=INDEPENDENT,
        predecessor_locale="fr",
        predecessor="Le tournoi d'échecs aura lieu ce week-end dans le salon "
        "#echecs.",
        candidate_locale="fr",
        candidate="Le tournoi de belote aura lieu ce week-end dans le salon "
        "#cartes.",
        rationale="Two distinct tournaments (chess, belote) in two distinct "
        "channels that merely share a weekend and sentence structure -- not "
        "the same event under any reading.",
    ),
    SupersessionCase(
        name="independent-channel-numeric-limit-different-subject-cross",
        category=INDEPENDENT,
        predecessor_locale="tr",
        predecessor="Reklam kanalında günde en fazla 1 mesaj paylaşılabilir.",
        candidate_locale="pl",
        candidate="Pytania kanału pomocy technicznej mogą być zadawane maksymalnie "
        "3 razy dziennie.",
        rationale="Both are '#channel: numeric daily limit' rules, which is "
        "exactly the shape that inflates embedding similarity, but the "
        "channels (ads, support) and the limited activity (posting ads, asking "
        "questions) are unrelated. Cross-locale: Turkish predecessor, Polish "
        "candidate.",
    ),
    SupersessionCase(
        name="independent-milestone-different-metric",
        category=INDEPENDENT,
        predecessor_locale="ja",
        predecessor="サーバーのフォロワー数が1000人を突破した。",
        candidate_locale="ja",
        candidate="サーバーのYouTubeチャンネル登録者数が1000人を突破した。",
        rationale="Same round number (1000), same 'surpassed X' template, "
        "completely different metric (server followers vs. a YouTube channel's "
        "subscribers) -- the classic embedding trap of two milestones sharing "
        "everything but their subject.",
    ),
    SupersessionCase(
        name="independent-prohibited-sharing-different-topic-cross",
        category=INDEPENDENT,
        predecessor_locale="pt-BR",
        predecessor="É proibido compartilhar links de outros servidores no chat "
        "geral.",
        candidate_locale="en-US",
        candidate="Sharing spoilers about ongoing anime series is prohibited in "
        "the #anime channel.",
        rationale="Shared 'X is prohibited in Y' template, entirely different "
        "prohibited activity and entirely different channel. Cross-locale: "
        "Portuguese predecessor, English candidate.",
    ),
    SupersessionCase(
        name="independent-decision-closure-different-channel",
        category=INDEPENDENT,
        predecessor_locale="es-ES",
        predecessor="Se decidió que el canal de música se cerrará los domingos "
        "por mantenimiento.",
        candidate_locale="es-ES",
        candidate="Se decidió que el canal de arte se cerrará los domingos para "
        "revisión de contenido.",
        rationale="Same 'closes on Sundays' template and decision framing "
        "applied to two different channels for two different reasons -- "
        "coincidental phrasing overlap, not the same decision.",
    ),
]

# =============================================================================
# 5. PHASE 3a-3 ATTACK CASES -- written AFTER the bake-off, against the two
#    rules the shipped prompt added (see aura.extraction.supersession).
#
#    The bake-off's own set could not test these, because the rules did not
#    exist when it ran. Each one attacks a rule from the direction that rule
#    makes newly dangerous:
#
#      * Rule 1 says a value change with no transition wording is a
#        contradiction, never a supersession. The over-correction that buys is
#        treating a REAL, clearly-signalled update as a conflict -- so the first
#        case is the bake-off's decisive contradiction pair with the signal put
#        back in, and it must come out the other way.
#      * The mirror-image over-correction is reading the presence of a signal as
#        evidence of succession, when the two facts are not even about the same
#        thing. The other two cases carry unmistakable transition wording
#        attached to something other than the disagreement.
#
#    All three are `boundary` and repeat 3x, the same discipline the bake-off
#    used: a rule that holds once and not twice has not held.
# =============================================================================
PHASE_3A3_ATTACK_CASES: list[SupersessionCase] = [
    SupersessionCase(
        name="supersession-pet-role-limit-with-signal",
        category=SUPERSESSION,
        predecessor_locale="de",
        predecessor="Mitglieder dürfen maximal 3 Haustier-Rollen gleichzeitig haben.",
        candidate_locale="de",
        candidate="Ab sofort dürfen Mitglieder maximal 5 Haustier-Rollen "
        "gleichzeitig haben.",
        rationale="Word-for-word CONTRADICTION-2 (contradiction-pet-role-limit, "
        "the case that decided the model bake-off) with 'Ab sofort' added and "
        "nothing else changed. The pair isolates exactly one variable: the "
        "transition wording. Rule 1 must turn on that wording and nothing else "
        "-- if this comes back as a contradiction, the rule has stopped "
        "distinguishing an update from a disagreement and merely made the model "
        "refuse both.",
        boundary=True,
        repeats=3,
    ),
    SupersessionCase(
        name="independent-upload-limit-different-channel-with-signal",
        category=INDEPENDENT,
        predecessor_locale="en-US",
        predecessor="The upload limit in the #resources channel is 25 MB.",
        candidate_locale="en-US",
        candidate="From now on, the upload limit in the #screenshots channel is "
        "10 MB.",
        rationale="INDEPENDENT-1 with 'From now on' added. The changed word is "
        "still the CHANNEL, so these remain two separate rules about two "
        "separate channels -- but the sentence now contains the strongest "
        "possible change signal, aimed at a rule that was never the "
        "predecessor's. A model reading 'signal present' as 'therefore a "
        "successor' proposes replacing a fact about #resources with one about "
        "#screenshots, which is a dangerous miss in the bake-off's sense: it "
        "invites a moderator to retire a rule that still holds.",
        boundary=True,
        repeats=3,
    ),
    SupersessionCase(
        name="contradiction-capacity-with-misattached-signal",
        category=CONTRADICTION,
        predecessor_locale="fr",
        predecessor="Le canal vocal principal peut accueillir jusqu'à 30 personnes.",
        candidate_locale="fr",
        candidate="Le salon #annonces est désormais réservé aux modérateurs, et "
        "le canal vocal principal peut accueillir jusqu'à 15 personnes.",
        rationale="The subtler half of the same attack: the candidate carries a "
        "real change signal ('désormais'), but it belongs to a DIFFERENT clause "
        "-- the announcements channel's posting rights, which the predecessor "
        "says nothing about. The capacity numbers still disagree with no "
        "explanation, so this is a contradiction. Getting it right requires "
        "attaching the signal to the claim it actually modifies rather than to "
        "the sentence as a whole, which a keyword check could never do and is "
        "the reason this judgement is a model call at all.",
        boundary=True,
        repeats=3,
    ),
]

ALL_CASES: list[SupersessionCase] = (
    SUPERSESSION_CASES
    + COMPLEMENTARY_CASES
    + CONTRADICTION_CASES
    + INDEPENDENT_CASES
    + PHASE_3A3_ATTACK_CASES
)

# The set Phase 3a-3 re-runs against the SHIPPED prompt (see
# scripts/supersession_reverify.py): every pair the bake-off measured a model
# getting wrong for a reason the prompt was then changed to address, one
# unchanged control, and the three attack cases above.
REVERIFICATION_CASE_NAMES: tuple[str, ...] = (
    # The four the bake-off's Section 4 identified, and the four the phase brief
    # names by hand.
    "contradiction-pet-role-limit",
    "supersession-level-requirement-raised-cross",
    "supersession-channel-closed",
    "complementary-voice-capacity-vs-recommendation",
    # Two controls, both already correct in the bake-off and both kept in the
    # set precisely because they could REGRESS. The first is the attack pair's
    # twin without the change signal, so "the signal did not break it" and "the
    # signal changed nothing" can be told apart. The second is the bake-off's
    # own "Attack It" case (the one Gemini failed) and the nearest neighbour to
    # the widened complementary definition: if widening it starts linking facts
    # that merely share a sentence shape, this is where that shows up.
    "independent-upload-limit-different-channel",
    "independent-new-member-rule-different-topic-cross",
    *(case.name for case in PHASE_3A3_ATTACK_CASES),
)

BOUNDARY_CASES: list[SupersessionCase] = [case for case in ALL_CASES if case.boundary]


def cases_by_category(category: str) -> list[SupersessionCase]:
    return [case for case in ALL_CASES if case.category == category]
