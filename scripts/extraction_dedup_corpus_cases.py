"""Hand-written pair corpus for calibrating EXTRACTION_DEDUP_SIMILARITY_THRESHOLD.

WHAT THIS MEASURES, and how it differs from scripts/supersession_bakeoff_cases.py.
That corpus calibrates a MODEL call: given a pair the dedup filter already
flagged, what IS the relationship (supersession / complementary / contradiction
/ independent)? This corpus calibrates the filter that decides whether a pair
ever reaches that call at all -- a free, local, embedding-only question with a
completely different shape: "does candidate B look enough like predecessor A
that a moderator should be shown a hint, and (since Phase 3a-3) that a paid
judgement call should be spent on it?"

That reframing is why this corpus uses four categories instead of the
bake-off's four, not the same four. The bake-off's "complementary" is
deliberately absent here: a complementary pair (a hard capacity limit and a
soft recommendation about the same voice channel, CLAUDE.md's own worked
example) is a case the DOWNSTREAM judgement call is supposed to receive and
correctly label "no action needed" -- but whether the pre-filter marks it is
not the question this threshold answers, and folding it in as a required
"should mark" case would conflate two different calibration questions. What
this threshold IS asked to separate is:

  1. DUPLICATE       -- should mark: the same fact, reworded or translated.
  2. SUPERSESSION     -- should mark: a genuine successor on the same detail.
  3. CONTRADICTION    -- should mark: same detail, different values.
  4. INDEPENDENT_RELATED / UNRELATED -- should NOT mark: no restatement
     relationship exists, whether the wording happens to look similar (4a) or
     not (4b).

HAND-WRITTEN, NOT GENERATED, for the same reason reports/phase-3a-2.txt
Section 5 and reports/phase-3a-1b.txt Section 9 give for their own corpora:
the boundary under test here (topically similar vs. actually the same claim)
is exactly the boundary a generator model would blur the same way the
mechanism under test does.

THE TWO NAMED ATTACK CASES from reports/supersession-bakeoff.json /
reports/supersession-reverify.json are imported here VERBATIM (not
retyped) via _imported(), per the calibration brief's explicit instruction:
independent-upload-limit-different-channel and
independent-new-member-rule-different-topic-cross. Both were written to
LOOK like a restatement while being about a different channel or a different
rule entirely -- exactly the shape this threshold has to hold below the
marking line. Five more cross-locale supersession/contradiction cases are
imported the same way, so the sweep can answer the brief's other explicit
question: would the cross-locale pairs the supersession work already found
tricky for a MODEL be reliably marked by this much cheaper filter in the
first place?
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from supersession_bakeoff_cases import ALL_CASES as _SUPERSESSION_CASES  # noqa: E402
from supersession_bakeoff_cases import SupersessionCase  # noqa: E402

_BY_NAME: dict[str, SupersessionCase] = {case.name: case for case in _SUPERSESSION_CASES}


class DedupCategory(Enum):
    """The four category labels this corpus assigns, not the bake-off's four."""

    DUPLICATE = "duplicate"
    SUPERSESSION = "supersession"
    CONTRADICTION = "contradiction"
    INDEPENDENT_RELATED = "independent_related"
    UNRELATED = "unrelated"


# What EXTRACTION_DEDUP_SIMILARITY_THRESHOLD is actually asked to do: mark a
# candidate as a possible predecessor hit whenever it is a duplicate, a
# supersession, or a contradiction of an existing fact, and hold it back
# otherwise. Ground truth for the sweep, not a property of any one case.
SHOULD_MARK_CATEGORIES = frozenset(
    {DedupCategory.DUPLICATE, DedupCategory.SUPERSESSION, DedupCategory.CONTRADICTION}
)


@dataclass(frozen=True)
class DedupPairCase:
    """One (predecessor, candidate) pair with its ground-truth marking label.

    Mirrors aura.extraction.pipeline's actual comparison: an existing active
    fact's text and a freshly distilled candidate's text, both embedded and
    compared by cosine similarity -- nothing else about either fact (channel,
    author, timestamp) enters that comparison, so nothing else belongs here
    either.
    """

    name: str
    category: DedupCategory
    predecessor_locale: str
    predecessor: str
    candidate_locale: str
    candidate: str
    rationale: str

    @property
    def cross_locale(self) -> bool:
        return self.predecessor_locale != self.candidate_locale

    @property
    def should_mark(self) -> bool:
        return self.category in SHOULD_MARK_CATEGORIES


def _imported(
    case_name: str, *, category: DedupCategory, name: str, rationale: str
) -> DedupPairCase:
    """Reuse a supersession-bake-off pair's exact text, under this corpus's own label.

    Text and locales come from the other module's data, not retyped, so the
    two corpora can never silently drift apart on what these specific
    sentences say.
    """
    source = _BY_NAME[case_name]
    return DedupPairCase(
        name=name,
        category=category,
        predecessor_locale=source.predecessor_locale,
        predecessor=source.predecessor,
        candidate_locale=source.candidate_locale,
        candidate=source.candidate,
        rationale=rationale,
    )


# =============================================================================
# 1. DUPLICATE -- the same fact, reworded or translated. Should mark.
#    18 same-locale paraphrases (2/locale) + 7 cross-locale translations,
#    since a translated duplicate is the harder case for a shared embedding
#    space and the one most worth stress-testing on its own.
# =============================================================================
DUPLICATE_CASES: list[DedupPairCase] = [
    DedupPairCase(
        name="dup-en-us-upload-limit",
        category=DedupCategory.DUPLICATE,
        predecessor_locale="en-US",
        predecessor="The upload limit in the #resources channel is 25 MB.",
        candidate_locale="en-US",
        candidate="You can upload files up to 25 MB in size over in #resources.",
        rationale="Same channel, same number, casual restatement -- nothing added or changed.",
    ),
    DedupPairCase(
        name="dup-en-us-boost-emoji",
        category=DedupCategory.DUPLICATE,
        predecessor_locale="en-US",
        predecessor="Thanks to boost level 2, the server now has 100 custom emoji slots.",
        candidate_locale="en-US",
        candidate="Since we hit boost tier 2, we've got 100 custom emoji slots available now.",
        rationale="Same claim (boost level 2 -> 100 emoji slots), reworded casually.",
    ),
    DedupPairCase(
        name="dup-es-es-voice-capacity",
        category=DedupCategory.DUPLICATE,
        predecessor_locale="es-ES",
        predecessor="El canal de voz principal admite hasta 30 personas.",
        candidate_locale="es-ES",
        candidate="En el canal de voz principal caben hasta 30 usuarios como máximo.",
        rationale="Same channel, same capacity number, restated.",
    ),
    DedupPairCase(
        name="dup-es-es-verification",
        category=DedupCategory.DUPLICATE,
        predecessor_locale="es-ES",
        predecessor="Los nuevos miembros deben verificar su correo antes de poder escribir.",
        candidate_locale="es-ES",
        candidate="Hay que confirmar el email para poder empezar a escribir si eres nuevo.",
        rationale="Same rule, casual restatement.",
    ),
    DedupPairCase(
        name="dup-pt-br-channel-status",
        category=DedupCategory.DUPLICATE,
        predecessor_locale="pt-BR",
        predecessor="O canal #sugestões está aberto para todos os membros.",
        candidate_locale="pt-BR",
        candidate="Qualquer membro pode postar no canal #sugestões, que está aberto.",
        rationale="Same open-status claim about the same channel, reworded.",
    ),
    DedupPairCase(
        name="dup-pt-br-tournament",
        category=DedupCategory.DUPLICATE,
        predecessor_locale="pt-BR",
        predecessor="O torneio de verão começa no sábado às 18h.",
        candidate_locale="pt-BR",
        candidate="No sábado, às 18h, dá o pontapé inicial no torneio de verão.",
        rationale="Same event and time, restated colloquially.",
    ),
    DedupPairCase(
        name="dup-de-level-gate",
        category=DedupCategory.DUPLICATE,
        predecessor_locale="de",
        predecessor="Mitglieder müssen Level 5 erreichen, um auf den Handelskanal zuzugreifen.",
        candidate_locale="de",
        candidate="Um den Handelskanal nutzen zu können, braucht man mindestens Level 5.",
        rationale="Same gate and level, restated.",
    ),
    DedupPairCase(
        name="dup-de-pet-role",
        category=DedupCategory.DUPLICATE,
        predecessor_locale="de",
        predecessor="Mitglieder dürfen maximal 3 Haustier-Rollen gleichzeitig haben.",
        candidate_locale="de",
        candidate="Man kann sich höchstens 3 Haustier-Rollen gleichzeitig zulegen.",
        rationale="Same cap, restated.",
    ),
    DedupPairCase(
        name="dup-fr-moderator",
        category=DedupCategory.DUPLICATE,
        predecessor_locale="fr",
        predecessor="Le modérateur en chef du serveur est Marc.",
        candidate_locale="fr",
        candidate="C'est Marc qui occupe le poste de modérateur en chef ici.",
        rationale="Same role holder, restated.",
    ),
    DedupPairCase(
        name="dup-fr-art-contest",
        category=DedupCategory.DUPLICATE,
        predecessor_locale="fr",
        predecessor="Les inscriptions au concours d'art se terminent le 30 août.",
        candidate_locale="fr",
        candidate="La date limite pour s'inscrire au concours d'art, c'est le 30 août.",
        rationale="Same deadline, restated.",
    ),
    DedupPairCase(
        name="dup-tr-upload-limit",
        category=DedupCategory.DUPLICATE,
        predecessor_locale="tr",
        predecessor="Sunucudaki dosya yükleme limiti 8 MB'dir.",
        candidate_locale="tr",
        candidate="Bu sunucuda en fazla 8 MB'lık dosya yükleyebilirsin.",
        rationale="Same limit, restated casually.",
    ),
    DedupPairCase(
        name="dup-tr-game-night",
        category=DedupCategory.DUPLICATE,
        predecessor_locale="tr",
        predecessor="Oyun gecesi her Cuma akşamı düzenlenir.",
        candidate_locale="tr",
        candidate="Her Cuma akşamı oyun gecesi yapıyoruz burada.",
        rationale="Same recurring schedule, restated.",
    ),
    DedupPairCase(
        name="dup-pl-onboarding",
        category=DedupCategory.DUPLICATE,
        predecessor_locale="pl",
        predecessor="Nowi członkowie otrzymują tymczasową rolę 'Gość' po dołączeniu.",
        candidate_locale="pl",
        candidate="Jak dołączysz, dostajesz najpierw tymczasową rolę 'Gość'.",
        rationale="Same onboarding step, restated.",
    ),
    DedupPairCase(
        name="dup-pl-giveaway",
        category=DedupCategory.DUPLICATE,
        predecessor_locale="pl",
        predecessor="Zwycięzcą konkursu na najlepszy fanart został użytkownik ArtKot.",
        candidate_locale="pl",
        candidate="To ArtKot wygrał konkurs na najlepszy fanart.",
        rationale="Same winner, restated.",
    ),
    DedupPairCase(
        name="dup-ja-tournament",
        category=DedupCategory.DUPLICATE,
        predecessor_locale="ja",
        predecessor="トーナメントは8月15日に開催される。",
        candidate_locale="ja",
        candidate="8月15日にトーナメントが行われます。",
        rationale="Same event date, restated with different word order and register.",
    ),
    DedupPairCase(
        name="dup-ja-milestone",
        category=DedupCategory.DUPLICATE,
        predecessor_locale="ja",
        predecessor="サーバーのフォロワー数が1000人を突破した。",
        candidate_locale="ja",
        candidate="サーバーのフォロワーがついに1000人を超えました。",
        rationale="Same milestone, restated.",
    ),
    DedupPairCase(
        name="dup-ko-voice-capacity",
        category=DedupCategory.DUPLICATE,
        predecessor_locale="ko",
        predecessor="음성 채널은 최대 20명까지 입장할 수 있다.",
        candidate_locale="ko",
        candidate="음성 채널에는 최대 20명까지만 들어올 수 있어요.",
        rationale="Same capacity, restated politely.",
    ),
    DedupPairCase(
        name="dup-ko-posting-rights",
        category=DedupCategory.DUPLICATE,
        predecessor_locale="ko",
        predecessor="이벤트 공지 채널은 현재 누구나 게시할 수 있다.",
        candidate_locale="ko",
        candidate="공지 채널은 지금 아무나 글 쓸 수 있어요.",
        rationale="Same posting-rights claim, restated.",
    ),
    # -- cross-locale translations of the same fact (7) --
    DedupPairCase(
        name="dup-cross-en-de-upload-limit",
        category=DedupCategory.DUPLICATE,
        predecessor_locale="en-US",
        predecessor="The upload limit in the #resources channel is 25 MB.",
        candidate_locale="de",
        candidate="Im Kanal #resources können Dateien bis maximal 25 MB hochgeladen werden.",
        rationale="Direct translation of the same fact -- the hardest duplicate shape, since "
        "no surface tokens are shared and only cross-lingual embedding quality can catch it.",
    ),
    DedupPairCase(
        name="dup-cross-es-en-voice-capacity",
        category=DedupCategory.DUPLICATE,
        predecessor_locale="es-ES",
        predecessor="El canal de voz principal admite hasta 30 personas.",
        candidate_locale="en-US",
        candidate="The main voice channel holds up to 30 people.",
        rationale="Same fact translated, no shared tokens.",
    ),
    DedupPairCase(
        name="dup-cross-fr-pt-moderator",
        category=DedupCategory.DUPLICATE,
        predecessor_locale="fr",
        predecessor="Le modérateur en chef du serveur est Marc.",
        candidate_locale="pt-BR",
        candidate="O moderador-chefe do servidor é o Marc.",
        rationale="Same fact translated; the name 'Marc' is the only shared token.",
    ),
    DedupPairCase(
        name="dup-cross-de-pl-level-gate",
        category=DedupCategory.DUPLICATE,
        predecessor_locale="de",
        predecessor="Mitglieder müssen Level 5 erreichen, um auf den Handelskanal zuzugreifen.",
        candidate_locale="pl",
        candidate="Aby uzyskać dostęp do kanału handlowego, trzeba osiągnąć poziom 5.",
        rationale="Same fact translated.",
    ),
    DedupPairCase(
        name="dup-cross-tr-ko-verification",
        category=DedupCategory.DUPLICATE,
        predecessor_locale="tr",
        predecessor="Yeni üyeler yazı yazmadan önce e-postalarını doğrulamak zorunda.",
        candidate_locale="ko",
        candidate="새로 온 멤버는 글을 쓰기 전에 이메일 인증을 해야 해요.",
        rationale="Same fact translated between two non-Latin-adjacent scripts.",
    ),
    DedupPairCase(
        name="dup-cross-ja-en-milestone",
        category=DedupCategory.DUPLICATE,
        predecessor_locale="ja",
        predecessor="サーバーのフォロワー数が1000人を突破した。",
        candidate_locale="en-US",
        candidate="The server just passed 1,000 followers.",
        rationale="Same fact translated.",
    ),
    DedupPairCase(
        name="dup-cross-pl-fr-giveaway",
        category=DedupCategory.DUPLICATE,
        predecessor_locale="pl",
        predecessor="Zwycięzcą konkursu na najlepszy fanart został użytkownik ArtKot.",
        candidate_locale="fr",
        candidate="C'est ArtKot qui a remporté le concours du meilleur fanart.",
        rationale="Same fact translated; 'ArtKot' is the only shared token.",
    ),
]

# =============================================================================
# 2. SUPERSESSION -- a genuine successor on the same specific detail. Should
#    mark. 18 same-locale + 7 cross-locale (3 imported verbatim from the
#    supersession work, since those are the cross-locale pairs the brief asks
#    to check this filter against; 4 newly written).
# =============================================================================
SUPERSESSION_CASES: list[DedupPairCase] = [
    DedupPairCase(
        name="super-en-us-upload-limit-increased",
        category=DedupCategory.SUPERSESSION,
        predecessor_locale="en-US",
        predecessor="The upload limit in the #resources channel is 25 MB.",
        candidate_locale="en-US",
        candidate="The upload limit in #resources has been increased to 50 MB.",
        rationale="Same channel/detail, value changed, explicit 'increased to'.",
    ),
    DedupPairCase(
        name="super-en-us-channel-renamed",
        category=DedupCategory.SUPERSESSION,
        predecessor_locale="en-US",
        predecessor="The announcements channel is called #news.",
        candidate_locale="en-US",
        candidate="The #news channel has been renamed to #server-news.",
        rationale="Explicit rename of the same channel; the old name no longer applies.",
    ),
    DedupPairCase(
        name="super-es-es-level-gate-raised",
        category=DedupCategory.SUPERSESSION,
        predecessor_locale="es-ES",
        predecessor="Hay que tener nivel 5 para entrar al canal de comercio.",
        candidate_locale="es-ES",
        candidate="Desde ahora se necesita nivel 8 para entrar al canal de comercio.",
        rationale="Same gate, new threshold, 'desde ahora' marks the change.",
    ),
    DedupPairCase(
        name="super-es-es-boost-tier-upgraded",
        category=DedupCategory.SUPERSESSION,
        predecessor_locale="es-ES",
        predecessor="El servidor alcanzó el nivel 2 de boost en julio.",
        candidate_locale="es-ES",
        candidate="El servidor subió al nivel 3 de boost esta semana.",
        rationale="Same metric, new value, progression language ('subió').",
    ),
    DedupPairCase(
        name="super-pt-br-tournament-rescheduled",
        category=DedupCategory.SUPERSESSION,
        predecessor_locale="pt-BR",
        predecessor="O torneio de verão começa no sábado às 18h.",
        candidate_locale="pt-BR",
        candidate="O torneio de verão, que era sábado, foi remarcado para domingo às 18h.",
        rationale="Explicit reschedule of the same event.",
    ),
    DedupPairCase(
        name="super-pt-br-pet-role-cap-raised",
        category=DedupCategory.SUPERSESSION,
        predecessor_locale="pt-BR",
        predecessor="Membros podem ter no máximo 3 cargos de bichinho ao mesmo tempo.",
        candidate_locale="pt-BR",
        candidate="A partir de agora, membros podem ter até 5 cargos de bichinho ao mesmo tempo.",
        rationale="Same cap, new number, 'a partir de agora' marks the change.",
    ),
    DedupPairCase(
        name="super-de-voice-capacity-increased",
        category=DedupCategory.SUPERSESSION,
        predecessor_locale="de",
        predecessor="Der Hauptsprachkanal fasst bis zu 30 Personen.",
        candidate_locale="de",
        candidate="Der Hauptsprachkanal wurde erweitert und fasst jetzt bis zu 50 Personen.",
        rationale="Same channel, capacity raised, 'wurde erweitert ... jetzt' marks the change.",
    ),
    DedupPairCase(
        name="super-de-moderator-handover",
        category=DedupCategory.SUPERSESSION,
        predecessor_locale="de",
        predecessor="Der Cheftechnikmoderator des Servers ist Paul.",
        candidate_locale="de",
        candidate="Paul hat sein Amt als Cheftechnikmoderator abgegeben; jetzt ist Nina zuständig.",
        rationale="Explicit handover of the same role.",
    ),
    DedupPairCase(
        name="super-fr-verification-retired",
        category=DedupCategory.SUPERSESSION,
        predecessor_locale="fr",
        predecessor="Les nouveaux membres doivent vérifier leur email avant de pouvoir écrire.",
        candidate_locale="fr",
        candidate="La vérification d'email n'est plus obligatoire pour les nouveaux membres.",
        rationale="Retirement of the same rule -- CLAUDE.md's own 'no longer true' status change.",
    ),
    DedupPairCase(
        name="super-fr-art-deadline-moved",
        category=DedupCategory.SUPERSESSION,
        predecessor_locale="fr",
        predecessor="Les inscriptions au concours d'art se terminent le 30 août.",
        candidate_locale="fr",
        candidate="La date limite du concours d'art a été repoussée au 15 septembre.",
        rationale="Same deadline, explicit 'a été repoussée'.",
    ),
    DedupPairCase(
        name="super-tr-upload-limit-changed",
        category=DedupCategory.SUPERSESSION,
        predecessor_locale="tr",
        predecessor="Sunucudaki dosya yükleme limiti 8 MB'dir.",
        candidate_locale="tr",
        candidate="Dosya yükleme limiti artık 20 MB'a çıkarıldı.",
        rationale="Same limit, 'artık ... çıkarıldı' marks the change.",
    ),
    DedupPairCase(
        name="super-tr-channel-closed",
        category=DedupCategory.SUPERSESSION,
        predecessor_locale="tr",
        predecessor="Öneri kanalı tüm üyelere açık.",
        candidate_locale="tr",
        candidate="Öneri kanalı kalıcı olarak kapatıldı.",
        rationale="Status flip on the same channel, explicit closure wording.",
    ),
    DedupPairCase(
        name="super-pl-game-night-day-changed",
        category=DedupCategory.SUPERSESSION,
        predecessor_locale="pl",
        predecessor="Wieczór gier odbywa się w każdy piątek.",
        candidate_locale="pl",
        candidate="Od tego tygodnia wieczór gier przenosimy na soboty.",
        rationale="Same recurring event, day changed, 'od tego tygodnia ... przenosimy' marks it.",
    ),
    DedupPairCase(
        name="super-pl-level-gate-lowered",
        category=DedupCategory.SUPERSESSION,
        predecessor_locale="pl",
        predecessor="Aby dołączyć do kanału VIP, trzeba mieć poziom 20.",
        candidate_locale="pl",
        candidate="Próg dołączenia do kanału VIP został obniżony do poziomu 10.",
        rationale="Same gate, value lowered, 'został obniżony' marks the change.",
    ),
    DedupPairCase(
        name="super-ja-tournament-moved",
        category=DedupCategory.SUPERSESSION,
        predecessor_locale="ja",
        predecessor="トーナメントは8月15日に開催される。",
        candidate_locale="ja",
        candidate="トーナメントは8月15日から8月22日に変更されました。",
        rationale="Same event, explicit date-change wording.",
    ),
    DedupPairCase(
        name="super-ja-channel-renamed",
        category=DedupCategory.SUPERSESSION,
        predecessor_locale="ja",
        predecessor="アナウンスチャンネルの名前は#お知らせです。",
        candidate_locale="ja",
        candidate="#お知らせチャンネルは#公式アナウンスに名前が変更されました。",
        rationale="Explicit rename of the same channel.",
    ),
    DedupPairCase(
        name="super-ko-voice-capacity-increased",
        category=DedupCategory.SUPERSESSION,
        predecessor_locale="ko",
        predecessor="음성 채널은 최대 20명까지 입장할 수 있다.",
        candidate_locale="ko",
        candidate="음성 채널 정원이 30명으로 늘어났습니다.",
        rationale="Same channel/detail, value increased, '늘어났습니다' marks the change.",
    ),
    DedupPairCase(
        name="super-ko-pet-role-changed",
        category=DedupCategory.SUPERSESSION,
        predecessor_locale="ko",
        predecessor="회원은 반려동물 역할을 최대 3개까지 가질 수 있다.",
        candidate_locale="ko",
        candidate="이제부터 반려동물 역할은 최대 5개까지 가질 수 있습니다.",
        rationale="Same cap, new value, '이제부터' marks the change.",
    ),
    # -- cross-locale (7): 3 imported verbatim from the supersession work --
    _imported(
        "supersession-level-requirement-raised-cross",
        category=DedupCategory.SUPERSESSION,
        name="super-cross-imported-level-requirement-raised",
        rationale="Imported verbatim from the supersession bake-off/re-verification "
        "(reports/phase-3a-3.txt Section 5): the cross-locale pair a downstream MODEL "
        "call was measured on. Checking whether the much cheaper embedding filter marks "
        "it at all is the pipeline question that call's evidence never answered.",
    ),
    _imported(
        "supersession-tournament-rescheduled-cross",
        category=DedupCategory.SUPERSESSION,
        name="super-cross-imported-tournament-rescheduled",
        rationale="Imported verbatim -- a second cross-locale supersession the bake-off "
        "corpus already used, for the same reason.",
    ),
    _imported(
        "supersession-upload-limit-increased-cross",
        category=DedupCategory.SUPERSESSION,
        name="super-cross-imported-upload-limit-increased",
        rationale="Imported verbatim -- a third cross-locale supersession, Turkish "
        "predecessor to Polish candidate.",
    ),
    DedupPairCase(
        name="super-cross-es-fr-boost-tier",
        category=DedupCategory.SUPERSESSION,
        predecessor_locale="es-ES",
        predecessor="El servidor alcanzó el nivel 2 de boost en julio.",
        candidate_locale="fr",
        candidate="Le serveur est passé au niveau de boost 3 ce mois-ci.",
        rationale="Same metric, new value, cross-locale.",
    ),
    DedupPairCase(
        name="super-cross-pt-ko-moderator-handover",
        category=DedupCategory.SUPERSESSION,
        predecessor_locale="pt-BR",
        predecessor="O moderador-chefe do servidor é o Marcos.",
        candidate_locale="ko",
        candidate="마르코스가 수석 모더레이터 자리에서 물러났고, 이제는 지연님이 맡고 있어요.",
        rationale="Explicit handover of the same role, cross-locale.",
    ),
    DedupPairCase(
        name="super-cross-de-tr-channel-closed",
        category=DedupCategory.SUPERSESSION,
        predecessor_locale="de",
        predecessor="Der Vorschläge-Kanal ist für alle Mitglieder geöffnet.",
        candidate_locale="tr",
        candidate="Öneri kanalı artık kalıcı olarak kapatıldı.",
        rationale="Status flip on the same channel, cross-locale.",
    ),
    DedupPairCase(
        name="super-cross-en-ja-pet-role-raised",
        category=DedupCategory.SUPERSESSION,
        predecessor_locale="en-US",
        predecessor="Members can have at most 3 pet roles at the same time.",
        candidate_locale="ja",
        candidate="会員が同時に持てるペットロールは、今では最大5個までになりました。",
        rationale="Same cap, new value, cross-locale.",
    ),
]

# =============================================================================
# 3. CONTRADICTION -- same specific detail, different values, no change
#    wording. Should mark. 18 same-locale + 7 cross-locale (3 imported
#    verbatim, 4 newly written).
# =============================================================================
CONTRADICTION_CASES: list[DedupPairCase] = [
    DedupPairCase(
        name="contra-en-us-upload-limit",
        category=DedupCategory.CONTRADICTION,
        predecessor_locale="en-US",
        predecessor="The upload limit in the #resources channel is 25 MB.",
        candidate_locale="en-US",
        candidate="The upload limit in the #resources channel is 10 MB.",
        rationale="Same channel/detail, different values, no change wording -- genuinely "
        "unclear which is current. Matched against indep-imported-upload-limit-channel below, "
        "which shares this predecessor but changes the CHANNEL instead of the number.",
    ),
    DedupPairCase(
        name="contra-en-us-founding-year",
        category=DedupCategory.CONTRADICTION,
        predecessor_locale="en-US",
        predecessor="The server was created in 2019.",
        candidate_locale="en-US",
        candidate="The server was created in 2021.",
        rationale="A fact that should not legitimately change, stated two ways, no "
        "correction language -- genuinely unclear which is accurate.",
    ),
    DedupPairCase(
        name="contra-es-es-voice-capacity",
        category=DedupCategory.CONTRADICTION,
        predecessor_locale="es-ES",
        predecessor="El canal de voz principal admite hasta 30 personas.",
        candidate_locale="es-ES",
        candidate="El canal de voz principal admite hasta 15 personas.",
        rationale="Same channel/detail, different values, no change wording.",
    ),
    DedupPairCase(
        name="contra-es-es-giveaway-winner",
        category=DedupCategory.CONTRADICTION,
        predecessor_locale="es-ES",
        predecessor="El ganador del sorteo de la semana fue @Laura.",
        candidate_locale="es-ES",
        candidate="El ganador del sorteo de la semana fue @Carlos.",
        rationale="Same contest, two different winners named, no correction wording.",
    ),
    DedupPairCase(
        name="contra-pt-br-level-gate",
        category=DedupCategory.CONTRADICTION,
        predecessor_locale="pt-BR",
        predecessor="É preciso ter nível 10 para acessar o canal VIP.",
        candidate_locale="pt-BR",
        candidate="É preciso ter nível 15 para acessar o canal VIP.",
        rationale="Same gate, different thresholds, no change wording.",
    ),
    DedupPairCase(
        name="contra-pt-br-boost-tier",
        category=DedupCategory.CONTRADICTION,
        predecessor_locale="pt-BR",
        predecessor="O servidor está atualmente no nível 3 de boost.",
        candidate_locale="pt-BR",
        candidate="O servidor está atualmente no nível 2 de boost.",
        rationale="Both claim the CURRENT tier, different values, no timestamp cue.",
    ),
    DedupPairCase(
        name="contra-de-posting-rights",
        category=DedupCategory.CONTRADICTION,
        predecessor_locale="de",
        predecessor="Im Ankündigungskanal darf aktuell jeder posten.",
        candidate_locale="de",
        candidate="Im Ankündigungskanal dürfen nur Moderatoren posten.",
        rationale="Same channel, mutually exclusive claims about who may post, no change wording.",
    ),
    DedupPairCase(
        name="contra-de-upload-limit",
        category=DedupCategory.CONTRADICTION,
        predecessor_locale="de",
        predecessor="Das Upload-Limit im Kanal #medien beträgt 20 MB.",
        candidate_locale="de",
        candidate="Das Upload-Limit im Kanal #medien beträgt 40 MB.",
        rationale="Same channel/detail, different numbers, no change wording.",
    ),
    DedupPairCase(
        name="contra-fr-voice-capacity",
        category=DedupCategory.CONTRADICTION,
        predecessor_locale="fr",
        predecessor="Le salon vocal des débutants peut accueillir jusqu'à 10 personnes.",
        candidate_locale="fr",
        candidate="Le salon vocal des débutants peut accueillir jusqu'à 20 personnes.",
        rationale="Same channel/detail, different values, no change wording.",
    ),
    DedupPairCase(
        name="contra-fr-moderator",
        category=DedupCategory.CONTRADICTION,
        predecessor_locale="fr",
        predecessor="Le modérateur en chef du serveur est Marc.",
        candidate_locale="fr",
        candidate="Le modérateur en chef du serveur est Julie.",
        rationale="Same role, two different names, no handover language.",
    ),
    DedupPairCase(
        name="contra-tr-level-gate",
        category=DedupCategory.CONTRADICTION,
        predecessor_locale="tr",
        predecessor="VIP kanalına girmek için seviye 20 gerekiyor.",
        candidate_locale="tr",
        candidate="VIP kanalına girmek için seviye 12 gerekiyor.",
        rationale="Same gate, different thresholds, no change wording.",
    ),
    DedupPairCase(
        name="contra-tr-founding-year",
        category=DedupCategory.CONTRADICTION,
        predecessor_locale="tr",
        predecessor="Bu sunucu 2018 yılında kuruldu.",
        candidate_locale="tr",
        candidate="Bu sunucu 2020 yılında kuruldu.",
        rationale="Conflicting founding years, no correction wording.",
    ),
    DedupPairCase(
        name="contra-pl-pet-role",
        category=DedupCategory.CONTRADICTION,
        predecessor_locale="pl",
        predecessor="Członkowie mogą mieć maksymalnie 4 role zwierzątek naraz.",
        candidate_locale="pl",
        candidate="Członkowie mogą mieć maksymalnie 6 ról zwierzątek naraz.",
        rationale="Same cap, different numbers, no change wording.",
    ),
    DedupPairCase(
        name="contra-pl-upload-limit",
        category=DedupCategory.CONTRADICTION,
        predecessor_locale="pl",
        predecessor="Limit przesyłania plików na kanale #media to 15 MB.",
        candidate_locale="pl",
        candidate="Limit przesyłania plików na kanale #media to 30 MB.",
        rationale="Same channel/detail, different values, no change wording.",
    ),
    DedupPairCase(
        name="contra-ja-voice-capacity",
        category=DedupCategory.CONTRADICTION,
        predecessor_locale="ja",
        predecessor="メイン音声チャンネルの定員は30人です。",
        candidate_locale="ja",
        candidate="メイン音声チャンネルの定員は15人です。",
        rationale="Same channel/detail, conflicting numbers, no change wording.",
    ),
    DedupPairCase(
        name="contra-ja-boost-tier",
        category=DedupCategory.CONTRADICTION,
        predecessor_locale="ja",
        predecessor="サーバーは現在ブーストレベル3です。",
        candidate_locale="ja",
        candidate="サーバーは現在ブーストレベル2です。",
        rationale="Both claim the current tier, different values, no timestamp cue.",
    ),
    DedupPairCase(
        name="contra-ko-level-gate",
        category=DedupCategory.CONTRADICTION,
        predecessor_locale="ko",
        predecessor="VIP 음성 채널에 참여하려면 레벨 20이 필요하다.",
        candidate_locale="ko",
        candidate="VIP 음성 채널에 참여하려면 레벨 12가 필요하다.",
        rationale="Same gate, different thresholds, no change wording.",
    ),
    DedupPairCase(
        name="contra-ko-giveaway-winner",
        category=DedupCategory.CONTRADICTION,
        predecessor_locale="ko",
        predecessor="이번 주 이벤트 우승자는 별빛님입니다.",
        candidate_locale="ko",
        candidate="이번 주 이벤트 우승자는 달빛님입니다.",
        rationale="Same contest, two different winners named, no correction wording.",
    ),
    # -- cross-locale (7): 3 imported verbatim, 4 newly written --
    _imported(
        "contradiction-announcement-posting-rights-cross",
        category=DedupCategory.CONTRADICTION,
        name="contra-cross-imported-posting-rights",
        rationale="Imported verbatim from the supersession bake-off (Japanese predecessor, "
        "Turkish candidate) -- one of the cross-locale pairs the downstream judge was "
        "measured on.",
    ),
    _imported(
        "contradiction-voice-capacity-cross",
        category=DedupCategory.CONTRADICTION,
        name="contra-cross-imported-voice-capacity",
        rationale="Imported verbatim -- French predecessor, English candidate.",
    ),
    _imported(
        "contradiction-vip-level-requirement-cross",
        category=DedupCategory.CONTRADICTION,
        name="contra-cross-imported-vip-level-requirement",
        rationale="Imported verbatim -- Polish predecessor, Korean candidate.",
    ),
    DedupPairCase(
        name="contra-cross-en-es-upload-limit",
        category=DedupCategory.CONTRADICTION,
        predecessor_locale="en-US",
        predecessor="The upload limit in the #media channel is 20 MB.",
        candidate_locale="es-ES",
        candidate="El límite de subida en el canal #media es de 40 MB.",
        rationale="Same channel/detail, conflicting values, no change wording, cross-locale.",
    ),
    DedupPairCase(
        name="contra-cross-de-pt-boost-tier",
        category=DedupCategory.CONTRADICTION,
        predecessor_locale="de",
        predecessor="Der Server ist aktuell auf Boost-Level 3.",
        candidate_locale="pt-BR",
        candidate="O servidor está atualmente no nível de boost 2.",
        rationale="Both claim the current tier, conflicting values, cross-locale.",
    ),
    DedupPairCase(
        name="contra-cross-fr-ja-moderator",
        category=DedupCategory.CONTRADICTION,
        predecessor_locale="fr",
        predecessor="Le modérateur en chef du serveur est Marc.",
        candidate_locale="ja",
        candidate="サーバーのチーフモデレーターはユキさんです。",
        rationale="Same role, two different names, no handover wording, cross-locale.",
    ),
    DedupPairCase(
        name="contra-cross-tr-pl-founding-year",
        category=DedupCategory.CONTRADICTION,
        predecessor_locale="tr",
        predecessor="Bu sunucu 2018 yılında kuruldu.",
        candidate_locale="pl",
        candidate="Ten serwer powstał w 2020 roku.",
        rationale="Conflicting founding years, cross-locale.",
    ),
]

# =============================================================================
# 4a. INDEPENDENT, RELATED -- similar wording/template, different subject.
#    Should NOT mark. This is the harder half of "should not mark": the
#    embedding trap CLAUDE.md and the supersession bake-off both name.
#    2 imported (the named attack cases) + 13 newly written.
# =============================================================================
INDEPENDENT_RELATED_CASES: list[DedupPairCase] = [
    _imported(
        "independent-upload-limit-different-channel",
        category=DedupCategory.INDEPENDENT_RELATED,
        name="indep-imported-upload-limit-channel",
        rationale="Imported verbatim -- the named Phase 3a-3 attack case. Near-identical "
        "wording to contra-en-us-upload-limit above, but the changed word is the CHANNEL, "
        "not the number: two separate rules, coincidentally similar enough in phrasing to "
        "risk clearing an embedding threshold.",
    ),
    _imported(
        "independent-new-member-rule-different-topic-cross",
        category=DedupCategory.INDEPENDENT_RELATED,
        name="indep-imported-new-member-rule-cross",
        rationale="Imported verbatim -- the named cross-locale Phase 3a-3 attack case "
        "(English predecessor, German candidate). Shares the 'new members must X before Y' "
        "template; X and Y name unrelated actions.",
    ),
    DedupPairCase(
        name="indep-en-us-voice-capacity-different-channel",
        category=DedupCategory.INDEPENDENT_RELATED,
        predecessor_locale="en-US",
        predecessor="The main voice channel can hold up to 30 people.",
        candidate_locale="en-US",
        candidate="The gaming voice channel can hold up to 15 people.",
        rationale="Same 'voice channel + capacity number' template, different channels -- two "
        "separate rules, not a conflict.",
    ),
    DedupPairCase(
        name="indep-es-es-level-gate-different-channel",
        category=DedupCategory.INDEPENDENT_RELATED,
        predecessor_locale="es-ES",
        predecessor="Hay que tener nivel 10 para entrar al canal VIP.",
        candidate_locale="es-ES",
        candidate="Hay que tener nivel 10 para entrar al canal de eventos.",
        rationale="Identical level number and template, but gates two different channels.",
    ),
    DedupPairCase(
        name="indep-pt-br-contest-deadline-different-contest",
        category=DedupCategory.INDEPENDENT_RELATED,
        predecessor_locale="pt-BR",
        predecessor="As inscrições do concurso de fanart terminam dia 30.",
        candidate_locale="pt-BR",
        candidate="As inscrições do concurso de cosplay terminam dia 30.",
        rationale="Same deadline date and template, two entirely different contests.",
    ),
    DedupPairCase(
        name="indep-de-cheating-different-competition",
        category=DedupCategory.INDEPENDENT_RELATED,
        predecessor_locale="de",
        predecessor="Betrug wird beim Schach-Turnier nicht toleriert und führt zur Disqualifikation.",
        candidate_locale="de",
        candidate="Betrug wird beim Quiz-Abend nicht toleriert und führt zur Disqualifikation.",
        rationale="Identical template sentence, two unrelated events.",
    ),
    DedupPairCase(
        name="indep-fr-tournament-different-game",
        category=DedupCategory.INDEPENDENT_RELATED,
        predecessor_locale="fr",
        predecessor="Le tournoi de Valorant a lieu ce week-end dans le salon #fps.",
        candidate_locale="fr",
        candidate="Le tournoi de Mario Kart a lieu ce week-end dans le salon #switch.",
        rationale="Two distinct tournaments sharing only the weekend and sentence shape.",
    ),
    DedupPairCase(
        name="indep-tr-daily-limit-different-channel",
        category=DedupCategory.INDEPENDENT_RELATED,
        predecessor_locale="tr",
        predecessor="Reklam kanalında günde en fazla 1 mesaj paylaşılabilir.",
        candidate_locale="tr",
        candidate="Yardım kanalında günde en fazla 1 mesaj paylaşılabilir.",
        rationale="Same 'once a day' template applied to two unrelated channels.",
    ),
    DedupPairCase(
        name="indep-pl-milestone-different-metric",
        category=DedupCategory.INDEPENDENT_RELATED,
        predecessor_locale="pl",
        predecessor="Serwer przekroczył 1000 obserwujących na Twitterze.",
        candidate_locale="pl",
        candidate="Serwer przekroczył 1000 subskrybentów na YouTube.",
        rationale="Same round number and 'surpassed X' template, different platform/metric.",
    ),
    DedupPairCase(
        name="indep-ja-posting-rights-different-channel",
        category=DedupCategory.INDEPENDENT_RELATED,
        predecessor_locale="ja",
        predecessor="#雑談チャンネルは現在誰でも投稿できる。",
        candidate_locale="ja",
        candidate="#お知らせチャンネルは現在誰でも投稿できる。",
        rationale="Same 'anyone can post' claim, two different channels with different purposes.",
    ),
    DedupPairCase(
        name="indep-ko-role-cap-different-role",
        category=DedupCategory.INDEPENDENT_RELATED,
        predecessor_locale="ko",
        predecessor="반려동물 역할은 최대 3개까지 가질 수 있다.",
        candidate_locale="ko",
        candidate="테마 역할은 최대 3개까지 가질 수 있다.",
        rationale="Identical cap number and template, two unrelated custom-role categories.",
    ),
    DedupPairCase(
        name="indep-cross-en-de-posting-rights-different-channel",
        category=DedupCategory.INDEPENDENT_RELATED,
        predecessor_locale="en-US",
        predecessor="Only moderators can post in the #announcements channel.",
        candidate_locale="de",
        candidate="Nur Moderatoren dürfen im Kanal #ankündigungen-events posten.",
        rationale="Same 'moderators only' template, two distinct announcement channels, "
        "cross-locale.",
    ),
    DedupPairCase(
        name="indep-cross-es-fr-contest-deadline",
        category=DedupCategory.INDEPENDENT_RELATED,
        predecessor_locale="es-ES",
        predecessor="Las inscripciones al concurso de música terminan el 15 de septiembre.",
        candidate_locale="fr",
        candidate="Les inscriptions au concours de cuisine se terminent le 15 septembre.",
        rationale="Same deadline and template, two unrelated contests, cross-locale.",
    ),
    DedupPairCase(
        name="indep-cross-pt-ko-upload-limit-different-channel",
        category=DedupCategory.INDEPENDENT_RELATED,
        predecessor_locale="pt-BR",
        predecessor="O limite de upload no canal #memes é de 10 MB.",
        candidate_locale="ko",
        candidate="#스크린샷 채널의 업로드 제한은 10MB입니다.",
        rationale="Identical number and template, two different channels, cross-locale.",
    ),
    DedupPairCase(
        name="indep-cross-tr-pl-level-gate-different-channel",
        category=DedupCategory.INDEPENDENT_RELATED,
        predecessor_locale="tr",
        predecessor="Sanat kanalına girmek için seviye 7 gerekiyor.",
        candidate_locale="pl",
        candidate="Aby wejść na kanał z muzyką, trzeba mieć poziom 7.",
        rationale="Identical level number and template, two unrelated channels, cross-locale.",
    ),
]

# =============================================================================
# 4b. UNRELATED -- no shared subject, template or wording at all. Should NOT
#    mark. The easy half of "should not mark", included so the sweep can show
#    this filter comfortably clears the ordinary case, not only the hard one.
# =============================================================================
UNRELATED_CASES: list[DedupPairCase] = [
    DedupPairCase(
        name="unrel-en-us-upload-vs-game-night",
        category=DedupCategory.UNRELATED,
        predecessor_locale="en-US",
        predecessor="The upload limit in the #resources channel is 25 MB.",
        candidate_locale="en-US",
        candidate="Game night happens every Friday evening.",
        rationale="No shared subject, channel, or template at all.",
    ),
    DedupPairCase(
        name="unrel-es-es-giveaway-vs-voice-capacity",
        category=DedupCategory.UNRELATED,
        predecessor_locale="es-ES",
        predecessor="El ganador del sorteo semanal fue @Laura.",
        candidate_locale="es-ES",
        candidate="El canal de voz principal admite hasta 30 personas.",
        rationale="Unrelated facts.",
    ),
    DedupPairCase(
        name="unrel-pt-br-founding-vs-onboarding",
        category=DedupCategory.UNRELATED,
        predecessor_locale="pt-BR",
        predecessor="O servidor foi criado em 2019.",
        candidate_locale="pt-BR",
        candidate="Os novos membros recebem o cargo temporário 'Visitante'.",
        rationale="Unrelated facts.",
    ),
    DedupPairCase(
        name="unrel-de-moderator-vs-art-contest",
        category=DedupCategory.UNRELATED,
        predecessor_locale="de",
        predecessor="Der Cheftechnikmoderator des Servers ist Paul.",
        candidate_locale="de",
        candidate="Der Kunstwettbewerb endet am 30. August.",
        rationale="Unrelated facts.",
    ),
    DedupPairCase(
        name="unrel-fr-tournament-vs-verification",
        category=DedupCategory.UNRELATED,
        predecessor_locale="fr",
        predecessor="Le tournoi d'échecs a lieu ce week-end.",
        candidate_locale="fr",
        candidate="Il faut vérifier son email avant de pouvoir écrire.",
        rationale="Unrelated facts.",
    ),
    DedupPairCase(
        name="unrel-tr-founding-vs-game-night",
        category=DedupCategory.UNRELATED,
        predecessor_locale="tr",
        predecessor="Sunucu 2018 yılında kuruldu.",
        candidate_locale="tr",
        candidate="Oyun gecesi her Cuma akşamı düzenlenir.",
        rationale="Unrelated facts.",
    ),
    DedupPairCase(
        name="unrel-pl-boost-vs-giveaway",
        category=DedupCategory.UNRELATED,
        predecessor_locale="pl",
        predecessor="Serwer osiągnął 2. poziom boost.",
        candidate_locale="pl",
        candidate="Zwycięzcą konkursu fanart został ArtKot.",
        rationale="Unrelated facts.",
    ),
    DedupPairCase(
        name="unrel-ja-tournament-vs-voice-capacity",
        category=DedupCategory.UNRELATED,
        predecessor_locale="ja",
        predecessor="トーナメントは8月15日に開催される。",
        candidate_locale="ja",
        candidate="メイン音声チャンネルの定員は30人です。",
        rationale="Unrelated facts.",
    ),
    DedupPairCase(
        name="unrel-ko-boost-vs-verification",
        category=DedupCategory.UNRELATED,
        predecessor_locale="ko",
        predecessor="서버는 현재 부스트 3단계이다.",
        candidate_locale="ko",
        candidate="새로 온 멤버는 이메일 인증을 해야 해요.",
        rationale="Unrelated facts.",
    ),
    DedupPairCase(
        name="unrel-en-us-voice-capacity-vs-giveaway",
        category=DedupCategory.UNRELATED,
        predecessor_locale="en-US",
        predecessor="The main voice channel can hold up to 30 people.",
        candidate_locale="en-US",
        candidate="The winner of last week's giveaway was @Sam.",
        rationale="Unrelated facts.",
    ),
    DedupPairCase(
        name="unrel-cross-es-de-boost-vs-art-contest",
        category=DedupCategory.UNRELATED,
        predecessor_locale="es-ES",
        predecessor="El servidor alcanzó el nivel 2 de boost en julio.",
        candidate_locale="de",
        candidate="Der Kunstwettbewerb endet am 30. August.",
        rationale="Unrelated facts, cross-locale.",
    ),
    DedupPairCase(
        name="unrel-cross-pt-fr-channel-status-vs-tournament",
        category=DedupCategory.UNRELATED,
        predecessor_locale="pt-BR",
        predecessor="O canal #sugestões está aberto para todos.",
        candidate_locale="fr",
        candidate="Le tournoi de belote a lieu ce week-end.",
        rationale="Unrelated facts, cross-locale.",
    ),
    DedupPairCase(
        name="unrel-cross-tr-ja-upload-vs-milestone",
        category=DedupCategory.UNRELATED,
        predecessor_locale="tr",
        predecessor="Dosya yükleme limiti 8 MB'dir.",
        candidate_locale="ja",
        candidate="サーバーのフォロワー数が1000人を突破した。",
        rationale="Unrelated facts, cross-locale.",
    ),
    DedupPairCase(
        name="unrel-cross-pl-ko-game-night-vs-level-gate",
        category=DedupCategory.UNRELATED,
        predecessor_locale="pl",
        predecessor="Wieczór gier odbywa się w każdy piątek.",
        candidate_locale="ko",
        candidate="VIP 음성 채널에 참여하려면 레벨 20이 필요하다.",
        rationale="Unrelated facts, cross-locale.",
    ),
    DedupPairCase(
        name="unrel-cross-de-en-level-gate-vs-tournament",
        category=DedupCategory.UNRELATED,
        predecessor_locale="de",
        predecessor="Mitglieder müssen Level 5 erreichen, um auf den Handelskanal zuzugreifen.",
        candidate_locale="en-US",
        candidate="The winter tournament starts Saturday at 6 PM in the #events channel.",
        rationale="Unrelated facts, cross-locale.",
    ),
]

ALL_CASES: list[DedupPairCase] = (
    DUPLICATE_CASES
    + SUPERSESSION_CASES
    + CONTRADICTION_CASES
    + INDEPENDENT_RELATED_CASES
    + UNRELATED_CASES
)

# The two named Phase 3a-3 attack cases, called out on their own so the sweep
# script and the report can check them individually rather than only as part
# of the INDEPENDENT_RELATED aggregate.
NAMED_ATTACK_CASE_NAMES: tuple[str, ...] = (
    "indep-imported-upload-limit-channel",
    "indep-imported-new-member-rule-cross",
)


def cases_by_category(category: DedupCategory) -> list[DedupPairCase]:
    return [case for case in ALL_CASES if case.category is category]
