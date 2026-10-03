"""A deterministic, invented server of any size, with labelled questions. Nothing here is real.

Built to answer one question the 11-fact diagnosis could not: does hybrid
retrieval stay precise when a guild has hundreds or thousands of facts, many of
them about similar things?

**Topics.** Every fact belongs to exactly one topic -- one invented event,
course or meetup -- and states one detail about it (when, where to sign up, who
runs it, the prize, ...). A topic has one to three facts. Topic names are built
from a modifier and a head noun per language, so the corpus is full of
deliberate near-collisions: "Sommerturnier" next to "Winterturnier" and
"Sommerkurs", "galaxy raid" next to "galaxy jam" and "pixel raid", plus a
handful of hand-made ones ("Serverwartung", "Serverregeln", "Serverumzug";
"Mentoriat" next to "Mentorprogramm").

**Questions.** A positive question asks about one topic in one register
(keyword, inflected, split compound, formal, colloquial, vague, typo) and its
relevant facts are exactly that topic's facts. Negative questions select
nothing in a perfect world: greetings and off-topic questions in every
language, sound-alikes of real topic words, and -- the hardest -- questions
about topics that were deliberately held out of the corpus although their
modifier and their head noun both occur in it ("Herbstturnier" when only
"Sommerturnier" and "Herbstkurs" exist).

**Deterministic.** The same seed and size give the same facts and questions on
every machine; a smaller corpus is a prefix of a larger one.

Imports nothing from aura.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass
from typing import Final

from ask_retrieval_eval.cases import EvalCase, InventedFact

DEFAULT_SEED: Final = 20261002

# Every modifier x noun pair whose combined index is divisible by this is held
# out of the corpus and used for template-trap negatives instead.
_HOLD_OUT_EVERY: Final = 7

_DAYS_DE: Final = ("Montag", "Dienstag", "Mittwoch", "Donnerstag", "Freitag", "Samstag", "Sonntag")
_DAYS_EN: Final = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
_DAYS_ES: Final = ("lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo")
_DAYS_FR: Final = ("lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche")
_DAYS_PT: Final = (
    "segunda-feira",
    "terça-feira",
    "quarta-feira",
    "quinta-feira",
    "sexta-feira",
    "sábado",
    "domingo",
)
_DAYS_TR: Final = ("pazartesi", "salı", "çarşamba", "perşembe", "cuma", "cumartesi", "pazar")
_DAYS_PL: Final = ("poniedziałek", "wtorek", "środę", "czwartek", "piątek", "sobotę", "niedzielę")
_DAYS_JA: Final = ("月曜日", "火曜日", "水曜日", "木曜日", "金曜日", "土曜日", "日曜日")
_DAYS_KO: Final = ("월요일", "화요일", "수요일", "목요일", "금요일", "토요일", "일요일")
_MONTHS_DE: Final = ("Januar", "März", "Mai", "Juli", "August", "Oktober", "Dezember")
_MONTHS_EN: Final = ("January", "March", "May", "July", "August", "October", "December")
_CHANNELS: Final = (
    "anmeldung",
    "events",
    "turniere",
    "ankuendigungen",
    "signups",
    "lobby",
    "treffpunkt",
    "orga",
    "schedule",
    "info-board",
)
_ROLES: Final = ("Eventteam", "Moderation", "Orga", "Helfer", "Host", "Crew", "Staff", "Kurator")
_PRIZES_DE: Final = (
    "einen Monat Nitro",
    "eine eigene Farbe",
    "einen Gutschein",
    "die Rolle Champion",
    "ein Steam-Spiel",
)
_PRIZES_EN: Final = (
    "one month of Nitro",
    "a custom colour",
    "a gift card",
    "the Champion role",
    "a Steam game",
)


@dataclass(frozen=True, slots=True)
class _GermanNoun:
    """A German head noun: the word, its plural, and its articles by case."""

    word: str
    plural: str
    nominative: str
    dative: str
    accusative: str


_DE_MODIFIERS: Final = (
    "Sommer",
    "Winter",
    "Frühlings",
    "Herbst",
    "Nacht",
    "Wochenend",
    "Monats",
    "Anfänger",
    "Profi",
    "Team",
    "Solo",
    "Duo",
    "Community",
    "Minecraft",
    "Valorant",
    "Schach",
    "Quiz",
    "Musik",
    "Film",
    "Kunst",
    "Foto",
    "Lese",
    "Koch",
    "Sprach",
    "Mathe",
    "Physik",
    "Programmier",
    "Karaoke",
    "Retro",
    "Speedrun",
    "Zeichen",
    "Gitarren",
    "Yoga",
    "Lauf",
    "Garten",
    "Brettspiel",
    "Podcast",
    "Rätsel",
    "Tanz",
    "Kletter",
)
_DE_NOUNS: Final = (
    _GermanNoun("Turnier", "Turniere", "das", "dem", "das"),
    _GermanNoun("Abend", "Abende", "der", "dem", "den"),
    _GermanNoun("Runde", "Runden", "die", "der", "die"),
    _GermanNoun("Treffen", "Treffen", "das", "dem", "das"),
    _GermanNoun("Wettbewerb", "Wettbewerbe", "der", "dem", "den"),
    _GermanNoun("Kurs", "Kurse", "der", "dem", "den"),
    _GermanNoun("Sprechstunde", "Sprechstunden", "die", "der", "die"),
    _GermanNoun("Workshop", "Workshops", "der", "dem", "den"),
    _GermanNoun("Liga", "Ligen", "die", "der", "die"),
    _GermanNoun("Umfrage", "Umfragen", "die", "der", "die"),
    _GermanNoun("Sitzung", "Sitzungen", "die", "der", "die"),
    _GermanNoun("Mentoriat", "Mentoriate", "das", "dem", "das"),
    _GermanNoun("Ausflug", "Ausflüge", "der", "dem", "den"),
)
_DE_FACTS: Final[tuple[str, ...]] = (
    "{Nom} {E} findet jeden {day} um {hour} Uhr statt.",
    "{Nom} {E} beginnt am {date}. {month_de} im Kanal #{channel}.",
    "Anmeldungen für {acc} {E} laufen über den Kanal #{channel}.",
    "{Nom} {E} wird von der Rolle @{role} organisiert.",
    "Bei {dat} {E} gibt es {prize_de} zu gewinnen.",
    "{Nom} {E} fällt im {month_de} aus.",
    "{Nom} {E} dauert etwa {n} Stunden.",
    "Für {acc} {E} braucht man die Rolle @{role}.",
)

_EN_MODIFIERS: Final = (
    "galaxy",
    "pixel",
    "midnight",
    "sunrise",
    "dragon",
    "neon",
    "cozy",
    "indie",
    "lofi",
    "anime",
    "esports",
    "ranked",
    "casual",
    "cosplay",
    "trivia",
    "puzzle",
    "rhythm",
    "survival",
    "sandbox",
    "strategy",
    "horror",
    "racing",
    "fishing",
    "crafting",
    "stealth",
)
_EN_NOUNS: Final = (
    "raid",
    "jam",
    "meetup",
    "giveaway",
    "scrim",
    "hangout",
    "marathon",
    "showcase",
    "bootcamp",
    "ladder",
    "draft",
    "watch party",
    "lan party",
    "speedrun race",
    "build contest",
)
_EN_FACTS: Final[tuple[str, ...]] = (
    "The {E} takes place every {day_en} at {hour}:00 UTC.",
    "Sign-ups for the {E} are open in #{channel}.",
    "The {E} is run by the @{role} team.",
    "The winner of the {E} receives {prize_en}.",
    "There is no {E} in {month_en}.",
    "The {E} usually lasts {n} hours.",
    "You need the @{role} role to join the {E}.",
)


@dataclass(frozen=True, slots=True)
class _SimpleLanguage:
    """A language with a smaller topic space: name templates, fact and question templates."""

    code: str
    modifiers: tuple[str, ...]
    nouns: tuple[str, ...]
    name: Callable[[str, str], str]
    days: tuple[str, ...]
    facts: tuple[str, ...]
    questions: tuple[tuple[str, str], ...]
    # Articles for {Art} / {art} / {of_art}, masculine then feminine; empty for
    # languages without articles.
    articles: tuple[tuple[str, str, str], tuple[str, str, str]] = (("", "", ""), ("", "", ""))
    feminine: frozenset[str] = frozenset()


_SIMPLE_LANGUAGES: Final = (
    _SimpleLanguage(
        code="es",
        modifiers=("ajedrez", "karaoke", "cine", "dibujo", "fútbol", "cocina", "trivia", "baile"),
        nouns=("torneo", "noche", "taller", "liga", "concurso", "curso", "encuentro", "maratón"),
        name=lambda modifier, noun: f"{noun} de {modifier}",
        days=_DAYS_ES,
        facts=(
            "{Art} {E} es cada {day} a las {hour}:00.",
            "Las inscripciones para {art} {E} están en #{channel}.",
            "{Art} {E} lo organiza el rol @{role}.",
            "{Art} {E} dura unas {n} horas.",
        ),
        questions=(
            ("keyword", "{E}"),
            ("formal", "¿Cuándo es {art} {E}?"),
            ("colloquial", "info {of_art} {E} porfa"),
        ),
        articles=(("El", "el", "del"), ("La", "la", "de la")),
        feminine=frozenset({"noche", "liga"}),
    ),
    _SimpleLanguage(
        code="fr",
        modifiers=("échecs", "karaoké", "cinéma", "dessin", "cuisine", "quiz", "danse", "lecture"),
        nouns=("tournoi", "soirée", "atelier", "ligue", "concours", "cours", "rencontre", "défi"),
        name=lambda modifier, noun: f"{noun} {modifier}",
        days=_DAYS_FR,
        facts=(
            "{Art} {E} a lieu chaque {day} à {hour}h.",
            "Les inscriptions pour {art} {E} se font dans #{channel}.",
            "{Art} {E} est organisé par le rôle @{role}.",
            "{Art} {E} dure environ {n} heures.",
        ),
        questions=(
            ("keyword", "{E}"),
            ("formal", "Quand a lieu {art} {E} ?"),
            ("colloquial", "c'est quand {art} {E}"),
        ),
        articles=(("Le", "le", "du"), ("La", "la", "de la")),
        feminine=frozenset({"soirée", "ligue", "rencontre"}),
    ),
    _SimpleLanguage(
        code="pt",
        modifiers=(
            "xadrez",
            "karaokê",
            "cinema",
            "desenho",
            "culinária",
            "quiz",
            "dança",
            "leitura",
        ),
        nouns=("torneio", "noite", "oficina", "liga", "concurso", "curso", "encontro", "desafio"),
        name=lambda modifier, noun: f"{noun} de {modifier}",
        days=_DAYS_PT,
        facts=(
            "{Art} {E} acontece toda {day} às {hour}h.",
            "As inscrições para {art} {E} ficam em #{channel}.",
            "{Art} {E} é organizado pelo cargo @{role}.",
            "{Art} {E} dura cerca de {n} horas.",
        ),
        questions=(
            ("keyword", "{E}"),
            ("formal", "Quando é {art} {E}?"),
            ("colloquial", "e {art} {E}, rola quando?"),
        ),
        articles=(("O", "o", "do"), ("A", "a", "da")),
        feminine=frozenset({"noite", "oficina", "liga"}),
    ),
    _SimpleLanguage(
        code="tr",
        modifiers=("satranç", "karaoke", "sinema", "çizim", "yemek", "bilgi", "dans", "kitap"),
        nouns=("turnuvası", "gecesi", "atölyesi", "ligi", "yarışması", "kursu", "buluşması"),
        name=lambda modifier, noun: f"{modifier} {noun}",
        days=_DAYS_TR,
        facts=(
            "{E} her {day} saat {hour}:00'da yapılır.",
            "{E} için kayıtlar #{channel} kanalında.",
            "{E} @{role} rolü tarafından düzenlenir.",
            "{E} yaklaşık {n} saat sürer.",
        ),
        questions=(
            ("keyword", "{E}"),
            ("formal", "{E} ne zaman?"),
            ("colloquial", "{E} hakkında bilgi var mı"),
        ),
    ),
    _SimpleLanguage(
        code="pl",
        modifiers=("szachowy", "karaoke", "filmowy", "rysunkowy", "kulinarny", "taneczny"),
        nouns=("turniej", "wieczór", "warsztat", "konkurs", "kurs", "maraton", "quiz"),
        name=lambda modifier, noun: f"{noun} {modifier}",
        days=_DAYS_PL,
        facts=(
            "{E} odbywa się w każdy {day} o {hour}:00.",
            "Zapisy na {E} są na kanale #{channel}.",
            "{E} organizuje rola @{role}.",
            "{E} trwa około {n} godzin.",
        ),
        questions=(
            ("keyword", "{E}"),
            ("formal", "Kiedy jest {E}?"),
            ("colloquial", "jest jakiś {E}?"),
        ),
    ),
    _SimpleLanguage(
        code="ja",
        modifiers=("チェス", "カラオケ", "映画", "お絵かき", "料理", "クイズ"),
        nouns=("大会", "ナイト", "教室", "リーグ", "コンテスト", "勉強会"),
        name=lambda modifier, noun: f"{modifier}{noun}",
        days=_DAYS_JA,
        facts=(
            "{E}は毎週{day}の{hour}時に開催されます。",
            "{E}の参加登録は #{channel} で受け付けています。",
            "{E}は @{role} ロールが運営しています。",
        ),
        questions=(("keyword", "{E}"), ("formal", "{E}はいつですか？")),
    ),
    _SimpleLanguage(
        code="ko",
        modifiers=("체스", "노래방", "영화", "그림", "요리", "퀴즈"),
        nouns=("대회", "모임", "강좌", "리그", "챌린지", "스터디"),
        name=lambda modifier, noun: f"{modifier} {noun}",
        days=_DAYS_KO,
        facts=(
            "{E}는 매주 {day} {hour}시에 열립니다.",
            "{E} 참가 신청은 #{channel} 채널에서 받습니다.",
            "{E}는 @{role} 역할이 운영합니다.",
        ),
        questions=(("keyword", "{E}"), ("formal", "{E} 언제 해요?")),
    ),
)

# Hand-made near-collisions, the shapes the diagnosis named: German compounds
# sharing "Server", and "Mentoriat" next to words that start like it.
_HANDMADE_TOPICS: Final[tuple[tuple[str, tuple[str, ...], tuple[tuple[str, str], ...]], ...]] = (
    (
        "Serverwartung",
        (
            "Die Serverwartung ist jeden Donnerstag um 5:00 MEZ.",
            "Bei Problemen beim Hoster kann es zusätzlich eine spontane Serverwartung geben.",
        ),
        (("keyword", "Serverwartung"), ("split", "Server Wartung"), ("keyword", "Wartung")),
    ),
    (
        "Serverregeln",
        ("Die Serverregeln stehen im Kanal #regeln und gelten für alle Mitglieder.",),
        (("keyword", "Serverregeln"), ("vague", "Was ist mit den Regeln?")),
    ),
    (
        "Serverumzug",
        ("Der Serverumzug auf den neuen Hoster ist für den 3. November geplant.",),
        (("keyword", "Serverumzug"), ("formal", "Wann zieht der Server um?")),
    ),
    (
        "Serverbackup",
        ("Ein Serverbackup wird jede Nacht um 4 Uhr erstellt.",),
        (("keyword", "Backup"), ("formal", "Wie oft gibt es ein Serverbackup?")),
    ),
    (
        "Mentorprogramm",
        ("Für das Mentorprogramm kann man sich bei der Rolle @Mentoren melden.",),
        (("keyword", "Mentorprogramm"), ("inflected", "Mentorprogramme")),
    ),
    (
        "Mentoriat",
        (
            "Das Mentoriat für Neulinge findet jeden Dienstag um 16 Uhr statt.",
            "Ein zweites Mentoriat gibt es donnerstags um 10 Uhr im Sprachkanal.",
        ),
        (
            ("keyword", "Mentoriate"),
            ("vague", "Was ist mit Mentoriaten?"),
            ("formal", "Wann finden Mentoriate statt?"),
        ),
    ),
)

_NEGATIVE_QUESTIONS: Final[tuple[tuple[str, str], ...]] = (
    ("greeting", "Hallo zusammen"),
    ("greeting", "Danke dir!"),
    ("greeting", "hi everyone"),
    ("greeting", "Hola, ¿qué tal?"),
    ("greeting", "Merci beaucoup"),
    ("greeting", "Teşekkürler"),
    ("greeting", "Dzięki wielkie"),
    ("greeting", "こんにちは"),
    ("greeting", "안녕하세요"),
    ("off-topic", "Wie wird das Wetter morgen?"),
    ("off-topic", "Was kostet Nitro?"),
    ("off-topic", "Wer ist der Owner vom Server?"),
    ("off-topic", "Was ist die Hauptstadt von Frankreich?"),
    ("off-topic", "Can you play music in voice?"),
    ("off-topic", "Tell me a joke"),
    ("off-topic", "How do I get the VIP role?"),
    ("off-topic", "¿Cuál es la contraseña del wifi?"),
    ("off-topic", "Quelle heure est-il ?"),
    ("off-topic", "Qual é o IP do servidor?"),
    ("off-topic", "Bugün hava nasıl?"),
    ("off-topic", "Ile kosztuje Nitro?"),
    ("off-topic", "今日の天気は？"),
    ("off-topic", "오늘 날씨 어때?"),
    ("off-topic", "Gibt es einen Fortnite-Kanal?"),
    ("off-topic", "Wo kann ich Bugs melden?"),
    ("sound-alike", "Turner"),
    ("sound-alike", "Mentos"),
    ("sound-alike", "Kegel"),
    ("sound-alike", "Sprechanlage"),
    ("sound-alike", "Kursiv"),
    ("sound-alike", "Ligatur"),
    ("sound-alike", "Streamer"),
    ("sound-alike", "draftsman"),
    ("sound-alike", "jamming"),
    ("sound-alike", "Wartezimmer"),
)


@dataclass(frozen=True, slots=True)
class _Topic:
    """One invented topic: its facts and the questions that ask about it."""

    name: str
    language: str
    facts: tuple[str, ...]
    questions: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class ScaleCorpus:
    """An invented guild and the questions labelled against it.

    Attributes
    ----------
    facts
        The facts, IDs 1..N.
    languages
        The language code of each fact, in the same order.
    cases
        Positive, then negative questions.
    """

    facts: tuple[InventedFact, ...]
    languages: tuple[str, ...]
    cases: tuple[EvalCase, ...]


def _typo(word: str, rng: random.Random) -> str | None:
    """Return a one-edit misspelling of a word of eight or more letters, or None."""
    if len(word) < 8 or not word.isalpha():
        return None
    position = rng.randrange(2, len(word) - 2)
    if rng.random() < 0.5:
        return word[:position] + word[position + 1] + word[position] + word[position + 2 :]
    return word[:position] + word[position + 1 :]


def _fill(template: str, entity: str, rng: random.Random, **extra: str) -> str:
    """Fill one fact template with an entity and random but plausible details."""
    values = {
        "E": entity,
        "day": "",
        "day_en": rng.choice(_DAYS_EN),
        "hour": str(rng.randrange(8, 23)),
        "date": str(rng.randrange(1, 29)),
        "month_de": rng.choice(_MONTHS_DE),
        "month_en": rng.choice(_MONTHS_EN),
        "channel": rng.choice(_CHANNELS),
        "role": rng.choice(_ROLES),
        "prize_de": rng.choice(_PRIZES_DE),
        "prize_en": rng.choice(_PRIZES_EN),
        "n": str(rng.randrange(1, 5)),
    }
    values.update(extra)
    return template.format(**values)


def _german_topics(rng: random.Random) -> tuple[list[_Topic], list[str]]:
    """Return the German topics and the held-out German names."""
    topics: list[_Topic] = []
    held_out: list[str] = []
    for modifier_index, modifier in enumerate(_DE_MODIFIERS):
        for noun_index, noun in enumerate(_DE_NOUNS):
            entity = modifier + noun.word.lower()
            if (modifier_index + noun_index) % _HOLD_OUT_EVERY == 0:
                held_out.append(f"Wann ist {noun.nominative} {entity}?")
                continue
            templates = rng.sample(_DE_FACTS, rng.randrange(1, 4))
            facts = tuple(
                _fill(
                    template,
                    entity,
                    rng,
                    Nom=noun.nominative.capitalize(),
                    acc=noun.accusative,
                    dat=noun.dative,
                    day=rng.choice(_DAYS_DE),
                )
                for template in templates
            )
            questions = [
                ("keyword", entity),
                ("inflected", modifier + noun.plural.lower()),
                ("split", f"{modifier} {noun.word}"),
                ("formal", f"Wann findet {noun.nominative} {entity} statt?"),
                ("colloquial", f"wann is {noun.nominative} {entity.lower()}"),
                ("vague", f"Was ist mit {noun.dative} {entity}?"),
            ]
            misspelled = _typo(entity, rng)
            if misspelled is not None:
                questions.append(("typo", misspelled))
            topics.append(_Topic(entity, "de", facts, tuple(questions)))
    return topics, held_out


def _english_topics(rng: random.Random) -> tuple[list[_Topic], list[str]]:
    """Return the English topics and the held-out English names."""
    topics: list[_Topic] = []
    held_out: list[str] = []
    for modifier_index, modifier in enumerate(_EN_MODIFIERS):
        for noun_index, noun in enumerate(_EN_NOUNS):
            entity = f"{modifier} {noun}"
            if (modifier_index + noun_index) % _HOLD_OUT_EVERY == 0:
                held_out.append(f"When is the {entity}?")
                continue
            templates = rng.sample(_EN_FACTS, rng.randrange(1, 4))
            facts = tuple(_fill(template, entity, rng) for template in templates)
            questions = [
                ("keyword", entity),
                ("inflected", f"{entity}s"),
                ("formal", f"When does the {entity} take place?"),
                ("colloquial", f"yo when's the {entity}"),
                ("vague", f"what about the {entity}?"),
            ]
            for word in (noun, modifier):
                misspelled = _typo(word, rng)
                if misspelled is not None:
                    questions.append(("typo", entity.replace(word, misspelled, 1)))
                    break
            topics.append(_Topic(entity, "en", facts, tuple(questions)))
    return topics, held_out


def _simple_topics(rng: random.Random) -> tuple[list[_Topic], list[str]]:
    """Return the topics and held-out questions of the smaller languages."""
    topics: list[_Topic] = []
    held_out: list[str] = []
    for language in _SIMPLE_LANGUAGES:
        for modifier_index, modifier in enumerate(language.modifiers):
            for noun_index, noun in enumerate(language.nouns):
                entity = language.name(modifier, noun)
                formal = dict(language.questions).get("formal", "{E}")
                if (modifier_index + noun_index) % _HOLD_OUT_EVERY == 0:
                    _, article, of_article = language.articles[noun in language.feminine]
                    held_out.append(formal.format(E=entity, art=article, of_art=of_article))
                    continue
                capitalized, article, of_article = language.articles[noun in language.feminine]
                grammar = {"Art": capitalized, "art": article, "of_art": of_article}
                templates = rng.sample(
                    language.facts, rng.randrange(1, min(3, len(language.facts)) + 1)
                )
                facts = tuple(
                    _fill(template, entity, rng, day=rng.choice(language.days), **grammar)
                    for template in templates
                )
                questions = tuple(
                    (register, template.format(E=entity, **grammar).strip())
                    for register, template in language.questions
                )
                topics.append(_Topic(entity, language.code, facts, questions))
    return topics, held_out


def _all_topics(seed: int) -> tuple[list[_Topic], list[str]]:
    """Return every topic in a fixed shuffled order, the hand-made ones first, and the held-out questions."""
    rng = random.Random(seed)
    german, german_held = _german_topics(rng)
    english, english_held = _english_topics(rng)
    simple, simple_held = _simple_topics(rng)
    generated = german + english + simple
    rng.shuffle(generated)
    handmade = [_Topic(name, "de", facts, questions) for name, facts, questions in _HANDMADE_TOPICS]
    held_out = german_held + english_held + simple_held
    rng.shuffle(held_out)
    return handmade + generated, held_out


def build_scale_corpus(
    fact_count: int,
    *,
    seed: int = DEFAULT_SEED,
    positive_questions: int = 200,
    held_out_questions: int = 40,
) -> ScaleCorpus:
    """Build an invented guild of exactly `fact_count` facts and its labelled questions.

    Parameters
    ----------
    fact_count
        How many facts. At most the number the generator can produce (about
        2,300); at least the hand-made topics' facts (8).
    seed
        Fixes every random choice.
    positive_questions
        How many positive questions to sample, spread over the topics present.
    held_out_questions
        How many template-trap negatives about held-out topics to include.

    Returns
    -------
    ScaleCorpus
        Facts with IDs 1..fact_count, then positive and negative questions.

    Raises
    ------
    ValueError
        If `fact_count` is outside what the generator can produce.
    """
    topics, held_out = _all_topics(seed)
    available = sum(len(topic.facts) for topic in topics)
    minimum = sum(len(facts) for _, facts, _ in _HANDMADE_TOPICS)
    if not minimum <= fact_count <= available:
        raise ValueError(f"fact_count must be between {minimum} and {available}")

    facts: list[InventedFact] = []
    languages: list[str] = []
    positives: list[EvalCase] = []
    rng = random.Random(seed + fact_count)
    candidate_questions: list[tuple[str, str, frozenset[int]]] = []
    for topic in topics:
        if len(facts) >= fact_count:
            break
        ids: list[int] = []
        for content in topic.facts[: fact_count - len(facts)]:
            facts.append(InventedFact(fact_id=len(facts) + 1, content=content))
            languages.append(topic.language)
            ids.append(len(facts))
        for register, question in topic.questions:
            candidate_questions.append((register, question, frozenset(ids)))

    handmade_count = sum(len(questions) for _, _, questions in _HANDMADE_TOPICS)
    handmade, generated = candidate_questions[:handmade_count], candidate_questions[handmade_count:]
    sampled = handmade + rng.sample(
        generated, min(len(generated), max(positive_questions - len(handmade), 0))
    )
    for index, (register, question, relevant) in enumerate(sampled):
        positives.append(
            EvalCase(
                case_id=f"pos-{index:04d}",
                query=question,
                kind="positive",
                register=register,
                relevant=relevant,
            )
        )
    negatives = [
        EvalCase(
            case_id=f"neg-{index:03d}",
            query=question,
            kind="negative",
            register=register,
            relevant=frozenset(),
        )
        for index, (register, question) in enumerate(
            list(_NEGATIVE_QUESTIONS)
            + [("held-out", question) for question in held_out[:held_out_questions]]
        )
    ]
    return ScaleCorpus(
        facts=tuple(facts), languages=tuple(languages), cases=tuple(positives + negatives)
    )
