"""Hand-written sample cards for the operator preview of the v2 answer format.

Every fact, question and answer here is INVENTED. The samples exist so the
operator can judge how the v2 answer card looks in a real Discord client --
desktop and phone, light and dark -- before any real answer uses it. They are
built through the real code: each sample answer passes
`aura.answer_contract.validate_contract`, and each card is rendered by
`aura.answer_card`, so the preview shows what an answer would look like, not a
mock-up of it.

Sample content is written in German and English (the operator's own language is
German); every label around it (sources, notes, footers, limit text) is the
locale text a real reader would see. No database, no model, no ledger: a sample
is a pure function of the locale, the guild and the clock.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, time, timedelta
from typing import Final

from aura.answer_card import (
    AnswerCard,
    build_answer_card,
    build_fact_list_card,
    build_notice_card,
)
from aura.answer_contract import validate_contract
from aura.db.models import Fact, FactStatus
from aura.i18n import t
from aura.theme import MessageKind

# Invented channels, so the sources block shows several channel names. The
# links built from them lead nowhere; the preview says so.
_CHANNELS: Final[dict[str, dict[int, str]]] = {
    "de": {1: "ankündigungen", 2: "events", 3: "regeln", 4: "lernen", 5: "allgemein"},
    "en": {1: "announcements", 2: "events", 3: "rules", 4: "learning", 5: "general"},
}

# key -> (channel number, sentence)
_FACTS: Final[dict[str, dict[str, tuple[int, str]]]] = {
    "de": {
        "film_time": (1, "Der Filmabend ist jeden Sonntag um 20 Uhr im Voice-Kanal Kino."),
        "film_vote": (2, "Für den Filmabend wird der Film am Freitag davor per Umfrage gewählt."),
        "film_rules": (3, "Beim Filmabend bleiben Mikrofone stumm, der Chat ist erlaubt."),
        "cup_18": (1, "Der Herbst-Cup startet um 18 Uhr."),
        "cup_19": (2, "Der Herbst-Cup startet um 19 Uhr."),
        "code_tue": (4, "Jeden Dienstag um 17 Uhr gibt es eine Programmier-Sprechstunde."),
        "code_sat": (
            5,
            "Eine Programmier-Sprechstunde findet jeden ersten Samstag im Monat statt.",
        ),
    },
    "en": {
        "film_time": (1, "Movie night is every Sunday at 20:00 in the Cinema voice channel."),
        "film_vote": (2, "The movie for movie night is chosen by a poll on the Friday before."),
        "film_rules": (3, "During movie night microphones stay muted; the chat is open."),
        "cup_18": (1, "The Autumn Cup starts at 18:00."),
        "cup_19": (2, "The Autumn Cup starts at 19:00."),
        "code_tue": (4, "Coding office hours are held every Tuesday at 17:00."),
        "code_sat": (5, "Coding office hours take place on the first Saturday of each month."),
    },
}


@dataclass(frozen=True)
class _SampleText:
    """The invented questions and answer sentences of one content language."""

    film_question: str
    film_lead: str
    film_points: tuple[tuple[str, tuple[str, ...]], ...]
    cup_question: str
    cup_lead: str
    code_question: str
    code_lead: str
    code_gap: str
    related_question: str
    limit_question: str


_TEXT: Final[dict[str, _SampleText]] = {
    "de": _SampleText(
        film_question="Wie läuft der Filmabend ab?",
        film_lead="Der Filmabend ist jeden Sonntag um 20 Uhr im Voice-Kanal Kino.",
        film_points=(
            ("Welcher Film läuft, wird am Freitag davor per Umfrage entschieden.", ("film_vote",)),
            ("Während des Films bleiben die Mikrofone stumm.", ("film_rules",)),
            ("Im Chat darf geschrieben werden.", ("film_rules",)),
        ),
        cup_question="Wann startet der Herbst-Cup?",
        cup_lead="Für den Herbst-Cup sind zwei Startzeiten vermerkt: 18 Uhr und 19 Uhr.",
        code_question="Wann und wo ist die Programmier-Sprechstunde?",
        code_lead=(
            "Zur Programmier-Sprechstunde sind zwei Termine vermerkt: dienstags um 17 Uhr "
            "und jeden ersten Samstag im Monat."
        ),
        code_gap="Ort der Sprechstunde",
        related_question="Gibt es einen Kinoabend mit Snacks?",
        limit_question="Wann ist der Filmabend?",
    ),
    "en": _SampleText(
        film_question="How does movie night work?",
        film_lead="Movie night is every Sunday at 20:00 in the Cinema voice channel.",
        film_points=(
            ("The movie is chosen by a poll on the Friday before.", ("film_vote",)),
            ("Microphones stay muted during the movie.", ("film_rules",)),
            ("The chat is open while it runs.", ("film_rules",)),
        ),
        cup_question="When does the Autumn Cup start?",
        cup_lead="Two start times are recorded for the Autumn Cup: 18:00 and 19:00.",
        code_question="When and where are the coding office hours?",
        code_lead=(
            "Two schedules are recorded for the coding office hours: Tuesdays at 17:00 and "
            "the first Saturday of each month."
        ),
        code_gap="location of the office hours",
        related_question="Is there a cinema night with snacks?",
        limit_question="When is movie night?",
    ),
}


@dataclass(frozen=True)
class PreviewSample:
    """One sample card and the caption the preview shows above it.

    Attributes
    ----------
    caption
        English, not translated: the preview is an operator tool.
    card
        The rendered card.
    """

    caption: str
    card: AnswerCard


def content_language(locale: str) -> str:
    """Return the sample content language for a locale: German for German, else English."""
    return "de" if locale.startswith("de") else "en"


def _facts(language: str, guild_id: int, keys: tuple[str, ...]) -> list[Fact]:
    facts: list[Fact] = []
    for number, key in enumerate(keys, start=1):
        channel_id, content = _FACTS[language][key]
        facts.append(
            Fact(
                id=number,
                guild_id=guild_id,
                channel_id=channel_id,
                message_id=number,
                content=content,
                embedding=b"",
                status=FactStatus.ACTIVE,
                created_at=datetime(2026, 8, 3 + 9 * number, 18, 0, tzinfo=UTC),
            )
        )
    return facts


def _reply(
    keys: tuple[str, ...],
    *,
    lead: str,
    points: tuple[tuple[str, tuple[str, ...]], ...] = (),
    relation: str | None = None,
    gap: tuple[str, ...] = (),
    answers_question: bool = True,
) -> dict[str, object]:
    """Return a contract reply as a model would write it, for the sample's facts."""
    number = {key: index for index, key in enumerate(keys, start=1)}
    return {
        "request_reading": "Sample request.",
        "fact_notes": [{"n": n, "covers": "sample"} for n in number.values()],
        "relations": [{"facts": list(number.values()), "kind": relation}] if relation else [],
        "not_covered_topics": list(gap),
        "tone": "casual",
        "lead": lead,
        "points": [{"text": text, "facts": [number[k] for k in ks]} for text, ks in points],
        "used_fact_numbers": list(number.values()),
        "answers_question": answers_question,
    }


def preview_samples(locale: str, *, guild_id: int, now: datetime) -> list[PreviewSample]:
    """Build the seven sample cards for one locale.

    Parameters
    ----------
    locale
        The operator's locale: picks the labels, and German or English content.
    guild_id
        The guild the preview runs in; the sample source links point into it.
    now
        The current time, for the limit sample's reset time.

    Returns
    -------
    list[PreviewSample]
        A normal answer with three points, a conflict, an "unclear if same"
        answer with the gap line, the related-facts reply, the limit notice, an
        error, and the proactive variant -- in that order.
    """
    language = content_language(locale)
    text = _TEXT[language]
    channels = _CHANNELS[language]

    def answer_card(
        keys: tuple[str, ...], reply: dict[str, object], question: str, *, proactive: bool = False
    ) -> AnswerCard:
        facts = _facts(language, guild_id, keys)
        return build_answer_card(
            validate_contract(reply, facts),
            facts,
            question=question,
            locale=locale,
            channel_names=channels,
            proactive=proactive,
        )

    film_keys = ("film_time", "film_vote", "film_rules")
    film_reply = _reply(film_keys, lead=text.film_lead, points=text.film_points)
    cup_keys = ("cup_18", "cup_19")
    code_keys = ("code_tue", "code_sat")
    reset = datetime.combine(now.astimezone(UTC).date() + timedelta(days=1), time(0), tzinfo=UTC)
    limit_note = t("ask_limit_guild_reached", locale, reset=f"<t:{int(reset.timestamp())}:R>")

    return [
        PreviewSample(
            "1/7 · answer with three points",
            answer_card(film_keys, film_reply, text.film_question),
        ),
        PreviewSample(
            "2/7 · two facts that conflict",
            answer_card(
                cup_keys,
                _reply(
                    cup_keys,
                    lead=text.cup_lead,
                    relation="same_detail_conflict",
                    answers_question=False,
                ),
                text.cup_question,
            ),
        ),
        PreviewSample(
            "3/7 · unclear whether both apply, with the 'not recorded' line",
            answer_card(
                code_keys,
                _reply(
                    code_keys, lead=text.code_lead, relation="unclear_if_same", gap=(text.code_gap,)
                ),
                text.code_question,
            ),
        ),
        PreviewSample(
            "4/7 · nothing exact, possibly related facts (no AI)",
            build_fact_list_card(
                MessageKind.RELATED,
                t("ask_no_info_related", locale),
                _facts(language, guild_id, ("film_time", "film_rules")),
                question=text.related_question,
            ),
        ),
        PreviewSample(
            "5/7 · daily limit reached (no AI; in real use only the asker sees it)",
            build_fact_list_card(
                MessageKind.LIMIT,
                limit_note,
                _facts(language, guild_id, ("film_time",)),
                question=text.limit_question,
            ),
        ),
        PreviewSample(
            "6/7 · error",
            build_notice_card(
                MessageKind.ERROR,
                t("ask_grounding_unverified", locale),
                question=text.film_question,
            ),
        ),
        PreviewSample(
            "7/7 · proactive answer (at most two points)",
            answer_card(film_keys, film_reply, text.film_question, proactive=True),
        ),
    ]


# --- P5: the card looks of the other message families ---------------------------

_P5_FACTS: Final[dict[str, dict[str, tuple[int, str]]]] = {
    "de": {
        "rule_ads": (3, "Werbung für andere Server ist in allen Kanälen verboten."),
        "rule_voice": (3, "Im Voice-Kanal Lounge ist Push-to-Talk Pflicht."),
        "status_memes": (1, "Der Kanal #memes ist geschlossen, Memes gehören nach #offtopic."),
        "event_quiz": (2, "Das Weihnachts-Quiz ist am 5. Dezember um 19 Uhr im Voice-Kanal Bühne."),
        "event_old": (2, "Das Bingo am Samstag beginnt um 19 Uhr."),
        "event_new": (2, "Das Bingo am Samstag wurde auf 20 Uhr verschoben."),
        "milestone": (5, "Der Server hat 2.000 Mitglieder erreicht."),
        "clips": (1, "Für Spiel-Highlights gibt es den neuen Kanal #clips."),
    },
    "en": {
        "rule_ads": (3, "Advertising other servers is not allowed in any channel."),
        "rule_voice": (3, "Push-to-talk is required in the Lounge voice channel."),
        "status_memes": (1, "The #memes channel is closed; memes go to #off-topic."),
        "event_quiz": (2, "The holiday quiz is on December 5 at 19:00 in the Stage voice channel."),
        "event_old": (2, "Saturday's bingo starts at 19:00."),
        "event_new": (2, "Saturday's bingo was moved to 20:00."),
        "milestone": (5, "The server reached 2,000 members."),
        "clips": (1, "There is a new #clips channel for gameplay highlights."),
    },
}


def _p5_fact(language: str, guild_id: int, key: str, number: int, day: int) -> Fact:
    channel_id, content = _P5_FACTS[language][key]
    return Fact(
        id=100 + number,
        guild_id=guild_id,
        channel_id=channel_id,
        message_id=100 + number,
        content=content,
        embedding=b"",
        status=FactStatus.ACTIVE,
        created_at=datetime(2026, 9, day, 17, 0, tzinfo=UTC),
    )


def card_look_samples(
    locale: str,
    *,
    guild_id: int,
    now: datetime,
    ask_caps: tuple[int, int, int],
    dashboard_url: str | None,
) -> list[PreviewSample]:
    """Build the samples of every family that has a card look since P5.

    Parameters
    ----------
    locale
        The operator's locale: picks the labels, and German or English content.
    guild_id
        The guild the preview runs in; the sample source links point into it.
    now
        The current time, for the digest period and the plan's dates.
    ask_caps
        (Free per-guild, Free per-member, Pro per-guild) /aura-ask caps, from
        the settings, so the plan samples show the real numbers.
    dashboard_url
        BILLING_DASHBOARD_URL, or None: the plan samples link to it exactly as
        a real /aura-plan would.

    Returns
    -------
    list[PreviewSample]
        The digest, the onboarding message, /aura-plan on Free and on Pro, the
        Pro-only refusal, and a confirmation -- in that order, each built by the
        real builders of aura.cards from invented content.
    """
    from aura.billing import (
        GuildPlan,
        PlanBasis,
        PlanTier,
        Standing,
        SubscriptionStanding,
    )
    from aura.cards import (
        build_digest_card,
        build_notice,
        build_onboarding_card,
        build_plan_card,
        build_pro_refusal_card,
    )
    from aura.commands.plan import standing_lines
    from aura.digest.builder import DigestChange, DigestContent
    from aura.digest.intervals import describe_interval
    from aura.onboarding.builder import OnboardingContent

    language = content_language(locale)
    channels = {**_CHANNELS[language], 5: "allgemein" if language == "de" else "general"}

    def fact(key: str, number: int, day: int) -> Fact:
        return _p5_fact(language, guild_id, key, number, day)

    digest = DigestContent(
        guild_id=guild_id,
        covered_from=now - timedelta(days=7),
        covered_until=now,
        new_facts=[fact("event_quiz", 1, 28), fact("clips", 2, 29)],
        milestones=[fact("milestone", 3, 30)],
        changes=[
            DigestChange(
                previous=fact("event_old", 4, 20),
                current=fact("event_new", 5, 30),
                changed_at=now - timedelta(days=2),
                collapsed_steps=0,
            )
        ],
    )
    onboarding = OnboardingContent(
        guild_id=guild_id,
        rules=[fact("rule_ads", 6, 2), fact("rule_voice", 7, 3)],
        status_changes=[fact("status_memes", 8, 12)],
        other=[fact("event_quiz", 9, 28), fact("clips", 10, 29)],
        total_eligible=5,
    )
    free_plan = GuildPlan(
        guild_id=guild_id,
        tier=PlanTier.FREE,
        basis=PlanBasis.SUBSCRIPTION,
        standing=SubscriptionStanding(
            standing=Standing.NO_SUBSCRIPTION,
            access_until=None,
            paid_through=None,
            shown_subscription_id=None,
            granting_subscription_ids=frozenset(),
        ),
    )
    paid_until = now + timedelta(days=21)
    pro_plan = GuildPlan(
        guild_id=guild_id,
        tier=PlanTier.PRO,
        basis=PlanBasis.SUBSCRIPTION,
        standing=SubscriptionStanding(
            standing=Standing.ACTIVE,
            access_until=paid_until,
            paid_through=paid_until,
            shown_subscription_id="sample",
            granting_subscription_ids=frozenset({"sample"}),
        ),
    )
    server_name = "Beispiel-Server" if language == "de" else "Example Server"
    return [
        PreviewSample(
            "P5 1/6 · digest (DIGEST_LOOK=card)",
            build_digest_card(
                digest,
                locale=locale,
                interval_label=describe_interval(7 * 24 * 3600, locale),
                channel_names=channels,
            ),
        ),
        PreviewSample(
            "P5 2/6 · onboarding (ONBOARDING_LOOK=card)",
            build_onboarding_card(
                onboarding, locale=locale, server_name=server_name, channel_names=channels
            ),
        ),
        PreviewSample(
            "P5 3/6 · /aura-plan on Free (PLAN_LOOK=card; only the admin sees it)",
            build_plan_card(
                free_plan,
                locale=locale,
                standing_lines=standing_lines(free_plan, locale=locale),
                dashboard_url=dashboard_url,
                ask_caps=ask_caps,
            ),
        ),
        PreviewSample(
            "P5 4/6 · /aura-plan on Pro (PLAN_LOOK=card)",
            build_plan_card(
                pro_plan,
                locale=locale,
                standing_lines=standing_lines(pro_plan, locale=locale),
                dashboard_url=dashboard_url,
                ask_caps=ask_caps,
            ),
        ),
        PreviewSample(
            "P5 5/6 · a Pro-only command on a Free server (NOTICE_LOOK=card)",
            build_pro_refusal_card(locale, dashboard_url=dashboard_url),
        ),
        PreviewSample(
            "P5 6/6 · a confirmation (NOTICE_LOOK=card)",
            build_notice(
                MessageKind.CONFIRM, t("pending_confirmed", locale, pending_id=7, fact_id=12)
            ),
        ),
    ]
