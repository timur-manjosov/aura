"""The scenario grid: which guilds exist, how big, about what, in which language.

Three axes, laid out as a deliberate grid rather than sampled, so no
combination is missing by accident:

* **size**   -- small and medium only (CLAUDE.md targets a few MB per server;
                a "huge" tier would generate evidence for a deployment shape
                this project explicitly does not have).
* **topic**  -- five structurally different community types, so the corpus
                cannot collapse into one template measured ten times.
* **locale** -- all nine supported locales, each carrying a full guild rather
                than a token example. Phase 2a-1/2a-2 calibrated against
                hand-written sentences that were majority English and German;
                a threshold tuned on that and deployed to a Japanese or Turkish
                server is exactly the kind of unexamined edge this project's
                core principle rules out.

`en-US` appears twice because five community types times two sizes is ten
guilds and there are nine locales. It gets the extra slot as Aura's mandatory
fallback locale -- the one every deployment can end up using regardless of what
its members speak.
"""
from __future__ import annotations

from dataclasses import dataclass

from synthetic_corpus.corpus_model import CommunityType, GuildSize

# What each size tier means concretely. Fact counts, not member counts, are what
# actually load Stage 2: retrieval ranks against every active fact in the guild,
# so a guild's fact count is the number that decides how hard the discrimination
# is. Member counts are recorded for realism in the generation prompt only.
FACTS_PER_SIZE: dict[GuildSize, int] = {
    GuildSize.SMALL: 16,
    GuildSize.MEDIUM: 32,
}

# Additional facts generated to contradict an existing one, per guild. Both
# stay ACTIVE -- an unsuperseded contradiction is the exact situation
# PROACTIVE_CONFIDENCE_GAP exists for, and the corpus has to contain it for the
# gap to be measurable at all.
CONTRADICTION_PAIRS_PER_GUILD = 6

# The base facts are partitioned, not shared: the LAST
# CONTRADICTION_PAIRS_PER_GUILD of them are "contested" -- each one gets a
# contradicting partner and is used only by the contradictory-question
# category -- and the rest are "clean", used by every other category.
#
# The partition is load-bearing, not tidiness. The first smoke run generated
# contradictions over the whole fact set, and the answered-question and
# partial-answer categories then happily targeted facts that had a live
# contradiction sitting next to them. Those cases were labelled "a fact
# genuinely answers this" while actually being ambiguous, which would have made
# the Stage 2 confidence gap look like it was rejecting true positives. A
# category's label has to be true of the whole knowledge state the message is
# evaluated against, not just of the one fact it was written from.

# How many messages of each category to generate per guild.
#
# partial_answer and contradictory_facts are deliberately generated at double
# the volume of every other category. Phase 2a-3's bake-off found those two
# categories were where both cheaper candidate models failed outright and where
# even the chosen model's margin was thinnest -- and the hand-measured samples
# behind the current thresholds contain exactly one case of each. That is where
# threshold precision actually decides something, so that is where the evidence
# has to be thickest.
MESSAGES_PER_CATEGORY: dict[str, int] = {
    "answered_question": 8,
    "off_topic_chatter": 6,
    "unanswered_question": 6,
    "partial_answer": 12,
    "contradictory_facts": 12,
    "adversarial_injection": 4,
    "adversarial_toxic": 4,
}

# Malformed cases are constructed programmatically rather than generated (see
# `malformed.py`): unicode edge cases are exactly specifiable, cost nothing, and
# carry no chance of an LLM inventing something unsafe.
MALFORMED_CASES_PER_GUILD = 6


@dataclass(frozen=True)
class GuildScenario:
    """One point on the size x topic x locale grid, before any content exists."""

    key: str
    index: int
    name: str
    community_type: CommunityType
    size: GuildSize
    locale: str
    member_count: int

    @property
    def fact_count(self) -> int:
        """How many base facts this guild's size tier calls for."""
        return FACTS_PER_SIZE[self.size]

    @property
    def clean_fact_count(self) -> int:
        """Base facts that will NOT get a contradicting partner."""
        return self.fact_count - CONTRADICTION_PAIRS_PER_GUILD


# Plain-English descriptions handed to the generator. They describe the *server*,
# never the labels -- the label comes from which prompt template is used, so a
# description that leaked category hints would blur the by-construction
# labelling this whole corpus depends on.
COMMUNITY_DESCRIPTIONS: dict[CommunityType, str] = {
    CommunityType.HOBBY_GAMING: (
        "a gaming community organised around co-op sessions, weekly tournaments, "
        "voice nights, and a roster of games the members play together"
    ),
    CommunityType.STUDY_EDUCATION: (
        "a study and exam-preparation server with per-subject channels, shared "
        "notes, revision sessions and a schedule built around a term calendar"
    ),
    CommunityType.CREATIVE_PROJECT: (
        "a creative-project server for artists, musicians and writers, with "
        "critique threads, collaboration calls, showcase channels and submission rules"
    ),
    CommunityType.TECH_SUPPORT: (
        "a peer tech-support server where members help each other troubleshoot "
        "software, hardware and configuration problems, with triage channels and "
        "documented workarounds"
    ),
    CommunityType.GENERAL_SOCIAL: (
        "a general-purpose social community with everyday chat, member-run events, "
        "hobby corners and a loose but enforced set of house rules"
    ),
}

LANGUAGE_NAMES: dict[str, str] = {
    "en-US": "English",
    "es-ES": "Spanish (Spain)",
    "pt-BR": "Brazilian Portuguese",
    "de": "German",
    "fr": "French",
    "tr": "Turkish",
    "pl": "Polish",
    "ja": "Japanese",
    "ko": "Korean",
}

SCENARIOS: tuple[GuildScenario, ...] = (
    GuildScenario(
        key="g01-gaming-small-en",
        index=1,
        name="Pixel Pact",
        community_type=CommunityType.HOBBY_GAMING,
        size=GuildSize.SMALL,
        locale="en-US",
        member_count=42,
    ),
    GuildScenario(
        key="g02-gaming-medium-ja",
        index=2,
        name="ギルド・ノクターン",
        community_type=CommunityType.HOBBY_GAMING,
        size=GuildSize.MEDIUM,
        locale="ja",
        member_count=780,
    ),
    GuildScenario(
        key="g03-study-small-de",
        index=3,
        name="Lerngruppe Nordlicht",
        community_type=CommunityType.STUDY_EDUCATION,
        size=GuildSize.SMALL,
        locale="de",
        member_count=61,
    ),
    GuildScenario(
        key="g04-study-medium-es",
        index=4,
        name="Aula Abierta",
        community_type=CommunityType.STUDY_EDUCATION,
        size=GuildSize.MEDIUM,
        locale="es-ES",
        member_count=540,
    ),
    GuildScenario(
        key="g05-creative-small-fr",
        index=5,
        name="Atelier Minuit",
        community_type=CommunityType.CREATIVE_PROJECT,
        size=GuildSize.SMALL,
        locale="fr",
        member_count=35,
    ),
    GuildScenario(
        key="g06-creative-medium-pt",
        index=6,
        name="Coletivo Aurora",
        community_type=CommunityType.CREATIVE_PROJECT,
        size=GuildSize.MEDIUM,
        locale="pt-BR",
        member_count=624,
    ),
    GuildScenario(
        key="g07-support-small-pl",
        index=7,
        name="Serwerownia",
        community_type=CommunityType.TECH_SUPPORT,
        size=GuildSize.SMALL,
        locale="pl",
        member_count=88,
    ),
    GuildScenario(
        key="g08-support-medium-tr",
        index=8,
        name="Teknik Destek Kulübü",
        community_type=CommunityType.TECH_SUPPORT,
        size=GuildSize.MEDIUM,
        locale="tr",
        member_count=910,
    ),
    GuildScenario(
        key="g09-social-small-ko",
        index=9,
        name="달빛 라운지",
        community_type=CommunityType.GENERAL_SOCIAL,
        size=GuildSize.SMALL,
        locale="ko",
        member_count=55,
    ),
    GuildScenario(
        key="g10-social-medium-en",
        index=10,
        name="The Long Table",
        community_type=CommunityType.GENERAL_SOCIAL,
        size=GuildSize.MEDIUM,
        locale="en-US",
        member_count=1204,
    ),
)


def describe_grid() -> str:
    """Return a human-readable table of the scenario grid, for the report header."""
    lines = [
        f"{'guild':<24} {'type':<17} {'size':<7} {'locale':<6} {'members':>7} {'facts':>6}",
        "-" * 72,
    ]
    for scenario in SCENARIOS:
        lines.append(
            f"{scenario.key:<24} {scenario.community_type:<17} {scenario.size:<7} "
            f"{scenario.locale:<6} {scenario.member_count:>7} "
            f"{scenario.fact_count + CONTRADICTION_PAIRS_PER_GUILD:>6}"
            f"  ({scenario.clean_fact_count} clean + "
            f"{CONTRADICTION_PAIRS_PER_GUILD} contested pairs)"
        )
    return "\n".join(lines)
