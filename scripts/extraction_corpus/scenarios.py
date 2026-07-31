"""The locale grid this corpus spans, and how many messages of each kind per locale.

One "scenario" per locale, all nine CLAUDE.md commits to -- no size or
community-type axis like scripts/synthetic_corpus.scenarios, because
fact-worthiness classification does not depend on server size or theme the
way Stage 2 retrieval depends on a guild's actual fact set. What DOES vary
per locale is the language the LLM is asked to write in; a community
backdrop (a study server, a gaming server) is still given per generation call
so the messages read like a real channel rather than isolated sentences, and
it is deliberately re-picked at random per call from a small set of themes
rather than fixed per locale -- so a locale's separation score cannot be
confounded with "this locale happened to get the easiest theme."
"""
from __future__ import annotations

from dataclasses import dataclass

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

# All nine of CLAUDE.md's supported locales, in the same order the table in
# CLAUDE.md's Internationalization section lists them.
LOCALES: tuple[str, ...] = (
    "en-US",
    "es-ES",
    "pt-BR",
    "de",
    "fr",
    "tr",
    "pl",
    "ja",
    "ko",
)

COMMUNITY_THEMES: tuple[str, ...] = (
    "a gaming community organised around co-op sessions and weekly tournaments",
    "a study and exam-preparation server with per-subject channels",
    "a creative-project server for artists, musicians and writers",
    "a peer tech-support server where members troubleshoot software and hardware",
    "a general-purpose social community with everyday chat and member-run events",
)

# Per-locale message counts. Ratio is the whole point of this corpus (see
# CLAUDE.md-adjacent design brief: "clearly noise-dominated... roughly 90%
# ordinary chat, 10% fact-worthy content"), and it is produced by the counts
# below exactly, not by a post-hoc downsample: 25 fact-worthy per locale (5
# per subcategory) against 230 not-fact-worthy per locale (150 ordinary,
# spread over six subcategories, plus 80 hard-negative, split evenly between
# the two adversarial-near-miss categories) is a 25:230 split, i.e. 9.8%
# positive.
#
# Phase 3a-1's original 5-per-locale grid (see reports/phase-3a-1.txt Section
# 8 and Section 9) was enough to show the mechanism separates real sentences
# at all, but explicitly too small to trust a per-locale number or the exact
# threshold position -- a single case moving from tp to fn swung recall by 20
# points on n=5. Phase 3a-1b (reports/phase-3a-1b.txt) scales every category
# 5x, proportionally, to reach a per-locale sample that can actually support
# the per-locale breakdown and threshold sweep the brief asks for, while
# keeping the same ~90/10 noise ratio 3a-1 established.
FACT_WORTHY_PER_LOCALE = 25
ORDINARY_NOT_FACT_WORTHY_PER_LOCALE = 150
HARD_NEGATIVE_PER_LOCALE_PER_CATEGORY = 40

TOTAL_PER_LOCALE = (
    FACT_WORTHY_PER_LOCALE
    + ORDINARY_NOT_FACT_WORTHY_PER_LOCALE
    + 2 * HARD_NEGATIVE_PER_LOCALE_PER_CATEGORY
)


@dataclass(frozen=True)
class LocaleScenario:
    """One locale's generation plan."""

    locale: str
    index: int

    @property
    def language_name(self) -> str:
        return LANGUAGE_NAMES[self.locale]


SCENARIOS: tuple[LocaleScenario, ...] = tuple(
    LocaleScenario(locale=locale, index=index) for index, locale in enumerate(LOCALES)
)


def describe_grid() -> str:
    """Human-readable summary of the corpus plan, for the report header."""
    lines = [
        f"{'locale':<8} {'fact-worthy':>12} {'ordinary noise':>15} {'hard negatives':>15} {'total':>7}",
        "-" * 64,
    ]
    hard = 2 * HARD_NEGATIVE_PER_LOCALE_PER_CATEGORY
    for scenario in SCENARIOS:
        lines.append(
            f"{scenario.locale:<8} {FACT_WORTHY_PER_LOCALE:>12} "
            f"{ORDINARY_NOT_FACT_WORTHY_PER_LOCALE:>15} {hard:>15} {TOTAL_PER_LOCALE:>7}"
        )
    total_messages = TOTAL_PER_LOCALE * len(SCENARIOS)
    total_positive = FACT_WORTHY_PER_LOCALE * len(SCENARIOS)
    lines.append("-" * 64)
    lines.append(
        f"TOTAL {total_messages} messages, {total_positive} fact-worthy "
        f"({total_positive / total_messages:.1%})"
    )
    return "\n".join(lines)
