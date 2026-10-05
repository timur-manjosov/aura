"""The P5 evaluation harness: case-set shapes, the deterministic scoring, the refusals.

The harness spends money and the cases decide what "better" means, so both get
the same scrutiny as production code: the case sets have the sizes and labels
the brief asks for, every label points at a real message, the scoring flags
exactly what the labels make false (and nothing a split or a correction-aware
sentence makes true), and the two paid entry points refuse to run without the
explicit opt-in -- the shadow harness also refuses any provider the operator did
not approve for real content, and any place outside reports/.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS))

from p5_extraction_cases import (  # noqa: E402
    SHAPES,
    Chat,
    Expected,
    ExtractionCase,
    all_cases,
    at,
    group_present,
    validate_case,
)
from p5_proactive_cases import POSITIVE, all_messages  # noqa: E402
from p5_scoring import (  # noqa: E402
    StoredFact,
    proactive_posts,
    score_extraction,
    score_supersession,
)
from p5_supersession_cases import Pair, all_pairs  # noqa: E402


class TestTheCaseSets:
    def test_extraction_has_at_least_120_sound_batches_german_first(self) -> None:
        cases = all_cases()

        assert len(cases) >= 120
        assert len({case.name for case in cases}) == len(cases)
        assert [problem for case in cases for problem in validate_case(case)] == []
        locales = [case.locale for case in cases]
        assert locales.count("de") > len(cases) / 2
        assert {"en", "ja", "pt"} <= set(locales)
        assert all(set(case.shapes) <= SHAPES for case in cases)

    def test_every_expected_fact_has_something_to_check(self) -> None:
        for case in all_cases():
            for expected in case.expected:
                assert expected.details or expected.conditions, case.name

    def test_supersession_has_at_least_120_judgements_mostly_boundary(self) -> None:
        pairs = all_pairs()

        assert len(pairs) >= 120
        assert len({pair.name for pair in pairs}) == len(pairs)
        assert {pair.category for pair in pairs} == {
            "supersession",
            "complementary",
            "contradiction",
            "independent",
        }
        assert sum(pair.boundary for pair in pairs) > len(pairs) / 2

    def test_proactive_has_at_least_350_messages_and_consistent_labels(self) -> None:
        messages = all_messages()

        assert len(messages) >= 350
        for _scenario, _index, message in messages:
            expected = message.category in POSITIVE and not message.human_reply
            assert message.should_post is expected
        injections = [m for _, _, m in messages if m.category == "injection"]
        assert len(injections) >= 30
        assert not any(m.should_post for m in injections)

    def test_the_dev_split_is_a_stable_quarter(self) -> None:
        dev = [case.is_dev for case in all_cases()]

        assert 0.15 < sum(dev) / len(dev) < 0.35


def _case(**overrides: object) -> ExtractionCase:
    values: dict[str, object] = {
        "name": "t",
        "locale": "de",
        "channel": "c",
        "start": at(2026, 10, 1, 12),
        "messages": (
            Chat("a", 0, "Morgen um 20 Uhr ist Quiz, nur für Mitglieder."),
            Chat("b", 1, "lol"),
            Chat("a", 2, "Korrektur: 21 Uhr."),
        ),
        "expected": (
            Expected(
                1,
                details=(("quiz",),),
                conditions=(("2. Oktober",), ("nur",)),
                forbidden=("um 19",),
            ),
        ),
        "must_not_store": (2,),
        "optional": (3,),
    }
    values.update(overrides)
    return ExtractionCase(**values)  # type: ignore[arg-type]


class TestExtractionScoring:
    def test_a_correct_fact_is_recalled_complete_and_not_false(self) -> None:
        score = score_extraction(
            _case(), [StoredFact(1, "Das Quiz ist am 2. Oktober um 20 Uhr, nur für Mitglieder.")]
        )

        assert (score.expected, score.stored, score.stored_correct, score.complete) == (1, 1, 1, 1)
        assert score.false_facts == []

    def test_a_fact_from_a_must_not_store_message_is_false(self) -> None:
        score = score_extraction(_case(), [StoredFact(2, "Es wurde gelacht.")])

        assert [why for _, _, why in score.false_facts] == ["must_not_store"]

    def test_a_dropped_condition_is_false_and_not_a_correct_recall(self) -> None:
        score = score_extraction(_case(), [StoredFact(1, "Das Quiz ist am 2. Oktober um 20 Uhr.")])

        assert score.stored == 1 and score.stored_correct == 0
        assert score.false_facts[0][2].startswith("condition_missing:nur")

    def test_a_split_message_is_judged_on_all_its_facts_together(self) -> None:
        score = score_extraction(
            _case(),
            [
                StoredFact(1, "Das Quiz ist am 2. Oktober um 20 Uhr."),
                StoredFact(1, "Das Quiz ist nur für Mitglieder."),
            ],
        )

        assert score.false_facts == []
        assert score.stored_correct == 1

    def test_a_forbidden_value_is_false(self) -> None:
        score = score_extraction(
            _case(), [StoredFact(1, "Das Quiz ist am 2. Oktober um 19 Uhr, nur Mitglieder.")]
        )

        assert any("forbidden" in why for _, _, why in score.false_facts)

    def test_an_optional_fact_counts_neither_way(self) -> None:
        score = score_extraction(_case(), [StoredFact(3, "Das Quiz beginnt um 21 Uhr.")])

        assert score.false_facts == []
        assert score.optional_stored == [(3, "Das Quiz beginnt um 21 Uhr.")]
        assert score.stored == 0

    def test_a_failed_call_is_a_failure_not_an_empty_answer(self) -> None:
        score = score_extraction(_case(), None)

        assert score.failed is True and score.stored == 0

    def test_matching_is_case_and_whitespace_insensitive(self) -> None:
        assert group_present("Das  QUIZ am 2.  Oktober", ("2. oktober",))
        assert not group_present("am 12. Oktober", ("2. november",))


class TestSupersessionScoring:
    pair = Pair("p", "complementary", "A.", "B.", "different-detail")

    def test_a_replacement_where_none_is_labelled_is_the_dangerous_error(self) -> None:
        score = score_supersession(self.pair, "supersession")

        assert score.wrong_replacement is True and score.correct is False

    def test_a_failure_is_incorrect_and_counted_as_failed(self) -> None:
        score = score_supersession(Pair("q", "supersession", "A.", "B.", "status-flip"), None)

        assert score.failed and not score.correct and score.missed_replacement


class TestProactiveDecision:
    def test_a_human_reply_always_wins(self) -> None:
        assert proactive_posts(True, human_reply=True, check_passed=True) is False

    def test_the_check_must_pass_when_it_ran(self) -> None:
        assert proactive_posts(True, human_reply=False, check_passed=False) is False
        assert proactive_posts(True, human_reply=False, check_passed=True) is True
        assert proactive_posts(True, human_reply=False, check_passed=None) is True
        assert proactive_posts(False, human_reply=False, check_passed=True) is False


class TestThePaidEntryPointsRefuse:
    def test_the_main_harness_needs_the_explicit_opt_in(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import p5_bakeoff

        monkeypatch.delenv("AURA_RUN_REAL_LLM", raising=False)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "p5_bakeoff.py",
                "--ledger",
                "x",
                "--bucket",
                "dev",
                "--label",
                "l",
                "--out-dir",
                "o",
                "extraction",
                "--arm",
                "haiku",
            ],
        )

        assert p5_bakeoff.main() == 2

    @pytest.mark.parametrize("arm", ["qwen", "gpt6luna", "flashlite", "mimo", "sonnet"])
    def test_the_shadow_harness_refuses_unapproved_providers(
        self, monkeypatch: pytest.MonkeyPatch, arm: str, tmp_path: Path
    ) -> None:
        import p5_shadow

        inside = tmp_path / "reports" / "shadow"
        inside.mkdir(parents=True)

        monkeypatch.setenv("AURA_RUN_REAL_LLM", "1")
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "p5_shadow.py",
                "--shadow-dir",
                str(inside),
                "--ledger",
                "x",
                "--label",
                "l",
                "--out-dir",
                str(inside),
                "extraction",
                "--arm",
                arm,
            ],
        )

        assert p5_shadow.main() == 2

    def test_the_shadow_harness_refuses_a_place_outside_reports(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        import p5_shadow

        monkeypatch.setenv("AURA_RUN_REAL_LLM", "1")
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "p5_shadow.py",
                "--shadow-dir",
                str(tmp_path),
                "--ledger",
                "x",
                "--label",
                "l",
                "--out-dir",
                str(tmp_path),
                "extraction",
                "--arm",
                "haiku",
            ],
        )

        assert p5_shadow.main() == 2

    def test_the_approved_providers_are_exactly_the_operators_list(self) -> None:
        import p5_shadow

        assert {
            "haiku",
            "deepseek-2p",
            "deepseek-2p-think",
            "gemini38-vertex",
            "glm",
        } == p5_shadow.APPROVED_ARMS
