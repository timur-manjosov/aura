"""Tests for the P4 bake-off harness: the metering ledger, the statistics, the scoring, the cases.

Measurement code, but code whose mistakes would either spend money past a
ceiling or report a result the data does not support -- so it is tested like
production code. Nothing here calls a model: litellm.acompletion is replaced by
a scripted fake under the metering wrapper.
"""

from __future__ import annotations

import asyncio
import json
import multiprocessing
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import litellm
import pytest

from answer_contract_cases import CASES, SHAPES, UNCLEAR_FORBIDDEN
from answer_contract_scoring import (
    SAFETY_FAILURES,
    contains_phrase,
    detect_language,
    has_cross_attribution,
    score_contract,
    score_legacy,
)
from bakeoff_arms import ARMS, prices
from bakeoff_stats import (
    blind_shuffle,
    intervals_overlap,
    mcnemar_exact,
    paired_bootstrap_difference,
    percentile,
    wilson_interval,
)
from extraction_german_cases import GERMAN_BATCHES
from grounding_verification_cases import ALL_CASES, CONTROL_CASES, FORGED_CASES, INVENTED_FACTS
from llm_metering import (
    HARNESS_MAX_TOKENS,
    ArmRouting,
    CeilingReachedError,
    Ledger,
    ModelPrice,
    install_metering,
    worst_case_usd,
)
from proactive_decision_cases import POSITIVE_CATEGORIES, SCENARIOS, all_messages

PRICE = ModelPrice(1.0, 5.0)
MODEL = "openrouter/fake/model"
MESSAGES = [{"role": "user", "content": "x" * 1000}]  # worst case: 1000 tokens in
WORST = worst_case_usd(PRICE, MESSAGES, 100)  # = 0.0015


def _response(cost: float | None = 0.0001, provider: str = "Prov") -> SimpleNamespace:
    usage = SimpleNamespace(
        prompt_tokens=10,
        completion_tokens=5,
        completion_tokens_details=SimpleNamespace(reasoning_tokens=3),
        cost=cost,
        model_extra={},
    )
    message = SimpleNamespace(content='{"ok": true}')
    return SimpleNamespace(
        usage=usage,
        choices=[SimpleNamespace(message=message, finish_reason="stop")],
        provider=provider,
    )


@pytest.fixture
def fake_completion(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    async def fake(**kwargs: Any) -> SimpleNamespace:
        calls.append(kwargs)
        await asyncio.sleep(0.01)
        return _response()

    monkeypatch.setattr(litellm, "acompletion", fake)
    return calls


def _ledger(tmp_path: Path, *, bucket: float = 1.0, total: float = 2.0) -> Ledger:
    return Ledger(tmp_path / "ledger.json", {"dev": bucket, "other": 1.0}, total)


async def _call(max_tokens: int | None = 100, **extra: Any) -> Any:
    kwargs: dict[str, Any] = {"model": MODEL, "messages": MESSAGES, **extra}
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    return await litellm.acompletion(**kwargs)


class TestTheLedger:
    async def test_a_call_is_recorded_with_the_providers_own_cost(
        self, tmp_path: Path, fake_completion: list[dict[str, Any]]
    ) -> None:
        ledger = _ledger(tmp_path)
        records: dict[str, Any] = {}
        restore = install_metering(
            ledger, bucket="dev", label="t", records=records, prices={MODEL: PRICE}
        )
        try:
            await _call()
        finally:
            restore()

        state = ledger.snapshot()
        assert state.spent() == pytest.approx(0.0001)
        assert state.calls[0]["usd_source"] == "provider usage.cost"
        assert state.calls[0]["provider"] == "Prov"
        assert state.calls[0]["reasoning_tokens"] == 3
        assert state.reservations == {}
        assert fake_completion[0]["extra_body"] == {"usage": {"include": True}}

    async def test_without_a_reported_cost_the_tokens_are_priced(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def fake(**_: Any) -> SimpleNamespace:
            return _response(cost=None)

        monkeypatch.setattr(litellm, "acompletion", fake)
        ledger = _ledger(tmp_path)
        restore = install_metering(
            ledger, bucket="dev", label="t", records={}, prices={MODEL: PRICE}
        )
        try:
            await _call()
        finally:
            restore()

        assert ledger.snapshot().spent() == pytest.approx((10 * 1.0 + 5 * 5.0) / 1e6)

    async def test_a_call_that_could_cross_the_bucket_ceiling_is_refused_before_it_leaves(
        self, tmp_path: Path, fake_completion: list[dict[str, Any]]
    ) -> None:
        ledger = _ledger(tmp_path, bucket=WORST * 0.99)
        restore = install_metering(
            ledger, bucket="dev", label="t", records={}, prices={MODEL: PRICE}
        )
        try:
            with pytest.raises(CeilingReachedError, match="'dev' ceiling"):
                await _call()
        finally:
            restore()

        assert fake_completion == []
        assert ledger.snapshot().calls == []

    async def test_the_total_ceiling_binds_across_buckets(
        self, tmp_path: Path, fake_completion: list[dict[str, Any]]
    ) -> None:
        ledger = Ledger(tmp_path / "l.json", {"a": 1.0, "b": 1.0}, WORST * 1.5)
        for bucket in ("a", "b"):
            restore = install_metering(
                ledger, bucket=bucket, label=bucket, records={}, prices={MODEL: PRICE}
            )
            try:
                if bucket == "a":
                    await _call()
                else:
                    await _call()
                    with pytest.raises(CeilingReachedError, match="total ceiling"):
                        for _ in range(50):
                            await _call()
            finally:
                restore()

        assert ledger.snapshot().spent() <= WORST * 1.5

    async def test_parallel_workers_never_overshoot_the_ceiling(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def expensive(**_: Any) -> SimpleNamespace:
            await asyncio.sleep(0.02)
            return _response(cost=WORST)  # every call really costs its worst case

        monkeypatch.setattr(litellm, "acompletion", expensive)
        ceiling = WORST * 7.5
        ledger = _ledger(tmp_path, bucket=ceiling, total=10.0)
        restore = install_metering(
            ledger, bucket="dev", label="t", records={}, prices={MODEL: PRICE}
        )

        async def one() -> bool:
            try:
                await _call()
            except CeilingReachedError:
                return False
            return True

        try:
            results = await asyncio.gather(*(one() for _ in range(40)))
        finally:
            restore()

        assert sum(results) == 7
        assert ledger.snapshot().spent() <= ceiling

    def test_parallel_processes_on_one_file_never_overshoot_the_ceiling(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "shared.json"
        ceiling = 0.0105  # room for exactly ten reservations of 0.001
        script = (
            "import sys, time\n"
            f"sys.path.insert(0, {str(Path(__file__).resolve().parent.parent / 'scripts')!r})\n"
            "from llm_metering import Ledger, CeilingReachedError\n"
            f"ledger = Ledger(__import__('pathlib').Path({str(path)!r}), {{'dev': {ceiling}}}, 1.0)\n"
            "granted = 0\n"
            "for _ in range(40):\n"
            "    try:\n"
            "        rid = ledger.reserve(bucket='dev', usd=0.001, model='m', label='p', tag='t')\n"
            "    except CeilingReachedError:\n"
            "        continue\n"
            "    time.sleep(0.002)\n"
            "    ledger.record(rid, {'bucket': 'dev', 'usd': 0.001, 'label': 'p', 'model': 'm'})\n"
            "    granted += 1\n"
            "print(granted)\n"
        )
        processes = [
            subprocess.Popen([sys.executable, "-c", script], stdout=subprocess.PIPE, text=True)
            for _ in range(4)
        ]
        granted = sum(int(p.communicate(timeout=60)[0].strip()) for p in processes)
        state = Ledger(path, {"dev": ceiling}, 1.0).snapshot()

        assert granted == 10
        assert state.spent() == pytest.approx(0.01)
        assert state.spent() <= ceiling
        assert state.reservations == {}

    def test_a_reservation_left_by_a_dead_process_is_booked_at_its_worst_case(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "crash.json"
        context = multiprocessing.get_context("spawn")
        child = context.Process(target=_reserve_and_die, args=(str(path),))
        child.start()
        child.join(timeout=60)
        assert child.exitcode == 3

        state = Ledger(path, {"dev": 1.0}, 1.0).snapshot()

        assert state.reservations == {}
        assert state.spent() == pytest.approx(0.25)
        assert state.calls[0]["finish_reason"] == "orphaned-reservation"

    async def test_a_resumed_ledger_keeps_earlier_spend_and_does_not_double_count(
        self, tmp_path: Path, fake_completion: list[dict[str, Any]]
    ) -> None:
        first = _ledger(tmp_path)
        restore = install_metering(
            first, bucket="dev", label="a", records={}, prices={MODEL: PRICE}
        )
        try:
            await _call()
        finally:
            restore()
        second = _ledger(tmp_path)

        assert second.snapshot().spent() == pytest.approx(0.0001)
        assert len(second.snapshot().calls) == 1

    async def test_a_call_that_raises_is_booked_at_its_worst_case(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def broken(**_: Any) -> None:
            raise ConnectionError("network")

        monkeypatch.setattr(litellm, "acompletion", broken)
        ledger = _ledger(tmp_path)
        restore = install_metering(
            ledger, bucket="dev", label="t", records={}, prices={MODEL: PRICE}
        )
        try:
            with pytest.raises(ConnectionError):
                await _call()
        finally:
            restore()

        state = ledger.snapshot()
        assert state.spent() == pytest.approx(WORST)
        assert state.calls[0]["finish_reason"] == "error:ConnectionError"
        assert state.reservations == {}

    async def test_a_model_without_a_price_is_refused(
        self, tmp_path: Path, fake_completion: list[dict[str, Any]]
    ) -> None:
        restore = install_metering(
            _ledger(tmp_path), bucket="dev", label="t", records={}, prices={}
        )
        try:
            with pytest.raises(CeilingReachedError, match="no known price"):
                await _call()
        finally:
            restore()
        assert fake_completion == []

    async def test_an_unknown_bucket_is_refused(
        self, tmp_path: Path, fake_completion: list[dict[str, Any]]
    ) -> None:
        restore = install_metering(
            _ledger(tmp_path), bucket="nope", label="t", records={}, prices={MODEL: PRICE}
        )
        try:
            with pytest.raises(CeilingReachedError, match="unknown budget bucket"):
                await _call()
        finally:
            restore()

    async def test_a_call_without_an_output_ceiling_gets_the_harness_bound(
        self, tmp_path: Path, fake_completion: list[dict[str, Any]]
    ) -> None:
        ledger = _ledger(tmp_path)
        restore = install_metering(
            ledger, bucket="dev", label="t", records={}, prices={MODEL: PRICE}
        )
        try:
            await _call(max_tokens=None)
        finally:
            restore()

        assert fake_completion[0]["max_tokens"] == HARNESS_MAX_TOKENS
        assert ledger.snapshot().calls[0]["harness_max_tokens_added"] is True

    async def test_an_arm_adds_its_routing_and_overrides(
        self, tmp_path: Path, fake_completion: list[dict[str, Any]]
    ) -> None:
        routing = ArmRouting(
            provider_order=("DeepInfra",),
            data_collection_deny=True,
            reasoning={"enabled": False},
            max_tokens=4000,
            temperature=0.3,
        )
        restore = install_metering(
            _ledger(tmp_path),
            bucket="dev",
            label="t",
            records={},
            prices={MODEL: PRICE},
            routing={MODEL: routing},
        )
        try:
            await _call(temperature=0.0)
        finally:
            restore()

        sent = fake_completion[0]
        assert sent["max_tokens"] == 4000
        assert sent["temperature"] == 0.3
        assert sent["extra_body"] == {
            "usage": {"include": True},
            "provider": {
                "order": ["DeepInfra"],
                "allow_fallbacks": False,
                "data_collection": "deny",
            },
            "reasoning": {"enabled": False},
        }

    def test_the_file_is_never_half_written(self, tmp_path: Path) -> None:
        ledger = _ledger(tmp_path)
        rid = ledger.reserve(bucket="dev", usd=0.01, model="m", label="l", tag="t")
        ledger.record(rid, {"bucket": "dev", "usd": 0.01, "label": "l", "model": "m"})

        payload = json.loads((tmp_path / "ledger.json").read_text(encoding="utf-8"))
        assert payload["spent_usd"] == pytest.approx(0.01)
        assert not list(tmp_path.glob("*.tmp"))

    def test_every_arm_has_a_price(self) -> None:
        table = prices()
        assert all(arm.model in table for arm in ARMS.values())
        assert all(price.input_usd > 0 and price.output_usd > 0 for price in table.values())


def _reserve_and_die(path: str) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    from llm_metering import Ledger as ChildLedger

    ChildLedger(Path(path), {"dev": 1.0}, 1.0).reserve(
        bucket="dev", usd=0.25, model="m", label="l", tag="t"
    )
    os._exit(3)


class TestStatistics:
    @pytest.mark.parametrize(
        ("k", "n", "low", "high"),
        [
            (0, 10, 0.0, 0.2775),
            (10, 10, 0.7225, 1.0),
            (5, 10, 0.2366, 0.7634),
            (58, 58, 0.9379, 1.0),
        ],
    )
    def test_wilson_matches_known_values(self, k: int, n: int, low: float, high: float) -> None:
        lo, hi = wilson_interval(k, n)

        assert lo == pytest.approx(low, abs=1e-3)
        assert hi == pytest.approx(high, abs=1e-3)

    def test_wilson_refuses_impossible_counts_and_handles_zero_trials(self) -> None:
        assert wilson_interval(0, 0) == (0.0, 1.0)
        for k, n in ((-1, 3), (4, 3), (0, -1)):
            with pytest.raises(ValueError):
                wilson_interval(k, n)

    @pytest.mark.parametrize(
        ("a", "b", "p"),
        [(0, 0, 1.0), (5, 0, 0.0625), (0, 6, 0.03125), (3, 3, 1.0), (10, 2, 0.03857)],
    )
    def test_exact_mcnemar(self, a: int, b: int, p: float) -> None:
        assert mcnemar_exact(a, b) == pytest.approx(p, abs=1e-4)

    def test_the_paired_bootstrap_is_reproducible_and_brackets_the_mean(self) -> None:
        a = [1.0] * 30 + [0.0] * 10
        b = [1.0] * 20 + [0.0] * 20
        first = paired_bootstrap_difference(a, b, seed=7)

        assert first == paired_bootstrap_difference(a, b, seed=7)
        mean, low, high = first
        assert mean == pytest.approx(0.25)
        assert low <= mean <= high
        with pytest.raises(ValueError):
            paired_bootstrap_difference([1.0], [1.0, 0.0])

    def test_overlap_and_percentile(self) -> None:
        assert intervals_overlap((0.1, 0.5), (0.5, 0.9))
        assert not intervals_overlap((0.1, 0.4), (0.5, 0.9))
        assert percentile([5, 1, 3, 2, 4], 0.95) == 5
        assert percentile([5, 1, 3, 2, 4], 0.5) == 3

    def test_the_blind_shuffle_hides_the_arm_and_keeps_the_key(self) -> None:
        outputs = [("haiku", "A text"), ("deepseek", "B text"), ("qwen", "C text")]
        items, key = blind_shuffle(outputs, seed=11)

        assert [item.label for item in items] == ["A", "B", "C"]
        assert sorted(key.values()) == ["deepseek", "haiku", "qwen"]
        assert {item.text for item in items} == {"A text", "B text", "C text"}
        assert {key[item.label]: item.text for item in items} == dict(outputs)
        assert blind_shuffle(outputs, seed=11) == (items, key)
        orders = {tuple(i.text for i in blind_shuffle(outputs, seed=s)[0]) for s in range(30)}
        assert len(orders) > 1
        with pytest.raises(ValueError):
            blind_shuffle([("a", "x"), ("a", "y")], seed=1)


def _case(name: str) -> Any:
    return next(case for case in CASES if case.name == name)


def _contract(case_name: str, **overrides: Any) -> dict[str, Any]:
    case = _case(case_name)
    base: dict[str, Any] = {
        "request_reading": "r",
        "fact_notes": [],
        "relations": [],
        "not_covered_topics": [],
        "tone": case.tones[0],
        "lead": "x",
        "points": [],
        "used_fact_numbers": list(case.must_cite),
        "answers_question": case.expected_answers_question,
    }
    if case.expected_relation:
        base["relations"] = [{"facts": list(case.must_cite), "kind": case.expected_relation}]
    base.update(overrides)
    return base


class TestScoring:
    def test_a_correct_unclear_answer_passes(self) -> None:
        score = score_contract(
            _case("de-unclear-sprechstunde"),
            _contract(
                "de-unclear-sprechstunde",
                lead="Zur Sprechstunde sind zwei Termine vermerkt: dienstags um 18 Uhr und jeden zweiten Samstag.",
            ),
        )

        assert score.passed, score.failures

    @pytest.mark.parametrize("phrase", ["zwei verschiedene", "wurde verlegt", "dasselbe"])
    def test_sameness_or_change_in_an_unclear_answer_is_a_safety_failure(self, phrase: str) -> None:
        score = score_contract(
            _case("de-unclear-sprechstunde"),
            _contract(
                "de-unclear-sprechstunde",
                lead=f"Es gibt {phrase} Sprechstunde am Dienstag um 18 Uhr und Samstag.",
            ),
        )

        assert "forbidden" in score.safety_failures

    def test_a_detail_moved_to_a_point_that_does_not_cite_it_is_cross_attribution(self) -> None:
        case = _case("de-unclear-sprechstunde")

        assert has_cross_attribution(case, [("Am Samstag um 18 Uhr ist Sprechstunde.", [2])])
        assert not has_cross_attribution(case, [("Dienstags um 18 Uhr.", [1])])

    def test_mentioning_a_fact_without_citing_it_is_a_safety_failure(self) -> None:
        score = score_contract(
            _case("en-single-shared-words"),
            _contract(
                "en-single-shared-words",
                lead="Beginners on 4 March at 15:00 UTC, pros on 11 March.",
            ),
        )

        assert "uncited_mention" in score.safety_failures

    def test_an_invented_quantifier_is_a_safety_failure_and_a_synonym_is_not(self) -> None:
        case = "de-register-formal-content-casual-question"
        invented = score_contract(
            _case(case),
            _contract(case, lead="Du kannst deine Daten jederzeit löschen lassen. Archiv 90 Tage."),
        )
        assert "invented_quantifier" in invented.safety_failures
        synonym = score_contract(
            _case("pt-injection-in-fact-sales"),
            _contract("pt-injection-in-fact-sales", lead="Sim, mas apenas no canal #mercado."),
        )
        assert "invented_quantifier" not in synonym.failures

    def test_an_invalid_reply_is_a_failure_never_a_skip(self) -> None:
        score = score_contract(_case("de-keyword-schachturnier"), None)

        assert score.failures == ["schema"]
        assert not score.passed
        assert score_legacy(_case("de-keyword-schachturnier"), None, [], None).failures == [
            "schema"
        ]

    def test_a_conflict_answer_is_judged_after_the_contracts_own_overrule(self) -> None:
        score = score_contract(
            _case("de-conflict-voice-event"),
            _contract(
                "de-conflict-voice-event",
                lead="Für das Voice-Event sind zwei Startzeiten vermerkt: 20 Uhr und 21 Uhr.",
                answers_question=True,
            ),
        )

        assert "answers_question" not in score.failures

    def test_the_ambiguous_case_accepts_either_reading(self) -> None:
        case = "de-unclear-treffpunkt"
        lead = "Für den Treffpunkt der Wandergruppe sind zwei Orte vermerkt: Bahnhof Nord und Parkplatz am Stadtwald."
        conflict = _contract(
            case,
            lead=lead,
            relations=[{"facts": [1, 2], "kind": "same_detail_conflict"}],
            answers_question=False,
        )

        assert score_contract(_case(case), conflict).passed
        assert score_contract(_case(case), _contract(case, lead=lead)).passed

    def test_phrase_matching_respects_word_boundaries_and_spaceless_scripts(self) -> None:
        assert contains_phrase("Es ist nur heute.", "nur")
        assert not contains_phrase("Die Nurse kommt.", "nur")
        assert contains_phrase("サーバーは閉鎖されました", "閉鎖")

    @pytest.mark.parametrize(
        ("text", "language"),
        [
            ("Der Filmabend ist jeden Freitag um 21 Uhr.", "de"),
            ("The event is on Friday in the evening.", "en-US"),
            ("O torneio acontece no dia 14 de junho.", "pt-BR"),
            ("配信は水曜日です。", "ja"),
            ("ok", None),
        ],
    )
    def test_language_detection(self, text: str, language: str | None) -> None:
        assert detect_language(text) == language

    def test_safety_failures_are_a_named_closed_set(self) -> None:
        assert {
            "forbidden",
            "relative_time",
            "cross_attribution",
            "uncited_mention",
            "invented_quantifier",
        } == SAFETY_FAILURES


class TestTheCaseSets:
    def test_synthesis_cases_cover_every_shape_language_and_difficulty(self) -> None:
        assert len(CASES) >= 60
        assert len({case.name for case in CASES}) == len(CASES)
        assert {case.shape for case in CASES} == set(SHAPES)
        assert {case.locale for case in CASES} == {"de", "en-US", "ja", "pt-BR"}
        assert {case.difficulty for case in CASES} == {"easy", "medium", "hard"}
        assert {case.register for case in CASES} == {"casual", "neutral", "formal"}

    def test_every_synthesis_case_is_well_formed(self) -> None:
        for case in CASES:
            numbers = set(range(1, len(case.facts) + 1))
            assert set(case.must_cite) <= numbers, case.name
            assert set(case.must_not_cite) <= numbers, case.name
            assert not set(case.must_cite) & set(case.must_not_cite), case.name
            assert all(fact.markers for fact in case.facts), case.name
            if case.expected_relation:
                assert len(case.must_cite) >= 2, case.name
            if case.shape == "unclear":
                assert set(UNCLEAR_FORBIDDEN[case.locale]) <= set(case.forbidden), case.name

    def test_no_case_uses_a_topic_of_the_prompts_worked_examples(self) -> None:
        text = " ".join(
            f"{c.question} {' '.join(f.text for f in c.facts)}" for c in CASES
        ).casefold()
        for topic in ("bastel", "emoji", "map rotation", "craft circle"):
            assert topic not in text

    def test_the_proactive_set_has_150_labeled_messages_of_every_category(self) -> None:
        messages = all_messages()
        assert len(messages) >= 150
        assert {message.category for _, _, message in messages} >= {
            "hit",
            "near_miss",
            "hard_negative",
            "conflict",
            "unclear",
            "injection",
            "elliptical",
        }
        for _, _, message in messages:
            assert message.should_post == (message.category in POSITIVE_CATEGORIES)
        assert all(len(scenario.facts) >= 10 for scenario in SCENARIOS)

    def test_the_grounding_corpus_is_the_134_case_set_from_invented_facts(self) -> None:
        assert len(ALL_CASES) == 134
        assert len(FORGED_CASES) == 76
        assert len(CONTROL_CASES) == 58
        assert len({case.name for case in ALL_CASES}) == 134
        for case in ALL_CASES:
            assert all(key in INVENTED_FACTS for key in case.fact_keys)
            assert (case.finding is not None) == (not case.expected_grounded)

    def test_the_german_extraction_batches_each_carry_a_control(self) -> None:
        for batch in GERMAN_BATCHES:
            assert batch.locale == "de"
            assert any(message.expect_fact for message in batch.messages), batch.name
