"""Tests for /aura-ask's cost bounds: the daily caps, the free answer, the operator brake.

Command callback invoked directly against mocked discord.Interaction objects
and a real in-memory database, as in test_ask_command.py. synthesize_answer and
verify_answer_grounded are mocked on the paid path; on the free path they, and
litellm itself, are tripwires -- the free answer must not reach a model at all.
"""

from __future__ import annotations

import logging
import re
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import aiosqlite
import discord
import pytest
from fastembed import TextEmbedding

from aura.billing import PlanGate
from aura.billing.entitlement import GracePolicy
from aura.commands.ask import ask_command
from aura.config import Settings
from aura.db.connection import utc_day, utc_now
from aura.db.models import Fact
from aura.db.proactive_state import try_acquire_escalation_slot
from aura.db.repository import init_schema
from aura.facts_service import add_fact
from aura.grounding import GroundingOutcome
from aura.synthesis import SynthesisResult

GUILD_A = 100000000000000001
GUILD_B = 200000000000000002
ALICE = 111
BOB = 222


@pytest.fixture
async def conn():
    connection = await aiosqlite.connect(":memory:")
    await init_schema(connection)
    yield connection
    await connection.close()


def _settings(**overrides: object) -> Settings:
    defaults: dict[str, object] = {
        "discord_token": "fake-token",
        "llm_api_key": "fake-key",
        "synthesis_model": "openrouter/fake/model",
        "similarity_threshold": 0.0,
    }
    defaults.update(overrides)
    return Settings(_env_file=None, **defaults)  # type: ignore[arg-type]


def _gate(*, pro: bool) -> MagicMock:
    gate = MagicMock(spec=PlanGate)
    gate.allows_pro = MagicMock(return_value=pro)
    return gate


def _interaction(
    conn: aiosqlite.Connection,
    embedding_model: TextEmbedding,
    *,
    settings: Settings | None = None,
    plan_gate: object | None = None,
    guild_id: int = GUILD_A,
    user_id: int = ALICE,
    locale: str = "en-US",
) -> MagicMock:
    interaction = MagicMock(spec=discord.Interaction)
    interaction.locale = locale
    interaction.guild_id = guild_id
    interaction.channel_id = 42
    interaction.user = MagicMock()
    interaction.user.id = user_id
    interaction.created_at = datetime.now(UTC)
    interaction.client = MagicMock()
    interaction.client.db = conn
    interaction.client.embedding_model = embedding_model
    interaction.client.settings = settings or _settings()
    interaction.client.plan_gate = plan_gate if plan_gate is not None else _gate(pro=True)
    interaction.response = MagicMock()
    interaction.response.defer = AsyncMock()
    interaction.response.is_done = MagicMock(return_value=True)
    interaction.followup = MagicMock()
    interaction.followup.send = AsyncMock()
    interaction.delete_original_response = AsyncMock()
    return interaction


async def _ask(interaction: MagicMock, question: str = "When does the event start?") -> None:
    await ask_command.callback(interaction, question)  # type: ignore[call-arg, arg-type]  # pyright: ignore


async def _add(
    conn: aiosqlite.Connection,
    embedding_model: TextEmbedding,
    content: str,
    *,
    guild_id: int = GUILD_A,
    message_id: int = 1,
) -> Fact:
    return await add_fact(
        conn,
        embedding_model,
        guild_id=guild_id,
        channel_id=11,
        message_id=message_id,
        content=content,
    )


async def _rows(conn: aiosqlite.Connection, guild_id: int | None = None) -> int:
    if guild_id is None:
        query, params = "SELECT COUNT(*) FROM ask_calls", ()
    else:
        query, params = "SELECT COUNT(*) FROM ask_calls WHERE guild_id = ?", (guild_id,)
    async with conn.execute(query, params) as cursor:
        row = await cursor.fetchone()
    assert row is not None
    return int(row[0])


def _paid_mocks(fact_ids: list[int] | None = None) -> tuple[AsyncMock, AsyncMock]:
    synth = AsyncMock(
        return_value=SynthesisResult(
            answer="Saturday at 18:00.", used_fact_ids=fact_ids or [], answers_question=True
        )
    )
    ground = AsyncMock(return_value=GroundingOutcome.GROUNDED)
    return synth, ground


async def _ask_paid(interaction: MagicMock, question: str = "When?") -> tuple[AsyncMock, AsyncMock]:
    synth, ground = _paid_mocks()
    with (
        patch("aura.commands.ask.synthesize_answer", synth),
        patch("aura.commands.ask.verify_answer_grounded", ground),
    ):
        await _ask(interaction, question)
    return synth, ground


class _ModelTripwires:
    """Every route to a model, armed: any call fails the test and is recorded."""

    def __init__(self) -> None:
        self.synth = AsyncMock(side_effect=AssertionError("synthesis reached"))
        self.ground = AsyncMock(side_effect=AssertionError("grounding reached"))
        self.litellm = AsyncMock(side_effect=AssertionError("litellm reached"))

    def __enter__(self) -> _ModelTripwires:
        self._patches = [
            patch("aura.commands.ask.synthesize_answer", self.synth),
            patch("aura.commands.ask.verify_answer_grounded", self.ground),
            patch("litellm.acompletion", self.litellm),
        ]
        for active in self._patches:
            active.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        for active in self._patches:
            active.stop()

    def assert_untouched(self) -> None:
        self.synth.assert_not_awaited()
        self.ground.assert_not_awaited()
        self.litellm.assert_not_awaited()


def _sent_embed(interaction: MagicMock) -> tuple[discord.Embed, dict[str, object]]:
    _, kwargs = interaction.followup.send.call_args
    return kwargs["embed"], kwargs


class TestWhenTheSlotIsClaimed:
    async def test_a_paid_answer_claims_exactly_one_slot(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        await _add(conn, embedding_model, "The event starts on Saturday at 18:00.")
        synth, ground = await _ask_paid(_interaction(conn, embedding_model))

        synth.assert_awaited_once()
        ground.assert_awaited_once()
        assert await _rows(conn) == 1

    async def test_the_slot_is_claimed_before_synthesis_runs(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        await _add(conn, embedding_model, "The event starts on Saturday at 18:00.")
        rows_seen_by_synthesis: list[int] = []

        async def synthesize(*_args: object, **_kwargs: object) -> SynthesisResult:
            rows_seen_by_synthesis.append(await _rows(conn))
            return SynthesisResult(answer="Saturday.", used_fact_ids=[], answers_question=True)

        with (
            patch("aura.commands.ask.synthesize_answer", AsyncMock(side_effect=synthesize)),
            patch(
                "aura.commands.ask.verify_answer_grounded",
                AsyncMock(return_value=GroundingOutcome.GROUNDED),
            ),
        ):
            await _ask(_interaction(conn, embedding_model))

        assert rows_seen_by_synthesis == [1]

    async def test_a_question_that_matches_nothing_claims_nothing(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        interaction = _interaction(conn, embedding_model)
        with _ModelTripwires() as tripwires:
            await _ask(interaction)

        tripwires.assert_untouched()
        assert await _rows(conn) == 0
        args, _ = interaction.followup.send.call_args
        assert "don't have any information" in args[0]

    async def test_an_unconfigured_deployment_claims_nothing(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        await _add(conn, embedding_model, "The event starts on Saturday at 18:00.")
        settings = _settings(llm_api_key=None, synthesis_model=None)
        await _ask(_interaction(conn, embedding_model, settings=settings))
        assert await _rows(conn) == 0

    @pytest.mark.parametrize(
        ("synthesis_result", "grounding"),
        [
            (None, GroundingOutcome.GROUNDED),
            (
                SynthesisResult(answer="x", used_fact_ids=[], answers_question=True),
                GroundingOutcome.UNGROUNDED,
            ),
            (
                SynthesisResult(answer="x", used_fact_ids=[], answers_question=True),
                GroundingOutcome.CHECK_FAILED,
            ),
        ],
    )
    async def test_a_failed_answer_keeps_its_slot(
        self,
        conn: aiosqlite.Connection,
        embedding_model: TextEmbedding,
        synthesis_result: SynthesisResult | None,
        grounding: GroundingOutcome,
    ) -> None:
        await _add(conn, embedding_model, "The event starts on Saturday at 18:00.")
        with (
            patch("aura.commands.ask.synthesize_answer", AsyncMock(return_value=synthesis_result)),
            patch("aura.commands.ask.verify_answer_grounded", AsyncMock(return_value=grounding)),
        ):
            await _ask(_interaction(conn, embedding_model))
        assert await _rows(conn) == 1

    async def test_the_paid_answer_is_still_public(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        await _add(conn, embedding_model, "The event starts on Saturday at 18:00.")
        interaction = _interaction(conn, embedding_model)
        await _ask_paid(interaction)

        _, kwargs = _sent_embed(interaction)
        assert kwargs.get("ephemeral", False) is False
        interaction.delete_original_response.assert_not_awaited()


class TestCapsByPlan:
    async def test_free_the_sixth_question_of_one_member_gets_the_free_answer(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        # The shipped defaults: Free 10 per guild, 5 per member.
        await _add(conn, embedding_model, "The event starts on Saturday at 18:00.")
        for _ in range(5):
            synth, _ = await _ask_paid(
                _interaction(conn, embedding_model, plan_gate=_gate(pro=False))
            )
            synth.assert_awaited_once()

        sixth = _interaction(conn, embedding_model, plan_gate=_gate(pro=False))
        with _ModelTripwires() as tripwires:
            await _ask(sixth)

        tripwires.assert_untouched()
        embed, kwargs = _sent_embed(sixth)
        assert kwargs["ephemeral"] is True
        assert embed.description is not None
        assert embed.description.startswith("You've used up your AI answers")
        assert await _rows(conn) == 5

    async def test_free_the_guild_cap_binds_across_members(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        await _add(conn, embedding_model, "The event starts on Saturday at 18:00.")
        for user_id in range(10):
            await _ask_paid(
                _interaction(conn, embedding_model, plan_gate=_gate(pro=False), user_id=user_id)
            )
        assert await _rows(conn) == 10

        eleventh = _interaction(conn, embedding_model, plan_gate=_gate(pro=False), user_id=99)
        with _ModelTripwires() as tripwires:
            await _ask(eleventh)

        tripwires.assert_untouched()
        embed, _ = _sent_embed(eleventh)
        assert embed.description is not None
        assert embed.description.startswith("Today's AI answers on this server are used up")

    async def test_pro_has_no_member_cap_and_a_guild_cap_of_25(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        await _add(conn, embedding_model, "The event starts on Saturday at 18:00.")
        for _ in range(25):
            synth, _ = await _ask_paid(
                _interaction(conn, embedding_model, plan_gate=_gate(pro=True))
            )
            synth.assert_awaited_once()

        twenty_sixth = _interaction(conn, embedding_model, plan_gate=_gate(pro=True))
        with _ModelTripwires() as tripwires:
            await _ask(twenty_sixth)
        tripwires.assert_untouched()
        embed, _ = _sent_embed(twenty_sixth)
        assert embed.description is not None
        assert embed.description.startswith("Today's AI answers on this server are used up")

    async def test_a_plan_change_mid_day_applies_to_the_next_question(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        settings = _settings(ask_daily_cap_free=2, ask_daily_cap_pro=4, ask_user_daily_cap_free=5)
        await _add(conn, embedding_model, "The event starts on Saturday at 18:00.")
        gate = _gate(pro=False)
        for _ in range(2):
            await _ask_paid(_interaction(conn, embedding_model, settings=settings, plan_gate=gate))
        with _ModelTripwires():
            await _ask(_interaction(conn, embedding_model, settings=settings, plan_gate=gate))
        assert await _rows(conn) == 2

        gate.allows_pro.return_value = True  # upgraded to Pro
        for _ in range(2):
            synth, _ = await _ask_paid(
                _interaction(conn, embedding_model, settings=settings, plan_gate=gate)
            )
            synth.assert_awaited_once()
        with _ModelTripwires() as tripwires:
            await _ask(_interaction(conn, embedding_model, settings=settings, plan_gate=gate))
        tripwires.assert_untouched()
        assert await _rows(conn) == 4  # the two Free answers counted against Pro's 4

    async def test_the_plan_is_asked_for_the_asking_guild(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        await _add(
            conn, embedding_model, "The event starts on Saturday at 18:00.", guild_id=GUILD_B
        )
        gate = _gate(pro=True)
        await _ask_paid(_interaction(conn, embedding_model, plan_gate=gate, guild_id=GUILD_B))
        gate.allows_pro.assert_called_once_with(GUILD_B)

    async def test_disabled_billing_is_pro(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        settings = _settings(ask_daily_cap_free=0, ask_daily_cap_pro=1)
        await _add(conn, embedding_model, "The event starts on Saturday at 18:00.")
        synth, _ = await _ask_paid(
            _interaction(conn, embedding_model, settings=settings, plan_gate=PlanGate.unenforced())
        )
        synth.assert_awaited_once()

    async def test_a_complimentary_guild_is_pro(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        settings = _settings(ask_daily_cap_free=0, ask_daily_cap_pro=1)
        gate = PlanGate(
            enforced=True,
            policy=GracePolicy(renewal_grace=timedelta(0), payment_failure_grace=timedelta(0)),
            complimentary_guild_ids=frozenset({GUILD_A}),
            records=(),
        )
        await _add(conn, embedding_model, "The event starts on Saturday at 18:00.")
        synth, _ = await _ask_paid(
            _interaction(conn, embedding_model, settings=settings, plan_gate=gate)
        )
        synth.assert_awaited_once()

    async def test_an_enforced_guild_without_a_subscription_is_free(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        settings = _settings(ask_daily_cap_free=0, ask_daily_cap_pro=1)
        gate = PlanGate(
            enforced=True,
            policy=GracePolicy(renewal_grace=timedelta(0), payment_failure_grace=timedelta(0)),
            complimentary_guild_ids=frozenset(),
            records=(),
        )
        await _add(conn, embedding_model, "The event starts on Saturday at 18:00.")
        with _ModelTripwires() as tripwires:
            await _ask(_interaction(conn, embedding_model, settings=settings, plan_gate=gate))
        tripwires.assert_untouched()

    async def test_a_zero_cap_never_calls_a_model(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        settings = _settings(ask_daily_cap_pro=0)
        await _add(conn, embedding_model, "The event starts on Saturday at 18:00.")
        interaction = _interaction(conn, embedding_model, settings=settings)
        with _ModelTripwires() as tripwires:
            await _ask(interaction)
        tripwires.assert_untouched()
        assert await _rows(conn) == 0
        _, kwargs = _sent_embed(interaction)
        assert kwargs["ephemeral"] is True

    async def test_a_failing_plan_gate_falls_back_to_the_free_caps(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        settings = _settings(ask_daily_cap_free=1, ask_daily_cap_pro=25)
        gate = MagicMock(spec=PlanGate)
        gate.allows_pro = MagicMock(side_effect=RuntimeError("gate broken"))
        await _add(conn, embedding_model, "The event starts on Saturday at 18:00.")

        synth, _ = await _ask_paid(
            _interaction(conn, embedding_model, settings=settings, plan_gate=gate)
        )
        synth.assert_awaited_once()
        with _ModelTripwires() as tripwires:
            await _ask(_interaction(conn, embedding_model, settings=settings, plan_gate=gate))
        tripwires.assert_untouched()


class TestTheFreeAnswer:
    async def _capped(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding, locale: str = "en-US"
    ) -> MagicMock:
        interaction = _interaction(
            conn, embedding_model, settings=_settings(ask_daily_cap_pro=0), locale=locale
        )
        with _ModelTripwires() as tripwires:
            await _ask(interaction)
        tripwires.assert_untouched()
        return interaction

    async def test_it_lists_at_most_three_facts_with_link_and_date(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        facts = [
            await _add(conn, embedding_model, f"Event fact number {n}.", message_id=100 + n)
            for n in range(5)
        ]
        interaction = await self._capped(conn, embedding_model)

        embed, _ = _sent_embed(interaction)
        assert embed.description is not None
        lines = [line for line in embed.description.split("\n") if line.startswith("• ")]
        assert len(lines) == 3
        for line in lines:
            match = re.fullmatch(
                r"• \[.+\]\(https://discord\.com/channels/(\d+)/(\d+)/(\d+)\) · <t:(\d+):d>", line
            )
            assert match is not None, line
            assert int(match.group(1)) == GUILD_A
            fact = next(f for f in facts if f.message_id == int(match.group(3)))
            assert int(match.group(4)) == int(fact.created_at.timestamp())

    async def test_it_lists_only_the_facts_retrieval_found_best_first(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        best = await _add(conn, embedding_model, "The event starts on Saturday at 18:00.")
        await _add(conn, embedding_model, "Unrelated: the bot prefix is !.", message_id=2)
        interaction = await self._capped(conn, embedding_model)

        embed, _ = _sent_embed(interaction)
        assert embed.description is not None
        first = next(line for line in embed.description.split("\n") if line.startswith("• "))
        assert f"/{best.channel_id}/{best.message_id})" in first

    async def test_fact_text_cannot_break_out_of_its_link(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        await _add(
            conn,
            embedding_model,
            "Event ](https://evil.example) [click\n\nhere\u200b now",
        )
        interaction = await self._capped(conn, embedding_model)

        embed, _ = _sent_embed(interaction)
        assert embed.description is not None
        line = next(line for line in embed.description.split("\n") if line.startswith("• "))
        assert "\\](https://evil.example) \\[click here" in line
        assert "\n" not in line
        # The only real link target is the Discord permalink.
        assert re.findall(r"(?<!\\)\]\((\S+?)\)", line) == [line.split("](")[-1].split(")")[0]]
        assert line.split("](")[-1].startswith("https://discord.com/channels/")

    async def test_a_huge_fact_still_fits_the_embed(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        for n in range(3):
            # Brackets double when escaped; the event words keep it retrievable.
            content = "When does the event start? " + "[]" * 1985
            await _add(conn, embedding_model, content, message_id=n + 1)
        interaction = await self._capped(conn, embedding_model)
        embed, _ = _sent_embed(interaction)
        assert embed.description is not None
        assert len(embed.description) <= 4096

    async def test_it_is_ephemeral_and_replaces_the_public_deferral(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        await _add(conn, embedding_model, "The event starts on Saturday at 18:00.")
        order: list[str] = []
        interaction = _interaction(conn, embedding_model, settings=_settings(ask_daily_cap_pro=0))
        interaction.delete_original_response = AsyncMock(side_effect=lambda: order.append("delete"))
        interaction.followup.send = AsyncMock(
            side_effect=lambda *_a, **_k: order.append("followup")
        )
        with _ModelTripwires():
            await _ask(interaction)

        assert order == ["delete", "followup"]
        _, kwargs = interaction.followup.send.call_args
        assert kwargs["ephemeral"] is True

    async def test_a_failed_delete_still_answers(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        await _add(conn, embedding_model, "The event starts on Saturday at 18:00.")
        interaction = _interaction(conn, embedding_model, settings=_settings(ask_daily_cap_pro=0))
        interaction.delete_original_response = AsyncMock(
            side_effect=discord.NotFound(MagicMock(status=404), "gone")
        )
        with _ModelTripwires():
            await _ask(interaction)
        interaction.followup.send.assert_awaited_once()

    async def test_it_writes_no_ledger_row_anywhere(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        await _add(conn, embedding_model, "The event starts on Saturday at 18:00.")
        await self._capped(conn, embedding_model)
        assert await _rows(conn) == 0
        for table in (
            "proactive_escalations",
            "extraction_calls",
            "supersession_calls",
            "backfill_calls",
        ):
            async with conn.execute(f"SELECT COUNT(*) FROM {table}") as cursor:
                assert await cursor.fetchone() == (0,)

    async def test_it_says_when_the_answers_come_back(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        await _add(conn, embedding_model, "The event starts on Saturday at 18:00.")
        before = utc_now()
        interaction = await self._capped(conn, embedding_model)
        embed, _ = _sent_embed(interaction)
        assert embed.description is not None
        reset = int(re.search(r"<t:(\d+):R>", embed.description).group(1))  # type: ignore[union-attr]
        midnight = datetime.fromtimestamp(reset, UTC)
        assert (midnight.hour, midnight.minute, midnight.second) == (0, 0, 0)
        assert midnight.date() == before.date() + timedelta(days=1) or (
            # the test straddled midnight itself
            midnight.date() == before.date() + timedelta(days=2)
        )

    async def test_it_speaks_the_askers_language(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        await _add(conn, embedding_model, "The event starts on Saturday at 18:00.")
        interaction = await self._capped(conn, embedding_model, locale="de")
        embed, _ = _sent_embed(interaction)
        assert embed.description is not None
        assert embed.description.startswith("Die KI-Antworten für heute")

    async def test_it_logs_no_content(
        self,
        conn: aiosqlite.Connection,
        embedding_model: TextEmbedding,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        await _add(conn, embedding_model, "The secret event starts on Saturday at 18:00.")
        with caplog.at_level(logging.DEBUG, logger="aura.commands.ask"):
            await self._capped(conn, embedding_model)
        text = "\n".join(record.getMessage() for record in caplog.records)
        assert "secret" not in text
        assert str(GUILD_A) not in text
        assert str(GUILD_A)[:4] in text


class TestOperatorBudget:
    async def _over_budget(self, conn: aiosqlite.Connection) -> None:
        attempt = await try_acquire_escalation_slot(
            conn,
            guild_id=GUILD_B,
            channel_id=1,
            message_id=1,
            cooldown_seconds=0.0,
            daily_cap=10,
            now=utc_now(),
        )
        assert attempt.granted

    async def test_hard_mode_over_budget_gives_the_free_answer(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        await self._over_budget(conn)
        await _add(conn, embedding_model, "The event starts on Saturday at 18:00.")
        settings = _settings(cross_guild_budget_mode="hard", cross_guild_daily_budget_usd=0.0)
        interaction = _interaction(conn, embedding_model, settings=settings)

        with _ModelTripwires() as tripwires:
            await _ask(interaction)

        tripwires.assert_untouched()
        assert await _rows(conn) == 0
        embed, kwargs = _sent_embed(interaction)
        assert kwargs["ephemeral"] is True
        assert embed.description is not None
        assert embed.description.startswith("Today's AI answers on this server are used up")

    async def test_warn_mode_over_budget_proceeds_and_logs(
        self,
        conn: aiosqlite.Connection,
        embedding_model: TextEmbedding,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        await self._over_budget(conn)
        await _add(conn, embedding_model, "The event starts on Saturday at 18:00.")
        settings = _settings(cross_guild_budget_mode="warn", cross_guild_daily_budget_usd=0.0)

        with caplog.at_level(logging.WARNING, logger="aura.db.cross_guild_budget"):
            synth, _ = await _ask_paid(_interaction(conn, embedding_model, settings=settings))

        synth.assert_awaited_once()
        assert await _rows(conn) == 1
        assert any("budget exceeded" in record.getMessage() for record in caplog.records)

    async def test_paid_answers_count_toward_the_operator_budget(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        # $0.004 per ask: one answer under a $0.004 budget is not over it, the
        # second pushes the total over, and the third is refused in HARD mode.
        await _add(conn, embedding_model, "The event starts on Saturday at 18:00.")
        settings = _settings(cross_guild_budget_mode="hard", cross_guild_daily_budget_usd=0.005)
        for user_id in (1, 2):
            synth, _ = await _ask_paid(
                _interaction(conn, embedding_model, settings=settings, user_id=user_id)
            )
            synth.assert_awaited_once()
        with _ModelTripwires() as tripwires:
            await _ask(_interaction(conn, embedding_model, settings=settings, user_id=3))
        tripwires.assert_untouched()
        assert await _rows(conn) == 2


class TestQuestionLength:
    @pytest.mark.parametrize(
        ("question", "expected"),
        [
            ("a" * 999, "a" * 999),
            ("a" * 1000, "a" * 1000),
            ("a" * 1001, "a" * 1000),
            ("a" * 6000, "a" * 1000),
            ("日" * 1001, "日" * 1000),
            ("ä" * 999 + "🎉🎉", "ä" * 999 + "🎉"),
            ("", ""),
        ],
        ids=["999", "1000", "1001", "6000", "cjk-1001", "multibyte-boundary", "empty"],
    )
    async def test_the_question_is_cut_before_retrieval_and_synthesis(
        self,
        conn: aiosqlite.Connection,
        embedding_model: TextEmbedding,
        question: str,
        expected: str,
    ) -> None:
        fact = await _add(conn, embedding_model, "The event starts on Saturday at 18:00.")
        find = AsyncMock(return_value=[(fact, 0.9)])
        synth, ground = _paid_mocks()
        with (
            patch("aura.commands.ask.find_similar_facts", find),
            patch("aura.commands.ask.synthesize_answer", synth),
            patch("aura.commands.ask.verify_answer_grounded", ground),
        ):
            await _ask(_interaction(conn, embedding_model), question)

        assert find.call_args.kwargs["query"] == expected
        assert synth.call_args.args[1] == expected

    async def test_the_day_key_is_utc(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        await _add(conn, embedding_model, "The event starts on Saturday at 18:00.")
        await _ask_paid(_interaction(conn, embedding_model))
        async with conn.execute("SELECT call_day, user_id FROM ask_calls") as cursor:
            assert await cursor.fetchone() == (utc_day(utc_now()), ALICE)
