"""Tests for aura.commands.proactive: /aura-debug-signals.

Command callback, permission check and error handler invoked directly against
mocked discord.Interaction objects and a real in-memory database, matching how
test_facts_commands.py exercises the other moderator-gated commands.

The embed's exact wording is not what these assert. What they assert is that
every number the gate decided with reaches the moderator, that a
short-circuited stage is visibly absent rather than shown as a zero, and that a
full page cannot exceed Discord's limits -- the three ways a debug view can
mislead rather than merely look different.
"""
from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock

import aiosqlite
import discord
import pytest
from discord import app_commands

from aura.commands.proactive import (
    _DEFAULT_SIGNAL_LIMIT,
    _EMBED_CHARACTER_BUDGET,
    _GRACE_DISPLAY,
    _MAX_SIGNAL_LIMIT,
    _NOT_EVALUATED,
    _VERDICT_DISPLAY,
    _handle_debug_signals_error,
    debug_signals_command,
)
from aura.db.proactive_signals import (
    DecisionTrail,
    GateVerdict,
    GracePeriodOutcome,
    record_signal,
    update_grace_outcome,
    update_synthesis_outcome,
)
from aura.db.proactive_state import try_acquire_escalation_slot
from aura.db.repository import init_schema
from aura.i18n import SUPPORTED_LOCALES, t
from aura.proactive.gate import ProactiveGateConfig

GUILD_A = 100000000000000001
GUILD_B = 200000000000000002

CONFIG = ProactiveGateConfig(
    question_threshold=0.0,
    similarity_threshold=0.5,
    cooldown_seconds=900.0,
    daily_cap=20,
)

ELIGIBLE_TRAIL = DecisionTrail(
    verdict=GateVerdict.ELIGIBLE,
    stage1_score=0.123,
    stage1_passed=True,
    stage2_top_score=0.812,
    stage2_runner_up_score=0.334,
    stage2_gap=0.478,
    stage2_passed=True,
    cooldown_seconds_remaining=0.0,
    daily_count=1,
    daily_cap=20,
)

REJECTED_TRAIL = DecisionTrail(
    verdict=GateVerdict.STAGE1_REJECTED,
    stage1_score=-0.456,
    stage1_passed=False,
)


@pytest.fixture
async def conn():
    connection = await aiosqlite.connect(":memory:")
    await init_schema(connection)
    yield connection
    await connection.close()


def _make_interaction(
    *,
    db: aiosqlite.Connection | None,
    locale: str = "en-US",
    guild_id: int = GUILD_A,
    gate_config: ProactiveGateConfig | None = CONFIG,
) -> MagicMock:
    """A mock Interaction exposing just what /aura-debug-signals actually touches."""
    interaction = MagicMock(spec=discord.Interaction)
    interaction.locale = locale
    interaction.guild_id = guild_id
    interaction.client = MagicMock()
    interaction.client.db = db
    interaction.client.gate_config = gate_config
    interaction.response = MagicMock()
    interaction.response.send_message = AsyncMock()
    interaction.response.is_done = MagicMock(return_value=False)
    interaction.followup = MagicMock()
    interaction.followup.send = AsyncMock()
    interaction.command = MagicMock()
    interaction.command.name = "aura-debug-signals"
    return interaction


async def _invoke(interaction: discord.Interaction, limit: int = _DEFAULT_SIGNAL_LIMIT) -> None:
    """Call the command's callback directly, bypassing its checks.

    Same discord.py CommandCallback union-typing gap the other command tests
    document (see test_facts_commands._invoke_list_facts): at runtime this
    callback only ever takes (interaction, limit).
    """
    await debug_signals_command.callback(interaction, limit)  # pyright: ignore[reportCallIssue, reportArgumentType]


async def _seed(
    conn: aiosqlite.Connection,
    count: int,
    *,
    guild_id: int = GUILD_A,
    channel_id: int = 4242,
    decision: DecisionTrail = REJECTED_TRAIL,
) -> None:
    for message_id in range(count):
        await record_signal(
            conn,
            guild_id=guild_id,
            channel_id=channel_id,
            message_id=message_id,
            decision=decision,
        )


def _sent_embed(interaction: MagicMock) -> discord.Embed:
    _, kwargs = interaction.response.send_message.call_args
    return kwargs["embed"]


class TestPermissionCheck:
    def test_rejects_a_non_moderator(self) -> None:
        fake_interaction = MagicMock(permissions=discord.Permissions(manage_guild=False))
        with pytest.raises(app_commands.MissingPermissions):
            for check in debug_signals_command.checks:
                check(fake_interaction)

    def test_allows_a_moderator(self) -> None:
        fake_interaction = MagicMock(permissions=discord.Permissions(manage_guild=True))
        for check in debug_signals_command.checks:
            assert check(fake_interaction) is True

    def test_the_command_is_guild_only(self) -> None:
        assert debug_signals_command.guild_only is True


class TestErrorHandler:
    async def test_missing_permissions_replies_ephemerally_and_localized(self) -> None:
        interaction = _make_interaction(db=None)

        await _handle_debug_signals_error(interaction, app_commands.MissingPermissions(["manage_guild"]))

        interaction.response.send_message.assert_awaited_once()
        args, kwargs = interaction.response.send_message.call_args
        assert "permission" in args[0].lower()
        assert kwargs.get("ephemeral") is True

    async def test_missing_permissions_uses_followup_if_already_responded(self) -> None:
        interaction = _make_interaction(db=None)
        interaction.response.is_done = MagicMock(return_value=True)

        await _handle_debug_signals_error(interaction, app_commands.MissingPermissions(["manage_guild"]))

        interaction.followup.send.assert_awaited_once()
        interaction.response.send_message.assert_not_awaited()

    async def test_unexpected_errors_are_logged_and_not_surfaced_to_the_user(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        interaction = _make_interaction(db=None)
        error = app_commands.CommandInvokeError(MagicMock(), ValueError("boom"))

        with caplog.at_level(logging.ERROR):
            await _handle_debug_signals_error(interaction, error)

        interaction.response.send_message.assert_not_awaited()
        interaction.followup.send.assert_not_awaited()
        assert any(record.levelno == logging.ERROR for record in caplog.records)


class TestEmptyState:
    async def test_no_signals_shows_a_localized_empty_message(
        self, conn: aiosqlite.Connection
    ) -> None:
        interaction = _make_interaction(db=conn)

        await _invoke(interaction)

        interaction.response.send_message.assert_awaited_once()
        args, kwargs = interaction.response.send_message.call_args
        assert "no question signals" in args[0].lower()
        assert kwargs.get("ephemeral") is True
        assert "embed" not in kwargs


class TestDecisionTrailRendering:
    async def test_an_eligible_decision_shows_every_stage_and_the_verdict(
        self, conn: aiosqlite.Connection
    ) -> None:
        await record_signal(
            conn, guild_id=GUILD_A, channel_id=11, message_id=22, decision=ELIGIBLE_TRAIL
        )
        interaction = _make_interaction(db=conn)

        await _invoke(interaction)

        [field] = _sent_embed(interaction).fields
        assert t("debug_signals_verdict_eligible", "en-US") in (field.name or "")
        value = field.value or ""
        assert "+0.123" in value  # Stage 1 score, with its sign
        assert "+0.812" in value  # Stage 2 top score
        assert "0.478" in value  # the confidence gap
        assert "1/20" in value  # daily cap usage
        assert f"https://discord.com/channels/{GUILD_A}/11/22" in value

    async def test_the_stage_one_outcome_is_shown_and_not_left_to_be_inferred(
        self, conn: aiosqlite.Connection
    ) -> None:
        # Phase 2a-1 showed the score alone because no threshold existed. Now
        # one does, and a moderator cannot see it -- so pass/fail has to be on
        # the row next to the number.
        await _seed(conn, 1, decision=REJECTED_TRAIL)
        interaction = _make_interaction(db=conn)

        await _invoke(interaction)

        [field] = _sent_embed(interaction).fields
        assert "-0.456" in (field.value or "")
        assert "✗" in (field.value or "")

    async def test_a_passing_stage_shows_a_pass_marker(
        self, conn: aiosqlite.Connection
    ) -> None:
        await _seed(conn, 1, decision=ELIGIBLE_TRAIL)
        interaction = _make_interaction(db=conn)

        await _invoke(interaction)

        assert "✓" in (_sent_embed(interaction).fields[0].value or "")

    async def test_an_unevaluated_stage_renders_as_absent_not_as_zero(
        self, conn: aiosqlite.Connection
    ) -> None:
        # The distinction that makes the trail worth reading: a Stage 2 that
        # never ran must not look like a Stage 2 that scored 0.000.
        await _seed(conn, 1, decision=REJECTED_TRAIL)
        interaction = _make_interaction(db=conn)

        await _invoke(interaction)

        value = _sent_embed(interaction).fields[0].value or ""
        assert _NOT_EVALUATED in value
        assert "0.000" not in value

    async def test_an_active_cooldown_is_reported_with_its_remaining_time(
        self, conn: aiosqlite.Connection
    ) -> None:
        trail = DecisionTrail(
            verdict=GateVerdict.COOLDOWN_ACTIVE,
            stage1_score=0.2,
            stage1_passed=True,
            stage2_top_score=0.9,
            stage2_runner_up_score=0.1,
            stage2_gap=0.8,
            stage2_passed=True,
            cooldown_seconds_remaining=842.7,
            daily_count=4,
            daily_cap=20,
        )
        await _seed(conn, 1, decision=trail)
        interaction = _make_interaction(db=conn)

        await _invoke(interaction)

        [field] = _sent_embed(interaction).fields
        assert "843" in (field.value or "")  # rounded for display
        assert t("debug_signals_verdict_cooldown_active", "en-US") in (field.name or "")

    async def test_a_cleared_cooldown_says_so_rather_than_showing_zero_seconds(
        self, conn: aiosqlite.Connection
    ) -> None:
        await _seed(conn, 1, decision=ELIGIBLE_TRAIL)
        interaction = _make_interaction(db=conn)

        await _invoke(interaction)

        assert t("debug_signals_cooldown_clear", "en-US") in (
            _sent_embed(interaction).fields[0].value or ""
        )

    async def test_a_guild_with_no_facts_says_so_instead_of_showing_a_score(
        self, conn: aiosqlite.Connection
    ) -> None:
        trail = DecisionTrail(
            verdict=GateVerdict.NO_MATCHING_FACT,
            stage1_score=0.2,
            stage1_passed=True,
            stage2_top_score=None,
            stage2_passed=False,
        )
        await _seed(conn, 1, decision=trail)
        interaction = _make_interaction(db=conn)

        await _invoke(interaction)

        assert t("debug_signals_no_facts", "en-US") in (
            _sent_embed(interaction).fields[0].value or ""
        )

    async def test_a_single_fact_with_no_runner_up_renders_without_a_gap(
        self, conn: aiosqlite.Connection
    ) -> None:
        trail = DecisionTrail(
            verdict=GateVerdict.ELIGIBLE,
            stage1_score=0.2,
            stage1_passed=True,
            stage2_top_score=0.9,
            stage2_runner_up_score=None,
            stage2_gap=None,
            stage2_passed=True,
            cooldown_seconds_remaining=0.0,
            daily_count=1,
            daily_cap=20,
        )
        await _seed(conn, 1, decision=trail)
        interaction = _make_interaction(db=conn)

        await _invoke(interaction)

        value = _sent_embed(interaction).fields[0].value or ""
        assert "+0.900" in value
        assert _NOT_EVALUATED in value  # the gap, absent rather than faked as 0

    @pytest.mark.parametrize("verdict", list(GateVerdict))
    async def test_every_verdict_renders_a_label_and_never_a_blank_field(
        self, conn: aiosqlite.Connection, verdict: GateVerdict
    ) -> None:
        trail = DecisionTrail(verdict=verdict, stage1_score=0.1, stage1_passed=True)
        await _seed(conn, 1, decision=trail)
        interaction = _make_interaction(db=conn)

        await _invoke(interaction)

        name = _sent_embed(interaction).fields[0].name or ""
        assert name.strip()
        assert "[" not in name  # not a missing-translation placeholder

    def test_the_verdict_display_map_covers_every_verdict(self) -> None:
        # A missing entry renders as a bullet with a raw enum value -- legible,
        # but a silent hole in the one view that explains the gate.
        assert set(_VERDICT_DISPLAY) == set(GateVerdict)

    def test_every_verdict_label_key_exists_in_every_locale(self) -> None:
        # t() degrades to "[key]" for a missing key rather than crashing, which
        # is the right runtime behaviour and exactly why a typo here would
        # otherwise never be noticed.
        for locale in SUPPORTED_LOCALES:
            for _, key in _VERDICT_DISPLAY.values():
                assert t(key, locale) != f"[{key}]", f"{key} missing for {locale}"

    def test_every_trail_translation_key_exists_in_every_locale(self) -> None:
        keys = [
            "debug_signals_trail",
            "debug_signals_stage2",
            "debug_signals_synthesis",
            "debug_signals_synthesis_no_result",
            "debug_signals_grace_pending",
            "debug_signals_grace_cancelled_by_human",
            "debug_signals_grace_expired_and_proceeded",
            "debug_signals_grace_stood_down_on_recheck",
            "debug_signals_no_facts",
            "debug_signals_cooldown_clear",
            "debug_signals_cooldown_remaining",
            "debug_signals_cap_today",
            "debug_signals_truncated_note",
            "debug_signals_title",
            "debug_signals_footer",
            "debug_signals_jump",
        ]
        for locale in SUPPORTED_LOCALES:
            for key in keys:
                assert t(key, locale) != f"[{key}]", f"{key} missing for {locale}"

    def test_the_trail_template_consumes_every_placeholder_it_is_given(self) -> None:
        # A template missing a placeholder silently drops that whole stage from
        # the view; t() cannot warn about it, since unused kwargs are legal.
        for locale in SUPPORTED_LOCALES:
            template = t("debug_signals_trail", locale)
            for placeholder in (
                "{stage1}", "{stage2}", "{cooldown}", "{cap}", "{grace}", "{synthesis}"
            ):
                assert placeholder in template, f"{placeholder} missing for {locale}"

    def test_the_synthesis_template_consumes_both_its_placeholders(self) -> None:
        for locale in SUPPORTED_LOCALES:
            template = t("debug_signals_synthesis", locale)
            for placeholder in ("{answers}", "{posted}"):
                assert placeholder in template, f"{placeholder} missing for {locale}"


class TestSynthesisOutcomeRendering:
    """Phase 2a-3 deliverable #7: the trail shows what synthesis decided and whether Aura posted."""

    async def _seed_eligible_with_synthesis(
        self,
        conn: aiosqlite.Connection,
        *,
        answers_question: bool | None,
        posted: bool,
    ) -> None:
        await record_signal(
            conn, guild_id=GUILD_A, channel_id=7, message_id=7, decision=ELIGIBLE_TRAIL
        )
        await update_synthesis_outcome(
            conn,
            channel_id=7,
            message_id=7,
            answers_question=answers_question,
            posted=posted,
        )

    async def test_a_posted_answer_shows_both_ticks(self, conn: aiosqlite.Connection) -> None:
        await self._seed_eligible_with_synthesis(conn, answers_question=True, posted=True)
        interaction = _make_interaction(db=conn)

        await _invoke(interaction)

        value = _sent_embed(interaction).fields[0].value or ""
        # "ans ✓ · post ✓" -- both the model's yes and the actual post.
        assert value.count("✓") >= 2

    async def test_a_confident_answer_that_was_not_posted_shows_answered_but_unposted(
        self, conn: aiosqlite.Connection
    ) -> None:
        # e.g. the channel was toggled off mid-flight: the model said yes, but
        # nothing was sent. Both facts have to be visible and distinct.
        await self._seed_eligible_with_synthesis(conn, answers_question=True, posted=False)
        interaction = _make_interaction(db=conn)

        await _invoke(interaction)

        value = _sent_embed(interaction).fields[0].value or ""
        assert "✓" in value and "✗" in value

    async def test_no_synthesis_result_shows_the_no_answer_label_not_two_crosses(
        self, conn: aiosqlite.Connection
    ) -> None:
        # answers_question None means the LLM produced nothing (failed or
        # unconfigured) -- distinct from the model answering "no".
        await self._seed_eligible_with_synthesis(conn, answers_question=None, posted=False)
        interaction = _make_interaction(db=conn)

        await _invoke(interaction)

        value = _sent_embed(interaction).fields[0].value or ""
        assert t("debug_signals_synthesis_no_result", "en-US") in value

    async def test_a_non_eligible_verdict_shows_synthesis_as_not_evaluated(
        self, conn: aiosqlite.Connection
    ) -> None:
        # Only ELIGIBLE messages reach synthesis; a Stage 1 rejection must not
        # render a misleading pair of crosses in the synthesis slot.
        await _seed(conn, 1, decision=REJECTED_TRAIL)
        interaction = _make_interaction(db=conn)

        await _invoke(interaction)

        # The synthesis slot renders as the not-evaluated dash. The trail always
        # ends with " · Synth {synthesis}", so the last token is that dash here.
        value = _sent_embed(interaction).fields[0].value or ""
        synth_segment = value.split("Synth")[1]
        assert _NOT_EVALUATED in synth_segment


class TestGraceOutcomeRendering:
    """Phase 2b-1 deliverable #5: the trail shows how the grace period was resolved."""

    async def _seed_eligible_with_grace_outcome(
        self, conn: aiosqlite.Connection, outcome: GracePeriodOutcome | None
    ) -> None:
        await record_signal(
            conn, guild_id=GUILD_A, channel_id=7, message_id=7, decision=ELIGIBLE_TRAIL
        )
        if outcome is not None:
            await update_grace_outcome(conn, channel_id=7, message_id=7, outcome=outcome)

    async def test_a_still_pending_wait_shows_the_pending_label(
        self, conn: aiosqlite.Connection
    ) -> None:
        await self._seed_eligible_with_grace_outcome(conn, GracePeriodOutcome.PENDING)
        interaction = _make_interaction(db=conn)

        await _invoke(interaction)

        value = _sent_embed(interaction).fields[0].value or ""
        assert t("debug_signals_grace_pending", "en-US") in value

    async def test_a_null_grace_outcome_on_an_eligible_row_also_reads_as_pending(
        self, conn: aiosqlite.Connection
    ) -> None:
        # Reachable for a database carried over from before Phase 2b-1's
        # migration, or the brief window before the first update_grace_outcome
        # write lands -- both are honestly "no terminal outcome yet".
        await self._seed_eligible_with_grace_outcome(conn, None)
        interaction = _make_interaction(db=conn)

        await _invoke(interaction)

        value = _sent_embed(interaction).fields[0].value or ""
        assert t("debug_signals_grace_pending", "en-US") in value

    async def test_cancelled_by_human_is_visible_and_distinct_from_expiry(
        self, conn: aiosqlite.Connection
    ) -> None:
        await self._seed_eligible_with_grace_outcome(
            conn, GracePeriodOutcome.CANCELLED_BY_HUMAN
        )
        interaction = _make_interaction(db=conn)

        await _invoke(interaction)

        value = _sent_embed(interaction).fields[0].value or ""
        assert t("debug_signals_grace_cancelled_by_human", "en-US") in value
        assert t("debug_signals_grace_expired_and_proceeded", "en-US") not in value

    async def test_expired_and_proceeded_is_visible(self, conn: aiosqlite.Connection) -> None:
        await self._seed_eligible_with_grace_outcome(
            conn, GracePeriodOutcome.EXPIRED_AND_PROCEEDED
        )
        interaction = _make_interaction(db=conn)

        await _invoke(interaction)

        value = _sent_embed(interaction).fields[0].value or ""
        assert t("debug_signals_grace_expired_and_proceeded", "en-US") in value

    async def test_stood_down_on_recheck_is_visible(self, conn: aiosqlite.Connection) -> None:
        await self._seed_eligible_with_grace_outcome(
            conn, GracePeriodOutcome.STOOD_DOWN_ON_RECHECK
        )
        interaction = _make_interaction(db=conn)

        await _invoke(interaction)

        value = _sent_embed(interaction).fields[0].value or ""
        assert t("debug_signals_grace_stood_down_on_recheck", "en-US") in value

    async def test_a_non_eligible_verdict_shows_grace_as_not_evaluated(
        self, conn: aiosqlite.Connection
    ) -> None:
        # A message that never reached the gate's ELIGIBLE verdict never enters
        # a grace period at all; it must not render a pending label as if one
        # were in flight.
        await _seed(conn, 1, decision=REJECTED_TRAIL)
        interaction = _make_interaction(db=conn)

        await _invoke(interaction)

        value = _sent_embed(interaction).fields[0].value or ""
        grace_segment = value.split("Grace")[1].split("Synth")[0]
        assert _NOT_EVALUATED in grace_segment

    def test_the_grace_display_map_covers_every_outcome(self) -> None:
        assert set(_GRACE_DISPLAY) == set(GracePeriodOutcome)

    def test_every_grace_label_key_exists_in_every_locale(self) -> None:
        for locale in SUPPORTED_LOCALES:
            for key in _GRACE_DISPLAY.values():
                assert t(key, locale) != f"[{key}]", f"{key} missing for {locale}"


class TestLiveCapUsage:
    async def test_the_header_reports_todays_usage_against_the_configured_cap(
        self, conn: aiosqlite.Connection
    ) -> None:
        # The per-message figures are historical by design; a moderator asking
        # "are we capped out right now?" needs the current answer too.
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc)
        for index in range(3):
            await try_acquire_escalation_slot(
                conn,
                guild_id=GUILD_A,
                channel_id=100 + index,
                message_id=index,
                cooldown_seconds=0.0,
                daily_cap=20,
                now=now,
            )
        await _seed(conn, 1)
        interaction = _make_interaction(db=conn)

        await _invoke(interaction)

        description = _sent_embed(interaction).description or ""
        assert "3/20" in description

    async def test_another_guilds_escalations_are_not_counted(
        self, conn: aiosqlite.Connection
    ) -> None:
        from datetime import datetime, timezone

        await try_acquire_escalation_slot(
            conn,
            guild_id=GUILD_B,
            channel_id=1,
            message_id=1,
            cooldown_seconds=0.0,
            daily_cap=20,
            now=datetime.now(timezone.utc),
        )
        await _seed(conn, 1, guild_id=GUILD_A)
        interaction = _make_interaction(db=conn, guild_id=GUILD_A)

        await _invoke(interaction)

        assert "0/20" in (_sent_embed(interaction).description or "")

    async def test_a_client_without_a_gate_config_still_renders(
        self, conn: aiosqlite.Connection
    ) -> None:
        # Unreachable once setup_hook has run, but this callback is also
        # directly callable, and a debug view that crashes is worse than one
        # that admits it does not know the cap.
        await _seed(conn, 1)
        interaction = _make_interaction(db=conn, gate_config=None)

        await _invoke(interaction)

        assert _sent_embed(interaction).description


class TestRendering:
    async def test_the_reply_is_ephemeral_and_marked_as_diagnostic(
        self, conn: aiosqlite.Connection
    ) -> None:
        await _seed(conn, 1)
        interaction = _make_interaction(db=conn)

        await _invoke(interaction)

        _, kwargs = interaction.response.send_message.call_args
        assert kwargs.get("ephemeral") is True
        assert "diagnostic" in (kwargs["embed"].footer.text or "").lower()

    async def test_the_footer_states_the_debug_view_itself_never_posts(
        self, conn: aiosqlite.Connection
    ) -> None:
        # Phase 2a-3 removed the old "no LLM is called" claim (the pipeline now
        # does both). The footer now scopes the "never posts" promise to this
        # diagnostic command itself, where a moderator reading it will see it.
        await _seed(conn, 1)
        interaction = _make_interaction(db=conn)

        await _invoke(interaction)

        footer = (_sent_embed(interaction).footer.text or "").lower()
        assert "never posts" in footer or "only visible to you" in footer

    async def test_results_are_newest_first(self, conn: aiosqlite.Connection) -> None:
        for message_id in range(5):
            trail = DecisionTrail(
                verdict=GateVerdict.STAGE1_REJECTED,
                stage1_score=message_id / 10,
                stage1_passed=False,
            )
            await record_signal(
                conn, guild_id=GUILD_A, channel_id=1, message_id=message_id, decision=trail
            )
        interaction = _make_interaction(db=conn)

        await _invoke(interaction)

        values = [field.value or "" for field in _sent_embed(interaction).fields]
        assert [v.split()[1] for v in values] == [
            "+0.400",
            "+0.300",
            "+0.200",
            "+0.100",
            "+0.000",
        ]

    async def test_a_negative_score_renders_without_mangling_the_sign(
        self, conn: aiosqlite.Connection
    ) -> None:
        trail = DecisionTrail(
            verdict=GateVerdict.STAGE1_REJECTED, stage1_score=-0.1234, stage1_passed=False
        )
        await _seed(conn, 1, decision=trail)
        interaction = _make_interaction(db=conn)

        await _invoke(interaction)

        assert "-0.123" in (_sent_embed(interaction).fields[0].value or "")

    async def test_the_floor_score_of_unscoreable_text_renders(
        self, conn: aiosqlite.Connection
    ) -> None:
        # -2.0 is a real value the detector returns, and it is one character
        # wider than every other score.
        trail = DecisionTrail(
            verdict=GateVerdict.STAGE1_REJECTED, stage1_score=-2.0, stage1_passed=False
        )
        await _seed(conn, 1, decision=trail)
        interaction = _make_interaction(db=conn)

        await _invoke(interaction)

        assert "-2.000" in (_sent_embed(interaction).fields[0].value or "")

    @pytest.mark.parametrize("locale", ["de", "ja", "ko", "pt-BR", "xx-INVALID"])
    async def test_any_locale_including_an_unsupported_one_renders_without_crashing(
        self, conn: aiosqlite.Connection, locale: str
    ) -> None:
        await _seed(conn, 1, decision=ELIGIBLE_TRAIL)
        interaction = _make_interaction(db=conn, locale=locale)

        await _invoke(interaction)

        embed = _sent_embed(interaction)
        assert embed.title  # falls back to en-US, never blank
        assert embed.fields[0].name
        assert embed.fields[0].value


class TestLimit:
    async def test_the_default_limit_applies_when_none_is_given(
        self, conn: aiosqlite.Connection
    ) -> None:
        await _seed(conn, _DEFAULT_SIGNAL_LIMIT + 10)
        interaction = _make_interaction(db=conn)

        await _invoke(interaction)

        assert len(_sent_embed(interaction).fields) == _DEFAULT_SIGNAL_LIMIT

    @pytest.mark.parametrize("limit", [1, 5, _MAX_SIGNAL_LIMIT])
    async def test_a_limit_inside_the_allowed_range_is_honoured(
        self, conn: aiosqlite.Connection, limit: int
    ) -> None:
        await _seed(conn, _MAX_SIGNAL_LIMIT + 5)
        interaction = _make_interaction(db=conn)

        await _invoke(interaction, limit=limit)

        assert len(_sent_embed(interaction).fields) == limit

    async def test_the_range_annotation_matches_discords_field_cap(self) -> None:
        # If these ever drift apart, Discord rejects the whole reply rather
        # than truncating it.
        (parameter,) = debug_signals_command.parameters
        assert parameter.min_value == 1
        assert parameter.max_value == _MAX_SIGNAL_LIMIT
        assert _MAX_SIGNAL_LIMIT == 25

    @pytest.mark.parametrize("limit", [0, -1, -1000])
    async def test_a_limit_below_the_range_is_clamped_rather_than_returning_everything(
        self, conn: aiosqlite.Connection, limit: int
    ) -> None:
        # Discord enforces the Range before a real invocation ever gets here,
        # but SQLite reads LIMIT -1 as "no limit", so the callback must not
        # depend on that enforcement to avoid dumping the whole table.
        await _seed(conn, 30)
        interaction = _make_interaction(db=conn)

        await _invoke(interaction, limit=limit)

        assert len(_sent_embed(interaction).fields) == 1

    @pytest.mark.parametrize("limit", [26, 1000, 10**9])
    async def test_a_limit_above_the_range_is_clamped(
        self, conn: aiosqlite.Connection, limit: int
    ) -> None:
        await _seed(conn, 40)
        interaction = _make_interaction(db=conn)

        await _invoke(interaction, limit=limit)

        assert len(_sent_embed(interaction).fields) <= _MAX_SIGNAL_LIMIT


class TestEmbedLimits:
    """A full page of trails is four times longer than Phase 2a-1's; it must still fit."""

    @staticmethod
    async def _seed_worst_case(conn: aiosqlite.Connection, count: int) -> None:
        # Snowflake-sized IDs make each permalink as long as it can really
        # get, and every stage populated makes each trail as long as it can
        # get -- so this is the worst realistic case, not a friendly one.
        worst = DecisionTrail(
            verdict=GateVerdict.DAILY_CAP_REACHED,
            stage1_score=-1.234567,
            stage1_passed=True,
            stage2_top_score=-0.987654,
            stage2_runner_up_score=-0.123456,
            stage2_gap=-0.864198,
            stage2_passed=True,
            cooldown_seconds_remaining=86399.99,
            daily_count=99999,
            daily_cap=99999,
        )
        for message_id in range(count):
            await record_signal(
                conn,
                guild_id=GUILD_A,
                channel_id=999999999999999999,
                message_id=888888888888888000 + message_id,
                decision=worst,
            )

    @pytest.mark.parametrize("locale", sorted(SUPPORTED_LOCALES))
    async def test_a_full_page_stays_within_discords_hard_limits_in_every_locale(
        self, conn: aiosqlite.Connection, locale: str
    ) -> None:
        # Per locale, because a translated verdict label is not the same length
        # as the English one, and Discord rejects an oversized embed outright
        # rather than truncating it. This is exactly why the field budget is
        # measured at render time instead of assumed from a field count.
        await self._seed_worst_case(conn, _MAX_SIGNAL_LIMIT)
        interaction = _make_interaction(db=conn, locale=locale)

        await _invoke(interaction, limit=_MAX_SIGNAL_LIMIT)

        embed = _sent_embed(interaction)
        assert len(embed.fields) <= 25
        assert all(len(field.name or "") <= 256 for field in embed.fields)
        assert all(len(field.value or "") <= 1024 for field in embed.fields)
        assert len(embed) <= 6000  # discord.py's own total-length accounting

    async def test_a_page_too_long_to_fit_is_truncated_and_says_so(
        self, conn: aiosqlite.Connection
    ) -> None:
        # Rather than silently dropping rows, or sending an embed Discord will
        # reject. Forced by shrinking the budget, so the test does not depend
        # on any particular locale being long enough to trigger it.
        await self._seed_worst_case(conn, 10)
        interaction = _make_interaction(db=conn)

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr("aura.commands.proactive._EMBED_CHARACTER_BUDGET", 900)
            await _invoke(interaction, limit=10)

        embed = _sent_embed(interaction)
        assert 0 < len(embed.fields) < 10
        assert "more" in (embed.description or "")
        assert len(embed) <= 6000

    async def test_a_budget_too_small_for_even_one_row_still_sends_a_valid_reply(
        self, conn: aiosqlite.Connection
    ) -> None:
        # The degenerate end of the same guard. An embed with no fields is
        # still valid as long as it has a description, so the reply must
        # explain itself rather than crash or send something Discord rejects.
        await self._seed_worst_case(conn, 5)
        interaction = _make_interaction(db=conn)

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr("aura.commands.proactive._EMBED_CHARACTER_BUDGET", 1)
            await _invoke(interaction, limit=5)

        embed = _sent_embed(interaction)
        assert embed.fields == []
        assert "more" in (embed.description or "")

    async def test_the_budget_leaves_room_for_the_truncation_note(self) -> None:
        assert _EMBED_CHARACTER_BUDGET < 6000


class TestGuildIsolation:
    async def test_a_guild_never_sees_another_guilds_signals(
        self, conn: aiosqlite.Connection
    ) -> None:
        await record_signal(
            conn, guild_id=GUILD_A, channel_id=1, message_id=1, decision=REJECTED_TRAIL
        )
        await record_signal(
            conn, guild_id=GUILD_B, channel_id=2, message_id=2, decision=ELIGIBLE_TRAIL
        )

        interaction = _make_interaction(db=conn, guild_id=GUILD_A)
        await _invoke(interaction)

        embed = _sent_embed(interaction)
        assert len(embed.fields) == 1
        assert "-0.456" in (embed.fields[0].value or "")
        assert str(GUILD_B) not in (embed.fields[0].value or "")

    async def test_a_guild_with_signals_elsewhere_still_sees_its_own_empty_state(
        self, conn: aiosqlite.Connection
    ) -> None:
        await record_signal(
            conn, guild_id=GUILD_B, channel_id=1, message_id=1, decision=ELIGIBLE_TRAIL
        )

        interaction = _make_interaction(db=conn, guild_id=GUILD_A)
        await _invoke(interaction)

        args, kwargs = interaction.response.send_message.call_args
        assert "embed" not in kwargs
        assert "no question signals" in args[0].lower()
