"""/aura-debug-signals: a moderator's live view of every proactive gate decision.

Phase 2a-1's inspection surface, extended to the whole decision trail now that
there is a decision to inspect: each recent message shows its Stage 1 score
*and* whether that passed, its Stage 2 match and confidence gap, the cooldown
and daily-cap state the gate saw, and the single verdict those produced.

Why the losing numbers are shown and not just the verdict: every threshold in
this pipeline is an explicit placeholder awaiting recalibration (see
aura.config), and "rejected" alone says nothing about which number to move.
"passed Stage 1 at +0.11, matched a fact at 0.68, held back because 0.68 is
under the bar" says exactly that. As of Phase 2a-3 the trail also carries the
synthesis outcome -- whether the LLM judged the facts an answer, and whether
Aura actually posted -- completing the decision trail from message to public
reply. Nothing here is a user-facing feature, and the whole module goes away
with the scaffolding it reads.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import discord
from discord import app_commands

from aura.db.connection import utc_now
from aura.db.proactive_signals import (
    GracePeriodOutcome,
    GateVerdict,
    ProactiveSignal,
    get_recent_signals,
)
from aura.db.proactive_state import count_escalations_on, utc_day
from aura.i18n import t

if TYPE_CHECKING:
    from aura.main import AuraClient

logger = logging.getLogger(__name__)

_DEFAULT_SIGNAL_LIMIT = 10
# Discord's own hard cap on the number of fields in one embed, not a choice
# made here.
_MAX_SIGNAL_LIMIT = 25

# Discord rejects an embed whose total rendered length exceeds 6000 characters
# outright rather than truncating it. Phase 2a-1 could ignore that: one field
# held a score and a link. A full decision trail is roughly four times longer,
# so 25 of them can genuinely exceed the limit -- and by how much depends on
# the viewer's locale, since a translated verdict label is not the same length
# as the English one. Hence a running character budget rather than a smaller
# hardcoded field count: the limit that actually binds is characters, and only
# measuring them is honest about it. The margin below Discord's 6000 leaves
# room for the truncation note appended after the loop.
_EMBED_CHARACTER_BUDGET = 5800

# One glyph per verdict, purely so a moderator can scan a page of these
# without reading. Deliberately not localized (a glyph has no language), while
# every word beside it is.
_VERDICT_DISPLAY: dict[GateVerdict, tuple[str, str]] = {
    GateVerdict.ELIGIBLE: ("✅", "debug_signals_verdict_eligible"),
    GateVerdict.STAGE1_REJECTED: ("➖", "debug_signals_verdict_stage1_rejected"),
    GateVerdict.NO_MATCHING_FACT: ("❔", "debug_signals_verdict_no_matching_fact"),
    # Kept for rows recorded before Phase 2b-4 retired this verdict; the gate
    # cannot produce it any more. See GateVerdict.AMBIGUOUS_FACTS.
    GateVerdict.AMBIGUOUS_FACTS: ("⚖️", "debug_signals_verdict_ambiguous_facts"),
    GateVerdict.COOLDOWN_ACTIVE: ("⏳", "debug_signals_verdict_cooldown_active"),
    GateVerdict.DAILY_CAP_REACHED: ("🚫", "debug_signals_verdict_daily_cap_reached"),
    GateVerdict.DUPLICATE_DELIVERY: ("🔁", "debug_signals_verdict_duplicate_delivery"),
}

# Phase 2b-1's grace-period outcome, keyed the same way _VERDICT_DISPLAY is:
# no glyph here, since a moderator scans the verdict glyph first and the
# grace outcome is finer detail read second, alongside the synthesis result.
_GRACE_DISPLAY: dict[GracePeriodOutcome, str] = {
    GracePeriodOutcome.PENDING: "debug_signals_grace_pending",
    GracePeriodOutcome.CANCELLED_BY_HUMAN: "debug_signals_grace_cancelled_by_human",
    GracePeriodOutcome.EXPIRED_AND_PROCEEDED: "debug_signals_grace_expired_and_proceeded",
    GracePeriodOutcome.STOOD_DOWN_ON_RECHECK: "debug_signals_grace_stood_down_on_recheck",
}

# Stands in for a number that was never computed, because the gate
# short-circuited before reaching that stage. A dash rather than 0.000, which
# would read as "evaluated, and scored zero" -- the opposite of the truth.
_NOT_EVALUATED = "—"

_PASSED = "✓"
_FAILED = "✗"


async def _handle_debug_signals_error(
    interaction: discord.Interaction[AuraClient], error: app_commands.AppCommandError
) -> None:
    """Turn the permission check's failure into a clean localized reply, log everything else.

    Attaching this via .error() stops CommandTree's default logging for this
    command (it only logs when a command has no local handler), so anything
    other than the permission failure is logged here rather than silently
    disappearing.
    """
    if isinstance(error, app_commands.MissingPermissions):
        locale = str(interaction.locale)
        message = t("debug_signals_permission_error", locale)
        if interaction.response.is_done():
            await interaction.followup.send(message, ephemeral=True)
        else:
            await interaction.response.send_message(message, ephemeral=True)
        return

    logger.error("Unhandled error in /aura-debug-signals", exc_info=error)


def _format_stage1(signal: ProactiveSignal) -> str:
    """Render the Stage 1 score together with the pass/fail it produced.

    Both, not just the score: Phase 2a-1 showed the number alone because no
    threshold existed yet, and a number whose verdict the reader has to infer
    from a config file they cannot see is a trail with a hole in it.
    """
    return f"{signal.stage1_score:+.3f} {_PASSED if signal.stage1_passed else _FAILED}"


def _format_stage2(signal: ProactiveSignal, locale: str) -> str:
    """Render the best fact match, its margin over the runner-up, and the outcome.

    The margin is shown as diagnostic context, not as a reason: since Phase
    2b-4 it decides nothing (see aura.proactive.gate), and only the top score
    is compared against anything. It stays on the trail because "how close were
    the top two facts?" is still the fastest way to see that a guild holds a
    near-duplicate or an unsuperseded pair -- a question a moderator can act on
    with /aura-supersede.
    """
    if signal.stage2_passed is None:
        return _NOT_EVALUATED
    if signal.stage2_top_score is None:
        # Stage 2 ran, but the guild has no active fact to compare against.
        return t("debug_signals_no_facts", locale)

    gap = _NOT_EVALUATED if signal.stage2_gap is None else f"{signal.stage2_gap:.3f}"
    scores = t("debug_signals_stage2", locale, top=f"{signal.stage2_top_score:+.3f}", gap=gap)
    return f"{scores} {_PASSED if signal.stage2_passed else _FAILED}"


def _format_cooldown(signal: ProactiveSignal, locale: str) -> str:
    """Render the channel's cooldown as the gate saw it, before this message's own claim."""
    if signal.cooldown_seconds_remaining is None:
        return _NOT_EVALUATED
    if signal.cooldown_seconds_remaining <= 0:
        return t("debug_signals_cooldown_clear", locale)
    return t(
        "debug_signals_cooldown_remaining",
        locale,
        seconds=round(signal.cooldown_seconds_remaining),
    )


def _format_cap(signal: ProactiveSignal) -> str:
    """Render the guild's daily cap usage as of this message."""
    if signal.daily_count is None or signal.daily_cap is None:
        return _NOT_EVALUATED
    return f"{signal.daily_count}/{signal.daily_cap}"


def _format_synthesis(signal: ProactiveSignal, locale: str) -> str:
    """Render the Phase 2a-3 synthesis outcome: did the model answer, and did Aura post?

    Only ELIGIBLE messages reach synthesis, so every other verdict renders as
    "not evaluated" rather than a misleading pair of crosses. An eligible
    message whose synthesis produced no result at all -- the LLM call failed,
    or none is configured for this trigger -- shows a distinct "no answer"
    label so a moderator can tell "the model declined" (answers ✗) apart from
    "there was no model to ask".
    """
    if signal.verdict is not GateVerdict.ELIGIBLE:
        return _NOT_EVALUATED
    if signal.synthesis_answers_question is None:
        return t("debug_signals_synthesis_no_result", locale)
    answers = _PASSED if signal.synthesis_answers_question else _FAILED
    posted = _PASSED if signal.synthesis_posted else _FAILED
    return t("debug_signals_synthesis", locale, answers=answers, posted=posted)


def _format_grace(signal: ProactiveSignal, locale: str) -> str:
    """Render Phase 2b-1's grace-period outcome: did a human get there first?

    Only ELIGIBLE messages ever enter a grace period, so every other verdict
    renders as "not evaluated" -- the same convention _format_synthesis uses,
    for the same reason. A NULL outcome on an ELIGIBLE row reads as PENDING
    rather than as "not evaluated": it means either the wait is genuinely
    still in flight (the common case a moderator would actually check this
    mid-wait for) or, on a database carried over from before Phase 2b-1's
    migration, that no grace period ever ran for it -- both are honestly
    described as "no terminal outcome yet".
    """
    if signal.verdict is not GateVerdict.ELIGIBLE:
        return _NOT_EVALUATED
    if signal.grace_period_outcome is None:
        return t("debug_signals_grace_pending", locale)
    return t(_GRACE_DISPLAY[signal.grace_period_outcome], locale)


def _render_signal(signal: ProactiveSignal, locale: str) -> tuple[str, str]:
    """Render one decision trail as an embed field's (name, value).

    Falls back to the raw verdict string for a verdict with no display entry,
    which cannot happen today (a test asserts the mapping is exhaustive) but
    would otherwise render as a blank field name -- a silent hole in a debug
    view is worse than an ugly one.
    """
    glyph, verdict_key = _VERDICT_DISPLAY.get(signal.verdict, ("•", ""))
    verdict_label = t(verdict_key, locale) if verdict_key else str(signal.verdict)

    permalink = (
        f"https://discord.com/channels/"
        f"{signal.guild_id}/{signal.channel_id}/{signal.message_id}"
    )
    # <t:...:R> renders as a relative time in each viewer's own client
    # locale and timezone -- correct in all nine of Aura's languages
    # without a date format string in any locale file.
    relative_time = f"<t:{int(signal.created_at.timestamp())}:R>"
    trail = t(
        "debug_signals_trail",
        locale,
        stage1=_format_stage1(signal),
        stage2=_format_stage2(signal, locale),
        cooldown=_format_cooldown(signal, locale),
        cap=_format_cap(signal),
        grace=_format_grace(signal, locale),
        synthesis=_format_synthesis(signal, locale),
    )

    name = f"{glyph} {verdict_label}"
    value = f"{trail}\n[{t('debug_signals_jump', locale)}]({permalink}) · {relative_time}"
    return name, value


@app_commands.command(
    name="aura-debug-signals",
    description="Diagnostic: show how Aura's proactive gate decided on recent messages.",
)
@app_commands.describe(limit="How many recent decisions to show (1-25, default 10).")
@app_commands.guild_only()
@app_commands.checks.has_permissions(manage_guild=True)
async def debug_signals_command(
    interaction: discord.Interaction[AuraClient],
    limit: app_commands.Range[int, 1, _MAX_SIGNAL_LIMIT] = _DEFAULT_SIGNAL_LIMIT,
) -> None:
    """List this guild's most recent proactive gate decisions in full, newest first."""
    assert interaction.guild_id is not None  # guaranteed by guild_only()
    locale = str(interaction.locale)

    db = interaction.client.db
    assert db is not None  # setup_hook always finishes before commands go live

    # Discord enforces the Range above on its own side, so a real invocation
    # cannot arrive outside it. Clamped again anyway because this callback is
    # also directly callable (tests, and any later internal reuse), and an
    # embed carrying more than _MAX_SIGNAL_LIMIT fields is rejected outright
    # by the API rather than truncated.
    requested = max(1, min(int(limit), _MAX_SIGNAL_LIMIT))

    signals = await get_recent_signals(db, guild_id=interaction.guild_id, limit=requested)

    if not signals:
        await interaction.response.send_message(t("debug_signals_empty", locale), ephemeral=True)
        return

    # Live cap usage, separate from the per-message figures below: those record
    # the state at evaluation time, which is what makes them useful for
    # tuning, but a moderator asking "are we capped out right now?" needs the
    # answer as of right now.
    today = utc_day(utc_now())
    spent_today = await count_escalations_on(db, guild_id=interaction.guild_id, day=today)
    gate_config = interaction.client.gate_config

    embed = discord.Embed(
        title=t("debug_signals_title", locale),
        description=t(
            "debug_signals_cap_today",
            locale,
            count=spent_today,
            cap=gate_config.daily_cap if gate_config is not None else _NOT_EVALUATED,
            day=today,
        ),
    )
    embed.set_footer(text=t("debug_signals_footer", locale))

    used_characters = len(embed)
    rendered = 0
    for signal in signals:
        name, value = _render_signal(signal, locale)
        if used_characters + len(name) + len(value) > _EMBED_CHARACTER_BUDGET:
            break
        embed.add_field(name=name, value=value, inline=False)
        used_characters += len(name) + len(value)
        rendered += 1

    if rendered < len(signals):
        embed.description = (
            f"{embed.description}\n"
            f"{t('debug_signals_truncated_note', locale, count=len(signals) - rendered)}"
        )

    # Ephemeral: this is moderator debug output about other people's messages.
    # The proactive answers themselves are public (see aura.proactive.responder);
    # this inspection surface for them is not.
    await interaction.response.send_message(embed=embed, ephemeral=True)


debug_signals_command.error(_handle_debug_signals_error)


def register_proactive_commands(tree: app_commands.CommandTree) -> None:
    """Register the proactive-detection debug command onto tree."""
    tree.add_command(debug_signals_command)
