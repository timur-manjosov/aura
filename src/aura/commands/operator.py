"""The operator's own commands: /aura-operator-budget and /aura-operator-preview.

/aura-operator-budget is the operator's view of Phase 4a-2's cross-guild brake,
and, since P5c, of the latest refusal of the shared LLM key (aura.llm_failures).
/aura-operator-preview (P4) renders hand-written sample cards of the v2 answer
format from invented facts, so the operator can judge the look in a real client
before any real answer uses it. It calls no model, claims no ledger slot and
writes nothing to the database; every reply is ephemeral. Both are gated the
same way, described below for the budget view.

Every other command in this package is moderator-facing and scoped to one
guild -- manage_guild is the right gate for "can configure this server."
Cross-guild spend is a different kind of question, asked by a different
person: the operator paying the shared LLM key behind every guild this
process serves, who may not moderate any one of them. Gating this on
manage_guild would let any moderator of any guild see every OTHER guild's
share of a combined total that is none of their business; gating it on a
single configured Discord user ID (OPERATOR_DISCORD_USER_ID, see
aura.config.Settings.operator_discord_user_id) answers "is this the person
who pays the bill" directly, without a Discord API round trip and without
branching on team-vs-solo application ownership this project does not use.

Deliberately NOT translated, unlike every other command's user-facing text.
CLAUDE.md's i18n section is about END USERS interacting with Aura in their own
language; this view is read by exactly one person, in the same role as
whoever reads this process's logs -- and every log line in this codebase is
English, untranslated, for that same audience. Treating this embed as an
extension of the log (which aura.db.cross_guild_budget already writes in
English) rather than as end-user UI is a deliberate scope decision, not an
oversight -- see reports/phase-4a-2.txt. The one exception is the permission
error below: unlike the embed's content, that message can be seen by any
member who tries this command out of curiosity, not only the operator, so it
goes through the same t() seam every other command's permission error does.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING, Final

import discord
from discord import app_commands

from aura.answer_card import card_to_embed, card_to_layout_view, components_v2_available
from aura.commands.preview_samples import PreviewSample, card_look_samples, preview_samples
from aura.db.connection import utc_day, utc_now
from aura.db.cross_guild_budget import get_cross_guild_status
from aura.i18n import t
from aura.llm_failures import KEY_ALARM, CallFailureKind, KeyAlarmStatus

if TYPE_CHECKING:
    from aura.main import AuraClient

logger = logging.getLogger(__name__)

# The preview's style choices: both card styles one after the other, or one.
_PREVIEW_STYLES: Final[tuple[str, ...]] = ("both", "embed", "container")

_PREVIEW_INTRO: Final = (
    "**Preview of the v2 answer cards** -- invented facts, no AI call, nothing "
    "stored, visible only to you. The source links lead nowhere. Each sample is "
    "shown as a classic embed and as a Components V2 container, so you can "
    "compare the two styles on desktop and phone, in dark and light theme."
)


def _discord_time(moment: datetime) -> str:
    """Render a moment as Discord's relative timestamp ("vor 12 Minuten" in the reader's language)."""
    return f"<t:{int(moment.timestamp())}:R>"


def key_alarm_line(status: KeyAlarmStatus) -> str:
    """Return the operator view's line about refusals of the LLM key (P5c).

    Parameters
    ----------
    status
        What aura.llm_failures recorded since this process started.

    Returns
    -------
    str
        One line: no refusal since start; or the latest refusal, its reason and
        the call it hit, how many there were, and whether a model call has
        succeeded since. English and untranslated, like the rest of this view.

    Notes
    -----
    Times are Discord timestamps, which every client renders in its own
    language and time zone. Nothing here is a secret or provider text: the
    purpose is a fixed label, the counts are counts.
    """
    if status.last_refusal_at is None:
        return f"No refusal of the LLM key since the bot started {_discord_time(status.watching_since)}."
    reason = (
        "spending limit or credits exhausted"
        if status.last_refusal_kind is CallFailureKind.KEY_LIMIT
        else "key invalid or revoked"
    )
    if status.succeeded_since_refusal:
        assert status.last_success_at is not None  # implied by succeeded_since_refusal
        since = f"model calls have succeeded again since {_discord_time(status.last_success_at)}"
        marker = "⚠️"
    else:
        since = "**no model call has succeeded since**"
        marker = "🚨"
    return (
        f"{marker} Last refusal of the LLM key: **{reason}** {_discord_time(status.last_refusal_at)} "
        f"(call: {status.last_refusal_purpose}); {status.refusals} refused call(s) since the bot "
        f"started; {since}."
    )


def _is_operator(interaction: discord.Interaction[AuraClient]) -> bool:
    """Whether the invoking user is this deployment's configured operator.

    False whenever OPERATOR_DISCORD_USER_ID is unset, which is the safe
    direction: an unconfigured deployment loses this diagnostic view, never
    exposes it to whoever happens to ask first.
    """
    operator_id = interaction.client.settings.operator_discord_user_id
    return operator_id is not None and interaction.user.id == operator_id


async def _handle_operator_budget_error(
    interaction: discord.Interaction[AuraClient], error: app_commands.AppCommandError
) -> None:
    """Turn the operator check's failure into a clean localized reply, log everything else.

    Attaching this via .error() stops CommandTree's default logging for this
    command (it only logs when a command has no local handler), so anything
    other than the check failure is logged here rather than silently
    disappearing.
    """
    if isinstance(error, app_commands.CheckFailure):
        locale = str(interaction.locale)
        message = t("operator_budget_permission_error", locale)
        if interaction.response.is_done():
            await interaction.followup.send(message, ephemeral=True)
        else:
            await interaction.response.send_message(message, ephemeral=True)
        return

    logger.error("Unhandled error in /aura-operator-budget", exc_info=error)


@app_commands.command(
    name="aura-operator-budget",
    description="Operator-only: today's estimated cross-guild spend across every ledger.",
)
@app_commands.guild_only()
@app_commands.check(_is_operator)
async def operator_budget_command(interaction: discord.Interaction[AuraClient]) -> None:
    """Show today's cross-guild call counts, the rough estimated spend, and the combined total.

    Parameters
    ----------
    interaction
        The command invocation. Carries the invoker's locale, the guild it was
        run in, and the client the database, models and plan gate hang off.

    Returns
    -------
    None

    Notes
    -----
    Read-only, and reads exactly the numbers aura.proactive.gate,
    aura.extraction.pipeline, aura.variants_service and aura.backfill.worker
    each check before claiming a per-guild slot (see
    aura.db.cross_guild_budget.get_cross_guild_status) -- so what enforcement
    acts on and what this command shows can never quietly disagree.
    """
    db = interaction.client.db
    assert db is not None  # setup_hook always finishes before commands go live
    settings = interaction.client.settings

    day = utc_day(utc_now())
    status = await get_cross_guild_status(
        db,
        day=day,
        budget_usd=settings.cross_guild_daily_budget_usd,
        mode=settings.cross_guild_budget_mode,
    )

    embed = discord.Embed(
        title="Cross-Guild Operator Budget (Phase 4a-2)",
        description=(
            f"UTC day {status.day} · mode = {status.mode.value}\n"
            f"Rough, conservative estimate from documented worst-case per-call "
            f"cost -- not a real-time cost meter. See aura.db.cross_guild_budget."
        ),
    )
    for entry in status.ledgers:
        embed.add_field(
            name=entry.ledger.value.capitalize(),
            # Four decimal places, not two: a single call on the cheapest
            # ledger (supersession, $0.001) already rounds to $0.00 at two
            # decimals, which would show real spend as if nothing happened.
            value=f"{entry.call_count} call(s) today · ~${entry.estimated_usd:.4f}",
            inline=True,
        )

    over_budget_note = " — **OVER BUDGET**" if status.over_budget else " — within budget"
    embed.add_field(
        name="Combined total (all guilds, all ledgers)",
        value=f"~${status.total_estimated_usd:.4f} of ${status.budget_usd:.2f}{over_budget_note}",
        inline=False,
    )
    # P5c: a refused key fails every model call; this is where the operator sees it.
    embed.add_field(name="LLM key", value=key_alarm_line(KEY_ALARM.status()), inline=False)
    embed.set_footer(
        text=(
            "Visible only to you. In WARN mode an over-budget total is logged, "
            "never enforced; in HARD mode new calls at every ledger are refused "
            "until the next UTC day."
        )
    )

    await interaction.response.send_message(embed=embed, ephemeral=True)


operator_budget_command.error(_handle_operator_budget_error)


async def _send_preview_sample(
    interaction: discord.Interaction[AuraClient], sample: PreviewSample, style: str
) -> None:
    """Send one sample in one card style, or a short note naming why Discord refused it.

    Parameters
    ----------
    interaction
        The deferred, ephemeral preview interaction.
    sample
        The sample to show.
    style
        "embed" or "container".

    Returns
    -------
    None

    Notes
    -----
    A refused sample is reported in the preview itself rather than ending it:
    the point of the preview is to find exactly such a refusal before a real
    answer meets it.
    """
    caption = f"**{sample.caption}** · {style}"
    if style == "container" and not components_v2_available():
        await interaction.followup.send(
            f"{caption}: the installed discord.py has no Components V2 support (2.6 or later "
            "needed), so this style cannot be shown.",
            ephemeral=True,
        )
        return
    try:
        if style == "embed":
            await interaction.followup.send(
                content=caption,
                embed=card_to_embed(sample.card),
                ephemeral=True,
                allowed_mentions=discord.AllowedMentions.none(),
            )
        else:
            await interaction.followup.send(
                view=card_to_layout_view(sample.card, caption=caption),
                ephemeral=True,
                allowed_mentions=discord.AllowedMentions.none(),
            )
    except discord.HTTPException as exc:
        logger.warning("Preview sample %r (%s) was refused: %s", sample.caption, style, exc.status)
        await interaction.followup.send(
            f"{caption}: Discord refused this sample (HTTP {exc.status}).", ephemeral=True
        )


@app_commands.command(
    name="aura-operator-preview",
    description="Operator-only: sample answer cards of the new format (invented facts, no AI).",
)
@app_commands.describe(style="Which card style to show (default: both)")
@app_commands.choices(
    style=[app_commands.Choice(name=name, value=name) for name in _PREVIEW_STYLES]
)
@app_commands.guild_only()
@app_commands.check(_is_operator)
async def operator_preview_command(
    interaction: discord.Interaction[AuraClient], style: app_commands.Choice[str] | None = None
) -> None:
    """Show the v2 answer card samples, visible only to the operator.

    Parameters
    ----------
    interaction
        The command invocation; its locale picks the labels and the content
        language of the samples.
    style
        "both" (the default), "embed" or "container".

    Returns
    -------
    None

    Notes
    -----
    Deferred and answered with followups, the same mechanism /aura-ask uses, so
    a card that Discord would refuse after a defer is refused here first. No
    model, no ledger, no database: the samples are a pure function of the
    locale, the guild and the clock (aura.commands.preview_samples).
    """
    assert interaction.guild_id is not None  # guaranteed by guild_only()
    await interaction.response.defer(ephemeral=True, thinking=True)
    chosen = style.value if style is not None else "both"
    styles = ("embed", "container") if chosen == "both" else (chosen,)
    await interaction.followup.send(_PREVIEW_INTRO, ephemeral=True)
    settings = interaction.client.settings
    now = utc_now()
    samples = [
        *preview_samples(str(interaction.locale), guild_id=interaction.guild_id, now=now),
        *card_look_samples(
            str(interaction.locale),
            guild_id=interaction.guild_id,
            now=now,
            ask_caps=(
                settings.ask_daily_cap_free,
                settings.ask_user_daily_cap_free,
                settings.ask_daily_cap_pro,
            ),
            dashboard_url=settings.billing_dashboard_url,
        ),
    ]
    for sample in samples:
        for one_style in styles:
            await _send_preview_sample(interaction, sample, one_style)


async def _handle_operator_preview_error(
    interaction: discord.Interaction[AuraClient], error: app_commands.AppCommandError
) -> None:
    """Refuse everyone but the operator with the same localized reply as the budget view."""
    if isinstance(error, app_commands.CheckFailure):
        await _handle_operator_budget_error(interaction, error)
        return
    logger.error("Unhandled error in /aura-operator-preview", exc_info=error)


operator_preview_command.error(_handle_operator_preview_error)


def register_operator_commands(tree: app_commands.CommandTree) -> None:
    """Register the operator-only commands (budget view and card preview) onto tree.

    Parameters
    ----------
    tree
        The command tree to register into.

    Returns
    -------
    None
    """
    tree.add_command(operator_budget_command)
    tree.add_command(operator_preview_command)
