"""/aura-operator-budget: the operator's own view of Phase 4a-2's cross-guild brake.

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
from typing import TYPE_CHECKING

import discord
from discord import app_commands

from aura.db.connection import utc_day, utc_now
from aura.db.cross_guild_budget import get_cross_guild_status
from aura.i18n import t

if TYPE_CHECKING:
    from aura.main import AuraClient

logger = logging.getLogger(__name__)


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
    embed.set_footer(
        text=(
            "Visible only to you. In WARN mode an over-budget total is logged, "
            "never enforced; in HARD mode new calls at every ledger are refused "
            "until the next UTC day."
        )
    )

    await interaction.response.send_message(embed=embed, ephemeral=True)


operator_budget_command.error(_handle_operator_budget_error)


def register_operator_commands(tree: app_commands.CommandTree) -> None:
    """Register the operator-only cross-guild budget command onto tree."""
    tree.add_command(operator_budget_command)
