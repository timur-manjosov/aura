"""/aura-plan, and the one refusal every Pro-only command gives on a Free server.

Phase 4c's promise to an admin is that a plan is never a silent state. Two
things in this module keep it:

  * pro_feature_refusal -- called by every command that would switch a Pro
    trigger ON (/aura-config's proactive and extraction switches,
    /aura-digest, /aura-onboarding, /aura-backfill start) before it writes
    anything. On a Free server the command answers with this refusal and
    changes nothing, rather than saving a setting that then does nothing.
    Switching a feature OFF is never refused: nobody should need a
    subscription to make Aura do less.

  * /aura-plan -- what plan the server is on, why, and until when. When a
    renewal payment has failed this is where an admin sees the date Pro ends
    unless the payment method is fixed, which is the in-Discord half of the
    grace-period communication (the other half is Stripe's own failed-payment
    email to the payer).

Dates are rendered as Discord timestamp markup (<t:...:f>), which every client
displays in its reader's own locale and time zone. That keeps date formats out
of nine locale files and out of this module entirely.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING

import discord
from discord import app_commands

from aura.billing import GuildPlan, PlanBasis, Standing
from aura.i18n import t

if TYPE_CHECKING:
    from aura.main import AuraClient

logger = logging.getLogger(__name__)

_STANDING_KEYS: dict[Standing, str] = {
    Standing.NO_SUBSCRIPTION: "plan_state_no_subscription",
    Standing.ENDED: "plan_state_ended",
    Standing.ACTIVE: "plan_state_active",
    Standing.RENEWAL_PENDING: "plan_state_renewal_pending",
    Standing.CANCELING: "plan_state_canceling",
    Standing.PAYMENT_GRACE: "plan_state_payment_grace",
}


def discord_timestamp(moment: datetime) -> str:
    """Discord's timestamp markup for a moment, rendered by each client in its own locale."""
    return f"<t:{int(moment.timestamp())}:f>"


def pro_feature_refusal(interaction: discord.Interaction[AuraClient]) -> str | None:
    """None if this guild may use Pro features now; otherwise the localized refusal to send.

    The refusal says three things, because a moderator who hits it needs all
    three: this is a Pro feature, nothing was changed, and what still works.
    The dashboard link is added only when the operator configured one.
    """
    assert interaction.guild_id is not None  # every caller is guild_only
    gate = interaction.client.plan_gate
    assert gate is not None  # setup_hook always finishes before commands go live
    if gate.allows_pro(interaction.guild_id):
        return None

    locale = str(interaction.locale)
    lines = [t("plan_pro_required", locale)]
    dashboard_url = interaction.client.settings.billing_dashboard_url
    if dashboard_url:
        lines.append(t("plan_upgrade_link", locale, url=dashboard_url))
    return "\n".join(lines)


def describe_plan(plan: GuildPlan, *, locale: str, dashboard_url: str | None) -> str:
    """The /aura-plan reply. Pure, so every standing is testable without Discord."""
    standing = plan.standing
    lines: list[str] = []

    if plan.basis is PlanBasis.BILLING_NOT_ENFORCED:
        lines.append(t("plan_state_not_enforced", locale))
    elif plan.basis is PlanBasis.COMPLIMENTARY:
        lines.append(t("plan_state_complimentary", locale))

    # The subscription line is shown whenever it decides the plan, and also
    # when a guild on a non-subscription basis is paying anyway -- otherwise a
    # complimentary guild with a forgotten subscription could never find out.
    if plan.basis is PlanBasis.SUBSCRIPTION or standing.grants_access:
        key = _STANDING_KEYS[standing.standing]
        shown_date = (
            standing.paid_through if standing.standing is Standing.ACTIVE else standing.access_until
        )
        if shown_date is None:
            lines.append(t(key, locale))
        else:
            lines.append(t(key, locale, date=discord_timestamp(shown_date)))

    if not plan.is_pro:
        lines.append(t("plan_free_includes", locale))

    granting = len(standing.granting_subscription_ids)
    if granting > 1:
        lines.append(t("plan_multiple_subscriptions", locale, count=granting))

    if dashboard_url:
        lines.append(t("plan_manage_link", locale, url=dashboard_url))
    return "\n".join(lines)


async def _handle_plan_command_error(
    interaction: discord.Interaction[AuraClient], error: app_commands.AppCommandError
) -> None:
    """Turn the permission check's failure into a clean localized reply, log everything else."""
    if isinstance(error, app_commands.MissingPermissions):
        locale = str(interaction.locale)
        message = t("plan_permission_error", locale)
        if interaction.response.is_done():
            await interaction.followup.send(message, ephemeral=True)
        else:
            await interaction.response.send_message(message, ephemeral=True)
        return

    logger.error("Unhandled error in /aura-plan", exc_info=error)


@app_commands.command(
    name="aura-plan",
    description="Show this server's Aura plan and what it includes (moderators only).",
)
@app_commands.guild_only()
@app_commands.checks.has_permissions(manage_guild=True)
async def plan_command(interaction: discord.Interaction[AuraClient]) -> None:
    """Reply with the server's plan, its standing and, when configured, where to manage it.

    Mod-gated like every other configuration command: whether a payment failed
    is the admins' business, not every member's. Ephemeral for the same reason.
    """
    assert interaction.guild_id is not None  # guaranteed by guild_only()
    gate = interaction.client.plan_gate
    assert gate is not None  # setup_hook always finishes before commands go live
    message = describe_plan(
        gate.plan_for(interaction.guild_id),
        locale=str(interaction.locale),
        dashboard_url=interaction.client.settings.billing_dashboard_url,
    )
    await interaction.response.send_message(message, ephemeral=True)


plan_command.error(_handle_plan_command_error)


def register_plan_command(tree: app_commands.CommandTree) -> None:
    """Register /aura-plan onto tree."""
    tree.add_command(plan_command)
