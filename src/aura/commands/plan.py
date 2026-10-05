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
from aura.card_delivery import reply_with_card
from aura.cards import build_plan_card, build_pro_refusal_card
from aura.config import MessageLook
from aura.i18n import t

if TYPE_CHECKING:
    from aura.main import AuraClient

logger = logging.getLogger(__name__)

_STANDING_KEYS: dict[Standing, str] = {
    Standing.NO_SUBSCRIPTION: "plan_state_no_subscription",
    Standing.ENDED: "plan_state_ended",
    Standing.ACTIVE: "plan_state_active",
    Standing.RENEWAL_PENDING: "plan_state_renewal_pending",
    Standing.PAYMENT_PENDING: "plan_state_payment_pending",
    Standing.CANCELING: "plan_state_canceling",
    Standing.PAYMENT_GRACE: "plan_state_payment_grace",
}


def discord_timestamp(moment: datetime) -> str:
    """Render a moment as Discord timestamp markup.

    Parameters
    ----------
    moment
        The instant to show.

    Returns
    -------
    str
        A `<t:UNIX:f>` token, rendered by each client in its own locale and
        timezone.
    """
    return f"<t:{int(moment.timestamp())}:f>"


def pro_feature_refusal(interaction: discord.Interaction[AuraClient]) -> str | None:
    """Return the localized refusal for a Pro feature, or None if it may run.

    Parameters
    ----------
    interaction
        The command invocation. Guild-only, so its `guild_id` is always set.

    Returns
    -------
    str or None
        None when this guild may use Pro features right now; otherwise the text
        to send. The refusal says three things, because a moderator who hits it
        needs all three: this is a Pro feature, nothing was changed, and what
        still works. The dashboard link is added only when the operator
        configured one.
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


def _shown_date(plan: GuildPlan) -> datetime | None:
    """The one date the standing's sentence carries, or None for a sentence without one.

    "Paid through" is only ever said of `paid_through`, which the entitlement
    rules set exactly when the period's invoice is paid; a payment still being
    confirmed has no date to promise, and every other standing names the moment
    Pro ends.
    """
    standing = plan.standing
    if standing.standing is Standing.ACTIVE:
        return standing.paid_through
    if standing.standing is Standing.PAYMENT_PENDING:
        return None
    return standing.access_until


def standing_lines(plan: GuildPlan, *, locale: str) -> list[str]:
    """Return the sentences that say which plan the server is on, and why.

    Parameters
    ----------
    plan
        The guild's decided plan and standing.
    locale
        Language to write in.

    Returns
    -------
    list[str]
        The basis sentence (billing not enforced, complimentary) when it
        applies, then the subscription's standing whenever it decides the plan
        or grants access anyway. Shared by the classic reply and the card
        (aura.cards.build_plan_card), so both say the same thing.
    """
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
        shown_date = _shown_date(plan)
        if shown_date is None:
            lines.append(t(key, locale))
        else:
            lines.append(t(key, locale, date=discord_timestamp(shown_date)))
    return lines


def describe_plan(plan: GuildPlan, *, locale: str, dashboard_url: str | None) -> str:
    """Build the /aura-plan reply text.

    Parameters
    ----------
    plan
        The guild's decided plan and standing.
    locale
        Language to write in.
    dashboard_url
        The operator's billing dashboard, or None when none is configured.

    Returns
    -------
    str
        The reply body. Pure, so every standing is testable without Discord.
    """
    standing = plan.standing
    lines = standing_lines(plan, locale=locale)

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
    Mod-gated like every other configuration command: whether a payment failed
    is the admins' business, not every member's. Ephemeral for the same reason.
    """
    assert interaction.guild_id is not None  # guaranteed by guild_only()
    gate = interaction.client.plan_gate
    assert gate is not None  # setup_hook always finishes before commands go live
    settings = interaction.client.settings
    plan = gate.plan_for(interaction.guild_id)
    locale = str(interaction.locale)
    if settings.plan_look is MessageLook.CARD:
        # P5's card look (aura.cards), switched by PLAN_LOOK: the same standing
        # sentences, with what the plan includes and the management link.
        card = build_plan_card(
            plan,
            locale=locale,
            standing_lines=standing_lines(plan, locale=locale),
            dashboard_url=settings.billing_dashboard_url,
            ask_caps=(
                settings.ask_daily_cap_free,
                settings.ask_user_daily_cap_free,
                settings.ask_daily_cap_pro,
            ),
        )
        await reply_with_card(interaction, card, style=settings.answer_card_style)
        return
    message = describe_plan(plan, locale=locale, dashboard_url=settings.billing_dashboard_url)
    await interaction.response.send_message(message, ephemeral=True)


async def send_pro_refusal(interaction: discord.Interaction[AuraClient], refusal: str) -> None:
    """Send the refusal `pro_feature_refusal` returned, in the configured look.

    Parameters
    ----------
    interaction
        The refused command invocation; not yet responded to.
    refusal
        The classic refusal text from `pro_feature_refusal`.

    Returns
    -------
    None

    Notes
    -----
    With NOTICE_LOOK=classic (the default) this is exactly the call every
    caller made before P5. With the card look it is the plan notice of
    aura.cards: the same sentence, and the upgrade link as a masked link.
    """
    settings = interaction.client.settings
    if settings.notice_look is MessageLook.CARD:
        card = build_pro_refusal_card(
            str(interaction.locale), dashboard_url=settings.billing_dashboard_url
        )
        await reply_with_card(interaction, card, style=settings.answer_card_style)
        return
    await interaction.response.send_message(refusal, ephemeral=True)


plan_command.error(_handle_plan_command_error)


def register_plan_command(tree: app_commands.CommandTree) -> None:
    """Register /aura-plan onto tree.

    Parameters
    ----------
    tree
        The command tree to register into.

    Returns
    -------
    None
    """
    tree.add_command(plan_command)
