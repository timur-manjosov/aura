"""Reacting to a member joining: CLAUDE.md's THIRD trigger, end to end.

The ordering below is the load-bearing part of this whole module, mirroring
aura.digest.scheduler._post_guild_digest's carefully-justified sequence for
the same reason -- every step fails in a direction that was chosen, not
inherited:

  0. Is this even a human? Bot joins are excluded (see handle_member_join).
  1. Is onboarding configured and on for this guild? A cheap read; an
     unconfigured or disabled guild costs one indexed lookup per join.
  2. Assemble the content. Read-only, no side effects, so it is safe to have
     done this and then abandon it if the guild has nothing to say yet.
  3. Resolve the target channel. Also read-only, and BEFORE the claim, for the
     same reason aura.digest.scheduler._resolve_target gives: a permanently
     broken channel should cost a cache lookup per join, not a written-and-
     immediately-orphaned bookkeeping row.
  4. CLAIM the send atomically -- before anything is posted. Guards against
     both a redelivered join event and a mass-join event exceeding the daily
     cap (see aura.db.onboarding_state).
  5. Send.

Unlike the digest, there is no retry-on-failure bookkeeping here: a digest is
retried by the NEXT scheduled tick, but a join is a one-shot Discord event
with no periodic sweep behind it, so a send that fails after being claimed is
simply logged and not reattempted for that join -- the same class of gap
_cannot_post_in on /aura-onboarding exists to catch in advance, and a
moderator who fixes permissions gets it right for every join from then on.
"""
from __future__ import annotations

import logging

import aiosqlite
import discord

from aura.config import Settings
from aura.db.connection import utc_iso, utc_now
from aura.db.onboarding_config import get_onboarding_config
from aura.db.onboarding_state import OnboardingSendOutcome, try_claim_onboarding_send
from aura.onboarding.builder import build_onboarding_content
from aura.onboarding.formatter import build_onboarding_embed, onboarding_locale
from aura.onboarding.gateway import OnboardingGateway

logger = logging.getLogger(__name__)


async def handle_member_join(
    member: discord.Member,
    *,
    db: aiosqlite.Connection,
    gateway: OnboardingGateway,
    settings: Settings,
) -> None:
    """Post one member's onboarding summary, if this guild wants one and has anything to say.

    Never raises: every branch below is an ordinary "do not send" outcome
    (unconfigured, empty, unresolvable channel, lost race, over the daily
    cap), and the one genuinely exceptional step -- the network call to
    Discord -- is wrapped so a transient failure logs instead of propagating
    into discord.py's own event-dispatch error handling.
    """
    if member.bot:
        # A bot joining a server (another utility bot, a moderation bot, Aura
        # itself being re-invited) is not a "new member" CLAUDE.md's
        # onboarding trigger is for -- it has no knowledge model to be caught
        # up on, and a summary posted at it would just be public noise aimed
        # at nobody who can read it.
        return

    guild = member.guild
    config = await get_onboarding_config(db, guild_id=guild.id)
    if config is None or not config.onboarding_enabled:
        return

    content = await build_onboarding_content(
        db, guild_id=guild.id, limit=settings.onboarding_fact_limit
    )
    if content.is_empty:
        logger.info(
            "No onboarding message for member %s in guild %s: no eligible active facts yet",
            member.id,
            guild.id,
        )
        return

    channel = await gateway.resolve_channel(config.channel_id)
    if channel is None:
        logger.warning(
            "Onboarding for guild %s could not be posted: channel %s is unavailable",
            guild.id,
            config.channel_id,
        )
        return

    if channel.guild.id != guild.id:
        # Not reachable through the slash command (Discord resolves the
        # channel option within the invoking guild), only through a
        # hand-edited configuration row -- the exact cross-guild leak
        # aura.digest.scheduler._resolve_target refuses, refused here for the
        # identical reason: posting one guild's current rules and status into
        # another guild's channel is a real data leak, not a hypothetical one.
        logger.error(
            "Refusing to post guild %s's onboarding message into channel %s, which "
            "belongs to guild %s",
            guild.id,
            config.channel_id,
            channel.guild.id,
        )
        return

    joined_at = member.joined_at or utc_now()
    outcome = await try_claim_onboarding_send(
        db,
        guild_id=guild.id,
        user_id=member.id,
        joined_at=utc_iso(joined_at),
        fact_count=content.shown_count,
        daily_cap=settings.onboarding_daily_cap,
        now=utc_now(),
    )
    if outcome is OnboardingSendOutcome.ALREADY_SENT:
        logger.info(
            "Onboarding for member %s in guild %s was already sent for this join; "
            "staying silent",
            member.id,
            guild.id,
        )
        return
    if outcome is OnboardingSendOutcome.DAILY_CAP_REACHED:
        logger.warning(
            "Onboarding for member %s in guild %s skipped: daily cap of %d reached",
            member.id,
            guild.id,
            settings.onboarding_daily_cap,
        )
        return

    locale = onboarding_locale(guild)
    embed = build_onboarding_embed(content, locale=locale)
    try:
        # Mentions are suppressed for the same reason the digest suppresses
        # them: a fact's text is written by a server member, and "@everyone"
        # reaching an automated post must be impossible by construction.
        await channel.send(embed=embed, allowed_mentions=discord.AllowedMentions.none())
    except Exception:
        logger.exception(
            "Onboarding post failed in channel %s (guild %s) for member %s",
            config.channel_id,
            guild.id,
            member.id,
        )
        return

    logger.info(
        "Posted onboarding for member %s in guild %s: %d fact(s) (%d rule(s), "
        "%d status change(s), %d other), %d omitted by the cap",
        member.id,
        guild.id,
        content.shown_count,
        len(content.rules),
        len(content.status_changes),
        len(content.other),
        content.omitted_count,
    )
