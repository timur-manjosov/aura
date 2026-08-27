"""/aura-link and /aura-unlink: a moderator declares two facts thematically related.

CLAUDE.md's fourth knowledge-model component has had a table since Phase 1b
and no way for a human to write a row into it. This is that way, and it is
deliberately the ONLY way: link detection is manual and mod-gated by design,
never inferred by a model. A link is an assertion about this specific server
that no amount of reading the two sentences can produce -- "the tournament
starts Saturday" and "the winner gets a month of Nitro" are one topic to the
person who runs the server and two unrelated sentences to an embedding model.

**Why there is no confirmation step, unlike /aura-supersede.** That command
asks before it commits because superseding a fact cannot be undone from
Discord: the command has no inverse. Linking does -- /aura-unlink is right
here, takes the same two IDs, and restores the exact prior state. A
confirmation dialog in front of a reversible action buys nothing and trains
moderators to click through the ones that matter.

**Both commands are ephemeral, like every other moderator tool here.** Linking
changes what Aura may cite later; it says nothing to the channel it was run in.

The pre-flight checks below duplicate what aura.db.repository re-checks
atomically at commit time, and that is not redundancy: these produce a
specific, friendly sentence naming what went wrong, while the repository's own
checks close the window between reading a fact and writing the link. Both are
needed, exactly as /aura-supersede uses both.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import aiosqlite
import discord
from discord import app_commands

from aura.db.models import Fact, FactStatus
from aura.db.repository import (
    FactNotActiveError,
    FactNotFoundError,
    get_fact_by_id,
    link_facts,
    unlink_facts,
)
from aura.i18n import t

if TYPE_CHECKING:
    from aura.main import AuraClient

logger = logging.getLogger(__name__)

# Comfortably under an embed field value's 1024-character hard cap, matching
# the display budget aura.commands.supersede and aura.commands.facts both use.
_FIELD_VALUE_DISPLAY_LIMIT = 200

# The range a fact ID can legitimately fall in, declared to Discord so it
# rejects anything outside it before an interaction is ever dispatched.
#
# Not cosmetic input validation. A fact ID is an AUTOINCREMENT rowid, so the
# lower bound is 1 -- 0 and negatives are always typos. The upper bound is
# Discord's own maximum for an integer option (2**53 - 1, the largest integer a
# JSON number carries exactly), and it is what keeps a pasted-in oversized
# number from reaching SQLite, which raises a bare OverflowError for anything
# past a signed 64-bit integer rather than reporting "no such fact". Declaring
# the range turns that into Discord telling the moderator their number is out
# of range, in their own language, with nothing logged.
_FactId = app_commands.Range[int, 1, 9007199254740991]


def _truncate(content: str, limit: int) -> str:
    """Truncate content to limit characters, appending an ellipsis if it was cut."""
    if len(content) <= limit:
        return content
    return content[: limit - 1] + "…"


async def _resolve_fact(
    db: aiosqlite.Connection, *, guild_id: int, fact_id: int, locale: str
) -> tuple[Fact | None, str | None]:
    """Look one fact up in this guild; return it, or the message explaining why not.

    Guild-scoped through get_fact_by_id, so a fact ID belonging to another
    server reads exactly like one that never existed -- a moderator must not
    be able to probe another guild's IDs through this command's error
    messages.
    """
    fact = await get_fact_by_id(db, guild_id=guild_id, fact_id=fact_id)
    if fact is None:
        return None, t("link_fact_not_found", locale, fact_id=fact_id)
    return fact, None


def _inactive_fact_message(fact: Fact, locale: str) -> str | None:
    """The message explaining why `fact` can't be linked, or None if it can be.

    A superseded fact gets the more useful of the two messages: it names the
    successor, because that successor is virtually always the fact the
    moderator actually meant to link. Falls back to the plain message if the
    chain is broken (superseded with no successor recorded), which is data
    only a hand edit can produce but which must still yield a sentence rather
    than a crash.
    """
    if fact.status is FactStatus.ACTIVE:
        return None
    if fact.superseded_by_id is not None:
        return t(
            "link_fact_superseded",
            locale,
            fact_id=fact.id,
            successor_id=fact.superseded_by_id,
        )
    return t("link_fact_not_active", locale, fact_id=fact.id)


def _build_link_embed(fact_a: Fact, fact_b: Fact, locale: str) -> discord.Embed:
    """Show both linked facts' content, so a moderator can verify the IDs they typed."""
    embed = discord.Embed(title=t("link_result_title", locale))
    for fact in (fact_a, fact_b):
        embed.add_field(
            name=t("link_fact_label", locale, fact_id=fact.id),
            value=_truncate(fact.content, _FIELD_VALUE_DISPLAY_LIMIT),
            inline=False,
        )
    return embed


async def _handle_link_command_error(
    interaction: discord.Interaction[AuraClient], error: app_commands.AppCommandError
) -> None:
    """Shared error handler for both link commands, matching every other mod-gated command's pattern.

    Attaching this via .error() stops CommandTree's default logging for these
    commands (it only logs when a command has no local handler), so anything
    other than the permission-check failure is logged here instead of
    silently disappearing.
    """
    if isinstance(error, app_commands.MissingPermissions):
        locale = str(interaction.locale)
        message = t("link_permission_error", locale)
        if interaction.response.is_done():
            await interaction.followup.send(message, ephemeral=True)
        else:
            await interaction.response.send_message(message, ephemeral=True)
        return

    command_name = interaction.command.name if interaction.command else "<unknown>"
    logger.error("Unhandled error in /%s", command_name, exc_info=error)


@app_commands.command(
    name="aura-link",
    description="Mark two facts as thematically related, so Aura can cite them together "
    "(moderators only).",
)
@app_commands.describe(
    fact_a_id="The ID (the #N shown by /aura-facts) of the first fact.",
    fact_b_id="The ID (the #N shown by /aura-facts) of the fact it belongs with.",
)
@app_commands.guild_only()
@app_commands.checks.has_permissions(manage_guild=True)
async def link_command(
    interaction: discord.Interaction[AuraClient], fact_a_id: _FactId, fact_b_id: _FactId
) -> None:
    """Validate both fact references, then link them; report whether anything changed."""
    assert interaction.guild_id is not None  # guaranteed by guild_only()
    locale = str(interaction.locale)
    guild_id = interaction.guild_id

    db = interaction.client.db
    assert db is not None  # setup_hook always finishes before commands go live

    if fact_a_id == fact_b_id:
        await interaction.response.send_message(t("link_self_error", locale), ephemeral=True)
        return

    facts: list[Fact] = []
    for fact_id in (fact_a_id, fact_b_id):
        fact, error_message = await _resolve_fact(
            db, guild_id=guild_id, fact_id=fact_id, locale=locale
        )
        if fact is None:
            assert error_message is not None  # _resolve_fact returns exactly one of the two
            await interaction.response.send_message(error_message, ephemeral=True)
            return
        facts.append(fact)

    for fact in facts:
        inactive_message = _inactive_fact_message(fact, locale)
        if inactive_message is not None:
            await interaction.response.send_message(inactive_message, ephemeral=True)
            return

    fact_a, fact_b = facts
    try:
        created = await link_facts(
            db, guild_id=guild_id, fact_id_1=fact_a.id, fact_id_2=fact_b.id
        )
    except FactNotFoundError:
        # Both facts existed a moment ago; one no longer does. Only a direct
        # database edit can produce this, but it must still be a sentence.
        await interaction.response.send_message(
            t("link_race_missing_error", locale), ephemeral=True
        )
        return
    except FactNotActiveError:
        # The realistic race: another moderator ran /aura-supersede on one of
        # these two between the check above and this write.
        await interaction.response.send_message(
            t("link_race_superseded_error", locale), ephemeral=True
        )
        return

    message_key = "link_success" if created else "link_already_linked"
    await interaction.response.send_message(
        content=t(message_key, locale, fact_a_id=fact_a.id, fact_b_id=fact_b.id),
        embed=_build_link_embed(fact_a, fact_b, locale),
        ephemeral=True,
    )


@app_commands.command(
    name="aura-unlink",
    description="Remove the thematic link between two facts (moderators only).",
)
@app_commands.describe(
    fact_a_id="The ID (the #N shown by /aura-facts) of the first fact.",
    fact_b_id="The ID (the #N shown by /aura-facts) of the fact to unlink it from.",
)
@app_commands.guild_only()
@app_commands.checks.has_permissions(manage_guild=True)
async def unlink_command(
    interaction: discord.Interaction[AuraClient], fact_a_id: _FactId, fact_b_id: _FactId
) -> None:
    """Remove the link between two facts of this guild, reporting whether one existed.

    Deliberately does NOT require either fact to still be active, mirroring
    aura.db.repository.unlink_facts: the links most worth cleaning up are
    exactly the ones whose facts have moved on since.
    """
    assert interaction.guild_id is not None  # guaranteed by guild_only()
    locale = str(interaction.locale)
    guild_id = interaction.guild_id

    db = interaction.client.db
    assert db is not None  # setup_hook always finishes before commands go live

    if fact_a_id == fact_b_id:
        await interaction.response.send_message(t("unlink_self_error", locale), ephemeral=True)
        return

    for fact_id in (fact_a_id, fact_b_id):
        fact, error_message = await _resolve_fact(
            db, guild_id=guild_id, fact_id=fact_id, locale=locale
        )
        if fact is None:
            assert error_message is not None  # _resolve_fact returns exactly one of the two
            await interaction.response.send_message(error_message, ephemeral=True)
            return

    removed = await unlink_facts(
        db, guild_id=guild_id, fact_id_1=fact_a_id, fact_id_2=fact_b_id
    )
    message_key = "unlink_success" if removed else "unlink_not_linked"
    await interaction.response.send_message(
        t(message_key, locale, fact_a_id=fact_a_id, fact_b_id=fact_b_id), ephemeral=True
    )


link_command.error(_handle_link_command_error)
unlink_command.error(_handle_link_command_error)


def register_link_commands(tree: app_commands.CommandTree) -> None:
    """Register /aura-link and /aura-unlink onto tree."""
    tree.add_command(link_command)
    tree.add_command(unlink_command)
