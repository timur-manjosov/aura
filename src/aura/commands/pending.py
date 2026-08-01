"""/aura-pending: a moderator's review of one automatically extracted fact candidate.

The human gate in front of Phase 3a-2's automatic path. Everything upstream --
the local filter, the batch, the distillation call, the staging table -- exists
to produce a sentence a moderator can accept or reject in one look; this is
where that acceptance happens, and it is the only way a candidate ever becomes
a real, citable fact.

**Deliberately one candidate at a time, not a bulk review queue.** The
originally-planned bundled review workflow exists for a volume problem that
belongs to a different sub-phase: backfilling a server's history can produce
hundreds of candidates at once, and reviewing those one by one would be
unusable. Live extraction produces candidates at a trickle -- a handful a day
in an active channel -- so the bundled workflow would be machinery built ahead
of the problem it solves. This is the smallest thing that makes the whole
pipeline testable end to end by a human, and no more.

Mod-gated on manage_guild, the same permission every other Aura configuration
and fact command uses, so "who may decide what Aura knows" has one consistent
answer.

**Phase 3a-3: the review now carries a judgement, and one of the four possible
judgements looks different from the other three.** Where the candidate may
restate an existing fact, a model has already been asked what that similarity
means (see aura.extraction.supersession), and the answer is shown here with the
reason it gave. Three of the four answers are ordinary information a moderator
reads and acts on. The fourth -- a contradiction, two facts that cannot both
hold with nothing in either wording saying which is current -- is the one case
where confirming without looking is actively harmful, so it is rendered
differently in three ways at once rather than with one extra word in the same
shape as everything else: the embed turns red, the field carries a warning icon,
and the hint tells the moderator to open both source messages instead of
offering them a next command to run. The three signals are deliberately
redundant, because any one of them alone is easy to skim past.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import aiosqlite
import discord
from discord import app_commands
from fastembed import TextEmbedding

from aura.db.pending_facts import (
    PendingFact,
    PendingFactAlreadyResolvedError,
    PendingFactNotFoundError,
    SupersessionRelationship,
    count_pending_facts,
    discard_pending_fact,
    get_pending_facts,
)
from aura.db.repository import get_fact_by_id
from aura.facts_service import confirm_fact
from aura.i18n import t

if TYPE_CHECKING:
    from aura.main import AuraClient

logger = logging.getLogger(__name__)

# Long enough to read a sentence, check its source message and decide; short
# enough that a forgotten prompt does not sit around holding a stale view of a
# candidate someone else may since have resolved. Same value, same reasoning,
# as /aura-supersede's confirmation.
_CONFIRMATION_TIMEOUT_SECONDS = 60.0

# Comfortably under an embed field value's 1024-character hard cap. A distilled
# candidate is capped at 500 characters upstream (see
# aura.extraction.distiller), so this only ever binds on a possible predecessor
# fact, which was entered by hand and has no such limit.
_FIELD_VALUE_DISPLAY_LIMIT = 200

# The colour a contradiction turns the whole embed. Set for that one judgement
# and for nothing else, so "red" carries exactly one meaning in this command
# rather than becoming decoration: these two facts disagree and Aura will not
# guess which of them is current.
_CONTRADICTION_COLOUR = discord.Colour.red()

# The second of the three redundant signals (see the module docstring). Kept out
# of the translation files on purpose: it is not text and must not vary by
# locale -- a moderator switching languages must still recognise the same
# marker, and a translator must not be able to soften it by leaving it out.
_CONTRADICTION_ICON = "⚠️"

# The judgement's reasoning sentence is capped at 600 characters where it is
# produced (aura.extraction.supersession), so this only binds on a row written
# by something else -- and it stays well under an embed field value's 1024-
# character hard cap either way.
_REASONING_DISPLAY_LIMIT = 600


def _truncate(content: str, limit: int) -> str:
    """Truncate content to limit characters, appending an ellipsis if it was cut."""
    if len(content) <= limit:
        return content
    return content[: limit - 1] + "…"


def _message_link(candidate: PendingFact) -> str:
    """The Discord permalink to the message a candidate was distilled from.

    The whole reason a candidate stores IDs instead of the source text: a
    moderator confirming a machine-written sentence must be able to read what
    it was written from, in one click, rather than trusting the distillation.
    """
    return (
        f"https://discord.com/channels/{candidate.guild_id}/"
        f"{candidate.channel_id}/{candidate.message_id}"
    )


def _add_relationship_fields(
    embed: discord.Embed, candidate: PendingFact, *, similar_fact_id: int, locale: str
) -> None:
    """Append what the judgement said about this pair, and what to do about it.

    Handles the un-judged case first because it is not an error state: a
    candidate carries no judgement whenever the daily cap was spent, the call
    failed, or no supersession model is configured at all, and in every one of
    those the correct thing to show is exactly what Phase 3a-2 showed -- the
    plain hint that this candidate may replace the fact above it, with the
    decision entirely the moderator's. A missing judgement must never read as
    "judged, and found nothing".

    The judged cases render three fields: what the relationship is, the model's
    own reasoning (labelled as the model's, not as Aura's finding), and what
    that means for the moderator's next action. Only the contradiction hint
    declines to name a next command, because there is no correct one to offer
    until a human has looked at both source messages.
    """
    relationship = candidate.relationship
    if relationship is None:
        embed.add_field(
            name="​",
            value=t("pending_similar_hint", locale, fact_id=similar_fact_id),
            inline=False,
        )
        return

    label = t("pending_relationship_label", locale)
    if relationship is SupersessionRelationship.CONTRADICTION:
        label = f"{_CONTRADICTION_ICON} {label}"
    embed.add_field(
        name=label,
        value=t(f"pending_relationship_{relationship.value}", locale),
        inline=False,
    )

    # Written in the candidate's own language rather than the moderator's (see
    # aura.extraction.supersession): Aura does not know a fact's language, so
    # the model writes its reason in the language of the sentence it judged,
    # exactly as the two facts above it are shown untranslated.
    if candidate.relationship_reasoning:
        embed.add_field(
            name=t("pending_relationship_reasoning_label", locale),
            value=_truncate(candidate.relationship_reasoning, _REASONING_DISPLAY_LIMIT),
            inline=False,
        )

    embed.add_field(
        name="​",
        value=t(
            f"pending_relationship_hint_{relationship.value}",
            locale,
            fact_id=similar_fact_id,
        ),
        inline=False,
    )


async def _build_candidate_embed(
    db: aiosqlite.Connection, candidate: PendingFact, *, remaining: int, locale: str
) -> discord.Embed:
    """Render one candidate for review: the sentence, its category, source, and dedup hint."""
    embed = discord.Embed(
        title=t("pending_review_title", locale, pending_id=candidate.id),
        description=candidate.content,
    )
    # The first of the three redundant contradiction signals, set on the whole
    # embed rather than on the field that says it: colour is the one property a
    # moderator sees before reading anything at all.
    if candidate.relationship is SupersessionRelationship.CONTRADICTION:
        embed.colour = _CONTRADICTION_COLOUR
    embed.add_field(
        name=t("pending_category_label", locale),
        value=t(f"pending_category_{candidate.category.value}", locale),
        inline=True,
    )
    embed.add_field(
        name=t("pending_source_label", locale),
        value=_message_link(candidate),
        inline=False,
    )

    # The advisory dedup hint. Resolved to the fact's actual text rather than
    # shown as a bare ID, because "this may replace #12" is only actionable if
    # the moderator can see what #12 says without running another command.
    if candidate.similar_fact_id is not None:
        similar = await get_fact_by_id(
            db, guild_id=candidate.guild_id, fact_id=candidate.similar_fact_id
        )
        if similar is not None:
            embed.add_field(
                name=t(
                    "pending_similar_label",
                    locale,
                    fact_id=similar.id,
                    score=f"{candidate.similar_fact_score:.2f}"
                    if candidate.similar_fact_score is not None
                    else "?",
                ),
                value=_truncate(similar.content, _FIELD_VALUE_DISPLAY_LIMIT),
                inline=False,
            )
            _add_relationship_fields(
                embed, candidate, similar_fact_id=similar.id, locale=locale
            )

    embed.set_footer(text=t("pending_review_footer", locale, remaining=remaining))
    return embed


class PendingReviewView(discord.ui.View):
    """Confirm/discard buttons for one staged candidate.

    interaction_check restricts both buttons to the moderator who ran the
    command, for the same belt-and-suspenders reason /aura-supersede's view
    does: the reply is already ephemeral, but visibility is not the property
    being enforced -- authority is.

    The `_resolved` flag guards a genuine double-fire on THIS view (a client
    sending Confirm twice before the first edit propagates). It is not what
    makes the two-moderator race safe -- two moderators have two views and two
    flags. That race is settled in the database, by the guarded UPDATE in
    aura.db.pending_facts, and the errors it raises are handled below.
    """

    def __init__(
        self,
        *,
        db: aiosqlite.Connection,
        model: TextEmbedding,
        guild_id: int,
        candidate: PendingFact,
        locale: str,
        invoker_id: int,
    ) -> None:
        super().__init__(timeout=_CONFIRMATION_TIMEOUT_SECONDS)
        self._db = db
        self._model = model
        self._guild_id = guild_id
        self._candidate = candidate
        self._locale = locale
        self._invoker_id = invoker_id
        # Set post-super().__init__(): only once _init_children() has run do
        # self.confirm/self.discard refer to the real Button instances rather
        # than the plain decorated functions, so labels can be localized here.
        self.confirm.label = t("pending_confirm_button", locale)
        self.discard.label = t("pending_discard_button", locale)
        self.message: discord.Message | None = None
        # Checked and set synchronously with no `await` in between, so two
        # concurrently dispatched callbacks on this same view cannot both pass.
        self._resolved = False

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        """Reject a button press from anyone but the moderator who ran the command."""
        if interaction.user.id != self._invoker_id:
            await interaction.response.send_message(
                t("pending_wrong_user_error", self._locale), ephemeral=True
            )
            return False
        return True

    @discord.ui.button(style=discord.ButtonStyle.success)
    async def confirm(self, interaction: discord.Interaction, _button: discord.ui.Button) -> None:
        """Turn the candidate into a real active fact, or report that someone else got there first."""
        if self._resolved:
            if not interaction.response.is_done():
                await interaction.response.defer()
            return
        self._resolved = True

        try:
            fact = await confirm_fact(
                self._db,
                self._model,
                guild_id=self._guild_id,
                pending_id=self._candidate.id,
                resolved_by_id=interaction.user.id,
            )
        except (PendingFactAlreadyResolvedError, PendingFactNotFoundError):
            # Another moderator confirmed or discarded this candidate between
            # this view being shown and this button being pressed. Nothing was
            # written by this call -- the guarded UPDATE saw to that -- so the
            # only thing left is to say so.
            await interaction.response.edit_message(
                content=t("pending_race_error", self._locale, pending_id=self._candidate.id),
                embed=None,
                view=None,
            )
            self.stop()
            return

        await interaction.response.edit_message(
            content=t(
                "pending_confirmed",
                self._locale,
                pending_id=self._candidate.id,
                fact_id=fact.id,
            ),
            embed=None,
            view=None,
        )
        self.stop()

    @discord.ui.button(style=discord.ButtonStyle.secondary)
    async def discard(self, interaction: discord.Interaction, _button: discord.ui.Button) -> None:
        """Reject the candidate. No fact is ever written, and the rejection is kept as evidence."""
        if self._resolved:
            if not interaction.response.is_done():
                await interaction.response.defer()
            return
        self._resolved = True

        try:
            await discard_pending_fact(
                self._db,
                guild_id=self._guild_id,
                pending_id=self._candidate.id,
                resolved_by_id=interaction.user.id,
            )
        except (PendingFactAlreadyResolvedError, PendingFactNotFoundError):
            await interaction.response.edit_message(
                content=t("pending_race_error", self._locale, pending_id=self._candidate.id),
                embed=None,
                view=None,
            )
            self.stop()
            return

        await interaction.response.edit_message(
            content=t("pending_discarded", self._locale, pending_id=self._candidate.id),
            embed=None,
            view=None,
        )
        self.stop()

    async def on_timeout(self) -> None:
        """A moderator who never answers leaves the candidate exactly as it was."""
        if self._resolved:
            # self.stop() cancels discord.py's own timeout task, so this
            # shouldn't normally fire after a button ran -- checked anyway so a
            # scheduling edge case cannot double-edit a finalized message.
            return
        self._resolved = True
        if self.message is not None:
            await self.message.edit(
                content=t("pending_expired", self._locale), embed=None, view=None
            )


async def _handle_pending_command_error(
    interaction: discord.Interaction[AuraClient], error: app_commands.AppCommandError
) -> None:
    """Shared error handler, matching every other mod-gated command's pattern.

    Attaching this via .error() stops CommandTree's default logging for this
    command (it only logs when a command has no local handler), so anything
    other than the permission failure is logged here rather than silently
    disappearing.
    """
    if isinstance(error, app_commands.MissingPermissions):
        locale = str(interaction.locale)
        message = t("pending_permission_error", locale)
        if interaction.response.is_done():
            await interaction.followup.send(message, ephemeral=True)
        else:
            await interaction.response.send_message(message, ephemeral=True)
        return

    logger.error("Unhandled error in /aura-pending", exc_info=error)


@app_commands.command(
    name="aura-pending",
    description="Review the next automatically extracted fact candidate (moderators only).",
)
@app_commands.guild_only()
@app_commands.checks.has_permissions(manage_guild=True)
async def pending_command(interaction: discord.Interaction[AuraClient]) -> None:
    """Show the oldest unreviewed candidate with confirm/discard buttons.

    Oldest first because this is a work queue: reviewing newest-first would let
    a candidate sit at the bottom indefinitely while newer ones keep landing on
    top of it. Running the command again after resolving one shows the next.
    """
    assert interaction.guild_id is not None  # guaranteed by guild_only()
    locale = str(interaction.locale)

    db = interaction.client.db
    assert db is not None  # setup_hook always finishes before commands go live
    model = interaction.client.embedding_model
    assert model is not None  # setup_hook always finishes before commands go live

    candidates = await get_pending_facts(db, guild_id=interaction.guild_id, limit=1)
    if not candidates:
        await interaction.response.send_message(t("pending_none", locale), ephemeral=True)
        return

    candidate = candidates[0]
    remaining = await count_pending_facts(db, guild_id=interaction.guild_id)
    embed = await _build_candidate_embed(db, candidate, remaining=remaining, locale=locale)

    view = PendingReviewView(
        db=db,
        model=model,
        guild_id=interaction.guild_id,
        candidate=candidate,
        locale=locale,
        invoker_id=interaction.user.id,
    )
    # Ephemeral: reviewing a candidate is the moderator's work, not an
    # announcement to the channel they happen to run it in.
    await interaction.response.send_message(embed=embed, view=view, ephemeral=True)
    view.message = await interaction.original_response()


pending_command.error(_handle_pending_command_error)


def register_pending_command(tree: app_commands.CommandTree) -> None:
    """Register /aura-pending onto tree."""
    tree.add_command(pending_command)
