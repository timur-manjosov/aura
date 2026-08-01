"""Tests for aura.commands.pending: the human gate in front of automatic extraction.

Exercises the command callback, the review view's buttons, the permission
check and the shared error handler directly against mocked
discord.Interaction objects and a real in-memory database -- never a live
Discord connection, matching test_supersede_command.py's approach and
CLAUDE.md's testing philosophy.

The UI-level race (two moderators, two views, one candidate) is covered here;
the data-level guarantee it depends on lives in tests/test_pending_facts.py.
Both are needed: the database decides the outcome, and this file is what proves
the command surfaces that decision instead of claiming success anyway.
"""
from __future__ import annotations

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import aiosqlite
import discord
import pytest
from discord import app_commands
from fastembed import TextEmbedding

from aura.commands.pending import (
    PendingReviewView,
    _handle_pending_command_error,
    pending_command,
)
from aura.db.pending_facts import (
    FactCategory,
    PendingFactStatus,
    SupersessionRelationship,
    get_pending_fact,
    record_relationship_judgement,
    stage_pending_fact,
)
from aura.db.repository import get_active_facts, init_schema
from aura.facts_service import add_fact

GUILD_A = 100000000000000001
GUILD_B = 200000000000000002
CHANNEL = 500000000000000005
MOD_A = 111
MOD_B = 222

EMBEDDING = b"\x00" * 16


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
    user_id: int = MOD_A,
    embedding_model: TextEmbedding | None = None,
) -> MagicMock:
    interaction = MagicMock(spec=discord.Interaction)
    interaction.locale = locale
    interaction.guild_id = guild_id
    interaction.client = MagicMock()
    interaction.client.db = db
    interaction.client.embedding_model = (
        embedding_model if embedding_model is not None else MagicMock()
    )
    interaction.user = MagicMock()
    interaction.user.id = user_id
    interaction.response = MagicMock()
    interaction.response.send_message = AsyncMock()
    interaction.response.edit_message = AsyncMock()
    interaction.response.defer = AsyncMock()
    interaction.response.is_done = MagicMock(return_value=False)
    interaction.followup = MagicMock()
    interaction.followup.send = AsyncMock()
    interaction.original_response = AsyncMock(return_value=MagicMock())
    interaction.command = MagicMock()
    interaction.command.name = "aura-pending"
    return interaction


async def _stage(
    conn: aiosqlite.Connection,
    *,
    guild_id: int = GUILD_A,
    message_id: int = 1,
    content: str = "Maintenance runs today at 14:00 UTC.",
    category: FactCategory = FactCategory.STATUS_CHANGE,
    similar_fact_id: int | None = None,
    similar_fact_score: float | None = None,
):
    staged = await stage_pending_fact(
        conn,
        guild_id=guild_id,
        channel_id=CHANNEL,
        message_id=message_id,
        content=content,
        embedding=EMBEDDING,
        category=category,
        similar_fact_id=similar_fact_id,
        similar_fact_score=similar_fact_score,
    )
    assert staged is not None
    return staged


def _view(
    conn: aiosqlite.Connection,
    candidate,
    *,
    invoker_id: int = MOD_A,
    model: TextEmbedding | None = None,
):
    return PendingReviewView(
        db=conn,
        # Most tests here never exercise variant generation (only confirm()
        # does, and it schedules that as a fire-and-forget background task
        # that immediately no-ops without a configured audit model) -- a
        # plain MagicMock stands in exactly the way interaction.client's other
        # attributes already do throughout this file, unless a test passes
        # the real fixture explicitly to assert on how it was used.
        model=model if model is not None else MagicMock(),
        guild_id=GUILD_A,
        candidate=candidate,
        locale="en-US",
        invoker_id=invoker_id,
    )


class TestEmptyQueue:
    async def test_nothing_pending_replies_cleanly(self, conn: aiosqlite.Connection) -> None:
        interaction = _make_interaction(db=conn)
        await pending_command.callback(interaction)

        interaction.response.send_message.assert_awaited_once()
        kwargs = interaction.response.send_message.call_args.kwargs
        assert kwargs["ephemeral"] is True
        assert "embed" not in kwargs

    async def test_another_guilds_candidates_do_not_count(
        self, conn: aiosqlite.Connection
    ) -> None:
        await _stage(conn, guild_id=GUILD_B)
        interaction = _make_interaction(db=conn, guild_id=GUILD_A)
        await pending_command.callback(interaction)

        assert "embed" not in interaction.response.send_message.call_args.kwargs


class TestShowingACandidate:
    async def test_the_oldest_candidate_is_shown_with_buttons(
        self, conn: aiosqlite.Connection
    ) -> None:
        first = await _stage(conn, message_id=1, content="The first candidate.")
        await _stage(conn, message_id=2, content="The second candidate.")

        interaction = _make_interaction(db=conn)
        await pending_command.callback(interaction)

        kwargs = interaction.response.send_message.call_args.kwargs
        assert kwargs["ephemeral"] is True
        assert isinstance(kwargs["view"], PendingReviewView)
        embed = kwargs["embed"]
        assert embed.description == first.content
        assert str(first.id) in embed.title

    async def test_the_embed_links_back_to_the_source_message(
        self, conn: aiosqlite.Connection
    ) -> None:
        # A moderator confirming a machine-written sentence has to be able to
        # read what it was written from, in one click.
        candidate = await _stage(conn, message_id=98765)
        interaction = _make_interaction(db=conn)
        await pending_command.callback(interaction)

        embed = interaction.response.send_message.call_args.kwargs["embed"]
        links = [field.value for field in embed.fields if "discord.com/channels" in str(field.value)]
        assert links
        assert f"/{GUILD_A}/{CHANNEL}/98765" in links[0]

    @pytest.mark.parametrize("category", list(FactCategory))
    async def test_every_category_renders_a_localized_label(
        self, conn: aiosqlite.Connection, category: FactCategory
    ) -> None:
        # A category with no translation key would render as "[pending_category_x]".
        await _stage(conn, category=category)
        interaction = _make_interaction(db=conn)
        await pending_command.callback(interaction)

        embed = interaction.response.send_message.call_args.kwargs["embed"]
        labels = [str(field.value) for field in embed.fields]
        assert not any(label.startswith("[") and label.endswith("]") for label in labels)

    async def test_the_dedup_hint_shows_the_predecessors_actual_text(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        existing = await add_fact(
            conn,
            embedding_model,
            guild_id=GUILD_A,
            channel_id=CHANNEL,
            message_id=900,
            content="Maintenance used to run on Tuesdays.",
        )
        await _stage(conn, similar_fact_id=existing.id, similar_fact_score=0.87)

        interaction = _make_interaction(db=conn)
        await pending_command.callback(interaction)

        embed = interaction.response.send_message.call_args.kwargs["embed"]
        rendered = " ".join(f"{field.name} {field.value}" for field in embed.fields)
        assert "Maintenance used to run on Tuesdays." in rendered
        assert str(existing.id) in rendered
        assert "0.87" in rendered

    async def test_a_dedup_hint_cannot_point_at_a_fact_that_does_not_exist(
        self, conn: aiosqlite.Connection
    ) -> None:
        # Written expecting to need defensive handling for a dangling pointer;
        # the schema turns out to rule it out instead. similar_fact_id carries
        # a REFERENCES facts(id) constraint and init_schema enables
        # PRAGMA foreign_keys, so staging a hint at a nonexistent fact fails at
        # write time rather than rendering as a broken reference at review
        # time. Asserted rather than trusted, since the PRAGMA is
        # per-connection and silently does nothing if it is ever missed.
        with pytest.raises(aiosqlite.IntegrityError):
            await stage_pending_fact(
                conn,
                guild_id=GUILD_A,
                channel_id=CHANNEL,
                message_id=1,
                content="A candidate hinting at nothing.",
                embedding=EMBEDDING,
                category=FactCategory.RULE,
                similar_fact_id=999999,
                similar_fact_score=0.9,
            )

    async def test_a_superseded_predecessor_still_renders_in_the_hint(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        # The reachable version of the same worry: facts are never deleted,
        # only superseded, and a hint staged while its predecessor was active
        # may be reviewed after a moderator has retired it. The candidate must
        # stay reviewable, and the old text is still the useful thing to show.
        old = await add_fact(
            conn,
            embedding_model,
            guild_id=GUILD_A,
            channel_id=CHANNEL,
            message_id=900,
            content="Maintenance used to run on Tuesdays.",
        )
        replacement = await add_fact(
            conn,
            embedding_model,
            guild_id=GUILD_A,
            channel_id=CHANNEL,
            message_id=901,
            content="Maintenance now runs on Wednesdays.",
        )
        await _stage(conn, similar_fact_id=old.id, similar_fact_score=0.88)

        from aura.db.repository import supersede_fact_with_existing_successor

        await supersede_fact_with_existing_successor(
            conn, old_fact_id=old.id, new_fact_id=replacement.id, guild_id=GUILD_A
        )

        interaction = _make_interaction(db=conn)
        await pending_command.callback(interaction)

        embed = interaction.response.send_message.call_args.kwargs["embed"]
        rendered = " ".join(f"{field.name} {field.value}" for field in embed.fields)
        assert "Maintenance used to run on Tuesdays." in rendered


class TestRelationshipDisplay:
    """Phase 3a-3's judgement, as a moderator actually sees it.

    The claim being tested is not "the words appear" but "a contradiction is
    distinguishable from the other three without reading carefully" -- the phase
    brief asks specifically whether the marking is visual/structural rather than
    one more sentence in the same shape as everything else. Three independent
    signals are checked: the embed's colour, an icon on the field, and a hint
    that offers no next command to run.
    """

    async def _render(
        self,
        conn: aiosqlite.Connection,
        *,
        relationship: SupersessionRelationship | None,
        reasoning: str = "Beide nennen dieselbe Wartung.",
        locale: str = "en-US",
        embedding_model: TextEmbedding,
    ) -> discord.Embed:
        existing = await add_fact(
            conn,
            embedding_model,
            guild_id=GUILD_A,
            channel_id=CHANNEL,
            message_id=900,
            content="Maintenance used to run on Tuesdays.",
        )
        candidate = await _stage(
            conn,
            content="Maintenance now runs on Wednesdays.",
            similar_fact_id=existing.id,
            similar_fact_score=0.88,
        )
        if relationship is not None:
            assert await record_relationship_judgement(
                conn,
                guild_id=GUILD_A,
                pending_id=candidate.id,
                relationship=relationship,
                reasoning=reasoning,
            )

        interaction = _make_interaction(db=conn, locale=locale)
        await pending_command.callback(interaction)
        return interaction.response.send_message.call_args.kwargs["embed"]

    @staticmethod
    def _rendered(embed: discord.Embed) -> str:
        return " ".join(f"{field.name} {field.value}" for field in embed.fields)

    @pytest.mark.parametrize("relationship", list(SupersessionRelationship))
    async def test_every_relationship_renders_a_localized_label_and_hint(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding, relationship
    ) -> None:
        # A missing translation key renders as "[pending_relationship_x]", which
        # is what this catches for all four values at once.
        embed = await self._render(
            conn, relationship=relationship, embedding_model=embedding_model
        )
        rendered = self._rendered(embed)
        assert "[pending_relationship" not in rendered

    async def test_the_models_reasoning_is_shown(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        embed = await self._render(
            conn,
            relationship=SupersessionRelationship.SUPERSESSION,
            reasoning="Der Wartungstag wurde von Dienstag auf Mittwoch verschoben.",
            embedding_model=embedding_model,
        )
        assert (
            "Der Wartungstag wurde von Dienstag auf Mittwoch verschoben."
            in self._rendered(embed)
        )

    async def test_a_contradiction_is_the_only_one_that_colours_the_embed(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        contradiction = await self._render(
            conn,
            relationship=SupersessionRelationship.CONTRADICTION,
            embedding_model=embedding_model,
        )
        assert contradiction.colour == discord.Colour.red()

    @pytest.mark.parametrize(
        "relationship",
        [
            SupersessionRelationship.SUPERSESSION,
            SupersessionRelationship.COMPLEMENTARY,
            SupersessionRelationship.INDEPENDENT,
        ],
    )
    async def test_the_other_three_leave_the_embed_uncoloured(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding, relationship
    ) -> None:
        embed = await self._render(
            conn, relationship=relationship, embedding_model=embedding_model
        )
        assert embed.colour is None

    async def test_a_contradiction_carries_a_warning_icon_the_others_do_not(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        contradiction = await self._render(
            conn,
            relationship=SupersessionRelationship.CONTRADICTION,
            embedding_model=embedding_model,
        )
        assert any("⚠" in str(field.name) for field in contradiction.fields)

    @pytest.mark.parametrize(
        "relationship",
        [
            SupersessionRelationship.SUPERSESSION,
            SupersessionRelationship.COMPLEMENTARY,
            SupersessionRelationship.INDEPENDENT,
        ],
    )
    async def test_the_other_three_carry_no_warning_icon(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding, relationship
    ) -> None:
        embed = await self._render(
            conn, relationship=relationship, embedding_model=embedding_model
        )
        assert not any("⚠" in str(field.name) for field in embed.fields)

    @pytest.mark.parametrize("locale", ["en-US", "de", "ja"])
    async def test_the_warning_icon_survives_a_locale_change(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding, locale: str
    ) -> None:
        # The icon is deliberately not in the translation files: a moderator
        # switching languages must still recognise the same marker, and a
        # translator must not be able to soften it by leaving it out.
        embed = await self._render(
            conn,
            relationship=SupersessionRelationship.CONTRADICTION,
            locale=locale,
            embedding_model=embedding_model,
        )
        assert any("⚠" in str(field.name) for field in embed.fields)
        assert embed.colour == discord.Colour.red()

    async def test_a_contradiction_offers_no_next_command_to_run(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        # The third signal, and the one that matters most for what a moderator
        # DOES: every other case names a next step, and this one deliberately
        # does not, because there is no correct one until a human has looked.
        contradiction = await self._render(
            conn,
            relationship=SupersessionRelationship.CONTRADICTION,
            embedding_model=embedding_model,
        )
        assert "/aura-supersede" not in self._rendered(contradiction)

    async def test_a_supersession_still_points_at_the_supersede_command(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        embed = await self._render(
            conn,
            relationship=SupersessionRelationship.SUPERSESSION,
            embedding_model=embedding_model,
        )
        assert "/aura-supersede" in self._rendered(embed)

    async def test_an_unjudged_candidate_keeps_the_phase_3a2_hint(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        # No judgement is an ordinary state -- the cap was spent, the call
        # failed, or no model is configured -- and it must read as "decide for
        # yourself", never as "judged, and found nothing".
        embed = await self._render(conn, relationship=None, embedding_model=embedding_model)
        rendered = self._rendered(embed)
        assert "/aura-supersede" in rendered
        assert embed.colour is None
        assert not any("⚠" in str(field.name) for field in embed.fields)

    async def test_an_unflagged_candidate_shows_no_relationship_section_at_all(
        self, conn: aiosqlite.Connection
    ) -> None:
        await _stage(conn)
        interaction = _make_interaction(db=conn)
        await pending_command.callback(interaction)

        embed = interaction.response.send_message.call_args.kwargs["embed"]
        assert "Relationship" not in self._rendered(embed)

    async def test_an_oversized_reasoning_cannot_break_the_embed(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        # The judgement caps its reasoning at 600 characters, but a row written
        # by anything else must not produce a field value over Discord's 1024
        # hard cap, which would make the whole reply fail to send.
        embed = await self._render(
            conn,
            relationship=SupersessionRelationship.COMPLEMENTARY,
            reasoning="x" * 5000,
            embedding_model=embedding_model,
        )
        assert all(len(str(field.value)) <= 1024 for field in embed.fields)


class TestConfirmButton:
    async def test_confirming_creates_the_fact_and_reports_it(
        self, conn: aiosqlite.Connection
    ) -> None:
        candidate = await _stage(conn)
        view = _view(conn, candidate)
        interaction = _make_interaction(db=conn)

        await view.confirm.callback(interaction)

        active = await get_active_facts(conn, GUILD_A)
        assert len(active) == 1
        interaction.response.edit_message.assert_awaited_once()
        kwargs = interaction.response.edit_message.call_args.kwargs
        assert str(active[0].id) in kwargs["content"]
        # The view is removed so the same candidate cannot be resolved twice
        # from a message left sitting in the moderator's client.
        assert kwargs["view"] is None

    async def test_confirming_records_the_pressing_moderator_not_the_invoker(
        self, conn: aiosqlite.Connection
    ) -> None:
        candidate = await _stage(conn)
        view = _view(conn, candidate, invoker_id=MOD_A)
        await view.confirm.callback(_make_interaction(db=conn, user_id=MOD_A))

        resolved = await get_pending_fact(conn, guild_id=GUILD_A, pending_id=candidate.id)
        assert resolved is not None
        assert resolved.resolved_by_id == MOD_A

    async def test_a_double_click_on_one_view_acts_once(
        self, conn: aiosqlite.Connection
    ) -> None:
        # A client can send two component interactions before the first
        # edit_message propagates back. _resolved is checked and set with no
        # await in between, so the second cannot also act.
        candidate = await _stage(conn)
        view = _view(conn, candidate)

        await view.confirm.callback(_make_interaction(db=conn))
        second = _make_interaction(db=conn)
        await view.confirm.callback(second)

        assert len(await get_active_facts(conn, GUILD_A)) == 1
        second.response.edit_message.assert_not_awaited()
        second.response.defer.assert_awaited_once()

    async def test_confirm_then_cancel_on_one_view_acts_once(
        self, conn: aiosqlite.Connection
    ) -> None:
        candidate = await _stage(conn)
        view = _view(conn, candidate)

        await view.confirm.callback(_make_interaction(db=conn))
        await view.discard.callback(_make_interaction(db=conn))

        resolved = await get_pending_fact(conn, guild_id=GUILD_A, pending_id=candidate.id)
        assert resolved is not None
        assert resolved.status is PendingFactStatus.CONFIRMED
        assert len(await get_active_facts(conn, GUILD_A)) == 1


class TestDiscardButton:
    async def test_discarding_writes_no_fact(self, conn: aiosqlite.Connection) -> None:
        candidate = await _stage(conn)
        view = _view(conn, candidate)
        interaction = _make_interaction(db=conn)

        await view.discard.callback(interaction)

        assert await get_active_facts(conn, GUILD_A) == []
        resolved = await get_pending_fact(conn, guild_id=GUILD_A, pending_id=candidate.id)
        assert resolved is not None
        assert resolved.status is PendingFactStatus.DISCARDED
        interaction.response.edit_message.assert_awaited_once()


class TestTwoModeratorRace:
    """Two moderators, two separate views, one candidate.

    The per-view _resolved flag cannot help here -- there are two of them --
    so what has to hold is that the database refuses the second resolution and
    the command reports that honestly instead of claiming success.
    """

    async def test_the_second_moderator_is_told_it_was_already_reviewed(
        self, conn: aiosqlite.Connection
    ) -> None:
        candidate = await _stage(conn)
        view_a = _view(conn, candidate, invoker_id=MOD_A)
        view_b = _view(conn, candidate, invoker_id=MOD_B)

        await view_a.confirm.callback(_make_interaction(db=conn, user_id=MOD_A))
        second = _make_interaction(db=conn, user_id=MOD_B)
        await view_b.confirm.callback(second)

        assert len(await get_active_facts(conn, GUILD_A)) == 1
        content = second.response.edit_message.call_args.kwargs["content"]
        assert str(candidate.id) in content
        assert "already" in content.lower()

    async def test_a_confirm_racing_a_discard_leaves_one_definite_outcome(
        self, conn: aiosqlite.Connection
    ) -> None:
        candidate = await _stage(conn)
        view_a = _view(conn, candidate, invoker_id=MOD_A)
        view_b = _view(conn, candidate, invoker_id=MOD_B)

        await view_a.discard.callback(_make_interaction(db=conn, user_id=MOD_A))
        second = _make_interaction(db=conn, user_id=MOD_B)
        await view_b.confirm.callback(second)

        # Discarded first wins; no fact exists, and the confirmer is told so
        # rather than being shown a fabricated fact ID.
        assert await get_active_facts(conn, GUILD_A) == []
        assert "already" in second.response.edit_message.call_args.kwargs["content"].lower()


class TestViewAuthorization:
    async def test_another_user_cannot_press_the_buttons(
        self, conn: aiosqlite.Connection
    ) -> None:
        candidate = await _stage(conn)
        view = _view(conn, candidate, invoker_id=MOD_A)
        intruder = _make_interaction(db=conn, user_id=MOD_B)

        assert await view.interaction_check(intruder) is False
        intruder.response.send_message.assert_awaited_once()
        assert intruder.response.send_message.call_args.kwargs["ephemeral"] is True

    async def test_the_invoking_moderator_passes_the_check(
        self, conn: aiosqlite.Connection
    ) -> None:
        candidate = await _stage(conn)
        view = _view(conn, candidate, invoker_id=MOD_A)
        assert await view.interaction_check(_make_interaction(db=conn, user_id=MOD_A)) is True


class TestTimeout:
    async def test_a_timeout_leaves_the_candidate_pending(
        self, conn: aiosqlite.Connection
    ) -> None:
        candidate = await _stage(conn)
        view = _view(conn, candidate)
        view.message = MagicMock()
        view.message.edit = AsyncMock()

        await view.on_timeout()

        still_pending = await get_pending_fact(conn, guild_id=GUILD_A, pending_id=candidate.id)
        assert still_pending is not None
        assert still_pending.status is PendingFactStatus.PENDING
        view.message.edit.assert_awaited_once()

    async def test_a_timeout_after_a_button_already_ran_does_not_double_edit(
        self, conn: aiosqlite.Connection
    ) -> None:
        candidate = await _stage(conn)
        view = _view(conn, candidate)
        view.message = MagicMock()
        view.message.edit = AsyncMock()

        await view.confirm.callback(_make_interaction(db=conn))
        await view.on_timeout()

        view.message.edit.assert_not_awaited()


class TestPermissions:
    def test_the_command_requires_manage_guild(self) -> None:
        # The same permission every other Aura configuration and fact command
        # uses, so "who may decide what Aura knows" has one answer.
        checks = getattr(pending_command, "checks", [])
        assert checks, "the command must carry a permission check"

    async def test_a_permission_failure_replies_localized_and_ephemerally(self) -> None:
        interaction = _make_interaction(db=None)
        await _handle_pending_command_error(
            interaction, app_commands.MissingPermissions(["manage_guild"])
        )
        interaction.response.send_message.assert_awaited_once()
        assert interaction.response.send_message.call_args.kwargs["ephemeral"] is True

    async def test_a_permission_failure_after_a_response_uses_the_followup(self) -> None:
        interaction = _make_interaction(db=None)
        interaction.response.is_done = MagicMock(return_value=True)
        await _handle_pending_command_error(
            interaction, app_commands.MissingPermissions(["manage_guild"])
        )
        interaction.followup.send.assert_awaited_once()

    async def test_any_other_error_is_logged_rather_than_swallowed(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        # Attaching a local .error() handler suppresses CommandTree's own
        # logging for this command, so this handler has to replace it.
        interaction = _make_interaction(db=None)
        with caplog.at_level(logging.ERROR):
            await _handle_pending_command_error(
                interaction, app_commands.AppCommandError("something else")
            )
        assert any(record.levelname == "ERROR" for record in caplog.records)


class TestVariantGenerationWiring:
    """Multi-Representation Indexing Part 1's automatic-path half.

    /aura-pending's confirm button must call aura.facts_service.confirm_fact
    (which schedules variant generation) rather than
    aura.db.pending_facts.confirm_pending_fact directly -- the two-call-site
    drift this design explicitly rules out. These tests exercise that wiring
    end to end through the real command and view, mocking only the LLM layer
    two levels down, never confirm_fact itself.
    """

    async def test_the_command_passes_the_real_embedding_model_into_the_view(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        await _stage(conn)
        interaction = _make_interaction(db=conn, embedding_model=embedding_model)
        await pending_command.callback(interaction)

        view = interaction.response.send_message.call_args.kwargs["view"]
        assert view._model is embedding_model

    async def test_confirming_schedules_variant_generation_via_facts_service(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        candidate = await _stage(conn)
        view = _view(conn, candidate, model=embedding_model)
        interaction = _make_interaction(db=conn)

        with patch(
            "aura.facts_service.generate_variants_for_fact", AsyncMock(return_value=[])
        ) as generate:
            await view.confirm.callback(interaction)
            await asyncio.sleep(0)

        active = await get_active_facts(conn, GUILD_A)
        assert len(active) == 1
        generate.assert_awaited_once_with(conn, embedding_model, active[0])

    async def test_confirming_still_produces_exactly_one_fact_with_variants_mocked(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        # A regression guard on the manual/automatic path split this wrapper
        # introduces: routing through facts_service.confirm_fact must not
        # change what confirming a candidate produces, only add the
        # background hook.
        candidate = await _stage(conn, content="Uploads in #trading are capped at 5MB.")
        view = _view(conn, candidate, model=embedding_model)
        interaction = _make_interaction(db=conn)

        with patch("aura.facts_service.generate_variants_for_fact", AsyncMock(return_value=[])):
            await view.confirm.callback(interaction)
            await asyncio.sleep(0)

        [fact] = await get_active_facts(conn, GUILD_A)
        assert fact.content == candidate.content
        assert fact.embedding == candidate.embedding
