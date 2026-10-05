"""The four look switches (P5): classic is exactly the old message, card is the new one.

DIGEST_LOOK, ONBOARDING_LOOK, PLAN_LOOK and NOTICE_LOOK default to `classic`,
and then every send is the call it was before P5 (the classic-path suites of
the digest, onboarding, plan and command modules run unchanged). Here: the
defaults, that `card` sends the family's card in ANSWER_CARD_STYLE with mentions
disabled, that a refused container falls back to an embed once, and that the
classic branch still sends the classic message when the switch is off.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import aiosqlite
import discord
import pytest

from aura.billing import PlanGate
from aura.card_delivery import reply_with_card, send_card_to_channel
from aura.cards import build_notice
from aura.commands.notices import edit_into_confirmation, send_confirmation
from aura.commands.plan import plan_command, send_pro_refusal
from aura.config import CardStyle, MessageLook, Settings
from aura.db.repository import init_schema
from aura.digest.scheduler import send_due_digests
from aura.onboarding.listener import handle_member_join
from aura.theme import ACCENT_COLORS, MessageKind


def _settings(**overrides: object) -> Settings:
    values: dict[str, object] = {"discord_token": "fake-token"}
    values.update(overrides)
    return Settings(_env_file=None, **values)  # type: ignore[arg-type]


@pytest.fixture
async def conn():
    """A fresh in-memory database with Aura's schema."""
    connection = await aiosqlite.connect(":memory:")
    await init_schema(connection)
    yield connection
    await connection.close()


class RecordingChannel:
    """A channel that records every send's keyword arguments; optionally refuses containers."""

    def __init__(self, channel_id: int, guild_id: int, *, refuse_views: bool = False) -> None:
        self.id = channel_id
        self.guild = MagicMock()
        self.guild.id = guild_id
        self.guild.preferred_locale = "de"
        self.guild.name = "Bastelstube"
        self.guild.get_channel = MagicMock(return_value=None)
        self.sent: list[dict[str, Any]] = []
        self._refuse_views = refuse_views

    async def send(self, **kwargs: Any) -> None:
        if self._refuse_views and "view" in kwargs:
            raise discord.HTTPException(MagicMock(status=400, reason="Bad Request"), "no")
        self.sent.append(kwargs)


class TestTheDefaults:
    def test_every_family_ships_classic(self) -> None:
        settings = _settings()

        assert settings.digest_look is MessageLook.CLASSIC
        assert settings.onboarding_look is MessageLook.CLASSIC
        assert settings.plan_look is MessageLook.CLASSIC
        assert settings.notice_look is MessageLook.CLASSIC

    def test_unknown_values_are_refused(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            _settings(digest_look="fancy")


class TestDelivery:
    @pytest.mark.parametrize(
        ("style", "key"), [(CardStyle.EMBED, "embed"), (CardStyle.CONTAINER, "view")]
    )
    async def test_a_channel_card_is_sent_in_the_style_with_mentions_disabled(
        self, style: CardStyle, key: str
    ) -> None:
        channel = RecordingChannel(1, 2)
        await send_card_to_channel(channel, build_notice(MessageKind.CONFIRM, "Ok."), style=style)  # type: ignore[arg-type]

        assert len(channel.sent) == 1
        assert key in channel.sent[0]
        assert channel.sent[0]["allowed_mentions"].everyone is False
        assert channel.sent[0]["allowed_mentions"].roles is False

    async def test_a_refused_container_goes_out_once_as_an_embed(self) -> None:
        channel = RecordingChannel(1, 2, refuse_views=True)
        await send_card_to_channel(
            channel,
            build_notice(MessageKind.CONFIRM, "Ok."),
            style=CardStyle.CONTAINER,  # type: ignore[arg-type]
        )

        assert [list(kwargs) for kwargs in channel.sent] == [["embed", "allowed_mentions"]]

    async def test_another_send_error_propagates(self) -> None:
        channel = MagicMock()
        channel.send = AsyncMock(
            side_effect=discord.HTTPException(MagicMock(status=403, reason="Forbidden"), "no")
        )
        with pytest.raises(discord.HTTPException):
            await send_card_to_channel(
                channel, build_notice(MessageKind.CONFIRM, "x"), style=CardStyle.CONTAINER
            )

    async def test_a_reply_uses_the_followup_once_the_response_is_done(self) -> None:
        interaction = MagicMock()
        interaction.response.is_done = MagicMock(return_value=True)
        interaction.followup.send = AsyncMock()
        await reply_with_card(
            interaction, build_notice(MessageKind.CONFIRM, "x"), style=CardStyle.EMBED
        )

        kwargs = interaction.followup.send.await_args.kwargs
        assert kwargs["ephemeral"] is True
        assert kwargs["embed"].colour.value == ACCENT_COLORS[MessageKind.CONFIRM]


class TestTheDigest:
    async def _post(
        self, conn: aiosqlite.Connection, look: MessageLook, style: CardStyle
    ) -> RecordingChannel:
        from tests.test_digest_scheduler import NOW, add_fact, backdate_enabled_at, configure

        await configure(conn)
        await backdate_enabled_at(conn, moment=NOW - timedelta(days=30))
        await add_fact(
            conn, content="@everyone Filmabend ist freitags.", created_at=NOW - timedelta(days=1)
        )
        channel = RecordingChannel(300000000000000003, 100000000000000001)

        class Gateway:
            async def resolve_channel(self, channel_id: int) -> Any:
                return channel

        posted = await send_due_digests(
            conn,
            Gateway(),
            now=NOW,
            plan_gate=PlanGate.unenforced(),
            look=look,
            card_style=style,  # type: ignore[arg-type]
        )
        assert posted == 1
        return channel

    async def test_classic_sends_the_classic_embed(self, conn: aiosqlite.Connection) -> None:
        channel = await self._post(conn, MessageLook.CLASSIC, CardStyle.CONTAINER)

        embed = channel.sent[0]["embed"]
        assert embed.title == "Was sich auf diesem Server geändert hat"
        assert "view" not in channel.sent[0]

    @pytest.mark.parametrize(
        ("style", "key"), [(CardStyle.EMBED, "embed"), (CardStyle.CONTAINER, "view")]
    )
    async def test_card_sends_the_digest_card(
        self, conn: aiosqlite.Connection, style: CardStyle, key: str
    ) -> None:
        channel = await self._post(conn, MessageLook.CARD, style)

        sent = channel.sent[0]
        assert key in sent
        assert sent["allowed_mentions"].everyone is False
        if key == "embed":
            assert sent["embed"].colour.value == ACCENT_COLORS[MessageKind.DIGEST]
            assert sent["embed"].fields[0].name == "Neu"


class TestOnboarding:
    async def _join(self, conn: aiosqlite.Connection, look: MessageLook) -> RecordingChannel:
        from tests.test_onboarding_listener import CHANNEL_A, GUILD_A, _member, add_fact, configure

        await configure(conn)
        await add_fact(conn, content="Keine Werbung.")
        channel = RecordingChannel(CHANNEL_A, GUILD_A)
        member = _member()
        member.guild = channel.guild

        class Gateway:
            async def resolve_channel(self, channel_id: int) -> Any:
                return channel

        await handle_member_join(
            member,
            db=conn,
            gateway=Gateway(),  # type: ignore[arg-type]
            settings=_settings(onboarding_look=look),
            plan_gate=PlanGate.unenforced(),
        )
        return channel

    async def test_classic_sends_the_classic_embed(self, conn: aiosqlite.Connection) -> None:
        channel = await self._join(conn, MessageLook.CLASSIC)

        assert channel.sent[0]["embed"].title == "Willkommen! Das gilt gerade auf diesem Server"

    async def test_card_greets_with_the_server_name(self, conn: aiosqlite.Connection) -> None:
        channel = await self._join(conn, MessageLook.CARD)

        embed = channel.sent[0]["embed"]
        assert embed.author.name == "👋 Willkommen auf Bastelstube!"
        assert embed.colour.value == ACCENT_COLORS[MessageKind.ONBOARDING]
        assert channel.sent[0]["allowed_mentions"].everyone is False


def _interaction(settings: Settings) -> MagicMock:
    interaction = MagicMock()
    interaction.locale = "de"
    interaction.guild_id = 1
    interaction.client.settings = settings
    interaction.response.is_done = MagicMock(return_value=False)
    interaction.response.send_message = AsyncMock()
    interaction.response.edit_message = AsyncMock()
    interaction.followup.send = AsyncMock()
    return interaction


class TestThePlanCommand:
    async def test_classic_sends_the_classic_text(self) -> None:
        interaction = _interaction(_settings())
        interaction.client.plan_gate = PlanGate.unenforced()
        await plan_command.callback(interaction)  # type: ignore[call-arg, arg-type]

        args = interaction.response.send_message.await_args
        assert args.args[0].startswith("Auf diesem Server sind alle Funktionen")
        assert args.kwargs == {"ephemeral": True}

    async def test_card_sends_the_plan_card_ephemerally(self) -> None:
        interaction = _interaction(_settings(plan_look="card"))
        interaction.client.plan_gate = PlanGate.unenforced()
        await plan_command.callback(interaction)  # type: ignore[call-arg, arg-type]

        kwargs = interaction.response.send_message.await_args.kwargs
        assert kwargs["ephemeral"] is True
        assert kwargs["embed"].colour.value == ACCENT_COLORS[MessageKind.PLAN]
        assert kwargs["embed"].fields[0].name == "Enthalten"


class TestNotices:
    async def test_classic_confirmation_is_the_old_call(self) -> None:
        interaction = _interaction(_settings())
        await send_confirmation(interaction, "Fakt #12 erstellt.")

        interaction.response.send_message.assert_awaited_once_with(
            "Fakt #12 erstellt.", ephemeral=True
        )

    async def test_card_confirmation_is_a_confirm_card(self) -> None:
        interaction = _interaction(_settings(notice_look="card", answer_card_style="container"))
        await send_confirmation(interaction, "Fakt #12 erstellt.")

        kwargs = interaction.response.send_message.await_args.kwargs
        assert "view" in kwargs
        assert kwargs["allowed_mentions"].everyone is False

    async def test_classic_edit_is_the_old_edit(self) -> None:
        interaction = _interaction(_settings())
        await edit_into_confirmation(interaction, "Vorschlag #7 verworfen.")

        interaction.response.edit_message.assert_awaited_once_with(
            content="Vorschlag #7 verworfen.", embed=None, view=None
        )

    async def test_card_edit_always_uses_an_embed(self) -> None:
        interaction = _interaction(_settings(notice_look="card", answer_card_style="container"))
        await edit_into_confirmation(interaction, "Vorschlag #7 verworfen.")

        kwargs = interaction.response.edit_message.await_args.kwargs
        assert kwargs["content"] is None and kwargs["view"] is None
        assert kwargs["embed"].colour.value == ACCENT_COLORS[MessageKind.CONFIRM]

    async def test_a_client_without_settings_falls_back_to_classic(self) -> None:
        interaction = _interaction(_settings())
        interaction.client = object()
        await send_confirmation(interaction, "x")

        interaction.response.send_message.assert_awaited_once_with("x", ephemeral=True)

    async def test_the_pro_refusal_classic_and_card(self) -> None:
        classic = _interaction(_settings())
        await send_pro_refusal(classic, "Refused.")
        card = _interaction(
            _settings(notice_look="card", billing_dashboard_url="https://example.com/d")
        )
        await send_pro_refusal(card, "Refused.")

        classic.response.send_message.assert_awaited_once_with("Refused.", ephemeral=True)
        embed = card.response.send_message.await_args.kwargs["embed"]
        assert embed.colour.value == ACCENT_COLORS[MessageKind.PLAN]
        assert "](<https://example.com/d>)" in (embed.description or "")
