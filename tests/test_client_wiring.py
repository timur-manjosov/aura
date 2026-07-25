"""Tests for AuraClient's own event wiring, without a gateway connection.

The listener's decisions are covered in test_proactive_listener.py; what is
covered here is the part that only exists in main.py -- that on_message is
actually reachable, actually delegates, and cannot blow up on a message that
arrives before startup has finished.
"""
from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import discord
import pytest

from aura.config import Settings
from aura.main import AuraClient, build_intents
from aura.proactive.gate import ProactiveGateConfig
from aura.proactive.grace import GraceRegistry

GUILD_A = 100000000000000001


def _settings() -> Settings:
    return Settings(_env_file=None, discord_token="fake-token")  # type: ignore[call-arg]


def _client() -> AuraClient:
    return AuraClient(intents=build_intents(), settings=_settings())


def _make_message() -> MagicMock:
    message = MagicMock(spec=discord.Message)
    message.content = "where are the rules?"
    message.guild = MagicMock()
    message.guild.id = GUILD_A
    message.channel = MagicMock()
    message.channel.id = 5
    message.id = 9
    message.author = MagicMock()
    message.author.bot = False
    message.webhook_id = None
    message.interaction_metadata = None
    message.type = discord.MessageType.default
    return message


class TestIntents:
    def test_message_content_intent_is_requested(self) -> None:
        # Without it every message arrives with empty content and the whole
        # detector silently scores nothing, forever.
        assert build_intents().message_content is True


_STARTUP_ATTRIBUTES = ("db", "question_detector", "embedding_model", "gate_config")


def _started_client() -> AuraClient:
    """A client with every dependency setup_hook would have installed."""
    client = _client()
    client.db = MagicMock()
    client.question_detector = MagicMock()
    client.embedding_model = MagicMock()
    client.gate_config = MagicMock()
    return client


class TestOnMessage:
    async def test_a_message_is_handed_to_the_listener_with_the_clients_own_dependencies(
        self,
    ) -> None:
        client = _started_client()
        message = _make_message()

        with patch("aura.main.handle_message", AsyncMock()) as handler:
            await client.on_message(message)

        handler.assert_awaited_once()
        args, kwargs = handler.call_args
        assert args[0] is message
        assert kwargs["db"] is client.db
        assert kwargs["detector"] is client.question_detector
        assert kwargs["model"] is client.embedding_model
        assert kwargs["config"] is client.gate_config
        # Phase 2a-3: the responder needs settings to resolve the proactive
        # model and check whether an LLM is configured at all.
        assert kwargs["settings"] is client.settings
        # Phase 2b-1: the listener needs the client's own long-lived grace
        # registry, not a fresh one per message, so cancellation state
        # actually carries across messages.
        assert kwargs["grace_registry"] is client.grace_registry

    @pytest.mark.parametrize("missing", _STARTUP_ATTRIBUTES)
    async def test_a_message_arriving_before_startup_finishes_is_skipped_not_crashed(
        self, missing: str, caplog: pytest.LogCaptureFixture
    ) -> None:
        # Unreachable in production (setup_hook completes before the gateway
        # delivers anything), so the cost of being wrong about that would be
        # an AttributeError on every single message. Parametrized over every
        # dependency, because a guard that checks three of four is a guard
        # that fails on the fourth.
        client = _started_client()
        setattr(client, missing, None)

        with patch("aura.main.handle_message", AsyncMock()) as handler:
            with caplog.at_level(logging.WARNING):
                await client.on_message(_make_message())

        handler.assert_not_awaited()
        assert any(record.levelno >= logging.WARNING for record in caplog.records)

    @pytest.mark.parametrize("attribute", _STARTUP_ATTRIBUTES)
    async def test_a_fresh_client_starts_with_nothing_installed(self, attribute: str) -> None:
        assert getattr(_client(), attribute) is None

    def test_a_fresh_client_already_has_a_grace_registry(self) -> None:
        # Unlike the setup_hook-installed dependencies above, the grace
        # registry needs no async setup and no settings -- it is built
        # eagerly in __init__, the same way a restart is meant to leave it:
        # empty, ready, and requiring no recovery step (see aura.proactive.grace).
        assert isinstance(_client().grace_registry, GraceRegistry)


class TestMessageDeleteAndEdit:
    """Phase 2b-1: a deleted or edited message must stand down its own grace period."""

    async def test_a_deleted_message_notifies_the_grace_registry(self) -> None:
        client = _started_client()
        message = _make_message()

        with patch.object(client.grace_registry, "notice_message_gone") as notice:
            await client.on_message_delete(message)

        notice.assert_called_once_with(channel_id=message.channel.id, message_id=message.id)

    async def test_an_uncached_raw_deletion_also_notifies_the_grace_registry(self) -> None:
        client = _started_client()
        payload = MagicMock(spec=discord.RawMessageDeleteEvent)
        payload.channel_id = 5
        payload.message_id = 9

        with patch.object(client.grace_registry, "notice_message_gone") as notice:
            await client.on_raw_message_delete(payload)

        notice.assert_called_once_with(channel_id=5, message_id=9)

    async def test_an_edited_message_notifies_the_grace_registry_using_the_after_state(
        self,
    ) -> None:
        client = _started_client()
        before = _make_message()
        after = _make_message()
        after.content = "a different question now"

        with patch.object(client.grace_registry, "notice_message_gone") as notice:
            await client.on_message_edit(before, after)

        notice.assert_called_once_with(channel_id=after.channel.id, message_id=after.id)


class TestGateConfiguration:
    def test_the_gate_config_is_built_from_the_clients_settings(self) -> None:
        # Built once in setup_hook, not per message. Asserted through the same
        # mapping production uses, so a setting renamed on one side and not the
        # other cannot pass.
        settings = _settings()

        config = ProactiveGateConfig.from_settings(settings)

        assert config.daily_cap == settings.proactive_daily_cap
        assert config.cooldown_seconds == settings.proactive_cooldown_seconds

    def test_the_shipped_defaults_produce_a_valid_gate_configuration(self) -> None:
        # The defaults in config.py are placeholders that will be retuned. A
        # retuning that lands outside the ranges the gate accepts must fail
        # here, at build time, rather than at a deployment's startup.
        assert ProactiveGateConfig.from_settings(_settings())
