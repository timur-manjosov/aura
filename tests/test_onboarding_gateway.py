"""Tests for aura.onboarding.gateway: turning a stored channel ID into
somewhere to post a new member's summary.

Structurally identical to tests/test_digest_gateway.py, for the module that
mirrors it exactly: the same three failure modes have to come back as None
with a logged reason rather than escaping into discord.py's own event
dispatch, which for on_member_join means a stack trace per join instead of a
skipped welcome.
"""
from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock

import discord
import pytest

from aura.onboarding.gateway import ClientOnboardingGateway

CHANNEL_A = 300000000000000003


def _client(*, cached=None, fetched=None, fetch_error: Exception | None = None) -> MagicMock:
    client = MagicMock(spec=discord.Client)
    client.get_channel = MagicMock(return_value=cached)
    client.fetch_channel = AsyncMock(return_value=fetched, side_effect=fetch_error)
    return client


def _text_channel() -> MagicMock:
    return MagicMock(spec=discord.TextChannel)


class TestResolveChannel:
    async def test_a_cached_channel_is_returned_without_any_http_call(self) -> None:
        channel = _text_channel()
        client = _client(cached=channel)

        resolved = await ClientOnboardingGateway(client).resolve_channel(CHANNEL_A)

        assert resolved is channel
        client.fetch_channel.assert_not_awaited()

    async def test_an_uncached_channel_is_fetched(self) -> None:
        channel = _text_channel()
        client = _client(cached=None, fetched=channel)

        resolved = await ClientOnboardingGateway(client).resolve_channel(CHANNEL_A)

        assert resolved is channel
        client.fetch_channel.assert_awaited_once_with(CHANNEL_A)

    async def test_a_deleted_or_forbidden_channel_resolves_to_nothing(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        response = MagicMock(status=404, reason="Not Found")
        client = _client(fetch_error=discord.NotFound(response, "Unknown Channel"))

        with caplog.at_level(logging.WARNING):
            resolved = await ClientOnboardingGateway(client).resolve_channel(CHANNEL_A)

        assert resolved is None
        assert any(str(CHANNEL_A) in record.getMessage() for record in caplog.records)

    async def test_a_transient_api_error_resolves_to_nothing_rather_than_raising(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        response = MagicMock(status=503, reason="Service Unavailable")
        client = _client(fetch_error=discord.DiscordServerError(response, "try later"))

        with caplog.at_level(logging.WARNING):
            resolved = await ClientOnboardingGateway(client).resolve_channel(CHANNEL_A)

        assert resolved is None

    @pytest.mark.parametrize(
        "spec", [discord.VoiceChannel, discord.CategoryChannel, discord.ForumChannel]
    )
    async def test_a_channel_that_is_not_a_text_channel_is_refused(
        self, spec: type, caplog: pytest.LogCaptureFixture
    ) -> None:
        client = _client(cached=MagicMock(spec=spec))

        with caplog.at_level(logging.WARNING):
            resolved = await ClientOnboardingGateway(client).resolve_channel(CHANNEL_A)

        assert resolved is None
        assert any("not a text channel" in record.getMessage() for record in caplog.records)

    async def test_a_fetched_non_text_channel_is_also_refused(self) -> None:
        client = _client(cached=None, fetched=MagicMock(spec=discord.VoiceChannel))

        assert await ClientOnboardingGateway(client).resolve_channel(CHANNEL_A) is None
