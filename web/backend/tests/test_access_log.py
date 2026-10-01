"""uvicorn's access log keeps every request's path and never its query string.

Driven through the real server, configured exactly as aura_web.__main__ does,
because the line in question is written by uvicorn's protocol code, not by the
application -- the ASGI transport the other tests use never produces it.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import socket
from collections.abc import AsyncIterator

import httpx
import pytest
import uvicorn

from aura_web.__main__ import uvicorn_config
from aura_web.access_log import (
    UVICORN_ACCESS_LOGGER,
    QueryStringRedactingFilter,
    install_access_log_redaction,
)
from aura_web.config import WebSettings
from fake_discord import FakeDiscordState, create_fake_discord
from helpers import build_app

CODE = "SeCrEtOaUtHcOdE0123456789"
STATE = "StAtEvAlUe-abcdefghijklmnopqrstuvwxyz"


def _free_port() -> int:
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as probe:
        probe.bind(("127.0.0.1", 0))
        port: int = probe.getsockname()[1]
        return port


@contextlib.asynccontextmanager
async def _serve(settings: WebSettings, discord_state: FakeDiscordState) -> AsyncIterator[str]:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=create_fake_discord(discord_state)),
        base_url="https://discord.test",
    ) as discord_http:
        app = build_app(settings, discord_http)
        port = _free_port()
        server = uvicorn.Server(uvicorn_config(app, host="127.0.0.1", port=port))
        task = asyncio.create_task(server.serve())
        try:
            for _ in range(500):
                if server.started:
                    break
                await asyncio.sleep(0.01)
            assert server.started
            yield f"http://127.0.0.1:{port}"
        finally:
            server.should_exit = True
            await task


def access_lines(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        record.getMessage() for record in caplog.records if record.name == UVICORN_ACCESS_LOGGER
    ]


class TestTheRealServer:
    async def test_the_oauth_callback_is_logged_without_its_code_or_state(
        self,
        web_settings: WebSettings,
        discord_state: FakeDiscordState,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        caplog.set_level(logging.INFO)
        async with (
            _serve(web_settings, discord_state) as base_url,
            httpx.AsyncClient(base_url=base_url) as client,
        ):
            await client.get(f"/api/auth/callback?code={CODE}&state={STATE}")
            await client.get("/api/health?x=1")
        lines = access_lines(caplog)
        assert len(lines) == 2
        assert '"GET /api/auth/callback HTTP/1.1"' in lines[0]
        assert '"GET /api/health HTTP/1.1" 200' in lines[1]
        assert not any(CODE in line or STATE in line or "?" in line for line in lines)

    async def test_a_newline_hidden_in_the_query_cannot_forge_a_line(
        self,
        web_settings: WebSettings,
        discord_state: FakeDiscordState,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        caplog.set_level(logging.INFO)
        async with (
            _serve(web_settings, discord_state) as base_url,
            httpx.AsyncClient(base_url=base_url) as client,
        ):
            await client.get("/api/health?a=%0a127.0.0.1:0%20-%20%22GET%20/forged%22%20200")
        (line,) = access_lines(caplog)
        assert "\n" not in line and "forged" not in line


class TestTheFilter:
    def record(self, *args: object) -> logging.LogRecord:
        return logging.LogRecord(
            UVICORN_ACCESS_LOGGER, logging.INFO, __file__, 1, '%s - "%s %s HTTP/%s" %d', args, None
        )

    def test_only_path_arguments_lose_their_query(self) -> None:
        record = self.record("203.0.113.7:0", "GET", "/a/b?code=x&state=y", "1.1", 303)
        assert QueryStringRedactingFilter().filter(record) is True
        assert record.getMessage() == '203.0.113.7:0 - "GET /a/b HTTP/1.1" 303'

    def test_a_record_without_a_query_or_without_arguments_is_untouched(self) -> None:
        plain = self.record("203.0.113.7:0", "GET", "/a", "1.1", 200)
        QueryStringRedactingFilter().filter(plain)
        assert plain.getMessage() == '203.0.113.7:0 - "GET /a HTTP/1.1" 200'
        bare = logging.LogRecord(
            UVICORN_ACCESS_LOGGER, logging.INFO, __file__, 1, "x?y", None, None
        )
        assert QueryStringRedactingFilter().filter(bare) is True
        assert bare.getMessage() == "x?y"

    def test_installing_twice_attaches_one_filter(self) -> None:
        install_access_log_redaction()
        install_access_log_redaction()
        filters = logging.getLogger(UVICORN_ACCESS_LOGGER).filters
        assert sum(isinstance(item, QueryStringRedactingFilter) for item in filters) == 1
