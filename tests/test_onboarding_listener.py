"""Tests for aura.onboarding.listener: on_member_join, end to end.

End to end over a real in-memory database and a fake gateway, mirroring
tests/test_digest_scheduler.py's approach for the same reason: the interesting
property after a failure is not "a message was sent" but "the claim was or was
not consumed", which only a fake that can be told to fail in each of the ways
a real channel does can show.

This file is also where the brief's "Attack It" section is answered directly:

  * TestBotJoins -- can a bot itself trigger onboarding.
  * TestCrossGuildRefusal -- the same leak aura.digest.scheduler refuses.
  * TestMassJoin -- many joins in quick succession; the daily cap and no
    unnecessary channel flood.
  * TestDuplicateJoin -- a redelivered on_member_join must not post twice.
  * TestMentionSuppression -- @everyone in a fact's text.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import cast
from unittest.mock import MagicMock

import aiosqlite
import discord
import pytest

from aura.config import Settings
from aura.db.connection import utc_iso
from aura.db.onboarding_config import set_onboarding_config
from aura.db.pending_facts import FactCategory
from aura.db.repository import init_schema
from aura.onboarding.listener import handle_member_join

GUILD_A = 100000000000000001
GUILD_B = 200000000000000002
CHANNEL_A = 300000000000000003
CHANNEL_B = 400000000000000004
MODERATOR = 4242

NOW = datetime(2026, 8, 16, 12, 0, 0, tzinfo=timezone.utc)

_next_message_id = iter(range(700000000000000000, 700000000000001000))
_next_user_id = iter(range(1, 100000))


@dataclass
class SentMessage:
    channel_id: int
    embed: discord.Embed
    allowed_mentions: discord.AllowedMentions | None


class FakeChannel:
    def __init__(self, channel_id: int, guild_id: int, sink: list[SentMessage]) -> None:
        self.id = channel_id
        self.guild = FakeGuild(guild_id)
        self._sink = sink
        self.raises: Exception | None = None

    async def send(self, *, embed: discord.Embed, allowed_mentions=None) -> None:
        if self.raises is not None:
            raise self.raises
        self._sink.append(SentMessage(self.id, embed, allowed_mentions))


class FakeGuild:
    def __init__(self, guild_id: int) -> None:
        self.id = guild_id
        self.preferred_locale = "en-US"


@dataclass
class FakeGateway:
    channels: dict[int, FakeChannel] = field(default_factory=dict)
    sent: list[SentMessage] = field(default_factory=list)
    resolve_calls: int = 0

    def add_channel(self, channel_id: int, guild_id: int) -> FakeChannel:
        channel = FakeChannel(channel_id, guild_id, self.sent)
        self.channels[channel_id] = channel
        return channel

    async def resolve_channel(self, channel_id: int) -> discord.TextChannel | None:
        self.resolve_calls += 1
        return cast("discord.TextChannel | None", self.channels.get(channel_id))


def _settings(*, fact_limit: int = 15, daily_cap: int = 20) -> Settings:
    return Settings(_env_file=None, discord_token="fake-token", onboarding_fact_limit=fact_limit, onboarding_daily_cap=daily_cap)  # type: ignore[call-arg]


async def _onboarding_sends_count(conn: aiosqlite.Connection, *, guild_id: int) -> int:
    """Total onboarding_sends rows for a guild, regardless of which UTC day they landed on.

    The listener always claims against utc_now() (the real clock), which the
    fixed `NOW` used to build test facts and members does not control -- so
    tests assert against a day-independent total rather than a specific
    send_day string.
    """
    async with conn.execute(
        "SELECT COUNT(*) FROM onboarding_sends WHERE guild_id = ?", (guild_id,)
    ) as cursor:
        row = await cursor.fetchone()
    return int(row[0]) if row else 0


def _member(
    *, user_id: int | None = None, bot: bool = False, guild_id: int = GUILD_A, joined_at: datetime | None = NOW
) -> MagicMock:
    member = MagicMock(spec=discord.Member)
    member.id = user_id if user_id is not None else next(_next_user_id)
    member.bot = bot
    member.guild = MagicMock()
    member.guild.id = guild_id
    member.joined_at = joined_at
    return member


@pytest.fixture
async def conn():
    connection = await aiosqlite.connect(":memory:")
    await init_schema(connection)
    yield connection
    await connection.close()


async def configure(
    conn: aiosqlite.Connection,
    *,
    guild_id: int = GUILD_A,
    channel_id: int = CHANNEL_A,
    enabled: bool = True,
) -> None:
    await set_onboarding_config(
        conn, guild_id=guild_id, channel_id=channel_id, enabled=enabled, updated_by_id=MODERATOR
    )


async def add_fact(
    conn: aiosqlite.Connection,
    *,
    content: str = "No spoilers outside #spoilers.",
    guild_id: int = GUILD_A,
    category: str | None = FactCategory.RULE,
) -> int:
    message_id = next(_next_message_id)
    cursor = await conn.execute(
        """
        INSERT INTO facts
            (guild_id, channel_id, message_id, content, embedding, status, created_at)
        VALUES (?, ?, ?, ?, ?, 'active', ?)
        """,
        (guild_id, CHANNEL_A, message_id, content, b"\x00\x00\x00\x00", utc_iso(NOW)),
    )
    await conn.commit()
    assert cursor.lastrowid is not None
    fact_id = cursor.lastrowid
    if category is not None:
        await conn.execute(
            """
            INSERT INTO pending_facts
                (guild_id, channel_id, message_id, content, embedding, category, status,
                 confirmed_fact_id, created_at)
            VALUES (?, ?, ?, ?, ?, ?, 'confirmed', ?, ?)
            """,
            (guild_id, CHANNEL_A, next(_next_message_id), f"candidate for {fact_id}",
             b"\x00\x00\x00\x00", category, fact_id, utc_iso(NOW)),
        )
        await conn.commit()
    return fact_id


async def ready_guild(
    conn: aiosqlite.Connection,
    gateway: FakeGateway,
    *,
    guild_id: int = GUILD_A,
    channel_id: int = CHANNEL_A,
    with_content: bool = True,
) -> FakeChannel:
    await configure(conn, guild_id=guild_id, channel_id=channel_id)
    if with_content:
        await add_fact(conn, guild_id=guild_id)
    return gateway.add_channel(channel_id, guild_id)


class TestNotConfigured:
    async def test_an_unconfigured_guild_gets_no_message(self, conn: aiosqlite.Connection) -> None:
        gateway = FakeGateway()

        await handle_member_join(_member(), db=conn, gateway=gateway, settings=_settings())

        assert gateway.sent == []
        assert gateway.resolve_calls == 0

    async def test_a_disabled_guild_gets_no_message(self, conn: aiosqlite.Connection) -> None:
        gateway = FakeGateway()
        await configure(conn, enabled=False)
        gateway.add_channel(CHANNEL_A, GUILD_A)

        await handle_member_join(_member(), db=conn, gateway=gateway, settings=_settings())

        assert gateway.sent == []


class TestEmptyContent:
    async def test_a_guild_with_no_active_facts_gets_no_message(
        self, conn: aiosqlite.Connection
    ) -> None:
        gateway = FakeGateway()
        await ready_guild(conn, gateway, with_content=False)

        await handle_member_join(_member(), db=conn, gateway=gateway, settings=_settings())

        assert gateway.sent == []
        # Read-only up to the point content is known empty -- no claim spent.
        assert await _onboarding_sends_count(conn, guild_id=GUILD_A) == 0

    async def test_a_guild_with_only_milestones_gets_no_message(
        self, conn: aiosqlite.Connection
    ) -> None:
        gateway = FakeGateway()
        await configure(conn)
        gateway.add_channel(CHANNEL_A, GUILD_A)
        await add_fact(conn, content="The server hit 500 members.", category=FactCategory.MILESTONE)

        await handle_member_join(_member(), db=conn, gateway=gateway, settings=_settings())

        assert gateway.sent == []


class TestHappyPath:
    async def test_a_configured_guild_with_content_gets_a_message(
        self, conn: aiosqlite.Connection
    ) -> None:
        gateway = FakeGateway()
        await ready_guild(conn, gateway)

        await handle_member_join(_member(), db=conn, gateway=gateway, settings=_settings())

        assert len(gateway.sent) == 1
        assert gateway.sent[0].channel_id == CHANNEL_A

    async def test_the_claim_is_recorded(self, conn: aiosqlite.Connection) -> None:
        gateway = FakeGateway()
        await ready_guild(conn, gateway)

        await handle_member_join(_member(), db=conn, gateway=gateway, settings=_settings())

        assert await _onboarding_sends_count(conn, guild_id=GUILD_A) == 1

    async def test_a_missing_joined_at_falls_back_to_now(self, conn: aiosqlite.Connection) -> None:
        # discord.py's own type allows joined_at to be None in edge cases
        # (e.g. a partial member object); the claim key must not crash on it.
        gateway = FakeGateway()
        await ready_guild(conn, gateway)

        await handle_member_join(
            _member(joined_at=None), db=conn, gateway=gateway, settings=_settings()
        )

        assert len(gateway.sent) == 1


class TestBotJoins:
    """Can a bot itself trigger onboarding, and should it."""

    async def test_a_bot_joining_gets_no_onboarding_message(
        self, conn: aiosqlite.Connection
    ) -> None:
        gateway = FakeGateway()
        await ready_guild(conn, gateway)

        await handle_member_join(
            _member(bot=True), db=conn, gateway=gateway, settings=_settings()
        )

        assert gateway.sent == []

    async def test_a_bot_join_does_not_even_touch_the_database(
        self, conn: aiosqlite.Connection
    ) -> None:
        gateway = FakeGateway()
        await ready_guild(conn, gateway)

        await handle_member_join(
            _member(bot=True), db=conn, gateway=gateway, settings=_settings()
        )

        assert gateway.resolve_calls == 0
        assert await _onboarding_sends_count(conn, guild_id=GUILD_A) == 0


class TestCrossGuildRefusal:
    """The exact leak aura.digest.scheduler._resolve_target refuses, refused here too."""

    async def test_a_channel_belonging_to_another_guild_is_refused(
        self, conn: aiosqlite.Connection, caplog: pytest.LogCaptureFixture
    ) -> None:
        gateway = FakeGateway()
        await configure(conn, guild_id=GUILD_A, channel_id=CHANNEL_A)
        await add_fact(conn, guild_id=GUILD_A)
        # The channel the row names actually belongs to GUILD_B -- reachable
        # only by a hand-edited config row, not through the slash command.
        gateway.add_channel(CHANNEL_A, GUILD_B)

        with caplog.at_level(logging.ERROR):
            await handle_member_join(
                _member(guild_id=GUILD_A), db=conn, gateway=gateway, settings=_settings()
            )

        assert gateway.sent == []
        assert any("Refusing" in record.getMessage() for record in caplog.records)
        # The claim must not be spent either -- fixing the row should let the
        # next join through.
        assert await _onboarding_sends_count(conn, guild_id=GUILD_A) == 0


class TestUnresolvableChannel:
    async def test_a_deleted_channel_leaves_the_claim_unspent(
        self, conn: aiosqlite.Connection, caplog: pytest.LogCaptureFixture
    ) -> None:
        gateway = FakeGateway()
        await configure(conn)
        await add_fact(conn)
        # Never added to the gateway -- resolve_channel returns None.

        with caplog.at_level(logging.WARNING):
            await handle_member_join(_member(), db=conn, gateway=gateway, settings=_settings())

        assert gateway.sent == []
        assert await _onboarding_sends_count(conn, guild_id=GUILD_A) == 0

    async def test_channel_resolution_happens_before_the_claim(
        self, conn: aiosqlite.Connection
    ) -> None:
        # Mirrors the digest's fix for unbounded bookkeeping on a permanently
        # broken channel: resolving first means a broken channel costs one
        # cache lookup per join and writes no row at all.
        gateway = FakeGateway()
        await configure(conn)
        await add_fact(conn)

        await handle_member_join(_member(), db=conn, gateway=gateway, settings=_settings())
        await handle_member_join(_member(), db=conn, gateway=gateway, settings=_settings())

        async with conn.execute("SELECT COUNT(*) FROM onboarding_sends") as cursor:
            assert await cursor.fetchone() == (0,)


class TestSendFailure:
    async def test_a_send_that_raises_does_not_crash_and_is_logged(
        self, conn: aiosqlite.Connection, caplog: pytest.LogCaptureFixture
    ) -> None:
        gateway = FakeGateway()
        channel = await ready_guild(conn, gateway)
        channel.raises = discord.Forbidden(MagicMock(status=403), "missing permission")

        with caplog.at_level(logging.ERROR):
            await handle_member_join(_member(), db=conn, gateway=gateway, settings=_settings())

        assert gateway.sent == []
        assert any(record.levelno >= logging.ERROR for record in caplog.records)

    async def test_the_claim_is_already_spent_when_the_send_fails(
        self, conn: aiosqlite.Connection
    ) -> None:
        # Unlike the digest, there is no retry-on-failure bookkeeping: a join
        # is one-shot, so a claim consumed before a failed send is not
        # reattempted for that join.
        gateway = FakeGateway()
        channel = await ready_guild(conn, gateway)
        channel.raises = RuntimeError("boom")

        await handle_member_join(_member(), db=conn, gateway=gateway, settings=_settings())

        assert await _onboarding_sends_count(conn, guild_id=GUILD_A) == 1


class TestDuplicateJoin:
    """A redelivered on_member_join for the SAME join must not post twice."""

    async def test_the_same_member_and_joined_at_posts_only_once(
        self, conn: aiosqlite.Connection
    ) -> None:
        gateway = FakeGateway()
        await ready_guild(conn, gateway)
        member = _member(user_id=999, joined_at=NOW)

        await handle_member_join(member, db=conn, gateway=gateway, settings=_settings())
        await handle_member_join(member, db=conn, gateway=gateway, settings=_settings())

        assert len(gateway.sent) == 1

    async def test_a_genuine_rejoin_with_a_new_joined_at_posts_again(
        self, conn: aiosqlite.Connection
    ) -> None:
        # A member who leaves and rejoins is treated as context-free as a new
        # one -- the deliberate product decision in aura.db.onboarding_state.
        gateway = FakeGateway()
        await ready_guild(conn, gateway)

        await handle_member_join(
            _member(user_id=999, joined_at=NOW), db=conn, gateway=gateway, settings=_settings()
        )
        await handle_member_join(
            _member(user_id=999, joined_at=NOW + timedelta(days=30)),
            db=conn,
            gateway=gateway,
            settings=_settings(),
        )

        assert len(gateway.sent) == 2


class TestMassJoin:
    """Many joins in quick succession: the daily cap, and no unbounded flood."""

    async def test_a_burst_of_joins_is_bounded_by_the_daily_cap(
        self, conn: aiosqlite.Connection
    ) -> None:
        gateway = FakeGateway()
        await ready_guild(conn, gateway)
        settings = _settings(daily_cap=5)
        members = [_member(joined_at=NOW + timedelta(seconds=i)) for i in range(30)]

        for member in members:
            await handle_member_join(member, db=conn, gateway=gateway, settings=settings)

        assert len(gateway.sent) == 5

    async def test_concurrent_joins_never_overshoot_the_cap(
        self, conn: aiosqlite.Connection
    ) -> None:
        gateway = FakeGateway()
        await ready_guild(conn, gateway)
        settings = _settings(daily_cap=5)
        members = [_member(joined_at=NOW + timedelta(seconds=i)) for i in range(30)]

        await asyncio.gather(
            *(
                handle_member_join(member, db=conn, gateway=gateway, settings=settings)
                for member in members
            )
        )

        assert len(gateway.sent) == 5

    async def test_joins_past_the_cap_get_no_message_but_no_error_either(
        self, conn: aiosqlite.Connection, caplog: pytest.LogCaptureFixture
    ) -> None:
        gateway = FakeGateway()
        await ready_guild(conn, gateway)
        settings = _settings(daily_cap=1)

        with caplog.at_level(logging.WARNING):
            await handle_member_join(
                _member(joined_at=NOW), db=conn, gateway=gateway, settings=settings
            )
            await handle_member_join(
                _member(joined_at=NOW + timedelta(seconds=1)), db=conn, gateway=gateway, settings=settings
            )

        assert len(gateway.sent) == 1
        assert any("cap" in record.getMessage() for record in caplog.records)


class TestMentionSuppression:
    async def test_at_everyone_in_a_facts_text_cannot_ping(
        self, conn: aiosqlite.Connection
    ) -> None:
        gateway = FakeGateway()
        await configure(conn)
        gateway.add_channel(CHANNEL_A, GUILD_A)
        await add_fact(conn, content="@everyone read the rules before posting.")

        await handle_member_join(_member(), db=conn, gateway=gateway, settings=_settings())

        assert len(gateway.sent) == 1
        mentions = gateway.sent[0].allowed_mentions
        assert mentions is not None
        assert mentions.everyone is False


class TestGuildIsolation:
    async def test_one_guilds_facts_never_appear_in_anothers_message(
        self, conn: aiosqlite.Connection
    ) -> None:
        gateway = FakeGateway()
        await ready_guild(conn, gateway, guild_id=GUILD_A, channel_id=CHANNEL_A)
        await ready_guild(conn, gateway, guild_id=GUILD_B, channel_id=CHANNEL_B)

        await handle_member_join(
            _member(guild_id=GUILD_A), db=conn, gateway=gateway, settings=_settings()
        )
        await handle_member_join(
            _member(guild_id=GUILD_B), db=conn, gateway=gateway, settings=_settings()
        )

        assert len(gateway.sent) == 2
        by_channel = {message.channel_id for message in gateway.sent}
        assert by_channel == {CHANNEL_A, CHANNEL_B}

    async def test_a_daily_cap_hit_in_one_guild_does_not_affect_another(
        self, conn: aiosqlite.Connection
    ) -> None:
        gateway = FakeGateway()
        await ready_guild(conn, gateway, guild_id=GUILD_A, channel_id=CHANNEL_A)
        await ready_guild(conn, gateway, guild_id=GUILD_B, channel_id=CHANNEL_B)
        settings = _settings(daily_cap=1)

        await handle_member_join(
            _member(guild_id=GUILD_A, joined_at=NOW), db=conn, gateway=gateway, settings=settings
        )
        await handle_member_join(
            _member(guild_id=GUILD_A, joined_at=NOW + timedelta(seconds=1)),
            db=conn,
            gateway=gateway,
            settings=settings,
        )
        await handle_member_join(
            _member(guild_id=GUILD_B, joined_at=NOW), db=conn, gateway=gateway, settings=settings
        )

        assert len(gateway.sent) == 2
        by_channel = [message.channel_id for message in gateway.sent]
        assert by_channel.count(CHANNEL_A) == 1
        assert by_channel.count(CHANNEL_B) == 1


class TestFactLimitFlowsThrough:
    async def test_the_configured_fact_limit_bounds_the_message(
        self, conn: aiosqlite.Connection
    ) -> None:
        gateway = FakeGateway()
        await configure(conn)
        gateway.add_channel(CHANNEL_A, GUILD_A)
        for index in range(10):
            await add_fact(conn, content=f"rule {index}")

        await handle_member_join(
            _member(), db=conn, gateway=gateway, settings=_settings(fact_limit=3)
        )

        assert len(gateway.sent) == 1
        embed = gateway.sent[0].embed
        rules_field = embed.fields[0]
        # All 3 shown fit within one field with no per-field truncation note
        # (the global cap already cut the other 7 before rendering); the
        # capped note in the footer is what names the 7 that never made it in.
        assert rules_field.name is not None and "(3)" in rules_field.name
        assert embed.footer.text is not None and "7" in embed.footer.text
