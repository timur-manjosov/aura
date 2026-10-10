"""P7a's visible surface: export safety, labels, notices, settings, logs and start-up wiring."""

from __future__ import annotations

import asyncio
import csv
import io
import logging
import re
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import aiosqlite
import discord
import pytest
from pydantic import BaseModel, ValidationError

from aura.answer_card import (
    AnswerCard,
    card_to_embed,
    label_legacy_embed,
    with_answer_labels,
)
from aura.config import Settings
from aura.db.models import Fact, FactStatus
from aura.log_safety import content_free_reason
from aura.logging_config import CONTENT_BEARING_LOGGERS, configure_logging
from aura.privacy.author_lookup import AuthorUnknown, LookupFailed, lookup_missing_authors
from aura.privacy.export import (
    MAX_EXPORT_FILE_BYTES,
    ExportTooLargeError,
    build_export,
    csv_safe_cell,
    render_csv,
    render_markdown,
)
from aura.privacy.notices import post_capture_notice_once
from aura.theme import MessageKind
from tests.privacy_data import BEFORE, CHANNEL_A, GUILD_A, MEMBER, OTHER, add_fact, open_database

SRC = Path(__file__).resolve().parent.parent / "src" / "aura"
NOW = datetime(2026, 10, 8, 12, 0, tzinfo=UTC)


def _fact(fact_id: int, content: str, **overrides: object) -> Fact:
    values: dict[str, object] = {
        "id": fact_id,
        "guild_id": GUILD_A,
        "channel_id": 5,
        "message_id": 1_000 + fact_id,
        "content": content,
        "embedding": b"",
        "status": FactStatus.ACTIVE,
        "created_at": datetime(2026, 9, 1, tzinfo=UTC),
    }
    values.update(overrides)
    return Fact(**values)  # type: ignore[arg-type]


class TestCsvExport:
    @pytest.mark.parametrize(
        "hostile",
        [
            '=HYPERLINK("http://evil","x")',
            "+1+1",
            "-2+3",
            "@SUM(A1)",
            "\t=1",
            "\r=1",
            "＝1+1",
            "＋1",
            "－1",
            "＠x",
        ],
    )
    def test_a_formula_never_starts_a_cell(self, hostile: str) -> None:
        assert csv_safe_cell(hostile).startswith("'")

    @pytest.mark.parametrize(
        "plain", ["Raid on Friday", "1+1 is two", " =not first", "", "émoji 🎉"]
    )
    def test_ordinary_text_is_untouched(self, plain: str) -> None:
        assert csv_safe_cell(plain) == plain

    def test_quotes_commas_newlines_and_nul_cannot_break_a_row(self) -> None:
        hostile = 'a,"b"\nc\r\nd\x00e'
        data = render_csv([_fact(1, hostile), _fact(2, "second")], [(1, 2)])
        assert data.startswith("﻿".encode())
        rows = list(csv.reader(io.StringIO(data.decode("utf-8-sig"))))
        assert len(rows) == 3
        assert rows[1][2] == 'a,"b"\nc\r\nd�e'
        assert rows[1][7] == "2" and rows[2][7] == "1"

    def test_a_removed_source_links_the_server(self) -> None:
        data = render_csv([_fact(1, "x", channel_id=0, message_id=0)], [])
        assert f'https://discord.com/channels/{GUILD_A}"' in data.decode("utf-8-sig")


class TestMarkdownExport:
    @pytest.mark.parametrize(
        "hostile",
        [
            "[click](http://evil.example)",
            "![img](http://tracker.example/p.png)",
            "<img src=x onerror=alert(1)>",
            "# heading",
            "- list",
            "**bold** `code` ~~strike~~ ||spoiler||",
            "@everyone look",
            "line one\nline two\n# injected heading",
            "\u202eevil reversed",
        ],
    )
    def test_no_markup_can_come_out_of_a_fact(self, hostile: str) -> None:
        text = render_markdown([_fact(1, hostile)], [], locale="en-US", exported_at=NOW).decode(
            "utf-8"
        )
        body = text.split("## Current facts", 1)[1].split("## Replaced facts", 1)[0]
        # The only link is the source link built from IDs.
        links = re.findall(r"(?<!\\)\]\((?P<url>[^)]*)\)", body)
        assert links == [f"https://discord.com/channels/{GUILD_A}/5/1001"]
        assert re.search(r"(?<!\\)<", body) is None and "\u202e" not in body
        assert not any(line.startswith(("# ", "- l", "line two")) for line in body.splitlines()[1:])
        assert "@everyone" not in body

    def test_history_shows_the_chain_and_a_deleted_replacement(self) -> None:
        old = _fact(
            1,
            "old rule",
            status=FactStatus.SUPERSEDED,
            superseded_by_id=2,
            superseded_at=datetime(2026, 9, 5, tzinfo=UTC),
        )
        gone = _fact(
            3,
            "older rule",
            status=FactStatus.SUPERSEDED,
            superseded_by_id=None,
            superseded_at=datetime(2026, 9, 6, tzinfo=UTC),
        )
        text = render_markdown(
            [old, _fact(2, "new rule"), gone], [(1, 2)], locale="de", exported_at=NOW
        ).decode("utf-8")
        assert "durch #2" in text and "der Nachfolger wurde gelöscht" in text
        assert "## Aktuelle Fakten" in text and "## Ersetzte Fakten" in text

    def test_files_over_the_attachment_limit_are_refused(self) -> None:
        huge = [_fact(index, "x" * 5000) for index in range(1, MAX_EXPORT_FILE_BYTES // 5000 + 10)]
        with pytest.raises(ExportTooLargeError):
            build_export(huge, [], locale="en-US", exported_at=NOW)

    def test_file_names_carry_only_the_date(self) -> None:
        csv_file, md_file = build_export([_fact(1, "x")], [], locale="en-US", exported_at=NOW)
        assert (csv_file.filename, md_file.filename) == (
            "aura-facts-2026-10-08.csv",
            "aura-facts-2026-10-08.md",
        )


def _card() -> AnswerCard:
    return AnswerCard(
        kind=MessageKind.ANSWER,
        top_line="❓ When is the raid?",
        paragraphs=("Friday.",),
        footer="Aura answers only from facts recorded on this server.",
    )


class TestLabels:
    def test_with_both_switches_off_the_card_is_the_same_object(self) -> None:
        card = _card()
        assert with_answer_labels(card, locale="en-US", ai_label=False, privacy_line=False) is card

    def test_the_label_leads_the_top_line_and_privacy_joins_the_footer(self) -> None:
        card = with_answer_labels(_card(), locale="de", ai_label=True, privacy_line=True)
        assert card.top_line == "🤖 KI-generiert · ❓ When is the raid?"
        assert card.footer is not None and card.footer.endswith(" · Datenschutz: /aura-privacy")
        assert card.paragraphs == _card().paragraphs  # the checked text is untouched

    def test_a_card_without_top_line_or_footer_gets_them(self) -> None:
        bare = AnswerCard(kind=MessageKind.PROACTIVE, top_line=None, paragraphs=("x",))
        labelled = with_answer_labels(bare, locale="en-US", ai_label=True, privacy_line=True)
        assert labelled.top_line == "🤖 AI-generated"
        assert labelled.footer == "Privacy: /aura-privacy"

    def test_the_label_survives_in_the_embed_author_line(self) -> None:
        long_question = AnswerCard(
            kind=MessageKind.ANSWER, top_line="❓ " + "q" * 300, paragraphs=("x",)
        )
        embed = card_to_embed(
            with_answer_labels(long_question, locale="en-US", ai_label=True, privacy_line=False)
        )
        assert embed.author.name is not None and embed.author.name.startswith("🤖 AI-generated · ")

    def test_legacy_embeds_get_the_same_label_and_line(self) -> None:
        embed = discord.Embed(description="x")
        embed.set_author(name="💡 Aura noticed a question in this channel")
        embed.set_footer(text="automatic")
        label_legacy_embed(embed, locale="en-US", ai_label=True, privacy_line=True)
        assert embed.author.name == "🤖 AI-generated · 💡 Aura noticed a question in this channel"
        assert embed.footer.text == "automatic · Privacy: /aura-privacy"
        untouched = discord.Embed(description="x")
        before = untouched.to_dict()
        label_legacy_embed(untouched, locale="en-US", ai_label=False, privacy_line=False)
        assert untouched.to_dict() == before


class TestWelcomeLine:
    def test_the_onboarding_footers_change_only_when_switched_on(self) -> None:
        from aura.cards import build_onboarding_card
        from aura.onboarding.builder import OnboardingContent
        from aura.onboarding.formatter import build_onboarding_embed

        content = OnboardingContent(
            guild_id=GUILD_A,
            rules=[_fact(1, "Be kind")],
            status_changes=[],
            other=[],
            total_eligible=1,
        )
        card_off = build_onboarding_card(content, locale="en-US", server_name="S", channel_names={})
        card_on = build_onboarding_card(
            content, locale="en-US", server_name="S", channel_names={}, privacy_line=True
        )
        assert (
            card_on.footer
            == f"{card_off.footer} Aura reads some channels here to keep the server's facts current — details: /aura-privacy"
        )
        embed_off = build_onboarding_embed(content, locale="en-US")
        embed_on = build_onboarding_embed(content, locale="en-US", privacy_line=True)
        assert embed_on.footer.text is not None and embed_off.footer.text is not None
        assert embed_on.footer.text.startswith(embed_off.footer.text)
        assert embed_on.footer.text.endswith("/aura-privacy")


@pytest.fixture
async def db() -> AsyncIterator[aiosqlite.Connection]:
    connection = await open_database()
    await connection.execute(
        "INSERT INTO extraction_channel_config (channel_id, guild_id, extraction_enabled, "
        "updated_by_id, updated_at) VALUES (?, ?, 1, 1, 'x')",
        (CHANNEL_A, GUILD_A),
    )
    await connection.commit()
    yield connection
    await connection.close()


def _channel(*, fail: bool = False) -> MagicMock:
    channel = MagicMock()
    channel.id = CHANNEL_A
    channel.send = AsyncMock(
        side_effect=discord.HTTPException(MagicMock(status=403), "no") if fail else None
    )
    return channel


class TestCaptureNotice:
    async def test_posted_once_ever(self, db: aiosqlite.Connection) -> None:
        channel = _channel()
        assert await post_capture_notice_once(db, channel, locale="de")
        assert not await post_capture_notice_once(db, channel, locale="de")
        channel.send.assert_awaited_once()
        text = channel.send.call_args.args[0]
        assert text.startswith("📝 Ab jetzt liest Aura") and "/aura-privacy" in text
        assert channel.send.call_args.kwargs["allowed_mentions"].everyone is False

    async def test_two_moderators_at_once_post_one_notice(self, db: aiosqlite.Connection) -> None:
        channel = _channel()
        results = await asyncio.gather(
            post_capture_notice_once(db, channel, locale="en-US"),
            post_capture_notice_once(db, channel, locale="en-US"),
        )
        assert sorted(results) == [False, True]
        channel.send.assert_awaited_once()

    async def test_a_failed_post_is_tried_again_next_time(self, db: aiosqlite.Connection) -> None:
        assert not await post_capture_notice_once(db, _channel(fail=True), locale="en-US")
        assert await post_capture_notice_once(db, _channel(), locale="en-US")

    async def test_a_channel_with_capture_off_gets_nothing(self, db: aiosqlite.Connection) -> None:
        await db.execute("UPDATE extraction_channel_config SET extraction_enabled = 0")
        await db.commit()
        channel = _channel()
        assert not await post_capture_notice_once(db, channel, locale="en-US")
        channel.send.assert_not_awaited()


class TestSettings:
    def _settings(self, **values: object) -> Settings:
        return Settings(_env_file=None, discord_token="t", **values)  # type: ignore[arg-type]

    def test_every_switch_is_off_by_default(self) -> None:
        settings = self._settings()
        assert settings.database_key_hex is None
        assert settings.data_purge_mode.value == "report"
        assert not settings.data_deletion_enabled and not settings.data_export_enabled
        assert not settings.privacy_info_enabled and not settings.ai_label_enabled
        assert settings.guild_purge_grace_days == 30

    def test_privacy_info_needs_its_link_and_contact(self) -> None:
        with pytest.raises(ValidationError):
            self._settings(privacy_info_enabled=True, privacy_contact="a@b.c")
        with pytest.raises(ValidationError):
            self._settings(privacy_info_enabled=True, privacy_policy_url="https://x.example")

    def test_deletion_needs_the_member_path(self) -> None:
        with pytest.raises(ValidationError):
            self._settings(data_deletion_enabled=True)

    @pytest.mark.parametrize(
        "url", ["http://x.example", "javascript:alert(1)", "https://", "https://a b.example", "x"]
    )
    def test_the_policy_link_must_be_https(self, url: str) -> None:
        with pytest.raises(ValidationError):
            self._settings(privacy_policy_url=url)

    @pytest.mark.parametrize("contact", ["a\nb", "x" * 201, "a\u202eb"])
    def test_the_contact_must_be_one_short_printable_line(self, contact: str) -> None:
        with pytest.raises(ValidationError):
            self._settings(privacy_contact=contact)

    def test_a_bad_key_is_refused_without_being_echoed(self) -> None:
        secret_like = "z" * 64
        with pytest.raises(ValidationError) as error:
            self._settings(database_encryption_key=secret_like)
        assert secret_like not in str(error.value)

    def test_the_ledger_cannot_be_the_database(self) -> None:
        with pytest.raises(ValidationError):
            self._settings(database_path="data/aura.db", deletion_ledger_path="data/./aura.db")

    def test_blank_values_mean_unset(self) -> None:
        settings = self._settings(
            database_encryption_key=" ", privacy_policy_url="", privacy_contact=" "
        )
        assert settings.database_key_hex is None
        assert settings.privacy_policy_url is None and settings.privacy_contact is None

    @pytest.mark.parametrize(
        "field",
        [
            "proactive_signal_retention_days",
            "ask_member_id_retention_days",
            "onboarding_send_retention_days",
            "guild_purge_grace_days",
        ],
    )
    def test_no_period_below_one_day(self, field: str) -> None:
        with pytest.raises(ValidationError):
            self._settings(**{field: 0})


class _Model(BaseModel):
    content: str


class TestLogsHoldNoContent:
    CANARY = "CANARY-p7a-log-91c2"

    def test_a_schema_error_is_described_without_its_input(self) -> None:
        with pytest.raises(ValidationError) as error:
            _Model.model_validate({"content": {"nested": self.CANARY}})
        reason = content_free_reason(error.value)
        assert self.CANARY not in reason and "schema error" in reason

    def test_third_party_loggers_never_go_below_info(self) -> None:
        configure_logging("DEBUG")
        try:
            for name in CONTENT_BEARING_LOGGERS:
                assert logging.getLogger(name).level == logging.INFO
            assert logging.getLogger().level == logging.DEBUG
        finally:
            configure_logging("INFO")

    def test_no_log_call_formats_a_question_answer_or_member(self) -> None:
        """A structural sweep over every logger call's arguments (the format string excepted).

        An argument is the value itself, a slice of it, or an attribute named
        like content: a question, an answer, the model's reasoning or claims, a
        fact's or message's content, or the ID of a member, user or author.
        Counts and flags derived from them (len(...), answer.answers_question)
        are fine.
        """
        import ast

        forbidden = {
            "question",
            "answer",
            "reasoning",
            "content",
            "change_signal",
            "unsupported_claim",
            "contradicted_claim",
            "invented_source",
        }
        person_bases = {"member", "user", "author"}

        def offending(expression: ast.expr) -> bool:
            while isinstance(expression, ast.Subscript):
                expression = expression.value
            if isinstance(expression, ast.Name):
                return expression.id in forbidden
            if isinstance(expression, ast.Attribute):
                if expression.attr in forbidden:
                    return True
                base = expression.value
                base_name = (
                    base.attr if isinstance(base, ast.Attribute) else getattr(base, "id", "")
                )
                return expression.attr == "id" and base_name in person_bases
            return False

        offenders = []
        for path in SRC.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "logger"
                ):
                    offenders.extend(
                        f"{path.name}:{node.lineno}:{ast.unparse(argument)}"
                        for argument in node.args[1:]
                        if offending(argument)
                    )
        assert offenders == []


class TestAuthorLookup:
    async def test_found_unknown_and_failed_are_stored_correctly(self) -> None:
        conn = await open_database()
        try:
            found = await add_fact(conn, guild_id=GUILD_A, author=None, when=BEFORE)
            gone = await add_fact(conn, guild_id=GUILD_A, author=None, when=BEFORE)
            flaky = await add_fact(conn, guild_id=GUILD_A, author=None, when=BEFORE)
            kept = await add_fact(conn, guild_id=GUILD_A, author=OTHER, when=BEFORE)
            async with conn.execute("SELECT id, message_id FROM facts") as cursor:
                message_of = {row[0]: row[1] for row in await cursor.fetchall()}
            answers = {
                message_of[found]: MEMBER,
                message_of[gone]: AuthorUnknown.UNKNOWN,
                message_of[flaky]: LookupFailed.FAILED,
            }
            source = MagicMock()
            source.author_of = AsyncMock(side_effect=lambda _channel, message: answers[message])
            result = await lookup_missing_authors(conn, source)
            assert (result.resolved, result.unknown, result.failed, result.remaining) == (
                1,
                1,
                1,
                1,
            )
            async with conn.execute("SELECT id, source_author_id FROM facts") as cursor:
                authors = {row[0]: row[1] for row in await cursor.fetchall()}
            assert authors == {found: MEMBER, gone: 0, flaky: None, kept: OTHER}
            assert source.author_of.await_count == 3  # a stored author is never asked again
        finally:
            await conn.close()


class TestStartupWiring:
    async def test_leaving_marks_and_returning_clears(self) -> None:
        from aura.main import AuraClient, build_intents

        settings = Settings(_env_file=None, discord_token="t")  # type: ignore[call-arg]
        client = AuraClient(intents=build_intents(), settings=settings)
        client.db = await open_database()
        try:
            guild = MagicMock()
            guild.id = GUILD_A
            await client.on_guild_remove(guild)
            async with client.db.execute("SELECT purge_after FROM guild_departures") as cursor:
                [(purge_after,)] = await cursor.fetchall()
            assert purge_after > datetime.now(UTC).isoformat()[:10]
            await client.on_guild_join(guild)
            async with client.db.execute("SELECT COUNT(*) FROM guild_departures") as cursor:
                assert (await cursor.fetchone()) == (0,)
        finally:
            await client.db.close()

    async def test_ready_reconciles_servers_left_while_offline(self) -> None:
        from aura.main import AuraClient, build_intents

        settings = Settings(_env_file=None, discord_token="t")  # type: ignore[call-arg]
        client = AuraClient(intents=build_intents(), settings=settings)
        client.db = await open_database()
        try:
            await add_fact(client.db, guild_id=GUILD_A, author=OTHER, when=BEFORE)
            with patch.object(AuraClient, "guilds", new=[]):
                await client.on_ready()
            async with client.db.execute("SELECT guild_id FROM guild_departures") as cursor:
                assert await cursor.fetchall() == [(GUILD_A,)]
        finally:
            await client.db.close()

    def test_a_database_that_does_not_open_stops_the_process_with_the_reason(self) -> None:
        from aura import main as main_module
        from aura.db.encryption import DatabaseOpenError

        with (
            patch.object(main_module, "configure_logging"),
            patch.object(main_module, "load_settings", return_value=MagicMock()),
            patch.object(main_module, "get_translator"),
            patch.object(main_module, "create_client") as create,
            pytest.raises(SystemExit) as stopped,
        ):
            create.return_value.run.side_effect = DatabaseOpenError("wrong key")
            main_module.main()
        assert stopped.value.code == 1


class TestStructure:
    def test_only_the_deletion_module_deletes_knowledge_rows(self) -> None:
        pattern = re.compile(
            r"DELETE FROM (facts|pending_facts|fact_variants|ask_calls|onboarding_sends)\b"
        )
        offenders = sorted(
            str(path.relative_to(SRC))
            for path in SRC.rglob("*.py")
            if pattern.search(path.read_text(encoding="utf-8"))
        )
        assert offenders == ["db/deletion.py"]

    def test_the_web_service_imports_nothing_new_from_the_bot(self) -> None:
        web = SRC.parent.parent / "web" / "backend" / "aura_web"
        for path in web.rglob("*.py"):
            assert "aura.privacy" not in path.read_text(encoding="utf-8")


class TestAuthorLookupPacing:
    """The background lookup respects Discord's limits, resumes, and never blocks the bot."""

    async def _missing(self, conn: aiosqlite.Connection, count: int) -> dict[int, int]:
        for _ in range(count):
            await add_fact(conn, guild_id=GUILD_A, author=None, when=BEFORE)
        async with conn.execute("SELECT message_id, id FROM facts") as cursor:
            return {row[0]: row[1] for row in await cursor.fetchall()}

    async def test_every_fetch_after_the_first_is_preceded_by_a_pause(self) -> None:
        from aura.privacy.author_lookup import LOOKUP_PAUSE_SECONDS

        conn = await open_database()
        try:
            await self._missing(conn, 4)
            pauses: list[float] = []

            async def record(seconds: float) -> None:
                pauses.append(seconds)

            source = MagicMock()
            source.author_of = AsyncMock(return_value=MEMBER)
            await lookup_missing_authors(conn, source, sleep=record)
            assert pauses == [LOOKUP_PAUSE_SECONDS] * 3
            assert LOOKUP_PAUSE_SECONDS >= 0.2  # at most five requests a second
        finally:
            await conn.close()

    async def test_a_rate_limit_ends_the_batch_and_the_run_backs_off(self) -> None:
        from aura.privacy.author_lookup import (
            RATE_LIMIT_BACKOFF_SECONDS,
            LookupRateLimited,
            run_author_lookup,
        )

        conn = await open_database()
        try:
            await self._missing(conn, 3)
            answers = iter([MEMBER, LookupRateLimited.RATE_LIMITED, MEMBER, MEMBER])
            source = MagicMock()
            source.author_of = AsyncMock(side_effect=lambda *_: next(answers))
            pauses: list[float] = []

            async def record(seconds: float) -> None:
                pauses.append(seconds)

            remaining = await run_author_lookup(conn, source, sleep=record)
            assert remaining == 0
            assert RATE_LIMIT_BACKOFF_SECONDS in pauses
            assert RATE_LIMIT_BACKOFF_SECONDS >= 30
            assert source.author_of.await_count == 4  # the rate-limited one is asked again
        finally:
            await conn.close()

    async def test_a_cancelled_run_resumes_where_it_stopped(self) -> None:
        from aura.privacy.author_lookup import run_author_lookup

        conn = await open_database()
        try:
            await self._missing(conn, 5)
            asked: list[int] = []
            gate = asyncio.Event()
            third_asked = asyncio.Event()

            async def answer(_channel: int, message: int) -> int:
                asked.append(message)
                if len(asked) == 3:
                    third_asked.set()
                    await gate.wait()  # cancelled here, mid-run
                return MEMBER

            source = MagicMock()
            source.author_of = answer

            async def no_pause(_seconds: float) -> None:
                return None

            task = asyncio.create_task(run_author_lookup(conn, source, sleep=no_pause))
            await asyncio.wait_for(third_asked.wait(), timeout=5)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            async with conn.execute(
                "SELECT COUNT(*) FROM facts WHERE source_author_id IS NULL"
            ) as cursor:
                assert (await cursor.fetchone()) == (3,)  # two stored before the cut
            first_run = list(asked)
            asked.clear()
            gate.set()
            assert await run_author_lookup(conn, source, sleep=no_pause) == 0
            assert set(asked).isdisjoint(first_run[:2])  # stored ones are never asked again
            assert len(asked) == 3
        finally:
            await conn.close()

    async def test_the_bot_keeps_using_the_database_while_a_lookup_waits_on_discord(self) -> None:
        from aura.db.repository import create_fact
        from aura.privacy.author_lookup import run_author_lookup

        conn = await open_database()
        try:
            await self._missing(conn, 2)
            release = asyncio.Event()

            async def slow(_channel: int, _message: int) -> int:
                await release.wait()
                return MEMBER

            source = MagicMock()
            source.author_of = slow

            async def no_pause(_seconds: float) -> None:
                return None

            task = asyncio.create_task(run_author_lookup(conn, source, sleep=no_pause))
            await asyncio.sleep(0)
            # While the lookup waits for Discord, an ordinary write completes.
            await asyncio.wait_for(
                create_fact(
                    conn,
                    guild_id=GUILD_A,
                    channel_id=1,
                    message_id=99,
                    content="written meanwhile",
                    embedding=bytes(384 * 4),
                ),
                timeout=2,
            )
            release.set()
            assert await asyncio.wait_for(task, timeout=5) == 0
        finally:
            await conn.close()

    async def test_the_gateway_maps_a_429_to_rate_limited(self) -> None:
        from aura.privacy.author_lookup import LookupRateLimited
        from aura.privacy.gateway import ClientMessageAuthorSource

        response = MagicMock(status=429)
        channel = MagicMock(spec=discord.TextChannel)
        channel.fetch_message = AsyncMock(side_effect=discord.HTTPException(response, "slow down"))
        client = MagicMock()
        client.get_channel = MagicMock(return_value=channel)
        answer = await ClientMessageAuthorSource(client).author_of(1, 2)
        assert answer is LookupRateLimited.RATE_LIMITED

    async def test_a_rate_limit_stops_the_batch_at_once(self) -> None:
        from aura.privacy.author_lookup import LookupRateLimited

        conn = await open_database()
        try:
            await self._missing(conn, 3)
            source = MagicMock()
            source.author_of = AsyncMock(
                side_effect=[LookupRateLimited.RATE_LIMITED, MEMBER, MEMBER]
            )

            async def no_pause(_seconds: float) -> None:
                return None

            result = await lookup_missing_authors(conn, source, sleep=no_pause)
            assert result.rate_limited and result.remaining == 3
            assert source.author_of.await_count == 1
        finally:
            await conn.close()

    async def test_an_author_stored_meanwhile_is_never_overwritten(self) -> None:
        conn = await open_database()
        try:
            [(message_id, fact_id)] = (await self._missing(conn, 1)).items()

            async def answer(_channel: int, _message: int) -> int:
                # Another writer stores the author while Discord is asked.
                await conn.execute(
                    "UPDATE facts SET source_author_id = ? WHERE id = ?", (OTHER, fact_id)
                )
                await conn.commit()
                return MEMBER

            source = MagicMock()
            source.author_of = answer

            async def no_pause(_seconds: float) -> None:
                return None

            await lookup_missing_authors(conn, source, sleep=no_pause)
            async with conn.execute(
                "SELECT source_author_id FROM facts WHERE message_id = ?", (message_id,)
            ) as cursor:
                assert (await cursor.fetchone()) == (OTHER,)
        finally:
            await conn.close()
