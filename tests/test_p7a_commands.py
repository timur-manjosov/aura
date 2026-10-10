"""The P7a commands: who may use them, what they touch, and that nobody can name another person."""

from __future__ import annotations

import csv
import io
from collections.abc import AsyncIterator
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import aiosqlite
import discord
import pytest
from discord import app_commands

import aura.commands.data_admin as data_admin
import aura.commands.privacy as privacy
from aura.commands.data_admin import (
    DeleteServerDataModal,
    ForgetFactView,
    delete_server_data_command,
    export_command,
    forget_command,
    names_match,
    register_data_admin_commands,
)
from aura.commands.operator_privacy import forget_member_command, parse_discord_id, privacy_group
from aura.commands.privacy import DeletionConfirmView, PrivacyView, privacy_command, privacy_text
from aura.config import Settings
from aura.db.deletion import MemberDeletionMode
from aura.privacy.ledger import DeletionKind, DeletionLedger, DeletionReason
from tests.privacy_data import (
    BEFORE,
    GUILD_A,
    GUILD_B,
    MEMBER,
    OTHER,
    add_fact,
    open_database,
    populate,
)

MODERATOR_PERMISSIONS = discord.Permissions(manage_guild=True)


def _settings(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "privacy_info_enabled": True,
        "privacy_policy_url": "https://example.org/privacy",
        "privacy_contact": "privacy@example.org",
        "data_deletion_enabled": True,
        "data_export_enabled": True,
    }
    values.update(overrides)
    return Settings(_env_file=None, discord_token="test", **values)  # type: ignore[arg-type]


@pytest.fixture
async def db() -> AsyncIterator[aiosqlite.Connection]:
    connection = await open_database()
    yield connection
    await connection.close()


@pytest.fixture
async def ledger(tmp_path: Path) -> AsyncIterator[DeletionLedger]:
    opened = await DeletionLedger.open(str(tmp_path / "ledger.db"), None)
    yield opened
    await opened.close()


@pytest.fixture(autouse=True)
def _fresh_cooldowns() -> None:
    privacy._last_request.clear()
    data_admin._last_export.clear()


def _interaction(
    db: aiosqlite.Connection,
    ledger: DeletionLedger,
    *,
    user_id: int = MEMBER,
    guild_id: int = GUILD_A,
    guild_name: str = "Test Server",
    manage_guild: bool = True,
    settings: Settings | None = None,
) -> MagicMock:
    interaction = MagicMock(spec=discord.Interaction)
    interaction.locale = "en-US"
    interaction.guild_id = guild_id
    interaction.guild = MagicMock()
    interaction.guild.id = guild_id
    interaction.guild.name = guild_name
    interaction.permissions = discord.Permissions(manage_guild=manage_guild)
    interaction.client = MagicMock()
    interaction.client.db = db
    interaction.client.deletion_ledger = ledger
    interaction.client.settings = settings or _settings()
    interaction.user = MagicMock()
    interaction.user.id = user_id
    interaction.response = MagicMock()
    interaction.response.send_message = AsyncMock()
    interaction.response.edit_message = AsyncMock()
    interaction.response.defer = AsyncMock()
    interaction.response.send_modal = AsyncMock()
    interaction.response.is_done = MagicMock(return_value=False)
    interaction.followup = MagicMock()
    interaction.followup.send = AsyncMock()
    interaction.edit_original_response = AsyncMock()
    interaction.original_response = AsyncMock(return_value=MagicMock())
    return interaction


async def _count(conn: aiosqlite.Connection, sql: str, *parameters: object) -> int:
    async with conn.execute(sql, parameters) as cursor:
        row = await cursor.fetchone()
    assert row is not None
    return int(row[0])


class TestPermissions:
    @pytest.mark.parametrize(
        "command", [forget_command, delete_server_data_command, export_command]
    )
    def test_admin_commands_refuse_members(self, command: app_commands.Command) -> None:  # type: ignore[type-arg]
        with pytest.raises(app_commands.MissingPermissions):
            for check in command.checks:
                check(MagicMock(permissions=discord.Permissions(manage_guild=False)))

    @pytest.mark.parametrize(
        "command", [forget_command, delete_server_data_command, export_command]
    )
    def test_admin_commands_allow_server_managers(self, command: app_commands.Command) -> None:  # type: ignore[type-arg]
        for check in command.checks:
            assert check(MagicMock(permissions=MODERATOR_PERMISSIONS)) is True

    def test_the_privacy_command_has_no_permission_gate(self) -> None:
        assert privacy_command.checks == []

    def test_every_operator_subcommand_requires_the_operator(self) -> None:
        for command in privacy_group.commands:
            assert command.checks, command.name
            interaction = MagicMock()
            interaction.client.settings.operator_discord_user_id = 42
            interaction.user.id = 7
            assert all(check(interaction) is False for check in command.checks)


class TestPrivacyCommand:
    async def test_the_summary_names_the_period_and_the_contact(
        self, db: aiosqlite.Connection, ledger: DeletionLedger
    ) -> None:
        interaction = _interaction(db, ledger)
        await privacy_command.callback(interaction)  # type: ignore[call-arg]
        kwargs = interaction.response.send_message.call_args.kwargs
        assert kwargs["ephemeral"] is True
        description = kwargs["embed"].description
        assert "30 days" in description and "privacy@example.org" in description
        view = kwargs["view"]
        labels = {getattr(item, "label", None) for item in view.children}
        assert {"Privacy policy", "Delete my data"} <= labels

    async def test_without_deletion_there_is_no_button_only_the_contact(
        self, db: aiosqlite.Connection, ledger: DeletionLedger
    ) -> None:
        settings = _settings(data_deletion_enabled=False)
        view = PrivacyView(settings=settings, locale="en-US", invoker_id=MEMBER, guild_id=GUILD_A)
        assert {getattr(item, "label", None) for item in view.children} == {"Privacy policy"}
        assert "write to privacy@example.org" in privacy_text(settings, "en-US")


class TestMemberDeletion:
    async def test_the_default_choice_deletes_the_members_facts_everywhere(
        self, db: aiosqlite.Connection, ledger: DeletionLedger
    ) -> None:
        ids = await populate(db)
        view = DeletionConfirmView(locale="en-US", invoker_id=MEMBER, guild_id=GUILD_A, contact="c")
        interaction = _interaction(db, ledger)
        await view.confirm.callback(interaction)
        assert (
            await _count(
                db,
                "SELECT COUNT(*) FROM facts WHERE id IN (?, ?)",
                ids["a_member_fact"],
                ids["b_member_fact"],
            )
            == 0
        )
        [entry] = await ledger.entries()
        assert entry.kind is DeletionKind.MEMBER and entry.guild_id is None
        assert entry.user_id == MEMBER and entry.mode is MemberDeletionMode.DELETE_FACTS
        assert entry.reason is DeletionReason.MEMBER_REQUEST
        message = interaction.edit_original_response.call_args.kwargs["content"]
        assert message.startswith("Done. 4 fact(s) deleted")  # every fact of theirs up to now

    async def test_only_the_invoking_members_id_is_ever_used(
        self, db: aiosqlite.Connection, ledger: DeletionLedger
    ) -> None:
        ids = await populate(db)
        view = DeletionConfirmView(locale="en-US", invoker_id=OTHER, guild_id=GUILD_A, contact="c")
        # The view belongs to OTHER; nothing anyone does with it can reach MEMBER.
        await view.confirm.callback(_interaction(db, ledger, user_id=OTHER))
        assert (
            await _count(db, "SELECT COUNT(*) FROM facts WHERE id = ?", ids["a_member_fact"]) == 1
        )
        assert await _count(db, "SELECT COUNT(*) FROM facts WHERE id = ?", ids["a_other_fact"]) == 0

    async def test_even_past_the_button_check_only_the_owners_id_is_used(
        self, db: aiosqlite.Connection, ledger: DeletionLedger
    ) -> None:
        ids = await populate(db)
        view = DeletionConfirmView(locale="en-US", invoker_id=OTHER, guild_id=GUILD_A, contact="c")
        # The callback reached directly with someone else's interaction: the
        # deletion still concerns the view's owner, never the clicker.
        await view.confirm.callback(_interaction(db, ledger, user_id=MEMBER))
        [entry] = await ledger.entries()
        assert entry.user_id == OTHER
        assert (
            await _count(db, "SELECT COUNT(*) FROM facts WHERE id = ?", ids["a_member_fact"]) == 1
        )

    async def test_someone_else_cannot_press_the_buttons(
        self, db: aiosqlite.Connection, ledger: DeletionLedger
    ) -> None:
        view = DeletionConfirmView(locale="en-US", invoker_id=MEMBER, guild_id=GUILD_A, contact="c")
        intruder = _interaction(db, ledger, user_id=OTHER)
        assert await view.interaction_check(intruder) is False
        privacy_view = PrivacyView(
            settings=_settings(), locale="en-US", invoker_id=MEMBER, guild_id=GUILD_A
        )
        assert await privacy_view._open_confirmation(intruder) is None
        assert "Only the person" in intruder.response.send_message.call_args.args[0]

    async def test_this_server_and_unlink_are_honoured(
        self, db: aiosqlite.Connection, ledger: DeletionLedger
    ) -> None:
        ids = await populate(db)
        view = DeletionConfirmView(locale="en-US", invoker_id=MEMBER, guild_id=GUILD_A, contact="c")
        view.scope = "this"
        view.mode = MemberDeletionMode.UNLINK
        await view.confirm.callback(_interaction(db, ledger))
        assert (
            await _count(db, "SELECT message_id FROM facts WHERE id = ?", ids["a_member_fact"]) == 0
        )
        assert (
            await _count(
                db,
                "SELECT COUNT(*) FROM facts WHERE id = ? AND message_id > 0",
                ids["b_member_fact"],
            )
            == 1
        )

    async def test_a_double_click_deletes_once(
        self, db: aiosqlite.Connection, ledger: DeletionLedger
    ) -> None:
        await populate(db)
        view = DeletionConfirmView(locale="en-US", invoker_id=MEMBER, guild_id=GUILD_A, contact="c")
        await view.confirm.callback(_interaction(db, ledger))
        await view.confirm.callback(_interaction(db, ledger))
        assert await ledger.count() == 1

    async def test_a_second_request_within_the_cooldown_is_refused(
        self, db: aiosqlite.Connection, ledger: DeletionLedger
    ) -> None:
        await populate(db)
        first = DeletionConfirmView(
            locale="en-US", invoker_id=MEMBER, guild_id=GUILD_A, contact="c"
        )
        await first.confirm.callback(_interaction(db, ledger))
        second = DeletionConfirmView(
            locale="en-US", invoker_id=MEMBER, guild_id=GUILD_A, contact="c"
        )
        interaction = _interaction(db, ledger)
        await second.confirm.callback(interaction)
        assert await ledger.count() == 1
        assert "wait a few minutes" in interaction.response.edit_message.call_args.kwargs["content"]

    async def test_cancel_deletes_nothing(
        self, db: aiosqlite.Connection, ledger: DeletionLedger
    ) -> None:
        await populate(db)
        view = DeletionConfirmView(locale="en-US", invoker_id=MEMBER, guild_id=GUILD_A, contact="c")
        await view.cancel.callback(_interaction(db, ledger))
        await view.confirm.callback(_interaction(db, ledger))  # after cancel: ignored
        assert await ledger.count() == 0


class TestForgetFact:
    async def test_another_servers_fact_cannot_be_reached(
        self, db: aiosqlite.Connection, ledger: DeletionLedger
    ) -> None:
        ids = await populate(db)
        interaction = _interaction(db, ledger, guild_id=GUILD_B)
        await forget_command.callback(interaction, ids["a_member_fact"])  # type: ignore[call-arg]
        assert "There is no fact" in interaction.response.send_message.call_args.args[0]
        view = ForgetFactView(
            locale="en-US", invoker_id=MEMBER, guild_id=GUILD_B, fact_id=ids["a_member_fact"]
        )
        await view.confirm.callback(_interaction(db, ledger, guild_id=GUILD_B))
        assert (
            await _count(db, "SELECT COUNT(*) FROM facts WHERE id = ?", ids["a_member_fact"]) == 1
        )
        assert await ledger.count() == 0

    async def test_confirming_deletes_the_fact_and_records_it(
        self, db: aiosqlite.Connection, ledger: DeletionLedger
    ) -> None:
        fact_id = await add_fact(
            db, guild_id=GUILD_A, author=OTHER, when=BEFORE, content="names *someone*"
        )
        interaction = _interaction(db, ledger)
        await forget_command.callback(interaction, fact_id)  # type: ignore[call-arg]
        shown = interaction.response.send_message.call_args.kwargs["embed"].description
        assert "names \\*someone\\*" in shown  # escaped, not rendered
        view = interaction.response.send_message.call_args.kwargs["view"]
        await view.confirm.callback(_interaction(db, ledger))
        assert await _count(db, "SELECT COUNT(*) FROM facts WHERE id = ?", fact_id) == 0
        [entry] = await ledger.entries()
        assert entry.kind is DeletionKind.FACT and entry.reason is DeletionReason.MODERATOR_REQUEST


class TestDeleteServerData:
    @pytest.mark.parametrize(
        ("typed", "name", "matches"),
        [
            ("Test Server", "Test Server", True),
            ("  test   SERVER ", "Test Server", True),
            ("ＴＥＳＴ Server", "Test Server", True),  # full-width, NFKC
            ("Test Serve", "Test Server", False),
            ("", "Test Server", False),
            ("", "", False),
            ("Test Server", "Test\u200b Server\u202e", True),  # invisible characters ignored
            ("\u200b", "\u200b", False),  # nothing visible is never a match
        ],
    )
    def test_the_typed_name_must_match(self, typed: str, name: str, matches: bool) -> None:
        assert names_match(typed, name) is matches

    async def _submit(
        self, db: aiosqlite.Connection, ledger: DeletionLedger, typed: str, **kwargs: object
    ) -> MagicMock:
        modal = DeleteServerDataModal(locale="en-US", guild_id=GUILD_A)
        modal.name_input._value = typed  # what Discord fills in on submit
        interaction = _interaction(db, ledger, **kwargs)  # type: ignore[arg-type]
        await modal.on_submit(interaction)
        return interaction

    async def test_the_right_name_deletes_only_this_server(
        self, db: aiosqlite.Connection, ledger: DeletionLedger
    ) -> None:
        await populate(db)
        interaction = await self._submit(db, ledger, "test server")
        assert await _count(db, "SELECT COUNT(*) FROM facts WHERE guild_id = ?", GUILD_A) == 0
        assert await _count(db, "SELECT COUNT(*) FROM facts WHERE guild_id = ?", GUILD_B) > 0
        assert (
            await _count(db, "SELECT COUNT(*) FROM guild_subscriptions WHERE guild_id = ?", GUILD_A)
            == 1
        )
        assert "Billing records are kept" in interaction.followup.send.call_args.args[0]

    async def test_a_wrong_name_deletes_nothing(
        self, db: aiosqlite.Connection, ledger: DeletionLedger
    ) -> None:
        await populate(db)
        await self._submit(db, ledger, "Another Server")
        assert await _count(db, "SELECT COUNT(*) FROM facts WHERE guild_id = ?", GUILD_A) > 0
        assert await ledger.count() == 0

    async def test_a_permission_lost_before_submitting_deletes_nothing(
        self, db: aiosqlite.Connection, ledger: DeletionLedger
    ) -> None:
        await populate(db)
        await self._submit(db, ledger, "Test Server", manage_guild=False)
        assert await _count(db, "SELECT COUNT(*) FROM facts WHERE guild_id = ?", GUILD_A) > 0

    async def test_a_modal_from_another_server_deletes_nothing(
        self, db: aiosqlite.Connection, ledger: DeletionLedger
    ) -> None:
        await populate(db)
        await self._submit(db, ledger, "Test Server", guild_id=GUILD_B)
        assert await ledger.count() == 0


class TestExport:
    async def test_the_admin_gets_two_ephemeral_files(
        self, db: aiosqlite.Connection, ledger: DeletionLedger
    ) -> None:
        await populate(db)
        interaction = _interaction(db, ledger)
        await export_command.callback(interaction)  # type: ignore[call-arg]
        kwargs = interaction.followup.send.call_args.kwargs
        assert kwargs["ephemeral"] is True
        names = sorted(file.filename for file in kwargs["files"])
        assert names[0].endswith(".csv") and names[1].endswith(".md")
        csv_file = next(file for file in kwargs["files"] if file.filename.endswith(".csv"))
        rows = list(csv.reader(io.StringIO(csv_file.fp.read().decode("utf-8-sig"))))
        guild_a_facts = await _count(db, "SELECT COUNT(*) FROM facts WHERE guild_id = ?", GUILD_A)
        assert len(rows) == guild_a_facts + 1  # only this server's facts

    async def test_a_second_export_within_the_cooldown_is_refused(
        self, db: aiosqlite.Connection, ledger: DeletionLedger
    ) -> None:
        await populate(db)
        await export_command.callback(_interaction(db, ledger))  # type: ignore[call-arg]
        again = _interaction(db, ledger)
        await export_command.callback(again)  # type: ignore[call-arg]
        assert "Please try again in 10 minute(s)" in again.response.send_message.call_args.args[0]
        other_server = _interaction(db, ledger, guild_id=GUILD_B)
        await export_command.callback(other_server)  # type: ignore[call-arg]
        assert other_server.followup.send.call_args.kwargs.get("files")

    async def test_an_empty_server_says_so(
        self, db: aiosqlite.Connection, ledger: DeletionLedger
    ) -> None:
        interaction = _interaction(db, ledger)
        await export_command.callback(interaction)  # type: ignore[call-arg]
        assert "no facts to export" in interaction.followup.send.call_args.args[0]


class TestOperator:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("123", 123),
            (" 123 ", 123),
            ("0", None),
            ("-5", None),
            ("1e5", None),
            ("١٢٣", None),  # non-ASCII digits
            (str(2**63), None),
            ("", None),
            (None, None),
        ],
    )
    def test_ids_are_parsed_strictly(self, raw: str | None, expected: int | None) -> None:
        assert parse_discord_id(raw) == expected

    async def test_a_malformed_id_deletes_nothing(
        self, db: aiosqlite.Connection, ledger: DeletionLedger
    ) -> None:
        await populate(db)
        interaction = _interaction(db, ledger)
        await forget_member_command.callback(interaction, "not-an-id")  # type: ignore[call-arg]
        assert await ledger.count() == 0
        await forget_member_command.callback(interaction, str(MEMBER), "x")  # type: ignore[call-arg]
        assert await ledger.count() == 0

    async def test_the_operator_runs_the_same_rule_with_its_own_reason(
        self, db: aiosqlite.Connection, ledger: DeletionLedger
    ) -> None:
        await populate(db)
        await forget_member_command.callback(_interaction(db, ledger), str(MEMBER), str(GUILD_A))  # type: ignore[call-arg]
        [entry] = await ledger.entries()
        assert entry.reason is DeletionReason.OPERATOR_REQUEST and entry.guild_id == GUILD_A


class TestRegistration:
    def test_nothing_is_registered_with_the_switches_off(self) -> None:
        tree = MagicMock()
        register_data_admin_commands(tree, deletion=False, export=False)
        tree.add_command.assert_not_called()

    def test_each_switch_registers_only_its_commands(self) -> None:
        tree = MagicMock()
        register_data_admin_commands(tree, deletion=False, export=True)
        assert [call.args[0].name for call in tree.add_command.call_args_list] == ["aura-export"]
