"""Tests for AuraClient's own event wiring, without a gateway connection.

The listener's decisions are covered in test_proactive_listener.py; what is
covered here is the part that only exists in main.py -- that on_message is
actually reachable, actually delegates, and cannot blow up on a message that
arrives before startup has finished.
"""
from __future__ import annotations

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import discord
import pytest

from aura.backfill import ClientBackfillGateway
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

    def test_members_intent_is_requested(self) -> None:
        # Without it, on_member_join (Phase 3d's onboarding trigger) never
        # fires at all -- no error, no log line, the bot just silently never
        # welcomes anyone. Both this and message_content must ALSO be enabled
        # in the Discord Developer Portal; see build_intents' docstring.
        assert build_intents().members is True


_STARTUP_ATTRIBUTES = (
    "db",
    "question_detector",
    "fact_worthiness_detector",
    "embedding_model",
    "gate_config",
)


def _started_client() -> AuraClient:
    """A client with every dependency setup_hook would have installed."""
    client = _client()
    client.db = MagicMock()
    client.question_detector = MagicMock()
    client.fact_worthiness_detector = MagicMock()
    client.embedding_model = MagicMock()
    client.gate_config = MagicMock()
    return client


class TestOnMessage:
    async def test_a_message_is_handed_to_the_listener_with_the_clients_own_dependencies(
        self,
    ) -> None:
        client = _started_client()
        message = _make_message()

        with patch("aura.main.handle_extraction_message", AsyncMock()):
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

    async def test_the_same_message_also_reaches_the_extraction_path_independently(
        self,
    ) -> None:
        # Phase 3a-2: both passive paths see the same raw message, as siblings
        # rather than as a chain. The dependencies each receives are what proves
        # they are genuinely separate -- extraction gets the fact-worthiness
        # detector and never the question detector, so neither path's gate can
        # be silently reused for the other's decision (the collision
        # reports/phase-3-pre-analysis.md Section 1c warned about).
        client = _started_client()
        message = _make_message()

        with patch("aura.main.handle_message", AsyncMock()):
            with patch("aura.main.handle_extraction_message", AsyncMock()) as extractor:
                await client.on_message(message)

        extractor.assert_awaited_once()
        args, kwargs = extractor.call_args
        assert args[0] is message
        assert kwargs["db"] is client.db
        assert kwargs["detector"] is client.fact_worthiness_detector
        assert kwargs["detector"] is not client.question_detector
        assert kwargs["settings"] is client.settings

    @pytest.mark.parametrize("missing", _STARTUP_ATTRIBUTES)
    async def test_a_message_arriving_before_startup_finishes_is_skipped_not_crashed(
        self, missing: str, caplog: pytest.LogCaptureFixture
    ) -> None:
        # Unreachable in production (setup_hook completes before the gateway
        # delivers anything), so the cost of being wrong about that would be
        # an AttributeError on every single message. Parametrized over every
        # dependency, because a guard that checks four of five is a guard
        # that fails on the fifth -- and Phase 3a-2 added the fifth.
        client = _started_client()
        setattr(client, missing, None)

        with patch("aura.main.handle_extraction_message", AsyncMock()) as extractor:
            with patch("aura.main.handle_message", AsyncMock()) as handler:
                with caplog.at_level(logging.WARNING):
                    await client.on_message(_make_message())

        handler.assert_not_awaited()
        extractor.assert_not_awaited()
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


def _make_member() -> MagicMock:
    member = MagicMock(spec=discord.Member)
    member.id = 42
    member.bot = False
    member.guild = MagicMock()
    member.guild.id = GUILD_A
    member.joined_at = None
    return member


_ONBOARDING_STARTUP_ATTRIBUTES = ("db", "onboarding_gateway")


class TestOnMemberJoin:
    async def test_a_join_is_handed_to_the_listener_with_the_clients_own_dependencies(
        self,
    ) -> None:
        client = _client()
        client.db = MagicMock()
        client.onboarding_gateway = MagicMock()
        member = _make_member()

        with patch("aura.main.handle_member_join", AsyncMock()) as handler:
            await client.on_member_join(member)

        handler.assert_awaited_once()
        args, kwargs = handler.call_args
        assert args[0] is member
        assert kwargs["db"] is client.db
        assert kwargs["gateway"] is client.onboarding_gateway
        assert kwargs["settings"] is client.settings

    @pytest.mark.parametrize("missing", _ONBOARDING_STARTUP_ATTRIBUTES)
    async def test_a_join_before_startup_finishes_is_skipped_not_crashed(
        self, missing: str, caplog: pytest.LogCaptureFixture
    ) -> None:
        client = _client()
        client.db = MagicMock()
        client.onboarding_gateway = MagicMock()
        setattr(client, missing, None)

        with patch("aura.main.handle_member_join", AsyncMock()) as handler:
            with caplog.at_level(logging.WARNING):
                await client.on_member_join(_make_member())

        handler.assert_not_awaited()
        assert any(record.levelno >= logging.WARNING for record in caplog.records)

    def test_a_fresh_client_has_no_onboarding_gateway_yet(self) -> None:
        assert _client().onboarding_gateway is None


class TestMessageDeleteAndEdit:
    """A deleted or edited message must stand down BOTH passive paths.

    Phase 2b-1 needed it to cancel a pending grace period; Phase 3a-2 needs it
    to withdraw the message from its pending extraction batch as well. Each
    test below asserts both, because the failure worth catching is one of the
    two being wired and the other quietly not.
    """

    async def test_a_deleted_message_notifies_both_passive_paths(self) -> None:
        client = _started_client()
        message = _make_message()

        with patch("aura.main.withdraw_message", AsyncMock()) as withdraw:
            with patch.object(client.grace_registry, "notice_message_gone") as notice:
                await client.on_message_delete(message)

        notice.assert_called_once_with(channel_id=message.channel.id, message_id=message.id)
        withdraw.assert_awaited_once_with(
            client.db, channel_id=message.channel.id, message_id=message.id
        )

    async def test_an_uncached_raw_deletion_also_notifies_both_passive_paths(self) -> None:
        client = _started_client()
        payload = MagicMock(spec=discord.RawMessageDeleteEvent)
        payload.channel_id = 5
        payload.message_id = 9

        with patch("aura.main.withdraw_message", AsyncMock()) as withdraw:
            with patch.object(client.grace_registry, "notice_message_gone") as notice:
                await client.on_raw_message_delete(payload)

        notice.assert_called_once_with(channel_id=5, message_id=9)
        withdraw.assert_awaited_once_with(client.db, channel_id=5, message_id=9)

    async def test_an_edited_message_notifies_both_passive_paths_using_the_after_state(
        self,
    ) -> None:
        client = _started_client()
        before = _make_message()
        after = _make_message()
        after.content = "a different question now"

        with patch("aura.main.withdraw_message", AsyncMock()) as withdraw:
            with patch.object(client.grace_registry, "notice_message_gone") as notice:
                await client.on_message_edit(before, after)

        notice.assert_called_once_with(channel_id=after.channel.id, message_id=after.id)
        withdraw.assert_awaited_once_with(
            client.db, channel_id=after.channel.id, message_id=after.id
        )

    async def test_a_deletion_before_startup_finishes_does_not_crash(self) -> None:
        # on_message_delete can fire before setup_hook has installed the
        # database (a gateway event racing startup), and an unguarded call
        # would be an AttributeError on a connection that is still None. The
        # grace registry exists from __init__ and is still notified.
        client = _client()

        with patch.object(client.grace_registry, "notice_message_gone") as notice:
            await client.on_message_delete(_make_message())

        notice.assert_called_once()


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


class TestBackgroundTasks:
    """Both long-lived tasks must be stopped before the connection they use is.

    Closing the database first lets an in-flight sweep or digest hit a closed
    connection and log an exception on the way out of a clean shutdown -- noise
    that reads exactly like a real fault in a container log after a restart.
    Phase 3e added a second task and Phase 3b a third, so this is also the guard
    against a future fourth one being created and never cancelled.
    """

    _BACKGROUND_TASKS = ("extraction_sweeper", "digest_scheduler", "backfill_worker")

    @pytest.mark.parametrize("attribute", _BACKGROUND_TASKS)
    def test_a_fresh_client_has_no_background_task_yet(self, attribute: str) -> None:
        assert getattr(_client(), attribute) is None

    async def test_close_cancels_every_background_task_before_closing_the_database(
        self,
    ) -> None:
        client = _started_client()
        client.db = AsyncMock()
        order: list[str] = []

        async def forever(name: str, running: asyncio.Event) -> None:
            running.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                order.append(name)
                raise

        running = {name: asyncio.Event() for name in self._BACKGROUND_TASKS}
        for name in self._BACKGROUND_TASKS:
            setattr(client, name, asyncio.create_task(forever(name, running[name])))
        # Every task must actually be RUNNING before close(): a task cancelled
        # before its first step never reaches its own except clause, which would
        # make the recorded order an artifact of the scheduler rather than of
        # close()'s own sequencing.
        await asyncio.gather(*(event.wait() for event in running.values()))
        client.db.close = AsyncMock(side_effect=lambda: order.append("db"))

        with patch.object(discord.Client, "close", AsyncMock()):
            await client.close()

        assert order == [*self._BACKGROUND_TASKS, "db"]
        assert all(getattr(client, name) is None for name in self._BACKGROUND_TASKS)

    async def test_close_works_before_startup_ever_created_the_tasks(self) -> None:
        # A process that fails during setup_hook still gets closed.
        client = _client()

        with patch.object(discord.Client, "close", AsyncMock()):
            await client.close()


class TestBackfillWiring:
    """Phase 3b's worker is started once, with the dependencies it must share.

    The one that matters is the detector. Backfill runs the SAME first filter
    live extraction runs, so it must be handed the fact-worthiness detector and
    provably not the question detector -- two instances of the same thing would
    be two places a future exemplar change has to land, and the wrong instance
    would silently score history against the wrong exemplars.
    """

    async def test_setup_hook_starts_the_worker_with_the_shared_detector(self) -> None:
        client = _client()
        client.db = MagicMock()
        client.embedding_model = MagicMock()
        fact_worthiness = MagicMock(name="fact_worthiness_detector")
        question = MagicMock(name="question_detector")

        started: dict[str, object] = {}

        async def fake_worker(db, model, gateway, detector, *, settings) -> None:
            started.update(
                db=db, model=model, gateway=gateway, detector=detector, settings=settings
            )
            await asyncio.Event().wait()

        with (
            patch("aura.main.aiosqlite.connect", AsyncMock(return_value=client.db)),
            patch("aura.main.init_schema", AsyncMock()),
            patch("aura.main.verify_signal_schema", AsyncMock()),
            patch("aura.main.verify_pending_facts_schema", AsyncMock()),
            patch("aura.main.asyncio.to_thread", AsyncMock(return_value=client.embedding_model)),
            patch(
                "aura.main.QuestionDetector.create", AsyncMock(return_value=question)
            ),
            patch(
                "aura.main.create_fact_worthiness_detector",
                AsyncMock(return_value=fact_worthiness),
            ),
            patch("aura.main.run_extraction_sweeper", AsyncMock()),
            patch("aura.main.run_digest_scheduler", AsyncMock()),
            patch("aura.main.run_backfill_worker", fake_worker),
            patch.object(client.tree, "sync", AsyncMock()),
        ):
            await client.setup_hook()
            # Let the freshly created task reach its first await.
            await asyncio.sleep(0)

            assert client.backfill_worker is not None
            assert started["detector"] is fact_worthiness
            assert started["detector"] is not question
            assert started["db"] is client.db
            assert started["model"] is client.embedding_model
            assert started["settings"] is client.settings
            assert isinstance(started["gateway"], ClientBackfillGateway)

            client.backfill_worker.cancel()
            with pytest.raises(asyncio.CancelledError):
                await client.backfill_worker

    async def test_the_backfill_command_group_is_registered(self) -> None:
        client = _client()
        client.db = MagicMock()
        client.embedding_model = MagicMock()

        with (
            patch("aura.main.aiosqlite.connect", AsyncMock(return_value=client.db)),
            patch("aura.main.init_schema", AsyncMock()),
            patch("aura.main.verify_signal_schema", AsyncMock()),
            patch("aura.main.verify_pending_facts_schema", AsyncMock()),
            patch("aura.main.asyncio.to_thread", AsyncMock(return_value=MagicMock())),
            patch("aura.main.QuestionDetector.create", AsyncMock()),
            patch("aura.main.create_fact_worthiness_detector", AsyncMock()),
            patch("aura.main.run_extraction_sweeper", AsyncMock()),
            patch("aura.main.run_digest_scheduler", AsyncMock()),
            patch("aura.main.run_backfill_worker", AsyncMock()),
            patch.object(client.tree, "sync", AsyncMock()),
        ):
            await client.setup_hook()

        names = {command.name for command in client.tree.get_commands()}
        assert "aura-backfill" in names

        for task_name in ("extraction_sweeper", "digest_scheduler", "backfill_worker"):
            task = getattr(client, task_name)
            if task is not None:
                task.cancel()
