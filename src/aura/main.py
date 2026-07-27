"""Entry point for the Aura Discord bot."""
from __future__ import annotations

import asyncio
import logging
import sys

import aiosqlite
import discord
from discord import app_commands
from fastembed import TextEmbedding

from aura.commands import (
    register_ask_command,
    register_config_command,
    register_fact_commands,
    register_proactive_commands,
    register_supersede_command,
)
from aura.config import ConfigurationError, ModelComponent, Settings, load_settings
from aura.db import init_schema
from aura.db.proactive_signals import OutdatedDiagnosticTableError, verify_signal_schema
from aura.i18n import DEFAULT_LOCALE, TranslationLoadError, Translator, get_translator
from aura.logging_config import configure_logging
from aura.proactive import GraceRegistry, ProactiveGateConfig, QuestionDetector, handle_message

logger = logging.getLogger(__name__)


class AuraClient(discord.Client):
    """Minimal Discord client that exposes only slash (application) commands."""

    def __init__(self, *, intents: discord.Intents, settings: Settings) -> None:
        super().__init__(intents=intents)
        self.tree = app_commands.CommandTree(self)
        self.settings = settings
        self.db: aiosqlite.Connection | None = None
        self.embedding_model: TextEmbedding | None = None
        self.question_detector: QuestionDetector | None = None
        # Built once in setup_hook rather than per message: it is derived
        # entirely from settings, and validating it on every incoming message
        # would be work with an identical answer every time.
        self.gate_config: ProactiveGateConfig | None = None
        # Phase 2b-1's in-memory grace-period tracker. Built once, here, and
        # shared across every message for the process's whole life -- the
        # same reasoning as question_detector and embedding_model above.
        # Deliberately NOT persisted anywhere: see aura.proactive.grace for
        # why a restart simply dropping whatever was pending is correct, not
        # a gap.
        self.grace_registry: GraceRegistry = GraceRegistry()

    async def setup_hook(self) -> None:
        """Open the database, load the embedding model, and sync commands -- once.

        Runs here rather than in on_ready because on_ready fires again on
        every gateway reconnect, and schema init, model load, and command
        sync must each happen exactly once per process, before Aura starts
        receiving events -- not be silently repeated (or raced) on a
        reconnect.
        """
        self.db = await aiosqlite.connect(self.settings.database_path)
        await init_schema(self.db)
        # CREATE TABLE IF NOT EXISTS cannot reshape a table that already
        # exists, so a database carried over from Phase 2a-1 would keep its
        # old diagnostic table and fail on every single write. Checked once,
        # loudly, here rather than being discovered one logged exception per
        # message later.
        await verify_signal_schema(self.db)
        logger.info("Database ready at %s", self.settings.database_path)

        # TextEmbedding's constructor does blocking file I/O (reading cached
        # model weights) and builds an ONNX Runtime session -- the same
        # class of blocking work embed_text offloads per call (see
        # aura.embeddings), offloaded here for this one-time load. Stored
        # explicitly on the client, not a module-level singleton, so
        # anything that needs it receives it as an argument -- same pattern
        # as self.db, and the same lesson Phase 1c's modal redesign already
        # applied (constructor argument over reaching into interaction.client
        # from a generic base-class override).
        self.embedding_model = await asyncio.to_thread(
            TextEmbedding, self.settings.embedding_model
        )
        logger.info("Embedding model ready: %s", self.settings.embedding_model)

        # Embeds the question exemplars once, here, for the same reason the
        # model itself loads here: it is one-time work that must be finished
        # before the first message arrives, not repeated per event.
        self.question_detector = await QuestionDetector.create(self.embedding_model)

        # Every threshold logged at startup, on purpose: they are placeholders
        # awaiting recalibration, so the numbers a given deployment is actually
        # running with have to be visible without reading its .env.
        gate_config = ProactiveGateConfig.from_settings(self.settings)
        self.gate_config = gate_config
        proactive_llm = (
            "on" if self.settings.is_llm_configured(ModelComponent.PROACTIVE) else "off"
        )
        logger.info(
            "Proactive gate ready: question>=%.3f, similarity>=%.2f, gap>=%.2f, "
            "cooldown %.0fs/channel, cap %d/guild/UTC-day, grace %.0fs, LLM %s "
            "(posts only in channels enabled via /aura-config)",
            gate_config.question_threshold,
            gate_config.similarity_threshold,
            gate_config.minimum_confidence_gap,
            gate_config.cooldown_seconds,
            gate_config.daily_cap,
            self.settings.proactive_grace_period_seconds,
            proactive_llm,
        )

        register_fact_commands(self.tree)
        register_ask_command(self.tree)
        register_proactive_commands(self.tree)
        register_config_command(self.tree)
        register_supersede_command(self.tree)

        # Global sync; Discord can take up to an hour to propagate new or
        # changed commands globally. Sync to a specific guild instead
        # during active development if commands need to appear immediately.
        await self.tree.sync()

    async def close(self) -> None:
        """Close the database connection before handing off to discord.py's own shutdown."""
        if self.db is not None:
            await self.db.close()
        await super().close()

    async def on_message(self, message: discord.Message) -> None:
        """Run each message through the proactive gate and record what it decided.

        A thin adapter, on purpose: every decision about what to evaluate and
        what to do with a failure lives in aura.proactive.listener, where it
        is testable without a gateway connection.

        discord.py dispatches each event as its own task, so messages
        arriving together are evaluated concurrently rather than queued behind
        one another. That concurrency is exactly why the cooldown and daily cap
        are acquired atomically in SQL rather than checked in Python (see
        aura.db.proactive_state), and the embedding inference underneath is
        offloaded to a worker thread (see aura.embeddings.embed_text), so a
        busy channel never stalls the event loop.

        The pipeline posts an answer only when the message's channel has been
        enabled via /aura-config, an LLM is configured, the full gate passes,
        nobody else answers during Phase 2b-1's grace period, and the LLM's
        own self-assessment agrees -- otherwise it stays silent.
        """
        if (
            self.db is None
            or self.question_detector is None
            or self.embedding_model is None
            or self.gate_config is None
        ):
            # Unreachable in practice -- setup_hook completes before the
            # gateway starts delivering events -- but a None here would
            # otherwise become an AttributeError on every single message.
            logger.warning("Skipping proactive evaluation: startup has not finished")
            return

        await handle_message(
            message,
            db=self.db,
            detector=self.question_detector,
            model=self.embedding_model,
            config=self.gate_config,
            settings=self.settings,
            grace_registry=self.grace_registry,
        )

    async def on_message_delete(self, message: discord.Message) -> None:
        """Stand down any grace period pending on a deleted message.

        Phase 2b-1: a message that no longer exists must never be answered
        into. discord.py's cache only guarantees `message` is populated for
        messages it had cached (see on_raw_message_delete below for the
        uncached case) -- but the only two fields this needs, channel and
        message ID, survive on an uncached partial message too.
        """
        self.grace_registry.notice_message_gone(
            channel_id=message.channel.id, message_id=message.id
        )

    async def on_raw_message_delete(self, payload: discord.RawMessageDeleteEvent) -> None:
        """Stand down a pending grace period for a deleted message discord.py's cache missed.

        on_message_delete only fires for a message discord.py already had
        cached; a message deleted after falling out of that cache (or one
        Aura never saw created, in principle) still needs its grace period
        cancelled if one is pending, and the raw event carries channel and
        message ID regardless of cache state.
        """
        self.grace_registry.notice_message_gone(
            channel_id=payload.channel_id, message_id=payload.message_id
        )

    async def on_message_edit(
        self, _before: discord.Message, after: discord.Message
    ) -> None:
        """Stand down any grace period pending on an edited message.

        Phase 2b-1: an edit may have changed the question into something the
        original Stage 1/2 scores no longer describe, and re-validating
        edited content is out of this phase's scope. Treated the same as a
        deletion -- cancel outright rather than risk answering a question
        that isn't the one that was asked.
        """
        self.grace_registry.notice_message_gone(
            channel_id=after.channel.id, message_id=after.id
        )

    async def on_ready(self) -> None:
        """Log a clear, greppable line once the gateway connection is live."""
        logger.info(
            "Aura is ready: logged in as %s, serving %d guild(s)",
            self.user,
            len(self.guilds),
        )


def build_intents() -> discord.Intents:
    """Build the gateway intents Aura requires.

    Message Content Intent is requested here, but it must ALSO be enabled
    for this bot in the Discord Developer Portal, under
    Bot > Privileged Gateway Intents. Without both, the client fails to
    connect with an intent-related error.
    """
    intents = discord.Intents.default()
    intents.message_content = True
    return intents


def create_client(translator: Translator, settings: Settings) -> AuraClient:
    """Construct the Aura Discord client and register its slash commands."""
    client = AuraClient(intents=build_intents(), settings=settings)

    # Command name/description are only resolved for en-US here. Localizing
    # them too (so e.g. German users see a German command description in
    # their picker) needs discord.py's own app_commands.Translator +
    # locale_str machinery (registered via CommandTree.set_translator),
    # which is a distinct mechanism from the t()-based lookup used for the
    # reply below. Deferred until a phase that exercises it against a real
    # sync cycle. The reply text, which the spec for this phase requires,
    # is fully localized via interaction.locale below.
    @client.tree.command(
        name=translator.t("ping_command_name", DEFAULT_LOCALE),
        description=translator.t("ping_command_description", DEFAULT_LOCALE),
    )
    async def ping(interaction: discord.Interaction) -> None:
        """Reply with a translated pong so users can confirm Aura is responsive."""
        locale = str(interaction.locale)
        await interaction.response.send_message(translator.t("ping_response", locale))

    return client


def main() -> None:
    """Load configuration and translations, then connect Aura to Discord.

    Fails fast with a clear, specific message for the two ways this phase
    can be misconfigured: missing/blank DISCORD_TOKEN and a bad Discord
    token, instead of letting either surface as a cryptic error deep inside
    discord.py.
    """
    configure_logging()

    try:
        settings = load_settings()
    except ConfigurationError as exc:
        logger.critical("Startup aborted: %s", exc)
        sys.exit(1)

    configure_logging(settings.log_level)
    logger.info("Configuration loaded.")

    try:
        translator = get_translator()
    except TranslationLoadError as exc:
        logger.critical("Startup aborted: %s", exc)
        sys.exit(1)

    client = create_client(translator, settings)

    try:
        client.run(settings.discord_token, log_handler=None)
    except discord.LoginFailure:
        logger.critical(
            "Discord rejected the bot token. Check DISCORD_TOKEN in .env — "
            "it may be invalid, revoked, or copied incorrectly."
        )
        sys.exit(1)
    except OutdatedDiagnosticTableError as exc:
        # Raised out of setup_hook, so this is a startup problem with an
        # operator-fixable cause -- reported like the other two rather than as
        # a traceback from inside discord.py's connection code.
        logger.critical("Startup aborted: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
