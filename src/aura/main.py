"""Entry point for the Aura Discord bot."""
from __future__ import annotations

import asyncio
import logging
import sys
from contextlib import suppress

import aiosqlite
import discord
from discord import app_commands
from fastembed import TextEmbedding

from aura.commands import (
    register_ask_command,
    register_config_command,
    register_fact_commands,
    register_pending_command,
    register_proactive_commands,
    register_supersede_command,
)
from aura.config import ConfigurationError, ModelComponent, Settings, load_settings
from aura.db import init_schema
from aura.db.pending_facts import verify_pending_facts_schema
from aura.db.proactive_signals import OutdatedDiagnosticTableError, verify_signal_schema
from aura.extraction import (
    create_fact_worthiness_detector,
    handle_extraction_message,
    run_extraction_sweeper,
    withdraw_message,
)
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
        # Phase 3a-2's second contrastive detector, over the fact-worthy /
        # not-fact-worthy exemplar pair instead of the question/statement one.
        # A separate instance rather than a second use of question_detector:
        # they answer different questions and are calibrated against different
        # thresholds (see aura.extraction.fact_worthiness).
        self.fact_worthiness_detector: QuestionDetector | None = None
        # The one background task in the process: it closes extraction batches
        # whose window has expired. Held so close() can stop it before the
        # database connection it uses goes away.
        self.extraction_sweeper: asyncio.Task[None] | None = None
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
        # The same class of problem for Phase 3a-3's two additive columns on
        # pending_facts, and the same one-shot reconciliation -- except purely
        # additive, so it migrates in place instead of asking an operator to
        # decide anything.
        await verify_pending_facts_schema(self.db)
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
            "Proactive gate ready: question>=%.3f, similarity>=%.2f, "
            "cooldown %.0fs/channel, cap %d/guild/UTC-day, grace %.0fs, LLM %s "
            "(posts only in channels enabled via /aura-config)",
            gate_config.question_threshold,
            gate_config.similarity_threshold,
            gate_config.cooldown_seconds,
            gate_config.daily_cap,
            self.settings.proactive_grace_period_seconds,
            proactive_llm,
        )

        # Phase 3a-2: the second detector, built here for exactly the reasons
        # the first one is -- one-time exemplar embedding that must be finished
        # before the first message arrives, not repeated per event.
        self.fact_worthiness_detector = await create_fact_worthiness_detector(
            self.embedding_model
        )
        extraction_llm = (
            "on" if self.settings.is_llm_configured(ModelComponent.EXTRACTION) else "off"
        )
        logger.info(
            "Extraction pipeline ready: fact-worthiness>=%.3f, %.0fs batch window, "
            "max %d message(s)/batch, cap %d call(s)/guild/UTC-day, LLM %s "
            "(reads only channels enabled via /aura-config, and only ever proposes "
            "candidates for /aura-pending review)",
            self.settings.extraction_fact_worthiness_threshold,
            self.settings.extraction_batch_window_seconds,
            self.settings.extraction_batch_max_messages,
            self.settings.extraction_daily_cap,
            extraction_llm,
        )
        # Phase 3a-3's judgment call, logged separately from the pipeline above
        # because it is a separate paid call site with a separate budget, and an
        # operator reading a container log should be able to see that it is on
        # (or off) without inferring it from extraction's line.
        supersession_llm = (
            "on" if self.settings.is_llm_configured(ModelComponent.SUPERSESSION) else "off"
        )
        logger.info(
            "Supersession judgement ready: fires only for candidates scoring >=%.2f "
            "against an existing fact, cap %d call(s)/guild/UTC-day, LLM %s "
            "(proposes only -- /aura-supersede remains the only way a fact is retired)",
            self.settings.extraction_dedup_similarity_threshold,
            self.settings.supersession_daily_cap,
            supersession_llm,
        )
        self.extraction_sweeper = asyncio.create_task(
            run_extraction_sweeper(self.db, self.embedding_model, settings=self.settings)
        )

        register_fact_commands(self.tree)
        register_ask_command(self.tree)
        register_proactive_commands(self.tree)
        register_config_command(self.tree)
        register_supersede_command(self.tree)
        register_pending_command(self.tree)

        # Global sync; Discord can take up to an hour to propagate new or
        # changed commands globally. Sync to a specific guild instead
        # during active development if commands need to appear immediately.
        await self.tree.sync()

    async def close(self) -> None:
        """Stop the sweeper, then close the database, then hand off to discord.py.

        The sweeper is stopped and awaited BEFORE the connection it uses is
        closed. Closing first would let an in-flight sweep hit a closed
        connection and log an exception on the way out of an otherwise clean
        shutdown -- noise that looks exactly like a real fault when read in a
        container log after a restart.
        """
        if self.extraction_sweeper is not None:
            self.extraction_sweeper.cancel()
            # The sweeper only ever ends by cancellation, so suppressing that
            # one exception here is awaiting it, not swallowing a failure.
            with suppress(asyncio.CancelledError):
                await self.extraction_sweeper
            self.extraction_sweeper = None
        if self.db is not None:
            await self.db.close()
        await super().close()

    async def on_message(self, message: discord.Message) -> None:
        """Hand each message to the two passive paths: proactive relief, then extraction.

        A thin adapter, on purpose: every decision about what to evaluate and
        what to do with a failure lives in aura.proactive.listener and
        aura.extraction.pipeline, where each is testable without a gateway
        connection.

        **The two calls are siblings, not a chain.** Trigger 2 and automatic
        extraction both observe the same raw message and neither can see or
        affect what the other decided -- separate channel switches, separate
        thresholds, separate spend ledgers, separate models. Extraction is
        deliberately NOT called from inside handle_message: doing so would
        inherit proactive relief's channel gate for a decision a moderator
        makes independently, which reports/phase-3-pre-analysis.md Section 1c
        identified as a real collision risk rather than a hypothetical one.
        Each call swallows its own exceptions, so neither path can take the
        other down.

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
            or self.fact_worthiness_detector is None
            or self.embedding_model is None
            or self.gate_config is None
        ):
            # Unreachable in practice -- setup_hook completes before the
            # gateway starts delivering events -- but a None here would
            # otherwise become an AttributeError on every single message.
            logger.warning("Skipping message evaluation: startup has not finished")
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

        await handle_extraction_message(
            message,
            db=self.db,
            detector=self.fact_worthiness_detector,
            settings=self.settings,
        )

    async def _notice_message_withdrawn(self, *, channel_id: int, message_id: int) -> None:
        """Tell both passive paths that a message was edited or deleted.

        The two reactions are independent and both matter: proactive relief
        stands down a grace period that was about to answer into a message that
        no longer says what it said (Phase 2b-1), and extraction drops the
        message from its pending batch so nothing is ever distilled from it
        (Phase 3a-2). Neither knows about the other; this method exists so the
        three Discord events that mean "this message changed" each notify both
        without any of them growing a second copy of the same two calls.
        """
        self.grace_registry.notice_message_gone(
            channel_id=channel_id, message_id=message_id
        )
        if self.db is not None:
            await withdraw_message(self.db, channel_id=channel_id, message_id=message_id)

    async def on_message_delete(self, message: discord.Message) -> None:
        """Withdraw a deleted message from both passive paths.

        discord.py's cache only guarantees `message` is populated for messages
        it had cached (see on_raw_message_delete below for the uncached case)
        -- but the only two fields this needs, channel and message ID, survive
        on an uncached partial message too.
        """
        await self._notice_message_withdrawn(
            channel_id=message.channel.id, message_id=message.id
        )

    async def on_raw_message_delete(self, payload: discord.RawMessageDeleteEvent) -> None:
        """Withdraw a deleted message discord.py's cache missed.

        on_message_delete only fires for a message discord.py already had
        cached; a message deleted after falling out of that cache (or one Aura
        never saw created, in principle) still needs its grace period cancelled
        and its queued extraction candidate removed, and the raw event carries
        channel and message ID regardless of cache state.

        Both events firing for the same deletion is harmless and expected: the
        grace cancellation is idempotent, and removing an already-removed queue
        row deletes nothing.
        """
        await self._notice_message_withdrawn(
            channel_id=payload.channel_id, message_id=payload.message_id
        )

    async def on_message_edit(
        self, _before: discord.Message, after: discord.Message
    ) -> None:
        """Treat an edit exactly like a deletion, for both passive paths.

        An edit may have changed the message into something the original scores
        no longer describe -- a question that is no longer that question, or a
        candidate whose text no longer says what cleared the fact-worthiness
        filter. Re-validating edited content is out of scope for both phases,
        and standing down is the conservative direction for each: proactive
        relief risks answering a question nobody asked, and extraction risks
        distilling a claim nobody made.
        """
        await self._notice_message_withdrawn(
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
