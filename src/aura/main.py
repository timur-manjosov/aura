"""Entry point for the Aura Discord bot."""
from __future__ import annotations

import logging
import sys

import aiosqlite
import discord
from discord import app_commands

from aura.commands import register_fact_commands
from aura.config import ConfigurationError, load_settings
from aura.db import init_schema
from aura.i18n import DEFAULT_LOCALE, TranslationLoadError, Translator, get_translator
from aura.logging_config import configure_logging

logger = logging.getLogger(__name__)


class AuraClient(discord.Client):
    """Minimal Discord client that exposes only slash (application) commands."""

    def __init__(self, *, intents: discord.Intents, database_path: str) -> None:
        super().__init__(intents=intents)
        self.tree = app_commands.CommandTree(self)
        self._database_path = database_path
        self.db: aiosqlite.Connection | None = None

    async def setup_hook(self) -> None:
        """Open the database connection and sync commands, once, before events flow.

        Runs here rather than in on_ready because on_ready fires again on
        every gateway reconnect, and both schema init and command sync must
        happen exactly once per process, before Aura starts receiving
        events -- not be silently repeated (or raced) on a reconnect.
        """
        self.db = await aiosqlite.connect(self._database_path)
        await init_schema(self.db)
        logger.info("Database ready at %s", self._database_path)

        register_fact_commands(self.tree)

        # Global sync; Discord can take up to an hour to propagate new or
        # changed commands globally. Sync to a specific guild instead
        # during active development if commands need to appear immediately.
        await self.tree.sync()

    async def close(self) -> None:
        """Close the database connection before handing off to discord.py's own shutdown."""
        if self.db is not None:
            await self.db.close()
        await super().close()

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


def create_client(translator: Translator, database_path: str) -> AuraClient:
    """Construct the Aura Discord client and register its slash commands."""
    client = AuraClient(intents=build_intents(), database_path=database_path)

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

    client = create_client(translator, settings.database_path)

    try:
        client.run(settings.discord_token, log_handler=None)
    except discord.LoginFailure:
        logger.critical(
            "Discord rejected the bot token. Check DISCORD_TOKEN in .env — "
            "it may be invalid, revoked, or copied incorrectly."
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
