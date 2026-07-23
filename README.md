# Aura

Aura is a Discord bot with exactly one function: **it knows the server.**

"Knowing the server" does not mean archiving every message ever written — that's what RAG-over-chat tools already do. It means maintaining a small, continuously distilled knowledge model of what's actually true about the server right now, including the information that something used to be true and no longer is. Aura answers questions from that knowledge model directly, proactively surfaces answers when it's confident enough to do so unprompted, greets new members with a live summary instead of a static welcome message, and posts a periodic digest of what's changed. It has no moderation authority — it never kicks, bans, deletes messages, or manages roles.

This repository is currently at **Phase 0**: the bot skeleton. It connects to Discord and responds to `/ping` with a translated message. The knowledge model (facts, embeddings, database) starts in Phase 1b.

## Running locally

1. Copy the example environment file and fill in real values:

   ```sh
   cp .env.example .env
   ```

   At minimum, set `DISCORD_TOKEN` to a bot token from the [Discord Developer Portal](https://discord.com/developers/applications). Under **Bot > Privileged Gateway Intents**, enable **Message Content Intent** — the bot also enables it in code, but it must be turned on in the portal too, or the bot fails to connect with an intent-related error.

2. Build and start the bot:

   ```sh
   docker compose up --build
   ```

3. Invite the bot to a server and run `/ping`. It replies with a translated "Pong!" based on your Discord client's language, falling back to English for any locale not yet translated.

The SQLite database (from Phase 1b onward) lives in `./data`, which is bind-mounted into the container so it survives rebuilds.

## Development

Run the test suite with:

```sh
pip install -r requirements.txt
pytest
```
