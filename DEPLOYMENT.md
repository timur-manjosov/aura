# Deployment Runbook: Netcup VPS

Aura runs 24/7 on the Netcup VPS in its own directory and container, next to
(but fully isolated from) Epiphyte. This document is the repeatable
procedure for the first deploy and for every redeploy after it. It assumes
no context beyond what's written here.

## Topology

- Host: `netcup-vps` (SSH alias; see `~/.ssh/config` on the deploying machine)
- Directory: `~/projects/aura` — sibling to `~/projects/epiphyte`, not nested
  inside it and not sharing any files with it
- Container: `aura-aura-1`, built from this repo's `Dockerfile`, brought up
  via this repo's `docker-compose.yml`
- Data: `~/projects/aura/data/aura.db` (SQLite), bind-mounted into the
  container at `/app/data` — survives container restarts and rebuilds
- No ports are published. Aura has no HTTP server; it only makes outbound
  connections (Discord gateway, OpenRouter via litellm).
- Both Aura and Epiphyte run under Docker Compose's `restart: unless-stopped`
  policy, independently. Neither shares a network, volume, or process with
  the other.

## First-time setup

These steps assume key-based SSH access from the deploying machine to
`netcup-vps` is already configured (the same access used for Epiphyte and
the portfolio site).

1. **Create the directory** (sibling to Epiphyte's, same ownership):
   ```
   ssh netcup-vps "mkdir -p ~/projects/aura"
   ```

2. **Clone the repo** (public repo, plain HTTPS — matches Epiphyte's
   git-based deployment convention, no credentials needed):
   ```
   ssh netcup-vps "cd ~/projects/aura && git clone https://github.com/timur-manjosov/aura.git ."
   ```
   Confirm `Dockerfile` and `docker-compose.yml` are present and match this
   repo's committed versions (`diff` them against the local copies — they
   should be byte-identical).

3. **Update the local `.env` first, manually, outside of any AI tooling** —
   this is where the real Discord bot token and the funded OpenRouter key
   live. Never commit this file or pass it through git.

4. **Transfer `.env` to the VPS via `scp`** (encrypted, direct copy — never
   via git, never through a logged channel):
   ```
   scp .env netcup-vps:~/projects/aura/.env
   ssh netcup-vps "chmod 600 ~/projects/aura/.env"
   ```
   Verify the transfer with a checksum comparison (`sha256sum` on both
   sides) and confirm `git check-ignore -v .env` reports it ignored on the
   VPS checkout, not just locally.

5. **Confirm the volume mount** in `docker-compose.yml` includes
   `./data:/app/data` — this is what makes the database durable across
   restarts/rebuilds. It's already in this repo's compose file; nothing to
   configure here beyond checking it wasn't accidentally dropped in transit.

6. **Stop any local (ThinkPad) instance completely** before bringing the VPS
   one up — two processes must never hold the same Discord token at once.
   Confirm with:
   ```
   docker ps --filter name=aura
   pgrep -af "python.*aura.main"
   ```
   Both should return nothing.

7. **Bring the container up:**
   ```
   ssh netcup-vps "cd ~/projects/aura && docker compose up -d --build"
   ```

8. **Verify:**
   ```
   ssh netcup-vps "docker ps --filter name=aura"
   ssh netcup-vps "docker logs aura-aura-1 --tail 40"
   ```
   Look for `Aura is ready: logged in as ...` with no errors/tracebacks
   above it. Then, in Discord: right-click a message → **Apps** → **"Add as
   Aura Fact"**, submit a test fact, and run `/aura-ask` to confirm it's
   retrieved and cited correctly from the live VPS instance.

9. **Confirm Epiphyte is unaffected** — check its container status, recent
   logs, and `data/` directory before and after the Aura deploy. It should
   show no restarts, no errors, and an unchanged data directory.

## Redeploying after a code change

```
ssh netcup-vps "cd ~/projects/aura && git pull && docker compose up -d --build"
```

The `data/` volume is untouched by this — the database persists across
rebuilds. No `.env` changes are needed unless the change itself requires a
new variable (in which case, repeat the `scp` step above with the updated
file).

## Restart policy: what `unless-stopped` actually guarantees

Both Aura and Epiphyte use `restart: unless-stopped`. This **does**
auto-recover, unattended, from:
- The containerized process crashing on its own (unhandled exception, OOM)
- A Docker daemon restart or VPS reboot

This **does not** auto-recover from an operator-issued `docker kill` or
`docker stop` — Docker treats that as explicit intent and will not restart
the container again until a manual `docker start` (or
`docker compose up -d`). This is standard, documented Docker behavior, not
a gap in this setup — verified directly: killing `aura-aura-1` via
`docker kill` left it in `Exited (137)` with `RestartCount=0` until manually
restarted, while the data in `data/aura.db` was confirmed intact throughout.
If you deliberately stop the container, you must deliberately start it
again.

## Troubleshooting checklist

- **Container won't start / exits immediately:** `docker logs aura-aura-1`
  first. Most likely cause is a missing or malformed `.env` value.
- **`/aura-ask` returns nothing / errors:** check `LLM_API_KEY` and
  `LLM_PROVIDER` in `.env`, and confirm the OpenRouter key is funded.
- **Suspect two instances are live on the same token:** check
  `docker logs aura-aura-1 | grep -i identify` for gateway resume/identify
  conflicts, and confirm no local ThinkPad process is running (see step 6).
- **Resource pressure:** `ssh netcup-vps "docker stats --no-stream"` and
  `free -h` — Aura's CPU-based embedding model adds a few hundred MB of RAM;
  confirm headroom before adding further services to the same box.
