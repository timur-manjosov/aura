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

**Pre-deploy intent check.** Before deploying any change that adds a new
gateway intent, confirm the matching setting is already enabled under Bot >
Privileged Gateway Intents in the Discord Developer Portal — *before*
running the deploy, not after it crash-loops. `build_intents()` in
`src/aura/main.py` currently requests:

| Intent (code) | Portal setting | Required by |
|---|---|---|
| `intents.message_content` | Message Content Intent | reading message text (fact capture, extraction, backfill) |
| `intents.members` | Server Members Intent | `on_member_join` (Phase 3d onboarding) |

Both are privileged: requesting either in code while its portal setting is
off makes Discord refuse the entire gateway connection
(`discord.errors.PrivilegedIntentsRequired`), taking down the whole bot in a
crash-loop under `restart: unless-stopped` — not just the feature that
needed it. This is not a hypothetical; it happened on 2026-08-27 (see
`reports/deployment-2026-08-27.txt`).

## Automatic fact extraction (Phase 3a)

Extraction runs as a sibling to Trigger 2 (proactive relief), not on top of
it — enabling one per channel via `/aura-config` does not enable the other.
It has its own five `.env` values, all documented in full in
`.env.example`; the operational summary an operator needs at deploy time:

- `EXTRACTION_MODEL` — the distillation model. Falls back to
  `SYNTHESIS_MODEL` when unset, same seam as `PROACTIVE_MODEL`.
- `EXTRACTION_BATCH_WINDOW_SECONDS` / `EXTRACTION_BATCH_MAX_MESSAGES` — how
  long candidate messages accumulate per channel, and the hard cap per
  distillation call, before one batch is distilled.
- `EXTRACTION_DAILY_CAP` — the per-guild ceiling on distillation calls per
  UTC day (0 disables automatic extraction entirely), the same kind of spend
  bound `PROACTIVE_DAILY_CAP` is for Trigger 2.
- `EXTRACTION_DEDUP_SIMILARITY_THRESHOLD` — similarity at or above which a
  new candidate is flagged in `/aura-pending` as possibly restating an
  existing active fact. Advisory only — it never blocks, stages, or
  supersedes anything on its own.
- `SUPERSESSION_MODEL` / `SUPERSESSION_DAILY_CAP` (Phase 3a-3) — the second
  paid call in this path and its own independent spend bound. It runs only
  for candidates that cleared the dedup threshold above, and judges what the
  similarity means: replacement, complement, conflict, or coincidence. The
  model falls back to `SYNTHESIS_MODEL` when unset; the cap is per guild per
  UTC day and, when it binds, costs nothing but the judgement — the candidate
  is still staged and still reviewed with the plain similarity hint. `0`
  turns the judgement off entirely and leaves the rest of extraction working.

None of these are new required values — a `.env` that predates Phase 3a
still starts cleanly, with extraction simply never enqueuing anything until
a moderator opts a channel in with `/aura-config`. A database that predates
Phase 3a-3 is migrated in place at startup (two nullable columns added to
`pending_facts`); there is nothing to run by hand and no data to move.

**`/aura-pending`** is the human gate in front of this path: mod-gated on
`manage_guild` (the same permission every other Aura configuration and fact
command uses), it shows the oldest unreviewed extracted candidate one at a
time — the distilled sentence, a permalink to its source message, and an
advisory dedup hint if it may restate an existing fact — with confirm/discard
buttons. Nothing extraction produces becomes a real, citable fact until a
moderator confirms it here; running the command again after resolving one
candidate shows the next. Worth exercising once after a first-time setup, the
same way step 8 above exercises `/aura-ask`: opt a channel in, post a
factual message, wait out `EXTRACTION_BATCH_WINDOW_SECONDS`, then run
`/aura-pending` and confirm the candidate appears.

Where a candidate may restate an existing fact, the review also shows the
Phase 3a-3 judgement and the model's reasoning. **A conflict looks
different on purpose:** the embed turns red, the relationship field carries a
⚠️ marker, and — unlike every other case — it offers no next command to run,
because the two facts disagree on the same detail and nothing in either says
which is current. That is the case to open both source messages before
deciding. The other three judgements are ordinary information: a replacement
suggests running `/aura-supersede` afterwards, a complement says no
supersession is needed, and an unrelated verdict means the similarity was a
false positive. **Aura never acts on any of them by itself.**

## Backfilling a channel's existing history (Phase 3b)

Extraction only ever sees messages written *after* a channel was opted in.
Backfill applies the same chain — first filter, distillation, dedup hint,
supersession proposal — to what was written before that, on a moderator's
explicit request.

```
/aura-backfill start channel:#announcements
/aura-backfill start channel:#announcements since:2025-03-14
/aura-backfill status
/aura-backfill pause  channel:#announcements
/aura-backfill cancel channel:#announcements
```

Mod-gated on `manage_guild`, like every other Aura configuration command, and
it **refuses a channel extraction is not already enabled for** — reading a
channel's whole history is a separate decision from reading its new messages,
and one command must not carry both. `since:` is a UTC date written
`YYYY-MM-DD`; leaving it off reads the whole available history. There is no
`resume` subcommand: `start` on a paused run resumes it exactly where it
stopped, and the reply says which of the two happened.

**Three `.env` values, all documented in full in `.env.example`:**

- `BACKFILL_DAILY_CAP` (default 30) — the per-guild ceiling on backfill
  distillation calls per UTC day, and deliberately **not** a share of
  `EXTRACTION_DAILY_CAP`. Sharing one number would let a backfill of a
  two-year channel consume the whole day's budget for extracting the messages
  members are writing right now, which from the outside is indistinguishable
  from extraction having broken. `0` stops every backfill without touching
  live extraction and without losing anyone's place.
- `BACKFILL_PAGE_PAUSE_SECONDS` (default 1.0) — the wait between two history
  page requests. discord.py already honours Discord's rate limits underneath
  this; the pause is what keeps Aura from having to be told.
- `BACKFILL_CHECK_INTERVAL_SECONDS` (default 30) — how often the worker looks
  for runs when idle. **Not** how fast a run progresses: an active run
  advances batch after batch.

**What to expect operationally.** A run over a large channel takes days, on
purpose. When the daily cap binds, the run keeps its position and resumes
after midnight UTC — `/aura-backfill status` shows today's budget alongside
each run's progress precisely so that "waiting on the cap" and "stalled" are
never the same picture. Progress is reported as a **position**, not a
percentage: Aura cannot know how many messages a channel holds without reading
all of them, so it names the date it has read up to and links the message it
stopped at.

**How long a run takes, and the lever if that is too long.** Measured against a
2,127-message corpus (`reports/phase-3b.txt` Section 7), scaled linearly:

| history | distillation calls | days at cap 30 | total spend |
|---|---|---|---|
| 10,000 messages | ~75 | 3 | ~$0.50 |
| 50,000 messages | ~376 | 13 | ~$2.49 |
| 200,000 messages | ~1,500 | 51 | ~$9.95 |

**The days column is the one to look at, not the spend column.** A backfill's
total cost is fixed by how much history there is; `BACKFILL_DAILY_CAP` only
decides how many days it takes to spend it. For a one-off backfill of a large
channel, raise the cap for the duration and put it back afterwards — the cap's
job is to stop a backfill dominating a day's API usage, not to stop it costing
$10 once. Spend figures are estimates from a token model at live OpenRouter
pricing, not billed figures.

**A restart mid-run costs nothing.** The position is written to the database
after every batch, never held in memory, so a `docker compose up -d --build`
in the middle of a multi-day backfill continues from exactly where it stopped.
No message is skipped, and no message is proposed twice.

**Nothing overlaps with live extraction.** A run's upper bound is the moment
it was started, and messages the live path is already holding — or has already
produced a candidate from — are skipped individually. A candidate a moderator
already confirmed or discarded is never put back in front of them.

**Everything it finds goes to `/aura-pending`**, exactly like live extraction,
and becomes a fact only when a moderator confirms it there. Expect
substantially more candidates than live traffic produces — a year of
announcements at once rather than a trickle — which is worth knowing before
starting a run on a busy channel.

Two states worth recognising in `/aura-backfill status`:

- **Stopped** — the channel could not be read (deleted, or Aura lost *Read
  Message History* there). Fix the permission and start a new run; the old
  one's position is kept as a record but is never resumed.
- **Paused** — a moderator stopped it. Anything it had already found stays in
  `/aura-pending`; pausing means "read no more", never "undo".

A `.env` that predates Phase 3b starts cleanly with all three values
defaulted, and the two new tables are created at startup like every other one.
Nothing runs until a moderator starts a run.

## Periodic digest (Phase 3e)

The fourth trigger: a summary of what changed in the knowledge model, posted
per guild on a cadence a moderator chooses. **It costs nothing to run** — the
digest is assembled from data Aura already has structured (each fact's
sentence, category, timestamp and supersession chain), so there is no LLM call
anywhere in it, no model to configure, and no grounding check needed (nothing
is generated, so nothing can be invented).

One new `.env` value, and it is optional:

- `DIGEST_CHECK_INTERVAL_SECONDS` (default `3600`) — how often the background
  scheduler wakes to ask which guilds are due. **Not** how often a digest is
  posted; that is per-guild state chosen with `/aura-digest`. This value only
  bounds how late a due digest can be.

Nothing else changes for an existing deployment: a `.env` that predates this
phase starts cleanly, and the two new tables (`digest_config`, `digest_runs`)
are created at startup by the usual `CREATE TABLE IF NOT EXISTS` — no
migration, nothing to run by hand, no data to move.

**`/aura-digest`** is the opt-in, mod-gated on `manage_guild` like every other
Aura configuration command. A server with no setting gets no digests at all.

    /aura-digest channel:#announcements interval:Weekly   # turn it on
    /aura-digest interval:Daily                           # change only the cadence
    /aura-digest enabled:False                            # stop; channel/interval kept

Every option is optional and they compose: anything not named keeps its
current value. Cadences offered are daily, weekly, every two weeks and every
30 days. If Aura cannot post in the chosen channel (it needs **Send Messages**
and **Embed Links**), the confirmation says so immediately rather than leaving
the first missing digest to explain it a week later.

Operational behaviour worth knowing before the first one arrives:

- **The first digest covers what changes from the moment it is switched on**,
  not the server's history — it will not repost the existing knowledge model.
  It therefore arrives one interval after setup, not immediately.
- **A period with nothing new posts nothing.** Silence means "nothing changed",
  not "broken". The schedule still advances, so the cadence stays regular.
- **Downtime is caught up exactly once.** A container that was off across a due
  window posts one digest covering everything since the last one when it comes
  back — never one per missed week, and never nothing. The schedule lives in
  the database (`digest_runs`), so a restart resumes mid-interval with no
  recovery step.
- **A failed post is retried, not skipped.** If the channel is gone or Aura
  cannot post, the window stays open and the next hourly check tries again;
  fixing the channel delivers the digest that was missed, in full.
- Re-enabling after a pause does not replay the silent period.

Troubleshooting is in the checklist below.

## Onboarding for new members (Phase 3d)

The third trigger: when a member joins, Aura posts a summary of the currently
active knowledge model to one configured channel — rules and policies first,
then current status, then everything else. Milestones are deliberately
excluded (they are retrospective, not actionable for someone with no
context). **It also costs nothing to run in the base case**, for the same
reason the digest does: the summary is assembled from facts already
structured, so there is no LLM call, no model to configure, and no grounding
check needed.

Requires the **Server Members Intent** enabled in the Discord Developer
Portal *before* deploying this feature (see the README). This is a
privileged intent: if the code requests it and the portal setting is off,
Discord refuses the entire gateway connection
(`discord.errors.PrivilegedIntentsRequired`) — the **whole bot** goes down in
a crash-loop under `restart: unless-stopped`, not just onboarding. This is
not a hypothetical: it took production down on 2026-08-27 (see
`reports/deployment-2026-08-27.txt`). Enable and confirm the intent before
running the deploy, not after it fails.

No new `.env` values are required; two optional ones tune it:

- `ONBOARDING_FACT_LIMIT` (default `15`) — the total number of facts one
  onboarding message may list, spent in priority order across all sections.
- `ONBOARDING_DAILY_CAP` (default `20`) — the per-guild, per-UTC-day ceiling
  on onboarding messages *sent*. Not a spend control (there is no LLM call) —
  a channel-flood control for a raid, a bot pile-on, or an invite spike.

The two new tables (`onboarding_config`, `onboarding_sends`) are created at
startup the same way as every other phase's — no migration, nothing to run
by hand.

**`/aura-onboarding`** is the opt-in, mod-gated on `manage_guild`. A server
with no setting gets no onboarding messages at all.

    /aura-onboarding channel:#welcome    # turn it on
    /aura-onboarding enabled:False       # stop; channel kept for later

Both options are optional and compose, same as `/aura-digest`. If Aura cannot
post in the chosen channel, the confirmation says so immediately.

Operational behaviour worth knowing:

- **Posts to a channel, never a DM.** An unsolicited private message to
  someone who just joined is a stronger interruption than a channel post.
- **A member who leaves and rejoins gets a fresh message.** Deliberate: a
  returning member is exactly as context-free as a new one, and Discord gives
  each join a distinct `joined_at`, which is what the dedup key is built on —
  so this is not the same case as a duplicate delivery of the *same* join
  (which is suppressed).
- **A guild with no eligible active facts yet gets no message.** Same
  deliberately-conservative stance as everywhere else in Aura: no empty
  shell of headings.
- **A failed post is not retried.** Unlike the digest, a join is a one-shot
  event with no periodic sweep behind it — a moderator who fixes a broken
  channel gets it right for every join from then on, but the one that failed
  is not replayed.

Troubleshooting is in the checklist below.

## Linked facts (`/aura-link`)

The fourth knowledge-model component, and the last one to reach production.
**Nothing about deployment changes**: no new `.env` value, no new table, no
migration. A database that predates this already has the `fact_links` table —
it has existed since Phase 1b and simply had no way for a human to write to it.

Two mod-gated commands (`manage_guild`, same as every other moderator tool),
both replying ephemerally:

    /aura-link   fact_a_id:12 fact_b_id:19   # these two belong in one answer
    /aura-unlink fact_a_id:12 fact_b_id:19   # take that back

The IDs are the `#N` values `/aura-facts` shows. Order does not matter — a link
is undirected, and linking the same pair twice tells you nothing changed rather
than creating a second one. There is no confirmation step, deliberately:
unlike `/aura-supersede`, this command has an inverse.

**What it changes at answer time.** When `/aura-ask` or proactive relief finds
a fact by similarity, the facts linked to it are handed to the synthesis model
as *additional candidates*. They are not automatically cited — the model still
decides on relevance, and the grounding check still verifies whatever it did
cite. This is for the case similarity structurally cannot reach: "the
tournament starts Saturday" and "the winner gets a month of Nitro" are one
topic to a member and two unrelated sentences to an embedding model.

**What it deliberately does not change:**

- **Eligibility.** A link never makes proactive relief speak up where it
  otherwise would not. A message that matches no fact still gets silence, and
  the escalation budget is untouched — links widen an answer Aura was already
  going to give, never authorize a new one.
- **Superseded facts.** `/aura-link` refuses a retired fact and names its
  replacement instead. Existing links are never rewritten when a fact is
  superseded: retrieval follows the supersession chain forward, so a link
  drawn months ago delivers whatever is current today, however many times it
  has been replaced since. `/aura-unlink` still works on a retired fact, which
  is the case worth cleaning up.
- **Prompt size, unboundedly.** At most five linked facts join one call, on
  top of the five similarity hits. A hub fact linked to fifty is capped, not
  obeyed, and expansion is one hop only — a chain A–B–C–D contributes B, never
  C and D.

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
  first. Most likely cause is a missing or malformed `.env` value. If the log
  instead shows `discord.errors.PrivilegedIntentsRequired`, this deploy added
  or already contains a request for a privileged intent (Message Content,
  Server Members) that isn't enabled yet under Bot > Privileged Gateway
  Intents in the Discord Developer Portal — see the pre-deploy intent check
  below. This takes down the **whole bot** in a crash-loop, not just the one
  feature that needed the intent; enable the portal setting and redeploy,
  it is not a code bug.
- **`/aura-ask` returns nothing / errors:** check `LLM_API_KEY` and
  `LLM_PROVIDER` in `.env`, and confirm the OpenRouter key is funded.
- **`/aura-pending` always reports nothing to review:** confirm the channel
  was actually opted into extraction via `/aura-config` — a channel enabled
  only for proactive relief never feeds the extraction queue. Also check
  `EXTRACTION_DAILY_CAP` hasn't hit 0 for the day and that `EXTRACTION_MODEL`
  (or its `SYNTHESIS_MODEL` fallback) is configured.
- **`/aura-pending` shows a dedup hint but no judgement:** expected whenever
  `SUPERSESSION_DAILY_CAP` is spent for the day, no supersession model
  resolves, or the call failed — all three degrade to the plain hint by
  design. `docker logs aura-aura-1 | grep -i "judge"` distinguishes them.
- **No digest ever arrives:** in order of likelihood — the guild was never
  opted in (`/aura-digest channel:#…`), the first interval has not elapsed yet
  (it never posts immediately), nothing changed in the period (an empty digest
  is deliberately not posted), or Aura cannot post in the chosen channel.
  `docker logs aura-aura-1 | grep -i digest` distinguishes all four: the
  scheduler logs a line for a skipped-empty window and a warning for an
  unavailable channel.
- **A digest arrived twice:** should be impossible — the window is claimed
  atomically before anything is sent. If it happens, check that only one
  container is live on the token (below); two processes sharing the same
  database file are still safe, but two processes on two *different* databases
  are not.
- **No onboarding message ever arrives, but the bot is otherwise up and
  healthy:** the **Server Members Intent** being disabled is *not* this
  case — see "Container won't start" above, since that failure takes the
  whole bot down instead. With the container confirmed running, in order of
  likelihood: the guild was never opted in
  (`/aura-onboarding channel:#…`), the guild currently has no eligible active
  facts (rules, status changes or other non-milestone facts), or Aura cannot
  post in the chosen channel. `docker logs aura-aura-1 | grep -i onboarding`
  distinguishes the three.
- **An onboarding message arrived twice for the same join:** should be
  impossible — the send is claimed atomically before anything is posted,
  keyed on the member's actual join event. A member who left and rejoined
  getting a second message is expected behaviour, not a bug (see above).
- **A linked fact never shows up in an answer:** expected in two cases and a
  problem in a third. Expected: the synthesis model judged it irrelevant to
  that particular question (it is offered, never forced), or more than five
  linked facts were available and it fell outside the cap. A problem: check
  the link actually exists by running `/aura-link` on the same pair again — it
  reports "already linked" if it does. Note that expansion is one hop, so a
  fact linked to a fact linked to the match is not a candidate, by design.
- **`/aura-link` says a fact is superseded:** that is the command working. The
  message names the replacement; link that ID instead. Aura resolves existing
  links forward automatically, so this only affects links you are creating now.
- **Suspect two instances are live on the same token:** check
  `docker logs aura-aura-1 | grep -i identify` for gateway resume/identify
  conflicts, and confirm no local ThinkPad process is running (see step 6).
- **Resource pressure:** `ssh netcup-vps "docker stats --no-stream"` and
  `free -h` — Aura's CPU-based embedding model adds a few hundred MB of RAM;
  confirm headroom before adding further services to the same box.
