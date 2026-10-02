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
- The web interface (`web/`) is a third, separate compose project
  (`aura-web`), reached at `https://aura.timurmanjosov.com` through the host's
  existing Caddy. See "The web interface on the server" below.

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

## Daily limits for `/aura-ask`

`/aura-ask` has its own daily ledger (`ask_calls`, created automatically at
startup like every other table — additive, no migration step, existing rows
untouched). One row is one **paid** answer: a synthesis call plus its
grounding check. A question that matches no fact gets the usual "no
information" reply and writes no row.

| Setting | Default | Meaning |
|---|---|---|
| `ASK_DAILY_CAP_FREE` | 10 | paid answers per guild per UTC day, Free plan |
| `ASK_DAILY_CAP_PRO` | 25 | paid answers per guild per UTC day, Pro plan |
| `ASK_USER_DAILY_CAP_FREE` | 5 | paid answers per member per guild per UTC day, Free plan only |
| `ASK_SYNTHESIS_MAX_OUTPUT_TOKENS` | 700 | output ceiling on every synthesis call (`/aura-ask` **and** proactive relief) |
| `GROUNDING_MAX_OUTPUT_TOKENS` | 300 | output ceiling on every grounding check (both paths) |

All five are optional; a server `.env` without them gets the defaults.

- **Which cap applies** is the plan gate's answer at the moment of the
  question. `BILLING_MODE=disabled` and every complimentary guild count as
  Pro. A plan change mid-day applies to the next question; answers already
  spent today keep counting.
- **When a cap is reached** the member is not refused: they get, visible only
  to them, a note that today's AI answers are used up (and when they come
  back), plus up to three of the facts retrieval found, each linked to its
  source message with its date. No model call, no row, no cost. A cap of `0`
  means every matched question gets this free answer.
- **The operator budget** counts these rows too (estimated at $0.004 each).
  In `hard` mode an over-budget day routes `/aura-ask` to the same free answer
  instead of refusing. `/aura-operator-budget` shows an "Ask" line.
- **Every question is cut to 1,000 characters**, and every fact to 1,000
  characters inside the synthesis prompt. A model response stopped at the
  output ceiling is treated like an unreadable one: `/aura-ask` shows its
  usual error, proactive relief stays silent, a cut-off grounding check fails
  closed.

**Reading real usage.** Every synthesis and grounding call writes one INFO
line with the model and the provider's token counts, never any content:

    docker logs aura-aura-1 2>&1 | grep "LLM usage:"
    # LLM usage: purpose=synthesis model=… prompt_tokens=1156 completion_tokens=98 finish_reason=stop

and every paid or capped question one line with the guild's first four
digits and today's count (`grep "/aura-ask in guild"`). Today's paid answers
per guild, from a backup copy (never the live file):

    sqlite3 -readonly <backup copy> \
      "SELECT substr(guild_id, 1, 4) || '…', COUNT(*) FROM ask_calls
       WHERE call_day = strftime('%Y-%m-%d', 'now') GROUP BY guild_id;"

## Plans and billing (Phase 4c)

Billing ships switched off (`BILLING_MODE=disabled`): a redeploy with this code
changes nothing for any guild until the steps below are taken on purpose.

**Redeploying `web/` after Phase 4c needs its billing values.** The web backend
has no mode without billing: it refuses to start unless
`AURA_WEB_STRIPE_SECRET_KEY`, `AURA_WEB_STRIPE_WEBHOOK_SECRET`,
`AURA_WEB_STRIPE_PRICE_ID`, `AURA_WEB_BOT_INTERNAL_API_URL` and
`AURA_WEB_BOT_INTERNAL_API_SECRET` are all set, and the container crash-loops
with the missing variable named in its log. A `web/` redeploy that only wants
the Phase 4b dashboard still needs steps 1–3 below first.

**The bot and the web backend move together.** They share one internal API
contract, and the Phase 4c audit fixes changed it (a required `on_pro_price`
field in every snapshot, a `payment_pending` standing). Deploy both from the
same commit. While they differ nothing is corrupted — the bot answers every
snapshot it does not understand with `400`, the web backend answers Stripe
with `503`, and Stripe redelivers for up to three days — but no plan changes
until both are current. The bot's own database needs no manual step: two
columns (`on_pro_price`, `past_due_since`) are added to `guild_subscriptions`
at startup, additively and idempotently, before the plan gate reads a row.
As for every schema change, dry-run the new image against a copy of the live
database backup first, twice (see the 2026-08-27 deployment report for the
procedure).

**Rolling back, and forward again.** Rolling back to an image from before these
fixes needs no database change: the old code names its columns explicitly and
simply ignores the two new ones. Rolling *forward* again afterwards needs one
statement first, because the old code never maintains `past_due_since` — an
anchor written before the rollback would survive a payment made during it and
deny the next failed payment its grace. With the bot stopped, before starting
the new image:

    docker compose run --rm --entrypoint python aura -c "import sqlite3; c = sqlite3.connect('data/aura.db'); c.execute('UPDATE guild_subscriptions SET past_due_since = NULL'); c.commit()"

Every `past_due` row then re-anchors at its own period start, exactly the
pre-fix meaning. `on_pro_price` needs nothing: the reconciliation a minute after
the web backend starts rewrites it for every subscription.

1. **Generate one shared secret** and put it in both files:
   `python -c "import secrets; print(secrets.token_urlsafe(48))"` →
   `INTERNAL_API_SECRET=` in `.env`, `AURA_WEB_BOT_INTERNAL_API_SECRET=` in
   `web/.env`.
2. **Bring the bot up first.** Its compose project creates the internal
   `aura-billing` network the web backend joins:
   `docker compose up -d --build`, then look for
   `Internal billing API listening on 0.0.0.0:8081` in `docker compose logs aura`.
   No host port is published; `ss -tlnp | grep 8081` on the host shows nothing.
   That does not make it unreachable from the host: any process on the VPS can
   still connect to the container's own IP on either network. The shared
   secret is what keeps it closed.
3. **Configure Stripe (test mode)** in `web/.env`: a restricted test key, the
   Pro Price ID and the webhook signing secret. Create the restricted key
   **from zero permissions** ("Create restricted key", every resource left at
   *None*) and grant exactly four:

   | Resource | Access | Why |
   |---|---|---|
   | Checkout Sessions | Write | creates the subscription checkout |
   | Subscriptions | Read | every webhook and every reconciliation re-fetches the subscription |
   | Invoices | Read | each of those fetches expands `latest_invoice`; without this Stripe refuses it (403), every webhook answers `503` and no paying guild ever reaches Pro |
   | Customer portal | Write | opens the payer's billing portal |

   Nothing else — in particular no Customers, Refunds or Webhook endpoints,
   and no write access to Subscriptions: the web container holds this key.
   The backend probes the key once at startup (`GET /v1/invoices?limit=1`) and
   logs one `ERROR` naming the missing permission if Invoices (read) is
   absent; an unreachable Stripe at that moment is only a warning. To check the
   whole set against a test key, run `python scripts/verify_stripe_sandbox.py`
   (its step A1 probes each permission without creating anything and refuses
   a key that can create payouts). Create the live key the same way.
   Point a webhook endpoint at
   `https://<your-domain>/api/stripe/webhook` with the events listed in
   `web/.env.example`, and apply the account settings in `web/README.md`
   ("Stripe account settings this code relies on"). Create a customer portal
   configuration with plan switching and quantity changes disabled and put its
   `bpc_…` ID in `AURA_WEB_STRIPE_PORTAL_CONFIGURATION_ID`. Checkout offers
   cards only (wallets such as Apple Pay and Google Pay included) and keeps
   Aura as merchant of record by switching Stripe's Managed Payments off for
   each session; both are set in code, not in the dashboard, so the account's
   "Managed Payments by default" setting does not matter. Under **Billing →
   Revenue recovery → Retries**, set "If all retries for a payment fail" to
   **Cancel the subscription** (see below).
4. **Bring the web interface up:** `docker compose -f web/docker-compose.yml up -d --build`.
   The backend log shows `Stripe billing ready: test mode, price price_…`.
5. **Enforce, with your own servers complimentary.** Put your own servers into
   `BILLING_COMPLIMENTARY_GUILD_IDS` first, then set `BILLING_MODE=enforced`
   and restart the bot (the full deploy discipline: online backup, integrity
   check, row counts, a tagged rollback image). Guilds without a subscription
   or a complimentary entry move to Free: their Pro settings are kept and pick
   up again with Pro. Enforcement comes first because the checkout refuses
   every guild whose plan no subscription decides (`409 nothing_to_buy`):
   with billing disabled, nothing can be bought.
6. **Subscribe a test server** that is not complimentary, from the dashboard
   with a Stripe test card, and check `/aura-plan` in that server: Free
   before, Pro paid through a date after.

Live keys are refused by the web backend unless
`AURA_WEB_STRIPE_ALLOW_LIVE_MODE=true` — a separate decision from everything
above, with its own checklist (Stripe's go-live checklist, tax registration,
the live webhook endpoint's own signing secret).

**The retry setting.** Stripe's "If all retries for a payment fail" has three
possible final actions, and Aura's entitlement is correct under each:

- **Cancel the subscription** (recommended, and the sandbox default: Smart
  Retries, 8 retries within 2 weeks, then cancel) — Stripe sends
  `customer.subscription.deleted`, the subscription is `canceled`, Pro has
  already ended with the 7-day payment grace and stays off.
- **Mark the subscription as unpaid** — status `unpaid`, which grants nothing
  from that moment.
- **Leave the subscription past-due** — Stripe keeps charging each new period.
  The payment grace is anchored at the oldest unpaid period and survives the
  rollover and any write-off, so no period earns a new grace; a written-off
  subscription that Stripe reports `active` again grants nothing until a
  payment is actually seen.

Cancel is recommended because it ends the subscription in Stripe too: nobody
keeps being invoiced for a service they no longer have, and the admin can
subscribe again cleanly.

**Troubleshooting.** `Stripe key self-check: the key lacks the Invoices (read)
permission` in the web log means exactly that: add the permission (step 3); no
restart is needed for syncs to recover, but the check itself only runs at
startup. `Stripe refused a checkout … (HTTP 400: …, param <name>)` names the
parameter Stripe objected to, when Stripe names one; some refusals (the
Managed Payments one among them) carry only their type, and Stripe's
Dashboard request log (Developers → Logs) shows the full message the
backend deliberately never writes. `The bot's billing API refused this service's shared secret`
in the web log means the two secrets differ. Webhooks answered `503` are safe:
Stripe redelivers them, and every redelivery is idempotent. If the web
interface was down for longer than Stripe's retry window, the reconciliation
that runs a minute after startup and every six hours re-syncs every
subscription.

## The web interface on the server

The web stack runs as its own compose project next to the bot and behind the
**Caddy that is already installed on the host** (a systemd service, not a
container) and already serves the portfolio site on ports 80 and 443. Aura's
site is added to that Caddy as one more site block; there is no second proxy,
and that Caddy is never stopped or restarted, only gracefully reloaded.

### What is reachable from where

| Component | Listens on | Reachable from |
|---|---|---|
| Caddy, site `aura.timurmanjosov.com` | host ports 80, 443 | the internet. Port 80 only redirects to HTTPS |
| Frontend (Next.js) | `127.0.0.1:3000` on the host | the host only. Caddy proxies to it |
| Backend (FastAPI) | port 8080 on the `aura-web` network | the frontend container. No host port |
| Bot's internal billing API | port 8081 on `aura-billing` (internal) | the web backend. No host port |

"No host port" is not "unreachable from the host": any process on the VPS can
connect to a container's own address. The backend trusts nothing on that basis
(see the client address below), and the bot's internal API is closed by its
shared secret, not by the network. Caddy's own admin API listens on
`localhost:2019` (its default, predating Aura); any local user can reconfigure
Caddy through it.

`web/docker-compose.yml` publishes the frontend on `127.0.0.1` on purpose:
Docker's published ports bypass the host firewall, so binding every interface
would put plain HTTP on the internet next to Caddy.

### Prerequisites

1. **DNS.** An `A` and an `AAAA` record for `aura.timurmanjosov.com` with the
   same addresses as the main domain. Check from anywhere:
   `getent ahosts aura.timurmanjosov.com` against `getent ahosts timurmanjosov.com`.
   Until the name resolves publicly, Caddy cannot obtain a certificate.
2. **Ports 80 and 443 open** on the host firewall and in Netcup's panel. They
   already are for the portfolio; the certificate challenge needs them.
3. **The bot is up** (it creates the `aura-billing` network the web backend
   joins).
4. **Discord:** under the application's OAuth2 → Redirects, add exactly
   `https://aura.timurmanjosov.com/api/auth/callback`.
5. **Stripe (test mode):** a webhook endpoint (below) and its signing secret.
6. **The `aura-web` subnet is free.** The compose file pins `172.16.86.0/28`.
   List every subnet already in use:
   `for n in $(docker network ls -q); do docker network inspect -f '{{.Name}} {{range .IPAM.Config}}{{.Subnet}} {{end}}' "$n"; done; ip -4 route`.
   Nothing may lie inside `172.16.86.0/28` or contain it. Docker's automatic
   pools start at `172.17.0.0/16`, so it never hands out this range by itself;
   only a hand-made network or a host route can collide. (Checked 2026-10-01:
   the host uses `172.17`–`172.20.0.0/16` and the public `/22`, nothing else.)

### `web/.env` on the server

Copy the laptop's `web/.env` with `scp` (never through git, never printed),
`chmod 600` it, then change exactly these lines (they are not secrets):

    AURA_WEB_OAUTH_REDIRECT_URI=https://aura.timurmanjosov.com/api/auth/callback
    AURA_WEB_POST_LOGIN_REDIRECT_URL=https://aura.timurmanjosov.com/
    AURA_WEB_CHECKOUT_SUCCESS_URL=https://aura.timurmanjosov.com/?checkout=success
    AURA_WEB_CHECKOUT_CANCEL_URL=https://aura.timurmanjosov.com/?checkout=cancelled
    AURA_WEB_BILLING_PORTAL_RETURN_URL=https://aura.timurmanjosov.com/

- `AURA_WEB_POST_LOGIN_REDIRECT_URL` is also what the billing routes compare a
  request's `Origin` against (`WebSettings.frontend_origin`): it must be
  exactly the public origin, `https`, no port.
- `AURA_WEB_SESSION_COOKIE_SECURE` stays unset (on).
- `AURA_WEB_STRIPE_WEBHOOK_SECRET` must be the signing secret of the **server's
  own endpoint**, not the `stripe listen` one from local testing.
- `AURA_WEB_DISCORD_BOT_TOKEN` must equal the bot's `DISCORD_TOKEN` (compare by
  hash, not by eye).
- `AURA_WEB_BOT_INTERNAL_API_URL` stays `http://aura-bot:8081`.
- Do not set `AURA_WEB_TRUSTED_PROXY_ADDRESSES`; the compose file sets it.

### Bringing it up

    cd ~/projects/aura && git pull --ff-only
    docker compose -f web/docker-compose.yml up -d --build

Only the `aura-web` project is touched. Check: both containers healthy with
`RestartCount` 0; the backend log shows `Stripe billing ready: test mode`,
`Stripe key self-check passed` and the `Request limits per client` line;
`ss -ltnp` shows `127.0.0.1:3000` and nothing new on a public address.

### Adding the site to Caddy

`/etc/caddy/Caddyfile` belongs to root, so the steps that change it need
`sudo`. The block to add is `web/deploy/Caddyfile.aura`: one site, no global
options, nothing that changes another site's behaviour.

1. **Back up** (as the normal user):

       TS=$(date -u +%Y%m%d-%H%M%S); B=~/backups/caddy-$TS-pre-aura-web
       mkdir -m 700 -p "$B" && cp -p /etc/caddy/Caddyfile "$B/Caddyfile.before"
       ls -la /etc/caddy > "$B/etc-caddy-listing.txt"

2. **Stage** the combined file, and confirm the existing sites are untouched:

       mkdir -p /tmp/aura-caddy
       { cat /etc/caddy/Caddyfile; printf '\n'; cat ~/projects/aura/web/deploy/Caddyfile.aura; } > /tmp/aura-caddy/Caddyfile.new
       chmod 644 /tmp/aura-caddy/Caddyfile.new

   `caddy adapt` both files and compare the JSON routes of every existing
   hostname: they must be identical except for the path of the configuration
   file itself, which `file_server` hides (and which is the same once the file
   is installed).

3. **Validate as the `caddy` user, install, reload gracefully** (with sudo):

       sudo -u caddy -H caddy validate --config /tmp/aura-caddy/Caddyfile.new --adapter caddyfile \
         && sudo cp -a /etc/caddy/Caddyfile /etc/caddy/Caddyfile.bak-$TS \
         && sudo install -m 644 -o root -g root /tmp/aura-caddy/Caddyfile.new /etc/caddy/Caddyfile \
         && sudo systemctl reload caddy

   Validate as `caddy`, not as root: validation opens the new access log, and
   a log file created by root is one the running Caddy cannot open. A reload
   that fails leaves the previous configuration running.

4. **Verify immediately:** the portfolio answers with exactly the status codes
   and headers it had before (compare against a snapshot taken in step 1);
   then `https://aura.timurmanjosov.com` serves the sign-in page with a valid
   certificate, `http://` redirects to `https://`, the response carries
   `X-Robots-Tag: noindex` and an HSTS header **without** `includeSubDomains`,
   and an unsigned `POST /api/stripe/webhook` is answered `400`.

**Rolling the Caddy change back** (portfolio not exactly as before, or anything
unexpected):

    sudo install -m 644 -o root -g root ~/backups/caddy-<TS>-pre-aura-web/Caddyfile.before /etc/caddy/Caddyfile && sudo systemctl reload caddy

The access log for this site is `/var/log/caddy/aura.timurmanjosov.com.access.log`
(rotated at 10 MiB, ten files, 30 days). It never contains bodies; Caddy redacts
cookies and `Authorization`, and the block additionally drops `Stripe-Signature`
and the OAuth callback's `code` and `state`.

### Registering the Stripe webhook

In the Stripe Dashboard, **in the sandbox** (the sandbox banner is visible):
Developers → Webhooks → add an endpoint at
`https://aura.timurmanjosov.com/api/stripe/webhook` with the 13 events listed in
`web/.env.example`. Reveal its signing secret, type it into the server's
`web/.env` as `AURA_WEB_STRIPE_WEBHOOK_SECRET` (with an editor on the server, so
it never passes through a chat or a terminal history on another machine), then
restart only the backend:

    docker compose -f web/docker-compose.yml up -d --force-recreate backend

### Connecting the bot

The web backend reaches the bot's internal billing API only once the bot has
`INTERNAL_API_SECRET` set, with the same value as
`AURA_WEB_BOT_INTERNAL_API_SECRET`. Adding it is a bot restart: take the full
deploy discipline (online backup of the database, integrity check, row counts,
a tagged rollback image), add the line, `docker compose up -d aura`, and look
for `Internal billing API listening on 0.0.0.0:8081`. `BILLING_MODE` is a
separate decision (see "Plans and billing").

### The client address and rate limiting

The backend limits requests per client in four buckets (sign-in, billing
actions, the Stripe webhook, everything else); the limits and the reasoning
behind each default are in `web/.env.example`, and every one is an
`AURA_WEB_RATE_LIMIT_*` setting. A client over its limit gets `429` with
`Retry-After`, before any request body is read. Only webhook deliveries that
fail signature verification count against the webhook's limit, so Stripe's own
deliveries are never refused by it.

The limit is keyed by the real client address, which the backend takes from
`X-Forwarded-For` **only** when the request comes from the frontend container's
pinned address, and then from the right-hand end. Caddy (with no
`trusted_proxies`) replaces whatever `X-Forwarded-For` a client sends with the
client's real address, and Next.js passes it on unchanged, so a client cannot
choose its own identity. `Forwarded` and `X-Real-IP` are never read. To raise a
limit, set the variable in `web/.env` and recreate the backend.

An IPv4 client is one address. An IPv6 client is its whole `/64`, the block a
single subscriber is normally given, because a client could otherwise rotate
through 2^64 addresses of its own. The domain has an `AAAA` record, so phones
on mobile data often arrive over IPv6. The other side of this choice: many
visitors behind one carrier-grade NAT IPv4 address share one budget.

Page and static-file requests are served by Next.js and are not rate-limited
(the stock Caddy build has no rate-limit module, and rebuilding the host's
Caddy with a plugin would put the other site at risk).

### Sessions stay in memory (decision)

Restarting the backend logs every user out. That is accepted: it costs a
re-login, it has no effect on billing (a webhook or a reconciliation never
depends on a session; verified in the Phase 4c audit, E4), and it means no
Discord token is ever written to disk. Durable sessions are deferred until a
feature needs them.

### Rolling the web stack back

- **Take the site off Caddy:** restore the backed-up Caddyfile and reload (above).
- **Stop the web stack:** `docker compose -f web/docker-compose.yml down`
  (removes only the `aura-web` containers and its own network; the external
  `aura-billing` network and the bot are untouched).
- **Disconnect the bot:** remove the `INTERNAL_API_SECRET` line from `.env` and
  `docker compose up -d aura`; the log then says the internal billing API is
  not started.

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
- **`/aura-ask` says today's AI answers are used up:** a daily cap was reached
  (see "Daily limits for `/aura-ask`"), or the operator budget is in `hard`
  mode and exceeded. `docker logs aura-aura-1 | grep "/aura-ask in guild"`
  shows which cap and the counts.
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
