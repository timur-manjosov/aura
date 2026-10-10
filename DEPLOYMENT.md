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

## How `/aura-ask` finds facts (hybrid retrieval)

`/aura-ask` scores every active fact two ways and hands a fact to the model
when **either** says it fits:

- its embedding similarity reaches `SIMILARITY_THRESHOLD` (unchanged), or
- the question's own words cover it -- inflected forms, compounds and typos in
  long words included ("Mentoriate", "Matheklausuren") -- by at least
  `ASK_LEXICAL_COVERAGE_THRESHOLD`, while its similarity is at least
  `ASK_LEXICAL_SIMILARITY_FLOOR`.

The facts that qualify are ranked by similarity plus
`ASK_LEXICAL_RANKING_WEIGHT` × coverage; the best five go to the model, then
linked facts as before. When nothing qualifies but some facts contain part of
the question, up to three of them are listed verbatim under "possibly related"
-- no model call, no `ask_calls` row, as visible as the plain "no information"
reply it replaces.

| Setting | Default | Meaning |
|---|---|---|
| `ASK_LEXICAL_COVERAGE_THRESHOLD` | 0.5 | share of the question's (rarity-weighted) words a fact must contain; above 0, at most 1 |
| `ASK_LEXICAL_SIMILARITY_FLOOR` | 0.05 | similarity a fact found by its words must still reach (-1 … 1) |
| `ASK_LEXICAL_RANKING_WEIGHT` | 0.5 | how much word coverage adds when facts are ranked (0 … 10) |

All three are optional; a server `.env` without them gets the defaults. They
apply to `/aura-ask` only -- proactive relief and the extraction duplicate
check still use embedding similarity alone, with their own thresholds.

**Data files.** The word matcher ignores function words listed in
`src/aura/retrieval/stopword_lists/<locale>.txt`, one file per supported
locale. They are part of `src/` and so of the image; nothing to configure.

**What to look for in the log.** At startup, once:

    /aura-ask retrieval ready: similarity>=0.40, or word coverage>=0.50 with similarity>=0.05; ranked by similarity + 0.50 x coverage; stopwords for 9 locale(s)

A WARNING `/aura-ask word matching unavailable` instead means the stopword
files could not be read: questions are then answered from similarity alone,
exactly as before hybrid retrieval, and nothing else is affected. Per question,
counts only (no question or fact text, first four guild digits):

    /aura-ask retrieval in guild 1000…: 2 of 11 active fact(s) selected, 2 by words alone, 0 possibly related

**Switching word matching off without a rollback.** Set
`ASK_LEXICAL_SIMILARITY_FLOOR` to the value of `SIMILARITY_THRESHOLD` (0.4)
and `ASK_LEXICAL_RANKING_WEIGHT=0`, then restart the bot
(`docker compose up -d --no-deps aura`): the selection is then exactly the
previous similarity-only one, and the "possibly related" list never appears
(asserted by a test).

**Memory and CPU.** Each guild's word index is built in a worker thread when
its facts change and kept in memory (at most 64 MB and 256 guilds in total,
least recently used dropped first); a question then costs well under a
millisecond of matching. Measured: index build about 50 ms for 2,000 facts,
about 0.4 MB of memory for 2,000 ordinary facts.

## The v2 answer format (P4) -- ships dark

`/aura-ask` can answer in a structured format instead of free text: the model
fills fixed fields (a lead, up to four points that each cite their own facts,
the relation between facts, what was asked but is not recorded), code renders
the "Not recorded: ..." line and the conflict or "unclear" caveat from
templates, a second model checks every displayed statement against exactly the
facts it cites, and the answer is sent as a card (question on top, lead, cited
points, sources with channel and recording date). Every other `/aura-ask`
reply in this format -- no information, possibly related, the daily limit, the
errors -- is a card of the same design.

**It is off by default.** With `ANSWER_FORMAT=legacy` (the default) nothing
changes: every prompt and every message is byte for byte what it was before.
Proactive relief has its own switch, `PROACTIVE_ANSWER_FORMAT`, which stays
`legacy` until the proactive path has had a calibration of its own.

| Setting | Default | Meaning |
|---|---|---|
| `ANSWER_FORMAT` | `legacy` | `/aura-ask`: `legacy` or `v2` |
| `PROACTIVE_ANSWER_FORMAT` | `legacy` | proactive relief: `legacy` or `v2` (not switched by P4) |
| `ANSWER_CARD_STYLE` | `embed` | `embed` (classic embed) or `container` (Components V2) |
| `ANSWER_V2_MODEL` | `SYNTHESIS_MODEL` | the model writing v2 answers |
| `ANSWER_V2_CHECK_MODEL` | `GROUNDING_CHECK_MODEL` | the model checking v2 answers -- never a synthesis model |
| `ANSWER_V2_MAX_OUTPUT_TOKENS` | 1000 | output ceiling of one v2 answer |
| `ANSWER_V2_CHECK_MAX_OUTPUT_TOKENS` | 600 | output ceiling of one v2 check |
| `ANSWER_V2_PROVIDERS` / `ANSWER_V2_CHECK_PROVIDERS` | empty | OpenRouter providers to pin (comma-separated, no fallback) |
| `ANSWER_V2_REASONING` / `ANSWER_V2_CHECK_REASONING` | empty | reasoning level: empty (model default), `off`, `low`, `medium`, `high` |
| `ANSWER_V2_DENY_DATA_COLLECTION` | `false` | use only providers that neither retain nor train on the data |

The route lines describe one model each. `ANSWER_V2_PROVIDERS`,
`ANSWER_V2_REASONING` and the data policy go with `/aura-ask`'s v2 answer only:
proactive relief's v2 path writes with `PROACTIVE_MODEL`, which the providers
pinned for `ANSWER_V2_MODEL` may not serve, so it is sent without them. The
`*_CHECK_*` lines and the data policy go with every v2 check, whichever trigger
asked for it.

`v2` on either switch **requires a checker model** (`ANSWER_V2_CHECK_MODEL` or
`GROUNDING_CHECK_MODEL`): without one the bot refuses to start, rather than send
unchecked answers. The legacy format's "no checker configured, send anyway"
behaviour does not carry over.

**Looking at it before anyone else does.** `/aura-operator-preview` (only for
`OPERATOR_DISCORD_USER_ID`, visible only to the operator) shows seven
hand-written sample cards from invented facts -- a normal answer with three
points, a conflict, an "unclear whether both apply" answer with the "not
recorded" line, the possibly-related reply, the daily-limit reply, an error, a
proactive answer -- once as a classic embed and once as a Components V2
container (option `style`: `both`, `embed`, `container`). No model, no ledger,
no database. It works whatever `ANSWER_FORMAT` says.

**Switching it on** (only after the checker has passed its acceptance; see the
P4 report for the measured model choices):

1. Take a backup as in "Redeploying after a code change".
2. Add the chosen lines to `~/projects/aura/.env`, for example
   `ANSWER_FORMAT=v2`, `ANSWER_V2_CHECK_MODEL=<checker>` and, where the
   report says so, the provider and reasoning lines. Key names only in any
   note or log you keep.
3. Restart the bot only: `docker compose up -d --force-recreate --no-deps aura`.
4. Check the log: `Aura is ready`, no Traceback or CRITICAL; after the first
   questions, one line per answer, counts only:

       /aura-ask v2 answer in guild 1000…: grounded (2 point(s), 3 source(s), answers_question=True)

   `ungrounded` or `check_failed` there means the asker got the honest
   "couldn't verify" notice instead of the answer.

**Switching it off in one step:** remove the added lines (or set
`ANSWER_FORMAT=legacy`) and run step 3 again. No database change is involved
either way; the v2 format writes the same single `ask_calls` row per paid answer
as the legacy one.

## Background functions and message looks (P5) -- ship dark

P5 measured fact extraction, the supersession judge and proactive relief on
new models (private report `reports/p5-background-functions-<date>.md`) and gave
the digest, onboarding, `/aura-plan` and the command confirmations a card look.
**With every new setting at its default the bot sends byte for byte what it
sent before**, with one deliberate exception: the background calls now carry an
output ceiling.

**Output ceilings (active by default).** Fact extraction (live and backfill),
the supersession judge and the two variant calls send `max_tokens` from
`EXTRACTION_MAX_OUTPUT_TOKENS` (4096), `SUPERSESSION_MAX_OUTPUT_TOKENS` (1024),
`VARIANT_MAX_OUTPUT_TOKENS` and `VARIANT_AUDIT_MAX_OUTPUT_TOKENS` (1024 each).
The largest reply measured was a fifth of these; a reply cut off at a ceiling
takes the existing failure path (the batch is skipped, the candidate keeps its
plain hint, no variants), never a half-stored result. Each call also writes one
`LLM usage:` line (purpose `extraction`, `supersession`, `extraction-verify`).

| Setting | Default | Meaning |
|---|---|---|
| `EXTRACTION_PROVIDERS` / `_REASONING` / `_DENY_DATA_COLLECTION` | empty / empty / false | route of `EXTRACTION_MODEL` (live, backfill; the data policy also covers the verification) |
| `SUPERSESSION_PROVIDERS` / `_REASONING` / `_DENY_DATA_COLLECTION` | empty | route of `SUPERSESSION_MODEL` |
| `PROACTIVE_PROVIDERS` / `_REASONING` / `_DENY_DATA_COLLECTION` | empty | route of `PROACTIVE_MODEL`, both answer formats |
| `PROACTIVE_MAX_OUTPUT_TOKENS` | unset (= `ANSWER_V2_MAX_OUTPUT_TOKENS`) | ceiling of proactive relief's v2 answer; a reasoning model needs ~4000 |
| `EXTRACTION_VERIFY_MODEL` | unset (= no verification) | second call that drops candidates the batch does not support; no fallback |
| `EXTRACTION_VERIFY_PROVIDERS` / `_REASONING` / `_MAX_OUTPUT_TOKENS` | empty / empty / 2048 | its route and ceiling |
| `EXTRACTION_VERIFY_MAX_ATTEMPTS` / `_RETRY_DELAY_SECONDS` | 4 / 600 | how often a batch whose extraction or verification CALL failed is tried, and the first pause (doubling); since P5c both calls share the attempts |
| `DIGEST_LOOK`, `ONBOARDING_LOOK`, `PLAN_LOOK`, `NOTICE_LOOK` | `classic` | `card` switches that family to its card look, drawn in `ANSWER_CARD_STYLE` |

**Proactive relief in v2** (`PROACTIVE_ANSWER_FORMAT=v2`) now uses the proactive
variant of the answer contract: the model first says what the message is
(`message_kind`), and only a sincere request may be answered; it is told the
posting date. The v2 check (`ANSWER_V2_CHECK_MODEL`) stays in the loop, and the
checker must be a different vendor from `PROACTIVE_MODEL`.

**A verification call that fails is retried; an unusable reply is not.** When
the call itself does not complete (a timeout, a provider or network error, a
refused key), the batch stays queued -- for backfill, the cursor stays -- and
is tried again after 10, 20 and 40 minutes (defaults), each attempt one
extraction slot. After the last attempt it is given up with an ERROR line
`Giving up a N-message ... batch ...: its verification failed on all 4
attempt(s)`. A reply that arrives but cannot be used (malformed, cut off, a
check missing) skips the batch at once, exactly like a failed extraction: at
temperature 0 a retry would most likely repeat it, and a batch crafted to break
the reply must not cost more than one slot. The counts live in memory; a
restart simply tries a held batch again. Since P5c a failed EXTRACTION call is
held the same way (see the P5c section below). Watch the `Extraction
verification kept N of M candidate(s)` and `The verification call failed ...
(attempt N of 4)` lines after switching it on.

**Looking first.** `/aura-operator-preview` now also shows the six card looks
(digest, onboarding, `/aura-plan` on Free and on Pro, the Pro-only refusal, a
confirmation), whatever the `*_LOOK` settings say.

**Switching one function** (each is its own step, with its own backup):

1. Backup as in "Redeploying after a code change" (online backup API, integrity
   check, `.env` copy mode 600, a rollback tag of the running image).
2. Append only that function's lines to `~/projects/aura/.env` (key names only
   in any note).
3. `docker compose up -d --force-recreate --no-deps aura`; check `Aura is
   ready`, no Traceback/CRITICAL, and the running settings.

**Switching it off in one step:** remove the appended lines and run step 3
again. No database change is involved for any of these settings.

## P5 open items closed (P5c)

P5c changed four things; the private report is `reports/p5c-cleanup-<date>.md`.

**1. The supersession judge's exception for changes limited in time (active
on deploy).** The judge's prompt gains one paragraph between Rule 2 and Rule
3: a change that Fact B's own wording limits to a time that ends ("tonight",
"only this weekend", "until Friday", "temporarily", "during the maintenance")
does not replace the standing fact -- the answer is `complementary` -- unless
the wording makes it lasting, a time word only says when a lasting change
starts, or Fact A is itself about that one occurrence; a hinted limit with no
end ("for now", "until further notice") keeps both facts. Nothing else in that
prompt and no other prompt changed (`scripts/p5c_byte_identity.py`). It takes
effect for whatever `SUPERSESSION_MODEL` is configured the moment the new image
starts. Measured with Haiku it removes every wrong replacement of a temporary
change but escalates a few genuine status changes as `contradiction` (a
moderator then decides); with Gemini 3.8 Flash it made no wrong replacement in
615 judgements. Switching the judge to Gemini, if chosen:

| Key | Value |
|---|---|
| `SUPERSESSION_MODEL` | `openrouter/google/gemini-3.8-flash` |
| `SUPERSESSION_PROVIDERS` | `Google` |
| `SUPERSESSION_REASONING` | `low` |
| `SUPERSESSION_DENY_DATA_COLLECTION` | `true` |
| `SUPERSESSION_MAX_OUTPUT_TOKENS` | `2000` |

Watch the `LLM usage: purpose=supersession` lines (the model named) and the
judgements in `/aura-pending`.

**2. A failed extraction call is held, not dropped (active on deploy).** A
distillation call that does not complete for a reason outside the batch -- a
timeout, a provider or network error, a refused or exhausted key -- now holds
the batch exactly like a failed verification call: the live batch stays
queued, the backfill cursor stays, and it is tried again after 10, 20 and 40
minutes (`EXTRACTION_VERIFY_RETRY_DELAY_SECONDS`, doubling), at most
`EXTRACTION_VERIFY_MAX_ATTEMPTS` attempts for both calls together, each one
extraction slot, then given up with one ERROR line `... its distillation call
failed and the batch failed on all 4 attempt(s) ...`. An unusable reply, and a
request the provider refuses because of its content (HTTP 400, a moderation
refusal, an unknown model), are never retried. The same narrowing now applies
to the verification call (an HTTP 400 there used to be retried).

**3. Proactive relief: a deadline, and no post into a conversation that moved
on.** Measured in P5c: the 30 seconds the answer calls pass to the HTTP client
are a limit per read, and OpenRouter keeps a slow request alive, so the
timeout never bounded a slow DeepSeek answer (one with an 8-second timeout
returned after 22 s). Two guards are active on deploy, in both answer formats:
an answer is not posted when, after the grace period ended, a different member
wrote in the channel or the question was edited or deleted; and it is not
posted later than the answer call's limit + 30 s (the check's deadline) + 15 s
after the grace period (75 s by default). Each withheld answer writes one INFO
line `Proactive answer withheld in channel <id>: the conversation moved on` /
`too late`. One setting, unset by default (= exactly the call of before):

| Key | Default | Meaning |
|---|---|---|
| `PROACTIVE_REQUEST_TIMEOUT_SECONDS` | unset | when set (5-300), the proactive answer call's client timeout AND a hard deadline; an answer not ready in time is silence with a WARNING. `/aura-ask` never reads it. P5c measured DeepSeek with reasoning at p99 36.5 s, max 43.4 s: `60` covers them; the posting limit becomes 105 s. |

**4. An alarm when the LLM key is refused (active on deploy).** When a model
call is refused because the key's spending limit or the account's credits are
exhausted (HTTP 402, or 403 "Key limit exceeded") or the key is invalid (401),
the bot logs one ERROR line `LLM key refused: ...` -- at most once an hour, with
the refused call's purpose and model, never the key or the provider's text --
and `/aura-operator-budget` shows an `LLM key` field: the latest refusal as a
Discord timestamp ("vor 12 Minuten" in your client's language), how many calls
were refused since the bot started, and whether a model call has succeeded
since. That field is English and untranslated, like the rest of the operator
view. State lives in memory; a restart clears it.

**Rolling P5c back:** retag the previous image as described in "Redeploying
after a code change" and recreate the bot; to undo only the supersession model
switch, remove its five lines from `.env` and recreate. No database change is
involved.

## Data obligations (P7a) -- ship dark

P7a builds what Discord's Developer Terms and the privacy groundwork require
before a public listing: encryption at rest, deletion on request, clean-up
after Aura leaves a server, rows kept no longer than needed, an export of the
knowledge base, a privacy summary for members and an "AI-generated" label. The
private report is `reports/p7a-data-obligations-<date>.md`.

**Active on deploy (no switch):**

- Three nullable columns and one table are added at start-up, in place:
  `facts.source_author_id`, `pending_facts.source_author_id`,
  `extraction_queue.author_id`, `extraction_channel_config.privacy_notice_posted_at`
  and `guild_departures`. From now on the author of a fact's source message is
  stored, so a member's deletion request can find it. Rollback: the previous
  image ignores the new columns (no data is changed by them).
- `on_guild_remove` marks a server Aura left; `on_guild_join` clears the mark;
  every `on_ready` reconciles servers left while the bot was offline. A mark
  deletes nothing.
- The purge job runs every `DATA_PURGE_CHECK_INTERVAL_SECONDS` in `report`
  mode: it logs what it WOULD delete (`Purge job (report only): ...`,
  `Purge job retention (report only, nothing deleted): ...`) and deletes
  nothing. It also re-applies the deletion ledger, which is empty until someone
  deletes something.
- `secure_delete` is set on every connection; the WAL is truncated after every
  deletion.
- Logs: no member IDs in the onboarding and backfill lines, no content in any
  model-failure line, and `discord`, `LiteLLM` and `aiosqlite` never log below
  INFO (their DEBUG output carries message text and prompts). The bot
  container's log is bounded to 5 × 10 MB.
- `data/deletion-ledger.db` is created (empty) next to the database.

**The switches (all off by default):**

| Key | Default | Meaning |
|---|---|---|
| `DATABASE_ENCRYPTION_KEY` | unset | 64 hex characters; the database and the ledger are SQLCipher files. A migration, not a toggle (below). |
| `DATA_PURGE_MODE` | `report` | `delete`: purge servers left more than `GUILD_PURGE_GRACE_DAYS` (30) days ago and apply the retention rules. |
| `PROACTIVE_SIGNAL_RETENTION_DAYS` / `ASK_MEMBER_ID_RETENTION_DAYS` / `ONBOARDING_SEND_RETENTION_DAYS` | 90 / 2 / 30 | diagnostic rows with message IDs deleted; the member ID on `/aura-ask` counters replaced by 0 (counts per server stay); welcome records deleted |
| `PRIVACY_INFO_ENABLED` | `false` | `/aura-privacy`, "Privacy: /aura-privacy" under every answer, a line in the welcome message, and a one-time notice in a channel when a moderator switches capture on there. Needs `PRIVACY_POLICY_URL` (https) and `PRIVACY_CONTACT`. |
| `DATA_DELETION_ENABLED` | `false` | the delete button in `/aura-privacy`, `/aura-forget`, `/aura-delete-server-data`, `/aura-operator-privacy`, and the one-time author lookup. Needs `PRIVACY_INFO_ENABLED`. |
| `DATA_EXPORT_ENABLED` | `false` | `/aura-export` (one per server per `EXPORT_COOLDOWN_SECONDS`, 600) |
| `AI_LABEL_ENABLED` | `false` | "🤖 AI-generated" in front of every `/aura-ask` and unprompted answer |

New commands appear only when their switch is on (the command list every
server sees is unchanged with the defaults).

**Backups from now on (both modes).** Keep backup folders at most 30 days:
at every deploy, after the new backup is verified, delete the `aura-*` folders
in `~/backups` older than 30 days (`find ~/backups -maxdepth 1 -name 'aura-*'
-mtime +30` lists them; delete after looking). A deletion therefore leaves the
live database at once and every backup within 30 days. **Never restore
`data/deletion-ledger.db` together with a database backup**: restore
`data/aura.db` only, then start the bot -- the ledger re-applies every
recorded deletion at start-up, bounded to data from before each deletion.

### Gate 3a: encryption at rest

1. Generate the key on the deploying machine:
   `python -c "import secrets; print(secrets.token_hex(32))"`. Put it in the
   password manager FIRST (entry "Aura database key"), then into the local
   `.env` as `DATABASE_ENCRYPTION_KEY=`.
2. Backup as in "Redeploying after a code change", plus a rollback tag.
3. Two dry runs on copies, the bot still running (key passed in the
   environment of a one-off container, never on the command line):
   ```
   cd ~/projects/aura && docker compose exec aura python -m aura.db.maintenance backup data/aura.db data/dry1-plain.db
   docker compose run --rm --no-deps --entrypoint python -e DATABASE_ENCRYPTION_KEY aura -m aura.db.maintenance encrypt data/dry1-plain.db data/dry1-enc.db
   ```
   (export `DATABASE_ENCRYPTION_KEY` in the shell first, from the password
   manager). Repeat as `dry2-*`. Both must print `integrity=ok cipher=ok` and
   the SAME schema hash and row counts as the plaintext copy. Delete the dry
   files afterwards.
4. Stop the bot: `docker compose stop aura`. `ls -la data/` must now show
   no `aura.db-wal` and no `deletion-ledger.db-wal` (a clean close folds the
   WAL into the file); if one is there, start and stop the bot once more.
5. Encrypt both files, keeping the plaintext ones until step 8:
   ```
   docker compose run --rm --no-deps --entrypoint python -e DATABASE_ENCRYPTION_KEY aura -m aura.db.maintenance encrypt data/aura.db data/aura.encrypted.db
   docker compose run --rm --no-deps --entrypoint python -e DATABASE_ENCRYPTION_KEY aura -m aura.db.maintenance encrypt data/deletion-ledger.db data/deletion-ledger.encrypted.db
   mv data/aura.db data/aura.plain.db && mv data/aura.encrypted.db data/aura.db
   mv data/deletion-ledger.db data/deletion-ledger.plain.db && mv data/deletion-ledger.encrypted.db data/deletion-ledger.db
   ```
   (`encrypt` refuses to write over an existing file, and checks that the
   copy has the source's schema and row counts before it reports success.)
6. Add `DATABASE_ENCRYPTION_KEY` to the server's `.env` (mode 600), start:
   `docker compose up -d --no-deps aura`. Check `Database ready at
   data/aura.db (encrypted at rest)` and no CRITICAL.
7. Encrypted backup, read back: `docker compose exec aura python -m
   aura.db.maintenance backup data/aura.db data/check-backup.db` must print
   `integrity=ok cipher=ok` and the expected counts; move it into the backup
   folder.
8. After Timur's "go": delete the plaintext files (`data/aura.plain.db`,
   `data/deletion-ledger.plain.db`) and the older plaintext backup folders and
   archives in `~/backups`.

**Rollback (before step 8):** stop the bot, `mv data/aura.plain.db
data/aura.db` (and the ledger), remove `DATABASE_ENCRYPTION_KEY` from `.env`,
start. After step 8: `python -m aura.db.maintenance decrypt` writes a
plaintext copy with the key.

**From then on, backups go through the container** (the host's `sqlite3`
cannot read the file): `docker compose exec aura python -m aura.db.maintenance
backup data/aura.db data/backup-<TS>.db`, then move the file into the backup
folder; `verify <file>` reads one back. Reads for reports use the same tool
(counts only) or a decrypted copy that is deleted right after.

**Rotating the key:** stop the bot; with `DATABASE_ENCRYPTION_KEY` (old) and
`DATABASE_ENCRYPTION_NEW_KEY` (new) in the one-off container's environment run
`maintenance rekey data/aura.db` and `maintenance rekey
data/deletion-ledger.db`; put the new key in `.env` and the password manager;
start. Old backups keep the old key -- keep it until they have aged out.

**Key loss.** With a wrong or missing key the bot refuses to start (`Startup
aborted: data/aura.db: the configured key does not open this file ...`) and
changes nothing; it never creates a new, empty database in place of the
encrypted one. Then: (1) look for the key in the password manager and in the
local `.env`; (2) if found, put it back and start; (3) if it is lost for good,
the database AND every backup made since the migration are unreadable --
there is no recovery. Stop the bot, move the unreadable files aside (do not
delete them, in case the key turns up), remove `DATABASE_ENCRYPTION_KEY` (or
generate a new one and run the steps above on an empty database), start the
bot with an empty knowledge base and tell the servers. Tested:
`tests/test_database_encryption.py::TestKeyLoss`.

### Gate 3b: the purge job

Read the `Purge job (report only)` lines for at least one cycle: they name the
servers that would be purged and the rows per table, and the retention counts.
Then set `DATA_PURGE_MODE=delete` and recreate the bot. Each purge writes a
`Deletion N (server, left_server): ...` line and a ledger entry. Rollback:
`DATA_PURGE_MODE=report` (what was deleted stays deleted -- restore from the
pre-switch backup only if a purge was wrong, and then expect the ledger to
re-apply it at start; remove the wrong entry from `data/deletion-ledger.db`
first, with the bot stopped).

### Gate 3c: deletion, export, privacy summary, AI label

Set `PRIVACY_POLICY_URL`, `PRIVACY_CONTACT`, `PRIVACY_INFO_ENABLED=true`,
`DATA_DELETION_ENABLED=true`, `DATA_EXPORT_ENABLED=true`,
`AI_LABEL_ENABLED=true`; recreate. Within a minute of the start the author
lookup runs once (`Author lookup: N resolved, ...`); `/aura-operator-privacy
status` shows how many sources are still without an author and `lookup-authors`
runs it again. Rollback: set the four switches back to `false` and recreate.

**When Aura stops for good** (Discord's terms): purge every server
(`/aura-operator-privacy forget-server` for each, or `DATA_PURGE_MODE=delete`
after removing the bot from every server and waiting the period), then delete
`data/` and every backup.

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
