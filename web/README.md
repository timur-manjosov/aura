# Aura Web Interface (Phases 4b and 4c)

Discord OAuth2 login, the list of servers a signed-in user may configure, and
— since Phase 4c — each server's plan: a Stripe-hosted checkout for Pro, the
Stripe billing portal for the person who paid, and the webhook that keeps the
bot's view of every subscription current. No fact editing yet (4d).

```
web/
  backend/    FastAPI: the OAuth2 flow, sessions, the guild filter
  frontend/   Next.js: a login button and a list
  docker-compose.yml   its own compose project, isolated from the bot
```

## Running it

1. Create a Discord application at <https://discord.com/developers/applications>.
   Under **OAuth2 → Redirects**, add `http://localhost:3000/api/auth/callback`
   exactly as written — Discord compares it byte for byte on both legs of the
   flow.
2. `cp web/.env.example web/.env` and fill in the three credentials.
3. `docker compose -f web/docker-compose.yml up -d --build`
4. Open <http://localhost:3000>.

For development without Docker:

```bash
# backend
PYTHONPATH=web/backend python -m aura_web            # reads web/.env
# frontend (next dev reads this at start-up; a production build bakes it in)
cd web/frontend && AURA_WEB_BACKEND_ORIGIN=http://127.0.0.1:8080 npm run dev
```

Tests are part of the repository's single suite: `pytest` from the root runs
the bot's tests and this service's together. The live OAuth2 round trip is
`python scripts/verify_oauth_flow.py`.

## Why the bot token, and not the database

The brief for this sub-phase asked whether the web backend needs to read
`data/aura.db` at all to know which guilds Aura is running on, and named two
candidate answers: a read-only connection against the same file, or asking
Discord with the bot's own token. **This service asks Discord, and never opens
the database.**

The deciding argument is not convenience, it is correctness: **Aura's database
has no record of which guilds it is in.** Every `guild_id` in `schema.sql` is a
side effect of activity — a fact was extracted, a channel was configured, a
digest ran. So deriving membership from it would be wrong in both directions
at once:

- a guild that invited Aura five minutes ago has no rows anywhere, and would
  be reported as "Aura is not here" while Aura sits in it;
- a guild that removed Aura last month keeps all of its rows forever (facts
  are never deleted, only superseded — that is the knowledge model's whole
  point), and would be reported as a live installation.

Adding a membership table would fix that, but it would mean the bot writing
one more thing and the web service reading it — a second source of truth for
something Discord already answers authoritatively, kept in sync by code that
can drift. Discord *is* the register of who is in which guild.

Choosing this also sidesteps, rather than solves, the SQLite question
`reports/phase-4a-multitenancy-audit.txt` Section 5 raises. That audit found
the load-bearing constraint is not row count but **write concurrency**: every
writer in the bot funnels through one connection-level lock, and a second
process against the same file adds WAL reader/writer interplay to a design
that has never needed it. A read-only connection would probably have been
fine. "Probably fine" is exactly the shape of thing CLAUDE.md's No Grey Areas
principle says not to ship, and here it was avoidable for free.

### What this costs, stated plainly

**The web container holds the bot token.** That is a real expansion of blast
radius and the honest downside of this choice: the token is full bot
authority, not a read-only scope, and Discord has no narrower credential for
"list my guilds". Compromise of the web container is therefore compromise of
the bot's Discord identity.

What bounds it:

- the token is used at exactly one call site
  (`DiscordClient.fetch_bot_guild_ids`), for one read-only endpoint, and is
  never accepted from or echoed to a request;
- the backend publishes no host port at all — it is reachable only from the
  frontend container over an internal network;
- the container holds no database, no volume and no filesystem state; a
  compromise of it cannot reach Aura's facts.

What does **not** bound it, and should be said rather than implied: nothing
here stops an attacker who obtains that token from using it against Discord
directly. If that trade stops being acceptable — most plausibly when this runs
somewhere less controlled than one VPS — the alternative is the membership
table above, accepting its staleness and its second writer. That decision is
recorded here so 4c/4d inherit the reasoning rather than the conclusion.

When **4d** needs real database access (it will — editing facts is the point),
the read-only-connection option comes back on the table on its own merits.
This sub-phase not needing it is not an argument that it never will.

## Why one origin, and what it buys

The frontend proxies `/api/*` to the backend (`next.config.mjs` → `rewrites`).
The browser therefore only ever talks to one origin, which is what lets the
session cookie stay `SameSite=Lax`.

The alternative — frontend and backend on separate origins — forces
`SameSite=None` on the session cookie so that cross-origin `fetch` carries it,
and `SameSite=None` is precisely the setting that lets any third-party page
make authenticated requests on the user's behalf. It would then need CORS with
`allow-credentials` and an exact origin allow-list to claw the protection
back. The proxy avoids needing the protection in the first place. There is
also no CORS middleware anywhere in this service, and that is deliberate: no
cross-origin request is made, so none has to be allowed.

One consequence, found by running it rather than by reading the docs: Next
serialises `rewrites()` into the build output, so `AURA_WEB_BACKEND_ORIGIN`
must be set at **build** time for a production build. The Dockerfile takes it
as a build `ARG` and `web/docker-compose.yml` passes the internal service
address. `next dev` reads it at start-up as you would expect.

## Security decisions in one place

| Decision | Where | Why |
|---|---|---|
| `state` stored server-side, single-use, TTL | `sessions.OAuthStateStore` | Proves this service issued it and it has not been replayed |
| `state` **also** mirrored in an httpOnly cookie | `routes/auth.py` | The store alone cannot tell whose login it was — an attacker's own valid state would otherwise bind a victim's browser to the attacker's account |
| Tokens never leave the process | `sessions.Session` | The browser gets an opaque identifier; there is no token in any response, verified on the wire |
| Session cookie: httpOnly, Secure, SameSite=Lax, Path=/ | `routes/auth.py::_set_cookie` | One place sets every cookie, so the flags cannot diverge per route |
| Store keys are SHA-256 digests | `sessions._digest` | A dump of the store is not a set of replayable credentials |
| Both in-memory stores are bounded | `WebSettings.max_*` | `/api/auth/login` is unauthenticated; an unbounded store is a memory lever |
| Post-login destination is config-only | `WebSettings.post_login_redirect_url` | A caller-chosen redirect target is the textbook open redirect in an OAuth callback |
| Logout is POST-only | `routes/auth.py::logout` | A GET logout is CSRF-able from any page with an `<img>` tag |
| `Cache-Control: no-store` on every response | `app.SecurityHeadersMiddleware` | `/api/guilds` is a per-session answer behind a cookie; a shared cache would hand it to the next person |
| Redirects are not followed | `app.create_app` | Following one would forward a bearer token to wherever it pointed |
| Guild list fails closed on a Discord outage | `routes/dashboard.py`, `bot_guilds.py` | Without a trustworthy membership list the filter cannot be applied; answering anyway would list servers Aura is not in |
| Backend returns error **codes**, never prose | `errors.ErrorCode` | CLAUDE.md forbids user-facing text in logic; the frontend translates through the same key mechanism the bot uses |

## Known limits of this sub-phase

- **Sessions live in memory.** Restarting the backend logs everyone out. In
  exchange, no Discord token is ever written to disk and this sub-phase adds
  no table. 4c/4d will need durable sessions; the two stores are behind small
  interfaces for that reason.
- **The session store's ceiling can evict a live session.** With `max_sessions`
  reached, the oldest login is dropped. That is recoverable (log in again);
  refusing all new logins would not be.
- **No rate limiting of its own.** `/api/auth/login` is unauthenticated and
  cheap, but it is not free. The store bounds memory; it does not bound
  request rate. A reverse proxy in front is the right place for that, and
  there is none yet in local development.

## Plans and billing (Phase 4c)

Two plans, Free and Pro. Free is permanent: `/aura-ask` and managing facts by
hand. Pro adds the automatic triggers — proactive relief, automatic fact
extraction, the digest, onboarding and backfill. Losing Pro moves a guild to
Free; it never switches the bot off.

### Where subscription state lives, and why

Unlike guild membership, a subscription is a fact Discord cannot answer, so it
has to be stored somewhere, and the bot has to read it on every message. Two
designs were considered:

- **(a)** this service writes a dedicated table in `data/aura.db` directly;
- **(b)** the bot stays the only writer of its database and exposes a small,
  internal, shared-secret API that this service calls with what Stripe says.

**This project uses (b).** The deciding arguments:

1. **The internet-facing container stays away from the data.** The web backend
   is the process that receives unauthenticated webhook requests from the
   internet. Under (a) it would need `../data` mounted — the whole knowledge
   base of every guild — to write one table. Under (b) it still has no volume
   and no database handle: compromising it does not reach a single fact.
2. **One writer, one lock.** Every writer in the bot goes through one
   connection and one lock (`reports/phase-4a-multitenancy-audit.txt`,
   Section 5). A second process writing the same SQLite file brings WAL
   locking across containers, a second user ID owning `-wal`/`-shm` files, and
   `database is locked` errors surfacing in bot code that has never had to
   expect them — "very likely fine" in exactly the way Phase 4b chose not to
   ship.
3. **One owner for the schema.** Under (a) this service would have to know a
   table layout owned by `src/aura/db/schema.sql` without importing it.
4. **Idempotency where the state is.** The bot records each Stripe event ID in
   the same transaction as the snapshot it caused. (a) could do that too, but
   (b) keeps it in the process that also decides the plan.

What (b) costs, stated plainly: the bot process gains a listening socket (bound
only on the internal `aura-billing` network, no host port, authenticated
before anything is read), and a second network hop exists that can fail. A
failed hop answers Stripe with a 503, and Stripe redelivers — for up to three
days in live mode. Beyond that horizon the periodic reconciliation below heals
it.

### How a plan changes

Every handled webhook — and every reconciliation — runs the same three steps
(`aura_web.billing_sync.sync_subscription`):

1. ask the bot whether this event was already applied, and which **version**
   of the subscription it holds;
2. fetch the subscription from Stripe **now** — the event's own copy is never
   used, because Stripe does not guarantee event order and its timestamps are
   whole seconds that distinct events share;
3. ask the bot to store that snapshot **only if the version is still the one
   read in step 1** (compare-and-swap).

A stored snapshot therefore always comes from the most recent fetch, however
events are ordered or interleaved, and a delayed write that lands late carries
a stale version and is refused. No clock is compared anywhere.

Every six hours (`AURA_WEB_STRIPE_RECONCILE_INTERVAL_SECONDS`) every Aura
subscription is re-synced through the same path, which recovers events Stripe
stopped retrying while something was down.

### When the bot decides a guild is Pro

In `aura.billing.entitlement`, tested to the microsecond. Every boundary
leans towards the paying guild by a fixed amount and no further.

| Stripe state | Pro until |
|---|---|
| `active` / `trialing` | period end **+ 72 h** (`BILLING_RENEWAL_GRACE_HOURS`) — the delay a renewal confirmation can have while Stripe is still retrying it. Shown as "paid through" only while the period's invoice is `paid`; before that (a renewal between draft and charge, a payment still processing) it is shown as **payment pending**, with the same access |
| … set to cancel at period end, or with a cancel date | exactly that date, no grace |
| `past_due` (a renewal payment failed) | start of the **oldest period still unpaid** **+ 7 days** (`BILLING_PAYMENT_GRACE_DAYS`), and never past a cancel date. The start is recorded the first time the bot stores the subscription `past_due` and kept until a **paid** period is seen: later failed retries, a late webhook, Stripe rolling the still-unpaid subscription into its next period, and a written-off invoice (which Stripe answers with `active`) all leave it where it is. One grace per lapsed payment, not one per period |
| `unpaid`, `canceled`, `incomplete`, `incomplete_expired`, `paused` | Free immediately |
| collection paused, or the invoice billing the current period (its first or a renewal invoice) `void` / `uncollectible` | Free immediately (a failed bank debit voids the invoice while the subscription still reads `active`); a voided proration or one-off invoice does not count |
| any item not on `AURA_WEB_STRIPE_PRICE_ID`, or at quantity 0 | Free immediately, whatever the status: the subscription is pushed to the bot as granting nothing (`on_pro_price=false`) and a warning is logged |

The payment-grace start is only as complete as what the bot has seen. A
subscription first seen `past_due` in its second unpaid period is anchored at
that period, and a payment made and missed entirely before the next failure
does not reset it. Either needs the web backend to miss every webhook retry and
every six-hourly reconciliation for a whole billing period.

When the status cannot be established — the web backend or the bot down, events
undelivered — the last stored snapshot keeps applying, and its own bound above
still ends it. The bot never reads subscription state from anywhere but its
own process memory, which is updated in the same step as each commit, so "the
subscription store was unreachable" is not a state a Pro trigger can observe.

### Security decisions

| Decision | Where | Why |
|---|---|---|
| Stripe-hosted Checkout and billing portal only | `stripe_api.py` | No card data ever touches this infrastructure; out of PCI-DSS scope |
| Guild checked server-side against a fresh Discord answer before any Stripe call | `routes/billing.py` | A guild ID in a request body is a request, not a fact |
| Guild and payer written into the **subscription's** metadata server-side | `StripeClient.create_checkout_session` | Every later event is resolved through data the customer cannot change |
| Webhook signature verified before the body is parsed, over the exact bytes, any failure refused | `routes/stripe_webhook.py` | The one control that stands between the internet and a plan change |
| Exactly one `Stripe-Signature` header, 1 MiB body ceiling, 5-minute timestamp tolerance | `routes/stripe_webhook.py` | Header pollution, memory levers and replay |
| Event ID recorded in the same transaction as the state | `aura.db.subscriptions` | Redelivery can neither grant nor revoke twice |
| Compare-and-swap on a per-subscription version | `aura.db.subscriptions`, `billing_sync.py` | Out-of-order and concurrent events converge to Stripe's current state |
| POST billing routes require `application/json`, a matching `Origin` and no `Sec-Fetch-Site: cross-site` | `routes/billing.py` | A third-party page cannot start a checkout or open a portal with the user's cookie |
| Billing portal only for the Discord user who paid | `routes/billing.py` | Another admin of the same server must not see the payer's billing details |
| The browser sees a plan's standing, never customer or subscription IDs | `routes/billing.browser_plan` | Nothing a page can leak |
| Every Stripe URL handed to a browser checked against Stripe's hosts, in the backend and again in the page | `stripe_api.py`, `lib/api.ts` | A misconfigured API base cannot become an open redirect |
| Live keys refused unless `AURA_WEB_STRIPE_ALLOW_LIVE_MODE=true`; live events refused in test mode | `config.py`, `routes/stripe_webhook.py` | Going live is a separate, deliberate step |
| `hide_input_in_errors` on both settings classes | `config.py`, `aura.config` | A refused setting never echoes a secret into a log |
| Every credential typed `SecretStr`, unwrapped only where a request is authenticated | `config.py`, `aura.config`, the three clients | `repr()`/`str()` of the settings (or anything holding them) prints `**********`, never a secret |
| Every subscription item checked against the configured Pro price, quantity ≥ 1 | `stripe_api.parse_subscription` | A portal plan switch or a hand-made subscription carrying Aura's metadata does not buy Pro |
| Checkout offers cards only | `StripeClient.create_checkout_session` | No delayed payment method can grant Pro before money arrives |
| Internal API: shared secret checked before routing, strict schemas, duplicate JSON keys refused; no listener can be built around a secret shorter than 32 characters | `aura.billing.internal_api` | It is the only way a plan changes |

### Stripe account settings this code relies on

- **Customer portal:** create a portal configuration that allows cancellation
  and payment-method updates and does **not** allow switching products or
  prices or changing the quantity, and set its `bpc_…` ID as
  `AURA_WEB_STRIPE_PORTAL_CONFIGURATION_ID`. Every portal session is then
  opened with it, so the portal's powers are pinned by this deployment rather
  than by the account's default configuration. Optional, deliberately: the
  code already refuses to count a subscription that is not on the Pro price at
  quantity ≥ 1, so a permissive portal can only cost the customer (a switch
  loses Pro), never the operator.
- **Payment methods:** nothing to configure. Checkout sends
  `payment_method_types[]=card` itself, so a bank debit enabled in the
  dashboard is never offered: it would make a subscription `active` days
  before its first payment settles.
- **Restricted key:** created from zero permissions with exactly Checkout
  Sessions (write), Subscriptions (read), Invoices (read) and Customer portal
  (write) — `web/.env.example` gives the reason for each. Invoices (read) is
  the one that is easy to miss: every sync expands `latest_invoice`, and
  without it every webhook answers `503`. The backend probes for it once at
  startup and logs one `ERROR` if it is missing.
- **Revenue recovery:** enable failed-payment emails to customers (the payer's
  half of the grace-period communication; `/aura-plan` and this dashboard are
  the admins' half) and choose what happens after the last retry
  (canceled or unpaid both end Pro immediately).
- **Webhook endpoint:** the event list in `web/.env.example`, pointed at
  `/api/stripe/webhook`. A reverse proxy in front must forward that path with
  the body byte-for-byte unchanged, or every signature fails.
- **Tax:** not configured by this code. Before charging real customers in the
  US or EU, set up Stripe Tax and a tax registration; enabling `automatic_tax`
  without a registration silently collects nothing.

### Known limits of Phase 4c

- **Two different admins paying for the same server within minutes** can
  create two subscriptions: the "already subscribed" check sees the first one
  only once its webhook has arrived. It is detected, not prevented — the bot
  logs a warning, and both `/aura-plan` and the dashboard show the count so
  the extra payer can cancel. Double clicks and second tabs of the same
  person reuse one Checkout Session (a 10-minute Stripe idempotency key).
- **A payer who has left the server** can still open their billing portal
  through the API, but the dashboard only lists servers they manage, so they
  have no button for it. Stripe's own receipt emails link to the portal.
- **`BILLING_MODE` defaults to `disabled`** in the bot. Turning it on is the
  operator's step (see DEPLOYMENT.md), as is switching to live Stripe keys.
- **"Upgrade to Pro" is offered only where it changes something**: a guild
  whose plan is decided by subscription and that has none granting yet. With
  `BILLING_MODE=disabled`, or on a complimentary guild, the button is not shown
  — which also means nobody can subscribe ahead of enforcement from the
  dashboard. The checkout route itself does not refuse those guilds: only a
  hand-crafted request reaches it, and it charges its own sender for a
  subscription that buys nothing extra.
