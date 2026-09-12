# Aura Web Interface (Phase 4b)

Discord OAuth2 login and the list of servers a signed-in user may configure.
Nothing else — no fact editing (4d), no payment (4c). This is the shell the
later sub-phases build on.

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
