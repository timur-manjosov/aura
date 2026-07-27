# CLAUDE.md

This file is read automatically at the start of every session. It defines the non-negotiable principles for this project. Every implementation decision should be checked against it.

## Project Overview

Aura is a Discord bot with exactly one function: **it knows the server.**

"Knowing the server" does not mean archiving every message ever written — that's what RAG-over-chat tools already do. It means maintaining a small, continuously distilled knowledge model of what's actually true about the server right now, including the information that something used to be true and no longer is. This knowledge model has exactly four components, and nothing else belongs in it:

- **Fact** — one distilled sentence, not a duplicated copy of the raw message, plus a reference to its origin via Discord channel ID + message ID. Discord permalinks make this reference resolvable, so the original text never has to be stored twice.
- **Timestamp** — when the fact was created or last changed.
- **Status** — active or superseded, chained to its successor fact. Old facts are never deleted, only marked superseded, so the history of what used to be true stays intact.
- **Link** — thematically related facts, even ones spread across time and different channels, are pulled together into one synthesized answer with multiple source citations, instead of being returned as isolated fragments.

Everything Aura does is **one mechanism with four different triggers**, all operating on this same knowledge model — there is no fifth function, and no function that works any other way:

1. **Direct query** — someone asks Aura directly; it answers from the current knowledge state with source references.
2. **Proactive relief** — the same logic, triggered unprompted when Aura can answer a repeat question with high confidence. Deliberately conservative, to avoid unwanted interruptions.
3. **Onboarding** — a new member receives the current knowledge state as a summary instead of a static welcome message.
4. **Periodic digest** — what has changed since the last update, based on facts' timestamps.

If a proposed feature is not one of these four triggers acting on the four-part knowledge model above, it doesn't belong in Aura — see Explicit Non-Goals below.

Aura has **no moderation authority.** It never kicks, bans, deletes messages, or manages roles. It is advisory only. This is a deliberate boundary, not a missing feature — it keeps Aura cleanly separated from automod tooling.

## Explicit Non-Goals

Do not add, even if it seems like a natural extension:
- Automod / moderation actions of any kind
- Economy or leveling systems
- Music playback
- General-purpose chit-chat / generic chatbot behavior
- Any feature that doesn't derive directly from the knowledge model (fact, timestamp, status/supersession, link)

## Tech Stack

- `discord.py` — Discord API, bot framework, slash commands
- `aiosqlite` — async SQLite access (never block the event loop with sync `sqlite3` calls)
- `pydantic` — data models for Fact, Status, Link
- `fastembed` — local CPU embedding model for semantic similarity
- `numpy` — cosine similarity over embeddings
- `litellm` — provider-agnostic LLM API access (BYO-key principle, no single provider hardcoded)
- `APScheduler` — scheduled digest generation
- `pytest`, `pytest-asyncio` — testing

## LLM Usage & Model Selection

Aura calls a large language model at several distinct points in the pipeline — fact extraction and supersession detection, answer synthesis from multiple linked facts, and later digest formatting. These are genuinely different tasks with different requirements, not one generic "the LLM does language stuff" job.

Because access goes through `litellm` (with OpenRouter as the planned provider, giving access to hundreds of models across dozens of providers), **never hardcode one model as "the" model for the whole project.** For each distinct LLM-calling component, reason explicitly about what that specific task needs before picking a model, and note the reasoning briefly in a comment near the call site:

- **Reasoning depth** — supersession detection ("does this new message contradict or update an existing fact?") needs real judgment; formatting an already-extracted digest does not.
- **Structured output reliability** — does the task require strict, parseable JSON (fact extraction) or is free-form prose fine (synthesis)? Favor models with strong native structured-output/tool-use support for the former.
- **Latency** — direct-query synthesis happens while a user is waiting in Discord and should feel responsive; background fact-extraction from batched messages can tolerate more latency.
- **Cost at expected volume** — automatic extraction (Phase 3a) can run on every incoming message across every connected server; on-demand synthesis runs only per query. A per-call cost difference that's irrelevant at synthesis volume can matter a great deal at extraction volume.
- **Multilingual robustness** — servers may converse in any of the locales Aura supports (see Internationalization); the model doing extraction and synthesis needs to actually be capable in those languages, not just in English.

Each task's model choice is its own config value (e.g. `EXTRACTION_MODEL`, `SYNTHESIS_MODEL`), never hardcoded, so it can be tuned or swapped per deployment without touching code — the same "new provider → zero code changes" principle from Scalability & Extensibility, applied per task instead of globally.

## Proactive Relief: Visibly Active by Design

Phase 2b-3 recalibrated `PROACTIVE_QUESTION_THRESHOLD`, `PROACTIVE_SIMILARITY_THRESHOLD`, `PROACTIVE_CONFIDENCE_GAP`, and `PROACTIVE_DAILY_CAP` well past their original Phase 2a conservative defaults (see `src/aura/config.py` for the per-field evidence, drawn from the 580-message synthetic-corpus simulation in `reports/phase-2b-2.txt`). This section records the product decision behind those numbers, so a future session reading the config cold understands *why* the thresholds look loose rather than assuming they drifted or were left un-tuned.

**Phase 2b-4 then retired `PROACTIVE_CONFIDENCE_GAP` entirely** — see "Competing facts go to the model, not to a threshold" below. Everything in this section still holds; one of its four numbers simply no longer exists.

**The decision.** Aura is deliberately tuned to be visibly, actively helpful for standard, easy-to-moderate informational questions — the kind a fact lookup is squarely built to answer — so that server members and moderators can see it doing real, noticeable work, rather than sitting almost entirely silent. Phase 2b-2's evidence showed the bottleneck was never answer quality (Stage 3 scored 12/12 on the hardest calibration cases in the model bake-off, and zero adversarial cases fooled it even with the earlier gate stages bypassed) — it was Stage 1 and Stage 2 holding back genuine, answerable traffic before a model ever saw it. Loosening those stages means more messages now reach paid synthesis, and Timur has explicitly accepted the resulting cost increase. Two things make that an acceptable trade rather than an open-ended one: `PROACTIVE_DAILY_CAP` bounds the worst case in hard dollar terms per guild per month, and the realistic near-term server count is a handful of test and community servers, not a large multi-guild deployment. Revisit this trade — the cap value, and possibly the thresholds themselves — before any large-scale rollout; the math changes at real scale even though it doesn't yet. Aura still defers to humans for individual, complex questions a fact lookup was never meant to answer; the recalibration widens what counts as "easy enough to answer on its own," it does not remove the judgment call documented below.

**Partial answers may post; false or contradictory ones never do.** The bar for Trigger 2 was never "give an unambiguous answer or say nothing" — it is "never state something false or outdated." A fact that only partially covers a question is still genuinely useful, the same way a human's partial answer would be, so the shared synthesis prompt (`aura.synthesis`) instructs the model to post partial coverage honestly — naming what is documented and what is not — rather than staying silent or silently glossing over the gap. A numeric confidence-gap threshold was deliberately not used to gate this: the full corpus shows median confidence gaps are nearly identical across answered, partial, and contradictory cases, so a gap threshold cannot tell them apart and would silence roughly half of all three indiscriminately.

### The model is paid for judgment, never for knowledge

This is the principle that makes the paragraph above safe rather than reckless, and it is worth stating on its own: **synthesis exists so an LLM can weigh multiple retrieved facts against each other, notice when two of them genuinely conflict, and phrase a natural answer — reasoning capability plain similarity scores cannot provide. That reasoning is what Aura pays for. It is never permission for the model to add anything from its own trained knowledge as content.**

This is enforced structurally, not just by prompt instruction — the synthesis call's context contains only the facts retrieved from this project's own database, nothing else, so there is no channel through which the model's general knowledge could enter an answer even if it tried. The prompt-level half of this is the contradiction check: the synthesis prompt explicitly asks the model to check whether the facts it would cite genuinely conflict on the same specific detail, and to set `answers_question: false` rather than guess which one is current when they do. That is the correctness backstop for stale or contradictory facts — a job for the model's judgment, deliberately not chased with a numeric proxy (see `PROACTIVE_CONFIDENCE_GAP`'s comment in `config.py` for why a threshold can't do this).

**The structural fix.** The knowledge model has had a `status: active/superseded` field with successor chaining since Phase 1b, and the mod-facing `/aura-supersede` command that finally makes it usable landed after Phase 2b-3. Where a moderator has actually run it, the superseded fact drops out of retrieval and the question of "which of these two currently holds" does not arise. Where nobody has run it yet, an old fact and its replacement still both sit in the database as active, and Aura still has no structural way to tell them apart — only the per-call inference described above, which is approximate by nature. So the command narrows this gap rather than closing it: the model's contradiction check remains the backstop for every pair a human has not yet resolved by hand.

### Competing facts go to the model, not to a threshold

Phase 2b-4 removed `PROACTIVE_CONFIDENCE_GAP` as a gate. It required the best-matching fact to beat the runner-up by a margin before Aura would escalate, on the theory that a near-tie meant "one of these is probably stale." The theory does not survive measurement, and the reason generalizes past this one threshold:

**A near-tie between two facts is exactly as likely to mean they are complementary as to mean one is stale, and the geometry is identical in both cases.** Two facts about the same topic score close together because they are about the same topic — which is what "one replaced the other" and "both are true and belong in one answer" have in common. Phase 2b-2's corpus showed it (median gaps of 0.075 / 0.069 / 0.075 across answered, partial and contradictory categories — indistinguishable), and Aura's first live day showed the other half: two genuinely complementary maintenance facts were refused escalation three times at gaps of 0.047, 0.012 and 0.014, while `/aura-ask` — which never had such a gate — combined that same pair correctly in two languages, citing both.

The consequence for design, beyond this one field: **when a decision requires distinguishing two situations that produce the same number, no threshold on that number can make it. Move the decision to where the meaning is legible, or don't make it at all.** Here that means Stage 3, which is asked directly whether the facts it would cite genuinely conflict on the same specific detail. This is the same principle as "the model is paid for judgment, never for knowledge," applied one stage earlier: judgment is also what decides *whether these facts belong together*, not just how to phrase them.

Two things make this safe rather than merely louder:

- **Every qualifying fact reaches the model, bounded.** All facts clearing `PROACTIVE_SIMILARITY_THRESHOLD` go into one synthesis call, capped at `SYNTHESIS_FACT_LIMIT` (5). Sending only the winner of a complementary pair produces a worse answer, not a safer one — and sending only one of a contradictory pair is what would let Aura state a side confidently. The model can only notice a conflict it can see.
- **Nothing about the spend limits changed.** `PROACTIVE_DAILY_CAP` and `PROACTIVE_COOLDOWN_SECONDS` are untouched, and because the cap counts *escalations* rather than answers, removing an earlier gate cannot raise the ceiling — it only makes the cap more often the thing that binds. The measured cost delta of the wider fact context is about $0.03 per guild per month, well inside the envelope already accepted above.

One correction this also forced: through Phase 2b-3 the responder selected facts using `SIMILARITY_THRESHOLD` (0.40, the direct-query bar) while the gate escalated on `PROACTIVE_SIMILARITY_THRESHOLD` (0.30). A message whose best fact fell between the two was granted a budget slot and then found nothing to answer from — 45 of 580 corpus cases, and 111 once the gap stopped blocking. Trigger 2 now uses one bar end to end.

## Core Principles

### Performance
- The bot is async end-to-end. Never run blocking I/O or CPU-bound work (embedding inference, synchronous DB calls, slow HTTP requests) directly inside an `async def` handler — offload with `asyncio.to_thread` or use async-native libraries throughout.
- Batch operations wherever more than one item is processed at once (e.g. backfill), rather than looping one-by-one against the database or an LLM API.
- Design the knowledge model for the data volume it actually has (a few MB per server). Do not introduce infrastructure — vector databases, message queues, etc. — sized for a scale problem Aura doesn't have.

### Clean Code (this project is open source)
- Descriptive names over explanatory comments; comments explain **why**, not what the code already says.
- Type hints everywhere — function signatures, pydantic models.
- Single-responsibility functions and modules.
- No hardcoded strings, magic numbers, or user-facing text inside business logic (see Internationalization).
- Every public function and class gets a docstring.

### Scalability & Extensibility
Code should be structured so none of the following ever require touching core logic:
- A new language → add one locale file.
- A new LLM provider → zero code changes (that's what `litellm` is for).
- A new fact-extraction rule → an isolated, independently testable unit.

### Internationalization (i18n)
- Development, code comments, commit messages, and the support server all operate in **English**.
- End users interact with Aura in their own language. Supported locales at launch:

  | Flag | Language | Code |
  |---|---|---|
  | 🇺🇸 | English (default) | `en-US` |
  | 🇪🇸 | Español | `es-ES` |
  | 🇧🇷 | Português (Brasil) | `pt-BR` |
  | 🇩🇪 | Deutsch | `de` |
  | 🇫🇷 | Français | `fr` |
  | 🇹🇷 | Türkçe | `tr` |
  | 🇵🇱 | Polski | `pl` |
  | 🇯🇵 | 日本語 | `ja` |
  | 🇰🇷 | 한국어 | `ko` |

- Architecture: a translation-key system, never hardcoded strings. Each locale is a JSON file (`locales/en-US.json`, `locales/de.json`, ...) mapping keys to translated strings. A single `t(key, locale, **kwargs)` function resolves the string, with **`en-US` as a mandatory fallback** — a missing translation key must never crash the bot or show a blank message.
- This also lowers the bar for community contributions: adding a language becomes "copy `en-US.json`, translate the values, open a PR" — no code changes, no gettext tooling required.
- Use Discord's native locale support (`interaction.locale`, and `name_localizations` / `description_localizations` on slash commands) as the default signal for a user's language, with room for a per-server or per-user override later.

## The Non-Negotiable Principle: No Grey Areas, No Bugs Found Later

The single most important rule on this project, learned the hard way on a previous bot: **implementing something "safely" and then later discovering a grey area — or worse, a critical bug — is the failure mode to design against, not to accept as normal.**

Therefore, for every implementation, without exception:
- **Slower and correct beats fast and fragile.** If a feature can be built quickly but leaves an edge case unexamined, it is not done yet.
- Before any implementation is considered complete, **put yourself in the role of a hacker and actively try to misuse and break what you just built.** Not a light review — a genuine attempt at destruction, the way someone hostile to this project would approach it. Feed it malformed input, empty input, oversized input, concurrent/simultaneous calls, unicode edge cases (this matters given multi-language support), unexpected API failures and timeouts, and race conditions.
- This adversarial pass is not optional polish — it is part of the definition of "done," the same way a passing test suite is. Write the adversarial test cases alongside the happy-path tests, not as an afterthought.
- **Only once every bug this adversarial pass turns up has actually been fixed** — not logged, not deferred, fixed — is the implementation done.
- The trade-off is explicit and accepted: **production is slower this way, but more effective.** A feature that took longer because it survived a genuine attempt to break it is worth more than one that shipped fast and left a grey area for someone to find later.

## Testing

- Every non-trivial feature ships with `pytest` / `pytest-asyncio` tests: the happy path **and** the adversarial cases above.
- Test fact-extraction and matching logic as pure functions/units, independent of Discord — a live Discord connection should never be required to verify this logic is correct.

## Open Items (deferred, tracked here so they are not lost)

- **Cross-guild shared budget for a future hosted free tier.** The proactive
  daily cap (Phase 2a-2) is per-guild, which is correct when each deployment
  brings its own OpenRouter key. A future hosted free tier would put many
  guilds behind one shared key, where per-guild caps no longer bound the
  operator's total spend — one busy guild, or many guilds together, could run
  up a shared bill. A cross-guild budget layer would be needed above the
  per-guild cap before offering that. Explicitly out of scope until such a tier
  exists; noted here so it is designed for, not discovered.
