CREATE TABLE IF NOT EXISTS facts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    guild_id INTEGER NOT NULL,
    channel_id INTEGER NOT NULL,
    message_id INTEGER NOT NULL,
    content TEXT NOT NULL,
    embedding BLOB NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('active', 'superseded')) DEFAULT 'active',
    superseded_by_id INTEGER REFERENCES facts(id),
    created_at TEXT NOT NULL,
    superseded_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_facts_guild_status ON facts(guild_id, status);

CREATE TABLE IF NOT EXISTS fact_links (
    fact_a_id INTEGER NOT NULL REFERENCES facts(id),
    fact_b_id INTEGER NOT NULL REFERENCES facts(id),
    created_at TEXT NOT NULL,
    PRIMARY KEY (fact_a_id, fact_b_id),
    CHECK (fact_a_id < fact_b_id)
);

-- ---------------------------------------------------------------------------
-- Everything above is the knowledge model (fact, timestamp, status, link).
-- Everything below is NOT, and must never be treated as if it were.
-- ---------------------------------------------------------------------------

-- Phase 2 diagnostic scaffolding: the gate's full decision trail per message.
-- A classifier score is none of the four things CLAUDE.md admits into the
-- knowledge model, so it gets its own table with no foreign key to facts in
-- either direction -- this table can be dropped wholesale once proactive
-- relief is tuned, without the knowledge model noticing.
--
-- Stores no message text, deliberately: guild/channel/message ID resolves to
-- a Discord permalink, exactly as a fact references its own origin, so raw
-- message content is never duplicated into Aura's database.
--
-- Grows by one ~120-byte row per human message Aura can see. Not pruned: this
-- is short-lived debugging scaffolding meant to be watched and then removed,
-- and a retention policy would be more machinery than the thing it protects.
--
-- Every stage column past Stage 1 is nullable because the gate short-circuits:
-- a message that fails Stage 1 has no Stage 2 score to record, and NULL says
-- "never evaluated" where 0.0 would falsely say "evaluated and scored zero".
-- The cooldown and cap columns record the state observed *at evaluation time*
-- rather than being recomputed later, since that state is exactly what cannot
-- be reconstructed after the fact.
CREATE TABLE IF NOT EXISTS proactive_signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    guild_id INTEGER NOT NULL,
    channel_id INTEGER NOT NULL,
    message_id INTEGER NOT NULL,
    -- Contrastive: question-exemplar similarity minus statement-exemplar
    -- similarity, so it spans [-2, 2] and not [0, 1]. Deliberately NOT named
    -- `score` like the Phase 2a-1 column it replaces: that one held a
    -- one-sided similarity, the two numbers are not comparable, and a shared
    -- name would have silently mixed both scales into one recalibration set.
    stage1_score REAL NOT NULL,
    stage1_passed INTEGER NOT NULL CHECK (stage1_passed IN (0, 1)),
    stage2_top_score REAL,
    stage2_runner_up_score REAL,
    stage2_gap REAL,
    stage2_passed INTEGER CHECK (stage2_passed IN (0, 1)),
    cooldown_seconds_remaining REAL,
    daily_count INTEGER,
    daily_cap INTEGER,
    -- Phase 2a-3 synthesis outcome, filled in AFTER the gate trail is first
    -- recorded (see update_synthesis_outcome). Both nullable, and NULL means
    -- three honest things at once: the message never reached synthesis (any
    -- verdict other than ELIGIBLE), or synthesis produced no result (the LLM
    -- call failed or is not configured), depending on the verdict beside them.
    -- These columns are additive over Phase 2a-2's shape and are back-filled by
    -- an ALTER TABLE migration on an existing table (see verify_signal_schema),
    -- NOT by dropping it: unlike the 2a-1 -> 2a-2 change, no existing column's
    -- meaning changes, so the collected tuning data stays valid.
    synthesis_answers_question INTEGER CHECK (synthesis_answers_question IN (0, 1)),
    synthesis_posted INTEGER CHECK (synthesis_posted IN (0, 1)),
    -- Phase 2b-1's grace-period outcome, filled in AFTER the gate trail is
    -- first recorded (see update_grace_outcome), same split-write pattern as
    -- the synthesis columns above and for the same reason: the ELIGIBLE trail
    -- must exist before the (now longer) wait begins, so a concurrent
    -- redelivery cannot win the (channel_id, message_id) ON CONFLICT and
    -- replace it with a DUPLICATE_DELIVERY artefact. NULL means "never
    -- reached a grace period" (any verdict other than ELIGIBLE); a non-NULL
    -- value distinguishes a wait still in flight (pending) from every way it
    -- can end. See GracePeriodOutcome for the closed set of values.
    grace_period_outcome TEXT CHECK (
        grace_period_outcome IN (
            'pending', 'cancelled_by_human', 'expired_and_proceeded', 'stood_down_on_recheck'
        )
    ),
    verdict TEXT NOT NULL,
    created_at TEXT NOT NULL,
    -- Discord's gateway can redeliver an event after a resumed session; the
    -- same message classified twice is one observation, not two, and a
    -- duplicate row would quietly skew every reading taken from this table.
    -- Scoped per channel rather than trusting snowflakes to be globally
    -- unique, since nothing here depends on that being true.
    UNIQUE (channel_id, message_id)
);

CREATE INDEX IF NOT EXISTS idx_proactive_signals_guild_id
    ON proactive_signals(guild_id, id DESC);

-- The durable budget ledger: one row per message that became eligible for
-- (future) paid synthesis. Deliberately one table serving both protections,
-- because both are questions about the same set of events:
--
--   * per-channel cooldown -- is there a row for this channel newer than
--     (now - cooldown)?
--   * per-guild daily cap   -- how many rows does this guild have for today?
--
-- Two separate mechanisms (say, a "last escalation" row per channel plus a
-- counter row per guild) would need to be kept in step with each other; one
-- append-only ledger cannot disagree with itself. It is also why both survive
-- a container restart for free: there is no in-memory state to lose.
--
-- Rows are written *before* any expensive work, never after it, so a crash
-- mid-synthesis spends the slot rather than silently refunding it -- the
-- conservative direction for something whose whole job is bounding spend.
CREATE TABLE IF NOT EXISTS proactive_escalations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    guild_id INTEGER NOT NULL,
    channel_id INTEGER NOT NULL,
    message_id INTEGER NOT NULL,
    escalated_at TEXT NOT NULL,
    -- The UTC calendar day (YYYY-MM-DD) escalated_at falls in, stored rather
    -- than derived per query so the daily count is one indexed equality
    -- lookup and cannot drift from how a query happens to parse timestamps.
    -- UTC, not local time: see aura.db.proactive_state for why.
    escalation_day TEXT NOT NULL,
    -- Idempotency against Discord redelivering the same message event after a
    -- resumed session: a duplicate must not consume a second slot from either
    -- the cooldown or the daily cap.
    --
    -- Load-bearing, not defensive decoration. Verified against the installed
    -- discord.py: ConnectionState.parse_message_create dispatches 'message'
    -- unconditionally, and the message deque it appends to afterwards is only a
    -- cache for get_message -- it is never consulted to suppress a repeat. So
    -- when the gateway replays events after a RESUME, on_message fires again
    -- with the same message ID and nothing upstream of this constraint stops
    -- the second delivery from spending a second slot.
    UNIQUE (channel_id, message_id)
);

CREATE INDEX IF NOT EXISTS idx_proactive_escalations_guild_day
    ON proactive_escalations(guild_id, escalation_day);

CREATE INDEX IF NOT EXISTS idx_proactive_escalations_channel_time
    ON proactive_escalations(channel_id, escalated_at DESC);

-- Per-channel on/off switch for proactive relief (CLAUDE.md's second
-- trigger), set by a moderator via /aura-config. NOT part of the knowledge
-- model -- it is a configuration of the existing Trigger 2 mechanism, not a
-- fifth mechanism, and like the two tables above it has no foreign key to
-- facts in either direction.
--
-- A channel with NO row here is OFF. Proactive relief is opt-in per channel by
-- design, not opt-out: this phase is the first time Aura spends money and
-- posts publicly, its thresholds are still un-recalibrated placeholders, and
-- CLAUDE.md requires the trigger to be "deliberately conservative, to avoid
-- unwanted interruptions." A bot that started volunteering answers in every
-- channel the moment it joined a server -- against placeholder thresholds --
-- is exactly the trust-damaging behaviour this phase is built to avoid. So a
-- moderator must consciously choose each channel Aura may speak up in.
--
-- channel_id is the primary key: a Discord channel (including a thread, which
-- gets its own channel ID) is globally unique, and the on/off state is a
-- property of the channel, not of the guild. guild_id is kept for isolation
-- queries and for showing a moderator their own channels.
CREATE TABLE IF NOT EXISTS proactive_channel_config (
    channel_id INTEGER PRIMARY KEY,
    guild_id INTEGER NOT NULL,
    proactive_enabled INTEGER NOT NULL CHECK (proactive_enabled IN (0, 1)),
    -- Who last changed it and when, for a moderator auditing why a channel is
    -- (or isn't) speaking. Not load-bearing; purely diagnostic.
    updated_by_id INTEGER NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_proactive_channel_config_guild
    ON proactive_channel_config(guild_id);
