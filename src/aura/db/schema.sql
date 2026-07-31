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

-- Per-channel on/off switch for automatic fact extraction (Phase 3a), set by a
-- moderator via /aura-config. Deliberately its OWN table, not a second column
-- read off proactive_channel_config: extraction and proactive relief are two
-- independent mechanisms that happen to both hang off on_message, and
-- reports/phase-3-pre-analysis.md Section 1c found a real collision risk in
-- reusing proactive relief's gate for extraction -- a channel a moderator
-- opted into *answering* questions in is not necessarily one they want every
-- message *read for facts*, and the reverse holds just as much. One shared
-- gate would silently couple two decisions CLAUDE.md treats as separate.
--
-- Same invariant as proactive_channel_config for the same reason: a channel
-- with NO row is OFF. Phase 3a-1 ships only the local, free first filter,
-- unwired from any live path (see aura.extraction) -- but the gate that will
-- eventually guard it is built now, opt-in by default, consistent with every
-- other Aura mechanism that posts or reads at volume.
CREATE TABLE IF NOT EXISTS extraction_channel_config (
    channel_id INTEGER PRIMARY KEY,
    guild_id INTEGER NOT NULL,
    extraction_enabled INTEGER NOT NULL CHECK (extraction_enabled IN (0, 1)),
    updated_by_id INTEGER NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_extraction_channel_config_guild
    ON extraction_channel_config(guild_id);

-- Phase 3a-2's batch collector: candidate messages waiting for their channel's
-- batch window to close (see aura.extraction.pipeline).
--
-- WHY THIS TABLE HOLDS RAW MESSAGE TEXT, when proactive_signals deliberately
-- does not and a Fact deliberately stores a distilled sentence instead of a
-- copy. This is the one place in Aura that keeps a verbatim message, and it is
-- a bounded, deliberate exception rather than a lapse:
--
--   * The batch MUST survive a container restart -- the same durability bar
--     the escalation ledger is held to -- so it cannot live in a Python dict.
--   * The alternative, storing only IDs and re-fetching each message from
--     Discord when the batch closes, trades one LLM call's worth of work for N
--     HTTP round trips, makes flushing depend on the gateway being healthy,
--     and re-reads content Aura already had in hand.
--   * The row's life is one batch window (minutes), and it is deleted the
--     moment the batch is distilled -- or earlier, if the message is edited or
--     deleted first. Nothing here is a record; it is a buffer.
--
-- The primary key is (channel_id, message_id) rather than a surrogate id, so a
-- redelivered gateway event cannot enqueue the same message twice and pay to
-- distill it twice within one batch.
--
-- enqueued_at is Aura's own clock; message_created_at is Discord's timestamp
-- for the message itself. Both are kept because they answer different
-- questions: the first decides when this channel's window closes, the second
-- is context the distillation model is actually shown (see the phase brief's
-- "channel context" decision).
CREATE TABLE IF NOT EXISTS extraction_queue (
    channel_id INTEGER NOT NULL,
    message_id INTEGER NOT NULL,
    guild_id INTEGER NOT NULL,
    channel_name TEXT NOT NULL,
    content TEXT NOT NULL,
    message_created_at TEXT NOT NULL,
    enqueued_at TEXT NOT NULL,
    PRIMARY KEY (channel_id, message_id)
);

CREATE INDEX IF NOT EXISTS idx_extraction_queue_enqueued
    ON extraction_queue(enqueued_at);

-- Phase 3a-2's staging area: fact CANDIDATES produced by the distillation
-- model, before any human has confirmed them.
--
-- WHY A SEPARATE TABLE RATHER THAN A THIRD `facts.status` VALUE. Both were on
-- the table; this is the reasoning, recorded because the choice is not obvious
-- either way:
--
--   1. Every read path over `facts` today filters on status = 'active' and
--      nothing else. Adding a 'pending' status would make each of those a
--      place where forgetting one predicate silently leaks an unconfirmed,
--      machine-written sentence into a cited public answer -- across
--      get_active_facts, find_similar_facts, /aura-facts, /aura-supersede and
--      both synthesis triggers. A separate table cannot be read by accident:
--      no existing query names it. That asymmetry is the whole argument.
--   2. A candidate carries fields a real fact must never have -- the model's
--      category, the possible-predecessor hint and its score, who resolved it
--      and when. On `facts` those are columns that are NULL for every genuine
--      row, and their presence invites the question of whether a fact might
--      legitimately have one.
--   3. CLAUDE.md admits exactly four things into the knowledge model. A
--      sentence no human has confirmed is not a fact yet; it is extraction
--      scaffolding, the same category as the two Phase 2 tables above, and it
--      belongs on this side of the line drawn at the top of this file.
--
-- Confirming a candidate creates an ordinary row in `facts` through the
-- ordinary insert path, so there stays exactly one way a fact comes into
-- existence. confirmed_fact_id records which one, so the trail from an
-- automatic fact back to the message and the model output that produced it
-- stays walkable.
--
-- Resolved candidates are kept rather than deleted, mirroring how a superseded
-- fact is kept: "the model proposed this and a moderator rejected it" is the
-- only evidence a later phase has for whether extraction is worth its cost.
--
-- The UNIQUE constraint makes staging idempotent. A batch re-distilled after a
-- crash (see aura.extraction.pipeline) produces the same sentences from the
-- same messages, and those must land as the same candidates, not as duplicates
-- a moderator has to reject one by one. It is deliberately (channel, message,
-- content) and not (channel, message): one message can legitimately assert two
-- separate things.
CREATE TABLE IF NOT EXISTS pending_facts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    guild_id INTEGER NOT NULL,
    channel_id INTEGER NOT NULL,
    message_id INTEGER NOT NULL,
    -- The DISTILLED sentence the model wrote, never a copy of the source
    -- message. Whether that is actually true of a given row is a property of
    -- the prompt, measured in reports/phase-3a-2.txt, not something this
    -- schema can enforce.
    content TEXT NOT NULL,
    embedding BLOB NOT NULL,
    category TEXT NOT NULL CHECK (
        category IN (
            'announcement', 'rule', 'decision', 'event', 'status_change', 'milestone'
        )
    ),
    status TEXT NOT NULL CHECK (status IN ('pending', 'confirmed', 'discarded'))
        DEFAULT 'pending',
    -- The possible predecessor this candidate may restate, and how similar it
    -- scored. Advisory: it is shown to the confirming moderator and decides
    -- nothing.
    similar_fact_id INTEGER REFERENCES facts(id),
    similar_fact_score REAL,
    -- Phase 3a-3: what a model judged that similarity to actually MEAN, and the
    -- one sentence it gave for why. Filled in at staging time, and only for a
    -- candidate whose similar_fact_id above is set -- the judgment call fires
    -- for nothing else, which is what keeps it a small slice of extraction's
    -- volume.
    --
    -- STILL ADVISORY, and this is the property the whole sub-phase rests on: a
    -- 'supersession' here is a PROPOSAL a moderator reads, never an action.
    -- Nothing in this codebase supersedes a fact from these columns; the only
    -- caller of supersede_fact is /aura-supersede, run by a human. Storing a
    -- judgment beside the candidate rather than acting on it is the entire
    -- design.
    --
    -- Both NULL means "never judged", which is an ordinary, expected state with
    -- three causes: the candidate was never flagged for dedup at all, the daily
    -- cap refused the call, or the call failed. All three fall back to Phase
    -- 3a-2's plain similarity hint rather than blocking the review.
    relationship TEXT CHECK (
        relationship IN (
            'supersession', 'complementary', 'contradiction', 'independent'
        )
    ),
    -- The model's own reasoning sentence, written in the candidate's language
    -- (see aura.extraction.supersession) and shown to the moderator as the
    -- model's words rather than as Aura's conclusion.
    relationship_reasoning TEXT,
    -- The real fact a confirmation produced, NULL until then.
    confirmed_fact_id INTEGER REFERENCES facts(id),
    created_at TEXT NOT NULL,
    resolved_at TEXT,
    resolved_by_id INTEGER,
    UNIQUE (channel_id, message_id, content)
);

CREATE INDEX IF NOT EXISTS idx_pending_facts_guild_status
    ON pending_facts(guild_id, status, id);

-- The distillation call's spend ledger: one row per paid call, per guild, per
-- UTC day. A deliberate structural twin of proactive_escalations above -- same
-- append-only shape, same stored UTC day key, same "written before the work it
-- authorizes, never refunded" rule -- so the two spend limits behave
-- identically and neither can be reasoned about in isolation from the other.
--
-- Two differences from proactive_escalations, both intentional:
--
--   * No cooldown column and no cooldown concept. The batch window already
--     bounds how often one channel can produce a call (at most one per
--     window), which is what a cooldown would have been for. Adding a second,
--     overlapping rate limit would mean two numbers that have to be kept
--     consistent with each other to mean anything.
--   * No UNIQUE (channel_id, message_id) idempotency key. There is no gateway
--     redelivery to absorb here: calls originate from Aura's own sweeper, not
--     from a Discord event. A crash between claiming a slot and finishing the
--     batch therefore spends the slot and re-does the work, which is the same
--     conservative direction proactive_escalations chose for the same reason:
--     for a spend limit, erring toward "already spent" is the only safe way to
--     err. Duplicate CANDIDATES from that retry are prevented by
--     pending_facts' own UNIQUE constraint instead.
CREATE TABLE IF NOT EXISTS extraction_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    guild_id INTEGER NOT NULL,
    channel_id INTEGER NOT NULL,
    message_count INTEGER NOT NULL,
    called_at TEXT NOT NULL,
    call_day TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_extraction_calls_guild_day
    ON extraction_calls(guild_id, call_day);

-- Phase 3a-3's spend ledger: one row per paid supersession-judgment call, per
-- guild, per UTC day. The third instance of the same shape as the two ledgers
-- above, and deliberately a third BUDGET rather than a share of extraction's:
-- two call sites drawing on one number would leave neither with a bound of its
-- own, and a burst of dedup-flagged candidates would eat the extraction budget
-- that produces them.
--
-- pending_fact_id, where extraction_calls carries message_count: this call
-- judges exactly one staged candidate, so the ledger can point at it, which
-- makes "what did this spend actually buy?" answerable by a join rather than by
-- correlating timestamps. The candidate is always staged before its slot is
-- claimed, so the reference is never dangling -- and candidates are never
-- deleted (see pending_facts above), so it stays resolvable forever.
CREATE TABLE IF NOT EXISTS supersession_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    guild_id INTEGER NOT NULL,
    pending_fact_id INTEGER NOT NULL REFERENCES pending_facts(id),
    called_at TEXT NOT NULL,
    call_day TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_supersession_calls_guild_day
    ON supersession_calls(guild_id, call_day);
