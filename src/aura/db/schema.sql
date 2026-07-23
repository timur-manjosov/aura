CREATE TABLE IF NOT EXISTS facts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    guild_id INTEGER NOT NULL,
    channel_id INTEGER NOT NULL,
    message_id INTEGER NOT NULL,
    content TEXT NOT NULL,
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
