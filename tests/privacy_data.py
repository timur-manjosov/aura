"""A small, fully populated database for the P7a deletion tests: two servers, three people.

Every table that can hold a server's or a person's data gets rows for both
servers and for more than one person, so a test can assert that a rule removed
exactly what it should and nothing else -- by comparing whole-table snapshots.

Invented IDs only. Message IDs are real snowflakes for fixed moments, so the
rules' "written before" bound is exercised exactly as in production.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Final

import aiosqlite

from aura.db.connection import utc_day, utc_iso
from aura.db.deletion import DISCORD_EPOCH_MS
from aura.db.privacy_schema import verify_data_obligations_schema
from aura.db.repository import init_schema

GUILD_A: Final = 111_000_000_000_000_001
GUILD_B: Final = 222_000_000_000_000_002
CHANNEL_A: Final = 111_000_000_000_000_101
CHANNEL_B: Final = 222_000_000_000_000_201
MEMBER: Final = 900_000_000_000_000_001  # the one who asks for deletion
OTHER: Final = 900_000_000_000_000_002  # another member
MODERATOR: Final = 900_000_000_000_000_003

# The deletion request's moment; rows dated after it must survive.
REQUEST: Final = datetime(2026, 10, 1, 12, 0, tzinfo=UTC)
BEFORE: Final = REQUEST - timedelta(days=3)
AFTER: Final = REQUEST + timedelta(hours=1)
EMBEDDING: Final = bytes(384 * 4)

# Tables whose rows are billing records: no rule may ever touch them.
BILLING_TABLES: Final = ("guild_subscriptions", "stripe_processed_events")

_sequence = 0


def snowflake(moment: datetime) -> int:
    """Return a unique message ID created at `moment`."""
    global _sequence
    _sequence += 1
    return ((int(moment.timestamp() * 1000) - DISCORD_EPOCH_MS) << 22) | (_sequence & 0xFFF)


async def open_database(path: str = ":memory:") -> aiosqlite.Connection:
    """Return a migrated, empty database."""
    conn = await aiosqlite.connect(path)
    await init_schema(conn)
    await verify_data_obligations_schema(conn)
    return conn


async def add_fact(
    conn: aiosqlite.Connection,
    *,
    guild_id: int,
    author: int | None,
    when: datetime,
    content: str = "a fact",
    channel_id: int | None = None,
) -> int:
    """Insert one active fact whose source message was written at `when`."""
    cursor = await conn.execute(
        """
        INSERT INTO facts (guild_id, channel_id, message_id, content, embedding, status,
                           created_at, source_author_id)
        VALUES (?, ?, ?, ?, ?, 'active', ?, ?)
        """,
        (
            guild_id,
            channel_id or (CHANNEL_A if guild_id == GUILD_A else CHANNEL_B),
            snowflake(when),
            content,
            EMBEDDING,
            utc_iso(when + timedelta(minutes=5)),
            author,
        ),
    )
    await conn.commit()
    assert cursor.lastrowid is not None
    return cursor.lastrowid


async def supersede(conn: aiosqlite.Connection, old: int, new: int, when: datetime) -> None:
    """Mark `old` superseded by `new`."""
    await conn.execute(
        "UPDATE facts SET status = 'superseded', superseded_by_id = ?, superseded_at = ? "
        "WHERE id = ?",
        (new, utc_iso(when), old),
    )
    await conn.commit()


async def add_candidate(
    conn: aiosqlite.Connection,
    *,
    guild_id: int,
    author: int | None,
    when: datetime,
    content: str = "a candidate",
    similar_fact_id: int | None = None,
    confirmed_fact_id: int | None = None,
    resolved_by: int | None = None,
) -> int:
    """Insert one candidate (and a supersession spend row pointing at it)."""
    status = "confirmed" if confirmed_fact_id else ("discarded" if resolved_by else "pending")
    cursor = await conn.execute(
        """
        INSERT INTO pending_facts (guild_id, channel_id, message_id, content, embedding,
            category, status, similar_fact_id, similar_fact_score, relationship,
            relationship_reasoning, confirmed_fact_id, created_at, resolved_at,
            resolved_by_id, source_author_id)
        VALUES (?, ?, ?, ?, ?, 'rule', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            guild_id,
            CHANNEL_A if guild_id == GUILD_A else CHANNEL_B,
            snowflake(when),
            content,
            EMBEDDING,
            status,
            similar_fact_id,
            0.9 if similar_fact_id else None,
            "complementary" if similar_fact_id else None,
            "the model's reasoning" if similar_fact_id else None,
            confirmed_fact_id,
            utc_iso(when + timedelta(minutes=1)),
            utc_iso(when + timedelta(minutes=2)) if resolved_by else None,
            resolved_by,
            author,
        ),
    )
    candidate_id = cursor.lastrowid
    await conn.execute(
        "INSERT INTO supersession_calls (guild_id, pending_fact_id, called_at, call_day) "
        "VALUES (?, ?, ?, ?)",
        (guild_id, candidate_id, utc_iso(when + timedelta(minutes=1)), utc_day(when)),
    )
    await conn.commit()
    assert candidate_id is not None
    return candidate_id


async def add_guild_rows(
    conn: aiosqlite.Connection, *, guild_id: int, when: datetime, moderator: int = MODERATOR
) -> None:
    """Insert one row into every other per-server table, for members MEMBER and OTHER."""
    channel = CHANNEL_A if guild_id == GUILD_A else CHANNEL_B
    stamp = utc_iso(when)
    day = utc_day(when)
    rows: list[tuple[str, tuple[object, ...]]] = [
        (
            "INSERT INTO proactive_signals (guild_id, channel_id, message_id, stage1_score, "
            "stage1_passed, verdict, created_at) VALUES (?, ?, ?, 0.1, 1, 'eligible', ?)",
            (guild_id, channel, snowflake(when), stamp),
        ),
        (
            "INSERT INTO proactive_escalations (guild_id, channel_id, message_id, escalated_at, "
            "escalation_day) VALUES (?, ?, ?, ?, ?)",
            (guild_id, channel, snowflake(when), stamp, day),
        ),
        (
            "INSERT INTO proactive_channel_config (channel_id, guild_id, proactive_enabled, "
            "updated_by_id, updated_at) VALUES (?, ?, 1, ?, ?)",
            (channel, guild_id, moderator, stamp),
        ),
        (
            "INSERT INTO extraction_channel_config (channel_id, guild_id, extraction_enabled, "
            "updated_by_id, updated_at) VALUES (?, ?, 1, ?, ?)",
            (channel, guild_id, moderator, stamp),
        ),
        (
            "INSERT INTO extraction_queue (channel_id, message_id, guild_id, channel_name, "
            "content, message_created_at, enqueued_at, author_id) VALUES (?, ?, ?, 'general', "
            "'raw text', ?, ?, ?)",
            (channel, snowflake(when), guild_id, stamp, stamp, MEMBER),
        ),
        (
            "INSERT INTO extraction_queue (channel_id, message_id, guild_id, channel_name, "
            "content, message_created_at, enqueued_at, author_id) VALUES (?, ?, ?, 'general', "
            "'raw text', ?, ?, ?)",
            (channel, snowflake(when), guild_id, stamp, stamp, OTHER),
        ),
        (
            "INSERT INTO extraction_calls (guild_id, channel_id, message_count, called_at, "
            "call_day) VALUES (?, ?, 3, ?, ?)",
            (guild_id, channel, stamp, day),
        ),
        (
            "INSERT INTO ask_calls (guild_id, user_id, created_at, call_day) VALUES (?, ?, ?, ?)",
            (guild_id, MEMBER, stamp, day),
        ),
        (
            "INSERT INTO ask_calls (guild_id, user_id, created_at, call_day) VALUES (?, ?, ?, ?)",
            (guild_id, OTHER, stamp, day),
        ),
        (
            "INSERT INTO digest_config (guild_id, channel_id, interval_seconds, digest_enabled, "
            "enabled_at, updated_by_id, updated_at) VALUES (?, ?, 86400, 1, ?, ?, ?)",
            (guild_id, channel, stamp, moderator, stamp),
        ),
        (
            "INSERT INTO digest_runs (guild_id, channel_id, covered_from, covered_until, "
            "new_fact_count, milestone_count, updated_fact_count, outcome, ran_at) "
            "VALUES (?, ?, ?, ?, 1, 0, 0, 'posted', ?)",
            (guild_id, channel, stamp, stamp, stamp),
        ),
        (
            "INSERT INTO onboarding_config (guild_id, channel_id, onboarding_enabled, "
            "updated_by_id, updated_at) VALUES (?, ?, 1, ?, ?)",
            (guild_id, channel, moderator, stamp),
        ),
        (
            "INSERT INTO onboarding_sends (guild_id, user_id, joined_at, send_day, fact_count, "
            "sent_at) VALUES (?, ?, ?, ?, 3, ?)",
            (guild_id, MEMBER, stamp, day, stamp),
        ),
        (
            "INSERT INTO onboarding_sends (guild_id, user_id, joined_at, send_day, fact_count, "
            "sent_at) VALUES (?, ?, ?, ?, 3, ?)",
            (guild_id, OTHER, stamp, day, stamp),
        ),
        (
            "INSERT INTO backfill_runs (guild_id, channel_id, state, until_message_id, "
            "requested_by_id, started_at, updated_at) VALUES (?, ?, 'completed', ?, ?, ?, ?)",
            (guild_id, channel, snowflake(when), moderator, stamp, stamp),
        ),
    ]
    for sql, parameters in rows:
        await conn.execute(sql, parameters)
    await conn.execute(
        "INSERT INTO backfill_calls (guild_id, run_id, message_count, called_at, call_day) "
        "VALUES (?, last_insert_rowid(), 5, ?, ?)",
        (guild_id, stamp, day),
    )
    await conn.commit()


async def add_billing_rows(conn: aiosqlite.Connection, *, guild_id: int) -> None:
    """Insert a subscription and a processed event for a server."""
    await conn.execute(
        """
        INSERT INTO guild_subscriptions (subscription_id, guild_id, customer_id,
            purchaser_user_id, status, cancel_at_period_end, collection_paused,
            current_period_start, current_period_end, livemode, version, first_seen_at,
            confirmed_at)
        VALUES (?, ?, 'cus_test', ?, 'active', 0, 0, 1, 2, 0, 1, 'x', 'x')
        """,
        (f"sub_{guild_id}", guild_id, MEMBER),
    )
    await conn.execute(
        "INSERT INTO stripe_processed_events (event_id, event_type, subscription_id, "
        "processed_at) VALUES (?, 'customer.subscription.updated', ?, 'x')",
        (f"evt_{guild_id}", f"sub_{guild_id}"),
    )
    await conn.commit()


async def populate(conn: aiosqlite.Connection) -> dict[str, int]:
    """Fill both servers; return the IDs tests refer to by name."""
    ids: dict[str, int] = {}
    for guild, prefix in ((GUILD_A, "a"), (GUILD_B, "b")):
        ids[f"{prefix}_member_fact"] = await add_fact(
            conn, guild_id=guild, author=MEMBER, when=BEFORE, content=f"{prefix} member fact"
        )
        ids[f"{prefix}_other_fact"] = await add_fact(
            conn, guild_id=guild, author=OTHER, when=BEFORE, content=f"{prefix} other fact"
        )
        ids[f"{prefix}_member_late_fact"] = await add_fact(
            conn, guild_id=guild, author=MEMBER, when=AFTER, content=f"{prefix} written later"
        )
        ids[f"{prefix}_unknown_fact"] = await add_fact(
            conn, guild_id=guild, author=None, when=BEFORE, content=f"{prefix} author unknown"
        )
        ids[f"{prefix}_member_candidate"] = await add_candidate(
            conn,
            guild_id=guild,
            author=MEMBER,
            when=BEFORE,
            similar_fact_id=ids[f"{prefix}_other_fact"],
            resolved_by=MODERATOR,
        )
        ids[f"{prefix}_other_candidate"] = await add_candidate(
            conn,
            guild_id=guild,
            author=OTHER,
            when=BEFORE,
            similar_fact_id=ids[f"{prefix}_member_fact"],
        )
        await conn.execute(
            "INSERT INTO fact_links (fact_a_id, fact_b_id, created_at) VALUES (?, ?, 'x')",
            tuple(sorted((ids[f"{prefix}_member_fact"], ids[f"{prefix}_other_fact"]))),
        )
        await conn.execute(
            "INSERT INTO fact_variants (fact_id, content, embedding, created_at) VALUES (?, 'v', ?, 'x')",
            (ids[f"{prefix}_member_fact"], EMBEDDING),
        )
        await conn.execute(
            "INSERT INTO variant_calls (guild_id, fact_id, called_at, call_day) VALUES (?, ?, ?, ?)",
            (guild, ids[f"{prefix}_member_fact"], utc_iso(BEFORE), utc_day(BEFORE)),
        )
        await conn.commit()
        await add_guild_rows(conn, guild_id=guild, when=BEFORE)
        await add_billing_rows(conn, guild_id=guild)
    return ids


async def snapshot(conn: aiosqlite.Connection) -> dict[str, list[tuple[object, ...]]]:
    """Return every row of every table, sorted -- for exact before/after comparisons."""
    async with conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name != 'sqlite_sequence' "
        "ORDER BY name"
    ) as cursor:
        tables = [row[0] for row in await cursor.fetchall()]
    result: dict[str, list[tuple[object, ...]]] = {}
    for table in tables:
        async with conn.execute(f"SELECT * FROM {table}") as cursor:
            result[table] = sorted((tuple(row) for row in await cursor.fetchall()), key=repr)
    return result
