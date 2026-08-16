"""Per-guild configuration for the periodic digest (CLAUDE.md's fourth trigger):
which channel it goes to, how often, and whether it runs at all.

A deliberate sibling of aura.db.proactive_channel_config and
aura.db.extraction_channel_config -- same opt-in-by-default invariant, same
upsert shape, same "who changed it and when" audit columns -- with one
structural difference stated up front because it is the thing a reader will
notice first: **this table is keyed by guild, not by channel.** The reasoning is
in schema.sql above the table; the short version is that "may Aura speak here"
is a property each channel answers for itself, while "where does this server's
digest go" is one question about one server with exactly one answer.

**A guild with no row is OFF.** get_enabled_digest_configs simply never returns
it, which is the only read the scheduler makes -- there is no code path where an
unconfigured guild has to be recognised as a special case, because it is
indistinguishable from a guild that does not exist.

This module owns exactly one table and imports nothing from aura.db.repository,
for the same isolation reason its two siblings give: a configuration switch is
none of the four things CLAUDE.md admits into the knowledge model, and it should
stay separable from the facts it schedules.
"""
from __future__ import annotations

import logging
import sqlite3
from datetime import datetime

import aiosqlite
from pydantic import BaseModel

from aura.db.connection import connection_lock, utc_now_iso

logger = logging.getLogger(__name__)

_CONFIG_COLUMNS = (
    "guild_id, channel_id, interval_seconds, digest_enabled, enabled_at, "
    "updated_by_id, updated_at"
)

# A stored interval outside this range cannot have come from the slash command,
# whose choices are a fixed set (see aura.digest.intervals). It can only come
# from a hand-edited database, and both ends of the range are guarded because
# both break something real: at or below zero every sweep would find the guild
# due and post a digest every hour forever, and a value past ten years makes the
# "next digest" arithmetic meaningless rather than merely long. Such a row is
# skipped with a warning rather than clamped -- Aura should not invent a
# schedule an operator did not choose.
MIN_INTERVAL_SECONDS = 60
MAX_INTERVAL_SECONDS = 10 * 365 * 24 * 60 * 60


class DigestConfig(BaseModel):
    """One guild's digest settings, as read back from the database.

    enabled_at is the baseline the FIRST digest is measured from, not a
    diagnostic timestamp: with no previous run to start from, a digest has to
    begin somewhere, and "when a moderator turned this on" is the only honest
    answer -- covering all of history instead is the onboarding trigger's job.
    It is preserved across a change of channel or interval and reset when
    digests are re-enabled after being off, so a pause is never retro-reported
    (see set_digest_config).
    """

    guild_id: int
    channel_id: int
    interval_seconds: int
    digest_enabled: bool
    enabled_at: datetime
    updated_by_id: int
    updated_at: datetime


def _row_to_config(row: sqlite3.Row) -> DigestConfig:
    return DigestConfig(
        guild_id=row[0],
        channel_id=row[1],
        interval_seconds=row[2],
        digest_enabled=bool(row[3]),
        enabled_at=row[4],
        updated_by_id=row[5],
        updated_at=row[6],
    )


async def set_digest_config(
    conn: aiosqlite.Connection,
    *,
    guild_id: int,
    channel_id: int,
    interval_seconds: int,
    enabled: bool,
    updated_by_id: int,
) -> None:
    """Write one guild's digest settings, recording who changed them.

    An upsert keyed on guild_id, the same shape its two channel-scoped siblings
    use: reconfiguring the same guild repeatedly leaves exactly one row, always
    reflecting the most recent decision.

    The CASE expression on enabled_at is the only non-obvious part, and it
    encodes a product decision rather than a storage detail. The baseline is
    preserved only while digests stay continuously on, so that changing the
    channel or the interval mid-week does not silently discard the changes
    accumulated since the last digest. Every other transition sets it to now:

      * off (or absent) -> on: the first digest covers the period starting now,
        not the entire history of the server. A moderator switching digests on
        is asking "tell me what changes from here", and dumping everything Aura
        has ever learned into a "what's new" post would be both wrong and, on a
        server with any history, unreadable.
      * on -> on again after a pause: the same reasoning. The silent period is
        deliberately not retro-reported -- a digest re-enabled in March should
        not open with three months of accumulated changes.
      * on -> off: irrelevant while off, and reset for whenever it comes back.

    Rejects an out-of-range interval here as well as at the reader, because a
    value this function accepts is a value some later reader has to defend
    against; refusing it at the one write path is what keeps that defence a
    guard rather than a policy.
    """
    if not MIN_INTERVAL_SECONDS <= interval_seconds <= MAX_INTERVAL_SECONDS:
        raise ValueError(
            f"interval_seconds must be between {MIN_INTERVAL_SECONDS} and "
            f"{MAX_INTERVAL_SECONDS}, got {interval_seconds}"
        )

    now = utc_now_iso()
    async with connection_lock(conn):
        await conn.execute(
            """
            INSERT INTO digest_config
                (guild_id, channel_id, interval_seconds, digest_enabled, enabled_at,
                 updated_by_id, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (guild_id) DO UPDATE SET
                channel_id = excluded.channel_id,
                interval_seconds = excluded.interval_seconds,
                digest_enabled = excluded.digest_enabled,
                enabled_at = CASE
                    WHEN digest_config.digest_enabled = 1 AND excluded.digest_enabled = 1
                    THEN digest_config.enabled_at
                    ELSE excluded.enabled_at
                END,
                updated_by_id = excluded.updated_by_id,
                updated_at = excluded.updated_at
            """,
            (guild_id, channel_id, interval_seconds, int(enabled), now, updated_by_id, now),
        )
        await conn.commit()


async def get_digest_config(
    conn: aiosqlite.Connection, *, guild_id: int
) -> DigestConfig | None:
    """Return one guild's digest settings, or None if it has never configured any.

    Returns a disabled row as a row, not as None: the slash command needs to
    tell "never set up" (where it must ask for a channel) apart from "set up and
    switched off" (where the previous channel and interval are still the
    sensible thing to switch back on).
    """
    async with connection_lock(conn):
        async with conn.execute(
            f"SELECT {_CONFIG_COLUMNS} FROM digest_config WHERE guild_id = ?",
            (guild_id,),
        ) as cursor:
            row = await cursor.fetchone()
    return None if row is None else _row_to_config(row)


async def get_enabled_digest_configs(conn: aiosqlite.Connection) -> list[DigestConfig]:
    """Return every guild whose digest is switched on and configured sanely.

    The scheduler's one read, and deliberately the only place a guild's digest
    settings enter the scheduling path: a guild that is off, or that has no row
    at all, is simply absent from the result rather than being represented and
    then filtered out somewhere downstream where the filter could be forgotten.

    A row with an out-of-range interval is dropped with a warning rather than
    clamped or defaulted. It cannot be produced by set_digest_config, so its
    only source is a hand-edited database -- and inventing a schedule for it
    would post unprompted messages on a cadence no moderator ever chose, which
    is worse than not posting. The warning names the guild so the row can be
    found and fixed.

    Not guild-scoped, unlike every other read in this project: the scheduler is
    the one caller, it runs for the whole process rather than on behalf of one
    server's moderator, and its whole job is asking "which guilds are due". Each
    config it returns is then acted on strictly within its own guild.
    """
    async with connection_lock(conn):
        async with conn.execute(
            f"SELECT {_CONFIG_COLUMNS} FROM digest_config WHERE digest_enabled = 1 "
            "ORDER BY guild_id",
        ) as cursor:
            rows = await cursor.fetchall()

    configs: list[DigestConfig] = []
    for row in rows:
        config = _row_to_config(row)
        if not MIN_INTERVAL_SECONDS <= config.interval_seconds <= MAX_INTERVAL_SECONDS:
            logger.warning(
                "Skipping the digest for guild %s: its stored interval of %s second(s) is "
                "outside the accepted range (%s-%s). Fix it with /aura-digest.",
                config.guild_id,
                config.interval_seconds,
                MIN_INTERVAL_SECONDS,
                MAX_INTERVAL_SECONDS,
            )
            continue
        configs.append(config)
    return configs
