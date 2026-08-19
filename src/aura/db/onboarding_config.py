"""Per-guild configuration for onboarding (CLAUDE.md's THIRD trigger): which
channel a new member's summary of the current knowledge model goes to, and
whether it runs at all.

A deliberate sibling of aura.db.digest_config -- same opt-in-by-default
invariant, same upsert shape, same "who changed it and when" audit columns,
and the same structural choice of being keyed by guild rather than by channel
(see schema.sql above the table for the reasoning, which is identical to
digest_config's: "where does this go" is one question per server with exactly
one answer). Simpler than digest_config in one respect: onboarding carries no
interval and no baseline timestamp, because it does not summarize a *window*
of change -- every onboarding message reflects the full active knowledge model
at the moment a member joins, so there is no "since when" to anchor.

**A guild with no row is OFF.** get_onboarding_config returning None is the
only signal the join handler needs -- there is no code path where an
unconfigured guild has to be recognised as a special case.

This module owns exactly one table and imports nothing from
aura.db.repository, for the same isolation reason its siblings give: a
configuration switch is none of the four things CLAUDE.md admits into the
knowledge model.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime

import aiosqlite
from pydantic import BaseModel

from aura.db.connection import connection_lock, utc_now_iso

_CONFIG_COLUMNS = "guild_id, channel_id, onboarding_enabled, updated_by_id, updated_at"


class OnboardingConfig(BaseModel):
    """One guild's onboarding settings, as read back from the database."""

    guild_id: int
    channel_id: int
    onboarding_enabled: bool
    updated_by_id: int
    updated_at: datetime


def _row_to_config(row: sqlite3.Row) -> OnboardingConfig:
    return OnboardingConfig(
        guild_id=row[0],
        channel_id=row[1],
        onboarding_enabled=bool(row[2]),
        updated_by_id=row[3],
        updated_at=row[4],
    )


async def set_onboarding_config(
    conn: aiosqlite.Connection,
    *,
    guild_id: int,
    channel_id: int,
    enabled: bool,
    updated_by_id: int,
) -> None:
    """Write one guild's onboarding settings, recording who changed them.

    An upsert keyed on guild_id, the same shape digest_config uses:
    reconfiguring the same guild repeatedly leaves exactly one row, always
    reflecting the most recent decision.
    """
    now = utc_now_iso()
    async with connection_lock(conn):
        await conn.execute(
            """
            INSERT INTO onboarding_config
                (guild_id, channel_id, onboarding_enabled, updated_by_id, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT (guild_id) DO UPDATE SET
                channel_id = excluded.channel_id,
                onboarding_enabled = excluded.onboarding_enabled,
                updated_by_id = excluded.updated_by_id,
                updated_at = excluded.updated_at
            """,
            (guild_id, channel_id, int(enabled), updated_by_id, now),
        )
        await conn.commit()


async def get_onboarding_config(
    conn: aiosqlite.Connection, *, guild_id: int
) -> OnboardingConfig | None:
    """Return one guild's onboarding settings, or None if it has never configured any.

    Returns a disabled row as a row, not as None: the slash command needs to
    tell "never set up" (where it must ask for a channel) apart from "set up
    and switched off" (where the previous channel is still the sensible thing
    to switch back on) -- exactly as get_digest_config does.
    """
    async with connection_lock(conn):
        async with conn.execute(
            f"SELECT {_CONFIG_COLUMNS} FROM onboarding_config WHERE guild_id = ?",
            (guild_id,),
        ) as cursor:
            row = await cursor.fetchone()
    return None if row is None else _row_to_config(row)
