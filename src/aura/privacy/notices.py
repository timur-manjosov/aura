"""The one-time notice in a channel whose messages Aura starts reading (P7a, R7).

When a moderator switches automatic fact capture on for a channel and
PRIVACY_INFO_ENABLED is on, Aura posts one short public message there: from now
on it reads new messages in this channel to suggest facts, moderators confirm
each one, and /aura-privacy says what is stored and how to delete it. Once per
channel, ever: the moment is recorded in `extraction_channel_config` and the
notice is not repeated when capture is switched off and on again.

The mark is claimed BEFORE the post (a guarded UPDATE), so two moderators
switching the same channel on at once post one notice, not two; a post that
fails gives the mark back so the next switch tries again.

Imports `aura.db.connection` and `aura.i18n`.
"""

from __future__ import annotations

import logging
from typing import Protocol

import aiosqlite
import discord

from aura.db.connection import connection_lock, utc_now_iso
from aura.i18n import t

logger = logging.getLogger(__name__)


class NoticeChannel(Protocol):
    """The part of a Discord channel the notice needs."""

    id: int

    async def send(self, content: str, *, allowed_mentions: discord.AllowedMentions) -> object:
        """Post a message."""
        ...


async def _claim(conn: aiosqlite.Connection, channel_id: int) -> bool:
    async with connection_lock(conn):
        cursor = await conn.execute(
            """
            UPDATE extraction_channel_config SET privacy_notice_posted_at = ?
            WHERE channel_id = ? AND extraction_enabled = 1 AND privacy_notice_posted_at IS NULL
            """,
            (utc_now_iso(), channel_id),
        )
        await conn.commit()
    return cursor.rowcount == 1


async def _release(conn: aiosqlite.Connection, channel_id: int) -> None:
    async with connection_lock(conn):
        await conn.execute(
            "UPDATE extraction_channel_config SET privacy_notice_posted_at = NULL WHERE channel_id = ?",
            (channel_id,),
        )
        await conn.commit()


async def post_capture_notice_once(
    conn: aiosqlite.Connection, channel: NoticeChannel, *, locale: str
) -> bool:
    """Post the capture notice in a channel unless it was posted there before.

    Parameters
    ----------
    conn
        The main database; the channel's capture switch must already be on.
    channel
        Where to post.
    locale
        The server's language (the notice is public).

    Returns
    -------
    bool
        True if the notice was posted now.

    Notes
    -----
    Never raises for a failed post: the switch the moderator asked for is
    already saved, and a missing notice must not turn that into an error. The
    failure is logged (no content) and the mark released for the next attempt.
    """
    if not await _claim(conn, channel.id):
        return False
    try:
        await channel.send(
            t("privacy_capture_notice", locale), allowed_mentions=discord.AllowedMentions.none()
        )
    except Exception:
        logger.warning(
            "Could not post the capture notice in channel %s; it is tried again the next time "
            "capture is switched on there",
            channel.id,
            exc_info=True,
        )
        await _release(conn, channel.id)
        return False
    return True
