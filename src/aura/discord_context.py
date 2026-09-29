"""Best-effort human-readable channel names for LLM prompt context.

Both /aura-ask and proactive relief (see `aura.commands.ask`,
`aura.proactive.responder`) resolve a channel to a name for the synthesis
prompt's question-channel and per-fact-channel context (see `aura.synthesis`'s
`question_channel_name` and `fact_channel_names` parameters). The graceful ID
fallback `aura.extraction.pipeline._channel_name` already uses, for the same
reason, lives here once instead of being copied at each call site.

Invariant this module maintains
-------------------------------
A resolution never fails and never returns an empty string. A channel name is
prompt context, never an identifier anything depends on, and a channel with no
usable name -- a partial object, an uncached lookup, or one deleted or renamed
since a fact was recorded -- must never be the reason an answer is not
produced. Callers therefore never have to special-case a missing entry.

Imports `discord` for the guild type only; it performs no I/O and reads only
discord.py's local cache.
"""

from __future__ import annotations

import discord


def channel_display_name(channel: object, channel_id: int) -> str:
    """Return a human-readable name for `channel`, falling back to its ID.

    Parameters
    ----------
    channel
        Any object that may carry a `name` attribute, or None. Typed as
        `object` rather than a discord.py union because the point is to
        tolerate whatever the cache returns, including a partial object.
    channel_id
        The channel's ID, used as the fallback label.

    Returns
    -------
    str
        The channel's name, or `channel_id` rendered as a string when the
        object has no name, has a blank one, or is None.
    """
    name = getattr(channel, "name", None)
    return str(name) if name else str(channel_id)


def fact_channel_names(guild: discord.Guild | None, channel_ids: set[int]) -> dict[int, str]:
    """Resolve each of `channel_ids` against `guild`'s channel cache.

    Parameters
    ----------
    guild
        The guild whose cache to read, or None when the guild itself is
        unknown. None is a supported input, not an error: every ID then
        resolves to its own string form.
    channel_ids
        The distinct channel IDs to resolve.

    Returns
    -------
    dict[int, str]
        One entry per requested ID, always. See `channel_display_name` for the
        per-entry fallback.

    Notes
    -----
    Keyed by channel ID rather than by fact ID because several facts routinely
    share one channel; a fact-keyed mapping would carry one entry per fact for
    the same repeated name.
    """
    return {
        channel_id: channel_display_name(
            guild.get_channel(channel_id) if guild is not None else None, channel_id
        )
        for channel_id in channel_ids
    }
