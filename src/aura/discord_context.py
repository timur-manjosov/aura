"""Best-effort human-readable channel names for LLM prompt context.

Both /aura-ask and proactive relief (see aura.commands.ask,
aura.proactive.responder) resolve a channel to a name for the synthesis
prompt's question-channel and per-fact-channel context (see
aura.synthesis's `question_channel_name` and `fact_channel_names`
parameters). The same graceful ID fallback aura.extraction.pipeline's
_channel_name already uses, for the same reason, lives here once instead of
being copied at each call site: a name is prompt context, never an
identifier anything depends on, and a channel with no name -- a partial
object, an uncached lookup, or one deleted or renamed since a fact was
recorded -- must never be why a call fails.
"""
from __future__ import annotations

import discord


def channel_display_name(channel: object, channel_id: int) -> str:
    """A human-readable name for `channel`, or `channel_id` as a string if unavailable."""
    name = getattr(channel, "name", None)
    return str(name) if name else str(channel_id)


def fact_channel_names(guild: discord.Guild | None, channel_ids: set[int]) -> dict[int, str]:
    """Resolve each of `channel_ids` against `guild`'s channel cache.

    Returns an entry for every ID regardless of whether the guild is known or
    a given lookup succeeds -- see channel_display_name's own fallback --
    so a caller building a synthesis prompt never has to special-case a
    missing key.
    """
    return {
        channel_id: channel_display_name(
            guild.get_channel(channel_id) if guild is not None else None, channel_id
        )
        for channel_id in channel_ids
    }
