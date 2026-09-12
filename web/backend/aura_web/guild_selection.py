"""Which of a user's guilds belong on their dashboard.

Both conditions, never one: the user must hold MANAGE_GUILD (or
ADMINISTRATOR, which subsumes it) in the guild, AND Aura must actually be a
member of it. Each alone is wrong in a way somebody notices -- permission
alone lists servers Aura has never seen, membership alone lists servers the
user has no business configuring -- so the conjunction is the feature, and it
is expressed once, here, rather than as two filters that could drift apart.

A pure function over already-fetched data, deliberately: it needs no HTTP, no
session and no Discord to test, so the case that matters most (a user with no
qualifying guild sees an empty list, not an error and not somebody else's
servers) is a unit test rather than a live OAuth round trip.
"""
from __future__ import annotations

from dataclasses import dataclass

from aura_web.discord_api import PartialGuild
from aura_web.permissions import has_manage_guild


@dataclass(frozen=True)
class ManageableGuild:
    """One guild the signed-in user may configure, as the browser will see it.

    Carries no permission bitmask. The browser is told *that* a guild is
    manageable, never the raw permission integer behind the decision: the
    decision is the server's to make, and shipping the input to it invites a
    later client-side reimplementation that disagrees.
    """

    id: str
    name: str
    icon: str | None


def select_manageable_guilds(
    user_guilds: list[PartialGuild], bot_guild_ids: frozenset[str]
) -> list[ManageableGuild]:
    """Return the guilds the user may manage AND Aura is present in, name-sorted.

    Sorted by name (case-insensitively, with the ID as a tiebreaker) so two
    calls with the same inputs produce the same order regardless of how
    Discord happened to page the response -- an unstable list reshuffles the
    page under the reader for no reason.
    """
    selected = [
        ManageableGuild(id=guild.id, name=guild.name, icon=guild.icon)
        for guild in user_guilds
        if guild.id in bot_guild_ids and has_manage_guild(guild.permissions)
    ]
    return sorted(selected, key=lambda guild: (guild.name.casefold(), guild.id))
