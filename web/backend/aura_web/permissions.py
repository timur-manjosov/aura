"""Deciding whether a user may manage a guild, from Discord's permission bitmask.

Pure functions over the raw JSON Discord sends, with no HTTP, no session and
no framework in sight -- the same separation CLAUDE.md's Testing section asks
for between fact-extraction logic and a live Discord connection, applied to
the one piece of authorization logic this service has. Every hostile shape
this module rejects is a unit test rather than a live OAuth round trip.
"""
from __future__ import annotations

# Discord serialises permissions as a decimal string of a variable-length
# integer (v8+), precisely so the set can outgrow 64 bits without breaking
# clients. Python's int is already arbitrary-precision, so the bit test below
# needs no special handling -- only the parsing does.
ADMINISTRATOR = 1 << 3
MANAGE_GUILD = 1 << 5

# A 64-bit permission set is at most 20 decimal digits; today's set is far
# short of that. 40 leaves generous room for Discord to extend the bitmask
# while still refusing a megabyte of digits outright -- CPython raises on
# int() past 4300 digits by default, but that is an interpreter setting a
# deployment can change, and an explicit bound here does not depend on it.
MAX_PERMISSIONS_DIGITS = 40

# Discord caps guild names at 100 characters. Truncating at the boundary
# keeps one malformed or hostile API response from turning into an unbounded
# string that this service then hands to a browser.
MAX_GUILD_NAME_LENGTH = 100

# Snowflakes are decimal strings; 20 digits covers the full 64-bit range with
# room to spare.
MAX_SNOWFLAKE_DIGITS = 20


def parse_permissions(raw: object) -> int | None:
    """Parse Discord's permission field into a non-negative int, or None if unusable.

    Returns None rather than raising, and rather than defaulting to zero,
    so the caller can distinguish "Discord said this user has no permissions"
    from "this value made no sense" -- the two deserve different log lines
    even though both must deny access.

    Accepts the documented string form and, defensively, a plain int: older
    API versions and some intermediaries emit the latter, and a type change
    upstream must not read as "nobody can manage anything."

    A negative value is rejected explicitly rather than being masked. In
    Python ``-1 & MANAGE_GUILD`` is ``MANAGE_GUILD`` because integers behave
    as infinite two's-complement -- so a "-1" arriving from a broken proxy
    would otherwise grant every permission that exists, which is the exact
    inversion of a safe failure.
    """
    if isinstance(raw, bool):
        # bool is an int subclass; True would parse as permission bit 0.
        return None
    if isinstance(raw, int):
        return raw if raw >= 0 else None
    if not isinstance(raw, str):
        return None

    candidate = raw.strip()
    if not candidate or len(candidate) > MAX_PERMISSIONS_DIGITS:
        return None
    # str.isdigit() is True for non-ASCII digits ('٣', '１') that int() also
    # accepts, which is harmless here but means this check is about rejecting
    # signs, separators and whitespace rather than about narrowing to ASCII.
    if not candidate.isdigit():
        return None
    try:
        value = int(candidate)
    except ValueError:  # pragma: no cover -- isdigit() already guarantees this parses
        return None
    return value if value >= 0 else None


def has_manage_guild(permissions: object) -> bool:
    """Whether this permission bitmask grants management of the guild.

    ADMINISTRATOR counts. Discord's computed bitmask does not fold the
    administrator grant into the other bits -- an administrator can have
    MANAGE_GUILD unset while being able to do everything the permission
    allows -- so checking MANAGE_GUILD alone would lock server owners and
    admins out of their own dashboard.
    """
    parsed = parse_permissions(permissions)
    if parsed is None:
        return False
    return bool(parsed & (ADMINISTRATOR | MANAGE_GUILD))


def parse_snowflake(raw: object) -> str | None:
    """Normalise a Discord ID to its decimal string form, or None if unusable.

    Kept as a string rather than an int throughout this service: IDs are only
    ever compared and displayed, never arithmetic, and JavaScript loses
    precision on integers past 2^53 -- which every Discord snowflake exceeds.
    """
    if isinstance(raw, bool):
        return None
    if isinstance(raw, int):
        return str(raw) if raw >= 0 else None
    if not isinstance(raw, str):
        return None
    candidate = raw.strip()
    if not candidate or len(candidate) > MAX_SNOWFLAKE_DIGITS or not candidate.isdigit():
        return None
    # Round-tripping through int normalises exotic digit forms and leading
    # zeros, so two spellings of one ID cannot look like two different guilds.
    return str(int(candidate))


def sanitize_guild_name(raw: object, fallback: str) -> str:
    """Clamp a guild name to something safe to store and hand to a browser.

    Control characters are stripped rather than escaped: they carry no
    meaning in a server name, and a name containing a line break or a
    bidirectional override is a display problem in every consumer this value
    reaches, not just the one being written today.
    """
    if not isinstance(raw, str):
        return fallback
    cleaned = "".join(
        character for character in raw if character.isprintable() or character == " "
    ).strip()
    if not cleaned:
        return fallback
    return cleaned[:MAX_GUILD_NAME_LENGTH]


def sanitize_icon_hash(raw: object) -> str | None:
    """Validate an icon hash, or None if it is absent or not a plain hash.

    The value is interpolated into a CDN path
    (``/icons/{guild_id}/{icon}.png``) by whoever renders it, so anything
    beyond the documented alphabet -- a slash, a dot, a query string -- turns
    a missing icon into a request at an attacker-chosen path. Validating here
    means no consumer has to remember to.
    """
    if not isinstance(raw, str):
        return None
    candidate = raw.strip()
    if not candidate or len(candidate) > 64:
        return None
    if not all(character.isascii() and (character.isalnum() or character == "_") for character in candidate):
        return None
    return candidate
