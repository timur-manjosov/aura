"""The closed set of digest intervals a moderator can choose, and how to name one.

A separate module from both the slash command that offers these choices and the
embed that names the chosen one, because both need the same vocabulary and
neither owns it -- the same reasoning aura.db.pending_facts gives for keeping
FactCategory beside the table that persists it rather than in the package that
produces it.

Deliberately a closed set rather than a free-form number of days. A digest is an
unprompted post, and an interval is the only thing standing between "a weekly
summary" and "a message every few minutes"; offering four named cadences means
the value written to the database is always one a human recognised, and the
range check in aura.db.digest_config only ever has to defend against a
hand-edited row rather than against a typo in a slash command.
"""
from __future__ import annotations

from enum import IntEnum

from aura.i18n import t

_SECONDS_PER_DAY = 24 * 60 * 60


class DigestInterval(IntEnum):
    """How often a guild's digest is posted, in seconds.

    An IntEnum rather than a StrEnum with a lookup table beside it: the value IS
    the number of seconds stored in digest_config.interval_seconds, so there is
    no mapping that could go out of step with the column. The member NAME
    doubles as the translation-key suffix (see describe_interval), which is what
    keeps adding a fifth cadence a change to this enum plus nine locale strings
    and nothing else.

    MONTHLY is 30 days, not "the calendar month". A digest interval is a
    duration measured against a stored timestamp, and calendar months are 28 to
    31 days long, so a real month would make the cadence drift by up to three
    days a year and make "is this guild due" depend on which month it is.
    """

    DAILY = _SECONDS_PER_DAY
    WEEKLY = 7 * _SECONDS_PER_DAY
    BIWEEKLY = 14 * _SECONDS_PER_DAY
    MONTHLY = 30 * _SECONDS_PER_DAY


def describe_interval(interval_seconds: int, locale: str) -> str:
    """Name an interval in the reader's language: "weekly", "alle zwei Wochen", ...

    Falls back to a localized "every N seconds" for a value that matches no
    member, which is unreachable through the slash command and reachable only
    through a hand-edited database. Reporting the raw number rather than
    rounding it to the nearest named cadence is deliberate: a moderator reading
    "weekly" about a row that actually says 900 seconds would have no way to
    discover why digests keep arriving, whereas the raw number points straight
    at the row that needs fixing.
    """
    for interval in DigestInterval:
        if interval.value == interval_seconds:
            return t(f"digest_interval_{interval.name.lower()}", locale)
    return t("digest_interval_custom", locale, seconds=interval_seconds)
