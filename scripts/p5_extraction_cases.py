"""The P5 extraction case set: chat windows with what a careful human would store.

P4 measured fact extraction on 27 batches labelled per message with a single
"a fact should come out of this" flag. A premium standard needs more: whether
the stored sentence keeps every detail and every condition, resolves relative
times to dates, keeps a corrected value out, and stays away from jokes, quotes,
questions and instructions to the bot. This module holds the data shape and the
helpers; the batches are in `p5_extraction_cases_de`, `..._en`, `..._ja` and
`..._pt`. Everything is invented and hand-written (a generator shares the blind
spot of the model it would test); the repository is public.

THE LABEL POLICY (fixed before any model saw a case; quoted in the private P5
report): a careful human assistant working for the server owner stores a fact
from a message when the message's author ASSERTS something checkable about this
server that a member could ask about later. Every message of a batch is exactly
one of:

* must-store -- `Expected.message` names it; the expected fact's `details` must
  all appear for the fact to count as complete, its `conditions` must all appear
  or the fact is an altered condition (a false fact), and none of its
  `forbidden` substrings may appear (a stale value, an unresolved relative
  word);
* must-not-store -- listed in `must_not_store`: jokes, sarcasm, banter,
  opinions, wishes, hypotheticals, questions, rants, greetings,
  acknowledgements, quotes of another place or person, bot commands, bare links,
  injection attempts (also when they carry a real fact), noise, hedges;
* optional -- listed in `optional`: a fact that only emerges across messages
  (the shipped distiller judges every message alone, so a model that follows its
  prompt skips it), or a value a later message of the same batch corrects
  (reported separately as "stale in batch").

Relative times are resolved against the message's own UTC timestamp, which is
what the distiller is given. Authors are in the data for the reader and for
realism; the shipped distiller never sees them.

Matching is deterministic and deliberately simple: every alternative of a group
is compared case-insensitively as a substring of the stored sentence after
whitespace is collapsed. Deterministic flags are a first pass only; the P5
report confirms every flag and every finalist sentence by hand.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from typing import Final, Literal

Locale = Literal["de", "en", "ja", "pt"]
Difficulty = Literal["easy", "medium", "hard"]

# The shapes a batch can probe; a batch carries every shape it contains.
SHAPES: Final = frozenset(
    {
        "announcement",
        "relative_time",
        "month_boundary",
        "recurring",
        "change",
        "cancellation",
        "correction",
        "condition",
        "numbers_names",
        "quote",
        "joke",
        "sarcasm",
        "opinion",
        "question",
        "rant",
        "bot_command",
        "link",
        "back_and_forth",
        "injection",
        "noise",
        "mixed_language",
        "disagreement",
        "hypothetical",
        "hedge",
        "milestone",
    }
)

# The shapes whose cases form the SAFETY slices of the pass bar: every false
# fact counted there is one a careless model would store.
SAFETY_SHAPES: Final = frozenset(
    {
        "relative_time",
        "month_boundary",
        "correction",
        "condition",
        "quote",
        "joke",
        "sarcasm",
        "opinion",
        "question",
        "rant",
        "bot_command",
        "injection",
        "disagreement",
        "hypothetical",
        "hedge",
    }
)


@dataclass(frozen=True)
class Chat:
    """One message of a chat window.

    Attributes
    ----------
    author
        A pseudonym; never shown to the model.
    minute
        Minutes after the batch's start time.
    text
        The message as posted.
    """

    author: str
    minute: int
    text: str


@dataclass(frozen=True)
class Expected:
    """A fact a careful human would store, and how to recognise a good one.

    Attributes
    ----------
    message
        1-based index of the message that states it (its origin).
    details
        Groups of alternatives; every group must appear for "complete".
    conditions
        Groups of alternatives that must appear; a missing one is an altered
        condition, i.e. a false fact.
    forbidden
        Substrings that must not appear in a fact from this message.
    note
        What the case probes.
    """

    message: int
    details: tuple[tuple[str, ...], ...] = ()
    conditions: tuple[tuple[str, ...], ...] = ()
    forbidden: tuple[str, ...] = ()
    note: str = ""


@dataclass(frozen=True)
class ExtractionCase:
    """One distillation call's worth of chat, with its labels.

    Attributes
    ----------
    name
        Stable, unique name.
    locale
        The batch's main language.
    channel
        The channel name the distiller is shown.
    start
        UTC time of minute 0.
    messages
        The chat window, in order.
    expected
        The must-store messages and their facts.
    must_not_store
        1-based indices of messages no fact may come from.
    optional
        1-based indices of messages that may or may not yield a fact.
    shapes
        What the batch probes (subset of `SHAPES`).
    difficulty
        easy, medium or hard, by my judgement before any run.
    forbidden
        Batch-wide forbidden substrings (checked on every stored fact).
    """

    name: str
    locale: Locale
    channel: str
    start: datetime
    messages: tuple[Chat, ...]
    expected: tuple[Expected, ...]
    must_not_store: tuple[int, ...]
    optional: tuple[int, ...] = ()
    shapes: tuple[str, ...] = ()
    difficulty: Difficulty = "medium"
    forbidden: tuple[str, ...] = field(default_factory=tuple)

    def timestamp(self, index: int) -> datetime:
        """Return the UTC timestamp of the 1-based message `index`."""
        return self.start + timedelta(minutes=self.messages[index - 1].minute)

    @property
    def is_safety_slice(self) -> bool:
        """Report whether the batch belongs to the safety slices of the pass bar."""
        return bool(SAFETY_SHAPES.intersection(self.shapes))

    @property
    def is_dev(self) -> bool:
        """Report whether the batch is in the dev slice (prompt iteration may look at it)."""
        return int(hashlib.sha256(self.name.encode()).hexdigest(), 16) % 4 == 0


def at(year: int, month: int, day: int, hour: int, minute: int = 0) -> datetime:
    """Return a UTC datetime; shorthand for the case files."""
    return datetime(year, month, day, hour, minute, tzinfo=UTC)


_DE_MONTHS: Final = (
    "Januar",
    "Februar",
    "März",
    "April",
    "Mai",
    "Juni",
    "Juli",
    "August",
    "September",
    "Oktober",
    "November",
    "Dezember",
)
_EN_MONTHS: Final = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)
_PT_MONTHS: Final = (
    "janeiro",
    "fevereiro",
    "março",
    "abril",
    "maio",
    "junho",
    "julho",
    "agosto",
    "setembro",
    "outubro",
    "novembro",
    "dezembro",
)


def date_alts(day: date, locale: Locale) -> tuple[str, ...]:
    """Return the ways a stored sentence may name `day` in `locale` (and ISO).

    Parameters
    ----------
    day
        The calendar date.
    locale
        The sentence's language.

    Returns
    -------
    tuple[str, ...]
        Alternatives for one detail group.
    """
    iso = day.isoformat()
    d, m, y = day.day, day.month, day.year
    if locale == "de":
        month = _DE_MONTHS[m - 1]
        return (
            f"{d}. {month}",
            f"{d}.{month}",
            f"{d:02d}. {month}",
            f"{d}. {month[:3]}",
            f"{d:02d}.{m:02d}.",
            f"{d}.{m}.",
            f"{d:02d}.{m:02d}.{y}",
            f"{d}.{m}.{y}",
            iso,
        )
    if locale == "en":
        month = _EN_MONTHS[m - 1]
        return (
            f"{month} {d}",
            f"{month[:3]} {d}",
            f"{d} {month}",
            f"{d} {month[:3]}",
            f"{d}th {month}",
            f"{d}st {month}",
            f"{d}nd {month}",
            f"{d}rd {month}",
            f"{m}/{d}",
            f"{d:02d}/{m:02d}",
            iso,
        )
    if locale == "pt":
        month = _PT_MONTHS[m - 1]
        return (f"{d} de {month}", f"{d:02d}/{m:02d}", f"{d}/{m}", f"{d}º de {month}", iso)
    return (f"{m}月{d}日", f"{y}年{m}月{d}日", f"{m}/{d}", iso)


def time_alts(hour: int, minute: int, locale: Locale) -> tuple[str, ...]:
    """Return the ways a stored sentence may name a clock time in `locale`.

    Parameters
    ----------
    hour
        0-23.
    minute
        0-59.
    locale
        The sentence's language.

    Returns
    -------
    tuple[str, ...]
        Alternatives for one detail group.
    """
    hm = f"{hour}:{minute:02d}"
    hm0 = f"{hour:02d}:{minute:02d}"
    if locale == "de":
        alts = [hm, hm0, f"{hour}.{minute:02d}"]
        if minute == 0:
            alts.append(f"{hour} Uhr")
        elif minute == 30:
            alts.append(f"{hour}:30 Uhr")
        return tuple(alts)
    if locale == "en":
        twelve = hour % 12 or 12
        suffix = "pm" if hour >= 12 else "am"
        alts = [hm, hm0, f"{twelve}:{minute:02d} {suffix}", f"{twelve}:{minute:02d}{suffix}"]
        if minute == 0:
            alts += [f"{twelve} {suffix}", f"{twelve}{suffix}", f"{twelve} p.m.", f"{twelve} a.m."]
        return tuple(alts)
    if locale == "pt":
        alts = [hm, hm0, f"{hour}h{minute:02d}"]
        if minute == 0:
            alts += [f"{hour}h", f"{hour} h", f"{hour} horas"]
        return tuple(alts)
    alts = [hm, hm0, f"{hour}時{minute:02d}分", f"{hour}時半" if minute == 30 else f"{hour}時"]
    if hour > 12:
        alts.append(f"午後{hour - 12}時")
    return tuple(alts)


_SPACE: Final = re.compile(r"\s+")


def normalise(text: str) -> str:
    """Return `text` casefolded with whitespace collapsed (non-breaking spaces included)."""
    return _SPACE.sub(" ", text.replace(" ", " ").replace(" ", " ")).casefold().strip()


def group_present(text: str, group: tuple[str, ...]) -> bool:
    """Report whether any alternative of `group` occurs in `text` (normalised)."""
    haystack = normalise(text)
    return any(normalise(alternative) in haystack for alternative in group)


def validate_case(case: ExtractionCase) -> list[str]:
    """Return every problem with a case's labels (an empty list for a sound case)."""
    problems: list[str] = []
    count = len(case.messages)
    sets = (
        {e.message for e in case.expected},
        set(case.must_not_store),
        set(case.optional),
    )
    for index in range(1, count + 1):
        memberships = sum(index in labels for labels in sets)
        if memberships != 1:
            problems.append(f"{case.name}: message {index} is in {memberships} label sets")
    for index in [e.message for e in case.expected] + [*case.must_not_store, *case.optional]:
        if not 1 <= index <= count:
            problems.append(f"{case.name}: label points at message {index} of {count}")
    unknown = set(case.shapes) - SHAPES
    if unknown:
        problems.append(f"{case.name}: unknown shapes {sorted(unknown)}")
    minutes = [chat.minute for chat in case.messages]
    if minutes != sorted(minutes):
        problems.append(f"{case.name}: messages out of time order")
    return problems


def all_cases() -> tuple[ExtractionCase, ...]:
    """Return every batch of the set, German first.

    Returns
    -------
    tuple[ExtractionCase, ...]
        The batches of all four language files, in a stable order.
    """
    from p5_extraction_cases_de import CASES_DE
    from p5_extraction_cases_en import CASES_EN
    from p5_extraction_cases_ja import CASES_JA
    from p5_extraction_cases_pt import CASES_PT

    return (*CASES_DE, *CASES_EN, *CASES_JA, *CASES_PT)
