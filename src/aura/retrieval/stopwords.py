"""The stopwords /aura-ask's word matching ignores: one data file per bot locale, applied as one set.

**The union of every locale, for every guild.** A guild's facts may be written
in any language and a question in another, and nothing records which language a
fact is in. So a word that is a function word in ANY supported locale is ignored
in every question. The cost is a word that is a function word in one language
and a subject in another; each list's header names those it left out on purpose
and those it kept anyway.

**IDF is the main protection, these lists only a safeguard.** A word that is
common among a guild's facts weighs almost nothing in a coverage whatever the
lists say (aura.retrieval.lexical.inverse_document_frequency). What the lists
catch is the case IDF cannot: a question word that appears in NO fact, which
counts at full weight in the denominator and would halve the coverage of a
two-word question ("¿Hay torneo?"). So a list holds the words that open and
frame questions, and nothing else.

**A new locale is one more file.** Every `*.txt` file in `stopword_lists/` is
read; none is named in code.

Invariants
----------
* Every word is stored normalized exactly like a question, by
  aura.retrieval.lexical.tokenize, so a list can be written with its natural
  spelling (accents, capitals) and still match a question typed without them.
* Loading never half-succeeds: a missing directory, an empty directory, or an
  unreadable or non-UTF-8 file raises `StopwordLoadError`, and nothing is
  cached until a load has fully succeeded.

Imports only aura.retrieval.lexical: no Discord, no database.
"""

from __future__ import annotations

import functools
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Final

from aura.retrieval.lexical import tokenize

STOPWORD_DIRECTORY: Final = Path(__file__).parent / "stopword_lists"
STOPWORD_FILE_SUFFIX: Final = ".txt"
_COMMENT_MARKER: Final = "#"


class StopwordLoadError(Exception):
    """The stopword lists are missing, empty or unreadable.

    Notes
    -----
    /aura-ask treats this as "word matching is unavailable" and answers from
    embedding similarity alone, exactly as before word matching existed (see
    aura.retrieval.hybrid); it never becomes an error reply.
    """


def parse_stopword_file(text: str) -> frozenset[str]:
    """Return the normalized words one stopword file lists.

    Parameters
    ----------
    text
        The file's content: words separated by whitespace, with "#" starting
        a comment that runs to the end of its line.

    Returns
    -------
    frozenset[str]
        Every token of every listed word, normalized. A listed word that
        normalization splits ("aujourd'hui") contributes each part.
    """
    words: set[str] = set()
    for line in text.splitlines():
        words.update(tokenize(line.split(_COMMENT_MARKER, 1)[0]))
    return frozenset(words)


def load_stopword_lists(directory: Path = STOPWORD_DIRECTORY) -> Mapping[str, frozenset[str]]:
    """Read every stopword file in a directory.

    Parameters
    ----------
    directory
        Where the `*.txt` files are. Defaults to the lists shipped with Aura.

    Returns
    -------
    Mapping[str, frozenset[str]]
        Normalized words per locale code (the file name without its suffix),
        read-only.

    Raises
    ------
    StopwordLoadError
        If the directory does not exist or holds no stopword file, or a file
        cannot be read as UTF-8.
    """
    try:
        paths = sorted(directory.glob(f"*{STOPWORD_FILE_SUFFIX}"))
    except OSError as error:
        raise StopwordLoadError(f"cannot list stopword files in {directory}") from error
    if not paths:
        raise StopwordLoadError(f"no stopword files (*{STOPWORD_FILE_SUFFIX}) in {directory}")
    lists: dict[str, frozenset[str]] = {}
    for path in paths:
        try:
            lists[path.stem] = parse_stopword_file(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError) as error:
            raise StopwordLoadError(f"cannot read stopword file {path.name}") from error
    return MappingProxyType(lists)


@functools.cache
def shipped_stopword_lists() -> Mapping[str, frozenset[str]]:
    """Return Aura's own stopword lists, read once per process.

    Returns
    -------
    Mapping[str, frozenset[str]]
        As `load_stopword_lists` for `STOPWORD_DIRECTORY`.

    Raises
    ------
    StopwordLoadError
        As `load_stopword_lists`. A failure is not cached, so the next call
        tries again.
    """
    return load_stopword_lists()


@functools.cache
def shipped_stopwords() -> frozenset[str]:
    """Return the union of Aura's stopword lists -- the set every question is filtered by.

    Returns
    -------
    frozenset[str]
        Every normalized word of every shipped list.

    Raises
    ------
    StopwordLoadError
        As `load_stopword_lists`. A failure is not cached.
    """
    return frozenset().union(*shipped_stopword_lists().values())
