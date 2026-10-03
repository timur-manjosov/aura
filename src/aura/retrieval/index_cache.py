"""A bounded, thread-safe memo of each guild's lexical index.

Building a guild's `LexicalIndex` tokenizes every active fact; scoring one
question against a built index takes well under a millisecond. So the index is
built once per version of a guild's facts and kept in memory until those facts
change.

**Keyed by the facts themselves, not by a guess at whether they changed.** An
entry is stored under a digest of every (fact ID, content) pair it was built
from, and is used only for a request whose current active facts produce the
same digest. Any change at all -- a fact added, superseded, deleted, its text
edited by hand in the database, a fact turned active again -- changes the
digest, so a stale index cannot be returned; there is no hook to forget and no
counter to drift. The price is hashing the guild's fact texts once per
question, about a millisecond at 2,000 facts, against facts the caller has
just read from the database anyway.

**Bounded.** At most `max_guilds` guilds and `max_bytes` of estimated index
memory are held, least recently used evicted first. A single guild whose index
alone exceeds the byte budget is still indexed and scored -- its facts are never
dropped -- but not kept, and that is logged once per guild per process.

Imports aura.retrieval.lexical and the standard library only.
"""

from __future__ import annotations

import hashlib
import logging
import threading
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from aura.retrieval.lexical import LexicalIndex

logger = logging.getLogger(__name__)

# How many guilds' indexes are kept. Far above the handful of guilds Aura
# serves today; the byte budget below is the bound that matters for memory,
# this one keeps the bookkeeping small however many tiny guilds exist.
DEFAULT_MAX_GUILDS: Final = 256

# How much estimated index memory is kept in total. Measured with tracemalloc:
# an index grows more slowly than its facts, because the n-grams a language
# uses are limited -- 4 MB for 2,000 facts of entirely distinct random words,
# 7 MB for 10,000 of them, 0.4 MB for 2,000 facts of ordinary, overlapping
# text. 64 MB therefore holds every guild of realistic size many times over,
# and bounds the worst case at a fraction of what the embedding model itself
# occupies.
DEFAULT_MAX_BYTES: Final = 64 * 1024 * 1024

# How many guild IDs the "too large to cache" note remembers, so a process
# running for months cannot grow this set without bound.
_MAX_REMEMBERED_UNCACHED_GUILDS: Final = 1024

# Digest size. 16 bytes makes an accidental collision between two versions of
# one guild's facts practically impossible (about 2**-64 even across billions
# of versions); the key is never exposed, so nobody can search for one.
_DIGEST_BYTES: Final = 16


def fact_set_digest(facts: Sequence[tuple[int, str]]) -> bytes:
    """Return a digest identifying one exact set of (fact ID, content) pairs.

    Parameters
    ----------
    facts
        The pairs, in any order.

    Returns
    -------
    bytes
        A BLAKE2b digest of the pairs sorted by ID, each framed by its ID and
        its content's length, so no two different sets share a byte stream.
    """
    hasher = hashlib.blake2b(digest_size=_DIGEST_BYTES)
    for fact_id, content in sorted(facts):
        encoded = content.encode("utf-8", "surrogatepass")
        hasher.update(fact_id.to_bytes(8, "big", signed=True))
        hasher.update(len(encoded).to_bytes(8, "big"))
        hasher.update(encoded)
    return hasher.digest()


@dataclass(frozen=True, slots=True)
class _Entry:
    digest: bytes
    index: LexicalIndex


@dataclass(frozen=True, slots=True)
class CacheStatistics:
    """What the cache has done since it was created.

    Attributes
    ----------
    hits
        Requests served from a stored index.
    builds
        Indexes built because none matched the request's facts.
    evictions
        Indexes dropped to stay within the bounds.
    guilds
        Guilds currently held.
    cached_bytes
        Estimated memory of the held indexes.
    """

    hits: int
    builds: int
    evictions: int
    guilds: int
    cached_bytes: int


class LexicalIndexCache:
    """One lexical index per guild, rebuilt whenever the guild's active facts differ.

    Parameters
    ----------
    max_guilds
        How many guilds to keep. At least 1.
    max_bytes
        How much estimated index memory to keep in total. At least 1.

    Raises
    ------
    ValueError
        If either bound is below 1.

    Notes
    -----
    Safe to call from several threads at once: the bookkeeping is done under a
    lock, the build outside it. Two concurrent requests for the same new
    version may both build it; both results are correct and the second simply
    replaces the first. A request still holding an older version of the facts
    may store its index after a newer one was stored; the next request with the
    newer facts then sees a different digest and rebuilds -- one wasted build,
    never a stale answer.
    """

    def __init__(
        self, *, max_guilds: int = DEFAULT_MAX_GUILDS, max_bytes: int = DEFAULT_MAX_BYTES
    ) -> None:
        if max_guilds < 1 or max_bytes < 1:
            raise ValueError("both cache bounds must be at least 1")
        self._max_guilds = max_guilds
        self._max_bytes = max_bytes
        self._entries: OrderedDict[int, _Entry] = OrderedDict()
        self._cached_bytes = 0
        self._hits = 0
        self._builds = 0
        self._evictions = 0
        self._noted_uncached_guilds: set[int] = set()
        self._lock = threading.Lock()

    def index_for(self, guild_id: int, facts: Sequence[tuple[int, str]]) -> LexicalIndex:
        """Return the index of exactly these facts, built now or kept from before.

        Parameters
        ----------
        guild_id
            The guild the facts belong to.
        facts
            The guild's current active facts as (fact ID, content) pairs.

        Returns
        -------
        LexicalIndex
            An index built from exactly `facts`.
        """
        digest = fact_set_digest(facts)
        with self._lock:
            entry = self._entries.get(guild_id)
            if entry is not None and entry.digest == digest:
                self._entries.move_to_end(guild_id)
                self._hits += 1
                return entry.index

        index = LexicalIndex.build(facts)

        with self._lock:
            self._builds += 1
            self._forget(guild_id)
            if index.estimated_bytes > self._max_bytes:
                self._note_uncached(guild_id, index.estimated_bytes)
                return index
            self._entries[guild_id] = _Entry(digest, index)
            self._cached_bytes += index.estimated_bytes
            while len(self._entries) > self._max_guilds or self._cached_bytes > self._max_bytes:
                self._forget(next(iter(self._entries)))
                self._evictions += 1
        return index

    def statistics(self) -> CacheStatistics:
        """Return a consistent snapshot of the cache's counters.

        Returns
        -------
        CacheStatistics
            Counters and current size, read under the lock.
        """
        with self._lock:
            return CacheStatistics(
                hits=self._hits,
                builds=self._builds,
                evictions=self._evictions,
                guilds=len(self._entries),
                cached_bytes=self._cached_bytes,
            )

    def clear(self) -> None:
        """Drop every kept index. The counters are kept."""
        with self._lock:
            self._entries.clear()
            self._cached_bytes = 0

    def _forget(self, guild_id: int) -> None:
        """Drop one guild's entry, if any. Caller holds the lock."""
        entry = self._entries.pop(guild_id, None)
        if entry is not None:
            self._cached_bytes -= entry.index.estimated_bytes

    def _note_uncached(self, guild_id: int, estimated_bytes: int) -> None:
        """Log, once per guild, that its index is too large to keep. Caller holds the lock."""
        if guild_id in self._noted_uncached_guilds:
            return
        if len(self._noted_uncached_guilds) >= _MAX_REMEMBERED_UNCACHED_GUILDS:
            self._noted_uncached_guilds.clear()
        self._noted_uncached_guilds.add(guild_id)
        logger.warning(
            "A guild's lexical index (about %d bytes) exceeds the index cache (%d bytes): "
            "it is rebuilt for every /aura-ask question. Every fact is still searched.",
            estimated_bytes,
            self._max_bytes,
        )
