"""Reading a channel's history safely: strict chronological order, enforced, and
a rate-limit layer over the one discord.py already has.

Two independent jobs live here, and neither belongs in the worker: the worker
decides WHAT to do with a page of history, and this module makes sure the page
it is handed is (a) actually in the order the worker assumes, and (b) actually
worth waiting for rather than hammering Discord to get.

------------------------------------------------------------------------------
ORDER IS ENFORCED, NOT ASSUMED, AND THE REASON IS NOT TIDINESS
------------------------------------------------------------------------------

Supersession judgements depend on temporal order. The whole point of Phase
3a-3's call is to decide whether a candidate is a LATER successor to an existing
fact -- and the dedup comparison behind it runs a candidate against the facts
active at the moment it is staged. Feed a channel's history in the wrong order
and an older statement is judged against a newer one as if it came second: a
rule that was raised from 3 to 5 and then lowered back to 3 would end its
backfill claiming 5, and nothing downstream could tell.

discord.py's `history(after=..., oldest_first=True)` does already return pages
oldest-first, and its own pagination is correct. That is not sufficient
justification to depend on it silently. This module sorts every page itself and
refuses anything at or below the cursor, so the guarantee is a property of
Aura's code and testable against a deliberately shuffled page -- which is what
`ordered_page` exists for and what tests/test_backfill_history.py feeds it.

Ordering by message ID and ordering by timestamp are THE SAME ORDERING here, and
that is exact rather than approximate: a Discord snowflake embeds its creation
time, and discord.py derives `Message.created_at` from the id (verified in
discord.py 2.7.1: `created_at` returns `utils.snowflake_time(self.id)`), never
the other way round. The sort key below names both anyway -- the id is what the
cursor and the API bounds are expressed in, the timestamp is what the ordering
actually means, and writing only one of them would leave a reader to work out
which.

------------------------------------------------------------------------------
RATE LIMITS: A SECOND LAYER, NOT A REPLACEMENT
------------------------------------------------------------------------------

discord.py's HTTPClient already does the right thing for the common case: it
tracks bucket headers, pre-emptively waits when a bucket is exhausted, sleeps
out a 429's `Retry-After` and retries the request, and retries 500/502/504/524
with a growing delay. None of that surfaces here at all.

What surfaces here is what it gives up on, and each needs a different answer:

  * `discord.RateLimited` -- raised instead of slept when a 429's Retry-After
    exceeds the client's configured `max_ratelimit_timeout`. It carries the
    exact seconds Discord asked for, so this layer honours that number rather
    than inventing one.
  * `discord.HTTPException` with status 429 -- a 429 discord.py declined to
    handle (a Cloudflare ban page, a response it could not read a Retry-After
    out of). Its `Retry-After` header is used when present, and a backoff when
    not.
  * `discord.DiscordServerError` and other transient HTTPExceptions -- Discord
    had a bad minute after discord.py's own retries were spent. Exponential
    backoff, bounded attempts.
  * `discord.Forbidden` / `discord.NotFound` -- permanent. Retrying a revoked
    permission forever is not resilience, it is a log full of the same line, so
    these are raised as ChannelUnreadable for the worker to end the run on.

The failure direction throughout is "fetch nothing this tick, keep the cursor",
which costs one page re-read and never costs a message.
"""
from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from itertools import pairwise

import discord

logger = logging.getLogger(__name__)

# How many times one page fetch is attempted before this tick gives up and
# leaves the run for the next one. Four is enough to ride out a bad minute
# (1 + 2 + 4 + 8 = 15 seconds of backoff) on top of discord.py's own five
# internal tries, and small enough that a genuinely broken channel is reported
# in the log rather than retried silently forever inside one tick. Giving up
# costs nothing: the cursor has not moved, so the next tick starts from exactly
# the same place.
_MAX_FETCH_ATTEMPTS = 4

# The backoff schedule's base and ceiling, in seconds. Deterministic rather than
# jittered, deliberately: there is exactly one backfill worker in a process and
# at most a handful of runs, so there is no thundering herd for jitter to break
# up -- and a deterministic delay is one a test can assert on instead of
# approximate.
_BACKOFF_BASE_SECONDS = 1.0
_BACKOFF_MAX_SECONDS = 30.0

# An upper bound on how long this layer will honour a Retry-After before giving
# up on the page and leaving it for the next tick. Discord does not normally ask
# for anything near this; the bound exists so a malformed or hostile header
# cannot park the worker task for hours holding a run open. Waiting less than
# asked is never done -- the choice is between waiting exactly as long as asked
# and not waiting at all.
_MAX_HONOURED_RETRY_AFTER_SECONDS = 300.0


class ChannelUnreadable(Exception):
    """Raised when a channel cannot be read and retrying will not change that.

    Deliberately distinct from "the fetch failed this time": a deleted channel
    and a revoked Read Message History permission are decisions someone made,
    and a backfill run over them should end with a state a moderator can see
    (BackfillState.FAILED) rather than retry hourly until someone notices the
    log.
    """


def ordered_page(
    messages: Sequence[discord.Message], *, after_message_id: int | None
) -> list[discord.Message]:
    """Sort one fetched page oldest-first and drop anything already covered.

    Returns a new list; never mutates the input. Two things happen here and both
    are part of the deliverable rather than defensive habit:

      * SORTING. The page is ordered by (created_at, id) regardless of what
        order it arrived in. If the input was not already sorted, that is logged
        at WARNING -- because the only ways it can happen are a discord.py
        change or a caller passing pages in the wrong order, and both are worth
        seeing rather than silently absorbing.
      * BOUNDING. Anything at or below `after_message_id` is dropped. The API is
        already asked for messages after that id, so this normally removes
        nothing; it is what makes re-processing a page after a crash idempotent
        at the boundary, and what stops a duplicate-delivery or an off-by-one in
        a future API change from walking the cursor backwards.

    Ties on created_at are broken by id, which cannot actually happen for real
    Discord messages -- two messages with the same creation time have different
    snowflakes, and the timestamp is derived from the snowflake -- but a sort
    key that is total by construction is one nobody has to reason about.
    """
    ordered = sorted(messages, key=lambda message: (message.created_at, message.id))
    if [message.id for message in ordered] != [message.id for message in messages]:
        logger.warning(
            "A history page arrived out of chronological order (%d message(s)); "
            "sorting it before processing. Backfill's supersession judgements depend "
            "on strict old-to-new order, so this is corrected rather than trusted.",
            len(messages),
        )

    if after_message_id is None:
        return ordered
    return [message for message in ordered if message.id > after_message_id]


def is_strictly_increasing(messages: Sequence[discord.Message]) -> bool:
    """Whether messages are in strictly ascending chronological order.

    Public because the guarantee it expresses is the deliverable, and a
    guarantee nothing can check from outside is not one: this is what the tests
    assert about a whole multi-page run, and what a future caller with a
    sequence of real messages in hand should use rather than writing the
    comparison again. Kept beside ordered_page so the check and the thing it
    checks cannot drift into disagreeing about what "chronological" means.

    The worker re-checks its assembled BATCH separately rather than through this
    function, because a batch is a list of QueuedMessage rather than of
    discord.Message -- two different shapes, deliberately not unified behind a
    protocol that would exist only to let one small comparison be written once.
    """
    return all(
        earlier.id < later.id and earlier.created_at <= later.created_at
        for earlier, later in pairwise(messages)
    )


def _retry_after_seconds(error: discord.HTTPException) -> float | None:
    """Discord's own requested wait for a 429, or None if it did not give one.

    Reads the header rather than guessing, and never raises: a header that is
    missing, empty or unparseable simply means "no number from Discord", which
    the caller answers with its ordinary backoff. `response` can be absent
    entirely on an exception built from a partial failure, which is why every
    step is defended rather than only the float conversion.
    """
    try:
        headers = error.response.headers  # type: ignore[union-attr]
        raw = headers.get("Retry-After")
    except Exception:
        return None
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _backoff_seconds(attempt: int) -> float:
    """The delay before retry number `attempt` (1-based), capped.

    Doubling from _BACKOFF_BASE_SECONDS: 1, 2, 4, 8 ... up to the ceiling. A
    function rather than an inline expression so the schedule is stated once and
    a test can assert the actual sequence instead of re-deriving it.
    """
    return min(_BACKOFF_MAX_SECONDS, _BACKOFF_BASE_SECONDS * (2 ** (attempt - 1)))


async def fetch_history_page(
    channel: discord.TextChannel,
    *,
    after_message_id: int | None,
    before_message_id: int,
    limit: int,
    sleep: object = asyncio.sleep,
) -> list[discord.Message] | None:
    """Fetch one page of history, oldest first, honouring Discord's rate limits.

    Returns the page (possibly empty, meaning the run has reached its upper
    bound), or None if every attempt failed transiently -- which the caller
    treats as "not this tick", leaving the cursor untouched.

    Raises ChannelUnreadable, and only that, for the permanent failures: a
    deleted channel, a revoked Read Message History permission. Everything else
    is retried or reported as None.

    `sleep` is injected so the backoff schedule is testable without a test suite
    that actually waits fifteen seconds to prove it waited fifteen seconds. It
    defaults to asyncio.sleep and production never passes anything else.

    The bounds are HALF-OPEN on both sides and stated in message ids rather than
    timestamps, which is exact: `after_message_id` is exclusive (discord.py's
    own `after` semantics) and `before_message_id` is exclusive too, so the
    upper bound -- the snowflake of the moment the run started -- cleanly
    separates backfill's territory from the live path's without either needing
    to know the other's rules.
    """
    for attempt in range(1, _MAX_FETCH_ATTEMPTS + 1):
        try:
            page = [
                message
                async for message in channel.history(
                    limit=limit,
                    after=discord.Object(id=after_message_id)
                    if after_message_id is not None
                    else None,
                    before=discord.Object(id=before_message_id),
                    oldest_first=True,
                )
            ]
        except (discord.Forbidden, discord.NotFound) as exc:
            # Permanent by nature: someone deleted the channel or took away Read
            # Message History. Retrying is not resilience here, it is a log line
            # repeated hourly, so the run ends in a state a moderator can act on.
            raise ChannelUnreadable(
                f"channel {channel.id} cannot be read: {exc}"
            ) from exc
        except discord.RateLimited as exc:
            # discord.py declined to sleep this one out because it exceeded the
            # client's max_ratelimit_timeout. It handed us Discord's own number;
            # honour exactly that rather than substituting a guess.
            if not await _wait(exc.retry_after, sleep=sleep, reason="a rate limit"):
                return None
            continue
        except discord.HTTPException as exc:
            if attempt == _MAX_FETCH_ATTEMPTS:
                logger.warning(
                    "Giving up on a history page for channel %s after %d attempt(s): %s. "
                    "The cursor is unchanged, so the next tick retries from the same place.",
                    channel.id,
                    attempt,
                    exc,
                )
                return None

            requested = _retry_after_seconds(exc) if exc.status == 429 else None
            delay = requested if requested is not None else _backoff_seconds(attempt)
            reason = "a rate limit" if exc.status == 429 else f"HTTP {exc.status}"
            logger.warning(
                "History page for channel %s failed with %s (attempt %d/%d); "
                "waiting %.1fs before retrying",
                channel.id,
                reason,
                attempt,
                _MAX_FETCH_ATTEMPTS,
                delay,
            )
            if not await _wait(delay, sleep=sleep, reason=reason):
                return None
            continue

        return ordered_page(page, after_message_id=after_message_id)

    # Unreachable: the loop either returns a page, returns None, or raises.
    # Written out rather than left to fall off the end, so a future edit to the
    # attempt bound cannot silently start returning None-by-omission.
    return None


async def _wait(delay: float, *, sleep: object, reason: str) -> bool:
    """Wait `delay` seconds. Returns False if the wait was refused as too long.

    Refusing rather than truncating is the only honest option: waiting less than
    Discord asked for is worse than not asking again at all, since it produces
    another 429 and another wait. A refused wait means the page is abandoned for
    this tick, which costs nothing -- the cursor has not moved.
    """
    if delay > _MAX_HONOURED_RETRY_AFTER_SECONDS:
        logger.warning(
            "Refusing to wait %.1fs for %s; abandoning this history page for now "
            "(the cursor is unchanged and the next tick retries)",
            delay,
            reason,
        )
        return False
    if delay > 0:
        await sleep(delay)  # type: ignore[operator]
    return True
