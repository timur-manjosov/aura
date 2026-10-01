"""Per-client request-rate limits, applied before any route reads a byte of the body.

WHY HERE AND NOT IN THE PROXY: the host's Caddy is the stock build, which has
no rate-limit module, and rebuilding it with a plugin would put the other site
it serves at risk. So the limit lives in this service, in front of every route,
keyed by the address aura_web.client_address resolved -- never by anything the
client could set itself.

FOUR BUCKETS, each a token bucket per client (a burst that refills at a steady
rate), each tuned to what one request in it costs:

  * ``auth``    -- /api/auth/*: the login start is cheap, but a callback is a
                   Discord token exchange and a user fetch, and a logout a
                   Discord revocation.
  * ``billing`` -- POST /api/billing/checkout and /portal: each is a Discord
                   call, a call to the bot and a Stripe call. The strictest.
  * ``webhook`` -- /api/stripe/*: see below.
  * ``global``  -- everything else, one ceiling per client.

THE WEBHOOK BUCKET COUNTS ONLY DELIVERIES THAT FAIL VERIFICATION. A legitimate
Stripe delivery must never be refused, and Stripe's traffic cannot be told
apart from anyone else's until its signature has been checked -- which means
reading the body, which is exactly the cost a flood should not get to impose.
So a token is taken before the body is read, as for every bucket, and handed
back the moment the signature verifies (refund_webhook_allowance, called by the
route). A client sending forgeries drains its bucket and is then refused
without a byte read; Stripe, whose deliveries all verify, never holds a token
for longer than one verification takes.

LOGGING is bounded per client and bucket, independently of the refusals: at
most one WARNING per REFUSAL_LOG_INTERVAL_SECONDS, carrying the number of
refusals left unlogged since the line before it, and one INFO line once the
client has gone a full interval without a refusal, carrying the rest. A flood
just above the refill rate otherwise ends and restarts its refusal run every
few seconds, and a line per run is a line every few seconds for as long as the
flood lasts. The 429 and its Retry-After never depend on any of this.

MEMORY is bounded per bucket by AURA_WEB_RATE_LIMIT_MAX_TRACKED_CLIENTS: the
least recently seen client is forgotten first. Forgetting is the safe
direction -- a forgotten client starts again from a full bucket -- and a flood
of distinct addresses large enough to force it is one a per-client limit could
not have stopped anyway. IPv6 clients are keyed by their /64, the block one
subscriber is normally assigned, so rotating addresses inside it buys nothing.

Imports aura_web.client_address (to parse the resolved address), aura_web.config
and aura_web.errors from this package, and nothing that reaches the network.
"""

from __future__ import annotations

import logging
import math
import time
from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from ipaddress import IPv6Address, IPv6Network
from typing import Final
from urllib.parse import quote

from starlette.requests import Request
from starlette.types import ASGIApp, Receive, Scope, Send

from aura_web.client_address import parse_address
from aura_web.config import WebSettings
from aura_web.errors import ErrorCode, error_response

logger = logging.getLogger(__name__)

# The block an IPv6 subscriber is normally given; keying any finer would hand
# one client 2**64 identities.
IPV6_CLIENT_PREFIX_LENGTH: Final[int] = 64

# The key for a request whose server reported no peer at all (never the case
# over TCP; possible over a Unix socket). One shared bucket, not a free pass.
UNKNOWN_CLIENT_KEY: Final[str] = "unknown"

# A peer that is not an IP address (only test transports produce one) is keyed
# by its text, cut to this length so it cannot become a memory lever.
MAX_NON_ADDRESS_KEY_LENGTH: Final[int] = 64

# Where the webhook bucket's refund is left for the route, in the per-request
# state Starlette exposes as `request.state`.
WEBHOOK_REFUND_STATE_KEY: Final[str] = "aura_rate_limit_webhook_refund"

AUTH_PATH_PREFIX: Final[str] = "/api/auth/"
BILLING_ACTION_PATHS: Final[frozenset[str]] = frozenset(
    {"/api/billing/checkout", "/api/billing/portal"}
)
STRIPE_PATH_PREFIX: Final[str] = "/api/stripe/"

SECONDS_PER_MINUTE: Final[float] = 60.0

# At most one WARNING per client and bucket in this many seconds; also how long
# a client must go without a refusal before its limit is reported lifted.
REFUSAL_LOG_INTERVAL_SECONDS: Final[float] = 60.0


class RateLimitBucket(StrEnum):
    """The four request classes, each limited separately per client."""

    AUTH = "auth"
    BILLING = "billing"
    WEBHOOK = "webhook"
    GLOBAL = "global"


@dataclass(frozen=True)
class BucketPolicy:
    """One bucket's limit: how many requests at once, and how fast it refills.

    Attributes
    ----------
    burst
        The bucket's capacity: requests a client may make back to back.
    per_minute
        The sustained rate the bucket refills at, in requests per minute.
    """

    burst: int
    per_minute: float

    @property
    def refill_per_second(self) -> float:
        """The refill rate in tokens per second.

        Returns
        -------
        float
            `per_minute` divided by sixty.
        """
        return self.per_minute / SECONDS_PER_MINUTE


@dataclass(frozen=True)
class RateLimitDecision:
    """The outcome of asking for one request's token, and whether it is worth a log line.

    Attributes
    ----------
    allowed
        Whether the request may proceed.
    retry_after_seconds
        Whole seconds until one token is available again; 0 when allowed.
    log_refusal
        True on a refusal that is to be logged at WARNING: the client's first
        in this bucket, or the first REFUSAL_LOG_INTERVAL_SECONDS or more after
        the last logged one. Never true on an allowed request.
    limit_lifted
        True on the first allowed request after the client went
        REFUSAL_LOG_INTERVAL_SECONDS or more without a refusal, following at
        least one refusal. Never true on a refusal.
    unlogged_refusals
        With `log_refusal` or `limit_lifted`: the refusals since the client's
        previous line in this bucket that no line has reported yet. 0 otherwise.
    """

    allowed: bool
    retry_after_seconds: int
    log_refusal: bool = False
    limit_lifted: bool = False
    unlogged_refusals: int = 0


@dataclass
class _Allowance:
    tokens: float
    updated_at: float
    # None while the client has no refusal awaiting its "lifted" line.
    last_refusal_at: float | None = None
    last_warning_at: float | None = None
    unlogged_refusals: int = 0


def classify_path(path: str) -> RateLimitBucket:
    """Return the bucket a request path is limited by.

    Parameters
    ----------
    path
        The decoded request path, exactly as the router will match it.

    Returns
    -------
    RateLimitBucket
        AUTH for anything under /api/auth/, BILLING for the two billing action
        paths, WEBHOOK for anything under /api/stripe/, GLOBAL otherwise.

    Notes
    -----
    Classified on the same decoded ``scope["path"]`` the router dispatches on,
    so no spelling of a path (percent-encoding, a trailing slash) can reach a
    route through a different bucket than its own: anything the router would
    not match exactly ends in a 404 or a redirect, at the cost of a token from
    whichever bucket its spelling falls in.
    """
    if path.startswith(AUTH_PATH_PREFIX):
        return RateLimitBucket.AUTH
    if path in BILLING_ACTION_PATHS:
        return RateLimitBucket.BILLING
    if path.startswith(STRIPE_PATH_PREFIX):
        return RateLimitBucket.WEBHOOK
    return RateLimitBucket.GLOBAL


def client_key(address: str | None) -> str:
    """Return the identity a client is limited under.

    Parameters
    ----------
    address
        The resolved client address (aura_web.client_address), or None.

    Returns
    -------
    str
        An IPv4 address as written; an IPv6 address's /64 network; the
        UNKNOWN_CLIENT_KEY for None; anything else cut to
        MAX_NON_ADDRESS_KEY_LENGTH characters.
    """
    if address is None:
        return UNKNOWN_CLIENT_KEY
    parsed = parse_address(address)
    if parsed is None:
        return address[:MAX_NON_ADDRESS_KEY_LENGTH]
    if isinstance(parsed, IPv6Address):
        return str(IPv6Network((parsed, IPV6_CLIENT_PREFIX_LENGTH), strict=False))
    return str(parsed)


class RateLimiter:
    """A bounded set of per-client token buckets.

    Parameters
    ----------
    policies
        One policy per bucket; every RateLimitBucket must have one.
    max_tracked_clients
        How many clients each bucket remembers before forgetting the least
        recently seen.
    clock
        A monotonic clock in seconds. Injectable so tests control time.

    Raises
    ------
    ValueError
        If a bucket has no policy, a policy is not positive, or
        `max_tracked_clients` is below one.

    Notes
    -----
    Every method is synchronous and awaits nothing, so under asyncio a
    check-and-take can never interleave with another request's: concurrent
    requests are serialised by the event loop, not by a lock.
    """

    def __init__(
        self,
        policies: Mapping[RateLimitBucket, BucketPolicy],
        *,
        max_tracked_clients: int,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        missing = set(RateLimitBucket) - set(policies)
        if missing:
            raise ValueError(f"no rate-limit policy for {sorted(missing)}")
        for bucket, policy in policies.items():
            if policy.burst < 1 or not policy.per_minute > 0:
                raise ValueError(f"the {bucket} rate-limit policy must be positive")
        if max_tracked_clients < 1:
            raise ValueError("max_tracked_clients must be at least 1")
        self._policies = dict(policies)
        self._max_tracked_clients = max_tracked_clients
        self._clock = clock
        self._allowances: dict[RateLimitBucket, OrderedDict[str, _Allowance]] = {
            bucket: OrderedDict() for bucket in RateLimitBucket
        }

    def policy(self, bucket: RateLimitBucket) -> BucketPolicy:
        """Return the policy a bucket is enforced with.

        Parameters
        ----------
        bucket
            The bucket.

        Returns
        -------
        BucketPolicy
            Its burst and refill rate.
        """
        return self._policies[bucket]

    def tracked_clients(self, bucket: RateLimitBucket) -> int:
        """Return how many clients a bucket currently remembers.

        Parameters
        ----------
        bucket
            The bucket.

        Returns
        -------
        int
            Never more than `max_tracked_clients`.
        """
        return len(self._allowances[bucket])

    def _refilled(self, bucket: RateLimitBucket, key: str) -> _Allowance:
        """Return the client's allowance, refilled up to now, as most recently used.

        Its `updated_at` is then the client's own "now": the clock reading, or
        the latest one this client has seen if the clock has stepped backwards.
        """
        policy = self._policies[bucket]
        allowances = self._allowances[bucket]
        now = self._clock()
        allowance = allowances.get(key)
        if allowance is None:
            allowance = _Allowance(tokens=float(policy.burst), updated_at=now)
            allowances[key] = allowance
            while len(allowances) > self._max_tracked_clients:
                allowances.popitem(last=False)
        else:
            # A clock that steps backwards refills nothing rather than
            # draining a bucket below what it held.
            elapsed = max(0.0, now - allowance.updated_at)
            allowance.tokens = min(
                float(policy.burst), allowance.tokens + elapsed * policy.refill_per_second
            )
            allowance.updated_at = max(allowance.updated_at, now)
            allowances.move_to_end(key)
        return allowance

    def acquire(self, bucket: RateLimitBucket, key: str) -> RateLimitDecision:
        """Take one token for a request, or refuse it.

        Parameters
        ----------
        bucket
            The bucket the request is limited by.
        key
            The client's identity (client_key).

        Returns
        -------
        RateLimitDecision
            Allowed with a token taken, or refused with the whole seconds until
            a token will be available (at least 1), plus whether the request is
            worth a log line. Whether it is allowed and its Retry-After are
            decided by the token bucket alone; the log fields never feed back.

        Notes
        -----
        Every refusal is either logged itself or counted in exactly one later
        line's `unlogged_refusals`, as long as the client is seen again after
        it: a WARNING once the interval has passed, or the "lifted" line once a
        full interval passed without a refusal. A client never seen again, or
        forgotten under memory pressure, takes its last uncounted refusals
        with it -- at most one interval's worth after its last WARNING.
        """
        policy = self._policies[bucket]
        allowance = self._refilled(bucket, key)
        now = allowance.updated_at
        if allowance.tokens >= 1.0:
            allowance.tokens -= 1.0
            if (
                allowance.last_refusal_at is None
                or now - allowance.last_refusal_at < REFUSAL_LOG_INTERVAL_SECONDS
            ):
                return RateLimitDecision(allowed=True, retry_after_seconds=0)
            unlogged = allowance.unlogged_refusals
            allowance.last_refusal_at = None
            allowance.unlogged_refusals = 0
            return RateLimitDecision(
                allowed=True,
                retry_after_seconds=0,
                limit_lifted=True,
                unlogged_refusals=unlogged,
            )
        retry_after = max(1, math.ceil((1.0 - allowance.tokens) / policy.refill_per_second))
        allowance.last_refusal_at = now
        if (
            allowance.last_warning_at is not None
            and now - allowance.last_warning_at < REFUSAL_LOG_INTERVAL_SECONDS
        ):
            allowance.unlogged_refusals += 1
            return RateLimitDecision(allowed=False, retry_after_seconds=retry_after)
        unlogged = allowance.unlogged_refusals
        allowance.last_warning_at = now
        allowance.unlogged_refusals = 0
        return RateLimitDecision(
            allowed=False,
            retry_after_seconds=retry_after,
            log_refusal=True,
            unlogged_refusals=unlogged,
        )

    def refund(self, bucket: RateLimitBucket, key: str) -> None:
        """Hand back one token taken by `acquire`, never past the bucket's capacity.

        Parameters
        ----------
        bucket
            The bucket the token was taken from.
        key
            The client's identity.

        Notes
        -----
        A client forgotten in between (evicted under memory pressure) is not
        re-created: it already starts from a full bucket next time.
        """
        allowance = self._allowances[bucket].get(key)
        if allowance is None:
            return
        allowance.tokens = min(float(self._policies[bucket].burst), allowance.tokens + 1.0)


def refund_webhook_allowance(request: Request) -> None:
    """Return the webhook token this request took, once its signature has verified.

    Parameters
    ----------
    request
        The webhook request whose delivery verified.

    Notes
    -----
    Single-use: the refund is removed from the request as it is called, so no
    code path can hand back more than the one token that was taken. A request
    the limiter never saw (a test calling the route directly) has none.
    """
    state = request.scope.get("state")
    if not isinstance(state, dict):
        return
    refund = state.pop(WEBHOOK_REFUND_STATE_KEY, None)
    if callable(refund):
        refund()


class RateLimitMiddleware:
    """Refuse a request over its client's limit before any route sees it.

    Parameters
    ----------
    app
        The ASGI application to wrap.
    limiter
        The buckets to enforce.

    Notes
    -----
    A pure ASGI middleware: it decides on the path and the resolved client
    address alone, so a refused request's body is never read -- the property
    that keeps a flood against the webhook cheap. It must sit inside
    aura_web.client_address.ClientAddressMiddleware, which is what makes
    ``scope["client"]`` the real client.
    """

    def __init__(self, app: ASGIApp, *, limiter: RateLimiter) -> None:
        self.app = app
        self.limiter = limiter

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Charge one token to the request's client, then pass it on or refuse it.

        Parameters
        ----------
        scope, receive, send
            The ASGI connection.
        """
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        bucket = classify_path(scope["path"])
        client = scope.get("client")
        key = client_key(client[0] if client is not None else None)
        decision = self.limiter.acquire(bucket, key)

        if decision.limit_lifted:
            logger.info(
                "Rate limit lifted for client %s in the %s bucket: no refusal for %ds; "
                "%d refusal(s) since the last line were not logged",
                key,
                bucket.value,
                REFUSAL_LOG_INTERVAL_SECONDS,
                decision.unlogged_refusals,
            )
        if not decision.allowed:
            if decision.log_refusal:
                # The method and path only: the query string can carry an OAuth
                # code or state, and nothing here needs it. The path is quoted
                # as uvicorn's access log quotes it: it arrives percent-decoded,
                # and a decoded %0a would otherwise start a forged log line.
                logger.warning(
                    "Rate limit reached for client %s in the %s bucket (%s %s); refusing for "
                    "%ds; %d refusal(s) since the last line were not logged",
                    key,
                    bucket.value,
                    scope.get("method", "?"),
                    quote(scope["path"]),
                    decision.retry_after_seconds,
                    decision.unlogged_refusals,
                )
            response = error_response(ErrorCode.RATE_LIMITED, status_code=429)
            response.headers["Retry-After"] = str(decision.retry_after_seconds)
            await response(scope, receive, send)
            return

        if bucket is RateLimitBucket.WEBHOOK:
            limiter = self.limiter

            def refund() -> None:
                limiter.refund(bucket, key)

            scope.setdefault("state", {})[WEBHOOK_REFUND_STATE_KEY] = refund
        await self.app(scope, receive, send)


def rate_limiter_from_settings(
    settings: WebSettings, *, clock: Callable[[], float] = time.monotonic
) -> RateLimiter:
    """Build the limiter the configuration describes.

    Parameters
    ----------
    settings
        Supplies every bucket's burst and rate, and the memory bound.
    clock
        A monotonic clock in seconds; tests pass a fake one.

    Returns
    -------
    RateLimiter
        The limiter, with no client tracked yet.
    """
    return RateLimiter(
        {
            RateLimitBucket.AUTH: BucketPolicy(
                burst=settings.rate_limit_auth_burst,
                per_minute=settings.rate_limit_auth_per_minute,
            ),
            RateLimitBucket.BILLING: BucketPolicy(
                burst=settings.rate_limit_billing_burst,
                per_minute=settings.rate_limit_billing_per_minute,
            ),
            RateLimitBucket.WEBHOOK: BucketPolicy(
                burst=settings.rate_limit_webhook_burst,
                per_minute=settings.rate_limit_webhook_per_minute,
            ),
            RateLimitBucket.GLOBAL: BucketPolicy(
                burst=settings.rate_limit_global_burst,
                per_minute=settings.rate_limit_global_per_minute,
            ),
        },
        max_tracked_clients=settings.rate_limit_max_tracked_clients,
        clock=clock,
    )
