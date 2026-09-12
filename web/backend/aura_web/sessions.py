"""Server-side session and OAuth-state storage.

Two stores, both in memory, both holding secrets only as digests.

The reason they are server-side at all is the sub-phase's central rule:
Discord's access and refresh tokens never leave this process. The browser
receives one opaque identifier and nothing else; every token stays in the
record that identifier points at. A signed-cookie or JWT session would have
put the token itself (or an encrypted copy of it) into the browser, which is
a different security posture no matter how good the cipher is.

IN-MEMORY, NOT PERSISTED, and that is a deliberate 4b trade-off rather than
an omission: a restart of this container logs everyone out, costing one click
on a login button, and in exchange the sub-phase ships with no new table, no
second writer against Aura's SQLite file (see web/README.md), and no
at-rest copy of anyone's Discord token. When 4c/4d give this service durable
storage of its own, moving these two stores behind the same interface is a
contained change; putting tokens in the browser would not have been.

Neither store awaits anywhere inside a method, so each method is atomic with
respect to the event loop: no request can observe a half-evicted store or
lose a write to an interleaving one, and no lock is needed to say so.
"""
from __future__ import annotations

import hashlib
import hmac
import logging
import secrets
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)

# 32 bytes of os.urandom, URL-safe base64 encoded. Both identifiers are bearer
# secrets travelling in cookies and query strings, so they are sized like
# session tokens rather than like database keys.
_TOKEN_BYTES = 32


def utc_now() -> datetime:
    """Current time as a timezone-aware UTC datetime.

    Deliberately the same shape as aura.db.connection.utc_now (which this
    service does not import -- see aura_web.config for why the two processes
    share no modules), so a reader moving between them finds one convention
    rather than two.
    """
    return datetime.now(timezone.utc)


def _digest(token: str) -> str:
    """Hash a bearer token for use as a store key.

    Storing the digest rather than the token means the store's keys are not
    themselves credentials: a log line, a heap dump or a debugger view of the
    store cannot be replayed as a login. Lookup hashes the presented token
    first, so the dict comparison never sees the secret either.
    """
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class DiscordTokens:
    """An OAuth2 token pair and its expiry. Never serialised toward the browser."""

    access_token: str
    refresh_token: str | None
    expires_at: datetime

    def is_expired(self, *, now: datetime, leeway_seconds: float = 60.0) -> bool:
        """Whether this access token is expired, or close enough to treat as expired.

        The leeway exists because the alternative is a race we would lose
        silently: a token with four seconds left passes a naive check, then
        expires in flight, and the user sees an unexplained 401 instead of a
        refresh.
        """
        return now >= self.expires_at - timedelta(seconds=leeway_seconds)


@dataclass(frozen=True)
class DiscordUser:
    """The identity fields this service keeps. A subset of Discord's user object.

    Only what the shell displays is retained. Everything else Discord sends
    for the ``identify`` scope -- email presence flags, MFA state, premium
    type, locale -- is dropped at the boundary rather than stored and
    forgotten about, so there is no second place to audit when asking what
    this service knows about a person.
    """

    id: str
    username: str
    global_name: str | None
    avatar: str | None


@dataclass
class Session:
    """One logged-in browser's server-side record."""

    user: DiscordUser
    tokens: DiscordTokens
    created_at: datetime
    expires_at: datetime


class SessionStore:
    """Opaque-identifier session storage with a TTL and a hard size ceiling."""

    def __init__(
        self,
        *,
        ttl_seconds: int,
        max_sessions: int,
        clock: Callable[[], datetime] = utc_now,
    ) -> None:
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        if max_sessions <= 0:
            raise ValueError("max_sessions must be positive")
        self._ttl = timedelta(seconds=ttl_seconds)
        self._max_sessions = max_sessions
        self._clock = clock
        # Insertion-ordered so eviction can drop the oldest record without
        # scanning for a minimum timestamp.
        self._sessions: OrderedDict[str, Session] = OrderedDict()

    def create(self, user: DiscordUser, tokens: DiscordTokens) -> str:
        """Store a new session and return its opaque identifier.

        A fresh identifier every time, never one supplied or influenced by the
        caller: session fixation needs an attacker-chosen identifier to
        survive the login, and there is no code path here that lets one in.
        """
        now = self._clock()
        self._purge_expired(now)
        self._evict_to_fit(self._max_sessions - 1)

        token = secrets.token_urlsafe(_TOKEN_BYTES)
        self._sessions[_digest(token)] = Session(
            user=user,
            tokens=tokens,
            created_at=now,
            expires_at=now + self._ttl,
        )
        return token

    def get(self, token: str | None) -> Session | None:
        """Resolve an identifier to its live session, or None.

        An expired session is deleted on the way out rather than merely
        hidden, so a store left running without traffic on a particular key
        does not accumulate records that can never be returned.
        """
        if not token:
            return None
        key = _digest(token)
        session = self._sessions.get(key)
        if session is None:
            return None
        if self._clock() >= session.expires_at:
            del self._sessions[key]
            return None
        return session

    def replace_tokens(self, token: str, tokens: DiscordTokens) -> bool:
        """Swap in a refreshed token pair, keeping the same session identifier.

        Rotating the identifier on refresh would log out every other tab of
        the same browser for no security gain -- the identifier is not the
        thing that expired.
        """
        session = self.get(token)
        if session is None:
            return False
        session.tokens = tokens
        return True

    def delete(self, token: str | None) -> Session | None:
        """Remove a session, returning it so the caller can revoke its tokens."""
        if not token:
            return None
        return self._sessions.pop(_digest(token), None)

    def __len__(self) -> int:
        return len(self._sessions)

    def _purge_expired(self, now: datetime) -> None:
        expired = [key for key, session in self._sessions.items() if now >= session.expires_at]
        for key in expired:
            del self._sessions[key]

    def _evict_to_fit(self, target_size: int) -> None:
        """Drop oldest-first until the store is no larger than target_size.

        Anyone can create sessions without authenticating past Discord, so an
        unbounded store is a memory-exhaustion lever. Evicting the oldest
        live session is the least-bad response: it logs somebody out, which is
        recoverable, rather than refusing all new logins, which is not
        recoverable by the person it happens to.
        """
        evicted = 0
        while len(self._sessions) > max(target_size, 0):
            self._sessions.popitem(last=False)
            evicted += 1
        if evicted:
            logger.warning(
                "Session store at capacity (%d); evicted %d oldest session(s)",
                self._max_sessions,
                evicted,
            )


class OAuthStateStore:
    """Single-use, TTL-bounded storage for the OAuth2 ``state`` parameter.

    The store alone does not complete the CSRF defence and is not meant to:
    it proves a state was issued by this service and has not been used, but
    not that it was issued to *this browser*. An attacker can start their own
    login and obtain a state this store considers perfectly valid. The second
    half -- a copy of the same value in an httpOnly cookie, compared on
    return -- is what binds the flow to one browser, and lives in
    aura_web.routes.auth where both halves are visible together.
    """

    def __init__(
        self,
        *,
        ttl_seconds: int,
        max_states: int,
        clock: Callable[[], datetime] = utc_now,
    ) -> None:
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        if max_states <= 0:
            raise ValueError("max_states must be positive")
        self._ttl = timedelta(seconds=ttl_seconds)
        self._max_states = max_states
        self._clock = clock
        self._states: OrderedDict[str, datetime] = OrderedDict()

    def issue(self) -> str:
        """Mint a new state value, evicting expired and then oldest entries to fit."""
        now = self._clock()
        self._purge_expired(now)
        self._evict_to_fit(self._max_states - 1)

        state = secrets.token_urlsafe(_TOKEN_BYTES)
        self._states[_digest(state)] = now + self._ttl
        return state

    def consume(self, state: str | None) -> bool:
        """Validate and burn a state value; False if unknown, expired or already used.

        Single-use by construction: the entry is removed before the expiry is
        checked, so a replay of an expired state cannot be distinguished from
        a replay of a live one by timing or by outcome, and neither succeeds.
        """
        if not state:
            return False
        expires_at = self._states.pop(_digest(state), None)
        if expires_at is None:
            return False
        return self._clock() < expires_at

    def __len__(self) -> int:
        return len(self._states)

    def _purge_expired(self, now: datetime) -> None:
        expired = [key for key, expires_at in self._states.items() if now >= expires_at]
        for key in expired:
            del self._states[key]

    def _evict_to_fit(self, target_size: int) -> None:
        evicted = 0
        while len(self._states) > max(target_size, 0):
            self._states.popitem(last=False)
            evicted += 1
        if evicted:
            logger.warning(
                "OAuth state store at capacity (%d); evicted %d oldest pending login(s)",
                self._max_states,
                evicted,
            )


def constant_time_equals(left: str | None, right: str | None) -> bool:
    """Compare two secrets without leaking their common prefix length via timing.

    Used for the state-cookie binding check. The values are short-lived and
    the endpoint is not a practical timing oracle, but ``==`` on a secret is
    the kind of detail that is free to get right here and expensive to notice
    is missing later.
    """
    if left is None or right is None:
        return False
    return hmac.compare_digest(left.encode("utf-8"), right.encode("utf-8"))
