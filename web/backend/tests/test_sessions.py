"""The two in-memory stores: lifetime, single use, bounds, and what they never hold.

Both stores are reachable by anyone who can hit /api/auth/login, so their
bounds are a denial-of-service surface rather than housekeeping, and are
tested as such.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from aura_web.sessions import (
    DiscordTokens,
    DiscordUser,
    OAuthStateStore,
    SessionStore,
    constant_time_equals,
)

START = datetime(2026, 9, 12, 12, 0, 0, tzinfo=timezone.utc)


class FrozenClock:
    """A hand-advanced clock, so expiry is tested exactly rather than by sleeping."""

    def __init__(self, now: datetime = START) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += timedelta(seconds=seconds)


def a_user(user_id: str = "5000") -> DiscordUser:
    return DiscordUser(id=user_id, username="moderator", global_name=None, avatar=None)


def some_tokens(clock: FrozenClock, *, lifetime: int = 3600) -> DiscordTokens:
    return DiscordTokens(
        access_token="access-secret",
        refresh_token="refresh-secret",
        expires_at=clock.now + timedelta(seconds=lifetime),
    )


class TestSessionStore:
    def test_a_created_session_is_retrievable_by_its_identifier(self) -> None:
        clock = FrozenClock()
        store = SessionStore(ttl_seconds=3600, max_sessions=10, clock=clock)

        token = store.create(a_user(), some_tokens(clock))

        session = store.get(token)
        assert session is not None
        assert session.user.id == "5000"

    def test_each_creation_mints_a_distinct_identifier(self) -> None:
        clock = FrozenClock()
        store = SessionStore(ttl_seconds=3600, max_sessions=100, clock=clock)

        tokens = {store.create(a_user(), some_tokens(clock)) for _ in range(50)}

        assert len(tokens) == 50

    def test_identifiers_are_long_enough_to_be_unguessable(self) -> None:
        clock = FrozenClock()
        store = SessionStore(ttl_seconds=3600, max_sessions=10, clock=clock)

        token = store.create(a_user(), some_tokens(clock))

        # 32 random bytes, URL-safe base64: 43 characters.
        assert len(token) >= 43

    def test_the_raw_identifier_is_never_a_key_in_the_store(self) -> None:
        """Keys are digests, so a heap dump or a log of the store is not replayable."""
        clock = FrozenClock()
        store = SessionStore(ttl_seconds=3600, max_sessions=10, clock=clock)

        token = store.create(a_user(), some_tokens(clock))

        assert token not in store._sessions

    @pytest.mark.parametrize("unknown", [None, "", "nope", "a" * 43])
    def test_an_unknown_identifier_resolves_to_nothing(self, unknown: str | None) -> None:
        clock = FrozenClock()
        store = SessionStore(ttl_seconds=3600, max_sessions=10, clock=clock)
        store.create(a_user(), some_tokens(clock))

        assert store.get(unknown) is None

    def test_a_session_expires_exactly_at_its_ttl(self) -> None:
        clock = FrozenClock()
        store = SessionStore(ttl_seconds=100, max_sessions=10, clock=clock)
        token = store.create(a_user(), some_tokens(clock))

        clock.advance(99)
        assert store.get(token) is not None
        clock.advance(1)
        assert store.get(token) is None

    def test_reading_an_expired_session_removes_it(self) -> None:
        """Otherwise a store with idle keys accumulates records nobody can reach."""
        clock = FrozenClock()
        store = SessionStore(ttl_seconds=100, max_sessions=10, clock=clock)
        token = store.create(a_user(), some_tokens(clock))

        clock.advance(101)
        store.get(token)

        assert len(store) == 0

    def test_deleting_returns_the_session_so_its_token_can_be_revoked(self) -> None:
        clock = FrozenClock()
        store = SessionStore(ttl_seconds=3600, max_sessions=10, clock=clock)
        token = store.create(a_user(), some_tokens(clock))

        deleted = store.delete(token)

        assert deleted is not None
        assert deleted.tokens.access_token == "access-secret"
        assert store.get(token) is None

    def test_deleting_twice_is_harmless(self) -> None:
        clock = FrozenClock()
        store = SessionStore(ttl_seconds=3600, max_sessions=10, clock=clock)
        token = store.create(a_user(), some_tokens(clock))

        store.delete(token)
        assert store.delete(token) is None

    def test_refreshing_tokens_keeps_the_same_identifier(self) -> None:
        """Rotating on refresh would log out every other tab for no security gain."""
        clock = FrozenClock()
        store = SessionStore(ttl_seconds=3600, max_sessions=10, clock=clock)
        token = store.create(a_user(), some_tokens(clock))

        replaced = store.replace_tokens(
            token,
            DiscordTokens(
                access_token="new-access",
                refresh_token="new-refresh",
                expires_at=clock.now + timedelta(hours=2),
            ),
        )

        assert replaced is True
        session = store.get(token)
        assert session is not None
        assert session.tokens.access_token == "new-access"

    def test_refreshing_an_unknown_session_reports_failure(self) -> None:
        clock = FrozenClock()
        store = SessionStore(ttl_seconds=3600, max_sessions=10, clock=clock)

        assert store.replace_tokens("nope", some_tokens(clock)) is False

    def test_the_store_never_grows_past_its_ceiling(self) -> None:
        """Anyone can create sessions; an unbounded store is a memory lever."""
        clock = FrozenClock()
        store = SessionStore(ttl_seconds=3600, max_sessions=5, clock=clock)

        for _ in range(500):
            store.create(a_user(), some_tokens(clock))

        assert len(store) <= 5

    def test_eviction_drops_the_oldest_session_first(self) -> None:
        clock = FrozenClock()
        store = SessionStore(ttl_seconds=3600, max_sessions=2, clock=clock)
        oldest = store.create(a_user("1"), some_tokens(clock))
        middle = store.create(a_user("2"), some_tokens(clock))

        newest = store.create(a_user("3"), some_tokens(clock))

        assert store.get(oldest) is None
        assert store.get(middle) is not None
        assert store.get(newest) is not None

    def test_expired_sessions_are_purged_before_a_live_one_is_evicted(self) -> None:
        """A quiet store full of dead records must not log a live user out."""
        clock = FrozenClock()
        store = SessionStore(ttl_seconds=100, max_sessions=3, clock=clock)
        for index in range(3):
            store.create(a_user(str(index)), some_tokens(clock))

        clock.advance(101)
        survivor = store.create(a_user("live"), some_tokens(clock))

        assert len(store) == 1
        assert store.get(survivor) is not None

    @pytest.mark.parametrize(("ttl", "maximum"), [(0, 10), (-1, 10), (10, 0), (10, -1)])
    def test_nonsense_bounds_are_rejected_at_construction(self, ttl: int, maximum: int) -> None:
        with pytest.raises(ValueError):
            SessionStore(ttl_seconds=ttl, max_sessions=maximum)


class TestTokenExpiry:
    def test_a_token_inside_its_lifetime_is_not_expired(self) -> None:
        tokens = DiscordTokens("a", "r", START + timedelta(hours=1))

        assert tokens.is_expired(now=START) is False

    def test_a_token_about_to_expire_counts_as_expired(self) -> None:
        """The leeway turns a race we would lose silently into a refresh."""
        tokens = DiscordTokens("a", "r", START + timedelta(seconds=30))

        assert tokens.is_expired(now=START, leeway_seconds=60) is True

    def test_a_token_past_its_expiry_is_expired(self) -> None:
        tokens = DiscordTokens("a", "r", START - timedelta(seconds=1))

        assert tokens.is_expired(now=START) is True


class TestOAuthStateStore:
    def test_an_issued_state_validates_once(self) -> None:
        clock = FrozenClock()
        store = OAuthStateStore(ttl_seconds=600, max_states=10, clock=clock)

        state = store.issue()

        assert store.consume(state) is True

    def test_a_state_cannot_be_consumed_twice(self) -> None:
        clock = FrozenClock()
        store = OAuthStateStore(ttl_seconds=600, max_states=10, clock=clock)
        state = store.issue()
        store.consume(state)

        assert store.consume(state) is False

    @pytest.mark.parametrize("unknown", [None, "", "never-issued"])
    def test_an_unknown_state_is_refused(self, unknown: str | None) -> None:
        clock = FrozenClock()
        store = OAuthStateStore(ttl_seconds=600, max_states=10, clock=clock)
        store.issue()

        assert store.consume(unknown) is False

    def test_an_expired_state_is_refused(self) -> None:
        clock = FrozenClock()
        store = OAuthStateStore(ttl_seconds=600, max_states=10, clock=clock)
        state = store.issue()

        clock.advance(601)

        assert store.consume(state) is False

    def test_an_expired_state_is_burnt_even_though_it_failed(self) -> None:
        """Removed before the expiry check, so a replay cannot be timed or retried."""
        clock = FrozenClock()
        store = OAuthStateStore(ttl_seconds=600, max_states=10, clock=clock)
        state = store.issue()
        clock.advance(601)

        store.consume(state)

        assert len(store) == 0

    def test_states_are_distinct_and_long(self) -> None:
        clock = FrozenClock()
        store = OAuthStateStore(ttl_seconds=600, max_states=1000, clock=clock)

        states = {store.issue() for _ in range(200)}

        assert len(states) == 200
        assert all(len(state) >= 43 for state in states)

    def test_the_store_never_grows_past_its_ceiling(self) -> None:
        """/api/auth/login is unauthenticated; the pending-state store must be bounded."""
        clock = FrozenClock()
        store = OAuthStateStore(ttl_seconds=600, max_states=4, clock=clock)

        for _ in range(1000):
            store.issue()

        assert len(store) <= 4

    def test_a_flood_of_logins_can_evict_a_pending_one(self) -> None:
        """Documented consequence of the bound: the victim retries, nothing is bypassed."""
        clock = FrozenClock()
        store = OAuthStateStore(ttl_seconds=600, max_states=2, clock=clock)
        first = store.issue()

        store.issue()
        store.issue()

        assert store.consume(first) is False

    def test_expired_states_are_purged_before_a_pending_one_is_evicted(self) -> None:
        clock = FrozenClock()
        store = OAuthStateStore(ttl_seconds=100, max_states=3, clock=clock)
        for _ in range(3):
            store.issue()

        clock.advance(101)
        survivor = store.issue()

        assert len(store) == 1
        assert store.consume(survivor) is True

    @pytest.mark.parametrize(("ttl", "maximum"), [(0, 10), (-5, 10), (10, 0)])
    def test_nonsense_bounds_are_rejected_at_construction(self, ttl: int, maximum: int) -> None:
        with pytest.raises(ValueError):
            OAuthStateStore(ttl_seconds=ttl, max_states=maximum)


class TestConstantTimeEquals:
    def test_equal_strings_match(self) -> None:
        assert constant_time_equals("abc", "abc") is True

    @pytest.mark.parametrize(
        ("left", "right"), [("abc", "abd"), ("abc", "ab"), ("", "a"), ("abc", "ABC")]
    )
    def test_different_strings_do_not_match(self, left: str, right: str) -> None:
        assert constant_time_equals(left, right) is False

    @pytest.mark.parametrize(("left", "right"), [(None, "a"), ("a", None), (None, None)])
    def test_a_missing_side_never_matches(self, left: str | None, right: str | None) -> None:
        """Two absent cookies must not compare equal and wave a callback through."""
        assert constant_time_equals(left, right) is False

    def test_non_ascii_values_compare_without_raising(self) -> None:
        assert constant_time_equals("héllo", "héllo") is True
        assert constant_time_equals("héllo", "hello") is False
