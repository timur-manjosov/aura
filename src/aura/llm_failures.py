"""Why a model call failed, as far as the bot acts on it -- and the alarm when the key is refused.

Every model call site in Aura catches its own failures and returns a safe value
(None, CHECK_FAILED, silence). Two decisions need more than "it failed":

* **Hold or drop** (aura.extraction): a batch whose call did not complete for a
  reason outside the batch -- a timeout, a provider or network error, a refused
  key -- is held and tried again later (aura.extraction.verify_retry). A call
  refused because of the request itself (HTTP 400, a moderation refusal, an
  unknown model) or an unusable reply is NOT retried: the same batch would most
  likely be refused again, and a batch crafted to be refused must not cost more
  than one attempt.
* **The key alarm** (operator): when the shared key is refused -- its spending
  limit or the account's credits are exhausted (OpenRouter: HTTP 402, or HTTP
  403 "Key limit exceeded"), or the key itself is invalid (HTTP 401) -- every
  model call fails until someone acts, and nothing else in the product says so.
  One ERROR line per hour at most, and /aura-operator-budget shows the latest
  refusal.

**Classified by status code, plus one anchored phrase.** litellm maps provider
errors to exceptions that carry the HTTP status (`status_code`); its exact
exception CLASS for OpenRouter's 402/403 differs between versions (measured: a
plain `APIError` with the status in litellm 1.93), so the class is never relied
on. A 403 alone is ambiguous -- OpenRouter also answers 403 to a moderation
refusal, whose body can echo the flagged input -- so the key limit is
recognised by OpenRouter's own phrase, accepted only as the START of the
provider's message (`"message":"Key limit exceeded` in the raw body, or
directly after litellm's "<Provider>Exception - " prefix). Text quoted from a
request sits inside an escaped JSON string and cannot produce either form.

**No secret, no provider text in any line this module writes.** The ERROR line
names the kind of refusal, the call's fixed purpose label and the configured
model string; never the exception text, which for OpenRouter contains a link
naming the key.

**State in memory only.** The alarm needs no durability: a restart forgets the
last refusal, and the next refused call records it again. Not thread-safe and
need not be: every caller runs on the bot's one event loop and nothing here
awaits.

Imports nothing from Aura, Discord or the database.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Final

import httpx
import openai

logger = logging.getLogger(__name__)


class CallFailureKind(Enum):
    """Why a model call failed, in the categories the bot acts on."""

    KEY_LIMIT = "key_limit"
    """The key's spending limit or the account's credits are exhausted."""

    KEY_INVALID = "key_invalid"
    """The key itself was refused (invalid, revoked, deleted)."""

    TRANSPORT = "transport"
    """The call did not complete for a reason outside the request: a timeout, a
    rate limit, a provider or network error."""

    REQUEST = "request"
    """Refused because of the request itself, or not recognised: never retried."""


# The kinds a held batch may be retried after: the call says nothing about the batch.
_RETRYABLE: Final[frozenset[CallFailureKind]] = frozenset(
    {CallFailureKind.KEY_LIMIT, CallFailureKind.KEY_INVALID, CallFailureKind.TRANSPORT}
)

# The kinds that mean every call fails until a human acts on the key.
_KEY_REFUSALS: Final[frozenset[CallFailureKind]] = frozenset(
    {CallFailureKind.KEY_LIMIT, CallFailureKind.KEY_INVALID}
)

# Statuses that say nothing about the request: request timeout, rate limit.
# Every 5xx is added in classify_call_failure.
_TRANSPORT_STATUSES: Final[frozenset[int]] = frozenset({408, 429})

# The start of OpenRouter's own message for a refused key limit or exhausted
# credits, in the two forms litellm puts it in an exception's text: inside the
# raw JSON body, or right after the "<Provider>Exception - " prefix. Anchored so
# that a phrase quoted from a request (escaped inside a JSON string) never
# matches; see the module docstring.
_KEY_LIMIT_MESSAGE: Final = re.compile(
    r'(?:"message"\s*:\s*"|Exception\s*-\s*)\s*'
    r"(?:key limit exceeded|insufficient credits|this request requires more credits)",
    re.IGNORECASE,
)

# Exception text beyond this is not searched: the phrase sits at the start of
# the provider's message, and an unbounded string must not cost unbounded time.
_MAX_SEARCHED_CHARS: Final = 2000

# The failures without a status code that mean the call never completed.
_TRANSPORT_EXCEPTIONS: Final[tuple[type[BaseException], ...]] = (
    TimeoutError,
    OSError,
    httpx.TransportError,
    openai.APIConnectionError,
)

# At most one alarm line per this interval, however often the key is refused.
ALARM_LOG_INTERVAL: Final = timedelta(hours=1)


def _status_code(exc: BaseException) -> int | None:
    """Return the HTTP status an exception carries, or None when it carries none."""
    for candidate in (
        getattr(exc, "status_code", None),
        getattr(getattr(exc, "response", None), "status_code", None),
    ):
        # bool is an int subclass; a True here would be nobody's status code.
        if isinstance(candidate, int) and not isinstance(candidate, bool):
            return candidate
    return None


def _names_key_limit(exc: BaseException) -> bool:
    """Report whether the provider's own message starts with the key-limit phrase."""
    try:
        text = str(exc)
    except Exception:
        # An exception whose own __str__ fails must not turn a handled call
        # failure into an unhandled one inside the caller's except block.
        return False
    return _KEY_LIMIT_MESSAGE.search(text[:_MAX_SEARCHED_CHARS]) is not None


def classify_call_failure(exc: BaseException) -> CallFailureKind:
    """Return why a model call raised `exc`.

    Parameters
    ----------
    exc
        What the call raised.

    Returns
    -------
    CallFailureKind
        KEY_LIMIT for HTTP 402, and for any failure whose provider message
        starts with the key-limit phrase (OpenRouter's 403); KEY_INVALID for
        HTTP 401; TRANSPORT for 408, 429, every 5xx, and a timeout or
        connection failure without a status; REQUEST for everything else,
        including 400, any other 403, 404 and 422.

    Notes
    -----
    Never raises. The phrase is checked whatever the status, so a litellm
    version that wraps the 403 differently (another class, another status)
    still raises the alarm; the anchoring makes that safe.

    Unknown failures are REQUEST, the direction that neither retries nor raises
    an alarm: a retry that should have happened loses one batch, as every
    failure did before P5; one that should not have can be triggered again by
    whoever crafted the batch.
    """
    status = _status_code(exc)
    if status == 402 or _names_key_limit(exc):
        return CallFailureKind.KEY_LIMIT
    if status == 401:
        return CallFailureKind.KEY_INVALID
    if status is not None:
        if status in _TRANSPORT_STATUSES or 500 <= status <= 599:
            return CallFailureKind.TRANSPORT
        return CallFailureKind.REQUEST
    if isinstance(exc, _TRANSPORT_EXCEPTIONS):
        return CallFailureKind.TRANSPORT
    return CallFailureKind.REQUEST


def is_retryable(kind: CallFailureKind) -> bool:
    """Report whether a batch whose call failed this way may be held and tried again."""
    return kind in _RETRYABLE


@dataclass(frozen=True)
class KeyAlarmStatus:
    """What the operator view shows about refusals of the key.

    Attributes
    ----------
    watching_since
        When this process started recording.
    refusals
        How many calls were refused because of the key since then.
    last_refusal_at
        When the latest one happened; None when there was none.
    last_refusal_kind
        KEY_LIMIT or KEY_INVALID; None when there was none.
    last_refusal_purpose
        The fixed purpose label of the refused call; None when there was none.
    last_success_at
        When a model call last returned a response; None when none has.
    """

    watching_since: datetime
    refusals: int
    last_refusal_at: datetime | None
    last_refusal_kind: CallFailureKind | None
    last_refusal_purpose: str | None
    last_success_at: datetime | None

    @property
    def succeeded_since_refusal(self) -> bool:
        """Report whether a model call returned a response after the latest refusal."""
        return (
            self.last_refusal_at is not None
            and self.last_success_at is not None
            and self.last_success_at > self.last_refusal_at
        )


_REFUSAL_REASONS: Final[dict[CallFailureKind, str]] = {
    CallFailureKind.KEY_LIMIT: "its spending limit or the account's credits are exhausted",
    CallFailureKind.KEY_INVALID: "the key is invalid or revoked",
}


class KeyAlarm:
    """Records refusals of the LLM key and raises a rate-limited ERROR line.

    Notes
    -----
    One instance per process (`KEY_ALARM`). Not thread-safe and need not be:
    see the module docstring.
    """

    def __init__(self, *, started_at: datetime | None = None) -> None:
        self._started_at = started_at or datetime.now(UTC)
        self._refusals = 0
        self._last_refusal_at: datetime | None = None
        self._last_refusal_kind: CallFailureKind | None = None
        self._last_refusal_purpose: str | None = None
        self._last_logged_at: datetime | None = None
        self._last_success_at: datetime | None = None

    def reset(self, *, started_at: datetime | None = None) -> None:
        """Forget everything recorded (tests)."""
        self._started_at = started_at or datetime.now(UTC)
        self._refusals = 0
        self._last_refusal_at = None
        self._last_refusal_kind = None
        self._last_refusal_purpose = None
        self._last_logged_at = None
        self._last_success_at = None

    def note_refusal(
        self, kind: CallFailureKind, *, purpose: str, model: str, now: datetime
    ) -> bool:
        """Record one refused call; log the alarm line unless one was logged within the hour.

        Parameters
        ----------
        kind
            KEY_LIMIT or KEY_INVALID; any other kind is ignored.
        purpose
            The call's fixed purpose label (e.g. "extraction"), never user input.
        model
            The configured model string.
        now
            Timezone-aware moment of the refusal.

        Returns
        -------
        bool
            Whether this call logged the ERROR line.
        """
        if kind not in _KEY_REFUSALS:
            return False
        self._refusals += 1
        self._last_refusal_at = now
        self._last_refusal_kind = kind
        self._last_refusal_purpose = purpose
        if self._last_logged_at is not None and now - self._last_logged_at < ALARM_LOG_INTERVAL:
            return False
        self._last_logged_at = now
        logger.error(
            "LLM key refused: %s. Every model call fails until this is fixed with "
            "the LLM provider (refused call: purpose=%s, model=%s). This line is "
            "repeated at most once an hour; /aura-operator-budget shows the latest "
            "refusal.",
            _REFUSAL_REASONS[kind],
            purpose,
            model,
        )
        return True

    def note_success(self, now: datetime) -> None:
        """Record that a model call returned a response."""
        self._last_success_at = now

    def status(self) -> KeyAlarmStatus:
        """Return what has been recorded, for the operator view."""
        return KeyAlarmStatus(
            watching_since=self._started_at,
            refusals=self._refusals,
            last_refusal_at=self._last_refusal_at,
            last_refusal_kind=self._last_refusal_kind,
            last_refusal_purpose=self._last_refusal_purpose,
            last_success_at=self._last_success_at,
        )


# The one instance every call site shares for the process's life.
KEY_ALARM: Final = KeyAlarm()


def record_call_failure(exc: BaseException, *, purpose: str, model: str) -> CallFailureKind:
    """Classify a failed model call and feed a refusal of the key to the alarm.

    Parameters
    ----------
    exc
        What the call raised.
    purpose
        The call site's fixed purpose label (the one aura.llm_usage logs).
    model
        The configured model string the call was made with.

    Returns
    -------
    CallFailureKind
        The classification (see `classify_call_failure`), for a caller that
        decides whether to hold its work.

    Notes
    -----
    Never raises, and never logs the exception: the caller logs its own line,
    as before.
    """
    kind = classify_call_failure(exc)
    KEY_ALARM.note_refusal(kind, purpose=purpose, model=model, now=datetime.now(UTC))
    return kind


def record_call_success() -> None:
    """Note that a model call returned a response (called from aura.llm_usage)."""
    KEY_ALARM.note_success(datetime.now(UTC))
