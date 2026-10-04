"""Metered, ceiling-guarded LLM calls for the paid harnesses under scripts/.

Every paid harness in this repository spends real money, and since P3 the rule
is that spend is metered and capped, never merely estimated in advance.
`install_metering` wraps `litellm.acompletion` for the rest of the process so
that every call -- made by the shipped code the harness drives, not by a copy of
it -- is checked against a ceiling BEFORE any request leaves, and recorded with
the provider's own token counts and cost afterwards.

The ledger is one JSON file shared by every invocation of one task, with a
ceiling per BUCKET (for example "dev", "bakeoff", "checker") and one TOTAL
ceiling over all buckets. A call is refused when the recorded spend, plus every
call still in flight anywhere, plus its own worst case, could cross either the
bucket's ceiling or the total.

Correct under three kinds of trouble:

* **Parallel workers in one process** (asyncio tasks): the check and the
  reservation happen with no await in between, so no two tasks can both pass a
  check that only one of them fits.
* **Parallel processes on one ledger**: every check, reservation and record
  happens under an exclusive `fcntl` lock on a sibling lock file, against the
  ledger as it is on disk at that moment, and reservations are written to the
  file, so a second process sees the first one's calls in flight.
* **A crash**: the file is replaced atomically (write, fsync, rename), so it is
  never half-written. A reservation left behind by a process that no longer
  runs is booked as SPENT at its worst case the next time the ledger is opened
  -- the provider may have charged for that call, and the safe direction is to
  count it rather than forget it. Results the harness already has are kept in
  its own result files, so a resumed run skips them instead of paying twice.

Worst case per call, for the guard: every prompt character counted as one token
(true for Japanese, a large overestimate for Latin script) and the full output
ceiling spent, at the model's input and output price. A model with no known
price is refused, never guessed at.

Not a test and never imported by the bot: tests/conftest.py keeps the suite
hermetic, and only a human who sets AURA_RUN_REAL_LLM runs a harness for real.
"""

from __future__ import annotations

import contextlib
import contextvars
import fcntl
import json
import os
import time
import uuid
from collections.abc import Awaitable, Callable, Generator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import litellm

RUN_REAL_LLM_ENV: Final = "AURA_RUN_REAL_LLM"

_SAFETY_TOKENS_PER_PROMPT_CHAR: Final = 1.0

# The output ceiling the metering adds to a call the shipped code sends WITHOUT
# one (fact extraction, the supersession judge and the variant calls ship
# unbounded). Without it the ledger has no worst case to guard. Recorded on the
# call's entry; no reply in the bake-off came near it.
HARNESS_MAX_TOKENS: Final = 4096

# The tag of the call the current task is about to make; harnesses set it per
# case so a record can be matched back to the case that caused it.
CURRENT_TAG: contextvars.ContextVar[str] = contextvars.ContextVar("current_tag", default="")


class CeilingReachedError(RuntimeError):
    """Raised before a call that could take a bucket or the total past its ceiling."""


@dataclass(frozen=True)
class ModelPrice:
    """A model's price per million tokens.

    Attributes
    ----------
    input_usd
        Price per million prompt tokens.
    output_usd
        Price per million completion tokens (reasoning tokens included).
    """

    input_usd: float
    output_usd: float


def worst_case_usd(
    price: ModelPrice, messages: Sequence[Mapping[str, Any]], max_tokens: int
) -> float:
    """Return the most one call could cost, for the ceiling guard.

    Parameters
    ----------
    price
        The model's price.
    messages
        The chat messages of the call.
    max_tokens
        The call's output ceiling.

    Returns
    -------
    float
        USD, with every prompt character counted as one token.
    """
    prompt_chars = sum(len(str(message.get("content", ""))) for message in messages)
    worst_prompt_tokens = prompt_chars * _SAFETY_TOKENS_PER_PROMPT_CHAR
    return (worst_prompt_tokens * price.input_usd + max_tokens * price.output_usd) / 1_000_000


def _pid_alive(pid: int) -> bool:
    """Report whether a process with this id is running (on this machine)."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


@dataclass
class LedgerState:
    """The ledger's content as stored on disk.

    Attributes
    ----------
    ceilings
        USD ceiling per bucket.
    total_ceiling
        USD ceiling over all buckets together.
    calls
        One entry per completed call (and per orphaned reservation booked at
        its worst case).
    reservations
        Calls in flight: reservation id -> {"pid", "bucket", "usd", "at"}.
    """

    ceilings: dict[str, float]
    total_ceiling: float
    calls: list[dict[str, Any]] = field(default_factory=list)
    reservations: dict[str, dict[str, Any]] = field(default_factory=dict)

    def spent(self, bucket: str | None = None) -> float:
        """Return recorded spend, for one bucket or (None) for all."""
        return sum(
            float(call["usd"]) for call in self.calls if bucket is None or call["bucket"] == bucket
        )

    def reserved(self, bucket: str | None = None) -> float:
        """Return the worst cases of calls in flight, for one bucket or all."""
        return sum(
            float(entry["usd"])
            for entry in self.reservations.values()
            if bucket is None or entry["bucket"] == bucket
        )


class Ledger:
    """A file-backed ledger with per-bucket ceilings and a total ceiling.

    Parameters
    ----------
    path
        The JSON file; created on first use. A sibling ``.lock`` file
        serializes every access across processes.
    ceilings
        USD ceiling per bucket. A call to a bucket not listed here is refused.
    total_ceiling
        USD ceiling over every bucket together.

    Notes
    -----
    The ceilings passed in are the ones enforced; they are also written to the
    file for the record. Every method that reads or writes the file takes the
    lock for its whole duration and reads the file fresh, so two processes
    never act on stale state.
    """

    def __init__(self, path: Path, ceilings: Mapping[str, float], total_ceiling: float) -> None:
        self.path = path
        self.lock_path = path.with_suffix(path.suffix + ".lock")
        self.ceilings = dict(ceilings)
        self.total_ceiling = total_ceiling
        with self._locked() as state:
            self._book_orphans(state)

    @contextlib.contextmanager
    def _locked(self) -> Generator[LedgerState]:
        """Hold the exclusive lock, yield the state from disk, write it back afterwards."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                state = self._read()
                yield state
                self._write(state)
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def _read(self) -> LedgerState:
        if not self.path.exists():
            return LedgerState(ceilings=dict(self.ceilings), total_ceiling=self.total_ceiling)
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        return LedgerState(
            ceilings=dict(self.ceilings),
            total_ceiling=self.total_ceiling,
            calls=list(payload.get("calls", [])),
            reservations=dict(payload.get("reservations", {})),
        )

    def _write(self, state: LedgerState) -> None:
        by_bucket: dict[str, dict[str, float]] = {}
        by_model: dict[str, dict[str, float]] = {}
        by_label: dict[str, dict[str, float]] = {}
        for call in state.calls:
            for key, table in (
                (call["bucket"], by_bucket),
                (call["model"], by_model),
                (call["label"], by_label),
            ):
                entry = table.setdefault(
                    key, {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "usd": 0.0}
                )
                entry["calls"] += 1
                entry["prompt_tokens"] += int(call.get("prompt_tokens") or 0)
                entry["completion_tokens"] += int(call.get("completion_tokens") or 0)
                entry["usd"] += float(call["usd"])
        payload = {
            "ceilings": state.ceilings,
            "total_ceiling": state.total_ceiling,
            "spent_usd": round(state.spent(), 6),
            "reserved_usd": round(state.reserved(), 6),
            "by_bucket": by_bucket,
            "by_model": by_model,
            "by_label": by_label,
            "reservations": state.reservations,
            "calls": state.calls,
        }
        temporary = self.path.with_suffix(self.path.suffix + f".{os.getpid()}.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=1)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, self.path)

    def _book_orphans(self, state: LedgerState) -> None:
        """Book every reservation of a process that no longer runs as spent at its worst case."""
        for reservation_id, entry in list(state.reservations.items()):
            if int(entry["pid"]) == os.getpid() or _pid_alive(int(entry["pid"])):
                continue
            state.calls.append(
                {
                    "at": entry.get("at", ""),
                    "bucket": entry["bucket"],
                    "label": entry.get("label", "orphaned"),
                    "tag": entry.get("tag", ""),
                    "model": entry.get("model", "?"),
                    "provider": None,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "reasoning_tokens": 0,
                    "finish_reason": "orphaned-reservation",
                    "seconds": 0.0,
                    "usd": float(entry["usd"]),
                    "usd_source": "worst-case (process ended mid-call)",
                }
            )
            del state.reservations[reservation_id]

    def reserve(self, *, bucket: str, usd: float, model: str, label: str, tag: str) -> str:
        """Reserve a call's worst case, or refuse it.

        Parameters
        ----------
        bucket
            The budget bucket the call belongs to.
        usd
            The call's worst case.
        model, label, tag
            Stored with the reservation, for the record.

        Returns
        -------
        str
            The reservation id, to pass to `record` or `release`.

        Raises
        ------
        CeilingReachedError
            If the bucket is unknown, or the call could cross the bucket's
            ceiling or the total.
        """
        if bucket not in self.ceilings:
            raise CeilingReachedError(f"unknown budget bucket {bucket!r}; refusing the call")
        with self._locked() as state:
            self._book_orphans(state)
            bucket_after = state.spent(bucket) + state.reserved(bucket) + usd
            total_after = state.spent() + state.reserved() + usd
            if bucket_after > self.ceilings[bucket]:
                raise CeilingReachedError(
                    f"next {model} call (worst case ${usd:.5f}) could cross the "
                    f"{bucket!r} ceiling of ${self.ceilings[bucket]:.2f} "
                    f"(${state.spent(bucket):.4f} spent)"
                )
            if total_after > self.total_ceiling:
                raise CeilingReachedError(
                    f"next {model} call (worst case ${usd:.5f}) could cross the total "
                    f"ceiling of ${self.total_ceiling:.2f} (${state.spent():.4f} spent)"
                )
            reservation_id = uuid.uuid4().hex
            state.reservations[reservation_id] = {
                "pid": os.getpid(),
                "bucket": bucket,
                "usd": usd,
                "model": model,
                "label": label,
                "tag": tag,
                "at": datetime.now(UTC).isoformat(timespec="seconds"),
            }
            return reservation_id

    def record(self, reservation_id: str, entry: dict[str, Any]) -> None:
        """Replace a reservation by the completed call's entry.

        Parameters
        ----------
        reservation_id
            The id `reserve` returned.
        entry
            The call's record; must carry "bucket" and "usd".
        """
        with self._locked() as state:
            state.reservations.pop(reservation_id, None)
            state.calls.append(entry)

    def release(self, reservation_id: str) -> None:
        """Drop a reservation whose call provably sent nothing (it raised before sending)."""
        with self._locked() as state:
            state.reservations.pop(reservation_id, None)

    def snapshot(self) -> LedgerState:
        """Return the current state from disk (under the lock)."""
        with self._locked() as state:
            return LedgerState(
                ceilings=dict(state.ceilings),
                total_ceiling=state.total_ceiling,
                calls=list(state.calls),
                reservations=dict(state.reservations),
            )


@dataclass(frozen=True)
class ArmRouting:
    """Request options one experiment arm adds to every call (OpenRouter's own fields).

    Attributes
    ----------
    provider_order
        Pin these OpenRouter providers, in order, with no fallback to others;
        empty means OpenRouter's default routing.
    data_collection_deny
        Ask OpenRouter to use only providers that do not retain or train on
        the data ("data_collection": "deny").
    reasoning
        OpenRouter's reasoning object for the call (for example
        {"enabled": False} or {"effort": "low"}), or None for the model's
        default.
    max_tokens
        Override of the call's output ceiling, or None to keep the shipped one
        (a reasoning arm needs room for its thinking tokens).
    temperature
        Override of the call's temperature, or None to keep the shipped 0.0.
    """

    provider_order: tuple[str, ...] = ()
    data_collection_deny: bool = False
    reasoning: Mapping[str, Any] | None = None
    max_tokens: int | None = None
    temperature: float | None = None

    def extra_body(self) -> dict[str, Any]:
        """Return the OpenRouter request fields for this arm, usage accounting always on."""
        body: dict[str, Any] = {"usage": {"include": True}}
        provider: dict[str, Any] = {}
        if self.provider_order:
            provider["order"] = list(self.provider_order)
            provider["allow_fallbacks"] = False
        if self.data_collection_deny:
            provider["data_collection"] = "deny"
        if provider:
            body["provider"] = provider
        if self.reasoning is not None:
            body["reasoning"] = dict(self.reasoning)
        return body


@dataclass(frozen=True)
class CallRecord:
    """What one metered call returned, keyed by the tag its caller set."""

    raw_content: str | None
    prompt_tokens: int
    completion_tokens: int
    reasoning_tokens: int
    finish_reason: str | None
    provider: str | None
    seconds: float
    usd: float
    usd_source: str


def _usage_numbers(response: Any) -> tuple[int | None, int | None, int, float | None]:
    """Return prompt, completion and reasoning tokens and the provider-reported cost."""
    usage = getattr(response, "usage", None)
    prompt = getattr(usage, "prompt_tokens", None)
    completion = getattr(usage, "completion_tokens", None)
    details = getattr(usage, "completion_tokens_details", None)
    reasoning = getattr(details, "reasoning_tokens", None) if details is not None else None
    cost = getattr(usage, "cost", None)
    if cost is None and usage is not None:
        extra = getattr(usage, "model_extra", None) or {}
        cost = extra.get("cost")
    return (
        prompt if isinstance(prompt, int) else None,
        completion if isinstance(completion, int) else None,
        reasoning if isinstance(reasoning, int) else 0,
        float(cost) if isinstance(cost, int | float) else None,
    )


def _provider_name(response: Any) -> str | None:
    """Return the OpenRouter provider that served a response, if litellm kept it."""
    for source in (
        getattr(response, "provider", None),
        (getattr(response, "model_extra", None) or {}).get("provider"),
        (getattr(response, "_hidden_params", None) or {}).get("provider"),
    ):
        if isinstance(source, str) and source:
            return source
    return None


def install_metering(
    ledger: Ledger,
    *,
    bucket: str,
    label: str,
    records: dict[str, CallRecord],
    prices: Mapping[str, ModelPrice],
    routing: Mapping[str, ArmRouting] | None = None,
) -> Callable[[], None]:
    """Wrap `litellm.acompletion` so every call is ceiling-checked and recorded.

    Parameters
    ----------
    ledger
        Receives one entry per completed call.
    bucket
        The budget bucket every call of this installation belongs to.
    label
        Stored with every entry, so a ledger shared by several runs stays
        readable per run.
    records
        Receives the raw response per call, keyed by the caller's tag
        (`CURRENT_TAG`).
    prices
        Price by model string. A model missing here is refused.
    routing
        Request options per model string (provider pinning, reasoning,
        overrides). A model missing here gets OpenRouter's defaults plus usage
        accounting.

    Returns
    -------
    Callable[[], None]
        Restores the original `litellm.acompletion`.

    Raises
    ------
    CeilingReachedError
        From inside the wrapped call, before any request is sent.

    Notes
    -----
    The cost recorded is the provider's own (`usage.cost`, OpenRouter usage
    accounting) whenever it is reported, and otherwise the token counts at the
    listed price; a response with no usage at all is booked at its worst case.
    A call that raises before a response arrives is booked at its worst case
    too, unless the exception proves nothing was sent (a ceiling refusal). A
    call the shipped code sends without an output ceiling gets
    `HARNESS_MAX_TOKENS` (or the arm's own ceiling), so its worst case exists.
    """
    original: Callable[..., Awaitable[Any]] = litellm.acompletion
    arm_routing = dict(routing or {})

    async def metered(*args: Any, **kwargs: Any) -> Any:
        model = str(kwargs.get("model", ""))
        price = prices.get(model)
        if price is None:
            raise CeilingReachedError(f"no known price for {model!r}; refusing to call it")
        options = arm_routing.get(model, ArmRouting())
        harness_bound = False
        if options.max_tokens is not None:
            kwargs["max_tokens"] = options.max_tokens
        elif "max_tokens" not in kwargs:
            kwargs["max_tokens"] = HARNESS_MAX_TOKENS
            harness_bound = True
        if options.temperature is not None:
            kwargs["temperature"] = options.temperature
        extra = dict(kwargs.get("extra_body") or {})
        extra.update(options.extra_body())
        kwargs["extra_body"] = extra
        worst = worst_case_usd(price, kwargs.get("messages", []), int(kwargs["max_tokens"]))
        tag = CURRENT_TAG.get()
        reservation = ledger.reserve(bucket=bucket, usd=worst, model=model, label=label, tag=tag)
        started = time.perf_counter()
        try:
            response = await original(*args, **kwargs)
        except BaseException as exc:
            seconds = time.perf_counter() - started
            ledger.record(
                reservation,
                {
                    "at": datetime.now(UTC).isoformat(timespec="seconds"),
                    "bucket": bucket,
                    "label": label,
                    "tag": tag,
                    "model": model,
                    "provider": None,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "reasoning_tokens": 0,
                    "finish_reason": f"error:{type(exc).__name__}",
                    "seconds": round(seconds, 3),
                    "usd": worst,
                    "usd_source": "worst-case (call raised; it may have been charged)",
                },
            )
            raise
        seconds = time.perf_counter() - started
        prompt, completion, reasoning, reported_cost = _usage_numbers(response)
        if reported_cost is not None:
            usd, source = reported_cost, "provider usage.cost"
        elif prompt is not None and completion is not None:
            usd = (prompt * price.input_usd + completion * price.output_usd) / 1_000_000
            source = "tokens at listed price"
        else:
            usd, source = worst, "worst-case (no usage reported)"
        choices = getattr(response, "choices", None)
        choice = choices[0] if choices else None
        finish_reason = getattr(choice, "finish_reason", None)
        content = getattr(getattr(choice, "message", None), "content", None)
        provider = _provider_name(response)
        ledger.record(
            reservation,
            {
                "at": datetime.now(UTC).isoformat(timespec="seconds"),
                "bucket": bucket,
                "label": label,
                "tag": tag,
                "model": model,
                "provider": provider,
                "prompt_tokens": prompt or 0,
                "completion_tokens": completion or 0,
                "reasoning_tokens": reasoning,
                "finish_reason": finish_reason,
                "seconds": round(seconds, 3),
                "usd": usd,
                "usd_source": source,
                "harness_max_tokens_added": harness_bound,
            },
        )
        records[tag] = CallRecord(
            raw_content=content if isinstance(content, str) else None,
            prompt_tokens=prompt or 0,
            completion_tokens=completion or 0,
            reasoning_tokens=reasoning,
            finish_reason=finish_reason if isinstance(finish_reason, str) else None,
            provider=provider,
            seconds=seconds,
            usd=usd,
            usd_source=source,
        )
        return response

    litellm.acompletion = metered

    def restore() -> None:
        litellm.acompletion = original

    return restore
