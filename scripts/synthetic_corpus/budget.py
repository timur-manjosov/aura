"""Hard, in-code caps on how many LLM calls this tooling may make and what it
may spend doing it.

A comment saying "this run should cost about twenty cents" is not a cap. This
is: `authorize()` raises before the call that would exceed the ceiling, and
`record()` raises after the call that pushed spend past it. Either way the run
stops -- it does not warn and continue, and it does not sample-and-hope.

Two ceilings rather than one, because they fail for different reasons. A call
count catches a runaway loop (a retry path that never gives up, a generator
asked for ten thousand items); a dollar ceiling catches a call that is
individually far more expensive than estimated (a model that answers a
600-token prompt with 60,000 tokens of output, or a price that moved since the
estimate was printed). Neither one subsumes the other.

Spend is computed from the provider's own reported token usage against a price
table captured at run time, not from an estimate: an estimate that drifts is
exactly the failure a cap exists to catch.
"""
from __future__ import annotations

from dataclasses import dataclass, field

_USD_PER_TOKEN_SCALE = 1_000_000


class BudgetExceededError(RuntimeError):
    """Raised when a run would exceed its call or spend ceiling.

    Its own type so the entry points can catch it, write out whatever was
    already generated, and report the partial result honestly -- rather than
    losing the work to a bare RuntimeError somewhere up the stack.
    """


@dataclass(frozen=True)
class ModelPrice:
    """USD per million prompt and completion tokens for one model.

    Populated from OpenRouter's live catalog at run time (see
    `pricing.fetch_model_prices`), never from a constant in this repository:
    the model bake-off already found one price in this project's own notes had
    gone stale by 3.7x, and a cap computed from a stale price is not a cap.
    """

    model: str
    usd_per_million_input: float
    usd_per_million_output: float

    def cost(self, *, input_tokens: int, output_tokens: int) -> float:
        """USD for one call with this token usage."""
        return (
            input_tokens * self.usd_per_million_input
            + output_tokens * self.usd_per_million_output
        ) / _USD_PER_TOKEN_SCALE


@dataclass
class CallBudget:
    """A call-count and spend ceiling, enforced at the moment of the call.

    Not thread-safe and not intended to be: every consumer of this class drives
    its calls sequentially from one coroutine, which is also what keeps the
    provider's rate limiter out of the measurement.
    """

    max_calls: int
    max_spend_usd: float
    calls: int = 0
    spent_usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    per_model_calls: dict[str, int] = field(default_factory=dict)
    # Per-model totals, kept because an aggregate hides the failure mode that
    # actually happened on the first run of this tooling: one model quietly
    # emitted 13,000 reasoning tokens per call for a two-field JSON verdict.
    # That is invisible in a combined figure and obvious in a split one.
    per_model_usage: dict[str, tuple[int, int, float]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.max_calls < 0:
            raise ValueError(f"max_calls must be non-negative, got {self.max_calls}")
        if self.max_spend_usd < 0:
            raise ValueError(f"max_spend_usd must be non-negative, got {self.max_spend_usd}")

    def authorize(self, model: str) -> None:
        """Reserve one call against the ceiling, raising if none is left.

        Called immediately before the request goes out, and counts the call
        whether or not it succeeds. Counting attempts rather than successes is
        deliberate: a failing call still costs the provider's tokens on some
        error paths, and a cap that only counted successes would let an
        infinite retry loop run forever for free-looking reasons.
        """
        if self.calls >= self.max_calls:
            raise BudgetExceededError(
                f"call cap reached: {self.calls}/{self.max_calls} calls already made "
                f"(next would be to {model}). Raise --max-calls deliberately if this "
                "run genuinely needs more."
            )
        self.calls += 1
        self.per_model_calls[model] = self.per_model_calls.get(model, 0) + 1

    def record(self, price: ModelPrice, *, input_tokens: int, output_tokens: int) -> None:
        """Book one call's measured token usage, raising if it breaks the spend ceiling.

        Raises *after* booking, so the reported total includes the call that
        broke the ceiling -- an honest number to put in the report, and the
        overrun is bounded by one call either way.
        """
        cost = price.cost(input_tokens=input_tokens, output_tokens=output_tokens)
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.spent_usd += cost

        previous = self.per_model_usage.get(price.model, (0, 0, 0.0))
        self.per_model_usage[price.model] = (
            previous[0] + input_tokens,
            previous[1] + output_tokens,
            previous[2] + cost,
        )

        if self.spent_usd > self.max_spend_usd:
            raise BudgetExceededError(
                f"spend cap reached: ${self.spent_usd:.4f} spent against a "
                f"${self.max_spend_usd:.4f} ceiling after {self.calls} calls."
            )

    @property
    def remaining_calls(self) -> int:
        """How many further calls the ceiling still permits."""
        return max(0, self.max_calls - self.calls)

    def summary(self) -> str:
        """One-line spend report, for the console and the written report."""
        return (
            f"{self.calls} calls, {self.input_tokens} input + {self.output_tokens} "
            f"output tokens, ${self.spent_usd:.4f} spent "
            f"(ceiling: {self.max_calls} calls / ${self.max_spend_usd:.2f})"
        )

    def per_model_summary(self) -> list[str]:
        """One line per model: calls, tokens and spend, for the written report."""
        lines: list[str] = []
        for model, (input_tokens, output_tokens, cost) in sorted(self.per_model_usage.items()):
            calls = self.per_model_calls.get(model, 0)
            per_call_output = output_tokens / calls if calls else 0.0
            lines.append(
                f"{model:<48} {calls:>4} calls  {input_tokens:>8} in  "
                f"{output_tokens:>8} out  ({per_call_output:>7.0f} out/call)  ${cost:.4f}"
            )
        return lines
