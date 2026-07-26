"""The one place this tooling talks to a model, so the caps sit on one door.

Every paid call in this package goes through `complete_json`, which authorizes
against the `CallBudget` before the request and books the provider's reported
token usage after it. There is no second path, which is what makes "the script
actually stops after N calls" a property of the code rather than a promise in a
docstring.

JSON parsing reuses `aura.synthesis`'s own parser rather than `json.loads`.
That is deliberate and the reason is measured: the model bake-off found that
`response_format={"type": "json_object"}` is a request, not a guarantee, and
that Anthropic models routed through OpenRouter return correct JSON wrapped in
a markdown fence. Duplicating a second, simpler parser here would mean this
tooling silently fails against exactly the provider the bot itself is
configured for. Importing the production one keeps a single source of truth for
provider quirks -- if that parser learns about a new quirk, this tooling gets it
for free.
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys

import litellm
from litellm.types.utils import ModelResponse

# Not a public name, imported knowingly: see this module's docstring for why a
# local copy of the fence-tolerant parser would be the worse choice.
from aura.synthesis import _parse_json_response
from synthetic_corpus.budget import CallBudget, ModelPrice

logger = logging.getLogger(__name__)

RUN_REAL_LLM_ENV = "AURA_RUN_REAL_LLM"

# Generous, because generation calls return long JSON arrays and a slow one is
# not a failure -- but bounded, because a hung request would otherwise stall a
# sequential run indefinitely with no output at all.
REQUEST_TIMEOUT_SECONDS = 120

# Retries are for a model that returned unusable output, not for network
# errors, which litellm already retries internally. Each attempt is authorized
# and booked separately, so a retry loop cannot outrun the budget.
MAX_ATTEMPTS = 3


class GenerationError(RuntimeError):
    """Raised when a call could not be turned into usable JSON within its attempts."""


def require_real_llm_optin(script_name: str) -> None:
    """Exit the process unless a human explicitly opted this run into paid calls.

    The same single signal `tests/conftest.py` and `scripts/model_bakeoff.py`
    key off, deliberately not `LLM_API_KEY`: importing litellm runs
    `load_dotenv()`, which pulls this repository's real key into the
    environment, so keying on the key would be an always-true check that
    silently spends money.
    """
    if os.environ.get(RUN_REAL_LLM_ENV):
        return
    print(
        f"refusing to run: {RUN_REAL_LLM_ENV} is not set. {script_name} makes real, "
        f"paid LLM calls.\nRe-run with: {RUN_REAL_LLM_ENV}=1 .venv/bin/python "
        f"scripts/{script_name}",
        file=sys.stderr,
    )
    raise SystemExit(1)


def resolve_api_key() -> str:
    """Return the configured provider key, raising if there isn't one.

    Read straight from the environment (which `load_dotenv` has already
    populated from `.env`) rather than through `Settings`, because this tooling
    has no business requiring a Discord token to generate test data.
    """
    key = os.environ.get("LLM_API_KEY")
    if not key:
        raise GenerationError(
            "LLM_API_KEY is not set; this tool cannot generate anything without it"
        )
    return key


async def complete_json(
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    budget: CallBudget,
    price: ModelPrice,
    api_key: str,
    temperature: float,
    max_attempts: int = MAX_ATTEMPTS,
) -> object:
    """Make one budgeted JSON call and return the parsed payload.

    Raises GenerationError when every attempt produced something unparseable,
    and lets BudgetExceededError propagate untouched -- a budget stop is not a
    generation failure to be retried past, it is the run ending.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    last_error = "no attempt was made"

    for attempt in range(1, max_attempts + 1):
        budget.authorize(model)
        try:
            response = await litellm.acompletion(
                model=model,
                api_key=api_key,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=temperature,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            last_error = f"call failed: {type(exc).__name__}: {exc}"
            logger.warning("attempt %d/%d %s", attempt, max_attempts, last_error)
            continue

        if not isinstance(response, ModelResponse):
            last_error = f"expected a ModelResponse, got {type(response).__name__}"
            continue

        usage = getattr(response, "usage", None)
        budget.record(
            price,
            input_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
        )

        content = response.choices[0].message.content
        if not content or not content.strip():
            last_error = "model returned empty content"
            logger.warning("attempt %d/%d %s", attempt, max_attempts, last_error)
            continue

        try:
            return _parse_json_response(content)
        except ValueError as exc:
            last_error = f"unparseable JSON: {exc}"
            logger.warning("attempt %d/%d %s", attempt, max_attempts, last_error)

    raise GenerationError(f"{model} produced no usable JSON in {max_attempts} attempts: {last_error}")
