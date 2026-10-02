"""What one answering-path model call actually used, and whether it was cut off.

Shared by the two calls behind every answer Aura sends -- synthesis
(aura.synthesis) and the grounding check (aura.grounding) -- so both report
usage in one format and recognise a truncated response the same way.

**The usage line replaces guessing.** Every cost figure this project had for
these calls was an estimate from token counts measured once. One INFO line per
call, carrying the provider's own `usage` numbers, makes real spend readable
from the log for later pricing. It deliberately carries no content: no
question, no fact text, no answer, no guild or user id -- only what the call
was for, which model, and how many tokens went in and out.

**A cut-off response is an unparsable one.** Both calls run with an output
ceiling (max_tokens). A response the provider stopped at that ceiling is
incomplete by definition, even in the rare case where what arrived happens to
parse, so both callers reject it on the same path they reject malformed JSON
on: synthesis returns None, the grounding check fails closed.

Imports only litellm's response type.
"""

from __future__ import annotations

import logging
from typing import Final

from litellm.types.utils import ModelResponse

logger = logging.getLogger(__name__)

# litellm normalises every provider's "stopped at the output limit" to the
# OpenAI value "length"; "max_tokens" is Anthropic's own spelling, accepted too
# so a provider passed through un-normalised still fails safe.
_CUT_OFF_FINISH_REASONS: Final = frozenset({"length", "max_tokens"})

# A finish reason is provider-controlled text, not Aura's; bounded so an
# unexpected value cannot land whole in a log line.
_MAX_LOGGED_FINISH_REASON_CHARS: Final = 32


def _finish_reason(response: ModelResponse) -> str | None:
    """Return the first choice's finish reason, or None when there is none."""
    if not response.choices:
        return None
    reason = getattr(response.choices[0], "finish_reason", None)
    return reason if isinstance(reason, str) else None


def _token_count(usage: object, field: str) -> int | None:
    """Return one integer token count from a usage object, or None if it is not one."""
    value = getattr(usage, field, None)
    # bool is an int subclass; a True here would be a provider bug, not a count.
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return None


def was_cut_off(response: ModelResponse) -> bool:
    """Report whether the provider stopped this response at the output limit.

    Parameters
    ----------
    response
        A completed, non-streaming model response.

    Returns
    -------
    bool
        True when the first choice finished because of the output ceiling.
        False for every other reason, including a missing one.
    """
    return _finish_reason(response) in _CUT_OFF_FINISH_REASONS


def log_llm_usage(response: ModelResponse, *, purpose: str, model: str) -> None:
    """Write one INFO line with the tokens a model call used.

    Parameters
    ----------
    response
        A completed, non-streaming model response.
    purpose
        What the call was for, e.g. "synthesis" or "grounding". A fixed label
        from the caller, never user input.
    model
        The configured model string the call was made with.

    Returns
    -------
    None

    Notes
    -----
    Never raises and never logs content: the line holds the purpose, the
    model, the prompt and completion token counts and the finish reason, and
    nothing else. When the provider sent no usage, or a usage without integer
    counts, the line says so instead -- a missing measurement is itself worth
    seeing.
    """
    finish_reason = _finish_reason(response)
    shown_reason = (
        "none" if finish_reason is None else finish_reason[:_MAX_LOGGED_FINISH_REASON_CHARS]
    )
    usage = getattr(response, "usage", None)
    prompt_tokens = _token_count(usage, "prompt_tokens")
    completion_tokens = _token_count(usage, "completion_tokens")
    if prompt_tokens is None or completion_tokens is None:
        logger.info(
            "LLM usage: purpose=%s model=%s usage=missing finish_reason=%s",
            purpose,
            model,
            shown_reason,
        )
        return
    logger.info(
        "LLM usage: purpose=%s model=%s prompt_tokens=%d completion_tokens=%d finish_reason=%s",
        purpose,
        model,
        prompt_tokens,
        completion_tokens,
        shown_reason,
    )
