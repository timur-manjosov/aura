"""Describing a failed model reply in a log line without quoting any of its content (P7a).

A pydantic `ValidationError`'s text includes the offending input -- for a model
reply, that is a distilled sentence, an answer, or the model's quotation of a
member's message. Logs must never hold message or fact text (a deletion would
otherwise leave copies behind in them), so every place that logs a parsing
failure goes through `content_free_reason` instead of formatting the exception.

Imports only pydantic.
"""

from __future__ import annotations

import json
from typing import Final

from pydantic import ValidationError

# The bound on an exception message that is our own wording (a ValueError
# raised by this codebase's validators, which name numbers and limits only).
_MAX_REASON_CHARS: Final = 300
_MAX_LOCATIONS: Final = 5


def content_free_reason(error: BaseException) -> str:
    """Return why a reply was unusable, without any of the reply's text.

    Parameters
    ----------
    error
        What parsing or validating the reply raised.

    Returns
    -------
    str
        For a schema error: the count and the field locations (names and
        indices only); for a JSON error: its type and position; otherwise the
        exception's type and its own message, bounded -- every other exception
        raised on these paths is this codebase's own wording, which names
        numbers and limits, never content.
    """
    if isinstance(error, ValidationError):
        locations = [entry["loc"] for entry in error.errors(include_input=False)][:_MAX_LOCATIONS]
        return f"{error.error_count()} schema error(s) at {locations}"
    if isinstance(error, json.JSONDecodeError):
        return f"JSONDecodeError at line {error.lineno} column {error.colno}"
    return f"{type(error).__name__}: {str(error)[:_MAX_REASON_CHARS]}"
