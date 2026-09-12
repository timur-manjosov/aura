"""Machine-readable error codes, and the JSON envelope that carries them.

This service never returns user-facing prose. CLAUDE.md forbids hardcoded
user-facing text inside business logic and requires every string a user reads
to come from a locale file with en-US as a mandatory fallback -- so the
backend emits a stable code and the frontend resolves it through the same
translation-key mechanism the bot's commands use (web/frontend/lib/i18n.ts).

The practical payoff is the same one the bot's t() seam already has: adding
a language is a new locale file and no backend change, and an error the
frontend has never heard of degrades to a visible key rather than a blank
screen.
"""
from __future__ import annotations

from enum import StrEnum

from fastapi.responses import JSONResponse


class ErrorCode(StrEnum):
    """Every error this service can return to a browser."""

    NOT_AUTHENTICATED = "not_authenticated"
    INVALID_STATE = "invalid_state"
    OAUTH_DENIED = "oauth_denied"
    OAUTH_FAILED = "oauth_failed"
    DISCORD_UNAVAILABLE = "discord_unavailable"


def error_response(code: ErrorCode, status_code: int) -> JSONResponse:
    """Build the one error shape this service emits.

    Deliberately just the code -- no message, no exception text, no upstream
    status. Everything a reader needs to diagnose a failure is in this
    process's logs, where it is not also handed to whoever triggered it.
    """
    return JSONResponse(status_code=status_code, content={"error": str(code)})
