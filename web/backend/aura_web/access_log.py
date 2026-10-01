"""Keep query strings out of uvicorn's access log.

The OAuth callback arrives as ``/api/auth/callback?code=...&state=...``, and
uvicorn's access log writes the full path with its query. Caddy's own access
log already deletes both parameters (web/deploy/Caddyfile.aura); the backend's
log wrote them anyway, which production showed on the first real login. The
code is single-use and already redeemed by the time the line is written, but a
credential does not belong in a log at all, and no route here needs its query
string recorded to be diagnosed.

Only the path stays: it is what an operator needs, and uvicorn has already
percent-encoded it, so it cannot start a forged log line.

May import only the standard library.
"""

from __future__ import annotations

import logging
from typing import Final

UVICORN_ACCESS_LOGGER: Final[str] = "uvicorn.access"


class QueryStringRedactingFilter(logging.Filter):
    """Cut the query string off every path argument of an access-log record.

    Notes
    -----
    Rewrites any string argument that starts with ``/`` rather than relying on
    its position in uvicorn's argument tuple, so a reordering in a later
    uvicorn release cannot quietly turn the redaction off. The record is always
    kept: dropping access lines would hide what this filter exists to keep
    safe to read.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        """Redact the record's path arguments in place and keep the record.

        Parameters
        ----------
        record
            An access-log record.

        Returns
        -------
        bool
            Always True.
        """
        if isinstance(record.args, tuple):
            record.args = tuple(
                argument.split("?", 1)[0]
                if isinstance(argument, str) and argument.startswith("/")
                else argument
                for argument in record.args
            )
        return True


def install_access_log_redaction() -> None:
    """Attach the redacting filter to uvicorn's access logger, once.

    Notes
    -----
    Idempotent, because the server configuration that calls it can be built
    more than once in one process (the test suite does).
    """
    access_logger = logging.getLogger(UVICORN_ACCESS_LOGGER)
    if not any(
        isinstance(existing, QueryStringRedactingFilter) for existing in access_logger.filters
    ):
        access_logger.addFilter(QueryStringRedactingFilter())
