"""Centralized logging configuration for Aura.

One format and one level for every logger in the process, including
discord.py's own hierarchy. Called twice during startup -- once with a
bootstrap default before configuration is readable, once with the configured
LOG_LEVEL -- which is why `configure_logging` is idempotent by construction.

Imports nothing from `aura`; it runs before configuration exists.
"""

from __future__ import annotations

import logging
import sys
from typing import Final

_LOG_FORMAT: Final = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

# Third-party loggers whose DEBUG output carries content: discord.py's gateway
# dumps every event (message text included), LiteLLM dumps whole prompts, and
# aiosqlite logs each statement with its parameters. P7a: they never log below
# INFO, whatever LOG_LEVEL says -- logs hold no message or fact text.
CONTENT_BEARING_LOGGERS: Final[tuple[str, ...]] = ("discord", "LiteLLM", "aiosqlite")
_DATE_FORMAT: Final = "%Y-%m-%d %H:%M:%S"

logger = logging.getLogger(__name__)


def configure_logging(level: str = "INFO") -> None:
    """Configure the root logger with a consistent format and the given level.

    Parameters
    ----------
    level
        A `logging` level name, case-insensitive (DEBUG, INFO, WARNING, ERROR,
        CRITICAL). Anything else, including an empty string, falls back to
        INFO.

    Returns
    -------
    None

    Notes
    -----
    Idempotent: safe to call more than once -- once with a bootstrap default
    before configuration is loaded, then again with the user-configured
    LOG_LEVEL -- because existing handlers are cleared rather than
    accumulated.

    Never raises on a bad level name. The fallback is logged at WARNING
    instead, so a typo in LOG_LEVEL is visible rather than silently discarded;
    refusing to start over a log level would be a worse trade than running one
    level noisier than asked.
    """
    named_level = getattr(logging, level.upper(), None) if level else None
    if isinstance(named_level, int):
        resolved_level, used_fallback = named_level, False
    else:
        resolved_level, used_fallback = logging.INFO, True

    root_logger = logging.getLogger()
    root_logger.setLevel(resolved_level)
    root_logger.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
    root_logger.addHandler(handler)

    # discord.py logs through its own "discord" logger hierarchy; keep it at
    # the same level so gateway events are as visible/quiet as everything else
    # -- but never below INFO (see CONTENT_BEARING_LOGGERS).
    for name in CONTENT_BEARING_LOGGERS:
        logging.getLogger(name).setLevel(max(resolved_level, logging.INFO))

    if used_fallback:
        logger.warning(
            "Unknown LOG_LEVEL %r; falling back to INFO. "
            "Valid values: DEBUG, INFO, WARNING, ERROR, CRITICAL.",
            level,
        )
