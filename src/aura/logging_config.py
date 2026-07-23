"""Centralized logging configuration for Aura."""
from __future__ import annotations

import logging
import sys

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

logger = logging.getLogger(__name__)


def configure_logging(level: str = "INFO") -> None:
    """Configure the root logger with a consistent format and the given level.

    Safe to call more than once — e.g. once with a bootstrap default before
    configuration is loaded, then again with the user-configured LOG_LEVEL —
    since existing handlers are cleared first rather than accumulated.

    An unrecognized level name falls back to INFO rather than raising, but
    the fallback itself is logged as a warning so a typo in LOG_LEVEL is
    visible instead of silently discarded.
    """
    resolved_level = getattr(logging, level.upper(), None) if level else None
    used_fallback = not isinstance(resolved_level, int)
    if used_fallback:
        resolved_level = logging.INFO

    root_logger = logging.getLogger()
    root_logger.setLevel(resolved_level)
    root_logger.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
    root_logger.addHandler(handler)

    # discord.py logs through its own "discord" logger hierarchy; keep it at
    # the same level so gateway events are as visible/quiet as everything else.
    logging.getLogger("discord").setLevel(resolved_level)

    if used_fallback:
        logger.warning(
            "Unknown LOG_LEVEL %r; falling back to INFO. "
            "Valid values: DEBUG, INFO, WARNING, ERROR, CRITICAL.",
            level,
        )
