"""Tests for aura.logging_config.

Not part of the knowledge model, but exercised here because it has real
branching behavior (an invalid LOG_LEVEL must be visible, not silently
swallowed) and no Discord dependency, so it's cheap to verify directly.
"""
from __future__ import annotations

import logging

import pytest

from aura.logging_config import configure_logging


@pytest.fixture(autouse=True)
def restore_root_logger():
    """configure_logging mutates global logging state; restore it for other test files."""
    root = logging.getLogger()
    original_level = root.level
    original_handlers = list(root.handlers)
    yield
    root.handlers.clear()
    for handler in original_handlers:
        root.addHandler(handler)
    root.setLevel(original_level)


def test_valid_level_is_applied() -> None:
    configure_logging("DEBUG")
    assert logging.getLogger().level == logging.DEBUG


def test_level_name_is_case_insensitive() -> None:
    configure_logging("warning")
    assert logging.getLogger().level == logging.WARNING


def test_invalid_level_falls_back_to_info() -> None:
    configure_logging("NOT_A_REAL_LEVEL")
    assert logging.getLogger().level == logging.INFO


def test_invalid_level_warning_is_visible_not_silent(capsys: pytest.CaptureFixture[str]) -> None:
    # The fallback handler writes to sys.stdout, which capsys has already
    # swapped in by the time this test body runs, so this is what actually
    # ends up on the container's stdout in production too.
    configure_logging("NOT_A_REAL_LEVEL")
    captured = capsys.readouterr()
    assert "NOT_A_REAL_LEVEL" in captured.out


def test_empty_string_level_falls_back_to_info() -> None:
    configure_logging("")
    assert logging.getLogger().level == logging.INFO


def test_repeated_calls_do_not_duplicate_handlers() -> None:
    configure_logging("INFO")
    configure_logging("INFO")
    assert len(logging.getLogger().handlers) == 1


def test_discord_logger_level_follows_configured_level() -> None:
    configure_logging("ERROR")
    assert logging.getLogger("discord").level == logging.ERROR
