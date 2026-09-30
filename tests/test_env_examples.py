"""Both .env.example files list exactly the settings their service reads.

An operator configures Aura from these two files and nothing else. A setting
missing from one is a setting nobody knows exists; a line for a setting that no
longer exists configures nothing, silently. The pair has drifted apart from
the code twice before, so the comparison is a test rather than a review step.

A variable counts as listed whether its line is active or commented out
(`# NAME=value`): a commented default is exactly how an optional setting is
documented in both files.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Final

from aura.config import Settings
from aura_web.config import WebSettings

REPO_ROOT: Final = Path(__file__).resolve().parent.parent
# An assignment at the start of a line, optionally commented out. Names carry
# an underscore, which keeps prose such as "F1=0.65" in a comment out of it.
ASSIGNMENT: Final = re.compile(r"^#?\s*([A-Z][A-Z0-9]*_[A-Z0-9_]+)=", re.MULTILINE)

# Read by web/docker-compose.yml to publish the frontend, not by WebSettings.
COMPOSE_ONLY_WEB_VARIABLES: Final = frozenset({"AURA_WEB_FRONTEND_PORT"})


def listed(path: Path) -> set[str]:
    return set(ASSIGNMENT.findall(path.read_text(encoding="utf-8")))


class TestTheBotsEnvExample:
    def test_it_lists_every_setting_and_nothing_else(self) -> None:
        expected = {name.upper() for name in Settings.model_fields}

        assert listed(REPO_ROOT / ".env.example") == expected


class TestTheWebEnvExample:
    def test_it_lists_every_setting_and_nothing_else(self) -> None:
        expected = {f"AURA_WEB_{name.upper()}" for name in WebSettings.model_fields}

        assert listed(REPO_ROOT / "web" / ".env.example") == expected | COMPOSE_ONLY_WEB_VARIABLES

    def test_the_portal_configuration_is_documented(self) -> None:
        assert "AURA_WEB_STRIPE_PORTAL_CONFIGURATION_ID" in listed(
            REPO_ROOT / "web" / ".env.example"
        )
