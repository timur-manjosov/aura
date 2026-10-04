"""The import boundaries of the v2 answer format, asserted statically (by AST).

The design (P4) places the new modules carefully:

* `aura.theme` is constants at the bottom of the graph: it imports nothing from
  aura.
* `aura.answer_contract` reaches the model but no Discord, database or retrieval
  module; `aura.answer_check` the same.
* `aura.answer_card` renders: it imports Discord's types but no LLM client, no
  database, no retrieval, no grounding.
* Only the two answering triggers (/aura-ask and the proactive responder) and
  the operator preview use the format. Extraction, backfill, the digest,
  onboarding, retrieval, billing and the database layer do not import any of
  it, so the format can never change what those paths do.
* The legacy grounding check (`aura.grounding`) and the legacy synthesis
  (`aura.synthesis`) import nothing of the new format: the legacy path does not
  depend on it at all.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parent.parent / "src" / "aura"
_NEW_MODULES = frozenset(
    {"aura.answer_contract", "aura.answer_card", "aura.answer_check", "aura.theme"}
)
_LLM_CLIENTS = frozenset({"litellm", "openai", "anthropic"})


def _imports(path: Path) -> Iterator[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            yield node.module


def _module_name(path: Path) -> str:
    return "aura." + ".".join(path.relative_to(_SRC).with_suffix("").parts)


def _files_importing_new_modules() -> set[str]:
    found = set()
    for path in _SRC.rglob("*.py"):
        name = _module_name(path)
        if name in _NEW_MODULES:
            continue
        if any(module in _NEW_MODULES for module in _imports(path)):
            found.add(name)
    return found


def test_the_theme_imports_nothing_from_aura() -> None:
    assert not [m for m in _imports(_SRC / "theme.py") if m == "aura" or m.startswith("aura.")]


@pytest.mark.parametrize("module", ["answer_contract.py", "answer_check.py"])
def test_the_model_calling_modules_reach_no_discord_database_or_retrieval(module: str) -> None:
    imported = set(_imports(_SRC / module))

    assert not {m for m in imported if m == "discord" or m.startswith("discord.")}
    assert not {
        m for m in imported if m.startswith(("aura.db.repository", "aura.retrieval", "aiosqlite"))
    }


def test_the_renderer_reaches_no_model_database_retrieval_or_grounding() -> None:
    imported = set(_imports(_SRC / "answer_card.py"))

    assert not imported & _LLM_CLIENTS
    assert not {
        m for m in imported if m.startswith(("aura.retrieval", "aura.grounding", "aiosqlite"))
    }
    assert not {m for m in imported if m.startswith("aura.db.") and m != "aura.db.models"}


def test_only_the_answering_triggers_and_the_preview_use_the_new_format() -> None:
    assert _files_importing_new_modules() == {
        "aura.commands.ask",
        "aura.commands.operator",
        "aura.commands.preview_samples",
        "aura.proactive.responder",
    }


@pytest.mark.parametrize(
    "package",
    ["extraction", "backfill", "digest", "onboarding", "retrieval", "billing", "db"],
)
def test_no_other_path_imports_the_new_format(package: str) -> None:
    for path in (_SRC / package).rglob("*.py"):
        assert not set(_imports(path)) & _NEW_MODULES, path


@pytest.mark.parametrize(
    "module", ["grounding.py", "synthesis.py", "embeddings.py", "llm_usage.py"]
)
def test_the_legacy_path_does_not_depend_on_the_new_format(module: str) -> None:
    assert not set(_imports(_SRC / module)) & _NEW_MODULES


def test_the_check_would_catch_a_violation(tmp_path: Path) -> None:
    probe = tmp_path / "probe.py"
    probe.write_text("from aura.answer_card import card_to_embed\n", encoding="utf-8")

    assert set(_imports(probe)) & _NEW_MODULES == {"aura.answer_card"}
