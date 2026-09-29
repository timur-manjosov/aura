"""The architectural import boundaries CLAUDE.md states in prose, asserted in code.

Three rules were documented but had nothing enforcing them; each was found to
survive a deliberate violation during the academic refactor's adversarial pass,
so each is pinned here.

  1. **The backfill worker reaches an LLM only through the extraction modules.**
     CLAUDE.md's Phase 3b brief requires backfill to REUSE `distill_facts` and
     `stage_distilled_candidates` rather than reimplement the recognition
     chain. A direct `litellm` import in the worker is the first step of
     reimplementing it -- a second call site with its own prompt, its own
     parsing and its own failure modes, none of which the extraction tests
     cover. The worker also has its own daily cap (`backfill_calls`), which
     only bounds spend as long as every call it makes goes through the one
     function that claims a slot.

  2. **The proactive gate reaches no LLM and writes no fact.** Already covered
     in depth by test_proactive_isolation.py; restated here only for the
     `litellm` half, so the two worker-side rules read together.

  3. **The web backend imports nothing from `src/aura` and opens no database.**
     The bot is the only writer of its SQLite file (see web/README.md, "Where
     subscription state lives, and why"); the web service reaches it only
     through the internal billing API. An import of `aura.*`, of `aiosqlite` or
     of `sqlite3` in `aura_web` is that separation being given up, and it would
     otherwise be noticed only in review.

Static, by AST, deliberately: an import that is never in scope cannot be called
by accident later, and the check does not need either service to be runnable.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent

# Every module an LLM could be reached through directly. `litellm` is the only
# client this project uses; the other two are listed so that swapping clients
# does not silently reopen the hole.
_LLM_CLIENT_MODULES = frozenset({"litellm", "openai", "anthropic"})

# What "opens a database" means for the web backend.
_DATABASE_MODULES = frozenset({"aiosqlite", "sqlite3"})


def _imported_modules(path: Path) -> Iterator[str]:
    """Yield every module name imported by one file, dotted and in full.

    Parameters
    ----------
    path
        The Python file to read.

    Yields
    ------
    str
        One module name per import. `import a.b` yields ``a.b``;
        `from a.b import c` yields ``a.b``. Relative imports yield the module
        they name, with the leading dots dropped.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            yield node.module


def _top_level(module: str) -> str:
    return module.split(".")[0]


def _python_files(directory: Path) -> list[Path]:
    return sorted(p for p in directory.rglob("*.py") if "__pycache__" not in p.parts)


class TestBackfillReachesTheModelOnlyThroughExtraction:
    """Rule 1: no direct LLM client import in the backfill package."""

    @pytest.mark.parametrize(
        "relative_path",
        ["backfill/worker.py", "backfill/history.py", "backfill/gateway.py"],
    )
    def test_no_llm_client_is_imported(self, relative_path: str) -> None:
        imported = {
            _top_level(module)
            for module in _imported_modules(_REPO_ROOT / "src" / "aura" / relative_path)
        }
        offending = imported & _LLM_CLIENT_MODULES
        assert not offending, (
            f"aura/{relative_path} imports {sorted(offending)} directly; backfill must "
            "reach the model only through aura.extraction.distiller, which is what "
            "claims a slot from the backfill daily cap"
        )

    def test_the_worker_does_import_the_sanctioned_extraction_entry_points(self) -> None:
        # Asserted positively so the rule above cannot be satisfied by a worker
        # that has stopped distilling altogether.
        source = (_REPO_ROOT / "src/aura/backfill/worker.py").read_text(encoding="utf-8")
        assert "from aura.extraction.distiller import distill_facts" in source
        assert "stage_distilled_candidates" in source


class TestProactiveGateReachesNoModel:
    """Rule 2: the free gate stages import no LLM client."""

    @pytest.mark.parametrize(
        "relative_path",
        ["proactive/gate.py", "proactive/question_detector.py", "proactive/grace.py"],
    )
    def test_no_llm_client_is_imported(self, relative_path: str) -> None:
        imported = {
            _top_level(module)
            for module in _imported_modules(_REPO_ROOT / "src" / "aura" / relative_path)
        }
        assert not imported & _LLM_CLIENT_MODULES


class TestWebBackendIsSeparateFromTheBot:
    """Rule 3: aura_web imports no bot module and no database driver."""

    def test_no_module_imports_anything_from_src_aura(self) -> None:
        offenders = {
            str(path.relative_to(_REPO_ROOT)): sorted(
                module for module in _imported_modules(path) if _top_level(module) == "aura"
            )
            for path in _python_files(_REPO_ROOT / "web/backend/aura_web")
        }
        offenders = {path: modules for path, modules in offenders.items() if modules}
        assert not offenders, (
            f"the web backend imports bot modules: {offenders}. It is a separate "
            "service with its own stack; subscription state reaches the bot through "
            "the internal billing API, never through a shared import"
        )

    def test_no_module_opens_a_database(self) -> None:
        offenders = {
            str(path.relative_to(_REPO_ROOT)): sorted(
                module
                for module in _imported_modules(path)
                if _top_level(module) in _DATABASE_MODULES
            )
            for path in _python_files(_REPO_ROOT / "web/backend/aura_web")
        }
        offenders = {path: modules for path, modules in offenders.items() if modules}
        assert not offenders, (
            f"the web backend imports a database driver: {offenders}. The bot is the "
            "only writer of its SQLite file (web/README.md)"
        )

    def test_the_boundary_check_actually_sees_this_tree(self) -> None:
        # A guard against the two tests above passing because they globbed an
        # empty directory -- the failure mode that makes a structural test
        # worthless without anyone noticing.
        files = _python_files(_REPO_ROOT / "web/backend/aura_web")
        assert len(files) >= 15
        assert any(path.name == "app.py" for path in files)
