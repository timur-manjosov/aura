"""Shared pytest fixtures across the test suite."""

from __future__ import annotations

import os

# The one signal that opts a run into real, paid LLM calls. Deliberately NOT
# LLM_API_KEY: a developer's .env holds one, and keying "is this a real run?"
# off it would silently let the suite spend money. This variable is set by
# nothing but a human who means it.
RUN_REAL_LLM_ENV = "AURA_RUN_REAL_LLM"

# Before anything imports litellm (V-04). Its import runs load_dotenv() unless
# LITELLM_MODE says otherwise, and python-dotenv finds the .env by walking up
# from litellm's own directory -- in a project venv, that is this repository's
# root .env, whatever directory pytest runs from. That copied DISCORD_TOKEN,
# LLM_API_KEY and INTERNAL_API_SECRET into every test process, and the last of
# them started a real listener in tests that build Settings(_env_file=None).
# setdefault, so an explicit LITELLM_MODE still wins; and not for an opt-in
# real-LLM run, which is the one run that is meant to use the developer's .env
# (its model names and key come from there).
if not os.environ.get(RUN_REAL_LLM_ENV):
    os.environ.setdefault("LITELLM_MODE", "PRODUCTION")

from collections.abc import Iterator  # noqa: E402 -- after LITELLM_MODE, on purpose
from typing import Final  # noqa: E402
from unittest.mock import patch  # noqa: E402

import pytest  # noqa: E402
from fastembed import TextEmbedding  # noqa: E402

from aura.config import Settings  # noqa: E402
from aura_web.config import WebSettings  # noqa: E402

EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# What the real-LLM opt-in keeps: the provider credentials and model choices it
# exists to use. Everything else a settings class reads is removed even then.
_REAL_LLM_SETTINGS: Final = frozenset(
    name for name in Settings.model_fields if name.startswith("llm_") or name.endswith("_model")
) - {"embedding_model"}


def settings_environment_names(*, keep_real_llm_settings: bool) -> frozenset[str]:
    """Every environment variable the bot's or the web service's settings read.

    Parameters
    ----------
    keep_real_llm_settings
        Leave out the LLM provider credentials and model names, for a run a
        human opted into real LLM calls.

    Returns
    -------
    frozenset[str]
        Upper-case variable names, derived from the two settings classes, so a
        setting added later is covered without anyone remembering this list.
    """
    bot = {
        name
        for name in Settings.model_fields
        if not (keep_real_llm_settings and name in _REAL_LLM_SETTINGS)
    }
    web_prefix = WebSettings.model_config.get("env_prefix", "")
    web = {f"{web_prefix}{name}" for name in WebSettings.model_fields}
    return frozenset(name.upper() for name in bot | web)


# Marks a test that exercises reading a .env file on purpose; it keeps the
# settings classes' own env_file (see hermetic_settings_environment).
READS_DOTENV_FILE_MARKER: Final = "reads_dotenv_file"


@pytest.fixture(autouse=True)
def hermetic_settings_environment(
    monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> None:
    """Keep the developer's .env and shell out of every test.

    Two routes, both closed per test:

    * the process environment: every variable the settings classes read is
      removed. LITELLM_MODE above closes the one known way a .env reached it;
      this closes every other one -- a variable exported in the shell, a value
      a CI runner injects, a library that loads a .env of its own;
    * the working directory: Settings and WebSettings read ``.env`` and
      ``web/.env`` relative to it on every construction, and production code
      calls load_settings() itself -- so a test run from the repository root
      read the developer's real .env, and one that set VARIANT_MODEL there
      turned three "no model configured" tests red. Their env_file is switched
      off, except for a test marked ``reads_dotenv_file``, which exercises the
      file reading itself (always after chdir to its own tmp_path).

    A test that needs a setting sets it itself (monkeypatch.setenv, or the
    constructor), so what it sees is what it wrote. The opt-in real-LLM run
    keeps the LLM credentials and the .env it takes them from.
    """
    keep = bool(os.environ.get(RUN_REAL_LLM_ENV))
    for name in settings_environment_names(keep_real_llm_settings=keep):
        monkeypatch.delenv(name, raising=False)
    if keep or request.node.get_closest_marker(READS_DOTENV_FILE_MARKER) is not None:
        return
    monkeypatch.setitem(Settings.model_config, "env_file", None)
    monkeypatch.setitem(WebSettings.model_config, "env_file", None)


@pytest.fixture(autouse=True)
def block_real_llm_calls() -> Iterator[None]:
    """Fail loudly if any test reaches a real LLM call without mocking it.

    A genuine .env with LLM_API_KEY and SYNTHESIS_MODEL exists in this repo;
    synthesize_answer reads the key through load_settings(), which reads that
    file from the working directory even with the environment scrubbed. So a test that reaches an un-mocked
    synthesis call would spend real money and hit the network -- exactly the
    grey area CLAUDE.md rules out, arriving through the test suite. This autouse
    guard replaces litellm.acompletion with one that raises; the tests that DO
    exercise synthesis re-patch it locally inside their own `with patch(...)`,
    which takes precedence within that scope.

    The guard steps aside only when a human explicitly opts in by exporting
    AURA_RUN_REAL_LLM, the signal the opt-in real-provider check also keys off.
    """
    if os.environ.get(RUN_REAL_LLM_ENV):
        yield
        return
    message = (
        "a test reached a real litellm.acompletion; mock it (see conftest.block_real_llm_calls)"
    )
    with patch("litellm.acompletion", side_effect=AssertionError(message)):
        yield


@pytest.fixture(scope="session")
def embedding_model() -> TextEmbedding:
    """Load the real embedding model once for the whole test session.

    Real, not mocked: several tests specifically verify real semantic
    behavior (e.g. two paraphrases scoring higher against each other than
    against unrelated text), which a mock can't meaningfully exercise.
    Session-scoped because loading it is the expensive part -- ONNX session
    init has real overhead even from a warm on-disk cache -- while inference
    itself, once loaded, is fast; reloading it per-test or per-module would
    make the suite slow for no verification benefit.
    """
    return TextEmbedding(EMBEDDING_MODEL_NAME)
