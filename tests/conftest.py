"""Shared pytest fixtures across the test suite."""
from __future__ import annotations

import os
from collections.abc import Iterator
from unittest.mock import patch

import pytest
from fastembed import TextEmbedding

EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


# The one signal that opts a run into real, paid LLM calls. Deliberately NOT
# LLM_API_KEY: `import litellm` calls load_dotenv(), which pulls this repo's
# real .env (LLM_API_KEY, SYNTHESIS_MODEL) into os.environ -- so keying "is this
# a real run?" off LLM_API_KEY would be always-true here and would silently let
# the suite spend money. This variable is set by nothing but a human who means
# it.
RUN_REAL_LLM_ENV = "AURA_RUN_REAL_LLM"


@pytest.fixture(autouse=True)
def block_real_llm_calls() -> Iterator[None]:
    """Fail loudly if any test reaches a real LLM call without mocking it.

    A genuine .env with LLM_API_KEY and SYNTHESIS_MODEL exists in this repo and
    is loaded into os.environ the moment litellm is imported; synthesize_answer
    reads the key through load_settings(). So a test that reaches an un-mocked
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
    message = "a test reached a real litellm.acompletion; mock it (see conftest.block_real_llm_calls)"
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
