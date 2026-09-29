"""Aura's test suite.

A package rather than a bare directory so the module names stay unambiguous
when pytest imports two files with the same basename from different trees
(`tests/conftest.py` and `web/backend/tests/conftest.py`).

Two rules hold throughout, both from CLAUDE.md:

* No test requires a live Discord connection. Command callbacks, checks and
  error handlers are invoked directly against mocked interactions.
* No test makes a real, paid LLM call. `conftest.block_real_llm_calls` replaces
  `litellm.acompletion` with one that raises, for every test that has not
  patched it itself; the handful of opt-in real-provider checks are skipped
  unless a human exports `AURA_RUN_REAL_LLM`.
"""
