"""Tooling for Phase 3a-1's fact-worthiness synthetic corpus and its calibration.

Sibling of scripts/synthetic_corpus, reusing its budget/pricing/LLM-call/
safety infrastructure (all fully generic) rather than duplicating it. Only the
corpus shape, the generation prompts, and the scenario grid are specific to
this phase -- see corpus_model.py for why this corpus is deliberately a
simpler shape than synthetic_corpus's own.
"""
