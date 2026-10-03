"""Hybrid fact retrieval for /aura-ask: embedding similarity plus the question's own words.

* `aura.retrieval.lexical` -- the pure word-matching scorer.
* `aura.retrieval.stopwords` -- the per-locale stopword data files.
* `aura.retrieval.index_cache` -- the bounded per-guild index memo.
* `aura.retrieval.hybrid` -- the gate and the ranking /aura-ask applies.

Used by /aura-ask only. Proactive relief and extraction dedup retrieve through
aura.embeddings with their own calibrated thresholds, and
tests/test_structural_boundaries.py keeps them from importing this package.
"""
