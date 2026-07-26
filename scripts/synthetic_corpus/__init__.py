"""Phase 2b-2 tooling: a synthetic Discord-scenario corpus and a pipeline simulator.

This package is *tooling*, not part of the bot. Nothing here is imported by
`aura`, nothing here runs at bot startup, and nothing here may write to the
production database (see `scratch_db`). It exists to produce the evidence base
Phase 2b-3 will pick threshold values from.

The split between the two entry points in `scripts/` matters:

* `generate_synthetic_corpus.py` costs money (it generates language) and runs
  once, gated behind AURA_RUN_REAL_LLM exactly like the model bake-off.
* `simulate_pipeline.py` is free for Stage 1 and Stage 2 -- local embeddings
  only -- and can therefore be re-run at real scale as often as a threshold
  question comes up. Only its optional Stage 3 passes spend anything, and those
  are separately gated and capped in code.
"""
