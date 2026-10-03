"""Offline evaluation of /aura-ask's fact retrieval through the production code path. Free.

* `cases` -- labelled question sets and the metrics computed over them.
* `production_path` -- runs questions through aura.embeddings.find_similar_facts
  and aura.retrieval.hybrid exactly as /aura-ask does, and through the
  similarity-only selection /aura-ask used before, for comparison.
* `scale_corpus` -- a deterministic, invented server of any size up to a few
  thousand facts in nine languages, with labelled questions.

Everything here is invented or read from a database copy the caller supplies;
nothing is written anywhere but where the caller asks. Imported by
scripts/evaluate_ask_retrieval.py and by the hermetic tests.
"""
