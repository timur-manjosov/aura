"""Reading and writing the fact-worthiness corpus as JSON.

No scratch database here, unlike scripts/synthetic_corpus.corpus_store: that
module exists to replay the corpus through the REAL production schema and
Stage 2 retrieval, which this corpus has no equivalent of (see
corpus_model.py's docstring). Calibration reads the JSON directly and scores
each message's content through aura.extraction.fact_worthiness -- there is
nothing here for a database to simulate.
"""
from __future__ import annotations

import json
from pathlib import Path

from extraction_corpus.corpus_model import SyntheticCorpus


class CorpusLoadError(RuntimeError):
    """Raised when a corpus file is unreadable or internally inconsistent."""


def read_corpus(path: Path) -> SyntheticCorpus:
    """Read and validate a corpus JSON file."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CorpusLoadError(f"could not read corpus at {path}: {exc}") from exc

    corpus = SyntheticCorpus.model_validate(raw)
    problems = corpus.check_referential_integrity()
    if problems:
        raise CorpusLoadError(
            f"corpus at {path} has {len(problems)} problem(s):\n  " + "\n  ".join(problems[:20])
        )
    return corpus


def write_corpus(corpus: SyntheticCorpus, path: Path) -> None:
    """Write a corpus to JSON, creating its directory if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(corpus.model_dump_json(indent=2, exclude_none=False), encoding="utf-8")
