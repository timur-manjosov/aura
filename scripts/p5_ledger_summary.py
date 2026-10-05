"""Print the P5 ledger by bucket, by label and by model (no content)."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

state = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
entries = state["calls"]
by_bucket: dict[str, list[float]] = defaultdict(lambda: [0, 0.0])
by_label: dict[str, list[float]] = defaultdict(lambda: [0, 0.0])
by_model: dict[str, list[float]] = defaultdict(lambda: [0, 0.0])
for entry in entries:
    for table, key in (
        (by_bucket, entry["bucket"]),
        (by_label, f"{entry['bucket']}/{entry['label']}"),
        (by_model, entry["model"]),
    ):
        table[key][0] += 1
        table[key][1] += entry["usd"]
for name, table in (("bucket", by_bucket), ("label", by_label), ("model", by_model)):
    print(f"--- by {name}")
    for key, (calls, usd) in sorted(table.items(), key=lambda item: -item[1][1]):
        print(f"{key:55s} {int(calls):6d} calls  ${usd:8.4f}")
print(
    f"TOTAL {len(entries)} calls ${sum(e['usd'] for e in entries):.4f}; open reservations: {len(state.get('reservations', {})) if isinstance(state, dict) else '?'}"
)
