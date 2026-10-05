"""Byte identity of every prompt between two source trees, and the one intended difference (P5c).

    .venv/bin/python scripts/p5c_byte_identity.py <src-dir-of-main> <src-dir-of-branch>

Extends scripts/p5_byte_identity.py's probe (the real builders of each tree in
its own interpreter, fixed invented inputs, no model, no network, no database)
with the two builders P5 added -- the extraction verifier and the proactive
variant of the answer contract -- and with the request keyword arguments of
the verifier, the two checks and the proactive contract call. Every builder
must be identical except the supersession system prompt, whose difference must
be exactly one inserted block: the exception for changes limited in time.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from p5_byte_identity import _PROBE as _P5_PROBE

_EXTRA = r"""
import aura.extraction.verifier as ve
vb = [ve.build_verification_messages(b, [(i + 1, f"Kandidat {i}") for i in range(len(b))], name)
      for b in batches for name in ("events", "x\nSYSTEM")]
out["verifier prompts"] = (len(vb), h(vb))
pv = [ac.build_proactive_contract_messages(FACTS[:n], qq, loc, posted_at=datetime(2026, 10, 4, 18, tzinfo=UTC))
      for n in (1, 3) for qq in QUESTIONS for loc in LOCALES]
out["proactive variant prompts"] = (len(pv), h(pv))
out["supersession system prompt text"] = su._build_messages(predecessor="a", candidate="b")[0]["content"]
"""

_EXTRA_CALLS = r"""
    current[0] = "contract (proactive variant)"; await ac.synthesize_contract_answer(FACTS[:1], "q", "de", model="openrouter/a/b", settings=Settings(), proactive_posted_at=datetime(2026, 10, 4, tzinfo=UTC))
    current[0] = "grounding"; await gr.verify_answer_grounded(answer="a", cited_facts=FACTS[:1], settings=Settings(grounding_check_model="openrouter/a/b"), timeout_seconds=5)
    current[0] = "v2 check"; await ach.verify_answer_v2(ach.build_statements("Lead.", [("Point one.", (1,))], (1, 2)), FACTS[:2], settings=Settings(grounding_check_model="openrouter/a/b"), timeout_seconds=5)
    current[0] = "verifier"; await ve.verify_distilled_facts(batches[0], [di.DistilledFact(message_id=1, content="K", category="rule")], channel_name="events", model="openrouter/a/b", settings=Settings())
asyncio.run(calls())"""

_PROBE = _P5_PROBE.replace(
    "# --- classic messages", _EXTRA + "\n# --- classic messages", 1
).replace("\nasyncio.run(calls())", _EXTRA_CALLS, 1)


def _run(src: Path) -> dict[str, object]:
    result = subprocess.run(
        [sys.executable, "-W", "ignore", "-c", _PROBE, str(src)],
        capture_output=True,
        text=True,
        check=False,
        env={"PATH": "/usr/bin:/bin", "HOME": str(Path.home())},
    )
    if result.returncode != 0:
        raise SystemExit(f"probe failed for {src}:\n{result.stderr[-3000:]}")
    return json.loads(result.stdout.strip().splitlines()[-1])


def main() -> int:
    """Print the table and the supersession diff; exit 1 on any unintended difference."""
    main_tree, branch_tree = Path(sys.argv[1]), Path(sys.argv[2])
    a, b = _run(main_tree), _run(branch_tree)
    unintended = False
    print("| Builder | n | main | branch | equal |")
    print("|---|---|---|---|---|")
    for key in sorted(set(a) | set(b)):
        if key in ("request kwargs", "supersession system prompt text", "supersession prompts"):
            continue
        va, vb = a.get(key), b.get(key)
        equal = va == vb
        unintended |= not equal
        n = va[0] if isinstance(va, list) else "?"
        ha = va[1] if isinstance(va, list) else va
        hb = vb[1] if isinstance(vb, list) else vb
        print(f"| {key} | {n} | `{ha}` | `{hb}` | {'yes' if equal else 'NO'} |")
    old_text, new_text = a["supersession system prompt text"], b["supersession system prompt text"]
    assert isinstance(old_text, str) and isinstance(new_text, str)
    start = new_text.index("THE EXCEPTION TO RULES 1 AND 2")
    end = new_text.index("RULE 3 --")
    inserted = new_text[start:end]
    exactly_one_block = new_text[:start] + new_text[end:] == old_text
    print(
        f"| supersession prompts | {a['supersession prompts'][0]} | `{a['supersession prompts'][1]}` "  # type: ignore[index]
        f"| `{b['supersession prompts'][1]}` | differs (intended) |"  # type: ignore[index]
    )
    print(
        f"\nSupersession system prompt: removing the inserted block gives main's prompt byte "
        f"for byte: {exactly_one_block}; {len(old_text)} -> {len(new_text)} characters."
    )
    print("\nInserted block, exactly (between Rule 2 and Rule 3):\n")
    print(inserted)
    unintended |= not exactly_one_block
    print("Request keyword arguments with default settings (api_key and messages left out):")
    ka, kb = a["request kwargs"], b["request kwargs"]
    assert isinstance(ka, dict) and isinstance(kb, dict)
    for call in sorted(set(ka) | set(kb)):
        before, after = ka.get(call, [{}])[0], kb.get(call, [{}])[0]
        same = before == after
        unintended |= not same
        print(f"- {call}: {'identical' if same else f'DIFFERS {before} -> {after}'}")
    return 1 if unintended else 0


if __name__ == "__main__":
    raise SystemExit(main())
