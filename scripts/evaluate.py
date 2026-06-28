#!/usr/bin/env python
"""Tiny eval harness. Reads a JSONL file of {question, expected_keywords, must_cite}
and reports how often the answer (a) contains the expected facts and (b) cites a
source. This is what the "85% accuracy" number in the README comes from -- it is
a keyword/citation check on a hand-written question set, not a formal benchmark.

    python scripts/evaluate.py scripts/eval_set.jsonl
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from rag.pipeline import answer_question


def run(path: Path) -> int:
    cases = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not cases:
        print("No cases found.")
        return 1

    hits = 0
    cited = 0
    for case in cases:
        ans = answer_question(case["question"])
        text = ans.text.lower()

        keywords = [k.lower() for k in case.get("expected_keywords", [])]
        ok = all(k in text for k in keywords) if keywords else True
        has_cite = bool(ans.sources)

        hits += ok
        cited += has_cite
        flag = "PASS" if ok else "FAIL"
        print(f"[{flag}] {case['question']}")
        if not ok:
            print(f"        expected: {keywords}")
            print(f"        got: {ans.text[:160]}")

    n = len(cases)
    print(f"\nAccuracy:  {hits}/{n} = {hits / n:.0%}")
    print(f"Cited:     {cited}/{n} = {cited / n:.0%}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(1)
    raise SystemExit(run(Path(sys.argv[1])))
