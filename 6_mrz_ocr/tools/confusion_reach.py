#!/usr/bin/env python3
"""Is the truth REACHABLE from the candidate sets the network produced?

WHY
---
`checksum_feasibility.py` showed the check-digit CONSTRAINT carries no
information for most failures: the true body and the predicted body
hash to the SAME digit, so no decoder that only enforces the checksum
can separate them.

This script answers the remaining question: is the true character even
present in the candidate list at each column?  That splits the failures
into two classes:

  * REACHABLE  -- the true glyph is among the top-K candidates.  The
    classifier saw it; the decoder threw it away (a decoder defect,
    fixable without retraining).
  * UNREACHABLE -- the true glyph is not in the top-K.  The classifier
    never produced it; only retraining can help.

Usage:
    python3 tools/confusion_reach.py [K]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DIAG_DIR = Path("/tmp/diags")
GT_JSON = ROOT / "10_ocr_research" / "gt_review_loop.json"

W = (7, 3, 1)
FIELDS = [
    ("passport", 0, 9, 9),
    ("birth", 13, 19, 19),
    ("expiry", 21, 27, 27),
    ("personal", 28, 42, 42),
]


def val(ch: str) -> int:
    if ch.isdigit():
        return ord(ch) - 48
    return ord(ch) - 55


def cd_of(seq: str) -> int:
    return sum(val(c) * W[i % 3] for i, c in enumerate(seq)) % 10


def cands(line: dict, col: int, k: int):
    return [(ch, p) for ch, p in line["chars"][col]["top"][:k] if ch]


def main():
    topk = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    gt = json.loads(GT_JSON.read_text())
    tot_reach = tot_unreach = 0
    per_sample = []
    for p in sorted(DIAG_DIR.glob("*.json")):
        stem = p.stem
        key = None
        for k in gt:
            if Path(k).stem == stem:
                key = k
                break
        if key is None:
            continue
        g1, g2 = gt[key]
        d = json.loads(p.read_text())
        out1, out2 = d.get("line1", ""), d.get("line2", "")
        misses = []
        for li, (pred, truth) in enumerate(((out1, g1), (out2, g2))):
            if pred == truth:
                continue
            line = d["lines"][li]
            for i, ch in enumerate(truth):
                if pred[i] == ch:
                    continue
                cl = cands(line, i, topk)
                if any(c == ch for c, _ in cl):
                    reach = True
                    pr = next((j for j, (c, _) in enumerate(cl) if c == ch), -1)
                    pp = next((pp for c, pp in cl if c == ch), 0.0)
                else:
                    reach = False
                    pr = pp = -1
                misses.append((li, i, pred[i], ch, reach, pr, pp))
        if not misses:
            continue
        nr = sum(1 for m in misses if m[4])
        tot_reach += nr
        tot_unreach += len(misses) - nr
        per_sample.append((stem, misses))
        print(f"=== {stem}  ({nr}/{len(misses)} reachable@top{topk})")
        for li, i, pred, truth, reach, pr, pp in misses:
            mark = "REACH" if reach else "MISS "
            print(f"   L{li+1} c{i:<3} pred={pred!r} gt={truth!r}  "
                  f"{mark} rank={pr} p={pp:.4f}")
    print(f"\nwrong chars: reachable={tot_reach} unreachable={tot_unreach}")
    print(f"samples with >=1 reachable: "
          f"{sum(1 for _, m in per_sample if any(x[4] for x in m))}"
          f"/{len(per_sample)}")


if __name__ == "__main__":
    main()
