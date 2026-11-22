#!/usr/bin/env python3
"""Score a `stem|l1|l2` dump against the photo GT -- no numpy.

Why this exists
---------------
`tools/band_sweep.py` imports numpy indirectly (through the offline
evaluator) and the sandbox stalls on some numpy operations, so the
fast A/B loop broke.  This scorer only needs the JSON GT and a text
dump, so it stays dependency-free.

Usage:
    python3 tools/score_lines.py /tmp/out_fielddp.txt
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GT_FILES = [
    ROOT / "tests" / "gt_photo_passport.json",
    ROOT / "tests" / "photo_auto_gt.json",
]


def load_gt():
    gt = {}
    for f in GT_FILES:
        if not f.exists():
            continue
        d = json.loads(f.read_text())
        g = d.get("gt", d) if isinstance(d, dict) else d
        for k, v in g.items():
            if not isinstance(v, dict):
                continue
            l1 = v.get("l1") or v.get("line1")
            l2 = v.get("l2") or v.get("line2")
            if l1 and l2 and len(l1) == 44 and len(l2) == 44:
                gt[Path(k).stem] = (l1, l2)
    return gt


def main():
    path = sys.argv[1]
    gt = load_gt()
    n = ex = l1x = l2x = 0
    c1 = c2 = 0
    fails = []
    for line in Path(path).read_text().splitlines():
        parts = line.split("|")
        if len(parts) != 3:
            continue
        stem, p1, p2 = parts
        if stem not in gt:
            continue
        g1, g2 = gt[stem]
        n += 1
        for k in range(44):
            c1 += 1 if (p1[k] if k < len(p1) else "_") == g1[k] else 0
            c2 += 1 if (p2[k] if k < len(p2) else "_") == g2[k] else 0
        a = p1 == g1
        b = p2 == g2
        l1x += a
        l2x += b
        if a and b:
            ex += 1
        else:
            fails.append(stem)
    print(f"n={n} exact={ex}/{n} l1={l1x} l2={l2x} "
          f"acc={100.0*c1/max(1,n*44):.2f}/{100.0*c2/max(1,n*44):.2f}")
    print("fails:", " ".join(fails))


if __name__ == "__main__":
    main()
