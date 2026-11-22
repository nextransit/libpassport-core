#!/usr/bin/env python3
"""Align a prediction with the GT and list the differing columns.

No numpy -- the sandbox stalls on some numpy operations.

Usage:
    python3 tools/align_diff.py <pred-file> [stem ...]
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
    pred_file = sys.argv[1]
    only = sys.argv[2:]
    gt = load_gt()
    for line in Path(pred_file).read_text().splitlines():
        parts = line.split("|")
        if len(parts) != 3:
            continue
        stem, p1, p2 = parts
        if stem not in gt:
            continue
        if only and stem not in only:
            continue
        g1, g2 = gt[stem]
        for tag, p, g in (("L1", p1, g1), ("L2", p2, g2)):
            diffs = [(i, g[i], p[i]) for i in range(44)
                     if (p[i] if i < len(p) else "_") != g[i]]
            if diffs:
                print(f"{stem:22s} {tag} diffs={len(diffs)}: " +
                      " ".join(f"c{i}:{t}->{q}" for i, t, q in diffs))
            else:
                print(f"{stem:22s} {tag} OK")


if __name__ == "__main__":
    main()
