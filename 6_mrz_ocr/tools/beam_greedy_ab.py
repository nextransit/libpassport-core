#!/bin/bin/env python3
"""Compare the greedy decode against the beam decode, per sample.

WHY
---
`mrz_beam_search()` starts with `best_score = -1e300`, i.e. the
incumbent is "nothing found yet".  Every leaf the walk reaches has a
DELTA score against the greedy decode, and a delta is always <= 0
(column j >= 1 is a lower-probability candidate).  So the beam accepts
its FIRST checksum-valid leaf even when it is strictly worse than the
greedy line -- and the greedy line is never a candidate.

That is only harmless while the greedy decode is checksum-INVALID (then
any valid reading is an improvement).  When the greedy decode is
already checksum-VALID, the beam can only make the line worse.

This script measures how often that happens: it runs the tool with the
beam on and off, checks the checksum status of the greedy line, and
reports the cases where the beam replaced a valid greedy line with a
worse one.

Usage:
    python3 tools/beam_greedy_ab.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "6_mrz_ocr" / "build" / "mrz_ocr_cnn_tool"
BANDS = Path("/tmp/holdout_export2")
GT_JSON = ROOT / "10_ocr_research" / "gt_review_loop.json"

W = (7, 3, 1)
FIELDS = [("passport", 0, 9, 9), ("birth", 13, 19, 19),
          ("expiry", 21, 27, 27), ("personal", 28, 42, 42)]


def val(ch: str) -> int:
    if ch.isdigit():
        return ord(ch) - 48
    return ord(ch) - 55


def cd_of(seq: str) -> int:
    return sum(val(c) * W[i % 3] for i, c in enumerate(seq)) % 10


def checksum_ok(l2: str) -> bool:
    if len(l2) != 44:
        return False
    for _, s, e, cd in FIELDS:
        if not l2[cd].isdigit():
            return False
        if cd_of(l2[s:e]) != int(l2[cd]):
            return False
    comp = l2[0:10] + l2[13:20] + l2[21:43]
    if not l2[43].isdigit():
        return False
    return cd_of(comp) == int(l2[43])


def run(ppm: Path, env_extra: dict) -> str:
    env = dict(os.environ)
    env.update({
        "MRZ_OCR_LINE_MODELS": "1", "MRZ_OCR_PAR": "0",
        "MRZ_OCR_CRNN": "0", "MRZ_OCR_FUSE": "0",
    })
    env.update(env_extra)
    out = subprocess.run([str(TOOL), str(ppm)], capture_output=True,
                         text=True, env=env).stdout
    for line in out.splitlines():
        if line.startswith("result.line2"):
            return line.split(":", 1)[1].strip()
    return ""


def main():
    gt = json.loads(GT_JSON.read_text())
    gt_by_stem = {}
    for k, v in gt.items():
        if isinstance(v, (list, tuple)) and len(v) == 2:
            gt_by_stem[Path(k).stem] = v

    n = 0
    same = 0
    worse = 0
    better = 0
    regress_list = []
    for ppm in sorted(BANDS.glob("*.ppm")):
        stem = ppm.stem
        if stem not in gt_by_stem:
            continue
        n += 1
        g1, g2 = gt_by_stem[stem]
        beam = run(ppm, {})
        greedy = run(ppm, {"MRZ_OCR_NOBEAM": "1"})
        if beam == greedy:
            same += 1
            continue
        gv = checksum_ok(greedy)
        if beam == g2 and greedy != g2:
            better += 1
            print(f"BETTER {stem}: beam fixed it")
        elif gv and greedy != g2:
            worse += 1
            regress_list.append((stem, greedy, beam, g2))
            print(f"WORSE  {stem}: greedy(valid)={greedy} -> beam={beam}")
            print(f"            gt={g2}")
        else:
            worse += 1
            print(f"DIFF  {stem}: greedy(valid={gv})={greedy} -> beam={beam}")
            print(f"            gt={g2}")
    print(f"\nn={n} same={same} changed={worse + better} "
          f"better={better} not-better={worse}")
    print(f"greedy was already checksum-valid on "
          f"{sum(1 for s in regress_list)} of the changed lines")


if __name__ == "__main__":
    main()
