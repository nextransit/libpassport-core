#!/usr/bin/env python3
"""Measure how complementary the CNN and the CRNN heads are.

The fused engine currently arbitrates only between the checksum beam's
winner and its runner-up, so a column the CNN reads with 0.97
confidence can never be overruled -- and the CRNN's opinion on that
column is discarded even when it is right.

This tool measures the complementarity that a better fusion could
exploit:

  * how often the CRNN is right where the CNN is wrong
  * how often the CNN is right where the CRNN is wrong
  * which columns each head rescues

Usage:
    python3 tools/head_complement.py --line 1
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "6_mrz_ocr" / "build" / "mrz_ocr_cnn_tool"
BANDS = Path("/tmp/holdout_export2")

LINE = 1


def run(band, crnn):
    env = dict(os.environ)
    env.update({"MRZ_OCR_LINE_MODELS": "1", "MRZ_OCR_PAR": "0",
                "MRZ_OCR_CRNN": "1" if crnn else "0",
                "MRZ_OCR_FUSE": "0"})
    p = subprocess.run([str(TOOL), str(band)], capture_output=True,
                       text=True, env=env, timeout=120)
    kv = {}
    for ln in p.stdout.splitlines():
        if ":" in ln:
            k, _, v = ln.partition(":")
            kv[k.strip()] = v.strip()
    return kv.get("result.line%d" % (LINE + 1), "")


def load_gt():
    gt = {}
    for f in (ROOT / "tests" / "gt_photo_passport.json",
              ROOT / "tests" / "photo_auto_gt.json"):
        p = Path(f)
        if not p.exists():
            continue
        d = json.loads(p.read_text())
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
    global LINE
    ap = argparse.ArgumentParser()
    ap.add_argument("--line", type=int, default=1)
    ap.add_argument("--bands", default=str(BANDS))
    args = ap.parse_args()
    LINE = args.line

    gt = load_gt()
    bands = sorted(Path(args.bands).glob("*.ppm"))
    n = 0
    cnn_only = 0
    crnn_only = 0
    neither = 0
    crnn_cols = Counter()
    cnn_cols = Counter()

    for b in bands:
        stem = b.stem
        if stem not in gt:
            continue
        truth = gt[stem][LINE]
        c_line = run(b, 0)
        s_line = run(b, 1)
        if len(c_line) != 44 or len(s_line) != 44:
            continue
        n += 1
        for c in range(44):
            tc = truth[c] == c_line[c]
            ts = truth[c] == s_line[c]
            if tc and ts:
                continue
            if tc:
                cnn_only += 1
                cnn_cols[c] += 1
            elif ts:
                crnn_only += 1
                crnn_cols[c] += 1
            else:
                neither += 1

    print("line %d over %d photos" % (LINE, n))
    print("  cells where only the CNN is right  : %d" % cnn_only)
    print("  cells where only the CRNN is right : %d" % crnn_only)
    print("  cells where neither is right       : %d" % neither)
    if cnn_only + crnn_only:
        print("  CRNN wins on %.1f%% of the cells the CNN loses" %
              (100.0 * crnn_only / max(1, cnn_only + crnn_only)))
    print()
    print("columns the CRNN rescues (CRNN right, CNN wrong):")
    for c, k in sorted(crnn_cols.items()):
        print("  col %2d : %d" % (c, k))
    print("columns only the CNN rescues:")
    for c, k in sorted(cnn_cols.items()):
        print("  col %2d : %d" % (c, k))


if __name__ == "__main__":
    main()
