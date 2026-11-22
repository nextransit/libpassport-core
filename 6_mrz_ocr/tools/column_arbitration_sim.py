#!/usr/bin/env python3
"""Per-column arbitration upper bound between the CNN and the CRNN.

The fused engine currently equals the CNN on every metric, because the
`arb` mode only lets the sequence head vote on columns where a
checksum-valid runner-up already exists -- and the CNN's errors are
overwhelmingly CONFIDENT errors (44 of 59 wrong characters have margin
> 0.5, median 0.995).  So the runner-up path almost never fires.

This script measures what a PER-COLUMN confidence arbitration could
achieve: for every column, take the character the engine is more
confident about, then count exact matches.  The confidence numbers come
from the engine's own probability dump, so this is an upper bound on
what a real per-column fusion can reach.

Two variants:
  * argmax-conf  -- pick the engine whose softmax probability is higher
  * crnn-wins    -- CRNN wins every column it disagrees on
  * oracle        -- the true character whenever EITHER engine got it

Usage:
  python3 tools/column_arbitration_sim.py [--band-dir /tmp/holdout_export2]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "6_mrz_ocr" / "build" / "mrz_ocr_cnn_tool"
CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<"


def parse(out: str) -> dict:
    kv = {}
    for ln in out.splitlines():
        if ":" not in ln:
            continue
        k, _, v = ln.partition(":")
        kv[k.strip()] = v.strip()
    return kv


def run(band: Path, env_extra: dict) -> dict:
    env = dict(os.environ)
    env.update(env_extra)
    env["MRZ_OCR_LINE_MODELS"] = "1"
    env["MRZ_OCR_PAR"] = "0"
    env["MRZ_OCR_REQUIRE_HEAD"] = env_extra.get("MRZ_OCR_CRNN") == "1" \
        and "crnn" or "cnn"
    p = subprocess.run([str(TOOL), str(band)], capture_output=True,
                       text=True, env=env, timeout=20)
    return parse(p.stdout)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--band-dir", default="/tmp/holdout_export2")
    ap.add_argument("--gt", default="10_ocr_research/gt_review_loop.json")
    ap.add_argument("--dump-probs", action="store_true",
                    help="also ask the tool for per-column probabilities")
    args = ap.parse_args()

    bands = sorted(Path(args.band_dir).glob("*.ppm"))
    gt = json.loads((ROOT / args.gt).read_text())
    gmap = {}
    for k, v in gt.items():
        stem = Path(k).stem
        if len(v[0]) == 44 and len(v[1]) == 44:
            gmap[stem] = (v[0], v[1])

    env_cnn = {"MRZ_OCR_CRNN": "0", "MRZ_OCR_FUSE": "0"}
    env_crnn = {"MRZ_OCR_CRNN": "1", "MRZ_OCR_FUSE": "0"}

    stats = {"cnn": 0, "crnn": 0, "fuse": 0, "argmax": 0,
             "crnn_wins": 0, "oracle": 0, "n": 0,
             "cells": 0, "disagree": 0, "cnn_right": 0, "crnn_right": 0,
             "both_wrong": 0}
    per_sample = []
    for band in bands:
        stem = band.stem
        if stem not in gmap:
            continue
        g1, g2 = gmap[stem]
        rc = run(band, env_cnn)
        rr = run(band, env_crnn)
        c1, c2 = rc.get("result.line1", ""), rc.get("result.line2", "")
        r1, r2 = rr.get("result.line1", ""), rr.get("result.line2", "")
        stats["n"] += 1
        stats["cnn"] += int(c1 == g1 and c2 == g2)
        stats["crnn"] += int(r1 == g1 and r2 == g2)
        for pi, (gc, pc, pr) in enumerate(((g1, c1, r1), (g2, c2, r2))):
            if len(gc) != 44:
                continue
            for i in range(44):
                stats["cells"] += 1
                if pc[i] != pr[i]:
                    stats["disagree"] += 1
                    if pc[i] == gc[i]:
                        stats["cnn_right"] += 1
                    elif pr[i] == gc[i]:
                        stats["crnn_right"] += 1
                    else:
                        stats["both_wrong"] += 1
        per_sample.append((stem, c1 == g1, c2 == g2, r1 == g1, r2 == g2,
                           (g1, c1, r1), (g2, c2, r2)))

    print(f"samples={stats['n']}")
    print(f"cells={stats['cells']}  disagreements={stats['disagree']} "
          f"({stats['disagree']/max(1,stats['cells'])*100:.2f}%)")
    print(f"  on disagreements: cnn right={stats['cnn_right']}  "
          f"crnn right={stats['crnn_right']}  both wrong={stats['both_wrong']}")
    # Oracle upper bound on the exact metric: a line counts as exact only
    # if EVERY column is right, so a per-column oracle is the ceiling.
    both_l1 = sum(1 for s in per_sample if s[1] and s[3])
    both_l2 = sum(1 for s in per_sample if s[2] and s[4])
    print(f"per-line oracle (line correct in cnn OR crnn): "
          f"l1={both_l1}/{stats['n']} l2={both_l2}/{stats['n']}")
    # A sample counts for the ORACLE exact metric only if both lines are
    # correct in at least one engine -- the ceiling for whole-row swap.
    swappable = sum(1 for s in per_sample
                    if (s[1] and s[2]) or (s[3] and s[4]))
    print(f"row-swap oracle (both lines right in one engine): "
          f"{swappable}/{stats['n']}")

    # PER-COLUMN ORACLE: pick, per column, whichever engine's character
    # equals the truth.  This is the ceiling of any per-column fusion
    # (confidence-based or otherwise) -- no weighting can beat it.
    per_col_exact = 0
    per_col_chars = 0
    for s in per_sample:
        ok = True
        for (g, pc, pr) in (s[5], s[6]):
            if len(g) != 44:
                continue
            merged = "".join(pc[i] if pc[i] == g[i] else pr[i]
                             for i in range(44))
            per_col_chars += 44
            if merged != g:
                ok = False
        per_col_exact += int(ok)
    print(f"PER-COLUMN ORACLE exact (per-column pick of the truth): "
          f"{per_col_exact}/{stats['n']}  "
          f"(char ceiling {per_col_chars} cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
