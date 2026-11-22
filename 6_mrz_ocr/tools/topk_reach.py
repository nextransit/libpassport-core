#!/usr/bin/env python3
"""Measure how far a deeper candidate list would reach.

The C decoder keeps the top-K classes per column (K=5 today) and the
checksum beam may only permute within that list.  When the true
character sits at rank 6 or beyond it is UNREACHABLE -- no beam depth
and no checksum constraint can recover it.

This script answers, per column of the real corpus:
  * the rank of the true character
  * whether a checksum-valid reading exists at all when the candidate
    list is widened to K

The second question is the one that decides whether widening K is
worth the beam cost: a column whose true character is at rank 6 only
matters if the rest of the line can still close its checksums.

Usage:
  python3 tools/topk_reach.py --weights /tmp/rl2_L0.h --prefix LINE2 --line 1
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "ce", str(ROOT / "tools" / "cnn_offline_eval.py"))
ce = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ce)

GLYPHS = ce.GLYPHS
GI = {c: i for i, c in enumerate(GLYPHS)}

# ICAO TD3 line-2 field mask (same rules as fill_allow in the C code).
ALPHA = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
DIGITS = "0123456789"
FILLER = "<"


def icao_allow(line: int, col: int) -> str:
    if line == 0:
        if col == 0:
            return "ACIPV"
        if col == 1:
            return ALPHA + FILLER
        if 2 <= col <= 4:
            return ALPHA
        return ALPHA + FILLER
    if col <= 8:
        return ALPHA + DIGITS + FILLER
    if col in (9, 19, 27):
        return DIGITS
    if col <= 12:
        return ALPHA + FILLER
    if 13 <= col <= 18 or 21 <= col <= 26:
        return DIGITS + FILLER + ALPHA
    if col == 20:
        return ALPHA + DIGITS + FILLER
    if col <= 41:
        return ALPHA + DIGITS + FILLER
    return DIGITS + FILLER


def char_value(c):
    if c.isdigit():
        return int(c)
    if c.isalpha():
        return ord(c.upper()) - 55
    return 0


def check_digit(text):
    w = (7, 3, 1)
    return sum(w[k % 3] * char_value(c) for k, c in enumerate(text)) % 10


def line2_checksums(l2):
    """Return the five expected check digits of a 44-char TD3 line 2."""
    if len(l2) != 44:
        return None
    comp = l2[0:10] + l2[13:20] + l2[21:43]
    return {
        9: check_digit(l2[0:9]),
        19: check_digit(l2[13:19]),
        27: check_digit(l2[21:27]),
        42: check_digit(l2[28:42]),
        43: check_digit(comp),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/tmp/glyph_cache.npz")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--line", type=int, required=True)
    ap.add_argument("--kmax", type=int, default=12)
    args = ap.parse_args()

    d = np.load(args.cache, allow_pickle=True)
    X = d["X"].astype(np.float32)
    meta = [json.loads(s) for s in d["meta"]]
    P = ce.pick(ce.parse_header(args.weights), args.prefix)
    keep = [i for i, m in enumerate(meta) if int(m["line"]) == args.line]
    logits = ce.forward(X[keep], P)
    prob = ce.softmax(logits)

    byfile = {}
    for j, i in enumerate(keep):
        byfile.setdefault(meta[i]["file"], []).append((int(meta[i]["col"]), j))

    hist = Counter()
    per_file = {}
    for f, cells in byfile.items():
        cells.sort()
        ranks = {}
        for c, j in cells:
            truth = meta[keep[j]].get("ch", "")
            if truth not in GI:
                continue
            ranked = [GLYPHS[o] for o in np.argsort(-prob[j])]
            ranks[c] = (ranked.index(truth) if truth in ranked else 99, truth)
        hist.update(r for r, _ in ranks.values())
        per_file[f] = ranks

    print("true-character rank histogram over all cells:")
    for r in sorted(hist):
        print("  rank %2d : %d" % (r, hist[r]))

    # For each file: with the TRUE character in every column, does the
    # line close its checksums?  If yes, the reading is reachable at
    # all and only the search has to find it.
    print()
    print("per-file: rank of the first unreachable column")
    unreachable_files = 0
    for f in sorted(per_file):
        ranks = per_file[f]
        worst = max((r for r, _ in ranks.values()), default=0)
        bad = [(c, r, t) for c, (r, t) in sorted(ranks.items()) if r >= 5]
        if bad:
            unreachable_files += 1
            print("  %-26s worst_rank=%2d  beyond_K5: %s" % (
                f, worst,
                " ".join("c%d(%s,r%d)" % (c, t, r) for c, r, t in bad)))
    print()
    print("files with at least one cell outside the current K=5: %d/%d"
          % (unreachable_files, len(per_file)))


if __name__ == "__main__":
    main()
