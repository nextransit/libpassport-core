#!/usr/bin/env python3
"""Report the classifier's top-k on the cells the offline score misses.

The question: when the argmax is wrong, is the TRUE character inside
the top-k at all?  That decides the fix:

  * true char in top-2/3 -> a better head can win it, and the
    checksum-constrained beam can be given the extra candidates
  * true char outside top-5 -> the information is not in the 16x12
    crop at all; the fix has to be resolution or geometry, not
    training

Also prints the margin, so a near-tie can be told apart from a
confident mistake.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/tmp/glyph_cache.npz")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--line", type=int, required=True)
    ap.add_argument("--file", default="")
    ap.add_argument("--rank-hist", action="store_true")
    args = ap.parse_args()

    d = np.load(args.cache, allow_pickle=True)
    X = d["X"].astype(np.float32)
    meta = [json.loads(s) for s in d["meta"]]

    P = ce.pick(ce.parse_header(args.weights), args.prefix)
    keep = [i for i, m in enumerate(meta) if int(m["line"]) == args.line]
    if args.file:
        keep = [i for i in keep if meta[i]["file"] == args.file]
    logits = ce.forward(X[keep], P)
    prob = ce.softmax(logits)
    order = np.argsort(-prob, axis=1)

    rank_hist = Counter()
    misses = []
    for j, i in enumerate(keep):
        truth = meta[i].get("ch", "")
        if truth not in GI:
            continue
        ti = GI[truth]
        ranked = [GLYPHS[o] for o in order[j]]
        if ranked[0] == truth:
            continue
        rank = ranked.index(truth) if truth in ranked else 99
        rank_hist[min(rank, 9)] += 1
        p_true = float(prob[j][ti])
        p_top = float(prob[j][order[j][0]])
        misses.append((meta[i]["file"], int(meta[i]["col"]), truth,
                       ranked[0], p_top, p_true, rank, ranked[:5]))

    print("misses: %d" % len(misses))
    if args.rank_hist:
        print("true-char rank histogram (9 = outside top-5):")
        for r in sorted(rank_hist):
            print("  rank %2d : %d" % (r, rank_hist[r]))
    print()
    print("%-26s %4s %3s %4s %7s %7s %5s  %s" %
          ("file", "col", "gt", "pred", "p_top", "p_true", "rank", "top5"))
    for f, c, truth, pred, pt, pgt, rank, top5 in misses:
        print("%-26s %4d %3s %4s %7.4f %7.4f %5d  %s" %
              (f, c, truth, pred, pt, pgt, rank, "".join(top5)))


if __name__ == "__main__":
    main()
