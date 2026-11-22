#!/usr/bin/env python3
"""Per-image, per-line error budget for a candidate CNN weight set.

Why this exists
---------------
Exact-match scoring says only "how many whole lines are perfect".  That
hides the difference between an image that is wrong in ONE cell (cheap:
fix one class, gain one exact line) and an image that is wrong in
TWELVE cells (expensive: the error is systematic, not a class issue).

For a fixed-weight classifier the per-cell errors are exactly
independent, so the "error budget" is the right prioritisation signal:
an image with 1 error is one class away from an exact line, an image
with 10 errors is not worth chasing with a classifier change.

Usage:
    python3 tools/error_budget.py --weights /tmp/rl1_L0.h /tmp/rl2_L0.h \
        --prefix LINE1 LINE2
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np

import cnn_offline_eval as ce


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", nargs="+", required=True)
    ap.add_argument("--prefix", nargs="+", required=True)
    ap.add_argument("--cache", default="/tmp/glyph_cache.npz")
    ap.add_argument("--top", type=int, default=40)
    args = ap.parse_args()

    d = np.load(args.cache, allow_pickle=True)
    meta = [json.loads(s) for s in d["meta"]]
    X = d["X"].astype(np.float32)
    G = ce.GLYPHS

    preds = []
    for w, p in zip(args.weights, args.prefix):
        P = ce.pick(ce.parse_header(w), p)
        pr = ce.softmax(ce.forward(X, P))
        preds.append(pr)

    # group the cache by (file, line)
    groups = defaultdict(list)
    for i, m in enumerate(meta):
        groups[(m["file"], int(m["line"]))].append(i)

    per_line = {0: [], 1: []}
    for (f, ln), idxs in sorted(groups.items()):
        errs = []
        for i in idxs:
            truth = meta[i]["ch"]
            if not truth:
                continue
            pred = G[preds[ln][i].argmax()]
            if pred != truth:
                col = int(meta[i]["col"])
                p_true = float(preds[ln][i][G.index(truth)]) if truth in G else 0.0
                errs.append((col, truth, pred, p_true))
        per_line[ln].append((f, len(idxs), errs))

    for ln in (0, 1):
        rows = per_line[ln]
        nexact = sum(1 for _, _, e in rows if not e)
        print(f"\n=== line{ln + 1}: {nexact}/{len(rows)} exact ===")
        rows.sort(key=lambda r: (len(r[2]), r[0]))
        bad = [r for r in rows if r[2]]
        for f, n, errs in bad[:args.top]:
            print(f"  {f:28s} errs={len(errs):2d}  " +
                  "  ".join(f"c{c}:{t}->{p}(p={pt:.3f})"
                            for c, t, p, pt in errs[:10]))
        hist = defaultdict(int)
        for _, _, e in rows:
            hist[len(e)] += 1
        print("  error-count histogram:",
              {k: hist[k] for k in sorted(hist)})


if __name__ == "__main__":
    main()
