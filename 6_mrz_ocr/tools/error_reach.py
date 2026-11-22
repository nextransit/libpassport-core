#!/usr/bin/env python3
"""Where is the truth for the cells that break exact match?

For every misclassified cell this reports the RANK of the true class in
the classifier's probability order, and whether the true class is even
in the top-K candidate set the C decoder works with.

This separates two very different failure modes:

  * truth at rank 2..K  -> the information is in the pixels AND in the
    candidate set; a better decoder (beam / checksum) can recover it.
  * truth beyond rank K -> the classifier is confidently wrong; no
    decoder can recover it without new weights.

Usage:
    python3 tools/error_reach.py --weights /tmp/rl1_L0.h /tmp/rl2_L0.h \
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
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    d = np.load(args.cache, allow_pickle=True)
    meta = [json.loads(s) for s in d["meta"]]
    X = d["X"].astype(np.float32)
    G = ce.GLYPHS

    preds = []
    for w, p in zip(args.weights, args.prefix):
        P = ce.pick(ce.parse_header(w), p)
        preds.append(ce.softmax(ce.forward(X, P)))

    groups = defaultdict(list)
    for i, m in enumerate(meta):
        groups[(m["file"], int(m["line"]))].append(i)

    reach = {0: [], 1: []}
    for (f, ln), idxs in groups.items():
        for i in idxs:
            truth = meta[i]["ch"]
            if not truth or truth not in G:
                continue
            pr = preds[ln][i]
            order = np.argsort(-pr)
            ranked = [G[o] for o in order]
            if ranked[0] == truth:
                continue
            r = ranked.index(truth)
            p_true = float(pr[G.index(truth)])
            p_top = float(pr[order[0]])
            reach[ln].append((f, int(meta[i]["col"]), truth, ranked[0],
                              r, p_true, p_top, ranked[:6]))

    for ln in (0, 1):
        rows = sorted(reach[ln], key=lambda x: (x[4], x[0], x[1]))
        inK = sum(1 for r in rows if r[4] < args.k)
        print(f"\n=== line{ln + 1}: {len(rows)} error cells, "
              f"truth within top{args.k}: {inK}/{len(rows)} ===")
        for f, c, t, p, r, pt, ptop, top6 in rows:
            mark = "OK " if r < args.k else "OUT"
            print(f"  {mark} {f:28s} c{c:02d} gt={t} pred={p} "
                  f"rank={r} p_true={pt:.4f} p_top={ptop:.4f} top6={''.join(top6)}")
        hist = defaultdict(int)
        for r in rows:
            hist[min(r[4], 20)] += 1
        print("  truth-rank histogram:",
              {k: hist[k] for k in sorted(hist)})


if __name__ == "__main__":
    main()
