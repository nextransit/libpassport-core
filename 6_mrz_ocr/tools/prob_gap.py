#!/usr/bin/env python3
"""Is the failure a SEARCH problem or a CLASSIFIER problem?

For every failing line this tool computes, from the exported
product-path slices:

  lp_greedy  -- the summed log-probability of the greedy (top-1) reading
  lp_gt      -- the summed log-probability of the ground-truth reading
                (using each column's probability for the GT glyph; columns
                where the GT glyph is not in the top-K contribute the floor)
  n_missing  -- how many columns do not even propose the GT glyph

The verdict:

  * lp_gt >> lp_greedy  -> the classifier RANKS the truth above the
    greedy reading, so the fault is the SEARCH: the beam did not visit
    the GT combination (pruning, candidate-column limit, depth).
  * lp_gt << lp_greedy  -> the classifier is CONFIDENT and WRONG; no
    search can recover it, the weights must be retrained.
  * n_missing > 0       -> the glyph is not even proposed; TOPK must grow
    or the classifier must be retrained.

Usage:
    python3 tools/prob_gap.py --npz /tmp/holdout_photo2.npz
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cnn_offline_eval as off  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="/tmp/holdout_photo2.npz")
    ap.add_argument("--w1", default="src/cnn_weights_line1.h")
    ap.add_argument("--w2", default="src/cnn_weights_line2.h")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--only", default="")
    ap.add_argument("--all", action="store_true",
                    help="also report the passing lines")
    args = ap.parse_args()

    d = np.load(args.npz, allow_pickle=True)
    X = d["X"].astype(np.float32)
    Y = d["Y"]
    meta = [json.loads(s) for s in d["meta"]]

    w1 = off.parse_header(args.w1)
    w2 = off.parse_header(args.w2)
    p1 = off.pick(w1, off.infer_prefix(w1))
    p2 = off.pick(w2, off.infer_prefix(w2))

    groups = defaultdict(list)
    for i, m in enumerate(meta):
        groups[(m["file"], int(m["line"]))].append(i)

    n_search = n_class = 0
    for (fname, line), idxs in sorted(groups.items()):
        if args.only and args.only not in fname:
            continue
        idxs = sorted(idxs, key=lambda i: meta[i]["col"])
        P = p1 if line == 0 else p2
        sm = off.softmax(off.forward(X[idxs], P))
        gt = "".join(off.GLYPHS[Y[i]] for i in idxs)
        greedy = "".join(off.GLYPHS[int(np.argmax(sm[j]))]
                         for j in range(len(idxs)))
        diff = [c for c in range(44) if greedy[c] != gt[c]]
        if not diff and not args.all:
            continue

        lp_g = lp_gt = 0.0
        missing = []
        for j in range(44):
            order = np.argsort(-sm[j])
            cls = int(order[0])
            pg = sm[j][cls]
            gtcls = int(Y[idxs[j]])
            # the probability the model gives the GT glyph
            pos = np.where(order == gtcls)[0]
            if len(pos):
                pt = sm[j][gtcls]
                rank = int(pos[0]) + 1
            else:
                pt = 1e-6
                rank = 99
                missing.append((c, gt[c], rank))
            lp_g += np.log(max(pg, 1e-9))
            lp_gt += np.log(max(pt, 1e-9))
        verdict = ("SEARCH" if lp_gt > lp_g else "CLASSIFIER")
        if missing:
            verdict += "+MISSING"
        if diff:
            n_search += verdict.startswith("SEARCH")
            n_class += verdict.startswith("CLASSIFIER")
        print(f"{fname:32s} L{line} ndiff={len(diff):2d} "
              f"lp_greedy={lp_g:8.2f} lp_gt={lp_gt:8.2f} "
              f"delta={lp_gt-lp_g:+7.2f}  -> {verdict}")
        if diff:
            bad = [(c, gt[c], greedy[c]) for c in range(44)
                   if greedy[c] != gt[c]]
            print(f"    cols: " + " ".join(
                f"{c}:{g}>{p}" for c, g, p in bad))
        if missing:
            print(f"    GT not in top-{args.topk}: " + " ".join(
                f"{c}({g})" for c, g, _ in missing))
    print(f"\nSUMMARY: search-limited={n_search}  classifier-limited={n_class}")


if __name__ == "__main__":
    main()
