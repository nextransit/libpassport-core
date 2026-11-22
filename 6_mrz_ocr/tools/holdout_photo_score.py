#!/usr/bin/env python3
"""Score a CNN weight pair on the HELD-OUT photos (Photo/3-8) and on the
training photos, using the product-path slices.

The holdout unit is the PHOTOGRAPH, not the slice: Photo/3-6 are byte
-identical files and Photo/8 is a different shot of the same document, so
a slice-level split would leak the answer.

Reports what the product cares about -- per-photo exact match of both
MRZ lines -- plus character accuracy, so a candidate weight set can be
rejected before the minute-scale bench.

Usage:
  python3 tools/holdout_photo_score.py --w1 a.h --w2 b.h --npz /tmp/holdout_photo2.npz
"""
from __future__ import annotations
import argparse, json, sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cnn_offline_eval as off

ALPHABET = off.GLYPHS


def load(w):
    h = off.parse_header(w)
    return off.pick(h, off.infer_prefix(h))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--w1", required=True)
    ap.add_argument("--w2", required=True)
    ap.add_argument("--npz", default="/tmp/holdout_photo2.npz")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--confusions", action="store_true")
    ap.add_argument("--split", default="holdout", choices=["holdout", "all", "train"])
    args = ap.parse_args()

    d = np.load(args.npz, allow_pickle=True)
    X = d["X"].astype(np.float32)
    Y = d["Y"].astype(np.int64)
    meta = [json.loads(s) for s in d["meta"]]

    p1, p2 = load(args.w1), load(args.w2)
    pred = np.zeros(len(Y), dtype=np.int64)
    for line, P in ((0, p1), (1, p2)):
        idx = [i for i, m in enumerate(meta) if int(m["line"]) == line]
        if idx:
            pred[idx] = off.forward(X[idx], P).argmax(1)

    def is_hold(m):
        return m["file"].startswith("IMG_")

    groups = {"holdout": [m for m in meta if is_hold(m)],
              "train": [m for m in meta if not is_hold(m)],
              "all": list(meta)}

    for gname in ([args.split] if args.split != "all" else ["holdout", "train", "all"]):
        idxs = [i for i, m in enumerate(meta) if m in groups[gname]] \
            if False else None
        sel = set()
        for i, m in enumerate(meta):
            if gname == "holdout" and is_hold(m):
                sel.add(i)
            elif gname == "train" and not is_hold(m):
                sel.add(i)
            elif gname == "all":
                sel.add(i)
        per = defaultdict(lambda: {0: [0, 0], 1: [0, 0]})
        for i in sel:
            m = meta[i]
            l = int(m["line"])
            per[m["file"]][l][1] += 1
            if pred[i] == Y[i]:
                per[m["file"]][l][0] += 1
        if not per:
            continue
        n_exact = sum(1 for v in per.values()
                      if v[0][0] == v[0][1] and v[1][0] == v[1][1])
        n_l1 = sum(1 for v in per.values() if v[0][0] == v[0][1])
        n_l2 = sum(1 for v in per.values() if v[1][0] == v[1][1])
        ok = sum(v[0][0] + v[1][0] for v in per.values())
        tot = sum(v[0][1] + v[1][1] for v in per.values())
        print(f"[{gname}] photos={len(per)} exact={n_exact}/{len(per)} "
              f"l1={n_l1} l2={n_l2} char_acc={ok/tot*100:.2f}%")
        if args.verbose:
            for f, v in sorted(per.items()):
                bad = (v[0][0] != v[0][1]) or (v[1][0] != v[1][1])
                if bad:
                    print(f"    {f:30s} l1={v[0][0]}/{v[0][1]} l2={v[1][0]}/{v[1][1]}")

    if args.confusions:
        conf = Counter()
        for i, m in enumerate(meta):
            if pred[i] != Y[i]:
                conf[(ALPHABET[Y[i]], ALPHABET[pred[i]])] += 1
        print("  confusions:", ", ".join(f"{g}->{p}:{n}" for (g, p), n
                                         in conf.most_common(15)))


if __name__ == "__main__":
    main()
