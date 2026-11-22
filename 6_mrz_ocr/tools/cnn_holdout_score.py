#!/usr/bin/env python3
"""Score a CNN weight pair on the REAL-path holdout slices.

Reads the slice npz produced by tools/export_holdout_slices.py and reports
the same metric the product bench reports -- per-photo EXACT match of both
MRZ lines -- but without recompiling or re-running the C tool.

This is the fast iteration loop: the slices are the C pipeline's own
`MRZ_OCR_DUMP` output for the product band, so a weight set's accuracy on
them is the accuracy the product will see (modulo the beam/CRNN layer,
which this script does not model -- it isolates the CNN head).

Usage:
  python3 tools/cnn_holdout_score.py \
      --w1 src/cnn_weights_line1.h --w2 src/cnn_weights_line2.h \
      --npz /tmp/holdout_photo.npz
"""
from __future__ import annotations
import argparse, json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import cnn_offline_eval as off


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--w1", required=True)
    ap.add_argument("--w2", required=True)
    ap.add_argument("--npz", required=True)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--confusions", action="store_true")
    args = ap.parse_args()

    d = np.load(args.npz, allow_pickle=True)
    X = d["X"].astype(np.float32)
    Y = d["Y"].astype(np.int64)
    meta = [json.loads(s) for s in d["meta"]]

    p1 = off.pick(off.parse_header(args.w1), off.infer_prefix(off.parse_header(args.w1)))
    p2 = off.pick(off.parse_header(args.w2), off.infer_prefix(off.parse_header(args.w2)))

    pred = np.zeros(len(Y), dtype=np.int64)
    for line, P in ((0, p1), (1, p2)):
        idx = [i for i, m in enumerate(meta) if int(m["line"]) == line]
        if not idx:
            continue
        logits = off.forward(X[idx], P)
        pred[idx] = logits.argmax(1)

    # per-photo exact: both lines fully correct
    per = defaultdict(lambda: {"ok": 0, "tot": 0, "line": {0: [0, 0], 1: [0, 0]}})
    for i, m in enumerate(meta):
        key = (m["file"], m["label"])
        d2 = per[key]
        d2["tot"] += 1
        d2["line"][int(m["line"])][1] += 1
        if pred[i] == Y[i]:
            d2["ok"] += 1
            d2["line"][int(m["line"])][0] += 1

    n_photo = len(per)
    n_exact = sum(1 for v in per.values() if v["ok"] == v["tot"])
    n_l1 = sum(1 for v in per.values() if v["line"][0][0] == 44)
    n_l2 = sum(1 for v in per.values() if v["line"][1][0] == 44)
    char_acc = sum(v["ok"] for v in per.values()) / max(1, sum(v["tot"] for v in per.values()))
    print(f"photos={n_photo}  exact={n_exact}/{n_photo}  "
          f"line1_full={n_l1}/{n_photo}  line2_full={n_l2}/{n_photo}  "
          f"char_acc={char_acc*100:.2f}%")

    if args.verbose:
        print("  per-photo:")
        for (f, lab), v in sorted(per.items()):
            if v["ok"] != v["tot"]:
                print(f"    {lab:52s} {v['ok']}/{v['tot']}  "
                      f"l1={v['line'][0][0]}/44 l2={v['line'][1][0]}/44")

    if args.confusions:
        conf = Counter()
        for i in range(len(Y)):
            if pred[i] != Y[i]:
                conf[(off.GLYPHS[Y[i]], off.GLYPHS[pred[i]])] += 1
        print("  confusions:")
        for (g, p), n in conf.most_common(20):
            print(f"    {g} -> {p} : {n}")


if __name__ == "__main__":
    main()
