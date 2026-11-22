#!/usr/bin/env python3
"""Split the product-path slice npz into a TRAIN set and a HOLDOUT set.

The split unit is the PHOTOGRAPH, not the slice: slices of one photo share
its font, lighting and segmentation, so a slice-level split leaks.

  * TRAIN    : the 41 tmp/PassPortPhoto photos (they have ground truth)
  * HOLDOUT  : the 6 Photo/ photos (never used for training)

Usage:
  python3 tools/split_product_slices.py --in /tmp/product_slices.npz \
      --train /tmp/prod_train.npz --holdout /tmp/prod_holdout.npz
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--holdout", required=True)
    ap.add_argument("--train-prefix", default="tmp/PassPortPhoto/")
    args = ap.parse_args()

    d = np.load(args.inp, allow_pickle=True)
    X = d["X"].astype(np.float32)
    Y = d["Y"].astype(np.int64)
    meta = [json.loads(s) for s in d["meta"]]

    tr = [i for i, m in enumerate(meta) if m["label"].startswith(args.train_prefix)]
    ho = [i for i, m in enumerate(meta) if not m["label"].startswith(args.train_prefix)]
    # guard: no photograph may appear in both halves
    trf = {meta[i]["file"] for i in tr}
    hof = {meta[i]["file"] for i in ho}
    overlap = trf & hof
    if overlap:
        raise SystemExit(f"photo leak: {sorted(overlap)}")

    for path, idx in ((args.train, tr), (args.holdout, ho)):
        np.savez(path,
                 X=X[idx], Y=Y[idx],
                 meta=np.array([json.dumps(meta[i]) for i in idx]))
        nfiles = len({meta[i]["file"] for i in idx})
        n1 = sum(1 for i in idx if meta[i]["line"] == 0)
        print(f"wrote {path}: {len(idx)} slices ({n1} line1) "
              f"from {nfiles} photos")


if __name__ == "__main__":
    main()
