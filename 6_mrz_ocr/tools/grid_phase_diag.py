#!/usr/bin/env python3
"""Measure the ink position inside each 16x12 glyph window.

The rendered glyphs of Photo/3 show ink pushed to one side of the
window (the `A` at col 40 has no left diagonal; the `D` at col 37 has
almost no top bar).  That is a GEOMETRY symptom, not a classifier
symptom, and the two have very different fixes.

For every cached cell this tool measures, on the 16x12 resample the
classifier actually sees:

  * the horizontal centre of mass of the ink, relative to the window
  * the vertical centre of mass
  * the left/right ink coverage (how much of each half carries ink)
  * the fill fraction (how much of the window carries any ink at all)

A systematic drift means the sampling window is off-centre with respect
to the glyph, so part of the letter is cut away and the model has to
guess the missing stroke -- which is exactly how `A` becomes `3` and
`D` becomes `0`.

Usage:
    python3 tools/grid_phase_diag.py --file IMG_1792133000_003 --line 1
"""
from __future__ import annotations

import argparse
import json

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/tmp/glyph_cache.npz")
    ap.add_argument("--file", default="")
    ap.add_argument("--line", type=int, default=1)
    ap.add_argument("--show", type=int, default=0,
                    help="print this many rows per cell")
    args = ap.parse_args()

    d = np.load(args.cache, allow_pickle=True)
    X = d["X"].astype(np.float32)
    meta = [json.loads(s) for s in d["meta"]]

    sel = [i for i, m in enumerate(meta)
           if int(m["line"]) == args.line
           and (not args.file or m["file"] == args.file)]
    sel.sort(key=lambda i: (meta[i]["file"], int(meta[i]["col"])))
    if not sel:
        print("no cells")
        return

    cur_file = None
    h_off, v_off = [], []
    for i in sel:
        m = meta[i]
        if m["file"] != cur_file:
            cur_file = m["file"]
            h_off, v_off = [], []
            if args.file:
                print("=== %s" % cur_file)
        g = X[i].reshape(12, 16)
        tot = g.sum()
        if tot <= 0:
            continue
        ys, xs = np.mgrid[0:12, 0:16]
        cx = float((xs * g).sum() / tot)
        cy = float((ys * g).sum() / tot)
        # the window centre is 7.5 / 5.5
        off = (cx - 7.5) / 16.0
        voff = (cy - 5.5) / 12.0
        h_off.append(off)
        v_off.append(voff)
        left = g[:, :8].sum() / max(tot, 1e-9)
        fill = float((g > 0.05).sum()) / g.size
        if args.show:
            print("  col %2d gt=%-2s  h_off %+.3f  v_off %+.3f  "
                  "left %.2f  fill %.2f" %
                  (int(m["col"]), m.get("ch", ""), off, voff, left, fill))

    if h_off:
        ho = np.asarray(h_off)
        vo = np.asarray(v_off)
        print("%-26s h_off mean %+.3f med %+.3f std %.3f | "
              "v_off mean %+.3f std %.3f | |h|>0.12: %d/%d" %
              (cur_file, ho.mean(), float(np.median(ho)), ho.std(),
               vo.mean(), vo.std(),
               int((np.abs(ho) > 0.12).sum()), len(ho)))


if __name__ == "__main__":
    main()
