#!/usr/bin/env python3
"""Measure the per-column grid phase drift across the corpus.

A glyph whose ink is off-centre inside its sampling window has lost
part of its strokes: the model then has to guess the missing part, and
guessing `D` as `0` or `A` as `3` is exactly what a cut-off `D` or a
cut-off `A` looks like.

This tool reports, for every line of every photo:

  * the horizontal ink offset per cell (as a fraction of the window)
  * a LINEAR FIT of the offset against the column index

A slope that is clearly non-zero is the smoking gun: the sampling grid
is not parallel to the text, so the phase error grows along the line
and the tail cells are the worst-hit.  A constant offset instead means
the whole line is shifted; a noisy pattern means the segmentation
itself is unstable.

Usage:
    python3 tools/phase_drift.py --line 1
    python3 tools/phase_drift.py --line 1 --worst
"""
from __future__ import annotations

import argparse
import json

import numpy as np


def cell_offsets(X, meta, idxs):
    """Return (cols, h_off, v_off) for the given cache indices."""
    cols, hoff, voff = [], [], []
    for i in idxs:
        g = X[i].reshape(12, 16)
        tot = g.sum()
        if tot <= 0:
            continue
        ys, xs = np.mgrid[0:12, 0:16]
        cols.append(int(meta[i]["col"]))
        hoff.append(float((xs * g).sum() / tot) - 7.5)
        voff.append(float((ys * g).sum() / tot) - 5.5)
    return np.array(cols), np.array(hoff), np.array(voff)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/tmp/glyph_cache.npz")
    ap.add_argument("--line", type=int, default=1)
    ap.add_argument("--worst", type=int, default=8,
                    help="how many lines to show at the end")
    ap.add_argument("--per-cell", action="store_true")
    args = ap.parse_args()

    d = np.load(args.cache, allow_pickle=True)
    X = d["X"].astype(np.float32)
    meta = [json.loads(s) for s in d["meta"]]

    byfile = {}
    for i, m in enumerate(meta):
        if int(m["line"]) == args.line:
            byfile.setdefault(m["file"], []).append(i)

    rows = []
    for f in sorted(byfile):
        idxs = sorted(byfile[f], key=lambda i: int(meta[i]["col"]))
        cols, hoff, voff = cell_offsets(X, meta, idxs)
        if len(cols) < 10:
            continue
        slope, intercept = np.polyfit(cols, hoff, 1)
        # residual scatter tells phase noise apart from a smooth drift
        resid = hoff - (slope * cols + intercept)
        rows.append((abs(slope), slope, intercept, resid.std(), f, cols,
                     hoff, voff))

    rows.sort(reverse=True)
    print("line %d  slope of the horizontal ink offset vs column index "
          "(pixels of the 16-wide window)" % args.line)
    print("%-26s %8s %8s %8s" % ("file", "slope", "intercept", "resid sd"))
    for a, slope, inter, sd, f, cols, hoff, voff in rows[:args.worst]:
        print("%-26s %+8.4f %+8.3f %8.3f" % (f, slope, inter, sd))
    slopes = np.array([r[1] for r in rows])
    print()
    print("across %d lines: slope mean %+.4f  median %+.4f  "
          "abs mean %.4f  max %.4f" %
          (len(slopes), slopes.mean(), float(np.median(slopes)),
           np.abs(slopes).mean(), np.abs(slopes).max()))

    if args.per_cell and rows:
        a, slope, inter, sd, f, cols, hoff, voff = rows[0]
        print()
        print("per-cell offsets of %s" % f)
        for c, h, v in zip(cols, hoff, voff):
            print("  col %2d  h %+6.2f  v %+6.2f" % (c, h, v))


if __name__ == "__main__":
    main()
