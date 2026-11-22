#!/usr/bin/env python3
"""Render cached glyph slices as ASCII so a mis-read cell can be judged.

The question this answers: when the classifier reads `M` as `N`, is the
reason a SEGMENTATION problem (the crop is offset, so the glyph is cut
and the missing stroke looks like another letter) or a RESOLUTION
problem (the crop is right but 16x12 grayscale cannot tell M from N)?

Those two diagnoses lead to opposite fixes -- the first is a geometry
bug, the second needs a bigger input or a different head -- so they
must be separated before any training is scheduled.

Usage:
    python3 tools/show_glyphs.py --file IMG_1792133000_003 --cols 12
    python3 tools/show_glyphs.py --file IMG_1792133000_003 --line 1 --cols 30,37
"""
from __future__ import annotations

import argparse
import json

import numpy as np

SHADES = " .:-=+*#%@"


def render(x, w=16, h=12):
    """x: (h*w,) flat float in [0,1] with 1 = ink."""
    g = x.reshape(h, w)
    out = []
    for r in range(h):
        out.append(''.join(SHADES[min(9, max(0, int(v * 9.99)))]
                           for v in g[r]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/tmp/glyph_cache.npz")
    ap.add_argument("--file", default="IMG_1792133000_003")
    ap.add_argument("--line", type=int, default=0)
    ap.add_argument("--cols", default="")
    ap.add_argument("--all-fail", action="store_true",
                    help="render every cell of the file")
    args = ap.parse_args()

    d = np.load(args.cache, allow_pickle=True)
    X = d["X"].astype(np.float32)
    meta = [json.loads(s) for s in d["meta"]]

    want = {int(c) for c in args.cols.split(",") if c.strip() != ""}
    hits = []
    for i, m in enumerate(meta):
        if m.get("file", "") != args.file:
            continue
        if int(m.get("line", -1)) != args.line:
            continue
        if want and int(m["col"]) not in want:
            continue
        hits.append(i)

    if not hits:
        print("no cached glyph for that file/line")
        return
    hits.sort(key=lambda i: int(meta[i]["col"]))

    for i in hits:
        m = meta[i]
        print("=== col %2d  gt=%r  meta=%s" % (
            int(m["col"]), m.get("ch", ""),
            {k: v for k, v in m.items() if k != "ch"}))
        for row in render(X[i]):
            print("   " + row)
        print()


if __name__ == "__main__":
    main()
