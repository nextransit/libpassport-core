#!/usr/bin/env python3
"""Show the real glyph slices the classifier gets wrong, side by side
with the glyph it should have been.

The point is to separate the two possible causes of a misread:

  * the SLICE is wrong (the cell window landed on the wrong ink, or the
    glyph is cropped) -- then the rendered shape simply does not look
    like the GT character;
  * the SLICE is right but the CLASSIFIER is wrong (the glyph leans, and
    the training corpus has no leaning examples) -- then the rendered
    shape DOES look like the GT character, but the network still picks
    something else.

Usage:
  python3 tools/cnn_holdout_glyphs.py --npz /tmp/holdout_photo.npz \
      --w1 ../src/cnn_weights_line1.h --w2 ../src/cnn_weights_line2.h \
      --file IMG_1792133000_003.jpg
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

import numpy as np

import cnn_offline_eval as off

RAMP = " .:-=+*#%@"


def render(g, w=2):
    """ASCII-render a 12x16 glyph."""
    rows = []
    for y in range(g.shape[0]):
        s = ""
        for x in range(g.shape[1]):
            v = float(g[y, x])
            k = int(min(len(RAMP) - 1, max(0, v * (len(RAMP) - 1))))
            s += RAMP[k] * w
        rows.append(s)
    return rows


def lean_of(g):
    xs = np.arange(g.shape[1])
    top = g[:4].sum(axis=0)
    bot = g[8:].sum(axis=0)
    if top.sum() <= 0 or bot.sum() <= 0:
        return 0.0
    return float((xs * bot).sum() / bot.sum() - (xs * top).sum() / top.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--w1", required=True)
    ap.add_argument("--w2", required=True)
    ap.add_argument("--file", default="")
    ap.add_argument("--line", type=int, default=None)
    ap.add_argument("--max", type=int, default=12)
    ap.add_argument("--only-wrong", action="store_true", default=True)
    args = ap.parse_args()

    d = np.load(args.npz, allow_pickle=True)
    X = d["X"].astype(np.float32)
    Y = d["Y"].astype(np.int64)
    meta = [json.loads(s) for s in d["meta"]]

    a1 = off.parse_header(args.w1)
    a2 = off.parse_header(args.w2)
    p1 = off.pick(a1, off.infer_prefix(a1))
    p2 = off.pick(a2, off.infer_prefix(a2))

    pred = np.zeros(len(Y), dtype=np.int64)
    for line, P in ((0, p1), (1, p2)):
        idx = [i for i, m in enumerate(meta) if int(m["line"]) == line]
        if idx:
            pred[idx] = off.forward(X[idx], P).argmax(1)

    shown = 0
    for i, m in enumerate(meta):
        if args.file and args.file not in m["file"]:
            continue
        if args.line is not None and int(m["line"]) != args.line:
            continue
        if pred[i] == Y[i]:
            continue
        g = X[i]
        rows = render(g)
        print("=" * 74)
        print(f"{m['label']}  line{m['line']} c{m['col']:02d}  GT={m['ch']}  "
              f"PRED={off.GLYPHS[pred[i]]}  lean={lean_of(g):+.2f}px  "
              f"ink={float(g.sum()):.1f}")
        for r in rows:
            print("    " + r)
        shown += 1
        if shown >= args.max:
            break


if __name__ == "__main__":
    main()
