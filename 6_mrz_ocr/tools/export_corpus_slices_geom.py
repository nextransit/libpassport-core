#!/usr/bin/env python3
"""Export glyph slices from a rendered corpus by GEOMETRY, not by reading.

Why this exists
---------------
`export_corpus_slices.py` finds the cell -> ground-truth correspondence
by aligning the TOOL's reading against the corpus text (Levenshtein
backtrace).  That works only while the model can read the image: on the
physical corpus the legacy-rendered model fails outright (measured:
173-243 of 375 images per shard returned a failed reading, and the
alignment dropped ~250-300 more), so the slices cannot be exported that
way -- and bootstrapping the labels from a model that cannot read the
image is exactly the wrong dependency.

The corpus is RENDERED, so its grid is known by construction:
`gen_mrz_image` draws line L at x = margin + i * (8*scale + hgap) with
a per-character skew, and the glyph cell is 8x12 source pixels scaled
by `scale`.  Cutting the cells analytically therefore gives EXACT
ground-truth labels with no dependence on any recogniser.

The cell is resampled to the 16x12 classifier input the same way the C
side resizes it (area/bilinear over the glyph box), and the ink is
inverted the same way (`1 - gray/255`).

Usage:
    python3 tools/export_corpus_slices_geom.py --corpus data/corpus_train_phys \
        --out data/corpus_train_phys/geom_slices.npz
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

GLYPHS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ<"
GI = {c: i for i, c in enumerate(GLYPHS)}
CELL_W_SRC = 8      # source glyph width in pixels (template.h)
CELL_H_SRC = 12
MARGIN = 12         # gen_mrz_image.c default
IN_W, IN_H = 16, 12  # classifier input


def cut_cell_scaled(gray: np.ndarray, x0: float, y0: float,
                    w: float, h: float) -> np.ndarray:
    """The 16x12 classifier input for one glyph, cut from the band.

    The glyph box is resampled to the network input (the C side resizes
    the glyph box to 16x12) and the ink is inverted: `1 - gray/255`.
    """
    xi, yi = int(round(x0)), int(round(y0))
    xi = max(0, xi); yi = max(0, yi)
    box = gray[yi:yi + max(1, int(round(h))), xi:xi + max(1, int(round(w)))]
    if box.size == 0:
        return np.zeros((IN_H, IN_W), np.float32)
    im = Image.fromarray(box.astype(np.uint8), mode="L")
    im = im.resize((IN_W, IN_H), Image.BILINEAR)
    a = np.asarray(im, dtype=np.float32) / 255.0
    return 1.0 - a


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    corpus = Path(args.corpus)
    data = json.loads((corpus / "corpus.json").read_text())
    recs = data["records"]
    if args.limit:
        recs = recs[: args.limit]

    X, Y, META = [], [], []
    for rec in recs:
        img_path = corpus / rec["image"]
        if not img_path.exists():
            continue
        g = np.asarray(Image.open(img_path).convert("L"), dtype=np.float32)
        # S5 rotates the whole band about its centre (both pipelines).
        # Invert it before cutting, or every cell of a rotated image is
        # sheared: the corpus record carries the angle, and at <=3 deg
        # the bilinear inverse leaves sub-pixel error.
        rot = float(rec.get("rot") or 0.0)
        if rot:
            im = Image.fromarray(g.astype(np.uint8), mode="L").rotate(
                -rot, resample=Image.BILINEAR, fillcolor=230,
                center=((g.shape[1] - 1) / 2.0, (g.shape[0] - 1) / 2.0))
            g = np.asarray(im, dtype=np.float32)
        scale = int(rec["scale"])
        hgap = int(rec.get("hgap", 2))
        vgap = int(rec.get("vgap", 6))
        skew = int(rec.get("skew", 0))
        margin = MARGIN
        glyph_w = CELL_W_SRC * scale
        glyph_h = CELL_H_SRC * scale
        n_lines = 2 if rec.get("line2") else 1
        n_chars = max(len(rec.get("line1") or ""), len(rec.get("line2") or ""))
        # The physical pipeline may downscale the band (S8).  The corpus
        # record does not carry that factor, so derive it from the image
        # size against the geometry the renderer would have produced.
        exp_w = margin * 2 + n_chars * glyph_w + max(0, n_chars - 1) * hgap
        exp_h = glyph_h * n_lines + vgap * (n_lines - 1) + margin * 2
        sx = g.shape[1] / float(exp_w)
        sy = g.shape[0] / float(exp_h)
        for li, key in ((0, "line1"), (1, "line2")):
            text = rec.get(key) or ""
            if not text:
                continue
            row_y = (margin + li * (glyph_h + vgap)) * sy
            n = len(text)
            for i, ch in enumerate(text):
                if ch not in GI:
                    continue
                skew_dx = skew * i // n if skew else 0
                x = (margin + i * (glyph_w + hgap) + skew_dx) * sx
                cell = cut_cell_scaled(g, x, row_y, glyph_w * sx,
                                       glyph_h * sy)
                X.append(cell.reshape(-1))
                Y.append(GI[ch])
                META.append({"file": rec["id"], "line": li, "col": i,
                              "ch": ch, "tier": "A", "weak": 0})
    if not X:
        print("no slices exported", flush=True)
        return 1
    Xa = np.asarray(X, dtype=np.float32)
    np.savez(args.out, X=Xa, Y=np.asarray(Y, dtype=np.int64),
             meta=np.array([json.dumps(m) for m in META]))
    print(f"exported {len(Y)} slices from {len({m['file'] for m in META})} "
          f"images -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
