#!/usr/bin/env python3
"""Sweep the resampling-window width per photo/line and report the ceiling.

Why
---
The C side chooses the resampling window from `medw + 8` (median ink-run
width + padding).  On photos whose glyphs touch, runs merge, `medw`
inflates, and each 16x12 slice carries one character plus a slice of its
neighbour.  `slice_width_probe.py` showed that on Photo/3 a wider window
buys +3 characters.

This tool runs that probe over EVERY photo/line in the corpus and answers
two questions:

  1. ORACLE  -- if we picked the best width per (photo,line) with hindsight,
                how many photos become exactly right?  This is the ceiling
                of any width rule.
  2. REALISTIC -- can a rule computable from the band itself (no GT) pick
                that width?  We search a family of candidate rules and
                score each by exact-match over the whole corpus.

Both are offline: the band pixels come from the product export and the
scoring uses the offline numpy forward in `cnn_offline_eval`.

Usage:
  python3 tools/width_oracle_sweep.py --bands /tmp/product_export
"""
from __future__ import annotations
import argparse, json, os, re, subprocess, sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path("/Users/zhouyong/Desktop/work/Decard/gitlab/passport")
sys.path.insert(0, str(ROOT / "6_mrz_ocr" / "tools"))
import cnn_offline_eval as off

ALPHABET = off.GLYPHS
GI = off.GI
TOOL = ROOT / "6_mrz_ocr/build/mrz_ocr_cnn_tool"


def grid_of(ppm):
    env = dict(os.environ)
    env.update({"MRZ_OCR_LINE_MODELS": "1", "MRZ_OCR_GRID": "1",
                "MRZ_OCR_LINES": "1", "MRZ_OCR_CRNN": "0", "MRZ_OCR_FUSE": "0"})
    r = subprocess.run([str(TOOL), str(ppm)], capture_output=True, text=True, env=env)
    grids, boxes, by = [], [], 0
    for l in r.stderr.splitlines():
        if l.startswith("GRID W="):
            grids.append(dict(kv.split("=", 1) for kv in l.split() if "=" in kv))
        elif l.startswith("LINES band="):
            for mm in re.finditer(r"l\d=\(y(\d+) h(\d+)\)", l):
                boxes.append((int(mm.group(1)), int(mm.group(2))))
    for l in r.stdout.splitlines():
        if l.startswith("band.x band.y"):
            by = int(l.split()[-3])
    return grids, boxes, by


def resample(gray, rx, ry, rw, rh):
    """Mirror resample_char_gray: area-average -> 3x3 smooth -> x centroid."""
    out = np.zeros((12, 16), dtype=np.float32)
    H, W = gray.shape
    for oy in range(12):
        sy0 = (oy * rh) // 12
        sy1 = max(sy0 + 1, ((oy + 1) * rh) // 12)
        for ox in range(16):
            sx0 = (ox * rw) // 16
            sx1 = max(sx0 + 1, ((ox + 1) * rw) // 16)
            s = 0.0
            n = 0
            for sy in range(sy0, sy1):
                yy = ry + sy
                if yy < 0 or yy >= H:
                    continue
                for sx in range(sx0, sx1):
                    xx = rx + sx
                    if xx < 0 or xx >= W:
                        continue
                    s += gray[yy, xx]
                    n += 1
            ink = s / n if n else 0.0
            out[oy, ox] = 1.0 - ink / 255.0
    tmp = out.copy()
    for y in range(12):
        for x in range(16):
            acc = 0.0
            c = 0
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < 12 and 0 <= nx < 16:
                        acc += tmp[ny, nx]
                        c += 1
            out[y, x] = acc / c
    xs = np.arange(16)
    v = out
    m = v > 0.35
    if m.sum() > 0:
        cx = float((xs[None, :] * v).sum() / v.sum())
        delta = int(round(cx - 7.5))
        delta = max(-2, min(2, delta))
        if delta:
            out = np.roll(out, -delta, axis=1)
            if delta > 0:
                out[:, :delta] = 0
            else:
                out[:, delta:] = 0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bands", default="/tmp/product_export")
    ap.add_argument("--w1", default=str(ROOT / "6_mrz_ocr/src/cnn_weights_line1.h"))
    ap.add_argument("--w2", default=str(ROOT / "6_mrz_ocr/src/cnn_weights_line2.h"))
    ap.add_argument("--gt", default=str(ROOT / "10_ocr_research/photo_direct_bench.json"))
    ap.add_argument("--out", default="/tmp/width_oracle.json")
    args = ap.parse_args()

    gts = {}
    for r in json.load(open(args.gt))["records"]:
        if r.get("gt"):
            gts[Path(r["path"]).name] = r["gt"]

    a1 = off.parse_header(args.w1)
    p1 = off.pick(a1, off.infer_prefix(a1))
    a2 = off.parse_header(args.w2)
    p2 = off.pick(a2, off.infer_prefix(a2))

    # width candidates: cell-relative and pitch-relative
    WIDTHS = list(range(20, 78))

    rows = []
    for p in sorted(Path(args.bands).glob("*.ppm")):
        gt = gts.get(p.stem + ".jpg")
        if not gt:
            continue
        grids, boxes, by = grid_of(p)
        if not grids or len(boxes) < 2:
            continue
        gray = np.asarray(Image.open(p)).astype(np.float32).mean(axis=2)
        for li, d in enumerate(grids):
            if li > 1:
                break
            c0 = float(d["c0"])
            L = float(d["L"])
            H = int(d["H"])
            cell = int(d.get("cell", 0))
            y0 = boxes[li][0] + by
            truth = gt["line1"] if li == 0 else gt["line2"]
            P = p1 if li == 0 else p2
            if len(truth) != 44:
                continue
            accs = {}
            for w in WIDTHS:
                Xs, Ys = [], []
                for c in range(44):
                    ch = truth[c]
                    if ch not in GI:
                        continue
                    cx = c0 + c * L
                    x0 = int(round(cx - w / 2.0))
                    g = resample(gray, x0, y0, w, H)
                    Xs.append(g)
                    Ys.append(GI[ch])
                if not Xs:
                    continue
                Xa = np.asarray(Xs, dtype=np.float32)
                Ya = np.asarray(Ys, dtype=np.int64)
                pred = off.forward(Xa, P).argmax(1)
                accs[w] = int((pred == Ya).sum())
            if not accs:
                continue
            rows.append(dict(file=p.stem, line=li, c0=c0, L=L, cell=cell, H=H,
                             accs={str(k): v for k, v in accs.items()}))

    json.dump(rows, open(args.out, "w"))

    # ---- oracle ceiling -------------------------------------------------
    byfile = defaultdict(dict)
    for r in rows:
        byfile[r["file"]][r["line"]] = r

    def score_at(r, w):
        return r["accs"].get(str(w), 0)

    n_photo = 0
    n_base = 0
    n_oracle = 0
    detail = []
    for f, lines in sorted(byfile.items()):
        if len(lines) < 2:
            continue
        n_photo += 1
        b = min(score_at(lines[0], lines[0]["cell"]),
                score_at(lines[1], lines[1]["cell"]))
        best = 0
        bw = (None, None)
        for w in WIDTHS:
            s = min(score_at(lines[0], w), score_at(lines[1], w))
            if s > best:
                best = s
                bw = (w, w)
        # per-line independent oracle is not allowed (same width per line,
        # but the two lines may pick different widths -- that IS allowed,
        # the C side decides per line)
        bl0 = max(WIDTHS, key=lambda w: score_at(lines[0], w))
        bl1 = max(WIDTHS, key=lambda w: score_at(lines[1], w))
        ind = min(score_at(lines[0], bl0), score_at(lines[1], bl1))
        if b == 44:
            n_base += 1
        if ind == 44:
            n_oracle += 1
        detail.append((f, b, bl0, score_at(lines[0], bl0), bl1,
                       score_at(lines[1], bl1), ind))
    print(f"photos={n_photo}  base(cell) exact={n_base}/{n_photo}  "
          f"ORACLE(width) exact={n_oracle}/{n_photo}")
    print("\nfile                          base  l0:w=acc    l1:w=acc")
    for f, b, w0, a0, w1, a1, ind in detail:
        mark = "  <== ORACLE FIX" if (b != 44 and ind == 44) else ""
        print(f"  {f:28s} {b:2d}/44   {w0:3d}:{a0:2d}   {w1:3d}:{a1:2d}{mark}")


if __name__ == "__main__":
    main()
