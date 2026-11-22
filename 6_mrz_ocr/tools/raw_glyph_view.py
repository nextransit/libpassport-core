#!/usr/bin/env python3
"""Render the FULL-RESOLUTION band region of one MRZ cell as ASCII.

Why this exists
---------------
The 16x12 glyph the classifier sees is an area-average downsample of a
~42x35 box of band pixels.  When a cell is misread, there are two
possible root causes:

  (a) RESOLUTION: the band pixels do not contain the distinguishing
      feature at all, so no classifier can recover it;
  (b) CLASSIFIER: the band pixels DO contain the feature, but the
      16x12 downsample plus the network's decision turns it into the
      wrong class.

The two look identical in the 16x12 cache, so the cache cannot tell
them apart.  This tool renders the SOURCE band pixels of the cell
(NOT the downsample), which settles the question directly.

Usage:
    python3 tools/raw_glyph_view.py --band /tmp/holdout_export2/IMG_1792133000_003.ppm \
        --line 0 --col 12 --width 42
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "6_mrz_ocr" / "build" / "mrz_ocr_cnn_tool"
DUMP_RE = re.compile(
    r"mrz_dump_(?P<tag>.+)_l(?P<line>\d+)_c(?P<col>\d+)"
    r"_x(?P<x>-?\d+)_r(?P<y>-?\d+)_w(?P<w>\d+)"
    r"(?:_h(?P<h>\d+))?(?:_b(?P<b>-?\d+))?\.ppm$")

SHADES = " .:-=+*#%@"


def render(gray, w, h, step=2):
    """ASCII art of a grayscale crop; darker = more ink."""
    lines = []
    for r in range(0, h, step):
        row = []
        for c in range(0, w, step):
            v = float(gray[r, c])
            # invert: ink (dark) -> bright
            ink = 1.0 - v
            row.append(SHADES[min(9, max(0, int(ink * 9.99)))])
        lines.append("".join(row))
    return lines


def geometry(band_path, line, col, dump_dir):
    for f in glob.glob(os.path.join(dump_dir, "mrz_dump_*")):
        os.remove(f)
    env = dict(os.environ)
    env.update({
        "MRZ_OCR_DUMP": "cache",
        "MRZ_OCR_DUMP_DIR": dump_dir,
        "MRZ_OCR_LINE_MODELS": "1",
        "MRZ_OCR_CRNN": "0",
        "MRZ_OCR_FUSE": "0",
        "MRZ_OCR_PAR": "0",
    })
    subprocess.run([str(TOOL), str(band_path)], capture_output=True,
                   text=True, env=env, timeout=120)
    for p in Path(dump_dir).glob(f"mrz_dump_*_l{line}_c{col:02d}_*.ppm"):
        m = DUMP_RE.match(p.name)
        if m:
            g = m.groupdict()
            return (int(g["x"]), int(g["y"]), int(g["w"]),
                    int(g["h"] or 0), int(g["b"] or 0))
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--band", required=True)
    ap.add_argument("--line", type=int, required=True)
    ap.add_argument("--col", type=int, required=True)
    ap.add_argument("--dump-dir", default="/tmp/rawview_dump")
    ap.add_argument("--step", type=int, default=2)
    ap.add_argument("--pad", type=int, default=4)
    args = ap.parse_args()

    dd = Path(args.dump_dir)
    dd.mkdir(parents=True, exist_ok=True)
    geo = geometry(args.band, args.line, args.col, dd)
    if geo is None:
        raise SystemExit("no dump geometry found")
    x, r, w, h, b = geo
    band = np.asarray(Image.open(args.band).convert("L"),
                      dtype=np.float32) / 255.0
    H, W = band.shape
    top = max(0, b + r - args.pad)
    bot = min(H, b + r + h + args.pad)
    left = max(0, x - args.pad)
    right = min(W, x + w + args.pad)
    crop = band[top:bot, left:right]
    if crop.size == 0:
        raise SystemExit("empty crop")
    print(f"{Path(args.band).stem}  line{args.line} col{args.col}")
    print(f"  cell box x={x} w={w}  row={b + r} h={h}   "
          f"crop {left}..{right} x {top}..{bot}")
    print(f"  ink mean={float((1.0 - crop).mean()):.4f}")
    for ln in render(crop, crop.shape[1], crop.shape[0], args.step):
        print("   " + ln)


if __name__ == "__main__":
    main()
