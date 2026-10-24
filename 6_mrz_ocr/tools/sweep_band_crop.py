#!/usr/bin/env python3
"""Does the assets set fail on the crop, on rotation, or on the glyphs?

The assets set has no trustworthy text label, so the only honest signal
is the ICAO check digits on line 2 -- and even that is weak, because the
decoder's beam search will walk the least-confident columns until the
five digits close.  It is still the best available ordering signal, and
it is used here only to RANK hypotheses, never to certify them.

Three hypotheses, each cheap to falsify:

  crop     catalog.json's loc.mrz clips the first/last glyph, so the
           44-slot grid starts or ends off the text
  rotate   the document is tilted inside the crop; a 44-character row
           drifts vertically by 44*pitch*tan(theta), i.e. 1.7 px per
           degree at 33 px/char -- enough to walk a glyph out of its
           row band long before the tilt is visible
  glyph    neither: the decode is simply wrong, and only training helps

Sweep the first two and see whether either moves the valid rate.

Usage:
    python3 tools/sweep_band_crop.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "build" / "mrz_ocr_cnn_tool"
ASSETS = ROOT.parent / "assets" / "pic"

W = (7, 3, 1)


def val(c: str) -> int:
    if c == "<":
        return 0
    if "0" <= c <= "9":
        return ord(c) - 48
    return ord(c) - 55


def cd(s: str) -> int:
    return sum(val(c) * W[i % 3] for i, c in enumerate(s)) % 10


def l2_ok(l: str) -> bool:
    if not l or len(l) != 44:
        return False
    try:
        for pos, (a, b) in ((9, (0, 9)), (19, (13, 19)), (27, (21, 27)),
                            (42, (28, 42))):
            if l[pos] != str(cd(l[a:b])):
                return False
        return l[43] == str(cd(l[0:10] + l[13:20] + l[21:28] + l[28:43]))
    except Exception:
        return False


ENV = dict(os.environ)
ENV.update(MRZ_OCR_LINE_MODELS="1", MRZ_OCR_PAR="0", MRZ_OCR_CRNN="0")


def read(ppm: Path) -> str:
    p = subprocess.run([str(TOOL), str(ppm)], capture_output=True, text=True,
                       env=ENV, timeout=120)
    for ln in p.stdout.splitlines():
        if ln.startswith("result.line2"):
            return ln.partition(":")[2].strip()
    return ""


def bands():
    cat = json.load(open(ASSETS / "catalog.json"))
    out = []
    for r in cat.get("records", []):
        name, loc = r.get("file"), (r.get("loc") or {}).get("mrz")
        if not name or not loc:
            continue
        img = cv2.imread(str(ASSETS / name)) if (ASSETS / name).exists() else None
        if img is None:
            continue
        out.append((Path(name).stem, img, loc))
    return out


def crop(img, loc, xpad, ypad):
    H, Wd = img.shape[:2]
    x0 = max(0, loc["x"] - xpad)
    x1 = min(Wd, loc["x"] + loc["w"] + xpad)
    y0 = max(0, loc["y"] - ypad)
    y1 = min(H, loc["y"] + loc["h"] + ypad)
    b = img[y0:y1, x0:x1]
    return b if b.shape[0] >= 16 and b.shape[1] >= 44 else None


def rot(img, deg):
    if deg == 0:
        return img
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="bc_"))
    items = bands()
    n = len(items)
    print(f"assets with loc.mrz: {n}")

    def run_all(xpad, ypad, deg):
        hits = 0
        for stem, img, loc in items:
            b = crop(img, loc, xpad, ypad)
            if b is None:
                continue
            b = rot(b, deg)
            ppm = tmp / f"{stem}.ppm"
            Image.fromarray(cv2.cvtColor(b, cv2.COLOR_BGR2RGB)).save(
                str(ppm), format="PPM")
            hits += l2_ok(read(ppm))
        return hits

    print("\n--- horizontal padding (ypad 6, no rotation) ---")
    for xpad in (0, 6, 12, 24, 40):
        print(f"  xpad={xpad:>3}  valid {run_all(xpad, 6, 0):>3}/{n}", flush=True)

    print("\n--- rotation (xpad 0, ypad 6) ---")
    for deg in (-3.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 3.0):
        print(f"  rot={deg:>5.1f}  valid {run_all(0, 6, deg):>3}/{n}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
