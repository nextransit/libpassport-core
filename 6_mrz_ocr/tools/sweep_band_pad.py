#!/usr/bin/env python3
"""Sweep the vertical pad around the two MRZ rows and measure both heads.

Why the pad is the variable that matters
----------------------------------------
The two heads are trained on DIFFERENT crop geometries:

  * per-character CNN  -> 16x12 glyph patches cut by the C segmenter
    (train_cnn_omtest.py); only the per-glyph normalisation reaches it,
    so the band pad changes which glyphs get cut, not the glyph shape;
  * CRNN               -> 44x96 whole-row feature sequences
    (corpus_real, built by build_real_corpus.isolate_mrz with a 6-7 px
    pad). The band pad changes the band-local Otsu threshold, the
    deskew angle, the line split and therefore every feature value.

Android's nativeBand() pads the band by sharedPad = max(8, min(25,
0.7*row_height)) -- i.e. 25 px on these photos, giving a ~118 px band.
Both host benchmarks use a tighter pad (bench_real: 4 px, ~85 px band;
corpus_real: 6-7 px, ~73 px band), so neither reproduces the device
geometry.

This script sweeps pad in {4, 12, 25, 40} px and reports char accuracy
and exact matches for both heads, which separates "head quality" from
"crop geometry mismatch".

Usage:
  /opt/homebrew/bin/python3 tools/sweep_band_pad.py [--pads 4,12,25,40]
"""
from __future__ import annotations
import argparse
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "build" / "mrz_ocr_cnn_tool"
OMTEST = ROOT.parent / "tmp" / "omtest"
sys.path.insert(0, str(ROOT / "tools"))
from cmp_crop_crnn import parse_result, l2_ok  # noqa: E402


def run(img: Path, crnn: int):
    env = dict(os.environ)
    env["MRZ_OCR_LINE_MODELS"] = "1"
    env["MRZ_OCR_PAR"] = "0"
    env["MRZ_OCR_SEGDBG"] = "1"
    env["MRZ_OCR_CRNN"] = "1" if crnn else "0"
    p = subprocess.run([str(TOOL), str(img)], capture_output=True, text=True,
                       timeout=60, env=env)
    kv = {}
    for ln in p.stdout.splitlines():
        if ":" in ln:
            k, _, v = ln.partition(":")
            kv[k.strip()] = v.strip()
    sc = [int(x) for x in re.findall(r"score=(\d+)", p.stderr)]
    return kv.get("result.line2", ""), (min(sc) if sc else -1)


def carr(a: str, b: str) -> float:
    n = max(len(a), len(b))
    return (sum(1 for i in range(n)
                if (a[i] if i < len(a) else "_") == (b[i] if i < len(b) else "_")) / n) if n else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pads", default="4,12,25,40")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    pads = [int(x) for x in args.pads.split(",")]

    tmp = Path(tempfile.mkdtemp(prefix="pad_"))
    samples = []
    for d in sorted(p for p in OMTEST.iterdir() if p.is_dir()):
        for txt in sorted(d.glob("*.jpg-result.txt")):
            img = txt.with_name(txt.name.replace("-result.txt", ""))
            if not img.exists():
                continue
            l1, l2, boxes = parse_result(txt)
            if not l1 or not l2 or len(l1) != 44 or len(l2) != 44 or len(boxes) < 2:
                continue
            rgb = cv2.imread(str(img))
            H, Wd = rgb.shape[:2]
            cy = H - (H // 3) - 50
            y1 = min(p[1] for p in boxes[0]) + cy
            y2 = max(p[1] for p in boxes[1]) + cy
            samples.append((f"{d.name}/{img.stem}", rgb, y1, y2, Wd, H, l1, l2))
    if args.limit:
        samples = samples[: args.limit]

    print(f"n={len(samples)}  rows y={samples[0][2]}..{samples[0][3]}\n")
    print(f"{'pad':>4} {'bandH':>6} {'head':>5} {'l2_acc':>8} {'l2_exact':>9} "
          f"{'cksum':>7} {'mean_seg':>9}")
    for pad in pads:
        agg = {"cnn": [], "crnn": []}
        for name, rgb, y1, y2, Wd, H, l1, l2 in samples:
            band = rgb[max(0, y1 - pad):min(H, y2 + 1 + pad), 0:Wd]
            ppm = tmp / f"{name.replace('/', '_')}_p{pad}.ppm"
            Image.fromarray(cv2.cvtColor(band, cv2.COLOR_BGR2RGB)).save(str(ppm), format="PPM")
            for head, crnn in (("cnn", 0), ("crnn", 1)):
                out, seg = run(ppm, crnn)
                agg[head].append((l2, out, seg, band.shape[0]))
        for head in ("cnn", "crnn"):
            v = agg[head]
            a = sum(carr(g, o) for g, o, _, _ in v) / len(v)
            ex = sum(1 for g, o, _, _ in v if g == o)
            ck = sum(1 for g, o, _, _ in v if l2_ok(o))
            seg = sum(s for _, _, s, _ in v) / len(v)
            h = v[0][3]
            print(f"{pad:4d} {h:6d} {head:>5} {a*100:7.2f}% {ex:6d}/{len(v)} "
                  f"{ck:6d}/{len(v)} {seg:8.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
