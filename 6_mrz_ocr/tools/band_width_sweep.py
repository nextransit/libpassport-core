#!/usr/bin/env python3
"""Does the band width drive the CNN's accuracy?

The six `Photo/3..8` failures are one passport band at 2392 px wide,
while every other holdout slice is 1440 px.  The CNN resamples every
cell to 12x16, so a wider band means each character is squeezed harder
before it ever reaches the classifier -- and the resampling is the only
place where the scale difference can be lost.

This sweeps the band width: every holdout slice is rescaled to a common
width and re-read, so the accuracy curve shows whether the shipped
weights are width-sensitive and where the optimum sits.

Usage:
  python3 tools/band_width_sweep.py --widths 1440 1800 2048 2392
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "6_mrz_ocr" / "build" / "mrz_ocr_cnn_tool"
CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ<"


def parse(out: str) -> dict:
    kv = {}
    for ln in out.splitlines():
        if ":" not in ln:
            continue
        k, _, v = ln.partition(":")
        kv[k.strip()] = v.strip()
    return kv


def read(band: Path, env_extra: dict) -> dict:
    env = dict(os.environ)
    env.update(env_extra)
    env["MRZ_OCR_LINE_MODELS"] = "1"
    env["MRZ_OCR_PAR"] = "0"
    p = subprocess.run([str(TOOL), str(band)], capture_output=True,
                       text=True, env=env, timeout=20)
    return parse(p.stdout)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--band-dir", default="/tmp/holdout_export2")
    ap.add_argument("--gt", default="10_ocr_research/gt_review_loop.json")
    ap.add_argument("--widths", default="1440,1800,2048,2392,2880")
    ap.add_argument("--engines", default="cnn,crnn")
    ap.add_argument("--engine-only", default="")
    args = ap.parse_args()

    gt = json.loads((ROOT / args.gt).read_text())
    gmap = {}
    for k, v in gt.items():
        stem = Path(k).stem
        if len(v[0]) == 44 and len(v[1]) == 44:
            gmap[stem] = (v[0], v[1])

    bands = sorted(Path(args.band_dir).glob("*.ppm"))
    tmp = Path(tempfile.mkdtemp(prefix="widthsweep_"))
    eng_env = {
        "cnn": {"MRZ_OCR_CRNN": "0", "MRZ_OCR_FUSE": "0"},
        "crnn": {"MRZ_OCR_CRNN": "1", "MRZ_OCR_FUSE": "0"},
        "fuse": {"MRZ_OCR_CRNN": "0", "MRZ_OCR_FUSE": "1"},
    }

    widths = [int(w) for w in args.widths.split(",") if w]
    print(f"{'engine':6s} {'width':>6s} {'exact':>8s} "
          f"{'l1acc':>7s} {'l2acc':>7s}  per-sample exact")
    for eng in args.engines.split(","):
        eng = eng.strip()
        if not eng:
            continue
        for w in widths:
            ex = 0
            n = 0
            c1 = c2 = 0
            tot = 0
            ok_samples = []
            for band in bands:
                stem = band.stem
                if stem not in gmap:
                    continue
                im = Image.open(band)
                r = w / im.size[0]
                nh = max(1, int(im.size[1] * r))
                dst = tmp / f"{stem}_{w}.ppm"
                im.resize((w, nh), Image.LANCZOS).save(dst)
                kv = read(dst, eng_env[eng])
                g1, g2 = gmap[stem]
                l1 = kv.get("result.line1", "")
                l2 = kv.get("result.line2", "")
                n += 1
                tot += 88
                c1 += sum(1 for a, b in zip(g1, l1) if a == b)
                c2 += sum(1 for a, b in zip(g2, l2) if a == b)
                hit = (l1 == g1 and l2 == g2)
                ex += int(hit)
                if hit:
                    ok_samples.append(stem)
            print(f"{eng:6s} {w:6d} {ex:3d}/{n:3d} "
                  f"{c1/tot*100:6.2f}% {c2/tot*100:6.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
