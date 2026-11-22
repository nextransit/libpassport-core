#!/usr/bin/env python3
"""Cache the product-path glyph slices of every band, for fast weight A/B.

Why this exists
---------------
Scoring a candidate CNN weight set end to end needs a recompile plus a
full pipeline run per point.  But the pipeline's SEGMENTATION does not
depend on the weights at all: `segment_line_grid()` works on the binary
band pixels, and the 16x12 glyph crop (`resample_char_gray`) is a pure
function of the band.  Only the CLASSIFIER consumes the weights.

So the glyphs can be extracted once and reused: dump them for every
band with the C tool, store them in one npz, and then score any weight
header in seconds with `tools/cnn_offline_eval.py`'s forward pass.

What this measures
------------------
Per-cell argmax on the product-path pixels.  That is the classifier's
own opinion, before the ICAO syntax mask and the checksum-constrained
decoding the C pipeline applies afterwards, so it is a strict LOWER
BOUND on the end-to-end result: it cannot credit a weight set for a
repair the post-processor would have made anyway.  Every candidate that
wins here must still be confirmed with `tools/band_sweep.py`.

Fidelity
--------
The dump is produced by the SAME binary and the SAME input band the
product path uses, so the pixels are the ones the device classifies.
"""
from __future__ import annotations
import argparse
import glob
import json
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tool", default=str(TOOL))
    ap.add_argument("--bands", default="/tmp/holdout_export2")
    ap.add_argument("--out", default="/tmp/glyph_cache.npz")
    ap.add_argument("--dump-dir", default="/tmp/glyph_cache_dump")
    ap.add_argument("--gt", default=str(ROOT / "tests" / "photo_auto_gt.json"))
    ap.add_argument("--keep-unlabeled", action="store_true",
                    help="keep photos with no GT (labels become -1)")
    args = ap.parse_args()

    gts = {}
    for f in (ROOT / "tests" / "gt_photo_passport.json",
              Path(args.gt)):
        if not Path(f).exists():
            continue
        d = json.loads(Path(f).read_text())
        g = d.get("gt", d) if isinstance(d, dict) else d
        for k, v in g.items():
            if not isinstance(v, dict):
                continue
            l1 = v.get("l1") or v.get("line1")
            l2 = v.get("l2") or v.get("line2")
            if l1 and l2 and len(l1) == 44 and len(l2) == 44:
                gts[Path(k).stem] = (l1, l2)

    dd = Path(args.dump_dir)
    dd.mkdir(parents=True, exist_ok=True)

    X, Y, meta = [], [], []
    bands = sorted(Path(args.bands).glob("*.ppm"))
    if not bands:
        raise SystemExit(f"no bands in {args.bands}")
    for bi, band in enumerate(bands):
        stem = band.stem
        for f in glob.glob(str(dd / "mrz_dump_*")):
            try:
                os.remove(f)
            except OSError:
                pass
        env = dict(os.environ)
        env.update({
            "MRZ_OCR_DUMP": "cache",
            "MRZ_OCR_DUMP_DIR": str(dd),
            "MRZ_OCR_LINE_MODELS": "1",
            "MRZ_OCR_CRNN": "0",
            "MRZ_OCR_FUSE": "0",
            "MRZ_OCR_PAR": "0",
        })
        try:
            subprocess.run([args.tool, str(band)], capture_output=True,
                           text=True, env=env, timeout=120)
        except subprocess.TimeoutExpired:
            print(f"  {stem}: timeout", flush=True)
            continue
        gt = gts.get(stem)
        files = {}
        for p in dd.glob("mrz_dump_cache_*.ppm"):
            m = DUMP_RE.match(p.name)
            if m:
                files[(int(m.group("line")), int(m.group("col")))] = p
        if not files:
            print(f"  {stem}: no glyphs", flush=True)
            continue
        for (li, c), p in sorted(files.items()):
            patch = np.asarray(Image.open(p).convert("L"),
                               dtype=np.float32) / 255.0
            ch = ""
            if gt:
                truth = gt[li]
                ch = truth[c] if c < len(truth) else ""
            X.append(patch)
            Y.append(ch)
            meta.append({"file": stem, "line": li, "col": c, "ch": ch})
        if (bi + 1) % 10 == 0:
            print(f"  {bi + 1}/{len(bands)} bands, {len(X)} glyphs",
                  flush=True)

    if not X:
        raise SystemExit("no glyphs collected")
    np.savez(args.out,
             X=np.array(X, dtype=np.float32).reshape(-1, 12, 16),
             Y=np.array(Y, dtype=object),
             meta=np.array([json.dumps(m) for m in meta]))
    print(f"wrote {args.out}: {len(X)} glyphs from "
          f"{len({m['file'] for m in meta})} bands")


if __name__ == "__main__":
    main()
