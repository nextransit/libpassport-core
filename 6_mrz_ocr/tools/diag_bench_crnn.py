#!/usr/bin/env python3
"""Diagnose the bench CRNN row: high char accuracy but exact == 0.

The bench measures CRNN on the band it cuts from the ORIGINAL photo
(`_rst_native_mrz_band` + `_ocr_pick_band` + guilloche removal), while
the holdout export slices (`/tmp/holdout_export2/*.ppm`) were cut by a
different tool.  On the holdout slices CRNN decodes 29-L perfectly; on
the bench band it does not.  This script re-runs the bench pipeline for
one photo, saves the band it actually feeds the engine, and decodes it
with all three heads so the two paths can be diffed byte for byte.

Usage:
  python3 tools/diag_bench_crnn.py [--photo tmp/PassPortPhoto/2026-9-29-L.jpg]
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests"))

os.environ.setdefault("MRZ_OCR_LINE_MODELS", "1")
os.environ.setdefault("MRZ_OCR_PAR", "0")

import gui  # noqa: E402

TOOL = ROOT / "6_mrz_ocr" / "build" / "mrz_ocr_cnn_tool"
ENGINES = {
    "cnn": {"MRZ_OCR_CRNN": "0", "MRZ_OCR_FUSE": "0"},
    "crnn": {"MRZ_OCR_CRNN": "1", "MRZ_OCR_FUSE": "0"},
    "fuse": {"MRZ_OCR_CRNN": "0", "MRZ_OCR_FUSE": "1"},
}


def run_head(band: Path, eng: str) -> dict:
    env = dict(os.environ)
    env.update(ENGINES[eng])
    env["MRZ_OCR_REQUIRE_HEAD"] = eng
    proc = subprocess.run([str(TOOL), str(band)], text=True,
                          capture_output=True, timeout=20, env=env)
    out = {}
    for line in proc.stdout.splitlines():
        if "=" not in line:
            continue
        k, _, v = line.partition("=")
        out[k.strip()] = v.strip()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--photo",
                    default="tmp/PassPortPhoto/2026-9-29-L.jpg")
    ap.add_argument("--out", default="/tmp/bench_band")
    args = ap.parse_args()

    photo = (ROOT / args.photo).resolve()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # The GUI class needs a Tk root for some paths; the band-cutting
    # helpers do not touch it, so a bare instance is enough.
    app = gui.MRZApp.__new__(gui.MRZApp)

    from PIL import Image
    rgb = Image.open(photo).convert("RGB")
    pair, scale, work2, work3 = app._rst_find_pair(rgb) \
        if hasattr(app, "_rst_find_pair") else (None, None, None, None)
    print(f"photo={photo}")
    print(f"pair={pair} scale={scale}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
