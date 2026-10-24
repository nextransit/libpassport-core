#!/usr/bin/env python3
"""Smoke test: the 9 assets records that clear all three gates.

Why only 9 of 84
----------------
The assets corpus cannot measure accuracy (see the report, §14): it mixes
TD3 passports, TD1 ID cards, specimen documents and screenshots, and its
glyphs are 12-20 px/char against a training range of 26-51.  What it CAN
do is catch a regression: these nine records satisfy the five ICAO check
digits, have structurally possible dates, and carry a real nationality
code.  If a change makes any of them stop decoding, something broke.

That is a one-sided test on purpose.  Passing does not mean the reading
is correct -- the content was never checked against the physical
document.  Failing does mean a previously-readable record is now
unreadable, which is a regression regardless of what the text says.

Usage:
    python3 tools/smoke_assets.py [--expect-fail]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "build" / "mrz_ocr_cnn_tool"
ASSETS = ROOT.parent / "assets" / "pic"
MANIFEST = ROOT / "data" / "smoke_assets.json"

ENV = dict(os.environ)
ENV.update(MRZ_OCR_LINE_MODELS="1", MRZ_OCR_PAR="0", MRZ_OCR_CRNN="0")


def band_of(rec):
    """Same crop the manifest was built with: loc.mrz plus 6 px of margin."""
    cat = {r["file"]: r for r in
           json.load(open(ASSETS / "catalog.json")).get("records", [])}
    r = cat.get(rec["file"])
    if not r:
        return None
    img = cv2.imread(str(ASSETS / rec["file"]))
    if img is None:
        return None
    loc = (r.get("loc") or {}).get("mrz")
    if not loc:
        return None
    H, W = img.shape[:2]
    b = img[max(0, loc["y"] - 6):min(H, loc["y"] + loc["h"] + 6),
            max(0, loc["x"]):min(W, loc["x"] + loc["w"])]
    return b if b.shape[0] >= 16 and b.shape[1] >= 44 else None


def run(ppm: Path):
    p = subprocess.run([str(TOOL), str(ppm)], capture_output=True, text=True,
                       env=ENV, timeout=120)
    kv = {}
    for ln in p.stdout.splitlines():
        if ":" in ln:
            k, _, v = ln.partition(":")
            kv[k.strip()] = v.strip()
    return kv


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--expect-fail", action="store_true",
                    help="invert the exit code (for negative controls)")
    args = ap.parse_args()

    if not MANIFEST.exists():
        print(f"missing {MANIFEST}", file=sys.stderr)
        return 2
    recs = json.load(open(MANIFEST))["records"]
    tmp = Path(tempfile.mkdtemp(prefix="smoke_"))

    bad = 0
    for rec in recs:
        b = band_of(rec)
        if b is None:
            print(f"MISSING  {rec['file']}")
            bad += 1
            continue
        ppm = tmp / (Path(rec["file"]).stem + ".ppm")
        Image.fromarray(cv2.cvtColor(b, cv2.COLOR_BGR2RGB)).save(
            str(ppm), format="PPM")
        kv = run(ppm)
        got = kv.get("result.line2", "")
        ok = (kv.get("result.struct2") == "1"
              and kv.get("result.natsuspect") == "0"
              and got == rec["line2"])
        if ok:
            print(f"ok       {rec['file']}")
        else:
            bad += 1
            print(f"CHANGED  {rec['file']}")
            print(f"         want {rec['line2']}")
            print(f"         got  {got}  struct={kv.get('result.struct2')} "
                  f"nat={kv.get('result.natsuspect')}")

    n = len(recs)
    print(f"\n{n - bad}/{n} unchanged")
    failed = bad != 0
    if args.expect_fail:
        failed = not failed
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
