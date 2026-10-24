#!/usr/bin/env python3
"""Re-run assets/pic with the MRZ region from catalog.json instead of the
C-side band locator.

Why: the end-to-end sweep shows the C locator returns the WHOLE image as
the band on these document scans (band.w == image width, band.h ~= image
height), and the two-line split then fails on 37/84. That is a locator
failure, not a recogniser failure -- and it hides whether the CNN/CRNN
heads can actually read these scans.

catalog.json already carries loc.mrz (x,y,w,h) for all 84 images,
produced by the Python locator (same code path as the GUI). This script
crops with that box, pads a little, and re-runs both heads so locator
and recogniser failures are separable.

Usage:
  /opt/homebrew/bin/python3 tools/sweep_assets_crop.py [--pad 6]
"""
from __future__ import annotations
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "build" / "mrz_ocr_cnn_tool"
PIC = ROOT.parent / "assets" / "pic"
sys.path.insert(0, str(ROOT / "tools"))
from cmp_crop_crnn import l2_ok  # noqa: E402


def run(img: Path, crnn: int):
    env = dict(os.environ)
    env["MRZ_OCR_LINE_MODELS"] = "1"
    env["MRZ_OCR_PAR"] = "0"
    env["MRZ_OCR_SEGDBG"] = "1"
    env["MRZ_OCR_CRNN"] = "1" if crnn else "0"
    p = subprocess.run([str(TOOL), str(img)], capture_output=True, text=True,
                       timeout=120, env=env)
    kv = {}
    for ln in p.stdout.splitlines():
        if ":" in ln:
            k, _, v = ln.partition(":")
            kv[k.strip()] = v.strip()
    sc = [int(x) for x in re.findall(r"score=(\d+)", p.stderr)]
    return {"ok": kv.get("result.ok", "?"), "l1": kv.get("result.line1", ""),
            "l2": kv.get("result.line2", ""), "c2": kv.get("result.conf2", "0"),
            "seg": min(sc) if sc else -1,
            "band": kv.get("band.x band.y band.w band.h", "")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pad", type=int, default=6)
    ap.add_argument("--out", default="/tmp/sweep_assets_crop.json")
    args = ap.parse_args()

    cat = json.loads((PIC / "catalog.json").read_text())["records"]
    tmp = Path(tempfile.mkdtemp(prefix="asg_"))
    rows = []
    for rec in cat:
        box = (rec.get("loc") or {}).get("mrz")
        if not box:
            continue
        src = PIC / rec["file"]
        im = Image.open(src).convert("RGB")
        W, H = im.size
        x = max(0, box["x"] - args.pad)
        y = max(0, box["y"] - args.pad)
        x2 = min(W, box["x"] + box["w"] + args.pad)
        y2 = min(H, box["y"] + box["h"] + args.pad)
        crop = im.crop((x, y, x2, y2))
        ppm = tmp / (src.stem + ".ppm")
        crop.save(str(ppm), format="PPM")
        c = run(ppm, 0)
        r = run(ppm, 1)
        rows.append({"file": rec["file"], "wh": (W, H), "crop_h": crop.size[1],
                     "c": c, "r": r})
        print(f"{rec['file']:34s} {W}x{H} crop_h={crop.size[1]:4d} "
              f"cnn={c['ok']:32s} seg={c['seg']:3d} "
              f"ck_c={int(l2_ok(c['l2']))} ck_r={int(l2_ok(r['l2']))}")

    n = len(rows)
    print(f"\n== assets/pic with loc.mrz crop (pad={args.pad}) n={n}")
    for tag, key in (("cnn", "c"), ("crnn", "r")):
        ok = sum(1 for x in rows if x[key]["ok"] == "OK")
        two = sum(1 for x in rows if len(x[key]["l1"]) == 44 and len(x[key]["l2"]) == 44)
        ck = sum(1 for x in rows if l2_ok(x[key]["l2"]))
        print(f"   {tag:5s} ok={ok:2d}/{n} two_lines_44={two:2d}/{n} cksum={ck:2d}/{n}")
    agree = sum(1 for x in rows if x["c"]["l2"] and x["c"]["l2"] == x["r"]["l2"])
    print(f"   cnn==crnn line2: {agree}/{n}")
    print(f"   ok-reason cnn: {dict(Counter(x['c']['ok'] for x in rows))}")
    Path(args.out).write_text(json.dumps(rows, indent=1, ensure_ascii=False))
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
