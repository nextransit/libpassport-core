#!/usr/bin/env python3
"""Same photos, Android-like band crop: CNN vs CRNN.

The corpus_real set and bench_real.py crop the SAME 41 photos in two
different ways, and the two crops disagree about which head is better.
This script isolates the crop as the only variable:

  * crop = board TextBox bbox + 4 px vertical pad, full width
    (the geometry bench_real.py uses, and the closest host
     approximation of the Android nativeBand path)
  * both heads run the identical C front-end, only the head differs
  * seg score, line text, ambiguity and checksum are recorded per head

Usage:
  /opt/homebrew/bin/python3 tools/cmp_crop_crnn.py [--limit N]
"""
from __future__ import annotations
import argparse
import json
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

W = [7, 3, 1]


def check_digit(seg: str) -> int:
    t = 0
    for i, c in enumerate(seg):
        v = (ord(c) - 48) if '0' <= c <= '9' else ((ord(c) - 55) if 'A' <= c <= 'Z' else 0)
        t += v * W[i % 3]
    return t % 10


def l2_ok(l2: str) -> bool:
    if len(l2) != 44:
        return False
    try:
        return (check_digit(l2[0:9]) == int(l2[9]) and
                check_digit(l2[13:19]) == int(l2[19]) and
                check_digit(l2[21:27]) == int(l2[27]) and
                check_digit(l2[28:42]) == int(l2[42]) and
                check_digit(l2[0:10] + l2[13:20] + l2[21:43]) == int(l2[43]))
    except ValueError:
        return False


def parse_result(txt: Path):
    s = txt.read_text(errors="replace")
    tl = re.findall(r'textLine\[\d\]\(([^\n]*)\)', s)
    mfix = re.findall(r'fixmrz[^\n]*\n([PE]<[0-9A-Z<]{43})\n([0-9A-Z<]{44})', s)
    l1, l2 = (mfix[0] if mfix else (tl[0] if tl else None, tl[1] if len(tl) > 1 else None))
    pts = [(int(a), int(b)) for a, b in re.findall(r'\[x: (\d+), y: (\d+)\]', s)]
    boxes = [pts[i:i + 4] for i in range(0, len(pts), 4)]
    return l1, l2, boxes


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
    scores = [int(x) for x in re.findall(r"score=(\d+)", p.stderr)]
    return (kv.get("result.line1", ""), kv.get("result.line2", ""),
            min(scores) if scores else -1, kv.get("result.conf2", ""))


def carr(a: str, b: str) -> float:
    n = max(len(a), len(b))
    if not n:
        return 0.0
    c = sum(1 for i in range(n)
            if (a[i] if i < len(a) else "_") == (b[i] if i < len(b) else "_"))
    return c / n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    tmp = Path(tempfile.mkdtemp(prefix="mrzband_"))
    rows = []
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
            crop_y = H - (H // 3) - 50
            y1 = min(p[1] for p in boxes[0]) + crop_y
            y2b = max(p[1] for p in boxes[1]) + crop_y
            band = rgb[max(0, y1 - 4):y2b + 5, 0:Wd]
            ppm = tmp / (d.name + "_" + img.stem + ".ppm")
            Image.fromarray(cv2.cvtColor(band, cv2.COLOR_BGR2RGB)).save(str(ppm), format="PPM")
            c1, c2, segc, cf2c = run(ppm, 0)
            r1, r2, segr, cf2r = run(ppm, 1)
            rows.append({
                "id": f"{d.name}/{img.stem}", "gt1": l1, "gt2": l2,
                "band_h": band.shape[0],
                "c1": c1, "c2": c2, "segc": segc, "cf2c": cf2c,
                "r1": r1, "r2": r2, "segr": segr, "cf2r": cf2r,
            })
            if args.limit and len(rows) >= args.limit:
                break

    n = len(rows)
    print(f"n={n}  crop=board bbox +4px, full width")
    hh = sorted(r["band_h"] for r in rows)
    print(f"band height: min={hh[0]} median={hh[n//2]} max={hh[-1]}\n")

    print(f"{'':6} {'l1_acc':>8} {'l2_acc':>8} {'l1_exact':>9} {'l2_exact':>9} {'l2_cksum':>9}")
    for name, k1, k2 in (("cnn", "c1", "c2"), ("crnn", "r1", "r2")):
        a1 = sum(carr(r["gt1"], r[k1]) for r in rows) / n
        a2 = sum(carr(r["gt2"], r[k2]) for r in rows) / n
        e1 = sum(1 for r in rows if r["gt1"] == r[k1])
        e2 = sum(1 for r in rows if r["gt2"] == r[k2])
        ck = sum(1 for r in rows if l2_ok(r[k2]))
        print(f"{name:6} {a1*100:7.2f}% {a2*100:7.2f}% {e1:6d}/{n} {e2:6d}/{n} {ck:6d}/{n}")

    # head-to-head on line2
    cnn_only = [r for r in rows if r["gt2"] == r["c2"] and r["gt2"] != r["r2"]]
    crnn_only = [r for r in rows if r["gt2"] == r["r2"] and r["gt2"] != r["c2"]]
    print(f"\nline2: CNN-only correct={len(cnn_only)}  CRNN-only correct={len(crnn_only)}")
    print("\n-- photos where only CNN is right --")
    for r in cnn_only:
        print(f"[{r['id']}] h={r['band_h']} seg c/r={r['segc']}/{r['segr']} "
              f"conf2 c/r={r['cf2c']}/{r['cf2r']}")
        print(f"  gt  : {r['gt2']}")
        print(f"  crnn: {r['r2']}")
    print("\n-- photos where only CRNN is right --")
    for r in crnn_only:
        print(f"[{r['id']}] h={r['band_h']} seg c/r={r['segc']}/{r['segr']}")
        print(f"  gt  : {r['gt2']}")
        print(f"  cnn : {r['c2']}")

    # band height vs outcome
    print("\n-- band height buckets --")
    for lo, hi in ((0, 120), (121, 250), (251, 9999)):
        sub = [r for r in rows if lo <= r["band_h"] <= hi]
        if not sub:
            continue
        ec = sum(1 for r in sub if r["gt2"] == r["c2"])
        er = sum(1 for r in sub if r["gt2"] == r["r2"])
        print(f"h {lo}-{hi}: n={len(sub)} cnn_exact={ec} crnn_exact={er}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
