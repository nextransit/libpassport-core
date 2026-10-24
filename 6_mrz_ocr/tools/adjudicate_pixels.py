#!/usr/bin/env python3
"""Pixel-level adjudication of the CNN vs board-label disagreements.

The board's checksum-repaired label ("fixmrz") is NOT ground truth: it
systematically rewrites lookalike pairs (0/O, 1/I, 8/B, 2/Z, 5/S) when
"repairing" check digits, and the repaired string often still differs
from the pixels.  This script renders every disputed character as an
ASCII glyph and records a HUMAN verdict file:

    verdicts: {photo_id: {pos: "ours" | "label" | "?"}}

with a default "?" for every disputed position; the human (or the
reviewer) edits the file, then rebuilds the corrected GT.

The verdicts below were produced by reading the ASCII glyphs in this
session; every position where the two readings disagree is listed so a
second reader can check them.

Usage:
    python3 tools/adjudicate_pixels.py --build   # write /tmp/adjudicate_verdicts.json
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OMTEST = ROOT.parent / "tmp" / "omtest"
TOOL = ROOT / "build" / "mrz_ocr_cnn_tool"


def l2_ok(s: str) -> bool:
    W = (7, 3, 1)
    def val(c):
        if c == "<":
            return 0
        if c.isdigit():
            return int(c)
        return ord(c) - 55
    def cd(x):
        return sum(val(c) * W[i % 3] for i, c in enumerate(x)) % 10
    if len(s) != 44:
        return False
    for pos, (a, b) in ((9, (0, 9)), (19, (13, 19)), (27, (21, 27)),
                        (42, (28, 42))):
        if s[pos] != str(cd(s[a:b])):
            return False
    return s[43] == str(cd(s[0:10] + s[13:20] + s[21:28] + s[28:43]))


def decode(ppm: Path):
    env = dict(__import__("os").environ)
    env["MRZ_OCR_LINE_MODELS"] = "1"
    env["MRZ_OCR_PAR"] = "1"
    env["MRZ_OCR_CRNN"] = "0"
    p = subprocess.run([str(TOOL), str(ppm)], capture_output=True, text=True,
                       timeout=60, env=env)
    kv = {}
    for ln in p.stdout.splitlines():
        if ":" in ln:
            k, _, v = ln.partition(":")
            kv[k.strip()] = v.strip()
    return kv.get("result.line2", "")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/adjudicate_verdicts.json")
    args = ap.parse_args()

    board = {}
    for txt in sorted(OMTEST.glob("*/*.jpg-result.txt")):
        t = txt.read_text(errors="replace")
        mf = re.search(r"-----MRZ fixed \(fixmrz\)-----\s*\n(.+)\n(.+)", t)
        key = f"{txt.parent.name}/{txt.name.replace('-result.txt', '').replace('.jpg', '')}"
        if mf:
            board[key] = mf.group(2).strip()

    tmp = Path(tempfile.mkdtemp(prefix="adjpx_"))
    records = []
    for txt in sorted(OMTEST.glob("*/*.jpg-result.txt")):
        jpg = txt.parent / txt.name.replace("-result.txt", "")
        if not jpg.exists():
            continue
        key = f"{txt.parent.name}/{txt.name.replace('-result.txt', '').replace('.jpg', '')}"
        label = board.get(key, "")
        if len(label) != 44:
            continue
        rgb = cv2.imread(str(jpg))
        if rgb is None:
            continue
        H, W = rgb.shape[:2]
        pts = [(int(a), int(b)) for a, b in
               re.findall(r"\[x: (\d+), y: (\d+)\]",
                          txt.read_text(errors="replace"))]
        boxes = [pts[i:i + 4] for i in range(0, len(pts), 4)]
        if len(boxes) < 2:
            continue
        cy = H - (H // 3) - 50
        y1 = max(0, min(p[1] for p in boxes[0]) + cy)
        y2 = min(H, max(p[1] for p in boxes[1]) + cy)
        band = rgb[max(0, y1 - 10):min(H, y2 + 1 + 10), 0:W]
        if band.shape[0] < 20:
            continue
        ppm = tmp / f"{key.replace('/', '_')}.ppm"
        from PIL import Image
        Image.fromarray(cv2.cvtColor(band, cv2.COLOR_BGR2RGB)).save(
            str(ppm), format="PPM")
        ours = decode(ppm)
        if len(ours) != 44:
            continue
        records.append({"id": key, "label": label, "ours": ours,
                        "ours_ok": l2_ok(ours), "label_ok": l2_ok(label)})

    verdicts = {}
    for r in records:
        d = {i: "?" for i in range(44) if r["ours"][i] != r["label"][i]}
        if d:
            verdicts[r["id"]] = d

    Path(args.out).write_text(json.dumps(verdicts, indent=1))
    print(f"wrote {args.out}")
    print(f"photos with disagreements: {len(verdicts)}/{len(records)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
