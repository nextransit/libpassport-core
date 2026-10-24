#!/usr/bin/env python3
"""Find the vertical offset that turns result-txt TextBox coords into pixels.

The old pipeline logs two TextBox rectangles in a coordinate frame that is
NOT the photo frame: the image was cropped to the bottom third
(crop_y = H - H//3 - 50) and then padded by 50 px on every side for DBNet
(ScaleParam sw = W+100).  Two sign/offset conventions are plausible
(-50 or 0), and guessing wrong moves the band by half a line height,
which silently destroys the decode.

Rather than reason about it, crop under every candidate offset and keep
the one where the decoder actually returns two 44-char rows.
"""
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


def decode(ppm, crnn=0):
    env = dict(os.environ)
    env["MRZ_OCR_LINE_MODELS"] = "1"
    env["MRZ_OCR_PAR"] = "0"
    env["MRZ_OCR_CRNN"] = str(crnn)
    p = subprocess.run([str(TOOL), str(ppm)], capture_output=True,
                       text=True, timeout=120, env=env)
    kv = {}
    for ln in p.stdout.splitlines():
        if ":" in ln:
            k, _, v = ln.partition(":")
            kv[k.strip()] = v.strip()
    return kv.get("result.line1", ""), kv.get("result.line2", "")


def main():
    offs = [-50, -25, 0, 25, 50]
    tmp = Path(tempfile.mkdtemp(prefix="offprobe_"))
    tally = {o: {"two44": 0, "one": 0, "n": 0} for o in offs}
    for txt in sorted(OMTEST.glob("*/*.jpg-result.txt"))[:12]:
        img = txt.with_name(txt.name.replace("-result.txt", ""))
        if not img.exists():
            continue
        s = txt.read_text(errors="replace")
        pts = [(int(a), int(b)) for a, b in
               re.findall(r"\[x: (\d+), y: (\d+)\]", s)]
        boxes = [pts[i:i + 4] for i in range(0, len(pts), 4)]
        if len(boxes) < 2:
            continue
        rgb = cv2.imread(str(img))
        H, W = rgb.shape[:2]
        crop_y = H - (H // 3) - 50
        for o in offs:
            y1 = min(p[1] for p in boxes[0]) + crop_y + o - 4
            y2 = max(p[1] for p in boxes[1]) + crop_y + o + 5
            band = rgb[max(0, y1):min(H, y2), :]
            if band.shape[0] < 20:
                continue
            ppm = tmp / f"{img.stem}_{o}.ppm"
            Image.fromarray(cv2.cvtColor(band, cv2.COLOR_BGR2RGB)).save(
                str(ppm), format="PPM")
            l1, l2 = decode(ppm)
            tally[o]["n"] += 1
            if len(l1) == 44 and len(l2) == 44:
                tally[o]["two44"] += 1
            elif len(l1) or len(l2):
                tally[o]["one"] += 1
    print(f"{'offset':>7} {'n':>4} {'two44':>6} {'partial':>8}")
    for o in offs:
        t = tally[o]
        print(f"{o:>7} {t['n']:>4} {t['two44']:>6} {t['one']:>8}")


if __name__ == "__main__":
    sys.exit(main())
