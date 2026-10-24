#!/usr/bin/env python3
"""Sweep the window-width factor (window = f * fitted pitch).

Theory (see mrz_cell_win_factor in mrz_ocr_cnn.c): with medw/pitch = 0.67
fixed by the font, ink/window = 0.67/f, and the training band is 0.69..0.82,
so f <= 0.97; clipping needs f >= ~0.92.  The prediction is a narrow plateau
around f = 0.94 that works at EVERY resolution, which would make the
upscaling unnecessary.

Reported for each f, with and without the 26 px/char upscaling:
  passport exact/char (objective GT) and assets checksum pass rate.
"""
import json
import os
import subprocess
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
TOOL = ROOT / "build" / "mrz_ocr_cnn_tool"
ASSETS = Path("/Users/zhouyong/gitlab/passport/assets/pic")
CORPUS = ROOT / "data" / "corpus_passport_photo"

W = (7, 3, 1)


def val(c):
    return 0 if c == "<" else (ord(c) - 48 if c.isdigit() else ord(c) - 55)


def cd(s):
    return sum(val(c) * W[i % 3] for i, c in enumerate(s)) % 10


def l2_ok(l):
    if not l or len(l) != 44:
        return False
    try:
        for p, (a, b) in ((9, (0, 9)), (19, (13, 19)), (27, (21, 27)), (42, (28, 42))):
            if l[p] != str(cd(l[a:b])):
                return False
        return l[43] == str(cd(l[0:10] + l[13:20] + l[21:28] + l[28:43]))
    except Exception:
        return False


def run(ppm, ppc, win, cell=None):
    env = dict(os.environ)
    env.update(MRZ_OCR_LINE_MODELS="1", MRZ_OCR_PAR="0", MRZ_OCR_CRNN="0",
               MRZ_OCR_PPC=str(ppc))
    if win:
        env["MRZ_OCR_CELL_WIN"] = str(win)
    else:
        env.pop("MRZ_OCR_CELL_WIN", None)
    if cell:
        env["MRZ_OCR_CELL"] = cell
    else:
        env.pop("MRZ_OCR_CELL", None)
    q = subprocess.run([str(TOOL), str(ppm)], capture_output=True, text=True, env=env)
    kv = {}
    for ln in q.stdout.splitlines():
        if ":" in ln:
            k, _, v = ln.partition(":")
            kv[k.strip()] = v.strip()
    return kv.get("result.line2", "")


def main():
    recs = json.load(open(CORPUS / "corpus.json"))["records"]
    cat = json.load(open(ASSETS / "catalog.json"))
    assets = []
    for r in cat["records"]:
        loc = (r.get("loc") or {}).get("mrz")
        img = ASSETS / r["file"]
        if not loc or not img.exists():
            continue
        rgb = cv2.imread(str(img))
        if rgb is None:
            continue
        H, Wd = rgb.shape[:2]
        pad = 12
        b = rgb[max(0, loc["y"] - pad):min(H, loc["y"] + loc["h"] + pad),
                max(0, loc["x"]):min(Wd, loc["x"] + loc["w"])]
        if b.shape[0] < 16 or b.shape[1] < 44:
            continue
        ppm = Path("/tmp") / f"w_{img.stem}.ppm"
        cv2.imwrite(str(ppm), b)
        assets.append(ppm)

    print(f"{'win':>5} {'PPC':>4} | {'passport exact':>15} {'char':>8} | {'assets cksum':>16}")
    for ppc in (8, 26):
        for win in (0, 0.90, 0.92, 0.94, 0.96, 0.98, 1.00):
            ex = 0
            ch = 0.0
            for r in recs:
                o = run(CORPUS / r["file"], ppc, win)
                ex += (o == r["line2"])
                ch += sum(1 for i in range(44) if len(o) == 44 and o[i] == r["line2"][i]) / 44.0
            hit = sum(l2_ok(run(p, ppc, win)) for p in assets)
            label = "legacy" if win == 0 else f"{win:.2f}"
            print(f"{label:>5} {ppc:>4} | {ex:>7}/{len(recs):<7} {ch / len(recs) * 100:>7.2f}% | "
                  f"{hit:>5}/{len(assets):<7} ({hit / len(assets) * 100:>5.1f}%)")
        print()


if __name__ == "__main__":
    main()
