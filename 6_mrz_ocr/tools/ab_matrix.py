#!/usr/bin/env python3
"""2x2 A/B: band upscaling (MRZ_OCR_PPC) x proportional glyph window (MRZ_OCR_CELL).

  PPC=8            -> upscaling disabled (factor clamped to 1.0)
  PPC=26           -> scale normalisation on (the section 13.3 change)
  CELL=prop        -> sample medw*4/3 of the cell instead of the whole cell
  CELL unset       -> legacy (whole cell, ratio 0.50 at 19 px/char)

The point of the matrix is to see whether the proportional window makes the
upscaling redundant: if the ratio is what matters, fixing it directly should
recover most of the gain without any resampling.
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


def run(ppm, ppc, cell):
    env = dict(os.environ)
    env.update(MRZ_OCR_LINE_MODELS="1", MRZ_OCR_PAR="0", MRZ_OCR_CRNN="0",
               MRZ_OCR_PPC=str(ppc))
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
    combos = [("legacy", 8, None), ("pitch", 8, "pitch"),
              ("pitch090", 8, "pitch090"), ("upscale", 26, None),
              ("up+pitch", 26, "pitch"), ("up+p090", 26, "pitch090")]

    recs = json.load(open(CORPUS / "corpus.json"))["records"]
    print(f"{'combo':10} {'PPC':>4} {'CELL':>6} | {'passport exact':>15} {'char':>8} | {'assets cksum':>13}")
    for name, ppc, cell in combos:
        ex = 0
        ch = 0.0
        for r in recs:
            o = run(CORPUS / r["file"], ppc, cell)
            ex += (o == r["line2"])
            ch += sum(1 for i in range(44) if len(o) == 44 and o[i] == r["line2"][i]) / 44.0
        n_p = len(recs)

        hit = n_a = 0
        cat = json.load(open(ASSETS / "catalog.json"))
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
            band = rgb[max(0, loc["y"] - pad):min(H, loc["y"] + loc["h"] + pad),
                       max(0, loc["x"]):min(Wd, loc["x"] + loc["w"])]
            if band.shape[0] < 16 or band.shape[1] < 44:
                continue
            ppm = Path("/tmp") / f"m_{img.stem}.ppm"
            cv2.imwrite(str(ppm), band)
            n_a += 1
            hit += l2_ok(run(ppm, ppc, cell))

        print(f"{name:10} {ppc:>4} {str(cell or '-'):>6} | "
              f"{ex:>7}/{n_p:<7} {ch / n_p * 100:>7.2f}% | "
              f"{hit:>5}/{n_a:<7} ({hit / n_a * 100:>5.1f}%)")


if __name__ == "__main__":
    main()
