#!/usr/bin/env python3
"""Which single px-per-character target is best? (one-line-change candidate)

The cascade result (49/84) came from pooling 16/20/26/32/38.  If one fixed
target gets close, the whole gain is a single resize instead of a re-decode
loop -- far lower risk.  Sweeps the target and reports, for each corpus, the
checksum pass rate (assets) or exact/char (passport, which has GT).
"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
TOOL = ROOT / "build" / "mrz_ocr_cnn_tool"
ASSETS = Path("/Users/zhouyong/gitlab/passport/assets/pic")
CORPUS = ROOT / "data" / "corpus_passport_photo"

W = (7, 3, 1)
TARGETS = (20, 24, 26, 28, 30, 32, 36, 40, 48, 56)


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


def run(ppm, crnn=0):
    env = dict(os.environ)
    env.update(MRZ_OCR_LINE_MODELS="1", MRZ_OCR_PAR="0", MRZ_OCR_CRNN=str(crnn))
    q = subprocess.run([str(TOOL), str(ppm)], capture_output=True, text=True, env=env)
    kv = {}
    for ln in q.stdout.splitlines():
        if ":" in ln:
            k, _, v = ln.partition(":")
            kv[k.strip()] = v.strip()
    return kv.get("result.line2", "")


def load():
    items = {"passport": [], "assets": []}
    for r in json.load(open(CORPUS / "corpus.json"))["records"]:
        b = cv2.imread(str(CORPUS / r["file"]))
        if b is not None:
            items["passport"].append((Path(r["file"]).stem, b, r["line2"]))
    for r in json.load(open(ASSETS / "catalog.json"))["records"]:
        loc = (r.get("loc") or {}).get("mrz")
        p = ASSETS / r["file"]
        if not loc or not p.exists():
            continue
        rgb = cv2.imread(str(p))
        if rgb is None:
            continue
        H, Wd = rgb.shape[:2]
        pad = 12
        b = rgb[max(0, loc["y"] - pad):min(H, loc["y"] + loc["h"] + pad),
                max(0, loc["x"]):min(Wd, loc["x"] + loc["w"])]
        if b.shape[0] >= 16 and b.shape[1] >= 44:
            items["assets"].append((p.stem, b, None))
    return items


def main():
    tmp = Path(tempfile.mkdtemp())
    items = load()
    for src in ("assets", "passport"):
        lst = items[src]
        print(f"\n=== {src} n={len(lst)} ===")
        print(f"{'target':>7} {'median s':>9} {'cksum/exact':>12} {'char':>8}")
        for t in TARGETS:
            hit = 0
            ch = 0.0
            ss = []
            for stem, band, gt in lst:
                ppc = band.shape[1] / 44.0
                s = max(1.0, t / ppc)
                ss.append(s)
                img = band if abs(s - 1.0) < 1e-6 else cv2.resize(
                    band, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
                p = tmp / f"{stem}_{t}.ppm"
                Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).save(str(p), format="PPM")
                o = run(p)
                if gt is None:
                    hit += l2_ok(o)
                else:
                    hit += (o == gt)
                    ch += sum(1 for i in range(44) if len(o) == 44 and o[i] == gt[i]) / 44.0
            ss.sort()
            med = ss[len(ss) // 2]
            if src == "assets":
                print(f"{t:>7} {med:>9.2f} {hit:>7}/{len(lst):<4} {'':>8}")
            else:
                print(f"{t:>7} {med:>9.2f} {hit:>7}/{len(lst):<4} {ch / len(lst) * 100:>7.2f}%")


if __name__ == "__main__":
    main()
