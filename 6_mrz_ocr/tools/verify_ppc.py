#!/usr/bin/env python3
"""End-to-end check of the C-side scale normalisation.

Feeds the ORIGINAL crops (no Python-side resize) so the upscaling under test is
the one inside mrz_ocr_recognise_cnn.  MRZ_OCR_PPC=26 is the new default;
MRZ_OCR_PPC=8 forces the factor to 1.0 and reproduces the old behaviour, which
makes the A/B exact.
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


def run(ppm, ppc):
    env = dict(os.environ)
    env.update(MRZ_OCR_LINE_MODELS="1", MRZ_OCR_PAR="0", MRZ_OCR_CRNN="0",
               MRZ_OCR_PPC=str(ppc))
    q = subprocess.run([str(TOOL), str(ppm)], capture_output=True, text=True, env=env)
    kv = {}
    for ln in q.stdout.splitlines():
        if ":" in ln:
            k, _, v = ln.partition(":")
            kv[k.strip()] = v.strip()
    return kv.get("result.line2", "")


def main():
    ppc = os.environ.get("PPC", "26")
    old = "8"
    print(f"MRZ_OCR_PPC={ppc} (new) vs {old} (factor clamped to 1.0 = old behaviour)\n")

    recs = json.load(open(CORPUS / "corpus.json"))["records"]
    print("=== PassPortPhoto (objective line-2 GT) ===")
    for tag, p in (("old", old), ("new", ppc)):
        ex = ch = 0.0
        for r in recs:
            o = run(CORPUS / r["file"], p)
            ex += (o == r["line2"])
            ch += sum(1 for i in range(44) if len(o) == 44 and o[i] == r["line2"][i]) / 44.0
        print(f"  {tag}: exact {int(ex)}/{len(recs)}  char {ch / len(recs) * 100:.2f}%")

    cat = json.load(open(ASSETS / "catalog.json"))
    print("\n=== assets/pic (checksum pass rate) ===")
    for tag, p in (("old", old), ("new", ppc)):
        n = hit = 0
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
            ppm = Path("/tmp") / f"v_{img.stem}.ppm"
            cv2.imwrite(str(ppm), band)
            n += 1
            hit += l2_ok(run(ppm, p))
        print(f"  {tag}: {hit}/{n} checksum-valid ({hit / n * 100:.1f}%)")


if __name__ == "__main__":
    main()
