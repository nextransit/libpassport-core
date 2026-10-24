#!/usr/bin/env python3
"""Two measurements that pin down the mechanism behind the scale fix.

1. Grid-misalignment sensitivity.  The claim is that the failure mode is the
   44-cell grid landing in the wrong phase basin, not the classifier being
   weak.  If that is true, shifting the crop horizontally by a FRACTION OF ONE
   CELL (band_w/44, i.e. ~33 px on the device corpus) must hurt, even though
   the glyphs themselves are untouched and no information is lost.  A
   classifier-side limitation would be indifferent to a shift this small.

2. Latency cost of the upscaling, measured through the real entry point.
"""
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

import cv2
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
TOOL = ROOT / "build" / "mrz_ocr_cnn_tool"
CORPUS = ROOT / "data" / "corpus_passport_photo"


def run(ppm, ppc="26"):
    env = dict(os.environ)
    env.update(MRZ_OCR_LINE_MODELS="1", MRZ_OCR_PAR="0", MRZ_OCR_CRNN="0",
               MRZ_OCR_PPC=str(ppc))
    t = time.perf_counter()
    q = subprocess.run([str(TOOL), str(ppm)], capture_output=True, text=True, env=env)
    ms = (time.perf_counter() - t) * 1000.0
    kv = {}
    for ln in q.stdout.splitlines():
        if ":" in ln:
            k, _, v = ln.partition(":")
            kv[k.strip()] = v.strip()
    return kv.get("result.line2", ""), ms


def main():
    recs = json.load(open(CORPUS / "corpus.json"))["records"]
    tmp = Path(tempfile.mkdtemp())

    print("=== grid phase sensitivity (device corpus, objective GT) ===")
    print(f"{'shift':>8} {'exact':>9} {'char':>8}")
    for frac in (0.0, 0.05, 0.10, 0.20, 0.35, 0.50):
        ex = 0
        ch = 0.0
        for r in recs:
            img = cv2.imread(str(CORPUS / r["file"]))
            if img is None:
                continue
            cell = img.shape[1] / 44.0
            dx = int(round(frac * cell))
            if dx:
                # Shift the crop, not the content: pad on one side, cut the
                # other.  This is exactly what a band-locator error does.
                img = cv2.copyMakeBorder(img, 0, 0, dx, 0, cv2.BORDER_REPLICATE)
                img = img[:, :img.shape[1] - dx]
            p = tmp / f"{Path(r['file']).stem}_{frac}.ppm"
            Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).save(str(p), format="PPM")
            o, _ = run(p)
            ex += (o == r["line2"])
            ch += sum(1 for i in range(44) if len(o) == 44 and o[i] == r["line2"][i]) / 44.0
        n = len(recs)
        print(f"{frac:>7.0%} {ex:>5}/{n:<3} {ch / n * 100:>7.2f}%")

    print("\n=== latency through mrz_ocr_recognise_cnn ===")
    for label, ppc in (("factor 1.0 (old)", "8"), ("normalised (new)", "26")):
        samples = []
        for r in recs[:6]:
            src = CORPUS / r["file"]
            img = cv2.imread(str(src))
            # Downscale to ~19 px/char so the new path actually has work to do.
            s = 19.0 / (img.shape[1] / 44.0)
            small = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            p = tmp / f"lat_{Path(r['file']).stem}_{ppc}.ppm"
            Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB)).save(str(p), format="PPM")
            samples.append(min(run(p, ppc)[1] for _ in range(3)))
        print(f"  {label:18} {sum(samples) / len(samples):6.2f} ms mean of per-image best-of-3")


if __name__ == "__main__":
    main()
