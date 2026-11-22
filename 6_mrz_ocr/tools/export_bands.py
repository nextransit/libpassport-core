#!/usr/bin/env python3
"""Export the PRODUCT-path MRZ band for every photo in one or more dirs.

Why this exists
---------------
The real-glyph augmentation pool is only as good as the real glyphs it
contains.  The 47-band cache under /tmp/holdout_export2 covers the
acceptance corpus (tmp/PassPortPhoto + Photo), but the repository also
ships 64 more real photographs under `assets/pic/`, and the automatic
GT has 108 44/44 rows in total.  Those 64 photos are real glyphs that
never reached the classifier, and the rare classes the failures live in
(line1 'J', line2 'C'/'M'/'D'/'P'/'7') are exactly the ones a bigger
real pool helps most.

Fidelity
--------
The band is taken from the SAME product call the bench uses:
`app._ocr_photo_engines(path)` -> pair / work2 / scale, then
`app._rst_native_mrz_band(rgb, pair, scale, source_bottom=...)` with the
identical `source_bottom` the bench computes.  Re-deriving the band
research-side produced a DIFFERENT crop for 11 of 41 photos, which made
the offline score disagree with the bench -- so the band must come from
the product path, not from a re-implementation.

Output: one PPM per photo in --out, named after the photo stem.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path("/Users/zhouyong/Desktop/work/Decard/gitlab/passport")
sys.path.insert(0, str(ROOT / "passport_photo_restore" / "src"))


def load_gui():
    spec = importlib.util.spec_from_file_location(
        "gui", str(ROOT / "tests" / "gui.py"))
    gui = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gui)
    gui.PassportGUI.AUTO_WARM_ENGINES = False
    app = gui.PassportGUI()
    app.update()
    app.withdraw()
    return gui, app


def band_for(app, path):
    import cv2
    rec = app._ocr_photo_engines(str(path))
    pair = rec.get("pair")
    work2 = rec.get("work2")
    scale = float(rec.get("scale") or 1.0)
    if not pair or work2 is None or scale <= 0:
        return None
    bgr = cv2.imread(str(path))
    if bgr is None:
        return None
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    band, _crop = app._rst_native_mrz_band(
        rgb, pair, scale, source_bottom=int(round(work2.shape[0] / scale)))
    return band


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True,
                    help="photo directories (top-level files)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--recursive", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    gui, app = load_gui()
    n_ok = n_skip = 0
    for d in args.dirs:
        base = Path(d)
        if not base.is_dir():
            print(f"  skip {d}: not a directory")
            continue
        pat = "**/*" if args.recursive else "*"
        for p in sorted(base.glob(pat)):
            if p.suffix.lower() not in (".jpg", ".jpeg", ".png", ".ppm"):
                continue
            try:
                band = band_for(app, p)
            except Exception as exc:
                print(f"  skip {p.name}: {type(exc).__name__}: {exc}")
                n_skip += 1
                continue
            if band is None:
                print(f"  skip {p.name}: no band")
                n_skip += 1
                continue
            band = np.asarray(band, dtype=np.uint8)
            dst = out / (p.stem + ".ppm")
            Image.fromarray(band).save(str(dst), format="PPM")
            n_ok += 1
    print(f"exported {n_ok} bands to {out} (skipped {n_skip})")


if __name__ == "__main__":
    main()
