#!/usr/bin/env python3
"""Render the RAW band pixels around a character range, with the fitted
grid overlaid, so the segmentation can be judged by eye.

The question: does every fitted cell contain exactly one glyph, or do
neighbouring glyphs bleed into each other's cell?  A 'C' that reads as
an 'E' is exactly what a 'C' with the next glyph's stem leaking in looks
like.

Usage:
  python3 tools/diag_raw_region.py --file IMG_1792133000_003 --line 1 \
      --from 40 --to 44
"""
from __future__ import annotations
import argparse, importlib.util, json, os, re, subprocess, sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path("/Users/zhouyong/Desktop/work/Decard/gitlab/passport")
sys.path.insert(0, str(ROOT / "passport_photo_restore" / "src"))
RAMP = " .:-=+*#%@"


def load_gui():
    spec = importlib.util.spec_from_file_location("gui", str(ROOT / "tests" / "gui.py"))
    gui = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gui)
    gui.PassportGUI.AUTO_WARM_ENGINES = False
    app = gui.PassportGUI()
    app.update()
    app.withdraw()
    return gui, app


def band_for(app, path):
    import cv2
    from restore import load, _gray, detect_structure, drop_reflection
    rgb = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
    work, scale = load(rgb, max_side=1250)
    g = _gray(work)
    st = detect_structure(g)
    _w2, st2, refl = drop_reflection(work, st)
    pair = st2.get("mrz_pair")
    sb = None
    if isinstance(refl, dict) and refl.get("crop_bottom"):
        sb = int(round(refl["crop_bottom"] / scale))
    if pair is None:
        return None
    band, _crop = app._rst_native_mrz_band(rgb, pair, scale, source_bottom=sb)
    return band


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--line", type=int, default=0)
    ap.add_argument("--from", dest="c0", type=int, default=0)
    ap.add_argument("--to", dest="c1", type=int, default=8)
    ap.add_argument("--tool", default=str(ROOT / "6_mrz_ocr/build/mrz_ocr_cnn_tool"))
    ap.add_argument("--gt", default=str(ROOT / "10_ocr_research/photo_direct_bench.json"))
    args = ap.parse_args()

    gui, app = load_gui()
    gts = {}
    for r in json.load(open(args.gt))["records"]:
        if r.get("gt"):
            gts[Path(r["path"]).name] = r["gt"]

    for label, path in gui.iter_passport_photos():
        if args.file not in label:
            continue
        gt = gts.get(Path(path).name)
        band = band_for(app, path)
        if band is None:
            print(f"{label}: no band")
            continue
        ppm = Path("/tmp/_raw_band.ppm")
        Image.fromarray(band).save(str(ppm), format="PPM")
        env = dict(os.environ)
        env.update({"MRZ_OCR_LINE_MODELS": "1", "MRZ_OCR_GRID": "1",
                    "MRZ_OCR_LINES": "1", "MRZ_OCR_CRNN": "0", "MRZ_OCR_FUSE": "0"})
        r = subprocess.run([args.tool, str(ppm)], capture_output=True, text=True, env=env)
        grids, boxes, by = [], [], 0
        for l in r.stderr.splitlines():
            if l.startswith("GRID W="):
                grids.append(dict(kv.split("=", 1) for kv in l.split() if "=" in kv))
            elif l.startswith("LINES band="):
                for mm in re.finditer(r"l\d=\(y(\d+) h(\d+)\)", l):
                    boxes.append((int(mm.group(1)), int(mm.group(2))))
        for l in r.stdout.splitlines():
            if l.startswith("band.x band.y"):
                by = int(l.split()[-3])
        gray = np.asarray(band).astype(np.float32).mean(axis=2)
        d = grids[args.line]
        g0 = float(d["c0"]); L = float(d["L"]); H = int(d["H"])
        cell = int(d.get("cell", 41))
        y0 = boxes[args.line][0] + by
        print("=" * 100)
        print(f"{label} line{args.line} c0={g0:.1f} L={L:.2f} cell={cell} "
              f"H={H} y0={y0} (band-relative)")
        if gt:
            t = gt["line1"] if args.line == 0 else gt["line2"]
            print(f"  GT[{args.c0}:{args.c1}] = {t[args.c0:args.c1]}")

        xa = int(round(g0 + args.c0 * L - L))
        xb = int(round(g0 + args.c1 * L + L))
        xa = max(0, xa); xb = min(gray.shape[1], xb)
        seg = gray[y0:y0 + H, xa:xb]
        # downsample rows by 2 for readability
        print(f"  raw region x[{xa}:{xb}] w={xb-xa}  (row 0 = band y {y0})")
        # column ruler: mark the fitted cell boundaries
        ruler = [" "] * (xb - xa)
        for c in range(args.c0, args.c1 + 1):
            cx = int(round(g0 + c * L))
            for d in range(-cell // 2, cell // 2 + 1):
                xi = cx - cell // 2 + d - xa
                if 0 <= xi < len(ruler):
                    ruler[xi] = "|" if d in (-cell // 2, cell // 2) else "."
        print("        " + "".join(ruler))
        for yy in range(0, seg.shape[0], 2):
            row = seg[yy]
            s = ""
            for v in row:
                k = int(min(len(RAMP) - 1, max(0, (255.0 - v) / 255.0 * (len(RAMP) - 1))))
                s += RAMP[k]
            print(f"  y{y0+yy:4d} {s}")
        # report where each fitted cell starts/ends
        print("  fitted cells:")
        for c in range(args.c0, args.c1 + 1):
            cx = g0 + c * L
            print(f"    c{c:02d}: centre={cx:8.1f} window=[{cx-cell/2:8.1f},"
                  f"{cx+cell/2:8.1f}]  GT={gt['line1'][c] if (gt and args.line==0) else (gt['line2'][c] if gt else '?')}")


if __name__ == "__main__":
    main()
