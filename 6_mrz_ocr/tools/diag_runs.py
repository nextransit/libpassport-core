#!/usr/bin/env python3
"""Diagnose the CHARACTER RUN structure of an MRZ line on the real path.

Reproduces the C side's `find_runs()` exactly (column projection, thr =
H/12, contiguous dark columns form one run, centre = (start+end)/2) and
prints, for every run:

    index, start column, end column, WIDTH, centre, and the gap to the
    next run

The question this answers: does every character own exactly ONE run, and
do the runs stay inside their own pitch-wide cell?  A run that is much
wider than the pitch, or a gap of 0 between two runs, means two glyphs
share a cell -- which is what makes a 'C' read as an 'E'.

Usage:
  python3 tools/diag_runs.py --file Photo/3/IMG_1792133000_003.jpg
"""
from __future__ import annotations
import argparse, importlib.util, json, os, re, subprocess, sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path("/Users/zhouyong/Desktop/work/Decard/gitlab/passport")
sys.path.insert(0, str(ROOT / "passport_photo_restore" / "src"))


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


def find_runs(gray, y0, h):
    """Exact reproduction of src/mrz_ocr_cnn.c find_runs()."""
    thr = max(1, h // 12)
    col = np.zeros(gray.shape[1], dtype=int)
    for x in range(gray.shape[1]):
        c = 0
        for y in range(y0, y0 + h):
            if y < 0 or y >= gray.shape[0]:
                continue
            if gray[y, x] < thr:
                c += 1
        col[x] = c
    runs = []
    s = -1
    for x in range(len(col)):
        if col[x] > 0:
            if s < 0:
                s = x
        else:
            if s >= 0:
                runs.append((s, x - 1))
                s = -1
    if s >= 0:
        runs.append((s, len(col) - 1))
    return runs, col, thr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True, help="substring of the photo label")
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
        ppm = Path("/tmp/_diag_band.ppm")
        Image.fromarray(band).save(str(ppm), format="PPM")
        env = dict(os.environ)
        env.update({"MRZ_OCR_LINE_MODELS": "1", "MRZ_OCR_GRID": "1",
                    "MRZ_OCR_LINES": "1", "MRZ_OCR_CRNN": "0", "MRZ_OCR_FUSE": "0"})
        r = subprocess.run([args.tool, str(ppm)], capture_output=True, text=True, env=env)
        grids = []
        boxes = []
        by = 0
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
        print("=" * 92)
        print(f"{label}   band h={gray.shape[0]} w={gray.shape[1]}  band.y={by}")
        if gt:
            print(f"  GT l1 = {gt.get('line1','')}")
            print(f"  GT l2 = {gt.get('line2','')}")
        for li, d in enumerate(grids):
            if li >= len(boxes):
                break
            c0 = float(d["c0"]); L = float(d["L"]); H = int(d["H"])
            lr = int(d.get("lr", 0))
            y0 = boxes[li][0] + by
            runs, col, thr = find_runs(gray, y0, H)
            print(f"  --- line{li}: c0={c0:.1f} L={L:.2f} H={H} lr={lr} "
                  f"runs={len(runs)} thr={thr}")
            if len(runs) != 44:
                print(f"      !!! run count {len(runs)} != 44 characters")
            for i, (s, e) in enumerate(runs):
                w = e - s + 1
                c = (s + e) / 2.0
                gap = (runs[i + 1][0] - e - 1) if i + 1 < len(runs) else -1
                # which character cell does this run centre land in?
                ch = int(round((c - c0) / L)) if L > 0 else -1
                flag = ""
                if w > L * 1.25:
                    flag += " WIDE"
                if gap == 0:
                    flag += " TOUCH"
                if 0 <= ch < 44:
                    # distance from the run centre to its own cell centre
                    dcell = c - (c0 + ch * L)
                    if abs(dcell) > L * 0.35:
                        flag += f" OFFCELL({dcell:+.1f})"
                else:
                    flag += " OUT"
                if flag or i < 3:
                    print(f"      run{i:2d} s={s:4d} e={e:4d} w={w:3d} "
                          f"c={c:7.1f} gap={gap:3d} cell={ch:3d}{flag}")


if __name__ == "__main__":
    main()
