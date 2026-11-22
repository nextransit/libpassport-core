#!/usr/bin/env python3
"""Re-slice the real product band at DIFFERENT cell widths and score the
result with the existing CNN weights, offline.

This is the decisive experiment for the cell-width question.  The C side
picks the resampling window from `medw + 8` (the median ink-RUN width plus
padding).  On a photo whose glyphs touch, several runs merge, the median
inflates, and the window ends up wider than one character -- so each 16x12
slice carries a character plus a slice of its neighbour.

Here we cut the band ourselves at a range of window widths around the
fitted grid centre and report, for each width, the character accuracy of
the CURRENT production weights.  If some width makes Photo/3-8 jump, the
window width is the bug and the fix belongs in the C side.

Usage:
  python3 tools/slice_width_probe.py --file IMG_1792133000_003 --line 1
"""
from __future__ import annotations
import argparse, importlib.util, json, os, re, subprocess, sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path("/Users/zhouyong/Desktop/work/Decard/gitlab/passport")
sys.path.insert(0, str(ROOT / "passport_photo_restore" / "src"))
sys.path.insert(0, str(ROOT / "6_mrz_ocr" / "tools"))
import cnn_offline_eval as off

ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ<"
GI = {c: i for i, c in enumerate(ALPHABET)}
TOOL = ROOT / "6_mrz_ocr/build/mrz_ocr_cnn_tool"


def load_gui():
    spec = importlib.util.spec_from_file_location("gui", str(ROOT / "tests" / "gui.py"))
    gui = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gui)
    gui.PassportGUI.AUTO_WARM_ENGINES = False
    app = gui.PassportGUI(); app.update(); app.withdraw()
    return gui, app


def band_for(app, path):
    import cv2
    from restore import load, _gray, detect_structure, drop_reflection
    rgb = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
    work, scale = load(rgb, max_side=1250)
    g = _gray(work); st = detect_structure(g)
    _w2, st2, refl = drop_reflection(work, st)
    pair = st2.get("mrz_pair")
    sb = None
    if isinstance(refl, dict) and refl.get("crop_bottom"):
        sb = int(round(refl["crop_bottom"] / scale))
    if pair is None:
        return None
    band, _c = app._rst_native_mrz_band(rgb, pair, scale, source_bottom=sb)
    return band


def grid_of(ppm):
    env = dict(os.environ)
    env.update({"MRZ_OCR_LINE_MODELS": "1", "MRZ_OCR_GRID": "1",
                "MRZ_OCR_LINES": "1", "MRZ_OCR_CRNN": "0", "MRZ_OCR_FUSE": "0"})
    r = subprocess.run([str(TOOL), str(ppm)], capture_output=True, text=True, env=env)
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
    return grids, boxes, by


def resample(gray, rx, ry, rw, rh):
    """Mirror resample_char_gray: area-average -> 3x3 smooth -> x centroid."""
    out = np.zeros((12, 16), dtype=np.float32)
    H, W = gray.shape
    for oy in range(12):
        sy0 = (oy * rh) // 12
        sy1 = max(sy0 + 1, ((oy + 1) * rh) // 12)
        for ox in range(16):
            sx0 = (ox * rw) // 16
            sx1 = max(sx0 + 1, ((ox + 1) * rw) // 16)
            s = 0.0; n = 0
            for sy in range(sy0, sy1):
                yy = ry + sy
                if yy < 0 or yy >= H:
                    continue
                for sx in range(sx0, sx1):
                    xx = rx + sx
                    if xx < 0 or xx >= W:
                        continue
                    s += gray[yy, xx]; n += 1
            ink = s / n if n else 0.0
            out[oy, ox] = 1.0 - ink / 255.0
    tmp = out.copy()
    for y in range(12):
        for x in range(16):
            acc = 0.0; c = 0
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < 12 and 0 <= nx < 16:
                        acc += tmp[ny, nx]; c += 1
            out[y, x] = acc / c
    # x centroid alignment, clamped +-2
    xs = np.arange(16)
    v = out
    m = v > 0.35
    if m.sum() > 0:
        cx = float((xs[None, :] * v).sum() / v.sum())
        delta = int(round(cx - 7.5))
        delta = max(-2, min(2, delta))
        if delta:
            out = np.roll(out, -delta, axis=1)
            if delta > 0:
                out[:, :delta] = 0
            else:
                out[:, delta:] = 0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--line", type=int, default=None)
    ap.add_argument("--bands", default="/tmp/product_export")
    ap.add_argument("--w1", default=str(ROOT / "6_mrz_ocr/src/cnn_weights_line1.h"))
    ap.add_argument("--w2", default=str(ROOT / "6_mrz_ocr/src/cnn_weights_line2.h"))
    ap.add_argument("--gt", default=str(ROOT / "10_ocr_research/photo_direct_bench.json"))
    ap.add_argument("--widths", default="")
    args = ap.parse_args()

    gts = {}
    for r in json.load(open(args.gt))["records"]:
        if r.get("gt"):
            gts[Path(r["path"]).name] = r["gt"]

    a1 = off.parse_header(args.w1); p1 = off.pick(a1, off.infer_prefix(a1))
    a2 = off.parse_header(args.w2); p2 = off.pick(a2, off.infer_prefix(a2))

    for p in sorted(Path(args.bands).glob("*.ppm")):
        if args.file not in p.stem:
            continue
        gt = gts.get(p.stem + ".jpg")
        if not gt:
            continue
        grids, boxes, by = grid_of(p)
        gray = np.asarray(Image.open(p)).astype(np.float32).mean(axis=2)
        for li, d in enumerate(grids):
            if args.line is not None and li != args.line:
                continue
            c0 = float(d["c0"]); L = float(d["L"]); H = int(d["H"])
            cell = int(d.get("cell", 0))
            y0 = boxes[li][0] + by
            truth = gt["line1"] if li == 0 else gt["line2"]
            P = p1 if li == 0 else p2
            widths = ([int(x) for x in args.widths.split(",")] if args.widths
                      else [cell - 12, cell - 8, cell - 6, cell - 4, cell - 2,
                            cell, cell + 2, cell + 4, cell + 6, cell + 8,
                            cell + 12, int(L), int(L) - 2, int(L) + 2])
            print("=" * 78)
            print(f"{p.stem} line{li}  c0={c0:.1f} L={L:.2f} cell={cell} "
                  f"H={H} y0={y0}")
            for w in sorted(set(widths)):
                if w < 6:
                    continue
                Xs, Ys = [], []
                for c in range(44):
                    ch = truth[c]
                    if ch not in GI:
                        continue
                    cx = c0 + c * L
                    x0 = int(round(cx - w / 2.0))
                    g = resample(gray, x0, y0, w, H)
                    Xs.append(g); Ys.append(GI[ch])
                if not Xs:
                    continue
                Xa = np.array(Xs, dtype=np.float32)
                Ya = np.array(Ys, dtype=np.int64)
                pred = off.forward(Xa, P).argmax(1)
                ok = int((pred == Ya).sum())
                bad = [(c, ALPHABET[Ya[i]], ALPHABET[pred[i]])
                       for i, c in enumerate(range(44)) if pred[i] != Ya[i]]
                tag = f"cell={cell}" if w == cell else (f"pitch" if abs(w - L) < 1 else "")
                print(f"  w={w:3d} {tag:8s} acc={ok:2d}/44  "
                      + " ".join(f"{g}->{p}" for _, g, p in bad[:8]))


if __name__ == "__main__":
    main()
