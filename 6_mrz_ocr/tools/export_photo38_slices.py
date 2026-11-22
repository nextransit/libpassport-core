#!/usr/bin/env python3
"""Export Photo/3-8 glyph slices with GT from gt_photo_camshots.json.

This is a modified version of export_holdout_slices.py that uses
gt_photo_camshots.json (which contains Photo/3-8 GT) instead of
photo_direct_bench.json (which only has PassPortPhoto GT).

Output: npz with X (N,12,16), Y (N,), meta (N,) json strings.
"""
from __future__ import annotations
import argparse, glob, importlib.util, json, os, subprocess, sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path("/Users/zhouyong/Desktop/work/Decard/gitlab/passport")
sys.path.insert(0, str(ROOT / "passport_photo_restore" / "src"))
ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ<"
GI = {c: i for i, c in enumerate(ALPHABET)}


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
    ap.add_argument("--tool", default=str(ROOT / "6_mrz_ocr" / "build" / "mrz_ocr_cnn_tool"))
    ap.add_argument("--gt", default=str(ROOT / "tests" / "gt_photo_camshots.json"))
    ap.add_argument("--out", default="/tmp/photo38_slices.npz")
    ap.add_argument("--only", default="Photo/", help="substring filter on the label")
    ap.add_argument("--dump-prefix", default="/tmp/mrz_dump")
    ap.add_argument("--band-dir", default="/tmp/photo38_export")
    args = ap.parse_args()

    gui, app = load_gui()

    # Load GT from gt_photo_camshots.json
    gt_data = json.load(open(args.gt))
    gts = {}
    for fname, info in gt_data.get("gt", {}).items():
        gts[fname] = {"line1": info["l1"], "line2": info["l2"]}
    print(f"Loaded {len(gts)} GT entries from {args.gt}")

    slices_x = []
    slices_y = []
    slices_meta = []
    out_dir = Path(args.band_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for label, path in gui.iter_passport_photos():
        if args.only and args.only not in label:
            continue
        gt = gts.get(Path(path).name)
        if not gt:
            print(f"  skip {label}: no GT")
            continue
        l1, l2 = gt.get("line1", ""), gt.get("line2", "")
        if len(l1) != 44 or len(l2) != 44:
            print(f"  skip {label}: GT not 44/44")
            continue
        try:
            band = band_for(app, path)
        except Exception as exc:
            print(f"  skip {label}: band failed {type(exc).__name__}: {exc}")
            continue
        if band is None:
            print(f"  skip {label}: no band")
            continue
        ppm = out_dir / (Path(path).stem + ".ppm")
        Image.fromarray(band).save(str(ppm), format="PPM")

        for f in glob.glob(f"{args.dump_prefix}_*"):
            try:
                os.remove(f)
            except OSError:
                pass
        env = dict(os.environ)
        env["MRZ_OCR_DUMP"] = "holdout"
        env["MRZ_OCR_LINE_MODELS"] = "1"
        env["MRZ_OCR_CRNN"] = "0"
        env["MRZ_OCR_FUSE"] = "0"
        r = subprocess.run([args.tool, str(ppm)], capture_output=True, text=True,
                           env=env, timeout=120)
        if r.returncode != 0:
            print(f"  skip {label}: tool rc={r.returncode}")
            continue

        n = 0
        for li, truth in ((0, l1), (1, l2)):
            for c, ch in enumerate(truth):
                if ch not in GI:
                    continue
                fs = glob.glob(f"{args.dump_prefix}_holdout_l{li}_c{c:02d}_x*_r*_w*.ppm")
                if not fs:
                    continue
                patch = np.asarray(Image.open(fs[0]).convert("L"),
                                   dtype=np.float32) / 255.0
                slices_x.append(patch)
                slices_y.append(GI[ch])
                slices_meta.append({"file": Path(path).name, "line": li,
                                    "col": c, "ch": ch, "label": label})
                n += 1
        print(f"  {label}: {n} slices", flush=True)

    if not slices_x:
        raise SystemExit("no slices exported")
    np.savez(args.out,
             X=np.array(slices_x, dtype=np.float32).reshape(-1, 12, 16),
             Y=np.array(slices_y, dtype=np.int64),
             meta=np.array([json.dumps(m) for m in slices_meta]))
    print(f"wrote {args.out}: {len(slices_x)} slices from "
          f"{len({m['file'] for m in slices_meta})} photos")


if __name__ == "__main__":
    main()
