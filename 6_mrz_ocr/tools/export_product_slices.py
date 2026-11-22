#!/usr/bin/env python3
"""Export CNN glyph slices from the PRODUCT band path (tests/gui.py).

Why
---
The training corpus (data/photo_train_passport/cnn_slices.npz) was
exported through `mrz_band.band_candidates()` -- a research-side crop
picker.  The product path (tests/gui.py::_rst_native_mrz_band) picks a
DIFFERENT crop for 11 of the 41 photos.  Measured on the same photos:

    30 photos : product crop == training crop   (pixel identical)
    11 photos : product crop != training crop   (mean |d| 0.03 .. 0.10)

So a model trained on the research path learns glyph shapes that do not
match what the product feeds it, and those 11 photos lose exact matches.
This exporter produces the same slice npz but sourced from the PRODUCT
path, so training and inference see the same pixels.

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
    ap.add_argument("--tool", default=str(ROOT / "6_mrz_ocr/build/mrz_ocr_cnn_tool"))
    ap.add_argument("--gt", default=str(ROOT / "10_ocr_research/photo_direct_bench.json"))
    ap.add_argument("--auto-gt", default=str(ROOT / "tests/photo_auto_gt.json"))
    ap.add_argument("--out", default="/tmp/product_slices.npz")
    ap.add_argument("--only", default="")
    ap.add_argument("--dump-prefix", default="/tmp/mrz_dump")
    ap.add_argument("--tag", default="prod")
    args = ap.parse_args()

    gui, app = load_gui()
    bench = {}
    for r in json.load(open(args.gt))["records"]:
        if r.get("gt"):
            bench[Path(r["path"]).name] = r["gt"]
    auto = json.load(open(args.auto_gt))
    auto = auto.get("gt", auto)

    sx, sy, smeta = [], [], []
    out_dir = Path("/tmp/product_export")
    out_dir.mkdir(parents=True, exist_ok=True)

    for label, path in gui.iter_passport_photos():
        if args.only and args.only not in label:
            continue
        name = Path(path).name
        truth = None
        # Prefer the bench GT (user-visible metric); fall back to auto_gt.
        if name in bench and len(bench[name].get("line1", "")) == 44 \
                and len(bench[name].get("line2", "")) == 44:
            truth = (bench[name]["line1"], bench[name]["line2"], "bench")
        elif name in auto:
            a = auto[name]
            if len(a.get("l1", "")) == 44 and len(a.get("l2", "")) == 44:
                truth = (a["l1"], a["l2"], "auto")
        if truth is None:
            print(f"  skip {label}: no 44/44 GT")
            continue
        l1, l2, src = truth
        band = band_for(app, path)
        if band is None:
            print(f"  skip {label}: no band")
            continue
        ppm = out_dir / (Path(path).stem + ".ppm")
        Image.fromarray(band).save(str(ppm), format="PPM")

        for f in glob.glob(f"{args.dump_prefix}_{args.tag}_*"):
            try:
                os.remove(f)
            except OSError:
                pass
        env = dict(os.environ)
        env.update({"MRZ_OCR_DUMP": args.tag, "MRZ_OCR_LINE_MODELS": "1",
                    "MRZ_OCR_CRNN": "0", "MRZ_OCR_FUSE": "0"})
        r = subprocess.run([args.tool, str(ppm)], capture_output=True, text=True,
                           env=env, timeout=120)
        if r.returncode != 0:
            print(f"  skip {label}: tool rc={r.returncode}")
            continue

        n = 0
        for li, t in ((0, l1), (1, l2)):
            for c, ch in enumerate(t):
                if ch not in GI:
                    continue
                fs = glob.glob(f"{args.dump_prefix}_{args.tag}_l{li}_c{c:02d}_x*_r*_w*.ppm")
                if not fs:
                    continue
                patch = np.asarray(Image.open(fs[0]).convert("L"),
                                   dtype=np.float32) / 255.0
                sx.append(patch)
                sy.append(GI[ch])
                smeta.append({"file": name, "line": li, "col": c, "ch": ch,
                              "label": label, "gt_src": src})
                n += 1
        print(f"  {label}: {n} slices (gt={src})", flush=True)

    if not sx:
        raise SystemExit("no slices exported")
    np.savez(args.out,
             X=np.array(sx, dtype=np.float32).reshape(-1, 12, 16),
             Y=np.array(sy, dtype=np.int64),
             meta=np.array([json.dumps(m) for m in smeta]))
    print(f"wrote {args.out}: {len(sx)} slices from "
          f"{len({m['file'] for m in smeta})} photos")


if __name__ == "__main__":
    main()
