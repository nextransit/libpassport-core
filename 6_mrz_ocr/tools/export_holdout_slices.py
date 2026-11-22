#!/usr/bin/env python3
"""Export the REAL product-path glyph slices for offline weight scoring.

Why this exists
---------------
The end-to-end bench (tests/photo_direct_bench.py) measures exact-match on
the whole photo, which needs a full recompile per weight change.  This
script exports the 16x12 glyph slices the SAME product path feeds to the
CNN, so a weight set can be scored on the corpus in seconds instead of
minutes.

Fidelity rule (learned the hard way)
------------------------------------
An earlier version re-derived the band by re-running `restore.load` /
`detect_structure` / `drop_reflection` itself.  That is NOT the band the
product uses: the product goes through `app._ocr_photo_engines()`, which
may pick a different reflection crop and a different scale.  The two
produced different bands for 11 of 41 photos, and the offline score
disagreed with the bench by 6 photos -- i.e. the offline channel was
optimising the wrong target.

So this script now takes the band from the SAME call the bench makes:
`app._ocr_photo_engines(path)` -> pair / work2 / scale, then
`app._rst_native_mrz_band(rgb, pair, scale, source_bottom=...)` with the
identical `source_bottom` the bench computes.

Output: <out> npz with X (N,12,16), Y (N,), meta (N,) json strings.
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
    """The band the product path uses -- identical to the bench's."""
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
    ap.add_argument("--tool", default=str(ROOT / "6_mrz_ocr/build/mrz_ocr_cnn_tool"))
    ap.add_argument("--gt", default=str(ROOT / "10_ocr_research/photo_direct_bench.json"))
    ap.add_argument("--out", default="/tmp/holdout_slices.npz")
    ap.add_argument("--only", default="", help="substring filter on the label")
    ap.add_argument("--dump-prefix", default="/tmp/mrz_dump",
                    help="base name; the C tool writes "
                         "<prefix>_<tag>_l<line>_c<col>_x.._r.._w..ppm")
    ap.add_argument("--band-dir", default="/tmp/holdout_export")
    args = ap.parse_args()

    gui, app = load_gui()
    gts = {}
    for r in json.load(open(args.gt))["records"]:
        if r.get("gt"):
            gts[Path(r["path"]).name] = r["gt"]

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
