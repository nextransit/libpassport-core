#!/usr/bin/env python3
"""Export real glyph slices from the FROZEN-GT eval band crops.

Why this exists
---------------
tools/export_omtest_slices.py dumps glyphs from the ANDROID fast-path
band (fast_pair locate + adaptive x-pad + width downscale), which is
the distribution the device feeds the network in production.
tools/eval_frozen.py however scores against a DIFFERENT band: the
frozen GT crop box with 10px of vertical padding and no downscale.
Weights trained only on the Android band must generalize across that
gap -- the 2026-10-24 line-head retrain showed what happens when they
do not (passport 37/41 -> 14/41, assets 21/32 -> 0/32).

This exporter produces slices from the EVAL band itself, so a trainer
can mix BOTH band distributions and the heads stop betting on one crop
path.  Labels come straight from the frozen GT (authoritative, human
adjudicated) -- no fixmrz repair heuristics.  Line 1 is only exported
for records that carry a line1 (gt_frozen_assets has none; join
gt_passport_manual.json for the passport set).

The band crop replicates eval_frozen.py exactly:
    band = rgb[max(0, y - pad) : min(H, y + h + pad),  x : x + w]
(horizontal is NOT padded), saved as PPM and handed to the same C tool
the eval harness uses.

Usage:
  python3 tools/export_gt_band_slices.py \
      --gt data/gt_frozen_passport_corrected.json \
      --gt data/gt_frozen_assets.json \
      --line1-gt data/gt_passport_manual.json \
      --out data/gt_band_slices.npz
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
GLYPHS = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ<")
GI = {c: i for i, c in enumerate(GLYPHS)}
OMTEST = ROOT.parent / "tmp" / "omtest"
ASSETS = ROOT.parent / "assets" / "pic"


def image_for(name: str) -> Path | None:
    p = ASSETS / name
    if p.exists():
        return p
    for d in sorted(OMTEST.iterdir()):
        if not d.is_dir():
            continue
        for ext in ("*.jpg", "*.png"):
            for jpg in d.glob(ext):
                if jpg.stem == Path(name).stem:
                    return jpg
    return None


def crop_band(img_path: Path, crop, pad: int) -> Image.Image:
    """The eval_frozen.py band crop, byte for byte."""
    im = Image.open(img_path).convert("RGB")
    W, H = im.size
    x, y, w, h = crop
    top = max(0, y - pad)
    bot = min(H, y + h + pad)
    left = max(0, x)
    right = min(W, x + w)
    band = im.crop((left, top, right, bot))
    if band.height < 20:
        raise ValueError(f"band too small for {img_path}")
    return band


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", action="append", default=[],
                    help="frozen-GT json (repeatable); records need "
                         "id/crop/pad/line2; line1 optional")
    ap.add_argument("--line1-gt", default="",
                    help="optional GT json with line1 labels (joined by id) "
                         "for records whose primary file lacks line1")
    ap.add_argument("--tool", default="build/mrz_ocr_cnn_tool")
    ap.add_argument("--out", default="data/gt_band_slices.npz")
    ap.add_argument("--tag", default="img")
    ap.add_argument("--dump-prefix", default="/tmp/mrz_dump")
    args = ap.parse_args()

    if not args.gt:
        print("no --gt given", file=sys.stderr)
        return 1

    line1_extra: dict[str, str] = {}
    if args.line1_gt:
        d = json.loads((ROOT / args.line1_gt).read_text())
        recs = d["records"] if isinstance(d, dict) and "records" in d else d
        for r in recs:
            if r.get("line1"):
                # ids may carry a filename suffix; join by stem.
                line1_extra[Path(r["id"]).stem] = r["line1"]

    X, Y, META = [], [], []
    skipped: dict[str, int] = {}
    tmp = Path("/tmp/gt_band_export")
    tmp.mkdir(exist_ok=True)

    for gt_path in args.gt:
        recs = json.loads((ROOT / gt_path).read_text())["records"]
        for rec in recs:
            img = image_for(rec["id"])
            if img is None:
                skipped["no_image"] = skipped.get("no_image", 0) + 1
                continue
            l1 = rec.get("line1") or line1_extra.get(Path(rec["id"]).stem, "")
            l2 = rec.get("line2", "")
            if len(l2) != 44:
                skipped["bad_line2"] = skipped.get("bad_line2", 0) + 1
                continue
            pad = int(rec.get("pad", 10))
            try:
                band = crop_band(img, rec["crop"], pad)
            except (ValueError, KeyError) as e:
                print(f"  skip {rec['id']}: {e}", file=sys.stderr)
                skipped["bad_crop"] = skipped.get("bad_crop", 0) + 1
                continue
            ppm = tmp / f"{Path(rec['id']).stem}_band.ppm"
            band.save(str(ppm), format="PPM")

            for f in glob.glob(f"{args.dump_prefix}_{args.tag}_l*_c*_x*_r*_w*.ppm"):
                os.remove(f)
            for f in glob.glob(f"/private/tmp/{args.dump_prefix.split('/')[-1]}_{args.tag}_l*_c*_x*_r*_w*.ppm"):
                os.remove(f)
            env = dict(os.environ)
            env["MRZ_OCR_DUMP"] = args.tag
            env["MRZ_OCR_LINE_MODELS"] = "1"
            r = subprocess.run([str(ROOT / args.tool), str(ppm)],
                               capture_output=True, text=True, timeout=60,
                               env=env)
            if r.returncode != 0:
                skipped["tool_fail"] = skipped.get("tool_fail", 0) + 1
                continue

            n_added = 0
            for li, truth in ((0, l1), (1, l2)):
                if not truth:
                    continue
                for c, ch in enumerate(truth):
                    if ch not in GI:
                        continue
                    files = glob.glob(
                        f"{args.dump_prefix}_{args.tag}_l{li}_c{c:02d}_x*_r*_w*.ppm")
                    if not files:
                        files = glob.glob(
                            f"/private/tmp/{args.dump_prefix.split('/')[-1]}_{args.tag}_l{li}_c{c:02d}_x*_r*_w*.ppm")
                    if not files:
                        continue
                    patch = np.asarray(Image.open(files[0]).convert("L"),
                                       dtype=np.float32) / 255.0
                    X.append(patch)
                    Y.append(GI[ch])
                    META.append({"file": f"{rec['id']}/evalband_pad{pad}",
                                 "line": li, "col": c, "ch": ch,
                                 "weak": False, "cand": ch})
                    n_added += 1
            if n_added == 0:
                skipped["dump_missing"] = skipped.get("dump_missing", 0) + 1

    if not X:
        print("no slices collected")
        return 1
    Xa = np.array(X, dtype=np.float32).reshape(-1, 12, 16)
    Ya = np.array(Y, dtype=np.int64)
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, X=Xa, Y=Ya, meta=np.array([json.dumps(m) for m in META]))
    n1 = sum(1 for m in META if m["line"] == 0)
    print(f"exported {len(X)} slices ({n1} line1) from "
          f"{len({m['file'].split('/')[0] for m in META})} photos -> {out}")
    print(f"  skipped: {skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
