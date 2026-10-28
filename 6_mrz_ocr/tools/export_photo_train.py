#!/usr/bin/env python3
"""Export CNN glyph slices + CRNN features from photo_auto_gt.json photos.

Reuses the Android fast-path band locator (fast_pair/adaptive_xpad/native_band)
exactly like build_gt_multiengine.locate_band, so the crop/pad used here match
what inference sees.  For each usable GT record (line1 and line2 both 44 chars)
we crop the band, then run the C recogniser twice:

  * MRZ_OCR_DUMP=tag  -> per-column 12x16 glyph patches -> CNN slices npz
  * MRZ_OCR_FEAT_DUMP -> per-cell 96-dim conv features  -> CRNN features npz

Tier filtering (default A,B) is applied per line via l1_tier / l2_tier so a
photo whose line1 is disputed can still contribute its frozen line2.

Outputs (all relative to ROOT):
  --out-dir <dir>/cnn_slices.npz and crnn_feats.npz (dir defaults to data/)
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_omtest_slices import adaptive_xpad, fast_pair, native_band

ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ<"
GI = {c: i for i, c in enumerate(ALPHABET)}
CNN_FLAT = 96
STEPS = 44


def locate_band(rgb: Image.Image):
    Wd, H = rgb.size
    arr = np.asarray(rgb, dtype=np.uint32)
    gray = ((arr[:, :, 0] * 299 + arr[:, :, 1] * 587 +
             arr[:, :, 2] * 114 + 500) // 1000).astype(np.uint8)
    scale = 1.0
    if Wd < 900:
        scale = 1500.0 / Wd
        gray = np.asarray(
            Image.fromarray(gray).resize(
                (int(Wd * scale), int(H * scale)), Image.BILINEAR),
            dtype=np.uint8)
    pair = fast_pair(gray)
    if pair is None:
        return None
    xp = adaptive_xpad(gray, pair)
    top, bot = native_band(gray, pair[0], pair[1], xp[0], xp[1])
    bx = int(xp[0] / scale)
    by = int(top / scale)
    bw = int(Wd - xp[0] / scale - xp[1] / scale)
    bh = int((bot - top) / scale)
    if bw < 400 or bh < 30:
        return None
    return bx, by, bw, bh


def read_feat(path: Path) -> np.ndarray | None:
    raw = path.read_bytes()
    if len(raw) < 4:
        return None
    (n,) = struct.unpack_from("i", raw, 0)
    need = 4 + n * CNN_FLAT * 4
    if n != STEPS or len(raw) < need:
        return None
    arr = np.frombuffer(raw, dtype="<f4", offset=4, count=n * CNN_FLAT)
    return arr.reshape(n, CNN_FLAT).astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", default="tests/photo_auto_gt.json")
    ap.add_argument("--tool", default="6_mrz_ocr/build/mrz_ocr_cnn_tool")
    ap.add_argument("--out-dir", default="6_mrz_ocr/data/photo_train")
    ap.add_argument("--tag", default="photo")
    ap.add_argument("--dump-prefix", default="/tmp/mrz_dump")
    ap.add_argument("--pad", type=int, default=10)
    ap.add_argument("--min-tier", default="A", choices=("A", "B"))
    ap.add_argument("--dispute-max", type=int, default=None,
                    help="max l1_dispute_n to accept a tier-C line1 (None = reject all C)")
    args = ap.parse_args()

    d = json.loads((ROOT / args.gt).read_text())
    gt = d["gt"] if isinstance(d, dict) and "gt" in d else d
    tmp = Path("/tmp/photo_train_export")
    tmp.mkdir(exist_ok=True)

    X, Y, META = [], [], []
    FX, FY, FID = [], [], []
    located = skipped_band = skipped_line = skipped_tool = 0

    for basename, rec in gt.items():
        label = rec.get("label")
        if not label:
            continue
        l1, l2 = rec.get("l1", ""), rec.get("l2", "")
        # Per-line tier gate: line1/line2 may be frozen independently. Accepted:
        #   * tier A/B -> always ok
        #   * tier C  -> line1 ok only up to --dispute-max (high-confidence cols),
        #                line2 ok only if l2_tier is A/B (line2 has no dispute)
        #   * no per-line tier (user-confirmed / engine-candidate 44-char GT) -> ok
        acc = set(("A", "B")) if args.min_tier == "B" else {"A"}
        l1_t = rec.get("l1_tier") or rec.get("tier") or ""
        l2_t = rec.get("l2_tier") or rec.get("tier") or ""
        dis = rec.get("l1_dispute_n")

        def line_ok(truth, t, is_l1):
            if not (len(truth) == STEPS):
                return False
            if t == "":
                return True  # no tier recorded (user/engine-confirmed GT)
            if t in acc:
                return True
            if t == "C" and is_l1 and args.dispute_max is not None:
                return dis is not None and dis <= args.dispute_max
            if t == "C" and not is_l1:
                return False  # un-frozen line2 mistrusted
            return False

        l1_ok = line_ok(l1, l1_t, True)
        l2_ok = line_ok(l2, l2_t, False)
        if not (l1_ok or l2_ok):
            continue
        img_path = ROOT / label
        if not img_path.exists():
            continue
        rgb = Image.open(img_path).convert("RGB")
        crop = locate_band(rgb)
        if crop is None:
            skipped_band += 1
            continue
        located += 1
        bx, by, bw, bh = crop
        pad = args.pad
        band = rgb.crop((bx, max(0, by - pad), bx + bw, min(rgb.height, by + bh + pad)))
        stem = Path(basename).stem
        ppm = tmp / f"{stem}_band.ppm"
        band.save(str(ppm), format="PPM")

        for f in glob.glob(f"{args.dump_prefix}_{args.tag}_l*_c*_x*_r*_w*.ppm"):
            os.remove(f)
        env = dict(os.environ)
        env["MRZ_OCR_DUMP"] = args.tag
        env["MRZ_OCR_LINE_MODELS"] = "1"
        env["MRZ_OCR_CRNN"] = "0"
        r = subprocess.run([str(ROOT / args.tool), str(ppm)],
                           capture_output=True, text=True, timeout=60, env=env)
        if r.returncode != 0:
            skipped_tool += 1
            continue

        n_added = 0
        for li, truth, ok, tierk in ((0, l1, l1_ok, "l1_tier"),
                                     (1, l2, l2_ok, "l2_tier")):
            if not (ok and truth):
                continue
            for c, ch in enumerate(truth):
                if ch not in GI:
                    continue
                files = glob.glob(
                    f"{args.dump_prefix}_{args.tag}_l{li}_c{c:02d}_x*_r*_w*.ppm")
                if not files:
                    continue
                patch = np.asarray(Image.open(files[0]).convert("L"),
                                   dtype=np.float32) / 255.0
                X.append(patch)
                Y.append(GI[ch])
                META.append({"file": basename, "line": li, "col": c, "ch": ch,
                             "tier": rec.get(tierk) or rec.get("tier")})
                n_added += 1
        if n_added == 0:
            skipped_line += 1

        # CRNN features (only when both lines frozen)
        if l1_ok and l2_ok:
            for f in glob.glob("/tmp/mrz_feat_crnn_*_l*.bin"):
                os.remove(f)
            fe = dict(os.environ)
            tag = f"crnn_{os.getpid()}"
            fe["MRZ_OCR_FEAT_DUMP"] = tag
            fe["MRZ_OCR_LINE_MODELS"] = "1"
            fe["MRZ_OCR_CRNN"] = "1"
            fe["MRZ_OCR_PAR"] = "0"
            r2 = subprocess.run([str(ROOT / args.tool), str(ppm)],
                                capture_output=True, text=True, timeout=60, env=fe)
            f0 = Path(f"/tmp/mrz_feat_{tag}_l0.bin")
            f1 = Path(f"/tmp/mrz_feat_{tag}_l1.bin")
            if f0.exists() and f1.exists():
                a = read_feat(f0)
                b = read_feat(f1)
                if a is not None and b is not None:
                    FX.append(np.stack([a, b]))
                    FY.append(np.stack([[GI[c] for c in l1], [GI[c] for c in l2]]))
                    FID.append(basename)

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    wrote = 0
    if X:
        Xa = np.array(X, dtype=np.float32).reshape(-1, 12, 16)
        Ya = np.array(Y, dtype=np.int64)
        np.savez(out_dir / "cnn_slices.npz", X=Xa, Y=Ya,
                 meta=np.array([json.dumps(m) for m in META]))
        wrote += 1
        n1 = sum(1 for m in META if m["line"] == 0)
        print(f"cnn: exported {len(X)} slices ({n1} line1) "
              f"from {len({m['file'] for m in META})} photos -> "
              f"{out_dir/'cnn_slices.npz'}")
    if FX:
        np.savez_compressed(out_dir / "crnn_feats.npz",
                            features=np.stack(FX), labels=np.stack(FY),
                            ids=np.array(FID))
        wrote += 1
        print(f"crnn: {len(FID)} samples shape={np.stack(FX).shape} -> "
              f"{out_dir/'crnn_feats.npz'}")
    print(f"located={located} skipped_band={skipped_band} "
          f"skipped_tool={skipped_tool} skipped_line={skipped_line}")
    return 0 if wrote else 1


if __name__ == "__main__":
    raise SystemExit(main())
