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
from export_omtest_slices import (
    adaptive_xpad, fast_pair, fast_single_row, native_band, native_band_single,
)
import mrz_band

ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ<"
GI = {c: i for i, c in enumerate(ALPHABET)}
CNN_FLAT = 96
STEPS = 44


def _locate_band_bottom(gray, scale, Wd, H):
    """Fallback: scan the bottom 30% of the image for MRZ text.
    Used when fast_pair/fast_single_row find the wrong text region
    (e.g. document text above the MRZ)."""
    h, w = gray.shape
    y0 = int(h * 0.70)
    y1 = int(h * 0.98)
    if y1 <= y0 or w < 80:
        return None
    sx0 = int(w * 0.03)
    sx1 = int(w * 0.97)
    stride = 4
    sample_w = (sx1 - sx0 + stride - 1) // stride
    if sample_w < 40:
        return None
    rows = gray[y0:y1]
    dark = ((rows[:, sx0:sx1:stride] < 170).sum(axis=1)).astype(int)
    min_ink = max(3, int(sample_w * 0.15))
    max_ink = int(sample_w * 0.88)
    runs = []
    start = -1
    gap = 0
    for i in range(len(dark)):
        if min_ink <= dark[i] <= max_ink:
            if start < 0:
                start = i
            gap = 0
        elif start >= 0:
            gap += 1
            if gap > 3:
                runs.append((start, i - gap))
                start = -1
                gap = 0
    if start >= 0:
        runs.append((start, len(dark) - 1))
    if not runs:
        return None
    best_pair = None
    for i in range(len(runs) - 1):
        a = runs[i]
        for j in range(i + 1, min(len(runs), i + 3)):
            b = runs[j]
            h1 = a[1] - a[0] + 1
            h2 = b[1] - b[0] + 1
            gap_rows = b[0] - a[1] - 1
            if h1 < 5 or h2 < 5 or h1 > 120 or h2 > 120:
                continue
            if h2 > 0 and not (0.3 <= h1 / h2 <= 3.0):
                continue
            max_gap = max(15, int(min(h1, h2) * 3))
            if gap_rows > max_gap:
                continue
            if best_pair is None or b[1] > best_pair[1][1]:
                best_pair = (a, b)
    if best_pair is not None:
        a, b = best_pair
        h1 = a[1] - a[0] + 1
        h2 = b[1] - b[0] + 1
        pad = max(8, min(25, int(min(h1, h2) * 0.70)))
        top = max(0, y0 + a[0] - pad)
        bot = min(h, y0 + b[1] + pad)
        bh = int((bot - top) / scale)
        if bh < 30:
            return None
        return (0, int(top / scale), int(Wd), bh)
    best_single = None
    for run in runs:
        rh = run[1] - run[0] + 1
        if rh < 5 or rh > 120:
            continue
        if best_single is None or rh > best_single[1] - best_single[0] + 1:
            best_single = run
    if best_single is not None:
        rh = best_single[1] - best_single[0] + 1
        pad = max(8, min(25, int(rh * 0.70)))
        top = max(0, y0 + best_single[0] - pad)
        bot = min(h, y0 + best_single[1] + pad)
        bh = int((bot - top) / scale)
        if bh < 20:
            return None
        return (0, int(top / scale), int(Wd), bh)
    return None


def _has_dark_separator(gray, y0, y1, thresh=100, min_rows=2):
    """True when a dark band separates two text regions.

    A passport lying on a glass table is mirrored below its own page edge,
    so the "bottom band" found by _locate_band_bottom is a reflection, not a
    second MRZ.  The page edge in between is a dark row run.
    """
    if y1 <= y0:
        return False
    rows = gray[int(y0):int(y1)].mean(axis=1)
    run = 0
    for v in rows:
        run = run + 1 if v < thresh else 0
        if run >= min_rows:
            return True
    return False


def locate_band(rgb: Image.Image, pad_hint: list | None = None):
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
    if pair is not None:
        xp = adaptive_xpad(gray, pair)
        top, bot = native_band(gray, pair[0], pair[1], xp[0], xp[1])
        bx = int(xp[0] / scale)
        by = int(top / scale)
        bw = int(Wd - xp[0] / scale - xp[1] / scale)
        bh = int((bot - top) / scale)
        if bw >= 400 and bh >= 30:
            result = (bx, by, bw, bh)
            bottom = _locate_band_bottom(gray, scale, Wd, H)
            if bottom is not None and bottom[1] > result[1] + max(int(result[3] * 0.55), 30):
                if _has_dark_separator(gray, (result[1] + result[3]) * scale,
                                       bottom[1] * scale):
                    # bottom band is the mirror reflection under the page edge
                    if pad_hint is not None:
                        pad_hint.append(max(int(result[3] * 0.25), 10))
                    return result
                return bottom
            return result
    single = fast_single_row(gray)
    if single is not None:
        xp = adaptive_xpad(gray, (single, single))
        top, bot, tl, tr = native_band_single(gray, single, xp[0], xp[1])
        bx = int(tl / scale)
        by = int(top / scale)
        bw = int((tr - tl) / scale)
        bh = int((bot - top) / scale)
        if bw >= 400 and bh >= 20:
            result = (bx, by, bw, bh)
            bottom = _locate_band_bottom(gray, scale, Wd, H)
            if bottom is not None and bottom[1] > result[1] + max(int(result[3] * 0.55), 30):
                if _has_dark_separator(gray, (result[1] + result[3]) * scale,
                                       bottom[1] * scale):
                    # bottom band is the mirror reflection under the page edge
                    if pad_hint is not None:
                        pad_hint.append(max(int(result[3] * 0.25), 10))
                    return result
                return bottom
            return result
    return _locate_band_bottom(gray, scale, Wd, H)


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
    ap.add_argument("--pad-map", default="",
                    help="JSON file mapping image basename (or label) to crop padding")
    args = ap.parse_args()

    pad_map = {}
    if args.pad_map:
        pm = json.loads((ROOT / args.pad_map).read_text())
        pad_map = {k: int(v) for k, v in pm.items()}

    d = json.loads((ROOT / args.gt).read_text())
    gt = d["gt"] if isinstance(d, dict) and "gt" in d else d
    tmp = Path("/tmp/photo_train_export")
    tmp.mkdir(exist_ok=True)

    X, Y, META = [], [], []
    FX, FY, FID = [], [], []
    located = skipped_band = skipped_line = skipped_tool = skipped_charset = 0

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
        pm = pad_map.get(basename, pad_map.get(label))
        cands = mrz_band.band_candidates(rgb, args.pad, pm)
        if not cands:
            skipped_band += 1
            continue
        located += 1
        stem = Path(basename).stem
        ppm = tmp / f"{stem}_band.ppm"
        # Score every candidate crop first (no glyph dump), then keep the one
        # whose line2 passes the ICAO check digits.  The chosen crop is the
        # one the evaluator scores, so training and scoring share a framing.
        scored = []
        for i, (_kind, rect) in enumerate(cands):
            cand_ppm = tmp / f"{stem}_cand{i}.ppm"
            rgb.crop(rect).save(str(cand_ppm), format="PPM")
            scored.append(mrz_band.run_tool(ROOT / args.tool, cand_ppm))
        pick = mrz_band.pick_candidate(scored, [k for k, _ in cands])
        band_kind, rect = cands[pick]
        rgb.crop(rect).save(str(ppm), format="PPM")

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
                try:
                    os.remove(f)
                except OSError:
                    pass
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
                    if all(c in GI for c in l1) and all(c in GI for c in l2):
                        FX.append(np.stack([a, b]))
                        FY.append(np.stack([[GI[c] for c in l1], [GI[c] for c in l2]]))
                        FID.append(basename)
                    else:
                        skipped_charset += 1

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
          f"skipped_tool={skipped_tool} skipped_line={skipped_line} "
          f"skipped_charset={skipped_charset}")
    return 0 if wrote else 1


if __name__ == "__main__":
    raise SystemExit(main())
