#!/usr/bin/env python3
"""Mine REAL glyph variants from the product-path band pixels.

Why this exists
---------------
The product-path corpus gives exactly ONE 16x12 glyph per character
position -- 2068 real slices for 47 photographs.  Its class histogram is
brutally lopsided: line 1 has 1463 '<' and only EIGHT 'J', thirty 'O',
four 'B'.  The measured failures are exactly those classes: 'O' read as
'2', 'J' read as 'H', '4' read as 'H', 'C' read as 'E', 'M' read as '8'.

Synthetic glyphs cannot fix that: they come from a renderer with a
different font, weight and slant than the real photographs, so the
network learns the synthetic appearance of 'J' and applies it to a
photographed 'J'.  Re-sampling the same synthetic glyph does not help
either -- it is the same appearance.

This tool mines the REAL pixels instead.  For every GT character position
it crops the glyph from the PRODUCT band at a range of sub-pixel offsets
and scales, and resamples each crop with the EXACT same code path the C
pipeline uses (see `resample_char_gray` in src/mrz_ocr_cnn.c):

    area-average the raw grayscale box down to 16x12, invert to ink
    3x3 box blur
    x-centroid alignment (+-2 px)
    y-centroid alignment (+-2 px)

Every generated sample is a real photographed glyph, seen at a slightly
different crop.  That multiplies the rare classes by 10-30x WITHOUT
introducing a synthetic domain -- which is the whole point.

Fidelity rules (learned the hard way)
-------------------------------------
* The band must be the one the PRODUCT path produces.  We reuse the
  bands exported by tools/export_holdout_slices.py rather than
  re-deriving them, because a re-derived band differs for 11 of 41
  photos and the offline score then disagrees with the bench.
* The pixel path must match the C one exactly.  An earlier version of
  this script applied a percentile normalisation; the C pipeline does
  no such thing (it inverts raw grayscale), so the generated samples
  lived in a different space than the real ones and hurt.
* The cell geometry comes from the C tool's own DUMP output.  The dump
  file name carries x, r, w -- and, since the h/b extension, the cell
  height and the row's base offset inside the band.

Candidate sets (`cand`)
-----------------------
The trainer consumes a per-slice `cand` list and turns it into a SOFT
target.  Without it every slice is a hard label, and a single misread
character in the automatic GT becomes a "correct" training label -- the
model then learns to reproduce the error.  With it, a cell whose GT is
uncertain is trained as "any of these characters", which is what the
evidence actually supports.

The candidate set for a cell is built from two sources:

* the ICAO 9303 field mask (a port of `fill_allow` in
  src/mrz_ocr_cnn.c): it removes classes that cannot legally appear in
  that slot at all.
* the check-digit constraint: for the cells that take part in a
  check-digit field, only the characters that keep the field's check
  digit consistent with the GT are kept.

The second source is the strong one.  A check digit over a 9-14
character field is a strong constraint: it typically leaves a handful of
alternatives instead of the full alphabet, so the soft target is
informative rather than vacuous.

Usage:
    python3 tools/aug_real_glyphs.py --out /tmp/real_aug.npz
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path("/Users/zhouyong/Desktop/work/Decard/gitlab/passport")
ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ<"
GI = {c: i for i, c in enumerate(ALPHABET)}

# The dump file name now carries the cell height and the row base:
#   mrz_dump_<tag>_l<line>_c<col>_x<x>_r<y>_w<w>_h<h>_b<base>.ppm
DUMP_RE = re.compile(
    r"mrz_dump_(?P<tag>.+)_l(?P<line>\d+)_c(?P<col>\d+)"
    r"_x(?P<x>-?\d+)_r(?P<y>-?\d+)_w(?P<w>\d+)"
    r"(?:_h(?P<h>\d+))?(?:_b(?P<b>-?\d+))?\.ppm$")

# ---------------------------------------------------------------------------
# ICAO 9303 field mask + check-digit constraint (candidate sets)
# ---------------------------------------------------------------------------
# A port of `fill_allow` in src/mrz_ocr_cnn.c (strict ICAO TD3 branch).
# Returns the set of characters that may legally appear at (line, col).
DIGITS = "0123456789"
ALPHA = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
FILLER = "<"


def icao_mask(line: int, col: int) -> str:
    """Characters allowed at (line, col) by the ICAO 9303 TD3 mask."""
    if line == 0:
        if col == 0:
            return "ACIPV"
        if col == 1:
            return ALPHA + FILLER
        if 2 <= col <= 4:
            return ALPHA
        return ALPHA + FILLER          # name field
    # line 2
    if col <= 8:
        return ALPHA + DIGITS + FILLER          # passport number
    if col in (9, 19, 27):
        return DIGITS                            # check digits
    if col <= 12:
        return ALPHA + FILLER                    # nationality
    if (col <= 18) or (21 <= col <= 26):
        return DIGITS + FILLER + ALPHA            # birth / expiry dates
    if col == 20:
        return ALPHA + DIGITS + FILLER            # sex
    if col <= 41:
        return ALPHA + DIGITS + FILLER            # personal number
    return DIGITS + FILLER                        # col 42/43


# The four check-digit fields of a TD3 line 2, as (first, last, cd_pos).
CD_FIELDS = ((0, 8, 9), (13, 18, 19), (21, 26, 27), (28, 41, 42))
# The composite check digit covers these ranges of the line.
CD_WEIGHTS = (7, 3, 1)


def _char_value(ch: str) -> int:
    if ch.isdigit():
        return int(ch)
    if ch in ALPHA:
        return ord(ch) - 55
    return 0


def check_digit(text: str) -> str:
    """ICAO 9303 check digit over `text`, weights 7-3-1 restarting at 7."""
    total = sum(CD_WEIGHTS[k % 3] * _char_value(ch) for k, ch in enumerate(text))
    return str(total % 10)


def _field_candidates(field, line2: str) -> dict:
    """For each column of a check-digit field, the characters that keep the
    field's check digit equal to the one written in the GT.

    Returns {col: set_of_chars}.  Columns not in any field are absent.
    """
    first, last, cd_pos = field
    target = line2[cd_pos]
    if not target.isdigit():
        return {}
    # The check digit itself is not part of the field.
    cols = list(range(first, last + 1))
    mask = {c: set(icao_mask(1, c)) for c in cols}
    # For every column, try every character the mask allows and keep the
    # ones that make the field's check digit come out right.
    out = {}
    for c in cols:
        keep = set()
        for ch in mask[c]:
            trial = list(line2[first:last + 1])
            trial[c - first] = ch
            if check_digit("".join(trial)) == target:
                keep.add(ch)
        if keep:
            out[c] = keep
    return out


def candidate_sets(line: int, truth: str) -> dict:
    """{col: sorted list of plausible characters} for one row.

    The GT character is always included: the automatic GT is the best
    evidence available, and the check digit only narrows the field.

    Line 1 has NO check digit, so the ICAO mask alone leaves ~26
    candidates in the name field -- far too weak to train on.  A soft
    target that wide is no target at all: measured, a line-1 head
    trained this way collapsed to a constant class (char accuracy
    0.97%).  So line 1 keeps the HARD label and only line 2, where the
    check digits narrow each field to 3-5 characters, gets a soft
    target.
    """
    res = {}
    if line == 0:
        for c, ch in enumerate(truth):
            res[c] = [ch]
        return res
    fields = {}
    for f in CD_FIELDS:
        fields.update(_field_candidates(f, truth))
    for c, ch in enumerate(truth):
        allowed = set(icao_mask(1, c))
        if c in fields:
            narrowed = fields[c]
            # The GT character must survive the narrowing, otherwise the
            # GT itself contradicts the check digit and the constraint
            # must not be applied to this cell.
            if ch in narrowed:
                allowed = narrowed
            else:
                allowed = allowed | {ch}
        else:
            # A column that no check digit covers keeps the HARD label.
            # The ICAO mask alone leaves 27 candidates in the
            # nationality and sex slots, and a soft target that wide
            # carries no information -- it just lets the model output
            # anything in the slot.
            res[c] = [ch]
            continue
        res[c] = sorted(allowed | {ch})
    return res


# --------------------------------------------------------------------------
# the C pipeline's pixel path, reproduced exactly
# --------------------------------------------------------------------------
def resample_char_gray(gray, W, H, rx, ry, rw, rh):
    """Port of resample_char_gray() from src/mrz_ocr_cnn.c.

    gray: (H, W) uint8 array.  Returns (12, 16) float32 with ink = 1 - v/255.

    The C version accumulates the area average with integer pixel loops;
    the numpy form below is the same arithmetic (an integer sum divided
    by the number of pixels that fell inside the image), just vectorised,
    because the augmentation needs thousands of these.
    """
    out = np.zeros((12, 16), dtype=np.float32)
    if rw <= 0 or rh <= 0:
        return out
    # integer pixel bounds of each output cell, exactly as the C code
    # computes them
    sy0 = (np.arange(12) * rh) // 12
    sy1 = ((np.arange(12) + 1) * rh) // 12
    sy1 = np.maximum(sy1, sy0 + 1)
    sx0 = (np.arange(16) * rw) // 16
    sx1 = ((np.arange(16) + 1) * rw) // 16
    sx1 = np.maximum(sx1, sx0 + 1)

    # clip to the image and accumulate
    yy0 = np.clip(ry + sy0, 0, H)
    yy1 = np.clip(ry + sy1, 0, H)
    total = np.zeros((12, 16), dtype=np.int64)
    count = np.zeros((12, 16), dtype=np.int64)
    for oy in range(12):
        a, b = yy0[oy], yy1[oy]
        if b <= a:
            continue
        strip = gray[a:b, :]                      # (rows, W)
        for ox in range(16):
            x0 = rx + sx0[ox]
            x1 = rx + sx1[ox]
            ca = max(0, x0)
            cb = min(W, x1)
            if cb <= ca:
                continue
            total[oy, ox] = strip[:, ca:cb].sum(dtype=np.int64)
            # the C loop visits every (row, column) pair inside the cell,
            # so the count is rows * columns -- not just the rows
            count[oy, ox] = (b - a) * (cb - ca)
    ink = np.where(count > 0, total / np.maximum(count, 1), 0.0)
    out = (1.0 - ink / 255.0).astype(np.float32)

    # 3x3 box blur on the final 16x12
    # The C code accumulates, for every output pixel, the neighbours
    # (y+dy, x+dx) that fall inside the image -- so a shift by dy=-1
    # means output y takes tmp[y-1], NOT tmp[y].
    tmp = out.copy()
    acc = np.zeros_like(out)
    c = np.zeros_like(out)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            oy0, oy1 = max(0, dy), 12 + min(0, dy)
            ox0, ox1 = max(0, dx), 16 + min(0, dx)
            sy0, sy1 = max(0, -dy), 12 + min(0, -dy)
            sx0, sx1 = max(0, -dx), 16 + min(0, -dx)
            acc[oy0:oy1, ox0:ox1] += tmp[sy0:sy1, sx0:sx1]
            c[oy0:oy1, ox0:ox1] += 1
    out = acc / np.maximum(c, 1)

    # x centroid alignment (clamped +-2 px)
    v = out * (out > 0.35)
    sw = v.sum()
    if sw > 1e-3:
        cx = (v * np.arange(16)[None, :]).sum() / sw
        delta = int(round(cx - (16 - 1) * 0.5))
        delta = max(-2, min(2, delta))
        if delta != 0:
            # the C code writes out[y][x] = tmp[y][x - delta]
            tmp = out.copy()
            out[:] = 0.0
            xs = slice(max(0, delta), 16 + min(0, delta))
            xsrc = slice(max(0, -delta), 16 + min(0, -delta))
            out[:, xs] = tmp[:, xsrc]
    # y centroid alignment (clamped +-2 px)
    v = out * (out > 0.35)
    syw = v.sum()
    if syw > 1e-3:
        cy = (v * np.arange(12)[:, None]).sum() / syw
        delta = int(round(cy - (12 - 1) * 0.5))
        delta = max(-2, min(2, delta))
        if delta != 0:
            tmp = out.copy()
            out[:] = 0.0
            ys = slice(max(0, delta), 12 + min(0, delta))
            ysrc = slice(max(0, -delta), 12 + min(0, -delta))
            out[ys, :] = tmp[ysrc, :]
    return out.astype(np.float32)


def variants(gray, H, W, x, y, w, h, base, shifts, scales):
    """Yield augmented 16x12 crops of the cell at (x, y, w, h).

    `base` is the row's offset inside the band, so the cell's absolute
    top-left in the band is (x, base + y).
    """
    outs = []
    for dx in range(-shifts, shifts + 1):
        for sc in scales:
            ww = max(4, int(round(w * sc)))
            x0 = x - (ww - w) // 2 + dx
            x1 = x0 + ww
            if x0 < 0 or x1 > W:
                continue
            ry = base + y
            if ry < 0 or ry + h > H:
                continue
            outs.append(resample_char_gray(gray, W, H, x0, ry, ww, h))
    return outs


def variant_plan(shifts, scales, dys):
    """Enumerate the geometric sampling plan as (dx, scale, dy) triples.

    The plan is built before any pixel is touched, so the caller can
    SUBSAMPLE it while keeping the crop count fixed.
    """
    plan = []
    for dx in range(-shifts, shifts + 1):
        for sc in scales:
            for dy in dys:
                plan.append((dx, sc, dy))
    return plan


def materialize(gray, H, W, x, y, w, h, base, plan):
    """Turn a sampling plan into 16x12 crops, skipping out-of-band cells."""
    outs = []
    for dx, sc, dy in plan:
        ww = max(4, int(round(w * sc)))
        x0 = x - (ww - w) // 2 + dx
        x1 = x0 + ww
        if x0 < 0 or x1 > W:
            continue
        ry = base + y + dy
        if ry < 0 or ry + h > H:
            continue
        outs.append(resample_char_gray(gray, W, H, x0, ry, ww, h))
    return outs


def subsample_uniform(items, k):
    """Pick exactly k items spread evenly over `items`.

    Even STRIDING (not random sampling) is used on purpose: the plan is
    ordered as (dx, scale, dy), so an even stride walks through the
    whole shift/scale/jitter space instead of clustering in one corner.
    """
    n = len(items)
    if k >= n:
        return list(items)
    if k <= 0:
        return []
    picked = []
    seen = set()
    for i in range(k):
        j = (i * n) // k
        while j in seen and j + 1 < n:
            j += 1
        if j in seen:
            continue
        seen.add(j)
        picked.append(items[j])
    return picked


def variants_wide(gray, H, W, x, y, w, h, base, shifts, scales):
    """A denser variant set for the RARE classes.

    The real corpus has 1485 `N` samples but only 180 `M` samples on
    line 1, and the classifier reads the minority class as the
    majority one (`M` -> `N`, 4 of 12 real `M` cells).  Class
    re-weighting and class re-sampling were both MEASURED AND REJECTED
    on this corpus (they push the false-accept rate up), so the fix has
    to be more DATA for the minority class, not a different weight on
    the same data.

    What this adds is geometric diversity for those cells only: a
    wider shift range, a finer scale ladder and a vertical jitter, so
    the minority class covers the sub-pixel placements the network
    actually sees at inference time.  The cells are the SAME real
    pixels -- only the sampling window moves.

    IMPORTANT (measured): the WIDE set must not simply REPLACE the
    ordinary one.  The ordinary set has (2*shifts+1) * len(scales)
    crops per cell, e.g. 5 * 3 = 15.  A wide plan of 9 * 7 * 3 = 189
    crops would multiply the corpus size by ~4x, and because the trainer
    samples the real half to a FIXED total, the extra slices crowd the
    majority classes out -- measured: line 2 collapsed to predicting
    digits only (exact 33 -> 0).  So the wide plan is SUBSAMPLED back
    to the ordinary crop count; the corpus size and the class priors
    stay exactly as they were, only the geometry coverage widens.
    """
    plan = variant_plan(shifts, scales, (-1, 0, 1))
    return materialize(gray, H, W, x, y, w, h, base, plan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tool", default=str(ROOT / "6_mrz_ocr/build/mrz_ocr_cnn_tool"))
    ap.add_argument("--gt", default=str(ROOT / "tests/photo_auto_gt.json"))
    ap.add_argument("--out", default="/tmp/real_aug.npz")
    ap.add_argument("--bands", default="/tmp/holdout_export2")
    ap.add_argument("--dump-prefix", default="/tmp/mrz_dump")
    ap.add_argument("--dump-dir", default="/tmp")
    ap.add_argument("--shifts", type=int, default=2)
    ap.add_argument("--scales", default="0.88,1.0,1.12")
    ap.add_argument("--only", default="")
    ap.add_argument("--rare-chars", default="",
                    help="comma-separated characters that get the denser "
                         "geometric sampling (e.g. 'M,J,Q,F,X,B,K,W,R,Z,D,S,L')")
    ap.add_argument("--rare-shifts", type=int, default=4)
    ap.add_argument("--rare-scales", default="0.80,0.88,0.94,1.0,1.06,1.12,1.20")
    args = ap.parse_args()

    rare_chars = {c.strip() for c in args.rare_chars.split(",") if c.strip()}
    rare_scales = [float(s) for s in args.rare_scales.split(",") if s.strip()]

    gts = {}
    gtmap = json.load(open(args.gt))["gt"]
    for k, v in gtmap.items():
        if not isinstance(v, dict):
            continue
        l1 = v.get("l1") or v.get("line1")
        l2 = v.get("l2") or v.get("line2")
        if l1 and l2 and len(l1) == 44 and len(l2) == 44:
            gts[Path(k).stem] = (l1, l2)

    scales = [float(s) for s in args.scales.split(",") if s.strip()]
    band_dir = Path(args.bands)

    X, Y, meta = [], [], []
    bands = sorted(band_dir.glob("*.ppm"))
    if args.only:
        bands = [b for b in bands if args.only in b.stem]
    if not bands:
        raise SystemExit(f"no bands matching {args.bands} / {args.only}")

    for band_path in bands:
        stem = band_path.stem
        if stem not in gts:
            continue
        l1, l2 = gts[stem]

        # clear the previous dump set
        for f in glob.glob(f"{args.dump_dir}/mrz_dump_*"):
            try:
                os.remove(f)
            except OSError:
                pass
        # The band the glyph sampler reads is NOT the input file: the
        # pipeline crops, upscales, deskews and possibly dewarpps it.
        # Ask the tool for the PROCESSED band so the augmentation works
        # on exactly the pixels the classifier saw.
        band_pgm = Path(args.dump_dir) / f"mrz_band_{stem}.pgm"
        env = dict(os.environ)
        env["MRZ_OCR_DUMP"] = "aug"
        env["MRZ_OCR_DUMP_DIR"] = args.dump_dir
        env["MRZ_OCR_BAND_DUMP"] = str(band_pgm)
        env["MRZ_OCR_LINE_MODELS"] = "1"
        env["MRZ_OCR_CRNN"] = "0"
        env["MRZ_OCR_FUSE"] = "0"
        env["MRZ_OCR_PAR"] = "0"
        try:
            subprocess.run([args.tool, str(band_path)], capture_output=True,
                           text=True, env=env, timeout=120)
        except subprocess.TimeoutExpired:
            print(f"  {stem}: tool timeout", flush=True)
            continue
        if not band_pgm.exists():
            print(f"  {stem}: no processed band dump", flush=True)
            continue
        gray = np.asarray(Image.open(band_pgm).convert("L"), dtype=np.uint8)
        H, W = gray.shape

        n = 0
        for li, truth in ((0, l1), (1, l2)):
            cands = candidate_sets(li, truth)
            for c, ch in enumerate(truth):
                if ch not in GI:
                    continue
                found = None
                for f in glob.glob(f"{args.dump_dir}/mrz_dump_aug_l{li}_c{c:02d}_*.ppm"):
                    m = DUMP_RE.match(Path(f).name)
                    if m:
                        found = m
                        break
                if not found:
                    continue
                x = int(found.group("x"))
                y = int(found.group("y"))
                w = int(found.group("w"))
                h = int(found.group("h") or 0)
                b = int(found.group("b") or 0)
                if h <= 0:
                    # the pre-h/b dump format: the segmenter always uses
                    # the full row height, which we cannot recover here
                    continue
                outs = variants(gray, H, W, x, y, w, h, b,
                                args.shifts, scales)
                # Rare classes get a denser geometric sampling: the
                # minority class must cover the same range of window
                # placements as the majority one, otherwise the network
                # learns "the common shape" and reads M as N.
                #
                # The wide plan is SUBSAMPLED back to the ordinary crop
                # count (measured, twice):
                #   * APPENDING the wide set inflates the corpus 62039
                #     -> 86819 and, because the trainer samples the real
                #     half to a FIXED total, the extra slices crowd the
                #     majority classes out -> line 2 collapsed to
                #     predicting digits only (exact 33 -> 0).
                #   * REPLACING with the full wide set has the same
                #     effect (the wide set is ~4x larger).
                # Subsampling keeps the corpus size and the class priors
                # byte-identical and only widens the geometric coverage
                # of the minority class.
                if ch in rare_chars:
                    plan = variant_plan(args.rare_shifts, rare_scales,
                                        (-1, 0, 1))
                    picked = subsample_uniform(plan, len(outs))
                    wide = materialize(gray, H, W, x, y, w, h, b, picked)
                    if wide:
                        outs = wide
                cand = cands.get(c, [ch])
                for o in outs:
                    X.append(o)
                    Y.append(GI[ch])
                    meta.append({"file": stem, "line": li, "col": c,
                                 "ch": ch, "cand": "".join(cand)})
                    n += 1
        print(f"  {stem}: {n} augmented slices", flush=True)

    if not X:
        raise SystemExit("no slices exported")
    np.savez(args.out,
             X=np.array(X, dtype=np.float32).reshape(-1, 12, 16),
             Y=np.array(Y, dtype=np.int64),
             meta=np.array([json.dumps(m) for m in meta]))
    cnt = Counter(m["ch"] for m in meta)
    print(f"\nwrote {args.out}: {len(X)} slices from "
          f"{len({m['file'] for m in meta})} photos")
    print("class histogram:",
          sorted(cnt.items(), key=lambda kv: -kv[1])[:40])


if __name__ == "__main__":
    main()
