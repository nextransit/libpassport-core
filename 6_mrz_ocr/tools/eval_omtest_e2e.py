#!/usr/bin/env python3
"""End-to-end evaluation of the photo -> MRZ pipeline on real photos.

This is the metric that matters for the product, and it is deliberately
NOT the per-glyph accuracy: a line can be mostly right and still be
worthless to a user, and a line can be wrong while the checksum happens
to accept it.

What it measures
----------------
  located     the two MRZ lines were found at all
  line1 exact the decoded line 1 equals the ICAO-corrected board label
  line2 in set the decoded line 2 equals one of the checksum-valid
              readings of the board label
  cksum       all five ICAO check digits of line 2 are satisfied
  TRUE        cksum AND in-set -- a usable, correct result
  FALSE       cksum but NOT in-set -- the pipeline accepted a line that
              is not in the truth set.  These are the dangerous ones:
              they look like a success to the caller.

Truth policy
------------
The board's own OCR is a hypothesis, not ground truth: for 27 of the 41
labelled photos its line 2 does not even satisfy the checksum, and for
a few of them the reading is ambiguous -- several characters are
equally consistent with the checksum.  The truth set for line 2 is
therefore the SET of checksum-valid readings reachable from the board
label within two confusion edits, and a prediction counts as correct
when it equals any member of that set.  Samples whose set has exactly
one member are unambiguous and are reported separately.

Usage
-----
  python3 tools/eval_omtest_e2e.py [--omtest DIR] [--tool PATH] [--json OUT]
"""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import export_omtest_slices as E  # noqa: E402

ROOT = HERE.parent
DEFAULT_OMTEST = ROOT.parent / "tmp" / "omtest"
DEFAULT_TOOL = ROOT / "build" / "mrz_ocr_cnn_tool"

W = [7, 3, 1]


def digit_for(seg: str) -> int:
    t = 0
    for i, c in enumerate(seg):
        v = int(c) if "0" <= c <= "9" else (ord(c) - 55 if "A" <= c <= "Z" else 0)
        t += v * W[i % 3]
    return t % 10


def line2_ok(l2: str) -> bool:
    if len(l2) < 44:
        return False
    try:
        return (
            digit_for(l2[0:9]) == int(l2[9])
            and digit_for(l2[13:19]) == int(l2[19])
            and digit_for(l2[21:27]) == int(l2[27])
            and digit_for(l2[28:42]) == int(l2[42])
            and digit_for(l2[0:10] + l2[13:20] + l2[21:43]) == int(l2[43])
        )
    except ValueError:
        return False


# ---------------------------------------------------------------- locating
# Mirrors PhotoMrzPipeline.fastPair: a row projection of the lower half
# of the photo, sampling every 4th column, then the run-picking rules.
def fast_pair(gray: np.ndarray):
    h, w = gray.shape
    if w < 80 or h < 120:
        return None
    y0 = int(h * 0.55)
    y1 = min(int(h * 0.96), h)
    sx0 = int(w * 0.03)
    sx1 = int(w * 0.97)
    stride = 4
    sample_w = (sx1 - sx0 + stride - 1) // stride
    if y1 <= y0 or sample_w < 40:
        return None
    dark = np.zeros(y1 - y0, dtype=np.int64)
    for row in range(y1 - y0):
        c = 0
        x = sx0
        while x < sx1:
            if gray[y0 + row, x] < 170:
                c += 1
            x += stride
        dark[row] = c
    min_ink = max(3, int(sample_w * 0.18))
    max_ink = int(sample_w * 0.88)
    runs = []
    start = -1
    gap = 0
    for i, d in enumerate(dark):
        active = min_ink <= d <= max_ink
        if active:
            if start < 0:
                start = i
            gap = 0
        elif start >= 0:
            gap += 1
            if gap > 2:
                runs.append((start, i - gap))
                start = -1
                gap = 0
    if start >= 0:
        runs.append((start, len(dark) - 1))
    solid_min = int(sample_w * 0.90)
    solid = []
    s = -1
    for i in range(len(dark) + 1):
        sol = i < len(dark) and dark[i] >= solid_min
        if sol:
            if s < 0:
                s = i
        elif s >= 0:
            if i - s >= 8:
                solid.append((s, i - 1))
            s = -1

    def pick(before=None):
        best = None

        def ok(a, b):
            if before is not None and b[1] >= before - 2:
                return False
            h1 = a[1] - a[0] + 1
            h2 = b[1] - b[0] + 1
            gr = b[0] - a[1] - 1
            if h1 < 5 or h1 > 100 or h2 < 5 or h2 > 100:
                return False
            if not (0.35 <= h2 / h1 <= 2.5):
                return False
            if gr < 0 or gr > max(12, int(min(h1, h2) * 2 + 4)):
                return False
            return True

        for i in range(len(runs) - 1):
            a = runs[i]
            h1 = a[1] - a[0] + 1
            if h1 < 5 or h1 > 100:
                continue
            for j in range(i + 1, min(len(runs) - 1, i + 3) + 1):
                b = runs[j]
                h2 = b[1] - b[0] + 1
                mx = max(2, min(h1, h2) // 3)
                if any(r[1] - r[0] + 1 > mx for r in runs[i + 1:j]):
                    continue
                if not ok(a, b):
                    continue
                if best is None or b[1] > best[1][1]:
                    best = (a, b)
        return best

    bb = None
    for so in solid:
        c = pick(so[0])
        if c is not None and (bb is None or c[1][1] > bb[1][1]):
            bb = c
    if bb is None:
        bb = pick()
    if bb is None:
        return None
    return (y0 + bb[0][0], y0 + bb[0][1]), (y0 + bb[1][0], y0 + bb[1][1])


def adaptive_xpad(gray, pair):
    h, w = gray.shape
    if w < 40:
        nominal = min(100, max(10, int(round(w * 0.08))))
        return nominal, nominal
    strip = max(4, min(12, w // 25))
    def frac(x0, x1):
        blk = gray[:, x0:x1]
        return float((blk < 160).sum()) / blk.size
    left_dark = frac(0, strip)
    right_dark = frac(w - strip, w)
    nominal = min(100, max(10, int(round(w * 0.08))))
    left = nominal if left_dark >= 0.9 else 0
    right = nominal if right_dark >= 0.9 else 0
    return left, right


def native_band(gray, first, second, xpad_l, xpad_r):
    h, w = gray.shape
    fy0, fy1 = first
    sy0, sy1 = second
    first_source_y1 = min(h, fy1 + 1)
    rows = []
    for (ry0, ry1) in (first, second):
        s0 = max(0, ry0)
        s1 = min(h, ry1 + 1)
        if (ry0, ry1) == second:
            s0 = max(s0, first_source_y1)
        row_h = max(1, s1 - s0)
        pad = max(8, min(25, int(round(row_h * 0.70))))
        rows.append((s0, s1, pad))
    first_top = max(0, fy0)
    second_bottom = min(h, sy1 + 1)
    shared_h = max((r[1] + r[2]) - (r[0] - r[2]) for r in rows)
    shared_pad = max(8, min(25, int(round(shared_h * 0.70))))
    return max(0, first_top - shared_pad), min(h, second_bottom + shared_pad)


FAST_NATIVE_MAX_WIDTH = 1600


def run_tool(tool: Path, band: Path):
    env = dict(os.environ)
    env["MRZ_OCR_LINE_MODELS"] = "1"
    r = subprocess.run([str(tool), str(band)], capture_output=True, text=True,
                       env=env, timeout=60)
    kv = {}
    for ln in r.stdout.splitlines():
        if ":" in ln:
            k, _, v = ln.partition(":")
            kv[k.strip()] = v.strip()
    return kv


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--omtest", default=str(DEFAULT_OMTEST))
    ap.add_argument("--tool", default=str(DEFAULT_TOOL))
    ap.add_argument("--json", default="")
    args = ap.parse_args()
    om = Path(args.omtest)
    tool = Path(args.tool)
    if not tool.exists():
        print(f"tool not found: {tool}", file=sys.stderr)
        return 2
    tmp = Path("/tmp/mrz_eval_e2e")
    tmp.mkdir(parents=True, exist_ok=True)

    rows = []
    for d in sorted(p for p in om.iterdir() if p.is_dir()):
        for txt in sorted((om / d.name).glob("*.jpg-result.txt")):
            img = txt.with_name(txt.name.replace("-result.txt", ""))
            if not img.exists():
                continue
            l1, l2 = E.parse_board(txt)
            if not l1 or not l2:
                continue
            truth1 = E.fix_line1(l1)
            # `truth1` is None when the board's reading of line 1 cannot
            # be repaired into a legal TD3 line (a misread document type
            # or a filler slot).  Such a sample has NO line-1 truth, so
            # counting it as a line-1 failure would understate the
            # metric: on the 41-photo corpus 7 of the samples have no
            # line-1 truth at all.
            cands = E.all_valid_variants(l2, max_edits=2)
            if not cands:
                continue
            im = Image.open(img).convert("RGB")
            a = np.asarray(im, dtype=np.uint32)
            gray = ((a[:, :, 0] * 299 + a[:, :, 1] * 587 + a[:, :, 2] * 114 + 500)
                    // 1000).astype(np.uint8)
            pair = fast_pair(gray)
            row = dict(file=f"{d.name}/{img.name}", n_cand=len(cands),
                       hard=(len(cands) == 1))
            if pair is None:
                row.update(located=False)
                rows.append(row)
                continue
            xl, xr = adaptive_xpad(gray, pair)
            top, bot = native_band(gray, pair[0], pair[1], xl, xr)
            band = im.crop((xl, top, im.width - xr, bot))
            if band.width > FAST_NATIVE_MAX_WIDTH:
                nh = max(1, int(round(band.height * FAST_NATIVE_MAX_WIDTH / band.width)))
                band = band.resize((FAST_NATIVE_MAX_WIDTH, nh), Image.BILINEAR)
            p = tmp / (img.stem + "_band.ppm")
            band.save(p)
            kv = run_tool(tool, p)
            pred1 = kv.get("result.line1", "")
            pred2 = kv.get("result.line2", "")
            row.update(located=True, line1=pred1, line2=pred2,
                       line1_exact=(pred1 == truth1) if truth1 else None,
                       l2_in=(pred2 in cands), cksum=line2_ok(pred2),
                       truth1=truth1)
            rows.append(row)

    located = [r for r in rows if r.get("located")]
    hard = [r for r in located if r.get("hard")]
    # Only the samples that HAVE a line-1 truth count towards the
    # line-1 metric.
    l1_eval = [r for r in located if r.get("truth1")]
    l1x = sum(1 for r in l1_eval if r.get("line1_exact"))
    l2i = sum(1 for r in located if r.get("l2_in"))
    ck = sum(1 for r in located if r.get("cksum"))
    ts = sum(1 for r in located if r.get("cksum") and r.get("l2_in"))
    fs = sum(1 for r in located if r.get("cksum") and not r.get("l2_in"))
    print(f"samples={len(rows)}  located={len(located)}  "
          f"unambiguous(line2 truth set size 1)={len(hard)}")
    print(f"  line1 exact        = {l1x}/{len(l1_eval)}"
          + (f"  (of {len(located)} located; {len(located) - len(l1_eval)}"
             f" have no line-1 truth)" if len(l1_eval) != len(located) else ""))
    print(f"  line2 in truth set = {l2i}/{len(located)}")
    print(f"  line2 checksum ok  = {ck}/{len(located)}")
    print(f"  TRUE  (cksum + in-set) = {ts}/{len(located)}")
    print(f"  FALSE (cksum, not in-set) = {fs}")
    if hard:
        print(f"  -- unambiguous subset: in-set="
              f"{sum(1 for r in hard if r.get('l2_in'))}/{len(hard)}  "
              f"false={sum(1 for r in hard if r.get('cksum') and not r.get('l2_in'))}")
    print()
    print("misses:")
    for r in rows:
        if r.get("located") and r.get("cksum") and r.get("l2_in"):
            continue
        why = "not located" if not r.get("located") else (
            "checksum failed" if not r.get("cksum") else "not in truth set")
        print(f"  {r['file']:32s} {why:20s} truth-set size {r['n_cand']}")
        if r.get("located"):
            print(f"      line2 = {r.get('line2')}")
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=1, default=str))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
