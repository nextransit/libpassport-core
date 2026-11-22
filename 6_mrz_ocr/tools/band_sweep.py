#!/usr/bin/env python3
"""Score a weight/knob configuration on the PRODUCT-path bands, fast.

Why this exists
---------------
`tests/photo_direct_bench.py` is the authority, but it re-runs the whole
GUI pipeline (band location, restore, four engines) for every photo, so
one sweep point costs ~8 s.  This tool reuses the product-path bands
exported by `tools/export_holdout_slices.py` (the SAME array the C tool
receives in the product) and only re-runs the C recogniser, so one sweep
point costs ~0.6 s.

It reports, per engine configuration:
  * exact     -- both MRZ lines match the GT exactly (the user metric)
  * line1/line2 exact
  * char accuracy

Usage:
    python3 tools/band_sweep.py --fuse-mode arb
    python3 tools/band_sweep.py --set MRZ_BEAM_DEPTH=8 MRZ_BEAM_COLS=24
    python3 tools/band_sweep.py --grid MRZ_BEAM_DEPTH=4,6,8
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "6_mrz_ocr" / "build" / "mrz_ocr_cnn_tool"
BANDS = Path("/tmp/holdout_export2")
GT_FILES = [
    ROOT / "tests" / "gt_photo_passport.json",
    ROOT / "tests" / "photo_auto_gt.json",
]


def load_gt():
    gt = {}
    for f in GT_FILES:
        if not f.exists():
            continue
        d = json.loads(f.read_text())
        g = d.get("gt", d) if isinstance(d, dict) else d
        for k, v in g.items():
            if not isinstance(v, dict):
                continue
            l1 = v.get("l1") or v.get("line1")
            l2 = v.get("l2") or v.get("line2")
            if l1 and l2 and len(l1) == 44 and len(l2) == 44:
                # The band files are named after the source photo stem
                # ("2026-9-29-A.ppm"), the GT after the photo ("...jpg").
                gt[Path(k).stem] = (l1, l2)
    return gt


def run_one(env_extra, bands, tool):
    env = dict(os.environ)
    env.update({
        "MRZ_OCR_LINE_MODELS": "1",
        "MRZ_OCR_PAR": "0",
        "MRZ_OCR_CRNN": "0",
        "MRZ_OCR_FUSE": "0",
    })
    env.update(env_extra)
    out = {}
    for band in bands:
        r = subprocess.run([str(tool), str(band)], capture_output=True,
                           text=True, env=env, timeout=60)
        d = {}
        for line in r.stdout.splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                k = k.strip()
                # the C tool prefixes its report keys with "result."
                if k.startswith("result."):
                    k = k[len("result."):]
                d[k] = v.strip()
        out[band.stem] = d
    return out


def score(out, gt):
    n = ex = l1x = l2x = 0
    c1 = c2 = 0
    fails = []
    for stem, d in out.items():
        if stem not in gt:
            continue
        g1, g2 = gt[stem]
        p1 = d.get("line1", "")
        p2 = d.get("line2", "")
        n += 1
        for k in range(44):
            c1 += 1 if (p1[k] if k < len(p1) else "_") == g1[k] else 0
            c2 += 1 if (p2[k] if k < len(p2) else "_") == g2[k] else 0
        a = p1 == g1
        b = p2 == g2
        l1x += a
        l2x += b
        if a and b:
            ex += 1
        else:
            fails.append(stem)
    return {
        "n": n, "exact": ex, "l1": l1x, "l2": l2x,
        "acc1": c1 / max(1, n * 44), "acc2": c2 / max(1, n * 44),
        "fails": fails,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bands", default=str(BANDS))
    ap.add_argument("--tool", default=str(TOOL))
    ap.add_argument("--set", nargs="*", default=[],
                    help="ENV=VAL assignments")
    ap.add_argument("--grid", nargs="*", default=[],
                    help="ENV=v1,v2,... sweeps (cartesian)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    bands = sorted(Path(args.bands).glob("*.ppm"))
    if not bands:
        print(f"no bands in {args.bands}")
        return 2
    gt = load_gt()
    base = {}
    for a in args.set:
        k, _, v = a.partition("=")
        base[k] = v

    # build the cartesian grid
    keys, values = [], []
    for a in args.grid:
        k, _, vs = a.partition("=")
        keys.append(k)
        values.append(vs.split(","))

    combos = [dict(base)]
    if keys:
        import itertools
        combos = []
        for prod in itertools.product(*values):
            c = dict(base)
            c.update(dict(zip(keys, prod)))
            combos.append(c)

    for c in combos:
        out = run_one(c, bands, args.tool)
        s = score(out, gt)
        desc = " ".join(f"{k}={v}" for k, v in sorted(c.items())) or "(default)"
        print(f"{desc:56s} exact={s['exact']:2d}/{s['n']} "
              f"l1={s['l1']:2d} l2={s['l2']:2d} "
              f"acc={s['acc1']*100:5.2f}/{s['acc2']*100:5.2f}")
        if not args.quiet and s["fails"]:
            print(f"    fails: {' '.join(s['fails'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
