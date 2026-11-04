#!/usr/bin/env python3
"""Can a per-column confidence floor separate wrong readings from right ones?

The line-level confidence (`conf2`) is an average over 44 columns, so a
single coin-flip column is invisible in it: `2026-9-30-AC` reads a wrong
personal number with conf2=96 and ambiguous=0, and is reported as PASS.

This script dumps the per-column top-3 for every image in the corrected
photo corpus, then compares the distribution of two candidate statistics
between the readings that are exactly right and the ones that are not:

    min_top1   the lowest top-1 probability on any payload column
    min_margin the smallest top1-top2 gap on any payload column

Payload columns are the ones the five check digits are supposed to
protect: 0-8 (document number), 13-18 (birth), 21-26 (expiry),
28-41 (personal number).  Check digits (9,19,27,42,43), nationality
(10-12) and sex (20) are excluded: the first group is derived, and the
other two are not protected by any check digit, so a low margin there
carries no information about the payload.

If the two distributions overlap, no threshold exists and the honest
conclusion is that the classifier needs better features -- not that the
gate should be tightened.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

import eval_omtest_e2e as EE  # noqa: E402
import eval_prod_path as EP  # noqa: E402

TOOL = ROOT / "build" / "mrz_ocr_cnn_tool"
OMTEST = EE.DEFAULT_OMTEST
FAST_NATIVE_MAX_WIDTH = EE.FAST_NATIVE_MAX_WIDTH
TMP = Path("/tmp/mrz_colstat")

PAYLOAD = list(range(0, 9)) + list(range(13, 19)) + list(range(21, 27)) + list(range(28, 42))


def make_band(img_path: Path) -> Image.Image | None:
    im = Image.open(img_path).convert("RGB")
    a = np.asarray(im, dtype=np.uint32)
    gray = ((a[:, :, 0] * 299 + a[:, :, 1] * 587 + a[:, :, 2] * 114 + 500) // 1000).astype(np.uint8)
    pair = EE.fast_pair(gray)
    if pair is None:
        return None
    xl, xr = EE.adaptive_xpad(gray, pair)
    top, bot = EE.native_band(gray, pair[0], pair[1], xl, xr)
    band = im.crop((xl, top, im.width - xr, bot))
    if band.width > FAST_NATIVE_MAX_WIDTH:
        nh = max(1, int(round(band.height * FAST_NATIVE_MAX_WIDTH / band.width)))
        band = band.resize((FAST_NATIVE_MAX_WIDTH, nh), Image.BILINEAR)
    return band


def run(band_path: Path):
    env = dict(os.environ)
    env["MRZ_OCR_LINE_MODELS"] = "1"
    env["MRZ_OCR_PAR"] = "1"
    env["MRZ_OCR_TOPK"] = "1"
    p = subprocess.run([str(TOOL), str(band_path)], capture_output=True,
                       text=True, env=env, timeout=120)
    t3: dict[int, list[tuple[str, float]]] = {}
    for line in p.stderr.splitlines():
        m = re.match(r"TOPK L1 c(\d+): (.)\(([\d.]+)\) (.)\(([\d.]+)\) (.)\(([\d.]+)\)", line)
        if m:
            c = int(m.group(1))
            t3[c] = [(m.group(2), float(m.group(3))),
                     (m.group(4), float(m.group(5))),
                     (m.group(6), float(m.group(7)))]
    kv = {}
    for ln in p.stdout.splitlines():
        if ":" in ln:
            k, _, v = ln.partition(":")
            kv[k.strip()] = v.strip()
    return t3, kv


def main() -> int:
    gt = {Path(r["id"]).name: r["line2"]
          for r in json.loads(EP.GT_CORRECTED.read_text())["records"]}
    TMP.mkdir(parents=True, exist_ok=True)

    rows = []
    for name, truth in sorted(gt.items()):
        hits = list(OMTEST.rglob(Path(name).stem + ".jpg"))
        if not hits:
            continue
        band = make_band(hits[0])
        if band is None:
            continue
        bp = TMP / (Path(name).stem + "_band.ppm")
        band.save(bp)
        t3, kv = run(bp)
        got = kv.get("result.line2", "")
        if len(got) != 44 or len(t3) < 44:
            continue
        top1 = [t3[c][0][1] for c in PAYLOAD]
        marg = [t3[c][0][1] - t3[c][1][1] for c in PAYLOAD]
        rows.append(dict(
            name=name, got=got, truth=truth, exact=(got == truth),
            conf2=int(kv.get("result.conf2", 0)),
            min_top1=min(top1), min_margin=min(marg),
            # which column is the weakest, and is it one of the wrong ones?
            weak_col=PAYLOAD[int(np.argmin(marg))],
            wrong_cols=[c for c in range(44) if got[c] != truth[c]],
        ))

    exact = [r for r in rows if r["exact"]]
    wrong = [r for r in rows if not r["exact"]]
    print(f"images={len(rows)}  exact={len(exact)}  wrong={len(wrong)}\n")

    for stat in ("min_top1", "min_margin", "conf2"):
        ev = sorted(r[stat] for r in exact)
        wv = sorted(r[stat] for r in wrong)
        print(f"{stat}:")
        print(f"  exact  min={ev[0]:.3f} p10={ev[len(ev)//10]:.3f} median={ev[len(ev)//2]:.3f}")
        if wv:
            print(f"  wrong  min={wv[0]:.3f} p10={wv[len(wv)//10]:.3f} median={wv[len(wv)//2]:.3f}")
            # how many exact readings fall below the highest wrong one?
            hi = wv[-1]
            below = sum(1 for v in ev if v < hi)
            print(f"  worst wrong={hi:.3f}; exact readings below it = {below}/{len(ev)}")
        print()

    print("per-image detail (sorted by min_margin):")
    for r in sorted(rows, key=lambda x: x["min_margin"]):
        flag = "OK " if r["exact"] else "BAD"
        print(f"  {flag} {r['name']:32s} conf2={r['conf2']:3d} "
              f"min_top1={r['min_top1']:.3f} min_margin={r['min_margin']:.3f} "
              f"weak_col={r['weak_col']:2d} wrong={r['wrong_cols']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
