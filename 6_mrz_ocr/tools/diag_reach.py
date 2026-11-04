#!/usr/bin/env python3
"""Reachability diagnostic for the residual line-2 errors.

For each failing image this dumps the per-column top-3 candidates
(MRZ_OCR_TOPK=1) on the SAME band the production path feeds the
recogniser, then checks whether the ground-truth character is among them.

Why this matters
----------------
The beam search can only find a correction the CNN already placed in its
top-3.  If the truth character is absent from a column's candidate list,
no amount of beam depth or column budget will recover it -- the fix has
to be in the classifier or the band, not in the search.  Separating the
two cases before touching the beam is what keeps the optimisation honest.

The band is produced by the exact same calls the production evaluation
uses (EE.fast_pair -> EE.adaptive_xpad -> EE.native_band), and the tool
runs with the same environment (MRZ_OCR_LINE_MODELS=1, MRZ_OCR_PAR=1), so
the `got` line here is byte-identical to eval_prod_path.py's.

Usage:
    python3 tools/diag_reach.py [image-name ...]
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

import eval_omtest_e2e as EE  # noqa: E402
import export_omtest_slices as E  # noqa: E402
import eval_prod_path as EP  # noqa: E402

TOOL = ROOT / "build" / "mrz_ocr_cnn_tool"
OMTEST = EE.DEFAULT_OMTEST
FAST_NATIVE_MAX_WIDTH = EE.FAST_NATIVE_MAX_WIDTH
TMP = Path("/tmp/mrz_reach")


def make_band(img_path: Path) -> Image.Image | None:
    """Reproduce eval_prod_path.py's band crop exactly."""
    im = Image.open(img_path).convert("RGB")
    a = np.asarray(im, dtype=np.uint32)
    gray = ((a[:, :, 0] * 299 + a[:, :, 1] * 587 + a[:, :, 2] * 114 + 500)
            // 1000).astype(np.uint8)
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


def run_topk(band_path: Path, extra_env: dict | None = None):
    env = dict(os.environ)
    env["MRZ_OCR_LINE_MODELS"] = "1"
    env["MRZ_OCR_PAR"] = "1"
    env["MRZ_OCR_TOPK"] = "1"
    if extra_env:
        env.update(extra_env)
    import subprocess
    p = subprocess.run([str(TOOL), str(band_path)], capture_output=True,
                       text=True, env=env, timeout=120)
    topk: dict[int, dict[int, list[tuple[str, float]]]] = {}
    for line in p.stderr.splitlines():
        m = re.match(r"TOPK L(\d) c(\d+): (.)\(([\d.]+)\) (.)\(([\d.]+)\) (.)\(([\d.]+)\)", line)
        if m:
            li, c = int(m.group(1)), int(m.group(2))
            topk.setdefault(li, {})[c] = [
                (m.group(3), float(m.group(4))),
                (m.group(5), float(m.group(6))),
                (m.group(7), float(m.group(8))),
            ]
    kv = {}
    for ln in p.stdout.splitlines():
        if ":" in ln:
            k, _, v = ln.partition(":")
            kv[k.strip()] = v.strip()
    return topk, kv, p


def find_image(name: str) -> Path | None:
    hits = list(OMTEST.rglob(Path(name).stem + ".jpg"))
    return hits[0] if hits else None


def main() -> int:
    data = json.loads(EP.GT_CORRECTED.read_text())
    recs = {}
    for r in data["records"]:
        recs[Path(r["id"]).name] = r["line2"]

    targets = sys.argv[1:] or [
        "2026-9-29-M.jpg",
        "2026-9-30-AC.jpg",
        "2026-9-30-AC_R.jpg",
        "2026-9-30-AG.jpg",
    ]
    TMP.mkdir(parents=True, exist_ok=True)

    for name in targets:
        truth = recs.get(Path(name).name)
        if not truth:
            print(f"{name}: not in corrected GT")
            continue
        img = find_image(name)
        if img is None:
            print(f"{name}: image not found")
            continue
        band = make_band(img)
        if band is None:
            print(f"{name}: band not located")
            continue
        bp = TMP / (Path(name).stem + "_band.ppm")
        band.save(bp)
        topk, kv, _ = run_topk(bp)
        got = kv.get("result.line2", "")

        print(f"\n=== {name} ===  band={band.width}x{band.height}")
        print(f"  got   {got}")
        print(f"  truth {truth}")
        if len(got) != 44:
            print(f"  got is not 44 chars (len={len(got)}) -- skipping")
            continue
        cols = [c for c in range(44) if got[c] != truth[c]]
        print(f"  differing columns: {cols}")
        absent = 0
        for c in cols:
            cands = topk.get(1, {}).get(c, [])
            chars = [x[0] for x in cands]
            present = truth[c] in chars
            if not present:
                absent += 1
            rank = chars.index(truth[c]) if present else -1
            tag = f"REACHABLE rank={rank}" if present else "*** ABSENT ***"
            print(f"    col {c:2d}: got={got[c]!r} truth={truth[c]!r} "
                  f"cands={[(x[0], round(x[1], 3)) for x in cands]} {tag}")
        print(f"  --> {len(cols)} diffs, {absent} absent from top-3")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
