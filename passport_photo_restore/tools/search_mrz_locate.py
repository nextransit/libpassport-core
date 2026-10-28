#!/usr/bin/env python3
"""Search the _mrz_pair_pass rule space for the best MRZ row locator.

Why
---
`tools/eval_mrz_locate.py` shows the working tree beats HEAD on the
board-verified assets/pic samples (35/48 vs 32/48), but both miss the
same handful.  The reason is structural, and this script makes it
explicit:

  * A real MRZ row is 44 separate glyphs: many short ink runs
    ("groups") and NO long continuous run.
  * A page border, a table rule, or a row of dense data text is the
    opposite: few groups and/or one very long continuous run.

So a row qualifies as an MRZ line when
    groups >= MIN_GROUPS   and   max_run <= MAX_RUN_FRAC * width
and the pair qualifies when the two rows have comparable heights and a
gap that is a sane fraction of the row height.

The knobs searched here are exactly those three numbers, plus the ink
window that decides which rows are even considered.  Everything is
scored against the board's own crop (see eval_mrz_locate.py) -- never
against a box derived by the same projection rule being tuned.

Usage
-----
    python3 tools/search_mrz_locate.py            # print the best combos
    python3 tools/search_mrz_locate.py --top 20
"""
from __future__ import annotations

import argparse
import itertools
import json
import pathlib
import sys

import cv2
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "passport_photo_restore" / "src"))

CATALOG = ROOT / "8_android" / "app" / "src" / "main" / "assets" / "pic" / "catalog.json"
PIC_DIR = ROOT / "8_android" / "app" / "src" / "main" / "assets" / "pic"


def run_groups(mask: np.ndarray):
    """Contiguous True runs -> list of (start, end)."""
    runs = []
    s = None
    for i, v in enumerate(mask):
        if v and s is None:
            s = i
        elif not v and s is not None:
            runs.append((s, i - 1))
            s = None
    if s is not None:
        runs.append((s, len(mask) - 1))
    return runs


class Params:
    def __init__(self, ink_lo, ink_hi, min_groups, max_run_frac,
                 gap_lo, gap_hi):
        self.ink_lo = ink_lo
        self.ink_hi = ink_hi
        self.min_groups = min_groups
        self.max_run_frac = max_run_frac
        self.gap_lo = gap_lo
        self.gap_hi = gap_hi


def detect(g, thr, p: Params):
    H, W = g.shape
    row_ink180 = (g < 180).sum(axis=1)
    E = np.abs(np.diff(g.astype(np.int16), axis=1)).sum(axis=1)
    row_ink_s = (g < thr).sum(axis=1)
    text = (row_ink_s > p.ink_lo * W) & (row_ink_s < p.ink_hi * W) & (E > 1800)
    runs = run_groups(text)
    runs = [(a, b) for a, b in runs if b - a + 1 >= 4]

    def row_ok(t):
        y0, y1 = t
        h = y1 - y0 + 1
        if not (5 <= h <= 48):
            return False
        col = (g[y0:y1 + 1] < thr).sum(axis=0)
        strong = col >= 4
        groups = run_groups(strong)
        if len(groups) < p.min_groups:
            return False
        mx = max((b - a + 1 for a, b in groups), default=0)
        if mx > max(160, int(W * p.max_run_frac)):
            return False
        return True

    cands = []
    for i, t1 in enumerate(runs[:-1]):
        t2 = runs[i + 1]
        h1 = t1[1] - t1[0] + 1
        h2 = t2[1] - t2[0] + 1
        gap = t2[0] - t1[1] - 1
        if not row_ok(t1) or not row_ok(t2):
            continue
        if not (0.4 <= h2 / h1 <= 1.6):
            continue
        ratio = gap / max(1, min(h1, h2))
        if not (p.gap_lo <= ratio <= p.gap_hi):
            continue
        cands.append(((-min(h1, h2), gap), t1, t2))
    if not cands:
        return None
    cands.sort(key=lambda x: x[0])
    return cands[0][1], cands[0][2]


def locate(g, p: Params):
    best = None
    for thr in (170, 160, 150):
        r = detect(g, thr, p)
        if r and (best is None or r[0][0] > best[0][0]):
            best = r
    return best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=12)
    args = ap.parse_args()

    cat = json.loads(CATALOG.read_text())
    samples = []
    for rec in cat["records"]:
        ocr = rec.get("ocr") or {}
        crop = ocr.get("crop")
        if not ocr.get("ok") or not crop:
            continue
        path = PIC_DIR / rec["file"]
        if not path.exists():
            continue
        rgb = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        work, scale = sys.modules["restore"].load(rgb)
        samples.append({
            "file": rec["file"],
            "g": sys.modules["restore"]._gray(work),
            "scale": scale,
            "crop": crop,
        })

    grid = {
        "ink_lo": [0.12, 0.15, 0.20],
        "ink_hi": [0.90, 0.95, 0.98],
        "min_groups": [30, 35, 40, 45],
        "max_run_frac": [0.30, 0.35, 0.45],
        "gap_lo": [0.20, 0.25, 0.30],
        "gap_hi": [1.6, 2.0, 2.5],
    }
    keys = list(grid)
    results = []
    for combo in itertools.product(*(grid[k] for k in keys)):
        p = Params(*combo)
        hits = 0
        for s in samples:
            r = locate(s["g"], p)
            if not r:
                continue
            y0 = s["crop"]["y"]
            y1 = y0 + s["crop"]["h"]
            src = tuple((int(a / s["scale"]), int(b / s["scale"])) for a, b in r)
            if y0 - 8 <= src[0][0] <= y1 and y0 - 8 <= src[1][0] <= y1:
                hits += 1
        results.append((hits, dict(zip(keys, combo))))

    results.sort(key=lambda x: -x[0])
    n = len(samples)
    print(f"样本 {n} 个 (板端 OCR 成功的 assets/pic)")
    print(f"HEAD 基线 32/{n}   当前工作区 35/{n}\n")
    print(f"{'命中':<10}参数")
    for hits, cfg in results[:args.top]:
        print(f"{hits}/{n:<8} {cfg}")
    return 0


if __name__ == "__main__":
    import restore  # noqa: E402  (needs sys.path set above)
    raise SystemExit(main())
