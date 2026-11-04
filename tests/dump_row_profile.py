#!/usr/bin/env python3
"""Dump row darkness profile for a specific photo's band crop."""
from __future__ import annotations
import os, sys, subprocess, re
from pathlib import Path
from PIL import Image
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "passport_photo_restore" / "src"))
from restore import _gray, detect_structure

def main():
    target = sys.argv[1] if len(sys.argv) > 1 else "IMG_1792133000_003.jpg"
    photo_dirs = [ROOT / "tmp" / "PassPortPhoto", ROOT / "Photo"]
    photo = None
    for d in photo_dirs:
        if d.exists():
            p = list(d.rglob(target))
            if p: photo = p[0]; break
    if not photo:
        print(f"Not found: {target}"); return

    rgb = np.array(Image.open(str(photo)).convert("RGB"))
    work = rgb
    if max(rgb.shape[:2]) > 1250:
        sf = 1250.0 / max(rgb.shape[:2])
        work = np.array(Image.fromarray(rgb).resize((int(rgb.shape[1]*sf), int(rgb.shape[0]*sf)), Image.BILINEAR))
    st = detect_structure(_gray(work))
    pair = st.get("mrz_pair")
    if not pair:
        single = st.get("mrz_single")
        if single: pair = (single, single)
        else: print("NO PAIR"); return
    is_single = pair[0] == pair[1]
    scale = work.shape[0] / max(1, rgb.shape[0])
    H, W = rgb.shape[:2]
    first_top = max(0, int(round(pair[0][0] / scale)))
    second_bottom = min(H, int(round((pair[1][1] + 1) / scale)))
    heights = [max(1, int(round((y1 + 1) / scale)) - int(round(y0 / scale))) for y0, y1 in pair]
    if is_single:
        top, bottom = first_top, second_bottom
    else:
        pad = max(8, min(25, int(round(max(heights) * 0.70))))
        top = max(0, first_top - pad); bottom = min(H, second_bottom + pad)
    band = rgb[top:bottom, 0:W]
    gray = np.mean(band, axis=2)
    Hb, Wb = gray.shape
    row_dark = np.sum(gray < 128, axis=1)
    peak = np.max(row_dark)
    gap_thr = max(4, peak // 6)

    print(f"Photo: {target}")
    print(f"Band: {Hb}x{Wb}, peak={peak}, gap_thr={gap_thr}")
    print(f"{'y':>3} {'dark':>5} {'pct':>5} {'gap?':>4}  {'bar'}")
    print("-" * 70)
    for y in range(Hb):
        pct = row_dark[y] / Wb * 100
        is_gap = row_dark[y] < gap_thr
        bar = "#" * int(pct)
        gap_mark = "GAP" if is_gap else ""
        print(f"{y:>3} {row_dark[y]:>5} {pct:>5.1f} {gap_mark:>4}  {bar}")

    print(f"\nCandidate gaps (gap_h >= {Hb//25}):")
    min_half = Hb // 5
    for y in range(1, Hb - 1):
        gh = 1
        while y + gh < Hb and row_dark[y + gh] < gap_thr: gh += 1
        if gh < Hb // 25: continue
        if y < min_half or (Hb - (y + gh)) < min_half: continue
        h1 = y
        h2 = Hb - (y + gh)
        ratio = h1 / h2 if h2 > 0 else 0
        even = min(h1, h2) / max(h1, h2) if max(h1, h2) > 0 else 0
        score = gh * even
        print(f"  y={y:>3} gap_h={gh:>2} h1={h1:>3} h2={h2:>3} ratio={ratio:.2f} even={even:.2f} score={score:.1f}")

if __name__ == "__main__":
    main()
