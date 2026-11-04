#!/usr/bin/env python3
"""Diagnose the 3 failing pic images: show band crop, row projection,
split_lines result, and segment_line result."""
import sys, os, subprocess, tempfile
from pathlib import Path
from PIL import Image
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_photo_train import locate_band

TOOL = ROOT / "6_mrz_ocr/build/mrz_ocr_cnn_tool_pic"
PAD = 10

FAILING = [
    "1792f7a2d79c7a8a_0017.jpg",
    "72ef4ce48bf27731_0002.jpg",
    "927cd546b5a8111d_0108.jpg",
]

def sauvola_binarize(gray, window=25, k=0.2):
    """Simple Sauvola binarization for diagnosis."""
    arr = np.asarray(gray, dtype=np.float64)
    from scipy.ndimage import uniform_filter
    mean = uniform_filter(arr, size=window, mode='reflect')
    sq_mean = uniform_filter(arr*arr, size=window, mode='reflect')
    std = np.sqrt(np.maximum(sq_mean - mean*mean, 0))
    threshold = mean * (1 + k * (std / 128 - 1))
    return (arr < threshold).astype(np.uint8)

def row_projection(bin_img):
    """Count dark pixels per row."""
    return bin_img.sum(axis=1)

def diagnose(name):
    img_path = ROOT / "assets/pic" / name
    rgb = Image.open(img_path).convert("RGB")
    print(f"\n{'='*70}")
    print(f"Image: {name}  size={rgb.size}")
    
    crop = locate_band(rgb)
    if crop is None:
        print("  locate_band FAILED (None)")
        return
    bx, by, bw, bh = crop
    print(f"  locate_band: x={bx} y={by} w={bw} h={bh}")
    
    band = rgb.crop((bx, max(0, by - PAD), bx + bw,
                     min(rgb.height, by + bh + PAD)))
    print(f"  band crop (with PAD={PAD}): size={band.size}")
    
    # Save band for inspection
    out_dir = ROOT / "tmp/diag_failing"
    out_dir.mkdir(parents=True, exist_ok=True)
    band.save(str(out_dir / f"{Path(name).stem}_band.png"))
    
    # Run C tool
    ppm = out_dir / f"{Path(name).stem}.ppm"
    band.save(str(ppm), format="PPM")
    
    env = dict(os.environ)
    env["MRZ_OCR_LINE_MODELS"] = "1"
    env["MRZ_OCR_PAR"] = "1"
    env["MRZ_OCR_CRNN"] = "0"
    p = subprocess.run([str(TOOL), str(ppm)], capture_output=True,
                       text=True, timeout=120, env=env)
    print(f"  C tool stdout:")
    for line in p.stdout.splitlines():
        print(f"    {line}")
    if p.stderr:
        print(f"  C tool stderr:")
        for line in p.stderr.splitlines()[:10]:
            print(f"    {line}")
    
    # Analyze binarization
    gray = band.convert("L")
    bin_img = sauvola_binarize(gray)
    proj = row_projection(bin_img)
    H, W = bin_img.shape
    print(f"\n  Binarization analysis (Sauvola): W={W} H={H}")
    print(f"  Row projection (dark pixels per row):")
    print(f"    peak = {proj.max()}")
    print(f"    mean = {proj.mean():.1f}")
    
    # Show projection as histogram
    peak = proj.max()
    gap_thr = max(peak // 6, 4)
    print(f"    gap_thr = peak/6 = {gap_thr}")
    
    # Find gap rows
    gap_rows = np.where(proj < gap_thr)[0]
    if len(gap_rows) > 0:
        # Find longest gap run in middle half
        min_half = H // 5
        best_gap_y, best_gap_h = -1, 0
        y = 0
        while y < H:
            if proj[y] < gap_thr:
                start = y
                while y < H and proj[y] < gap_thr:
                    y += 1
                gap_h = y - start
                if gap_h >= H // 25 and start >= min_half and (H - y) >= min_half:
                    if gap_h > best_gap_h:
                        best_gap_h = gap_h
                        best_gap_y = start
            else:
                y += 1
        if best_gap_y >= 0:
            h1 = best_gap_y
            h2 = H - (best_gap_y + best_gap_h)
            print(f"    Step 1 gap: y={best_gap_y} h={best_gap_h}  -> h1={h1} h2={h2}")
            if h1 * 2 >= h2 and h2 * 2 >= h1:
                print(f"    -> ACCEPTED (h1*2={h1*2} >= h2={h2}, h2*2={h2*2} >= h1={h1})")
            else:
                print(f"    -> REJECTED height ratio (h1*2={h1*2} vs h2={h2}, h2*2={h2*2} vs h1={h1})")
        else:
            print(f"    Step 1: no valid gap found in middle half")
    
    # Show projection histogram (every 4th row)
    print(f"\n  Projection histogram (every 3rd row, '*' = dark, '.' = gap):")
    for y in range(0, H, 3):
        bar = int(proj[y] / max(peak, 1) * 50)
        marker = "GAP" if proj[y] < gap_thr else "   "
        print(f"    y={y:3d} [{marker}] {'#' * bar} ({int(proj[y])})")

for name in FAILING:
    diagnose(name)
