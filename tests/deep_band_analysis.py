#!/usr/bin/env python3
"""Deep analysis of band segmentation for FAIL cases.

Reports the full pipeline coordinates:
  - Python-side: pair detection, native band crop, padding
  - C-side: locate_band result, split_lines result, upscale factor
  - Row darkness profile at the band edges
"""
from __future__ import annotations
import os, sys, subprocess, re
from pathlib import Path
from PIL import Image
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "passport_photo_restore" / "src"))
from restore import drop_reflection, _gray, detect_structure

TOOL = ROOT / "6_mrz_ocr" / "build" / "mrz_ocr_cnn_tool"


def run_cnn_tool_verbose(ppm_path):
    env = dict(os.environ)
    env["MRZ_OCR_LINES"] = "1"
    env["MRZ_OCR_FUSE"] = "1"
    env["MRZ_OCR_LINE_MODELS"] = "1"
    env["MRZ_OCR_GRID"] = "1"
    r = subprocess.run([str(TOOL), str(ppm_path)], capture_output=True, text=True, env=env, timeout=10)
    info = {}
    for line in r.stdout.splitlines():
        if line.startswith("band.x"):
            m = re.match(r"band\.x band\.y band\.w band\.h : (\d+) (\d+) (\d+) (\d+)", line)
            if m: info["band"] = tuple(int(x) for x in m.groups())
        if line.startswith("result.line1"): info["line1"] = line.split(":", 1)[1].strip()
        if line.startswith("result.line2"): info["line2"] = line.split(":", 1)[1].strip()
        if line.startswith("result.layoutname"): info["layout"] = line.split(":", 1)[1].strip()
    for line in r.stderr.splitlines():
        if line.startswith("LINES"):
            m = re.match(r"LINES band=(\d+)x(\d+) n=(\d+) l0=\(y(\d+) h(\d+)\)(?: l1=\(y(\d+) h(\d+)\))?(?: ratio=([\d.]+))?", line)
            if m:
                info["lines"] = {
                    "bw": int(m.group(1)), "bh": int(m.group(2)), "n": int(m.group(3)),
                    "l0_y": int(m.group(4)), "l0_h": int(m.group(5)),
                    "l1_y": int(m.group(6)) if m.group(6) else None,
                    "l1_h": int(m.group(7)) if m.group(7) else None,
                    "ratio": float(m.group(8)) if m.group(8) else None,
                }
        if line.startswith("SKEW"):
            info["skew"] = line.strip()
    return info


def native_band_with_details(rgb, pair, scale, single=False):
    H, W = rgb.shape[:2]
    first_top = max(0, int(round(pair[0][0] / scale)))
    second_bottom = min(H, int(round((pair[1][1] + 1) / scale)))
    heights = [max(1, int(round((y1 + 1) / scale)) - int(round(y0 / scale))) for y0, y1 in pair]
    if single:
        top, bottom = first_top, second_bottom
        min_h = 30
        actual_h = bottom - top
        if actual_h < min_h:
            extra = (min_h - actual_h + 1) // 2
            top = max(0, top - extra); bottom = min(H, bottom + extra)
        pad = 0
    else:
        pad = max(8, min(25, int(round(max(heights) * 0.70))))
        top = max(0, first_top - pad); bottom = min(H, second_bottom + pad)
    band = rgb[top:bottom, 0:W]
    return band, {"top": top, "bottom": bottom, "first_top": first_top,
                  "second_bottom": second_bottom, "pad": pad, "heights": heights,
                  "pair": pair, "scale": scale}


def analyze_edges(band_np):
    gray = np.mean(band_np, axis=2)
    H, W = gray.shape
    row_dark = np.sum(gray < 128, axis=1)
    peak = np.max(row_dark)
    print(f"  Band shape: {band_np.shape}")
    print(f"  Row darkness peak: {peak} ({peak/W*100:.1f}% of width)")
    print(f"  Top 5 rows darkness: {row_dark[:5].tolist()}")
    print(f"  Bottom 5 rows darkness: {row_dark[-5:].tolist()}")
    dense = row_dark >= max(4, peak // 6)
    first_dense = np.argmax(dense) if np.any(dense) else -1
    last_dense = H - 1 - np.argmax(dense[::-1]) if np.any(dense) else -1
    print(f"  First dense row: {first_dense} (margin from top: {first_dense})")
    print(f"  Last dense row: {last_dense} (margin from bottom: {H-1-last_dense})")
    return row_dark


def main():
    targets = [
        "2026-9-29-U.jpg", "2026-9-29-W.jpg",
        "2026-9-30-I.jpg", "2026-9-30-J.jpg", "2026-9-30-K.jpg",
        "2026-9-30-H.jpg", "2026-9-30-F.jpg", "2026-9-30-G.jpg",
        "IMG_1792133000_003.jpg", "IMG_1792133649_007.jpg", "IMG_1792133682_008.jpg",
    ]
    photo_dirs = [ROOT / "tmp" / "PassPortPhoto", ROOT / "Photo"]
    all_photos = {}
    for d in photo_dirs:
        if d.exists():
            for p in sorted(d.rglob("*.jpg")):
                all_photos[p.name] = p

    for target in targets:
        photo = all_photos.get(target)
        if not photo:
            print(f"\n{'='*60}\n{target}: NOT FOUND")
            continue
        print(f"\n{'='*60}")
        print(f"Photo: {target}")
        rgb = np.array(Image.open(str(photo)).convert("RGB"))
        print(f"  Original size: {rgb.shape}")
        work = rgb
        if max(rgb.shape[:2]) > 1250:
            sf = 1250.0 / max(rgb.shape[:2])
            work = np.array(Image.fromarray(rgb).resize((int(rgb.shape[1]*sf), int(rgb.shape[0]*sf)), Image.BILINEAR))
        st = detect_structure(_gray(work))
        pair = st.get("mrz_pair")
        if not pair:
            single = st.get("mrz_single")
            if single: pair = (single, single)
            else:
                print("  NO PAIR FOUND")
                continue
        is_single = pair[0] == pair[1]
        scale = work.shape[0] / max(1, rgb.shape[0])
        band_np, details = native_band_with_details(rgb, pair, scale, single=is_single)
        print(f"  Pair (work coords): {details['pair']}")
        print(f"  Scale: {scale:.4f}")
        print(f"  Line heights (native): {details['heights']}")
        print(f"  Padding: {details['pad']}px")
        print(f"  first_top={details['first_top']}, second_bottom={details['second_bottom']}")
        print(f"  Crop: top={details['top']}, bottom={details['bottom']}, height={details['bottom']-details['top']}")
        if band_np.size == 0:
            print("  EMPTY BAND")
            continue
        row_dark = analyze_edges(band_np)
        tmp_ppm = ROOT / "tmp" / "band_debug" / f"_tmp_{target}.ppm"
        Image.fromarray(band_np).save(str(tmp_ppm), format="PPM")
        cnn_info = run_cnn_tool_verbose(tmp_ppm)
        tmp_ppm.unlink(missing_ok=True)
        print(f"  C-side band: {cnn_info.get('band', 'N/A')}")
        print(f"  C-side lines: {cnn_info.get('lines', 'N/A')}")
        print(f"  Layout: {cnn_info.get('layout', 'N/A')}")
        print(f"  Skew: {cnn_info.get('skew', 'N/A')}")
        print(f"  L1: {cnn_info.get('line1', 'N/A')}")
        print(f"  L2: {cnn_info.get('line2', 'N/A')}")
        lines_info = cnn_info.get("lines", {})
        if lines_info:
            bw = lines_info.get("bw", 0)
            bh = lines_info.get("bh", 0)
            print(f"  C-band vs P-band: C-side uses {bw}x{bh} of {band_np.shape[1]}x{band_np.shape[0]} PPM")
            if lines_info.get("l1_y") is not None:
                l0_y = lines_info["l0_y"]; l0_h = lines_info["l0_h"]
                l1_y = lines_info["l1_y"]; l1_h = lines_info["l1_h"]
                gap = l1_y - (l0_y + l0_h)
                print(f"  Line 0: y={l0_y} h={l0_h} (bottom={l0_y+l0_h})")
                print(f"  Line 1: y={l1_y} h={l1_h} (bottom={l1_y+l1_h})")
                print(f"  Inter-line gap: {gap}")
                print(f"  Top margin in C-band: {l0_y}")
                print(f"  Bottom margin in C-band: {bh - (l1_y + l1_h)}")
                print(f"  Height ratio l0/l1: {l0_h/l1_h:.2f}" if l1_h else "  Height ratio: N/A")


if __name__ == "__main__":
    main()
