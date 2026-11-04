#!/usr/bin/env python3
"""Dump band crop + C-side line split visualization for all photos.

For each photo:
  1. Run detect_structure to get the MRZ pair (Python-side crop)
  2. Crop the native band (_rst_native_mrz_band)
  3. Save the band as PNG
  4. Run the CNN tool on the band with MRZ_OCR_LINES=1
  5. Parse the C-side band/line coordinates
  6. Draw the C-side line boundaries on the band crop
  7. Save the annotated visualization

Output: tmp/band_debug/<filename>.png
"""
from __future__ import annotations
import os, sys, subprocess, json, re, struct
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "passport_photo_restore" / "src"))
from restore import drop_reflection, _gray, detect_structure

TOOL = ROOT / "6_mrz_ocr" / "build" / "mrz_ocr_cnn_tool"
OUT_DIR = ROOT / "tmp" / "band_debug"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_rgb(path):
    img = Image.open(str(path)).convert("RGB")
    return np.array(img)


def adaptive_xpad(rgb, pair, scale, nominal):
    H, W = rgb.shape[:2]
    return 0, 0


def native_band(rgb, pair, scale, source_bottom=None, single=False):
    H, W = rgb.shape[:2]
    first_top = max(0, int(round(pair[0][0] / scale)))
    second_bottom = min(H, int(round((pair[1][1] + 1) / scale)))
    heights = [
        max(1, int(round((y1 + 1) / scale)) - int(round(y0 / scale)))
        for y0, y1 in pair
    ]
    if single:
        top, bottom = first_top, second_bottom
        min_h = 30
        actual_h = bottom - top
        if actual_h < min_h:
            extra = (min_h - actual_h + 1) // 2
            top = max(0, top - extra)
            bottom = min(H, bottom + extra)
    else:
        pad = max(8, min(25, int(round(max(heights) * 0.70))))
        top = max(0, first_top - pad)
        bottom = min(H, second_bottom + pad)
    if source_bottom is not None:
        bottom = min(bottom, max(second_bottom, source_bottom))
    band = rgb[top:bottom, 0:W]
    return band, {"x": 0, "y": top, "w": W, "h": bottom - top}


def run_cnn_tool(ppm_path):
    env = dict(os.environ)
    env["MRZ_OCR_LINES"] = "1"
    env["MRZ_OCR_FUSE"] = "1"
    env["MRZ_OCR_LINE_MODELS"] = "1"
    r = subprocess.run(
        [str(TOOL), str(ppm_path)],
        capture_output=True, text=True, env=env, timeout=10
    )
    info = {}
    for line in r.stdout.splitlines():
        if line.startswith("band.x"):
            m = re.match(r"band\.x band\.y band\.w band\.h : (\d+) (\d+) (\d+) (\d+)", line)
            if m:
                info["band"] = tuple(int(x) for x in m.groups())
        if line.startswith("result.line1"):
            info["line1"] = line.split(":", 1)[1].strip()
        if line.startswith("result.line2"):
            info["line2"] = line.split(":", 1)[1].strip()
    for line in r.stderr.splitlines():
        if line.startswith("LINES"):
            m = re.match(r"LINES band=(\d+)x(\d+) n=(\d+) l0=\(y(\d+) h(\d+)\)(?: l1=\(y(\d+) h(\d+)\))?", line)
            if m:
                info["lines"] = {
                    "bw": int(m.group(1)), "bh": int(m.group(2)),
                    "n": int(m.group(3)),
                    "l0_y": int(m.group(4)), "l0_h": int(m.group(5)),
                    "l1_y": int(m.group(6)) if m.group(6) else None,
                    "l1_h": int(m.group(7)) if m.group(7) else None,
                }
    return info


def draw_visualization(band_np, cnn_info, out_path, filename, gt=None):
    img = Image.fromarray(band_np).convert("RGB")
    draw = ImageDraw.Draw(img)
    W, H = img.size
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 14)
    except Exception:
        font = ImageFont.load_default()

    lines_info = cnn_info.get("lines")
    if lines_info:
        bw = lines_info["bw"]
        bh = lines_info["bh"]
        scale_x = W / bw if bw > 0 else 1
        scale_y = H / bh if bh > 0 else 1
        l0_y = int(lines_info["l0_y"] * scale_y)
        l0_h = int(lines_info["l0_h"] * scale_y)
        l0_bot = l0_y + l0_h
        draw.line([(0, l0_y), (W, l0_y)], fill=(0, 255, 0), width=2)
        draw.line([(0, l0_bot), (W, l0_bot)], fill=(0, 255, 0), width=2)
        draw.text((2, l0_y - 16), f"L0 y={l0_y} h={l0_h}", fill=(0, 255, 0), font=font)
        if lines_info["l1_y"] is not None:
            l1_y = int(lines_info["l1_y"] * scale_y)
            l1_h = int(lines_info["l1_h"] * scale_y)
            l1_bot = l1_y + l1_h
            draw.line([(0, l1_y), (W, l1_y)], fill=(255, 0, 0), width=2)
            draw.line([(0, l1_bot), (W, l1_bot)], fill=(255, 0, 0), width=2)
            draw.text((2, l1_y - 16), f"L1 y={l1_y} h={l1_h}", fill=(255, 0, 0), font=font)

    band_info = cnn_info.get("band")
    if band_info:
        bx, by, bw, bh = band_info
        draw.text((2, 2), f"C-band x={bx} y={by} w={bw} h={bh}", fill=(255, 255, 0), font=font)

    l1 = cnn_info.get("line1", "")
    l2 = cnn_info.get("line2", "")
    draw.text((2, H - 30), f"L1: {l1}", fill=(0, 255, 255), font=font)
    draw.text((2, H - 14), f"L2: {l2}", fill=(0, 255, 255), font=font)
    if gt:
        draw.text((W // 2, H - 30), f"GT1: {gt.get('line1','')}", fill=(255, 255, 0), font=font)
        draw.text((W // 2, H - 14), f"GT2: {gt.get('line2','')}", fill=(255, 255, 0), font=font)

    img.save(str(out_path))


def load_gt():
    gt_map = {}
    try:
        import openpyxl
        wb = openpyxl.load_workbook(ROOT / "10_ocr_research" / "gt_review_loop.xlsx", read_only=True)
        ws = wb["GT对比"]
        for row in ws.iter_rows(min_row=2, values_only=True):
            fname = row[1]
            if fname:
                gt_map[str(fname)] = {"line1": row[6] or "", "line2": row[7] or ""}
    except Exception as e:
        print(f"GT load error: {e}")
    return gt_map


def main():
    gt_map = load_gt()
    print(f"Loaded {len(gt_map)} GT entries")

    photo_dirs = [ROOT / "tmp" / "PassPortPhoto", ROOT / "Photo"]
    photos = []
    for d in photo_dirs:
        if d.exists():
            photos.extend(sorted(d.rglob("*.jpg")))
    print(f"Found {len(photos)} photos")

    import tempfile
    for i, photo in enumerate(photos):
        fname = photo.name
        try:
            rgb = load_rgb(photo)
            work = rgb
            if max(rgb.shape[:2]) > 1250:
                scale_f = 1250.0 / max(rgb.shape[:2])
                work = np.array(Image.fromarray(rgb).resize(
                    (int(rgb.shape[1] * scale_f), int(rgb.shape[0] * scale_f)),
                    Image.BILINEAR))
            st = detect_structure(_gray(work))
            pair = st.get("mrz_pair")
            if not pair:
                single = st.get("mrz_single")
                if single:
                    pair = (single, single)
                else:
                    print(f"  [{i+1}/{len(photos)}] {fname}: NO PAIR")
                    continue
            is_single = pair[0] == pair[1]
            scale = work.shape[0] / max(1, rgb.shape[0])
            band_np, crop_rect = native_band(rgb, pair, scale, single=is_single)
            if band_np.size == 0:
                print(f"  [{i+1}/{len(photos)}] {fname}: EMPTY BAND")
                continue
            tmp_ppm = OUT_DIR / f"_tmp_{fname}.ppm"
            Image.fromarray(band_np).save(str(tmp_ppm), format="PPM")
            cnn_info = run_cnn_tool(tmp_ppm)
            tmp_ppm.unlink(missing_ok=True)
            out_path = OUT_DIR / f"{fname}.png"
            gt = gt_map.get(fname)
            draw_visualization(band_np, cnn_info, out_path, fname, gt)
            l1 = cnn_info.get("line1", "")
            l2 = cnn_info.get("line2", "")
            match = ""
            if gt:
                m1 = l1 == gt.get("line1", "")
                m2 = l2 == gt.get("line2", "")
                match = "PASS" if (m1 and m2) else f"FAIL({'L1ok' if m1 else 'L1bad'}/{'L2ok' if m2 else 'L2bad'})"
            print(f"  [{i+1}/{len(photos)}] {fname}: {match} band={band_np.shape} {cnn_info.get('lines',{}).get('n','?')}lines")
        except Exception as e:
            print(f"  [{i+1}/{len(photos)}] {fname}: ERROR {e}")

    print(f"\nDone. Visualizations in {OUT_DIR}")


if __name__ == "__main__":
    main()
