#!/usr/bin/env python3
"""Analyze band segmentation quality numerically.

For each photo, reports:
  - Python-side crop height and padding
  - Row darkness profile (to see if text is clipped at edges)
  - C-side band/line coordinates
  - Whether top/bottom rows have significant ink (indicating clipped text)
  - Line height ratio (should be ~1.0 for TD3)
  - Inter-line gap (should be ~0.3-0.5x line height)
"""
from __future__ import annotations
import os, sys, subprocess, re, json
from pathlib import Path
from PIL import Image
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "passport_photo_restore" / "src"))
from restore import drop_reflection, _gray, detect_structure

TOOL = ROOT / "6_mrz_ocr" / "build" / "mrz_ocr_cnn_tool"


def run_cnn_tool(ppm_path):
    env = dict(os.environ)
    env["MRZ_OCR_LINES"] = "1"
    env["MRZ_OCR_FUSE"] = "1"
    env["MRZ_OCR_LINE_MODELS"] = "1"
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
    return info


def analyze_row_profile(band_np):
    gray = np.mean(band_np, axis=2)
    H, W = gray.shape
    row_dark = np.sum(gray < 128, axis=1)
    peak = np.max(row_dark)
    threshold = max(4, peak // 6)
    dense_rows = row_dark >= threshold
    first_dense = np.argmax(dense_rows) if np.any(dense_rows) else 0
    last_dense = H - 1 - np.argmax(dense_rows[::-1]) if np.any(dense_rows) else H - 1
    top_3_avg = np.mean(row_dark[:3]) if H >= 3 else 0
    bot_3_avg = np.mean(row_dark[-3:]) if H >= 3 else 0
    top_clip = top_3_avg > threshold * 0.5
    bot_clip = bot_3_avg > threshold * 0.5
    gap_rows = np.where(~dense_rows)[0]
    if len(gap_rows) > 0:
        gaps = []
        in_gap = False
        gap_start = 0
        for y in range(H):
            if not dense_rows[y]:
                if not in_gap: gap_start = y; in_gap = True
            else:
                if in_gap: gaps.append((gap_start, y - 1, y - gap_start)); in_gap = False
        if in_gap: gaps.append((gap_start, H - 1, H - gap_start))
        gaps.sort(key=lambda g: -g[2])
    else:
        gaps = []
    return {
        "H": H, "W": W, "peak": int(peak), "threshold": int(threshold),
        "first_dense": int(first_dense), "last_dense": int(last_dense),
        "top_3_avg": float(top_3_avg), "bot_3_avg": float(bot_3_avg),
        "top_clip": bool(top_clip), "bot_clip": bool(bot_clip),
        "top_margin": int(first_dense), "bot_margin": int(H - 1 - last_dense),
        "biggest_gaps": [{"y0": g[0], "y1": g[1], "h": g[2]} for g in gaps[:3]],
    }


def load_gt():
    gt_map = {}
    try:
        import openpyxl
        wb = openpyxl.load_workbook(ROOT / "10_ocr_research" / "gt_review_loop.xlsx", read_only=True)
        ws = wb["GT对比"]
        for row in ws.iter_rows(min_row=2, values_only=True):
            fname = row[1]
            if fname: gt_map[str(fname)] = {"line1": row[6] or "", "line2": row[7] or ""}
    except Exception as e:
        print(f"GT load error: {e}")
    return gt_map


def native_band(rgb, pair, scale, single=False):
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
    else:
        pad = max(8, min(25, int(round(max(heights) * 0.70))))
        top = max(0, first_top - pad); bottom = min(H, second_bottom + pad)
    band = rgb[top:bottom, 0:W]
    return band, top, bottom, first_top, second_bottom


def main():
    gt_map = load_gt()
    photo_dirs = [ROOT / "tmp" / "PassPortPhoto", ROOT / "Photo"]
    photos = []
    for d in photo_dirs:
        if d.exists(): photos.extend(sorted(d.rglob("*.jpg")))

    print(f"{'photo':<35} {'bh':>3} {'pad_t':>5} {'pad_b':>5} {'clipT':>5} {'clipB':>5} {'nLn':>3} {'l0h':>3} {'l1h':>3} {'gap':>3} {'ratio':>5} {'result':<8} {'issue'}")
    print("-" * 130)

    issues = []
    for photo in photos:
        fname = photo.name
        try:
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
                else:
                    print(f"{fname:<35} NO PAIR")
                    continue
            is_single = pair[0] == pair[1]
            scale = work.shape[0] / max(1, rgb.shape[0])
            band_np, top, bottom, ft, sb = native_band(rgb, pair, scale, single=is_single)
            if band_np.size == 0: continue

            tmp_ppm = ROOT / "tmp" / "band_debug" / f"_tmp_{fname}.ppm"
            Image.fromarray(band_np).save(str(tmp_ppm), format="PPM")
            cnn_info = run_cnn_tool(tmp_ppm)
            tmp_ppm.unlink(missing_ok=True)

            prof = analyze_row_profile(band_np)
            lines_info = cnn_info.get("lines", {})
            n_lines = lines_info.get("n", "?")
            l0h = lines_info.get("l0_h", 0)
            l1h = lines_info.get("l1_h", 0)
            l1y = lines_info.get("l1_y", 0)
            l0y = lines_info.get("l0_y", 0)
            gap = (l1y - (l0y + l0h)) if (l1y is not None and l0h) else 0
            ratio = lines_info.get("ratio", 0) or 0

            l1 = cnn_info.get("line1", "")
            l2 = cnn_info.get("line2", "")
            gt = gt_map.get(fname, {})
            m1 = l1 == gt.get("line1", "")
            m2 = l2 == gt.get("line2", "")
            result = "PASS" if (m1 and m2) else "FAIL"

            issue = ""
            if prof["top_clip"]: issue += "TOP_CLIP "
            if prof["bot_clip"]: issue += "BOT_CLIP "
            if n_lines == 1 and not is_single: issue += "MISSING_L1 "
            if l0h and l1h and (max(l0h, l1h) > 2 * min(l0h, l1h)): issue += "HEIGHT_RATIO "
            if prof["top_margin"] < 2: issue += "TIGHT_TOP "
            if prof["bot_margin"] < 2: issue += "TIGHT_BOT "
            if band_np.shape[0] < 60 and not is_single: issue += "SHORT_BAND "

            pad_t = ft - top
            pad_b = bottom - sb

            print(f"{fname:<35} {band_np.shape[0]:>3} {pad_t:>5} {pad_b:>5} {'YES' if prof['top_clip'] else 'no':>5} {'YES' if prof['bot_clip'] else 'no':>5} {str(n_lines):>3} {l0h:>3} {l1h:>3} {gap:>3} {ratio:>5.2f} {result:<8} {issue}")

            if issue:
                issues.append((fname, issue, result))

        except Exception as e:
            print(f"{fname:<35} ERROR {e}")

    print(f"\n=== Issues found ({len(issues)}) ===")
    for fname, issue, result in issues:
        print(f"  {fname:<35} {result} {issue}")


if __name__ == "__main__":
    main()
