#!/usr/bin/env python3
"""诊断 MRZ band 裁剪几何 — 检查 band 是否完全覆盖 line1+line2。

对每个样本输出：
  - 图片尺寸
  - pair (y0a, y1a, y0b, y1b) 在 native 坐标
  - band 裁剪区域 (y0, y1, height)
  - line1 高度 (y1a - y0a), line2 高度 (y1b - y0b)
  - 行间距 (y0b - y1a)
  - band 顶部 margin (y0a - 0), 底部 margin (img_h - y1b)
  - band 区域内顶部/底部的暗像素比例（判断是否截断了文字）

用法:
    python3 tests/diag_band_geometry.py --dirs tmp/PassPortPhoto,Photo
    python3 tests/diag_band_geometry.py --dirs assets/pic
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))


def _load_gui():
    spec = importlib.util.spec_from_file_location(
        "gui", ROOT / "tests" / "gui.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _ink_ratio(gray: np.ndarray, row_range: tuple) -> float:
    """某行区间内暗像素（<128）的比例。"""
    y0, y1 = row_range
    y0 = max(0, y0)
    y1 = min(gray.shape[0], y1)
    if y1 <= y0:
        return 0.0
    strip = gray[y0:y1, :]
    return float(np.mean(strip < 128))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", default="")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)

    gui = _load_gui()
    gui.PassportGUI.AUTO_WARM_ENGINES = False
    app = gui.PassportGUI()
    app.update()
    app.withdraw()

    app.rst_method_both.set(True)
    app._rst_selected_methods = lambda: ["fuse"]
    app._rst_resident = True

    dirs = [d.strip() for d in args.dirs.split(",") if d.strip()] if args.dirs else []
    photos = []
    for label, path in gui.iter_passport_photos():
        if dirs and not any(label == d or label.startswith(d + "/") for d in dirs):
            continue
        photos.append((label, path))
    if args.limit:
        photos = photos[:args.limit]

    gt_map = app._gt_map()

    print(f"{'文件名':<30} {'img_h':>5} {'band_y0':>7} {'band_y1':>7} "
          f"{'band_h':>6} {'l1_h':>5} {'l2_h':>5} {'gap':>5} "
          f"{'top_ink':>8} {'bot_ink':>8} {'问题':<20}")
    print("-" * 120)

    issues = []
    for idx, (label, path) in enumerate(photos, 1):
        name = path.name
        bgr = cv2.imread(str(path))
        if bgr is None:
            continue
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        H, W = gray.shape

        try:
            rec = app._ocr_photo_engines(str(path))
        except Exception:
            continue

        pair = rec.get("pair")
        if not pair:
            print(f"{name:<30} {H:>5} {'—':>7} {'—':>7} {'—':>6} "
                  f"{'—':>5} {'—':>5} {'—':>5} {'—':>8} {'—':>8} {'NO_PAIR':<20}")
            issues.append((name, "NO_PAIR"))
            continue

        (a0, a1), (b0, b1) = pair
        scale = float(rec.get("scale") or 1.0)

        # native 坐标
        ny0a = int(round(a0 / scale))
        ny1a = int(round(a1 / scale))
        ny0b = int(round(b0 / scale))
        ny1b = int(round(b1 / scale))

        l1_h = ny1a - ny0a
        l2_h = ny1b - ny0b
        gap = ny0b - ny1a
        band_h = ny1b - ny0a

        # 检查 band 顶部/底部是否有文字被截断
        # 顶部：检查 band 上方 3px 是否有暗像素（如果有，说明 band 顶部截断了文字）
        top_margin = 3
        top_ink = _ink_ratio(gray, (ny0a - top_margin, ny0a)) if ny0a > top_margin else 0.0
        # 底部：检查 band 下方 3px 是否有暗像素
        bot_ink = _ink_ratio(gray, (ny1b, ny1b + top_margin)) if ny1b + top_margin < H else 0.0

        # 也检查 band 内部顶部/底部 2px 的暗像素比例
        inner_top_ink = _ink_ratio(gray, (ny0a, ny0a + 2))
        inner_bot_ink = _ink_ratio(gray, (ny1b - 2, ny1b))

        # 判断问题
        probs = []
        if top_ink > 0.15:
            probs.append(f"顶部截断(上方{top_ink:.0%})")
        if bot_ink > 0.15:
            probs.append(f"底部截断(下方{bot_ink:.0%})")
        if l1_h < 8:
            probs.append(f"L1太矮({l1_h}px)")
        if l2_h < 8:
            probs.append(f"L2太矮({l2_h}px)")
        if gap < 0:
            probs.append(f"行重叠({gap}px)")
        if gap > l1_h * 1.5:
            probs.append(f"行间距过大({gap}px)")
        if band_h < 20:
            probs.append(f"band太矮({band_h}px)")

        # GT 匹配
        gt = gt_map.get(name)
        ocr = rec.get("ocr") or {}
        l1 = ocr.get("line1") or ""
        l2 = ocr.get("line2") or ""
        gt_l1 = gt[0] if gt else ""
        gt_l2 = gt[1] if gt else ""
        gt_match = "✓" if (gt and l1 == gt_l1 and l2 == gt_l2) else "✗"

        prob_str = ",".join(probs) if probs else "OK"
        if probs:
            issues.append((name, prob_str))

        print(f"{name:<30} {H:>5} {ny0a:>7} {ny1b:>7} "
              f"{band_h:>6} {l1_h:>5} {l2_h:>5} {gap:>5} "
              f"{top_ink:>7.1%} {bot_ink:>7.1%} {gt_match} {prob_str}")

    print(f"\n问题汇总: {len(issues)}/{len(photos)} 样本有潜在分割问题")
    for name, prob in issues:
        print(f"  {name}: {prob}")

    app.destroy()
    return 0


if __name__ == "__main__":
    sys.exit(main())
