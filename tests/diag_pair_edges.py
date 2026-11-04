#!/usr/bin/env python3
"""精确诊断 pair 边界 — 输出边界处每行墨量，判断是否截断了文字。

对每个样本输出 pair 边界上下5行的墨量分布：
  - L1 top 上下5行：判断 line1 顶部是否截断
  - L1 bot / L2 top 上下5行：判断行间分隔
  - L2 bot 上下5行：判断 line2 底部是否截断

用法:
    python3 tests/diag_pair_edges.py --dirs tmp/PassPortPhoto,Photo
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
            continue

        (a0, a1), (b0, b1) = pair
        scale = float(rec.get("scale") or 1.0)
        ny0a = max(0, int(round(a0 / scale)))
        ny1a = min(H, int(round(a1 / scale)))
        ny0b = max(0, int(round(b0 / scale)))
        ny1b = min(H, int(round(b1 / scale)))

        # GT 匹配
        gt = app._gt_map().get(name)
        ocr = rec.get("ocr") or {}
        l1 = ocr.get("line1") or ""
        l2 = ocr.get("line2") or ""
        gt_l1 = gt[0] if gt else ""
        gt_l2 = gt[1] if gt else ""
        match = "✓" if (gt and l1 == gt_l1 and l2 == gt_l2) else "✗"

        # 计算每行墨量
        row_ink = (gray < 128).sum(axis=1).astype(float) / W

        # 输出边界上下5行的墨量
        def _ctx(y, n=5):
            rows = []
            for dy in range(-n, n + 1):
                ry = y + dy
                if 0 <= ry < H:
                    marker = ">>" if dy == 0 else "  "
                    rows.append(f"  {marker} y={ry:4d} ink={row_ink[ry]:.3f}")
            return "\n".join(rows)

        print(f"\n{'='*60}")
        print(f"{name}  {match}  (L1:{ny0a}-{ny1a} h={ny1a-ny0a}, "
              f"L2:{ny0b}-{ny1b} h={ny1b-ny0b}, gap={ny0b-ny1a})")
        print(f"--- L1 top (y={ny0a}) ---")
        print(_ctx(ny0a))
        print(f"--- L1 bot / L2 top (gap {ny0b-ny1a}px) ---")
        print(f"  L1 bot y={ny1a} ink={row_ink[ny1a]:.3f}")
        for y in range(ny1a + 1, ny0b):
            print(f"     y={y:4d} ink={row_ink[y]:.3f}")
        print(f"  L2 top y={ny0b} ink={row_ink[ny0b]:.3f}")
        print(f"--- L2 bot (y={ny1b}) ---")
        print(_ctx(ny1b))

    app.destroy()
    return 0


if __name__ == "__main__":
    sys.exit(main())
