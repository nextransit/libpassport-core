#!/usr/bin/env python3
"""passport_photo_restore/eval — 对 Photo/ 图集跑固定矫正+去倒影，输出结构定位评估。

用法：
  python3 src/eval.py <Photo_dir> [<out_md>]

输出 markdown：每图 MRZ 带 / solid 黑带 / 定位耗时。
"""
from __future__ import annotations
import sys
import time
from pathlib import Path

import cv2

from restore import load, _gray, detect_structure, restore  # noqa: E402

HERE = Path(__file__).resolve().parent


def main():
    photo_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else         Path(__file__).resolve().parents[2] / "Photo"
    out_md = Path(sys.argv[2]) if len(sys.argv) > 2 else         HERE / "EVAL.md"
    files = sorted(photo_dir.glob("*/*.jpg")) + sorted(photo_dir.glob("*.jpg"))
    total = len(files)
    ok_mrz = 0
    ok_solid = 0
    rows = []
    t_all = time.perf_counter()
    for f in files:
        r = restore(f)
        mrz = r["mrz"]
        if mrz:
            ok_mrz += 1
        solid = r["solid"] or []
        if any(b[1] - b[0] >= 10 for b in solid):
            ok_solid += 1
        rows.append((f.name, r))
    total_ms = (time.perf_counter() - t_all) * 1000

    md = []
    md.append("# Passport 拍照图 — 固定矫正结构先验 + 去倒影 评估")
    md.append("")
    md.append("> 语料：`%s` 共 %d 张。工具：`src/restore.py`（固定算法，自包含 cv2/numpy）。"
              % (photo_dir, total))
    md.append("> 全量定位耗时：%d ms（约 %d ms/张）。" % (int(total_ms), int(total_ms / max(1, total))))
    md.append("")
    md.append("**结果：MRZ 结构定位 %d/%d；solid 黑带定位 %d/%d。**"
              % (ok_mrz, total, ok_solid, total))
    md.append("")
    md.append("| 文件 | MRZ 带(y0-y1, 高px) | solid 黑带 | 定位耗时(ms) |")
    md.append("|---|---|---|---|")
    for name, r in rows:
        mrz = r["mrz"]
        mrz_txt = "y%d-%d (%dpx)" % (mrz[0], mrz[1], mrz[1] - mrz[0] + 1) if mrz else "**未定位**"
        solid_txt = "; ".join("%d-%d" % (a, b) for a, b in (r["solid"] or [])[-2:]) or "-"
        md.append("| %s | %s | %s | %.1f |" % (name, mrz_txt, solid_txt, r["ms"]))
    md.append("")
    md.append("## 倒影切除判定")
    md.append("- MRZ 带 = solid 黑带下沿之后连续 text 段（gap<=15px）。")
    md.append("- 当下一段与上一段 gap>15px 判定进入倒影区 → crop 天然不含倒影。")
    md.append("- 单应矫正：MRZ 带四边形 -> 标准 44格x2行 矩形（INTER_LANCZOS4）。")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(md))
    print("wrote", out_md)
    print("MRZ locate:", ok_mrz, "/", total, "| solid:", ok_solid, "/", total)
    return 0


if __name__ == "__main__":
    sys.exit(main())
