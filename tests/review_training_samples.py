#!/usr/bin/env python3
"""训练样本人工确认工具 — 生成 HTML 报告供逐条审查。

对每张图片：
  1. 检测 MRZ band 并裁剪保存为 PNG（放大 2× 便于阅读）
  2. 运行 fuse 识别
  3. 从 gt_review_loop.xlsx 读取 GT
  4. 逐字符对比 GT vs 识别结果，标记差异
  5. 汇总到 HTML 报告

报告输出到 10_ocr_research/training_sample_review.html

用法:
    python3 tests/review_training_samples.py
    python3 tests/review_training_samples.py --dirs tmp/PassPortPhoto,Photo
    python3 tests/review_training_samples.py --dirs assets/pic
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import cv2
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))


def _load_gui():
    spec = importlib.util.spec_from_file_location(
        "gui", ROOT / "tests" / "gui.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _char_diff(pred: str, gt: str) -> list:
    """逐字符对比，返回 [(idx, pred_ch, gt_ch, match_bool), ...]。"""
    n = max(len(pred), len(gt))
    result = []
    for i in range(n):
        p = pred[i] if i < len(pred) else "·"
        g = gt[i] if i < len(gt) else "·"
        result.append((i, p, g, p == g))
    return result


def _draw_h_line(img, y, color, label=""):
    """在图片上画水平线并标注文字。"""
    H, W = img.shape[:2]
    if 0 <= y < H:
        cv2.line(img, (0, y), (W - 1, y), color, 1)
        if label:
            cv2.putText(img, label, (2, max(0, y - 2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)


def _diff_html(pred: str, gt: str) -> str:
    """生成逐字符对比的 HTML 片段。"""
    diffs = _char_diff(pred, gt)
    cells = []
    for idx, p, g, ok in diffs:
        if ok:
            cells.append(f'<span class="ok">{g}</span>')
        else:
            cells.append(
                f'<span class="err" title="pos {idx}: pred={p} gt={g}">'
                f'{g}</span>')
    return "".join(cells)


def _read_gt_from_xlsx() -> dict:
    """从 gt_review_loop.xlsx 读取 GT。"""
    import openpyxl
    xlsx = ROOT / "10_ocr_research" / "gt_review_loop.xlsx"
    if not xlsx.exists():
        return {}
    wb = openpyxl.load_workbook(str(xlsx), read_only=True, data_only=True)
    ws = wb["GT对比"]
    m = {}
    for row in ws.iter_rows(min_row=2, values_only=True):
        fn = row[1] or ""
        gt1 = str(row[6] or "").strip()
        gt2 = str(row[7] or "").strip()
        if fn:
            m[str(fn).strip()] = (gt1, gt2)
    wb.close()
    return m


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", default="",
                    help="逗号分隔的目录，默认全部")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "10_ocr_research" / "training_sample_review.html")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)

    gui = _load_gui()
    gui.PassportGUI.AUTO_WARM_ENGINES = False
    app = gui.PassportGUI()
    app.update()
    app.withdraw()

    # 预加载引擎
    sys.path.insert(0, str(ROOT / "tests"))
    import mrz_engine_pool
    mrz_engine_pool.warmup_all(gui_module=gui)

    app.rst_method_both.set(True)
    app._rst_selected_methods = lambda: ["fuse"]
    app._rst_resident = True

    # 收集图片
    dirs = [d.strip() for d in args.dirs.split(",") if d.strip()] if args.dirs else []
    photos = []
    for label, path in gui.iter_passport_photos():
        if dirs and not any(label == d or label.startswith(d + "/") for d in dirs):
            continue
        photos.append((label, path))
    if args.limit:
        photos = photos[:args.limit]

    gt_map = _read_gt_from_xlsx()

    # 输出目录
    band_dir = args.out.parent / "review_bands"
    band_dir.mkdir(parents=True, exist_ok=True)

    rows_html = []
    stats = {"total": 0, "has_gt": 0, "gt_match": 0, "gt_mismatch": 0,
             "no_gt": 0, "no_pair": 0}

    for idx, (label, path) in enumerate(photos, 1):
        stats["total"] += 1
        name = path.name
        print(f"[{idx}/{len(photos)}] {name} ...", end="", flush=True)

        # 读取图片
        bgr = cv2.imread(str(path))
        if bgr is None:
            rows_html.append(_row_html(name, label, None, None, None,
                                       None, None, "无法读取图片"))
            stats["no_pair"] += 1
            print(" 读取失败")
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        H, W = rgb.shape[:2]

        # 运行识别
        t0 = time.perf_counter()
        try:
            rec = app._ocr_photo_engines(str(path))
        except Exception as exc:
            rec = {"ocr": {"line1": "", "line2": ""},
                   "pair": None, "loc": {}, "engines": [],
                   "reason": f"{type(exc).__name__}: {exc}"}
        ms = (time.perf_counter() - t0) * 1000.0

        ocr = rec.get("ocr") or {}
        l1 = ocr.get("line1") or ""
        l2 = ocr.get("line2") or ""
        pair = rec.get("pair")

        # 裁剪 MRZ band — 标注 pair 边界线
        band_png = None
        band_info = None
        if pair:
            (a0, a1), (b0, b1) = pair
            scale = float(rec.get("scale") or 1.0)
            ny0a = max(0, int(round(a0 / scale)))
            ny1a = min(H, int(round(a1 / scale)))
            ny0b = max(0, int(round(b0 / scale)))
            ny1b = min(H, int(round(b1 / scale)))
            # padding (与 _rst_native_mrz_band 一致)
            h1 = max(1, ny1a - ny0a)
            h2 = max(1, ny1b - ny0b)
            pad = max(8, min(25, int(round(max(h1, h2) * 0.70))))
            band_top = max(0, ny0a - pad)
            band_bot = min(H, ny1b + pad)
            # 裁剪更大区域（上下各加20px margin）以显示上下文
            margin = 20
            ctx_top = max(0, band_top - margin)
            ctx_bot = min(H, band_bot + margin)
            band_crop = rgb[ctx_top:ctx_bot, :].copy()
            if band_crop.size > 0:
                # 标注线
                _draw_h_line(band_crop, ny0a - ctx_top, (255, 0, 0), "L1 top")
                _draw_h_line(band_crop, ny1a - ctx_top, (255, 0, 0), "L1 bot")
                _draw_h_line(band_crop, ny0b - ctx_top, (0, 0, 255), "L2 top")
                _draw_h_line(band_crop, ny1b - ctx_top, (0, 0, 255), "L2 bot")
                _draw_h_line(band_crop, band_top - ctx_top, (0, 255, 0), "band top")
                _draw_h_line(band_crop, band_bot - ctx_top, (0, 255, 0), "band bot")
                # 放大 2× 便于阅读
                band_big = cv2.resize(band_crop, None, fx=2.0, fy=2.0,
                                      interpolation=cv2.INTER_LINEAR)
                band_png = f"review_bands/{name}.png"
                cv2.imwrite(str(args.out.parent / band_png),
                            cv2.cvtColor(band_big, cv2.COLOR_RGB2BGR))
                band_info = {
                    "y0": ny0a, "y1": ny1b, "h": ny1b - ny0a,
                    "l1_h": h1, "l2_h": h2, "gap": ny0b - ny1a,
                    "pad": pad, "band_h": band_bot - band_top,
                    "scale": scale,
                }
        else:
            band_raw = ocr.get("band", "")
            lines_raw = ocr.get("lines", "")
            crop = ocr.get("crop") or {}
            if band_raw and lines_raw:
                try:
                    bx, by, bw, bh = (int(float(x)) for x in band_raw.split())
                    l1y, l1h, l2y, l2h = (int(float(x)) for x in lines_raw.split())
                    cy = int(crop.get("y", 0))
                    band_y_abs = cy + by
                    ny0a = max(0, band_y_abs + l1y)
                    ny1a = min(H, band_y_abs + l1y + l1h)
                    ny0b = max(0, band_y_abs + l2y)
                    ny1b = min(H, band_y_abs + l2y + l2h)
                    band_top = max(0, band_y_abs)
                    band_bot = min(H, band_y_abs + bh)
                    margin = 20
                    ctx_top = max(0, band_top - margin)
                    ctx_bot = min(H, band_bot + margin)
                    band_crop = rgb[ctx_top:ctx_bot, :].copy()
                    if band_crop.size > 0:
                        _draw_h_line(band_crop, ny0a - ctx_top, (255, 0, 0), "L1 top")
                        _draw_h_line(band_crop, ny1a - ctx_top, (255, 0, 0), "L1 bot")
                        _draw_h_line(band_crop, ny0b - ctx_top, (0, 0, 255), "L2 top")
                        _draw_h_line(band_crop, ny1b - ctx_top, (0, 0, 255), "L2 bot")
                        _draw_h_line(band_crop, band_top - ctx_top, (0, 255, 0), "band top")
                        _draw_h_line(band_crop, band_bot - ctx_top, (0, 255, 0), "band bot")
                        band_big = cv2.resize(band_crop, None, fx=2.0, fy=2.0,
                                              interpolation=cv2.INTER_LINEAR)
                        band_png = f"review_bands/{name}.png"
                        cv2.imwrite(str(args.out.parent / band_png),
                                    cv2.cvtColor(band_big, cv2.COLOR_RGB2BGR))
                        band_info = {
                            "y0": ny0a, "y1": ny1b, "h": ny1b - ny0a,
                            "l1_h": l1h, "l2_h": l2h, "gap": ny0b - ny1a,
                            "pad": 0, "band_h": bh, "scale": 1,
                        }
                except Exception:
                    pass

        # GT 对比
        gt = gt_map.get(name)
        gt_l1 = gt[0] if gt else ""
        gt_l2 = gt[1] if gt else ""

        if gt and (gt_l1 or gt_l2):
            stats["has_gt"] += 1
            l1_match = (l1 == gt_l1) if gt_l1 else True
            l2_match = (l2 == gt_l2) if gt_l2 else True
            if l1_match and l2_match:
                stats["gt_match"] += 1
            else:
                stats["gt_mismatch"] += 1
        elif not gt:
            stats["no_gt"] += 1

        if not pair:
            stats["no_pair"] += 1

        # 生成行 HTML
        row = _row_html(
            name, label, (W, H), band_png, band_info,
            (gt_l1, gt_l2), (l1, l2), None,
            ms=ms, pair=pair, reason=rec.get("reason", ""))
        rows_html.append(row)
        match_tag = "✓" if (gt and l1 == gt_l1 and l2 == gt_l2) else "✗"
        print(f" {ms:.0f}ms {match_tag}")

    # 生成 HTML
    html = _html_template(rows_html, stats)
    args.out.write_text(html, encoding="utf-8")
    print(f"\n报告已生成: {args.out}")
    print(f"统计: {stats['total']} 张图片, "
          f"{stats['has_gt']} 有GT, "
          f"{stats['gt_match']} GT匹配, "
          f"{stats['gt_mismatch']} GT不匹配, "
          f"{stats['no_gt']} 无GT, "
          f"{stats['no_pair']} 未定位")

    app.destroy()
    return 0


def _row_html(name, label, dims, band_png, band_info,
              gt, pred, error, ms=0, pair=None, reason=""):
    """生成单行 HTML。"""
    if error:
        return f'<tr class="err-row"><td>{name}</td><td colspan="7">{error}</td></tr>'

    dim_str = f"{dims[0]}×{dims[1]}" if dims else "-"
    band_cell = ""
    if band_png:
        band_cell = f'<img src="{band_png}" loading="lazy">'
    band_meta = ""
    if band_info:
        band_meta = (f"L1:{band_info['l1_h']}px L2:{band_info['l2_h']}px "
                     f"gap:{band_info['gap']}px pad:{band_info['pad']}px "
                     f"band:{band_info['band_h']}px")

    gt_l1, gt_l2 = gt if gt else ("", "")
    p_l1, p_l2 = pred if pred else ("", "")

    l1_diff = _diff_html(p_l1, gt_l1) if gt_l1 else f'<span class="nogt">{p_l1}</span>'
    l2_diff = _diff_html(p_l2, gt_l2) if gt_l2 else f'<span class="nogt">{p_l2}</span>'

    l1_ok = (p_l1 == gt_l1) if gt_l1 else None
    l2_ok = (p_l2 == gt_l2) if gt_l2 else None

    status = ""
    if gt_l1 or gt_l2:
        if (l1_ok is not False) and (l2_ok is not False):
            status = '<span class="pass">PASS</span>'
        else:
            status = '<span class="fail">FAIL</span>'
    else:
        status = '<span class="nogt">NO_GT</span>'

    pair_str = "✓" if pair else "✗"

    return f"""<tr>
<td class="fname">{name}</td>
<td>{dim_str}</td>
<td class="band">{band_cell}<div class="meta">{band_meta}</div></td>
<td class="mono">{gt_l1 or '—'}</td>
<td class="mono">{gt_l2 or '—'}</td>
<td class="mono diff">{l1_diff}</td>
<td class="mono diff">{l2_diff}</td>
<td>{status}</td>
<td>{pair_str}</td>
<td>{ms:.0f}ms</td>
</tr>"""


def _html_template(rows, stats):
    total = stats["total"] or 1
    match_pct = stats["gt_match"] / stats["has_gt"] * 100 if stats["has_gt"] else 0
    return f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>训练样本审查报告</title>
<style>
body {{ font-family: -apple-system, "Helvetica Neue", sans-serif; margin: 20px; }}
h1 {{ font-size: 18px; }}
.summary {{ background: #f0f0f0; padding: 12px 16px; border-radius: 8px;
            margin-bottom: 16px; font-size: 14px; }}
.summary span {{ margin-right: 20px; }}
table {{ border-collapse: collapse; width: 100%; font-size: 12px; }}
th, td {{ border: 1px solid #ddd; padding: 4px 6px; text-align: left; vertical-align: top; }}
th {{ background: #4a90d9; color: white; position: sticky; top: 0; }}
tr:nth-child(even) {{ background: #f9f9f9; }}
tr.err-row {{ background: #ffe0e0; }}
.fname {{ font-weight: bold; white-space: nowrap; }}
.mono {{ font-family: "Menlo", "Courier New", monospace; font-size: 11px;
        white-space: pre; letter-spacing: 0.5px; }}
.diff {{ word-break: break-all; }}
.ok {{ color: #333; }}
.err {{ color: #d32f2f; font-weight: bold; text-decoration: underline;
       background: #fff3cd; padding: 0 1px; }}
.nogt {{ color: #999; font-style: italic; }}
.pass {{ color: #4caf50; font-weight: bold; }}
.fail {{ color: #f44336; font-weight: bold; }}
.band img {{ max-width: 400px; max-height: 80px; display: block;
            border: 1px solid #ccc; }}
.meta {{ font-size: 10px; color: #888; margin-top: 2px; }}
td:nth-child(1) {{ min-width: 120px; }}
</style>
</head>
<body>
<h1>训练样本审查报告</h1>
<div class="summary">
<span>总计: <b>{stats['total']}</b></span>
<span>有GT: <b>{stats['has_gt']}</b></span>
<span>GT匹配: <b style="color:#4caf50">{stats['gt_match']}</b></span>
<span>GT不匹配: <b style="color:#f44336">{stats['gt_mismatch']}</b></span>
<span>匹配率: <b>{match_pct:.1f}%</b></span>
<span>无GT: {stats['no_gt']}</span>
<span>未定位: {stats['no_pair']}</span>
</div>
<table>
<tr>
<th>文件名</th><th>分辨率</th><th>MRZ Band</th>
<th>GT line1</th><th>GT line2</th>
<th>识别 line1 (差异标红)</th><th>识别 line2 (差异标红)</th>
<th>状态</th><th>定位</th><th>耗时</th>
</tr>
{"".join(rows)}
</table>
<p style="margin-top:16px;font-size:11px;color:#888;">
说明: 绿色=PASS, 红色=FAIL, 灰色=无GT。识别列中黄色下划线字符=与GT不匹配。<br>
Band 图片标注: <span style="color:red">红线=L1 top/bot (pair边界)</span>,
<span style="color:blue">蓝线=L2 top/bot (pair边界)</span>,
<span style="color:green">绿线=band top/bot (padding后裁剪边界)</span>。
图片上下各多留20px上下文。2× 放大。<br>
生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}
</p>
</body>
</html>"""


if __name__ == "__main__":
    sys.exit(main())
