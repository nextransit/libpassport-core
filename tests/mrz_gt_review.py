#!/usr/bin/env python3
"""GT vs PRED 详细对比资料生成器（闭环第 2 步）。

输入：
  --bench   photo_direct_bench 输出的 json（默认取 10_ocr_research/archive/ 下最新 train*.json，
            可用 --bench 显式指定）
  --gt      tests/photo_auto_gt.json（唯一权威真值）
  --out     输出文件。以 .xlsx 结尾 → Excel 工作簿（含文件名 + 图片缩略图列 + 逐引擎 GT/PRED
            + 引擎汇总 + 需人工核对清单三个 sheet）；其它后缀 → 纯文本 markdown。

xlsx 结构：
  Sheet「GT对比」：每照片一行。图片缩略图列 + 文件名 + 官方判定/chosen + 官方 OCR
      + GT line1/line2 + 每引擎两列（PRED line1 / PRED line2，exact 绿底、有差异黄底、
      空灰底、decode_ok 蓝字加粗）。
  Sheet「引擎汇总」：每引擎 ran/decode_ok/exact/逐字符准确率。
  Sheet「需人工核对」：GT 缺失或全引擎不匹配的照片清单。
"""
import argparse
import io
import json
import os
import sys
from pathlib import Path

try:
    import openpyxl
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    from openpyxl.utils import get_column_letter
    from PIL import Image as PILImage
    _HAS_XLSX = True
except Exception:
    _HAS_XLSX = False

ROOT = Path(__file__).resolve().parents[1]
GT_DEFAULT = ROOT / "10_ocr_research" / "archive" / "photo_auto_gt_all.json"
ARCHIVE = ROOT / "10_ocr_research" / "archive"

# 验收门槛（与 mrz-gt-train-loop skill 的验收标准一致）
ACCEPT_MIN_PASS = 105          # 官方 PASS 张数下限
ACCEPT_MIN_RAW_FUSE = 84       # raw fuse decode 张数下限
ACCEPT_MIN_RATE = 2.0          # 官方单张路径速率下限（张/秒，约 ≤500ms/张）

STEPS = 44

CELL_W = 350    # 图片缩放到适配 A 列 cell（列宽 50ch ≈ 350px）
CELL_H = 280    # 图片高度适配行高 210pt ≈ 280px
ROW_H = 210     # 图片行高（Excel 行高单位 pt）

FILL_EXACT = PatternFill("solid", fgColor="C6EFCE")   # 绿色：与 GT 完全一致
FILL_DIFF = PatternFill("solid", fgColor="FFEB9C")    # 黄色：有差异
FILL_EMPTY = PatternFill("solid", fgColor="F2F2F2")   # 灰色：引擎无输出
FILL_SKIP = PatternFill("solid", fgColor="E7E6E6")    # 深灰：未运行/无 band
FONT_HEAD = Font(bold=True, color="FFFFFF")
FILL_HEAD = PatternFill("solid", fgColor="4472C4")
FONT_OK = Font(color="1F4E79", bold=True)             # decode_ok 蓝字加粗
FONT_LINE = Font(size=20)                              # 官方 line1/line2 大字
ALIGN_LEFT = Alignment(horizontal="left", vertical="center")
SIDE = Side(style="thin", color="BFBFBF")
BORDER = Border(left=SIDE, right=SIDE, top=SIDE, bottom=SIDE)


def load_gt(path):
    d = json.load(open(path, encoding="utf-8"))
    return d.get("gt") or d.get("records") or {}


def pick_bench(path):
    if path:
        p = Path(path)
        if not p.exists():
            sys.exit(f"bench json not found: {p}")
        return p
    cands = sorted(ARCHIVE.glob("photo_direct_bench_*.json"), reverse=True)
    if not cands:
        sys.exit("no photo_direct_bench_*.json found in archive/; use --bench")
    return cands[0]


def char_diff(gt, pred):
    n = max(len(gt), len(pred))
    g = gt.ljust(n)
    p = pred.ljust(n)
    mism = [i for i in range(n) if g[i] != p[i]]
    caret = "".join("^" if i in mism else " " for i in range(n))
    return caret, len(mism), n


def photo_thumb(path, max_w=None, max_h=None):
    """读照片返回可写入 openpyxl 的 BytesIO PIL 图。当 max_w/max_h 均给定且
    原图超界时，STRETCH（非等比）到该 cell 尺寸填满宽高（保持 LANCZOS）；
    max_w/max_h 与原图不符比例时仍强制拉伸；失败返回 None。"""
    try:
        p = Path(path)
        if not p.exists():
            p = ROOT / path
        if not p.exists():
            return None
        im = PILImage.open(p).convert("RGB")
        w, h = im.width, im.height
        if max_w and max_h and (w > max_w or h > max_h or (w, h) != (max_w, max_h)):
            im = im.resize((max_w, max_h), PILImage.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        buf.seek(0)
        return buf, im.width, im.height
    except Exception:
        return None


def collect_rows(recs, gt_map, methods):
    """收集逐照片数据 → [(label, base, path, official, gt1, gt2,
    [(m, e, l1, l2, ok1, ok2, nm1, nm2, dk, status)])]，并汇总 per_engine。"""
    rows = []
    per_engine = {m: {"ran": 0, "decode": 0, "exact": 0, "chars": 0,
                      "total": 0, "lines": 0, "ms": []} for m in methods}
    need_review = []

    for r in recs:
        label = r.get("label", "?")
        base = label.split("/")[-1]
        path = r.get("path") or label
        gt = gt_map.get(base) or {}
        if isinstance(gt, dict):
            gt1, gt2 = gt.get("l1"), gt.get("l2")
        else:
            gt1, gt2 = gt
        fallback_gt = r.get("gt")
        if not gt1 and isinstance(fallback_gt, dict):
            gt1, gt2 = fallback_gt.get("line1"), fallback_gt.get("line2")
        elif not gt1 and isinstance(fallback_gt, (list, tuple)):
            gt1, gt2 = fallback_gt
        official = r.get("official") or {}
        raw = r.get("raw") or {}

        engines = []
        matched = 0
        for m in methods:
            e = raw.get(m)
            if not e or e.get("status") != "run":
                engines.append((m, e, None, None, False, False, 0, 0, False, e.get("status") if e else "no_record"))
                continue
            dk = bool(e.get("decode_ok"))
            l1, l2 = e.get("line1"), e.get("line2")
            ok1 = ok2 = False
            nm1 = nm2 = 0
            if gt1:
                _, nm1, _ = char_diff(gt1, l1 or "")
                ok1 = (gt1 == l1)
            if gt2:
                _, nm2, _ = char_diff(gt2, l2 or "")
                ok2 = (gt2 == l2)
            pe = per_engine[m]
            pe["ran"] += 1
            pe["ms"].append(float(e.get("ms") or 0.0))
            if dk:
                pe["decode"] += 1
            if gt1 and l1 == gt1 and gt2 and l2 == gt2:
                pe["exact"] += 1
                matched += 1
            for gv, pv in ((gt1, l1), (gt2, l2)):
                if gv:
                    _, nm, _ = char_diff(gv, pv or "")
                    pe["chars"] += nm
                    pe["lines"] += 1
            engines.append((m, e, l1, l2, ok1, ok2, nm1, nm2, dk, "run"))
        rows.append((label, base, path, official, gt1, gt2, engines))
        if gt1 or gt2:
            need_review.append((base, gt1, gt2, matched, label))

    return rows, per_engine, need_review


def _pct(values, p):
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = int(round((p / 100.0) * (len(ordered) - 1)))
    return ordered[max(0, min(len(ordered) - 1, idx))]


def acceptance_summary(data, recs, per_engine, methods, baseline=None):
    """汇总验收指标：速率（吞吐）+ 通过率（官方/GT）+ 门槛判定 + 退化对比。

    返回 dict，含官方 PASS/REJECT、官方与各引擎的速率（张/秒）、
    通过率（官方 PASS 率、GT 精确率、逐字符准确率）以及与基线的差异。
    """
    official = data.get("official") or {}
    raw = data.get("raw") or {}
    photos = len(recs) or official.get("photos") or 0
    grades = official.get("grades") or {}
    pass_n = grades.get("PASS", 0)
    reject_n = grades.get("REJECT", 0)
    elapsed = float(data.get("elapsed_s") or 0.0)
    bench_throughput = (photos / elapsed) if elapsed > 0 else 0.0

    # 官方路径单张耗时（毫秒）→ 速率
    official_ms = [float((r.get("official") or {}).get("ms", 0.0)) for r in recs]
    official_ms = [v for v in official_ms if v > 0]
    official_mean_ms = sum(official_ms) / len(official_ms) if official_ms else 0.0
    # 速率口径：官方单张识别路径（1000 / 平均耗时），与机器负载无关；
    # 批测墙钟吞吐另列，仅作参考。
    official_rate = (1000.0 / official_mean_ms) if official_mean_ms > 0 else 0.0

    engines = {}
    for m in methods:
        st = raw.get(m)
        if not isinstance(st, dict):
            continue
        ms = [float(v) for v in (st.get("ms") or [])]
        ran = st.get("ran") or 0
        decode = st.get("decode") or 0
        located = st.get("located") or 0
        mean_ms = sum(ms) / len(ms) if ms else 0.0
        engines[m] = {
            "ran": ran,
            "decode": decode,
            "located": located,
            "mean_ms": round(mean_ms, 2),
            "p95_ms": round(_pct(ms, 95), 2),
            "throughput": round((1000.0 / mean_ms) if mean_ms > 0 else 0.0, 3),
            "decode_rate": round(decode / photos * 100.0, 1) if photos else 0.0,
            "locate_rate": round(located / photos * 100.0, 1) if photos else 0.0,
            "gt_total": st.get("gt_total") or 0,
            "gt_exact": st.get("gt_exact") or 0,
            "char_acc1": round(sum(st.get("acc1") or []) / len(st["acc1"]) * 100.0, 2)
                          if st.get("acc1") else 0.0,
            "char_acc2": round(sum(st.get("acc2") or []) / len(st["acc2"]) * 100.0, 2)
                          if st.get("acc2") else 0.0,
        }

    fuse = raw.get("fuse") or {}
    raw_fuse_decode = fuse.get("decode") or 0

    summary = {
        "generated_at": data.get("generated_at") or "",
        "bench": Path(data.get("__path__", "")).name,
        "photos": photos,
        "elapsed_s": round(elapsed, 3),
        "official": {
            "pass": pass_n,
            "reject": reject_n,
            "suspect": grades.get("SUSPECT", 0),
            "verified": official.get("verified") or 0,
            "pass_rate": round(pass_n / photos * 100.0, 2) if photos else 0.0,
            "ms_mean": round(official_mean_ms, 2),
            "ms_median": round(float(official.get("ms_median") or 0.0), 2),
            "ms_p95": round(float(official.get("ms_p95") or 0.0), 2),
            "rate": round(official_rate, 3),
            "bench_throughput": round(bench_throughput, 3),
        },
        "raw_fuse_decode": raw_fuse_decode,
        "engines": engines,
    }

    checks = [
        ("官方 PASS 张数", pass_n, str(ACCEPT_MIN_PASS), "ge"),
        ("raw fuse decode 张数", raw_fuse_decode, str(ACCEPT_MIN_RAW_FUSE), "ge"),
        ("官方路径速率（张/秒）", round(official_rate, 3),
         str(ACCEPT_MIN_RATE), "ge"),
    ]
    summary["checks"] = [
        {"name": n, "value": v, "threshold": t, "op": op,
         "pass": (v >= float(t)) if op == "ge" else (v <= float(t))}
        for n, v, t, op in checks
    ]
    summary["all_pass"] = all(c["pass"] for c in summary["checks"])

    if baseline:
        b = baseline.get("official") or {}
        b_raw = baseline.get("raw") or {}
        b_fuse = (b_raw.get("fuse") or {}).get("decode") or 0
        b_elapsed = float(baseline.get("elapsed_s") or 0.0)
        b_photos = len(baseline.get("records") or []) or b.get("photos") or 0
        b_ms = float(b.get("ms_mean") or 0.0)
        b_rate = (1000.0 / b_ms) if b_ms > 0 else 0.0
        summary["baseline"] = {
            "bench": baseline.get("__name__", ""),
            "pass": b.get("grades", {}).get("PASS", 0),
            "raw_fuse_decode": b_fuse,
            "rate": round(b_rate, 3),
            "bench_throughput": round(b_photos / b_elapsed, 3) if b_elapsed > 0 else 0.0,
            "ms_mean": round(b_ms, 2),
            "delta_pass": pass_n - b.get("grades", {}).get("PASS", 0),
            "delta_raw_fuse": raw_fuse_decode - b_fuse,
            "delta_rate": round(official_rate - b_rate, 3),
            "raw": b_raw,
        }
        # 逐张退化：基线 PASS 且本次非 PASS
        regressed = []
        base_by_name = {}
        for r in baseline.get("records") or []:
            name = str(r.get("label", "")).split("/")[-1]
            base_by_name[name] = (r.get("official") or {}).get("grade")
        for r in recs:
            name = str(r.get("label", "")).split("/")[-1]
            now = (r.get("official") or {}).get("grade")
            if base_by_name.get(name) == "PASS" and now != "PASS":
                regressed.append((name, base_by_name.get(name), now))
        summary["regressed"] = regressed
        summary["regressed_count"] = len(regressed)
    return summary


def write_xlsx(out_path, bench, recs, methods, rows, per_engine, need_review, gt_name,
                summary=None):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "GT对比"

    # 表头：A 图片、B 文件名、C 官方判定、D chosen、E 官方 line1、F 官方 line2、
    #       G GT line1、H GT line2，然后每引擎两列: {m} l1 / {m} l2
    headers = ["图片", "文件名", "官方判定", "chosen", "官方 line1", "官方 line2",
               "GT line1", "GT line2"]
    width_map = [50, 34, 10, 14, STEPS + 4, STEPS + 4, STEPS + 3, STEPS + 3]
    for m in methods:
        headers.append(f"{m} · l1")
        headers.append(f"{m} · l2")
        width_map += [STEPS + 2, STEPS + 2]

    for c, h in enumerate(headers, 1):
        cell = ws.cell(row=1, column=c, value=h)
        cell.font = FONT_HEAD
        cell.fill = FILL_HEAD
        cell.alignment = Alignment(horizontal="center", vertical="center")
        cell.border = BORDER
    for c, w in enumerate(width_map, 1):
        ws.column_dimensions[get_column_letter(c)].width = min(w, 60)

    ws.freeze_panes = "C2"   # 冻结 图片/文件名 两列

    R, C_BASE = 2, 8          # 数据从第 2 行开始；引擎列起始 = C_BASE+1
    next_col = C_BASE + 1

    for idx, (label, base, path, official, gt1, gt2, engines) in enumerate(rows):
        ws.cell(row=R, column=1)   # A 列占位（true 图片锚定在 A 列）
        ws.cell(row=R, column=2, value=base)
        ws.cell(row=R, column=3, value=official.get("grade", ""))
        ws.cell(row=R, column=4, value=official.get("chosen") or "")
        cell5 = ws.cell(row=R, column=5, value=(official.get("line1") or "")[:STEPS])
        cell5.font = FONT_LINE
        cell5.alignment = ALIGN_LEFT
        cell6 = ws.cell(row=R, column=6, value=(official.get("line2") or "")[:STEPS])
        cell6.font = FONT_LINE
        cell6.alignment = ALIGN_LEFT
        ws.cell(row=R, column=7, value=(gt1 or "")[:STEPS])
        ws.cell(row=R, column=8, value=(gt2 or "")[:STEPS])

        tb = photo_thumb(path, max_w=CELL_W, max_h=CELL_H)
        if tb is not None:
            buf, iw, ih = tb
            img = openpyxl.drawing.image.Image(buf)
            img.width = iw
            img.height = ih
            ws.add_image(img, f"A{R}")
            ws.row_dimensions[R].height = ROW_H
        else:
            ws.row_dimensions[R].height = ROW_H

        for m, e, l1, l2, ok1, ok2, nm1, nm2, dk, status in engines:
            col = next_col
            next_col += 2
            for sub, val, ok, nm in ((0, l1, ok1, nm1), (1, l2, ok2, nm2)):
                cell = ws.cell(row=R, column=col + sub)
                if status != "run":
                    cell.value = ""
                    cell.fill = FILL_SKIP
                else:
                    cell.value = (val or "")[:STEPS]
                    if ok:
                        cell.fill = FILL_EXACT
                    elif val:
                        cell.fill = FILL_DIFF
                    else:
                        cell.fill = FILL_EMPTY
                cell.border = BORDER
                cell.alignment = Alignment(vertical="center")
                if dk:
                    cell.font = FONT_OK
        R += 1
        next_col = C_BASE + 1

    # ---- Sheet「引擎汇总」 ----
    ws2 = wb.create_sheet("引擎汇总")
    for c, h in enumerate(["engine", "ran", "decode_ok", "exact(GT)", "char acc"], 1):
        cell = ws2.cell(row=1, column=c, value=h)
        cell.font = FONT_HEAD
        cell.fill = FILL_HEAD
        cell.border = BORDER
    rr = 2
    for m in methods:
        pe = per_engine[m]
        if pe["ran"] == 0:
            continue
        acc = (1 - pe["chars"] / max(1, pe["lines"] * STEPS)) * 100 if pe["lines"] else 0
        for c, v in enumerate([m, pe["ran"], pe["decode"], pe["exact"], round(acc, 1)], 1):
            cell = ws2.cell(row=rr, column=c, value=v)
            cell.border = BORDER
        rr += 1

    # ---- Sheet「需人工核对」 ----
    ws3 = wb.create_sheet("需人工核对")
    for c, h in enumerate(["文件名", "GT line1", "GT line2", "匹配引擎数", "图片"], 1):
        cell = ws3.cell(row=1, column=c, value=h)
        cell.font = FONT_HEAD
        cell.fill = FILL_HEAD
        cell.border = BORDER
    ws3.column_dimensions["A"].width = 34
    ws3.column_dimensions["B"].width = STEPS + 3
    ws3.column_dimensions["C"].width = STEPS + 3
    ws3.column_dimensions["D"].width = 12
    ws3.column_dimensions["E"].width = 50
    for rr, (base, gt1, gt2, matched, label) in enumerate(need_review, 2):
        ws3.cell(row=rr, column=1, value=base)
        ws3.cell(row=rr, column=2, value=gt1 or "")
        ws3.cell(row=rr, column=3, value=gt2 or "")
        ws3.cell(row=rr, column=4, value=matched)
        tb = photo_thumb(label, max_w=CELL_W, max_h=CELL_H)
        if tb is not None:
            buf, iw, ih = tb
            img = openpyxl.drawing.image.Image(buf)
            img.width = iw
            img.height = ih
            ws3.add_image(img, f"E{rr}")
        ws3.row_dimensions[rr].height = ROW_H

    # ---- Sheet「全算法验收」 ----
    # One row per engine: 速率 + 通过率 + GT 精确率 + 与基线差值.
    # The engine list comes from the bench payload, so every engine that
    # produced a record is covered (including indra_fullpage).
    ws5 = wb.create_sheet("全算法验收", 1)
    head = ["engine", "ran", "decode_ok", "decode率(%)", "GT精确率(%)",
            "逐字符准确率l1(%)", "逐字符准确率l2(%)",
            "平均耗时(ms)", "P95耗时(ms)", "速率(张/秒)", "基线decode", "Δdecode"]
    for c, h in enumerate(head, 1):
        cell = ws5.cell(row=1, column=c, value=h)
        cell.font = FONT_HEAD
        cell.fill = FILL_HEAD
        cell.border = BORDER
        cell.alignment = Alignment(horizontal="center", vertical="center")
    for c, w in enumerate([16, 7, 10, 12, 12, 16, 16, 13, 13, 13, 12, 10], 1):
        ws5.column_dimensions[get_column_letter(c)].width = w
    ws5.freeze_panes = "B2"

    baseline_raw = (summary.get("baseline", {}) or {}).get("raw") or {}
    rr = 2
    for m in methods:
        st = summary["engines"].get(m)
        if not st:
            continue
        b_st = baseline_raw.get(m) or {}
        b_decode = b_st.get("decode")
        values = [
            m,                                   # 1 engine
            st["ran"],                           # 2 ran
            st["decode"],                        # 3 decode_ok
            st["decode_rate"],                   # 4 decode率
            round(st["gt_exact"] / st["gt_total"] * 100.0, 1)
            if st["gt_total"] else 0,            # 5 GT精确率
            st["char_acc1"],                     # 6 逐字符 l1
            st["char_acc2"],                     # 7 逐字符 l2
            st["mean_ms"],                       # 8 平均耗时
            st["p95_ms"],                        # 9 P95耗时
            st["throughput"],                    # 10 速率
            b_decode if b_decode is not None else "-",   # 11 基线decode
            (st["decode"] - b_decode) if b_decode is not None else "-",  # 12 Δ
        ]
        for c, v in enumerate(values, 1):
            cell = ws5.cell(row=rr, column=c, value=v)
            cell.border = BORDER
        delta = values[11] if len(values) > 11 else None
        if isinstance(delta, int):
            if delta < 0:
                ws5.cell(row=rr, column=12).fill = FILL_DIFF
            elif delta > 0:
                ws5.cell(row=rr, column=12).fill = FILL_EXACT
        rr += 1

    # ---- Sheet「验收汇总」 ----
    summary = summary or {}
    ws4 = wb.create_sheet("验收汇总", 0)
    for c, h in enumerate(["项目", "数值", "门槛", "判定"], 1):
        cell = ws4.cell(row=1, column=c, value=h)
        cell.font = FONT_HEAD
        cell.fill = FILL_HEAD
        cell.border = BORDER
    ws4.column_dimensions["A"].width = 34
    ws4.column_dimensions["B"].width = 18
    ws4.column_dimensions["C"].width = 14
    ws4.column_dimensions["D"].width = 10

    def put(rr, name, value, threshold="", verdict=""):
        cells = (name, value, threshold, verdict)
        for c, v in enumerate(cells, 1):
            cell = ws4.cell(row=rr, column=c, value=v)
            cell.border = BORDER
            if verdict == "PASS":
                cell.fill = FILL_EXACT
            elif verdict == "FAIL":
                cell.fill = FILL_DIFF

    rr = 2
    off = summary["official"]
    put(rr, "批测来源", summary["bench"]); rr += 1
    put(rr, "生成时间", summary["generated_at"]); rr += 1
    put(rr, "照片总数", summary["photos"]); rr += 1
    put(rr, "批测总耗时（s）", summary["elapsed_s"]); rr += 1
    rr += 1
    ws4.cell(row=rr, column=1, value="速率").font = Font(bold=True)
    rr += 1
    put(rr, "【产品路径】平均耗时（ms）", off["ms_mean"]); rr += 1
    put(rr, "【产品路径】中位耗时（ms）", off["ms_median"]); rr += 1
    put(rr, "【产品路径】P95 耗时（ms）", off["ms_p95"]); rr += 1
    put(rr, "【产品路径】速率（张/秒）", off["rate"]); rr += 1
    put(rr, "【产品路径】批测墙钟吞吐（张/秒，参考）",
        off["bench_throughput"]); rr += 1
    put(rr, "【单引擎最快】速率（张/秒）",
        max((e["throughput"] for e in summary["engines"].values()),
            default=0.0)); rr += 1
    rr += 1
    ws4.cell(row=rr, column=1, value="各引擎单次识别速率").font = Font(bold=True)
    rr += 1
    for m, e in summary["engines"].items():
        put(rr, f"{m} 速率（张/秒）", e["throughput"]); rr += 1
    rr += 1
    ws4.cell(row=rr, column=1, value="通过率").font = Font(bold=True)
    rr += 1
    put(rr, "官方 PASS 张数", off["pass"]); rr += 1
    put(rr, "官方 PASS 率（%）", off["pass_rate"]); rr += 1
    put(rr, "官方 REJECT 张数", off["reject"]); rr += 1
    put(rr, "官方 verified 张数", off["verified"]); rr += 1
    put(rr, "raw fuse decode 张数", summary["raw_fuse_decode"]); rr += 1
    for m, e in summary["engines"].items():
        put(rr, f"{m} decode 率（%）", e["decode_rate"]); rr += 1
        put(rr, f"{m} GT 精确率（%）",
            round(e["gt_exact"] / e["gt_total"] * 100.0, 1) if e["gt_total"] else 0.0)
        rr += 1
        put(rr, f"{m} 逐字符准确率 l1/l2（%）",
            f"{e['char_acc1']}/{e['char_acc2']}"); rr += 1
    rr += 1
    ws4.cell(row=rr, column=1, value="门槛判定").font = Font(bold=True)
    rr += 1
    for chk in summary["checks"]:
        put(rr, chk["name"], chk["value"], chk["threshold"],
            "PASS" if chk["pass"] else "FAIL")
        rr += 1
    put(rr, "总体结论", "PASS" if summary["all_pass"] else "FAIL", "",
        "PASS" if summary["all_pass"] else "FAIL")
    rr += 2
    if summary.get("baseline"):
        b = summary["baseline"]
        ws4.cell(row=rr, column=1, value="与基线对比").font = Font(bold=True)
        rr += 1
        put(rr, "基线批测", b["bench"]); rr += 1
        put(rr, "PASS 差值（张）", b["delta_pass"]); rr += 1
        put(rr, "raw fuse 差值（张）", b["delta_raw_fuse"]); rr += 1
        put(rr, "速率差值（张/秒）", b["delta_rate"]); rr += 1
        put(rr, "退化张数（基线 PASS → 本次非 PASS）",
            summary["regressed_count"], "0",
            "PASS" if summary["regressed_count"] == 0 else "FAIL")
        rr += 1
        for name, was, now in summary["regressed"]:
            put(rr, f"退化: {name}", f"{was} → {now}"); rr += 1

    wb.save(out_path)
    print(f"wrote {out_path}  ({len(recs)} photos, {len(rows)} rows; "
          f"engine sheets: 汇总 {len(per_engine)} / 核对 {len(need_review)})")


def write_md(out_path, bench, recs, methods, rows, per_engine, need_review, gt_name,
             summary=None):
    out = []
    out.append(f"# MRZ GT vs PRED 对比（来源: {bench.name}）\n")
    out.append(f"照片总数 {len(recs)}  真值来源 {Path(gt_name).name}\n")

    summary = summary or {}
    if summary:
        off = summary["official"]
        out.append("## 验收汇总\n")
        out.append("| 项目 | 数值 | 门槛 | 判定 |")
        out.append("|---|---|---|---|")
        for chk in summary["checks"]:
            out.append(f"| {chk['name']} | {chk['value']} | >= {chk['threshold']} | "
                       f"{'PASS' if chk['pass'] else 'FAIL'} |")
        out.append(f"| 总体结论 | {'PASS' if summary['all_pass'] else 'FAIL'} | | "
                   f"{'PASS' if summary['all_pass'] else 'FAIL'} |")
        out.append("")
        fastest = max((e["throughput"] for e in summary["engines"].values()),
                      default=0.0)
        out.append(f"- 速率【产品路径（全引擎并发）】：{off['rate']} 张/秒，"
                   f"平均 {off['ms_mean']} ms，中位 {off['ms_median']} ms，"
                   f"P95 {off['ms_p95']} ms")
        out.append(f"- 速率【单引擎最快】：{fastest} 张/秒；"
                   f"批测墙钟吞吐 {off['bench_throughput']} 张/秒（参考）")
        out.append(f"- 通过率：官方 PASS {off['pass']}/{summary['photos']} "
                   f"= {off['pass_rate']}%，REJECT {off['reject']}，"
                   f"raw fuse decode {summary['raw_fuse_decode']}")
        if summary.get("baseline"):
            b = summary["baseline"]
            out.append(f"- 基线对比（{b['bench']}）：PASS {b['delta_pass']:+d}，"
                       f"raw fuse {b['delta_raw_fuse']:+d}，"
                       f"速率 {b['delta_rate']:+} 张/秒，"
                       f"退化 {summary['regressed_count']} 张")
        out.append("")

    for label, base, path, official, gt1, gt2, engines in rows:
        out.append(f"\n## {base}")
        out.append(f"- 官方判定: {official.get('grade')}  verified={official.get('verified')}  "
                   f"chosen={official.get('chosen')}  ms={official.get('ms', 0):.0f}")
        if official.get("line1"):
            out.append(f"- 官方 line1: {official['line1']}")
        if official.get("line2"):
            out.append(f"- 官方 line2: {official['line2']}")
        if official.get("reason"):
            out.append(f"- 理由: {official['reason']}")
        if gt1:
            out.append(f"\n- GT line1: `{gt1}`")
        if gt2:
            out.append(f"- GT line2: `{gt2}`")

        for m, e, l1, l2, ok1, ok2, nm1, nm2, dk, status in engines:
            if status != "run":
                out.append(f"\n### {m}  [{status}]")
                continue
            verdict = "DECODE-OK" if dk else ("decoded" if (l1 or l2) else "FAIL")
            conf = f" conf1={e.get('conf1')}% conf2={e.get('conf2')}%" if (e and (e.get("conf1") or e.get("conf2"))) else ""
            out.append(f"\n### {m}  [{verdict}]{conf}")
            if e and e.get("backend"):
                out.append(f"- backend: {e['backend']}  ({e.get('ms', 0):.0f} ms)")
            if e and e.get("error"):
                out.append(f"- error: {e['error']}")
            for gv, pv, tag in ((gt1, l1, "line1"), (gt2, l2, "line2")):
                if not gv:
                    continue
                if pv:
                    caret, nm, n = char_diff(gv, pv)
                    out.append(f"GT  : {gv.ljust(n)}")
                    out.append(f"PRED: {pv.ljust(n)}")
                    out.append(f"      {caret}   [{nm} mismatch]")
                else:
                    out.append(f"- PRED {tag}: <空>")

    out.append(f"\n## 引擎汇总\n")
    out.append("| engine | ran | decode_ok | exact(GT) | char acc |")
    out.append("|---|---|---|---|---|")
    for m in methods:
        pe = per_engine[m]
        if pe["ran"] == 0:
            continue
        acc = (1 - pe["chars"] / max(1, pe["lines"] * STEPS)) * 100 if pe["lines"] else 0
        out.append(f"| {m} | {pe['ran']} | {pe['decode']} | {pe['exact']} | {acc:.1f}% |")

    out.append(f"\n## 需人工核对的照片（{len(need_review)} 张，GT 已缺失或全引擎不匹配）\n")
    for base, gt1, gt2, matched, label in need_review:
        mstr = matched if matched else "无引擎完全匹配"
        out.append(f"- [{base}] ({mstr})")
        if gt1:
            out.append(f"    l1=`{gt1}`")
        if gt2:
            out.append(f"    l2=`{gt2}`")

    text = "\n".join(out) + "\n"
    Path(out_path).write_text(text, encoding="utf-8")
    print(f"wrote {out_path}  ({len(recs)} photos)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench")
    ap.add_argument("--gt", default=str(GT_DEFAULT))
    ap.add_argument("--out")
    ap.add_argument("--baseline",
                    help="基线 bench json（用于速率/通过率对比与退化清单）")
    ap.add_argument("--min-scale", type=float, default=0.0,
                    help="只输出字符准确率的阈值以上照片（调参用，未实现）")
    args = ap.parse_args()

    gt_map = load_gt(args.gt)
    bench = pick_bench(args.bench)
    data = json.load(open(bench, encoding="utf-8"))
    recs = data.get("records", [])
    methods = data.get("methods") or ["traditional", "cnn", "crnn", "fuse",
                                      "tesseract", "paddle", "indra",
                                      "indra_fullpage"]
    # Every engine that actually produced a record must be evaluated,
    # even when it is not in the declared method list (e.g.
    # "indra_fullpage", the extra full-photo Indra pass).
    for key in (data.get("raw") or {}):
        if key not in methods:
            methods.append(key)

    rows, per_engine, need_review = collect_rows(recs, gt_map, methods)

    baseline = None
    if args.baseline:
        bp = Path(args.baseline)
        if not bp.exists():
            sys.exit(f"baseline bench json not found: {bp}")
        baseline = json.load(open(bp, encoding="utf-8"))
        baseline["__name__"] = bp.name
    data["__path__"] = str(bench)
    summary = acceptance_summary(data, recs, per_engine, methods, baseline)

    out_path = args.out
    if out_path and str(out_path).lower().endswith(".xlsx"):
        if not _HAS_XLSX:
            sys.exit("xlsx 输出需要 openpyxl + Pillow，请先安装")
        write_xlsx(out_path, bench, recs, methods, rows, per_engine, need_review,
                   args.gt, summary)
    else:
        target = out_path or sys.stdout
        if target is sys.stdout:
            write_md(target, bench, recs, methods, rows, per_engine, need_review,
                     args.gt, summary)
        else:
            write_md(out_path, bench, recs, methods, rows, per_engine, need_review,
                     args.gt, summary)


if __name__ == "__main__":
    main()
