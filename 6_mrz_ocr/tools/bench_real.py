#!/usr/bin/env python3
"""真实护照照片评测（板端 bbox 裁 band + 板端 fixmrz 弱标签对比）。

数据源: tmp/omtest/pic_scan{1,2,3}/*.jpg-result.txt
  - textLine[0]/[1] = 板端 PaddleOCR 原始两行
  - fixmrz 段 = 板端修正后两行 (若存在, 优先作为参考标签)
  - TextBox[0]/[1] + 板端 onlymrz 裁剪偏移 = 真实 MRZ band 的 bbox

对每个可用样本:
  1. 用 bbox 从原图裁出 MRZ band (全宽, 上下 pad 4px)
  2. 跑 mrz_ocr_cnn_tool (当前权重) / mrz_ocr_tool (传统) / paddle_v6 / tesseract
  3. 输出与弱标签逐字符对比 line1/line2 acc 与 line2 ICAO 校验

用法:
  python3 tools/bench_real.py [OMTEST_DIR] [--out md]
"""
from __future__ import annotations
import argparse, json, re, subprocess, sys, time
from pathlib import Path
import cv2
from PIL import Image

W = [7, 3, 1]

def check_digit(seg: str) -> int:
    t = 0
    for i, c in enumerate(seg):
        v = (ord(c) - 48) if '0' <= c <= '9' else ((ord(c) - 55) if 'A' <= c <= 'Z' else 0)
        t += v * W[i % 3]
    return t % 10

def l2_cca(l2: str) -> bool:
    if len(l2) != 44: return False
    try:
        return (check_digit(l2[0:9]) == int(l2[9]) and
                check_digit(l2[13:19]) == int(l2[19]) and
                check_digit(l2[21:27]) == int(l2[27]) and
                check_digit(l2[28:42]) == int(l2[42]) and
                check_digit(l2[0:10] + l2[13:20] + l2[21:43]) == int(l2[43]))
    except ValueError:
        return False

def char_acc(a, b):
    n = max(len(a), len(b)); c = 0
    for i in range(n):
        g = a[i] if i < len(a) else "_"; h = b[i] if i < len(b) else "_"
        c += g == h
    return c / n if n else 0.0

def parse_result(txt: Path):
    s = txt.read_text(errors="replace")
    tl = re.findall(r'textLine\[\d\]\(([^\n]*)\)', s)
    mfix = re.findall(r'fixmrz[^\n]*\n([PE]<[0-9A-Z<]{43})\n([0-9A-Z<]{44})', s)
    l1, l2 = (mfix[0] if mfix else (tl[0] if tl else None, tl[1] if len(tl) > 1 else None))
    pts = [(int(a), int(b)) for a, b in re.findall(r'\[x: (\d+), y: (\d+)\]', s)]
    boxes = [pts[i:i + 4] for i in range(0, len(pts), 4)]
    return l1, l2, boxes

def run_tool(tool: Path, ppm: Path, is_py=False):
    t0 = time.perf_counter()
    try:
        cmd = [sys.executable, str(tool), str(ppm)] if is_py else [str(tool), str(ppm)]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        ms = (time.perf_counter() - t0) * 1000
    except Exception:
        return {}, False, 0.0
    kv = {}
    for ln in r.stdout.splitlines():
        if ":" not in ln: continue
        k, _, v = ln.partition(":")
        kv[k.strip()] = v.strip()
    return kv, r.returncode == 0, ms

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("omtest", nargs="?", default="../tmp/omtest")
    ap.add_argument("--out", default="")
    ap.add_argument("--tools", default="cnn,paddle,tess")
    a = ap.parse_args()
    ROOT = Path(__file__).resolve().parents[1]
    omtest = Path(a.omtest)
    tools = {
        "cnn": (ROOT / "build" / "mrz_ocr_cnn_tool", False),
        "trad": (ROOT / "build" / "mrz_ocr_tool", False),
        "paddle": (ROOT.parent.parent / "6_1_PaddleOCR" / "paddle_v6.py", True),
        "tess": (ROOT.parent.parent / "6_2_Tesseract" / "tesseract_tool.py", True),
    }
    want = [t.strip() for t in a.tools.split(",") if t.strip() in tools]

    rows = []
    for d in sorted(p for p in omtest.iterdir() if p.is_dir()):
        for txt in sorted((omtest / d.name).glob("*.jpg-result.txt")):
            img = txt.with_name(txt.name.replace("-result.txt", ""))
            if not img.exists(): continue
            l1, l2, boxes = parse_result(txt)
            if not l1 or not l2 or len(l1) != 44 or len(l2) != 44 or len(boxes) < 2:
                continue
            rgb = cv2.imread(str(img)); H, W = rgb.shape[:2]
            crop_y = H - (H // 3) - 50   # 板端 onlymrz: bottom 1/3 minus 50px
            y1 = min(p[1] for p in boxes[0]) + crop_y
            y2b = max(p[1] for p in boxes[1]) + crop_y
            band = rgb[max(0, y1 - 4):y2b + 5, 0:W]
            ppm = ROOT / "data" / "_real_bench.ppm"
            Image.fromarray(cv2.cvtColor(band, cv2.COLOR_BGR2RGB)).save(str(ppm), format="PPM")
            entry = {"file": img.name, "dir": d.name, "gt1": l1, "gt2": l2,
                     "gt_cksum": l2_cca(l2), "engines": {}}
            for name in want:
                tool, is_py = tools[name]
                kv, ok, ms = run_tool(tool, ppm, is_py)
                o1, o2 = kv.get("result.line1", ""), kv.get("result.line2", "")
                entry["engines"][name] = {
                    "ok": kv.get("result.ok", "?"), "line1": o1, "line2": o2,
                    "conf1": kv.get("result.conf1", ""), "conf2": kv.get("result.conf2", ""),
                    "ms": round(ms, 1), "l1_acc": round(char_acc(l1, o1), 3),
                    "l2_acc": round(char_acc(l2, o2), 3),
                    "full": o1 == l1 and o2 == l2, "cksum": l2_cca(o2),
                }
            rows.append(entry)
            print(f"{d.name}/{entry['file']} gt_cksum={entry['gt_cksum']} " +
                  " ".join(f"{n}:{entry['engines'][n]['ok']} "
                           f"l1={entry['engines'][n]['l1_acc']}"
                           f"/l2={entry['engines'][n]['l2_acc']}" for n in want))

    n = len(rows)
    print(f"\nromtest samples with full two-line GT: {n}")
    for name in want:
        ok = sum(1 for r in rows if r["engines"][name]["cksum"])
        full = sum(1 for r in rows if r["engines"][name]["full"])
        a1 = sum(r["engines"][name]["l1_acc"] for r in rows) / n
        a2 = sum(r["engines"][name]["l2_acc"] for r in rows) / n
        print(f"  {name:6s}: l1_acc={a1:.3f} l2_acc={a2:.3f} full={full}/{n} "
              f"l2_cksum_valid={ok}/{n}")
    if n and a.out:
        md = ["# 真实护照照片评测\n",
              f"> {omtest} pic_scan{{1,2,3}} n={n}; 弱标签来自板端 fixmrz.\n",
              "| file | gt_cksum | " + " | ".join(f"{m} l1/l2/full" for m in want) + " |"]
        for r in rows:
            cols = [f"{r['dir']}/{r['file']}", str(r["gt_cksum"])]
            for m in want:
                e = r["engines"][m]
                cols.append(f"{e['l1_acc']}/{e['l2_acc']}/{'Y' if e['full'] else 'N'}")
            md.append("| " + " | ".join(cols) + " |")
        md.append("")
        for m in want:
            ok = sum(1 for r in rows if r["engines"][m]["cksum"])
            full = sum(1 for r in rows if r["engines"][m]["full"])
            a1 = sum(r["engines"][m]["l1_acc"] for r in rows) / n
            a2 = sum(r["engines"][m]["l2_acc"] for r in rows) / n
            md.append(f"- **{m}**: l1_acc={a1:.3f} l2_acc={a2:.3f} "
                      f"full={full}/{n} cksum={ok}/{n}")
        Path(a.out).parent.mkdir(exist_ok=True)
        Path(a.out).write_text("\n".join(md) + "\n")
        print("wrote", a.out)
    return 0

if __name__ == "__main__":
    sys.exit(main())
