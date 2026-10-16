#!/usr/bin/env python3
"""探针：对 Photo/ 护照机照片跑仓库内 4 套 OCR 后端（含矫正后 crop），
输出每张的 decode 结果 + conf + 耗时，作为 10_ocr_research 调研第一手数据。

后端：
  0) cnn      6_mrz_ocr/build/mrz_ocr_cnn_tool
  1) template 6_mrz_ocr/build/mrz_ocr_tool
  2) tesseract 6_2_Tesseract/tesseract_tool.py (ocrb)
  3) paddle    6_1_PaddleOCR/paddle_v6.py
"""
from __future__ import annotations
import json
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "passport_photo_restore" / "src"))
from restore import load, _gray, detect_structure, rectify  # noqa: E402

PHOTO = ROOT / "Photo"
TOOLS = {
    "cnn": ROOT / "6_mrz_ocr" / "build" / "mrz_ocr_cnn_tool",
    "template": ROOT / "6_mrz_ocr" / "build" / "mrz_ocr_tool",
    "tesseract": ROOT / "6_2_Tesseract" / "tesseract_tool.py",
    "paddle": ROOT / "6_1_PaddleOCR" / "paddle_v6.py",
}


def run(tool: Path, img: Path) -> dict:
    t0 = time.perf_counter()
    args = [sys.executable, str(tool), str(img)] if tool.name.endswith(".py") else [str(tool), str(img)]
    try:
        r = subprocess.run(args, text=True, capture_output=True, timeout=80)
        ms = (time.perf_counter() - t0) * 1000
    except subprocess.TimeoutExpired:
        return {"ok": False, "line1": "", "line2": "", "ms": -1, "err": "timeout"}
    out = {}
    for ln in r.stdout.splitlines():
        if ":" in ln:
            k, _, v = ln.partition(":")
            out[k.strip()] = v.strip()
    return {
        "ok": r.returncode == 0 and out.get("result.ok", "") == "OK",
        "line1": out.get("result.line1", ""),
        "line2": out.get("result.line2", ""),
        "conf1": out.get("result.conf1", ""),
        "conf2": out.get("result.conf2", ""),
        "ms": round(ms, 1),
        "err": out.get("result.ok", r.stderr.strip()[:80]),
    }


def write_ppm(numpy_rgb, p: Path):
    from PIL import Image
    Image.fromarray(numpy_rgb).save(p, format="PPM")


def main():
    imgs = sorted(PHOTO.glob("*/*.jpg")) + sorted(PHOTO.glob("*.jpg"))
    results = []
    tmp_ppm = ROOT / "10_ocr_research" / "_probe_src.ppm"
    for img in imgs:
        entry = {"file": str(img.relative_to(ROOT))}
        # C 工具只吃 PPM：原始图先转 PPM 再喂所有后端（公平对比）
        from PIL import Image
        Image.open(img).convert("RGB").save(tmp_ppm, format="PPM")
        for name, tool in TOOLS.items():
            if not tool.exists():
                entry[name] = {"ok": False, "err": "tool missing"}
                continue
            entry[name] = run(tool, tmp_ppm)
        # 矫正 crop 后再跑（cnn/template/paddle 吃 crop；tesseract 自带 locate）
        rgb = cv2.cvtColor(cv2.imread(str(img)), cv2.COLOR_BGR2RGB)
        work, _ = load(rgb)
        st = detect_structure(_gray(work))
        rc = rectify(work, st)
        if rc is not None:
            p = ROOT / "10_ocr_research" / "_probe_crop.ppm"
            write_ppm(rc, p)
            entry["rect_cnn"] = run(TOOLS["cnn"], p)
            entry["rect_paddle"] = run(TOOLS["paddle"], p)
        else:
            entry["rect_cnn"] = {"ok": False, "err": "no mrz for rectify"}
            entry["rect_paddle"] = {"ok": False, "err": "no mrz for rectify"}
        results.append(entry)
        print(entry["file"])
        for k, v in entry.items():
            if k == "file":
                continue
            if isinstance(v, dict):
                print(f"  {k}: ok={v['ok']} dec?={bool(v.get('line1')) and bool(v.get('line2'))} "
                      f"l1={v.get('line1','')[:36]!r} l2={v.get('line2','')[:36]!r} "
                      f"conf=({v.get('conf1','')},{v.get('conf2','')}) ms={v.get('ms')} err={v.get('err','')[:30]!r}")
    json.dump(results, open(ROOT / "10_ocr_research" / "probe_results.json", "w"),
              ensure_ascii=False, indent=1)
    print("\nsaved 10_ocr_research/probe_results.json")
    # 简单汇总：每后端 decode 率（line1/line2 都非空且非全 <）
    from collections import Counter
    for backend in ["cnn", "template", "tesseract", "paddle", "rect_cnn", "rect_paddle"]:
        n_ok = sum(1 for e in results if e.get(backend, {}).get("ok"))
        n_both = sum(1 for e in results if (e.get(backend, {}).get("line1") or "").strip("<") and (e.get(backend, {}).get("line2") or "").strip("<"))
        print(f"{backend}: tool-ok={n_ok}/{len(results)}  both-lines={n_both}/{len(results)}")


if __name__ == "__main__":
    main()
