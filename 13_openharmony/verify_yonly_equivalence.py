#!/usr/bin/env python3
"""
verify_yonly_equivalence.py - Y-only 路径等价性验证 + 详细 OCR 报告

对每张 PassPortPhoto:
  路径A (原):   RGB → locate_band → band PPM → mrz_ocr_cnn_tool → MRZ_A
  路径B (Y-only): RGB → BT.601 Y → 灰度RGB(Y,Y,Y) → locate_band → band PPM → tool → MRZ_B

输出: 每张图详情 (GT/A/B/conf/一致性/准确率) + 汇总 + JSON 报告
"""
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "6_mrz_ocr" / "tools"))
from export_photo_train import locate_band  # noqa: E402
from eval_weight_sets import run  # noqa: E402

GT = ROOT / "tests/gt_photo_passport_official.json"
TOOL = ROOT / "6_mrz_ocr" / "build" / "mrz_ocr_cnn_tool"
PAD = 10


def to_gray_rgb(rgb_img):
    arr = np.array(rgb_img)
    r, g, b = arr[:, :, 0].astype(float), arr[:, :, 1].astype(float), arr[:, :, 2].astype(float)
    y = (0.299 * r + 0.587 * g + 0.114 * b).clip(0, 255).astype(np.uint8)
    return Image.fromarray(np.stack([y, y, y], axis=2), "RGB")


def run_path(img, tmp, tag):
    crop = locate_band(img)
    if crop is None:
        return None
    bx, by, bw, bh = crop
    band = img.crop((bx, max(0, by - PAD), bx + bw, min(img.height, by + bh + PAD)))
    ppm = tmp / (tag + ".ppm")
    band.save(str(ppm), format="PPM")
    return run(TOOL, ppm)


def char_acc(pred, gt):
    if len(gt) != 44:
        return None, 0, 0
    tot = 44
    if len(pred) != 44:
        return False, 0, tot
    ok = sum(a == b for a, b in zip(pred, gt))
    return ok == tot, ok, tot


def main():
    d = json.loads(GT.read_text())
    gt = d["gt"] if isinstance(d, dict) and "gt" in d else d
    tmp = Path(tempfile.mkdtemp(prefix="yonly_eq_"))

    rows = []
    print("=" * 100)
    print(f"{'photo':<22} {'A==B':>4} {'A==GT':>5} {'B==GT':>5}  GT_l1 / A_l1 / B_l1")
    print("-" * 100)

    for name, rec in sorted(gt.items()):
        label = rec.get("label") or ""
        img_path = ROOT / label
        if not img_path.exists():
            continue
        rgb = Image.open(img_path).convert("RGB")
        gray_rgb = to_gray_rgb(rgb)

        rA = run_path(rgb, tmp, name + "_A")
        rB = run_path(gray_rgb, tmp, name + "_B")
        l1_gt = rec.get("l1") or ""
        l2_gt = rec.get("l2") or ""

        if rA is None or rB is None:
            rows.append({"id": name, "located": False})
            print(f"{name:<22}  BAND-FAIL")
            continue

        a_eq_b = (rA["line1"] == rB["line1"] and rA["line2"] == rB["line2"])
        l1a_ok, l1a_n, l1a_t = char_acc(rA["line1"], l1_gt)
        l2a_ok, l2a_n, l2a_t = char_acc(rA["line2"], l2_gt)
        l1b_ok, l1b_n, l1b_t = char_acc(rB["line1"], l1_gt)
        l2b_ok, l2b_n, l2b_t = char_acc(rB["line2"], l2_gt)

        rows.append({
            "id": name, "located": True,
            "gt1": l1_gt, "gt2": l2_gt,
            "A": {"line1": rA["line1"], "line2": rA["line2"], "conf1": rA["conf1"], "conf2": rA["conf2"], "layout": rA["layout"]},
            "B": {"line1": rB["line1"], "line2": rB["line2"], "conf1": rB["conf1"], "conf2": rB["conf2"], "layout": rB["layout"]},
            "a_eq_b": a_eq_b,
            "a_l1_char": l1a_n, "a_l2_char": l2a_n,
            "b_l1_char": l1b_n, "b_l2_char": l2b_n,
        })

        flag = "Y" if a_eq_b else "N"
        a_gt = "Y" if (l1a_ok and l2a_ok) else "N"
        b_gt = "Y" if (l1b_ok and l2b_ok) else "N"
        print(f"{name:<22} {flag:>4} {a_gt:>5} {b_gt:>5}  "
              f"GT:{l1_gt[:20]}  A:{rA['line1'][:20]}  B:{rB['line1'][:20]}")

    located = [r for r in rows if r.get("located")]
    match = sum(1 for r in located if r["a_eq_b"])
    mismatch = len(located) - match

    def acc_sum(key):
        l1 = sum(r.get(key, 0) for r in located)
        return l1

    a_l1 = acc_sum("a_l1_char"); a_l2 = acc_sum("a_l2_char")
    b_l1 = acc_sum("b_l1_char"); b_l2 = acc_sum("b_l2_char")
    tot_l1 = sum(44 for r in located if len(r.get("gt1", "")) == 44)
    tot_l2 = sum(44 for r in located if len(r.get("gt2", "")) == 44)
    a_both = sum(1 for r in located if r["a_l1_char"] == 44 and r["a_l2_char"] == 44
                 and len(r.get("gt1", "")) == 44 and len(r.get("gt2", "")) == 44)
    b_both = sum(1 for r in located if r["b_l1_char"] == 44 and r["b_l2_char"] == 44
                 and len(r.get("gt1", "")) == 44 and len(r.get("gt2", "")) == 44)

    print("=" * 100)
    print(f"\n=== Equivalence (A vs B) ===")
    print(f"  match={match} mismatch={mismatch} (of {len(located)} located)")
    print(f"\n=== Accuracy vs GT ===")
    print(f"  Path A (original RGB):")
    print(f"    line1 char {a_l1}/{tot_l1} = {a_l1/tot_l1 if tot_l1 else 0:.4f}  "
          f"line2 char {a_l2}/{tot_l2} = {a_l2/tot_l2 if tot_l2 else 0:.4f}  "
          f"both-exact {a_both}/{len(located)}")
    print(f"  Path B (Y-only gray RGB):")
    print(f"    line1 char {b_l1}/{tot_l1} = {b_l1/tot_l1 if tot_l1 else 0:.4f}  "
          f"line2 char {b_l2}/{tot_l2} = {b_l2/tot_l2 if tot_l2 else 0:.4f}  "
          f"both-exact {b_both}/{len(located)}")
    print(f"  tool={TOOL.name}")

    out_json = ROOT / "13_openharmony" / "yonly_report.json"
    out_json.write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    print(f"\n  detailed JSON: {out_json}")
    return 1 if mismatch else 0


if __name__ == "__main__":
    sys.exit(main())
