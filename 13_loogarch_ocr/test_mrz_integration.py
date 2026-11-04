#!/usr/bin/env python3
"""
test_mrz_integration.py - 验证 loogarch_ocr MRZ 集成正确性

对每张 PassPortPhoto:
  路径A (CLI 参照):  RGB → locate_band → mrz_ocr_cnn_tool → MRZ_A
  路径B (Y-only 库): RGB → loogarch_ocr(gray=1, RGB888) → MRZ_B
  路径C (Color 库):  RGB → loogarch_ocr(RGB888, 无 gray) → MRZ_C

验证 B==A (Y-only 等价) 且 C==A (Color 等价)
"""
import ctypes
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "6_mrz_ocr" / "tools"))
from export_photo_train import locate_band
from eval_weight_sets import run

GT = ROOT / "tests/gt_photo_passport_official.json"
TOOL = ROOT / "6_mrz_ocr" / "build" / "mrz_ocr_cnn_tool"
LIB = ROOT / "13_loogarch_ocr" / "libloogarch_ocr_mrz.so"
LIB_OPT = ROOT / "13_loogarch_ocr" / "libloogarch_ocr_mrz_optimized.so"
PAD = 10

RGB888 = 6  # LOOGARCH_PIXEL_FORMAT_RGB888 (NV21=0,NV12=1,I420=2,I422=3,I444=4,ARGB=5,RGB888=6)


class LoogarchOcrResult(ctypes.Structure):
    _fields_ = [
        ("text", ctypes.c_void_p),
        ("confidence", ctypes.c_int),
        ("width", ctypes.c_int),
        ("height", ctypes.c_int),
        ("format", ctypes.c_int),
        ("line1", ctypes.c_char * 64),
        ("line2", ctypes.c_char * 64),
        ("conf1", ctypes.c_int),
        ("conf2", ctypes.c_int),
        ("band_x", ctypes.c_int),
        ("band_y", ctypes.c_int),
        ("band_w", ctypes.c_int),
        ("band_h", ctypes.c_int),
        ("layout", ctypes.c_int),
        ("status", ctypes.c_int),
    ]


def load_lib(path):
    lib = ctypes.CDLL(str(path))
    lib.loogarch_ocr_init.restype = ctypes.c_void_p
    lib.loogarch_ocr_init.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_char_p]
    lib.loogarch_ocr_process.restype = LoogarchOcrResult
    lib.loogarch_ocr_process.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint8),
                                         ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    lib.loogarch_ocr_cleanup.restype = None
    lib.loogarch_ocr_cleanup.argtypes = [ctypes.c_void_p]
    lib.loogarch_ocr_get_version.restype = ctypes.c_char_p
    return lib


def run_lib_path(lib, rgb_img, gray_only):
    crop = locate_band(rgb_img)
    if crop is None:
        return None
    bx, by, bw, bh = crop
    band = rgb_img.crop((bx, max(0, by - PAD), bx + bw, min(rgb_img.height, by + bh + PAD)))
    w, h = band.size
    arr = np.ascontiguousarray(np.array(band, dtype=np.uint8))
    buf = arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
    opts = b"gray=1" if gray_only else b""
    handle = lib.loogarch_ocr_init(RGB888, w, h, opts)
    if not handle:
        return None
    res = lib.loogarch_ocr_process(handle, buf, 0, 0, 0, 0)
    lib.loogarch_ocr_cleanup(handle)
    return {
        "line1": res.line1.decode("ascii", errors="replace"),
        "line2": res.line2.decode("ascii", errors="replace"),
        "conf1": res.conf1,
        "conf2": res.conf2,
        "layout": res.layout,
        "status": res.status,
    }


def run_cli_path(img, tmp, tag):
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
    os.environ['MRZ_OCR_LINE_MODELS'] = '1'
    os.environ['MRZ_OCR_PAR'] = '1'
    os.environ['MRZ_OCR_CRNN'] = '0'

    d = json.loads(GT.read_text())
    gt = d["gt"] if isinstance(d, dict) and "gt" in d else d
    tmp = Path(tempfile.mkdtemp(prefix="mrz_integ_"))

    lib = load_lib(LIB)
    lib_opt = load_lib(LIB_OPT)
    print(f"lib version: {lib.loogarch_ocr_get_version().decode()}")
    print(f"lib_opt version: {lib_opt.loogarch_ocr_get_version().decode()}")

    rows = []
    print("=" * 120)
    print(f"{'photo':<22} {'A==B':>4} {'A==C':>4} {'B==C':>4} {'A==GT':>5} {'B==GT':>5}  GT_l1 / A_l1 / B_l1(y) / C_l1(rgb)")
    print("-" * 120)

    for name, rec in sorted(gt.items()):
        label = rec.get("label") or ""
        img_path = ROOT / label
        if not img_path.exists():
            continue
        rgb = Image.open(img_path).convert("RGB")

        try:
            rA = run_cli_path(rgb, tmp, name + "_A")
            rB = run_lib_path(lib, rgb, True)
            rC = run_lib_path(lib, rgb, False)
        except Exception as e:
            print(f"{name:<22}  EXCEPTION: {e}")
            rows.append({"id": name, "ok": False})
            continue
        l1_gt = rec.get("l1") or ""
        l2_gt = rec.get("l2") or ""

        if rA is None or rB is None or rC is None:
            rows.append({"id": name, "ok": False})
            print(f"{name:<22}  FAIL (A={rA is not None} B={rB is not None} C={rC is not None})")
            continue

        a_eq_b = (rA["line1"] == rB["line1"] and rA["line2"] == rB["line2"])
        a_eq_c = (rA["line1"] == rC["line1"] and rA["line2"] == rC["line2"])
        b_eq_c = (rB["line1"] == rC["line1"] and rB["line2"] == rC["line2"])

        l1a_ok, l1a_n, _ = char_acc(rA["line1"], l1_gt)
        l2a_ok, l2a_n, _ = char_acc(rA["line2"], l2_gt)
        l1b_ok, l1b_n, _ = char_acc(rB["line1"], l1_gt)
        l2b_ok, l2b_n, _ = char_acc(rB["line2"], l2_gt)

        rows.append({
            "id": name, "ok": True,
            "gt1": l1_gt, "gt2": l2_gt,
            "A": rA, "B": rB, "C": rC,
            "a_eq_b": a_eq_b, "a_eq_c": a_eq_c, "b_eq_c": b_eq_c,
            "a_l1": l1a_n, "a_l2": l2a_n,
            "b_l1": l1b_n, "b_l2": l2b_n,
        })

        f_ab = "Y" if a_eq_b else "N"
        f_ac = "Y" if a_eq_c else "N"
        f_bc = "Y" if b_eq_c else "N"
        f_agt = "Y" if (l1a_ok and l2a_ok) else "N"
        f_bgt = "Y" if (l1b_ok and l2b_ok) else "N"
        print(f"{name:<22} {f_ab:>4} {f_ac:>4} {f_bc:>4} {f_agt:>5} {f_bgt:>5}  "
              f"GT:{l1_gt[:18]}  A:{rA['line1'][:18]}  B:{rB['line1'][:18]}  C:{rC['line1'][:18]}")

    ok_rows = [r for r in rows if r.get("ok")]
    n = len(ok_rows)
    match_ab = sum(1 for r in ok_rows if r["a_eq_b"])
    match_ac = sum(1 for r in ok_rows if r["a_eq_c"])
    match_bc = sum(1 for r in ok_rows if r["b_eq_c"])

    a_l1 = sum(r["a_l1"] for r in ok_rows)
    a_l2 = sum(r["a_l2"] for r in ok_rows)
    b_l1 = sum(r["b_l1"] for r in ok_rows)
    b_l2 = sum(r["b_l2"] for r in ok_rows)
    tot_l1 = sum(44 for r in ok_rows if len(r.get("gt1", "")) == 44)
    tot_l2 = sum(44 for r in ok_rows if len(r.get("gt2", "")) == 44)
    a_both = sum(1 for r in ok_rows if r["a_l1"] == 44 and r["a_l2"] == 44
                 and len(r.get("gt1", "")) == 44 and len(r.get("gt2", "")) == 44)
    b_both = sum(1 for r in ok_rows if r["b_l1"] == 44 and r["b_l2"] == 44
                 and len(r.get("gt1", "")) == 44 and len(r.get("gt2", "")) == 44)

    print("=" * 120)
    print(f"\n=== Equivalence ===")
    print(f"  A (CLI) vs B (lib Y-only):  {match_ab}/{n}")
    print(f"  A (CLI) vs C (lib Color):   {match_ac}/{n}")
    print(f"  B (lib Y-only) vs C (lib Color): {match_bc}/{n}")

    print(f"\n=== Accuracy vs GT ===")
    print(f"  A (CLI reference):")
    print(f"    line1 {a_l1}/{tot_l1} = {a_l1/tot_l1:.4f}  line2 {a_l2}/{tot_l2} = {a_l2/tot_l2:.4f}  both-exact {a_both}/{n}")
    print(f"  B (lib Y-only):")
    print(f"    line1 {b_l1}/{tot_l1} = {b_l1/tot_l1:.4f}  line2 {b_l2}/{tot_l2} = {b_l2/tot_l2:.4f}  both-exact {b_both}/{n}")

    out_json = ROOT / "13_loogarch_ocr" / "mrz_integration_report.json"
    out_json.write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    print(f"\n  detailed JSON: {out_json}")

    all_match = (match_ab == n and match_ac == n)
    return 0 if all_match else 1


if __name__ == "__main__":
    sys.exit(main())
