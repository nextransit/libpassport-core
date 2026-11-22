#!/usr/bin/env python3
"""Test EasyOCR supplement path — lightweight, no full GUI.

Directly calls the photo recognition pipeline by instantiating a
minimal stub object that provides only the attributes/methods that
_ocr_photo_engines and _ocr_photo_hypothesis need.
"""
import sys
import os
import time
import json
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))

os.environ["MRZ_FAST_STRICT"] = "0"

# We'll monkey-patch PassportGUI to skip UI construction and only
# keep the OCR-related methods.
import tkinter as tk

# Patch _build_* methods to no-ops BEFORE importing
import gui

for name in dir(gui.PassportGUI):
    if name.startswith("_build_") or name.startswith("_create_") or name.startswith("_setup_"):
        original = getattr(gui.PassportGUI, name, None)
        if original and callable(original):
            setattr(gui.PassportGUI, name, lambda self, *a, **kw: None)

# Also patch _face_render_preview and other UI helpers
for name in ["_face_render_preview", "_ocr_reset_results", "_ocr_setup_ui",
             "_rst_setup_ui", "_update_preview_image", "_fit_columns",
             "_load_preview", "_show_diff", "_load_ocrb_font"]:
    if hasattr(gui.PassportGUI, name):
        setattr(gui.PassportGUI, name, lambda self, *a, **kw: None)

root = tk.Tk()
root.withdraw()

try:
    app = gui.PassportGUI()
except Exception as e:
    print(f"GUI init failed: {e}")
    # Fall back to creating a bare object with the needed methods
    app = object.__new__(gui.PassportGUI)
    # Initialize essential attributes
    app._rst_resident = False
    app._ocr_result_cache = {}
    app._ocr_tk = None
    app._gt_map_cache = None

# GT data
GT_PHOTO_3_8 = {
    "3": ("POCHNJIA<<LIMIN<<<<<<<<<<<<<<<<<<<<<<<<<<<<<",
           "EE20426536CHN7410024M2808303LMNGMBKCMDPBA970"),
    "4": ("POCHNJIA<<LIMIN<<<<<<<<<<<<<<<<<<<<<<<<<<<<<",
           "EE20426536CHN7410024M2808303LMNGMBKCMDPBA970"),
    "5": ("POCHNJIA<<LIMIN<<<<<<<<<<<<<<<<<<<<<<<<<<<<<",
           "EE20426536CHN7410024M2808303LMNGMBKCMDPBA970"),
    "6": ("POCHNJIA<<LIMIN<<<<<<<<<<<<<<<<<<<<<<<<<<<<<",
           "EE20426536CHN7410024M2808303LMNGMBKCMDPBA970"),
    "7": ("POCHNJIA<<LIMIN<<<<<<<<<<<<<<<<<<<<<<<<<<<<<",
           "EE20426536CHN7410024M2808303LMNGMBKCMDPBA970"),
    "8": ("POCHNJIA<<LIMIN<<<<<<<<<<<<<<<<<<<<<<<<<<<<<",
           "EE20426536CHN7410024M2808303LMNGMBKCMDPBA970"),
}

gt_path = ROOT / "tests" / "gt_photo_passport_official.json"
GT_PASSPORT = {}
if gt_path.exists():
    with open(gt_path) as f:
        gt_data = json.load(f)
    gt_entries = gt_data.get("gt", gt_data)
    for name, entry in gt_entries.items():
        l1 = entry.get("l1", "")
        l2 = entry.get("l2", "")
        if name and l1 and l2:
            GT_PASSPORT[name] = (l1, l2)

def char_accuracy(pred, gt):
    if not gt:
        return 0.0
    n = min(len(pred), len(gt))
    correct = sum(1 for i in range(n) if pred[i] == gt[i])
    return correct / len(gt) * 100.0

all_results = []

# Test Photo/3-8
print(f"\n{'='*60}")
print(f"Testing Photo/3-8 (CS/CD scans)")
print(f"{'='*60}")

photo_base = ROOT / "Photo"
for i in range(3, 9):
    subdir = photo_base / str(i)
    if not subdir.is_dir():
        print(f"  Photo/{i}: directory not found")
        continue
    photo_files = (list(subdir.glob("*.jpg")) + list(subdir.glob("*.JPG"))
                   + list(subdir.glob("*.png")) + list(subdir.glob("*.PNG")))
    if not photo_files:
        print(f"  Photo/{i}: no image found")
        continue
    photo_path = photo_files[0]
    gt_l1, gt_l2 = GT_PHOTO_3_8[str(i)]

    t0 = time.perf_counter()
    try:
        rec = app._ocr_photo_engines(str(photo_path), methods=["cnn", "fuse"])
    except Exception as e:
        print(f"  Photo/{i}: ERROR: {e}")
        continue
    elapsed = (time.perf_counter() - t0) * 1000.0

    engines = rec.get("engines", [])
    easyocr_engines = [e for e in engines if e.get("method") == "easyocr"]
    easyocr_triggered = bool(easyocr_engines)

    best_l1 = rec.get("ocr", {}).get("line1", "")
    best_l2 = rec.get("ocr", {}).get("line2", "")
    grade = rec.get("grade", "")

    l1_acc = char_accuracy(best_l1, gt_l1)
    l2_acc = char_accuracy(best_l2, gt_l2)
    exact = (best_l1 == gt_l1 and best_l2 == gt_l2)

    print(f"\n  Photo/{i} ({photo_path.name}):")
    print(f"    grade={grade} exact={exact} l1_acc={l1_acc:.1f}% l2_acc={l2_acc:.1f}% ms={elapsed:.0f}")
    print(f"    best_l1={best_l1}")
    print(f"    gt_l1  ={gt_l1}")
    print(f"    best_l2={best_l2}")
    print(f"    gt_l2  ={gt_l2}")
    print(f"    eo_triggered={easyocr_triggered}")
    if easyocr_triggered:
        eo = easyocr_engines[0]
        print(f"    EasyOCR: l1={eo.get('line1','')}")
        print(f"             l2={eo.get('line2','')}")
        print(f"             decode_ok={eo.get('decode_ok')} ms={eo.get('ms',0):.0f}")

    all_results.append({
        "name": f"Photo/{i}",
        "exact": exact,
        "l1_acc": l1_acc,
        "l2_acc": l2_acc,
        "easyocr_triggered": easyocr_triggered,
        "ms": elapsed,
    })

# Test PassPortPhoto sample
print(f"\n{'='*60}")
print(f"Testing PassPortPhoto sample (5 photos)")
print(f"{'='*60}")

passport_dir = ROOT / "tmp" / "PassPortPhoto"
if passport_dir.is_dir():
    sample_names = list(GT_PASSPORT.keys())[:5]
    for name in sample_names:
        gt_l1, gt_l2 = GT_PASSPORT[name]
        photo_path = passport_dir / name
        if not photo_path.exists():
            continue

        t0 = time.perf_counter()
        try:
            rec = app._ocr_photo_engines(str(photo_path), methods=["cnn", "fuse"])
        except Exception as e:
            print(f"  {name}: ERROR: {e}")
            continue
        elapsed = (time.perf_counter() - t0) * 1000.0

        engines = rec.get("engines", [])
        easyocr_engines = [e for e in engines if e.get("method") == "easyocr"]
        easyocr_triggered = bool(easyocr_engines)

        best_l1 = rec.get("ocr", {}).get("line1", "")
        best_l2 = rec.get("ocr", {}).get("line2", "")
        grade = rec.get("grade", "")

        l1_acc = char_accuracy(best_l1, gt_l1)
        l2_acc = char_accuracy(best_l2, gt_l2)
        exact = (best_l1 == gt_l1 and best_l2 == gt_l2)

        print(f"\n  {name}: grade={grade} exact={exact} "
              f"l1_acc={l1_acc:.1f}% l2_acc={l2_acc:.1f}% "
              f"ms={elapsed:.0f} eo_triggered={easyocr_triggered}")

        all_results.append({
            "name": name,
            "exact": exact,
            "l1_acc": l1_acc,
            "l2_acc": l2_acc,
            "easyocr_triggered": easyocr_triggered,
            "ms": elapsed,
        })

# Final summary
print(f"\n{'='*60}")
print(f"FINAL SUMMARY")
print(f"{'='*60}")
n = len(all_results)
n_exact = sum(1 for r in all_results if r["exact"])
n_eo = sum(1 for r in all_results if r["easyocr_triggered"])
avg_ms = sum(r["ms"] for r in all_results) / n if n else 0
print(f"Total: {n_exact}/{n} exact, {n_eo}/{n} EasyOCR triggered, avg={avg_ms:.0f}ms")

try:
    root.destroy()
except Exception:
    pass
