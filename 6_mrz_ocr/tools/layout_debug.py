#!/usr/bin/env python3
"""Debug script: run CNN tool with LAYOUT dump on specific images,
using the same preprocessing as eval_all_engines.py."""
import os, sys, subprocess, tempfile
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_photo_train import locate_band

try:
    _restore_src = ROOT / "passport_photo_restore" / "src"
    if str(_restore_src) not in sys.path:
        sys.path.insert(0, str(_restore_src))
    from restore import remove_guilloche as _remove_guilloche
except Exception:
    _remove_guilloche = None

PAD = 10
TOOL = ROOT / "6_mrz_ocr/build/mrz_ocr_cnn_tool_pic"
GT = ROOT / "tests/gt_assets_pic_official.json"

import json
gt_data = json.loads(GT.read_text())
gt = gt_data["gt"] if isinstance(gt_data, dict) and "gt" in gt_data else gt_data

targets = ["1c6acd092dee7031_0222.png", "acf14e0c631ec935_0030.png",
           "1792f7a2d79c7a8a_0017.jpg", "61d671ec161f22e3_0042.png"]

tmp = Path(tempfile.mkdtemp(prefix="layout_dbg_"))

for basename in targets:
    rec = gt.get(basename)
    if not rec:
        print(f"=== {basename}: NOT IN GT ===")
        continue
    l1 = rec.get("l1", "")
    l2 = rec.get("l2", "")
    label = rec.get("label", "")
    is_cscd = len(l1) == 30 and not l2

    img_path = ROOT / label
    if not img_path.exists():
        print(f"=== {basename}: IMAGE NOT FOUND {img_path} ===")
        continue

    rgb = Image.open(str(img_path)).convert("RGB")
    crop = locate_band(rgb)
    if crop is None:
        print(f"=== {basename}: BAND NOT FOUND ===")
        continue

    bx, by, bw, bh = crop
    band = rgb.crop((bx, max(0, by - PAD), bx + bw, min(rgb.height, by + bh + PAD)))

    if is_cscd and _remove_guilloche is not None:
        band_arr = np.array(band)
        band_arr = _remove_guilloche(band_arr)
        band = Image.fromarray(band_arr)

    ppm = tmp / f"{Path(basename).stem}.ppm"
    band.save(str(ppm), format="PPM")

    env = dict(os.environ)
    env["MRZ_OCR_LAYOUT"] = "1"
    env["MRZ_OCR_LINE_MODELS"] = "1"
    env["MRZ_OCR_PAR"] = "1"
    env["MRZ_OCR_CRNN"] = "0"

    p = subprocess.run([str(TOOL), str(ppm)], capture_output=True,
                       text=True, timeout=60, env=env)

    print(f"=== {basename} (is_cscd={is_cscd}) ===")
    print(f"  GT L1: {l1}")
    print(f"  GT L2: {l2}")
    print(f"  STDERR (LAYOUT dump):")
    for line in p.stderr.splitlines():
        if line.startswith("LAYOUT"):
            print(f"    {line}")
    # Parse stdout
    for line in p.stdout.splitlines():
        if "result.line1" in line or "result.line2" in line or "result.ok" in line:
            print(f"  {line.strip()}")
    print()
