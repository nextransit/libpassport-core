#!/usr/bin/env python3
"""截获产品 band 并以 crnn / fuse 模式喂 C 工具，诊断 7 张难照。

复刻 _ocr_photo_engines -> _ocr_photo_hypothesis(A假设) 的 band 提取:
load(1250) -> detect_structure -> drop_reflection -> pair/single ->
_rst_native_mrz_band -> remove_guilloche(单行) -> _ocr_pick_band。
"""
import os
import subprocess
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "passport_photo_restore" / "src"))
import gui  # noqa: E402

gui.PassportGUI.AUTO_WARM_ENGINES = False
app = gui.PassportGUI()
app.update()
app.withdraw()

from restore import (load, _gray, detect_structure,  # noqa: E402
                     drop_reflection)

PHOTOS = [
    "tmp/PassPortPhoto/2026-9-29-U.jpg",
    "tmp/PassPortPhoto/2026-9-30-AG.jpg",
    "tmp/PassPortPhoto/2026-9-30-H.jpg",
    "tmp/PassPortPhoto/2026-9-30-I.jpg",
    "tmp/PassPortPhoto/2026-9-30-J.jpg",
    "tmp/PassPortPhoto/2026-9-30-K.jpg",
    "Photo/3/IMG_1792133000_003.jpg",
    "Photo/8/IMG_1792133682_008.jpg",
    # 对照组：正常 PASS 照
    "tmp/PassPortPhoto/2026-9-29-A.jpg",
]


def band_for(path):
    rgb = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
    work, scale = load(rgb, max_side=1250)
    g = _gray(work)
    st = detect_structure(g)
    work2, st2, refl = drop_reflection(work, st)
    src_bottom = None
    try:
        cb = refl.get("crop_bottom") if isinstance(refl, dict) else None
        if cb:
            src_bottom = int(round(int(cb) / max(1e-6, scale)))
    except Exception:
        src_bottom = None
    pair = st2.get("mrz_pair")
    is_single = False
    if not pair:
        single = st2.get("mrz_single")
        if single and (single[1] - single[0]) >= 20:
            pair = (single, single)
            is_single = True
    if pair is None:
        return rgb, scale, None, True, None, "no pair/single"
    native_band, native_crop = app._rst_native_mrz_band(
        rgb, pair, scale, source_bottom=src_bottom, single=is_single)
    if is_single:
        try:
            from restore import remove_guilloche
            native_band = remove_guilloche(native_band)
        except Exception:
            pass
    native_band, native_crop = app._ocr_pick_band(
        rgb, native_band, native_crop, pair, scale, work2,
        is_single, None, src_bottom)
    return rgb, scale, native_band, is_single, pair, None


def run_tool(band_path, mode):
    env = dict(os.environ)
    env["MRZ_OCR_LINE_MODELS"] = "1"
    env["MRZ_OCR_CRNN"] = "1" if mode == "crnn" else "0"
    env.pop("MRZ_OCR_FUSE", None)
    if mode == "fuse":
        env["MRZ_OCR_FUSE"] = "1"
    if os.environ.get("DIAG_TOPK"):
        env["MRZ_OCR_TOPK"] = os.environ["DIAG_TOPK"]
        env["MRZ_OCR_CAND_K"] = os.environ.get("DIAG_CAND_K", "8")
    tool = app._ocr_weights_tool()
    proc = subprocess.run([str(tool), str(band_path)], capture_output=True,
                          text=True, timeout=15, env=env)
    return proc


def kv(stdout):
    out = {}
    for line in stdout.splitlines():
        if ":" in line:
            k, _, v = line.partition(":")
            out[k.strip()] = v.strip()
    return out


def main():
    outdir = ROOT / "10_ocr_research"
    for rel in PHOTOS:
        path = ROOT / rel
        rgb, scale, band, is_single, pair, err = band_for(path)
        name = path.stem.replace("IMG_1792133000_003", "P3").replace(
            "IMG_1792133682_008", "P8")
        print("=" * 92)
        print(f"{rel}  single={is_single} pair={pair} err={err}")
        if band is None:
            print("  !! no band"); continue
        print(f"  band shape: {band.shape}")
        band_path = outdir / f"_diag1120_{name}.ppm"
        try:
            from PIL import Image
            Image.fromarray(band).save(str(band_path), format="PPM")
        except Exception as e:
            print("  !! save fail", e); continue
        for mode in ("crnn", "fuse"):
            p = run_tool(band_path, mode)
            d = kv(p.stdout)
            print(f"  [{mode:4s}] head={d.get('result.head','?'):10s} "
                  f"ok={d.get('result.ok','?'):5s} "
                  f"conf={d.get('result.conf1','?')}/{d.get('result.conf2','?')} "
                  f"ms={d.get('result.ms','?')}")
            print(f"         l1: {d.get('result.line1','')}")
            print(f"         l2: {d.get('result.line2','')}")
            if p.returncode != 0:
                print(f"         rc={p.returncode} stderr={p.stderr[:200]}")


if __name__ == "__main__":
    main()
