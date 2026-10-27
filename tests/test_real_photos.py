#!/usr/bin/env python3
"""Regression test for the real-photo MRZ pipeline.

Locks the behaviour that the 2026-10 GUI rework established:

  1. A photo goes through structure detection -> reflection removal ->
     native band -> multi-engine OCR, and the verdict follows the REAL
     decoder (mrz_tool), not "the engine reported OK".
  2. mrz_tool exits 0 even when it REJECTS the lines (it prints
     'decode failed: ...' on stderr). A result with empty lines must
     therefore never be reported as verified.
  3. Known sample images must reproduce their expected MRZ, so a
     regression in detection / cropping / decoding is caught.

Run:
    python3 tests/test_real_photos.py            # representative subset
    python3 tests/test_real_photos.py --full     # every photo in assets/pic
    python3 tests/test_real_photos.py --list     # show the expectations
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

# Expectations: file -> (expected line1 substring, expected line2 substring).
# The substring form keeps the test robust to the tail filler while
# still pinning the identity fields.
# Truth read off the source photos (not from an OCR run).
EXPECTED = {
    # Czech specimen: P<CZESPECIMEN<<VZOR / 99009054<4CZE...
    "16c07a99a0fb81e4_0013.jpg": ("P<CZESPECIMEN<<VZOR", "99009054<4CZE"),
    # French specimen: PMCHEDE<MAIENFELD<<PIERRE / S2100023<6CHE...
    "08806d29a6268e1f_0003.jpg": ("MAIENFELD<<PIERRE", "CHE"),
    # Polish: P<POLKOWALSKI<<JAN / ...
    "1cce0f3fc387c893_0001.jpg": ("POLKOWALSKI", ""),
    # Ukrainian: P<UKRGRYTSENKO<<IVAN / ...
    "23f16a7d9b6ab9da_0062.jpg": ("UKRGRYTSENKO", ""),
    # Philippine: P<PHLDELAKCRUZ<<MARIA / ...
    "28ccd101a752acef_0016.jpg": ("P<PHLDELA<CRUZ<<MARIA", "P0000000A0PHL8003166F"),
    # Latvian: P<LVAOZOLKALNA<<LIENE<MARA / ...
    "2c793efe12276f9b_0258.jpg": ("LVAOZOLKALNA", ""),
    # Chinese: P<CHNZHENGJIAN<<YANGBEN / ...
    "1792f7a2d79c7a8a_0017.jpg": ("CHNZHENGJIAN", ""),
    # Chinese: P<SCHNYANG<<ZHA0 / ...
    "239e042aba0c3ab2_0102.jpg": ("SCHNYANG", ""),
}
# Photos whose MRZ the current detector cannot isolate: the US sample-book
# image is a two-page composite whose data page text interleaves with the
# MRZ, and the Taiwanese permits carry a single-line (TD1) MRZ.
KNOWN_HARD = ("0f9780b8e1e3cc21_0008.jpg", "1c6acd092dee7031_0222.png",
               "61d671ec161f22e3_0042.png", "acf14e0c631ec935_0030.png",
               "cdcc06ad360b7849_0019.png", "f385b0113d785dfd_0040.png",
               "31611582cbf14d7f_0018.jpg", "4dbd1b9e24b069f4_0023.png",
               "556644d1d8bf4672_0215.jpg", "6b91117492dc24f3_0022.jpg",
               "927cd546b5a8111d_0108.jpg", "e564d4372e28b9ed_0018.jpg")

MIN_VERIFIED_SUBSET = 6      # of the checked subset
MIN_VERIFIED_FULL = 50       # of the non-hard photos in assets/pic


def _gui():
    import importlib.util
    spec = importlib.util.spec_from_file_location("gui", ROOT / "tests" / "gui.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true",
                    help="run every photo in assets/pic (slow)")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args(argv)

    pic_dir = ROOT / "assets" / "pic"
    if a.list:
        for k, v in EXPECTED.items():
            print(f"{k}: {v}")
        return 0

    gui = _gui()
    import tkinter as tk
    app = gui.PassportGUI()
    app.update()
    try:
        app.rst_method_both.set(True)
        methods = app._rst_selected_methods()
        print(f"engines: {methods}")

        failures = []
        checked = 0
        verified = 0

        names = list(EXPECTED) if not a.full else sorted(
            p.name for p in pic_dir.iterdir()
            if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        if a.full:
            hard = [n for n in names if n in KNOWN_HARD]
            names = [n for n in names if n not in KNOWN_HARD]
            print(f"known-hard photos excluded from the floor: {len(hard)}")
        for name in names:
            path = pic_dir / name
            if not path.exists():
                failures.append(f"{name}: 样本缺失")
                continue
            checked += 1
            t0 = time.perf_counter()
            rec = app._ocr_photo_engines(str(path))
            ms = (time.perf_counter() - t0) * 1000.0
            ocr = rec.get("ocr") or {}
            l1 = ocr.get("line1") or ""
            l2 = ocr.get("line2") or ""
            if rec.get("verified"):
                verified += 1
            exp1, exp2 = EXPECTED.get(name, ("", ""))
            ok = True
            if exp1 and exp1 not in l1:
                ok = False
            if exp2 and exp2 not in l2:
                ok = False
            # 2. an empty result must never be 'verified'
            if rec.get("verified") and not (l1 and l2):
                ok = False
                failures.append(
                    f"{name}: 空结果被判为校验通过 (decode 判据失效)")
            if not ok:
                failures.append(
                    f"{name}: 期望 {exp1!r}/{exp2!r}, 实得 {l1[:44]!r}/{l2[:44]!r}")
            print(f"  {name[:34]:34s} {rec['grade']:8s} "
                  f"verified={rec.get('verified')!s:5s} {ms:6.0f} ms")

        floor = MIN_VERIFIED_FULL if a.full else MIN_VERIFIED_SUBSET
        print(f"\nchecked={checked} verified={verified} (floor {floor})")
        if verified < floor:
            failures.append(
                f"verified 数量 {verified} < 下限 {floor}（管线退化）")
        if failures:
            print("\n[FAIL]")
            for f in failures:
                print("  -", f)
            return 1
        print("\n[OK] real-photo pipeline regression passed")
        return 0
    finally:
        app.destroy()


if __name__ == "__main__":
    sys.exit(main())
