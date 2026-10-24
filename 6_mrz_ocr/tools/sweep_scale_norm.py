#!/usr/bin/env python3
"""Is the decode failure on small documents a resolution problem?

Motivation
----------
assets/pic has a median of 19.4 px per MRZ character column (band width /
44), while the machine-position photos sit at exactly 32.7 (1440 / 44).
Only 12 of 84 assets images produce a line 2 whose check digits verify,
against 17 of 18 on PassPortPhoto.

The question this script answers: is that gap caused by the glyphs simply
being too small, and if so, what is the right way to express the rule?

Three things are measured:

  1. verify rate as a function of a fixed vertical scale factor
  2. verify rate as a function of a fixed *line height*
  3. verify rate as a function of a fixed *px per character column*

(3) wins: px/char is the quantity that actually predicts success, because
it is the horizontal sampling density the CNN glyph input sees.  Line
height is the wrong knob -- PassPortPhoto decodes fine at 46 px line
height while assets fail at the same 45.5 px, because passport has 32.7
px/char and assets has 19.4.

Finally the deployable rule is tested: upscale only, target 26 px/char,
never downscale.  On PassPortPhoto every band is already above the target
so the rule is a no-op there, which is exactly why it is safe.

Usage:
    python3 tools/sweep_scale_norm.py
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

import cv2
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "build" / "mrz_ocr_cnn_tool"
ASSETS = ROOT.parent / "assets" / "pic"
CORPUS = ROOT / "data" / "corpus_passport_photo"

WEIGHTS = (7, 3, 1)


def char_value(c: str) -> int:
    if c == "<":
        return 0
    if "0" <= c <= "9":
        return ord(c) - 48
    return ord(c) - 55


def check_digit(seg: str) -> int:
    return sum(char_value(c) * WEIGHTS[i % 3] for i, c in enumerate(seg)) % 10


def verifies(line2: str) -> bool:
    """All five TD3 line-2 check digits verify.

    Necessary, not sufficient -- see tools/checksum_nullspace.py.  It is
    still the only label-free signal available on an unlabelled set.
    """
    if len(line2) != 44:
        return False
    try:
        for pos, (a, b) in ((9, (0, 9)), (19, (13, 19)),
                            (27, (21, 27)), (42, (28, 42))):
            if line2[pos] != str(check_digit(line2[a:b])):
                return False
        return line2[43] == str(
            check_digit(line2[0:10] + line2[13:20] + line2[21:28] + line2[28:43]))
    except (ValueError, IndexError):
        return False


def decode(ppm: Path, crnn: int) -> str:
    env = dict(os.environ)
    env.update(MRZ_OCR_LINE_MODELS="1", MRZ_OCR_PAR="0",
               MRZ_OCR_CRNN=str(crnn))
    p = subprocess.run([str(TOOL), str(ppm)], capture_output=True,
                       text=True, timeout=120, env=env)
    kv = {}
    for ln in p.stdout.splitlines():
        if ":" in ln:
            k, _, v = ln.partition(":")
            kv[k.strip()] = v.strip()
    return kv.get("result.line2", "")


def collect():
    """(label, band, group) for both sets."""
    out = []
    cat = json.load(open(ASSETS / "catalog.json"))
    for rec in cat.get("records", []):
        loc = (rec.get("loc") or {}).get("mrz")
        path = ASSETS / rec["file"]
        if not loc or not path.exists():
            continue
        rgb = cv2.imread(str(path))
        if rgb is None:
            continue
        h, w = rgb.shape[:2]
        pad = 12
        band = rgb[max(0, loc["y"] - pad):min(h, loc["y"] + loc["h"] + pad),
                   max(0, loc["x"]):min(w, loc["x"] + loc["w"])]
        if band.shape[0] >= 16 and band.shape[1] >= 44:
            out.append((path.stem, band, "assets"))
    for rec in json.load(open(CORPUS / "corpus.json"))["records"]:
        band = cv2.imread(str(CORPUS / rec["file"]))
        if band is not None:
            out.append((Path(rec["file"]).stem, band, "passport"))
    return out


def scaled(band, factor):
    if abs(factor - 1.0) < 1e-9:
        return band
    return cv2.resize(band, None, fx=factor, fy=factor,
                      interpolation=cv2.INTER_CUBIC)


def measure(items, name, factor_fn, tmp):
    """Verify count per group under a given scaling rule."""
    tally = {}
    for label, band, group in items:
        factor = factor_fn(band)
        img = scaled(band, factor)
        if img.shape[0] < 16 or img.shape[1] < 44:
            continue
        ppm = tmp / f"{name}_{label}.ppm"
        Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).save(
            str(ppm), format="PPM")
        hit = verifies(decode(ppm, 0))
        n, v = tally.get(group, (0, 0))
        tally[group] = (n + 1, v + hit)
    return tally


def main() -> int:
    items = collect()
    tmp = Path(tempfile.mkdtemp(prefix="scale_"))
    groups = sorted({g for _, _, g in items})
    print(f"n={len(items)}  " + "  ".join(
        f"{g}={sum(1 for _, _, gg in items if gg == g)}" for g in groups))
    print()

    def row(title, tally):
        print(f"{title:>18}  " + "  ".join(
            f"{g}: {tally[g][1]}/{tally[g][0]}" for g in groups
            if g in tally))

    print("--- 固定竖直倍率（行高不是正确的旋钮）---")
    for f in (1.0, 1.5, 2.0, 2.5, 3.0):
        row(f"x{f}", measure(items, f"f{f}", lambda b, f=f: f, tmp))

    print("\n--- 固定行高 ---")
    for target in (60, 80, 100, 120):
        row(f"line_h {target}px",
            measure(items, f"h{target}",
                    lambda b, t=target: t / (b.shape[0] / 2.0), tmp))

    print("\n--- 固定每字符宽度（正确旋钮）---")
    for target in (20, 26, 32, 40):
        row(f"{target} px/char",
            measure(items, f"c{target}",
                    lambda b, t=target: t / (b.shape[1] / 44.0), tmp))

    print("\n--- 部署规则：只放大不缩小，目标 26 px/char ---")
    print(f"{'':18}  " + "  ".join(groups))
    for crnn, tag in ((0, "cnn"), (1, "crnn")):
        cells = []
        for g in groups:
            sub = [(l, b) for l, b, gg in items if gg == g]
            raw = norm = 0
            for label, band in sub:
                ppm = tmp / f"raw_{label}.ppm"
                Image.fromarray(cv2.cvtColor(band, cv2.COLOR_BGR2RGB)).save(
                    str(ppm), format="PPM")
                raw += verifies(decode(ppm, crnn))
                factor = max(1.0, 26.0 / (band.shape[1] / 44.0))
                img = scaled(band, factor)
                ppm2 = tmp / f"norm_{label}.ppm"
                Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).save(
                    str(ppm2), format="PPM")
                norm += verifies(decode(ppm2, crnn))
            cells.append(f"{raw} -> {norm}/{len(sub)}")
        print(f"{tag:>18}  " + "  ".join(cells))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
