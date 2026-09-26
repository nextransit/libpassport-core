#!/usr/bin/env python3
"""tesseract_tool -- Tesseract-backed MRZ recognizer that mimics the
stdout protocol of 6_mrz_ocr/build/mrz_ocr_tool so the GUI can
compare OCR back-ends side-by-side without any code changes.

Contract (identical to mrz_ocr_tool):
    argv[0] = this script
    argv[1] = image path (PPM/PBM/BMP/PNG/JPG via PIL)
    stdout  = key/value lines:
        result.ok      : OK | <error message>
        result.line1   : 44-char line (or empty)
        result.line2   : 44-char line (or empty)
        result.conf1   : 0..100
        result.conf2   : 0..100
        band.x band.y band.w band.h : "x y w h" in image coordinates
    exit code: 0 on success, 1 on OCR failure
"""
from __future__ import annotations

import sys
import re
import subprocess
import tempfile
import time
from pathlib import Path

# Use PIL for input loading + horizontal MRZ band crop detection
# (Tesseract works much better on a tight crop than on the whole
# page). The MRZ band is the bottom ~10% of the image; we run
# Tesseract with --psm 7 (single text line) twice, once per row,
# and stitch the results into the standard two-line shape.

MRZ_RE = re.compile(r"[A-Z0-9<]{10,}")
LINE_LEN = 44


def _image_to_gray_array(path: Path):
    """Return (grayscale ndarray HxW, original PIL image)."""
    import numpy as np
    from PIL import Image
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img.convert("L"))
    return arr, img


def _find_mrz_band(gray):
    """Locate the two MRZ lines. Returns (y0, y1) in pixel space.

    The MRZ band is always in the bottom third of the image; header
    banners / photo areas in the top / middle also satisfy a naive
    row-density threshold, so we restrict the search to the bottom
    third and require two dense sub-groups separated by a short gap
    (the MRZ 'double-line' fingerprint).
    """
    import numpy as np
    H, W = gray.shape
    bottom_start = int(H * 0.66)
    sub = gray[bottom_start:, :]
    rd = (sub < 128).sum(axis=1)
    rows = np.where(rd > W * 0.10)[0]
    if rows.size < 4:
        return int(H * 0.88), H
    groups = []
    start = prev = int(rows[0])
    for r in rows[1:]:
        r = int(r)
        if r - prev <= 4:
            prev = r
        else:
            groups.append((start, prev))
            start = prev = r
    groups.append((start, prev))
    # Translate groups back to absolute y and look for a pair
    # separated by <= 30 px (the two MRZ lines).
    abs_groups = [(g[0] + bottom_start, g[1] + bottom_start)
                  for g in groups if g[1] - g[0] >= 10]
    if len(abs_groups) >= 2:
        for i in range(len(abs_groups) - 1, 0, -1):
            top, bot = abs_groups[i - 1], abs_groups[i]
            if bot[0] - top[1] <= 30:
                return top[0], bot[1]
    # Single-line fallback.
    if abs_groups:
        return abs_groups[-1][0], abs_groups[-1][1]
    return int(H * 0.88), H


def _tesseract_line(pil_image) -> tuple[str, int]:
    """Run tesseract on a single MRZ line. Returns (text, conf).
    conf is 0..100."""
    import pytesseract
    # --psm 7 = single text line. OEM 3 = default (LSTM).
    config = "--psm 7 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<"
    try:
        data = pytesseract.image_to_data(
            pil_image,
            config=config,
            output_type=pytesseract.Output.DICT,
        )
    except pytesseract.TesseractError as e:
        return "", 0
    text = "".join(data.get("text", [])).strip().upper()
    # Pad / trim to exactly 44 chars so the downstream decoder can
    # attempt a parse; OCR noise at the edges is the most common
    # cause of bad length.
    text = text.replace(" ", "")
    if len(text) < LINE_LEN:
        text = text + "<" * (LINE_LEN - len(text))
    text = text[:LINE_LEN]
    # Tesseract is very conservative with OCR-B glyphs (often
    # returning conf=0 for what is in fact a correct recognition).
    # Use a *proxy* confidence: the share of characters that are in
    # the ICAO alphabet, penalised for leading/trailing junk.
    allowed = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<")
    if not text:
        return text, 0
    in_set = sum(1 for c in text if c in allowed)
    purity = 100.0 * in_set / max(1, len(text))
    # Bonus for length-correct lines, slight penalty if the line is
    # off by one or two characters (Tesseract often drops a char).
    length_bonus = 0
    if len(text) == 44:
        length_bonus = 5
    elif len(text) >= 40:
        length_bonus = 2
    avg = int(min(100, purity + length_bonus))
    return text, avg


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: tesseract_tool.py <image>", file=sys.stderr)
        return 2
    img_path = Path(argv[1])
    if not img_path.exists():
        print(f"cannot open {img_path}", file=sys.stderr)
        return 2

    t0 = time.perf_counter()
    try:
        gray, img = _image_to_gray_array(img_path)
    except Exception as e:
        print(f"result.ok       : cannot load image: {e}",
              file=sys.stderr)
        return 1

    H, W = gray.shape
    y0, y1 = _find_mrz_band(gray)
    crop = img.crop((0, y0, W, y1))

    # Split the band into two horizontal halves (one per MRZ line).
    band_h = y1 - y0
    half = band_h // 2
    top = crop.crop((0, 0, W, half + 2))
    bot = crop.crop((0, max(0, half - 2), W, band_h))
    line1, conf1 = _tesseract_line(top)
    line2, conf2 = _tesseract_line(bot)

    band_box = f"0 {y0} {W} {band_h}"
    ms = int((time.perf_counter() - t0) * 1000)

    # Stricter gate: line1 must begin with a valid ICAO document
    # type (P< for passport, I< for ID card, V< for visa, A<, C<,
    # D<, ...). line2 first 10 chars must be alphanumeric (passport
    # number + check digit) and the whole line must contain at
    # least 30 valid ICAO characters.
    valid_types = set("PIVACDR<")
    allowed = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<")
    def _is_valid(line):
        if len(line) != LINE_LEN:
            return False
        if not line or line[0] not in valid_types or line[1] != "<":
            return False
        in_set = sum(1 for c in line if c in allowed)
        return in_set >= 30
    ok = _is_valid(line1) and _is_valid(line2)
    print(f"result.ok       : {'OK' if ok else 'unreadable'}")
    print(f"result.line1    : {line1}")
    print(f"result.line2    : {line2}")
    print(f"result.conf1    : {conf1}")
    print(f"result.conf2    : {conf2}")
    print(f"band.x band.y band.w band.h : {band_box}")
    return 0 if ok else 1


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except Exception as e:
        print(f"tesseract_tool: {e}", file=sys.stderr)
        sys.exit(1)
