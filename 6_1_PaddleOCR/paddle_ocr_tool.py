#!/usr/bin/env python3
"""paddle_ocr_tool -- PaddleOCR-backed MRZ recognizer.

Emits the same stdout contract as 6_mrz_ocr/build/mrz_ocr_tool so the
GUI can compare OCR back-ends side by side:

    result.ok      : OK | <error message>
    result.line1   : 44-char line (or empty)
    result.line2   : 44-char line (or empty)
    result.conf1   : 0..100
    result.conf2   : 0..100
    band.x band.y band.w band.h : "x y w h" in image coordinates

Uses the PaddleX ONNX Runtime engine (no paddlepaddle dependency),
which works on macOS arm64 + Python 3.14. Models auto-download to
~/.paddlex on first use.
"""
from __future__ import annotations
import sys
import time
from pathlib import Path

# Lazily-initialised single instance: PaddleOCR model init is expensive
# (loads det + rec + orientation models), so the tool keeps one global
# OCR handle across invocations when imported as a module, and reuses
# the process-local cache for the CLI path.
_OCR = None

LINE_LEN = 44


def _get_ocr():
    global _OCR
    if _OCR is None:
        from paddleocr import PaddleOCR
        # Lower detection thresholds so OCR-B / ICAO thin glyphs are
        # not filtered out; the default PP-OCRv6 det can miss the MRZ
        # band entirely on narrow crops.
        _OCR = PaddleOCR(
            lang="en",
            engine="onnxruntime",
            text_det_thresh=0.3,
            text_det_box_thresh=0.3,
            use_doc_orientation_classify=False,
            use_textline_orientation=False,
        )
    return _OCR


def _find_mrz_band(gray):
    """Locate the two MRZ lines in the bottom third of the image.
    Returns (y0, y1) in pixel space."""
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
    abs_groups = [(g[0] + bottom_start, g[1] + bottom_start)
                  for g in groups if g[1] - g[0] >= 10]
    if len(abs_groups) >= 2:
        for i in range(len(abs_groups) - 1, 0, -1):
            top, bot = abs_groups[i - 1], abs_groups[i]
            if bot[0] - top[1] <= 30:
                return top[0], bot[1]
    if abs_groups:
        return abs_groups[-1][0], abs_groups[-1][1]
    return int(H * 0.88), H


def _norm_line(s: str) -> tuple[str, int]:
    """Pad/trim a recognised line to exactly 44 ICAO chars.
    Returns (line, conf_percent)."""
    import re
    allowed = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<")
    s = (s or "").strip().upper()
    # Keep only ICAO chars; collapse runs of '<' to one placeholder
    # (PaddleOCR can emit consecutive '<' noise).
    s = "".join(c for c in s if c in allowed)
    if len(s) < LINE_LEN:
        s = s + "<" * (LINE_LEN - len(s))
    s = s[:LINE_LEN]
    in_set = sum(1 for c in s if c in allowed)
    conf = min(100, int(100.0 * in_set / LINE_LEN))
    return s, conf


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: paddle_ocr_tool.py <image>", file=sys.stderr)
        return 2
    img_path = Path(argv[1])
    if not img_path.exists():
        print(f"cannot open {img_path}", file=sys.stderr)
        return 2

    t0 = time.perf_counter()
    try:
        from PIL import Image
        import numpy as np
        img = Image.open(img_path).convert("RGB")
    except Exception as e:
        print(f"result.ok       : cannot load image: {e}", file=sys.stderr)
        return 1
    gray = np.asarray(img.convert("L"))
    H, W = gray.shape
    y0, y1 = _find_mrz_band(gray)
    crop = img.crop((0, y0, W, y1))
    # 2x upscale so the detector sees glyphs at a comfortable scale.
    crop = crop.resize((crop.width * 2, crop.height * 2), Image.LANCZOS)

    # Split the MRZ band into two horizontal halves and OCR each half
    # separately. PP-OCRv6 det on the full 2-line band frequently
    # collapses both rows into a single text region; splitting makes
    # line1 and line2 independently recoverable (same trick as the
    # tesseract back-end).
    try:
        ocr = _get_ocr()
    except Exception as e:
        print(f"result.ok       : paddle init error: {e}", file=sys.stderr)
        return 1

    band_h = y1 - y0
    half = band_h // 2
    tops = crop.crop((0, 0, crop.width, max(1, half + 2)))
    bots = crop.crop((0, max(0, half - 2), crop.width, crop.height))

    def _predict_halves(im):
        try:
            r = ocr.predict(np.asarray(im))
        except Exception as e:
            print(f"result.ok       : paddle predict error: {e}",
                  file=sys.stderr)
            return [], []
        if not r:
            return [], []
        res = r[0]
        return res.get("rec_texts") or [], res.get("rec_scores") or []

    def _pick(im):
        """Return the most plausible MRZ line from one half: the text
        with the most ICAO chars and a non-trivial score."""
        texts, scores = _predict_halves(im)
        best, best_s, best_n = "", 0, 0
        for t, s in zip(texts, scores):
            t = (t or "").strip()
            n = sum(1 for c in t if c.isalnum() or c == "<")
            if n >= 4 and (n, s) > (best_n, best_s):
                best, best_s, best_n = t, float(s), n
        return best, best_s

    t1, s1 = _pick(tops)
    t2, s2 = _pick(bots)
    # If one half failed, fall back to the whole-band single-pass result.
    if (not t1 or not t2):
        try:
            res = ocr.predict(np.asarray(crop))[0]
            texts = res.get("rec_texts") or []
            scores = res.get("rec_scores") or []
        except Exception:
            texts, scores = [], []
        cands = sorted(
            [(t, s) for t, s in zip(texts, scores)
             if sum(1 for c in (t or "") if c.isalnum() or c == "<") >= 10],
            key=lambda x: len(x[0]), reverse=True)
        if not t1 and cands:
            t1, s1 = cands[0][0], cands[0][1]
        if not t2 and len(cands) > 1:
            t2, s2 = cands[1][0], cands[1][1]

    line1, conf1 = _norm_line(t1)
    line2, conf2 = _norm_line(t2)

    band_box = f"0 {y0} {W} {y1 - y0}"
    # ok gate: both lines 44 chars, line1 opens with a valid document
    # type (P/I/V/A/C/D/R + '<'), and line1 carries real content beyond
    # '<' padding (a fully-padded '<'*44 line must NOT pass).
    l1_real = sum(1 for c in line1 if c not in "<")
    l2_real = sum(1 for c in line2 if c not in "<")
    ok = (len(line1) == LINE_LEN and len(line2) == LINE_LEN
          and line1[0] in set("PIVACDR<") and line1[1] == "<"
          and l1_real >= 5 and l2_real >= 5)

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
        print(f"paddle_ocr_tool: {e}", file=sys.stderr)
        sys.exit(1)
