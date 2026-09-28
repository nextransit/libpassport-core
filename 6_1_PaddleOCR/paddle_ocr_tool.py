#!/usr/bin/env python3
"""paddle_ocr_tool -- MRZ recognizer with two interchangeable back-ends.

Back-end selection is explicit (env MRZ_BACKEND, default "local"):

  * local     : paddle_local.py — pure-local Tesseract OCR-B decoder.
                Empirically more accurate on dense MRZ strips than the
                PP-OCR det+rec path, which frequently collapses both
                MRZ rows into a single region and loses Line 1.
  * paddleocr : PaddleOCR det+rec (needs paddleocr + onnxruntime in the
                active environment). Kept as a comparison back-end; it
                is also used automatically when "local" yields nothing.

Both paths emit the same stdout contract so the GUI can compare OCR
back-ends side-by-side:

    result.backend  : local | paddleocr
    result.ok       : OK | unreadable | <error message>
    result.line1    : 44-char line (or empty)
    result.line2    : 44-char line (or empty)
    result.conf1    : 0..100
    result.conf2    : 0..100
    band.x band.y band.w band.h : "x y w h" in image coordinates

Confidence comes from the selected back-end (tesseract TSV per-line
means for "local", recognition scores for "paddleocr"); this file no
longer overwrites it with a constant character-legality rate.
"""
from __future__ import annotations
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import paddle_local  # noqa: E402

LINE_LEN = 44
DOC_TYPES = set("PIVACDR")


def _have_paddleocr():
    """True iff paddleocr + onnxruntime are importable in this env."""
    try:
        import paddleocr  # noqa: F401
        import onnxruntime  # noqa: F401
        return True
    except ImportError:
        return False


def _get_ocr_paddle():
    """Lazy-init PaddleOCR (comparison path)."""
    from paddleocr import PaddleOCR
    return PaddleOCR(
        lang="en",
        engine="onnxruntime",
        text_det_thresh=0.3,
        text_det_box_thresh=0.3,
        use_doc_orientation_classify=False,
        use_textline_orientation=False,
    )


def _pad_line(s: str) -> str:
    """Pad/trim a recognised line to exactly 44 ICAO chars."""
    s = (s or "").strip().upper()
    s = "".join(c for c in s if c in paddle_local.ICAO)
    if len(s) < LINE_LEN:
        s = s + "<" * (LINE_LEN - len(s))
    return s[:LINE_LEN]


def _predict_paddle(ocr, np_img):
    """Run paddleocr on the given image. Returns (texts, scores)."""
    r = ocr.predict(np_img)
    if not r:
        return [], []
    return r[0].get("rec_texts") or [], r[0].get("rec_scores") or []


def _pick_l1_l2(texts, scores):
    """Pick the two best MRZ-shaped lines from a paddleocr result.

    Heuristic: choose the two longest lines; if one starts with a doc
    type ('P','I','V','A','C','D','R') and the other with a digit /
    'L', keep that order. Otherwise order by the doc-type heuristic
    fallback so the GUI's line1 / line2 contract holds.
    """
    if not texts:
        return "", "", 0, 0
    pairs = [(t.strip(), float(s)) for t, s in zip(texts, scores)]
    pairs.sort(key=lambda p: len(p[0]), reverse=True)
    chosen = pairs[:2]
    if len(chosen) < 2:
        chosen.append(("", 0.0))
    a, sa = chosen[0]
    b, sb = chosen[1]

    def is_l1(s):
        return bool(s) and s[0] in DOC_TYPES

    def is_l2(s):
        return bool(s) and (s[0].isdigit() or s[0] == "L")

    if is_l1(a) and is_l2(b):
        l1, sl1, l2, sl2 = a, sa, b, sb
    elif is_l1(b) and is_l2(a):
        l1, sl1, l2, sl2 = b, sb, a, sa
    elif is_l1(a):
        l1, sl1, l2, sl2 = a, sa, "", 0.0
    else:
        l1, sl1, l2, sl2 = "", 0.0, a, sa
    return l1, l2, int(sl1 * 100), int(sl2 * 100)


def _run_paddle(img_path: Path):
    """PaddleOCR det+rec path. Returns (l1, l2, c1, c2, band_box, err).

    PP-OCRv6's text detector frequently collapses both MRZ lines into
    one (line2 only) on dense strips. We try multiple splits:
    1. 3x whole-image det (best on strips where line1 has many '<' pads)
    2. 2x band split top/bottom (catches line1 cleanly)
    The l1/l2 picking heuristic then sorts the union into the right
    slots by doc-type prefix.
    """
    from PIL import Image
    import numpy as np
    img = Image.open(img_path).convert("RGB")
    gray = np.asarray(img.convert("L"))
    H, W = gray.shape
    y0, y1 = paddle_local._find_mrz_band(gray)
    band_box = f"0 {y0} {W} {y1 - y0}"
    band = img.crop((0, y0, W, y1))

    ocr = _get_ocr_paddle()

    # Pass 1: 3x whole image — picks up both MRZ lines if det accepts.
    # Filter out short / empty text boxes (e.g. single '1' or '<' noise).
    img3 = img.resize((W * 3, H * 3), Image.LANCZOS)
    all_texts, all_scores = _predict_paddle(ocr, np.asarray(img3))
    texts = []
    scores = []
    for t, sc in zip(all_texts, all_scores):
        if t and len(t.strip()) >= 5:
            texts.append(t)
            scores.append(sc)
    n_lines = len(texts)

    # Pass 2 (fallback): 2x band split — line1 often only detected here
    if n_lines < 2:
        band2 = band.resize((W * 2, (y1 - y0) * 2), Image.LANCZOS)
        band_h = y1 - y0
        half = band_h // 2
        tops = band2.crop((0, 0, W * 2, max(1, half * 2 + 4)))
        bots = band2.crop((0, max(0, half * 2 - 4), W * 2, band_h * 2))
        texts_top, scores_top = _predict_paddle(ocr, np.asarray(tops))
        texts_bot, scores_bot = _predict_paddle(ocr, np.asarray(bots))
        merged_texts = list(texts_top) + list(texts_bot)
        merged_scores = list(scores_top) + list(scores_bot)
        if len(merged_texts) > n_lines:
            texts, scores = merged_texts, merged_scores

    l1, l2, c1, c2 = _pick_l1_l2(texts, scores)
    return l1, l2, c1, c2, band_box, ""


def _run_local(img_path: Path):
    """Tesseract OCR-B path (paddle_local.decode_image)."""
    d = paddle_local.decode_image(str(img_path))
    return (d["line1"], d["line2"], d["conf1"], d["conf2"],
            d["band_box"], d["err"])


def main(argv):
    if len(argv) < 2:
        print("usage: paddle_ocr_tool.py <image>", file=sys.stderr)
        return 2
    img_path = Path(argv[1])
    if not img_path.exists():
        print(f"cannot open {img_path}", file=sys.stderr)
        return 2

    t0 = time.perf_counter()
    backend = os.environ.get("MRZ_BACKEND", "local").lower()
    if backend not in ("local", "paddleocr"):
        backend = "local"
    err = ""
    if backend == "local":
        line1, line2, conf1, conf2, band_box, err = _run_local(img_path)
        if not (line1 or line2) and _have_paddleocr():
            # local produced nothing usable — last-resort paddleocr pass.
            try:
                line1, line2, conf1, conf2, band_box, err = \
                    _run_paddle(img_path)
                backend = "paddleocr"
            except Exception as e:
                err = f"{backend} error: {e}"
    else:
        try:
            line1, line2, conf1, conf2, band_box, err = _run_paddle(img_path)
        except Exception as e:
            line1 = line2 = band_box = ""
            conf1 = conf2 = 0
            err = f"{backend} error: {e}"
        if not (line1 or line2):
            line1, line2, conf1, conf2, band_box, err2 = _run_local(img_path)
            err = err or err2
            backend = "local"

    line1, line2 = _pad_line(line1), _pad_line(line2)
    l1_real = sum(1 for c in line1 if c not in "<")
    l2_real = sum(1 for c in line2 if c not in "<")
    ok = (line1[0] in DOC_TYPES and line1[1] == "<"
          and l1_real >= 5 and l2_real >= 5)

    print(f"result.backend  : {backend}")
    print(f"result.ok       : {'OK' if ok else 'unreadable'}")
    print(f"result.line1    : {line1}")
    print(f"result.line2    : {line2}")
    print(f"result.conf1    : {conf1}")
    print(f"result.conf2    : {conf2}")
    print(f"band.x band.y band.w band.h : {band_box}")
    if err:
        print(f"paddle_ocr_tool: {err}", file=sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except Exception as e:
        print(f"paddle_ocr_tool: {e}", file=sys.stderr)
        sys.exit(1)
