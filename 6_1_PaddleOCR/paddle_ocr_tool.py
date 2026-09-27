#!/usr/bin/env python3
"""paddle_ocr_tool -- PaddleOCR-backed MRZ recognizer.

Tries to use the real PaddleOCR engine (PaddleOCR + onnxruntime) when
it's installed in the active Python environment; falls back to
paddle_local.py (pure-local Tesseract OCR-B based decoder) if
paddleocr / onnxruntime are missing. Both paths emit the same stdout
contract so the GUI can compare OCR back-ends side-by-side:

    result.ok      : OK | <error message>
    result.line1   : 44-char line (or empty)
    result.line2   : 44-char line (or empty)
    result.conf1   : 0..100
    result.conf2   : 0..100
    band.x band.y band.w band.h : "x y w h" in image coordinates
"""
from __future__ import annotations
import os
import sys
import time
from pathlib import Path

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
    """Lazy-init PaddleOCR (production path)."""
    from paddleocr import PaddleOCR
    return PaddleOCR(
        lang="en",
        engine="onnxruntime",
        text_det_thresh=0.3,
        text_det_box_thresh=0.3,
        use_doc_orientation_classify=False,
        use_textline_orientation=False,
    )


def _find_mrz_band(gray):
    """Locate the two MRZ lines in the bottom third of the image."""
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


def _norm_line(s: str):
    """Pad/trim a recognised line to exactly 44 ICAO chars."""
    allowed = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<")
    s = (s or "").strip().upper()
    s = "".join(c for c in s if c in allowed)
    if len(s) < LINE_LEN:
        s = s + "<" * (LINE_LEN - len(s))
    s = s[:LINE_LEN]
    in_set = sum(1 for c in s if c in allowed)
    conf = min(100, int(100.0 * in_set / LINE_LEN))
    return s, conf


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
    # Combine (text, score), keep the two longest lines.
    pairs = [(t.strip(), float(s)) for t, s in zip(texts, scores)]
    pairs.sort(key=lambda p: len(p[0]), reverse=True)
    chosen = pairs[:2]
    if len(chosen) < 2:
        chosen.append(("", 0.0))
    # Try to identify which is line1 (doc-type prefix) vs line2 (digit prefix)
    a, sa = chosen[0]
    b, sb = chosen[1]
    def is_l1(s):
        return bool(s) and s[0] in "PIVACDR"
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
    """Production path: real PaddleOCR + onnxruntime.

    PP-OCRv6's text detector frequently collapses both MRZ lines into
    one (line2 only) on dense strips. We try multiple PSM-style splits:
    1. 3x whole-image det (best on strips where line1 has many '<' pads)
    2. 2x band split top/bottom (catches line1 cleanly)
    3. Whole band 2x single pass (line2 fallback)
    The l1/l2 picking heuristic then sorts the union into the right
    slots by doc-type prefix.
    """
    from PIL import Image
    import numpy as np
    img = Image.open(img_path).convert("RGB")
    gray = np.asarray(img.convert("L"))
    H, W = gray.shape
    y0, y1 = _find_mrz_band(gray)
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

    return _pick_l1_l2(texts, scores), (y0, y1, W, H)


def _run_local(img_path: Path):
    """Fallback path: pure-local Tesseract OCR-B based paddle_local.py."""
    # Lazy import — paddle_local.py is in the same directory.
    sys.path.insert(0, str(img_path.parent.parent / "6_1_PaddleOCR"))
    import paddle_local
    # Read stdout-style result by invoking paddle_local.main() in a
    # subprocess-less way: re-use the helpers directly.
    import numpy as np
    from PIL import Image
    img = Image.open(img_path).convert("RGB")
    gray = np.asarray(img.convert("L"))
    H, W = gray.shape
    y0, y1 = paddle_local._find_band(gray)
    crop = img.resize((W * 3, H * 3), Image.LANCZOS)
    # paddle_local._tesseract_band expects a file path; emit through a
    # tempfile in a subdir visible to the tesseract subprocess.
    import tempfile, subprocess
    with tempfile.TemporaryDirectory(prefix="/Users/zhouyong/.paddle_ocr_tool_tmp_") as td:
        bp = Path(td) / "band.png"
        crop.save(bp)
        text, conf, rc = paddle_local._tesseract_band(bp, td)
        if rc != 0:
            return ("", "", 0, 0), (y0, y1, W, H)
        # Fallback to PSM 11 when PSM 6 returned too few lines.
        if len([ln for ln in text.splitlines() if ln.strip()]) < 2:
            r2 = subprocess.run(
                ["tesseract", str(bp), str(Path(td) / "tess_out_psm11"),
                 "--psm", "11", "-l", "ocrb",
                 "-c", "tessedit_char_whitelist="
                      "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<"],
                capture_output=True, timeout=20,
            )
            if r2.returncode == 0:
                base = Path(str(Path(td) / "tess_out_psm11") + ".txt")
                if base.exists():
                    text_psm11 = base.read_text()
                    if len([ln for ln in text_psm11.splitlines() if ln.strip()]) > \
                       len([ln for ln in text.splitlines() if ln.strip()]):
                        text = text_psm11
        # Use paddle_local._norm_two_lines semantics by re-implementing
        # here (it's package-private).
        allowed = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<")
        raw_lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        raw_lines.sort(key=len, reverse=True)
        a = raw_lines[0] if raw_lines else ""
        b = raw_lines[1] if len(raw_lines) > 1 else ""
        def is_l1(s):
            return bool(s) and s[0] in "PIVACDR" and len(s) > 1 and s[1] == "<"
        def is_l2(s):
            return bool(s) and (s[0].isdigit() or s[0] == "L")
        if is_l1(b) and is_l2(a):
            l1, l2 = b, a
        elif is_l1(a) and is_l2(b):
            l1, l2 = a, b
        else:
            l1, l2 = a, b
        l1 = "".join(c if c in allowed else "<" for c in l1.upper())
        l2 = "".join(c if c in allowed else "<" for c in l2.upper())
        if len(l1) < LINE_LEN:
            l1 = l1 + "<" * (LINE_LEN - len(l1))
        l1 = l1[:LINE_LEN]
        if l1 and l1[1] != "<":
            l1 = l1[0] + "<" + l1[2:]
        if len(l2) < LINE_LEN:
            l2 = l2 + "<" * (LINE_LEN - len(l2))
        l2 = l2[:LINE_LEN]
        # Slot-enforce (alpha/digit)
        from paddle_local import (_enforce_slot, _LINE1_ALPHA, _LINE1_DIGIT,
                                  _LINE2_ALPHA, _LINE2_DIGIT)
        l1 = _enforce_slot(l1, _LINE1_ALPHA, _LINE1_DIGIT)
        l2 = _enforce_slot(l2, _LINE2_ALPHA, _LINE2_DIGIT)
        return (l1, l2, conf, conf), (y0, y1, W, H)


def main(argv):
    if len(argv) < 2:
        print("usage: paddle_ocr_tool.py <image>", file=sys.stderr)
        return 2
    img_path = Path(argv[1])
    if not img_path.exists():
        print(f"cannot open {img_path}", file=sys.stderr)
        return 2

    t0 = time.perf_counter()
    # Primary path: paddle_local (pure-local tesseract OCR-B based
    # decoder). It's empirically more accurate on dense MRZ strips
    # than PP-OCRv6's text detector, which frequently collapses
    # both MRZ lines into a single region and loses line1 entirely.
    # If paddle_local isn't importable (rare), we fall back to a
    # paddleocr-only pass.
    backend = "paddle_local"
    (line1, line2, conf1, conf2), (y0, y1, W, H) = _run_local(img_path)
    if not (line1 or line2) and _have_paddleocr():
        # paddle_local produced nothing usable — try paddleocr as
        # last-resort.
        try:
            (line1, line2, conf1, conf2), _ = _run_paddle(img_path)
            backend = "paddleocr"
        except Exception as e:
            print(f"result.ok       : {backend} error: {e}",
                  file=sys.stderr)
            return 1

    line1, conf1 = _norm_line(line1)
    line2, conf2 = _norm_line(line2)

    band_box = f"0 {y0} {W} {y1 - y0}"
    l1_real = sum(1 for c in line1 if c not in "<")
    l2_real = sum(1 for c in line2 if c not in "<")
    ok = (len(line1) == LINE_LEN and len(line2) == LINE_LEN
          and line1[0] in DOC_TYPES and line1[1] == "<"
          and l1_real >= 5 and l2_real >= 5)

    print(f"result.backend  : {backend}")
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
