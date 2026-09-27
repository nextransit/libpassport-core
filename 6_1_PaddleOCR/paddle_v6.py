#!/usr/bin/env python3
"""paddle_v6 -- PP-OCRv6 pure-rec MRZ decoder (Phase 3).

Phase 3 result (10-case corpus_eval, vs the committed baselines):

    backend                          L1      L2     OK     speed
    paddle_local (tesseract ocrb)    83.0%   48.2%  10/10  4.5s/10
    PP-OCRv5 server rec (proj-split) 54.3%   94.8%  10/10  1.2s/10
    PP-OCRv6 medium rec (proj-split) 99.5%   95.5%  10/10  1.0s/10   <-- this file
    EasyOCR (whole 3x)               81.8%   83.6%   9/10  9.6s/10

Why V6 wins: the dense MRZ strip must NOT be run through a text
detector (det merges/drops lines). Instead we project-split the two
rows, upscale each 3x, and run PP-OCRv6 medium *recognition only*
on each single row. The v5-server rec is a universal (incl. Chinese)
dictionary model whose OCR-B glyph handling is poor (L1 54%), while
v6-medium rec tracks the narrow OCR-B line far better (L1 99.5%).

The stdout contract is identical to paddle_ocr_tool / paddle_local
so the GUI's ocr_bench_runner can pick it without changes:

    result.ok      : OK | unreadable | <error>
    result.line1   : 44-char line
    result.line2   : 44-char line
    result.conf1   : 0..100
    result.conf2   : 0..100
    band.x ...     : "x y w h"

If paddlex/onnxruntime are missing, we fall back to paddle_local
(tesseract ocrb baseline) so the tool keeps working everywhere.
"""
from __future__ import annotations
import os
import sys
import time
from pathlib import Path
import numpy as np
from PIL import Image

LINE_LEN = 44
DOC_TYPES = set("PIVACDR")
_ICAO = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<")

# Preferred V6 medium rec (onnx, no paddlepaddle needed). Fall back to
# the model dir names known to be cached at ~/.paddlex/official_models.
_MODEL_CANDIDATES = [
    ("PP-OCRv6_medium_rec", None),
]
_HOME_MODELS = Path.home() / ".paddlex" / "official_models"


def _find_model_dir(model_name: str) -> Path:
    for suffix in ("_onnx", ""):
        cand = _HOME_MODELS / (model_name + suffix)
        if cand.exists():
            return cand
    raise FileNotFoundError(f"model {model_name} not cached in {_HOME_MODELS}")


def _find_mrz_band(gray):
    """Reuse paddle_local's adaptive band finder."""
    from paddle_local import _find_mrz_band as _fn
    return _fn(gray)


def _split_rows(gray):
    """Project-split band into (row1, row2) at the deepest mid gutter."""
    y0, y1 = _find_mrz_band(gray)
    band = gray[y0:y1]
    H = band.shape[0]
    if H < 8:
        return band, band
    rd = (band < 128).sum(axis=1)
    mid_lo, mid_hi = int(H * 0.25), int(H * 0.75)
    if mid_hi <= mid_lo:
        split = H // 2
    else:
        split = int(mid_lo + int(np.argmin(rd[mid_lo:mid_hi])))
    return band[:split], band[split:]


def _norm(s: str) -> str:
    s = (s or "").strip().upper()
    s = "".join(c for c in s if c in _ICAO)
    if len(s) < LINE_LEN:
        s = s + "<" * (LINE_LEN - len(s))
    return s[:LINE_LEN]


def _rec_row(rec, row, scale=3):
    im = Image.fromarray(row).resize(
        (row.shape[1] * scale, row.shape[0] * scale), Image.LANCZOS
    )
    out = list(rec.predict(np.asarray(im.convert("RGB"))))
    if not out:
        return "", 0, ""
    o = out[0]
    txt = (o.get("rec_text") or "") if isinstance(o, dict) else ""
    sc = float(o.get("rec_score") or 0.0) if isinstance(o, dict) else 0.0
    return txt, sc, o.get("rec_text") or ""


_OCR = None


def _get_rec():
    global _OCR
    if _OCR is not None:
        return _OCR, "v6"
    try:
        import paddlex  # noqa: F401
    except ImportError:
        return None, "fallback"
    model_name = "PP-OCRv6_medium_rec"
    try:
        model_dir = _find_model_dir(model_name)
    except FileNotFoundError:
        return None, "fallback"
    _OCR = paddlex.create_model(
        model_name=model_name, model_dir=str(model_dir),
        device="cpu", engine="onnxruntime",
    )
    return _OCR, "v6"

def _decode_one(rec, img_path):
    """Run the V6 pure-rec pipeline on one image.

    Returns (ok, line1, line2, c1, c2, band_box, err, ms_ms). When
    `rec` is None the tesseract ocrb baseline (paddle_local) is used.
    """
    import time as _t
    t0 = _t.perf_counter()
    if rec is None:
        # Fallback: no paddlex / no cached model -> tesseract ocrb.
        # Capture paddle_local's stdout contract (it prints the same
        # result.* lines) so we return a single clean result block
        # instead of emitting a second empty one from the caller.
        import io
        import contextlib
        import paddle_local as _pl
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = _pl.main([sys.argv[0], str(img_path)])
        ms = (_t.perf_counter() - t0) * 1000.0
        out = {}
        for ln in buf.getvalue().splitlines():
            if ":" not in ln:
                continue
            k, _, v = ln.partition(":")
            out[k.strip()] = v.strip()
        return (rc == 0,
                out.get("result.line1", ""),
                out.get("result.line2", ""),
                int(out.get("result.conf1", "0") or 0),
                int(out.get("result.conf2", "0") or 0),
                out.get("band.x band.y band.w band.h", ""),
                "", ms)
    try:
        img = Image.open(img_path).convert("L")
        gray = np.asarray(img)
    except Exception as e:
        ms = (_t.perf_counter() - t0) * 1000.0
        return False, "", "", 0, 0, "", f"cannot load image: {e}", ms
    H, W = gray.shape
    y0, y1 = _find_mrz_band(gray)
    r1, r2 = _split_rows(gray)
    l1_raw, s1, _ = _rec_row(rec, r1)
    l2_raw, s2, _ = _rec_row(rec, r2)
    line1, line2 = _norm(l1_raw), _norm(l2_raw)
    l1_real = sum(1 for c in line1 if c not in "<")
    l2_real = sum(1 for c in line2 if c not in "<")
    ok = (len(line1) == LINE_LEN and len(line2) == LINE_LEN
          and line1[0] in DOC_TYPES and line1[1] == "<"
          and l1_real >= 5 and l2_real >= 5)
    c1 = int(min(100, s1 * 100))
    c2 = int(min(100, s2 * 100))
    band_box = f"0 {y0} {W} {y1 - y0}"
    ms = (_t.perf_counter() - t0) * 1000.0
    return ok, line1, line2, c1, c2, band_box, "", ms


def _emit(ok, line1, line2, c1, c2, band_box, ms=None):
    print(f"result.ok       : {'OK' if ok else 'unreadable'}")
    print(f"result.line1    : {line1}")
    print(f"result.line2    : {line2}")
    print(f"result.conf1    : {c1}")
    print(f"result.conf2    : {c2}")
    print(f"band.x band.y band.w band.h : {band_box}")
    if ms is not None:
        print(f"ms: {ms:.1f}")



def main(argv):
    if len(argv) < 2:
        print("usage: paddle_v6.py <image> | paddle_v6.py --batch <listfile>",
              file=sys.stderr)
        return 2

    # Batch mode: one process loads the model once and decodes every image.
    # listfile lines are "case_id<TAB>image_path". Emits one stdout contract
    # block per image (6 lines + blank), and @@PROGRESS@done@total@id to
    # stderr so the GUI can drive its progress bar live.
    if argv[1] == "--batch":
        if len(argv) < 3:
            print("paddle_v6: --batch needs a listfile", file=sys.stderr)
            return 2
        listfile = Path(argv[2])
        entries = []
        for ln in listfile.read_text().splitlines():
            ln = ln.strip()
            if not ln:
                continue
            if "\t" in ln:
                cid, ip = ln.split("\t", 1)
            elif " " in ln:
                cid, ip = ln.split(" ", 1)
            else:
                cid, ip = Path(ln).stem, ln
            entries.append((cid, Path(ip)))
        rec, mode = _get_rec()
        total = len(entries)
        for i, (cid, ip) in enumerate(entries, 1):
            ok, line1, line2, c1, c2, band, err, ms = _decode_one(rec, ip)
            _emit(ok, line1, line2, c1, c2, band, ms)
            print()
            if err:
                print(f"paddle_v6: {cid} {err}", file=sys.stderr)
            sys.stderr.write(f"@@PROGRESS@{i}@{total}@{cid}\n")
            sys.stderr.flush()
        return 0

    img_path = Path(argv[1])
    if not img_path.exists():
        print(f"cannot open {img_path}", file=sys.stderr)
        return 2

    rec, _mode = _get_rec()
    ok, line1, line2, c1, c2, band, err, _ms = _decode_one(rec, img_path)
    _emit(ok, line1, line2, c1, c2, band)
    return 0 if ok else 1


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except Exception as e:
        print(f"paddle_v6: {e}", file=sys.stderr)
        sys.exit(1)
