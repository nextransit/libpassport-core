#!/usr/bin/env python3
"""Shared MRZ band candidates and candidate selection.

The training-slice exporter (export_photo_train.py) and the weight-set
evaluator (eval_weight_sets.py) must cut the SAME crop out of every photo.
Otherwise the glyph framing the CNN is trained on differs from the framing
it is scored on and the measurement is a train/inference mismatch instead of
the model.  Both therefore go through this module.

Two band locators are used:

* ``fast_pair`` (export_photo_train.locate_band) -- the historical locator.
  It is right for flat scans but sometimes returns the page body instead of
  the MRZ (assets/pic/1792f7a2d79c7a8a_0017.jpg) or a band whose top edge
  carries a dark strip, which makes the C tool frame line 1 over that strip
  and read the whole row wrong (assets/pic/0f9780b8e1e3cc21_0008.jpg).
* the text-row pair from passport_photo_restore.restore.find_mrz_pair --
  accurate on photo pages, weaker on flat scans.

Instead of trusting one of them, both are cut at a few paddings and the
candidate whose line2 passes the ICAO 9303 check digits -- and, when both
pass, whose confidences are higher -- is kept.  No ground truth is used.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "passport_photo_restore" / "src"))
sys.path.insert(0, str(ROOT / "tests"))

# Paddings tried around the fast_pair band and around the text-row pair.
# The first candidate of every list is the historical crop, so a tie keeps
# the previous behaviour.
BASE_PAD = 10
FAST_PADS = (4, 6, 0)
RES_PADS = (4, 6, 0, 10)
# Paddings of the row-scan fallback (see sensitive_band_rows).
SENS_PADS = (2, 6)

_WEIGHT = {str(d): d for d in range(10)}
_WEIGHT.update({chr(c): c - 55 for c in range(65, 91)})
_WEIGHT["<"] = 0


def check_digit(s: str) -> str:
    """ICAO 9303 7-3-1 weighted check digit."""
    total = 0
    for i, ch in enumerate(s):
        total += _WEIGHT.get(ch, 0) * (7, 3, 1)[i % 3]
    return str(total % 10)


def line2_bad_checks(line2: str) -> list:
    """Names of the TD3 line2 check digits that do not match."""
    if len(line2) != 44:
        return ["length"]
    bad = []
    if check_digit(line2[0:9]) != line2[9]:
        bad.append("document_number")
    if check_digit(line2[13:19]) != line2[19]:
        bad.append("birth_date")
    if check_digit(line2[21:27]) != line2[27]:
        bad.append("expiry_date")
    if check_digit(line2[28:42]) != line2[42]:
        bad.append("personal_number")
    composite = line2[0:10] + line2[13:20] + line2[21:28] + line2[28:43]
    if check_digit(composite) != line2[43]:
        bad.append("composite")
    return bad


def restore_pair_rows(rgb: Image.Image):
    """Text-row pair of the MRZ in native pixels, or None."""
    try:
        import cv2
        import restore
    except Exception:
        return None
    try:
        arr = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)
        work, scale = restore.load(arr, max_side=1250)
        gray = restore._gray(work)
        st = restore.detect_structure(gray)
    except Exception:
        return None
    pair = st.get("mrz_pair")
    if not pair:
        return None
    (a0, a1), (b0, b1) = pair
    return (int(a0 / scale), int(a1 / scale), int(b0 / scale), int(b1 / scale))


def _otsu_threshold(a: np.ndarray) -> int:
    hist, _ = np.histogram(a, bins=256, range=(0, 256))
    p = hist.astype(float) / a.size
    w = np.cumsum(p)
    m = np.cumsum(p * np.arange(256))
    mt = m[-1]
    den = w * (1 - w)
    den[den == 0] = 1e-12
    sb = (mt * w - m) ** 2 / den
    sb[den < 1e-11] = 0
    return int(np.argmax(sb))


def text_row_groups(gray: np.ndarray, thr: int | None = None,
                    min_seg: int = 20):
    """Group the rows of a page into text lines.

    A text line is a run of rows that carry many small dark segments (one
    per glyph); a photo page carries 40+ such segments on an MRZ row, while
    borders, shadows and solid bars carry one or two.  Returns a list of
    ``(y0, y1, max_segs, max_ink, x0, x1)`` in native pixels.
    """
    H, W = gray.shape
    t = thr if thr is not None else _otsu_threshold(gray)
    dark = gray <= t
    segs = np.zeros(H, dtype=np.int32)
    ink = np.zeros(H, dtype=np.int32)
    x0 = np.full(H, -1, dtype=np.int32)
    x1 = np.full(H, -1, dtype=np.int32)
    for y in range(H):
        row = dark[y]
        segs[y] = int((np.diff(row.astype(np.int8)) != 0).sum()) // 2 + (1 if row.any() else 0)
        ink[y] = int(row.sum())
        xs = np.where(row)[0]
        if xs.size:
            x0[y] = int(xs[0])
            x1[y] = int(xs[-1])
    groups = []
    for y in range(H):
        if segs[y] < min_seg:
            continue
        if groups and y - groups[-1][1] <= 2:
            g = groups[-1]
            groups[-1] = (g[0], y, max(g[2], int(segs[y])), max(g[3], int(ink[y])),
                          min(g[4], int(x0[y])) if x0[y] >= 0 else g[4],
                          max(g[5], int(x1[y])))
        else:
            groups.append((y, y, int(segs[y]), int(ink[y]),
                           int(x0[y]) if x0[y] >= 0 else 0, int(x1[y])))
    return groups


def sensitive_band_rows(rgb: Image.Image, min_seg: int = 20, min_h: int = 6):
    """Rows of the bottom-most pair of text lines, in native pixels.

    The two locators above both need a clear MRZ; on a photo of a passport
    that fills a small part of the frame (a social-media screenshot) the MRZ
    is only ~15 px high and neither of them finds it.  The MRZ is always the
    last thing on the page, so the bottom-most two text lines are a usable
    fallback.  Returns (y0a, y1a, y0b, y1b, x0a, x1a, x0b, x1b) or None.
    """
    try:
        gray = np.asarray(rgb.convert("L"))
    except Exception:
        return None
    groups = [g for g in text_row_groups(gray, min_seg=min_seg)
              if g[1] - g[0] + 1 >= min_h]
    if len(groups) < 2:
        return None
    a, b = groups[-2], groups[-1]
    return (a[0], a[1], b[0], b[1], a[4], a[5], b[4], b[5])


def band_candidates(rgb: Image.Image, pad: int = BASE_PAD,
                    pad_map_value: int | None = None):
    """Candidate crops for one photo.

    Returns a list of ``(kind, rect)`` with rect = (x0, y0, x1, y1) in
    native pixels.  The first entry is the historical crop, so an
    unchanged selection keeps the previous framing.
    """
    from export_photo_train import locate_band

    W, H = rgb.size
    hint: list = []
    band = locate_band(rgb, hint)
    cands = []
    if band:
        bx, by, bw, bh = band
        special = max(pad, hint[0] if hint else 0, pad_map_value or 0)
        cands.append((f"fast{special}",
                      (bx, max(0, by - special), bx + bw, min(H, by + bh + special))))
        for p in FAST_PADS:
            if p == special:
                continue
            cands.append((f"fast{p}",
                          (bx, max(0, by - p), bx + bw, min(H, by + bh + p))))
    rows = restore_pair_rows(rgb)
    if rows:
        y0a, _y1a, _y0b, y1b = rows
        for p in RES_PADS:
            cands.append((f"res{p}", (0, max(0, y0a - p), W, min(H, y1b + p))))
    sens = sensitive_band_rows(rgb)
    if sens:
        y0a, _y1a, _y0b, y1b, x0a, x1a, x0b, x1b = sens
        x0 = max(0, min(x0a, x0b))
        x1 = min(W, max(x1a, x1b))
        for p in SENS_PADS:
            cands.append((f"sens{p}",
                          (max(0, x0 - p), max(0, y0a - p),
                           min(W, x1 + p), min(H, y1b + p))))
    return cands


def parse_kv(out: str) -> dict:
    kv = {}
    for ln in out.splitlines():
        if ":" in ln:
            k, _, v = ln.partition(":")
            kv[k.strip()] = v.strip()
    return kv


def parse_result(kv: dict) -> dict:
    """Map the tool's raw ``result.*`` keys to the flat shape run_tool returns.

    A resident tool over a pipe returns the same stdout block as a
    one-shot run, so the caller can parse it without spawning a
    process per candidate.
    """
    return {
        "ok": "OK" in kv.get("result.ok", ""),
        "line1": kv.get("result.line1", ""),
        "line2": kv.get("result.line2", ""),
        "conf1": int(kv.get("result.conf1", 0) or 0),
        "conf2": int(kv.get("result.conf2", 0) or 0),
        "layout": kv.get("result.layoutname", "?"),
    }


def run_tool(tool, ppm: Path, env_extra: dict | None = None,
             timeout: int = 120) -> dict:
    env = dict(os.environ)
    env["MRZ_OCR_LINE_MODELS"] = "1"
    env["MRZ_OCR_CRNN"] = "0"
    env["MRZ_OCR_FUSE"] = "0"
    if env_extra:
        env.update(env_extra)
    try:
        p = subprocess.run([str(tool), str(ppm)], capture_output=True,
                           text=True, timeout=timeout, env=env)
    except subprocess.TimeoutExpired:
        return {}
    return parse_result(parse_kv(p.stdout))


def candidate_score(result: dict):
    """Rank a candidate: fewer broken check digits first, then confidence.

    A line2 that is not 44 characters long is treated as fully broken --
    an empty or short line carries no check-digit information at all.
    """
    line2 = result.get("line2") or ""
    if len(line2) != 44:
        ck = 10
    else:
        ck = len(line2_bad_checks(line2))
    conf = int(result.get("conf1") or 0) + int(result.get("conf2") or 0)
    return (ck, -conf)


def pick_candidate(results: list, kinds: list | None = None):
    """Index of the best candidate; ties keep the earlier (historical) one.

    ``sens`` candidates come from a heuristic row scan and can latch onto
    unrelated text near the bottom of the page, so they are only considered
    when no primary candidate produced a line2 with valid ICAO check digits.
    """
    if not results:
        return 0
    scores = [candidate_score(r) for r in results]
    primary = [i for i in range(len(results))
               if kinds is None or not kinds[i].startswith("sens")]
    pool = primary
    if not primary or not any(scores[i][0] == 0 for i in primary):
        pool = list(range(len(results)))
    best = pool[0]
    best_score = scores[best]
    for i in pool[1:]:
        if scores[i] < best_score:
            best_score = scores[i]
            best = i
    return best
