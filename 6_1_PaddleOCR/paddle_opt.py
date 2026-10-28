#!/usr/bin/env python3
"""Optimised PaddleOCR MRZ pipeline (opt-in variant of paddle_v6.py).

Differences against 6_1_PaddleOCR/paddle_v6.py:

  * Row geometry copied from the Android app: each MRZ row is cut from the
    source photo with pad = max(8, min(25, 0.7 * rowHeight)), the second row
    starts below the first row's bottom, and the recognition width is
    48 * cropWidth / cropHeight * widthFactor.  paddle_v6 instead upscales
    every row 3x with LANCZOS and then clamps the width to 3200px, which
    both distorts the glyphs and wastes compute.
  * Tuned ONNX Runtime session (intra_op threads, ORT_ENABLE_ALL, mem
    arena/pattern) instead of the library defaults.
  * The two rows are recognised concurrently.
  * Optional multi-hypothesis reading (several width factors) with
    checksum-driven selection.
  * Optional two-model cascade: the small v4 mobile recogniser proposes,
    the v6 medium recogniser referees when the proposal is weak.

The stdout contract is identical to paddle_v6.py, so the GUI and
tests/ocr_bench_runner.py can consume it unchanged.
"""

import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
for p in (HERE, os.path.join(ROOT, "11_indra"), os.path.join(ROOT, "passport_photo_restore/src")):
    if p not in sys.path:
        sys.path.insert(0, p)

import onnxruntime as ort  # noqa: E402
import paddle_v6 as P  # noqa: E402  (read-only reference implementation)

V6_DIR = os.path.expanduser(
    os.environ.get("MRZ_PADDLE_MODEL", "~/.paddlex/official_models/PP-OCRv6_medium_rec_onnx"))
V4M_ONNX = os.path.join(ROOT, "8_android/app/src/main/assets/models/ppocrv4_mobile_rec.onnx")
V4M_DICT = os.path.join(ROOT, "8_android/app/src/main/assets/models/ppocrv4_mobile_rec.dict.txt")

WF_ROW1 = (0.9,)
WF_ROW2 = (0.75,)
WF_ROW1_EXT = (0.75, 1.0)
WF_ROW2_EXT = (0.9, 1.0)
L1_MIN_SCORE = 0.80
L2_MIN_SCORE = 0.85


# ----------------------------------------------------------------------
# Model wrapper
# ----------------------------------------------------------------------
class Rec:
    """Recognition-only wrapper around a PaddleOCR rec model."""

    def __init__(self, model_path, characters, height=48, max_width=4096, threads=6):
        so = ort.SessionOptions()
        so.intra_op_num_threads = int(os.environ.get("MRZ_THREADS", threads))
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.enable_cpu_mem_arena = True
        so.enable_mem_pattern = True
        self._session = ort.InferenceSession(model_path, so, providers=["CPUExecutionProvider"])
        self._characters = characters
        self._height = height
        self._max_width = max_width

    def prepare(self, gray: np.ndarray, width_factor: float = 1.0) -> np.ndarray:
        gray = trim_frame(gray)
        h, w = gray.shape[:2]
        width = int(math.ceil(self._height * w / float(h) * width_factor))
        width = max(16, min(self._max_width, width))
        img = Image.fromarray(gray).resize((width, self._height), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = (arr - 0.5) / 0.5
        arr = np.repeat(arr[:, :, None], 3, axis=2)
        return arr.transpose(2, 0, 1)[None]

    def decode(self, output: np.ndarray):
        idx = output[0].argmax(axis=1)
        prob = output[0].max(axis=1)
        chars = self._characters
        text = []
        prev = -1
        total = 0.0
        for i, k in enumerate(idx):
            k = int(k)
            if k != prev and k != 0 and k < len(chars):
                text.append(chars[k])
                total += float(prob[i])
            prev = k
        score = total / len(text) if text else 0.0
        return "".join(text), score

    def predict(self, gray: np.ndarray, width_factor: float = 1.0):
        tensor = self.prepare(gray, width_factor)
        out = self._session.run(None, {self._session.get_inputs()[0].name: tensor})[0]
        return self.decode(out)


def load_v6():
    model = os.path.join(V6_DIR, "inference.onnx")
    if not os.path.exists(model):
        return None
    height, characters = 48, None
    yml = os.path.join(V6_DIR, "inference.yml")
    if os.path.exists(yml):
        try:
            import yaml
            meta = yaml.safe_load(open(yml, encoding="utf-8")) or {}
            for op in meta.get("PreProcess", {}).get("transform_ops", []):
                shape = op.get("RecResizeImg", {}).get("image_shape")
                if shape:
                    height = int(shape[1])
            characters = list(meta.get("PostProcess", {}).get("character_dict", []))
        except Exception:
            characters = None
    if not characters:
        return None
    return Rec(model, ["blank"] + characters, height=height)


def load_v4m():
    if not (os.path.exists(V4M_ONNX) and os.path.exists(V4M_DICT)):
        return None
    with open(V4M_DICT, encoding="utf-8") as fh:
        chars = [line.rstrip("\n") for line in fh if line.strip("\n")]
    return Rec(V4M_ONNX, ["blank"] + chars, height=48)


# ----------------------------------------------------------------------
# Row extraction
# ----------------------------------------------------------------------
def _adaptive_xpad(gray, nominal):
    """Drop the dark camera-frame border when the photo has one."""
    h, w = gray.shape[:2]
    strip = max(4, min(12, w // 25))
    for name, cols in (("left", slice(0, strip)), ("right", slice(w - strip, w))):
        band = gray[:, cols]
        dark = float((band < 160).mean()) if band.size else 0.0
        if dark >= 0.9:
            return nominal if name == "left" else nominal
    return 0


def _otsu_ink(gray: np.ndarray):
    import cv2

    _, thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return (thr > 0).astype(np.uint8)


def _ink_bands(ink: np.ndarray, min_ink: int = 3):
    """Text bands (row runs with ink) of a binarised page."""
    proj = ink.sum(axis=1)
    on = proj > min_ink
    bands = []
    y = 0
    h = len(on)
    while y < h:
        if on[y]:
            y0 = y
            while y < h and on[y]:
                y += 1
            bands.append((y0, y))
        else:
            y += 1
    merged = []
    for b in bands:
        if merged and b[0] - merged[-1][1] <= 3:
            merged[-1] = (merged[-1][0], b[1])
        else:
            merged.append(b)
    return merged


def _band_mrz_score(ink: np.ndarray, y0: int, y1: int):
    """Score how much a text band looks like a 44 character MRZ line."""
    band = ink[y0:y1]
    if band.size == 0:
        return -1e9, 0
    col = band.sum(axis=0)
    on = col > 0
    starts = []
    x = 0
    w = len(on)
    while x < w:
        if on[x]:
            starts.append(x)
            while x < w and on[x]:
                x += 1
        else:
            x += 1
    n = len(starts)
    if n < 20:
        return -1e9, n
    d = np.diff(np.asarray(starts, dtype=float))
    jitter = float(d.std() / max(1e-6, d.mean())) if d.size > 2 else 9.9
    score = -abs(n - 44) * 1.0 - jitter * 12.0 - 0.01 * max(1, y1 - y0)
    return score, n


def profile_row_pairs(gray: np.ndarray, limit: int = 2):
    """Rank row pairs that look like the MRZ by their ink run profile.

    The rectifier sometimes locks onto a neighbouring caption (e.g. the
    Chinese "有效期至" row) instead of the MRZ.  A MRZ line shows ~44 evenly
    spaced ink groups, so scoring the run profile finds the real lines
    without an OCR pass.
    """
    ink = _otsu_ink(gray)
    bands = _ink_bands(ink)
    scored = []
    for b in bands:
        score, n = _band_mrz_score(ink, b[0], b[1])
        scored.append((b, score, n))
    pairs = []
    for i in range(len(scored) - 1):
        (b0, s0, n0) = scored[i]
        (b1, s1, n1) = scored[i + 1]
        if n0 < 28 or n1 < 28:
            continue
        gap = b1[0] - b0[1]
        if gap > 3.0 * max(b1[1] - b1[0], b0[1] - b0[0]):
            continue
        score = s0 + s1 + (4.0 if abs(n0 - n1) <= 4 else 0.0)
        pairs.append((score, (b0, b1)))
    pairs.sort(key=lambda item: -item[0])
    return [(b0, b1) for _, (b0, b1) in pairs[:limit]]


def _crop_row(gray, top, bot, xpad, w):
    row_h = max(4, bot - top)
    pad = max(8, min(25, int(row_h * 0.70)))
    top = max(0, top - pad)
    bot = min(gray.shape[0], bot + pad)
    if xpad:
        return gray[top:bot, xpad:max(xpad + 1, w - xpad)]
    return gray[top:bot]


def find_rows(rgb: np.ndarray):
    """Return [(row_gray, geometry), ...] for the two MRZ lines.

    Uses the rectification pipeline of passport_photo_restore (the same one
    the Android app uses) and cuts the rows from the source photo.  Falls
    back to paddle_v6's band split for images that are already cropped
    bands (the synthetic corpus).
    """
    import restore as R

    gray_src = np.asarray(Image.fromarray(rgb).convert("L"))
    try:
        work, scale = R.load(rgb)
        g = R._gray(work)
        pair = R.find_mrz_pair(g)
    except Exception:
        pair = None
    if pair is not None:
        h, w = gray_src.shape[:2]
        nominal = min(100, max(10, int(round(w * 0.08))))
        xpad = _adaptive_xpad(gray_src, nominal)
        rows = []
        first_bottom = None
        for idx, (y0, y1) in enumerate(pair):
            top = int(y0 / scale)
            bot = min(h, int(y1 / scale) + 1)
            row_h = max(4, bot - top)
            if idx == 1 and first_bottom is not None:
                top = max(top, first_bottom)
            first_bottom = bot
            pad = max(8, min(25, int(row_h * 0.70)))
            top = max(0, top - pad)
            bot = min(h, bot + pad)
            if xpad:
                row = gray_src[top:bot, xpad:max(xpad + 1, w - xpad)]
            else:
                row = gray_src[top:bot]
            if row.size:
                rows.append(row)
        if len(rows) == 2:
            return rows, "restore"
    # fallback: paddle_v6 band split (corpus bands)
    band = P._find_mrz_band(gray_src)
    if band is not None:
        y0, y1 = band
        pieces = P._split_rows(gray_src, y0, y1)
        if len(pieces) == 2:
            return [p[0] if isinstance(p, tuple) else p for p in pieces], "band"
    return [], "none"


# ----------------------------------------------------------------------
# Recognition + selection
# ----------------------------------------------------------------------
def _line_shape(g, band, thr, W):
    """Shape test of restore._mrz_pair_pass, plus the stroke count.

    An OCR-B MRZ row is made of ~44 separated strokes; a caption row has
    far fewer.  The stroke count is what tells the two apart.
    """
    import restore as R

    y0, y1 = band
    h = y1 - y0 + 1
    if h < 5 or h > 48:
        return None
    mask = g[y0:y1 + 1] < thr
    col_ink = mask.sum(axis=0)
    strong = col_ink >= 4
    groups = R._run_groups(strong, min_len=1)
    max_group = max((b - a + 1 for a, b in groups), default=0)
    density = float(strong.mean())
    if density > 0.90 or max_group > max(160, int(W * 0.45)):
        return None
    return density, len(groups), max_group


def _mrz_candidate_pairs(g, limit=4):
    """Enumerate MRZ row-pair candidates on the rectified page.

    restore.find_mrz_pair returns a single pair and prefers the tallest
    glyphs, which locks onto caption blocks such as "有效期至 / Date of
    expiry" instead of the MRZ.  Here every adjacent text-run pair is
    enumerated and ranked by the total stroke count, so the 44-character
    MRZ rows win.
    """
    import restore as R

    H, W = g.shape[:2]
    row_ink180 = (g < 180).sum(axis=1)
    gx = np.abs(np.diff(g.astype(np.int16), axis=1))
    gx = np.pad(gx, ((0, 0), (1, 0)))
    E = gx.sum(axis=1)
    solid = row_ink180 > 0.9 * W
    solid_b = R._run_groups(solid, min_len=5)
    solid_cand = [b for b in solid_b
                  if int(H * 0.3) <= b[0] <= int(H * 0.9) and b[1] - b[0] >= 10]
    s_top = solid_cand[-1][1] + 1 if solid_cand else None

    scored = []
    seen = set()
    for thr in (180, 170, 160, 150):
        row_ink_s = (g < thr).sum(axis=1)
        text = (row_ink_s > 0.2 * W) & (row_ink_s < 0.9 * W) & (E > 1800)
        bands = R._run_groups(text, min_len=4)
        pools = [bands]
        if s_top is not None:
            above = [b for b in bands if b[1] < s_top - 1]
            if len(above) >= 2:
                pools = [above]
        for pool in pools:
            for i, t1 in enumerate(pool[:-1]):
                h1 = t1[1] - t1[0] + 1
                if h1 < 4:
                    continue
                s1 = _line_shape(g, t1, thr, W)
                if s1 is None:
                    continue
                t2 = pool[i + 1]
                h2 = t2[1] - t2[0] + 1
                gap = t2[0] - t1[1] - 1
                s2 = _line_shape(g, t2, thr, W)
                if s2 is None:
                    continue
                if not (0.25 <= h2 / h1 <= 2.5 and 0 <= gap <= max(8, int(3.0 * h1))):
                    continue
                key = (t1, t2)
                if key in seen:
                    continue
                seen.add(key)
                scored.append((-(s1[1] + s2[1]), t1, t2))
        if scored:
            break
    scored.sort(key=lambda item: item[0])
    return [(t1, t2) for _, t1, t2 in scored[:limit]]


def find_row_candidates(rgb: np.ndarray):
    """Return candidate row pairs, best guess first.

    The rectified page is mined for MRZ-like row pairs (stroke count);
    the single pair restore picks is kept as the last fallback.
    """
    gray_src = np.asarray(Image.fromarray(rgb).convert("L"))
    h, w = gray_src.shape[:2]
    nominal = min(100, max(10, int(round(w * 0.08))))
    xpad = _adaptive_xpad(gray_src, nominal)
    out = []
    try:
        import restore as R

        work, scale = R.load(rgb)
        g = R._gray(work)
        for idx, (t1, t2) in enumerate(_mrz_candidate_pairs(g)):
            rows = []
            first_bottom = None
            for k, (y0, y1) in enumerate((t1, t2)):
                top = int(y0 / scale)
                bot = min(h, int(y1 / scale) + 1)
                row_h = max(4, bot - top)
                if k == 1 and first_bottom is not None:
                    top = max(top, first_bottom)
                first_bottom = bot
                pad = max(8, min(25, int(row_h * 0.70)))
                top = max(0, top - pad)
                bot = min(h, bot + pad)
                if xpad:
                    row = gray_src[top:bot, xpad:max(xpad + 1, w - xpad)]
                else:
                    row = gray_src[top:bot]
                if row.size:
                    rows.append(row)
            if len(rows) == 2:
                out.append((rows, f"enum{idx}"))
    except Exception:
        pass
    rows, source = find_rows(rgb)
    if len(rows) == 2:
        out.append((rows, source))
    return out



def trim_frame(gray: np.ndarray, min_dark: int = 8) -> np.ndarray:
    """Drop the black camera-frame columns that surround some MRZ rows.

    A photo taken through a scanner keeps a black border left and right of
    the passport.  The border is dark over the full row height, while a
    '<' filler only darkens a few rows, so a column-mean test separates
    them cleanly.
    """
    if gray is None or gray.size == 0:
        return gray
    col_mean = gray.mean(axis=0)
    dark = col_mean < 70
    left = 0
    while left < len(dark) and dark[left]:
        left += 1
    right = len(dark) - 1
    while right > left and dark[right]:
        right -= 1
    if left < min_dark and (len(dark) - 1 - right) < min_dark:
        return gray
    return gray[:, left:right + 1]


def _read_row(rec, row, width_factor):
    try:
        text, score = rec.predict(row, width_factor)
    except Exception:
        return "", 0.0
    return text, score


def _eval_line1(raw, score):
    try:
        line, score = P._line1_eval(raw, score)
    except Exception:
        return "", score
    if len(line) > 44:
        line = line[:44]
    return line, score


def _eval_line2(raw):
    try:
        return P._line2_eval(raw)
    except Exception:
        return "", 0, False


def _country_align(line1, line2):
    """Make L1[2:5] and L2[10:13] agree when one of them can be replaced."""
    if not line1 or not line2 or len(line1) < 5 or len(line2) < 13:
        return line1, line2
    c1, c2 = line1[2:5], line2[10:13]
    if c1 == c2:
        return line1, line2
    if c1.isalpha() and len(c2) == 3:
        patched = line2[:10] + c1 + line2[13:]
        fixed, n, _, ok = P.repair_line2(patched)
        if ok and fixed == patched:
            return line1, patched
    if c2.isalpha() and len(c1) == 3:
        return line1[:2] + c2 + line1[5:], line2
    return line1, line2


def _plausibility(line1, line2, score1, score2, ok2):
    """How much does this reading look like a real TD3 MRZ pair?"""
    score = 0.0
    if len(line1) == 44:
        score += 2.0
    if line1 and line1[0] in "PIVACDRQ":
        score += 2.0
    if len(line2) == 44:
        score += 2.0
    if ok2:
        score += 4.0
    if line1 and line2 and len(line1) >= 5 and len(line2) >= 13:
        if line1[2:5] == line2[10:13]:
            score += 1.5
    icao = sum(1 for ch in (line1 + line2)
               if ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<")
    if icao:
        score += 2.0 * icao / max(1, len(line1) + len(line2))
    score += 0.5 * score1 + 0.5 * score2
    return score


def decode_image(img_path, recs, threads=2, wf1=None, wf2=None, cascade=False):
    """Run the full pipeline on one image and return the KV payload."""
    t0 = time.perf_counter()
    import cv2

    bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(str(img_path))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    candidates = find_row_candidates(rgb)
    t_struct = (time.perf_counter() - t0) * 1000.0
    if not candidates:
        ms = (time.perf_counter() - t0) * 1000.0
        return {"ok": False, "line1": "", "line2": "", "conf1": 0.0, "conf2": 0.0,
                "band": "", "ms": ms, "note": "no-mrz-rows",
                "row_source": "none", "ms_struct": t_struct}

    fast = recs[0]

    def evaluate(rows, wfs1, wfs2):
        hypotheses = []
        for row, wfs in ((rows[0], wfs1), (rows[1], wfs2)):
            hypotheses.append([_read_row(fast, row, wf) for wf in wfs])

        cands = []
        for text, score in hypotheses[0]:
            line, _ = _eval_line1(text, score)
            if line:
                cands.append((line, score))
        line1 = ""
        score1 = 0.0
        if cands:
            merged = P._merge_line1(cands)
            line1 = merged[:44] if merged else ""
            score1 = max(c[1] for c in cands)

        best = None
        for text, score in hypotheses[1]:
            line, nfix, ok = _eval_line2(text)
            if not line:
                continue
            key = (1 if ok else 0, score, -nfix)
            if best is None or key > best[0]:
                best = (key, line, score, nfix, ok)
        if best is None:
            line2, score2, nfix, ok2 = "", 0.0, 0, False
        else:
            _, line2, score2, nfix, ok2 = best

        if cascade and len(recs) > 1:
            ref = recs[1]
            if score1 < L1_MIN_SCORE or not line1:
                for wf in (wf1 or WF_ROW1):
                    text, score = _read_row(ref, rows[0], wf)
                    line, _ = _eval_line1(text, score)
                    if line and score > score1:
                        line1, score1 = line, score
            if score2 < L2_MIN_SCORE or not ok2:
                for wf in (wf2 or WF_ROW2):
                    text, score = _read_row(ref, rows[1], wf)
                    line, nf, ok = _eval_line2(text)
                    if line and (1 if ok else 0, score, -nf) > ((1 if ok2 else 0), score2, -nfix):
                        line2, score2, nfix, ok2 = line, score, nf, ok

        line1, line2 = _country_align(line1, line2)
        return line1, line2, score1, score2, nfix, ok2

    base1 = (wf1,) if wf1 and not isinstance(wf1, (tuple, list)) else (wf1 or WF_ROW1)
    base2 = (wf2,) if wf2 and not isinstance(wf2, (tuple, list)) else (wf2 or WF_ROW2)
    ext1 = (wf1,) if wf1 and not isinstance(wf1, (tuple, list)) else (wf1 or WF_ROW1_EXT)
    ext2 = (wf2,) if wf2 and not isinstance(wf2, (tuple, list)) else (wf2 or WF_ROW2_EXT)

    best = None
    source = "none"
    for idx, (rows, label) in enumerate(candidates):
        line1, line2, score1, score2, nfix, ok2 = evaluate(rows, base1, base2)
        plaus = _plausibility(line1, line2, score1, score2, ok2)
        if not (ok2 and line1 and line1[0] in "PIVACDRQ" and len(line1) == 44):
            l1b, l2b, s1b, s2b, nfb, ok2b = evaluate(rows, ext1, ext2)
            plausb = _plausibility(l1b, l2b, s1b, s2b, ok2b)
            if plausb > plaus:
                line1, line2, score1, score2, nfix, ok2 = l1b, l2b, s1b, s2b, nfb, ok2b
                plaus = plausb
        if best is None or plaus > best[0]:
            best = (plaus, line1, line2, score1, score2, nfix, ok2)
            source = label
        if ok2 and line1 and line1[0] in "PIVACDRQ" and len(line1) == 44:
            break
    plaus, line1, line2, score1, score2, nfix, ok2 = best

    t_rec = (time.perf_counter() - t0) * 1000.0 - t_struct
    ok = bool(line1) and bool(line2)
    ms = (time.perf_counter() - t0) * 1000.0
    return {"ok": ok, "line1": line1, "line2": line2,
            "conf1": round(score1 * 100, 1), "conf2": round(score2 * 100, 1),
            "band": "", "ms": ms,
            "note": f"src={source} l2fix={nfix} l2ok={int(ok2)}",
            "row_source": source, "ms_struct": round(t_struct, 1),
            "ms_rec": round(t_rec, 1)}


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def emit(result):
    out = [
        "result.backend  : paddle_opt",
        f"result.ok       : {'OK' if result['ok'] else 'unreadable'}",
        f"result.line1    : {result['line1']}",
        f"result.line2    : {result['line2']}",
        f"result.conf1    : {int(round(result['conf1']))}",
        f"result.conf2    : {int(round(result['conf2']))}",
    ]
    if result.get("band"):
        out.append(f"band.x band.y band.w band.h : {result['band']}")
    if result.get("note"):
        out.append(f"result.note     : {result['note']}")
    out.append(f"ms: {result['ms']:.1f}")
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("image", nargs="?")
    ap.add_argument("--batch")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--wf1", default="")
    ap.add_argument("--wf2", default="")
    ap.add_argument("--model", default="v6", choices=["v6", "v4m", "cascade"])
    ap.add_argument("--dump", action="store_true")
    args = ap.parse_args(argv)

    def _wfs(text, default):
        if not text:
            return default
        return tuple(float(x) for x in text.split(",") if x.strip())

    wf1 = _wfs(args.wf1, WF_ROW1)
    wf2 = _wfs(args.wf2, WF_ROW2)
    if args.model == "v6":
        recs = [r for r in [load_v6()] if r is not None]
    elif args.model == "v4m":
        recs = [r for r in [load_v4m()] if r is not None]
    else:
        recs = [r for r in (load_v4m(), load_v6()) if r is not None]
    if not recs:
        print("no recognition model available", file=sys.stderr)
        return 2
    cascade = args.model == "cascade" and len(recs) > 1
    if args.image:
        print(emit(decode_image(args.image, recs, wf1=wf1, wf2=wf2, cascade=cascade)))
        return 0
    if args.batch:
        for line in open(args.batch, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                print(emit(decode_image(line, recs, wf1=wf1, wf2=wf2, cascade=cascade)))
            except Exception as exc:  # keep the batch alive
                print(f"result.ok       : error\nresult.note     : {exc}")
            print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
