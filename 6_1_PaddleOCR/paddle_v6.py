#!/usr/bin/env python3
"""paddle_v6 -- Paddle pure-rec MRZ decoder (Phase 3+).

Pipeline: adaptive MRZ band -> mid-gutter row split -> per-row LANCZOS
upscale (3x, plus 2x/4x when the checksums disagree) -> PP-OCRv6 medium
*recognition only* -> ICAO filter -> TD3 slot enforcement -> Mod-10
checksum repair -> cross-scale per-position vote for Line 1.

Why recognition-only: the dense MRZ strip must NOT go through a text
detector (det merges/drops lines). Instead we project-split the two
rows, upscale each, and run PP-OCRv6 medium rec on each single row.
The v5-server rec is used only when the v6 ONNX model is unavailable.

Error recovery (two independent layers):
  * mrz_slots.enforce_slots  -- slot-domain swaps along the OCR-B
    confusion classes (0<->O, 5<->S, 1<->I ...). Line 1 is alpha-only,
    so digits in names become letters; Line 2 digit slots become
    digits, and the alpha nationality field loses stray digits.
  * mrz_checksum.repair_line2 -- Mod-10 search over the four checked
    fields (single/double substitutions + "the data is right, the
    check glyph is misread"), with 43/45-char lines aligned by a
    checksum-guided insertion/deletion first. A candidate is only
    preferred when it turns Line 2 fully check-valid.

Mod-10 cannot see confusions whose ICAO values differ by a multiple of
10 (L<->1, 6<->G, U<->0, T<->1); those stay for the confusion classes
and the multi-scale vote. Env switches for A/B: MRZ_TTA=0 disables the
multi-scale pass, MRZ_REPAIR=0 disables checksum repair.

The stdout contract is identical to paddle_ocr_tool / paddle_local
so the GUI's ocr_bench_runner can pick it without changes:

    result.ok      : OK | unreadable | <error>
    result.line1   : 44-char line
    result.line2   : 44-char line
    result.conf1   : 0..100
    result.conf2   : 0..100
    band.x ...     : "x y w h"

If the cached ONNX model or onnxruntime is missing, we fall back to
paddle_local (tesseract ocrb baseline) so the tool keeps working everywhere.

Measured on 6_mrz_ocr/data/corpus_eval with PP-OCRv6 medium rec:
  * stratified n=200 (100 clean + 100 realistic): L1 88.82%
    (exact 48/200), L2 94.70% (exact 78/200),
    chk4 = chk5 = 200/200, ~132 ms/img.
  * full corpus n=1380: L1 84.91% (exact 326), L2 89.29% (exact 543),
    chk4 = chk5 = 1335/1380 (96.7%), ~190 ms/img.
The same n=200 sample scores L1 87.55% / L2 92.18% with only 31/200
check-valid lines before the slot + checksum layers, so the repair
pass is what makes the output usable downstream.
"""
from __future__ import annotations
import math
import os
import sys
import time
from pathlib import Path
import numpy as np
from PIL import Image

from mrz_slots import (
    ALPHA_L1, ALPHA_L2, DIGIT_L1, DIGIT_L2, DOC_TYPES, ICAO, LINE_LEN,
    enforce_slots,
)
from mrz_checksum import repair_line2

_TTA = os.environ.get("MRZ_TTA", "1") != "0"
_REPAIR = os.environ.get("MRZ_REPAIR", "1") != "0"

# Prefer the locally cached/registerable ONNX model name. Older PaddleX
# builds expose the same files under the name without the `_onnx` suffix.
_MODEL_NAMES = tuple(
    name for name in (
        os.environ.get("MRZ_PADDLE_MODEL", "").strip(),
        "PP-OCRv6_medium_rec_onnx",
        "PP-OCRv6_medium_rec",
        "PP-OCRv5_server_rec",
        "PP-OCRv5_mobile_rec",
        "en_PP-OCRv5_mobile_rec",
    ) if name
)
_HOME_MODELS = Path.home() / ".paddlex" / "official_models"


def _find_model_dir(model_name: str) -> tuple[str, Path]:
    """Return PaddleX's registered name and its cached directory.

    PaddleX may cache an ONNX model under a ``*_onnx`` directory while the
    directory metadata keeps the base registered name.  Passing the cache
    directory name as ``model_name`` makes PaddleX reject the model.
    """
    registered_name = model_name.removesuffix("_onnx")
    candidates = (model_name, registered_name)
    if not model_name.endswith("_onnx"):
        candidates = (model_name + "_onnx", model_name)
    for candidate_name in candidates:
        cand = _HOME_MODELS / candidate_name
        if cand.is_dir():
            return registered_name, cand
    raise FileNotFoundError(f"model {model_name} not cached in {_HOME_MODELS}")


def _find_mrz_band(gray):
    """Reuse paddle_local's adaptive band finder."""
    from paddle_local import _find_mrz_band as _fn
    return _fn(gray)


def _split_rows(gray, y0=None, y1=None):
    """Project-split band into (row1, row2) at the deepest mid gutter."""
    if y0 is None or y1 is None:
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


def _filter(s: str) -> str:
    """Uppercase + ICAO whitelist, no padding."""
    s = (s or "").strip().upper()
    return "".join(c for c in s if c in ICAO)


def _pad(s: str) -> str:
    """Fit a filtered line to exactly LINE_LEN chars with '<' padding."""
    if len(s) < LINE_LEN:
        s = s + "<" * (LINE_LEN - len(s))
    return s[:LINE_LEN]


def _rec_row(rec, row, scale=3):
    im = Image.fromarray(row).resize(
        (row.shape[1] * scale, row.shape[0] * scale), Image.LANCZOS
    )
    out = list(rec.predict(np.asarray(im.convert("RGB"))))
    if not out:
        return "", 0.0, ""
    o = out[0]
    txt = (o.get("rec_text") or "") if isinstance(o, dict) else ""
    sc = float(o.get("rec_score") or 0.0) if isinstance(o, dict) else 0.0
    return txt, sc, txt


def _line1_eval(raw: str, score: float):
    """Filter + slot-enforce a Line 1 candidate (alpha-only line)."""
    return enforce_slots(_pad(_filter(raw)), ALPHA_L1, DIGIT_L1), score


def _line2_eval(raw: str):
    """Filter + slot/checksum-repair a Line 2 candidate.

    Returns (line44, n_fixed, fully_valid). Repair sees the raw length
    first (43/45-char lines get a checksum-guided alignment), then the
    padded line with digit-slot letters pre-swapped. Nationality digits
    are cleaned to letters only after success, so a valid line is never
    broken by the slot pass.
    """
    raw = _filter(raw)
    if _REPAIR:
        line, n, _rep, ok = repair_line2(raw)
        if ok:
            return enforce_slots(line, ALPHA_L2, set()), n, True
    padded = enforce_slots(_pad(raw), set(), DIGIT_L2)
    if _REPAIR:
        line, n, _rep, ok = repair_line2(padded)
        return enforce_slots(line, ALPHA_L2, set()), n, ok
    return padded, 0, False


def _merge_line1(cands):
    """Score-weighted per-position vote over 44-char Line 1 candidates."""
    if not cands:
        return "<" * LINE_LEN
    if len({t for t, _ in cands}) == 1:
        return cands[0][0]
    out = []
    for i in range(LINE_LEN):
        votes = {}
        for t, sc in cands:
            w = sc if sc > 0.0 else 0.01
            votes[t[i]] = votes.get(t[i], 0.0) + w
        out.append(max(votes, key=votes.get))
    return "".join(out)


_OCR = None
_OCR_ATTEMPTED = False
_OCR_ERROR = ""


class _OnnxRec:
    """Small PP-OCR recognition adapter for cached ONNX models.

    PaddleX 3.3.x does not register PP-OCRv6 recognition yet, although the
    downloaded model and its ONNX runtime files are valid.  Loading the
    model directly keeps the recognition-only path independent of that
    registry.
    """

    def __init__(self, model_dir: Path):
        import cv2
        import onnxruntime as ort
        import yaml

        config_path = model_dir / "inference.yml"
        model_path = model_dir / "inference.onnx"
        config = yaml.safe_load(config_path.read_text())
        shape = next(
            op["RecResizeImg"]["image_shape"]
            for op in config["PreProcess"]["transform_ops"]
            if "RecResizeImg" in op
        )
        chars = config["PostProcess"]["character_dict"]
        self._cv2 = cv2
        self._height = int(shape[1])
        self._max_width = 3200
        self._characters = ["blank"] + list(chars)
        self._session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"])
        self._input_name = self._session.get_inputs()[0].name

    def _prepare(self, image):
        image = np.asarray(image)
        if image.ndim == 2:
            image = np.repeat(image[:, :, None], 3, axis=2)
        if image.ndim != 3 or image.shape[2] < 3:
            raise ValueError(f"invalid recognition image shape: {image.shape}")
        image = image[:, :, :3]
        h, w = image.shape[:2]
        if h <= 0 or w <= 0:
            raise ValueError("empty recognition image")
        width = min(self._max_width, max(
            1, int(math.ceil(self._height * w / float(h)))))
        image = self._cv2.resize(
            image, (width, self._height),
            interpolation=self._cv2.INTER_LINEAR,
        )
        image = image.astype("float32").transpose((2, 0, 1)) / 255.0
        image = (image - 0.5) / 0.5
        return image[None, ...].astype("float32")

    def _decode(self, output):
        output = np.asarray(output)
        if output.ndim == 3:
            output = output[0]
        indices = output.argmax(axis=-1)
        probabilities = output.max(axis=-1)
        selected = np.ones(len(indices), dtype=bool)
        if len(indices) > 1:
            selected[1:] = indices[1:] != indices[:-1]
        selected &= indices != 0
        text = "".join(
            self._characters[int(index)]
            for index in indices[selected]
            if int(index) < len(self._characters)
        )
        score = float(probabilities[selected].mean()) \
            if selected.any() else 0.0
        return text, score

    def predict(self, image):
        batch = image if isinstance(image, (list, tuple)) else [image]
        inputs = np.concatenate([self._prepare(item) for item in batch], axis=0)
        output = self._session.run(None, {self._input_name: inputs})[0]
        return [
            {"rec_text": text, "rec_score": score}
            for text, score in (self._decode(item) for item in output)
        ]


def _get_rec():
    global _OCR, _OCR_ATTEMPTED, _OCR_ERROR
    if _OCR is not None:
        return _OCR, "paddle-onnxruntime"
    if _OCR_ATTEMPTED:
        return None, "fallback"
    _OCR_ATTEMPTED = True

    missing = []
    selected = None
    for model_name in _MODEL_NAMES:
        try:
            registered_name, model_dir = _find_model_dir(model_name)
        except FileNotFoundError:
            missing.append(model_name)
            continue
        if (model_dir / "inference.onnx").is_file():
            selected = (registered_name, model_dir)
            break

    if selected is None:
        _OCR_ERROR = (
            "no cached ONNX recognition model; tried " +
            ", ".join(missing or _MODEL_NAMES)
        )
        return None, "fallback"

    model_name, model_dir = selected
    try:
        _OCR = _OnnxRec(model_dir)
        return _OCR, "paddle-onnxruntime"
    except Exception as exc:
        _OCR_ERROR = f"{model_name} ONNX init failed: {exc}"
        return None, "fallback"


def _decode_strip_fallback(img_path):
    """Use fast per-row OCR when the Paddle model is unavailable.

    The GUI passes a native-resolution, two-row MRZ strip for external
    backends. PSM 6 on that strip can swap the rows or keep a leading
    camera-frame glyph. Split the two ink runs first, recognize each row
    with PSM 7, then apply the same ICAO/checksum post-processing.
    """
    import shutil
    import subprocess
    import tempfile

    if not shutil.which("tesseract"):
        return None
    try:
        gray = np.asarray(Image.open(img_path).convert("L"))
    except Exception:
        return None
    H, W = gray.shape
    if H < 100 or W <= 3 * H:
        return None

    row_ink = (gray < 128).sum(axis=1)
    rows = np.where(row_ink > W * 0.10)[0]
    if rows.size < 4:
        return None
    groups = []
    start = previous = int(rows[0])
    for y in rows[1:]:
        y = int(y)
        if y - previous <= 4:
            previous = y
        else:
            groups.append((start, previous))
            start = previous = y
    groups.append((start, previous))
    groups = [g for g in groups if g[1] - g[0] >= 6]
    if len(groups) < 2:
        return None
    first, second = groups[-2:]
    pad = max(15, min(30, int(round(
        max(1, second[0] - first[1]) * 0.38))))
    crops = [
        gray[max(0, first[0] - pad):min(H, first[1] + pad + 1)],
        gray[max(0, second[0] - pad):min(H, second[1] + pad + 1)],
    ]

    texts = []
    with tempfile.TemporaryDirectory(prefix="_paddle_fallback_") as td:
        for idx, row in enumerate(crops):
            image_path = Path(td) / f"line{idx + 1}.png"
            Image.fromarray(row).resize(
                (row.shape[1] * 2, row.shape[0] * 2),
                Image.Resampling.LANCZOS,
            ).save(image_path)
            output_base = Path(td) / f"ocr{idx + 1}"
            proc = subprocess.run(
                [shutil.which("tesseract"), str(image_path),
                 str(output_base), "--psm", "7", "-l", "ocrb",
                 "-c",
                 "tessedit_char_whitelist="
                 "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<"],
                text=True, capture_output=True, timeout=10,
            )
            text_path = Path(str(output_base) + ".txt")
            texts.append(text_path.read_text() if (
                proc.returncode == 0 and text_path.exists()) else "")

    line1_raw = _filter(texts[0])
    line1_start = next(
        (i for i, ch in enumerate(line1_raw) if ch in DOC_TYPES),
        -1,
    )
    if line1_start < 0:
        return None
    line1, _score = _line1_eval(line1_raw[line1_start:line1_start + 44],
                                 0.85)

    line2_raw = _filter(texts[1])
    choices = []
    for start in range(len(line2_raw)):
        candidate = line2_raw[start:start + 44]
        if len(candidate) < 40:
            continue
        line2, n_fixed, valid = _line2_eval(candidate)
        if valid:
            choices.append((n_fixed, start, line2))
    if not choices:
        return None
    _n_fixed, _start, line2 = min(choices)
    ok = (
        line1[0] in DOC_TYPES and
        sum(ch not in "<" for ch in line1) >= 5 and
        sum(ch not in "<" for ch in line2) >= 5
    )
    return {
        "ok": ok,
        "line1": line1,
        "line2": line2,
        "conf1": 85,
        "conf2": 85,
        "band_box": f"0 0 {W} {H}",
        "err": "",
    }


def _decode_one(rec, img_path, init_error=""):
    """Run the V6 pure-rec pipeline on one image.

    Returns (ok, line1, line2, c1, c2, band_box, err, ms_ms). When
    `rec` is None the tesseract ocrb baseline (paddle_local) is used.
    """
    t0 = time.perf_counter()
    if rec is None:
        # Fallback: use the native two-row path when called by the GUI;
        # retain paddle_local for full-page/batch inputs.
        strip = _decode_strip_fallback(img_path)
        if strip is not None:
            d = strip
        else:
            from paddle_local import decode_image
            d = decode_image(str(img_path))
        ms = (time.perf_counter() - t0) * 1000.0
        return (bool(d.get("ok")), d.get("line1", ""), d.get("line2", ""),
                int(d.get("conf1", 0) or 0), int(d.get("conf2", 0) or 0),
                d.get("band_box", ""),
                "; ".join(x for x in (init_error, d.get("err", "")) if x),
                ms)
    try:
        gray = np.asarray(Image.open(img_path).convert("L"))
    except Exception as e:
        ms = (time.perf_counter() - t0) * 1000.0
        return False, "", "", 0, 0, "", f"cannot load image: {e}", ms

    H, W = gray.shape
    y0, y1 = _find_mrz_band(gray)
    r1, r2 = _split_rows(gray, y0, y1)

    scales = [3, 2] if _TTA else [3]
    l1_cands = []
    best2 = None
    best2_key = None
    got_valid = False
    idx = 0
    while idx < len(scales):
        sc = scales[idx]
        idx += 1
        t1, s1, _ = _rec_row(rec, r1, sc)
        t2, s2, _ = _rec_row(rec, r2, sc)
        l1_cands.append(_line1_eval(t1, s1))
        l2, nfix, valid = _line2_eval(t2)
        key = (valid, s2, -nfix)
        if best2_key is None or key > best2_key:
            best2_key, best2 = key, (l2, nfix, valid, s2)
        # A high-confidence checksum-valid result does not benefit from
        # another full inference pass. Keep TTA for low-confidence repairs
        # and checksum failures, where a second scale can still help.
        if valid and (nfix <= 2 or s2 >= 0.85):
            got_valid = True
            break
        # Hard case: try one finer scale before giving up.
        if (_TTA and not got_valid and idx == len(scales)
                and scales[-1] == 2):
            scales.append(4)

    line1 = _merge_line1(l1_cands)
    line2, _nfix, _valid, s2sel = best2
    l1_real = sum(1 for c in line1 if c not in "<")
    l2_real = sum(1 for c in line2 if c not in "<")
    ok = (line1[0] in DOC_TYPES
          and l1_real >= 5 and l2_real >= 5 and _valid)
    c1 = int(min(100, max((s for _t, s in l1_cands), default=0.0) * 100))
    c2 = int(min(100, s2sel * 100))
    band_box = f"0 {y0} {W} {y1 - y0}"
    ms = (time.perf_counter() - t0) * 1000.0
    return ok, line1, line2, c1, c2, band_box, "", ms


def _emit(ok, line1, line2, c1, c2, band_box, ms=None,
          backend="", note=""):
    if backend:
        print(f"result.backend  : {backend}")
    print(f"result.ok       : {'OK' if ok else 'unreadable'}")
    print(f"result.line1    : {line1}")
    print(f"result.line2    : {line2}")
    print(f"result.conf1    : {c1}")
    print(f"result.conf2    : {c2}")
    print(f"band.x band.y band.w band.h : {band_box}")
    if note:
        print(f"result.note     : {note}")
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
            ok, line1, line2, c1, c2, band, err, ms = _decode_one(
                rec, ip, _OCR_ERROR if mode == "fallback" else "")
            _emit(
                ok, line1, line2, c1, c2, band, ms,
                backend=mode,
                note=_OCR_ERROR if mode == "fallback" else "",
            )
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

    rec, mode = _get_rec()
    ok, line1, line2, c1, c2, band, err, _ms = _decode_one(
        rec, img_path, _OCR_ERROR if mode == "fallback" else "")
    _emit(
        ok, line1, line2, c1, c2, band,
        backend=mode,
        note=_OCR_ERROR if mode == "fallback" else "",
    )
    return 0 if ok else 1


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except Exception as e:
        print(f"paddle_v6: {e}", file=sys.stderr)
        sys.exit(1)
