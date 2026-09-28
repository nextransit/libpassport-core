#!/usr/bin/env python3
"""paddle_local -- Pure-local MRZ decoder compatible with paddle_ocr_tool's
stdout contract. No paddleocr dependency.

Implementation:
  1. Locate the MRZ band via row-projection (_find_mrz_band)
  2. Crop to that band and 3x upscale (gives comfortable OCR-B glyphs)
  3. Run tesseract --psm 6 -l ocrb on the band; it auto-detects the
     two MRZ lines and returns them in order (PSM 11 fallback when it
     only found one line)
  4. Normalise to the 44-char ICAO whitelist
  5. TD3 slot enforcement (mrz_slots): Line 1 is alpha-only (stray
     digits become letters); Line 2 digit slots become digits, the
     alpha nationality field loses stray digits
  6. Mod-10 checksum repair of Line 2 (mrz_checksum.repair_line2):
      43/45-char reads get a checksum-guided insertion/deletion first,
      then the four checked fields are repaired within an edit budget
  7. Per-line confidence is a *structure-plausibility* score, not an
     engine confidence: tesseract's OCR-B CLI reports 0.0 word
     confidences for this model/whitelist, so we score how well the
     raw text respects the TD3 slots (line 1) and how many of the five
     TD3 checks pass after repair (line 2, minus a small per-edit
     penalty)

TD3 layout (this is the standard passport MRZ; earlier revisions of
this file carried a TD1 layout with Line-1 digit slots at 13..43,
which corrupted every name it touched):

    line1  [0] doc type alpha      [1] '<'          [2..4] issuing alpha
           [5..43] name alpha
    line2  [0..8] passport_no alnum    [9] cd
           [10..12] nationality alpha  [13..18] dob   [19] cd
           [20] sex M/F/<              [21..26] expiry [27] cd
           [28..41] personal_no alnum  [42] cd        [43] composite cd

This is a *stand-in* decoder for environments where paddleocr isn't
installed; for production accuracy install paddleocr (preferred) or
use tesseract_tool.py directly. The CLI contract emitted is identical.
Set MRZ_REPAIR=0 to disable the checksum repair pass (A/B testing).
"""
from __future__ import annotations
import os
import subprocess
import sys
import time
from pathlib import Path

from mrz_slots import (
    ALPHA_L1, ALPHA_L2, DIGIT_L1, DIGIT_L2, DOC_TYPES, ICAO, LINE_LEN,
    enforce_slots,
)
from mrz_checksum import compute_check_digit, repair_line2

_REPAIR = os.environ.get("MRZ_REPAIR", "1") != "0"

_WHITELIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<"


def _find_mrz_band(gray):
    """Find y0, y1 of the MRZ band.

    Two operating modes — auto-detected from the ink distribution:
      - Pre-cropped strip: >70% of rows are ink rows; return the full
        image (with a 4-px safety inset) so the inter-line gutter is
        included in the band.
      - Full passport page: only the bottom 1/3 has ink; use the
        bottom-1/3 ink-group heuristic to find the two-line band.

    The pre-cropped path was previously broken: the bottom-1/3 logic
    returned only the bottom ink group, missing line1 and the gutter.
    """
    import numpy as np
    H, W = gray.shape
    if H == 0 or W == 0:
        return 0, 0
    # Robust pre-cropped strip signature: an MRZ strip is a wide, short
    # image (W/H > 3, H < 400) regardless of ink coverage. The old
    # ink-coverage test (>70% ink rows) failed on short-name samples
    # whose line-1 is mostly '<' fillers (e.g. img_1237_x49_v0), which
    # then fell into the bottom-1/3 branch and lost most of line 1.
    if W > 3 * H and H < 400:
        return 0, H
    rd = (gray < 128).sum(axis=1)
    ink_rows = np.where(rd > W * 0.10)[0]
    # Pre-cropped strip signature: most of the image is ink text.
    if ink_rows.size > 0.7 * H:
        first = max(0, int(ink_rows[0]) - 4)
        last = min(H, int(ink_rows[-1]) + 4)
        return first, last
    # Otherwise: bottom-1/3 of a full passport page.
    bottom_start = int(H * 0.55)
    sub = gray[bottom_start:, :]
    rd = (sub < 128).sum(axis=1)
    rows = np.where(rd > W * 0.10)[0]
    if rows.size < 4:
        return int(H * 0.80), H
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
                  for g in groups if g[1] - g[0] >= 6]
    if len(abs_groups) >= 2:
        best, best_gap = None, 10**9
        for i in range(len(abs_groups) - 1, 0, -1):
            top, bot = abs_groups[i - 1], abs_groups[i]
            gap = bot[0] - top[1]
            if gap <= 30 and gap < best_gap:
                best = (top[0], bot[1])
                best_gap = gap
        if best:
            return best
    if abs_groups:
        return abs_groups[-1][0], abs_groups[-1][1]
    return int(H * 0.80), H


def _tesseract_run(in_path, tmpdir, psm, tag):
    """Run tesseract OCR-B on `in_path`; return (text, rc).

    The CLI's per-word confidences are unusable with this ocrb model
    (TSV and hOCR report 0.0 for every word), so the caller scores
    structure plausibility instead (see _slot_score / _checks_score).
    """
    base = Path(tmpdir) / tag
    r = subprocess.run(
        ["tesseract", str(in_path), str(base),
         "--psm", str(psm), "-l", "ocrb",
         "-c", f"tessedit_char_whitelist={_WHITELIST}"],
        capture_output=True, timeout=20,
    )
    txt_path = Path(str(base) + ".txt")
    text = txt_path.read_text() if txt_path.exists() else ""
    return text, r.returncode


def _tesseract_band(in_path, tmpdir):
    """PSM 6 pass over the band (auto two-row split)."""
    return _tesseract_run(in_path, tmpdir, 6, "tess_out")


def _work_dir(img_path):
    """Repo-local scratch dir (the sandbox hides /tmp from tesseract)."""
    argv_path = Path(img_path).resolve()
    candidate = argv_path.parent
    found = None
    while candidate != candidate.parent:
        if (candidate / "tests" / "ocr_bench_runner.py").exists():
            found = candidate
            break
        candidate = candidate.parent
    tmpdir = (found / "tests" / ".paddle_local_tmp") if found \
        else Path.cwd() / ".paddle_local_tmp"
    tmpdir.mkdir(parents=True, exist_ok=True)
    return tmpdir


def _clean(s):
    """Uppercase + map non-ICAO chars to '<', preserving positions."""
    return "".join(c if c in ICAO else "<" for c in (s or "").upper())


def _pad(s):
    if len(s) < LINE_LEN:
        s = s + "<" * (LINE_LEN - len(s))
    return s[:LINE_LEN]


def _norm_two_lines(text):
    """Parse tesseract output into two ICAO lines.

    Tesseract PSM 6 returns both MRZ rows; we classify each non-empty
    line by its first char:
      - line1 starts with P / I / V / A / C / D / R + '<'  (doc type)
      - line2 starts with a digit / 'L' / '<' (passport no.)
    If classification is unambiguous, lines are reordered so the
    returned lines always satisfy the ICAO layout. Returns
    (line1, line2, raw1): line1 is padded to 44 and slot-enforced,
    line2 is left un-padded so the caller can checksum-align 43/45-char
    reads before padding, and raw1 is the cleaned pre-enforcement
    line 1 used for the plausibility score.
    """
    raw_lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    raw_lines.sort(key=len, reverse=True)
    a = raw_lines[0] if len(raw_lines) >= 1 else ""
    b = raw_lines[1] if len(raw_lines) >= 2 else ""

    def is_l1(s):
        s = s.upper()
        return bool(s) and s[0] in DOC_TYPES and len(s) > 1 and s[1] == "<"

    def is_l2(s):
        s = s.upper()
        return bool(s) and (s[0].isdigit() or s[0] == "L")

    if is_l1(b) and is_l2(a):
        line1, line2 = b, a
    elif is_l1(a) and is_l2(b):
        line1, line2 = a, b
    elif is_l1(a) and not is_l2(b):
        line1, line2 = a, b
    else:
        line1, line2 = a, b

    line1 = _clean(line1)
    line2 = _clean(line2)
    raw1 = line1
    line1 = enforce_slots(_pad(line1), ALPHA_L1, DIGIT_L1)
    line1 = (line1[0] if line1 else "P") + "<" + line1[2:]
    return line1, line2, raw1


def _repair_line2(line2):
    """Checksum-align + repair a raw Line 2; returns (line, n_fixed)."""
    if _REPAIR:
        line, n, _rep, ok = repair_line2(line2)
        if ok:
            return enforce_slots(line, ALPHA_L2, set()), n
    padded = enforce_slots(_pad(line2), set(), DIGIT_L2)
    if _REPAIR:
        line, n, _rep, _ok = repair_line2(padded)
        return enforce_slots(line, ALPHA_L2, set()), n
    return enforce_slots(padded, ALPHA_L2, set()), 0


def _slot_score(line, alpha_set, digit_set):
    """Plausibility 0..100: share of real chars in valid TD3 slots."""
    real = [(i, c) for i, c in enumerate(line) if c not in "<"]
    if not real:
        return 0
    ok = 0
    for i, c in real:
        if i in digit_set:
            ok += c.isdigit()
        elif i in alpha_set:
            ok += c.isalpha()
        else:
            ok += 1
    return int(round(100.0 * ok / len(real)))


def _checks_score(line2, n_fixed):
    """Plausibility 0..100 from the five TD3 checks on a 44-char line 2."""
    if len(line2) != LINE_LEN:
        return 0
    n = 0
    for start, end, cd in ((0, 9, 9), (13, 19, 19), (21, 27, 27),
                           (28, 42, 42)):
        n += compute_check_digit(line2[start:end]) == line2[cd]
    body = line2[0:10] + line2[13:20] + line2[21:28] + line2[28:43]
    n += compute_check_digit(body) == line2[43]
    return max(0, int(round(100.0 * n / 5)) - 3 * n_fixed)


def decode_image(img_path):
    """Decode one image; shared by main() and paddle_ocr_tool.

    Returns a dict: ok, line1, line2, conf1, conf2, band_box, err.
    conf1/conf2 are structure-plausibility scores (0..100), not engine
    confidences (see the module docstring, point 7).
    """
    try:
        from PIL import Image
        import numpy as np
        img = Image.open(img_path).convert("RGB")
        gray = np.asarray(img.convert("L"))
    except Exception as e:
        return {"ok": False, "line1": "", "line2": "", "conf1": 0,
                "conf2": 0, "band_box": "", "err": f"cannot load image: {e}"}

    H, W = gray.shape
    y0, y1 = _find_mrz_band(gray)
    if y1 <= y0:
        y0, y1 = 0, H
    crop = img.crop((0, y0, W, y1)).resize(
        (W * 3, max(1, y1 - y0) * 3), Image.LANCZOS)
    tmpdir = _work_dir(img_path)
    band_path = tmpdir / "band.png"
    crop.save(band_path)

    text, rc = _tesseract_band(band_path, tmpdir)
    err = ""
    if rc != 0:
        err = f"tesseract failed (rc={rc})"
    elif len([ln for ln in text.splitlines() if ln.strip()]) < 2:
        # PSM 6 dropped the second row: retry with PSM 11 (sparse text),
        # which preserves both rows for clean strips.
        t11, rc11 = _tesseract_run(band_path, tmpdir, 11, "tess_out_psm11")
        if rc11 == 0:
            n6 = len([ln for ln in text.splitlines() if ln.strip()])
            n11 = len([ln for ln in t11.splitlines() if ln.strip()])
            if n11 > n6:
                text = t11

    line1, line2, raw1 = _norm_two_lines(text)
    line2, n_fixed = _repair_line2(line2)

    conf1 = _slot_score(raw1, ALPHA_L1, DIGIT_L1)
    conf2 = _checks_score(line2, n_fixed)
    l1_real = sum(1 for c in line1 if c not in "<")
    l2_real = sum(1 for c in line2 if c not in "<")
    ok = (line1[0] in DOC_TYPES and line1[1] == "<"
          and l1_real >= 5 and l2_real >= 5)
    return {"ok": ok, "line1": line1, "line2": line2,
            "conf1": conf1, "conf2": conf2,
            "band_box": f"0 {y0} {W} {y1 - y0}", "err": err}


def main(argv):
    if len(argv) < 2:
        print("usage: paddle_local.py <image>", file=sys.stderr)
        return 2
    img_path = Path(argv[1])
    if not img_path.exists():
        print(f"cannot open {img_path}", file=sys.stderr)
        return 2

    t0 = time.perf_counter()
    d = decode_image(img_path)
    ms = (time.perf_counter() - t0) * 1000.0
    print(f"result.ok       : {'OK' if d['ok'] else 'unreadable'}")
    print(f"result.line1    : {d['line1']}")
    print(f"result.line2    : {d['line2']}")
    print(f"result.conf1    : {d['conf1']}")
    print(f"result.conf2    : {d['conf2']}")
    print(f"band.x band.y band.w band.h : {d['band_box']}")
    print(f"ms: {ms:.1f}")
    if d["err"]:
        print(f"paddle_local: {d['err']}", file=sys.stderr)
        return 1
    return 0 if d["ok"] else 1


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except Exception as e:
        print(f"paddle_local: {e}", file=sys.stderr)
        sys.exit(1)
