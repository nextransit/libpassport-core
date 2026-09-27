#!/usr/bin/env python3
"""paddle_local -- Pure-local MRZ decoder compatible with paddle_ocr_tool's
stdout contract. No paddleocr dependency.

Implementation:
  1. Locate MRZ band via row-projection (bottom-third scan)
  2. 3x upscale (gives ~38px glyph height — comfortable for OCR-B)
  3. Run tesseract --psm 6 -l ocrb on the full band; it auto-detects
     the two MRZ lines and returns them in order
  4. Apply ICAO ambiguity table (O<->0, I<->1, B<->8, S<->5, Z<->2)
     using slot-domain heuristics:
       - document type slot (line1[0])              → P/I/V/A/C/D/R
       - issuing-state slot (line1[2..4])           → alpha-only
       - date slots (line1[13..19] line1[21..27])   → digit-only
       - composite check digits (line1[9], line2[9],
                                line1[19], line2[19],
                                line1[27], line2[27],
                                line1[43], line2[43]) → digit-only
  5. Normalise to 44-char ICAO whitelist and pad/trim
  6. Confidence from tesseract TSV (mean per-symbol)

This is a *stand-in* decoder for environments where paddleocr isn't
installed; for production accuracy install paddleocr (preferred) or
use tesseract_tool.py directly. The CLI contract emitted is identical.
"""
from __future__ import annotations
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

LINE_LEN = 44
DOC_TYPES = set("PIVACDR")

# ICAO ambiguous pairs (master -> candidate swap)
AMBIG = {"O": "0", "0": "O", "I": "1", "1": "I", "B": "8", "8": "B",
         "S": "5", "5": "S", "Z": "2", "2": "Z"}

# ICAO MRZ slot layout — only digit or alpha is enforced here.
# line1: [0] doc_type  [1] filler '<'
#        [2..4]  issuing_state (alpha-3)          [5..13] name
#        [14..19] dob YYMMDD                        [20] filler '<'
#        [21] sex M/F/<                             [22..27] exp YYMMDD
#        [28..42] personal_no
#        [29] check_digit
#        [30..42] composite
#        [43] final_check_digit
# line2: [0..6] passport_no   [7] check_digit
#        [8..13] nationality (alpha-3)
#        [14..19] dob YYMMDD                       [20] check_digit
#        [21..27] expiry YYMMDD                     [28] check_digit
#        [29..41] personal_no                       [42] check_digit
#        [43] composite_check_digit
_LINE1_ALPHA = set(range(2, 5)) | set(range(5, 14)) | {21}      # name + sex + state
_LINE1_DIGIT = {13, 14, 15, 16, 17, 18, 19, 21, 22, 23, 24, 25,
                26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37,
                38, 39, 40, 41, 42, 43}
_LINE2_ALPHA = set(range(8, 14))
_LINE2_DIGIT = {0, 1, 2, 3, 4, 5, 6, 7, 14, 15, 16, 17, 18, 19,
                20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
                32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43}


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


def _tesseract_band(in_path, tmpdir):
    """Run tesseract OCR-B on a single image and return (text, conf).

    Returns the raw multi-line text (joined with '\n') and the mean
    per-symbol confidence (0..100). Uses --psm 6 to let tesseract
    auto-detect the two MRZ lines.
    """
    base = Path(tmpdir) / "tess_out"
    # PSM 6: assume a uniform block of text. Lets tesseract split the
    # 2-line MRZ band into the two rows automatically. Tried 3/4/6/11/12;
    # 6 and 4 gave the best MRZ results.
    r = subprocess.run(
        ["tesseract", str(in_path), str(base),
         "--psm", "6", "-l", "ocrb",
         "-c", "tessedit_char_whitelist="
              "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<"],
        capture_output=True, timeout=20,
    )
    txt_path = Path(str(base) + ".txt")
    text = txt_path.read_text() if txt_path.exists() else ""
    # TSV confidence
    conf = 0
    tsv_path = Path(str(base) + ".tsv")
    if tsv_path.exists():
        confs = []
        for ln in tsv_path.read_text().splitlines()[1:]:
            cols = ln.split("\t")
            if len(cols) >= 12:
                try:
                    c = int(cols[10])
                    if c >= 0:
                        confs.append(c)
                except ValueError:
                    pass
        if confs:
            conf = sum(confs) // len(confs)
    return text, conf, r.returncode


def _enforce_slot(line, alpha_set, digit_set):
    """Map ambiguous O/0 etc. to the canonical ICAO charset for the slot.

    Strategy: for each position, if the slot is digit-only and the OCR
    returned a letter, swap via AMBIG. Conversely for alpha-only slots.
    Other glyphs ('<') are left as-is.
    """
    out = list(line)
    for i, ch in enumerate(out):
        if ch in "<" or not ch:
            continue
        swap = AMBIG.get(ch)
        if swap is None:
            continue
        if i in digit_set and ch.isalpha():
            out[i] = swap   # alpha in digit slot → swap to digit
        elif i in alpha_set and ch.isdigit():
            out[i] = swap   # digit in alpha slot → swap to alpha
    return "".join(out)


def _norm_two_lines(text, conf):
    """Parse tesseract output into two 44-char lines + per-line conf.

    Tesseract PSM 6 returns both MRZ rows; we classify each non-empty
    line by its first char:
      - line1 starts with P / I / V / A / C / D / R + '<'  (doc type)
      - line2 starts with a digit / 'L' / '<' (passport no.)
    If classification is unambiguous, lines are reordered so the
    returned (line1, line2) tuple always satisfies the ICAO layout.
    """
    allowed = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<")
    allowed = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<")
    # Pick the two longest lines
    raw_lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    raw_lines.sort(key=len, reverse=True)
    a = raw_lines[0] if len(raw_lines) >= 1 else ""
    b = raw_lines[1] if len(raw_lines) >= 2 else ""
    # Classify: line1 starts with a doc-type letter + '<', line2 starts
    # with digit (passport no.). If tesseract emitted them reversed,
    # swap so the output always matches the ICAO contract.
    def is_l1(s):
        s = s.upper()
        return bool(s) and s[0] in "PIVACDR" and len(s) > 1 and s[1] == "<"
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
        # ambiguous — keep original order
        line1, line2 = a, b

    def _clean(s):
        s = s.upper()
        s = "".join(c if c in allowed else "<" for c in s)
        if len(s) < LINE_LEN:
            s = s + "<" * (LINE_LEN - len(s))
        return s[:LINE_LEN]

    line1 = _clean(line1)
    line2 = _clean(line2)
    # Slot-enforce ICAO charset
    line1 = _enforce_slot(line1, _LINE1_ALPHA, _LINE1_DIGIT)
    line2 = _enforce_slot(line2, _LINE2_ALPHA, _LINE2_DIGIT)
    # Force slot 1 of line1 to '<'
    line1 = (line1[0] if line1 else "P") + "<" + line1[2:]
    # Pad if short
    if len(line1) < LINE_LEN:
        line1 = line1 + "<" * (LINE_LEN - len(line1))
    if len(line2) < LINE_LEN:
        line2 = line2 + "<" * (LINE_LEN - len(line2))
    return line1, line2, conf


def main(argv):
    if len(argv) < 2:
        print("usage: paddle_local.py <image>", file=sys.stderr)
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
        gray = np.asarray(img.convert("L"))
    except Exception as e:
        print(f"result.ok       : cannot load image: {e}", file=sys.stderr)
        return 1

    H, W = gray.shape
    # Whole-image 3x upscale; tesseract --psm 6 -l ocrb auto-locates
    # the two MRZ lines. _find_band is still computed for the band_box
    # metadata in the output contract.
    y0, y1 = _find_mrz_band(gray)
    crop = img.resize((W * 3, H * 3), Image.LANCZOS)

    # IMPORTANT: the sandbox hides /tmp from the tesseract subprocess,
    # so we put the work file inside the repo tree where it's visible.
    # Use the project's top-level tests/ directory — that's where
    # gui.py + ocr_bench_runner.py live and where the runner expects
    # to find helper artefacts.
    argv_path = Path(argv[1]).resolve()
    # Walk up to find the project root (the dir that contains tests/).
    candidate = argv_path.parent
    found = None
    while candidate != candidate.parent:
        if (candidate / "tests" / "ocr_bench_runner.py").exists():
            found = candidate
            break
        candidate = candidate.parent
    if found is not None:
        tmpdir = found / "tests" / ".paddle_local_tmp"
    else:
        tmpdir = Path.cwd() / ".paddle_local_tmp"
    tmpdir.mkdir(parents=True, exist_ok=True)
    band_path = tmpdir / "band.png"
    crop.save(band_path)

    text, conf, rc = _tesseract_band(band_path, tmpdir)
    if rc != 0:
        print(f"result.ok       : tesseract failed (rc={rc})", file=sys.stderr)
        return 1
    # Fallback: if PSM 6 returned a single line only (the second MRZ
    # line was discarded), re-run with PSM 11 (sparse text) which
    # preserves both rows for clean strips.
    if len([ln for ln in text.splitlines() if ln.strip()]) < 2:
        r2 = subprocess.run(
            ["tesseract", str(band_path), str(Path(tmpdir) / "tess_out_psm11"),
             "--psm", "11", "-l", "ocrb",
             "-c", "tessedit_char_whitelist="
                  "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<"],
            capture_output=True, timeout=20,
        )
        if r2.returncode == 0:
            base11 = Path(str(Path(tmpdir) / "tess_out_psm11") + ".txt")
            if base11.exists():
                text_psm11 = base11.read_text()
                # Use PSM 11 if it returned more lines (>=2) than PSM 6.
                lines_psm6 = [l for l in text.splitlines() if l.strip()]
                lines_psm11 = [l for l in text_psm11.splitlines() if l.strip()]
                if len(lines_psm11) > len(lines_psm6):
                    text = text_psm11

    line1, line2, mean_conf = _norm_two_lines(text, conf)

    band_box = f"0 {y0} {W} {y1 - y0}"
    l1_real = sum(1 for c in line1 if c not in "<")
    l2_real = sum(1 for c in line2 if c not in "<")
    ok = (len(line1) == LINE_LEN and len(line2) == LINE_LEN
          and line1[0] in DOC_TYPES and line1[1] == "<"
          and l1_real >= 5 and l2_real >= 5)

    print(f"result.ok       : {'OK' if ok else 'unreadable'}")
    print(f"result.line1    : {line1}")
    print(f"result.line2    : {line2}")
    print(f"result.conf1    : {mean_conf}")
    print(f"result.conf2    : {mean_conf}")
    print(f"band.x band.y band.w band.h : {band_box}")
    return 0 if ok else 1


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except Exception as e:
        print(f"paddle_local: {e}", file=sys.stderr)
        sys.exit(1)
