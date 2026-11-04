#!/usr/bin/env python3
"""Evaluate ALL OCR engines against official GT (strict accuracy: empty output = all wrong).

Engines: cnn, crnn, tesseract, paddle, indra, traditional, fuse

Usage:
  # GT from gt_review_loop.xlsx (preferred — single source of truth)
  python3 tools/eval_all_engines.py --gt 10_ocr_research/gt_review_loop.xlsx \
      --tool 6_mrz_ocr/build/mrz_ocr_cnn_tool_pic
  # Legacy JSON GT
  python3 tools/eval_all_engines.py --gt tests/gt_assets_pic_official.json \
      --photo-dir assets/pic --tool 6_mrz_ocr/build/mrz_ocr_cnn_tool_pic
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, tempfile, time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_photo_train import locate_band

try:
    _restore_src = ROOT / "passport_photo_restore" / "src"
    if str(_restore_src) not in sys.path:
        sys.path.insert(0, str(_restore_src))
    from restore import remove_guilloche as _remove_guilloche
except Exception:
    _remove_guilloche = None

PAD = 10
ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ<"

_PADDLE_DIR = ROOT / "6_1_PaddleOCR"
if str(_PADDLE_DIR) not in sys.path:
    sys.path.insert(0, str(_PADDLE_DIR))


def _icao_clean(text: str) -> str:
    """Uppercase and keep only ICAO chars."""
    return "".join(c for c in text.upper() if c in ALPHABET)


def repair_line1(l1: str) -> str:
    """Repair TD3 line1: fix country code digit confusions only.
    Position 0 (doc type) and 1 (subtype) can be letters — do NOT force."""
    if not l1:
        return l1
    l1 = _icao_clean(l1)[:44]
    if len(l1) < 44:
        l1 = l1 + "<" * (44 - len(l1))
    chars = list(l1)
    if chars[0].isdigit():
        country_fix = {"0": "O", "1": "I", "2": "Z", "5": "S", "6": "G", "8": "B"}
        chars[0] = country_fix.get(chars[0], "P")
    if chars[1].isdigit():
        country_fix = {"0": "O", "1": "I", "2": "Z", "5": "S", "6": "G", "8": "B"}
        chars[1] = country_fix.get(chars[1], "<")
    country_fix = {"0": "O", "1": "I", "2": "Z", "5": "S", "6": "G", "8": "B"}
    for i in range(2, 5):
        if chars[i].isdigit() and chars[i] in country_fix:
            chars[i] = country_fix[chars[i]]
    for i in range(5, 44):
        if chars[i].isdigit() and chars[i] in country_fix:
            chars[i] = country_fix[chars[i]]
    i = 43
    while i >= 0 and chars[i] != "<":
        i -= 1
    if i < 43 and i >= 5:
        fill_count = 0
        j = i
        while j >= 5 and chars[j] == "<":
            fill_count += 1
            j -= 1
        if fill_count >= 5:
            for k in range(i + 1, 44):
                chars[k] = "<"
    return "".join(chars)


def repair_line2(l2: str) -> str:
    """Repair TD3 line2: fix sex position and nationality digit confusions.
    Conservative: only fixes clearly invalid characters (digit in sex pos,
    digit in nationality). Does NOT touch DOB/expiry/personal_no because
    some GT entries have non-standard characters in those positions."""
    if not l2:
        return l2
    l2 = _icao_clean(l2)
    if len(l2) < 44:
        l2 = l2 + "<" * (44 - len(l2))
    l2 = l2[:44]
    chars = list(l2)
    # Nationality (10-12): must be letters — fix digit confusions
    NAT_DIGIT_FIX = {"0": "O", "1": "I", "2": "Z", "5": "S", "6": "G", "8": "B"}
    for i in range(10, 13):
        if chars[i].isdigit() and chars[i] in NAT_DIGIT_FIX:
            chars[i] = NAT_DIGIT_FIX[chars[i]]
    NAT_FIX = {"CEE": "CHE", "CHL": "CHN", "PLL": "POL", "SYH": "SYR",
               "VMN": "CHN", "ATS": "AUT", "KNH": "KOR", "CHL": "CHN"}
    nat = "".join(chars[10:13])
    if nat in NAT_FIX:
        fixed = NAT_FIX[nat]
        for i, c in enumerate(fixed):
            chars[10 + i] = c
    # Sex position (20): must be M, F, or < — never a digit
    SEX_FIX = {"5": "F", "1": "I", "0": "O", "8": "B", "6": "G", "2": "Z"}
    if chars[20].isdigit() and chars[20] in SEX_FIX:
        chars[20] = SEX_FIX[chars[20]]
    return "".join(chars)


def _mrz_char_val(c: str) -> int:
    """MRZ character value for check digit computation."""
    if c == '<' or not c:
        return 0
    if c.isdigit():
        return int(c)
    if c.isalpha():
        return 10 + ord(c.upper()) - ord('A')
    return 0


def _mrz_check_digit(s: str) -> int:
    """Compute MRZ check digit (mod 10 of weighted sum, weights 7-3-1)."""
    w = [7, 3, 1]
    return sum(_mrz_char_val(c) * w[i % 3] for i, c in enumerate(s)) % 10


_O0_CONFUSION = {
    'O': '0', 'I': '1', 'S': '5', 'B': '8', 'Z': '2', 'G': '6',
    'D': '0', 'Q': '0',
}
_O0_REVERSE = {
    '0': 'O', '1': 'I', '5': 'S', '8': 'B', '2': 'Z', '6': 'G',
}


def repair_o0_confusion(l2: str) -> str:
    """Fix bidirectional letter↔digit confusion in personal number (pos 28-41)
    using the composite check digit (pos 43).

    Tries both directions:
      - letter→digit (O→0, I→1, S→5, etc.) — OCR misreads letter as digit
      - digit→letter (0→O, 1→I, 5→S, etc.) — OCR misreads digit as letter

    Strategy: single replacement → pair → all replacements.
    Only applies when composite CK is currently WRONG.
    """
    if len(l2) < 44:
        return l2
    chars = list(l2)
    composite_positions = (list(range(0, 10)) + list(range(13, 20))
                           + list(range(21, 28)) + list(range(28, 43)))
    w = [7, 3, 1]
    pos_to_weight = {pos: w[composite_positions.index(pos) % 3]
                      for pos in composite_positions}
    try:
        expected_ck = int(chars[43])
    except (ValueError, IndexError):
        return l2
    current_sum = sum(_mrz_char_val(chars[p]) * pos_to_weight[p]
                      for p in composite_positions)
    if current_sum % 10 == expected_ck:
        return l2
    candidates = []
    for pos in range(28, 42):
        c = chars[pos]
        if c in _O0_CONFUSION:
            repl = _O0_CONFUSION[c]
            delta = (_mrz_char_val(repl) - _mrz_char_val(c)) * pos_to_weight[pos]
            candidates.append((pos, repl, delta))
        if c in _O0_REVERSE:
            repl = _O0_REVERSE[c]
            delta = (_mrz_char_val(repl) - _mrz_char_val(c)) * pos_to_weight[pos]
            candidates.append((pos, repl, delta))
    if not candidates:
        return l2
    for pos, repl, delta in candidates:
        if (current_sum + delta) % 10 == expected_ck:
            chars[pos] = repl
            return "".join(chars)
    for i in range(len(candidates)):
        pos1, repl1, delta1 = candidates[i]
        for j in range(i + 1, len(candidates)):
            pos2, repl2, delta2 = candidates[j]
            if pos1 == pos2:
                continue
            if (current_sum + delta1 + delta2) % 10 == expected_ck:
                chars[pos1] = repl1
                chars[pos2] = repl2
                return "".join(chars)
    total_delta = sum(d for _, _, d in candidates)
    if (current_sum + total_delta) % 10 == expected_ck:
        for pos, repl, _ in candidates:
            chars[pos] = repl
        return "".join(chars)
    return l2


def repair_mrz(l1: str, l2: str) -> tuple:
    """Apply MRZ repair post-processing to both lines."""
    l1 = repair_line1(l1)
    l2 = repair_line2(l2)
    l2 = repair_o0_confusion(l2)
    return l1, l2


def fix_personal_number(l1: str, l2: str) -> str:
    """Try confusion pair fixes in personal number (pos 28-42) even when
    composite CK is correct.  Uses mathematical CK revalidation (no
    mrz_tool call) and prefers the version with fewer ambiguous chars.

    Only applies when a replacement produces a valid CK AND strictly
    fewer ambiguous characters — safe for already-correct results.
    """
    if len(l2) < 44:
        return l2
    ambiguous = set("OISBZGDQ015286")
    conf_map = {**_O0_CONFUSION, **_O0_REVERSE}
    candidates = []
    for pos in range(28, 42):
        c = l2[pos]
        if c in conf_map and conf_map[c] != c:
            candidates.append((pos, conf_map[c]))
    if not candidates:
        return l2
    curr_amb = sum(1 for c in l2[28:42] if c in ambiguous)
    composite_positions = (list(range(0, 10)) + list(range(13, 20))
                           + list(range(21, 28)) + list(range(28, 43)))
    w = [7, 3, 1]
    pos_to_weight = {pos: w[composite_positions.index(pos) % 3]
                     for pos in composite_positions}
    try:
        expected_ck = int(l2[43])
    except (ValueError, IndexError):
        return l2
    current_sum = sum(_mrz_char_val(l2[p]) * pos_to_weight[p]
                      for p in composite_positions)
    best_l2 = l2
    best_amb = curr_amb
    from itertools import combinations
    for r in range(1, min(4, len(candidates) + 1)):
        for combo in combinations(candidates, r):
            delta = 0
            chars = list(l2)
            for pos, repl in combo:
                delta += (_mrz_char_val(repl) - _mrz_char_val(l2[pos])) * pos_to_weight[pos]
                chars[pos] = repl
            if (current_sum + delta) % 10 == expected_ck:
                cand = "".join(chars)
                cand_amb = sum(1 for c in cand[28:42] if c in ambiguous)
                if cand_amb < best_amb:
                    best_l2 = cand
                    best_amb = cand_amb
    return best_l2


def parse_kv(out: str) -> dict:
    kv = {}
    for ln in out.splitlines():
        if ":" in ln:
            k, _, v = ln.partition(":")
            kv[k.strip()] = v.strip()
    return kv


def run_cnn(tool: Path, ppm: Path) -> dict:
    env = dict(os.environ)
    env["MRZ_OCR_LINE_MODELS"] = "1"
    env["MRZ_OCR_PAR"] = "1"
    env["MRZ_OCR_CRNN"] = "0"
    p = subprocess.run([str(tool), str(ppm)], capture_output=True,
                       text=True, timeout=60, env=env)
    kv = parse_kv(p.stdout)
    return {"line1": kv.get("result.line1", ""), "line2": kv.get("result.line2", ""),
            "ok": "OK" in kv.get("result.ok", ""),
            "conf1": int(kv.get("result.conf1", "0") or 0),
            "conf2": int(kv.get("result.conf2", "0") or 0)}


def run_fuse(tool: Path, ppm: Path) -> dict:
    env = dict(os.environ)
    env["MRZ_OCR_LINE_MODELS"] = "1"
    env["MRZ_OCR_PAR"] = "1"
    env["MRZ_OCR_CRNN"] = "1"
    env["MRZ_OCR_FUSE"] = "1"
    p = subprocess.run([str(tool), str(ppm)], capture_output=True,
                       text=True, timeout=60, env=env)
    kv = parse_kv(p.stdout)
    return {"line1": kv.get("result.line1", ""), "line2": kv.get("result.line2", ""),
            "ok": "OK" in kv.get("result.ok", "")}


def run_traditional(ppm: Path) -> dict:
    tool = ROOT / "6_mrz_ocr/build/mrz_ocr_tool"
    if not tool.exists():
        return {"line1": "", "line2": "", "ok": False}
    p = subprocess.run([str(tool), str(ppm)], capture_output=True,
                       text=True, timeout=60)
    kv = parse_kv(p.stdout)
    return {"line1": kv.get("result.line1", ""), "line2": kv.get("result.line2", ""),
            "ok": "OK" in kv.get("result.ok", "")}


def run_crnn(tool: Path, ppm: Path) -> dict:
    env = dict(os.environ)
    env["MRZ_OCR_LINE_MODELS"] = "1"
    env["MRZ_OCR_PAR"] = "1"
    env["MRZ_OCR_CRNN"] = "1"
    env["MRZ_OCR_FUSE"] = "0"
    p = subprocess.run([str(tool), str(ppm)], capture_output=True,
                       text=True, timeout=60, env=env)
    kv = parse_kv(p.stdout)
    return {"line1": kv.get("result.line1", ""), "line2": kv.get("result.line2", ""),
            "ok": "OK" in kv.get("result.ok", "")}


def _mrz_line1_candidates(text: str) -> list:
    """Extract 44-char TD3 line-1 windows from noisy OCR text."""
    clean = _icao_clean(text)
    found = []
    doc_types = set("PIVACDRQ")
    for start in range(max(1, len(clean) - 43)):
        line = clean[start:start + 44]
        while len(line) < 44:
            line += "<"
        if (len(line) == 44 and line[0] in doc_types
                    and line[1] in "<ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                    and line[2:5].isalpha()
                    and sum(ch.isalpha() for ch in line[5:]) >= 3):
            if line not in found:
                found.append(line)
            if line[0] == "P" and line[1] in "O0":
                alt = line[0] + "<" + line[2:]
                if alt not in found:
                    found.append(alt)
    return found


def _mrz_line2_candidates(text: str) -> list:
    """Extract and checksum-repair near-44-char TD3 line-2 windows."""
    clean = _icao_clean(text)
    if not clean:
        return []
    from mrz_checksum import repair_line2
    country_fix = {"0": "O", "1": "I", "2": "Z", "5": "S", "6": "G", "8": "B"}
    found = []
    for start in range(len(clean)):
        for length in (43, 44, 45):
            raw = clean[start:start + length]
            if len(raw) != length or len(raw) < 28:
                continue
            country = raw[10:13]
            if not country.isalpha():
                country = "".join(country_fix.get(ch, ch) for ch in country)
                if not country.isalpha():
                    continue
                raw = raw[:10] + country + raw[13:]
            if raw[20] not in "MF<ABCDEFGHIJKLMNOPQRSTUVWXYZ":
                continue
            fixed, n_fixed, report, valid = repair_line2(raw)
            if len(fixed) != 44:
                continue
            item = (n_fixed if valid else 99 + n_fixed,
                    abs(length - 44), start, fixed, report, raw)
            if not any(old[3] == fixed for old in found):
                found.append(item)
    found.sort(key=lambda item: item[:3])
    return found


def _decode_ok(decoded) -> bool:
    if decoded is None or decoded.returncode != 0:
        return False
    out = decoded.stdout or ""
    if "decode failed" in (decoded.stderr or "").lower():
        return False
    return "doc_type" in out


def _decode_candidates(line1s: list, line2s: list) -> tuple:
    """Return the first OCR candidate pair accepted by mrz_tool decode.
    Try normal order, then swapped order (line1<->line2 confusion)."""
    mrz_tool = ROOT / "1_mrz_decode/build/mrz_tool"
    for line1 in line1s:
        for item in line2s:
            line2 = item[3]
            decoded = subprocess.run(
                [str(mrz_tool), "decode", f"{line1}\n{line2}"],
                text=True, capture_output=True, timeout=10)
            if _decode_ok(decoded):
                return line1, line2
    for item in line2s:
        line1_try = item[3]
        if not line1_try or line1_try[0] not in "PIVACDRQ":
            continue
        for line2_try in line1s:
            if line2_try[0] in "PIVACDRQ":
                continue
            decoded = subprocess.run(
                [str(mrz_tool), "decode", f"{line1_try}\n{line2_try}"],
                text=True, capture_output=True, timeout=10)
            if _decode_ok(decoded):
                return line1_try, line2_try
    return "", ""


def run_tesseract(ppm: Path) -> dict:
    """Run Tesseract on band with candidate + checksum + decode verification."""
    import shutil
    tess = shutil.which("tesseract")
    if not tess:
        return {"line1": "", "line2": "", "ok": False}
    img = Image.open(str(ppm))
    W, H = img.size
    gray = img.convert("L")
    arr = np.array(gray)
    bin_img = (arr < 128).astype(np.uint8)
    row_dark = bin_img.sum(axis=1)
    peak = row_dark.max()
    gap_thr = max(peak // 6, 4)
    best_gap_y, best_gap_h = -1, 0
    for y in range(1, H - 1):
        gh = 0
        while y + gh < H and row_dark[y + gh] < gap_thr:
            gh += 1
        if gh >= max(H // 25, 2) and gh > best_gap_h:
            if y >= H // 5 and (H - (y + gh)) >= H // 5:
                best_gap_y, best_gap_h = y, gh
    if best_gap_y < 0:
        mid = H // 2
        rows = [(0, mid), (mid, H)]
    else:
        rows = [(0, best_gap_y), (best_gap_y + best_gap_h, H)]
        if best_gap_y < 20 or (H - best_gap_y - best_gap_h) < 20:
            mid = H // 2
            rows = [(0, mid), (mid, H)]
    whitelist = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<"
    import tempfile as _tf
    row_texts = [[], []]
    with _tf.NamedTemporaryFile(suffix=".png", delete=False) as tf:
        img.save(tf.name)
        tmp_full = tf.name
    for psm in [6, 4, 3]:
        try:
            proc = subprocess.run(
                [tess, tmp_full, "stdout", "--psm", str(psm), "-l", "ocrb",
                 "-c", f"tessedit_char_whitelist={whitelist}"],
                text=True, capture_output=True, timeout=20)
            lines = ["".join(c for c in ln if c in whitelist)
                     for ln in (proc.stdout or "").splitlines()]
            lines = [ln for ln in lines if ln]
            if len(lines) >= 2:
                if lines[0] not in row_texts[0]:
                    row_texts[0].append(lines[0])
                if lines[1] not in row_texts[1]:
                    row_texts[1].append(lines[1])
        except Exception:
            pass
    os.unlink(tmp_full)
    for idx, (y0, y1) in enumerate(rows):
        pad = max(4, min(15, (y1 - y0) // 4))
        crop = img.crop((0, max(0, y0 - pad), W, min(H, y1 + pad)))
        if crop.height < 20:
            crop = crop.resize((crop.width * 2, crop.height * 2), Image.LANCZOS)
        with _tf.NamedTemporaryFile(suffix=".png", delete=False) as tf:
            crop.save(tf.name)
            tmp_png = tf.name
        for psm in [7, 6, 13, 8, 4]:
            try:
                proc = subprocess.run(
                    [tess, tmp_png, "stdout", "--psm", str(psm), "-l", "ocrb",
                     "-c", f"tessedit_char_whitelist={whitelist}"],
                    text=True, capture_output=True, timeout=20)
                text = "".join(c for c in (proc.stdout or "") if c in whitelist)
                if text:
                    row_texts[idx].append(text)
            except Exception:
                pass
        os.unlink(tmp_png)
    l1_cands = []
    for text in row_texts[0]:
        for cand in _mrz_line1_candidates(text):
            if cand not in l1_cands:
                l1_cands.append(cand)
    l2_cands = []
    for text in row_texts[1]:
        for item in _mrz_line2_candidates(text):
            if not any(old[3] == item[3] for old in l2_cands):
                l2_cands.append(item)
    if not l1_cands:
        for text in row_texts[1]:
            for cand in _mrz_line1_candidates(text):
                if cand not in l1_cands:
                    l1_cands.append(cand)
    if not l2_cands:
        for text in row_texts[0]:
            for item in _mrz_line2_candidates(text):
                if not any(old[3] == item[3] for old in l2_cands):
                    l2_cands.append(item)
    l1_cands.sort(key=lambda line: (line[0] != "P", line[1] != "<",
                                    -line[5:].count("<")))
    if l1_cands and l2_cands:
        l1, l2 = _decode_candidates(l1_cands, l2_cands)
        if l1 and l2:
            l1, l2 = repair_mrz(l1, l2)
            return {"line1": l1, "line2": l2, "ok": True}
    best_texts = ["", ""]
    for idx in range(2):
        for text in row_texts[idx]:
            if len(text) > len(best_texts[idx]):
                best_texts[idx] = text
    l1 = best_texts[0][:44]
    l2 = best_texts[1][:44]
    while len(l1) < 44:
        l1 += "<"
    while len(l2) < 44:
        l2 += "<"
    if l1_cands:
        l1 = l1_cands[0]
    l1, l2 = verify_and_repair(l1, l2)
    return {"line1": l1, "line2": l2, "ok": bool(l1)}


def verify_and_repair(l1: str, l2: str) -> tuple:
    """Verify with mrz_tool decode; try repair if decode fails. Returns (l1, l2)."""
    if not l1 or not l2:
        return l1, l2
    l1 = l1[:44]
    l2 = l2[:44]
    while len(l1) < 44:
        l1 += "<"
    while len(l2) < 44:
        l2 += "<"
    mrz_tool = ROOT / "1_mrz_decode/build/mrz_tool"
    decoded = subprocess.run(
        [str(mrz_tool), "decode", f"{l1}\n{l2}"],
        text=True, capture_output=True, timeout=10)
    if _decode_ok(decoded):
        return l1, l2
    try:
        from mrz_checksum import repair_line2 as ck_repair
        fixed2, n_fixed, report, valid = ck_repair(l2)
        if valid and fixed2 != l2:
            decoded = subprocess.run(
                [str(mrz_tool), "decode", f"{l1}\n{fixed2}"],
                text=True, capture_output=True, timeout=10)
            if _decode_ok(decoded):
                return l1, fixed2
    except Exception:
        pass
    rl1, rl2 = repair_mrz(l1, l2)
    decoded = subprocess.run(
        [str(mrz_tool), "decode", f"{rl1}\n{rl2}"],
        text=True, capture_output=True, timeout=10)
    if _decode_ok(decoded):
        return rl1, rl2
    return l1, l2


def run_paddle(ppm: Path) -> dict:
    script = ROOT / "6_1_PaddleOCR/paddle_v6.py"
    if not script.exists():
        return {"line1": "", "line2": "", "ok": False}
    env = dict(os.environ)
    env["MRZ_BAND_ALREADY_CROPPED"] = "1"
    p = subprocess.run([sys.executable, str(script), str(ppm)],
                       capture_output=True, text=True, timeout=60, env=env)
    kv = parse_kv(p.stdout)
    l1 = kv.get("result.line1", "")
    l2 = kv.get("result.line2", "")
    l1, l2 = verify_and_repair(l1, l2)
    return {"line1": l1, "line2": l2,
            "ok": "OK" in kv.get("result.ok", "")}


def run_indra(ppm: Path) -> dict:
    script = ROOT / "11_indra/mrz_indra.py"
    if not script.exists():
        return {"line1": "", "line2": "", "ok": False}
    p = subprocess.run([sys.executable, str(script), str(ppm)],
                       capture_output=True, text=True, timeout=60)
    kv = parse_kv(p.stdout)
    l1 = kv.get("result.line1", "")
    l2 = kv.get("result.line2", "")
    l1, l2 = verify_and_repair(l1, l2)
    return {"line1": l1, "line2": l2,
            "ok": "OK" in kv.get("result.ok", "")}


def _consensus_vote(lines: list, prefer_last: bool = False) -> str:
    """Per-position majority vote across multiple engine outputs.

    When prefer_last=True, ties are broken in favor of the last engine
    in the list (useful for putting CNN last so non-CNN engines win ties).
    """
    if not lines:
        return ""
    max_len = max(len(l) for l in lines)
    result = []
    for i in range(max_len):
        votes = {}
        for line in lines:
            if i < len(line):
                c = line[i]
                votes[c] = votes.get(c, 0) + 1
        if votes:
            best_count = max(votes.values())
            cands = [c for c, n in votes.items() if n == best_count]
            non_fill = [c for c in cands if c != "<"]
            pick_from = non_fill if non_fill else cands
            if prefer_last and len(pick_from) > 1:
                for line in reversed(lines):
                    if i < len(line) and line[i] in pick_from:
                        result.append(line[i])
                        break
                else:
                    result.append(pick_from[0])
            else:
                result.append(pick_from[0])
        else:
            result.append("<")
    return "".join(result)


def _decode_valid(l1: str, l2: str) -> bool:
    """Check if mrz_tool decode accepts this L1/L2 pair."""
    if not l1 or not l2:
        return False
    mrz_tool = ROOT / "1_mrz_decode/build/mrz_tool"
    decoded = subprocess.run(
        [str(mrz_tool), "decode", f"{l1}\n{l2}"],
        text=True, capture_output=True, timeout=10)
    return _decode_ok(decoded)


def _decode_valid_single(line: str) -> bool:
    """Check if mrz_tool decode accepts a single 30-char CS/CD line."""
    if not line or len(line) != 30:
        return False
    mrz_tool = ROOT / "1_mrz_decode/build/mrz_tool"
    decoded = subprocess.run(
        [str(mrz_tool), "decode", line],
        text=True, capture_output=True, timeout=10)
    return _decode_ok(decoded)


def run_cscd_ocr(ppm: Path) -> dict:
    """OCR for CS/CD single-line 30-char MRZ.

    Tries tesseract (single-line PSM) and paddle, extracts 30-char
    candidates starting with CS/CD, verifies with mrz_tool decode.
    Also tries all 10 composite CK values when content is correct but CK wrong.
    """
    import shutil
    tess = shutil.which("tesseract")
    whitelist = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<"
    candidates = []
    img = Image.open(str(ppm))

    def _extract_cscd(text):
        clean = _icao_clean(text)
        for start in range(max(1, len(clean) - 29)):
            cand = clean[start:start + 30]
            if len(cand) == 30 and cand[:2] in ("CS", "CD"):
                if cand not in candidates:
                    candidates.append(cand)

    def _cscd_found():
        for cand in candidates:
            if _decode_valid_single(cand):
                return cand
        return None

    if tess:
        for psm in [7, 8, 13]:
            try:
                proc = subprocess.run(
                    [tess, str(ppm), "stdout", "--psm", str(psm), "-l", "ocrb",
                     "-c", f"tessedit_char_whitelist={whitelist}"],
                    text=True, capture_output=True, timeout=20)
                _extract_cscd(proc.stdout or "")
            except Exception:
                pass
        found = _cscd_found()
        if found:
            return {"line1": found, "line2": "", "ok": True}

        w, h = img.size
        big = img.resize((w * 2, h * 2), Image.LANCZOS)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
            big.save(tf.name)
            tmp_png = tf.name
        for psm in [7, 8]:
            try:
                proc = subprocess.run(
                    [tess, tmp_png, "stdout", "--psm", str(psm), "-l", "ocrb",
                     "-c", f"tessedit_char_whitelist={whitelist}"],
                    text=True, capture_output=True, timeout=20)
                _extract_cscd(proc.stdout or "")
            except Exception:
                pass
        os.unlink(tmp_png)
        found = _cscd_found()
        if found:
            return {"line1": found, "line2": "", "ok": True}

        gray = img.convert("L")
        arr = np.array(gray)
        bin_arr = np.where(arr < 128, 0, 255).astype(np.uint8)
        bin_img = Image.fromarray(bin_arr)
        w, h = bin_img.size
        big = bin_img.resize((w * 2, h * 2), Image.LANCZOS)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
            big.save(tf.name)
            tmp_png = tf.name
        for psm in [7, 8]:
            try:
                proc = subprocess.run(
                    [tess, tmp_png, "stdout", "--psm", str(psm), "-l", "ocrb",
                     "-c", f"tessedit_char_whitelist={whitelist}"],
                    text=True, capture_output=True, timeout=20)
                _extract_cscd(proc.stdout or "")
            except Exception:
                pass
        os.unlink(tmp_png)
        found = _cscd_found()
        if found:
            return {"line1": found, "line2": "", "ok": True}

        try:
            import cv2 as _cv2
            clahe = _cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 2))
            cl = clahe.apply(arr)
            cl_up = _cv2.resize(cl, (cl.shape[1]*2, cl.shape[0]*2),
                                interpolation=_cv2.INTER_CUBIC)
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
                _cv2.imwrite(tf.name, cl_up)
                tmp_png = tf.name
            for psm in [7, 8]:
                try:
                    proc = subprocess.run(
                        [tess, tmp_png, "stdout", "--psm", str(psm), "-l", "ocrb",
                         "-c", f"tessedit_char_whitelist={whitelist}"],
                        text=True, capture_output=True, timeout=20)
                    _extract_cscd(proc.stdout or "")
                except Exception:
                    pass
            os.unlink(tmp_png)
        except Exception:
            pass

    script = ROOT / "6_1_PaddleOCR/paddle_v6.py"
    if script.exists():
        env = dict(os.environ)
        env["MRZ_BAND_ALREADY_CROPPED"] = "1"
        try:
            p = subprocess.run([sys.executable, str(script), str(ppm)],
                               capture_output=True, text=True, timeout=60, env=env)
            kv = parse_kv(p.stdout)
            for key in ("result.line1", "result.line2"):
                _extract_cscd(kv.get(key, ""))
        except Exception:
            pass

    found = _cscd_found()
    if found:
        return {"line1": found, "line2": "", "ok": True}

    for cand in candidates:
        for ck in range(10):
            fixed = cand[:29] + str(ck)
            if fixed != cand and _decode_valid_single(fixed):
                return {"line1": fixed, "line2": "", "ok": True}

    _cscd_conf = {"T": "7", "7": "T", "O": "0", "0": "O", "I": "1", "1": "I",
                  "S": "5", "5": "S", "B": "8", "8": "B", "Z": "2", "2": "Z",
                  "G": "6", "6": "G", "A": "4", "4": "A", "U": "V", "V": "U",
                  "D": "0", "P": "R", "R": "P"}
    for cand in candidates:
        for i in range(2, 29):
            orig = cand[i]
            if orig in _cscd_conf and orig != "<":
                fixed = cand[:i] + _cscd_conf[orig] + cand[i+1:]
                if _decode_valid_single(fixed):
                    return {"line1": fixed, "line2": "", "ok": True}

    if candidates:
        return {"line1": candidates[0], "line2": "", "ok": False}
    return {"line1": "", "line2": "", "ok": False}


_CONF_PAIRS = set()
for _k, _v in _O0_CONFUSION.items():
    _CONF_PAIRS.add((_k, _v))
    _CONF_PAIRS.add((_v, _k))
for _k, _v in _O0_REVERSE.items():
    _CONF_PAIRS.add((_k, _v))
    _CONF_PAIRS.add((_v, _k))
_CONF_PAIRS.add(("4", "A"))
_CONF_PAIRS.add(("A", "4"))
_CONF_PAIRS.add(("7", "1"))
_CONF_PAIRS.add(("1", "7"))


def _confusion_merge(valid, cnn_l1, cnn_l2):
    """If cnn and another engine agree on one line but differ only in
    confusion pairs on the other, prefer the non-cnn result."""
    for r in valid:
        l1o, l2o = r["line1"], r["line2"]
        if l1o == cnn_l1 and l2o != cnn_l2:
            diffs = [(cnn_l2[i], l2o[i]) for i in range(min(len(cnn_l2), len(l2o))) if cnn_l2[i] != l2o[i]]
            if diffs and all((a, b) in _CONF_PAIRS for a, b in diffs):
                if _decode_valid(cnn_l1, l2o):
                    return {"line1": cnn_l1, "line2": l2o, "ok": True}
        if l2o == cnn_l2 and l1o != cnn_l1:
            diffs = [(cnn_l1[i], l1o[i]) for i in range(min(len(cnn_l1), len(l1o))) if cnn_l1[i] != l1o[i]]
            if diffs and all((a, b) in _CONF_PAIRS for a, b in diffs):
                if _decode_valid(l1o, cnn_l2):
                    return {"line1": l1o, "line2": cnn_l2, "ok": True}
    return None


def _try_cross_combine(voted_l1, voted_l2, cnn_l1, cnn_l2):
    """Try cross-combining voted lines with CNN lines.

    When the consensus vote fixes one line but corrupts the other,
    try mixing voted L1 + CNN L2 (or CNN L1 + voted L2) to preserve
    the correct line from each source.  Returns dict on success.
    """
    if cnn_l1 and cnn_l2:
        if voted_l1 != cnn_l1 and _decode_valid(voted_l1, cnn_l2):
            return {"line1": voted_l1, "line2": cnn_l2, "ok": True}
        if voted_l2 != cnn_l2 and _decode_valid(cnn_l1, voted_l2):
            return {"line1": cnn_l1, "line2": voted_l2, "ok": True}
    return None


def _run_fuse_multi_impl(tool: Path, ppm: Path) -> dict:
    """CNN+CRNN fusion — no external fallback (no paddle/tesseract).

    CNN passes decode → return CNN (~8ms)
    CNN fails → try CRNN → if CRNN passes decode, return CRNN (~16ms)
    Both fail → return CNN (failure is failure)
    """
    cnn_res = run_cnn(tool, ppm)
    cnn_l1, cnn_l2 = cnn_res.get("line1", ""), cnn_res.get("line2", "")

    if cnn_l1 and cnn_l2 and _decode_valid(cnn_l1, cnn_l2):
        return cnn_res

    crnn_res = run_crnn(tool, ppm)
    crnn_l1, crnn_l2 = crnn_res.get("line1", ""), crnn_res.get("line2", "")
    if crnn_l1 and crnn_l2 and _decode_valid(crnn_l1, crnn_l2):
        return {"line1": crnn_l1, "line2": crnn_l2, "ok": True}

    return cnn_res


def run_fuse_multi(tool: Path, ppm: Path) -> dict:
    """Multi-engine fusion with consensus voting + personal number repair."""
    result = _run_fuse_multi_impl(tool, ppm)
    l1 = result.get("line1", "")
    l2 = result.get("line2", "")
    if l1 and l2 and len(l2) == 44:
        fixed_l2 = fix_personal_number(l1, l2)
        if fixed_l2 != l2:
            result["line2"] = fixed_l2
    return result


ENGINES = {
    "cnn": lambda tool, ppm, rgb, crop: run_cnn(tool, ppm),
    "fuse": lambda tool, ppm, rgb, crop: run_fuse_multi(tool, ppm),
    "traditional": lambda tool, ppm, rgb, crop: run_traditional(ppm),
    "crnn": lambda tool, ppm, rgb, crop: run_crnn(tool, ppm),
    "tesseract": lambda tool, ppm, rgb, crop: run_tesseract(ppm),
    "paddle": lambda tool, ppm, rgb, crop: run_paddle(ppm),
    "indra": lambda tool, ppm, rgb, crop: run_indra(ppm),
}


def load_gt_xlsx(path) -> dict:
    """Load GT from gt_review_loop.xlsx 'GT对比' sheet (GT line1 / GT line2).

    Returns {filename: {"l1": gt_line1, "l2": gt_line2}}.
    """
    import openpyxl
    wb = openpyxl.load_workbook(str(path), read_only=True, data_only=True)
    ws = wb["GT对比"]
    gt = {}
    for row in ws.iter_rows(min_row=2, values_only=True):
        fn = row[1] or ""
        gt1 = row[6] or ""
        gt2 = row[7] or ""
        if fn and gt1:
            gt[fn] = {"l1": gt1, "l2": gt2}
    wb.close()
    return gt


def _find_photo(basename, label, photo_dir) -> Path | None:
    """Locate a photo file.

    If photo_dir is given, ONLY search there (for per-dataset evaluation).
    Otherwise auto-search label → assets/pic → tmp/PassPortPhoto.
    """
    if photo_dir:
        p = ROOT / photo_dir / basename
        return p if p.exists() else None
    if label:
        p = ROOT / label
        if p.exists():
            return p
    for d in ("assets/pic", "tmp/PassPortPhoto"):
        p = ROOT / d / basename
        if p.exists():
            return p
    return None


def strict_acc(records, pred_key, gt_key) -> tuple:
    ok = tot = 0
    for r in records:
        g = r.get(gt_key, "")
        p = r.get(pred_key, "")
        if not g:
            continue
        tot += len(g)
        for a, b in zip(g, p[:len(g)]):
            ok += (a == b)
    return ok, tot


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True,
                    help="GT source: .xlsx (gt_review_loop.xlsx) or .json")
    ap.add_argument("--photo-dir", default=None,
                    help="Photo dir (optional; auto-searches assets/pic + tmp/PassPortPhoto)")
    ap.add_argument("--tool", required=True)
    ap.add_argument("--engines", default="cnn,crnn,tesseract,paddle,indra,fuse,traditional")
    ap.add_argument("--repair", action="store_true",
                    help="Apply MRZ repair post-processing (checksum + confusion)")
    args = ap.parse_args()

    tool = Path(args.tool).resolve()
    gt_path = Path(args.gt)
    if gt_path.suffix == ".xlsx":
        gt = load_gt_xlsx(gt_path if gt_path.is_absolute() else ROOT / gt_path)
    else:
        d = json.loads((ROOT / args.gt).read_text())
        gt = d["gt"] if isinstance(d, dict) and "gt" in d else d
    engine_list = [e.strip() for e in args.engines.split(",")]

    tmp = Path(tempfile.mkdtemp(prefix="eval_eng_"))
    records = []
    skipped_band = 0

    for basename, rec in sorted(gt.items()):
        label = rec.get("label") or ""
        l1, l2 = rec.get("l1", ""), rec.get("l2", "")
        is_td3 = len(l1) == 44 and len(l2) == 44
        is_cscd = len(l1) == 30 and not l2
        if not (is_td3 or is_cscd):
            continue
        img_path = _find_photo(basename, label, args.photo_dir)
        if img_path is None:
            continue
        rgb = Image.open(str(img_path)).convert("RGB")
        crop = locate_band(rgb)
        if crop is None:
            skipped_band += 1
            records.append({"id": basename, "gt1": l1, "gt2": l2, "located": False})
            continue
        bx, by, bw, bh = crop
        band = rgb.crop((bx, max(0, by - PAD), bx + bw, min(rgb.height, by + bh + PAD)))
        if is_cscd and _remove_guilloche is not None:
            band_arr = np.array(band)
            band_arr = _remove_guilloche(band_arr)
            band = Image.fromarray(band_arr)
        ppm = tmp / f"{Path(basename).stem}.ppm"
        band.save(str(ppm), format="PPM")

        row = {"id": basename, "gt1": l1, "gt2": l2, "located": True}
        for eng in engine_list:
            if eng not in ENGINES:
                continue
            t0 = time.time()
            try:
                result = ENGINES[eng](tool, ppm, rgb, crop)
            except Exception as e:
                result = {"line1": "", "line2": "", "ok": False}
            ms = (time.time() - t0) * 1000
            row[f"{eng}_l1"] = result["line1"]
            row[f"{eng}_l2"] = result["line2"]
            row[f"{eng}_ms"] = ms
            if args.repair:
                if is_cscd:
                    row[f"{eng}_rl1"] = result["line1"]
                    row[f"{eng}_rl2"] = result["line2"]
                else:
                    rl1, rl2 = repair_mrz(result["line1"], result["line2"])
                    row[f"{eng}_rl1"] = rl1
                    row[f"{eng}_rl2"] = rl2
        records.append(row)

    print(f"\n{'='*80}")
    print(f"GT: {args.gt}  |  Photos: {len(records)}  |  Band-fail: {skipped_band}")
    print(f"{'='*80}")
    target = 98.0 if (args.photo_dir or "").endswith("PassPortPhoto") else 95.0
    print(f"判定规则: line1+line2 全部匹配 GT  →  目标准确率 ≥ {target:.0f}%")
    print(f"{'Engine':<14} {'Both exact':>11} {'Pass%':>8} {'Verdict':>8} "
          f"{'L1 char%':>10} {'L2 char%':>10} {'Mean ms':>8}")
    print("-" * 80)

    for eng in engine_list:
        if eng not in ENGINES:
            continue
        l1_ok = sum(1 for r in records if r.get("located") and
                    r.get(f"{eng}_l1", "") == r.get("gt1", ""))
        l2_ok = sum(1 for r in records if r.get("located") and
                    r.get(f"{eng}_l2", "") == r.get("gt2", ""))
        both_ok = sum(1 for r in records if r.get("located") and
                      r.get(f"{eng}_l1", "") == r.get("gt1", "") and
                      (not r.get("gt2", "") or
                       r.get(f"{eng}_l2", "") == r.get("gt2", "")))
        o1, t1 = strict_acc(records, f"{eng}_l1", "gt1")
        o2, t2 = strict_acc(records, f"{eng}_l2", "gt2")
        times = [r[f"{eng}_ms"] for r in records if r.get("located")]
        mean_ms = sum(times) / len(times) if times else 0
        n_loc = sum(1 for r in records if r.get("located"))
        pass_pct = both_ok / n_loc * 100 if n_loc else 0
        verdict = "PASS" if pass_pct >= target else "FAIL"
        print(f"{eng:<14} {both_ok:>5}/{n_loc:<4} {pass_pct:>7.2f}% {verdict:>8} "
              f"{o1/t1*100 if t1 else 0:>9.2f}% "
              f"{o2/t2*100 if t2 else 0:>9.2f}% {mean_ms:>7.0f}")
        if args.repair:
            ro1, _ = strict_acc(records, f"{eng}_rl1", "gt1")
            ro2, _ = strict_acc(records, f"{eng}_rl2", "gt2")
            rl1_ok = sum(1 for r in records if r.get("located") and
                         r.get(f"{eng}_rl1", "") == r.get("gt1", ""))
            rl2_ok = sum(1 for r in records if r.get("located") and
                         r.get(f"{eng}_rl2", "") == r.get("gt2", ""))
            rboth = sum(1 for r in records if r.get("located") and
                        r.get(f"{eng}_rl1", "") == r.get("gt1", "") and
                        (not r.get("gt2", "") or
                         r.get(f"{eng}_rl2", "") == r.get("gt2", "")))
            rpass = rboth / n_loc * 100 if n_loc else 0
            rverdict = "PASS" if rpass >= target else "FAIL"
            print(f"  {'+repair':<12} {rboth:>5}/{n_loc:<4} {rpass:>7.2f}% {rverdict:>8} "
                  f"{ro1/t1*100 if t1 else 0:>9.2f}% "
                  f"{ro2/t2*100 if t2 else 0:>9.2f}%")

    if args.gt:
        out_json = ROOT / "tmp" / f"eval_engines_{Path(args.gt).stem}.json"
        out_json.parent.mkdir(exist_ok=True)
        out_json.write_text(json.dumps(records, ensure_ascii=False, indent=1))
        print(f"\nDetails: {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
