#!/usr/bin/env python3
"""tesseract_tool -- industrial Tesseract MRZ engine (ocrb model).

Implements the 5-stage pipeline:
  1. geometry pre-process : row-projection MRZ band locate + exact
     per-line split + character-height rescale (~32px glyph, 48px
     line band) + per-line contrast stretch
  2. tesseract params      : PSM_SINGLE_LINE, all Dawgs disabled,
     char whitelist, textord_no_rejects, preserve_interword_spaces=0
  3. model                 : Shreeshrii tessdata_ocrb (OCR-B fine-tuned)
  4. API                   : resident PyTessBaseAPI, per-symbol
     confidence via ResultIterator RIL_SYMBOL
  5. post-process          : 44-char fixed-length regularization,
     slot whitelist hard-fix (dates O->0 I->1 B->8 S->5 Z->2),
     composite-checksum repair (single flips then pairs over the
     lowest-confidence positions)

Stdout contract (identical to 6_mrz_ocr/build/mrz_ocr_tool):
    argv[1] = image path (PPM/PBM/BMP/PNG/JPG via PIL)
    stdout  = key/value lines:
        result.ok      : OK | <error message>
        result.line1   : 44-char line (or empty)
        result.line2   : 44-char line (or empty)
        result.conf1   : 0..100
        result.conf2   : 0..100
        band.x band.y band.w band.h : "x y w h" in image coordinates
    exit code: 0 on success, 1 on OCR failure
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

LINE_LEN = 44
WHITELIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<"
TARGET_LINE_H = 48          # output line-band height (glyph ~32px)
DOC_TYPES = set("PIVACDR")
AMBIG = {"O": "0", "0": "O", "I": "1", "1": "I", "B": "8", "8": "B",
         "S": "5", "5": "S", "Z": "2", "2": "Z"}
CONFUSION = {"0": ["<", "O", "D"], "<": ["0", "C"], "O": ["0", "D"],
             "1": ["I", "L"], "I": ["1"], "8": ["B"], "B": ["8"],
             "5": ["S"], "S": ["5"], "Z": ["2"], "2": ["Z"]}
_DATE_SLOTS = frozenset(range(13, 20)) | frozenset(range(21, 28))
# ISO 3166-1 alpha-3 issuing states + common agency codes (UTO etc.)
ISO_CODES = frozenset("""ABW AFG AGO AIA ALB AND ANT ARE ARG ARM ASM ATA ATF
ATG AUS AUT AZE BDI BEL BEN BES BFA BGD BGR BHR BHS BIH BLM BLR BLZ BMU
BOL BRA BRB BRN BTN BVT BWA CAF CAN CCK CHE CHL CHN CIV CMR COD COG COK
COL COM CPV CRI CUB CUW CXR CYM CYP CZE DEU DJI DMA DNK DOM DZA ECU EGY
ERI ESH ESP EST ETH FIN FJI FLK FRA FRO FSM GAB GBR GEO GGY GHA GIB GIN
GLP GMB GNB GNQ GRC GRD GRL GTM GUF GUM GUY HKG HMD HND HRV HTI HUN IDN
IMN IND IOT IRL IRN IRQ ISL ISR ITA JAM JEY JOR JPN KAZ KEN KGZ KHM KIR
KNA KOR KWT LAO LBN LBR LBY LCA LIE LKA LSO LTU LUX LVA MAC MAF MAR MCO
MDA MDG MDV MEX MHL MKD MLI MLT MMR MNE MNG MNP MOZ MRT MSR MTQ MUS MWI
MYS MYT NAM NCL NER NFK NGA NIC NIU NLD NOR NPL NRU NZL OMN PAK PAN PCN
PER PHL PLW PNG POL PRI PRK PRT PRY PSE PYF QAT REU ROU RUS RWA SAU SDN
SEN SGP SGS SHN SJM SLB SLE SLV SMR SOM SPM SRB SSD STP SUR SVK SVN SWE
SWZ SXM SYC SYR TCA TCD TGO THA TJK TKL TKM TLS TON TTO TUN TUR TUV TWN
TZA UGA UKR UMI URY USA UZB VAT VCT VEN VGB VIR VNM VUT WLF WSM YEM ZAF
ZMB ZWE UTO EUE EUU XXA XXB XXX XIN UNO""".split())


def _load_gray(path: Path):
    """Return (HxW uint8 ndarray, original PIL RGB image)."""
    import numpy as np
    from PIL import Image
    img = Image.open(path).convert("RGB")
    return np.asarray(img.convert("L")), img


def _mrz_lines(gray, pad: int = 2):
    """Locate the two MRZ lines (whole-page scan, bottom-preferring).

    Returns (y0, y1, [(l1_top, l1_bot), (l2_top, l2_bot)]) or None.
    Row boundaries are padded by `pad` px each side so glyph
    ascenders/descenders are never clipped (tight rows lose the
    top/bottom strokes and recognition collapses).
    """
    import numpy as np
    H, W = gray.shape
    rd = (gray < 128).sum(axis=1)
    rows = np.where(rd > W * 0.10)[0]
    if rows.size < 4:
        return None
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
    abs_groups = [g for g in groups if g[1] - g[0] >= 6]

    def pair_score(top, bot):
        """None unless heights match and gap is sane (MRZ-like pair)."""
        h1 = top[1] - top[0]
        h2 = bot[1] - bot[0]
        if abs(h1 - h2) > max(6, h1 // 2):
            return None
        gap = bot[0] - top[1]
        if gap > max(30, 2 * max(h1, h2)):
            return None
        return True

    def padded(top, bot):
        return [
            (max(0, g[0] - pad), min(H, g[1] + pad))
            for g in (top, bot)
        ]

    # Case A: the whole image IS a bare MRZ band (synthetic corpus
    # images, tight crops): two tall lines covering >25% of the page.
    # On tight crops (GUI PPM path, wide aspect ratio >4) the band IS
    # the page: 12px lines on a 50px crop never exceed 0.25*H, so relax
    # to 0.20*H there. Full pages keep the 0.25*H guard: the data zone
    # can contain two tall-ish text blocks (16c07a99: 101/99px bands)
    # that would otherwise be mispaired as MRZ lines.
    strip = (W / H) > 4.0
    thr_a = H * (0.20 if strip else 0.25)
    for i in range(len(abs_groups) - 1, 0, -1):
        top, bot = abs_groups[i - 1], abs_groups[i]
        if (top[1] - top[0]) > thr_a and (bot[1] - bot[0]) > thr_a:
            if pair_score(top, bot):
                return top[0], bot[1], padded(top, bot)

    # Case B: normal full page - two-line fingerprint in the bottom
    # third only (data-region text above must not participate; e.g.
    # b43dea2a has a 58/57px data block pair that mimics MRZ lines).
    bottom_start = int(H * 0.66)
    cands = [g for g in abs_groups if g[0] >= bottom_start]
    for i in range(len(cands) - 1, 0, -1):
        top, bot = cands[i - 1], cands[i]
        if pair_score(top, bot):
            return top[0], bot[1], padded(top, bot)

    # Case C: single-line fallback: lowest band in the bottom third,
    # else the lowest band on the page.
    g = None
    for cand in reversed(abs_groups):
        if cand[0] >= bottom_start:
            g = cand
            break
    if g is None and abs_groups:
        g = abs_groups[-1]
    if g is not None:
        h = g[1] - g[0]
        half = h // 2
        return g[0], g[1], [(max(0, g[0] - pad), g[0] + half + pad),
                            (max(0, g[0] + half - pad), g[1] + pad)]
    return None


def _rescale_line(pil_line, line_h: int):
    """Rescale every line to a canonical 24px glyph height.

    Measured (ocrb model): native real-photo lines (~12px) are
    unaffected by a 24px rescale (44/44 line2 kept), while synthetic
    corpus glyphs (48px bitmap templates) jump 21->40 / 10->36 chars
    correct. 24px is the sweet spot for both inputs; 32px hurts the
    photo case (29->25)."""
    from PIL import Image
    w, h = pil_line.size
    if h <= 0 or h == 24:
        return pil_line
    return pil_line.resize(
        (max(1, int(round(w * 24.0 / h))), 24), Image.LANCZOS)


def _init_api(tessdata_dir: Path, lang: str):
    from tesserocr import PyTessBaseAPI, PSM, OEM
    api = PyTessBaseAPI(path=str(tessdata_dir), lang=lang,
                        psm=PSM.SINGLE_LINE, oem=OEM.LSTM_ONLY)
    for d in ("system", "freq", "punc", "number", "bigram", "unambig"):
        api.SetVariable(f"load_{d}_dawg", "false")
    api.SetVariable("tessedit_char_whitelist", WHITELIST)
    api.SetVariable("textord_no_rejects", "1")
    api.SetVariable("preserve_interword_spaces", "0")
    api.SetVariable("lstm_choice_mode", "2")
    api.SetVariable("lstm_choice_iterations", "10")
    return api


def _recognize_line(api, pil_line, want_choices=False):
    """Recognize one line. Returns (text, [per-symbol conf 0..100],
    {col_idx: [alt chars]} -- the last one empty unless
    want_choices=True, which enables lstm_choice_mode=2 and returns
    per-timestep physical candidates (score ascending; the CTC blank
    is always the lowest-score entry and is skipped)."""
    from tesserocr import RIL
    api.SetImage(pil_line)
    api.Recognize()
    text = ""
    confs = []
    ri = api.GetIterator()
    while ri:
        try:
            txt = ri.GetUTF8Text(RIL.SYMBOL) or ""
        except RuntimeError:
            txt = ""
        text += txt
        confs.append(ri.Confidence(RIL.SYMBOL))
        if not ri.Next(RIL.SYMBOL):
            break
    choices = {}
    if want_choices:
        try:
            lst = api.GetBestLSTMSymbolChoices()
            if lst and lst[0]:
                for k, grp in enumerate(lst[0][:LINE_LEN]):
                    alts = []
                    for ch, sc in grp:
                        if ch not in WHITELIST or sc < 10.0:
                            continue  # skip CTC blank (score ~1.x)
                        if ch not in alts:
                            alts.append(ch)
                        if len(alts) >= 4:
                            break
                    if alts:
                        choices[k] = alts
        except Exception:
            pass
    return text, confs, choices


def _normalize(text: str, confs: list, line_no: int):
    """44-char fixed-length regularization + slot whitelist hard-fix.

    Returns (line44, confs44).
    """
    raw = "".join(ch for ch in text if ch in WHITELIST)
    confs = confs[: len(raw)]
    # Structural positions that must never be dropped when the line
    # is over-long: doc code first char (0), check digits (19/27/43),
    # sex (20).
    protect = {0, 19, 20, 27, 43}
    # 1) Strip trailing phantom letters (OCR hallucination past the
    #    filler zone, e.g. "...<<<<<<SAC"). The line tail is filler
    #    '<' plus a digit check char, so a trailing letter is noise.
    while len(raw) > LINE_LEN and raw[-1].isalpha():
        raw = raw[:-1]
        confs.pop()
    # 2) Drop slot-violating chars first (date-field non-digits on
    #    line 2, non-letters in the doc-code zone on line 1), then
    #    non-filler low-confidence chars, then '<' fillers.
    while len(raw) > LINE_LEN:
        if len(confs) < len(raw):
            confs = confs + [0] * (len(raw) - len(confs))

        def _violates(i: int, c: str) -> bool:
            if line_no == 2 and i in _DATE_SLOTS:
                return c not in "0123456789OIBZS"
            if line_no == 1 and i < 5:
                return c not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ<"
            return False

        cands = [(i, c, confs[i]) for i, c in enumerate(raw)
                 if i not in protect]
        if not cands:
            break
        cands.sort(key=lambda t: (not _violates(t[0], t[1]), t[1] == "<", t[2]))
        i = cands[0][0]
        raw = raw[:i] + raw[i + 1:]
        confs.pop(i)
    while len(raw) < LINE_LEN:
        raw += "<"
        confs.append(0)
    # Slot whitelist hard-fix.
    line = list(raw)
    if line_no == 1:
        # line 1: pos0-1 doc code (P<, PM, PS, Q< ...), pos2-4 issuing
        # state letters. Only guard chars that MUST be letters; never
        # force '<' (PM/PS/Q< are valid ICAO doc codes).
        for i in range(0, 5):
            if line[i] not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ<":
                line[i] = "<"
    else:
        # line 2: date fields (dob 13-18, expiry 21-26) and their
        # check digits (19, 27) are hard digit slots. Document number
        # (0-9), composite and personal numbers may legitimately
        # contain letters, so they are left untouched.
        for i in set(range(13, 20)) | set(range(21, 28)):
            c = line[i]
            if c in "OIBZS":
                line[i] = AMBIG[c]
        # nationality (10-12) must be letters
        for i in range(10, 13):
            if line[i] not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
                line[i] = "<"
        # sex (20) must be M/F/<
        if line[20] not in "MF<":
            line[20] = "<"
    return "".join(line), confs


def _composite_ok(l2: str) -> bool:
    """ICAO 9303 composite: cd(l2[0:10]+l2[13:20]+l2[21:43]) == l2[43]."""
    if len(l2) != LINE_LEN or not l2[43].isdigit():
        return False
    comp = l2[0:10] + l2[13:20] + l2[21:43]
    s = 0
    for i, c in enumerate(comp):
        if "0" <= c <= "9":
            v = ord(c) - 48
        elif "A" <= c <= "Z":
            v = ord(c) - 55
        elif c == "<":
            v = 0
        else:
            return False
        s += v * (7, 3, 1)[i % 3]
    return s % 10 == ord(l2[43]) - 48


def _repair_line2(l2: str, confs: list, choices: dict = None) -> str:
    """Composite-checksum beam search.

    Search order (all candidates verified against the TD3 composite
    formula, 7-3-1 weighted):
      1. optional-data zone (col 28-42) single flips, lowest conf first
      2. whole-line single flips, lowest conf first
      3. optional-data zone pairs
      4. whole-line pairs over the 6 least-confident slots
    Candidate chars come from the LSTM physical alternatives
    (lstm_choice_mode=2) when present, merged with the static
    confusion map (0/<, O/D, 1/I/L, 8/B ...)."""
    if _composite_ok(l2):
        return l2

    def cands(i):
        out = []
        if choices and i in choices:
            out += [c for c in choices[i] if c in WHITELIST]
        out += CONFUSION.get(l2[i], [])
        seen = []
        for c in out:
            if c != l2[i] and c not in seen:
                seen.append(c)
        return seen

    def try_flip(ia, ca, ib=None, cb=None):
        cand = list(l2)
        cand[ia] = ca
        if ib is not None:
            cand[ib] = cb
        s = "".join(cand)
        return s if _composite_ok(s) else None

    zone = sorted(range(28, 43), key=lambda i: confs[i])
    # 1) optional-data zone singles
    for i in zone:
        for c in cands(i):
            s = try_flip(i, c)
            if s:
                return s
    # 2) whole-line singles
    order = sorted(range(LINE_LEN), key=lambda i: confs[i])
    for i in order:
        for c in cands(i):
            s = try_flip(i, c)
            if s:
                return s
    # 3) optional-data zone pairs
    for a in range(len(zone)):
        ia = zone[a]
        for ca in cands(ia):
            for b in range(a + 1, len(zone)):
                ib = zone[b]
                for cb in cands(ib):
                    s = try_flip(ia, ca, ib, cb)
                    if s:
                        return s
    # 4) whole-line pairs (6 least confident)
    for a in range(len(order)):
        ia = order[a]
        for ca in cands(ia):
            for b in range(a + 1, min(len(order), 6)):
                ib = order[b]
                for cb in cands(ib):
                    s = try_flip(ia, ca, ib, cb)
                    if s:
                        return s
    return l2


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: tesseract_tool.py <image>", file=sys.stderr)
        return 2
    img_path = Path(argv[1])
    if not img_path.exists():
        print(f"cannot open {img_path}", file=sys.stderr)
        return 2

    tool_dir = Path(__file__).resolve().parent
    tessdata = tool_dir / "data"
    lang = "ocrb" if (tessdata / "ocrb.traineddata").exists() else "eng"

    t0 = time.perf_counter()
    try:
        gray, img = _load_gray(img_path)
    except Exception as e:
        print(f"result.ok       : cannot load image: {e}", file=sys.stderr)
        return 1

    H, W = gray.shape
    found = _mrz_lines(gray)
    if found is None:
        print(f"result.ok       : MRZ band not found")
        print(f"result.line1    : ")
        print(f"result.line2    : ")
        print(f"result.conf1    : 0")
        print(f"result.conf2    : 0")
        print(f"band.x band.y band.w band.h : 0 0 {W} {H}")
        return 1
    y0, y1, lines = found
    band_box = f"0 {y0} {W} {y1 - y0}"

    try:
        api = _init_api(tessdata, lang)
    except Exception as e:
        print(f"result.ok       : tesseract init failed: {e}",
              file=sys.stderr)
        return 1

    results = []
    try:
        for idx, (lt, lb) in enumerate(lines):
            strip = img.crop((0, lt, W, lb))
            strip = _rescale_line(strip, lb - lt)
            text, confs, choices = _recognize_line(
                api, strip, want_choices=(idx == 1))
            line44, confs44 = _normalize(text, confs, len(results) + 1)
            avg_conf = int(sum(confs44) / max(1, len(confs44)))
            results.append((line44, avg_conf, confs44, choices))
    except Exception as e:
        print(f"result.ok       : OCR failed: {e}", file=sys.stderr)
        return 1
    finally:
        try:
            api.End()
        except Exception:
            pass

    line1, conf1, _, _ = results[0]
    line2, conf2, confs2, choices2 = results[1]
    if not _composite_ok(line2):
        line2 = _repair_line2(line2, confs2, choices2)
    # Leading '<' with a letter behind it usually means the first glyph
    # was misread ('<P...' for 'P<...'); swap the pair so the gate can
    # re-evaluate with a valid doc code.
    if len(line1) >= 2 and line1[0] == "<" and line1[1].isalpha():
        line1 = line1[1] + "<" + line1[2:]

    # Gate (ICAO 9303 Part 7 doc codes, unbounded):
    # 1. first char must be an uppercase letter (P V A C D R I S T Q B
    #    are the traditional core categories -> direct pass)
    # 2. otherwise verify col 2-4 is a valid ISO 3166-1 alpha-3
    #    issuing state / agency code ('<'-padded, e.g. D<<, UN<)
    def _is_valid(l1, l2):
        if len(l1) != LINE_LEN or len(l2) != LINE_LEN:
            return False
        if not (l1[0].isalpha() and l1[0].isupper()):
            return False
        if l1[0] in "PVACDRISTQB":
            pass
        else:
            iso = l1[2:5].replace("<", "")
            if iso not in ISO_CODES:
                return False
        return _composite_ok(l2)

    ok = _is_valid(line1, line2)
    ms = int((time.perf_counter() - t0) * 1000)
    print(f"result.ok       : {'OK' if ok else 'unreadable'}")
    print(f"result.line1    : {line1}")
    print(f"result.line2    : {line2}")
    print(f"result.conf1    : {conf1}")
    print(f"result.conf2    : {conf2}")
    print(f"band.x band.y band.w band.h : {band_box}")
    if not ok:
        print(f"// {ms} ms, lang={lang}", file=sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except Exception as e:
        print(f"tesseract_tool: {e}", file=sys.stderr)
        sys.exit(1)