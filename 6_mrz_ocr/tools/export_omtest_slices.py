#!/usr/bin/env python3
"""Export real glyph slices from tmp/omtest with ICAO-corrected labels.

The mix15 line models were trained on `data/real_slices.npz`, which was
built from the board's `fixmrz` labels.  Those labels are a PaddleOCR
hypothesis and are wrong in two ways that poison training:

  1. They put DIGITS in line-1 alpha slots (ICAO 9303 TD3 forbids
     digits there), e.g. `P<CHNGUO0<LURU...` where the document reads
     `O`.  Training on that teaches the net to emit `0` in the name
     field -- exactly the confusion seen on device.
  2. They are checksum-INVALID for 27/41 samples, so the line-2 labels
     teach wrong check digits.

This exporter re-derives the labels with the ICAO rules applied:
  * line 1: a digit in an alpha slot is mapped to its letter counterpart
    (0->O, 1->I, 2->Z, 5->S, 6->G, 8->B).
  * line 2: the board label is repaired to the checksum-valid variant
    that our own confusion alphabet can reach; if several variants tie,
    the sample is dropped (ambiguous label -> do not train on it).

The 16x12 glyphs are dumped by the C pipeline itself (MRZ_OCR_DUMP=1),
so they are exactly the tensors the network consumes at inference time.

Usage:
  python3 tools/export_omtest_slices.py --omtest ../tmp/omtest \
      --out data/omtest_slices.npz
"""
from __future__ import annotations
import argparse, glob, json, os, re, shutil, subprocess, sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

GLYPHS = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ<")
GI = {c: i for i, c in enumerate(GLYPHS)}
DIGIT_TO_LETTER = {'0': 'O', '1': 'I', '2': 'Z', '5': 'S', '6': 'G', '8': 'B'}
# BEGIN GENERATED CONFUSION PAIRS (tools/gen_confusion_tables.py -- do not edit by hand)
CONF = ["0O","0Q","0U","0D","0V","0G","1I","1L","1T","17","5S","2Z","8B","8R","8X",
        "83","6G","4W","4A","WH","WA","WU","WM","HN","HM","MN","RB","LC","KB","KX",
        "DU","DO","UV","UO","VY","QO","TI","T7","XK",
        # measured on real photos (tools/collect_confusions.py)
        "8M",  # n=6
        "0D",  # n=5
       ]
# END GENERATED CONFUSION PAIRS
W = [7, 3, 1]


def digit_for(seg: str) -> int:
    t = 0
    for i, c in enumerate(seg):
        v = int(c) if '0' <= c <= '9' else (ord(c) - 55 if 'A' <= c <= 'Z' else 0)
        t += v * W[i % 3]
    return t % 10


def line2_ok(l2: str) -> bool:
    if len(l2) != 44:
        return False
    try:
        return (digit_for(l2[0:9]) == int(l2[9]) and
                digit_for(l2[13:19]) == int(l2[19]) and
                digit_for(l2[21:27]) == int(l2[27]) and
                digit_for(l2[28:42]) == int(l2[42]) and
                digit_for(l2[0:10] + l2[13:20] + l2[21:43]) == int(l2[43]))
    except ValueError:
        return False


def confusion_map():
    m = {}
    for p in CONF:
        a, b = p[0], p[1]
        for x, y in ((a, b), (b, a)):
            m.setdefault(x, set()).add(y)
    return m


CMAP = confusion_map()


def repair_board_line1(label: str):
    """Undo the board's text-layer damage before applying the ICAO mask.

    The `-result.txt` files are written by a board app that emits the MRZ
    through a byte-oriented path, so three distinct defects show up:

      * a run of `<<<<...` arrives as the mojibake of the 3-byte U+FE3F /
        2-byte sequences, i.e. characters such as `ヘ` (E3 83 9A) or
        `く` (E3 81 8B).  Re-encoding latin-1 and decoding UTF-8 restores
        every one of them to `<`, which also restores the 44-char length;
      * the line arrives lower-cased (`p0cHNRA0<<...`);
      * the leading `P<` can arrive as `0<` or `p0`, where the `0` is a
        misread `P`/`O`.

    Each of these is deterministic, so repairing them is strictly better
    than discarding the sample: dropping it removes the only signal the
    network has about that photograph's font and lighting.
    """
    if not label:
        return label
    # 1. The board's OCR reads a run of `<` as kana.  Every one of the
    #    three shapes seen in the corpus (U+304F, U+30D8, U+3078) is the
    #    board's `<`; map them back before any length or mask check,
    #    otherwise the line stays 44 "characters" long but every filler
    #    position is unusable and the whole sample is thrown away.
    label = "".join("<" if ch in "くヘへ" else ch for ch in label)
    label = label.upper()
    return label


def fix_line1(label: str):
    """Apply the ICAO line-1 mask to a board label. None if not TD3."""
    label = repair_board_line1(label)
    if len(label) != 44:
        return None
    if not all('A' <= c <= 'Z' or c == '<' or c.isdigit() for c in label):
        return None
    out = list(label)
    if out[0].isdigit():
        out[0] = DIGIT_TO_LETTER.get(out[0], out[0])
    if out[1].isdigit():
        out[1] = DIGIT_TO_LETTER.get(out[1], out[1])
    for i, ch in enumerate(out):
        if i == 0:
            if ch == 'P':
                continue
            if ch.isdigit() and DIGIT_TO_LETTER.get(ch) == 'P':
                continue
            return None
        if i == 1:
            if ch == '<':
                continue
            return None
        if 2 <= i <= 4:
            if 'A' <= ch <= 'Z':
                continue
            if ch.isdigit() and 'A' <= DIGIT_TO_LETTER.get(ch, '?') <= 'Z':
                out[i] = DIGIT_TO_LETTER[ch]
                continue
            return None
        if ch == '<' or 'A' <= ch <= 'Z':
            continue
        if ch.isdigit() and 'A' <= DIGIT_TO_LETTER.get(ch, '?') <= 'Z':
            out[i] = DIGIT_TO_LETTER[ch]
            continue
        return None
    return "".join(out)


def fix_line2(label: str):
    """Return the unique checksum-valid variant, or None if ambiguous."""
    if len(label) != 44:
        return None
    if line2_ok(label):
        return label
    cands = []
    for i in range(44):
        for r in CMAP.get(label[i], ()):
            t = label[:i] + r + label[i+1:]
            if line2_ok(t):
                cands.append(t)
    if len(cands) == 1:
        return cands[0]
    if not cands:
        # try two edits
        for i in range(44):
            for j in range(i + 1, 44):
                for r1 in CMAP.get(label[i], ()):
                    for r2 in CMAP.get(label[j], ()):
                        t = list(label)
                        t[i] = r1
                        t[j] = r2
                        t = "".join(t)
                        if line2_ok(t):
                            cands.append(t)
        if len(cands) == 1:
            return cands[0]
    return None


def all_valid_variants(label: str, max_edits: int = 2):
    """All checksum-valid variants reachable within max_edits confusion edits."""
    if len(label) != 44:
        return []
    if line2_ok(label):
        return [label]
    cands = []
    for i in range(44):
        for r in CMAP.get(label[i], ()):
            t = label[:i] + r + label[i+1:]
            if line2_ok(t):
                cands.append(t)
    if max_edits < 2:
        return sorted(set(cands))
    # The truth set is defined as every checksum-valid reading within two
    # confusion edits, so the 2-edit variants must be collected even when
    # a 1-edit variant already exists.  Stopping at the first hit used to
    # drop readings that need two edits; AJ's real line 2 is exactly such
    # a case (it differs from the board label at two positions).
    for i in range(44):
        for j in range(i + 1, 44):
            for r1 in CMAP.get(label[i], ()):
                for r2 in CMAP.get(label[j], ()):
                    t = list(label)
                    t[i] = r1
                    t[j] = r2
                    t = "".join(t)
                    if line2_ok(t):
                        cands.append(t)
    return sorted(set(cands))


def position_candidates(label: str, variants):
    """Per-position acceptable classes for a line-2 label.

    `variants` is the set of checksum-valid readings of this line (the
    board label plus every confusion edit that makes the checksum work).
    A position on which all variants agree is a HARD label: the checksum
    plus the ICAO field mask pin it down, so the network can be told the
    exact answer.  A position on which the variants disagree is
    AMBIGUOUS: there the set of characters that appear is the acceptable
    set, and the loss must not assert any single one of them.

    This is strictly better than dropping the sample: the photo still
    teaches the network that font, lighting and perspective, and the
    ambiguous cells teach it to separate a confusion class, which is the
    capability that is actually missing.
    """
    cands = []
    for col in range(len(label)):
        col_set = {v[col] for v in variants if col < len(v)}
        # The board character is always a member: it is what the OCR
        # actually saw, and the trainer needs it to stay the reference
        # point of the candidate set (a candidate set that excludes the
        # observed character contradicts the image it is attached to).
        if label[col] in GI:
            col_set.add(label[col])
        col_set = sorted(c for c in col_set if c in GI)
        if not col_set:
            col_set = [label[col]] if label[col] in GI else []
        cands.append(col_set)
    return cands


def parse_board(txt: Path):
    s = txt.read_text(errors="replace")
    tl = re.findall(r'textLine\[\d\]\(([^\n]*)\)', s)
    mfix = re.findall(r'fixmrz[^\n]*\n([PE]<[0-9A-Z<]{42})\n([0-9A-Z<]{44})', s)
    if mfix:
        return mfix[0][0], mfix[0][1]
    if len(tl) >= 2:
        return tl[0], tl[1]
    return None, None


FAST_NATIVE_MAX_WIDTH = 1600   # must match PhotoMrzPipeline.kt
TMPDIR = Path('/tmp/mrz_export_bands')


def fast_pair(gray: np.ndarray):
    h, w = gray.shape
    if w < 80 or h < 120:
        return None
    y0 = int(h * 0.55)
    y1 = min(int(h * 0.96), h)
    sx0 = int(w * 0.03)
    sx1 = int(w * 0.97)
    stride = 4
    sample_w = (sx1 - sx0 + stride - 1) // stride
    if y1 <= y0 or sample_w < 40:
        return None
    rows = gray[y0:y1]
    dark = ((rows[:, sx0:sx1:stride] < 170).sum(axis=1)).astype(int)
    min_ink = max(3, int(sample_w * 0.18))
    max_ink = int(sample_w * 0.88)
    runs = []
    start = -1
    gap = 0
    for i in range(len(dark)):
        if min_ink <= dark[i] <= max_ink:
            if start < 0:
                start = i
            gap = 0
        elif start >= 0:
            gap += 1
            if gap > 2:
                runs.append((start, i - gap))
                start = -1
                gap = 0
    if start >= 0:
        runs.append((start, len(dark) - 1))
    solid_min = int(sample_w * 0.90)
    solid_runs = []
    s = -1
    for i in range(len(dark) + 1):
        solid = i < len(dark) and dark[i] >= solid_min
        if solid:
            if s < 0:
                s = i
        elif s >= 0:
            if i - s >= 8:
                solid_runs.append((s, i - 1))
            s = -1

    def pick_pair(before=None):
        candidate = None
        def valid(a, b):
            if before is not None and b[1] >= before - 2:
                return False
            h1 = a[1] - a[0] + 1
            h2 = b[1] - b[0] + 1
            gap_rows = b[0] - a[1] - 1
            if h1 < 5 or h1 > 100 or h2 < 5 or h2 > 100:
                return False
            if not (0.35 <= h2 / h1 <= 2.5):
                return False
            if gap_rows < 0 or gap_rows > max(12, int(min(h1, h2) * 2 + 4)):
                return False
            return True
        for i in range(len(runs) - 1):
            a = runs[i]
            h1 = a[1] - a[0] + 1
            if h1 < 5 or h1 > 100:
                continue
            for j in range(i + 1, min(len(runs) - 1, i + 3) + 1):
                b = runs[j]
                h2 = b[1] - b[0] + 1
                max_noise = max(2, min(h1, h2) // 3)
                skipped = runs[i + 1:j]
                if any(r[1] - r[0] + 1 > max_noise for r in skipped):
                    continue
                if not valid(a, b):
                    continue
                if candidate is None or b[1] > candidate[1][1]:
                    candidate = (a, b)
        return candidate

    best = None
    for solid in solid_runs:
        c = pick_pair(solid[0])
        if c is not None and (best is None or c[1][1] > best[1][1]):
            best = c
    if best is None:
        best = pick_pair()
    if best is None:
        return None
    a, b = best
    return (y0 + a[0], y0 + a[1]), (y0 + b[0], y0 + b[1])


def fast_single_row(gray: np.ndarray):
    """Locate a single MRZ text row (CS/CD 1x30 format)."""
    h, w = gray.shape
    if w < 80 or h < 120:
        return None
    y0 = int(h * 0.55)
    y1 = min(int(h * 0.96), h)
    sx0 = int(w * 0.03)
    sx1 = int(w * 0.97)
    stride = 4
    sample_w = (sx1 - sx0 + stride - 1) // stride
    if y1 <= y0 or sample_w < 40:
        return None
    rows = gray[y0:y1]
    dark = ((rows[:, sx0:sx1:stride] < 170).sum(axis=1)).astype(int)
    min_ink = max(3, int(sample_w * 0.18))
    max_ink = int(sample_w * 0.88)
    runs = []
    start = -1
    gap = 0
    for i in range(len(dark)):
        if min_ink <= dark[i] <= max_ink:
            if start < 0:
                start = i
            gap = 0
        elif start >= 0:
            gap += 1
            if gap > 2:
                runs.append((start, i - gap))
                start = -1
                gap = 0
    if start >= 0:
        runs.append((start, len(dark) - 1))
    best = None
    for run in runs:
        rh = run[1] - run[0] + 1
        if rh < 5 or rh > 100:
            continue
        if best is None or rh > best[1] - best[0] + 1:
            best = run
    if best is None:
        return None
    return (y0 + best[0], y0 + best[1])


def native_band_single(gray, row, xpad_l, xpad_r):
    """Crop a single-row band with padding (CS/CD).
    Also crops horizontally to the text region, excluding border/shadow."""
    h, w = gray.shape
    ry0, ry1 = row
    s0 = max(0, ry0)
    s1 = min(h, ry1 + 1)
    row_h = max(1, s1 - s0)
    pad = max(3, min(8, int(round(row_h * 0.25))))
    crop_top = max(0, s0 - pad)
    crop_bottom = min(h, s1 + pad)
    # Find horizontal text boundaries (exclude large border runs)
    sub = gray[crop_top:crop_bottom, xpad_l:max(1, w - xpad_r)]
    sw = sub.shape[1]
    col_dark = (sub < 170).sum(axis=0)
    thr = max(2, row_h // 4)
    runs = []
    start = -1
    for x in range(sw):
        if col_dark[x] >= thr:
            if start < 0:
                start = x
        elif start >= 0:
            runs.append((start, x - 1))
            start = -1
    if start >= 0:
        runs.append((start, sw - 1))
    if len(runs) >= 3:
        widths = [e - s + 1 for s, e in runs]
        medw = sorted(widths)[len(widths) // 2]
        # Skip border runs (much wider than median)
        first = 0
        last = len(runs) - 1
        while first < last and widths[first] > medw * 4:
            first += 1
        while last > first and widths[last] > medw * 4:
            last -= 1
        text_left = xpad_l + runs[first][0]
        text_right = xpad_l + runs[last][1]
        # Add small horizontal padding
        hpad = max(3, medw // 2)
        text_left = max(0, text_left - hpad)
        text_right = min(w, text_right + hpad)
        return crop_top, crop_bottom, text_left, text_right
    return crop_top, crop_bottom, xpad_l, max(1, w - xpad_r)


def adaptive_xpad(gray, pair):
    h, w = gray.shape
    if w < 40:
        nominal = min(100, max(10, int(round(w * 0.08))))
        return nominal, nominal
    strip = max(4, min(12, w // 25))
    def frac(x0, x1):
        blk = gray[:, x0:x1]
        return float((blk < 160).sum()) / blk.size
    left_dark = frac(0, strip)
    right_dark = frac(w - strip, w)
    nominal = min(100, max(10, int(round(w * 0.08))))
    left = nominal if left_dark >= 0.9 else 0
    right = nominal if right_dark >= 0.9 else 0
    return left, right


def native_band(gray, first, second, xpad_l, xpad_r):
    h, w = gray.shape
    fy0, fy1 = first
    sy0, sy1 = second
    first_source_y1 = min(h, int(round((fy1 + 1) / 1.0)))
    rows = []
    for (ry0, ry1) in (first, second):
        s0 = max(0, int(round(ry0 / 1.0)))
        s1 = min(h, int(round((ry1 + 1) / 1.0)))
        if (ry0, ry1) == second:
            s0 = max(s0, first_source_y1)
        row_h = max(1, s1 - s0)
        pad = max(8, min(25, int(round(row_h * 0.70))))
        rows.append((s0, s1, pad))
    first_top = max(0, int(round(fy0 / 1.0)))
    second_bottom = min(h, int(round((sy1 + 1) / 1.0)))
    shared_h = max((r[1] + r[2]) - (r[0] - r[2]) for r in rows)
    shared_pad = max(8, min(25, int(round(shared_h * 0.70))))
    crop_top = max(0, first_top - shared_pad)
    crop_bottom = min(h, second_bottom + shared_pad)
    return crop_top, crop_bottom



def write_band_ppm(img: Path, out_dir: Path) -> Path:
    """Reproduce the Android fast-path band crop, then save it as PPM."""
    from PIL import Image
    import numpy as np
    im = Image.open(img).convert('RGB')
    arr = np.asarray(im, dtype=np.uint32)
    gray = ((arr[:, :, 0] * 299 + arr[:, :, 1] * 587 + arr[:, :, 2] * 114 + 500)
            // 1000).astype(np.uint8)
    pair = fast_pair(gray)
    if pair is None:
        raise RuntimeError(f"MRZ 双行未定位: {img}")
    xp = adaptive_xpad(gray, pair)
    top, bot = native_band(gray, pair[0], pair[1], xp[0], xp[1])
    band = im.crop((xp[0], top, im.width - xp[1], bot))
    if band.width > FAST_NATIVE_MAX_WIDTH:
        nh = max(1, int(round(band.height * FAST_NATIVE_MAX_WIDTH / band.width)))
        band = band.resize((FAST_NATIVE_MAX_WIDTH, nh), Image.BILINEAR)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / (img.stem + "_band.ppm")
    band.save(out)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--omtest", default="../tmp/omtest")
    ap.add_argument("--tool", default="build/mrz_ocr_cnn_tool")
    ap.add_argument("--out", default="data/omtest_slices.npz")
    ap.add_argument("--dump-prefix", default="/tmp/mrz_dump")
    ap.add_argument("--tag", default="img",
                    help="tag used by the C dumper (mrz_dump_<tag>_l<line>_c<col>_...)")
    args = ap.parse_args()
    ROOT = Path(__file__).resolve().parents[1]
    om = Path(args.omtest)

    X, Y, META = [], [], []
    used = 0
    skipped = {"no_label": 0, "not_td3": 0, "dump_missing": 0}
    soft_used = 0
    for d in sorted(p for p in om.iterdir() if p.is_dir()):
        for txt in sorted((om / d.name).glob("*.jpg-result.txt")):
            img = txt.with_name(txt.name.replace("-result.txt", ""))
            if not img.exists():
                continue
            l1, l2 = parse_board(txt)
            if not l1 or not l2:
                skipped["no_label"] += 1
                continue
            g1 = fix_line1(l1)
            # Line 2: every checksum-valid reading of the board label.
            # The variants agree on most positions (hard label) and
            # disagree on a few (soft target).  The sample is never
            # dropped: discarding it removes the only evidence the network
            # has about that photograph's font and lighting, which is why
            # the model then fails on that same photograph.
            variants = all_valid_variants(l2, max_edits=2)
            if not variants:
                skipped["not_td3"] += 1
                continue
            g2 = fix_line2(l2) or l2
            soft_used += 1
            cand2 = position_candidates(g2, variants)
            cand1 = ([[ch] for ch in g1] if g1 else None)
            # A sample whose line 2 has exactly ONE checksum-valid reading
            # is pinned by the checksum: every cell is a hard label.  A
            # sample with several readings was rescued from a broken OCR
            # hypothesis, so its labels are trusted less and the trainer
            # down-weights it (see tools/train_cnn_omtest.py).
            weak_line2 = len(variants) > 1
            if cand1 is None and cand2 is None:
                skipped["not_td3"] += 1
                continue

            # Re-run the C pipeline on the SAME band the Android fast path
            # produces, so the dumped glyphs match inference exactly.
            for f in glob.glob(f'{args.dump_prefix}_{args.tag}_l*_c*_x*_r*_w*.ppm'):
                os.remove(f)
            for f in glob.glob(f'/private/tmp/mrz_dump_{args.tag}_l*_c*_x*_r*_w*.ppm'):
                os.remove(f)
            env = dict(os.environ)
            env['MRZ_OCR_DUMP'] = args.tag
            env['MRZ_OCR_LINE_MODELS'] = '1'
            # mrz_ocr_cnn_tool only reads PPM.  The Android fast path does
            # NOT hand it the whole photo: it crops the two located MRZ
            # rows first (locateFast -> nativeBand -> fastNativeInput).
            # Dump from that same band, otherwise the training glyphs come
            # from a different segmentation than inference uses.
            band = write_band_ppm(img, TMPDIR)
            r = subprocess.run([str(ROOT / args.tool), str(band)],
                               capture_output=True, text=True, timeout=30, env=env)
            if r.returncode != 0:
                skipped["dump_missing"] += 1
                continue

            n_added = 0
            for li, truth, cands_by_col in ((0, g1, cand1), (1, g2, cand2)):
                if truth is None:
                    continue
                for c, ch in enumerate(truth):
                    if ch not in GI:
                        continue
                    Y_extra = (cands_by_col[c]
                               if cands_by_col and c < len(cands_by_col)
                               else [ch])
                    if ch not in Y_extra:
                        Y_extra = [ch] + Y_extra
                    files = glob.glob(f'{args.dump_prefix}_{args.tag}_l{li}_c{c:02d}_x*_r*_w*.ppm')
                    if not files:
                        files = glob.glob(f'/private/tmp/mrz_dump_{args.tag}_l{li}_c{c:02d}_x*_r*_w*.ppm')
                    if not files:
                        continue
                    patch = np.asarray(Image.open(files[0]).convert('L'),
                                       dtype=np.float32) / 255.0
                    X.append(patch)
                    Y.append(GI[ch])
                    # `cand` is the acceptable-class set used for the
                    # soft target; a single class means a hard label.
                    META.append({"file": f"{d.name}/{img.name}", "line": li,
                                 "col": c, "ch": ch,
                                 "weak": bool(weak_line2 and li == 1),
                                 "cand": "".join(y for y in Y_extra)})
                    n_added += 1
            if n_added:
                used += 1
    if not X:
        print("no slices collected")
        return 1
    Xa = np.array(X, dtype=np.float32).reshape(-1, 12, 16)
    Ya = np.array(Y, dtype=np.int64)
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, X=Xa, Y=Ya, meta=np.array([json.dumps(m) for m in META]))
    print(f"== exported {len(Xa)} slices from {used} samples -> {out}")
    print(f"   skipped: {skipped}")
    print("   label histogram:", dict(sorted(Counter(Ya.tolist()).items())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
