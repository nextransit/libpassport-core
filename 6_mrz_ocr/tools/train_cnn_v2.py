#!/usr/bin/env python3
"""Train the Tiny ConvNet OCR-B classifier and emit float32 C weights.

v2 IMPROVEMENTS over train_cnn.py (root-cause fixes for 0/O, 1/I, 5/S,
2/Z, ... confusions seen in the 500-image corpus benchmark):

  1. **Train/inference distribution match.** The C pipeline resamples
     each char by area-averaging a binary 8*scale x 12*scale ink box
     down to 16x12 (then 3x3 box smooth + centroid shift). Old
     `render_augment` used bilinear-translate on a 4x upscale which
     produced 40%+ "middle values" never seen at inference. v2 draws
     ink into a real w x h box with the SAME `w/h` distribution as the
     corpus (scale=3..6, hgap variance), applies the same area-average
     resample, then the same 3x3 box smooth.
  2. **Skew augmentation.** Skew shifts row y by +y*skew/scale px
     when sampling, mirroring gen_mrz_image's global skew.
  3. **Noise augmentation.** Salt/pepper at the binary-ink level
     BEFORE resample, matching gen_mrz_image's noise probability
     (0..0.05 range).
  4. **Centroid pre-shift.** Sample-level centroid pre-aligns each
     glyph so the network never has to compensate for the inference
     centroid stage (training distribution already matches).
  5. **Expanded hard-pair list.** Covers every (gt, got) confusion
     seen at scale in the corpus: 0/O, 1/I, 2/Z, 3/D, 4/A, 5/S, 6/O,
     7/Z, 8/O, 9/S, B/8, F/6, G/6, P/7, J/3, E/5, R/8, T/6, U/V,
     N/M, Q/O, K/X.
  6. **Bigger sample budget + longer schedule.** 1500/class * 37 +
     extras for hard pairs; cosine LR; up to 60 epochs; 95/5
     train/val split with best-val selection.

Architecture unchanged (must match cnn.h):
  input : 12 rows x 16 cols grayscale float (raw 0..1, soft edges)
  conv1 : 3x3, 8 filters, stride 1, pad 1 -> ReLU
  pool1 : 2x1 max (height only)            -> 6 x 16 x 8
  conv2 : 3x3, 16 filters, stride 1, pad 1 -> ReLU
  pool2 : 2x2 max                          -> 3 x 8 x 16
  conv3 : 1x1, 4 filters                   -> 3 x 8 x 4
  fc1   : 96  -> 64, ReLU
  fc2   : 64  -> 37 (softmax)
"""
from __future__ import annotations
import math, random, os
from pathlib import Path
import numpy as np

# ---- OCR-B template bank (must match src/template.c exactly) ----
GLYPHS = [
    ('0', [0x3C, 0x66, 0x66, 0x6E, 0x76, 0x66, 0x66, 0x66, 0x66, 0x6E, 0x66, 0x3C]),
    ('1', [0x18, 0x38, 0x78, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x7E]),
    ('2', [0x3C, 0x66, 0x66, 0x06, 0x06, 0x0C, 0x18, 0x30, 0x60, 0x60, 0x60, 0x7E]),
    ('3', [0x3C, 0x66, 0x66, 0x06, 0x06, 0x1C, 0x06, 0x06, 0x06, 0x66, 0x66, 0x3C]),
    ('4', [0x0C, 0x1C, 0x3C, 0x6C, 0x6C, 0x66, 0x66, 0x7E, 0x06, 0x06, 0x06, 0x06]),
    ('5', [0x7E, 0x60, 0x60, 0x60, 0x7C, 0x66, 0x06, 0x06, 0x06, 0x66, 0x66, 0x3C]),
    ('6', [0x3C, 0x66, 0x60, 0x60, 0x7C, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x3C]),
    ('7', [0x7E, 0x66, 0x06, 0x06, 0x0C, 0x18, 0x18, 0x18, 0x18, 0x30, 0x30, 0x30]),
    ('8', [0x3C, 0x66, 0x66, 0x66, 0x66, 0x3C, 0x66, 0x66, 0x66, 0x66, 0x66, 0x3C]),
    ('9', [0x3C, 0x66, 0x66, 0x66, 0x66, 0x66, 0x3E, 0x06, 0x06, 0x06, 0x66, 0x3C]),
    ('A', [0x18, 0x18, 0x3C, 0x3C, 0x66, 0x66, 0x7E, 0x7E, 0x66, 0x66, 0x66, 0x66]),
    ('B', [0x7C, 0x66, 0x66, 0x66, 0x66, 0x7C, 0x66, 0x66, 0x66, 0x66, 0x66, 0x7C]),
    ('C', [0x3C, 0x66, 0x66, 0x60, 0x60, 0x60, 0x60, 0x60, 0x60, 0x66, 0x66, 0x3C]),
    ('D', [0x78, 0x6C, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x6C, 0x78]),
    ('E', [0x7E, 0x60, 0x60, 0x60, 0x60, 0x7C, 0x60, 0x60, 0x60, 0x60, 0x60, 0x7E]),
    ('F', [0x7E, 0x60, 0x60, 0x60, 0x60, 0x7C, 0x60, 0x60, 0x60, 0x60, 0x60, 0x60]),
    ('G', [0x3C, 0x66, 0x66, 0x60, 0x60, 0x60, 0x6E, 0x66, 0x66, 0x66, 0x66, 0x3C]),
    ('H', [0x66, 0x66, 0x66, 0x66, 0x66, 0x7E, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66]),
    ('I', [0x3C, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x3C]),
    ('J', [0x3E, 0x0C, 0x0C, 0x0C, 0x0C, 0x0C, 0x0C, 0x0C, 0x0C, 0x6C, 0x6C, 0x38]),
    ('K', [0x66, 0x6C, 0x78, 0x70, 0x78, 0x6C, 0x66, 0x66, 0x66, 0x66, 0x6C, 0x66]),
    ('L', [0x60, 0x60, 0x60, 0x60, 0x60, 0x60, 0x60, 0x60, 0x60, 0x60, 0x60, 0x7E]),
    ('M', [0x66, 0x7E, 0x7E, 0x7E, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66]),
    ('N', [0x66, 0x66, 0x76, 0x7E, 0x7E, 0x6E, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66]),
    ('O', [0x3C, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x3C]),
    ('P', [0x7C, 0x66, 0x66, 0x66, 0x66, 0x7C, 0x60, 0x60, 0x60, 0x60, 0x60, 0x60]),
    ('Q', [0x3C, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x6E, 0x6C, 0x36, 0x1E]),
    ('R', [0x7C, 0x66, 0x66, 0x66, 0x66, 0x7C, 0x6C, 0x66, 0x66, 0x66, 0x66, 0x66]),
    ('S', [0x3E, 0x60, 0x60, 0x60, 0x3C, 0x06, 0x06, 0x06, 0x06, 0x06, 0x66, 0x3C]),
    ('T', [0x7E, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18]),
    ('U', [0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x3C]),
    ('V', [0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x3C, 0x3C, 0x18, 0x18]),
    ('W', [0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x7E, 0x7E, 0x7E, 0x66, 0x66]),
    ('X', [0x66, 0x66, 0x66, 0x3C, 0x3C, 0x18, 0x3C, 0x3C, 0x66, 0x66, 0x66, 0x66]),
    ('Y', [0x66, 0x66, 0x66, 0x66, 0x3C, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18]),
    ('Z', [0x7E, 0x06, 0x06, 0x06, 0x0C, 0x18, 0x30, 0x60, 0x60, 0x60, 0x60, 0x7E]),
    ('<', [0x00, 0x00, 0x06, 0x0C, 0x18, 0x30, 0x60, 0x30, 0x18, 0x0C, 0x06, 0x00]),
]
assert len(GLYPHS) == 37
NUM_CLASSES = len(GLYPHS)
GLYPH_INDEX = {c: i for i, (c, _) in enumerate(GLYPHS)}
# Pre-built glyph bits (LONG FLOAT) for vectorised batch expansion.
GLYPH_BITS = np.zeros((NUM_CLASSES, 12, 8), dtype=np.float32)
for _ci, (_ch, _rows) in enumerate(GLYPHS):
    for _y in range(12):
        for _x in range(8):
            if (_rows[_y] >> (7 - _x)) & 1:
                GLYPH_BITS[_ci, _y, _x] = 1.0

# Network geometry (must match include/cnn.h).
IN_H, IN_W = 12, 16
C1, C2, C3 = 8, 16, 4
KH, KW = 3, 3
HID = 64

# Hard confusable pairs (extended; covers every (gt, got) seen in
# 500-sample corpus benchmark). Each pair gets extra samples so the
# model sees both sides under skew/noise.
HARD_PAIRS = [
    ("0", "O"), ("1", "I"), ("2", "Z"), ("3", "D"), ("4", "A"),
    ("5", "S"), ("6", "O"), ("7", "Z"), ("8", "O"), ("9", "S"),
    ("B", "8"), ("F", "6"), ("G", "6"), ("P", "7"), ("J", "3"),
    ("E", "5"), ("R", "8"), ("T", "6"), ("M", "1"), ("Q", "O"),
    ("<", "0"), ("U", "V"), ("N", "M"), ("K", "X"),
]


def draw_ink_box(ch, scale):
    """Render a binary ink box of size (12*scale) x (8*scale).
    Each template bit becomes an `scale x scale` ink patch (1.0).
    Background is 0.0. This is what gen_mrz_image writes into the
    PPM image (binary, scale px per bit)."""
    rows = GLYPHS[GLYPH_INDEX[ch]][1]
    h, w = 12 * scale, 8 * scale
    box = np.zeros((h, w), dtype=np.float32)
    for y in range(12):
        for x in range(8):
            if (rows[y] >> (7 - x)) & 1:
                box[y*scale:(y+1)*scale, x*scale:(x+1)*scale] = 1.0
    return box


def add_noise(box, p):
    """Salt/pepper on the ink box: each pixel flipped with prob p."""
    if p <= 0:
        return box
    mask = np.random.random(box.shape) < p
    out = box.copy()
    out[mask] = 1.0 - out[mask]
    return out


def skew_box(box, skew):
    """Mimic gen_mrz_image's global row skew: row y shifts right by
    ~y*skew/total_rows px. gen_mrz_image uses abs(skew) as the
    TOTAL band shift over the whole height, scaled by band width."""
    if skew == 0:
        return box
    h, w = box.shape
    sk = h * abs(skew) // max(w, 1)
    if sk == 0:
        return box
    new = np.zeros((h, w + sk), dtype=np.float32)
    for y in range(h):
        off = (y * abs(skew) * w) // (h * w) if skew > 0 else 0
        row = box[y]
        if off > 0:
            new[y, off:off + w] = row
        else:
            new[y, :w] = row
    return new


def area_resample(box, oh=IN_H, ow=IN_W):
    """Area-average downsample a binary (or noisy) ink box to (oh, ow).
    Same shape as the C-side resample_char_gray.  NOTE: callers must
    pass the FULL window (ink + inter-char pad) -- we never crop the
    ink bbox here, to mirror the C pipeline exactly."""
    h, w = box.shape
    out = np.zeros((oh, ow), dtype=np.float32)
    for oy in range(oh):
        sy0 = (oy * h) // oh
        sy1 = ((oy + 1) * h) // oh
        if sy1 <= sy0:
            sy1 = sy0 + 1
        for ox in range(ow):
            sx0 = (ox * w) // ow
            sx1 = ((ox + 1) * w) // ow
            if sx1 <= sx0:
                sx1 = sx0 + 1
            out[oy, ox] = box[sy0:sy1, sx0:sx1].mean()
    return out


def box_smooth_3x3(img):
    """Mirror the C-side 3x3 box smooth on the 16x12 glyph."""
    h, w = img.shape
    out = np.zeros_like(img)
    for y in range(h):
        for x in range(w):
            acc = 0.0
            cnt = 0
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w:
                        acc += img[ny, nx]
                        cnt += 1
            out[y, x] = acc / max(cnt, 1)
    return out


def centroid_shift(img, axis, max_shift):
    """Pre-align the glyph along `axis` ('x' or 'y') so the network
    sees the same input the C pipeline sees AFTER its own centroid
    stage. Mirrors resample_char_gray's clamp +-2 px alignment."""
    h, w = img.shape
    thresh = 0.35
    if axis == 'x':
        mask = img > thresh
        sw = mask.sum()
        if sw < 1e-3:
            return img
        xs = np.arange(w).reshape(1, w)
        cx = (mask.astype(np.float64) * xs).sum() / sw
        target = (w - 1) * 0.5
        delta = int(round(cx - target))
    else:
        mask = img > thresh
        sw = mask.sum()
        if sw < 1e-3:
            return img
        ys = np.arange(h).reshape(h, 1)
        cy = (mask.astype(np.float64) * ys).sum() / sw
        target = (h - 1) * 0.5
        delta = int(round(cy - target))
    delta = max(-max_shift, min(max_shift, delta))
    if delta == 0:
        return img
    out = np.zeros_like(img)
    if axis == 'x':
        if delta > 0:
            out[:, delta:] = img[:, :w - delta]
        else:
            out[:, :w + delta] = img[:, -delta:]
    else:
        if delta > 0:
            out[delta:, :] = img[:h - delta, :]
        else:
            out[:h + delta, :] = img[-delta:, :]
    return out


def render_realistic(ch, seed):
    """Generate ONE 12x16 ink-high glyph matching the C inference
    distribution: real binary ink box at variable scale, optional
    skew / noise, area-resample to 16x12, 3x3 box smooth, centroid
    pre-align."""
    rnd = random.Random(seed)
    scale = rnd.choice([3, 4, 4, 4, 5, 6])
    box = draw_ink_box(ch, scale)
    # C-side segment_line_cnn fixes the window to medw+8 = 8s+8 px
    # (ink box 8s wide centred in an 8s+8 cell).  We must feed the
    # SAME window so inference and training see identical aspect
    # ratios; the old 0..2 px/side padding made training glyphs ~10-25%
    # fatter than inference, which is why retrains never helped.
    # Keep the window 8s+8 exactly and jitter the ink position by
    # +-1 px so the net learns float placement, matching centroid.
    pad_l = rnd.randint(3, 5)
    pad_r = 8 - pad_l  # window == 6s+8 exactly (C cell = medw+8)
    if pad_l or pad_r:
        h, w = box.shape
        # crop to real ink span, centre in a 6s+8 window
        ink = box[:, np.any(box, axis=0)]
        iw = ink.shape[1]
        win_w = 6 * scale + 8
        padded = np.zeros((h, win_w), dtype=np.float32)
        margin = pad_l + (win_w - 8 - iw) // 2
        padded[:, margin:margin + iw] = ink
        box = padded
    # Skew (mirrors gen_mrz_image's `skew` parameter; up to +-2 px).
    sk = rnd.choice([0, 0, 0, 0, 1, -1, 2, -2])
    box = skew_box(box, sk)
    # Noise (mirrors gen_mrz_image's `noise` parameter; 0..0.05).
    nz = rnd.choice([0.0, 0.0, 0.0, 0.0, 0.005, 0.01, 0.02, 0.03, 0.05])
    box = add_noise(box, nz)
    glyph = area_resample(box, oh=IN_H, ow=IN_W)
    glyph = box_smooth_3x3(glyph)
    glyph = centroid_shift(glyph, 'x', max_shift=2)
    glyph = centroid_shift(glyph, 'y', max_shift=2)
    glyph = np.clip(glyph, 0.0, 1.0).astype(np.float32)
    return glyph


def _vec_render_batch(chars, scales, pad_ls, pad_rs, skews, noises):
    """Vectorised batch renderer. Returns (N, 12, 16) float32."""
    N = len(chars)
    labels = np.array([GLYPH_INDEX[c] for c in chars], dtype=np.int64)
    bits_all = GLYPH_BITS[labels]  # (N, 12, 8)

    # Build per-sample ink boxes (variable size, padded to per-batch max).
    # C-side segment_line_cnn: cell = medw + 8 where medw is the MEDIAN
    # ink-run width.  OCR-B median is 6 template cols wide => medw=6s,
    # so the C window is (12*scale) x (6*scale+8).  We replicate that:
    # ink is cropped to its real template span, then centred in the
    # (12s) x (6s+8) window with pad_l = 3..5 px (pad_r = 8-pad_l).
    max_h = max(12 * s for s in scales)
    max_w = max(6 * s + 8 + (s * abs(sk) // max(s, 1)) if sk else 6 * s + 8
                for s, sk in zip(scales, skews))
    boxes = np.zeros((N, max_h, max_w), dtype=np.float32)

    # Group by scale to batch the np.repeat expansion.
    for s in sorted(set(int(x) for x in scales)):
        m = (scales == s)
        big = np.repeat(np.repeat(bits_all[m], s, axis=1), s, axis=2)
        Nm = int(m.sum())
        idxs = np.where(m)[0]
        for i in range(Nm):
            ii = idxs[i]
            pl, pr = int(pad_ls[ii]), int(pad_rs[ii])
            assert pl + pr == 8, "window must be ink+8 to match C-side"
            sk = int(skews[ii])
            h, w = big[i].shape   # 12*scale x 8*scale ink box
            # Crop the ink to its real template span (median 6s wide;
            # 'I' is narrower).  Skip the always-empty outer columns.
            rowink = np.where(big[i].any(axis=0))[0]
            x0, x1 = int(rowink[0]), int(rowink[-1]) + 1
            ink = big[i][:, x0:x1]           # e.g. 6*s wide for most glyphs
            iw = ink.shape[1]
            win_w = 6 * s + 8                # matches C cell (medw+8)
            shift = (h * abs(sk) // max(win_w, 1)) if sk else 0
            # centre the ink in the (6s+8) window (left margin varies
            # with pad_l so the net sees float placement).
            margin = pl + (win_w - 8 - iw) // 2
            for y in range(h):
                off = (y * abs(sk) * iw // max(h * iw, 1)) if sk != 0 else 0
                if sk < 0:
                    off = shift - off
                row = ink[y]
                boxes[ii, y, margin + off:margin + off + iw] = row

    # Apply per-sample noise.
    for i in range(N):
        nz = float(noises[i])
        if nz > 0:
            mask = np.random.random((max_h, max_w)) < nz
            boxes[i][mask] = 1.0 - boxes[i][mask]

    # Area-average the FULL padded window down to (12, 16).  This is
    # the critical distribution-match fix: the C-side segmenter returns
    # a full-cell window (medw+8 = 8*scale+8 px) and resample_char_gray
    # stretches THAT WHOLE WINDOW to 16 cols.  The old code cropped the
    # ink bounding box (pad stripped), making every training glyph ~10-25%
    # fatter than inference -- val_acc looked great (self-consistent) but
    # the retrain scored worse on real inference.  Now we resize the full
    # box including pad, exactly like the C side.
    out = np.zeros((N, IN_H, IN_W), dtype=np.float32)
    for i in range(N):
        # Extract each sample's OWN full window: height = 12*scale rows
        # (full cell), width = 8*scale + 8 + skew_shift cols.  Do NOT
        # crop to the ink bbox and do NOT reuse the max-padded box --
        # both distort the aspect ratio relative to the C pipeline.
        s_i = int(scales[i])
        h = 12 * s_i
        w = 6 * s_i + 8                                # == C cell (medw+8)
        box = boxes[i][:h, :w]
        if not (box != 0).any():
            out[i] = 0
            continue
        oh, ow = IN_H, IN_W
        cell = np.zeros((oh, ow), dtype=np.float32)
        for oy in range(oh):
            sy0 = (oy * h) // oh
            sy1 = ((oy + 1) * h) // oh
            if sy1 <= sy0:
                sy1 = sy0 + 1
            for ox in range(ow):
                sx0 = (ox * w) // ow
                sx1 = ((ox + 1) * w) // ow
                if sx1 <= sx0:
                    sx1 = sx0 + 1
                cell[oy, ox] = box[sy0:sy1, sx0:sx1].mean()
        out[i] = cell

    # 3x3 box smooth (vectorised).
    padded = np.pad(out, ((0, 0), (1, 1), (1, 1)), mode='edge')
    acc = np.zeros_like(out)
    cnt_arr = np.zeros_like(out)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            acc += padded[:, 1 + dy:1 + dy + IN_H, 1 + dx:1 + dx + IN_W]
            cnt_arr += 1
    out = acc / cnt_arr

    # Centroid shift (X then Y) per sample (mirrors C-side clamping).
    for i in range(N):
        img = out[i]
        mask = img > 0.35
        if mask.sum() > 1e-3:
            xs = np.arange(IN_W).reshape(1, IN_W)
            cx = (mask.astype(np.float64) * xs).sum() / mask.sum()
            delta = int(round(cx - (IN_W - 1) * 0.5))
            delta = max(-2, min(2, delta))
            if delta:
                tmp = img.copy()
                img[:] = 0
                if delta > 0:
                    img[:, delta:] = tmp[:, :IN_W - delta]
                else:
                    img[:, :IN_W + delta] = tmp[:, -delta:]
        mask = img > 0.35
        if mask.sum() > 1e-3:
            ys = np.arange(IN_H).reshape(IN_H, 1)
            cy = (mask.astype(np.float64) * ys).sum() / mask.sum()
            delta = int(round(cy - (IN_H - 1) * 0.5))
            delta = max(-2, min(2, delta))
            if delta:
                tmp = img.copy()
                img[:] = 0
                if delta > 0:
                    img[delta:, :] = tmp[:IN_H - delta, :]
                else:
                    img[:IN_H + delta, :] = tmp[-delta:, :]
        out[i] = img
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def build_dataset(n_per_class=1500, hard_extra=750):
    """Build a large batch dataset using the vectorised renderer."""
    import random as _r
    chars, scales, pad_ls, pad_rs, skews, noises = [], [], [], [], [], []
    for idx, (ch, _) in enumerate(GLYPHS):
        for s in range(n_per_class):
            seed = idx * 1_000_000 + s
            r = _r.Random(seed)
            chars.append(ch)
            scales.append(r.choice([3, 4, 4, 4, 5, 6]))
            pad_ls.append(r.randint(3, 5))
            pad_rs.append(8 - pad_ls[-1])
            skews.append(r.choice([0, 0, 0, 0, 1, -1, 2, -2]))
            noises.append(r.choice([0.0, 0.0, 0.0, 0.0, 0.005, 0.01, 0.02, 0.03, 0.05]))
        for a, b in HARD_PAIRS:
            if a == ch or b == ch:
                for s in range(hard_extra):
                    seed = 900_000 + idx * 1000 + s
                    r = _r.Random(seed)
                    chars.append(ch)
                    scales.append(r.choice([3, 4, 4, 4, 5, 6]))
                    pad_ls.append(r.randint(3, 5))
                    pad_rs.append(8 - pad_ls[-1])
                    skews.append(r.choice([0, 0, 0, 0, 1, -1, 2, -2]))
                    noises.append(r.choice([0.0, 0.0, 0.0, 0.0, 0.005, 0.01, 0.02, 0.03, 0.05]))

    chars = np.array(chars)
    scales = np.array(scales)
    pad_ls = np.array(pad_ls)
    pad_rs = np.array(pad_rs)
    skews = np.array(skews)
    noises = np.array(noises)

    N = len(chars)
    out = np.zeros((N, IN_H, IN_W), dtype=np.float32)
    bs = 4096
    for s in range(0, N, bs):
        e = min(s + bs, N)
        out[s:e] = _vec_render_batch(
            chars[s:e].tolist(), scales[s:e], pad_ls[s:e], pad_rs[s:e],
            skews[s:e], noises[s:e])
    Y = np.array([GLYPH_INDEX[c] for c in chars], dtype=np.int64)
    return out, Y


# ---------- numpy conv primitive (im2col) ----------
def conv_fwd(X, W, b, pad):
    """X: (N, H, W, Cin) -> (N, Ho, Wo, Cout). W: (Kh,Kw,Cin,Cout)."""
    N, H, Wd, Cin = X.shape
    Kh, Kw = W.shape[0], W.shape[1]
    Cout = W.shape[3]
    Ho = H + 2 * pad - Kh + 1
    Wo = Wd + 2 * pad - Kw + 1
    Xp = np.pad(X, ((0, 0), (pad, pad), (pad, pad), (0, 0)), mode="constant")
    cols = np.zeros((N, Ho * Wo, Kh * Kw * Cin), dtype=np.float32)
    for i in range(Kh):
        for j in range(Kw):
            cols[:, :, (i * Kw + j) * Cin:(i * Kw + j + 1) * Cin] = \
                Xp[:, i:i+Ho, j:j+Wo, :].reshape(N, Ho * Wo, Cin)
    Wflat = W.reshape(-1, Cout)
    out = cols @ Wflat + b.reshape(1, Cout)
    return out.reshape(N, Ho, Wo, Cout), cols


def conv_bwd(dout, cols, W, X, pad):
    """dout: (N,Ho,Wo,Cout); cols: im2col; W: (Kh,Kw,Cin,Cout)."""
    N, Ho, Wo, Cout = dout.shape
    Kh, Kw, Cin, _ = W.shape
    dWflat = cols.transpose(1, 0, 2).reshape(Ho * Wo, N * Kh * Kw * Cin) \
                 .T @ dout.reshape(N * Ho * Wo, Cout)
    dW = dWflat.reshape(Kh, Kw, Cin, Cout)
    db = dout.sum(axis=(0, 1, 2))
    dXp = np.zeros((N, dout.shape[1] + 2*pad, dout.shape[2] + 2*pad, Cin),
                   dtype=np.float32)
    dd = dout.reshape(N, Ho * Wo, Cout)
    Wflat = W.reshape(-1, Cout)
    dcol = dd @ Wflat.T
    dcol = dcol.reshape(N, Ho, Wo, Kh, Kw, Cin)
    for i in range(Kh):
        for j in range(Kw):
            dXp[:, i:i+Ho, j:j+Wo, :] += dcol[:, :, :, i, j, :]
    if pad:
        return dXp[:, pad:-pad, pad:-pad, :]
    return dXp


def train(epochs=60, lr=0.02, batch=512, n_per_class=1500, hard_extra=750, out_path="src/cnn_weights.h"):
    import torch
    import torch.nn as nn
    torch.manual_seed(1)
    np.random.seed(1)
    random.seed(1)

    Xn, Yn = build_dataset(n_per_class=n_per_class, hard_extra=hard_extra)
    N = Xn.shape[0]
    print(f"dataset: {N} samples x {IN_H*IN_W}px, {NUM_CLASSES} classes",
          flush=True)

    class TinyConvNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, C1, 3, stride=1, padding=1)
            self.conv2 = nn.Conv2d(C1, C2, 3, stride=1, padding=1)
            self.conv3 = nn.Conv2d(C2, C3, 1)
            self.fc1 = nn.Linear(C3 * 3 * 8, HID)
            self.fc2 = nn.Linear(HID, NUM_CLASSES)
        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = torch.nn.functional.max_pool2d(x, (2, 1))
            x = torch.relu(self.conv2(x))
            x = torch.nn.functional.max_pool2d(x, (2, 2))
            x = self.conv3(x)
            x = x.flatten(1)
            x = torch.relu(self.fc1(x))
            return self.fc2(x)

    model = TinyConvNet()
    Xc = Xn.reshape(-1, 1, IN_H, IN_W).astype(np.float32)
    Xt = torch.from_numpy(Xc)
    Yt = torch.from_numpy(Yn.astype(np.int64))
    n = Xt.shape[0]

    # 95/5 train/val split for early stopping.
    perm = torch.randperm(n)
    n_val = max(int(0.05 * n), 1000)
    val_idx = perm[:n_val]
    tr_idx = perm[n_val:]

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_val = 0.0
    best = None
    for ep in range(epochs):
        model.train()
        perm_tr = tr_idx[torch.randperm(len(tr_idx))]
        correct = tot = 0
        for s in range(0, len(perm_tr), batch):
            idx = perm_tr[s:s + batch]
            xb, yb = Xt[idx], Yt[idx]
            logits = model(xb)
            loss = torch.nn.functional.cross_entropy(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            correct += int((logits.argmax(1) == yb).sum())
            tot += len(idx)
        sched.step()
        model.eval()
        with torch.no_grad():
            v_correct = 0
            for s in range(0, len(val_idx), batch):
                idx = val_idx[s:s + batch]
                v_correct += int((model(Xt[idx]).argmax(1) == Yt[idx]).sum())
        val_acc = v_correct / len(val_idx)
        if val_acc > best_val:
            best_val = val_acc
            best = {k: v.detach().cpu().numpy().copy()
                    for k, v in model.state_dict().items()}
        if ep % 2 == 0 or ep == epochs - 1:
            print(f"  ep {ep:3d} tr={correct / tot:.4f} val={val_acc:.4f} "
                  f"best={best_val:.4f} lr={opt.param_groups[0]['lr']:.4f}",
                  flush=True)
        if val_acc > 0.9999 and ep > 10:
            break

    # Per-class accuracy on val split with best weights.
    model.load_state_dict({k: torch.from_numpy(v) for k, v in best.items()})
    model.eval()
    with torch.no_grad():
        pred = model(Xt[val_idx]).argmax(1).numpy()
    per_class = {}
    for ci in range(NUM_CLASSES):
        m = (Yn[val_idx.numpy()] == ci)
        g = GLYPHS[ci][0]
        per_class[g] = float((pred[m] == ci).sum()) / max(int(m.sum()), 1)

    W1 = best["conv1.weight"].transpose(2, 3, 1, 0)
    b1 = best["conv1.bias"]
    W2 = best["conv2.weight"].transpose(2, 3, 1, 0)
    b2 = best["conv2.bias"]
    W3 = best["conv3.weight"].transpose(2, 3, 1, 0)
    b3 = best["conv3.bias"]
    Wf1 = best["fc1.weight"]
    bf1 = best["fc1.bias"]
    Wf2 = best["fc2.weight"]
    bf2 = best["fc2.bias"]
    mean_v = np.zeros(IN_H * IN_W, dtype=np.float32)
    std_v = np.ones(IN_H * IN_W, dtype=np.float32)
    return (W1, b1, W2, b2, W3, b3, Wf1, bf1, Wf2, bf2,
            mean_v, std_v, best_val, per_class)


def emit_header(params, out_path="src/cnn_weights.h"):
    W1, b1, W2, b2, W3, b3, Wf1, bf1, Wf2, bf2, mean, std, best_acc = params[:13]

    def fmt(arr):
        return ", ".join(f"{v:.8f}f" for v in np.asarray(arr).ravel())

    out = ("/* Auto-generated by tools/train_cnn_v2.py. Do not edit. */\n"
           "#ifndef MRZ_OCR_CNN_WEIGHTS_H\n"
           "#define MRZ_OCR_CNN_WEIGHTS_H\n\n"
           "#include <stdint.h>\n"
           "#include \"cnn.h\"\n\n"
           f"static const float DEFAULT_IN_MEAN[CNN_IN_H * CNN_IN_W] = {{\n")
    for i in range(0, len(mean), 8):
        out += "    " + fmt(mean[i:i + 8]) + ",\n"
    out += "};\n"
    out += f"static const float DEFAULT_IN_STD[CNN_IN_H * CNN_IN_W] = {{\n"
    for i in range(0, len(std), 8):
        out += "    " + fmt(std[i:i + 8]) + ",\n"
    out += "};\n\n"

    out += f"/* conv1: {KH}x{KW} x (1 -> {C1}), pad=1 */\n"
    out += "static const float DEFAULT_CONV1_W[CNN_K*CNN_K*1*CNN_C1] = {\n"
    for i in range(0, 3 * 3 * 1 * C1, 8):
        out += "    " + fmt(W1.ravel()[i:i + 8]) + ",\n"
    out += "};\n"
    out += "static const float DEFAULT_CONV1_B[CNN_C1] = {" + fmt(b1) + "};\n\n"
    out += f"/* conv2: {KH}x{KW} x ({C1} -> {C2}), pad=1 */\n"
    out += "static const float DEFAULT_CONV2_W[CNN_K*CNN_K*CNN_C1*CNN_C2] = {\n"
    for i in range(0, 3 * 3 * 8 * 16, 8):
        out += "    " + fmt(W2.ravel()[i:i + 8]) + ",\n"
    out += "};\n"
    out += "static const float DEFAULT_CONV2_B[CNN_C2] = {" + fmt(b2) + "};\n\n"
    out += "/* 1x1 conv (16 -> 4) after pool2 */\n"
    out += "static const float DEFAULT_CONV3_W[CNN_C2*CNN_C3] = {\n"
    for i in range(0, W3.ravel().size, 8):
        out += "    " + fmt(W3.ravel()[i:i + 8]) + ",\n"
    out += "};\n"
    out += "static const float DEFAULT_CONV3_B[CNN_C3] = {" + fmt(b3) + "};\n\n"
    out += "static const float DEFAULT_FC1_W[CNN_FLAT*CNN_HIDDEN] = {\n"
    for i in range(0, Wf1.ravel().size, 8):
        out += "    " + fmt(Wf1.ravel()[i:i + 8]) + ",\n"
    out += "};\n"
    out += "static const float DEFAULT_FC1_B[CNN_HIDDEN] = {" + fmt(bf1) + "};\n\n"
    out += "static const float DEFAULT_FC2_W[CNN_HIDDEN*CNN_OUT] = {\n"
    for i in range(0, HID * 37, 8):
        out += "    " + fmt(Wf2.ravel()[i:i + 8]) + ",\n"
    out += "};\n"
    out += "static const float DEFAULT_FC2_B[CNN_OUT] = {" + fmt(bf2) + "};\n\n"
    out += "#endif\n"
    Path(out_path).write_text(out)
    print(f"wrote src/cnn_weights.h (best_val_acc={best_acc:.4f})",
          flush=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--n-per-class", type=int, default=1500)
    ap.add_argument("--hard-extra", type=int, default=750)
    ap.add_argument("--out", default="src/cnn_weights.h")
    a = ap.parse_args()
    params = train(epochs=a.epochs, lr=0.02, batch=512,
                   n_per_class=a.n_per_class, hard_extra=a.hard_extra,
                   out_path=a.out)
    emit_header(params, a.out)
    print("\nper-class accuracy (val, balanced):")
    for ch in "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ<":
        if ch in params[-1]:
            print(f"  {ch}: {params[-1][ch] * 100:.2f}%")
