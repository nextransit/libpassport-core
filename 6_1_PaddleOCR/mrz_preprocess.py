#!/usr/bin/env python3
"""mrz_preprocess -- MRZ-strip pre-processing primitives.

Phase 3 keeps only the two primitives that are still useful:
  * binarise_otsu(gray) / binarise_adaptive(gray) — strip the input
    to a binary mask so rec engines see a clean OCR-B glyph map.
  * clean_morphology(binary) — open/close to remove scan speckle
    and bridge hairline gaps inside glyphs.

The earlier equiwidth column-resample (`find_char_columns` +
`equiwidth_resample`) was empirically counterproductive on the
corpus_eval baseline (L1 83% -> 50%, L2 48% -> 14%) and is removed.
"""
from __future__ import annotations


def binarise_otsu(gray):
    """numpy-only Otsu binarisation. Returns (binary uint8, threshold)."""
    import numpy as np
    hist, _ = np.histogram(gray, bins=256, range=(0, 256))
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total == 0:
        return (gray < 128).astype("uint8"), 128
    p = hist / total
    omega = np.cumsum(p)
    mu = np.cumsum(p * np.arange(256))
    mu_t = mu[-1]
    sigma_b = (mu_t * omega - mu) ** 2 / (omega * (1 - omega) + 1e-12)
    thr = int(np.argmax(sigma_b))
    return (gray < thr).astype("uint8"), thr


def binarise_adaptive(gray, block=15, c=5):
    """Gaussian adaptive threshold — better for unevenly-lit inputs.

    Implemented in numpy (skimage-free) so the dependency surface
    stays minimal. block must be odd.
    """
    import numpy as np
    if block % 2 == 0:
        block += 1
    h, w = gray.shape
    pad = block // 2
    padded = np.pad(gray.astype(np.float64), pad, mode="reflect")
    cs = np.zeros((h + 2 * pad + 1, w + 2 * pad + 1), dtype=np.float64)
    cs[1:, 1:] = np.cumsum(np.cumsum(padded, axis=0), axis=1)
    sum_kernel = (
        cs[block:, block:] - cs[:-block, block:]
        - cs[block:, :-block] + cs[:-block, :-block]
    )
    area = float(block * block)
    local_mean = sum_kernel / area
    binary = (gray.astype(np.float64) < (local_mean - c)).astype("uint8")
    return binary


def clean_morphology(binary, open_px=1, close_px=1):
    """Open (erode+dilate) then close (dilate+erode) a binary image.

    Removes isolated dark speckes (open) and bridges narrow gaps inside
    glyphs (close). Pixel radius = open_px / close_px. Uses a square
    structuring element (4-neighbour + diagonals).
    """
    import numpy as np
    b = binary.copy()
    if open_px > 0:
        for _ in range(open_px):
            b = _erode_sq(b)
        for _ in range(open_px):
            b = _dilate_sq(b)
    if close_px > 0:
        for _ in range(close_px):
            b = _dilate_sq(b)
        for _ in range(close_px):
            b = _erode_sq(b)
    return b


def _erode_sq(binary):
    """Square 3x3 erosion (numpy)."""
    import numpy as np
    b = np.pad(binary, 1, mode="constant", constant_values=0)
    out = np.zeros_like(binary)
    for dy in range(3):
        for dx in range(3):
            out &= b[dy:dy + binary.shape[0], dx:dx + binary.shape[1]]
    return out


def _dilate_sq(binary):
    """Square 3x3 dilation (numpy)."""
    import numpy as np
    b = np.pad(binary, 1, mode="constant", constant_values=0)
    out = np.zeros_like(binary)
    for dy in range(3):
        for dx in range(3):
            out |= b[dy:dy + binary.shape[0], dx:dx + binary.shape[1]]
    return out
