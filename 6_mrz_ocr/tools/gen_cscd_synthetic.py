#!/usr/bin/env python3
"""Generate synthetic CS/CD MRZ glyph patches with guilloche backgrounds.

CS/CD (港澳台通行证) cards have a guilloche security-printing background
that defeats binarization-based glyph extraction.  This script renders
OCR-B characters directly at 12x16 resolution with synthetic guilloche
backgrounds + noise, producing training data that teaches the CNN to
ignore the texture and focus on the character.

Output format matches train_cnn_real.py's gather() expectations:
  <workdir>/img_XXXXX/
    meta.json          {"line1": "CSC...", "line2": ""}
    mrz_dump_img_l0_c00_x0_r0_w8.ppm   (12x16 grayscale)
    ...
    mrz_dump_img_l0_c29_x0_r0_w8.ppm

Usage:
  python3 gen_cscd_synthetic.py --n-images 500 --out /tmp/mrz_cscd_synth
  python3 train_cnn_real.py --corpus /tmp/mrz_cscd_synth --workdir /tmp/mrz_cscd_glyphs --out src/cnn_weights_cscd.h
"""
from __future__ import annotations
import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_cnn_v2 import GLYPHS, GLYPH_INDEX  # noqa: E402

ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ<"
GI = {c: i for i, c in enumerate(ALPHABET)}
CELL_W = 16
CELL_H = 12
PPM_W = 8


def mrz_check_digit(s: str) -> str:
    weights = [7, 3, 1]
    total = 0
    for i, ch in enumerate(s):
        if ch == '<':
            v = 0
        elif ch.isdigit():
            v = int(ch)
        else:
            v = ord(ch) - 55
        total += v * weights[i % 3]
    return str(total % 10)


def gen_cscd_line() -> str:
    doc_type = random.choice(['CS', 'CD'])
    state_chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    issuing = ''.join(random.choices(state_chars, k=2))
    doc_num = ''.join(random.choices('0123456789', k=7))
    doc_ck_input = issuing + doc_num
    doc_ck = mrz_check_digit(doc_ck_input)
    by = f"{random.randint(60, 90):02d}"
    bm = f"{random.randint(1, 12):02d}"
    bd = f"{random.randint(1, 28):02d}"
    birth = by + bm + bd
    birth_ck = mrz_check_digit(birth)
    ey = f"{random.randint(25, 35):02d}"
    em = f"{random.randint(1, 12):02d}"
    ed = f"{random.randint(1, 28):02d}"
    expiry = ey + em + ed
    expiry_ck = mrz_check_digit(expiry)
    comp_input = doc_ck_input + doc_ck + birth + birth_ck + expiry + expiry_ck
    comp_ck = mrz_check_digit(comp_input)
    line = f"{doc_type}{issuing}{doc_num}{doc_ck}<{birth}{birth_ck}<{expiry}{expiry_ck}<{comp_ck}"
    assert len(line) == 30, f"len={len(line)} line={line}"
    return line


def guilloche_bg(h: int, w: int, amplitude: float,
                 freq_x: float, freq_y: float, phase: float) -> np.ndarray:
    x = np.linspace(0, 2 * math.pi, w)
    y = np.linspace(0, 2 * math.pi, h)
    X, Y = np.meshgrid(x, y)
    return amplitude * (
        np.sin(freq_x * X + phase) * np.cos(freq_y * Y + phase * 1.3)
        + 0.5 * np.sin((freq_x + freq_y) * X + phase * 0.7)
    )


def render_glyph_12x16(ch: str, cfg: dict) -> np.ndarray:
    """Render one OCR-B glyph at 12x16 with guilloche background + noise."""
    rows = GLYPHS[GLYPH_INDEX[ch]][1]
    scale = cfg['scale']
    h_hr = CELL_H * scale
    w_hr = CELL_W * scale
    bg_val = cfg['bg_val']
    ink_val = cfg['ink_val']
    img = np.full((h_hr, w_hr), bg_val, dtype=np.float32)
    sx = w_hr / 8.0
    sy = h_hr / 12.0
    for y in range(12):
        for x in range(8):
            if (rows[y] >> (7 - x)) & 1:
                y0 = int(y * sy)
                y1 = int((y + 1) * sy)
                x0 = int(x * sx)
                x1 = int((x + 1) * sx)
                img[y0:y1, x0:x1] = ink_val
    if cfg['guilloche_amp'] > 0:
        bg = guilloche_bg(h_hr, w_hr, cfg['guilloche_amp'],
                          cfg['freq_x'], cfg['freq_y'], cfg['phase'])
        text_mask = np.zeros((h_hr, w_hr), dtype=np.float32)
        for y in range(12):
            for x in range(8):
                if (rows[y] >> (7 - x)) & 1:
                    y0 = int(y * sy)
                    y1 = int((y + 1) * sy)
                    x0 = int(x * sx)
                    x1 = int((x + 1) * sx)
                    text_mask[y0:y1, x0:x1] = 1.0
        img = img * (1 - text_mask * 0.6) + bg * (1 - text_mask * 0.6) + img * text_mask
    if cfg['noise'] > 0:
        img += np.random.normal(0, cfg['noise'], img.shape)
    if cfg['blur'] > 0:
        from PIL import ImageFilter
        pil = Image.fromarray(np.clip(img * 255, 0, 255).astype(np.uint8))
        pil = pil.filter(ImageFilter.GaussianBlur(cfg['blur']))
        img = np.asarray(pil, dtype=np.float32) / 255.0
    pil = Image.fromarray(np.clip(img * 255, 0, 255).astype(np.uint8))
    pil = pil.resize((CELL_W, CELL_H), Image.BILINEAR)
    patch = np.asarray(pil, dtype=np.float32) / 255.0
    if cfg.get('centroid_shift', True):
        ink = patch < 0.5
        if ink.any():
            ys, xs = np.where(ink)
            cy = ys.mean() - (CELL_H - 1) / 2
            cx = xs.mean() - (CELL_W - 1) / 2
            shift_y = int(round(cy))
            shift_x = int(round(cx))
            if shift_y or shift_x:
                patch = np.roll(patch, (-shift_y, -shift_x), axis=(0, 1))
    return patch


def random_cfg() -> dict:
    return {
        'scale': random.randint(3, 6),
        'bg_val': random.uniform(0.80, 0.95),
        'ink_val': random.uniform(0.05, 0.20),
        'guilloche_amp': random.uniform(0.08, 0.25),
        'freq_x': random.uniform(1.5, 4.0),
        'freq_y': random.uniform(1.0, 3.0),
        'phase': random.uniform(0, 2 * math.pi),
        'noise': random.uniform(0.01, 0.05),
        'blur': random.uniform(0.0, 0.8),
        'centroid_shift': True,
    }


def save_ppm(path: Path, patch: np.ndarray):
    arr = np.clip(patch * 255, 0, 255).astype(np.uint8)
    Image.fromarray(arr, mode='L').save(str(path), format='PPM')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-images', type=int, default=500)
    ap.add_argument('--out', default='/tmp/mrz_cscd_synth')
    ap.add_argument('--seed', type=int, default=42)
    a = ap.parse_args()

    random.seed(a.seed)
    np.random.seed(a.seed)
    out = Path(a.out)
    if out.exists():
        import shutil
        shutil.rmtree(out)
    out.mkdir(parents=True)

    for i in range(a.n_images):
        line = gen_cscd_line()
        imgdir = out / f"img_{i+1:05d}"
        imgdir.mkdir(parents=True)
        meta = {"line1": line, "line2": "", "synthetic": True}
        (imgdir / 'meta.json').write_text(json.dumps(meta))
        for col, ch in enumerate(line):
            cfg = random_cfg()
            patch = render_glyph_12x16(ch, cfg)
            fname = f"mrz_dump_img_l0_c{col:02d}_x0_r0_w{PPM_W}.ppm"
            save_ppm(imgdir / fname, patch)

    print(f"Generated {a.n_images} synthetic CS/CD images at {out}")
    print(f"Each image has 30 glyph patches (12x16) with guilloche backgrounds")
    print(f"To train: python3 train_cnn_real.py --corpus {out} "
          f"--workdir /tmp/mrz_cscd_glyphs --out src/cnn_weights_cscd.h")


if __name__ == '__main__':
    main()
