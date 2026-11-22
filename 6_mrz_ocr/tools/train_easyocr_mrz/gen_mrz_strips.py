#!/usr/bin/env python3
"""Generate single-line MRZ strip images for EasyOCR custom training.

Renders ICAO 9303 TD3 line1/line2 strings using the OCR-B font,
applies physical augmentation (shear, rotation, blur, illumination,
noise), and writes images + labels.csv in the EasyOCR trainer format.

Usage:
  python3 gen_mrz_strips.py --train --out data/mrz_train --n 20000
  python3 gen_mrz_strips.py --eval  --out data/mrz_val   --n 2000
"""
from __future__ import annotations
import argparse
import csv
import json
import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
FONT_PATH = ROOT / "14_ultimateMRZ-SDK" / "assets" / "fonts" / "OcrB Regular.ttf"

ALNUM = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
ALPHA = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
DIGIT = "0123456789"
PERS_ALNUM_POOL = "".join(ch * (3 if ch in "GDKMLR" else 1) for ch in ALNUM)

CHARSET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ<"


def check_digit_of(seg: str) -> str:
    w = [7, 3, 1]
    total = 0
    for i, c in enumerate(seg):
        v = (ord(c) - 48) if '0' <= c <= '9' else (
            (ord(c) - 55) if 'A' <= c <= 'Z' else 0)
        total += v * w[i % 3]
    return str(total % 10)


def build_line2(pn, nat, bd, sex, ex, pers):
    l2 = (pn + check_digit_of(pn) + nat + bd + check_digit_of(bd) +
          sex + ex + check_digit_of(ex) + pers + check_digit_of(pers))
    comp = l2[0:10] + l2[13:20] + l2[21:43]
    l2 += check_digit_of(comp)
    return l2


def build_line1(issuer, surname, given):
    name = surname + "<<" + given if given else surname + "<"
    l1 = ("P<" + issuer + name).ljust(44, "<")
    return l1


def iso3_pool():
    return ["AFG", "CHN", "DEU", "GBR", "USA", "JPN", "KOR", "FRA", "ITA",
            "ESP", "RUS", "IND", "BRA", "CAN", "AUS", "UTO", "CHE", "NLD",
            "BEL", "SWE", "NOR", "DNK", "FIN", "POL", "CZE", "AUT", "PRT",
            "GRC", "TUR", "EGY", "ZAF", "MEX", "ARG", "THA", "MYS", "SGP",
            "PHL", "VNM", "IDN", "IRL", "ISR", "KAZ", "UKR", "BLR", "LVA",
            "EST", "LTU", "SVN", "SVK", "HRV", "BGR", "ROU", "HUN", "ALB",
            "MKD", "BIH", "MNE", "SRB", "LUX", "ISL", "LIE", "MLT", "CYP"]


def random_date(rnd, lo_year, hi_year):
    y = rnd.randint(lo_year, hi_year)
    m = rnd.randint(1, 12)
    leap = (y % 4 == 0 and y % 100 != 0) or y % 400 == 0
    dmax = [31, 29 if leap else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][m - 1]
    return f"{y % 100:02d}{m:02d}{rnd.randint(1, dmax):02d}"


def random_pair(rnd):
    pool = iso3_pool()
    issuer = rnd.choice(pool)
    nat = issuer if rnd.random() < 0.8 else rnd.choice(pool)
    surname = "".join(rnd.choice(ALPHA) for _ in range(rnd.randint(3, 12)))
    given = "".join(rnd.choice(ALPHA) for _ in range(rnd.randint(0, 15)))
    l1 = build_line1(issuer, surname, given)
    r = rnd.random()
    if r < 0.6:
        pn = "".join(rnd.choice(DIGIT) for _ in range(9))
    elif r < 0.8:
        pn = "".join(rnd.choice(ALNUM) for _ in range(9))
    else:
        k = rnd.randint(7, 8)
        pn = ("".join(rnd.choice(ALNUM) for _ in range(k))).ljust(9, "<")
    bd = random_date(rnd, 1940, 2009)
    ex = random_date(rnd, 2015, 2039)
    u = rnd.random()
    if u < 0.35:
        pers = "<" * 14
    elif u < 0.60:
        pers = "".join(rnd.choice(DIGIT) for _ in range(14))
    elif u < 0.90:
        pers = "".join(rnd.choice(PERS_ALNUM_POOL) for _ in range(14))
    else:
        pers = ("".join(rnd.choice(PERS_ALNUM_POOL)
                        for _ in range(rnd.randint(6, 12)))).ljust(14, "<")
    l2 = build_line2(pn, nat, bd, rnd.choice("MF<"), ex, pers)
    return l1, l2


def render_strip(text, font, img_h=32, char_w=25, augment=True, rnd=None):
    """Render a single MRZ line as a grayscale strip image.

    char_w=25 matches real passport photos where each character is ~25px
    wide after scaling to imgH=32 (real photos: 32.7px char width at 42px
    line height → 32.7*32/42 ≈ 25px).
    """
    if rnd is None:
        rnd = random.Random()
    img_w = len(text) * char_w + 4
    img = Image.new('L', (img_w, img_h), 255)
    draw = ImageDraw.Draw(img)
    # Center text vertically
    bbox = draw.textbbox((0, 0), text, font=font)
    text_h = bbox[3] - bbox[1]
    y_off = max(0, (img_h - text_h) // 2 - 1)
    draw.text((2, y_off), text, font=font, fill=0)

    if augment:
        # Rotation (-3 to +3 degrees)
        angle = rnd.uniform(-3, 3)
        img = img.rotate(angle, fillcolor=255, expand=False)

        # Shear (horizontal skew, -0.08 to +0.08)
        shear = rnd.uniform(-0.08, 0.08)
        img = img.transform(
            (img_w, img_h), Image.AFFINE,
            (1, shear, 0, 0, 1, 0), fillcolor=255)

        # Gaussian blur (radius 0 to 1.5)
        blur_r = rnd.uniform(0, 1.5)
        if blur_r > 0.1:
            img = img.filter(ImageFilter.GaussianBlur(radius=blur_r))

        # Illumination gradient
        arr = np.array(img, dtype=np.float32)
        if rnd.random() < 0.4:
            strength = rnd.uniform(20, 80)
            direction = rnd.choice(['h', 'v', 'd'])
            if direction == 'h':
                grad = np.linspace(0, strength, img_w)[None, :]
            elif direction == 'v':
                grad = np.linspace(0, strength, img_h)[:, None]
            else:
                grad = np.linspace(0, strength, img_w)[None, :] * \
                       np.linspace(0, 1, img_h)[:, None]
            arr += grad
        # Gaussian noise
        if rnd.random() < 0.3:
            noise = np.random.normal(0, rnd.uniform(2, 10), arr.shape)
            arr += noise
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr, 'L')

    return img


def generate_dataset(out_dir, n_samples, seed, augment=True):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "images").mkdir(exist_ok=True)

    rnd = random.Random(seed)
    np.random.seed(seed)

    # Load OCR-B font at multiple sizes — larger sizes to match real photos
    fonts = {}
    for sz in [20, 22, 24, 26, 28]:
        fonts[sz] = ImageFont.truetype(str(FONT_PATH), sz)

    labels = []
    for i in range(n_samples):
        l1, l2 = random_pair(rnd)

        # Choose line and font size
        line_choice = rnd.choice(['l1', 'l2'])
        text = l1 if line_choice == 'l1' else l2
        font_sz = rnd.choice([20, 22, 24, 26, 28])
        font = fonts[font_sz]
        # char_w=25 matches real passport photos (32.7px * 32/42 ≈ 25px)
        char_w = 25

        img = render_strip(text, font, img_h=32, char_w=char_w,
                           augment=augment, rnd=rnd)

        fname = f"{i:06d}_{line_choice}.png"
        img.save(out_dir / "images" / fname)
        labels.append((fname, text))

    # Write labels.csv
    with open(out_dir / "labels.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'words'])
        for fname, text in labels:
            writer.writerow([f"images/{fname}", text])

    # Write metadata
    meta = {
        "n_samples": n_samples,
        "seed": seed,
        "augment": augment,
        "charset": CHARSET,
        "font": str(FONT_PATH),
        "font_sizes": [16, 18, 20, 22, 24],
    }
    with open(out_dir / "meta.json", 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"Generated {n_samples} samples in {out_dir}")
    print(f"  labels.csv: {len(labels)} entries")
    print(f"  charset: {CHARSET} ({len(CHARSET)} chars)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--out", required=True)
    parser.add_argument("--n", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=1001)
    parser.add_argument("--no-augment", action="store_true")
    args = parser.parse_args()

    augment = not args.no_augment
    if args.train:
        generate_dataset(args.out, args.n, args.seed, augment=augment)
    elif args.eval:
        generate_dataset(args.out, args.n, args.seed + 1000, augment=augment)
    else:
        generate_dataset(args.out, args.n, args.seed, augment=augment)
