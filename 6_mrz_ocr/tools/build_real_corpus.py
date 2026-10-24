#!/usr/bin/env python3
"""Build a real-photo CRNN training corpus from板端 (PaddleOCR) captures.

The 41 machine-position photos ship with `<name>.jpg-result.txt` logs that
contain the board-side detector output. Those logs carry two labels:

  * `textLine[0/1]`      — the detector's raw per-character scores; usable
                           only when every score is high (>= --min-score).
  * the `-----MRZ fixed (fixmrz)-----` block — the board's checksum-repaired
                           guess, which is NOT reliable on its own.

This script prefers the raw detector text when it is high-confidence and
falls back to fixmrz otherwise, then writes:

  <out>/corpus.json   records with line1/line2 + source image
  <out>/*.ppm         board-cropped MRZ band, converted from JPEG

The PPM conversion keeps the exact crop the board used, so the Native
pipeline's feature extraction sees the same geometry the board saw. That is
what makes these samples usable for training the sequence head on the real
domain.
"""
from __future__ import annotations
import argparse
import json
import re
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

LINE_RE = re.compile(r"^textLine\[(\d+)\]\((.*)\)\s*$")
SCORE_RE = re.compile(r"^textScores\[(\d+)\]\{(.*)\}\s*$")
SCORES_ALL_RE = re.compile(r"textScores\[\d+\]\{(.*)\}")
FIX_RE = re.compile(r"-----MRZ fixed \(fixmrz\)-----\s*\n(.+)\n(.+)", re.M)


def parse_log(path: Path):
    text = path.read_text(errors="replace")
    lines = {}
    for m in LINE_RE.finditer(text):
        lines[int(m.group(1))] = m.group(2).strip()
    # Prefer the minimum score across every row: a single weak character
    # makes the whole line unusable as a label.
    mins = [min(float(v) for v in m.group(1).split(","))
            for m in SCORES_ALL_RE.finditer(text)]
    min_score = min(mins) if mins else 0.0
    fixed = None
    fm = FIX_RE.search(text)
    if fm:
        fixed = (fm.group(1).strip(), fm.group(2).strip())
    # The board sometimes prints the un-repaired lines just before the
    # fixmrz block; take those as a third candidate.
    raw_pair = None
    tail = text.split("=====End detect=====")[-1]
    cand = [l.strip() for l in tail.splitlines()
            if l.strip() and re.fullmatch(r"[A-Z0-9<]{40,44}", l.strip())]
    if len(cand) >= 2:
        raw_pair = (cand[0], cand[1])
    return lines, min_score, fixed, raw_pair


def valid(s: str) -> bool:
    return len(s) == 44 and all(c.isalnum() or c == "<" for c in s)


def isolate_mrz(im: Image.Image):
    """Cut the two MRZ text rows out of a machine-position photo.

    MRZ is the pair of dense, short dark rows that sit ABOVE the solid
    observation band of the passport. The band is a wide contiguous dark
    block; the MRZ rows are much narrower in height but carry character
    density. We therefore find the solid band first and keep the text rows
    that lie above it.
    """
    g = np.asarray(im.convert("L"), dtype=np.int16)
    h, w = g.shape
    dark = (g < 120).sum(axis=1)
    # The observation band is a WIDE contiguous dark block. The photo may
    # also have a dark border at the very top, so take the LAST such block
    # rather than the first one.
    band_rows = np.where(dark > w * 0.85)[0]
    limit = h
    if len(band_rows):
        gaps = np.where(np.diff(band_rows) > 1)[0]
        starts = np.concatenate(([0], gaps + 1))
        ends = np.concatenate((gaps, [len(band_rows) - 1]))
        blocks = [(band_rows[s], band_rows[e]) for s, e in zip(starts, ends)]
        # The band must be reasonably tall to be the observation window.
        blocks = [b for b in blocks if b[1] - b[0] >= 8]
        if blocks:
            limit = blocks[-1][0]
    # Candidate text rows above the band: moderate dark density.
    text_rows = [r for r in range(0, limit)
                 if w * 0.02 < dark[r] < w * 0.85]
    if not text_rows:
        return None, "none"
    # Group contiguous candidate rows into bands, keep the two largest
    # groups that look like text lines.
    groups = []
    start = prev = text_rows[0]
    for r in text_rows[1:]:
        if r - prev > 3:
            groups.append((start, prev))
            start = r
        prev = r
    groups.append((start, prev))
    groups = [g for g in groups if g[1] - g[0] >= 4]
    if not groups:
        return None, "none"
    # Prefer the LOWEST two groups (MRZ is the last text before the band).
    groups.sort(key=lambda g: -g[0])
    chosen = sorted(groups[:2])
    top = max(0, chosen[0][0] - 6)
    bot = min(h, chosen[-1][1] + 7)
    return im.crop((0, top, w, bot)), "isolated"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="tmp/omtest")
    ap.add_argument("--out", default="data/corpus_real")
    ap.add_argument("--min-score", type=float, default=0.98)
    ap.add_argument("--scale", type=int, default=0,
                    help="downscale long side to this (0 = keep)")
    args = ap.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    records = []
    kept = dropped = 0
    for log in sorted(src.rglob("*.jpg-result.txt")):
        stem = log.name[: -len("-result.txt")]
        jpg = log.parent / stem
        # Prefer the board's own rectified MRZ crop when it exists: it is
        # exactly the geometry the board's detector saw, which is what we
        # want the sequence head to be robust to.
        crop_jpg = log.parent / (stem + "-result.jpg")
        if not jpg.exists():
            continue
        lines, min_score, fixed, raw_pair = parse_log(log)

        l1 = l2 = None
        source = None
        if 0 in lines and 1 in lines and valid(lines[0]) and valid(lines[1]) \
                and min_score >= args.min_score:
            l1, l2, source = lines[0], lines[1], "detector"
        elif fixed and valid(fixed[0]) and valid(fixed[1]):
            l1, l2, source = fixed[0], fixed[1], "fixmrz"
        elif raw_pair and valid(raw_pair[0]) and valid(raw_pair[1]):
            l1, l2, source = raw_pair[0], raw_pair[1], "raw"
        if not l1:
            dropped += 1
            continue

        # The board writes `<name>.jpg-result.jpg` as a DETECTION OVERLAY
        # (it still contains the observation band), so it is not a clean
        # MRZ crop. Instead locate the two text rows directly: the MRZ is
        # the pair of dense dark rows above the solid observation band.
        im = Image.open(jpg).convert("RGB")
        crop, crop_src = isolate_mrz(im)
        if crop is None:
            dropped += 1
            continue
        if args.scale and max(crop.size) > args.scale:
            r = args.scale / max(crop.size)
            crop = crop.resize((int(crop.width * r), int(crop.height * r)),
                               Image.LANCZOS)
        name = f"real_{kept:04d}.ppm"
        crop.save(out / name)

        records.append({
            "id": jpg.parent.name + "/" + jpg.stem,
            "image": name,
            "line1": l1, "line2": l2,
            "profile": "real", "scale": 0, "hgap": 0,
            "skew": 0.0, "noise": 0.0, "rot": 0.0,
            "label_source": source, "min_score": round(min_score, 4),
            "crop_source": crop_src,
        })
        kept += 1

    (out / "corpus.json").write_text(json.dumps({"records": records}, indent=1))
    by_src = {}
    for r in records:
        by_src[r["label_source"]] = by_src.get(r["label_source"], 0) + 1
    print(f"wrote {out} kept={kept} dropped={dropped} sources={by_src}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())