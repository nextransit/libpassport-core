#!/usr/bin/env python3
"""Export CNN glyph slices from SYNTHETIC MRZ band renders.

Why this exists
---------------
`train_cnn_v2.build_dataset` renders ONE glyph at a time and area-averages
the C-side cell window (medw+8) down to 12x16.  That is close to the
inference distribution, but it never exercises the *band* pipeline: the
cell crop that `segment_line_grid` actually produces for a rendered MRZ
line (glyph pitch, inter-glyph margins, vertical alignment of the two
rows).  The gap shows up as a systematic E/F confusion on small scales:

    $ mrz_ocr_cnn_tool  (scale=3 render of "PEPEPEPE<<PEPEPEEP<<PEPE...")
    PFPFPFPF<<PFPFPFFP<<PFPF...        <- every E read as F

and as the `mrz_golden` regression ("cnn scale=3 line1": E read as F).

This exporter closes the gap the same way the photo exporter does for
real images: render a complete MRZ band with `gen_mrz_image`, let the C
pipeline cut it into cells (MRZ_OCR_DUMP), and label each dumped patch
with the character that was rendered at that column.  Alignment is
verified model-independently: a render is only accepted when the tool
produced exactly 44 cells per line and the cell x-positions advance with
a constant pitch.

Usage:
    python3 tools/export_synth_slices.py --out data/photo_train_synth/cnn_slices.npz
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

GLYPHS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ<"
GI = {c: i for i, c in enumerate(GLYPHS)}
W = [7, 3, 1]

ROOT = Path(__file__).resolve().parents[2]


def check_digit(s: str) -> str:
    total = 0
    for i, c in enumerate(s):
        v = int(c) if c.isdigit() else (ord(c) - 55 if c.isupper() else 0)
        total += v * W[i % 3]
    return str(total % 10)


SURNAMES = ["ERIKSSON", "MULLER", "NOVAK", "SCHMIDT", "GARCIA", "SMITH",
             "JANSSEN", "PETROV", "KOWALSKI", "ANDERSSON", "BENNETT",
             "DELGADO", "FONTAINE", "GRUBER", "HENRIksen".upper(),
             "IORGA", "JENSEN", "KELLER", "LINDQVIST", "MORENO"]
GIVENS = ["ANNA", "MARIA", "JOHN", "ELENA", "PAUL", "ANNA", "LARS", "SOFIA",
          "PETRA", "JAMES", "OLIVIA", "MICHEL", "HANNA", "NIELS", "CLARA"]


def rand_name(rng: random.Random) -> str:
    sur = rng.choice(SURNAMES)
    giv = rng.choice(GIVENS)
    if rng.random() < 0.3:
        giv += " " + rng.choice(GIVENS)
    name = f"{sur}<<{giv}".replace(" ", "<")
    return name[:39]


def rand_alnum(rng: random.Random, n: int) -> str:
    return "".join(rng.choice("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
                   for _ in range(n))


def make_record(rng: random.Random) -> tuple[str, str]:
    """Return an ICAO-valid (line1, line2) pair."""
    docno = rand_alnum(rng, 9)
    nat = "".join(rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in range(3))
    birth = f"{rng.randint(0, 99):02d}{rng.randint(1, 12):02d}{rng.randint(1, 28):02d}"
    expiry = f"{rng.randint(0, 99):02d}{rng.randint(1, 12):02d}{rng.randint(1, 28):02d}"
    sex = rng.choice("MF")
    personal = "".join(rng.choice("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ<")
                       for _ in range(14))
    l2 = (docno + check_digit(docno) + "<" + nat + birth + check_digit(birth) +
          sex + expiry + check_digit(expiry) + personal)
    l2 += check_digit(l2)
    l1 = ("P<" + rand_name(rng)).ljust(44, "<")[:44]
    return l1, l2


def dump_files(prefix: str, tag: str, line: int) -> list[str]:
    return sorted(glob.glob(f"{prefix}_{tag}_l{line}_c*_x*_r*_w*.ppm"))


def parse_dump(path: str) -> tuple[int, int, int]:
    m = re.search(r"_c(\d+)_x(\d+)_r(\d+)_w(\d+)", Path(path).name)
    return int(m.group(1)), int(m.group(2)), int(m.group(4))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/photo_train_synth/cnn_slices.npz")
    ap.add_argument("--renders", type=int, default=60,
                    help="number of random MRZ records")
    ap.add_argument("--scales", default="3,4,5,6")
    ap.add_argument("--seed", type=int, default=20261103)
    ap.add_argument("--tool", default="6_mrz_ocr/build/mrz_ocr_cnn_tool")
    ap.add_argument("--gen", default="6_mrz_ocr/build/gen_mrz_image")
    ap.add_argument("--dump-prefix", default="/tmp/mrz_dump")
    ap.add_argument("--hgap", type=int, default=2)
    ap.add_argument("--vgap", type=int, default=6)
    ap.add_argument("--margin", type=int, default=12)
    args = ap.parse_args()

    scales = [int(s) for s in args.scales.split(",") if s.strip()]
    tool = ROOT / args.tool
    gen = ROOT / args.gen
    tmp = Path("/tmp/mrz_synth_export")
    tmp.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    X, Y, META = [], [], []
    n_render = n_skip = n_cells = 0
    for i in range(args.renders):
        l1, l2 = make_record(rng)
        for scale in scales:
            ppm = tmp / f"synth_{i}_{scale}.ppm"
            r = subprocess.run(
                [str(gen), str(ppm), l1, l2,
                 f"scale={scale}", f"hgap={args.hgap}", f"vgap={args.vgap}",
                 f"margin={args.margin}"],
                capture_output=True, text=True, timeout=60)
            if r.returncode != 0:
                n_skip += 1
                continue
            tag = f"syn{i}s{scale}"
            for f in glob.glob(f"{args.dump_prefix}_{tag}_*"):
                os.remove(f)
            env = dict(os.environ)
            env["MRZ_OCR_DUMP"] = tag
            env["MRZ_OCR_LINE_MODELS"] = "1"
            env["MRZ_OCR_CRNN"] = "0"
            r = subprocess.run([str(tool), str(ppm)], capture_output=True,
                               text=True, timeout=60, env=env)
            if r.returncode != 0:
                n_skip += 1
                continue
            ok = True
            per_line = {}
            for li, truth in ((0, l1), (1, l2)):
                files = dump_files(args.dump_prefix, tag, li)
                if len(files) != 44:
                    ok = False
                    break
                cells = sorted((parse_dump(f) for f in files))
                if [c for c, _, _ in cells] != list(range(44)):
                    ok = False
                    break
                xs = [x for _, x, _ in cells]
                dx = [xs[j + 1] - xs[j] for j in range(len(xs) - 1)]
                if min(dx) <= 0:
                    ok = False
                    break
                pitch = float(np.median(dx))
                if pitch <= 0 or max(dx) - min(dx) > 0.2 * pitch:
                    ok = False
                    break
                per_line[li] = (files, cells)
            if not ok:
                n_skip += 1
                continue
            n_render += 1
            for li, truth in ((0, l1), (1, l2)):
                files, cells = per_line[li]
                for (col, _x, _w), f in zip(cells, files):
                    ch = truth[col]
                    if ch not in GI:
                        continue
                    patch = np.asarray(Image.open(f).convert("L"),
                                       dtype=np.float32) / 255.0
                    if patch.shape != (12, 16):
                        patch = np.asarray(
                            Image.open(f).convert("L").resize((16, 12)),
                            dtype=np.float32) / 255.0
                    X.append(patch)
                    Y.append(GI[ch])
                    META.append({"file": f"synth_{i}", "line": li, "col": col,
                                 "ch": ch, "scale": scale, "synth": True})
                    n_cells += 1
            for f in glob.glob(f"{args.dump_prefix}_{tag}_*"):
                os.remove(f)

    if not X:
        print("no slices exported", file=sys.stderr)
        return 1
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, X=np.array(X, dtype=np.float32).reshape(-1, 12, 16),
             Y=np.array(Y, dtype=np.int64),
             meta=np.array([json.dumps(m) for m in META]))
    n1 = sum(1 for m in META if m["line"] == 0)
    print(f"synth: {len(X)} slices ({n1} line1) from {n_render} renders "
          f"({n_skip} skipped) -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
