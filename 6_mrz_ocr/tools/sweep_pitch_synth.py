#!/usr/bin/env python3
"""Char accuracy vs px-per-character on synthetic bands with exact GT.

Why synthetic: the two photo sets have no trustworthy text label.  The
renderer does -- it was handed the string it drew.  That makes it the
only instrument that can answer "does accuracy depend on glyph size,
and where does it break" without a GT argument.

The knob is the renderer's `scale` (px per template bit) and `hgap`
(inter-character gap).  One character is 8 template bits plus the gap,
so

    px/char = 8*scale + hgap

and the training corpus only ever drew scale 3..6, i.e. 26..51 px/char.
Anything smaller is out of distribution by construction.

Two conditions per point:
  raw    MRZ_OCR_PPC=8   -- the model sees the glyphs at native size
  prod   default         -- the pipeline upscales to 26 px/char first

If `prod` is flat where `raw` falls off, the upscaler is doing its job.
If both fall off, the model itself cannot read small glyphs and the fix
has to be in training, not in preprocessing.

Usage:
    python3 tools/sweep_pitch_synth.py [--scales 2,3,4,5,6,7,8]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GEN = ROOT / "build" / "gen_mrz_image"
TOOL = ROOT / "build" / "mrz_ocr_cnn_tool"
CORPUS = ROOT / "data" / "corpus" / "corpus.json"

# Ten real-looking TD3 records, distinct enough that no two share a
# check-digit structure (so a systematic swap cannot look like a hit).
RECORDS = [
    ("P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<",
     "L898902C36UTO6908061F9406236ZE184226B<<<<<18"),
    ("P<CHEDE<MASENFELD<<PSE<RE<<<<<<<<<<<<<<<<<<<",
     "5210002356CHE7108019M2002293<5<55<<<<<<<<<<6"),
    ("P<GBRDENT<<JAMES<ALEXANDER<<<<<<<<<<<<<<<<<<",
     "7408122A5GBR8001011M2501012<<<<<<<<<<<<<<<02"),
    ("P<FRADUPONT<<JEAN<PIERRE<<<<<<<<<<<<<<<<<<<<",
     "13AA45678FRA8505123M3008154ABCDEF123<<<<<57"),
    ("P<DEUMUELLER<<HANS<JOSEF<<<<<<<<<<<<<<<<<<<<",
     "C01X00T47DEU9002287M2204198<<<<<<<<<<<<<<<91"),
    ("P<ITAROSSI<<MARCO<<<<<<<<<<<<<<<<<<<<<<<<<<<",
     "YA1234567ITA7509302F2711206<<<<<<<<<<<<<<<44"),
    ("P<ESPGARCIA<<LUIS<MIGUEL<<<<<<<<<<<<<<<<<<<<",
     "PA9876543ESP6804159M2406073<<<<<<<<<<<<<<<83"),
    ("P<NLDVANDIJK<<PIETER<<<<<<<<<<<<<<<<<<<<<<<<",
     "NX1234567NLD9207115F2903157<<<<<<<<<<<<<<<27"),
    ("P<POLNOWAK<<ANNA<KATARZYNA<<<<<<<<<<<<<<<<<<",
     "ZS4455667POL8812034F2601248<<<<<<<<<<<<<<<65"),
    ("P<SWELINDQVIST<<ERIK<<<<<<<<<<<<<<<<<<<<<<<",
     "KU7788990SWE9506261M2810119<<<<<<<<<<<<<<<39"),
]


def decode(ppm: Path, ppc: str = "") -> str:
    env = dict(os.environ)
    env.update(MRZ_OCR_LINE_MODELS="1", MRZ_OCR_PAR="0", MRZ_OCR_CRNN="0")
    if ppc:
        env["MRZ_OCR_PPC"] = ppc
    p = subprocess.run([str(TOOL), str(ppm)], capture_output=True, text=True,
                       env=env, timeout=120)
    for ln in p.stdout.splitlines():
        if ln.startswith("result.line2"):
            return ln.partition(":")[2].strip()
    return ""


def char_acc(out: str, gt: str) -> float:
    if len(out) != 44:
        return 0.0
    return sum(1 for a, b in zip(out, gt) if a == b) / 44.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scales", default="2,3,4,5,6,7,8")
    ap.add_argument("--hgaps", default="1,2,3")
    ap.add_argument("--out", default="/tmp/pitch_sweep.json")
    args = ap.parse_args()

    tmp = Path(tempfile.mkdtemp(prefix="ps_"))
    rows = []
    for scale in [int(x) for x in args.scales.split(",")]:
        for hgap in [int(x) for x in args.hgaps.split(",")]:
            ppc = 8 * scale + hgap
            accs = {"raw": [], "prod": []}
            nok = {"raw": 0, "prod": 0}
            for i, (l1, l2) in enumerate(RECORDS):
                ppm = tmp / f"s{scale}_h{hgap}_{i}.ppm"
                subprocess.run([str(GEN), str(ppm), l1, l2,
                                f"scale={scale}", f"hgap={hgap}",
                                "vgap=6", "margin=12"],
                               check=True, capture_output=True)
                for cond, ppc_env in (("raw", "8"), ("prod", "")):
                    out = decode(ppm, ppc_env)
                    accs[cond].append(char_acc(out, l2))
                    nok[cond] += (out == l2)
            n = len(RECORDS)
            rows.append({
                "scale": scale, "hgap": hgap, "ppc": ppc, "n": n,
                "raw": sum(accs["raw"]) / n, "prod": sum(accs["prod"]) / n,
                "raw_exact": nok["raw"], "prod_exact": nok["prod"],
            })
            r = rows[-1]
            print(f"scale={scale} hgap={hgap} px/char={ppc:>3}  "
                  f"raw {r['raw']*100:>6.2f}% ({r['raw_exact']}/{n})   "
                  f"prod {r['prod']*100:>6.2f}% ({r['prod_exact']}/{n})",
                  flush=True)

    json.dump(rows, open(args.out, "w"), indent=1)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
