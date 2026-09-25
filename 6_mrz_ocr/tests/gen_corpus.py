#!/usr/bin/env python3
"""Build a corpus of >= 100 MRZ test images for the OCR module.

For each combination we:
  1. Pick a (line1, line2) ground truth (canonical ERIKSSON sample or
     one of the ICAO examples we synthesised in module 1).
  2. Render it via gen_mrz_image with various (scale, hgap, vgap, skew,
     noise) settings.
  3. Write the PPM and a meta.json file listing the expected text.

After generation, run mrz_ocr_tool on every PPM and report recognition
accuracy. The total image count is designed to exceed 100.
"""
from __future__ import annotations
import json, os, subprocess, sys, itertools, random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GEN  = ROOT / "build" / "gen_mrz_image"
OCR  = ROOT / "build" / "mrz_ocr_tool"
DATA = ROOT / "data" / "corpus"
DATA.mkdir(parents=True, exist_ok=True)

# Ground-truth MRZ pairs (TD3). Each entry must be exactly 44 chars per
# line. We include:
#   - 5 base samples from the 1_mrz_decode module's data/test_cases.json
#   - additional samples we generate on the fly
BASE_PAIRS = [
    ("P<UTOERIKSSON<ANNA<MARIA<<<<<<<<<<<<<<<<<<<<",
     "L898902C36UTO6908061F9406236ZE184226B<<<<<18"),
    ("P<CHNWANG<<LI<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<",
     "E123456782CHN900101M3012311234567890<<<<<<<2"),
    ("P<USASMITH<<JOHN<<<<<<<<<<<<<<<<<<<<<<<<<<<<",
     "1234567897USA850215M250215A1B2C3D4E5<<<<<09<"),
    ("P<DEUMULLER<<HANS<<<<<<<<<<<<<<<<<<<<<<<<<<<",
     "C01X00T478DEU000101<32010100000000000000<1<<"),
    ("P<GBROBRIEN<<PATRICK<<<<<<<<<<<<<<<<<<<<<<<<",
     "GBR1234567IRL990101F291231GBRDOCUMENT0123<8<"),
]
# Sanity: must all be 44 chars.
for a, b in BASE_PAIRS:
    assert len(a) == 44 and len(b) == 44, (a, b)

# Image variants: each tuple (scale, hgap, vgap, skew, noise).
VARIANTS = [
    (4, 2, 6, 0, 0.00),
    (4, 2, 6, 0, 0.01),
    (4, 2, 6, 2, 0.00),
    (4, 2, 6, -2, 0.00),
    (5, 3, 8, 0, 0.00),
    (5, 3, 8, 0, 0.02),
    (6, 2, 10, 0, 0.00),
    (6, 3, 10, 1, 0.01),
    (4, 4, 8, 0, 0.00),
    (3, 2, 6, 0, 0.00),
    (4, 2, 6, 0, 0.03),
    (4, 2, 6, 0, 0.05),
]

def render(name: str, l1: str, l2: str, scale: int, hgap: int, vgap: int,
           skew: int, noise: float) -> Path:
    p = DATA / f"{name}.ppm"
    subprocess.check_call([
        str(GEN), str(p), l1, l2,
        f"scale={scale}", f"hgap={hgap}", f"vgap={vgap}",
        f"skew={skew}", f"noise={noise}",
    ])
    return p

def main():
    if not GEN.exists():
        print("gen_mrz_image not built -- run cmake --build build first",
              file=sys.stderr)
        return 1

    records = []
    random.seed(42)
    i = 0
    # We do all (pair, variant) combinations => 5 * 12 = 60. Then we add
    # variants of extra synthetic pairs to push past 100.
    for pair_idx, (l1, l2) in enumerate(BASE_PAIRS):
        for var_idx, (s, h, v, sk, n) in enumerate(VARIANTS):
            i += 1
            name = f"img_{i:04d}_p{pair_idx}_v{var_idx}"
            render(name, l1, l2, s, h, v, sk, n)
            records.append({
                "id": name,
                "line1": l1, "line2": l2,
                "scale": s, "hgap": h, "vgap": v,
                "skew": sk, "noise": n,
                "image": f"{name}.ppm",
            })
    # Extra synthetic pairs (8 more) * 6 light variants = 48 more.
    EXTRA_PAIRS = []
    for n in range(8):
        # Pseudo-random but deterministic MRZ-shaped strings.
        rnd = random.Random(1000 + n)
        chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<"
        l1 = "P<" + "".join(rnd.choice(chars) for _ in range(41))
        # line2: passport_no 9 + ck + nationality 3 + birth 6 + ck +
        # sex + expiry 6 + ck + personal_no 14 + ck + composite 1
        l2 = "".join(rnd.choice(chars) for _ in range(44))
        EXTRA_PAIRS.append((l1, l2))
    EXTRA_VARIANTS = [
        (4, 2, 6, 0, 0.00),
        (4, 2, 6, 0, 0.01),
        (5, 3, 8, 0, 0.02),
        (6, 2, 10, 1, 0.00),
        (4, 2, 6, 0, 0.04),
        (4, 4, 8, -1, 0.01),
    ]
    for pair_idx, (l1, l2) in enumerate(EXTRA_PAIRS):
        for var_idx, (s, h, v, sk, n) in enumerate(EXTRA_VARIANTS):
            i += 1
            name = f"img_{i:04d}_x{pair_idx}_v{var_idx}"
            render(name, l1, l2, s, h, v, sk, n)
            records.append({
                "id": name,
                "line1": l1, "line2": l2,
                "scale": s, "hgap": h, "vgap": v,
                "skew": sk, "noise": n,
                "image": f"{name}.ppm",
            })

    out = DATA / "corpus.json"
    out.write_text(json.dumps({"version": 1, "count": len(records),
                                "records": records}, indent=2))
    print(f"generated {len(records)} test images")
    return 0

if __name__ == "__main__":
    sys.exit(main())
