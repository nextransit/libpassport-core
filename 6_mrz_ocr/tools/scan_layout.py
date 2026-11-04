#!/usr/bin/env python3
"""Layout-classifier regression sweep (TD3 2x44 vs CS/CD 1x30 vs dropped frame).

Why this is a file and not a one-off heredoc: the three-way decision
(classify the bottom row, then decide whether the frame is usable at all)
is the one place where a "sensible looking" threshold change silently
turns a dropped frame into a confidently wrong 30-character decode.  Every
time the cascade changed, the sweep had to be retyped by hand, and the
numbers in the report came from slightly different matrices each time.

The matrix here is fixed:

    categories x scales {3,4,5,6} x conditions {clean, noise, rot, noise+rot}
    = 16 renders per category, 48 total

Expected verdicts:

    permit30    (one 30-cell row)          -> CS_CD_30
    td3         (two 44-cell rows)         -> TD3_44
    td3_trunc   (one 44-cell row, no pair) -> UNKNOWN   (frame dropped)

The third category is the safety property, not an accuracy target: a TD3
passport whose line 1 was lost to glare must be dropped, never mistaken
for a 30-character travel permit.

Usage:
  python3 tools/scan_layout.py [--tool build/mrz_ocr_cnn_tool]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
BUILD = HERE.parent / "build"

# ICAO specimen lines.  Line 2 is the standard L898902C3... example; line 1
# is its matching TD3 header.
TD3_L1 = "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<"
TD3_L2 = "L898902C36UTO6908061F9406236ZE184226B<<<<<18"
# CS/CD travel permit: one 30-cell row.  Same alphabet, no filler run to
# speak of -- 30 cells fill up fast.
PERMIT = "CSCH81234567<9501011<0300101<2"

SCALES = (3, 4, 5, 6)
CONDITIONS = (
    ("clean", []),
    ("noise", ["noise=0.02", "noise-seed=7"]),
    ("rot", ["rot=1.0"]),
    ("noise+rot", ["noise=0.02", "noise-seed=7", "rot=1.0"]),
)

CATEGORIES = (
    ("permit30", PERMIT, "", 1, "CS_CD_30"),
    ("td3", TD3_L1, TD3_L2, 2, "TD3_44"),
    ("td3_trunc", TD3_L1, "", 1, "UNKNOWN"),
)


def run_case(gen: Path, tool: Path, out: Path, l1: str, l2: str,
             nlines: int, opts: list[str]) -> tuple[str, dict]:
    cmd = [str(gen), str(out), l1, l2, f"lines={nlines}"] + opts
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        return "GEN_FAIL", {}
    p = subprocess.run([str(tool), str(out)], capture_output=True, text=True,
                       env={"MRZ_OCR_LINE_MODELS": "1", "MRZ_OCR_CRNN": "0",
                            "PATH": "/usr/bin:/bin"})
    kv = {}
    for ln in p.stdout.splitlines():
        if ":" in ln:
            k, _, v = ln.partition(":")
            kv[k.strip()] = v.strip()
    return kv.get("result.layoutname", "?"), kv


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tool", default=str(BUILD / "mrz_ocr_cnn_tool"))
    ap.add_argument("--gen", default=str(BUILD / "gen_mrz_image"))
    args = ap.parse_args()

    tool = Path(args.tool).resolve()
    gen = Path(args.gen).resolve()
    for p in (tool, gen):
        if not p.exists():
            print(f"missing: {p}", file=sys.stderr)
            return 2

    tmp = Path(tempfile.mkdtemp(prefix="layout_"))
    total_ok = total = 0
    for name, l1, l2, nlines, want in CATEGORIES:
        ok = 0
        fails = []
        for scale in SCALES:
            for cname, copts in CONDITIONS:
                out = tmp / f"{name}_s{scale}_{cname.replace('+', '_')}.ppm"
                got, kv = run_case(gen, tool, out, l1, l2, nlines,
                                   [f"scale={scale}"] + copts)
                total += 1
                if got == want:
                    ok += 1
                    total_ok += 1
                else:
                    fails.append(f"s{scale}/{cname}->{got}")
        print(f"{name:11} {ok:2}/{len(SCALES)*len(CONDITIONS)}  -> {want}")
        if fails:
            print(f"            miss: {', '.join(fails)}")
    print(f"\ntotal {total_ok}/{total}")
    return 0 if total_ok == total else 1


if __name__ == "__main__":
    sys.exit(main())
