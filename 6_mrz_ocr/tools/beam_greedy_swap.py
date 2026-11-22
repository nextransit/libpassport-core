#!/usr/bin/env python3
"""Does the greedy line beat the beam's winner on the samples it lost?

WHY
---
`mrz_beam_search()` seeds the incumbent with "nothing found" and only
accepts leaves that satisfy the checksum.  The greedy decode is NOT a
candidate.  So when the greedy line is ALREADY checksum-valid and the
beam finds a different valid line, the beam can only replace it with a
strictly lower-probability reading -- the beam's score is a DELTA
against the greedy decode, and every alternative it may switch to has a
lower probability.

Concretely on 2026-9-29-L: the greedy line has '0' at column 29
(p = 0.82, the top-1 class) and the beam replaced it with 'O'
(p = 0.18) because 'O' made the checksum close.

This script measures, for every sample, whether the greedy line or the
beam line has the higher total log-probability, and whether the higher
one matches the ground truth.

Usage:
    python3 tools/beam_greedy_swap.py
"""
from __future__ import annotations

import json
import math
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "6_mrz_ocr" / "build" / "mrz_ocr_cnn_tool"
BANDS = Path("/tmp/holdout_export2")
DIAG_DIR = Path("/tmp/diags")
GT_JSON = ROOT / "10_ocr_research" / "gt_review_loop.json"


def logprob(line: str, diag_line: dict) -> float:
    """Total log-probability of a line under the network's own output."""
    s = 0.0
    for c in range(44):
        ch = line[c]
        p = 1e-6
        for cc, pp in diag_line["chars"][c]["top"]:
            if cc == ch:
                p = max(pp, 1e-6)
                break
        s += math.log(p)
    return s


def main():
    gt = {}
    for k, v in json.loads(GT_JSON.read_text()).items():
        if isinstance(v, (list, tuple)) and len(v) == 2:
            gt[Path(k).stem] = v

    n = 0
    beam_wins = 0
    greedy_wins = 0
    tie = 0
    for ppm in sorted(BANDS.glob("*.ppm")):
        stem = ppm.stem
        if stem not in gt:
            continue
        dp = DIAG_DIR / f"{stem}.json"
        if not dp.exists():
            continue
        d = json.loads(dp.read_text())
        beam = d.get("line2", "")
        greedy = d.get("line2", "")
        # the diag was produced with the beam ON, so re-read the greedy
        # line from the tool with the beam OFF
        env = dict(os.environ)
        env.update({"MRZ_OCR_LINE_MODELS": "1", "MRZ_OCR_PAR": "0",
                    "MRZ_OCR_CRNN": "0", "MRZ_OCR_FUSE": "0",
                    "MRZ_OCR_NOBEAM": "1"})
        out = subprocess.run([str(TOOL), str(ppm)], capture_output=True,
                             text=True, env=env).stdout
        for line in out.splitlines():
            if line.startswith("result.line2"):
                greedy = line.split(":", 1)[1].strip()
        if beam == greedy:
            continue
        n += 1
        lb = logprob(beam, d["lines"][1])
        lg = logprob(greedy, d["lines"][1])
        g2 = gt[stem][1]
        if abs(lb - lg) < 1e-9:
            tie += 1
            verdict = "TIE"
        elif lb > lg:
            beam_wins += 1
            verdict = "BEAM more likely"
        else:
            greedy_wins += 1
            verdict = "GREEDY more likely"
        print(f"{stem:<24} {verdict:<16} "
              f"beam_lp={lb:9.4f} greedy_lp={lg:9.4f} "
              f"beam_exact={beam == g2} greedy_exact={greedy == g2}")
    print(f"\nchanged lines: {n}  beam-more-likely={beam_wins} "
          f"greedy-more-likely={greedy_wins} tie={tie}")


if __name__ == "__main__":
    main()
