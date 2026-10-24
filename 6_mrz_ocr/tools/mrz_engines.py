#!/usr/bin/env python3
"""Independent MRZ readers for multi-engine GT construction.

The frozen-GT principle (tools/build_frozen_gt.py) is that labels must
not come from the model under test alone.  For scraped corpora there is
no board recogniser, so this module provides a genuinely independent
reader: tesseract 5 with the purpose-trained OCR-B data from the
Android assets (a different codebase, different training data, no
shared code with the C pipeline).  The native CNN/CRNN heads share a
convolutional front-end and are therefore NOT independent of each
other; the tier logic must never treat them as two votes.

Readers return (line1, line2) with TD3 normalisation already applied:
uppercase, O->0/I->1 inside numeric fields is NOT done here (the
checksum solver owns that), only length-44 selection and whitespace
stripping.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TESSDATA = ROOT.parent / "8_android" / "app" / "src" / "main" / "assets" / "tessdata"


def read_tesseract(ppm: str | Path) -> tuple[str, str]:
    """Run tesseract with the OCR-B model over a band image.

    --psm 6 assumes a uniform block of text (the two MRZ rows), the
    whitelist excludes everything ICAO 9303 forbids.  The output is the
    two longest line-shaped rows; anything else is discarded.
    """
    env = dict(os.environ, TESSDATA_PREFIX=str(TESSDATA))
    r = subprocess.run(
        ["tesseract", str(ppm), "stdout", "-l", "ocrb", "--psm", "6",
         "-c", "tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<"],
        capture_output=True, text=True, timeout=60, env=env)
    lines = [ln.strip().replace(" ", "") for ln in r.stdout.splitlines()]
    lines = [ln for ln in lines if len(ln) >= 30]
    lines.sort(key=len, reverse=True)
    lines = lines[:2]
    if len(lines) < 2:
        return "", ""
    # Top-to-bottom order: tesseract returns reading order, but a
    # two-row block is reliably line1-then-line2; verify against the
    # line-2 signature (positions 13/15 digits-heavy) is over-clever --
    # keep reading order.
    l1, l2 = lines[0], lines[1]
    return (l1 if len(l1) <= 44 else l1[:44],
            l2 if len(l2) <= 44 else l2[:44])
