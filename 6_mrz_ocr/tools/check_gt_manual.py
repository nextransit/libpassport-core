#!/usr/bin/env python3
"""Validate a hand-read MRZ transcription against the ICAO check digits.

Line 2 carries five check digits over four fields plus a composite, so a
transcription that satisfies all five is correct with probability
1 - 1e-5 -- far stronger evidence than any OCR consensus.  Line 1 has no
such protection; the only structural tests are "starts with P", "country
code at 2..4", "name field is A-Z and '<'".  That asymmetry is the whole
reason the product gates on line 2.

Usage:
    python3 tools/check_gt_manual.py [path.json]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

W = (7, 3, 1)
FIELDS = ((9, 0, 9), (19, 13, 19), (27, 21, 27), (42, 28, 42))


def val(c: str) -> int:
    if c == "<":
        return 0
    if "0" <= c <= "9":
        return ord(c) - 48
    return ord(c) - 55


def cd(s: str) -> int:
    return sum(val(c) * W[i % 3] for i, c in enumerate(s)) % 10


def check(line: str):
    """Return (ok, [(pos, got, want, field)])."""
    bad = []
    if len(line) != 44:
        return False, [("len", len(line), 44, "")]
    for pos, a, b in FIELDS:
        want = str(cd(line[a:b]))
        if line[pos] != want:
            bad.append((pos, line[pos], want, line[a:b]))
    comp = str(cd(line[0:10] + line[13:20] + line[21:28] + line[28:43]))
    if line[43] != comp:
        bad.append((43, line[43], comp, "composite"))
    return not bad, bad


def main() -> int:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/gt_manual.json")
    recs = json.load(open(path))["records"]
    nok = 0
    for r in recs:
        ok, bad = check(r["line2"])
        nok += ok
        if not ok:
            print(f"{r['id']:<16} FAIL {bad}")
        l1 = r.get("line1", "")
        if len(l1) != 44:
            print(f"{r['id']:<16} line1 len {len(l1)}")
        elif not (l1[0] == "P" and l1[1] in "<ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
            print(f"{r['id']:<16} line1 head {l1[:6]!r}")
    print(f"\nline2 checksum: {nok}/{len(recs)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
