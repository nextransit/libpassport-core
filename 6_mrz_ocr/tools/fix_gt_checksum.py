#!/usr/bin/env python3
"""Repair a hand-read line 2 using the ICAO check digits as a solver.

Reading 44 glyphs off a photo by eye gets ~1 character wrong per line;
the check digits then fail, but they also *localise* the error.  This
tool searches the neighbourhood of the hand reading for the strings
that satisfy all five check digits, and reports exactly which positions
had to change.

That is not circular.  The hand reading supplies the hypothesis, the
check digits supply an independent arithmetic constraint with a 1e-5
false-accept rate, and the reported delta is then confirmed by eye on a
4x crop of those positions.  A search that only succeeds at Hamming 2
inside the optional-data field is telling you the hand reading was off
by two, not that the check digits are weak.

Search order (stop at the first tier that yields a solution):
    0  delete one character      (hand count was off by one)
    1  substitute one character  (any of 44 positions)
    2  substitute two characters (optional data 28..42 only -- that is
       where the eye fails, because those glyphs are arbitrary letters)

Usage:
    python3 tools/fix_gt_checksum.py /tmp/gt_manual.json /tmp/gt_fixed.json
"""
from __future__ import annotations

import json
import string
import sys
from pathlib import Path

ALPHA = string.digits + string.ascii_uppercase + "<"
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


def ok(line: str) -> bool:
    if len(line) != 44:
        return False
    for pos, a, b in FIELDS:
        if line[pos] != str(cd(line[a:b])):
            return False
    return line[43] == str(cd(line[0:10] + line[13:20] + line[21:28]
                              + line[28:43]))


def solve(line: str):
    """Return (fixed, tier, delta) or (None, None, None)."""
    if len(line) == 45:
        for i in range(45):
            cand = line[:i] + line[i + 1:]
            if ok(cand):
                return cand, 0, [(i, line[i], "del")]
        return None, None, None
    if len(line) != 44:
        return None, None, None
    if ok(line):
        return line, -1, []
    for i in range(44):
        for c in ALPHA:
            if c == line[i]:
                continue
            cand = line[:i] + c + line[i + 1:]
            if ok(cand):
                return cand, 1, [(i, line[i], c)]
    for i in range(28, 43):
        for c in ALPHA:
            if c == line[i]:
                continue
            s1 = line[:i] + c + line[i + 1:]
            for j in range(i + 1, 43):
                for d in ALPHA:
                    if d == s1[j]:
                        continue
                    cand = s1[:j] + d + s1[j + 1:]
                    if ok(cand):
                        return cand, 2, [(i, line[i], c), (j, line[j], d)]
    return None, None, None


def main() -> int:
    src = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/gt_manual.json")
    dst = Path(sys.argv[2] if len(sys.argv) > 2 else "/tmp/gt_fixed.json")
    recs = json.load(open(src))["records"]
    tiers = {0: 0, 1: 0, 2: 0, -1: 0}
    out = []
    for r in recs:
        fixed, tier, delta = solve(r["line2"])
        if fixed is None:
            print(f"{r['id']:<16} UNSOLVED  {r['line2']}")
            out.append(dict(r, line2=None))
            continue
        tiers[tier] += 1
        tag = {-1: "clean", 0: "del1", 1: "sub1", 2: "sub2"}[tier]
        ds = " ".join(f"[{i}]{a}->{b}" for i, a, b in delta)
        print(f"{r['id']:<16} {tag:<5} {ds}")
        out.append(dict(r, line2=fixed, fix=tag))
    json.dump({"records": out}, open(dst, "w"), indent=1)
    print(f"\n{'  '.join(f'{k}:{v}' for k, v in tiers.items())}")
    print(f"wrote {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
