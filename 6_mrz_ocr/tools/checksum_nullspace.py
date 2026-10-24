#!/usr/bin/env python3
"""How much does a verifying TD3 line 2 actually constrain the string?

Motivation
----------
`line2_checksum_ok()` is used in three places in mrz_ocr_cnn.c: to gate
the checksum beam search, to pick between candidate rows, and (in the
tooling) as a proxy for "this decode is correct".  All three assume the
check digits are a strong filter.

They are not.  ICAO 9303 weights are 7,3,1 cycling and values are
0-9 -> 0-9, A-Z -> 10-35, '<' -> 0.  A substitution at position i changes
the sum by d*w(i), where d is the value difference.  The change is
invisible to the check digit whenever d*w(i) = 0 (mod 10).  Because
w(i) in {7,3,1} and d is a value difference, whole families of
substitutions are undetectable -- including the ones OCR actually makes:

    d = 24  (0 <-> O)   needs w-sum = 0 (mod 5)  -> e.g. 3+7 = 10
    d = 20  (1 <-> L, 0 <-> K)  invisible at ANY single position
    d = 10  (any digit <-> the letter 10 above it)  invisible singly

This script enumerates the null space exactly, so the report can state
"a verifying line 2 admits N variants that also verify" instead of
implying verification equals correctness.

Usage:
    python3 tools/checksum_nullspace.py "E62588I562CHN9505209F3503269G1Z152520121S504"
    python3 tools/checksum_nullspace.py --stats /tmp/gt_l2_passport.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys

ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ<"
W = (7, 3, 1)


def val(c: str) -> int:
    if c == "<":
        return 0
    if "0" <= c <= "9":
        return ord(c) - 48
    return ord(c) - 55


def cd(s: str) -> int:
    return sum(val(c) * W[i % 3] for i, c in enumerate(s)) % 10


def l2_ok(l2: str) -> bool:
    if len(l2) != 44:
        return False
    return all(l2[p] == str(cd(l2[a:b])) for p, (a, b) in
               ((9, (0, 9)), (19, (13, 19)), (27, (21, 27)), (42, (28, 42)))) \
        and l2[43] == str(cd(l2[0:10] + l2[13:20] + l2[21:28] + l2[28:43]))


# Positions covered by each check digit, and the check digit position.
GROUPS = [((0, 9), 9), ((13, 19), 19), ((21, 27), 27), ((28, 42), 42)]
OVERALL = (43, "0:10|13:20|21:28|28:43")


def group_of(i: int):
    for (a, b), p in GROUPS:
        if a <= i < b:
            return (a, b), p
    if i == 43:
        return "overall", 43
    if i in (9, 19, 27, 42):
        return "checkdigit", i
    return None  # 10,11,12,20 -> uncovered


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("line2", nargs="?")
    ap.add_argument("--stats")
    args = ap.parse_args()

    if args.stats:
        gt = json.load(open(args.stats))
        n_amb = 0
        tot = 0
        worst = []
        for name, rec in gt.items():
            l2 = rec["l2"]
            if not l2_ok(l2):
                continue
            tot += 1
            # count single-position alternatives that keep all 5 digits
            free = 0
            for i in range(44):
                if group_of(i) is None:
                    continue
                for c in ALPHABET:
                    if c == l2[i]:
                        continue
                    t = l2[:i] + c + l2[i + 1:]
                    if l2_ok(t):
                        free += 1
            if free:
                n_amb += 1
            worst.append((free, name, l2))
        worst.sort(reverse=True)
        print(f"verifying samples: {tot}, with >=1 single-char "
              f"variant that also verifies: {n_amb}")
        print(f"{'#alts':>6}  sample / line2")
        for f, n, l2 in worst[:10]:
            print(f"{f:>6}  {n}\n        {l2}")
        return 0

    if not args.line2:
        ap.error("need a line2 or --stats")

    l2 = args.line2.strip()
    print(f"line2: {l2}")
    print(f"verifies: {l2_ok(l2)}")
    print("\nuncovered by any check digit (free to vary):")
    free = [i for i in range(44) if group_of(i) is None]
    print("  positions", free, "=", "".join(l2[i] for i in free))

    print("\nper-position alternatives that still verify "
          "(single substitution):")
    total_alt = 1
    for i in range(44):
        if group_of(i) is None:
            continue
        alts = [c for c in ALPHABET if c != l2[i]
                and l2_ok(l2[:i] + c + l2[i + 1:])]
        if alts:
            total_alt *= (1 + len(alts))
            print(f"  idx {i:>2} w={W[i % 3]} {l2[i]!r} -> "
                  f"{''.join(alts)}   ({len(alts)})")
    print(f"\nsingle-substitution null space size: {total_alt} "
          f"(the true string is 1 of them)")

    # the pair case that bit us: 0<->O at weights 3 and 7
    print("\ndemonstration: 0 <-> O at two positions whose weights sum "
          "to 10 is invisible")
    demo = "E062480684CHN8201113F251225501S8IBBG51B52Z66"
    alt = demo[:1] + "O" + demo[2:6] + "O" + demo[7:]
    print(f"  {demo}  verifies={l2_ok(demo)}")
    print(f"  {alt}  verifies={l2_ok(alt)}   <- same five check digits")
    return 0


if __name__ == "__main__":
    sys.exit(main())
