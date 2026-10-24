#!/usr/bin/env python3
"""Validate / normalise a hand-transcribed TD3 MRZ ground-truth file.

The point of this script is that a hand-transcribed GT is only worth as much
as the checks applied to it.  Everything below is a *structural* check that
follows from ICAO 9303 part 3 (TD3), so a violation is always a transcription
error and never a legitimate MRZ.

    line 1   P<ISSUINGSTATE<SURNAME<<GIVEN<NAMES...
             idx 0        must be 'P'  (passport)
             idx 1        must be '<'
             idx 2..4     three upper-case letters (issuing state)
             idx 5..43    [A-Z<] only, no digits

    line 2   DOCNUM(9) CD NAT(3) DOB(6) CD SEX EXP(6) CD PERSONAL(14) CD CD
             idx 9        check digit over 0..8
             idx 10..12   three upper-case letters (nationality)
             idx 13..18   YYMMDD, all digits
             idx 19       check digit over 13..18
             idx 20       'M' | 'F' | '<'
             idx 21..26   YYMMDD, all digits
             idx 27       check digit over 21..26
             idx 42       check digit over 28..41
             idx 43       check digit over 0..9 + 13..19 + 21..27 + 28..42
             idx 0..43    [A-Z0-9<] only

Usage:
    python3 validate_gt.py gt.json [--fix-prefix] [--write out.json]
"""
import argparse
import json
import sys

ALNUM = set("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ<")
ALPHA = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ<")
DIGIT = set("0123456789")

W = (7, 3, 1)


def check_digit(s: str) -> int:
    """ICAO 9303 check digit: sum of value*weight mod 10, '<' counts as 0."""
    total = 0
    for i, ch in enumerate(s):
        if ch == "<":
            v = 0
        elif ch.isdigit():
            v = ord(ch) - 48
        else:
            v = ord(ch) - 55
        total += v * W[i % 3]
    return total % 10


def cd_at(line: str, pos: int, start: int, end: int) -> bool:
    return line[pos] == str(check_digit(line[start:end]))


def validate(name: str, rec: dict) -> list[str]:
    errs: list[str] = []
    l1 = rec.get("l1", "")
    l2 = rec.get("l2", "")

    if len(l1) != 44:
        errs.append(f"l1 length {len(l1)} != 44")
    if len(l2) != 44:
        errs.append(f"l2 length {len(l2)} != 44")
    if len(l1) != 44 or len(l2) != 44:
        return errs

    # ---- line 1 grammar ----
    if l1[0] != "P":
        errs.append(f"l1[0]={l1[0]!r} should be 'P'")
    if l1[1] != "<":
        errs.append(f"l1[1]={l1[1]!r} should be '<'")
    if not set(l1[2:5]) <= ALPHA - {"<"}:
        errs.append(f"l1[2:5]={l1[2:5]!r} should be 3 letters")
    bad = sorted(set(l1[5:]) - ALPHA)
    if bad:
        errs.append(f"l1[5:] has illegal chars {bad}")

    # ---- line 2 grammar ----
    bad = sorted(set(l2) - ALNUM)
    if bad:
        errs.append(f"l2 has illegal chars {bad}")
    if not set(l2[10:13]) <= ALPHA - {"<"}:
        errs.append(f"l2[10:13]={l2[10:13]!r} should be 3 letters")
    if not set(l2[13:19]) <= DIGIT:
        errs.append(f"l2[13:19]={l2[13:19]!r} DOB should be 6 digits")
    if l2[20] not in "MF<":
        errs.append(f"l2[20]={l2[20]!r} sex should be M/F/<")
    if not set(l2[21:27]) <= DIGIT:
        errs.append(f"l2[21:27]={l2[21:27]!r} expiry should be 6 digits")

    # ---- check digits ----
    for pos, (a, b), label in (
        (9, (0, 9), "docnum"),
        (19, (13, 19), "dob"),
        (27, (21, 27), "expiry"),
        (42, (28, 42), "personal"),
    ):
        if not cd_at(l2, pos, a, b):
            errs.append(
                f"l2[{pos}]={l2[pos]!r} != {check_digit(l2[a:b])} (check digit over {label} {l2[a:b]!r})"
            )
    overall_src = l2[0:10] + l2[13:20] + l2[21:28] + l2[28:43]
    if l2[43] != str(check_digit(overall_src)):
        errs.append(f"l2[43]={l2[43]!r} != {check_digit(overall_src)} (overall)")
    return errs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("gt")
    ap.add_argument("--fix-prefix", action="store_true",
                    help="force l1[0:2]='P<' when only the prefix is wrong")
    ap.add_argument("--write")
    args = ap.parse_args()

    gt = json.load(open(args.gt))
    clean, dirty = {}, {}
    for name, rec in sorted(gt.items()):
        if args.fix_prefix and rec["l1"][:1] == "P" and rec["l1"][1] != "<":
            rec = dict(rec)
            rec["l1"] = "P<" + rec["l1"][2:]
        errs = validate(name, rec)
        if errs:
            dirty[name] = errs
        else:
            clean[name] = rec

    for name, errs in dirty.items():
        print(f"\n{name}")
        for e in errs:
            print(f"    {e}")

    # how many are checksum-clean (all errors gone) vs only structural
    n = len(gt)
    print(f"\n=== {len(clean)}/{n} fully clean, {len(dirty)} with issues ===")
    if args.write:
        json.dump(clean, open(args.write, "w"), indent=1, ensure_ascii=False)
        print(f"wrote {len(clean)} clean records -> {args.write}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
