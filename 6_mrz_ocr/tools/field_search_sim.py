#!/usr/bin/env python3
"""Simulate a FIELD search that TRUSTS the check digit the network read.

WHY
---
The C decoder derives the check digits from the field bodies, so the
checksum never rejects a wrong reading.  This simulation does the
opposite: it takes the digit the network actually READ at the check
column (col 9/19/27/42) as a CONSTRAINT and searches the field body
for the reading whose ICAO hash equals it.

That is the classic MRZ error-correction step, and it is strictly
stronger than the current decoder:
  * the constraint is an INDEPENDENT observation (the check-digit cell
    is a different cell from the body), so it can rule a reading out;
  * the current decoder's derived digit can never rule anything out.

The search is an exact DP over (own residue, composite residue) --
100 states -- so it is exact, not a beam.

Usage:
    python3 tools/field_search_sim.py [TOPK]
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DIAG_DIR = Path("/tmp/diags")
GT_JSON = ROOT / "10_ocr_research" / "gt_review_loop.json"

W = (7, 3, 1)
# field name, body start, body end (exclusive), check-digit column
FIELDS = [
    ("passport", 0, 9, 9),
    ("birth", 13, 19, 19),
    ("expiry", 21, 27, 27),
    ("personal", 28, 42, 42),
]
COMPOSITE_COL = 43


def val(ch: str) -> int:
    if ch.isdigit():
        return ord(ch) - 48
    return ord(ch) - 55


def cd_of(seq: str) -> int:
    return sum(val(c) * W[i % 3] for i, c in enumerate(seq)) % 10


def comp_index(c: int) -> int:
    if c <= 9:
        return c
    if c <= 19:
        return c - 3
    return c - 4


def comp_weight(c: int) -> int:
    return W[comp_index(c) % 3]


def field_weights(start: int, n: int):
    """The field-local weight sequence: 7,3,1 restarting at the body."""
    return [W[k % 3] for k in range(n)]


def solve_field(cands, start, n, cd_col, cd_cands=None):
    """Exact DP over (own residue, composite residue).

    cands: list per column of [(char, logprob), ...]
    Returns (best_score, best_body) for the body whose own hash matches
    the digit READ at cd_col, plus the best body overall.
    """
    # state index = own * 10 + comp
    cur = {}
    for j in range(n):
        c = start + j
        w = field_weights(start, n)[j]
        cw = comp_weight(c)
        nxt = {}
        for (own, comp), (sc, body) in cur.items():
            for ch, lp in cands[j]:
                no = (own + val(ch) * w) % 10
                nc = (comp + val(ch) * cw) % 10
                ni = (no, nc)
                s = sc + lp
                if ni not in nxt or s > nxt[ni][0]:
                    nxt[ni] = (s, body + ch)
        cur = nxt
    # the read digit at the check column
    read = None
    for ch, lp in (cd_cands or []):
        if ch.isdigit():
            read = int(ch)
            break
    best_ok = None
    best_any = None
    for (own, comp), (sc, body) in cur.items():
        if best_any is None or sc > best_any[0]:
            best_any = (sc, body, own, comp)
        if read is not None and own == read:
            if best_ok is None or sc > best_ok[0]:
                best_ok = (sc, body, own, comp)
    return best_ok, best_any, read


def main():
    topk = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    gt = {}
    for k, v in json.loads(GT_JSON.read_text()).items():
        if isinstance(v, (list, tuple)) and len(v) == 2:
            gt[Path(k).stem] = v

    nfix = 0
    ntot = 0
    for p in sorted(DIAG_DIR.glob("*.json")):
        stem = p.stem
        if stem not in gt:
            continue
        ntot += 1
        d = json.loads(p.read_text())
        g1, g2 = gt[stem]
        line = d["lines"][1]
        base = d.get("line2", "")
        if base == g2:
            continue

        # candidate list per column
        def cands(c):
            return [(ch, math.log(max(pp, 1e-9)))
                    for ch, pp in line["chars"][c]["top"][:topk] if ch]

        # only touch the fields whose OWN check digit fails against the
        # digit the network READ
        new = list(base)
        changed = False
        for name, s, e, cd in FIELDS:
            body = base[s:e]
            read = None
            for ch, lp in cands(e):
                if ch.isdigit():
                    read = int(ch)
                    break
            if read is None:
                continue
            if cd_of(body) == read:
                continue        # this field is already consistent
            ok, any_, rd = solve_field([cands(s + k) for k in range(e - s)],
                                        s, e - s, cd, cands(e))
            if ok is None:
                continue
            sc, body_new, own, comp = ok
            new[s:e] = body_new
            changed = True
            print(f"{stem:<24} {name:<10} body {body} -> {body_new} "
                  f"(read={read}, own={own})")
        if changed:
            new[42] = str(cd_of("".join(new[28:42])))
            comp_seq = new[0:10] + new[13:20] + new[21:43]
            new[43] = str(cd_of(comp_seq))
            ok = "".join(new) == g2
            nfix += 1
            print(f"{stem:<24} => {''.join(new)}")
            print(f"{'':<24}    gt {g2}   {'MATCH' if ok else 'differs'}")
    print(f"\n{full if False else ''}attempted field repair on {nfix} samples "
          f"of {ntot}")


if __name__ == "__main__":
    main()
