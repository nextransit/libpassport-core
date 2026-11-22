#!/usr/bin/env python3
"""Use the COMPOSITE check digit as an INDEPENDENT constraint.

WHY
---
The three identity fields (passport / birth / expiry) have their check
digits READ CORRECTLY in every failing sample, and the composite digit
at col 43 is derived correctly in 10 of the 11 failing samples.  The
composite covers 39 characters spread over four fields, so it is a
constraint that spans the whole line -- exactly the thing a
per-field decoder does not have.

This script simulates a decoder that:
  1. keeps the identity fields as read (their check digits are trusted),
  2. searches the PERSONAL NUMBER field (cols 28..41) for the body
     whose ICAO hash equals the digit READ at col 42,
  3. additionally requires the composite over cols 0..9 / 13..19 /
     21..42 to equal the digit READ at col 43.

It is an exact DP over (own residue, composite residue) -- 100 states.

Usage:
    python3 tools/composite_constraint_sim.py [TOPK]
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


def read_digit(cands):
    for ch, _lp in cands:
        if ch.isdigit():
            return int(ch)
    return None


def solve_pn(cands_body, start, n, read42, read43, fixed_prefix):
    """Exact DP over (own residue, composite residue) for the pn body.

    cands_body: per column [(char, logprob), ...] for cols start..start+n-1
    fixed_prefix: the composite-residue contribution of the columns that
                  are NOT part of this field (cols 0..9, 13..19, 21..27
                  and the field's own check-digit column).
    """
    cur = {(0, 0): (0.0, "")}
    for j in range(n):
        c = start + j
        w = W[j % 3]
        cw = comp_weight(c)
        nxt = {}
        for (own, comp), (sc, body) in cur.items():
            for ch, lp in cands_body[j]:
                no = (own + val(ch) * w) % 10
                nc = (comp + val(ch) * cw) % 10
                ni = (no, nc)
                s = sc + lp
                if ni not in nxt or s > nxt[ni][0]:
                    nxt[ni] = (s, body + ch)
        cur = nxt

    best = None
    for (own, comp), (sc, body) in cur.items():
        if read42 is not None and own != read42:
            continue
        if read43 is not None:
            # the composite includes the field's own check digit, which
            # is the derived one, plus the fixed prefix contribution
            total = (comp + own * comp_weight(42) + fixed_prefix) % 10
            if total != read43:
                continue
        if best is None or sc > best[0]:
            best = (sc, body)
    return best


def main():
    topk = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    gt = {}
    for k, v in json.loads(GT_JSON.read_text()).items():
        if isinstance(v, (list, tuple)) and len(v) == 2:
            gt[Path(k).stem] = v

    nfixed = ntry = 0
    for p in sorted(DIAG_DIR.glob("*.json")):
        stem = p.stem
        if stem not in gt:
            continue
        d = json.loads(p.read_text())
        g1, g2 = gt[stem]
        base = d.get("line2", "")
        if base == g2:
            continue
        line = d["lines"][1]

        def cands(c):
            return [(ch, math.log(max(pp, 1e-9)))
                    for ch, pp in line["chars"][c]["top"][:topk] if ch]

        read42 = read_digit(cands(42))
        read43 = read_digit(cands(43))
        # composite contribution of everything except the pn body
        fixed = 0
        for c in range(44):
            if 28 <= c <= 41:
                continue
            if c <= 9 or (13 <= c <= 19) or (21 <= c <= 42):
                fixed += val(base[c]) * comp_weight(c)
        fixed %= 10

        ntry += 1
        best = solve_pn([cands(28 + k) for k in range(14)], 28, 14,
                        read42, read43, fixed)
        if best is None:
            print(f"{stem:<24} no solution under the composite constraint")
            continue
        sc, body = best
        new = list(base)
        new[28:42] = body
        new[42] = str(cd_of("".join(new[28:42])))
        comp_seq = "".join(new[0:10]) + "".join(new[13:20]) + \
                   "".join(new[21:43])
        new[43] = str(cd_of(comp_seq))
        out = "".join(new)
        match = out == g2
        nfixed += 1
        print(f"{stem:<24} pn {base[28:42]} -> {body}  read42={read42} "
              f"read43={read43}  {'MATCH' if match else 'differs'}")
        if not match:
            print(f"{'':<24} got {out}")
            print(f"{'':<24} gt  {g2}")
    print(f"\nconstraint applied to {ntry} samples, produced a reading on "
          f"{nfixed}")


if __name__ == "__main__":
    main()
