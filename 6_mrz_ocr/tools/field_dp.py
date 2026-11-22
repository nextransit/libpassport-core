#!/usr/bin/env python3
"""Field-scoped CONSTRAINED decode over line 2, solved EXACTLY by DP.

WHY THIS EXISTS
---------------
The shipped beam changes at most MRZ_BEAM_DEPTH (=4) columns and picks
the columns it may change by ascending top-2 probability margin.  A cell
the network is *confidently* wrong about has a huge margin, so it is
never selected -- and a passport number whose five characters are all
misread needs five simultaneous changes, which exceeds the depth anyway.

The check digits localise the error far better.  Line 2 carries five
independent check digits:

    doc        cols 0..8   check at 9
    birth      cols 13..18 check at 19
    expiry     cols 21..26 check at 27
    personal   cols 28..41 check at 42
    composite  the four fields above, check at 43

Two properties make the search exact instead of heuristic:

  1. THE CHECK DIGIT IS DERIVED, NEVER ENUMERATED.  For every candidate
     field body the implied digit is computed, so the decoder only has
     to win on the FIELD, not on the individual character.  This is what
     lets it repair a cell the classifier is confidently wrong about.

  2. THE FIELD IS A SMALL FINITE-STATE PROBLEM.  A field body
     determines two residues mod 10:
        own  -- the field's own weighted sum, i.e. its check digit
        comp -- the body's contribution to the composite digit
     so a partial body is fully characterised by (own, comp) -- 100
     states.  Two prefixes with the same (own, comp) have IDENTICAL
     continuations, so keeping only the better-scoring one is EXACT.

The four fields are coupled only through the composite digit, so the
joint problem is a second DP whose state is the accumulated composite
residue (10 states).  Total: exact, and cheap.

The composite digit covers columns 0..9, 13..19 and 21..42 -- 39
characters -- INCLUDING each field's own check-digit column.  Its 7/3/1
weights restart over that 39-character sequence, so a column's
composite weight is NOT its field-local weight.

Usage:
    python3 tools/field_dp.py --weights /tmp/rl2_L0.h --prefix LINE2 \
        --topk 5 --verbose
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np

import cnn_offline_eval as ce

WEIGHTS = (7, 3, 1)

# (name, body start, body end, check-digit column)
FIELDS = (
    ("doc", 0, 9, 9),
    ("birth", 13, 19, 19),
    ("expiry", 21, 27, 27),
    ("pn", 28, 42, 42),
)


def char_value(ch: str) -> int:
    if ch == "<":
        return 0
    if ch.isdigit():
        return ord(ch) - ord("0")
    return ord(ch) - ord("A") + 10


def check_digit(value: str) -> int:
    return sum(char_value(c) * WEIGHTS[i % 3]
               for i, c in enumerate(value)) % 10


def composite_seq(line: str) -> str:
    return line[0:10] + line[13:20] + line[21:28] + line[28:43]


def comp_index(col: int) -> int:
    if col <= 9:
        return col
    if col <= 19:
        return col - 3
    return col - 4


def comp_weight(col: int) -> int:
    return WEIGHTS[comp_index(col) % 3]


def icao_allow(line: int, col: int) -> set:
    """Faithful port of `fill_allow` (strict ICAO branch) in
    src/mrz_ocr_cnn.c."""
    A = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    D = set("0123456789")
    F = {"<"}
    if line == 0:
        if col == 0:
            return {"A", "C", "I", "P", "V"}
        if col == 1:
            return A | F
        if 2 <= col <= 4:
            return A
        return A | F
    if col <= 8:
        return A | D | F
    if col in (9, 19, 27):
        return D
    if col <= 12:
        return A | F
    if (13 <= col <= 18) or (21 <= col <= 26):
        return D | F | A
    if col == 20:
        return A | D | F
    if col <= 41:
        return A | D | F
    return D | F


def field_dp(cands, lps, s, e):
    """Exact DP over one field body.

    Returns {(own, comp): (score, body)} where
      own  = the check digit the body implies
      comp = the body's contribution to the composite digit
    (the check-digit column's own composite contribution is added by the
    caller, because the digit is only known at the end).
    """
    states = {(0, 0): (0.0, "")}
    for k, c in enumerate(range(s, e)):
        w = WEIGHTS[k % 3]          # field-local weight
        cw = comp_weight(c)         # composite weight
        nxt = {}
        for (own, comp), (score, body) in states.items():
            for j, ch in enumerate(cands[c]):
                v = char_value(ch)
                key = ((own + v * w) % 10, (comp + v * cw) % 10)
                sc = score + lps[c][j]
                cur = nxt.get(key)
                if cur is None or sc > cur[0]:
                    nxt[key] = (sc, body + ch)
        states = nxt
    return states


def solve_line2(cands, lps, greedy):
    """Return the best checksum-valid line, or None.

    Fields whose own check digit fails are re-solved exactly; the
    others are left alone.  The joint choice is a DP over the
    accumulated composite residue.
    """
    bad = []
    for name, s, e, cdp in FIELDS:
        cur = greedy[cdp]
        if not cur.isdigit() or check_digit(greedy[s:e]) != int(cur):
            bad.append((name, s, e, cdp))

    # The composite digit printed in the line is the constraint.  If it
    # is not a digit there is no constraint, and the derived value is
    # used instead.
    target = int(greedy[43]) if greedy[43].isdigit() else None

    # Per-field option lists: (own, comp_body, score, body).
    opts = []
    for name, s, e, cdp in bad:
        st = field_dp(cands, lps, s, e)
        options = []
        for (own, comp), (score, body) in st.items():
            options.append((own, comp, score, body))
        if not options:
            return None
        opts.append((s, e, cdp, options))

    # Joint DP over the accumulated composite residue.
    # state: (accumulated residue) -> (score, [chosen option index per field])
    cur = {0: (0.0, [])}
    for s, e, cdp, options in opts:
        nxt = {}
        for acc, (score, picks) in cur.items():
            for oi, (own, comp, sc, body) in enumerate(options):
                # the derived check digit also feeds the composite
                new_acc = (acc + comp + own * comp_weight(cdp)) % 10
                ns = score + sc
                key = new_acc
                old = nxt.get(key)
                if old is None or ns > old[0]:
                    nxt[key] = (ns, picks + [oi])
        cur = nxt

    best = None
    best_key = None
    for acc, (score, picks) in cur.items():
        if target is not None and acc != target:
            continue
        if best is None or score > best[0]:
            best = (score, picks)
            best_key = acc
    if best is None:
        return None

    score, picks = best
    cand = list(greedy)
    for (s, e, cdp, options), oi in zip(opts, picks):
        own, comp, sc, body = options[oi]
        for k, ch in enumerate(body):
            cand[s + k] = ch
        cand[cdp] = str(own)
    line = "".join(cand)
    if target is None:
        line = line[:43] + str(check_digit(composite_seq(line)))
    return score, line, [b[0] for b in bad]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--prefix", default="LINE2")
    ap.add_argument("--cache", default="/tmp/glyph_cache.npz")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--only", default="")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    # NB: no allow_pickle -- the sandbox's pickle scanner stalls the
    # load, and the arrays are plain numeric/string arrays anyway.
    d = np.load(args.cache)
    meta = [json.loads(s) for s in d["meta"]]
    X = d["X"].astype(np.float32)
    G = ce.GLYPHS

    P = ce.pick(ce.parse_header(args.weights), args.prefix)
    pr = ce.softmax(ce.forward(X, P))

    groups = defaultdict(list)
    for i, m in enumerate(meta):
        groups[(m["file"], int(m["line"]))].append(i)

    agg = defaultdict(int)
    for (f, ln), idxs in sorted(groups.items()):
        if ln != 1:
            continue
        if args.only and args.only not in f:
            continue
        idxs.sort(key=lambda i: int(meta[i]["col"]))
        truth = "".join(meta[i]["ch"] for i in idxs)
        if len(truth) != 44:
            continue
        cands, lps = [], []
        for i in idxs:
            col = int(meta[i]["col"])
            allow = icao_allow(1, col)
            order = np.argsort(-pr[i])
            cl, ll = [], []
            for o in order:
                ch = G[o]
                if ch not in allow:
                    continue
                cl.append(ch)
                ll.append(float(np.log(max(float(pr[i][o]), 1e-12))))
                if len(cl) >= args.topk:
                    break
            if not cl:
                cl, ll = ["<"], [0.0]
            cands.append(cl)
            lps.append(ll)
        greedy = "".join(c[0] for c in cands)
        if greedy == truth:
            agg["exact"] += 1
            continue

        res = solve_line2(cands, lps, greedy)
        if res is None:
            agg["no_solution"] += 1
            if args.verbose:
                print(f"{f:30s} NO SOLUTION")
            continue
        _, line, badnames = res
        if line == truth:
            agg["fixed"] += 1
            if args.verbose:
                print(f"{f:30s} FIXED  bad={badnames}")
        else:
            agg["wrong"] += 1
            if args.verbose:
                print(f"{f:30s} WRONG  bad={badnames} got={line}")
                print(f"{'':30s}        want={truth}")

    print(dict(agg))


if __name__ == "__main__":
    main()
