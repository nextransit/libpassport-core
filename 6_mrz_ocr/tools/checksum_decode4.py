#!/usr/bin/env python3
"""Joint constrained decode of MRZ line 2: five check digits at once.

Why this exists
---------------
`tools/prob_gap.py` proved every failure is CLASSIFIER-limited, so a
deeper beam over the same candidate list cannot help.  TD3's printed
check digits can.

Measured on this corpus, every failing row's ground truth satisfies ALL
five check digits, while the greedy reading violates at least one.
So the arithmetic is a precise pointer to the truth.

The five constraints
--------------------
    doc@9       = cd(l2[0:9])
    birth@19    = cd(l2[13:19])
    exp@27      = cd(l2[21:27])
    pn@42       = cd(l2[28:42])
    composite@43 = cd(l2[0:10] + l2[13:20] + l2[21:43])

Each field's 7-3-1 weight cycle RESTARTS at the field start (as in the
C pipeline, MRZ_CD_SEG*_W).  The composite concatenates three ranges
into one 39-character string and restarts the cycle once, at position 0.

Because the composite's weight sequence over l2[21:43] continues the
cycle of the concatenated string (10 + 7 = 17 chars before it, and
17 mod 3 = 2), the composite is NOT the sum of the four field
checksums -- but it is still an independent constraint, and using all
five together over-determines the solution far more than any single
one.

The solver
----------
A field's checksum is a weighted sum mod 10 -- a 10-state automaton --
so each field is solvable exactly by a Viterbi DP over its columns.
The composite couples three of the fields, so it is solved by a DP over
the SAME columns but with the composite's weight vector, run only on
the columns the composite covers.

Strategy: keep the greedy reading whenever it satisfies the check
digit of that field (the constraint is a FILTER, not an objective --
replacing a checksum-valid reading with a merely more probable one
loses information).  Only the fields the greedy reading violates are
re-solved by DP.

Usage:
    python3 tools/checksum_decode4.py --npz /tmp/holdout_photo2.npz
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cnn_offline_eval as off  # noqa: E402

WT = (7, 3, 1)
GL = off.GLYPHS

FIELDS = [
    (0, 9, 9),     # document number
    (13, 19, 19),  # date of birth
    (21, 27, 27),  # date of expiry
    (28, 42, 42),  # personal number
]
COMPOSITE_RANGES = ((0, 10), (13, 20), (21, 43))


def cval(c):
    return 0 if c == "<" else (int(c) if c.isdigit() else ord(c) - 55)


def weighted(chars):
    """Checksum of a string; the 7-3-1 cycle restarts at position 0."""
    return sum(cval(c) * WT[i % 3] for i, c in enumerate(chars)) % 10


def composite_of(l2):
    s = list(l2[0:10]) + list(l2[13:20]) + list(l2[21:43])
    return weighted(s)


# ---- ICAO 9303 line-2 field masks (mirrors fill_allow in the C code) ----
DIGITS = set("0123456789")
ALNUM = set("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
ALPHA = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def line2_mask(col):
    if col == 0:
        return set("P")
    if col == 1:
        return ALPHA | {"<"}
    if 2 <= col <= 8:
        return ALNUM
    if col == 9:
        return DIGITS
    if 10 <= col <= 12:
        return ALPHA
    if 13 <= col <= 19:
        return DIGITS
    if col == 20:
        return set("MF<")
    if 21 <= col <= 27:
        return DIGITS
    if 28 <= col <= 41:
        return ALNUM | {"<"}
    if col in (42, 43):
        return DIGITS
    return ALNUM


def dp_field(cols, weights, cands, target):
    """Exact max-log-prob assignment over `cols` with checksum == target."""
    NEG = float("-inf")
    dp = [(NEG, None)] * 10
    dp[0] = (0.0, ())
    for k in range(len(cols)):
        nxt = [(NEG, None)] * 10
        for r in range(10):
            lp, back = dp[r]
            if lp == NEG:
                continue
            for cls, lpc in cands[k]:
                nr = (r + cval(GL[cls]) * weights[k]) % 10
                if lp + lpc > nxt[nr][0]:
                    nxt[nr] = (lp + lpc, back + (cls,))
        dp = nxt
    lp, back = dp[target % 10]
    if lp == NEG:
        return None, None
    return lp, back


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="/tmp/holdout_photo2.npz")
    ap.add_argument("--w2", default="src/cnn_weights_line2.h")
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--cd-topk", type=int, default=8)
    ap.add_argument("--only", default="")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    d = np.load(args.npz, allow_pickle=True)
    X = d["X"].astype(np.float32)
    Y = d["Y"]
    meta = [json.loads(s) for s in d["meta"]]

    w2 = off.parse_header(args.w2)
    p2 = off.pick(w2, off.infer_prefix(w2))

    groups = defaultdict(list)
    for i, m in enumerate(meta):
        groups[(m["file"], int(m["line"]))].append(i)

    masks = {c: line2_mask(c) for c in range(44)}

    n_rows = n_base = n_fixed = n_broke = 0
    for (fname, line), idxs in sorted(groups.items()):
        if args.only and args.only not in fname:
            continue
        if line != 1:
            continue
        idxs = sorted(idxs, key=lambda i: meta[i]["col"])
        if len(idxs) != 44:
            continue
        sm = off.softmax(off.forward(X[idxs], p2))
        gt = "".join(GL[int(Y[i])] for i in idxs)
        greedy = "".join(GL[int(np.argmax(sm[j]))] for j in range(44))
        n_rows += 1
        ok_base = greedy == gt
        n_base += ok_base

        def cands_for(cols, k):
            out = []
            for c in cols:
                order = np.argsort(-sm[c])[:k]
                cand = [(int(o), float(np.log(max(sm[c][o], 1e-12))))
                        for o in order if GL[int(o)] in masks[c]]
                if not cand:
                    cand = [(int(order[0]),
                             float(np.log(max(sm[c][order[0]], 1e-12))))]
                out.append(cand)
            return out

        new = list(greedy)

        # ---- step 1: repair the fields the greedy reading violates ----
        for (a, b, cdpos) in FIELDS:
            gv = greedy[cdpos]
            if gv.isdigit() and weighted(greedy[a:b]) == int(gv):
                continue          # greedy already valid -> keep it
            cols = list(range(a, b))
            weights = [WT[k % 3] for k in range(len(cols))]
            cm = cands_for(cols, args.topk)
            best = (float("-inf"), None, None)
            for cls, cdlp in cands_for([cdpos], args.cd_topk)[0]:
                if not GL[cls].isdigit():
                    continue
                lp, assign = dp_field(cols, weights, cm, int(GL[cls]))
                if assign is None:
                    continue
                score = lp + cdlp
                if score > best[0]:
                    best = (score, assign, cls)
            if best[1] is not None:
                for k, c in enumerate(cols):
                    new[c] = GL[best[1][k]]
                new[cdpos] = GL[best[2]]

        # ---- step 2: the composite check digit ----
        gv = new[43]
        if not (gv.isdigit() and composite_of(new) == int(gv)):
            cols, weights = [], []
            for a, b in COMPOSITE_RANGES:
                for k in range(a, b):
                    cols.append(k)
                    weights.append(WT[len(weights) % 3])
            cm = cands_for(cols, args.topk)
            best = (float("-inf"), None, None)
            for cls, cdlp in cands_for([43], args.cd_topk)[0]:
                if not GL[cls].isdigit():
                    continue
                lp, assign = dp_field(cols, weights, cm, int(GL[cls]))
                if assign is None:
                    continue
                score = lp + cdlp
                if score > best[0]:
                    best = (score, assign, cls)
            if best[1] is not None:
                for k, c in enumerate(cols):
                    new[c] = GL[best[1][k]]
                new[43] = GL[best[2]]

        s_new = "".join(new)
        comp_ok = (s_new[43].isdigit() and composite_of(s_new) == int(s_new[43]))

        if s_new == gt and not ok_base:
            n_fixed += 1
            if args.verbose:
                diff = [i for i in range(44) if greedy[i] != gt[i]]
                print(f"  FIXED {fname} cols={diff} comp_ok={comp_ok}")
                print(f"    greedy={greedy}")
                print(f"    cdec  ={s_new}")
                print(f"    gt    ={gt}")
        elif ok_base and s_new != gt:
            n_broke += 1
            if args.verbose:
                print(f"  BROKE {fname} comp_ok={comp_ok}")
                print(f"    greedy={greedy}")
                print(f"    cdec  ={s_new}")
                print(f"    gt    ={gt}")

    print(f"line2 rows         : {n_rows}")
    print(f"greedy exact       : {n_base}")
    print(f"fixed by checksum  : {n_fixed}")
    print(f"broken by checksum : {n_broke}")
    print(f"net                : {n_base + n_fixed - n_broke}")


if __name__ == "__main__":
    main()
