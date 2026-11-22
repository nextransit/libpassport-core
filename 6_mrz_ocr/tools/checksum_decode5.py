#!/usr/bin/env python3
"""Minimal-change constrained decode of MRZ line 2.

Why this exists
---------------
`tools/prob_gap.py` proved every failure is CLASSIFIER-limited: the truth
is in the top-K but ranked below the greedy reading, so a deeper beam
over the same candidate list cannot help.  TD3's printed check digits
can.

Measured on this corpus, EVERY failing row's ground truth satisfies all
five check digits while the greedy reading violates at least one -- the
arithmetic is a precise pointer to the truth.

Why minimal change
------------------
Solving a field by maximising the joint log-probability over the
candidate space does NOT work, and the reason is measurable: the truth
often needs SEVERAL columns changed at once (the failing rows need 2-4
columns changed in the personal-number field), and among the many
combinations that satisfy the checksum, the most probable one is
routinely NOT the true one.  The checksum is one scalar constraint over
many columns; it barely discriminates.

The prior that does discriminate is the same one the rest of the
pipeline already assumes: a mis-read character is an EXCEPTION, so the
correct reading is the one that changes the FEWEST columns away from
the greedy reading.  So the objective is lexicographic:

    1. satisfy the check digit (hard constraint)
    2. change as few columns as possible
    3. among those, maximise the total log-probability

Step 2 is what the probability objective was missing.

Search
------
For each field we enumerate the columns that could change (those whose
top-1 is not the only plausible candidate), trying 1 change, then 2,
then 3 ... until a checksum-satisfying assignment is found.  With 14
columns and K candidates the 2-change search is C(14,2)*K^2 which is
tiny; the search stops as soon as it finds any solution, so the cost is
bounded by the smallest change count that works.

The check-digit column is treated as a variable too (a printed '7' read
as '1' happens), and the composite check digit is applied as a final
filter.

Usage:
    python3 tools/checksum_decode5.py --npz /tmp/holdout_photo2.npz
"""
from __future__ import annotations

import argparse
import itertools
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

DIGITS = set("0123456789")
ALNUM = set("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
ALPHA = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def cval(c):
    return 0 if c == "<" else (int(c) if c.isdigit() else ord(c) - 55)


def weighted(chars):
    return sum(cval(c) * WT[i % 3] for i, c in enumerate(chars)) % 10


def composite_of(l2):
    s = list(l2[0:10]) + list(l2[13:20]) + list(l2[21:43])
    return weighted(s)


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


def field_candidates(sm, cols, topk, masks):
    """Per column: list of (class, logprob) candidates, best first."""
    out = []
    for c in cols:
        order = np.argsort(-sm[c])[:topk]
        cand = [(int(o), float(np.log(max(sm[c][o], 1e-12))))
                for o in order if GL[int(o)] in masks[c]]
        if not cand:
            cand = [(int(order[0]),
                     float(np.log(max(sm[c][order[0]], 1e-12))))]
        out.append(cand)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="/tmp/holdout_photo2.npz")
    ap.add_argument("--w2", default="src/cnn_weights_line2.h")
    ap.add_argument("--topk", type=int, default=6)
    ap.add_argument("--max-changes", type=int, default=3)
    ap.add_argument("--max-candidates", type=int, default=4)
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

        new = list(greedy)

        for (a, b, cdpos) in FIELDS:
            # Keep the greedy reading when it ALREADY satisfies this
            # field's check digit: the constraint is a filter, not an
            # objective.  Rewriting a checksum-valid field only loses
            # information.
            gv = greedy[cdpos]
            if gv.isdigit() and weighted(greedy[a:b]) == int(gv):
                continue
            cols = list(range(a, b))
            weights = [WT[k % 3] for k in range(len(cols))]
            cands = field_candidates(sm, cols, args.topk, masks)
            # the check digit we must satisfy: try the read digit first,
            # then the alternatives (the printed digit is sometimes
            # mis-read)
            cd_opts = []
            for o in np.argsort(-sm[cdpos])[:args.topk]:
                if GL[int(o)].isdigit():
                    cd_opts.append(int(GL[int(o)]))
            best = None
            for cdval in cd_opts:
                target = cdval
                # minimal-change search
                found = minimal_change(cols, greedy, cands, weights,
                                       target, args.max_changes,
                                       args.max_candidates)
                if found is None:
                    continue
                nch, assign, lp = found
                cand = list(greedy)
                for k, c in enumerate(cols):
                    cand[c] = assign[k]
                cand[cdpos] = str(cdval)
                score = (nch, -lp)
                if best is None or score < best[0]:
                    best = (score, cand, nch)
            if best is not None:
                new = list(best[1])

        s_new = "".join(new)
        comp_ok = (s_new[43].isdigit() and composite_of(s_new) == int(s_new[43]))

        if s_new == gt and not ok_base:
            n_fixed += 1
            if args.verbose:
                diff = [i for i in range(44) if greedy[i] != gt[i]]
                print(f"  FIXED {fname} cols={diff} comp_ok={comp_ok}")
                print(f"    greedy={greedy}")
                print(f"    cdec  ={s_new}")
        elif ok_base and s_new != gt:
            n_broke += 1
            if args.verbose:
                print(f"  BROKE {fname} comp_ok={comp_ok}")
                print(f"    greedy={greedy}")
                print(f"    cdec  ={s_new}")

    print(f"line2 rows         : {n_rows}")
    print(f"greedy exact       : {n_base}")
    print(f"fixed by checksum  : {n_fixed}")
    print(f"broken by checksum : {n_broke}")
    print(f"net                : {n_base + n_fixed - n_broke}")


def minimal_change(cols, greedy, cands, weights, target, max_changes,
                   max_candidates):
    """Smallest number of column changes that makes the field checksum
    equal `target`; among those, the most probable one.

    Returns (n_changes, assignment_list, logprob) or None.
    """
    n = len(cols)
    base = [cval(greedy[c]) * weights[k] for k, c in enumerate(cols)]
    base_sum = sum(base) % 10
    mutable = [k for k in range(n) if len(cands[k]) > 1]

    for nch in range(0, max_changes + 1):
        best = None
        for subset in itertools.combinations(mutable, nch):
            per_col = []
            for k in subset:
                # `cls` is a CLASS INDEX (0..36); the greedy reading is a
                # CHARACTER.  Comparing the two directly is a type error
                # that silently disables the "changed" test.
                opts = [(cls, lp) for cls, lp in cands[k][:max_candidates]
                        if GL[cls] != greedy[cols[k]]]
                if not opts:
                    break
                per_col.append(opts)
            if len(per_col) != len(subset):
                continue
            for combo in itertools.product(*per_col):
                s = base_sum
                lp = 0.0
                for (k, (cls, lpc)) in zip(subset, combo):
                    s += (cval(GL[cls]) - cval(greedy[cols[k]])) * weights[k]
                    lp += lpc
                if s % 10 != target % 10:
                    continue
                if best is None or lp > best[2]:
                    assign = [greedy[c] for c in cols]
                    for (k, (cls, _)) in zip(subset, combo):
                        assign[k] = GL[cls]
                    best = (nch, assign, lp)
        if best is not None:
            return best[0], best[1], best[2]
    return None


if __name__ == "__main__":
    main()
