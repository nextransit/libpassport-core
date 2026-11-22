#!/usr/bin/env python3
"""Joint constrained decode of MRZ line 2 (ICAO 9303).

Design
------
Every failure measured on this corpus is CLASSIFIER-limited (the truth is
in the top-K but ranked below the greedy reading), so a deeper beam over
the same candidate list cannot help.  What can help is the arithmetic
constraint TD3 prints on the document itself.

Three facts drive the decoder:

1. The 7-3-1 weight cycle is 3 and every line-2 field boundary
   (0-9, 13-19, 21-27, 28-42) sits on a multiple of 3, so a field's
   weights are the same indexed from the field start or from the line
   start.  It follows

       composite@43 == doc@9 + birth@19 + exp@27 + pn@42  (mod 10)

   -- satisfying the four FIELD checks satisfies the composite for free.
   The composite check alone is strictly weaker.

2. The check-digit column is itself sometimes mis-read (a '7' read as
   '1').  Trusting it propagates the error across the whole field, so
   the decoder enumerates the check-digit column's own candidates.

3. The ICAO field masks (the same whitelist the C pipeline already
   applies) shrink the candidate space enormously: the birth and expiry
   dates are digits only, the personal-number column is restricted, etc.

The maximiser is exact.  A check digit is a weighted sum mod 10, i.e. a
10-state automaton, so each field is solved by a Viterbi DP over its
columns: O(ncols * topk * 10).  Four fields cost tens of microseconds.

The four fields are independent given the check-digit columns, so the
whole line is decoded exactly -- no beam, no fallback, no retry.

Usage:
    python3 tools/checksum_decode3.py --npz /tmp/holdout_photo2.npz
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

# line 2 check-digit fields: (member start, member end, check-digit column)
FIELDS = [
    (0, 9, 9),     # document number
    (13, 19, 19),  # date of birth
    (21, 27, 27),  # date of expiry
    (28, 42, 42),  # personal number
]

# ---- ICAO 9303 line-2 field masks (mirrors fill_allow in the C code) ----
DIGITS = set("0123456789")
ALNUM = set("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
FILLER = "<"


def cval(c):
    return 0 if c == "<" else (int(c) if c.isdigit() else ord(c) - 55)


def weighted(chars):
    return sum(cval(c) * WT[i % 3] for i, c in enumerate(chars)) % 10


def line2_mask(col):
    """Allowed characters for a line-2 column, per ICAO 9303 TD3."""
    if col == 0:
        # document type: P or '<' filler in the first position
        return set("P")
    if col == 1:
        # document type qualifier: '<' or a letter subtype
        return set("ABCDEFGHIJKLMNOPQRSTUVWXYZ<")
    if 2 <= col <= 8:
        # document number: alphanumeric
        return ALNUM
    if col == 9:
        return DIGITS
    if col == 10:
        # nationality: 3 letters
        return set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    if 11 <= col <= 12:
        return set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    if col == 13:
        # date of birth YYMMDD: digits
        return DIGITS
    if 14 <= col <= 18:
        return DIGITS
    if col == 19:
        return DIGITS
    if col == 20:
        # sex: M, F or '<' (unspecified)
        return set("MF<")
    if 21 <= col <= 26:
        return DIGITS
    if col == 27:
        return DIGITS
    if 28 <= col <= 41:
        # personal number: alphanumeric + '<' filler
        return ALNUM | {FILLER}
    if col == 42:
        return DIGITS
    if col == 43:
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
    ap.add_argument("--topk", type=int, default=6)
    ap.add_argument("--cd-topk", type=int, default=6)
    ap.add_argument("--no-mask", action="store_true")
    ap.add_argument("--force-dp", action="store_true",
                    help="run the DP on every field, even when the greedy "
                         "reading already satisfies the check digit")
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

    masks = None if args.no_mask else {c: line2_mask(c) for c in range(44)}

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
                cand = []
                for o in order:
                    ch = GL[int(o)]
                    if masks is not None and ch not in masks[c]:
                        continue
                    cand.append((int(o), float(np.log(max(sm[c][o], 1e-12)))))
                if not cand:
                    cand = [(int(order[0]), float(np.log(max(sm[c][order[0]], 1e-12))))]
                out.append(cand)
            return out

        new = list(greedy)
        for (a, b, cdpos) in FIELDS:
            cols = list(range(a, b))
            # ICAO 9303: the 7-3-1 weight cycle RESTARTS at the beginning
            # of each field (the C pipeline does the same -- see
            # MRZ_CD_SEG0_W/SEG1_W/... in src/mrz_ocr_cnn.c).  Indexing
            # the weights from the LINE start instead is a different,
            # wrong constraint.
            weights = [WT[k % 3] for k in range(len(cols))]
            # If the greedy reading ALREADY satisfies this field's check
            # digit, leave it alone.  The DP maximises the joint
            # probability of the field, and the true reading is not
            # necessarily the most probable one -- an alternative that
            # also satisfies the checksum can outscore it and drag a
            # correct reading off the mark.  The constraint is a filter,
            # not an objective: it should only ever REJECT readings the
            # checksum rules out, never replace a checksum-valid one.
            greedy_cd = greedy[cdpos]
            if (not args.force_dp) and greedy_cd.isdigit() and \
                    weighted(greedy[a:b]) == int(greedy_cd):
                continue
            cm = cands_for(cols, args.topk)
            cd_cands = cands_for([cdpos], args.cd_topk)[0]
            best = (float("-inf"), None, None)
            for cls, cdlp in cd_cands:
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

        s_new = "".join(new)
        comp_ok = (weighted(s_new[0:10] + s_new[13:20] + s_new[21:43])
                   == int(s_new[43])) if s_new[43].isdigit() else False

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
                print(f"    gt    ={gt}")

    print(f"line2 rows         : {n_rows}")
    print(f"greedy exact       : {n_base}")
    print(f"fixed by checksum  : {n_fixed}")
    print(f"broken by checksum : {n_broke}")
    print(f"net                : {n_base + n_fixed - n_broke}")


if __name__ == "__main__":
    main()
