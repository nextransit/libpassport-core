#!/usr/bin/env python3
"""Decode MRZ line 2 with the ICAO 9303 check digits as HARD constraints.

Why this exists
---------------
`tools/prob_gap.py` proved every failure is CLASSIFIER-limited: the ground
truth is inside the top-K but ranked below the greedy reading, so a deeper
beam cannot fix it.

TD3 hands us something stronger than a deeper beam: FIVE check digits on
line 2.  Verified against the corpus:

    doc@9       cols[0:9]    104/107 valid
    birth@19    cols[13:19]  105/107 valid
    exp@27      cols[21:27]  105/107 valid
    pn@42       cols[28:42]  104/107 valid
    composite@43 (l2 only)  104/107 valid

Line 1 has NO check digit in this corpus (position 44 is '<' in 107/107
rows), so only line 2 can be repaired this way.

The maximiser is exact and cheap.  The check digit is a weighted sum mod
10, so the constraint is a finite-state automaton with 10 states: we run
a Viterbi-style DP over the columns of a field, keeping the best
assignment per (column, residue).  Cost is O(ncols * topk * 10), so the
whole line 2 is decoded in microseconds -- no beam, no fallback.

Usage:
    python3 tools/checksum_decode.py --npz /tmp/holdout_photo2.npz
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

# line 2 check-digit fields: (start, end, check-digit column)
FIELDS = [
    (0, 9, 9),     # document number
    (13, 19, 19),  # date of birth
    (21, 27, 27),  # date of expiry
    (28, 42, 42),  # personal number
]
# the composite field concatenates three ranges of line 2
COMPOSITE_RANGES = ((0, 10), (13, 20), (21, 43))


def cval(c):
    return 0 if c == "<" else (int(c) if c.isdigit() else ord(c) - 55)


def weighted(chars):
    """Weighted 7-3-1 checksum over a sequence of characters."""
    t = 0
    for i, c in enumerate(chars):
        t += cval(c) * WT[i % 3]
    return t % 10


def composite_chars(s):
    out = []
    for a, b in COMPOSITE_RANGES:
        out.extend(s[a:b])
    return out


def dp_field(cols, weights, cands, target):
    """Exact max-log-prob assignment over `cols` with checksum == target.

    `cands[k]` is a list of (class_index, logprob) for cols[k].
    `weights[k]` is the 7-3-1 weight of cols[k] inside the field.
    Returns (best_logprob, best_class_list) or (None, None).
    """
    # dp[r] = (best_lp, backpointer) for residue r
    dp = [(float("-inf"), None)] * 10
    dp[0] = (0.0, [])
    for k in range(len(cols)):
        nxt = [(float("-inf"), None)] * 10
        for r in range(10):
            lp, back = dp[r]
            if lp == float("-inf"):
                continue
            for cls, lpc in cands[k]:
                nr = (r + cval(GL[cls]) * weights[k]) % 10
                cand_lp = lp + lpc
                if cand_lp > nxt[nr][0]:
                    nxt[nr] = (cand_lp, back + [cls])
        dp = nxt
    lp, back = dp[target % 10]
    if lp == float("-inf"):
        return None, None
    return lp, back


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="/tmp/holdout_photo2.npz")
    ap.add_argument("--w1", default="src/cnn_weights_line1.h")
    ap.add_argument("--w2", default="src/cnn_weights_line2.h")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--only", default="")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    d = np.load(args.npz, allow_pickle=True)
    X = d["X"].astype(np.float32)
    Y = d["Y"]
    meta = [json.loads(s) for s in d["meta"]]

    w1 = off.parse_header(args.w1)
    w2 = off.parse_header(args.w2)
    p1 = off.pick(w1, off.infer_prefix(w1))
    p2 = off.pick(w2, off.infer_prefix(w2))

    groups = defaultdict(list)
    for i, m in enumerate(meta):
        groups[(m["file"], int(m["line"]))].append(i)

    n_rows = n_base = n_fixed = n_broke = 0
    for (fname, line), idxs in sorted(groups.items()):
        if args.only and args.only not in fname:
            continue
        if line != 1:
            continue
        idxs = sorted(idxs, key=lambda i: meta[i]["col"])
        if len(idxs) != 44:
            continue
        P = p2
        sm = off.softmax(off.forward(X[idxs], P))
        gt = "".join(GL[int(Y[i])] for i in idxs)
        greedy = "".join(GL[int(np.argmax(sm[j]))] for j in range(44))
        n_rows += 1
        ok_base = greedy == gt
        n_base += ok_base
        print(f"  row {fname} greedy_ok={ok_base}", flush=True)

        def topk_cands(cols):
            out = []
            for c in cols:
                order = np.argsort(-sm[c])[:args.topk]
                out.append([(int(o), float(np.log(max(sm[c][o], 1e-12))))
                            for o in order])
            return out

        new = list(greedy)

        # --- the four single-field check digits ---
        for (a, b, cdpos) in FIELDS:
            if not greedy[cdpos].isdigit():
                continue
            cols = list(range(a, b))
            weights = [WT[(a + k) % 3] for k in range(len(cols))]
            lp, assign = dp_field(cols, weights, topk_cands(cols),
                                  int(greedy[cdpos]))
            if assign:
                for k, c in enumerate(cols):
                    new[c] = GL[assign[k]]

        # --- the composite check digit ---
        if greedy[43].isdigit():
            cols, weights = [], []
            for a, b in COMPOSITE_RANGES:
                for k in range(a, b):
                    cols.append(k)
                    weights.append(WT[len(weights) % 3])
            lp, assign = dp_field(cols, weights, topk_cands(cols),
                                  int(greedy[43]))
            if assign:
                for k, c in enumerate(cols):
                    new[c] = GL[assign[k]]

        s_new = "".join(new)
        if s_new == gt and not ok_base:
            n_fixed += 1
            if args.verbose:
                diff = [i for i in range(44) if greedy[i] != gt[i]]
                print(f"  FIXED {fname} cols={diff}")
                print(f"    greedy={greedy}")
                print(f"    cdec  ={s_new}")
        elif ok_base and s_new != gt:
            n_broke += 1
            if args.verbose:
                print(f"  BROKE {fname}")
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
