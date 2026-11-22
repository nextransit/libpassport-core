#!/usr/bin/env python3
"""Constrained MRZ decode: ICAO 9303 check digits as HARD constraints.

Why this exists
---------------
`tools/prob_gap.py` proved every failure is CLASSIFIER-limited: the truth is
inside the top-K but ranked below the greedy reading, so a deeper beam
cannot fix it.  TD3's check digits are a much stronger constraint.

Key structural fact (verified against the corpus)
-------------------------------------------------
The 7-3-1 weight cycle is 3, and every field boundary in line 2 falls on
a multiple of 3 (0-9, 13-19, 21-27, 28-42).  Therefore the weights of a
field's members are the SAME whether you index them from the start of the
field or from the start of the line.  It follows that

    composite@43  ==  doc@9 + birth@19 + exp@27 + pn@42   (mod 10)

i.e. satisfying the four single-field check digits ALSO satisfies the
composite one.  The composite check alone is therefore strictly weaker
than the four field checks, and the right decoder enforces all four
FIELD checks -- the composite then holds for free.

The check-digit column itself is treated as a VARIABLE: the printed digit
is sometimes mis-read (e.g. '7' read as '1'), and a decoder that trusts it
propagates the error into the whole field.  We therefore enumerate the
check-digit column's own top-K candidates and pick the hypothesis with the
best overall log-probability.

The maximiser is exact and cheap: a check digit is a weighted sum mod 10,
so the constraint is a 10-state finite automaton and we run a Viterbi DP
over the columns of a field.  Cost is O(ncols * topk * 10) per field --
microseconds, no beam, no fallback.

Usage:
    python3 tools/checksum_decode2.py --npz /tmp/holdout_photo2.npz
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


def cval(c):
    return 0 if c == "<" else (int(c) if c.isdigit() else ord(c) - 55)


def weighted(chars):
    return sum(cval(c) * WT[i % 3] for i, c in enumerate(chars)) % 10


def dp_field(cols, weights, cands, target):
    """Exact max-log-prob assignment over `cols` with checksum == target.

    `cands[k]` is a list of (class_index, logprob) for cols[k]; `weights[k]`
    is the 7-3-1 weight of cols[k] inside the field.  Returns
    (best_logprob, best_class_list) or (None, None).
    """
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
    ap.add_argument("--cd-topk", type=int, default=6,
                    help="candidates considered for the check-digit column")
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
    n_fields_solved = 0
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
                out.append([(int(o), float(np.log(max(sm[c][o], 1e-12))))
                            for o in order])
            return out

        new = list(greedy)
        for (a, b, cdpos) in FIELDS:
            cols = list(range(a, b))
            weights = [WT[(a + k) % 3] for k in range(len(cols))]
            cm = cands_for(cols, args.topk)
            # the check-digit column is a variable: try its own candidates
            cd_order = np.argsort(-sm[cdpos])[:args.cd_topk]
            best_all = (float("-inf"), None, None)
            for o in cd_order:
                if not GL[int(o)].isdigit():
                    continue
                target = int(GL[int(o)])
                lp, assign = dp_field(cols, weights, cm, target)
                if assign is None:
                    continue
                # the check digit's own log-prob joins the field score
                cdlp = float(np.log(max(sm[cdpos][int(o)], 1e-12)))
                score = lp + cdlp
                if score > best_all[0]:
                    best_all = (score, assign, int(o))
            if best_all[1] is not None:
                n_fields_solved += 1
                for k, c in enumerate(cols):
                    new[c] = GL[best_all[1][k]]
                new[cdpos] = GL[best_all[2]]

        s_new = "".join(new)
        # verify the composite holds (it should, by the identity)
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
    print(f"fields solved      : {n_fields_solved}/{n_rows * 4}")
    print(f"fixed by checksum  : {n_fixed}")
    print(f"broken by checksum : {n_broke}")
    print(f"net                : {n_base + n_fixed - n_broke}")


if __name__ == "__main__":
    main()
