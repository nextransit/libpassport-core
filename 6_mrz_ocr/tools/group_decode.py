#!/usr/bin/env python3
"""Can the ICAO check digits RESCUE the failing lines, group by group?

The C beam searches the top-14 least-confident columns with a depth
budget of 4, so a reading that needs five or six simultaneous
corrections is mathematically out of reach.  But the five line-2 check
digits partition the line into FOUR independent groups:

    seg0  cols  0..8   -> check digit at col  9
    seg1  cols 13..18  -> check digit at col 19
    seg2  cols 21..26  -> check digit at col 27
    seg3  cols 28..41  -> check digit at col 42

plus the composite digit at col 43, which covers
cols 0..9 + 13..19 + 21..42.

Each group can be solved SEPARATELY: enumerate the readings of that
group's columns (the check digit column is forced by the group), keep
the ones whose weighted sum matches, and rank them by log-probability.
Because the groups are independent, the required corrections per group
are usually 1-3 -- well inside the depth the beam already supports.

This tool measures, offline, what such a group-wise decode would yield:
the exactness of the reconstructed line and the number of groups that
the greedy reading already satisfies.

Usage:
    python3 tools/group_decode.py --npz /tmp/holdout_photo2.npz
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


def v(c):
    if "0" <= c <= "9":
        return ord(c) - 48
    if "A" <= c <= "Z":
        return ord(c) - 55
    return 0


def cd(s):
    return sum(v(c) * WT[i % 3] for i, c in enumerate(s)) % 10


GROUPS = [
    (list(range(0, 9)), 9),
    (list(range(13, 19)), 19),
    (list(range(21, 27)), 27),
    (list(range(28, 42)), 42),
]


def solve_group(cands, cols, cd_col, max_change, top_cols=None):
    """Best readings of `cols` whose weighted sum matches the check digit.

    `top_cols` limits the search to that many columns of the group --
    the least confident ones -- so the enumeration stays bounded.  The
    rest are held at their greedy value.

    Returns a list of (log_prob, tuple_of_class_indices), best first.
    """
    cd_vals = {v(off.GLYPHS[c[0]]) for c in cands[cd_col]}
    base = [cands[c][0][0] for c in cols]
    lps = []
    vals = []
    for c in cols:
        col = []
        for k, p in cands[c]:
            col.append((k, p))
        lps.append(col)
        vals.append([v(off.GLYPHS[k]) for k, _ in col])

    # rank the group's columns by the top-2 margin: the least confident
    # ones are the only realistic candidates for a correction.
    if top_cols is not None and top_cols < len(cols):
        margin = [(cands[c][0][1] - cands[c][1][1], i)
                  for i, c in enumerate(cols) if len(cands[c]) > 1]
        margin.sort()
        keep = {i for _, i in margin[:top_cols]}
        search_cols = [i for i in range(len(cols)) if i in keep]
    else:
        search_cols = list(range(len(cols)))

    base_sum = sum(vals[i][0] * WT[i % 3] for i in range(len(cols)))
    out = []
    for k in range(0, max_change + 1):
        for combo in itertools.combinations(search_cols, k):
            for choices in itertools.product(*[range(1, len(vals[i]))
                                              for i in combo]):
                s = base_sum
                for i, j in zip(combo, choices):
                    s += (vals[i][j] - vals[i][0]) * WT[i % 3]
                if s % 10 not in cd_vals:
                    continue
                work = list(base)
                lp = 0.0
                for i in range(len(cols)):
                    j = choices[combo.index(i)] if i in combo else 0
                    work[i] = lps[i][j][0]
                    lp += np.log(max(lps[i][j][1], 1e-9))
                out.append((lp, tuple(work)))
    out.sort(key=lambda r: -r[0])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="/tmp/holdout_photo2.npz")
    ap.add_argument("--w1", default="src/cnn_weights_line1.h")
    ap.add_argument("--w2", default="src/cnn_weights_line2.h")
    ap.add_argument("--max-change", type=int, default=4)
    ap.add_argument("--top-cols", type=int, default=6,
                    help="per group, how many of the least confident "
                         "columns may change")
    ap.add_argument("--only", default="")
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

    n_gt_won = n_total = 0
    for (fname, line), idxs in sorted(groups.items()):
        if line != 1:
            continue
        if args.only and args.only not in fname:
            continue
        idxs = sorted(idxs, key=lambda i: meta[i]["col"])
        sm = off.softmax(off.forward(X[idxs], p2))
        cands = []
        for j in range(len(idxs)):
            order = np.argsort(-sm[j])
            cands.append([(int(order[k]), float(sm[j][order[k]]))
                          for k in range(5)])
        gt = "".join(off.GLYPHS[Y[i]] for i in idxs)
        greedy = "".join(off.GLYPHS[c[0][0]] for c in cands)

        # solve each group
        best = list(greedy)
        detail = []
        for cols, cd_col in GROUPS:
            sols = solve_group(cands, cols, cd_col, args.max_change,
                               args.top_cols)
            if not sols:
                detail.append(f"g{cols[0]}:NOSOL")
                continue
            lp, cls = sols[0]
            for i, c in enumerate(cols):
                best[c] = off.GLYPHS[cls[i]]
            best[cd_col] = off.GLYPHS[cands[cd_col][0][0]]
            detail.append(f"g{cols[0]}:{len(sols)}")
        # the composite digit is the 5th constraint: force it
        comp = best[0:10] + best[13:20] + best[21:43]
        best[43] = off.GLYPHS[cands[43][0][0]]
        comp_ok = cd(comp) == v(best[43])
        # if the composite does not match, try the check-digit candidates
        if not comp_ok:
            for k, _ in cands[43]:
                cand = off.GLYPHS[k]
                if cd(best[0:10] + best[13:20] + best[21:43]) == v(cand):
                    best[43] = cand
                    break
        got = "".join(best)
        ok = got == gt
        n_total += 1
        n_gt_won += ok
        g_ok = greedy == gt
        print(f"{fname:32s} greedy={'OK ' if g_ok else 'BAD'} "
              f"group={'OK ' if ok else 'BAD'} comp_ok={comp_ok} "
              f"ndiff={sum(1 for a,b in zip(got,gt) if a!=b)} [{' '.join(detail)}]")
        if not ok and g_ok:
            print(f"    GT    : {gt}")
            print(f"    group : {got}")
    print(f"\ngroup decode wins {n_gt_won}/{n_total} (greedy baseline differs)")


if __name__ == "__main__":
    main()
