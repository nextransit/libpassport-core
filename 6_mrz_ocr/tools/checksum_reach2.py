#!/usr/bin/env python3
"""REACHABILITY of the GT under the ICAO check-digit constraints.

The C beam accepts a reading only when all five line-2 check digits
validate, so the question is: does a checksum-valid reading that spells
the GT exist inside the CNN's top-K candidates per column?

The naive enumeration (all subsets of changed columns) explodes, so this
tool solves it as a CONSTRAINT PROBLEM instead:

  * the five check digits constrain the following column groups:
      seg0  cols 0..8   -> col 9
      seg1  cols 13..18 -> col 19
      seg2  cols 21..26 -> col 27
      seg3  cols 28..41 -> col 42
      seg4  composite   -> col 43
  * Each group is independent of the others except through the
    composite digit, so the groups can be solved separately and the
    check-digit column itself is treated as a free variable whose
    value is FORCED by the group.

For each group we enumerate only the combinations whose weighted sum
matches the check-digit column's candidate set, which cuts the space
by ~10x per group.  The composite digit is then checked against the
union of the group solutions.

Usage:
    python3 tools/checksum_reach2.py --npz /tmp/holdout_photo2.npz
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


def cd_value(ch):
    if "0" <= ch <= "9":
        return ord(ch) - 48
    if "A" <= ch <= "Z":
        return ord(ch) - 55
    return 0


def check_digit(s):
    return sum(cd_value(c) * WT[i % 3] for i, c in enumerate(s)) % 10


GROUPS = [
    # (data cols, check col)
    (list(range(0, 9)), 9),
    (list(range(13, 19)), 19),
    (list(range(21, 27)), 27),
    (list(range(28, 42)), 42),
]
COMP = list(range(0, 10)) + list(range(13, 20)) + list(range(21, 43))


def group_solutions(cands, cols, cd_col, max_change):
    """All readings of `cols` (with the check column forced) whose
    weighted sum matches one of the check column's candidates."""
    # candidate values of the check column
    cd_vals = {cd_value(off.GLYPHS[c[0]]) for c in cands[cd_col]}
    # the free columns of the group, in the order the weights apply
    base = [cands[c][0][0] for c in cols]
    free = list(range(len(cols)))

    out = []
    # We enumerate the SUBSET of group columns that deviate from the
    # greedy reading.  A group whose greedy reading already satisfies
    # its check digit contributes one solution with no change.
    def weight_of(i):
        return WT[i % 3]

    base_sum = sum(cd_value(off.GLYPHS[base[i]]) * weight_of(i)
                   for i in range(len(cols)))

    # candidate value sets per column
    vals = []
    for c in cols:
        vals.append([cd_value(off.GLYPHS[k]) for k, p in cands[c]])

    for k in range(0, max_change + 1):
        for combo in itertools.combinations(free, k):
            for choices in itertools.product(*[range(1, len(vals[i]))
                                              for i in combo]):
                s = base_sum
                for i, j in zip(combo, choices):
                    s += (vals[i][j] - vals[i][0]) * weight_of(i)
                if s % 10 in cd_vals:
                    # rebuild the reading
                    work = list(base)
                    for i, j in zip(combo, choices):
                        work[i] = cands[cols[i]][j][0]
                    out.append((tuple(work), combo, choices))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="/tmp/holdout_photo2.npz")
    ap.add_argument("--w1", default="src/cnn_weights_line1.h")
    ap.add_argument("--w2", default="src/cnn_weights_line2.h")
    ap.add_argument("--max-change", type=int, default=6)
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

    for (fname, line), idxs in sorted(groups.items()):
        if line != 1:
            continue
        if args.only and args.only not in fname:
            continue
        idxs = sorted(idxs, key=lambda i: meta[i]["col"])
        P = p2
        sm = off.softmax(off.forward(X[idxs], P))
        cands = []
        for j in range(len(idxs)):
            order = np.argsort(-sm[j])
            cands.append([(int(order[k]), float(sm[j][order[k]]))
                          for k in range(5)])
        gt = "".join(off.GLYPHS[Y[i]] for i in idxs)
        greedy = "".join(off.GLYPHS[c[0][0]] for c in cands)

        # solve each group independently
        per_group = []
        for cols, cd_col in GROUPS:
            sols = group_solutions(cands, cols, cd_col, args.max_change)
            per_group.append((cols, cd_col, sols))
        sizes = [len(s) for _, _, s in per_group]
        total = 1
        for s in sizes:
            total *= s
        # the GT itself must be one of the group solutions
        gt_in = []
        for (cols, cd_col, sols) in per_group:
            want = [gt[c] for c in cols]
            ok = any(all(off.GLYPHS[g] == w
                         for g, w in zip(sol[0], want)) for sol in sols)
            gt_in.append(ok)
        # composite
        comp_gt = check_digit(gt[0:10] + gt[13:20] + gt[21:43])
        gt_comp_ok = comp_gt == cd_value(gt[43])
        greedy_comp_ok = (check_digit(greedy[0:10] + greedy[13:20] +
                                      greedy[21:43]) == cd_value(greedy[43]))
        print(f"{fname:32s} group_sizes={sizes} total={total} "
              f"GT_in_groups={gt_in} GT_comp_ok={gt_comp_ok} "
              f"greedy_comp_ok={greedy_comp_ok}")
        if not all(gt_in):
            print(f"    GT    : {gt}")
            print(f"    greedy: {greedy}")


if __name__ == "__main__":
    main()
