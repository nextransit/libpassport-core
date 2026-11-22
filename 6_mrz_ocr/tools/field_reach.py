#!/usr/bin/env python3
"""Field-scoped reachability for the line-2 checksum decoder.

Why this exists
---------------
The end-to-end beam picks its candidate COLUMNS by ascending top-2
probability margin.  A cell the network is *confidently* wrong about has
a huge margin, so it lands at the very end of that ordering and is never
searched.  That is why widening the candidate width (TOPK) changed
nothing: the extra candidates belonged to columns that were never
visited.

But the check digits give a MUCH better way to localise the problem.
Line 2 has five independent check digits:

    doc        cols 0..8   check at 9
    birth      cols 13..18 check at 19
    expiry     cols 21..26 check at 27
    passport   cols 28..41 check at 42
    composite  the four fields above, check at 43

If exactly ONE of those check digits is violated, then exactly ONE field
holds the error -- and the search should be confined to that field.  A
field is at most 14 columns wide, which is affordable to search
exhaustively even when the network is confidently wrong inside it.

This tool measures, per failing line, which fields violate their check
digit and whether the TRUE field content is inside the classifier's
top-K for every one of its columns.  That is the ceiling of a
field-scoped decoder.

Usage:
    python3 tools/field_reach.py --weights /tmp/rl2_L0.h --prefix LINE2 \
        --topk 5
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
    total = 0
    for index, char in enumerate(value):
        total += char_value(char) * WEIGHTS[index % 3]
    return total % 10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--prefix", default="LINE2")
    ap.add_argument("--cache", default="/tmp/glyph_cache.npz")
    ap.add_argument("--topk", type=int, default=5)
    args = ap.parse_args()

    d = np.load(args.cache, allow_pickle=True)
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
        idxs.sort(key=lambda i: int(meta[i]["col"]))
        truth = "".join(meta[i]["ch"] for i in idxs)
        if len(truth) != 44:
            continue
        sets = []
        for i in idxs:
            order = np.argsort(-pr[i])[:args.topk]
            sets.append({G[o] for o in order})

        bad = []
        for name, s, e, cd in FIELDS:
            body = truth[s:e]
            want = str(check_digit(body))
            if truth[cd] != want:
                bad.append((name, s, e, cd, want))
        comp = truth[0:10] + truth[13:20] + truth[21:28] + truth[28:43]
        comp_bad = truth[43] != str(check_digit(comp))

        if not bad and not comp_bad:
            agg["gt_ok"] += 1
            continue
        agg["gt_bad"] += 1
        names = [b[0] for b in bad]
        # is the truth inside the top-K for every column of the bad fields?
        miss = []
        for name, s, e, cd, want in bad:
            for c in range(s, e):
                if truth[c] not in sets[c]:
                    miss.append(f"{name}.c{c}:{truth[c]}")
        if comp_bad and not miss:
            miss.append("composite")
        verdict = "FIELD-FIXABLE" if (bad and not miss) else (
            "COMPOSITE-ONLY" if (not bad and comp_bad) else "NOT-IN-TOPK")
        agg[verdict] += 1
        print(f"{f:30s} bad_fields={names or ['-']} "
              f"comp_bad={int(comp_bad)} miss={miss[:8]} -> {verdict}")

    print(f"\ngt checksum-ok={agg['gt_ok']} gt checksum-bad={agg['gt_bad']}")
    print(f"FIELD-FIXABLE={agg['FIELD-FIXABLE']} "
          f"COMPOSITE-ONLY={agg['COMPOSITE-ONLY']} "
          f"NOT-IN-TOPK={agg['NOT-IN-TOPK']}")


if __name__ == "__main__":
    main()
