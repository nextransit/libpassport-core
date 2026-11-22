#!/usr/bin/env python3
"""Is the ground truth REACHABLE under the ICAO checksum constraints?

The C beam search accepts a reading only when all five line-2 check
digits validate, and it prunes a branch as soon as its log-probability
bound falls below the incumbent.  Two different questions follow:

  (a) REACHABILITY -- does a checksum-valid reading that matches the GT
      character for character exist inside the CNN's top-K candidates?
      If not, no amount of beam depth or checksum cleverness can help:
      the classifier never proposed the right glyph.

  (b) RANKING -- among the checksum-valid readings, does the GT one
      score higher than the one the beam actually picked?  If it does
      not, the beam is behaving correctly and the fault is the
      classifier's probability calibration, not the search.

This tool answers both, offline, from the exported product-path slices.

Usage:
    python3 tools/checksum_reach.py --npz /tmp/holdout_photo2.npz
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


def l2_checks(l):
    """The five TD3 line-2 check digits, as the C code computes them."""
    if len(l) != 44:
        return False
    comp = l[0:10] + l[13:20] + l[21:43]
    return (check_digit(l[0:9]) == cd_value(l[9]) and
            check_digit(l[13:19]) == cd_value(l[19]) and
            check_digit(l[21:27]) == cd_value(l[27]) and
            check_digit(l[28:42]) == cd_value(l[42]) and
            check_digit(comp) == cd_value(l[43]))


def l1_checks(l):
    """Line 1 has only two check digits: col 9 and the composite col 43."""
    if len(l) != 44:
        return False
    comp = l[0:10] + l[13:43]
    return (check_digit(l[0:9]) == cd_value(l[9]) and
            check_digit(comp) == cd_value(l[43]))


def field_protected(col):
    """Is column `col` covered by a check digit in line 2?"""
    return (col <= 8 or col == 9 or 13 <= col <= 19 or
            21 <= col <= 27 or 28 <= col <= 42 or col == 43)


def enumerate_readings(cands, probs, max_change, limit=400000):
    """Enumerate readings within `max_change` substitutions of the greedy
    reading, over the per-column candidate lists.

    `cands[c]` is the list of (class index, probability) for column c,
    ordered by descending probability.  Returns a list of
    (log_prob, reading) for every checksum-valid reading found.
    """
    base = [c[0][0] for c in cands]
    n = len(base)
    cols = [c for c in range(n) if len(cands[c]) > 1]
    out = []
    seen = 0
    for k in range(0, max_change + 1):
        for combo in itertools.combinations(cols, k):
            for choices in itertools.product(*[range(1, len(cands[c]))
                                              for c in combo]):
                work = list(base)
                for c, j in zip(combo, choices):
                    work[c] = cands[c][j][0]
                seen += 1
                if seen > limit:
                    return out, seen, True
                lp = 0.0
                for c in range(n):
                    cls = work[c]
                    p = next((p for k2, p in cands[c] if k2 == cls), 1e-6)
                    lp += np.log(max(p, 1e-9))
                s = "".join(off.GLYPHS[k] for k in work)
                if l2_checks(s):
                    out.append((lp, s))
    return out, seen, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="/tmp/holdout_photo2.npz")
    ap.add_argument("--w1", default="src/cnn_weights_line1.h")
    ap.add_argument("--w2", default="src/cnn_weights_line2.h")
    ap.add_argument("--max-change", type=int, default=8)
    ap.add_argument("--only", default="", help="substring filter on file")
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
        if args.only and args.only not in fname:
            continue
        idxs = sorted(idxs, key=lambda i: meta[i]["col"])
        P = p1 if line == 0 else p2
        sm = off.softmax(off.forward(X[idxs], P))
        gt = "".join(off.GLYPHS[Y[i]] for i in idxs)
        cands = []
        for j in range(len(idxs)):
            order = np.argsort(-sm[j])
            cands.append([(int(order[k]), float(sm[j][order[k]]))
                          for k in range(5)])
        greedy = "".join(off.GLYPHS[c[0][0]] for c in cands)
        if line == 0:
            reach = "n/a (line 1)"
            rank = ""
        else:
            readings, seen, truncated = enumerate_readings(
                cands, sm, args.max_change)
            valid = [r for r in readings]
            gt_entries = [r for r in valid if r[1] == gt]
            if truncated:
                reach = f"TRUNCATED after {seen}"
            elif not valid:
                reach = "NO checksum-valid reading in the space"
            elif not gt_entries:
                reach = (f"reachable set has {len(valid)} valid readings but "
                         f"the GT is NOT among them")
            else:
                best = max(valid, key=lambda r: r[0])
                gt_lp = gt_entries[0][0]
                rank = (f"  GT lp={gt_lp:8.3f}  best lp={best[0]:8.3f}  "
                        f"-> {'GT WINS' if gt_lp > best[0] else 'GT LOSES'}")
                reach = f"{len(valid)} valid readings, GT present"
        diff = sum(1 for a, b in zip(greedy, gt) if a != b)
        print(f"{fname:32s} L{line} greedy_diff={diff:2d}  {reach}{rank}")
        if diff and line == 1:
            print(f"    GT    : {gt}")
            print(f"    greedy: {greedy}")


if __name__ == "__main__":
    main()
