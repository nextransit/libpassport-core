#!/usr/bin/env python3
"""Is the TRUE string reachable from the classifier's candidate set?

This is the decisive question for every failing line:

  * truth reachable  -> the pixels carry the information; the failure is
    a DECODER failure (wrong column chosen, wrong depth, wrong
    branching), and no amount of extra training is needed.
  * truth NOT reachable -> the classifier does not even rank the true
    class inside its top-K; the failure is a CLASSIFIER failure and only
    new/better weights can fix it.

Reachability is decided per line:
  * the truth string must satisfy the line's ICAO constraints
    (line 1: syntax only -- it has NO check digits; line 2: the five
    TD3 check digits), and
  * every character of the truth must appear in that column's top-K
    candidate list.

Usage:
    python3 tools/truth_reachable.py --weights /tmp/rl1_L0.h /tmp/rl2_L0.h \
        --prefix LINE1 LINE2 --topk 5
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np

import cnn_offline_eval as ce

WEIGHTS = (7, 3, 1)
CD_FIELDS = ((9, 0, 9), (19, 13, 19), (27, 21, 27), (42, 28, 42))


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


def line1_ok(line: str) -> bool:
    if len(line) != 44 or line[0] != "P":
        return False
    if not all("A" <= c <= "Z" for c in line[2:5]):
        return False
    return all(c == "<" or "A" <= c <= "Z" for c in line[5:])


def line2_ok(line: str) -> bool:
    for pos, start, end in CD_FIELDS:
        if line[pos] != str(check_digit(line[start:end])):
            return False
    comp = line[0:10] + line[13:20] + line[21:28] + line[28:43]
    return line[43] == str(check_digit(comp))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", nargs="+", required=True)
    ap.add_argument("--prefix", nargs="+", required=True)
    ap.add_argument("--cache", default="/tmp/glyph_cache.npz")
    ap.add_argument("--topk", type=int, default=5)
    args = ap.parse_args()

    d = np.load(args.cache, allow_pickle=True)
    meta = [json.loads(s) for s in d["meta"]]
    X = d["X"].astype(np.float32)
    G = ce.GLYPHS

    preds = []
    for w, p in zip(args.weights, args.prefix):
        P = ce.pick(ce.parse_header(w), p)
        preds.append(ce.softmax(ce.forward(X, P)))

    groups = defaultdict(list)
    for i, m in enumerate(meta):
        groups[(m["file"], int(m["line"]))].append(i)

    print(f"{'file':30s} {'ln':>2s} {'errs':>4s} {'miss':>4s}  "
          f"{'misscols':38s} verdict")
    for ln in (0, 1):
        agg = defaultdict(int)
        for (f, l), idxs in sorted(groups.items()):
            if l != ln:
                continue
            idxs.sort(key=lambda i: int(meta[i]["col"]))
            truth = "".join(meta[i]["ch"] for i in idxs)
            if len(truth) != 44:
                continue
            sets = []
            errs = []
            for i in idxs:
                pr = preds[ln][i]
                order = np.argsort(-pr)[:args.topk]
                s = {G[o] for o in order}
                sets.append(s)
                if truth[int(meta[i]["col"])] not in s:
                    errs.append((int(meta[i]["col"]),
                                 truth[int(meta[i]["col"])]))
            cons_ok = line1_ok(truth) if ln == 0 else line2_ok(truth)
            if not errs and cons_ok:
                agg["reachable"] += 1
                agg["reachable_exact"] += 1
                continue
            agg["total"] += 1
            reach = cons_ok and not errs
            if reach:
                agg["reachable"] += 1
                agg["reachable_exact"] += 1
            else:
                agg["unreachable"] += 1
            miss = ",".join(f"c{c}:{ch}" for c, ch in errs[:6])
            verdict = "REACHABLE" if reach else (
                "CHECKSUM-BAD" if not cons_ok else "NOT-IN-TOPK")
            if errs or not cons_ok:
                print(f"{f:30s} {ln+1:2d} {len(errs):4d} {len(errs):4d}  "
                      f"{miss:38s} {verdict}")
        print(f"  line{ln+1}: total={agg['total']} "
              f"reachable={agg['reachable']} "
              f"reachable_AND_exact={agg['reachable_exact']} "
              f"unreachable={agg['unreachable']}")
        print()


if __name__ == "__main__":
    main()
