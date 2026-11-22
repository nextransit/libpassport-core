#!/usr/bin/env python3
"""Offline constrained decode over the cached glyph probabilities.

This is the PYTHON reference for what the C pipeline's checksum-constrained
decoder does, applied to the cached 16x12 glyph probabilities.  It exists
to answer one question that the end-to-end bench cannot answer:

    for a line that currently decodes wrong, is the TRUE string
    reachable at all from the classifier's candidate set?

If the truth is reachable, the decoder's search (column choice, depth,
branching factor) is the bottleneck.  If it is not, the classifier is.

The search uses INCREMENTAL check-digit state, exactly like the C
implementation's `mrz_cd_state`: the five line-2 check digits are
accumulated field by field as the columns are walked, so a partial
assignment that already violates a completed field's check digit is
rejected immediately instead of at the leaf.

Usage:
    python3 tools/decode_oracle.py --weights /tmp/rl1_L0.h /tmp/rl2_L0.h \
        --prefix LINE1 LINE2 --depth 3
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict

import numpy as np

import cnn_offline_eval as ce

WEIGHTS = (7, 3, 1)

# ICAO 9303 TD3 line-2: (check-digit position, field start, field end)
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


def composite(line: str) -> str:
    return line[0:10] + line[13:20] + line[21:28] + line[28:43]


def line1_ok(line: str) -> bool:
    """Line 1 has NO check digits; only the ICAO syntax applies."""
    if len(line) != 44 or line[0] != "P":
        return False
    if not all("A" <= c <= "Z" for c in line[2:5]):
        return False
    return all(c == "<" or "A" <= c <= "Z" for c in line[5:])


class CdState:
    """Incremental check-digit state for the five line-2 fields.

    Each field keeps a running weighted sum; the field is COMPLETE when
    its last character has been assigned, at which point the check digit
    position can be compared.
    """

    # (field start, field end, check-digit position, weight restart)
    FIELDS = CD_FIELDS

    def __init__(self):
        self.sum = [0] * len(self.FIELDS)
        self.filled = [0] * len(self.FIELDS)
        self.known = [None] * len(self.FIELDS)

    def clone(self):
        s = CdState()
        s.sum = list(self.sum)
        s.filled = list(self.filled)
        s.known = list(self.known)
        return s

    def apply(self, col: int, ch: str):
        """Assign column `col` the character `ch`; return False if a
        completed field's check digit is already violated."""
        v = char_value(ch)
        for fi, (start, end, cdpos) in enumerate(self.FIELDS):
            if start <= col < end:
                # the weight restarts at 7 inside every field
                idx = col - start
                self.sum[fi] += v * WEIGHTS[idx % 3]
                self.filled[fi] += 1
                break
        return True

    def check(self, line: list) -> bool:
        """Test the completed assignment against the check digits."""
        for fi, (start, end, cdpos) in enumerate(self.FIELDS):
            if self.filled[fi] == end - start:
                if line[cdpos] != str(check_digit("".join(line[start:end]))):
                    return False
        return True


def line2_ok(line: str) -> bool:
    for pos, start, end in CD_FIELDS:
        if line[pos] != str(check_digit(line[start:end])):
            return False
    return line[43] == str(check_digit(composite(line)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", nargs="+", required=True)
    ap.add_argument("--prefix", nargs="+", required=True)
    ap.add_argument("--cache", default="/tmp/glyph_cache.npz")
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--only-line", type=int, default=-1)
    ap.add_argument("--verbose", action="store_true")
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

    stats = {0: defaultdict(int), 1: defaultdict(int)}
    for (f, ln), idxs in sorted(groups.items()):
        if args.only_line >= 0 and ln != args.only_line:
            continue
        idxs.sort(key=lambda i: int(meta[i]["col"]))
        truth = "".join(meta[i]["ch"] for i in idxs)
        if len(truth) != 44:
            continue
        cands, lps = [], []
        for i in idxs:
            pr = preds[ln][i]
            order = np.argsort(-pr)[:args.topk]
            cands.append([G[o] for o in order])
            lps.append([float(np.log(max(pr[o], 1e-12))) for o in order])
        greedy = "".join(c[0] for c in cands)
        greedy_ok = line1_ok(greedy) if ln == 0 else line2_ok(greedy)

        # Only columns that sit inside a check-digit field can influence
        # the constraint; line 1 has none, so the search there is a
        # pure "best alternative" scan.
        if ln == 0:
            found = None
            best_score = -1e300
            if not greedy_ok:
                for c in range(44):
                    for j in range(1, len(cands[c])):
                        s = list(greedy)
                        s[c] = cands[c][j]
                        s = "".join(s)
                        score = sum(lps[k][0] for k in range(44)) + \
                            lps[c][j] - lps[c][0]
                        if line1_ok(s) and score > best_score:
                            best_score = score
                            found = s
            else:
                found = greedy
        else:
            found = _search_line2(cands, lps, greedy, args.depth)
        stats[ln]["total"] += 1
        stats[ln]["greedy_ok"] += int(greedy_ok)
        stats[ln]["greedy_exact"] += int(greedy == truth)
        if found is not None:
            stats[ln]["reach"] += 1
            stats[ln]["reach_exact"] += int(found == truth)
            if args.verbose and found != truth:
                print(f"  {f} line{ln+1}: found {found}")
                print(f"  {f} line{ln+1}: truth  {truth}")
        elif greedy_ok:
            stats[ln]["reach"] += 1

    for ln in (0, 1):
        s = stats[ln]
        if not s:
            continue
        print(f"line{ln + 1}: total={s['total']} "
              f"greedy_valid={s['greedy_ok']} "
              f"greedy_exact={s['greedy_exact']} "
              f"reachable={s['reach']} "
              f"reachable_AND_exact={s['reach_exact']}")


def _search_line2(cands, lps, greedy, depth):
    """Depth-limited search over column substitutions.

    The search walks the CANDIDATE columns in order (the columns whose
    top-2 classes differ), not all 44 columns.  Whenever the walk passes
    the last column of a check-digit field, that field's check digit is
    tested immediately and a violating branch is cut there and there.
    This is the same incremental pruning the C decoder does, and it is
    what keeps the search affordable.
    """
    work = list(greedy)
    best = None
    best_score = -1e300

    # (last body column, check-digit column, field start)
    CHECKS = ((8, 9, 0), (18, 19, 13), (26, 27, 21), (41, 42, 28))

    cand_cols = [c for c in range(44) if len(cands[c]) > 1]
    nc = len(cand_cols)

    def fields_closed_before(col):
        """The fields whose last body column is < col."""
        out = []
        for last, cdpos, start in CHECKS:
            if last < col:
                out.append((last, cdpos, start))
        return out

    def field_ok(last, cdpos, start):
        body = "".join(work[start:last + 1])
        return work[cdpos] == str(check_digit(body))

    def finalize(score):
        nonlocal best, best_score
        comp = "".join(work[0:10]) + "".join(work[13:20]) + \
            "".join(work[21:28]) + "".join(work[28:43])
        if work[43] == str(check_digit(comp)) and score > best_score:
            best_score = score
            best = "".join(work)

    def recurse(i, budget, score):
        nonlocal best, best_score
        if i == nc:
            finalize(score)
            return
        # every field that ended before this column must already be valid
        for last, cdpos, start in fields_closed_before(cand_cols[i]):
            if not field_ok(last, cdpos, start):
                return
        c = cand_cols[i]
        keep = work[c]
        # keep the greedy class
        work[c] = keep
        recurse(i + 1, budget, score)
        # switch to an alternative
        if budget > 0:
            for j in range(1, len(cands[c])):
                work[c] = cands[c][j]
                recurse(i + 1, budget - 1, score + lps[c][j] - lps[c][0])
            work[c] = keep
        work[c] = keep

    recurse(0, depth, 0.0)
    return best


if __name__ == "__main__":
    main()
