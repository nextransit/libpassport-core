#!/usr/bin/env python3
"""Field-scoped search with a DERIVED check digit -- feasibility probe.

Why this exists
---------------
The shipped beam changes at most MRZ_BEAM_DEPTH (=4) columns, and it
picks the columns it may change by ascending top-2 probability margin.
A cell the network is *confidently* wrong about has a huge margin, so it
is never selected -- and a passport number whose five characters are
all misread needs five simultaneous changes, which exceeds the depth
anyway.

The check digits localise the error much better.  Line 2 carries five
independent check digits (doc 0..8/9, birth 13..18/19, expiry 21..26/27,
personal number 28..41/42, composite 43).  When one field's check
digit fails, the error is inside that field, and the field can be
replaced WHOLE -- no depth limit, and the columns are chosen by the
check digit rather than by the margin.

The key property is that the check digit is DERIVED from the field
body, not chosen: for every candidate body the implied digit is
computed, so the decoder only has to win on the FIELD, not on the
individual character.

This probe measures the ceiling of that approach offline.

Usage:
    python3 tools/field_search.py --weights /tmp/rl2_L0.h --prefix LINE2 \
        --topk 8 --beam 40000
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


def composite(line: str) -> str:
    return line[0:10] + line[13:20] + line[21:28] + line[28:43]


def icao_allow(line: int, col: int) -> set:
    """Faithful port of `fill_allow` (strict ICAO branch) in
    src/mrz_ocr_cnn.c."""
    A = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    D = set("0123456789")
    F = {"<"}
    if line == 0:
        if col == 0:
            return {"A", "C", "I", "P", "V"}
        if col == 1:
            return A | F
        if 2 <= col <= 4:
            return A
        return A | F
    if col <= 8:
        return A | D | F
    if col in (9, 19, 27):
        return D
    if col <= 12:
        return A | F
    if (13 <= col <= 18) or (21 <= col <= 26):
        return D | F | A
    if col == 20:
        return A | D | F
    if col <= 41:
        return A | D | F
    return D | F


def field_bodies(cands, lps, s, e, beam):
    """Beam-pruned enumeration of the field body.

    Returns a list of (score, body, implied_digit) sorted by score.
    """
    cols = list(range(s, e))
    states = [(0.0, "", 0)]
    for c in cols:
        nxt = []
        w = WEIGHTS[(c - s) % 3]
        for score, body, wsum in states:
            for j, ch in enumerate(cands[c]):
                nxt.append((score + lps[c][j], body + ch,
                            (wsum + char_value(ch) * w) % 10))
        if len(nxt) > beam:
            nxt.sort(key=lambda x: -x[0])
            nxt = nxt[:beam]
        states = nxt
    states.sort(key=lambda x: -x[0])
    return [(sc, body, str(ws)) for sc, body, ws in states]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--prefix", default="LINE2")
    ap.add_argument("--cache", default="/tmp/glyph_cache.npz")
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--beam", type=int, default=40000)
    ap.add_argument("--max-per-field", type=int, default=4000)
    ap.add_argument("--only", default="")
    ap.add_argument("--verbose", action="store_true")
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
        if args.only and args.only not in f:
            continue
        idxs.sort(key=lambda i: int(meta[i]["col"]))
        truth = "".join(meta[i]["ch"] for i in idxs)
        if len(truth) != 44:
            continue
        cands, lps = [], []
        for i in idxs:
            col = int(meta[i]["col"])
            allow = icao_allow(1, col)
            order = np.argsort(-pr[i])
            cl, ll = [], []
            for o in order:
                ch = G[o]
                if ch not in allow:
                    continue
                cl.append(ch)
                ll.append(float(np.log(max(float(pr[i][o]), 1e-12))))
                if len(cl) >= args.topk:
                    break
            if not cl:
                cl, ll = ["<"], [0.0]
            cands.append(cl)
            lps.append(ll)
        greedy = "".join(c[0] for c in cands)
        if greedy == truth:
            agg["exact"] += 1
            continue

        bad = [(n, s, e, cdp) for n, s, e, cdp in FIELDS
               if greedy[cdp] != str(check_digit(greedy[s:e]))]
        if not bad:
            agg["composite_only"] += 1
            continue

        # Solve each bad field; the check digit is DERIVED.
        field_solutions = []
        for name, s, e, cdp in bad:
            sols = field_bodies(cands, lps, s, e, args.beam)
            field_solutions.append((name, s, e, cdp, sols[:args.max_per_field]))

        # Cross product, filtered by the composite digit.
        best, best_score = None, -1e300
        lists = [fs[4] for fs in field_solutions]
        for combo in _product(lists):
            cand = list(greedy)
            score = 0.0
            for (name, s, e, cdp, _), (sc, body, dig) in zip(
                    field_solutions, combo):
                for kk, ch in enumerate(body):
                    cand[s + kk] = ch
                cand[cdp] = dig
                score += sc
            s2 = "".join(cand)
            if s2[43] != str(check_digit(composite(s2))):
                continue
            ok = True
            for name, s, e, cdp in FIELDS:
                if any(name == fs[0] for fs in field_solutions):
                    continue
                if s2[cdp] != str(check_digit(s2[s:e])):
                    ok = False
                    break
            if not ok:
                continue
            if score > best_score:
                best_score = score
                best = s2
        if best is None:
            agg["no_solution"] += 1
            if args.verbose:
                print(f"{f:30s} NO SOLUTION bad={[b[0] for b in bad]}")
            continue
        if best == truth:
            agg["fixed"] += 1
            if args.verbose:
                print(f"{f:30s} FIXED  bad={[b[0] for b in bad]}")
        else:
            agg["wrong"] += 1
            if args.verbose:
                print(f"{f:30s} WRONG  got={best}")
                print(f"{'':30s}        want={truth}")

    print(dict(agg))


def _product(lists):
    import itertools
    return itertools.product(*lists)


if __name__ == "__main__":
    main()
