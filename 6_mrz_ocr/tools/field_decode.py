#!/usr/bin/env python3
"""FIELD-SCOPED exhaustive decoder for the line-2 check-digit fields.

Why this exists
---------------
The shipped beam picks its candidate columns by ASCENDING top-2
probability margin.  A cell the network is *confidently* wrong about has
a huge margin, so it sorts to the very end of that list and is never
searched -- which is why raising TOPK from 5 to 10 changed nothing:
the extra candidates belonged to columns that were never visited.

The check digits localise the error far better than the margin does.
Line 2 carries five independent check digits:

    doc        cols 0..8   check at 9
    birth      cols 13..18 check at 19
    expiry     cols 21..26 check at 27
    passport   cols 28..41 check at 42
    composite  the four fields above, check at 43

When exactly ONE field violates its check digit, exactly ONE field holds
the error, and the search can be confined to that field's columns.  The
field is at most 14 columns wide, so an exhaustive search over its
candidate alternatives is affordable -- even when the network is
confidently wrong inside it.

Key property: the check digit is NOT enumerated.  It is COMPUTED from
the field body.  That is what makes the search able to fix a cell the
network is confidently wrong about -- the decoder is not asked to
out-vote the classifier on the character, only on the field.

The composite check digit couples the four fields, so the fields are
solved independently first and the composite digit then selects between
the surviving combinations.

Usage:
    python3 tools/field_decode.py --weights /tmp/rl2_L0.h --prefix LINE2 \
        --topk 4
"""
from __future__ import annotations

import argparse
import itertools
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

ALNUM = set("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ<")
# line-2 column -> allowed characters (ICAO 9303 TD3)
DIGITS = set("0123456789")
ALPHA = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
FILLER = {"<"}


def icao_allow(line: int, col: int) -> set:
    """Characters allowed at (line, col) by the ICAO 9303 TD3 mask.

    This is a faithful port of `fill_allow` in src/mrz_ocr_cnn.c
    (strict ICAO branch).  The port matters: an earlier hand-written
    mask in this file was wrong and produced garbage decodes, which
    looked like a decoder bug but was a mask bug.
    """
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
    # line 2
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


def composite(line) -> str:
    return (line[0:10] + line[13:20] + line[21:28] + line[28:43])


def solve_field(cands, lps, s, e, beam=200000):
    """Enumerate the field body assignments, pruning with a beam.

    Returns a list of (score, body) sorted by score descending.  The
    check digit is not enumerated -- it is computed by the caller.
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
    return [(score, body) for score, body, _ in states]


def solve_field_cd(cands, lps, s, e, beam=200000, keep=64):
    """Field bodies together with the check digit THEY IMPLY.

    The check digit is DERIVED from the body, so every body yields
    exactly one digit.  The caller keeps only the bodies whose implied
    digit equals the digit the classifier read at the check position
    (the common case: the digit is right and only the body is wrong).
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
    out = []
    for score, body, wsum in states:
        out.append((score, body, str(wsum)))
    out.sort(key=lambda x: -x[0])
    return out[:keep]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--prefix", default="LINE2")
    ap.add_argument("--cache", default="/tmp/glyph_cache.npz")
    ap.add_argument("--topk", type=int, default=4)
    ap.add_argument("--beam", type=int, default=200000)
    ap.add_argument("--max-per-field", type=int, default=4000)
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

        bad = []
        for name, s, e, cdpos in FIELDS:
            if greedy[cdpos] != str(check_digit(greedy[s:e])):
                bad.append((name, s, e, cdpos))
        if not bad:
            agg["composite_only"] += 1
            continue

        # Solve each bad field.  The check digit is DERIVED from the
        # body, so the field contributes (body, implied_digit) pairs.
        # A pair is a candidate only when the implied digit equals the
        # digit the classifier read at the check position -- that is the
        # common case (the digit is right, the body is wrong) and it
        # prunes the search by a factor of ten.
        field_solutions = []
        for name, s, e, cdpos in bad:
            read_cd = greedy[cdpos]
            sols = [(sc, body, dig) for sc, body, dig in
                    solve_field_cd(cands, lps, s, e, args.beam,
                                   args.max_per_field)
                    if dig == read_cd]
            if not sols:
                # the check digit column itself is wrong: keep the
                # bodies and let the digit be corrected
                sols = [(sc, body, dig) for sc, body, dig in
                        solve_field_cd(cands, lps, s, e, args.beam,
                                       args.max_per_field)]
            field_solutions.append((name, s, e, cdpos, sols))

        # Cross-product over the bad fields.  Each field contributes
        # (score, body, derived_digit); the derived digit replaces the
        # digit the classifier read at the check position.  The
        # composite digit at column 43 is then checked against the
        # reassembled line.
        best = None
        best_score = -1e300
        lists = [fs[4] for fs in field_solutions]
        for combo in itertools.product(*lists):
            cand = list(greedy)
            score = 0.0
            for (name, s0, e0, cdpos, _), (sc, body, dig) in zip(
                    field_solutions, combo):
                for kk, ch in enumerate(body):
                    cand[s0 + kk] = ch
                cand[cdpos] = dig
                score += sc
            s2 = "".join(cand)
            if s2[43] != str(check_digit(composite(s2))):
                continue
            ok = True
            for name, s0, e0, cdpos in FIELDS:
                if any(name == fs[0] for fs in field_solutions):
                    continue
                if s2[cdpos] != str(check_digit(s2[s0:e0])):
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
                print(f"{f:30s} NO SOLUTION (bad={[b[0] for b in bad]})")
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


if __name__ == "__main__":
    main()
