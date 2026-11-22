#!/usr/bin/env python3
"""Minimal-change constrained decode, measured on the real corpus.

The C decoder's checksum beam maximises the TOTAL LOG-PROBABILITY of a
reading subject to the check digits closing.  That objective was measured
to fail on this corpus: among the many readings that satisfy the five
check digits, the most probable one is routinely NOT the true one, because
the truth often needs SEVERAL columns changed at once.

The prior that does discriminate is the one the rest of the pipeline
already assumes: a mis-read character is an EXCEPTION, so the correct
reading changes the FEWEST columns away from the greedy reading.  The
objective is therefore lexicographic:

    1. satisfy the check digits (hard constraint)
    2. change as few columns as possible
    3. among those, maximise the total log-probability

This script measures what that would buy, BEFORE any of it is ported to
C.  It uses the cached product-path glyphs and the exact numpy forward
pass, so the numbers are directly comparable with the offline scores.

Usage:
    python3 tools/minimal_change_decode.py --weights /tmp/rl2_L0.h \
        --prefix LINE2 --line 1
"""
from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "ce", str(ROOT / "tools" / "cnn_offline_eval.py"))
ce = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ce)

GLYPHS = ce.GLYPHS
WT = (7, 3, 1)

ALPHA = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
DIGITS = set("0123456789")
ALNUM = ALPHA | DIGITS

FIELDS = [
    (list(range(0, 9)), 9),
    (list(range(13, 19)), 19),
    (list(range(21, 27)), 27),
    (list(range(28, 42)), 42),
]
COMPOSITE = list(range(0, 10)) + list(range(13, 20)) + list(range(21, 43))


def cval(c):
    if c == "<":
        return 0
    if c.isdigit():
        return int(c)
    return ord(c) - 55


def weighted(chars):
    return sum(cval(c) * WT[i % 3] for i, c in enumerate(chars)) % 10


def line2_mask(col):
    # TD3 line 2 starts with the passport NUMBER, so col 0 is an
    # ordinary alphanumerics slot -- not the document-type letter of
    # line 1.
    if col <= 8:
        return ALNUM
    if col == 9:
        return DIGITS
    if 10 <= col <= 12:
        return ALPHA
    if 13 <= col <= 19:
        return DIGITS
    if col == 20:
        return set("MF<")
    if 21 <= col <= 27:
        return DIGITS
    if 28 <= col <= 41:
        return ALNUM | {"<"}
    return DIGITS


def load_gt():
    gt = {}
    for f in (ROOT.parent / "tests" / "gt_photo_passport.json",
              ROOT.parent / "tests" / "photo_auto_gt.json"):
        p = Path(f)
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        g = d.get("gt", d) if isinstance(d, dict) else d
        for k, v in g.items():
            if not isinstance(v, dict):
                continue
            l1 = v.get("l1") or v.get("line1")
            l2 = v.get("l2") or v.get("line2")
            if l1 and l2 and len(l1) == 44 and len(l2) == 44:
                gt[Path(k).stem] = (l1, l2)
    return gt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/tmp/glyph_cache.npz")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--line", type=int, default=1)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--max-changes", type=int, default=4)
    ap.add_argument("--max-candidates", type=int, default=3)
    ap.add_argument("--gt", default="")
    args = ap.parse_args()

    d = np.load(args.cache, allow_pickle=True)
    X = d["X"].astype(np.float32)
    meta = [json.loads(s) for s in d["meta"]]
    P = ce.pick(ce.parse_header(args.weights), args.prefix)
    keep = [i for i, m in enumerate(meta) if int(m["line"]) == args.line]
    logits = ce.forward(X[keep], P)
    prob = ce.softmax(logits)
    order = np.argsort(-prob, axis=1)

    mask = [line2_mask(c) for c in range(44)]
    gts = load_gt()

    byfile = {}
    for j, i in enumerate(keep):
        byfile.setdefault(meta[i]["file"], []).append((int(meta[i]["col"]), j, i))

    base_exact = 0
    fixed_exact = 0
    improved = 0
    regressed = 0
    nfiles = 0
    details = []

    for f in sorted(byfile):
        cells = sorted(byfile[f])
        if len(cells) < 44:
            continue
        nfiles += 1
        cands = {}
        for c, j, i in cells:
            lst = []
            for o in order[j][:args.topk]:
                ch = GLYPHS[int(o)]
                if ch in mask[c]:
                    lst.append((ch, float(np.log(max(prob[j][int(o)], 1e-12)))))
            if not lst:
                lst = [(GLYPHS[int(order[j][0])], -20.0)]
            cands[c] = lst
        base = [cands[c][0][0] for c in range(44)]

        def all_ok(line):
            for p_range, cd in FIELDS:
                if weighted([line[c] for c in p_range]) != int(line[cd]):
                    return False
            return weighted([line[c] for c in COMPOSITE]) == int(line[43])

        out = list(base)
        if not all_ok(out):
            # search the whole line jointly: which columns to change, and
            # to what, so that every check digit closes.
            cols = [c for c in range(44)
                    if (c in mask and len(cands[c]) > 1)]
            found = None
            for nch in range(1, args.max_changes + 1):
                for combo in itertools.combinations(cols, nch):
                    pools = [cands[c][:args.max_candidates] for c in combo]
                    for pick in itertools.product(*pools):
                        line = list(base)
                        for c, (ch, _) in zip(combo, pick):
                            line[c] = ch
                        if all_ok(line):
                            found = line
                            break
                    if found:
                        break
                if found:
                    break
            if found:
                out = found

        truth = gts.get(f)
        if truth:
            gt = truth[args.line]
            b_ok = (base == list(gt))
            o_ok = (out == list(gt))
            base_exact += 1 if b_ok else 0
            fixed_exact += 1 if o_ok else 0
            if o_ok and not b_ok:
                improved += 1
                details.append("  FIXED   %s  %s -> %s" %
                               (f, "".join(base), "".join(out)))
            elif b_ok and not o_ok:
                regressed += 1
                details.append("  BROKE   %s  %s -> %s" %
                               (f, "".join(base), "".join(out)))
            elif not o_ok:
                diff = [c for c in range(44) if out[c] != gt[c]]
                details.append("  still   %s  wrong cols %s" %
                               (f, diff))

    print("files scored            : %d" % nfiles)
    print("greedy line exact       : %d" % base_exact)
    print("minimal-change exact    : %d" % fixed_exact)
    print("improved / regressed    : %d / %d" % (improved, regressed))
    print()
    for dline in details:
        print(dline)


if __name__ == "__main__":
    main()
