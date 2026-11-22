#!/usr/bin/env python3
"""Personal-number-field decode: can the check digit alone find the truth?

The failing rows need SEVERAL columns of the personal number changed at
once (Photo/3-8 needs five), which is why the whole-line beam (depth 4)
and the whole-line minimal-change search both fail.

But the personal number has its OWN check digit (col 42).  Inside the
14-column field that is a tight constraint: a reading of 14 characters
whose 7-3-1 checksum equals the printed digit is very unlikely to be
produced by chance.  So the question this script answers is narrow and
decisive:

    inside cols 28..41, does the reading that changes the FEWEST
    columns and satisfies check_digit(field) == printed digit coincide
    with the ground truth?

If yes, the fix is local -- a field-scoped minimal-change search -- and
it does not need the whole line to be right at once.

Usage:
    python3 tools/pn_field_decode.py --weights /tmp/rl2_L0.h --prefix LINE2
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


def cval(c):
    if c == "<":
        return 0
    if c.isdigit():
        return int(c)
    return ord(c) - 55


def weighted(chars):
    return sum(cval(c) * WT[i % 3] for i, c in enumerate(chars)) % 10


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
    ap.add_argument("--prefix", default="LINE2")
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--max-candidates", type=int, default=3)
    ap.add_argument("--max-changes", type=int, default=6)
    ap.add_argument("--files", default="")
    args = ap.parse_args()

    d = np.load(args.cache, allow_pickle=True)
    X = d["X"].astype(np.float32)
    meta = [json.loads(s) for s in d["meta"]]
    P = ce.pick(ce.parse_header(args.weights), args.prefix)
    keep = [i for i, m in enumerate(meta) if int(m["line"]) == 1]
    logits = ce.forward(X[keep], P)
    prob = ce.softmax(logits)
    order = np.argsort(-prob, axis=1)

    gts = load_gt()
    wanted = set(args.files.split(",")) if args.files else None

    byfile = {}
    for j, i in enumerate(keep):
        byfile.setdefault(meta[i]["file"], []).append((int(meta[i]["col"]), j, i))

    print("%-26s %5s %5s  %s" % ("file", "greedy", "min-ch", "verdict"))
    print("-" * 78)

    n_fixed = n_broke = n_same = 0
    for f in sorted(byfile):
        if wanted and f not in wanted:
            continue
        cells = sorted(byfile[f])
        if len(cells) < 44:
            continue
        cands = {}
        for c, j, i in cells:
            lst = []
            for o in order[j][:args.topk]:
                ch = GLYPHS[int(o)]
                if c == 20:
                    ok = ch in set("MF<")
                elif 28 <= c <= 41:
                    ok = ch in ALNUM or ch == "<"
                elif c in (9, 19, 27, 42, 43):
                    ok = ch in DIGITS
                elif 10 <= c <= 12:
                    ok = ch in ALPHA
                elif 13 <= c <= 19 or 21 <= c <= 27:
                    ok = ch in DIGITS
                else:
                    ok = ch in ALNUM
                if ok:
                    lst.append(ch)
            cands[c] = lst or [GLYPHS[int(order[j][0])]]

        base = [cands[c][0] for c in range(44)]
        field = base[28:42]
        printed = base[42]
        truth = gts.get(f)
        gt = truth[1] if truth else None

        greedy_ok = (gt is not None and base == list(gt))
        best = None
        best_nch = None
        if weighted(field) != int(printed):
            # Only the columns whose SECOND choice still has real
            # probability mass are worth trying: a column the network
            # is sure about cannot be the mis-read one, and including
            # them makes the combination count explode.
            cols = []
            for c in range(28, 42):
                if len(cands[c]) <= 1:
                    continue
                j = [jj for cc, jj, ii in cells if cc == c][0]
                p0 = float(prob[j][order[j][0]])
                p1 = float(prob[j][order[j][1]]) if len(order[j]) > 1 else 0.0
                if p1 / max(p0, 1e-9) > 0.01:
                    cols.append(c)
            for nch in range(1, args.max_changes + 1):
                for combo in itertools.combinations(cols, nch):
                    pools = [cands[c][:args.max_candidates] for c in combo]
                    for pick in itertools.product(*pools):
                        fld = list(field)
                        for c, ch in zip(combo, pick):
                            fld[c - 28] = ch
                        if weighted(fld) != int(printed):
                            continue
                        best = list(base)
                        for c, ch in zip(combo, pick):
                            best[c] = ch
                        best_nch = nch
                        break
                    if best:
                        break
                if best:
                    break
        else:
            best = list(base)
            best_nch = 0

        if best is None:
            print("%-26s %5s %5s  no checksum-valid reading found" %
                  (f, "-" if not greedy_ok else "OK", "-"))
            continue

        # the composite check digit at 43 follows from the rest
        comp = weighted([best[c] for c in list(range(0, 10)) +
                         list(range(13, 20)) + list(range(21, 43))])
        best[43] = str(comp)

        out_ok = (gt is not None and best == list(gt))
        if out_ok and not greedy_ok:
            verdict = "FIXED"
            n_fixed += 1
        elif greedy_ok and not out_ok:
            verdict = "BROKE"
            n_broke += 1
        elif out_ok:
            verdict = "still ok"
            n_same += 1
        else:
            verdict = "still wrong cols %s" % [
                c for c in range(44) if best[c] != gt[c]]
        print("%-26s %5s %5s  %s" % (
            f, "OK" if greedy_ok else "bad", best_nch, verdict))

    print()
    print("fixed=%d broke=%d still-ok=%d" % (n_fixed, n_broke, n_same))


if __name__ == "__main__":
    main()
