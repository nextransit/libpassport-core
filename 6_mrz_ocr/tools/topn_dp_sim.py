#!/usr/bin/env python3
"""Top-N prefix DP: does keeping more than one prefix per residue help?

WHY
---
The exact DP merges all prefixes that share a residue pair (own, comp)
and keeps only the highest-scoring one.  That is exact for the OPTIMUM
but destroys any path whose characters are individually unlikely: a
ground-truth character with p = 0.000 loses to a wrong character with
p = 0.8 at the same residue, and the ground-truth path is gone.

This script keeps the TOP-N prefixes per residue pair and re-runs the
constraint check.  If the ground truth becomes reachable, the fix is a
small change to mrz_field_solve() in the C decoder.

Usage:
    python3 tools/topn_dp_sim.py [TOPK] [KEEP]
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DIAG_DIR = Path("/tmp/diags")
GT_JSON = ROOT / "10_ocr_research" / "gt_review_loop.json"

W = (7, 3, 1)


def val(ch: str) -> int:
    if ch.isdigit():
        return ord(ch) - 48
    return ord(ch) - 55


def cd_of(seq: str) -> int:
    return sum(val(c) * W[i % 3] for i, c in enumerate(seq)) % 10


def ci(c: int) -> int:
    if c <= 9:
        return c
    if c <= 19:
        return c - 3
    return c - 4


def cw(c: int) -> int:
    return W[ci(c) % 3]


def read_digit(cands):
    for ch, _lp in cands:
        if ch.isdigit():
            return int(ch)
    return None


def solve(cands_body, read42, read43, fixed, keep):
    """DP keeping the top `keep` prefixes per (own, comp) residue pair."""
    cur = {(0, 0): [(0.0, "")]}
    for j in range(14):
        c = 28 + j
        w = W[j % 3]
        cweight = cw(c)
        nxt = {}
        for (own, comp), plist in cur.items():
            for sc, body in plist:
                for ch, lp in cands_body[j]:
                    no = (own + val(ch) * w) % 10
                    nc = (comp + val(ch) * cweight) % 10
                    s = sc + lp
                    bucket = nxt.setdefault((no, nc), [])
                    bucket.append((s, body + ch))
        # prune each bucket to the top `keep`
        cur = {}
        for key, bucket in nxt.items():
            bucket.sort(key=lambda x: -x[0])
            cur[key] = bucket[:keep]
    res = []
    for (own, comp), plist in cur.items():
        for sc, body in plist:
            if read42 is not None and own != read42:
                continue
            if read43 is not None:
                if (comp + own * cw(42) + fixed) % 10 != read43:
                    continue
            res.append((sc, body))
    res.sort(key=lambda x: -x[0])
    return res


def main():
    topk = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    keep = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    gt = {}
    for k, v in json.loads(GT_JSON.read_text()).items():
        if isinstance(v, (list, tuple)) and len(v) == 2:
            gt[Path(k).stem] = v

    nreach = ntry = 0
    for p in sorted(DIAG_DIR.glob("*.json")):
        stem = p.stem
        if stem not in gt:
            continue
        d = json.loads(p.read_text())
        g1, g2 = gt[stem]
        base = d.get("line2", "")
        if base == g2:
            continue
        line = d["lines"][1]

        def cands(c):
            return [(ch, math.log(max(pp, 1e-9)))
                    for ch, pp in line["chars"][c]["top"][:topk] if ch]

        read42 = read_digit(cands(42))
        read43 = read_digit(cands(43))
        fixed = sum(val(base[c]) * cw(c) for c in range(44)
                    if not (28 <= c <= 41)
                    and (c <= 9 or (13 <= c <= 19) or (21 <= c <= 42))) % 10
        ntry += 1
        res = solve([cands(28 + k) for k in range(14)], read42, read43,
                    fixed, keep)
        gtbody = g2[28:42]
        hit = any(b == gtbody for _, b in res)
        if hit:
            nreach += 1
        print(f"{stem:<24} nsol={len(res):<6} gt_reachable={hit}  "
              f"gt={gtbody}")
        if res:
            print(f"{'':<24} top={res[0][1]}")
    print(f"\ntopk={topk} keep={keep}: gt reachable on {nreach}/{ntry} "
          f"failing samples")


if __name__ == "__main__":
    main()
