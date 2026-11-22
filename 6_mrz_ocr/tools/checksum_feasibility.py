#!/usr/bin/env python3
"""Checksum-constrained decoding feasibility analysis (no numpy).

WHY THIS EXISTS
---------------
The C decoder DERIVES the check digits (col 9/19/27/42/43) from the
field bodies instead of matching the digits the network actually read
at those columns.  A derived check digit is always self-consistent, so
`line2_checksum_ok()` can never reject a wrong reading -- the checksum
carries ZERO information about whether the body is right.

This script asks the opposite question: if we treat the check digit as
a CONSTRAINT (trust the digit the network read at col 9/19/27/42 and
require the body to hash to it), can the failing samples be recovered
from the candidate sets the network already produced?

GT source: 10_ocr_research/gt_review_loop.json -- the reviewed sheet
the official bench reads (tests/gui.py::_gt_map).  The two JSON files
under tests/ carry STALE transcriptions of the same rows and must not
be used as the reference.

Usage:
    python3 tools/checksum_feasibility.py            # failing samples
    python3 tools/checksum_feasibility.py 2026-9-29-L
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DIAG_DIR = Path("/tmp/diags")
GT_JSON = ROOT / "10_ocr_research" / "gt_review_loop.json"

# field name, body start, body end (exclusive), check-digit column
FIELDS = [
    ("passport", 0, 9, 9),
    ("birth", 13, 19, 19),
    ("expiry", 21, 27, 27),
    ("personal", 28, 42, 42),
]
COMPOSITE_COL = 43
W = (7, 3, 1)


def val(ch: str) -> int:
    if ch.isdigit():
        return ord(ch) - 48
    return ord(ch) - 55


def cd_of(seq: str) -> int:
    return sum(val(c) * W[i % 3] for i, c in enumerate(seq)) % 10


def composite_seq(l2: str) -> str:
    """The 39-character sequence the composite digit covers."""
    return l2[0:10] + l2[13:20] + l2[21:43]


def load_gt() -> dict:
    d = json.loads(GT_JSON.read_text())
    gt = {}
    for k, v in d.items():
        if isinstance(v, (list, tuple)) and len(v) == 2:
            l1, l2 = v
            if l1 and l2 and len(l1) == 44 and len(l2) == 44:
                gt[Path(k).stem] = (l1, l2)
    return gt


def candidates(line: dict, col: int):
    """[(char, prob, rank)] for one column, best first."""
    out = []
    for rank, (ch, p) in enumerate(line["chars"][col]["top"]):
        if ch:
            out.append((ch, p, rank))
    return out


def read_digit(line: dict, col: int):
    """The digit the network actually READ at a check column."""
    for ch, p, rank in candidates(line, col):
        if ch.isdigit():
            return int(ch), rank, p
    return None, None, None


def analyse(stem: str, gt2: str, line2: dict, out2: str):
    print(f"=== {stem}")
    print(f"  gt   {gt2}")
    print(f"  pred {out2}")
    for name, s, e, cd in FIELDS:
        gt_body = gt2[s:e]
        pr_body = out2[s:e]
        gt_calc = cd_of(gt_body)
        pr_calc = cd_of(pr_body)
        read, read_rank, read_p = read_digit(line2, cd)
        if gt_calc == pr_calc and read is None:
            continue
        gt_ok = (read is not None and gt_calc == read)
        pr_ok = (read is not None and pr_calc == read)
        verdict = "REACHABLE" if (gt_ok and not pr_ok) else (
            "both-ok" if (gt_ok and pr_ok) else "unreachable")
        print(f"  {verdict:<10} {name:<10} gt[{gt_body}] calc={gt_calc} | "
              f"pred[{pr_body}] calc={pr_calc} | read={read} "
              f"(rank {read_rank}, p={read_p})")
        ranks = []
        for i, ch in enumerate(gt_body):
            col = s + i
            found = None
            for c, p, r in candidates(line2, col):
                if c == ch:
                    found = (r, p)
                    break
            ranks.append(found)
        bad = [(s + i, ch, r) for i, (ch, r) in
               enumerate(zip(gt_body, ranks)) if r is None or r[0] > 2]
        if bad:
            desc = ", ".join(
                f"c{col}:{ch}={'MISS' if r is None else 'rk%d p=%.4f' % r}"
                for col, ch, r in bad)
            print(f"             gt chars outside top3: {desc}")
    gcomp = cd_of(composite_seq(gt2))
    pcomp = cd_of(composite_seq(out2))
    rcomp, rrank, rp = read_digit(line2, COMPOSITE_COL)
    verdict = "REACHABLE" if (gcomp == rcomp and pcomp != rcomp) else (
        "both-ok" if (gcomp == rcomp and pcomp == rcomp) else "unreachable")
    print(f"  {verdict:<10} composite   gt calc={gcomp} pred calc={pcomp} "
          f"read={rcomp} (rank {rrank}, p={rp})")


def main():
    gt = load_gt()
    only = sys.argv[1] if len(sys.argv) > 1 else None
    nfail = 0
    for p in sorted(DIAG_DIR.glob("*.json")):
        stem = p.stem
        if stem not in gt:
            continue
        if only and only not in stem:
            continue
        d = json.loads(p.read_text())
        g2 = gt[stem][1]
        out2 = d.get("line2", "")
        if out2 == g2:
            continue
        nfail += 1
        analyse(stem, g2, d["lines"][1], out2)
    print(f"\nfailing line2 samples: {nfail}")


if __name__ == "__main__":
    main()
