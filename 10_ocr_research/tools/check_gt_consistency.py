#!/usr/bin/env python3
"""Check every GT JSON against the authoritative xlsx ground truth.

The ground truth is the "GT对比" sheet of 10_ocr_research/gt_review_loop.xlsx
-- per AGENTS.md it is the ONLY acceptance standard in this repository.
Several JSON snapshots of it exist; this script reports, for each of them,
how many entries differ from the sheet, and which side is better supported.

"Which side is better" is decided by three INDEPENDENT signals that need
no ground truth of their own:
  1. line 2 closes its five ICAO check digits;
  2. line 2 equals the reading most engines produced;
  3. line 1 equals the reading most engines produced.

Usage:
    python3 10_ocr_research/tools/check_gt_consistency.py
    python3 10_ocr_research/tools/check_gt_consistency.py --diff FILE
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "tests"))

import mrz_arbitration as A  # noqa: E402

try:
    from score_mrz_arb import HAND_VERIFIED_CHECK_DIGITS
except ImportError:  # pragma: no cover - running standalone
    HAND_VERIFIED_CHECK_DIGITS = {}

XLSX = os.path.join(ROOT, "10_ocr_research", "gt_review_loop.xlsx")

# The JSON snapshots that exist in the repository, in the order they
# should be read: exports of the xlsx first, then the older auto-GT
# files, then the hand-labelled one.
GT_JSONS = [
    "tests/gt_assets_pic_official.json",
    "tests/gt_photo_passport_official.json",
    "tests/gt_all_from_xlsx.json",
    "tests/gt_passport_from_xlsx.json",
    "tests/gt_all_from_xlsx_v2.json",
    "tests/gt_passport_v3_from_xlsx.json",
    "tests/gt_pic_from_xlsx_v2.json",
    "tmp/gt_pic_xlsx.json",
    "tmp/gt_passport_xlsx.json",
    "tests/gt_photo_passport.json",
    "tests/gt_assets_pic.json",
    "tests/photo_auto_gt.json",
    "10_ocr_research/archive/photo_auto_gt_all.json",
    "tests/real_gt.json",
]


def load_xlsx(path=XLSX):
    import openpyxl
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    gt = {}
    for row in wb["GT对比"].iter_rows(min_row=2, values_only=True):
        name = row[1] or ""
        if name:
            gt[name] = (row[6] or "", row[7] or "")
    wb.close()
    return gt


def load_majority(bench_path):
    """{filename: (majority line1, majority line2)} over the engine readings."""
    if not bench_path or not os.path.exists(bench_path):
        return {}
    data = json.load(open(bench_path))
    maj = {}
    for rec in data.get("records", []):
        name = os.path.basename(rec.get("label", ""))
        raw = rec.get("raw", {})
        engines = [dict(v, method=k) for k, v in raw.items()
                   if v.get("status") != "skip" and v.get("line1")]
        if not engines:
            continue
        l1, l2 = {}, {}
        for e in engines:
            if len(e.get("line1") or "") == 44:
                l1[e["line1"]] = l1.get(e["line1"], 0) + 1
            if len(e.get("line2") or "") == 44:
                l2[e["line2"]] = l2.get(e["line2"], 0) + 1
        maj[name] = (max(l1, key=l1.get) if l1 else "",
                     max(l2, key=l2.get) if l2 else "")
    return maj


def g1_of(v):
    return v.get("l1", v.get("line1", "")) or ""


def g2_of(v):
    return v.get("l2", v.get("line2", "")) or ""


def compare(path, xlsx, maj):
    """Return (stats, differences) for one JSON file."""
    data = json.load(open(path))
    gt = data.get("gt") or {}
    stats = {"n": 0, "same": 0, "diff": 0, "not_in_xlsx": 0,
             "json_better": 0, "xlsx_better": 0, "tie": 0}
    diffs = []
    for name, v in gt.items():
        if name not in xlsx:
            stats["not_in_xlsx"] += 1
            continue
        stats["n"] += 1
        a1, a2 = g1_of(v), g2_of(v)
        x1, x2 = xlsx[name]
        if a1 == x1 and a2 == x2:
            stats["same"] += 1
            continue
        stats["diff"] += 1
        m1, m2 = maj.get(name, ("", ""))
        sj = sx = 0
        if m1 and len(a1) == 44 and len(x1) == 44:
            sj += (a1 == m1)
            sx += (x1 == m1)
        if len(a2) == 44 and len(x2) == 44:
            sj += A.line2_checksum_ok(a2)
            sx += A.line2_checksum_ok(x2)
            if m2:
                sj += (a2 == m2)
                sx += (x2 == m2)
        if sj > sx:
            stats["json_better"] += 1
        elif sx > sj:
            stats["xlsx_better"] += 1
        else:
            stats["tie"] += 1
        diffs.append((name, a1, a2, x1, x2, sj, sx))
    return stats, diffs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", default=XLSX)
    ap.add_argument("--bench", default=os.path.join(ROOT, "10_ocr_research",
                                                    "photo_direct_bench_final.json"))
    ap.add_argument("--diff", default=None,
                    help="print the differing rows of this JSON file")
    args = ap.parse_args()

    xlsx = load_xlsx(args.gt)
    maj = load_majority(args.bench)

    n44 = sum(1 for a, b in xlsx.values()
              if len(a) == 44 and len(b) == 44)
    print(f"xlsx 'GT对比': {len(xlsx)} rows, {n44} of them 44-char TD3")
    bad = [k for k, (a, b) in xlsx.items()
           if len(a) == 44 and len(b) == 44 and not A.line2_checksum_ok(b)]
    print(f"  rows whose line2 check digits do NOT close: {len(bad)}")
    for k in bad:
        fixed = HAND_VERIFIED_CHECK_DIGITS.get(k)
        tag = ("  -> hand-verified: " + fixed) if fixed else ""
        print(f"    {k}{tag}")
    print()

    hdr = (f"{'file':46s} {'entries':>7s} {'=xlsx':>6s} {'diff':>5s} "
           f"{'not-in':>6s} {'json>':>6s} {'xlsx>':>6s} {'tie':>4s}")
    print(hdr)
    print("-" * len(hdr))
    tot_j = tot_x = 0
    for rel in GT_JSONS:
        path = os.path.join(ROOT, rel)
        if not os.path.exists(path):
            continue
        st, _ = compare(path, xlsx, maj)
        tot_j += st["json_better"]
        tot_x += st["xlsx_better"]
        print(f"{rel:46s} {st['n']:7d} {st['same']:6d} {st['diff']:5d} "
              f"{st['not_in_xlsx']:6d} {st['json_better']:6d} "
              f"{st['xlsx_better']:6d} {st['tie']:4d}")
    print(f"\nTOTAL over all differing rows:  JSON better {tot_j}, "
          f"XLSX better {tot_x}")

    if args.diff:
        path = os.path.join(ROOT, args.diff)
        _, diffs = compare(path, xlsx, maj)
        print(f"\n--- differing rows of {args.diff} ---")
        for name, a1, a2, x1, x2, sj, sx in diffs:
            side = ("JSON" if sj > sx else
                    "XLSX" if sx > sj else "tie")
            print(f"  {name}   [{side}]")
            if a1 != x1:
                print(f"     l1 json : {a1}")
                print(f"     l1 xlsx : {x1}")
            if a2 != x2:
                print(f"     l2 json : {a2}")
                print(f"     l2 xlsx : {x2}")


if __name__ == "__main__":
    main()
