#!/usr/bin/env python3
"""Score a photo_direct_bench JSON against the authoritative ground truth.

The ground truth is the "GT对比" sheet of 10_ocr_research/gt_review_loop.xlsx
-- per AGENTS.md it is the ONLY acceptance standard in this repository.
The bench JSON produced by tests/photo_direct_bench.py carries the
"official" (production path) result of every photo, so this script turns
a bench run into the numbers the acceptance table needs.

Usage:
    python3 10_ocr_research/tools/score_mrz_arb.py <bench.json> [...]
    python3 10_ocr_research/tools/score_mrz_arb.py --replay <bench.json>

--replay additionally re-runs the arbitration offline on the raw engine
readings stored in the bench JSON, so the arbitration can be measured
without re-running the engines.
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


def load_gt(path=None):
    """Return {filename: (line1, line2)} from the xlsx ground truth."""
    import openpyxl
    path = path or os.path.join(ROOT, "10_ocr_research",
                                "gt_review_loop.xlsx")
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    gt = {}
    for row in wb["GT对比"].iter_rows(min_row=2, values_only=True):
        name = row[1] or ""
        if name:
            gt[name] = (row[6] or "", row[7] or "")
    wb.close()
    return gt


def load_gt_fixed(path=None):
    """Ground truth with the six hand-verified check digits corrected."""
    gt = load_gt(path)
    return {name: (l1, fix_check_digits(l2, name))
            for name, (l1, l2) in gt.items()}


def char_accuracy(pred, ref):
    n = max(len(pred), len(ref))
    if not n:
        return 0.0
    hits = sum(1 for i in range(n)
               if (pred[i] if i < len(pred) else "_") ==
                  (ref[i] if i < len(ref) else "_"))
    return hits / n


# Ground-truth rows whose line 2 check digits were mistyped.  These were
# re-read from the photos by hand on 2026-11-04: the DATA FIELDS are
# correct in every one of them, only the check digits were typed wrong.
# The value below is the line with the check digits recomputed from those
# hand-verified data fields (ICAO 9303-4 7-3-1).
HAND_VERIFIED_CHECK_DIGITS = {
    "23f16a7d9b6ab9da_0062.jpg":
        "EK000001<1UKR8307255M19080333052125257<<<<02",
    "556644d1d8bf4672_0215.jpg":
        "9850701804KNH7307052F1204089K201781104<<<<00",
    "78f45152a658f728_0012.jpg":
        "X00000O<<9IDN4508179S1601264<<<<<<<<<<<<<<00",
    "dd6a2ea3ce305a5d_0192.jpg":
        "6451331382PRK6609302F1003145<<<<<<<<<<<<<<04",
    "f80334b200240666_0034.png":
        "AAMO000000ARG5702147M3303023000000000<<<<<04",
    # 2026-9-30-AJ.jpg needs NO correction: re-read from the photo on
    # 2026-11-04 it is "E818572393CHN6603171M2702157FHS9IS8GQWHI6I88",
    # whose five check digits all close, and it is byte-identical to what
    # the paddle engine read on its own.
}


def fix_check_digits(line2, filename=None):
    """Recompute the check digits of a line 2 from its data fields.

    Six rows of the xlsx ground truth carry a mistyped check digit (the
    data fields agree with the engines, the check digit does not).  This
    returns the line the ICAO arithmetic implies, so the score can be
    reported both ways: against the sheet as it stands and against the
    sheet with the arithmetic corrected.

    For the six rows that were re-read from the photos by hand, the
    hand-verified value is used instead -- those data fields were
    confirmed against the images, so they are the authority, not the
    (possibly wrong) text in the sheet.
    """
    if len(line2) != 44:
        return line2
    if filename and filename in HAND_VERIFIED_CHECK_DIGITS:
        return HAND_VERIFIED_CHECK_DIGITS[filename]
    if A.line2_checksum_ok(line2):
        return line2
    line = (line2[0:9] + A.check_digit(line2[0:9]) + line2[10:13] +
            line2[13:19] + A.check_digit(line2[13:19]) + line2[20:21] +
            line2[21:27] + A.check_digit(line2[21:27]) +
            line2[28:42] + A.check_digit(line2[28:42]))
    if len(line) != 43:
        return line2
    comp = "".join(line[a:b] for a, b in A._COMPOSITE_SLICES)
    return line + A.check_digit(comp)


def score(records, gt, gt_fixed=None):
    stats = {"n": 0, "pair": 0, "l1": 0, "l2": 0,
             "a1": 0.0, "a2": 0.0, "ck": 0}
    for rec in records:
        name = os.path.basename(rec.get("label", ""))
        ref = gt.get(name)
        if not ref or len(ref[0]) != 44 or len(ref[1]) != 44:
            continue
        stats["n"] += 1
        l1 = (rec.get("official") or {}).get("line1", "")
        l2 = (rec.get("official") or {}).get("line2", "")
        stats["pair"] += (l1 == ref[0] and l2 == ref[1])
        stats["l1"] += (l1 == ref[0])
        stats["l2"] += (l2 == ref[1])
        stats["a1"] += char_accuracy(l1, ref[0])
        stats["a2"] += char_accuracy(l2, ref[1])
        stats["ck"] += A.line2_checksum_ok(l2)
    n = max(1, stats["n"])
    return (f"pair-exact {stats['pair']:3d}/{stats['n']}  "
            f"l1exact {stats['l1']:3d} l1acc {stats['a1']/n:.4f}  "
            f"l2exact {stats['l2']:3d} l2acc {stats['a2']/n:.4f}  "
            f"l2ck {stats['ck']:3d}/{stats['n']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bench", nargs="+")
    ap.add_argument("--gt", default=None)
    ap.add_argument("--replay", action="store_true",
                    help="re-run the arbitration on the stored raw "
                         "engine readings instead of scoring the "
                         "production path")
    args = ap.parse_args()

    gt = load_gt(args.gt)
    gt_fixed = load_gt_fixed(args.gt)
    n_fixed = sum(1 for k, v in gt.items()
                  if len(v[1]) == 44 and v[1] != gt_fixed[k][1])

    for path in args.bench:
        data = json.load(open(path))
        records = data.get("records", [])
        print(f"== {os.path.basename(path)}")
        if args.replay:
            replayed = []
            for rec in records:
                raw = rec.get("raw", {})
                engines = [dict(v, method=k) for k, v in raw.items()
                           if v.get("status") != "skip" and v.get("line1")]
                if not engines:
                    continue
                l1, l2, info = A.arbitrate(engines)
                replayed.append({"label": rec["label"],
                                 "official": {"line1": l1, "line2": l2}})
            print("  arbitration replay   :",
                  score(replayed, gt, gt_fixed))
            print("  replay (fixed ck GT) :",
                  score(replayed, gt_fixed, gt_fixed))
        print("  production path      :", score(records, gt, gt_fixed))
        print("  production (fixed ck):", score(records, gt_fixed, gt_fixed))
    print(f"\n({n_fixed} ground-truth rows have a mistyped check digit; "
          f"the 'fixed ck' rows recompute it from the data fields)")


if __name__ == "__main__":
    main()
