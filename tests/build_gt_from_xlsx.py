#!/usr/bin/env python3
"""Rebuild the official ground-truth JSON files from gt_review_loop.xlsx.

The xlsx is the single source of truth (AGENTS.md).  The *_official.json
files under tests/ are the machine-readable projection of it that the
evaluation tools consume (6_mrz_ocr/tools/eval_weight_sets.py).

Tiers are derived, not copied from the sheet, so that a stale or partially
corrected row can never silently count as trusted truth:

    l1_tier / l2_tier = A   line is 44 characters and structurally sane
                      = B   line is 44 characters but an ICAO 7-3-1 check
                            digit does not match -> the text is used, the
                            row is reported as suspect
                      = S   line is a complete CS/CD single line document
                            of 30 characters (birth and expiry check
                            digits verified) -> reported on its own, never
                            folded into the TD3 character accuracy
                      = C   line is not a 44 character TD3 line (a
                            truncated 43 character line, or the empty
                            second line of a CS/CD document) -> not counted
                            at all

Usage:
    python3 tests/build_gt_from_xlsx.py
    python3 tests/build_gt_from_xlsx.py --check     # diff only, no writes
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import openpyxl

ROOT = Path(__file__).resolve().parents[1]
XLSX = ROOT / "10_ocr_research" / "gt_review_loop.xlsx"

# directory -> output file
OUTPUTS = {
    "assets/pic": ROOT / "tests" / "gt_assets_pic_official.json",
    "tmp/PassPortPhoto": ROOT / "tests" / "gt_photo_passport_official.json",
}

WEIGHT = {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7,
          "8": 8, "9": 9, "A": 10, "B": 11, "C": 12, "D": 13, "E": 14,
          "F": 15, "G": 16, "H": 17, "I": 18, "J": 19, "K": 20, "L": 21,
          "M": 22, "N": 23, "O": 24, "P": 25, "Q": 26, "R": 27, "S": 28,
          "T": 29, "U": 30, "V": 31, "W": 32, "X": 33, "Y": 34, "Z": 35,
          "<": 0}


def check_digit(s: str) -> str:
    """ICAO 9303 7-3-1 weighted check digit."""
    total = 0
    for i, ch in enumerate(s):
        total += WEIGHT.get(ch, 0) * (7, 3, 1)[i % 3]
    return str(total % 10)


def line2_bad_checks(line2: str) -> list:
    """Names of the ICAO check digits of a TD3 line2 that do not match."""
    if len(line2) != 44:
        return ["length"]
    bad = []
    if check_digit(line2[0:9]) != line2[9]:
        bad.append("document_number")
    if check_digit(line2[13:19]) != line2[19]:
        bad.append("birth_date")
    if check_digit(line2[21:27]) != line2[27]:
        bad.append("expiry_date")
    if check_digit(line2[28:42]) != line2[42]:
        bad.append("personal_number")
    composite = line2[0:10] + line2[13:20] + line2[21:28] + line2[28:43]
    if check_digit(composite) != line2[43]:
        bad.append("composite")
    return bad


def line1_tier(line1: str) -> str:
    # TD3 line1 starts with the issuing state code.  The second character is
    # the document type: '<' for the ordinary form, but PM/PO/PD/PW are
    # legitimate variants, so only the length and the leading letter count.
    if len(line1) != 44:
        return "C"
    if not line1[0].isalpha():
        return "B"
    return "A"


def cscd_line_tier(line: str) -> str:
    """Tier of a CS/CD (Mainland travel permit) single machine readable row.

    Those documents carry one 30 character row instead of the two 44
    character TD3 rows:

        [0:3]   CSC / CDL          document family + country code
        [3:12]  document number     9 characters, no check digit
        [12]    '<'
        [13:19] date of birth       6 digits
        [19]    check digit          ICAO 7-3-1 over [13:19]
        [20]    '<'
        [21:27] date of expiry       6 digits
        [27]    check digit          ICAO 7-3-1 over [21:27]
        [28]    '<'
        [29]    composite check digit

    The birth and expiry check digits follow the usual ICAO 7-3-1 cycle
    (verified on every CS/CD sample in the sheet).  The trailing composite
    digit does not: no field set and weight cycle reproduces it across the
    samples, so it is carried verbatim but never treated as verified.

    Tier "S" marks such a row as a complete, structurally verified single
    line document.  It is not a TD3 line, so it is reported on its own and
    never folded into the TD3 character accuracy.
    """
    if len(line) != 30:
        return ""
    if line[0:2] not in ("CS", "CD"):
        return ""
    if line[12] != "<" or line[20] != "<" or line[28] != "<":
        return ""
    if any(ch not in WEIGHT for ch in line):
        return ""
    if check_digit(line[13:19]) != line[19]:
        return ""
    if check_digit(line[21:27]) != line[27]:
        return ""
    return "S"


def line2_tier(line2: str) -> str:
    if len(line2) != 44:
        return "C"
    bad = line2_bad_checks(line2)
    if not bad:
        return "A"
    if all(b in ("personal_number", "composite") for b in bad):
        # Optional data is filled with filler characters whose check digit
        # is not always computed by the issuing authority.
        return "B"
    return "B"


def build_label_index() -> dict:
    idx = {}
    for folder in ("assets/pic", "tmp/PassPortPhoto", "Photo"):
        for path in sorted(glob.glob(f"{ROOT}/{folder}/**/*.*", recursive=True)):
            if os.path.isfile(path):
                idx.setdefault(os.path.basename(path), os.path.relpath(path, ROOT))
    return idx


def read_sheet() -> list:
    wb = openpyxl.load_workbook(XLSX, data_only=True)
    ws = wb["GT对比"]
    rows = []
    for r in range(2, ws.max_row + 1):
        name = ws.cell(r, 2).value
        if not name:
            continue
        l1 = str(ws.cell(r, 7).value or "").strip()
        l2 = str(ws.cell(r, 8).value or "").strip()
        rows.append((str(name).strip(), l1, l2))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="only report what would change")
    args = ap.parse_args()

    labels = build_label_index()
    rows = read_sheet()

    buckets = {folder: {} for folder in OUTPUTS}
    missing = []
    for name, l1, l2 in rows:
        label = labels.get(name)
        if label is None:
            missing.append(name)
            continue
        for folder in OUTPUTS:
            if label.startswith(folder + "/"):
                buckets[folder][name] = {
                    "label": label,
                    "l1": l1,
                    "l2": l2,
                    "tier": "C",
                    "l1_tier": cscd_line_tier(l1) or line1_tier(l1),
                    "l2_tier": line2_tier(l2),
                }
                break
    for folder in OUTPUTS:
        for name, rec in buckets[folder].items():
            order = {"A": 0, "B": 1, "S": 2, "C": 3}
            if rec["l1_tier"] == "S":
                # Single line CS/CD document: the 30 character row is the
                # whole machine readable zone, there is no second line to
                # score, so the record keeps the single line tier.
                rec["tier"] = "S"
            else:
                rec["tier"] = max((rec["l1_tier"], rec["l2_tier"]),
                                  key=lambda t: order[t])

    if missing:
        print(f"warning: {len(missing)} sheet rows have no image on disk: "
              f"{', '.join(missing)}")

    changed = False
    for folder, out in OUTPUTS.items():
        gt = dict(sorted(buckets[folder].items()))
        payload = {
            "_comment": (f"Official ground truth rebuilt from "
                         f"10_ocr_research/gt_review_loop.xlsx (sheet "
                         f"GT对比) by tests/build_gt_from_xlsx.py. "
                         f"Source folder: {folder}/. Tiers are derived: A = "
                         f"44 char line with matching ICAO 7-3-1 check digits, "
                         f"B = 44 char line with a suspect check digit, "
                         f"S = complete CS/CD single line document of 30 "
                         f"characters (reported separately), C = not "
                         f"a 44 char TD3 line (excluded from the metrics)."),
            "gt": gt,
        }
        text = json.dumps(payload, ensure_ascii=False, indent=1) + "\n"
        if out.exists() and out.read_text() == text:
            print(f"{out.relative_to(ROOT)}: unchanged ({len(gt)} entries)")
            continue
        changed = True
        old = json.loads(out.read_text())["gt"] if out.exists() else {}
        added = [k for k in gt if k not in old]
        removed = [k for k in old if k not in gt]
        modified = [k for k in gt if k in old and old[k] != gt[k]]
        print(f"{out.relative_to(ROOT)}: {len(old)} -> {len(gt)} entries; "
              f"+{len(added)} added, -{len(removed)} removed, "
              f"{len(modified)} modified")
        for k in added:
            print(f"    + {k}  l2={gt[k]['l2_tier']}")
        for k in modified:
            o, n = old[k], gt[k]
            print(f"    ~ {k}")
            if o["l1"] != n["l1"]:
                print(f"        l1 {o['l1']}  ->  {n['l1']}")
            if o["l2"] != n["l2"]:
                print(f"        l2 {o['l2']}  ->  {n['l2']}")
        if args.check:
            continue
        out.write_text(text)
        print(f"    written")

    return 1 if (changed and args.check) else 0


if __name__ == "__main__":
    sys.exit(main())
