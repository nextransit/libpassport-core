#!/usr/bin/env python3
"""Run OCR on every image in the corpus and report per-record and
overall accuracy.

For each entry we:
  1. Run mrz_ocr_tool on the image.
  2. Compare the recognised line1 / line2 to the ground truth.
  3. Count correct characters at every position.
  4. Tolerate small variations: lines may be slightly shorter or
     longer than 44 chars; we pad / truncate to match. Position-level
     comparison.
"""
from __future__ import annotations
import json, os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OCR  = ROOT / "build" / "mrz_ocr_tool"
DATA = ROOT / "data" / "corpus"
CORPUS = DATA / "corpus.json"

def recognize(p: Path) -> dict:
    """Parse mrz_ocr_tool stdout into a dict."""
    r = subprocess.run([str(OCR), str(p)], text=True, capture_output=True,
                       timeout=10)
    out = {}
    for line in r.stdout.splitlines():
        if ":" not in line:
            continue
        k, _, v = line.partition(":")
        out[k.strip()] = v.strip()
    return out

def compare(gt: str, got: str) -> tuple[int, int]:
    """Return (correct_chars, total_chars) compared position by position,
    clipping to the shorter length."""
    n = min(len(gt), len(got))
    correct = sum(1 for i in range(n) if gt[i] == got[i])
    return correct, len(gt)

def main():
    if not OCR.exists():
        print("mrz_ocr_tool not built", file=sys.stderr)
        return 1
    corpus = json.loads(CORPUS.read_text())
    total_l1_correct = total_l1_chars = 0
    total_l2_correct = total_l2_chars = 0
    failed = 0
    n_total = len(corpus["records"])
    n_full_match = 0
    for rec in corpus["records"]:
        p = DATA / rec["image"]
        result = recognize(p)
        if result.get("result.ok") != "OK":
            failed += 1
            continue
        l1 = result.get("result.line1", "")
        l2 = result.get("result.line2", "")
        c1, n1 = compare(rec["line1"], l1)
        c2, n2 = compare(rec["line2"], l2)
        total_l1_correct += c1
        total_l1_chars += n1
        total_l2_correct += c2
        total_l2_chars += n2
        if c1 == n1 and c2 == n2:
            n_full_match += 1
    print(f"=== OCR corpus results ===")
    print(f"records          : {n_total}")
    print(f"pipeline failures: {failed}")
    print(f"full matches     : {n_full_match}")
    print(f"line1 accuracy   : {total_l1_correct}/{total_l1_chars} "
          f"= {100.0*total_l1_correct/total_l1_chars:.1f}%")
    print(f"line2 accuracy   : {total_l2_correct}/{total_l2_chars} "
          f"= {100.0*total_l2_correct/total_l2_chars:.1f}%")
    return 0

if __name__ == "__main__":
    sys.exit(main())
