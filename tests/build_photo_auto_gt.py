#!/usr/bin/env python3
"""Generate auto-GT for every passport photo via the multi-engine consensus
labeller (6_mrz_ocr/tools/build_gt_multiengine.py), then merge the frozen
records into tests/photo_auto_gt.json.

The labeller freezes line 2 only when its five ICAO check digits close AND
both reader families (our C conv pipeline + tesseract OCR-B) agree; line 1
freezes when no column is disputed across the two families.  Tier C records
carry no trustable GT; we still store every record that produced a non-empty
l1/l2 so a human can review and fix them afterwards (records without a located
band are stored with empty l1/l2 and tier "no_band").

Output:
  tests/photo_auto_gt.json
    { "<basename>": { "label": <input label>, "l1": str, "l2": str,
                      "tier": "A"|"B"|"C"|..., "l2_tier": ..., "l1_tier": ...,
                      "l2_votes": int, "l2_families": [..],
                      "l1_dispute_n": int } , ... }
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "6_mrz_ocr" / "tools"))

import gui  # noqa: E402   iter_passport_photos / PASSPORT_PHOTO_DIRS / ROOT

BUILDER = ROOT / "6_mrz_ocr" / "tools" / "build_gt_multiengine.py"
OUT = ROOT / "tests" / "photo_auto_gt.json"

EXT = (".jpg", ".jpeg", ".png")


def collect(max_items: int | None) -> list[tuple[str, Path]]:
    items = list(gui.iter_passport_photos())
    if max_items:
        items = items[:max_items]
    return [(label, Path(path)) for label, path in items]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--keep-tmp", action="store_true")
    args = ap.parse_args()

    items = collect(args.limit)
    tmp = Path(tempfile.mkdtemp(prefix="auto_gt_"))
    try:
        for idx, (label, path) in enumerate(items):
            src = tmp / f"auto{idx:04d}_{path.name}"
            try:
                os.symlink(os.fspath(path.resolve()), os.fspath(src))
            except FileExistsError:
                pass
        json_out = tmp / "gt_multiengine_photos.json"
        p = subprocess.run(
            [sys.executable, str(BUILDER), "--dir", str(tmp),
             "--out", str(json_out)],
            capture_output=True, text=True, timeout=1800)
        if p.returncode != 0:
            print("builder failed:\n", p.stdout, p.stderr, file=sys.stderr)
            return 1
        print(p.stdout, flush=True)
        raw = json.loads(json_out.read_text())
    finally:
        if not args.keep_tmp:
            # symlinks are ordinary files later; clean the whole tree
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)

    records = raw.get("records", [])
    by_label: dict[str, dict] = {}
    have_record: set[str] = set()
    for rec in records:
        stem = rec.get("id", "")
        prefix, _, name = stem.partition("_")
        if not prefix.startswith("auto"):
            continue
        idx = int(prefix[4:])
        if idx >= len(items):
            continue
        label, _path = items[idx]
        base = os.path.basename(str(_path))
        have_record.add(base)
        l1, l2 = rec.get("l1", ""), rec.get("l2", "")
        entry = {
            "label": label,
            "l1": l1,
            "l2": l2,
            "tier": rec.get("tier"),
            "l2_tier": rec.get("l2_tier"),
            "l1_tier": rec.get("l1_tier"),
            "l2_votes": rec.get("l2_votes"),
            "l2_families": rec.get("l2_families"),
            "l1_dispute_n": rec.get("l1_dispute_n"),
        }
        # keep every record that produced a usable (non-empty) line pair;
        # no_band / GT-less photos stay with empty lines and their real tier
        if not (l1 and l2):
            entry["l1_frozen"] = bool(rec.get("l1_frozen"))
            entry["l2_frozen"] = bool(rec.get("l2_frozen"))
        by_label[base] = entry

    # photos the labeller never saw get a placeholder row so the coverage
    # table is complete for all 115 inputs
    for _label, _path in items:
        base = os.path.basename(str(_path))
        if base not in by_label and base not in have_record:
            by_label[base] = {
                "label": _label, "l1": "", "l2": "",
                "tier": "no_record",
                "l1_dispute_n": None,
            }

    Path(args.out).write_text(
        json.dumps({"_comment": "auto-GT by multi-engine consensus "
                   "(build_gt_multiengine.py); ALL records stored incl. "
                   "tier C / no_band, ready for human review & fix",
                    "gt": by_label}, indent=1, ensure_ascii=False))
    tiers = {}
    for r in records:
        tiers[r.get("tier")] = tiers.get(r.get("tier"), 0) + 1
    print(f"builder tiers: {tiers}")
    print(f"written {args.out}: {len(by_label)} of {len(items)} photos "
          f"(all records incl. tier C kept for manual review)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())