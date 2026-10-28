"""benchmark: re-run the locator on assets/pic, score against the
human-reviewed xlsx, and emit a fresh annotations_review_v2.xlsx.

Usage:
  PYTHONPATH=. python3 -m tests.passport_layout.benchmark

  This:
    1. enumerates assets/pic/*.jpg|*.png
    2. runs tests.passport_pipeline.locate(image) (current pipeline)
    3. scores each image against the human verdict from the xlsx
       (missing / wrong / OK) using the failure categories
    4. writes docs/passport_layout/benchmark_v1.csv with per-image
       per-region status. No boxes are drawn -- the human still has
       to judge.

The score is qualitative (matches the human coarse notes), not
IoU. IoU would need GT boxes that the existing xlsx does not provide.
"""
from __future__ import annotations
import sys
import time
from collections import Counter
from pathlib import Path

from openpyxl import load_workbook

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests"))

from passport_pipeline import locate  # type: ignore  # noqa: E402
from passport_layout.eval import parse_xlsx  # type: ignore  # noqa: E402

PIC_DIR = ROOT / "assets" / "pic"
REVIEW_XLSX = PIC_DIR / "annotations_review.xlsx"
OUT_CSV = ROOT / "docs" / "passport_layout" / "benchmark_v1.csv"
HEADER = ["fname", "region", "pred_x", "pred_y", "pred_w", "pred_h", "pred_conf", "human_missing", "human_note", "status"]
NL = chr(10)


def run_locator():
    out = {}
    for p in sorted(PIC_DIR.iterdir()):
        if p.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        if p.name in ("catalog.json", "corpus.json"):
            continue
        t0 = time.perf_counter()
        try:
            loc = locate(p)
        except Exception as e:
            out[p.name] = {"_error": str(e)}
            continue
        out[p.name] = {k: v for k, v in loc.items()} if loc else {}
        out[p.name]["_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    return out


def status_for(region, pred, verdict):
    p = pred.get(region)
    human_missing = region in verdict.missing
    note = verdict.notes.get(region, "") or ""
    if human_missing and not p:
        return ("agree_missing", "")
    if human_missing and p:
        return ("false_positive", "locator found " + region + " but human says missing")
    if not human_missing and not p:
        return ("missed", "locator did not find " + region)
    bad = ("第2行" in note) or ("覆盖" in note and region == "data") or ("位置错误" in note)
    return (("wrong_bbox" if bad else "agree"), note)


def main():
    verdicts = {v.fname: v for v in parse_xlsx(REVIEW_XLSX)}
    preds = run_locator()
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    counter = Counter()
    with OUT_CSV.open("w") as f:
        f.write(",".join(HEADER) + NL)
        for fname, verdict in sorted(verdicts.items()):
            pred = preds.get(fname, {})
            if "_error" in pred:
                err = pred.get("_error", "").replace(",", ";")
                for region in ("photo", "data", "mrz"):
                    row = [fname, region, "", "", "", "", "", "", "", "locator_error:" + err]
                    f.write(",".join(row) + NL)
                continue
            for region in ("photo", "data", "mrz"):
                status, note = status_for(region, pred, verdict)
                counter[(region, status)] += 1
                p = pred.get(region) or {}
                row = [
                    fname,
                    region,
                    str(p.get("x", "")),
                    str(p.get("y", "")),
                    str(p.get("w", "")),
                    str(p.get("h", "")),
                    str(p.get("confidence", "")),
                    str(int(region in verdict.missing)),
                    note.replace(NL, " "),
                    status,
                ]
                row = [v.replace(",", ";") for v in row]
                f.write(",".join(row) + NL)
    print("scored", len(verdicts), "images, locator timing in", OUT_CSV)
    print("per-region agreement:")
    for (region, status), n in sorted(counter.items()):
        print(" ", region.ljust(6), status.ljust(18), n)
    return 0


if __name__ == "__main__":
    sys.exit(main())