#!/usr/bin/env python3
"""Closed-loop training orchestration for MRZ photo OCR.

Pipeline (automated test -> GT review -> human fix -> train -> test):
  1. test      : python3 tests/photo_direct_bench.py --json <bench>
  2. review    : python3 tests/mrz_gt_review.py --bench <bench> --gt <gt> --out <review.xlsx>
  3. fixcheck  : python3 tests/check_photo_auto_gt.py <gt>  (rc=1 -> stop for human fix)
  4. export    : python3 6_mrz_ocr/tools/export_photo_train.py --gt <gt> --out-dir ... --min-tier B --dispute-max 20
  5. train cnn : python3 6_mrz_ocr/tools/train_cnn_mixed.py ...
  6. train crnn: <crnn-env>/python 6_mrz_ocr/tools/train_crnn_seq.py ...
  7. apply+rebuild : cp weights -> src/, cmake --build 6_mrz_ocr/build -j8
  8. re-test   : back to step 1

Run only selected steps with --steps 1,3,4 style flags (default: all).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GT = ROOT / "10_ocr_research" / "archive" / "photo_auto_gt_all.json"
REPO = ROOT
STEPS = [1, 2, 3, 4, 5, 6, 7, 8]
CRNN_ENV = Path.home() / ".workbuddy-ai" / "binaries" / "python" / "envs" / "crnn" / "bin" / "python"


def run(cmd, *args, **kw):
    print(f"$ {cmd}", flush=True)
    res = subprocess.run(cmd, *args, **kw)
    if res.returncode != 0 and kw.get("check") is not False:
        sys.exit(res.returncode)
    return res


def step_test(bench, gt):
    run([
        sys.executable, str(ROOT / "tests" / "photo_direct_bench.py"),
        "--json", str(bench),
    ])


def step_review(bench, gt, review):
    run([
        sys.executable, str(ROOT / "tests" / "mrz_gt_review.py"),
        "--bench", str(bench),
        "--gt", str(gt),
        "--out", str(review),
    ])


def step_fixcheck(gt):
    res = run([
        sys.executable, str(ROOT / "tests" / "check_photo_auto_gt.py"), str(gt),
    ], check=False)
    if res.returncode != 0:
        print("FIX-STOP: check_photo_auto_gt reported issues; edit " + str(gt) +
              " then re-run this orchestrator (or --steps 3,4,5,6,7,8).",
              flush=True)
        sys.exit(2)


def step_export(gt, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    run([
        sys.executable, str(ROOT / "6_mrz_ocr" / "tools" / "export_photo_train.py"),
        "--gt", str(gt),
        "--out-dir", str(out_dir),
        "--min-tier", "B",
        "--dispute-max", "20",
    ])


def step_train_cnn(out_dir):
    run([
        sys.executable, str(ROOT / "6_mrz_ocr" / "tools" / "train_cnn_mixed.py"),
        "--epochs", "40",
        "--n-per-class", "1500",
        "--hard-extra", "750",
        "--real-npz", str(out_dir / "cnn_slices.npz"),
        "--real-frac", "0.50",
        "--out", "/tmp/cnn_loop.h",
    ])


def step_train_crnn(out_dir, synth):
    if not CRNN_ENV.exists():
        raise SystemExit(f"CRNN env missing: {CRNN_ENV}")
    if not synth.exists():
        run([
            str(CRNN_ENV), str(ROOT / "6_mrz_ocr" / "tools" / "export_crnn_features.py"),
            "--corpus", str(ROOT / "6_mrz_ocr" / "data" / "corpus_train"),
            "--tool", str(ROOT / "6_mrz_ocr" / "build" / "mrz_ocr_cnn_tool"),
            "--out", str(synth),
            "--crnn", "1", "--parallel", "0", "--line-models", "1",
        ])
    run([
        str(CRNN_ENV), str(ROOT / "6_mrz_ocr" / "tools" / "train_crnn_seq.py"),
        "--train", str(synth),
        "--extra", str(out_dir / "crnn_feats.npz"),
        "--extra-repeat", "60",
        "--out", "/tmp/crnn_loop.h",
        "--epochs", "60",
        "--arch", "bi",
        "--weight-decay", "0.01",
    ])


def step_apply():
    src = ROOT / "6_mrz_ocr" / "src"
    for target in ("cnn_weights.h", "crnn_weights.h"):
        b = src / target
        if b.exists():
            shutil.copy2(b, b.with_suffix(".h.bak-loop"))
        shutil.copy2(Path("/tmp") / target.replace("_weights.h", "_loop.h"), b)
    run(["cmake", "--build", str(ROOT / "6_mrz_ocr" / "build"), "-j8"])


def main():
    p = argparse.ArgumentParser(description="MRZ closed-loop train/test orchestrator")
    p.add_argument("--steps", default="1,2,3,4,5,6,7,8",
                   help="comma-separated step numbers to run (default all)")
    p.add_argument("--bench", default=str(ROOT / "10_ocr_research" / "photo_direct_bench_loop.json"))
    p.add_argument("--gt", default=str(DEFAULT_GT))
    p.add_argument("--review", default=str(ROOT / "10_ocr_research" / "gt_review_loop.xlsx"))
    p.add_argument("--out-dir", default=str(ROOT / "6_mrz_ocr" / "data" / "photo_train"))
    p.add_argument("--synth", default="/tmp/synth_crnn_feats.npz")
    a = p.parse_args()

    bench = Path(a.bench)
    gt = Path(a.gt)
    review = Path(a.review)
    out_dir = Path(a.out_dir)
    synth = Path(a.synth)

    steps = {int(s) for s in a.steps.split(",") if s.strip()}

    if 1 in steps:
        step_test(bench, gt)
    if 2 in steps:
        step_review(bench, gt, review)
    if 3 in steps:
        step_fixcheck(gt)
    if 4 in steps:
        step_export(gt, out_dir)
    if 5 in steps:
        step_train_cnn(out_dir)
    if 6 in steps:
        step_train_crnn(out_dir, synth)
    if 7 in steps:
        step_apply()
    print("done")


if __name__ == "__main__":
    main()