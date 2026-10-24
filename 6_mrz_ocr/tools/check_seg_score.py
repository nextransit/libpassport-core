#!/usr/bin/env python3
"""Check whether the pitch-grid alignment score predicts recognition quality.

Runs the native recogniser over a corpus with MRZ_OCR_SEGDBG=1, pairs each
sample's segmentation score with the character accuracy it achieved, and
reports accuracy for samples above and below a candidate threshold.

This is the evidence needed to justify the Android-side gate that switches
between the CRNN sequence head and the per-character CNN.
"""
from __future__ import annotations
import argparse
import json
import os
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/corpus_real")
    ap.add_argument("--tool", default="build/mrz_ocr_cnn_tool")
    ap.add_argument("--threshold", type=int, default=70)
    ap.add_argument("--crnn", type=int, default=1)
    args = ap.parse_args()

    corpus = ROOT / args.corpus
    records = json.loads((corpus / "corpus.json").read_text())["records"]
    env = dict(os.environ)
    env["MRZ_OCR_LINE_MODELS"] = "1"
    env["MRZ_OCR_CRNN"] = "1" if args.crnn else "0"
    env["MRZ_OCR_PAR"] = "0"
    env["MRZ_OCR_SEGDBG"] = "1"

    rows = []
    for rec in records:
        p = subprocess.run([str(ROOT / args.tool), str(corpus / rec["image"])],
                           capture_output=True, text=True, timeout=30, env=env)
        scores = [int(s) for s in re.findall(r"score=(\d+)", p.stderr)]
        if not scores:
            continue
        m1 = re.search(r"result\.line1\s*:\s*(.*)", p.stdout)
        m2 = re.search(r"result\.line2\s*:\s*(.*)", p.stdout)
        rows.append({
            "seg": min(scores),
            "l1": m1.group(1).strip() if m1 else "",
            "l2": m2.group(1).strip() if m2 else "",
            "gt1": rec["line1"], "gt2": rec["line2"],
        })

    if not rows:
        print("no scored samples")
        return 1

    def acc(sub, got, gt):
        c = t = 0
        for r in sub:
            a, b = r[got], r[gt]
            for i in range(min(len(a), len(b))):
                t += 1
                c += 1 if a[i] == b[i] else 0
        return (c / t if t else 0.0)

    segs = [r["seg"] for r in rows]
    print(f"samples={len(rows)} seg min={min(segs)} max={max(segs)} "
          f"mean={sum(segs)/len(segs):.1f}")
    hi = [r for r in rows if r["seg"] >= args.threshold]
    lo = [r for r in rows if r["seg"] < args.threshold]
    for name, sub in (("seg>=%d" % args.threshold, hi), ("seg<%d" % args.threshold, lo)):
        if not sub:
            print(f"{name}: n=0")
            continue
        print(f"{name}: n={len(sub)} "
              f"l1_acc={acc(sub,'l1','gt1'):.3f} l2_acc={acc(sub,'l2','gt2'):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
