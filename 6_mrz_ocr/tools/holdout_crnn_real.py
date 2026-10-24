#!/usr/bin/env python3
"""Held-out test: is CRNN's real-photo accuracy generalisation or memorisation?

The shipped weights are trained on corpus_real, and every published
real-photo number is measured on that same corpus -- there is no
held-out split anywhere in rebuild_crnn.sh. So "CRNN 87%" could be
either real generalisation or 35 memorised images.

This script splits the real features BY PHOTO GROUP and trains two
disjoint variants:

  C: synthetic + pic_scan1      -> evaluated on pic_scan2/3  (unseen)
  D: synthetic + pic_scan2/3    -> evaluated on pic_scan1    (unseen)

Comparing C/D against the pure-synthetic baseline (A) and the
train-on-everything baseline (B) separates the two effects.

Usage:
  /opt/homebrew/bin/python3 tools/holdout_crnn_real.py
"""
from __future__ import annotations
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "build" / "mrz_ocr_cnn_tool"
OMTEST = ROOT.parent / "tmp" / "omtest"
WORK = Path("/tmp/crnn_build")
PY = Path.home() / ".workbuddy-ai/binaries/python/envs/crnn/bin/python"
sys.path.insert(0, str(ROOT / "tools"))
from cmp_crop_crnn import parse_result, l2_ok  # noqa: E402


def build():
    subprocess.run(["cmake", "--build", "build", "-j8"], cwd=ROOT,
                   check=True, capture_output=True)


def train(train_npz: Path, extra: Path | None, out: Path, repeat: int = 20):
    cmd = [str(PY), "tools/train_crnn_seq.py", "--train", str(train_npz),
           "--out", str(out), "--epochs", "60"]
    if extra:
        cmd += ["--extra", str(extra), "--extra-repeat", str(repeat)]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    print("   ", r.stdout.strip().splitlines()[-1] if r.stdout else r.stderr[-200:])


def run(img: Path, crnn: int):
    env = dict(os.environ)
    env["MRZ_OCR_LINE_MODELS"] = "1"
    env["MRZ_OCR_PAR"] = "0"
    env["MRZ_OCR_CRNN"] = "1" if crnn else "0"
    p = subprocess.run([str(TOOL), str(img)], capture_output=True, text=True,
                       timeout=60, env=env)
    kv = {}
    for ln in p.stdout.splitlines():
        if ":" in ln:
            k, _, v = ln.partition(":")
            kv[k.strip()] = v.strip()
    return kv.get("result.line2", "")


def carr(a: str, b: str) -> float:
    n = max(len(a), len(b))
    return (sum(1 for i in range(n)
                if (a[i] if i < len(a) else "_") == (b[i] if i < len(b) else "_")) / n) if n else 0.0


def main() -> int:
    d = np.load(WORK / "corpus_real.npz", allow_pickle=True)
    feats, labels, ids = d["features"], d["labels"], d["ids"]
    g1 = np.array([i.startswith("pic_scan1/") for i in ids])
    print(f"real features n={len(ids)}  scan1={g1.sum()}  scan2/3={(~g1).sum()}")

    def save(mask, path):
        np.savez_compressed(path, features=feats[mask], labels=labels[mask], ids=ids[mask])

    save(g1, Path("/tmp/real_scan1.npz"))
    save(~g1, Path("/tmp/real_scan23.npz"))

    # ---- evaluation set: board bbox crop, pad 12 (mid of the sweep) ----
    tmp = Path(tempfile.mkdtemp(prefix="hold_"))
    samples = []
    for dd in sorted(p for p in OMTEST.iterdir() if p.is_dir()):
        for txt in sorted(dd.glob("*.jpg-result.txt")):
            img = txt.with_name(txt.name.replace("-result.txt", ""))
            if not img.exists():
                continue
            l1, l2, boxes = parse_result(txt)
            if not l1 or not l2 or len(l1) != 44 or len(l2) != 44 or len(boxes) < 2:
                continue
            rgb = cv2.imread(str(img))
            H, W = rgb.shape[:2]
            cy = H - (H // 3) - 50
            y1 = min(p[1] for p in boxes[0]) + cy
            y2 = max(p[1] for p in boxes[1]) + cy
            pad = 12
            band = rgb[max(0, y1 - pad):min(H, y2 + 1 + pad), 0:W]
            ppm = tmp / (dd.name + "_" + img.stem + ".ppm")
            Image.fromarray(cv2.cvtColor(band, cv2.COLOR_BGR2RGB)).save(str(ppm), format="PPM")
            samples.append((f"{dd.name}/{img.stem}", ppm, l2))
    print(f"eval samples={len(samples)}")

    variants = [
        ("A pure-synth", None),
        ("B all-real x20", WORK / "corpus_real.npz"),
        ("C scan1-only x20", Path("/tmp/real_scan1.npz")),
        ("D scan23-only x20", Path("/tmp/real_scan23.npz")),
    ]

    backup = Path("/tmp/crnn_weights_B_repeat20.h")
    results = {}
    for name, extra in variants:
        print(f"\n== {name}")
        train(WORK / "corpus_train.npz", extra, Path("/tmp/variant.h"))
        shutil.copy("/tmp/variant.h", ROOT / "src/crnn_weights.h")
        build()
        rows = []
        for sid, ppm, gt in samples:
            out = run(ppm, 1)
            rows.append((sid, gt, out))
        results[name] = rows
        for group, pred in (("scan1", lambda s: s.startswith("pic_scan1/")),
                            ("scan2/3", lambda s: not s.startswith("pic_scan1/"))):
            sub = [r for r in rows if pred(r[0])]
            a = sum(carr(g, o) for _, g, o in sub) / len(sub)
            ex = sum(1 for _, g, o in sub if g == o)
            print(f"   {group:8} n={len(sub):2d} l2_acc={a*100:6.2f}% exact={ex:2d}/{len(sub)}")

    shutil.copy(backup, ROOT / "src/crnn_weights.h")
    build()
    print("\nrestored shipped weights")

    # ---- cross table: which variant wins where ----
    print("\n-- held-out comparison (the honest numbers) --")
    for name in ("A pure-synth", "B all-real x20", "C scan1-only x20", "D scan23-only x20"):
        rows = results[name]
        for group, pred in (("scan1 (seen by C)", lambda s: s.startswith("pic_scan1/")),
                            ("scan2/3 (seen by D)", lambda s: not s.startswith("pic_scan1/"))):
            sub = [r for r in rows if pred(r[0])]
            a = sum(carr(g, o) for _, g, o in sub) / len(sub)
            print(f"  {name:18} on {group:20} l2_acc={a*100:6.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
