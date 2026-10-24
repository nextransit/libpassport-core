#!/usr/bin/env python3
"""Grouped cross-validation of the MRZ recognition heads.

WHY THIS SCRIPT EXISTS
----------------------
Every published "CRNN accuracy" number in this repository was measured
on photos that were ALSO in the training set.  The real corpus has 41
photos from 3 capture sessions; the shipped sequence head was trained on
all of them, so a number measured on them is memorisation, not
generalisation.

This script does the only honest measurement available with 41 photos:
it trains on 2 capture sessions and evaluates on the third, so every
evaluated photo was never seen by the model that scored it.  The three
folds are pooled.

The band geometry used for training is the SAME band the evaluation
feeds (the board's TextBox crop with 10 px of padding), because the
training corpus used to be cut from the ISOLATED MRZ rows, which is a
different geometry from what the recogniser sees at inference time.

Usage:
    python3 tools/cv_crnn.py [--epochs 60] [--synth-subsample 1500]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OMTEST = ROOT.parent / "tmp" / "omtest"
WORK = Path("/tmp/crnn_build")
PY = Path.home() / ".workbuddy-ai/binaries/python/envs/crnn/bin/python"

W = (7, 3, 1)


def val(c: str) -> int:
    if c == "<":
        return 0
    if c.isdigit():
        return int(c)
    return ord(c) - 55


def cd(s: str) -> int:
    return sum(val(c) * W[i % 3] for i, c in enumerate(s)) % 10


def l2_ok(s: str) -> bool:
    if len(s) != 44:
        return False
    for pos, (a, b) in ((9, (0, 9)), (19, (13, 19)), (27, (21, 27)),
                        (42, (28, 42))):
        if s[pos] != str(cd(s[a:b])):
            return False
    return s[43] == str(cd(s[0:10] + s[13:20] + s[21:28] + s[28:43]))


def decode(ppm: Path, env_extra: dict):
    env = dict(os.environ)
    env["MRZ_OCR_LINE_MODELS"] = "1"
    env["MRZ_OCR_PAR"] = "1"
    env["MRZ_OCR_CRNN"] = "0"
    env.update(env_extra)
    p = subprocess.run([str(ROOT / "build" / "mrz_ocr_cnn_tool"), str(ppm)],
                       capture_output=True, text=True, timeout=60, env=env)
    kv = {}
    for ln in p.stdout.splitlines():
        if ":" in ln:
            k, _, v = ln.partition(":")
            kv[k.strip()] = v.strip()
    return kv.get("result.line1", ""), kv.get("result.line2", "")


def parse_board_log(txt: Path):
    """The board's own reading of the photo (an independent recogniser)."""
    t = txt.read_text(errors="replace")
    ls = {}
    for m in re.finditer(r"^textLine\[(\d+)\]\((.*?)\)\s*$", t, re.M):
        ls[int(m.group(1))] = m.group(2).strip()
    mf = re.search(r"-----MRZ fixed \(fixmrz\)-----\s*\n(.+)\n(.+)", t)
    fix = (mf.group(1).strip(), mf.group(2).strip()) if mf else None
    return ls, fix


def build_corpus(out: Path):
    """Cut the evaluation band for every photo, with the board's label."""
    from PIL import Image
    import cv2
    out.mkdir(parents=True, exist_ok=True)
    records = []
    for txt in sorted(OMTEST.glob("*/*.jpg-result.txt")):
        jpg = txt.parent / txt.name.replace("-result.txt", "")
        if not jpg.exists():
            continue
        ls, fix = parse_board_log(txt)
        l1 = fix[0] if fix else ls.get(0, "")
        l2 = fix[1] if fix else ls.get(1, "")
        if len(l1) != 44 or len(l2) != 44:
            continue
        pts = [(int(a), int(b)) for a, b in
               re.findall(r"\[x: (\d+), y: (\d+)\]",
                          txt.read_text(errors="replace"))]
        boxes = [pts[i:i + 4] for i in range(0, len(pts), 4)]
        if len(boxes) < 2:
            continue
        rgb = cv2.imread(str(jpg))
        if rgb is None:
            continue
        H, Wd = rgb.shape[:2]
        cy = H - (H // 3) - 50
        y1 = max(0, min(p[1] for p in boxes[0]) + cy)
        y2 = min(H, max(p[1] for p in boxes[1]) + cy)
        band = rgb[max(0, y1 - 10):min(H, y2 + 1 + 10), 0:Wd]
        if band.shape[0] < 20:
            continue
        name = f"c{len(records):04d}.ppm"
        Image.fromarray(cv2.cvtColor(band, cv2.COLOR_BGR2RGB)).save(
            str(out / name), format="PPM")
        records.append({
            "id": f"{jpg.parent.name}/{jpg.stem}", "image": name,
            "line1": l1, "line2": l2, "profile": "real",
            "scale": 0, "hgap": 0, "skew": 0.0, "noise": 0.0, "rot": 0.0,
        })
    (out / "corpus.json").write_text(json.dumps({"records": records}, indent=1))
    return records


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--synth-subsample", type=int, default=1500)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--real-repeat", type=int, default=20)
    args = ap.parse_args()

    corpus_dir = Path("/tmp/cv_corpus")
    records = build_corpus(corpus_dir)
    groups = {}
    for r in records:
        groups.setdefault(r["id"].split("/")[0], []).append(r["id"])
    print("groups:", {k: len(v) for k, v in groups.items()})

    backup = ROOT / "src" / "crnn_weights.h"
    saved = Path("/tmp/crnn_weights_cv_backup.h")
    shutil.copy(backup, saved)
    fuse_env = {"MRZ_OCR_FUSE": "1"}
    results = {}
    try:
        for hold in groups:
            train_ids = [i for g, v in groups.items() if g != hold for i in v]
            test_ids = groups[hold]
            print(f"\n=== fold: hold out {hold} "
                  f"(train {len(train_ids)}, test {len(test_ids)}) ===", flush=True)
            # export the features of the TRAINING photos only
            subprocess.run([str(PY), "tools/export_crnn_features.py",
                            "--corpus", str(corpus_dir),
                            "--out", "/tmp/cv_feats.npz", "--crnn", "1",
                            "--parallel", "0"], cwd=ROOT, capture_output=True)
            d = np.load("/tmp/cv_feats.npz", allow_pickle=True)
            ids = np.array([str(i) for i in d["ids"]])
            m = np.isin(ids, train_ids)
            np.savez_compressed("/tmp/cv_feats_sub.npz",
                                features=d["features"][m], labels=d["labels"][m],
                                ids=ids[m])
            # train
            cmd = [str(PY), "tools/train_crnn_seq.py",
                   "--train", str(WORK / "corpus_train.npz"),
                   "--out", "/tmp/cv_w.h", "--epochs", str(args.epochs),
                   "--synth-subsample", str(args.synth_subsample),
                   "--weight-decay", str(args.weight_decay),
                   "--extra", "/tmp/cv_feats_sub.npz",
                   "--extra-repeat", str(args.real_repeat)]
            r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
            if r.returncode:
                print(r.stderr[-800:])
                return 1
            shutil.copy("/tmp/cv_w.h", backup)
            subprocess.run(["cmake", "--build", "build", "-j8"], cwd=ROOT,
                           capture_output=True)
            by_id = {r["id"]: r for r in records}
            for label, env in (("CNN", {}),
                               ("CRNN", {"MRZ_OCR_CRNN": "1"}),
                               ("FUSE", fuse_env)):
                n = ex = ck = ckw = 0
                ch = 0.0
                for i in test_ids:
                    rec = by_id[i]
                    _, l2 = decode(corpus_dir / rec["image"], env)
                    if not l2:
                        continue
                    n += 1
                    gt = rec["line2"]
                    if l2 == gt:
                        ex += 1
                    ch += sum(1 for k in range(44)
                              if k < len(l2) and l2[k] == gt[k]) / 44
                    if l2_ok(l2):
                        ck += 1
                        if l2 != gt:
                            ckw += 1
                print(f"   {label:5} exact={ex}/{n} char={ch / n * 100 if n else 0:6.2f}% "
                      f"cksum={ck}/{n} ckw={ckw}/{n}")
                results.setdefault(label, []).append(
                    (n, ex, ch, ck, ckw))
    finally:
        shutil.copy(saved, backup)
        subprocess.run(["cmake", "--build", "build", "-j8"], cwd=ROOT,
                       capture_output=True)

    print("\n=== POOLED (grouped CV) ===")
    for label, rs in results.items():
        N = sum(x[0] for x in rs)
        E = sum(x[1] for x in rs)
        C = sum(x[2] for x in rs)
        K = sum(x[3] for x in rs)
        W_ = sum(x[4] for x in rs)
        ch = sum(x[2] / x[0] * x[0] for x in rs) / N if N else 0
        print(f"  {label:5} exact={E}/{N} ({E / N * 100:.1f}%)  char={ch:.2f}%  "
              f"cksum={K}/{N}  ckw={W_}/{N}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
