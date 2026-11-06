#!/usr/bin/env python3
"""Export classifier-input slices via the TOOL's dump path, labels by POSITION.

Why this exists (two failed exporters before it)
------------------------------------------------
1. `export_corpus_slices.py` aligns the tool's READING against the corpus
   text.  On the physical corpus the model cannot read the images, so the
   alignment drops nearly everything -- and bootstrapping labels from a
   model that cannot read the image is the wrong dependency anyway.
2. `export_corpus_slices_geom.py` cuts cells analytically from the band.
   The labels are exact, but the CELLS are raw grayscale crops -- while
   the classifier at inference consumes the tool's binarized/resampled
   patch (window width + weights + ink gate are one calibration,
   mrz_ocr_cnn.c:869-873).  Training on raw crops violates the §5
   train-infer contract: measured, a model trained on geometric slices
   scores 1.25% full-exact on the very corpus it trained on.

This exporter runs the tool with MRZ_OCR_DUMP so every emitted cell is
EXACTLY what inference feeds the network, and labels the cells by
POSITION: dumped cell i of line L is ground-truth character i -- no
dependence on the tool's classification.  An image is used only when the
segmenter emitted exactly as many cells as the ground truth has
characters (otherwise the segmentation, not the label, is wrong, and the
image is skipped and counted).

Usage:
  python3 tools/export_corpus_slices_pos.py --corpus data/corpus_train_phys \
      --out data/corpus_train_phys/pos_slices.npz [--tool ...]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

GI = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ<"


def parse_ok(stdout: str) -> bool:
    for ln in stdout.splitlines():
        if ln.strip().startswith("result.ok"):
            return "OK" in ln
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tool", default="build/mrz_ocr_cnn_tool")
    ap.add_argument("--dump-dir", default="/tmp/mrz_pos_dump")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    corpus = Path(args.corpus)
    if not corpus.is_absolute():
        corpus = Path.cwd() / corpus
    data = json.loads((corpus / "corpus.json").read_text())
    recs = data["records"]
    if args.limit:
        recs = recs[: args.limit]

    tool = Path(args.tool)
    if not tool.is_absolute():
        tool = Path.cwd() / tool
    dump_dir = Path(args.dump_dir)
    dump_dir.mkdir(parents=True, exist_ok=True)
    tag = "pos"

    env = dict(os.environ)
    env["MRZ_OCR_DUMP"] = tag
    env["MRZ_OCR_DUMP_DIR"] = str(dump_dir)
    env["MRZ_OCR_LINE_MODELS"] = "1"
    env["MRZ_OCR_CRNN"] = "0"
    env.pop("MRZ_OCR_FUSE", None)

    X, Y, META = [], [], []
    used = skipped_seg = skipped_noimg = 0
    per_line_used = Counter()
    for rec in recs:
        img = corpus / rec["image"]
        if not img.exists():
            skipped_noimg += 1
            continue
        for f in glob.glob(str(dump_dir / f"mrz_dump_{tag}_l*_c*.ppm")):
            os.remove(f)
        r = subprocess.run([str(tool), str(img)], capture_output=True,
                           text=True, timeout=60, env=env)
        # Positional labelling: for every line, the cell count must equal
        # the ground-truth length.  The tool's READING is irrelevant --
        # only its segmentation is trusted here.
        ok_lines = []
        for li, key in ((0, "line1"), (1, "line2")):
            truth = rec.get(key) or ""
            if not truth:
                ok_lines.append(None)
                continue
            cells = sorted(
                glob.glob(str(dump_dir / f"mrz_dump_{tag}_l{li}_c*.ppm")),
                key=lambda p: int(p.rsplit("_c", 1)[1].split("_")[0]))
            if len(cells) != len(truth):
                ok_lines.append(None)
                continue
            ok_lines.append((truth, cells))
        if all(v is None for v in ok_lines):
            skipped_seg += 1
            continue
        used += 1
        for li, ok in enumerate(ok_lines):
            if ok is None:
                continue
            truth, cells = ok
            for c, path in enumerate(cells):
                ch = truth[c]
                if ch not in GI:
                    continue
                a = np.asarray(Image.open(path).convert("L"),
                               dtype=np.float32) / 255.0
                if a.shape != (12, 16):
                    continue
                X.append(a.reshape(-1))
                Y.append(GI.index(ch))
                META.append({"file": rec["id"], "line": li, "col": c,
                             "ch": ch, "tier": "A", "weak": 0})
            per_line_used[li] += 1

    if not X:
        print("no slices exported", flush=True)
        return 1
    np.savez(args.out, X=np.asarray(X, dtype=np.float32),
             Y=np.asarray(Y, dtype=np.int64),
             meta=np.array([json.dumps(m) for m in META]))
    print(f"images: used={used} skipped_seg={skipped_seg} "
          f"noimg={skipped_noimg} | line0 imgs={per_line_used[0]} "
          f"line1 imgs={per_line_used[1]} | slices={len(Y)} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
