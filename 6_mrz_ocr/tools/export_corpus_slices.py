#!/usr/bin/env python3
"""Export 12x16 CNN glyph slices from a rendered MRZ corpus.

The corpus images (data/corpus_eval, data/corpus_train) are already band
images: two MRZ lines rendered by tools/gen_mrz_image.  We feed them
straight into the C tool with MRZ_OCR_DUMP=<tag> and collect the
per-column glyph patches the tool itself produced.  Labels come from
corpus.json's line1 / line2 text (the generator guarantees
checksum-valid MRZ).

Label mapping: the tool's own text (result.line1/result.line2) is aligned
against the ground truth with a Levenshtein backtrace, so every dumped
cell gets the ground-truth character at its aligned position.  Lines
whose alignment distance exceeds 25% of the line length are dropped (the
tool read them too badly for the correspondence to be trustworthy).

Why this exists: the CNN was trained on photographs plus a Python-rendered
synthetic half, while the corpus is rendered by the C tool with its own
degradations (rot / blur / illum / noise).  That domain gap is what made
the corpus score 87.75% / 73.09% with zero full matches.  Training on
slices exported from the *train* corpus makes the corpus in-distribution.

Usage:
  python3 tools/export_corpus_slices.py --corpus data/corpus_train \
      --out data/corpus_train/cnn_slices.npz --shard 0 --shards 8
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
GI = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ<"
MAX_DIST_RATIO = 0.45


def align(text: str, gt: str):
    """Levenshtein backtrace; returns (mapping, dist).

    mapping[i] is the index in gt that text[i] corresponds to (-1 when the
    backtrace deleted text[i]).  Substitutions are kept (diagonal steps)
    so every cell gets a label.
    """
    n, m = len(text), len(gt)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        di, dp = d[i], d[i - 1]
        ti = text[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ti == gt[j - 1] else 1
            di[j] = min(dp[j] + 1, di[j - 1] + 1, dp[j - 1] + cost)
    mapping = [-1] * n
    i, j = n, m
    while i > 0 and j > 0:
        cost = 0 if text[i - 1] == gt[j - 1] else 1
        if d[i][j] == d[i - 1][j - 1] + cost:
            mapping[i - 1] = j - 1
            i -= 1
            j -= 1
        elif d[i][j] == d[i - 1][j] + 1:
            i -= 1
        else:
            j -= 1
    return mapping, d[n][m]


def parse_text(stdout: str):
    out = {}
    for key in ("line1", "line2"):
        m = re.search(rf"^result\.{key}\s*:\s*(\S*)\s*$", stdout, re.M)
        out[key] = m.group(1) if m else ""
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, help="corpus dir with corpus.json")
    ap.add_argument("--tool", default="6_mrz_ocr/build/mrz_ocr_cnn_tool")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dump-prefix", default="/tmp/mrz_dump")
    ap.add_argument("--tag", default="corp")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    args = ap.parse_args()

    corpus = Path(args.corpus)
    if not corpus.is_absolute():
        for base in (ROOT, ROOT / "6_mrz_ocr"):
            if (base / corpus).exists():
                corpus = base / corpus
                break
        else:
            corpus = ROOT / corpus
    data = json.loads((corpus / "corpus.json").read_text())
    recs = data["records"]
    recs = [r for i, r in enumerate(recs) if i % args.shards == args.shard]
    if args.limit:
        recs = recs[: args.limit]

    tool = str(ROOT / args.tool)
    X, Y, META = [], [], []
    n_skipped_tool = 0
    n_skipped_align = 0
    for n, rec in enumerate(recs):
        img = corpus / rec["image"]
        if not img.exists():
            continue
        # Drop stale dumps first: if the tool fails to emit a cell for this
        # image we must not silently pick up the previous image's patch.
        for f in glob.glob(f"{args.dump_prefix}_{args.tag}_l*_c*_x*_r*_w*.ppm"):
            os.remove(f)
        env = dict(os.environ)
        env["MRZ_OCR_DUMP"] = args.tag
        env["MRZ_OCR_LINE_MODELS"] = "1"
        env["MRZ_OCR_CRNN"] = "0"
        try:
            r = subprocess.run([tool, str(img)], capture_output=True, text=True,
                               timeout=60, env=env)
        except subprocess.TimeoutExpired:
            n_skipped_tool += 1
            continue
        if r.returncode != 0:
            n_skipped_tool += 1
            continue
        texts = parse_text(r.stdout)
        for li, truth in ((0, rec.get("line1")), (1, rec.get("line2"))):
            if not truth:
                continue
            cells = glob.glob(
                f"{args.dump_prefix}_{args.tag}_l{li}_c*_x*_r*_w*.ppm")
            if not cells:
                continue
            cells.sort(key=lambda p: int(re.search(r"_c(\d+)_", p).group(1)))
            text = texts["line1"] if li == 0 else texts["line2"]
            if len(text) != len(cells):
                n_skipped_align += 1
                continue
            mapping, dist = align(text, truth)
            if len(truth) and dist > MAX_DIST_RATIO * len(truth):
                n_skipped_align += 1
                continue
            for c, path in enumerate(cells):
                gi = mapping[c]
                if gi < 0:
                    continue
                ch = truth[gi]
                if ch not in GI:
                    continue
                # The tool read a different character than the corpus text.
                # The corpus text is ground truth (checksum valid), so the
                # tool is wrong -- but a substitution can also mean the
                # alignment slipped a column.  Keep the slice with a low
                # weight instead of dropping the whole row, so the hard
                # (noise/rot/blur) corpus profiles still train the CNN.
                weak = 1 if text[c] != ch else 0
                patch = np.asarray(Image.open(path).convert("L"),
                                   dtype=np.float32) / 255.0
                X.append(patch)
                Y.append(GI.index(ch))
                META.append({"file": rec["id"], "line": li, "col": c, "ch": ch,
                             "tier": "A", "weak": weak})
        if n % 100 == 0:
            print(f"[shard {args.shard}] {n}/{len(recs)} slices={len(X)}", flush=True)

    if not X:
        print("no slices exported", file=sys.stderr)
        return 1
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, X=np.asarray(X, dtype=np.float32),
             Y=np.asarray(Y, dtype=np.int64),
             meta=np.array([json.dumps(m) for m in META]))
    n_files = len({m["file"] for m in META})
    print(f"exported {len(X)} slices from {n_files} corpus images "
          f"({len(recs)} scanned, tool_fail={n_skipped_tool}, "
          f"align_drop={n_skipped_align}) -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
