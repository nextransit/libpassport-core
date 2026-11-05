#!/usr/bin/env python3
"""Export fixed-44-step CRNN training features from the native pipeline.

Runs the C recogniser over a corpus with MRZ_OCR_FEAT_DUMP enabled so the
exported numbers are byte-identical to what the device runtime feeds the
classifier: band locate -> deskew -> line split -> 44-cell grid -> 16x12
resample -> conv1/pool1/conv2/pool2/conv3 (CNN_FLAT = 96 values per cell).

Output: an .npz with
  features  (N, 2, 44, 96) float32   (line0 = MRZ line 1, line1 = line 2)
  labels    (N, 2, 44)    int64
  ids       (N,)           unicode
Images whose pipeline fails or whose cell count is not 44 are skipped, so the
sequence length contract is always exact.
"""
from __future__ import annotations
import argparse
import glob
import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ<"
GLYPH_INDEX = {c: i for i, c in enumerate(ALPHABET)}
CNN_FLAT = 96
STEPS = 44

_W = [7, 3, 1]


def _cd(seg: str) -> int:
    t = 0
    for i, c in enumerate(seg):
        v = (ord(c) - 48) if c.isdigit() else ((ord(c) - 55) if 'A' <= c <= 'Z' else 0)
        t += v * _W[i % 3]
    return t % 10


def line2_checksum_ok(l2: str) -> bool:
    """TD3 line2 five-segment ICAO check.  A checksum-invalid label can
    never be ground truth, so training on one poisons the head (the
    corpus_real fixmrz pollution: 5/41 labels failed, oversampled x20-50)."""
    if len(l2) != 44:
        return False
    try:
        if _cd(l2[0:9]) != int(l2[9]): return False
        if _cd(l2[13:19]) != int(l2[19]): return False
        if _cd(l2[21:27]) != int(l2[27]): return False
        if _cd(l2[28:42]) != int(l2[42]): return False
        return _cd(l2[0:10] + l2[13:20] + l2[21:43]) == int(l2[43])
    except (ValueError, IndexError):
        return False


def read_feat(path: Path) -> np.ndarray | None:
    raw = path.read_bytes()
    if len(raw) < 4:
        return None
    (n,) = struct.unpack_from("i", raw, 0)
    need = 4 + n * CNN_FLAT * 4
    if n != STEPS or len(raw) < need:
        return None
    arr = np.frombuffer(raw, dtype="<f4", offset=4, count=n * CNN_FLAT)
    return arr.reshape(n, CNN_FLAT).astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, help="corpus dir with corpus.json + PPMs")
    ap.add_argument("--tool", default="build/mrz_ocr_cnn_tool")
    ap.add_argument("--out", required=True, help="output .npz")
    ap.add_argument("--max-images", type=int, default=0)
    ap.add_argument("--line-models", type=int, default=1)
    ap.add_argument("--crnn", type=int, default=1,
                    help="export in CRNN mode (sequential path) so the "
                         "features match CRNN inference exactly")
    ap.add_argument("--parallel", type=int, default=0)
    args = ap.parse_args()

    corpus = Path(args.corpus)
    meta = json.loads((corpus / "corpus.json").read_text())
    records = meta["records"] if isinstance(meta, dict) else meta
    if args.max_images and args.max_images < len(records):
        records = records[: args.max_images]

    env = dict(os.environ)
    # The env VALUE is the dump tag used in the output filenames; pin it so
    # the reader knows exactly which prefix to look for. A per-run tag keeps
    # concurrent exports from clobbering each other's /tmp files.
    tag = f"crnn_{os.getpid()}"
    env["MRZ_OCR_FEAT_DUMP"] = tag
    env["MRZ_OCR_LINE_MODELS"] = "1" if args.line_models else "0"
    # CRITICAL: the CRNN path forces the sequential code path, which changes
    # the scratchpad allocation order and therefore the feature values.
    # Export with the SAME backend mode that inference will use, otherwise
    # training data and inference features silently diverge (observed as a
    # 95% -> 34% character-accuracy collapse).
    env["MRZ_OCR_CRNN"] = "1" if args.crnn else "0"
    env["MRZ_OCR_PAR"] = "1" if args.parallel else "0"

    feats, labels, ids = [], [], []
    skipped = 0
    for i, rec in enumerate(records):
        img = corpus / rec["image"]
        if not img.exists():
            skipped += 1
            continue
        for f in glob.glob(f"/tmp/mrz_feat_{tag}_l*.bin"):
            os.remove(f)
        r = subprocess.run([args.tool, str(img)], capture_output=True, text=True,
                           timeout=60, env=env)
        f0 = Path(f"/tmp/mrz_feat_{tag}_l0.bin")
        f1 = Path(f"/tmp/mrz_feat_{tag}_l1.bin")
        if not f0.exists() or not f1.exists():
            skipped += 1
            continue
        a = read_feat(f0)
        b = read_feat(f1)
        if a is None or b is None:
            skipped += 1
            continue
        l1, l2 = rec["line1"], rec["line2"]
        if len(l1) != STEPS or len(l2) != STEPS:
            skipped += 1
            continue
        if not line2_checksum_ok(l2):
            print(f"  SKIP {rec.get('id', rec.get('image'))}: line2 label "
                  f"fails the ICAO checksum -- not ground truth, never train "
                  f"on it", file=sys.stderr)
            skipped += 1
            continue
        try:
            y1 = [GLYPH_INDEX[c] for c in l1]
            y2 = [GLYPH_INDEX[c] for c in l2]
        except KeyError:
            skipped += 1
            continue
        feats.append(np.stack([a, b]))
        labels.append(np.stack([y1, y2]))
        ids.append(rec.get("id", rec["image"]))
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(records)} kept={len(ids)} skipped={skipped}", flush=True)

    if not ids:
        print("no usable samples", file=sys.stderr)
        return 1
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Feature fingerprint: the features ARE the line-head conv output of
    # this exact tool binary.  Any rebuild of the conv/geometry front-end
    # changes them and silently invalidates every weight trained on this
    # npz (the 10-28 "CRNN 0/41" incident).  The md5 travels with the
    # data so train_crnn_seq.py can stamp it into the emitted header and
    # eval scripts can refuse a weights/tool mismatch.
    tool_path = Path(args.tool)
    tool_md5 = hashlib.md5(tool_path.read_bytes()).hexdigest() if tool_path.exists() else "unknown"
    np.savez_compressed(out, features=np.stack(feats), labels=np.stack(labels),
                        ids=np.array(ids),
                        tool_md5=np.array(tool_md5),
                        tool_path=np.array(str(tool_path)),
                        tool_env=np.array(
                            f"line_models={args.line_models} crnn={args.crnn} "
                            f"par={args.parallel}"))
    print(f"wrote {out} samples={len(ids)} skipped={skipped} "
          f"shape={np.stack(feats).shape}")
    print(f"  feature fingerprint: tool={tool_path.name} md5={tool_md5} "
          f"env=line_models={args.line_models} crnn={args.crnn} par={args.parallel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
