#!/usr/bin/env python3
"""Export CRNN training features from the native pipeline.

Runs the C recogniser over a corpus with MRZ_OCR_FEAT_DUMP enabled so the
exported numbers are byte-identical to what the device runtime feeds the
classifier: band locate -> deskew -> line split -> cell grid -> 16x12
resample -> conv1/pool1/conv2/pool2/conv3 (CNN_FLAT = 96 values per cell).

TD3 mode (default, --steps 44):
  features  (N, 2, 44, 96) float32   (line0 = MRZ line 1, line1 = line 2)
  labels    (N, 2, 44)    int64

CS/CD mode (--steps 30):
  features  (N, 1, 30, 96) float32   (single line, 30 chars)
  labels    (N, 1, 30)    int64

Images whose pipeline fails or whose cell count does not match --steps
are skipped, so the sequence length contract is always exact.
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


def read_feat(path: Path, steps: int = 44) -> np.ndarray | None:
    raw = path.read_bytes()
    if len(raw) < 4:
        return None
    (n,) = struct.unpack_from("i", raw, 0)
    need = 4 + n * CNN_FLAT * 4
    if n != steps or len(raw) < need:
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
    ap.add_argument("--steps", type=int, default=44,
                    help="sequence length: 44 for TD3 (default), 30 for CS/CD")
    args = ap.parse_args()

    steps = args.steps
    is_cscd = (steps == 30)

    corpus = Path(args.corpus)
    meta = json.loads((corpus / "corpus.json").read_text())
    records = meta["records"] if isinstance(meta, dict) else meta
    if args.max_images and args.max_images < len(records):
        records = records[: args.max_images]

    env = dict(os.environ)
    tag = f"crnn_{os.getpid()}"
    env["MRZ_OCR_FEAT_DUMP"] = tag
    env["MRZ_OCR_LINE_MODELS"] = "1" if args.line_models else "0"
    env["MRZ_OCR_CRNN"] = "1" if args.crnn else "0"
    env["MRZ_OCR_PAR"] = "1" if args.parallel else "0"
    if is_cscd:
        env["MRZ_OCR_FORCE_LAYOUT"] = "30"

    tool_path = Path(args.tool)
    from weight_fingerprint import feature_fingerprint
    tool_md5 = feature_fingerprint()
    print(f"feature fingerprint: tool={tool_path.name} fp={tool_md5[:12]} "
          f"env=line_models={args.line_models} crnn={args.crnn} par={args.parallel} "
          f"steps={steps}",
          flush=True)

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
        if not f0.exists():
            skipped += 1
            continue
        a = read_feat(f0, steps)
        if a is None:
            skipped += 1
            continue

        l1 = rec["line1"]
        if len(l1) != steps:
            skipped += 1
            continue

        if is_cscd:
            # CS/CD: single line, no line2, no TD3 checksum check
            try:
                y1 = [GLYPH_INDEX[c] for c in l1]
            except KeyError:
                skipped += 1
                continue
            feats.append(a[np.newaxis, ...])
            labels.append(np.array([y1], dtype=np.int64))
            ids.append(rec.get("id", rec["image"]))
        else:
            # TD3: two lines, require line2 checksum
            f1 = Path(f"/tmp/mrz_feat_{tag}_l1.bin")
            if not f1.exists():
                skipped += 1
                continue
            b = read_feat(f1, steps)
            if b is None:
                skipped += 1
                continue
            l2 = rec["line2"]
            if len(l2) != steps:
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
    from weight_fingerprint import feature_fingerprint as _fp
    tool_md5_end = _fp()
    if tool_md5_end != tool_md5:
        print(f"ERROR: the tool binary changed during the export "
              f"({tool_md5[:12]} -> {tool_md5_end[:12]}); the npz would mix "
              f"two builds.  Re-run the export.", file=sys.stderr)
        return 2
    np.savez_compressed(out, features=np.stack(feats), labels=np.stack(labels),
                        ids=np.array(ids),
                        tool_md5=np.array(tool_md5),
                        tool_path=np.array(str(tool_path)),
                        tool_env=np.array(
                            f"line_models={args.line_models} crnn={args.crnn} "
                            f"par={args.parallel} steps={steps}"))
    print(f"wrote {out} samples={len(ids)} skipped={skipped} "
          f"shape={np.stack(feats).shape}")
    print(f"  feature fingerprint: tool={tool_path.name} md5={tool_md5} "
          f"env=line_models={args.line_models} crnn={args.crnn} par={args.parallel} "
          f"steps={steps}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
