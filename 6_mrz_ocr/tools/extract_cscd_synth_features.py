#!/usr/bin/env python3
"""Extract 96-dim CRNN features from synthetic CS/CD glyph patches.

Loads CNN conv weights from the C header (line2 head, which the C code
uses for CS/CD), implements the CNN front-end in PyTorch, and extracts
the 96-dim feature vector (conv3 output) from each 12x16 glyph patch.

This bridges gen_cscd_synthetic.py (which renders patches) and
train_crnn_seq.py (which needs 96-dim features) without requiring the
full C pipeline (band locate -> deskew -> grid split).

Output: .npz with
  features  (N, 1, 30, 96) float32
  labels    (N, 1, 30)    int64
  ids       (N,)           unicode
"""
from __future__ import annotations
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ<"
GLYPH_INDEX = {c: i for i, c in enumerate(ALPHABET)}
CNN_IN_H, CNN_IN_W = 12, 16
CNN_C1, CNN_C2, CNN_C3 = 8, 16, 4
CNN_FLAT = 96


def parse_c_header(path: Path, prefix: str = "LINE2") -> dict:
    text = path.read_text()
    arrays = {}
    pattern = re.compile(
        r'static\s+const\s+float\s+' + prefix + r'_(\w+)\s*\[[^\]]*\]\s*=\s*\{([^}]+)\}',
        re.MULTILINE
    )
    for m in pattern.finditer(text):
        name = m.group(1)
        vals_str = m.group(2)
        vals = [float(v.strip().rstrip('f')) for v in vals_str.split(',') if v.strip()]
        arrays[name] = np.array(vals, dtype=np.float32)
    return arrays


class CNNFrontend(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('in_mean', torch.zeros(CNN_IN_H, CNN_IN_W))
        self.register_buffer('in_std', torch.ones(CNN_IN_H, CNN_IN_W))
        self.conv1 = nn.Conv2d(1, CNN_C1, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(CNN_C1, CNN_C2, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(CNN_C2, CNN_C3, 1)

    @staticmethod
    def from_c_weights(w: dict) -> "CNNFrontend":
        net = CNNFrontend()
        net.in_mean.data = torch.from_numpy(w["IN_MEAN"].reshape(CNN_IN_H, CNN_IN_W))
        net.in_std.data = torch.from_numpy(w["IN_STD"].reshape(CNN_IN_H, CNN_IN_W))
        net.conv1.weight.data = torch.from_numpy(
            w["CONV1_W"].reshape(3, 3, 1, CNN_C1).transpose(3, 2, 0, 1).copy())
        net.conv1.bias.data = torch.from_numpy(w["CONV1_B"])
        net.conv2.weight.data = torch.from_numpy(
            w["CONV2_W"].reshape(3, 3, CNN_C1, CNN_C2).transpose(3, 2, 0, 1).copy())
        net.conv2.bias.data = torch.from_numpy(w["CONV2_B"])
        net.conv3.weight.data = torch.from_numpy(
            w["CONV3_W"].reshape(1, 1, CNN_C2, CNN_C3).transpose(3, 2, 0, 1).copy())
        net.conv3.bias.data = torch.from_numpy(w["CONV3_B"])
        net.eval()
        return net

    def forward(self, x):
        x = (x - self.in_mean) / (self.in_std + 1e-8)
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, (2, 1))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, (2, 2))
        x = self.conv3(x)
        x = x.flatten(1)
        return x


def load_ppm_patch(path: Path) -> np.ndarray:
    img = Image.open(str(path))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    if arr.shape != (CNN_IN_H, CNN_IN_W):
        img = img.resize((CNN_IN_W, CNN_IN_H), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
    # Invert polarity: gen_cscd_synthetic.py renders ink=dark (low values)
    # on bg=light (high values), but the C pipeline's resample_char_gray
    # outputs 1.0 - gray/255 (ink=high, bg=low).  Without this flip the
    # CRNN trains on features whose polarity is opposite to inference.
    arr = 1.0 - arr
    return arr


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth-dir", required=True,
                    help="directory from gen_cscd_synthetic.py")
    ap.add_argument("--cnn-header", default="src/cnn_weights_line2.h",
                    help="C header with CNN weights (line2 head for CS/CD)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-images", type=int, default=0)
    args = ap.parse_args()

    synth_dir = Path(args.synth_dir)
    header_path = Path(args.cnn_header)
    if not header_path.exists():
        print(f"CNN header not found: {header_path}", file=sys.stderr)
        return 1

    w = parse_c_header(header_path, prefix="LINE2")
    needed = ["IN_MEAN", "IN_STD", "CONV1_W", "CONV1_B", "CONV2_W", "CONV2_B",
              "CONV3_W", "CONV3_B"]
    for k in needed:
        if k not in w:
            print(f"Missing {k} in {header_path}", file=sys.stderr)
            return 1
    print(f"Loaded CNN weights from {header_path}: {len(w)} arrays", flush=True)

    net = CNNFrontend.from_c_weights(w)

    img_dirs = sorted(synth_dir.glob("img_*"))
    if args.max_images:
        img_dirs = img_dirs[:args.max_images]
    print(f"Processing {len(img_dirs)} synthetic images", flush=True)

    feats, labels, ids = [], [], []
    skipped = 0
    for i, imgdir in enumerate(img_dirs):
        meta_path = imgdir / "meta.json"
        if not meta_path.exists():
            skipped += 1
            continue
        meta = json.loads(meta_path.read_text())
        line1 = meta.get("line1", "")
        if len(line1) != 30:
            skipped += 1
            continue
        patches = []
        ok = True
        for col in range(30):
            ppm_path = imgdir / f"mrz_dump_img_l0_c{col:02d}_x0_r0_w8.ppm"
            if not ppm_path.exists():
                ok = False
                break
            patches.append(load_ppm_patch(ppm_path))
        if not ok:
            skipped += 1
            continue
        patch_tensor = torch.from_numpy(np.stack(patches)).unsqueeze(1)
        with torch.no_grad():
            feat = net(patch_tensor)
        try:
            y1 = [GLYPH_INDEX[c] for c in line1]
        except KeyError:
            skipped += 1
            continue
        feats.append(feat.numpy()[np.newaxis, ...])
        labels.append(np.array([y1], dtype=np.int64))
        ids.append(imgdir.name)
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(img_dirs)} kept={len(ids)} skipped={skipped}", flush=True)

    if not ids:
        print("no usable samples", file=sys.stderr)
        return 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out,
                        features=np.stack(feats),
                        labels=np.stack(labels),
                        ids=np.array(ids))
    print(f"wrote {out} samples={len(ids)} skipped={skipped} "
          f"shape={np.stack(feats).shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
