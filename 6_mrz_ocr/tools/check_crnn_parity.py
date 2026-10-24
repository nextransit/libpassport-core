#!/usr/bin/env python3
"""Numeric parity check between the PyTorch CRNN head and the C one.

The C side is exercised through build/test_crnn_parity, which reads a
(44 x 96) feature matrix from a binary file and prints the per-step
argmax plus the first few probabilities of each line head.

This script rebuilds the SAME head in numpy from the EXPORTED header
(crnn_weights.h) and compares step by step.  It is the guard that
catches a weight-export layout mistake: the two sides consume the same
numbers, so any mismatch is an export or an indexing bug, not a
modelling difference.

Usage:
    python3 tools/check_crnn_parity.py [--features features.bin]
"""
from __future__ import annotations

import argparse
import re
import struct
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
H, CLASSES, STEPS, FEATURES = 32, 37, 44, 96
CTX = 2 * H


def load_header(path: Path):
    text = path.read_text()
    out = {}
    for name, body in re.findall(r"static const float (\w+)\[\d+\] = \{([^}]*)\};",
                                 text, re.S):
        vals = np.array([float(v.rstrip("f")) for v in body.replace("\n", " ").split(",")
                         if v.strip()], dtype=np.float32)
        out[name] = vals
    return out


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -12, 12)))


def gru_forward(w, x, reverse):
    """One GRU direction, using the C index convention (flat [in][out])."""
    h = np.zeros(H, np.float64)
    outs = []
    order = range(STEPS - 1, -1, -1) if reverse else range(STEPS)
    for t in order:
        v = x[t]
        z = np.zeros(H)
        r = np.zeros(H)
        for j in range(H):
            sz = w["BZ"][j]
            sr = w["BR"][j]
            for i in range(H):
                sz += v[i] * w["WZ"][i * H + j] + h[i] * w["UZ"][i * H + j]
                sr += v[i] * w["WR"][i * H + j] + h[i] * w["UR"][i * H + j]
            z[j] = sigmoid(sz)
            r[j] = sigmoid(sr)
        n = np.zeros(H)
        for j in range(H):
            sn = w["BH_IN"][j]
            sh = w["BH_HN"][j]
            for i in range(H):
                sn += v[i] * w["WH"][i * H + j]
                sh += h[i] * w["UH"][i * H + j]
            n[j] = np.tanh(sn + r[j] * sh)
        h = (1 - z) * n + z * h
        outs.append((t, h.copy()))
    outs.sort()
    return np.stack([o[1] for o in outs])


def head_probs(W, prefix, x):
    proj_w = W[prefix + "_PROJ_W"].reshape(FEATURES, H)
    proj_b = W[prefix + "_PROJ_B"]
    pos_w = W[prefix + "_POS_W"].reshape(STEPS, H)
    emb = np.maximum(x @ proj_w + proj_b, 0.0) + pos_w

    def dirw(suffix):
        return {
            "WZ": W[prefix + "_WZ" + suffix], "UZ": W[prefix + "_UZ" + suffix],
            "BZ": W[prefix + "_BZ" + suffix], "WR": W[prefix + "_WR" + suffix],
            "UR": W[prefix + "_UR" + suffix], "BR": W[prefix + "_BR" + suffix],
            "WH": W[prefix + "_WH" + suffix], "UH": W[prefix + "_UH" + suffix],
            "BH_IN": W[prefix + "_BH_IN" + suffix],
            "BH_HN": W[prefix + "_BH_HN" + suffix],
        }

    fwd = gru_forward(dirw(""), emb, False)
    bwd = gru_forward(dirw("B"), emb, True)
    cat = np.concatenate([fwd, bwd], -1)
    out_w = W[prefix + "_OUT_W"].reshape(CTX, CLASSES)
    logits = cat @ out_w + W[prefix + "_OUT_B"]
    logits -= logits.max(-1, keepdims=True)
    e = np.exp(logits)
    return e / e.sum(-1, keepdims=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="")
    ap.add_argument("--tool", default="build/test_crnn_parity")
    args = ap.parse_args()

    W = load_header(ROOT / "src" / "crnn_weights.h")
    if args.features:
        feat_path = Path(args.features)
    else:
        rng = np.random.default_rng(11)
        x = rng.standard_normal((STEPS, FEATURES)).astype(np.float32) * 0.7
        feat_path = Path("/tmp/mrz_parity_features.bin")
        feat_path.write_bytes(struct.pack("i", STEPS) +
                              x.astype("<f4").tobytes())
        print(f"wrote {feat_path}")

    raw = feat_path.read_bytes()
    (n,) = struct.unpack_from("i", raw, 0)
    x = np.frombuffer(raw, dtype="<f4", offset=4,
                      count=n * FEATURES).reshape(n, FEATURES).astype(np.float64)

    proc = subprocess.run([str(ROOT / args.tool), str(feat_path)],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stdout, proc.stderr)
        return 1

    ok = True
    line_idx = -1
    for line in proc.stdout.strip().splitlines():
        if line.startswith("line"):
            line_idx += 1
            prefix = f"CRNN{line_idx + 1}"
            P = head_probs(W, prefix, x)
            c_args = [int(v) for v in line.split(":")[1].split()]
            t_args = P.argmax(-1).tolist()
            match = c_args == t_args
            ok &= match
            print(f"  {prefix}: argmax match = {match}")
            if not match:
                bad = [(i, a, b) for i, (a, b) in enumerate(zip(c_args, t_args)) if a != b]
                print("    first mismatches:", bad[:6])
        elif line.startswith("p"):
            parts = line.split(":")
            t = int(parts[0][1:])
            c_vals = [float(v) for v in parts[1].split()]
            P = head_probs(W, f"CRNN{line_idx + 1}", x)
            d = max(abs(a - b) for a, b in zip(c_vals, P[t][:5].tolist()))
            ok &= d < 2e-5
            print(f"  {parts[0]} max abs diff = {d:.2e} "
                  f"{'OK' if d < 2e-5 else 'MISMATCH'}")
    print("PARITY", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
