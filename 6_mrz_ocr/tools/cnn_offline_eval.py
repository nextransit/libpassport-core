#!/usr/bin/env python3
"""Offline evaluation of a CNN weight header on exported glyph slices.

Why this exists
---------------
The end-to-end loop (train -> recompile -> export CRNN features -> retrain
CRNN -> rebuild -> bench) takes minutes per iteration.  Most of the
questions we need answered are much narrower: "does this weight set read
the real photo glyphs correctly, and which characters does it get wrong?"
Those can be answered directly on the exported 16x12 slices.

This module parses a `cnn_weights_line{1,2}.h` header, re-implements the
C-side forward pass exactly (src/cnn.c is the reference), and scores a
weight set against a slice npz.  If the forward pass drifts from the C
one, the numbers lie -- so the layout details are spelled out below.

Layouts (all taken from src/cnn.c):
  * conv1_w / conv2_w are (KH, KW, CIN, COUT), index
    ((kh*K+kw)*CIN + ci)*COUT + co
  * conv3_w is (CIN, COUT): conv3_w[c*COUT + c3]
  * the conv3 output is flattened as (c3, y, x) -- c3 outermost -- which
    matches torch's flatten(1) on an (N,C3,H,W) tensor
  * normalisation is (g - in_mean) / in_std, elementwise, no abs

Usage:
  python3 tools/cnn_offline_eval.py --weights src/cnn_weights_line1.h \
      --npz data/photo_train_passport/cnn_slices.npz --line 0
"""
from __future__ import annotations
import argparse, json, re
from pathlib import Path

import numpy as np

GLYPHS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ<"
GI = {c: i for i, c in enumerate(GLYPHS)}
C1, C2, C3, HID = 8, 16, 4, 64
IN_H, IN_W = 12, 16
NUM_CLASSES = 37


# --------------------------------------------------------------------------
# header parsing
# --------------------------------------------------------------------------
def parse_header(path):
    """Return {symbol: np.ndarray} for every `static const float X[]`."""
    text = Path(path).read_text()
    out = {}
    for m in re.finditer(
            r"static\s+const\s+float\s+(\w+)\s*\[[^\]]*\]\s*=\s*\{(.*?)\}\s*;",
            text, re.S):
        name, body = m.group(1), m.group(2)
        vals = []
        for v in body.split(","):
            v = v.split("//")[0].split("/*")[0].strip()
            if not v:
                continue
            vals.append(float(v.rstrip("fF")))
        out[name] = np.asarray(vals, dtype=np.float32)
    return out


def pick(arrs, prefix):
    names = {
        "conv1_w": f"{prefix}_CONV1_W", "conv1_b": f"{prefix}_CONV1_B",
        "conv2_w": f"{prefix}_CONV2_W", "conv2_b": f"{prefix}_CONV2_B",
        "conv3_w": f"{prefix}_CONV3_W", "conv3_b": f"{prefix}_CONV3_B",
        "fc1_w": f"{prefix}_FC1_W", "fc1_b": f"{prefix}_FC1_B",
        "fc2_w": f"{prefix}_FC2_W", "fc2_b": f"{prefix}_FC2_B",
        "in_mean": f"{prefix}_IN_MEAN", "in_std": f"{prefix}_IN_STD",
    }
    missing = [k for k, v in names.items() if v not in arrs]
    if missing:
        raise SystemExit(f"header is missing {missing}; symbols: {sorted(arrs)[:24]}")
    return {k: arrs[v] for k, v in names.items()}


def infer_prefix(arrs, explicit=None):
    if explicit:
        return explicit
    cands = sorted({k.split("_")[0] for k in arrs})
    if len(cands) == 1:
        return cands[0]
    for c in cands:
        if c.upper().startswith("LINE"):
            return c
    raise SystemExit(f"cannot infer prefix, pass --prefix; candidates {cands}")


# --------------------------------------------------------------------------
# forward pass (numpy, mirroring src/cnn.c)
# --------------------------------------------------------------------------
def _conv_k3(x, w, b):
    """x:(N,H,W,CIN); w:(3,3,CIN,COUT); b:(COUT,) -> (N,H,W,COUT)."""
    n, h, wd, cin = x.shape
    xp = np.zeros((n, h + 2, wd + 2, cin), dtype=np.float32)
    xp[:, 1:h + 1, 1:wd + 1] = x
    out = np.zeros((n, h, wd, w.shape[3]), dtype=np.float32)
    for kh in range(3):
        for kw in range(3):
            patch = xp[:, kh:kh + h, kw:kw + wd, :]          # (n,h,wd,cin)
            out += np.tensordot(patch, w[kh, kw], axes=([3], [0]))
    return out + b[None, None, None, :]


def forward(x, P):
    """x:(N,12,16) raw glyph values -> logits (N,37)."""
    n = x.shape[0]
    mean = P["in_mean"].reshape(1, IN_H, IN_W)
    std = P["in_std"].reshape(1, IN_H, IN_W)
    z = (x - mean) / std
    z = z[:, :, :, None]                                    # (n,12,16,1)

    w1 = P["conv1_w"].reshape(3, 3, 1, C1)
    a = _conv_k3(z, w1, P["conv1_b"])
    a = np.maximum(a, 0.0)
    a = np.maximum(a[:, 0::2, :, :], a[:, 1::2, :, :])       # pool 2x1 (H)

    w2 = P["conv2_w"].reshape(3, 3, C1, C2)
    b = _conv_k3(a, w2, P["conv2_b"])
    b = np.maximum(b, 0.0)
    p2 = np.maximum(np.maximum(b[:, 0::2, 0::2, :], b[:, 0::2, 1::2, :]),
                    np.maximum(b[:, 1::2, 0::2, :], b[:, 1::2, 1::2, :]))

    # conv3 is 1x1: v[c3] = b[c3] + sum_c p2[c] * conv3_w[c*C3 + c3]
    c3 = (p2 @ P["conv3_w"].reshape(C2, C3)) + P["conv3_b"][None, None, None, :]
    # flatten (c3, y, x) -- c3 outermost, like torch flatten on (N,C3,H,W)
    feat = c3.transpose(0, 3, 1, 2).reshape(n, -1)

    # fc1: C reads fc1_w + j*CNN_FLAT, i.e. the header layout is
    # (HIDDEN, FLAT).  fc2: fc2_w + o*CNN_HIDDEN, i.e. (OUT, HIDDEN).
    w1 = P["fc1_w"].reshape(HID, -1)
    w2 = P["fc2_w"].reshape(NUM_CLASSES, HID)
    h = np.maximum(feat @ w1.T + P["fc1_b"][None, :], 0.0)
    return h @ w2.T + P["fc2_b"][None, :]


def softmax(l):
    l = l - l.max(axis=1, keepdims=True)
    e = np.exp(l)
    return e / e.sum(axis=1, keepdims=True)


# --------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------
def load_slices(npz, line=None):
    d = np.load(npz, allow_pickle=True)
    X = d["X"].astype(np.float32)
    Y = d["Y"].astype(np.int64)
    meta = []
    if "meta" in d.files:
        meta = [json.loads(s) for s in d["meta"]]
    keep = list(range(len(Y)))
    if line is not None and meta:
        keep = [i for i, m in enumerate(meta) if int(m.get("line", -1)) == line]
    return X[keep], Y[keep], [meta[i] for i in keep]


def report(X, Y, meta, P, show_conf=14, show_photos=12):
    logits = forward(X, P)
    pred = logits.argmax(1)
    prob = softmax(logits)
    acc = float((pred == Y).mean())
    print(f"  acc={acc*100:.2f}%  n={len(Y)}")
    conf = {}
    for i in range(len(Y)):
        if pred[i] != Y[i]:
            k = (GLYPHS[Y[i]], GLYPHS[pred[i]])
            conf[k] = conf.get(k, 0) + 1
    if conf:
        print("  confusions (GT->pred):")
        for (g, p), n in sorted(conf.items(), key=lambda kv: -kv[1])[:show_conf]:
            print(f"    {g} -> {p} : {n}")
    byfile = {}
    for i, m in enumerate(meta):
        f = str(m.get("file", "?"))
        d = byfile.setdefault(f, [0, 0])
        d[1] += 1
        if pred[i] == Y[i]:
            d[0] += 1
    worst = sorted(byfile.items(), key=lambda kv: kv[1][0] / max(1, kv[1][1]))
    if worst:
        print("  per-photo (worst first):")
        for f, (c, t) in worst[:show_photos]:
            flag = "" if c == t else "  <-- "
            print(f"    {f}: {c}/{t}{flag}")
    return acc, pred, prob


def lean_of(g):
    """Italic lean of a 16x12 glyph: lower-third centroid - upper-third."""
    xs = np.arange(g.shape[1])
    top = g[:4].sum(axis=0)
    bot = g[8:].sum(axis=0)
    if top.sum() <= 0 or bot.sum() <= 0:
        return 0.0
    return float((xs * bot).sum() / bot.sum() - (xs * top).sum() / top.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--prefix", default=None)
    ap.add_argument("--npz", required=True)
    ap.add_argument("--line", type=int, default=None)
    ap.add_argument("--lean", action="store_true")
    args = ap.parse_args()

    arrs = parse_header(args.weights)
    prefix = infer_prefix(arrs, args.prefix)
    P = pick(arrs, prefix)
    print(f"weights={args.weights} prefix={prefix}")
    X, Y, meta = load_slices(args.npz, args.line)
    report(X, Y, meta, P)

    if args.lean:
        leans = np.array([lean_of(g) for g in X])
        print(f"  lean: min={leans.min():+.2f} max={leans.max():+.2f} "
              f"mean={leans.mean():+.2f} |lean|>1px: "
              f"{int((np.abs(leans) > 1).sum())}/{len(leans)}")


if __name__ == "__main__":
    main()
