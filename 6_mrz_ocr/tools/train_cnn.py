#!/usr/bin/env python3
"""Train the tiny OCR-B MLP (raw 12x8 pixels only) and emit float32 weights.

ARCHITECTURE (matched by C side):
   fc1 96 -> 32, ReLU
   fc2 32 -> 37

INPUT: 96 raw pixel values for a 12x8 binary OCR-B glyph.

TRAINING:
- For each glyph, include exactly one "clean" sample plus a handful
  of small geometric augmentations.
- Adam-style SGD with cosine LR.
"""
from __future__ import annotations
import math, random
from pathlib import Path

GLYPHS = [
    ('0', [0x3C, 0x66, 0x66, 0x6E, 0x76, 0x66, 0x66, 0x66, 0x66, 0x6E, 0x66, 0x3C]),
    ('1', [0x18, 0x38, 0x78, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x7E]),
    ('2', [0x3C, 0x66, 0x66, 0x06, 0x06, 0x0C, 0x18, 0x30, 0x60, 0x60, 0x60, 0x7E]),
    ('3', [0x3C, 0x66, 0x66, 0x06, 0x06, 0x1C, 0x06, 0x06, 0x06, 0x66, 0x66, 0x3C]),
    ('4', [0x0C, 0x1C, 0x3C, 0x6C, 0x6C, 0x66, 0x66, 0x7E, 0x06, 0x06, 0x06, 0x06]),
    ('5', [0x7E, 0x60, 0x60, 0x60, 0x7C, 0x66, 0x06, 0x06, 0x06, 0x66, 0x66, 0x3C]),
    ('6', [0x3C, 0x66, 0x60, 0x60, 0x7C, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x3C]),
    ('7', [0x7E, 0x66, 0x06, 0x06, 0x0C, 0x18, 0x18, 0x18, 0x18, 0x30, 0x30, 0x30]),
    ('8', [0x3C, 0x66, 0x66, 0x66, 0x66, 0x3C, 0x66, 0x66, 0x66, 0x66, 0x66, 0x3C]),
    ('9', [0x3C, 0x66, 0x66, 0x66, 0x66, 0x66, 0x3E, 0x06, 0x06, 0x06, 0x66, 0x3C]),
    ('A', [0x18, 0x18, 0x3C, 0x3C, 0x66, 0x66, 0x7E, 0x7E, 0x66, 0x66, 0x66, 0x66]),
    ('B', [0x7C, 0x66, 0x66, 0x66, 0x66, 0x7C, 0x66, 0x66, 0x66, 0x66, 0x66, 0x7C]),
    ('C', [0x3C, 0x66, 0x66, 0x60, 0x60, 0x60, 0x60, 0x60, 0x60, 0x66, 0x66, 0x3C]),
    ('D', [0x78, 0x6C, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x6C, 0x78]),
    ('E', [0x7E, 0x60, 0x60, 0x60, 0x60, 0x7C, 0x60, 0x60, 0x60, 0x60, 0x60, 0x7E]),
    ('F', [0x7E, 0x60, 0x60, 0x60, 0x60, 0x7C, 0x60, 0x60, 0x60, 0x60, 0x60, 0x60]),
    ('G', [0x3C, 0x66, 0x66, 0x60, 0x60, 0x60, 0x6E, 0x66, 0x66, 0x66, 0x66, 0x3C]),
    ('H', [0x66, 0x66, 0x66, 0x66, 0x66, 0x7E, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66]),
    ('I', [0x3C, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x3C]),
    ('J', [0x3E, 0x0C, 0x0C, 0x0C, 0x0C, 0x0C, 0x0C, 0x0C, 0x0C, 0x6C, 0x6C, 0x38]),
    ('K', [0x66, 0x6C, 0x78, 0x70, 0x78, 0x6C, 0x66, 0x66, 0x66, 0x66, 0x6C, 0x66]),
    ('L', [0x60, 0x60, 0x60, 0x60, 0x60, 0x60, 0x60, 0x60, 0x60, 0x60, 0x60, 0x7E]),
    ('M', [0x66, 0x7E, 0x7E, 0x7E, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66]),
    ('N', [0x66, 0x66, 0x76, 0x7E, 0x7E, 0x6E, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66]),
    ('O', [0x3C, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x3C]),
    ('P', [0x7C, 0x66, 0x66, 0x66, 0x66, 0x7C, 0x60, 0x60, 0x60, 0x60, 0x60, 0x60]),
    ('Q', [0x3C, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x6E, 0x6C, 0x36, 0x1E]),
    ('R', [0x7C, 0x66, 0x66, 0x66, 0x66, 0x7C, 0x6C, 0x66, 0x66, 0x66, 0x66, 0x66]),
    ('S', [0x3E, 0x60, 0x60, 0x60, 0x3C, 0x06, 0x06, 0x06, 0x06, 0x06, 0x66, 0x3C]),
    ('T', [0x7E, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18]),
    ('U', [0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x3C]),
    ('V', [0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x3C, 0x3C, 0x18, 0x18]),
    ('W', [0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x7E, 0x7E, 0x7E, 0x66, 0x66]),
    ('X', [0x66, 0x66, 0x66, 0x3C, 0x3C, 0x18, 0x3C, 0x3C, 0x66, 0x66, 0x66, 0x66]),
    ('Y', [0x66, 0x66, 0x66, 0x66, 0x3C, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18]),
    ('Z', [0x7E, 0x06, 0x06, 0x06, 0x0C, 0x18, 0x30, 0x60, 0x60, 0x60, 0x60, 0x7E]),
    ('<', [0x00, 0x00, 0x06, 0x0C, 0x18, 0x30, 0x60, 0x30, 0x18, 0x0C, 0x06, 0x00]),
]
assert len(GLYPHS) == 37
NUM_CLASSES = len(GLYPHS)
FEATURE_DIM = 96      # raw 12x8 bitmaps only (no pooling)
HIDDEN = 96


def glyph_bitmap(ch):
    idx = next(i for i, (c, _) in enumerate(GLYPHS) if c == ch)
    rows = GLYPHS[idx][1]
    return [[(rows[y] >> (7 - x)) & 1 for x in range(8)] for y in range(12)]


def render_high(bmp, scale):
    H, W = 12 * scale, 8 * scale
    out = [[0] * W for _ in range(H)]
    for y in range(12):
        for x in range(8):
            v = bmp[y][x]
            for dy in range(scale):
                for dx in range(scale):
                    out[y*scale+dy][x*scale+dx] = v
    return out


def box_downsample(big, oh, ow, threshold_div=8):
    H, W = len(big), len(big[0])
    out = [[0]*ow for _ in range(oh)]
    for y in range(oh):
        sy0 = y * H // oh
        sy1 = (y+1) * H // oh
        if sy1 <= sy0: sy1 = sy0 + 1
        for x in range(ow):
            sx0 = x * W // ow
            sx1 = (x+1) * W // ow
            if sx1 <= sx0: sx1 = sx0 + 1
            ink = total = 0
            for sy in range(sy0, sy1):
                for sx in range(sx0, sx1):
                    ink += big[sy][sx]; total += 1
            out[y][x] = 1 if (total > 0 and ink * threshold_div >= total) else 0
    return out


def augment(bmp, seed):
    """Tiny noise on the 12x8 binary glyph. Matches the kind of
    variation introduced by the resample+binarise step in the OCR
    pipeline: a few stray ink or background flips per character."""
    rnd = random.Random(seed)
    out = [row[:] for row in bmp]
    n_flip = rnd.randint(0, 3)
    for _ in range(n_flip):
        ry = rnd.randint(0, 11)
        rx = rnd.randint(0, 7)
        out[ry][rx] ^= 1
    return out


def extract_raw(bmp):
    return [float(bmp[y][x]) for y in range(12) for x in range(8)]


DIM = len(extract_raw(glyph_bitmap('A')))
assert DIM == FEATURE_DIM, f"{DIM} vs {FEATURE_DIM}"


def main():
    import numpy as np
    random.seed(0)
    np.random.seed(1)
    X, Y = [], []
    for idx, (ch, _) in enumerate(GLYPHS):
        base = glyph_bitmap(ch)
        # 30 clean + 30 augmented (90% clean shifts + pixel noise).
        # Most samples are clean because the OCR binarisation step
        # largely preserves the glyph; a few are lightly perturbed
        # to introduce robustness against segmenter boundary jitter.
        for _ in range(30):
            X.append(extract_raw(base))
            Y.append(idx)
        for s in range(30):
            bmp = augment(base, seed=idx * 1000 + s)
            X.append(extract_raw(bmp))
            Y.append(idx)
    print(f"built training set: {len(X)} x {DIM}", flush=True)

    Xn = np.asarray(X, dtype=np.float32)
    Yn = np.asarray(Y, dtype=np.int64)
    n, D = Xn.shape
    H = HIDDEN
    K = NUM_CLASSES
    rng = np.random.default_rng(1)
    # NO mean/std normalization for raw binary input -- it's already 0/1.
    W1 = rng.normal(0.0, math.sqrt(2.0 / D), size=(H, D)).astype(np.float32)
    b1 = np.zeros(H, dtype=np.float32)
    W2 = rng.normal(0.0, math.sqrt(2.0 / H), size=(K, H)).astype(np.float32)
    b2 = np.zeros(K, dtype=np.float32)
    best_acc = 0.0
    best = (W1.copy(), b1.copy(), W2.copy(), b2.copy())
    lr = 0.1; l2 = 1e-4; batch = 64; epochs = 600
    print("training MLP (raw 96 -> 32 -> 37)...", flush=True)
    for ep in range(epochs):
        perm = rng.permutation(n)
        correct = 0
        cur_lr = lr * 0.5 * (1 + math.cos(math.pi * ep / epochs))
        for s in range(0, n, batch):
            idx = perm[s:s+batch]
            xb = Xn[idx]; yb = Yn[idx]
            h_pre = xb @ W1.T + b1
            h_act = np.maximum(h_pre, 0)
            logits = h_act @ W2.T + b2
            logits -= logits.max(axis=1, keepdims=True)
            ez = np.exp(logits)
            p = ez / ez.sum(axis=1, keepdims=True)
            oh = np.zeros_like(p)
            oh[np.arange(len(idx)), yb] = 1
            dlogits = (p - oh) / len(idx)
            dW2 = dlogits.T @ h_act
            db2 = dlogits.sum(axis=0)
            dh = dlogits @ W2
            dh_relu = dh * (h_pre > 0)
            dW1 = dh_relu.T @ xb
            db1 = dh_relu.sum(axis=0)
            W2 -= cur_lr * (dW2 + l2 * W2)
            b2 -= cur_lr * db2
            W1 -= cur_lr * (dW1 + l2 * W1)
            b1 -= cur_lr * db1
            correct += int((p.argmax(axis=1) == yb).sum())
        acc = correct / n
        if acc > best_acc:
            best_acc = acc
            best = (W1.copy(), b1.copy(), W2.copy(), b2.copy())
        if ep % 20 == 0 or ep == epochs-1 or acc > 0.999:
            print(f"  ep {ep:4d} acc={acc:.4f} best={best_acc:.4f}", flush=True)
        if acc > 0.999:
            break
    W1, b1, W2, b2 = best

    out = (f"/* Auto-generated by tools/train_cnn.py. */\n"
           f"#ifndef MRZ_OCR_CNN_WEIGHTS_H\n"
           f"#define MRZ_OCR_CNN_WEIGHTS_H\n\n"
           f"#include <stdint.h>\n"
           f"#include \"cnn.h\"\n\n"
           f"/* No conv weights - input is raw pixels only. */\n"
           f"static const int8_t DEFAULT_CONV_W[CNN_NF * CNN_K * CNN_K]"
           f" = {{ 0 }};\n"
           f"static const int8_t DEFAULT_CONV_B[CNN_NF] = {{ 0 }};\n\n")

    def fmt(arr):
        return ", ".join(f"{v:.8f}f" for v in arr)

    fc1_w_flat = [float(W1[j][i]) for j in range(HIDDEN) for i in range(FEATURE_DIM)]
    fc1_b_flat = [float(b1[j]) for j in range(HIDDEN)]
    out += f"/* fc1 {FEATURE_DIM} -> {HIDDEN}, ReLU. Float32. */\n"
    out += "static const float DEFAULT_FC1_W[CNN_FEATURE * CNN_HIDDEN] = {\n"
    for i in range(0, len(fc1_w_flat), 8):
        out += "    " + fmt(fc1_w_flat[i:i+8]) + ",\n"
    out += "};\n"
    out += "static const float DEFAULT_FC1_B[CNN_HIDDEN] = {\n"
    for i in range(0, len(fc1_b_flat), 8):
        out += "    " + fmt(fc1_b_flat[i:i+8]) + ",\n"
    out += "};\n\n"

    fc2_w_flat = [float(W2[o][j]) for o in range(NUM_CLASSES) for j in range(HIDDEN)]
    fc2_b_flat = [float(b2[o]) for o in range(NUM_CLASSES)]
    out += f"/* fc2 {HIDDEN} -> {NUM_CLASSES}. Float32. */\n"
    out += "static const float DEFAULT_FC2_W[CNN_HIDDEN * CNN_OUT] = {\n"
    for i in range(0, len(fc2_w_flat), 8):
        out += "    " + fmt(fc2_w_flat[i:i+8]) + ",\n"
    out += "};\n"
    out += "static const float DEFAULT_FC2_B[CNN_OUT] = {\n"
    for i in range(0, len(fc2_b_flat), 8):
        out += "    " + fmt(fc2_b_flat[i:i+8]) + ",\n"
    out += "};\n\n"
    out += "static const float DEFAULT_FC1_SCALE = 1.0f;\n"
    out += "static const float DEFAULT_FC2_SCALE = 1.0f;\n\n"
    out += "#endif\n"
    Path("src/cnn_weights.h").write_text(out)
    print(f"wrote src/cnn_weights.h (best_acc={best_acc:.4f})")


if __name__ == "__main__":
    main()
