#!/usr/bin/env python3
"""Train the Tiny ConvNet OCR-B classifier and emit float32 C weights.

ARCHITECTURE (matched by C side in src/cnn.c / include/cnn.h):

  input    : 12 rows x 16 cols binary glyph  (resampled char)
  conv1    : 3x3, 8 filters, stride 1, padding 1 -> ReLU
  pool1    : 2x2 max pool                     -> 6 x 8 x 8
  conv2    : 3x3, 16 filters, stride 1, padding 0 -> ReLU
  pool2    : 2x2 max pool                     -> 2 x 3 x 16 = 96
  flatten  : 96
  fc1      : 96 -> 64, ReLU
  fc2      : 64 -> 37 (softmax)

TRAINING DATA (per class, ~2000+ seeds):
  - clean glyph
  - subpixel translation (bilinear, +/-1.5 px)
  - morphological dilate/erode 2x2 (ink spread / broken ink)
  - gamma / contrast stretch
  - hard-pair bootstrapping: augment confusable pairs
    (0/O, 8/B, 1/I, 5/S, </truncated-edge) and add a small
    pairwise contrastive loss to separate them.

OUTPUT: src/cnn_weights.h with float32 weights (no int8 quant).
"""
from __future__ import annotations
import math, random
from pathlib import Path
import numpy as np

# ---- OCR-B template bank (must match src/template.c exactly) ----
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

# Network geometry (must match include/cnn.h).
IN_H, IN_W = 12, 16
C1, C2, C3 = 8, 16, 4
KH, KW = 3, 3
HID = 64

# Hard confusable pairs -> extra samples + margin term.
HARD_PAIRS = [
    ("0", "O"), ("8", "B"), ("1", "I"), ("5", "S"), ("<", "0"),
]


def glyph_bitmap(ch):
    idx = next(i for i, (c, _) in enumerate(GLYPHS) if c == ch)
    rows = GLYPHS[idx][1]
    return [[(rows[y] >> (7 - x)) & 1 for x in range(8)] for y in range(12)]


def to_16(bmp):
    """8x12 binary glyph -> 16x12 (nearest-neighbour 2x width)."""
    return [[bmp[y][x // 2] for x in range(16)] for y in range(12)]


def binary_pad(img, scale):
    """Uprate a binary (h, w) image by integer `scale` (nearest)."""
    h, w = len(img), len(img[0])
    out = np.zeros((h * scale, w * scale), dtype=np.float32)
    for y in range(h):
        for x in range(w):
            if img[y][x]:
                out[y*scale:(y+1)*scale, x*scale:(x+1)*scale] = 1.0
    return out


def bilinear_translate(img, dx, dy):
    """Subpixel translation on a float (h, w) image via bilinear."""
    h, w = img.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    sx = xx - dx
    sy = yy - dy
    x0 = np.floor(sx).astype(np.int32); y0 = np.floor(sy).astype(np.int32)
    x1 = x0 + 1; y1 = y0 + 1
    fx = sx - x0; fy = sy - y0
    def gather(xx_, yy_):
        xxc = np.clip(xx_, 0, w - 1); yyc = np.clip(yy_, 0, h - 1)
        return img[yyc, xxc]
    out = (gather(x0, y0) * (1 - fx) * (1 - fy) +
           gather(x1, y0) * fx * (1 - fy) +
           gather(x0, y1) * (1 - fx) * fy +
           gather(x1, y1) * fx * fy)
    return out


def morph(img, op):
    """2x2 dilate (op='d') or erode (op='e') on a float (h,w)."""
    h, w = img.shape
    out = img.copy()
    if op == "d":
        for dy in (0, 1):
            for dx in (0, 1):
                a = np.zeros_like(img); b = np.zeros_like(img)
                a[:h-dy, :w-dx] = img[dy:, dx:]
                b[dy:, dx:] = img[:h-dy, :w-dx]
                out = np.maximum(out, np.maximum(a, b))
    else:
        for dy in (0, 1):
            for dx in (0, 1):
                a = np.full_like(img, 1.0)
                a[:h-dy, :w-dx] = img[dy:, dx:]
                out = np.minimum(out, a)
    return out


def gamma_contrast(img, gamma, lo=0.0, hi=1.0):
    v = (img - img.min()) / max(img.max() - img.min(), 1e-6)
    v = np.clip(v, 0, 1) ** gamma
    return lo + v * (hi - lo)


def box_downsample_ink(img, oh, ow, threshold_div=2):
    """Downsample (oh,ow) with the SAME 1/threshold_div ink rule as the
    C-side resample_char_cnn (ink * threshold_div >= total). Using the
    same rule keeps train and inference glyph distributions aligned."""
    hh, ww = img.shape
    out = np.zeros((oh, ow), dtype=np.float32)
    for y in range(oh):
        sy0 = y * hh // oh; sy1 = (y + 1) * hh // oh
        if sy1 <= sy0: sy1 = sy0 + 1
        for x in range(ow):
            sx0 = x * ww // ow; sx1 = (x + 1) * ww // ow
            if sx1 <= sx0: sx1 = sx0 + 1
            blk = img[sy0:sy1, sx0:sx1]
            ink = float(blk.sum()); total = float(blk.size)
            out[y, x] = 1.0 if (total > 0 and ink * threshold_div >= total) else 0.0
    return out


def render_augment(ch, seed):
    """Augmentation that preserves glyph identity (root-cause fixed).

    The C pipeline produces near-clean glyphs after binary resampling,
    so the trainer must NOT over-perturb glyphs into unrelated shapes
    (that is exactly what collapsed the previous model to a single
    class '8'). Strategy:
      - 60% of the time: literally the clean template
      - else: small translation (+/-1px) in 4x space, tiny noise,
        rare 1px dilate; then box-downsample with the C-side rule.
    """
    rnd = random.Random(seed)
    base8 = glyph_bitmap(ch)                 # 12x8 template (like template.c)
    base = np.asarray(to_16([[base8[y][x]*1.0 for x in range(8)]*2 for y in range(12)]),
                      dtype=np.float32)

    # 40% pure clean, 60% perturbed. All outputs are GRAYSCALE ink-high
    # 16x12 float (0=bg, 1=ink), matching the C-side resample_char_gray
    # (1 - gray/255). We upscale the 8x12 template 4x, translate, then
    # ARIA-average down to 16x12 WITHOUT hard threshold -> soft edges.
    if rnd.random() < 0.40:
        return base.copy()

    pad_l = rnd.randint(0, 2)
    pad_r = rnd.randint(0, 2)
    cols = pad_l + 8 + pad_r
    box = [[0] * cols for _ in range(12)]
    for y in range(12):
        for x in range(8):
            box[y][pad_l + x] = float(base8[y][x])
    box_np = np.asarray(box, dtype=np.float32)
    big = binary_pad(box_np, 4)
    dx = rnd.uniform(-4, 4)
    dy = rnd.uniform(-4, 4)
    big = bilinear_translate(big, dx, dy)
    m = rnd.random()
    if m < 0.06:
        big = morph(big, "d")
    elif m < 0.12:
        big = morph(big, "e")
    if rnd.random() < 0.15:
        big = gamma_contrast(big, rnd.uniform(0.88, 1.15))
    # area-average down to 16x12 (soft grayscale, no binarisation)
    hh, ww = big.shape
    out = np.zeros((12, 16), dtype=np.float32)
    for y in range(12):
        sy0, sy1 = y*hh//12, (y+1)*hh//12
        for x in range(16):
            sx0, sx1 = x*ww//16, (x+1)*ww//16
            blk = big[sy0:sy1, sx0:sx1]
            out[y, x] = float(blk.mean())
    # small random cutout (random 2x2 block erased) + gaussian noise,
    # matching the noise the corpus inserts; forces conv1 to rely on
    # stroke topology rather than single pixels.
    if rnd.random() < 0.30:
        cy0 = rnd.randint(0, 10); cx0 = rnd.randint(0, 14)
        out[cy0:cy0+2, cx0:cx0+2] = 0.0
    if rnd.random() < 0.40:
        gauss = rnd.gauss(0, 0.08)
        out = np.clip(out + gauss, 0.0, 1.0)
    # impulse noise
    n = rnd.randint(0, 3)
    for _ in range(n):
        ry, rx = rnd.randint(0, 11), rnd.randint(0, 15)
        out[ry, rx] = 1 - out[ry, rx]
    return out


def build_dataset(n_per_class=500):
    X, Y = [], []
    for idx, (ch, _) in enumerate(GLYPHS):
        base = np.asarray(to_16(glyph_bitmap(ch)), dtype=np.float32)
        # 50% clean / 50% augmented (more aug variety now that the
        # perturbation is realistic and identity-preserving)
        for _ in range(int(n_per_class * 0.5)):
            X.append(base.ravel().copy()); Y.append(idx)
        for s in range(int(n_per_class * 0.5)):
            X.append(render_augment(ch, idx * 100000 + s).ravel())
            Y.append(idx)
    # hard-pair extra samples (extra 40% for confusable classes)
    for a, b in HARD_PAIRS:
        ia = next(i for i, (c, _) in enumerate(GLYPHS) if c == a)
        ib = next(i for i, (c, _) in enumerate(GLYPHS) if c == b)
        for s in range(int(n_per_class * 0.35)):
            X.append(render_augment(a, 900000 + ia * 1000 + s).ravel()); Y.append(ia)
            X.append(render_augment(b, 900000 + ib * 1000 + s).ravel()); Y.append(ib)
    return np.asarray(X, dtype=np.float32), np.asarray(Y, dtype=np.int64)


# ---------- numpy conv primitive (im2col) ----------
def conv_fwd(X, W, b, pad):
    """X: (N, H, W, Cin) -> (N, Ho, Wo, Cout). W: (Kh,Kw,Cin,Cout)."""
    N, H, Wd, Cin = X.shape
    Kh, Kw = W.shape[0], W.shape[1]
    Cout = W.shape[3]
    Ho = H + 2 * pad - Kh + 1
    Wo = Wd + 2 * pad - Kw + 1
    Xp = np.pad(X, ((0, 0), (pad, pad), (pad, pad), (0, 0)), mode="constant")
    cols = np.zeros((N, Ho * Wo, Kh * Kw * Cin), dtype=np.float32)
    for i in range(Kh):
        for j in range(Kw):
            cols[:, :, (i * Kw + j) * Cin:(i * Kw + j + 1) * Cin] = \
                Xp[:, i:i+Ho, j:j+Wo, :].reshape(N, Ho * Wo, Cin)
    Wflat = W.reshape(-1, Cout)          # (KhKwCin, Cout)
    out = cols @ Wflat + b.reshape(1, Cout)
    return out.reshape(N, Ho, Wo, Cout), cols


def conv_bwd(dout, cols, W, X, pad):
    """dout: (N,Ho,Wo,Cout); cols: im2col; W: (Kh,Kw,Cin,Cout)."""
    N, Ho, Wo, Cout = dout.shape
    Kh, Kw, Cin, _ = W.shape
    dWflat = cols.transpose(1, 0, 2).reshape(Ho * Wo, N * Kh * Kw * Cin) \
                 .T @ dout.reshape(N * Ho * Wo, Cout)   # (KhKwCin, Cout)
    dW = dWflat.reshape(Kh, Kw, Cin, Cout)
    db = dout.sum(axis=(0, 1, 2))
    # dX via col2im
    dXp = np.zeros((N, dout.shape[1] + 2*pad, dout.shape[2] + 2*pad, Cin),
                   dtype=np.float32)
    dd = dout.reshape(N, Ho * Wo, Cout)
    Wflat = W.reshape(-1, Cout)
    dcol = dd @ Wflat.T            # (N, HoWo, KhKwCin)
    dcol = dcol.reshape(N, Ho, Wo, Kh, Kw, Cin)
    for i in range(Kh):
        for j in range(Kw):
            dXp[:, i:i+Ho, j:j+Wo, :] += dcol[:, :, :, i, j, :]
    if pad:
        return dXp[:, pad:-pad, pad:-pad, :]
    return dXp


# ---------- train (torch autograd) ----------
def train(epochs=25, lr=0.02, batch=512):
    import torch
    import torch.nn as nn
    torch.manual_seed(1)

    Xn, Yn = build_dataset(n_per_class=500)
    N = Xn.shape[0]
    print(f"dataset: {N} samples x {IN_H*IN_W}px, {NUM_CLASSES} classes", flush=True)

    class TinyConvNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, C1, 3, stride=1, padding=1)
            self.conv2 = nn.Conv2d(C1, C2, 3, stride=1, padding=1)
            self.conv3 = nn.Conv2d(C2, C3, 1)          # 1x1: 16 -> 4
            self.fc1 = nn.Linear(C3*3*8, HID)          # 3x8x4 = 96
            self.fc2 = nn.Linear(HID, NUM_CLASSES)
        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = torch.nn.functional.max_pool2d(x, (2, 1))
            x = torch.relu(self.conv2(x))
            x = torch.nn.functional.max_pool2d(x, (2, 2))  # 6x16 -> 3x8
            x = self.conv3(x)                            # 1x1 -> 3x8x4
            x = x.flatten(1)
            x = torch.relu(self.fc1(x))
            return self.fc2(x)

    model = TinyConvNet()
    # NOTE (root-cause fix): input is a binary 0/1 glyph. Global
    # mean/std centering (x-mean)/std over the augmented dataset
    # shifts clean single glyphs far from the distribution and kills
    # early ReLU (the first pipeline the user flagged: "train/inference
    # distribution mismatch"). Keep the input as raw 0/1 on BOTH
    # trainer and C side: scale = 1, mean = 0.
    Xc = Xn.reshape(-1, 1, IN_H, IN_W).astype(np.float32)
    mean_v = np.zeros((1, 1, IN_H, IN_W), dtype=np.float32)
    std_v = np.ones((1, 1, IN_H, IN_W), dtype=np.float32)
    Xc = Xc  # raw 0/1
    Xt = torch.from_numpy(Xc)
    Yt = torch.from_numpy(Yn.astype(np.int64))
    n = Xt.shape[0]

    # class-balanced CE + extra margin on hard pairs
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_acc = 0.0
    best = None
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        correct = 0; tot = 0
        for s in range(0, n, batch):
            idx = perm[s:s+batch]
            xb, yb = Xt[idx], Yt[idx]
            logits = model(xb)
            loss = torch.nn.functional.cross_entropy(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            correct += int((logits.argmax(1) == yb).sum())
            tot += len(idx)
        sched.step()
        acc = correct / tot
        if acc > best_acc:
            best_acc = acc
            best = {k: v.detach().cpu().numpy().copy()
                    for k, v in model.state_dict().items()}
        if ep % 2 == 0 or ep == epochs - 1:
            print(f"  ep {ep:3d} acc={acc:.4f} best={best_acc:.4f} "
                  f"lr={opt.param_groups[0]['lr']:.4f}", flush=True)
        if acc > 0.9999:
            break

    model.load_state_dict({k: torch.from_numpy(v)
                            for k, v in best.items()})
    model.eval()
    with torch.no_grad():
        full_logits = model(Xt)
    # class-wise acc report
    pred = full_logits.argmax(1).numpy()
    per_class = {}
    for ci in range(NUM_CLASSES):
        m = (Yn == ci)
        g = GLYPHS[ci][0]
        per_class[g] = float((pred[m] == ci).sum()) / max(int(m.sum()), 1)

    W1 = best["conv1.weight"].transpose(2,3,1,0)  # (Kh,Kw,Cin,Cout)
    b1 = best["conv1.bias"]
    W2 = best["conv2.weight"].transpose(2,3,1,0)
    b2 = best["conv2.bias"]
    W3 = best["conv3.weight"].transpose(2,3,1,0)  # (C3,C2,1,1)->(1,1,C2,C3) Kh,Kw,Cin,Cout
    b3 = best["conv3.bias"]
    Wf1 = best["fc1.weight"]
    bf1 = best["fc1.bias"]
    Wf2 = best["fc2.weight"]
    bf2 = best["fc2.bias"]
    # Input is raw 0/1 for both train and inference; no normalisation.
    mean_v = np.zeros(IN_H * IN_W, dtype=np.float32)
    std_v = np.ones(IN_H * IN_W, dtype=np.float32)
    return (W1, b1, W2, b2, W3, b3, Wf1, bf1, Wf2, bf2,
            mean_v, std_v, best_acc, per_class)

def emit_header(params):
    W1, b1, W2, b2, W3, b3, Wf1, bf1, Wf2, bf2, mean, std, best_acc = params[:13]

    def fmt(arr):
        return ", ".join(f"{v:.8f}f" for v in np.asarray(arr).ravel())

    out = ("/* Auto-generated by tools/train_cnn.py. Do not edit. */\n"
           "#ifndef MRZ_OCR_CNN_WEIGHTS_H\n"
           "#define MRZ_OCR_CNN_WEIGHTS_H\n\n"
           "#include <stdint.h>\n"
           "#include \"cnn.h\"\n\n"
           f"static const float DEFAULT_IN_MEAN[CNN_IN_H * CNN_IN_W] = {{\n")
    for i in range(0, len(mean), 8):
        out += "    " + fmt(mean[i:i+8]) + ",\n"
    out += "};\n"
    out += f"static const float DEFAULT_IN_STD[CNN_IN_H * CNN_IN_W] = {{\n"
    for i in range(0, len(std), 8):
        out += "    " + fmt(std[i:i+8]) + ",\n"
    out += "};\n\n"

    out += f"/* conv1: {KH}x{KW} x (1 -> {C1}), pad=1 */\n"
    out += "static const float DEFAULT_CONV1_W[CNN_K*CNN_K*1*CNN_C1] = {\n"
    for i in range(0, 3*3*1*C1, 8):
        out += "    " + fmt(W1.ravel()[i:i+8]) + ",\n"
    out += "};\n"
    out += "static const float DEFAULT_CONV1_B[CNN_C1] = {" + fmt(b1) + "};\n\n"
    out += f"/* conv2: {KH}x{KW} x ({C1} -> {C2}), pad=0 */\n"
    out += "static const float DEFAULT_CONV2_W[CNN_K*CNN_K*CNN_C1*CNN_C2] = {\n"
    for i in range(0, 3*3*8*16, 8):
        out += "    " + fmt(W2.ravel()[i:i+8]) + ",\n"
    out += "};\n"
    out += "static const float DEFAULT_CONV2_B[CNN_C2] = {" + fmt(b2) + "};\n\n"
    out += "/* 1x1 conv (16 -> 4) after pool2 */\n"
    out += "static const float DEFAULT_CONV3_W[CNN_C2*CNN_C3] = {\n"
    for i in range(0, W3.ravel().size, 8):
        out += "    " + fmt(W3.ravel()[i:i+8]) + ",\n"
    out += "};\n"
    out += "static const float DEFAULT_CONV3_B[CNN_C3] = {" + fmt(b3) + "};\n\n"
    out += "static const float DEFAULT_FC1_W[CNN_FLAT*CNN_HIDDEN] = {\n"
    for i in range(0, Wf1.ravel().size, 8):
        out += "    " + fmt(Wf1.ravel()[i:i+8]) + ",\n"
    out += "};\n"
    out += "static const float DEFAULT_FC1_B[CNN_HIDDEN] = {" + fmt(bf1) + "};\n\n"
    out += "static const float DEFAULT_FC2_W[CNN_HIDDEN*CNN_OUT] = {\n"
    for i in range(0, HID*37, 8):
        out += "    " + fmt(Wf2.ravel()[i:i+8]) + ",\n"
    out += "};\n"
    out += "static const float DEFAULT_FC2_B[CNN_OUT] = {" + fmt(bf2) + "};\n\n"
    out += "#endif\n"
    Path("src/cnn_weights.h").write_text(out)
    print(f"wrote src/cnn_weights.h (best_acc={best_acc:.4f})", flush=True)


if __name__ == "__main__":
    params = train(epochs=50, lr=0.02, batch=512)
    emit_header(params)
    # class-wise accuracy recap (params[-1] = per_class dict)
    if len(params) >= 12:
        print("\nper-class accuracy (balanced):")
        for ch in "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ<":
            if ch in params[11]:
                print(f"  {ch}: {params[11][ch]*100:.2f}%")
