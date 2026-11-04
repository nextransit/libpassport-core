"""Train CNN weights directly from cnn_slices.npz exported by export_photo_train.py.

This skips the corpus -> dump -> gather pipeline in train_cnn_real.py and
trains directly on the pre-exported glyph slices, using the GT from
gt_review_loop.xlsx as ground truth.

Usage:
  python3 train_from_slices.py --slices data/photo_train_passport/cnn_slices.npz \
      --out src/cnn_weights_line1.h --line-index 0 --epochs 200
"""
import json, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--slices", required=True, help="cnn_slices.npz from export_photo_train.py")
ap.add_argument("--out", required=True, help="output C header file")
ap.add_argument("--line-index", type=int, choices=[0, 1], default=-1,
                help="train only line 1 (0) or line 2 (1); -1 mixes both")
ap.add_argument("--epochs", type=int, default=100)
ap.add_argument("--val-frac", type=float, default=0.15)
ap.add_argument("--mix-synthetic", type=int, default=1000,
                help="blend N synthetic v2-renderer samples per class")
ap.add_argument("--lr", type=float, default=0.01)
ap.add_argument("--augment", type=int, default=1,
                help="apply online augmentation to real slices (0=off)")
a = ap.parse_args()

SRC = Path(__file__).resolve().parents[1]
GLYPHS = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ<")
GI = {c: i for i, c in enumerate(GLYPHS)}

d = np.load(a.slices)
X_all = d['X']
Y_all = d['Y']
meta_all = [json.loads(m) for m in d['meta']]

if a.line_index >= 0:
    mask = np.array([m['line'] == a.line_index for m in meta_all])
    X_all = X_all[mask]
    Y_all = Y_all[mask]
    meta_all = [m for i, m in enumerate(meta_all) if mask[i]]

files = sorted(set(m['file'] for m in meta_all))
np.random.seed(42)
np.random.shuffle(files)
n_val = max(1, int(len(files) * a.val_frac))
val_files = set(files[:n_val])

tr_idx = [i for i, m in enumerate(meta_all) if m['file'] not in val_files]
val_idx = [i for i, m in enumerate(meta_all) if m['file'] in val_files]

Xtr = torch.from_numpy(X_all[tr_idx].astype(np.float32).reshape(-1, 1, 12, 16))
Ytr = torch.tensor(Y_all[tr_idx])
Xval = torch.from_numpy(X_all[val_idx].astype(np.float32).reshape(-1, 1, 12, 16))
Yval = torch.tensor(Y_all[val_idx])

print(f"slices: {len(X_all)} (line_index={a.line_index})", flush=True)
print(f"train: {len(Xtr)} from {len(files) - len(val_files)} images", flush=True)
print(f"val:   {len(Xval)} from {len(val_files)} images", flush=True)

if a.mix_synthetic > 0:
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parent))
    import train_cnn_v2 as v2
    Xsyn, Ysyn = v2.build_dataset(n_per_class=a.mix_synthetic, hard_extra=a.mix_synthetic // 2)
    Xsyn_t = torch.from_numpy(Xsyn.reshape(-1, 1, 12, 16))
    Ysyn_t = torch.from_numpy(Ysyn.astype(np.int64))
    print(f"mixed {Xsyn.shape[0]} synthetic -> train {len(Xtr) + Xsyn.shape[0]}", flush=True)
else:
    Xsyn_t = Ysyn_t = None


def augment_batch(xb):
    """Vectorized online augmentation: rotation + translation + noise.

    Glyphs are 12x16. Transforms are subtle to preserve character shape:
    rotation ±3°, translation ±1px, Gaussian + salt/pepper noise.
    """
    import math
    out = xb.clone()
    N = out.shape[0]

    # Rotation (±3°) — 30% of samples, vectorized via affine_grid
    rot_mask = torch.rand(N) < 0.3
    if rot_mask.any():
        n = rot_mask.sum().item()
        angles = (torch.rand(n) - 0.5) * 6.0
        theta = angles * math.pi / 180.0
        cos_t, sin_t = torch.cos(theta), torch.sin(theta)
        aff = torch.zeros(n, 2, 3)
        aff[:, 0, 0] = cos_t;  aff[:, 0, 1] = -sin_t
        aff[:, 1, 0] = sin_t;  aff[:, 1, 1] = cos_t
        sub = out[rot_mask]
        grid = torch.nn.functional.affine_grid(aff, sub.shape, align_corners=False)
        out[rot_mask] = torch.nn.functional.grid_sample(
            sub, grid, padding_mode='border', align_corners=False)

    # Translation (±1px) — 30% of samples
    shift_mask = torch.rand(N) < 0.3
    if shift_mask.any():
        n = shift_mask.sum().item()
        tx = torch.randint(-1, 2, (n,)).float() * (2.0 / 15.0)
        ty = torch.randint(-1, 2, (n,)).float() * (2.0 / 11.0)
        aff = torch.zeros(n, 2, 3)
        aff[:, 0, 0] = 1.0;  aff[:, 1, 1] = 1.0
        aff[:, 0, 2] = tx;   aff[:, 1, 2] = ty
        sub = out[shift_mask]
        grid = torch.nn.functional.affine_grid(aff, sub.shape, align_corners=False)
        out[shift_mask] = torch.nn.functional.grid_sample(
            sub, grid, padding_mode='border', align_corners=False)

    # Gaussian noise — 40% of samples
    noise_mask = torch.rand(out.shape[0]) < 0.4
    if noise_mask.any():
        n = noise_mask.sum().item()
        out[noise_mask] += torch.randn(n, 1, 12, 16) * 0.03

    # Salt/pepper noise — 20% of samples
    flip_mask = torch.rand(out.shape[0]) < 0.2
    if flip_mask.any():
        pmask = torch.rand(flip_mask.sum().item(), 1, 12, 16) < 0.05
        sub = out[flip_mask]
        sub[pmask] = 1.0 - sub[pmask]
        out[flip_mask] = sub

    return torch.clamp(out, 0.0, 1.0)


C1, C2, C3, HID = 8, 16, 4, 64

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, C1, 3, padding=1)
        self.conv2 = nn.Conv2d(C1, C2, 3, padding=1)
        self.conv3 = nn.Conv2d(C2, C3, 1)
        self.fc1 = nn.Linear(C3 * 3 * 8, HID)
        self.fc2 = nn.Linear(HID, 37)

    def forward(self, x):
        x = F.relu(self.conv1(x)); x = F.max_pool2d(x, (2, 1))
        x = F.relu(self.conv2(x)); x = F.max_pool2d(x, (2, 2))
        x = self.conv3(x); x = x.flatten(1)
        x = F.relu(self.fc1(x)); return self.fc2(x)

torch.manual_seed(1)
model = Net()
opt = torch.optim.Adam(model.parameters(), lr=a.lr)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
bs = 512
best = 0.0; best_state = None

if Xsyn_t is not None:
    Xtr_full = torch.cat([Xtr, Xsyn_t])
    Ytr_full = torch.cat([Ytr, Ysyn_t])
else:
    Xtr_full, Ytr_full = Xtr, Ytr

n_real = len(Xtr)
print(f"total train: {len(Xtr_full)} (real={n_real}, syn={len(Xtr_full) - n_real})", flush=True)

for ep in range(a.epochs):
    model.train()
    perm = torch.randperm(len(Xtr_full))
    ctr = tot = 0
    for s in range(0, len(perm), bs):
        idx = perm[s:s + bs]
        xb, yb = Xtr_full[idx], Ytr_full[idx]
        if a.augment:
            real_mask = idx < n_real
            if real_mask.any():
                xb_real = xb[real_mask]
                xb_aug = augment_batch(xb_real)
                xb = xb.clone()
                xb[real_mask] = xb_aug
        lg = model(xb); loss = F.cross_entropy(lg, yb)
        opt.zero_grad(); loss.backward(); opt.step()
        ctr += int((lg.argmax(1) == yb).sum()); tot += len(idx)
    sched.step()
    model.eval()
    with torch.no_grad():
        val = 0
        for s in range(0, len(Xval), bs):
            v = model(Xval[s:s + bs])
            val += int((v.argmax(1) == Yval[s:s + bs]).sum())
        va = val / len(Yval) if len(Yval) > 0 else 0
    if va > best:
        best = va; best_state = {k: v.detach().cpu().numpy().copy() for k, v in model.state_dict().items()}
    if ep % 10 == 0 or ep == a.epochs - 1:
        print(f"ep{ep:3d} tr={ctr/tot:.4f} val={va:.4f} best={best:.4f} lr={opt.param_groups[0]['lr']:.5f}", flush=True)

print(f"best val: {best:.4f}", flush=True)

def emit(best_state, path):
    W1 = best_state['conv1.weight'].transpose(2, 3, 1, 0)
    b1 = best_state['conv1.bias']
    W2 = best_state['conv2.weight'].transpose(2, 3, 1, 0)
    b2 = best_state['conv2.bias']
    W3 = best_state['conv3.weight'].transpose(2, 3, 1, 0)
    b3 = best_state['conv3.bias']
    Wf1 = best_state['fc1.weight']; bf1 = best_state['fc1.bias']
    Wf2 = best_state['fc2.weight']; bf2 = best_state['fc2.bias']
    mean = np.zeros(12 * 16, np.float32); std = np.ones(12 * 16, np.float32)
    def fmt(a): return ", ".join(f"{v:.8f}f" for v in np.asarray(a).ravel())
    prefix = "DEFAULT" if a.line_index < 0 else f"LINE{a.line_index + 1}"
    guard = f"MRZ_OCR_CNN_WEIGHTS_{prefix}_H"
    out = [f"/* Auto-generated from cnn_slices.npz, line_index={a.line_index}. Do not edit. */",
           f"#ifndef {guard}", f"#define {guard}", "",
           "#include <stdint.h>", "#include \"cnn.h\"", "",
           f"static const float {prefix}_IN_MEAN[CNN_IN_H * CNN_IN_W] = {{"]
    for i in range(0, len(mean), 8): out.append("    " + fmt(mean[i:i + 8]) + ",")
    out.append("};"); out.append(f"static const float {prefix}_IN_STD[CNN_IN_H*CNN_IN_W] = {{")
    for i in range(0, len(std), 8): out.append("    " + fmt(std[i:i + 8]) + ",")
    out.append("};")
    out.append(f"static const float {prefix}_CONV1_W[CNN_K*CNN_K*1*CNN_C1] = {{" +
               ",".join(fmt(W1.ravel()[i:i + 8]) for i in range(0, 9 * C1, 8)) + ",};")
    out.append(f"static const float {prefix}_CONV1_B[CNN_C1] = {{" + fmt(b1) + "};")
    out.append(f"static const float {prefix}_CONV2_W[CNN_K*CNN_K*CNN_C1*CNN_C2] = {{" +
               ",".join(fmt(W2.ravel()[i:i + 8]) for i in range(0, 9 * C1 * C2, 8)) + ",};")
    out.append(f"static const float {prefix}_CONV2_B[CNN_C2] = {{" + fmt(b2) + "};")
    out.append(f"static const float {prefix}_CONV3_W[CNN_C2*CNN_C3] = {{" + fmt(W3) + "};")
    out.append(f"static const float {prefix}_CONV3_B[CNN_C3] = {{" + fmt(b3) + "};")
    out.append(f"static const float {prefix}_FC1_W[CNN_FLAT*CNN_HIDDEN] = {{" +
               ",".join(fmt(Wf1.ravel()[i:i + 8]) for i in range(0, Wf1.size, 8)) + ",};")
    out.append(f"static const float {prefix}_FC1_B[CNN_HIDDEN] = {{" + fmt(bf1) + "};")
    out.append(f"static const float {prefix}_FC2_W[CNN_HIDDEN*CNN_OUT] = {{" +
               ",".join(fmt(Wf2.ravel()[i:i + 8]) for i in range(0, HID * 37, 8)) + ",};")
    out.append(f"static const float {prefix}_FC2_B[CNN_OUT] = {{" + fmt(bf2) + "};")
    out.append("#endif")
    Path(path).write_text("\n".join(out) + "\n")
    print("wrote", path, "best_val", best, flush=True)

emit(best_state, a.out)
