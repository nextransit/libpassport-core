#!/usr/bin/env python3
"""Train the Tiny CNN on SYNTHETIC v2 renderer + REAL photo glyph slices.

Rationale:
  * Pure-synthetic weights (src/cnn_weights.h) score ~95%/98% on the
    frozen eval corpus but ~54%/10% on real passport band crops.
  * Pure-real training destroys the synthetic distribution (eval
    collapsed to 55%/17% in a full-replacement run).
  * Mixing real glyphs (weak labels from board-side fixmrz) at a
    controlled weight keeps the synthetic distribution intact while
    nudging the classifier toward the real-band input domain.

Usage:
  python3 tools/train_cnn_mixed.py --epochs 40 --out /tmp/cnn_mixed.h
"""
from __future__ import annotations
import argparse, random
from pathlib import Path
import numpy as np

import train_cnn_v2 as v2
import torch
import torch.nn as nn
import torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument("--epochs", type=int, default=40)
ap.add_argument("--n-per-class", type=int, default=1500)
ap.add_argument("--hard-extra", type=int, default=750)
ap.add_argument("--real-npz", default="data/real_slices.npz")
ap.add_argument("--real-frac", type=float, default=0.30,
                help="real glyphs as fraction of the train set (0..1)")
ap.add_argument("--out", default="/tmp/cnn_mixed.h")
a = ap.parse_args()

C1, C2, C3, HID = 8, 16, 4, 64
IN_H, IN_W = 12, 16
NUM_CLASSES = 37

torch.manual_seed(1); np.random.seed(1); random.seed(1)

# ---- synthetic renderer ----
Xs, Ys = v2.build_dataset(n_per_class=a.n_per_class, hard_extra=a.hard_extra)
Xs = Xs.reshape(-1, 1, IN_H, IN_W).astype(np.float32)
Ys = Ys.astype(np.int64)
print(f"synthetic: {Xs.shape[0]}", flush=True)

# ---- real slices ----
d = np.load(a.real_npz)
Xr = d["X"].astype(np.float32).reshape(-1, 1, IN_H, IN_W)
Yr = d["Y"].astype(np.int64)
print(f"real: {Xr.shape[0]}", flush=True)

# target: real_frac of TRAIN samples (val stays synthetic-only to keep
# the gate honest)
n_train_syn = int(len(Ys) * 0.95)
perm = torch.randperm(len(Ys))
syn_tr_idx, syn_val_idx = perm[:n_train_syn], perm[n_train_syn:]
Xtr_syn, Ytr_syn = torch.from_numpy(Xs[syn_tr_idx]), torch.from_numpy(Ys[syn_tr_idx])
Xval, Yval = torch.from_numpy(Xs[syn_val_idx]), torch.from_numpy(Ys[syn_val_idx])

# upsample real to real_frac of the synthetic train count
n_real_target = int(len(Ytr_syn) * a.real_frac)
reps = (n_real_target + len(Yr) - 1) // len(Yr)
Xr_r = torch.from_numpy(np.repeat(Xr, reps, axis=0)[:n_real_target])
Yr_r = torch.from_numpy(np.repeat(Yr, reps)[:n_real_target])
print(f"train syn={len(Ytr_syn)} real={len(Yr_r)} val(syn)={len(Yval)} "
      f"real_frac={len(Yr_r)/(len(Ytr_syn)+len(Yr_r)):.2f}", flush=True)

Xall = torch.cat([Xtr_syn, Xr_r])
Yall = torch.cat([Ytr_syn, Yr_r])

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, C1, 3, padding=1)
        self.conv2 = nn.Conv2d(C1, C2, 3, padding=1)
        self.conv3 = nn.Conv2d(C2, C3, 1)
        self.fc1 = nn.Linear(C3 * 3 * 8, HID)
        self.fc2 = nn.Linear(HID, NUM_CLASSES)
    def forward(self, x):
        x = F.relu(self.conv1(x)); x = F.max_pool2d(x, (2, 1))
        x = F.relu(self.conv2(x)); x = F.max_pool2d(x, (2, 2))
        x = self.conv3(x); x = x.flatten(1)
        x = F.relu(self.fc1(x)); return self.fc2(x)

model = Net()
opt = torch.optim.Adam(model.parameters(), lr=0.02)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
bs = 512
best = 0.0; best_state = None
for ep in range(a.epochs):
    model.train()
    perm_tr = torch.randperm(len(Xall))
    correct = tot = 0
    for s in range(0, len(perm_tr), bs):
        idx = perm_tr[s:s + bs]
        lg = model(Xall[idx]); loss = F.cross_entropy(lg, Yall[idx])
        opt.zero_grad(); loss.backward(); opt.step()
        correct += int((lg.argmax(1) == Yall[idx]).sum()); tot += len(idx)
    sched.step()
    model.eval()
    with torch.no_grad():
        va = 0
        for s in range(0, len(Xval), bs):
            v = model(Xval[s:s + bs]); va += int((v.argmax(1) == Yval[s:s + bs]).sum())
        val_acc = va / len(Yval)
    if val_acc > best:
        best = val_acc
        best_state = {k: v.detach().cpu().numpy().copy() for k, v in model.state_dict().items()}
    print(f"ep{ep:2d} tr={correct/tot:.4f} val={val_acc:.4f} best={best:.4f}", flush=True)

v2.emit_header(
    (best_state["conv1.weight"].transpose(2,3,1,0), best_state["conv1.bias"],
     best_state["conv2.weight"].transpose(2,3,1,0), best_state["conv2.bias"],
     best_state["conv3.weight"].transpose(2,3,1,0), best_state["conv3.bias"],
     best_state["fc1.weight"], best_state["fc1.bias"],
     best_state["fc2.weight"], best_state["fc2.bias"],
     np.zeros(IN_H*IN_W, np.float32), np.ones(IN_H*IN_W, np.float32),
     best), a.out)
print("wrote", a.out, "best_val", best, flush=True)
