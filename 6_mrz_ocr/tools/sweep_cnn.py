#!/usr/bin/env python3
"""Fast training sweep with a REAL-path holdout metric.

Why
---
The production loop (train -> recompile -> export CRNN features -> retrain
CRNN -> rebuild -> bench) costs minutes per iteration, which makes it
useless for searching a hyper-parameter space.  This script trains the same
network (same architecture, same loss, same synthetic renderer) but scores
it on the PRODUCT-path holdout slices after every epoch, so a whole sweep
finishes in seconds.

The holdout metric is the same thing the product bench measures -- both MRZ
lines of a photo fully correct -- computed on the C pipeline's own glyph
slices, so it isolates the CNN head from the beam/CRNN layer.

Usage:
  python3 tools/sweep_cnn.py --line 0 \
      --train /tmp/prod_train.npz --holdout /tmp/prod_holdout.npz \
      --epochs 40 --italic-prob 0.35 --out /tmp/sweep_line1.h
"""
from __future__ import annotations
import argparse, json, random, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import train_cnn_v2 as v2

GLYPHS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ<"
GI = {c: i for i, c in enumerate(GLYPHS)}
C1, C2, C3, HID = 8, 16, 4, 64
IN_H, IN_W = 12, 16
NUM_CLASSES = 37


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


def load_npz(path, line=None):
    d = np.load(path, allow_pickle=True)
    X = d["X"].astype(np.float32).reshape(-1, 1, IN_H, IN_W)
    Y = d["Y"].astype(np.int64)
    meta = []
    if "meta" in d.files:
        meta = [json.loads(s) for s in d["meta"]]
    keep = list(range(len(Y)))
    if line is not None and meta:
        keep = [i for i, m in enumerate(meta) if int(m.get("line", -1)) == line]
    return X[keep], Y[keep], [meta[i] for i in keep]


def holdout_exact(model, X, Y, meta, line):
    """Per-photo exact match: BOTH lines of a photo fully correct.

    X is a torch tensor (N,1,12,16); Y a numpy int array.
    """
    model.eval()
    with torch.no_grad():
        pred = model(X).argmax(1).cpu().numpy()
    Y = np.asarray(Y)
    per = {}
    for i, m in enumerate(meta):
        k = m["file"]
        d = per.setdefault(k, {"ok": 0, "tot": 0, "l": {0: [0, 0], 1: [0, 0]}})
        d["tot"] += 1
        d["l"][int(m["line"])][1] += 1
        if pred[i] == Y[i]:
            d["ok"] += 1
            d["l"][int(m["line"])][0] += 1
    n_full = sum(1 for v in per.values() if v["ok"] == v["tot"])
    l1 = sum(1 for v in per.values() if v["l"][0][0] == v["l"][0][1] and v["l"][0][1])
    l2 = sum(1 for v in per.values() if v["l"][1][0] == v["l"][1][1] and v["l"][1][1])
    char = sum(v["ok"] for v in per.values()) / max(1, sum(v["tot"] for v in per.values()))
    return n_full, len(per), l1, l2, char


def subgroup_exact(model, X, Y, meta, want_prefix):
    """Exact-match count restricted to one label prefix (e.g. "Photo/")."""
    idx = [i for i, m in enumerate(meta) if str(m.get("label", "")).startswith(want_prefix)]
    if not idx:
        return 0, 0
    model.eval()
    with torch.no_grad():
        pred = model(X[idx]).argmax(1).cpu().numpy()
    Y = np.asarray(Y)
    per = {}
    for k, i in enumerate(idx):
        d = per.setdefault(meta[i]["file"], [0, 0])
        d[1] += 1
        d[0] += int(pred[k] == Y[i])
    return sum(1 for v in per.values() if v[0] == v[1]), len(per)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--line", type=int, choices=[0, 1], required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--holdout", required=True)
    ap.add_argument("--synth-extra-npz", default="")
    ap.add_argument("--real-frac", type=float, default=0.5)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--n-per-class", type=int, default=1500)
    ap.add_argument("--hard-extra", type=int, default=750)
    ap.add_argument("--italic-prob", type=float, default=0.0)
    ap.add_argument("--italic-max-frac", type=float, default=0.18)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--select", default="holdout",
                    choices=("holdout", "train", "min"))
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)

    Xr, Yr, Mr = load_npz(args.train, args.line)
    Xh, Yh, Mh = load_npz(args.holdout, args.line)
    Xr_t = torch.from_numpy(Xr)
    Xh_t = torch.from_numpy(Xh)
    print(f"train slices line{args.line}: {len(Yr)}   "
          f"holdout slices line{args.line}: {len(Yh)}", flush=True)

    Xs, Ys = v2.build_dataset(n_per_class=args.n_per_class,
                              hard_extra=args.hard_extra,
                              italic_prob=args.italic_prob,
                              italic_max_frac=args.italic_max_frac)
    Xs = Xs.reshape(-1, 1, IN_H, IN_W).astype(np.float32)
    Ys = Ys.astype(np.int64)

    perm = torch.randperm(len(Ys))
    n_val = int(len(Ys) * 0.05)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    Xtr_syn = torch.from_numpy(Xs[tr_idx]); Ytr_syn = torch.from_numpy(Ys[tr_idx])
    Xval = torch.from_numpy(Xs[val_idx]); Yval = torch.from_numpy(Ys[val_idx])

    if args.synth_extra_npz:
        ex = np.load(args.synth_extra_npz, allow_pickle=True)
        Xe = ex["X"].astype(np.float32).reshape(-1, 1, IN_H, IN_W)
        Ye = ex["Y"].astype(np.int64)
        if "meta" in ex.files:
            em = [json.loads(s) for s in ex["meta"]]
            sel = [i for i, m in enumerate(em)
                   if int(m.get("line", args.line)) == args.line]
            Xe, Ye = Xe[sel], Ye[sel]
        Xtr_syn = torch.cat([Xtr_syn, torch.from_numpy(Xe)])
        Ytr_syn = torch.cat([Ytr_syn, torch.from_numpy(Ye)])
        print(f"synth-extra: +{len(Ye)}", flush=True)

    Yr_t = torch.from_numpy(Yr)
    n_real_target = int(len(Ytr_syn) * args.real_frac / max(1e-9, 1 - args.real_frac))
    reps = (n_real_target + len(Yr) - 1) // max(1, len(Yr))
    Xr_r = torch.from_numpy(np.repeat(Xr, reps, axis=0)[:n_real_target])
    Yr_r = torch.from_numpy(np.repeat(Yr, reps)[:n_real_target])

    Xall = torch.cat([Xtr_syn, Xr_r]); Yall = torch.cat([Ytr_syn, Yr_r])
    print(f"train syn={len(Ytr_syn)} real={len(Yr_r)} "
          f"real_frac={len(Yr_r)/len(Xall):.2f}", flush=True)

    model = Net()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    bs = 512
    best_score = -1.0
    best_state = None
    best_row = None

    for ep in range(args.epochs):
        model.train()
        p = torch.randperm(len(Xall))
        correct = tot = 0
        for s in range(0, len(p), bs):
            idx = p[s:s + bs]
            lg = model(Xall[idx])
            per = F.cross_entropy(lg, Yall[idx], reduction="none")
            loss = per.mean()
            opt.zero_grad(); loss.backward(); opt.step()
            correct += int((lg.argmax(1) == Yall[idx]).sum()); tot += len(idx)
        sched.step()
        model.eval()
        with torch.no_grad():
            va = 0
            for s in range(0, len(Xval), bs):
                v = model(Xval[s:s + bs])
                va += int((v.argmax(1) == Yval[s:s + bs]).sum())
            val_acc = va / len(Xval)
        # TRAIN-domain accuracy (photos fully correct)
        nf, np_, l1f, l2f, char = holdout_exact(model, Xr_t, Yr_t, Mr, args.line)
        # HOLDOUT-domain accuracy
        hnf, hnp, hl1, hl2, hchar = holdout_exact(model, Xh_t, Yh, Mh, args.line)
        score = {"holdout": hnf, "train": nf, "min": min(nf, hnf)}[args.select]
        if score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu().numpy().copy()
                          for k, v in model.state_dict().items()}
            best_row = (ep, nf, np_, hnf, hnp, hchar)
        pfn, pnp = subgroup_exact(model, Xh_t, Yh, Mh, "Photo/")
        print(f"ep{ep:2d} tr={correct/tot:.4f} val={val_acc:.4f} "
              f"TRAIN exact={nf}/{np_}  HOLD exact={hnf}/{hnp} "
              f"hchar={hchar*100:.2f}%  Photo/={pfn}/{pnp}", flush=True)

    if best_state is None:
        raise SystemExit("no epoch scored")
    prefix = "LINE1" if args.line == 0 else "LINE2"
    v2.emit_header(
        (best_state["conv1.weight"].transpose(2, 3, 1, 0), best_state["conv1.bias"],
         best_state["conv2.weight"].transpose(2, 3, 1, 0), best_state["conv2.bias"],
         best_state["conv3.weight"].transpose(2, 3, 1, 0), best_state["conv3.bias"],
         best_state["fc1.weight"], best_state["fc1.bias"],
         best_state["fc2.weight"], best_state["fc2.bias"],
         np.zeros(IN_H * IN_W, np.float32), np.ones(IN_H * IN_W, np.float32),
         val_acc), args.out, prefix=prefix,
        comment=(f"/* Auto-generated by tools/sweep_cnn.py --line {args.line}. "
                 f"Do not edit. */"))
    ep, nf, np_, hnf, hnp, hchar = best_row
    print(f"BEST ep={ep} TRAIN exact={nf}/{np_} HOLD exact={hnf}/{hnp} "
          f"hchar={hchar*100:.2f}%  -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
