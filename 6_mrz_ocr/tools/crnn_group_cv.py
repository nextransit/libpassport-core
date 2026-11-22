#!/usr/bin/env python3
"""Leave-one-photo-group-out CV for the CRNN sequence head.

The head shipped in src/crnn_weights.h reaches only 28/47 exact on the
held-out real photos and 27/41 on its OWN training slices, so it is
weak in absolute terms, not merely domain-shifted.  This script measures
what a head trained on the SAME features can actually reach, with a
strict photo-group split: no photo group appears in both train and
eval, so a line never leaks across the split.

It answers three questions the A/B on the shipped weights cannot:
  1. how much of the gap is the ARCHITECTURE (bi vs uni GRU)?
  2. how much is the TRAINING SET SIZE (real photos only vs +synthetic)?
  3. is the head limited by the 96-value conv features or by the
     sequence model itself?

Usage:
  python3 tools/crnn_group_cv.py --npz /tmp/crnn_build/corpus_real2.npz \
      [--synth /tmp/crnn_build/corpus_train.npz] [--epochs 120]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

FEATURES, HIDDEN, CLASSES, STEPS = 96, 32, 37, 44
CTX = 2 * HIDDEN


class CrnnSeq(nn.Module):
    """Must stay numerically aligned with src/crnn.c."""

    def __init__(self, arch: str = "bi"):
        super().__init__()
        self.arch = arch
        bidir = (arch == "bi")
        self.proj = nn.Linear(FEATURES, HIDDEN)
        self.pos = nn.Embedding(STEPS, HIDDEN) if bidir else None
        self.gru = nn.GRU(HIDDEN, HIDDEN, batch_first=True, bidirectional=bidir)
        self.out = nn.Linear(CTX if bidir else HIDDEN, CLASSES)

    def forward(self, x):
        h = torch.relu(self.proj(x))
        if self.pos is not None:
            pos = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
            h = h + self.pos(pos.expand(x.shape[0], -1))
        out, _ = self.gru(h)
        return self.out(out)


def group_of(ids: np.ndarray) -> np.ndarray:
    return np.array([str(i).split("/")[0] for i in ids])


def flat(feats: np.ndarray, labels: np.ndarray):
    """(N, 2, 44, F) -> (N*2, 44, F); line0 then line1 per photo.

    The head runs one 44-step row at a time, so the photo axis and the
    line axis merge into the batch axis.
    """
    n, two, steps, f = feats.shape
    return (feats.reshape(n * two, steps, f),
            labels.reshape(n * two, steps))


def evaluate(model, feats, labels) -> tuple:
    """Return (char_acc_l1, char_acc_l2, l1_exact, l2_exact, both_exact, n)."""
    model.eval()
    n = feats.shape[0]
    X = torch.from_numpy(feats.reshape(n * 2, 44, feats.shape[-1]))
    with torch.no_grad():
        pred = model(X).argmax(-1).numpy()
    pred = pred.reshape(n, 2, 44)
    gt = labels
    c1 = c2 = 0
    e1 = e2 = eb = 0
    for i in range(n):
        l1o = pred[i, 0] == gt[i, 0]
        l2o = pred[i, 1] == gt[i, 1]
        c1 += int(l1o.sum())
        c2 += int(l2o.sum())
        e1 += int(l1o.all())
        e2 += int(l2o.all())
        eb += int(l1o.all() and l2o.all())
    cells = n * 44 * 2
    return (c1 / (n * 44), c2 / (n * 44), e1, e2, eb, n)


def train_pair(train_f, train_l, val_f, val_l, arch, epochs, lr, wd, seed):
    """Train ONE head per line, exactly like tools/train_crnn_seq.py.

    The shipped script fits line 1 and line 2 with SEPARATE heads
    (m1/m2).  A shared head over the merged photo axis is a different
    model and does not reproduce the shipped accuracy, so the CV has to
    use the same split.
    """
    models = []
    for line in (0, 1):
        torch.manual_seed(seed + line)
        np.random.seed(seed + line)
        model = CrnnSeq(arch)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        X = torch.from_numpy(train_f[:, line])
        Y = torch.from_numpy(train_l[:, line])
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        n = X.shape[0]
        bs = min(128, n)
        for ep in range(epochs):
            model.train()
            perm = torch.randperm(n)
            for i in range(0, n, bs):
                idx = perm[i:i + bs]
                logits = model(X[idx])
                loss = nn.functional.cross_entropy(
                    logits.reshape(-1, CLASSES), Y[idx].reshape(-1))
                opt.zero_grad()
                loss.backward()
                opt.step()
            sched.step()
        models.append(model)
    return models


def evaluate_pair(models, feats, labels) -> tuple:
    model.eval() if False else None
    n = feats.shape[0]
    preds = []
    for line, model in enumerate(models):
        model.eval()
        with torch.no_grad():
            p = model(torch.from_numpy(feats[:, line])).argmax(-1).numpy()
        preds.append(p)
    pred = np.stack(preds, axis=1)
    gt = labels
    c1 = c2 = 0
    e1 = e2 = eb = 0
    for i in range(n):
        l1o = pred[i, 0] == gt[i, 0]
        l2o = pred[i, 1] == gt[i, 1]
        c1 += int(l1o.sum())
        c2 += int(l2o.sum())
        e1 += int(l1o.all())
        e2 += int(l2o.all())
        eb += int(l1o.all() and l2o.all())
    return (c1 / (n * 44), c2 / (n * 44), e1, e2, eb, n)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="/tmp/crnn_build/corpus_real2.npz")
    ap.add_argument("--synth", default="")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--wd", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--repeat", type=int, default=0,
                    help="oversample the real rows this many times")
    args = ap.parse_args()

    d = np.load(args.npz, allow_pickle=True)
    feats, labels, ids = d["features"], d["labels"], d["ids"]
    groups = group_of(ids)
    uniq = sorted(set(groups.tolist()))
    print(f"real npz: {feats.shape}  groups={ {g: int((groups==g).sum()) for g in uniq} }")

    synth = None
    if args.synth and Path(args.synth).exists():
        s = np.load(args.synth, allow_pickle=True)
        synth = (s["features"], s["labels"])
        print(f"synth npz: {synth[0].shape}")

    rows = []
    for arch in ("bi", "uni"):
        for use_synth in ((False, True) if synth is not None else (False,)):
            for hold in uniq:
                tr = groups != hold
                va = groups == hold
                tf, tl = feats[tr], labels[tr]
                if use_synth:
                    tf = np.concatenate([tf, synth[0]])
                    tl = np.concatenate([tl, synth[1]])
                if args.repeat:
                    tf = np.concatenate([tf] + [feats[tr]] * args.repeat)
                    tl = np.concatenate([tl] + [labels[tr]] * args.repeat)
                models = train_pair(tf, tl, feats[va], labels[va],
                                    arch, args.epochs, args.lr, args.wd,
                                    args.seed)
                c1, c2, e1, e2, eb, n = evaluate_pair(models, feats[va],
                                                     labels[va])
                rows.append((arch, use_synth, hold, n, c1, c2, e1, e2, eb))
                print(f"  {arch:3s} synth={int(use_synth)} hold={hold:10s} "
                      f"n={n:2d} char l1={c1*100:5.1f}% l2={c2*100:5.1f}% "
                      f"exact l1={e1:2d} l2={e2:2d} both={eb:2d}", flush=True)
        sel = [r for r in rows if r[0] == arch and r[1] == use_synth]
        tot_n = sum(r[3] for r in sel)
        # char accuracy is per-fold RATIO -> weight it by the fold size
        wsum = sum(r[3] for r in sel)
        a1 = sum(r[4] * r[3] for r in sel) / wsum
        a2 = sum(r[5] * r[3] for r in sel) / wsum
        e1 = sum(r[6] for r in sel)
        e2 = sum(r[7] for r in sel)
        eb = sum(r[8] for r in sel)
        print(f"== arch={arch} synth={int(use_synth)}: "
              f"char l1={a1*100:.2f}% l2={a2*100:.2f}% | "
              f"exact l1={e1}/{tot_n} l2={e2}/{tot_n} both={eb}/{tot_n}",
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
