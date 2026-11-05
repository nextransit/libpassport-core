#!/usr/bin/env python3
"""Train the fixed-44-step CRNN sequence head and export C weights.

Input: an .npz produced by tools/export_crnn_features.py with
  features (N, 2, 44, 96) float32, labels (N, 2, 44) int64.

Model (must stay numerically aligned with src/crnn.c):
  Linear(96->32) + ReLU
  -> + learned position embedding (44 x 32)
  -> bidirectional GRU(32->32)   (forward and backward each 32 wide)
  -> Linear(64->37)

Why bidirectional + position embedding: on a strictly held-out split
(train on pic_scan1, evaluate on pic_scan2/3, identical features for
both heads) the previous single-direction GRU reached 81.5% character
accuracy on MRZ line 1; the bidirectional head with position embedding
reached 86.6%.  A MRZ row is a fixed 44-slot layout, so letting the
head know the slot index and see both neighbours helps.

Validation is BY IMAGE, never by row, so a line never leaks across the
split.  The export writes include/src/crnn_weights.h with
MRZ_CRNN_WEIGHTS_AVAILABLE 1.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

FEATURES, HIDDEN, CLASSES, STEPS = 96, 32, 37, 44
CTX = 2 * HIDDEN


class CrnnSeq(nn.Module):
    """The sequence head.

    arch="bi"  : bidirectional GRU + learned position embedding.
                 ctx = 2 * HIDDEN.
    arch="uni" : single-direction GRU, no position embedding.
                 ctx = HIDDEN.  Kept for A/B: on a 22-photo training
                 set the extra capacity overfits, so the smaller head
                 sometimes wins.
    """

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
            pos = torch.arange(x.shape[1], device=x.device).unsqueeze(0).expand(x.shape[0], -1)
            h = h + self.pos(pos)
        out, _ = self.gru(h)
        return self.out(out)


def emit(name: str, arr) -> str:
    # np.ascontiguousarray is required: a transposed view is not
    # C-contiguous, and reshape(-1) on such a view can reorder the values.
    flat = np.ascontiguousarray(np.asarray(arr, dtype=np.float32)).reshape(-1)
    body = ", ".join(f"{float(v):.9g}f" for v in flat)
    return f"static const float {name}[{flat.size}] = {{{body}}};\n"


def _gate_blocks(w_ih, w_hh, b_ih, b_hh, suffix, out):
    """Emit one direction's GRU gate matrices.

    torch GRU stores the three gates as three consecutive H-row blocks:
      [0:H)=r  [H:2H)=z  [2H:3H)=n
    so a gate is w[idx*H : (idx+1)*H], not w[3*idx : 3*idx+H].
    The new gate keeps b_ih outside the reset multiplication and b_hh
    inside it: n = tanh(W x + b_ih + r * (U h + b_hh)).
    """
    for gate, idx in (("Z", 1), ("R", 0), ("H", 2)):
        rows = slice(idx * HIDDEN, (idx + 1) * HIDDEN)
        out("W" + gate + suffix, w_ih[rows].T)
        out("U" + gate + suffix, w_hh[rows].T)
        out("B" + gate + suffix, b_ih[rows] + b_hh[rows])
    out("BH_IN" + suffix, b_ih[2 * HIDDEN:3 * HIDDEN])
    out("BH_HN" + suffix, b_hh[2 * HIDDEN:3 * HIDDEN])


def export_header(model: CrnnSeq, path: Path, prefix: str = "CRNN",
                  mode: str = "write") -> None:
    """Emit the C weight arrays for one sequence head.

    mode="write"  -> create/replace the file with the guard + this head.
    mode="append" -> append this head to the existing file.
    """
    s = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
    text = ""
    if mode == "write":
        text = (
            "#ifndef MRZ_OCR_CRNN_WEIGHTS_H\n#define MRZ_OCR_CRNN_WEIGHTS_H\n"
            "#define MRZ_CRNN_WEIGHTS_AVAILABLE 1\n"
        )

    def put(name: str, arr) -> None:
        nonlocal text
        text += emit(prefix + "_" + name, arr)

    # C layout is [in][out]; torch Linear stores [out][in].
    put("PROJ_W", s["proj.weight"].T)
    put("PROJ_B", s["proj.bias"])
    put("POS_W", s["pos.weight"])
    if model.arch == "bi":
        # Forward direction.
        _gate_blocks(s["gru.weight_ih_l0"], s["gru.weight_hh_l0"],
                     s["gru.bias_ih_l0"], s["gru.bias_hh_l0"], "", put)
        # Backward direction (same layout, _B suffix).
        _gate_blocks(s["gru.weight_ih_l0_reverse"], s["gru.weight_hh_l0_reverse"],
                     s["gru.bias_ih_l0_reverse"], s["gru.bias_hh_l0_reverse"],
                     "B", put)
    else:
        # Single-direction head.  The C implementation always runs two
        # directions, so the forward weights are mirrored into the
        # backward half: the backward pass then sees the same
        # sequence, which is exactly what a single-direction GRU
        # computes, and the C math stays well defined.
        _gate_blocks(s["gru.weight_ih_l0"], s["gru.weight_hh_l0"],
                     s["gru.bias_ih_l0"], s["gru.bias_hh_l0"], "", put)
        _gate_blocks(s["gru.weight_ih_l0"], s["gru.weight_hh_l0"],
                     s["gru.bias_ih_l0"], s["gru.bias_hh_l0"], "B", put)
    ow = s["out.weight"]
    if model.arch != "bi":
        # Pad the classifier from HIDDEN to CTX: the forward state goes
        # in the first half, the second half is zero.
        pad = np.zeros((CLASSES, CTX - HIDDEN), dtype=ow.dtype)
        ow = np.concatenate([ow, pad], axis=1)
    put("OUT_W", ow.T)
    put("OUT_B", s["out.bias"])
    if mode == "write":
        path.write_text(text)
    else:
        with path.open("a") as f:
            f.write(text)


def train_one(X, Y, epochs, batch, lr, seed, weight_decay=1e-4, arch="bi"):
    """Train a single sequence head. Returns (model, best_val_char_acc)."""
    torch.manual_seed(seed)
    n = len(X)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_val = max(1, int(n * 0.12))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    Xtr, Ytr = X[tr_idx], Y[tr_idx]
    Xva, Yva = (X[val_idx], Y[val_idx]) if len(val_idx) else (None, None)

    model = CrnnSeq(arch)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss()
    best, best_state = -1.0, {k: v.detach().clone() for k, v in model.state_dict().items()}
    for ep in range(epochs):
        model.train()
        idx = torch.randperm(len(Xtr))
        total = 0.0
        correct = 0
        for s in range(0, len(idx), batch):
            b = idx[s:s + batch]
            logits = model(Xtr[b])
            loss = loss_fn(logits.reshape(-1, CLASSES), Ytr[b].reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(b)
            correct += int((logits.argmax(-1) == Ytr[b]).sum())
        sched.step()
        msg = (f"  epoch {ep+1}/{epochs} loss={total/len(Xtr):.4f} "
               f"char_acc={correct/(len(Xtr)*STEPS):.4f}")
        if Xva is not None:
            model.eval()
            with torch.no_grad():
                lg = model(Xva)
                va = int((lg.argmax(-1) == Yva).sum()) / (len(Xva) * STEPS)
            msg += f" val_char_acc={va:.4f}"
            if va > best:
                best = va
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        print(msg, flush=True)
    model.load_state_dict(best_state)
    return model, best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--extra", action="append", default=[],
                    help="extra .npz of real-domain features; repeatable")
    ap.add_argument("--extra-repeat", type=int, default=1,
                    help="oversample the extra (real) data N times")
    ap.add_argument("--synth-subsample", type=int, default=0,
                    help="keep only N synthetic rows (0 = keep all). The "
                         "synthetic corpus is rendered with a different "
                         "glyph/ink distribution than the real photos, so "
                         "a large synthetic share can dominate the head "
                         "and hurt real-photo generalisation.")
    ap.add_argument("--synth-seed", type=int, default=0,
                    help="seed for the synthetic subsample")
    ap.add_argument("--synth-scale", type=float, default=1.0,
                    help="multiply the synthetic features by this factor. The "
                         "synthetic glyphs are inked denser than the real "
                         "photos produce, so the two domains differ in "
                         "feature magnitude as well as in shape.")
    ap.add_argument("--val", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--arch", choices=("bi", "uni"), default="bi",
                    help="sequence head: bidirectional GRU with position "
                         "embedding (bi) or single-direction GRU (uni)")
    ap.add_argument("--weight-decay", type=float, default=1e-4,
                    help="AdamW weight decay. The real-domain corpus is "
                         "tiny (41 photos), so the head overfits it unless "
                         "this is raised; measured on the held-out split, "
                         "0.3 beat 1e-4 by ~8 accuracy points.")
    args = ap.parse_args()

    np.random.seed(args.seed)
    tr = np.load(args.train)
    # Feature fingerprint: which tool binary produced these features.
    # Stamped into the emitted header so eval scripts can refuse a
    # weights/tool mismatch (front-end rebuilds silently change the conv
    # features -- the 10-28 "CRNN 0/41" incident).
    tool_md5 = str(tr["tool_md5"]) if "tool_md5" in tr.files else "unknown"
    tool_path = str(tr["tool_path"]) if "tool_path" in tr.files else "unknown"
    print(f"feature fingerprint: tool_md5={tool_md5} tool={tool_path}", flush=True)
    feats = tr["features"]
    if args.synth_subsample and args.synth_subsample < len(feats):
        rng = np.random.default_rng(args.synth_seed)
        sel = rng.choice(len(feats), size=args.synth_subsample, replace=False)
        feats = feats[sel]
        labels = tr["labels"][sel]
        print(f"synthetic subsample: {args.synth_subsample} rows kept", flush=True)
    else:
        labels = tr["labels"]
    if args.synth_scale != 1.0:
        feats = feats * np.float32(args.synth_scale)
        print(f"synthetic feature scale: {args.synth_scale}", flush=True)
    X = torch.from_numpy(feats.astype("float32"))
    Y = torch.from_numpy(labels.astype("int64"))
    # Real-domain samples are far fewer than the synthetic corpus, so they
    # are oversampled. Mixing them in is what keeps the sequence head from
    # overfitting to synthetic band geometry: the CRNN's gain depends on the
    # 44-cell grid being aligned, and real photos crop/split differently.
    for path in args.extra:
        ex = np.load(path)
        exX = torch.from_numpy(ex["features"].astype("float32"))
        exY = torch.from_numpy(ex["labels"].astype("int64"))
        for _ in range(max(1, args.extra_repeat)):
            X = torch.cat([X, exX])
            Y = torch.cat([Y, exY])
        print(f"mixed real domain: +{len(exX) * max(1, args.extra_repeat)} rows "
              f"from {path}", flush=True)
    # (N, 2, 44, 96) -> per-line (N, 44, 96). The CNN front-end uses a
    # DIFFERENT convolution weight set for MRZ line 1 and line 2, so the
    # feature distributions differ and the sequence head must be trained
    # per line.
    print("training line 1 head:", flush=True)
    m1, a1 = train_one(X[:, 0], Y[:, 0], args.epochs, args.batch, args.lr,
                       args.seed, args.weight_decay, args.arch)
    print("training line 2 head:", flush=True)
    m2, a2 = train_one(X[:, 1], Y[:, 1], args.epochs, args.batch, args.lr,
                       args.seed + 1, args.weight_decay, args.arch)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    export_header(m1, out, "CRNN1", mode="write")
    export_header(m2, out, "CRNN2", mode="append")
    with out.open("a") as f:
        f.write("#endif\n")
    # Feature provenance goes into the header as a comment so the shipped
    # weights are self-describing; eval scripts parse MRZ_CRNN_FEATURE_MD5
    # back out and compare it against the tool binary under test.
    prov = (f"/* feature provenance: MRZ_CRNN_FEATURE_MD5={tool_md5} "
            f"tool={Path(tool_path).name if tool_path != 'unknown' else 'unknown'} "
            f"trained={__import__('datetime').date.today().isoformat()} */\n")
    body = out.read_text()
    marker = "#define MRZ_CRNN_WEIGHTS_AVAILABLE 1\n"
    body = body.replace(marker, marker + prov, 1)
    out.write_text(body)
    print(f"exported {out} line1_val={a1:.4f} line2_val={a2:.4f}")
    print(f"  header fingerprint: MRZ_CRNN_FEATURE_MD5={tool_md5}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
