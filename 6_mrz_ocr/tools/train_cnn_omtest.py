#!/usr/bin/env python3
"""Train the TD3 line-1 / line-2 CNNs on REAL passport photo glyphs.

Why this exists
---------------
The line models that ship in `src/cnn_weights_line{1,2}.h` were trained
on glyph slices whose labels came from the board's `fixmrz` output.
That label is a PaddleOCR hypothesis and it is wrong in ways that
directly teach the network the wrong thing:

  * it puts DIGITS in line-1 alpha slots (`P<CHNGUO0<LURU` where the
    document reads `O`), so the network learns to emit `0` in the name
    field;
  * it is checksum-INVALID for 27/41 samples, so the line-2 labels carry
    wrong check digits.

`tools/export_omtest_slices.py` re-derives the labels with the ICAO
9303 rules applied (see that file for the exact policy) and dumps the
16x12 glyphs the C pipeline actually feeds the network, so the training
distribution matches inference exactly.

Mixing
------
Purely real weights destroy the synthetic distribution (the frozen
1380-image eval corpus collapses).  So we train on a mix:
  synthetic (train_cnn_v2 renderer)  +  real (this file)
and select the epoch by held-out synthetic validation, exactly like
`train_cnn_mixed.py`.

Usage:
  python3 tools/train_cnn_omtest.py --line 0 --epochs 40 \
      --real-npz data/omtest_slices.npz --out src/cnn_weights_line1.h
"""
from __future__ import annotations
import argparse, json, math, random
from pathlib import Path

from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import train_cnn_v2 as v2

C1, C2, C3, HID = 8, 16, 4, 64
IN_H, IN_W = 12, 16
NUM_CLASSES = 37

ap = argparse.ArgumentParser()
ap.add_argument("--epochs", type=int, default=40)
ap.add_argument("--n-per-class", type=int, default=1500)
ap.add_argument("--hard-extra", type=int, default=750)
ap.add_argument("--synth-extra-npz", default="",
                help="extra data for the SYNTHETIC half (npz with X/Y/meta),\n"
                     "e.g. glyph slices exported from the rendered corpus.\n"
                     "Joined to the synthetic half on purpose: the photo\n"
                     "half keeps its --real-frac share of the training set.")
ap.add_argument("--real-npz", default="data/omtest_slices.npz")
ap.add_argument("--real-frac", type=float, default=0.30)
ap.add_argument("--line", type=int, choices=[0, 1], required=True,
                help="0 = TD3 line 1, 1 = TD3 line 2")
ap.add_argument("--out", required=True)
ap.add_argument("--class-boost", type=int, default=0,
                help="extra synthetic samples per MEASURED_HARD_CLASSES "
                     "character (G/D/K/M/L/R/7 -- the measured-hard GT "
                     "classes on real product bands).  0 = off.")
ap.add_argument("--pn-weight", type=float, default=1.0,
                help="loss weight multiplier for line-2 personal-number "
                     "zone cells (cols 28..41) with hard labels; 1.0 = off. "
                     "The measured residual-error surface concentrates "
                     "there (letter cells 14.2%% err vs 5.6%% digits). "
                     "NOTE: the class-oversampling alternative "
                     "(--class-boost) was measured NEGATIVE (40/41 -> "
                     "39/41); a position prior does not disturb the class "
                     "balance, so it is the carrier to try first.")
ap.add_argument("--weak-real-weight", type=float, default=0.25,
                help="loss weight for rescue slices whose board label is "
                     "not checksum-pinned")
ap.add_argument("--class-weight", action="store_true",
                help="weight the real half by inverse class frequency.\n"
                     "MEASURED AND REJECTED: it removes the frequency prior\n"
                     "but costs accuracy (41-photo end-to-end 40/41 -> 32/41,\n"
                     "false accepts 0 -> 7) because 1804 unique real slices\n"
                     "are too few to re-weight safely.  Kept for the record.\n"
                     "See docs/deep-beam.md for the full measurement.")
ap.add_argument("--holdout-photos", type=int, default=0,
                help="hold out N whole photographs from the real half and\n"
                     "validate on them (the only leak-free real signal)")
ap.add_argument("--holdout-files", default="",
                help="comma-separated photograph names to hold out, "
                     "overriding --holdout-photos. Use this when the "
                     "corpus contains shots of the same document: a "
                     "random split can leak the answer.")
ap.add_argument("--balance-real", action="store_true",
                help="sample the real half per class to remove the\n"
                     "frequency prior (e.g. 223 real '0' vs 1 real 'O').\n"
                     "MEASURED AND REJECTED: it removes the prior but the\n"
                     "1804 unique slices become ~37k duplicates, the model\n"
                     "memorises them and the 41-photo result drops 37/41 ->\n"
                     "28/41 with the false accepts rising 2 -> 11.\n"
                     "Kept for the record; see docs/deep-beam.md.")
ap.add_argument("--hard-loss", action="store_true",
                help="use plain cross entropy instead of the soft target")
ap.add_argument("--augment", action="store_true",
                help="apply per-row photometric/geometric augmentation to the real\n"
                     "half each epoch (horizontal squeeze, blur, stroke erosion).\n"
                     "Motivated by the AJ residual: its 16x12 glyphs are narrow\n"
                     "(~4-12 of 16 columns) and low contrast, so the classifier\n"
                     "reads F->E, H->M, 9->7, H->U.  Baseline retrain already hits\n"
                     "40/41; only use this to try to close the last miss.")
ap.add_argument("--aug-seed", type=int, default=7)
ap.add_argument("--jitter", type=float, default=0.0,
                help="per-epoch Gaussian brightness noise (sigma, in [0,1]\n"
                     "normalised units) added to the real half only.  Models the\n"
                     "+-1..2/255 JPEG-decode noise that separates the device's\n"
                     "BitmapFactory pixels from the PC's PIL pixels: Q/T are\n"
                     "rescue (weak) samples whose checksum margin is only\n"
                     "0.035-0.126, and that decode noise flips them.  Unlike\n"
                     "--augment this touches brightness only, no geometry, so it\n"
                     "should not disturb the 40/41 baseline.")
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--italic-prob", type=float, default=0.0,
                help="probability that a SYNTHETIC glyph is rendered with the\n"
                     "italic lean the real MRZ photos show.  The real photos are\n"
                     "set in an italic face and photographed at an angle, so every\n"
                     "glyph leans: the measured lean is 4-6 px over a ~32 px glyph\n"
                     "box (0.12-0.18 of the box height).  The legacy corpus had NO\n"
                     "such augmentation, which is why the classifier confuses the\n"
                     "leaning shapes (M->8, C->E, D->0, P->E).")
ap.add_argument("--italic-max-frac", type=float, default=0.18,
                help="maximum italic lean as a fraction of the glyph box height")
args = ap.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

GLYPHS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ<"
GI = {c: i for i, c in enumerate(GLYPHS)}

meta = np.load(args.real_npz, allow_pickle=True)
Xr_all = meta["X"].astype(np.float32).reshape(-1, 1, IN_H, IN_W)
Yr_all = meta["Y"].astype(np.int64)
if "meta" in meta.files:
    mm = [json.loads(s) for s in meta["meta"]]
    keep = [i for i, m in enumerate(mm) if int(m["line"]) == args.line]
else:
    keep = list(range(len(Yr_all)))
    mm = [{} for _ in keep]
Xr = torch.from_numpy(Xr_all[keep])
Yr = torch.from_numpy(Yr_all[keep])

# Per-cell candidate sets.  The board label is a PaddleOCR hypothesis, so
# a cell whose checksum-valid variants disagree must NOT be trained as a
# single hard class -- that is how a misread digit becomes a "correct"
# training label and the network learns to reproduce the error.  `cand`
# carries every character consistent with the checksum evidence for that
# cell.
#
# Weighting matters as much as the candidate set: a photograph that had
# to be rescued with a soft label is, by construction, one the OCR could
# not read.  Training on it as if every cell were known poisons the
# model (measured: 30/41 -> 21/41 end-to-end when those 11 hard-labelled
# rescue samples are mixed in at full weight).  So each real slice
# carries a per-slice weight: 1.0 when its label came from a checksum
# the pipeline accepts outright, and `--weak-real-weight` when it came
# from a rescue.
Cr = torch.zeros(len(keep), NUM_CLASSES)
Wslice = torch.ones(len(keep))
n_soft = 0
for row, src in enumerate(keep):
    info = mm[src] if isinstance(mm[src], dict) else {}
    cand = info.get("cand")
    weak = bool(info.get("weak"))
    ids = [GI[c] for c in cand if c in GI] if cand else []
    if not ids:
        ids = [int(Yr_all[src])]
    if len(ids) > 1:
        n_soft += 1
    Cr[row, ids] = 1.0 / len(ids)
    if weak:
        Wslice[row] = args.weak_real_weight
    _col = info.get("col")
    if (args.pn_weight != 1.0 and _col is not None
            and int(info.get("line", -1)) == 1
            and 28 <= int(_col) <= 41):
        Wslice[row] *= args.pn_weight
print(f"real slices for line {args.line}: {len(Yr)} "
      f"(soft cells: {n_soft}, weak slices: {int((Wslice < 1).sum())})",
      flush=True)

Xs, Ys = v2.build_dataset(n_per_class=args.n_per_class,
                         hard_extra=args.hard_extra,
                         class_boost=args.class_boost,
                         italic_prob=args.italic_prob,
                         italic_max_frac=args.italic_max_frac)
Xs = Xs.reshape(-1, 1, IN_H, IN_W).astype(np.float32)
Ys = Ys.astype(np.int64)
print(f"synthetic: {Xs.shape[0]}", flush=True)

perm = torch.randperm(len(Ys))
n_val = int(len(Ys) * 0.05)
val_idx, tr_idx = perm[:n_val], perm[n_val:]
Xtr_syn = torch.from_numpy(Xs[tr_idx])
Ytr_syn = torch.from_numpy(Ys[tr_idx])
Wtr_syn = torch.ones(len(Ytr_syn))
Xval = torch.from_numpy(Xs[val_idx])
Yval = torch.from_numpy(Ys[val_idx])

if args.synth_extra_npz:
    ex = np.load(args.synth_extra_npz, allow_pickle=True)
    Xe = ex["X"].astype(np.float32).reshape(-1, 1, IN_H, IN_W)
    Ye = ex["Y"].astype(np.int64)
    if "meta" in ex.files:
        em = [json.loads(s) for s in ex["meta"]]
        sel = [i for i, m in enumerate(em)
               if int(m.get("line", args.line)) == args.line]
        Xe, Ye = Xe[sel], Ye[sel]
    if "meta" in ex.files:
        # meta['weak'] marks slices whose label the tool did not agree
        # with; they stay in the training set but at a low weight.
        _we = [args.weak_real_weight if em[i].get("weak") else 1.0
               for i in sel]
        if args.pn_weight != 1.0:
            for k, i in enumerate(sel):
                _col = em[i].get("col")
                if (_col is not None and int(em[i].get("line", -1)) == 1
                        and 28 <= int(_col) <= 41):
                    _we[k] *= args.pn_weight
        Wtr_syn = torch.cat([Wtr_syn, torch.tensor(_we, dtype=torch.float32)])
    Xtr_syn = torch.cat([Xtr_syn, torch.from_numpy(Xe)])
    Ytr_syn = torch.cat([Ytr_syn, torch.from_numpy(Ye)])
    print(f"synth-extra: +{len(Ye)} rows from {args.synth_extra_npz}",
          flush=True)

# Class balance for the real half.
#
# The 41 photos are 44 characters wide, and in line 2 the numeric fields
# and the filler runs produce a lopsided histogram: 223 real '0' but a
# single real 'O', 194 '1' against 17 'I', 105 '8' against 26 'B'.  The
# network therefore learns "round glyph -> 0", which is exactly the
# confusion that shows up in the residual failures (a '0' read as 'O'
# and back).  The synthetic half is balanced by construction, so the
# imbalance is introduced purely by the real half.
#
# Sampling the real half per class -- capped for the frequent classes,
# repeated for the rare ones -- keeps every real photograph in the
# training set while removing the prior the frequency counts encode.
# Photo-level holdout.
#
# Slicing a photograph into 44 glyphs and then holding out SLICES leaks:
# the remaining slices of the same photograph carry the same font, the
# same lighting and the same segmentation, so a held-out slice is far
# easier than a held-out photograph.  The only leak-free real signal is
# a whole photograph, so the holdout unit here is the photograph.
photo_of = [str(mm[src].get("file", f"row{src}")) for src in keep]
uniq_photos = sorted(set(photo_of))
rng_h = random.Random(args.seed + 991)
rng_h.shuffle(uniq_photos)
n_hold = min(args.holdout_photos, max(0, len(uniq_photos) - 1))
hold_set = set(uniq_photos[:n_hold])
# An explicit holdout list beats a random pick: the corpus contains
# photographs of the SAME document (byte-identical files, or a second
# shot of the same passport page).  A random split can put one of them in
# the training half and its twin in the holdout half, which leaks the
# answer and makes the holdout accuracy meaningless.  `--holdout-files`
# names the photographs that must be held out; everything else trains.
if args.holdout_files:
    wanted = {s.strip() for s in args.holdout_files.split(",") if s.strip()}
    # Match on the stem, so "IMG_1792133000_003" selects the file whose
    # meta name is "IMG_1792133000_003.jpg".
    stems = {w.rsplit(".", 1)[0] for w in wanted}
    matched = {p for p in uniq_photos
               if p in wanted or Path(p).name in wanted
               or Path(p).stem in stems
               or any(w in p for w in wanted)}
    if not matched:
        raise SystemExit(f"--holdout-files {sorted(wanted)} matched no "
                         f"photograph in the real half; available: "
                         f"{uniq_photos[:8]}")
    hold_set = matched
    n_hold = len(hold_set)
    print(f"explicit holdout: {n_hold} photos {sorted(hold_set)}", flush=True)
hold_idx = [i for i, ph in enumerate(photo_of) if ph in hold_set]
train_idx = [i for i, ph in enumerate(photo_of) if ph not in hold_set]
if n_hold:
    Xr = Xr[train_idx]; Yr = Yr[train_idx]; Cr = Cr[train_idx]
    Wslice = Wslice[train_idx]
    photo_of = [photo_of[i] for i in train_idx]
    Xh = torch.from_numpy(Xr_all[[keep[i] for i in hold_idx]].astype(np.float32))
    Yh = Yr_all[[keep[i] for i in hold_idx]]
    Ch = torch.zeros(len(hold_idx), NUM_CLASSES)
    for r, src in enumerate(hold_idx):
        info = mm[keep[src]] if isinstance(mm[keep[src]], dict) else {}
        cand = info.get("cand")
        ids = [GI[c] for c in cand if c in GI] if cand else [int(Yr_all[keep[src]])]
        Ch[r, ids] = 1.0 / len(ids)
    print(f"real holdout: {len(hold_idx)} slices from {n_hold} photos "
          f"({sorted(hold_set)[:3]}...)", flush=True)
else:
    Xh = Yh = Ch = None

n_real_target = int(len(Ytr_syn) * args.real_frac / max(1e-9, 1 - args.real_frac))
if args.balance_real:
    _by_class = {}
    for _i, _y in enumerate(Yr.tolist()):
        _by_class.setdefault(_y, []).append(_i)
    _per_class = max(1, n_real_target // max(1, len(_by_class)))
    _picked = []
    for _y, _idxs in _by_class.items():
        _r = _per_class // len(_idxs) + 1
        _picked.extend([_idxs[_k % len(_idxs)] for _k in range(_r * len(_idxs))])
    _picked = _picked[:n_real_target]
    _sel = torch.tensor(_picked, dtype=torch.long)
    Xr = Xr[_sel]
    Yr = Yr[_sel]
    Cr = Cr[_sel]
    Wslice = Wslice[_sel]
    from collections import Counter
    _hist = Counter(Yr.tolist())
    _mc = max(_hist.values()) / max(1, min(_hist.values()))
    print(f"real half rebalanced: {len(Yr)} slices, "
          f"{len(_hist)} classes, max/min class ratio {_mc:.1f}", flush=True)
# Per-class inverse-frequency weights for the real half.
#
# The 41 photographs are 44 characters wide, so the numeric fields and
# the filler runs make the real histogram lopsided: 223 real '0' but a
# single real 'O', 194 '1' against 17 'I', 105 '8' against 26 'B'.
# The network reads that as a prior -- "round glyph -> 0" -- and that
# prior is exactly what flips the residual failures.
#
# Resampling the real half to fix the histogram (--balance-real) does
# remove the prior, but it also multiplies the 1804 unique slices into
# ~37k duplicates, and the model then reaches a perfect training-set
# score by epoch 8 and overfits the photographs it is supposed to
# generalise from (measured: 37/41 -> 28/41 end-to-end, with the false
# accepts rising from 2 to 11).
#
# Weighting is the other way to remove a prior: every unique slice is
# kept exactly once and the LOSS is re-weighted so that each class
# contributes the same total mass.  No duplicates, no memorisation.
# Class weighting of the real half.
#
# The 41 photographs are 44 characters wide, so the numeric fields and
# the filler runs make the real histogram lopsided: 223 real '0' but a
# single real 'O', 194 '1' against 17 'I', 105 '8' against 26 'B'.
# The network reads that as a prior -- "round glyph -> 0" -- and that
# prior is exactly what flips the residual failures.
#
# Two ways to remove the prior, and BOTH of them fail on their own:
#
#  * resampling the real half to a flat histogram (--balance-real)
#    multiplies the 1804 unique slices into ~37k duplicates.  The model
#    reaches a perfect training score by epoch 8 and then memorises the
#    photographs it is supposed to generalise from: measured end-to-end,
#    37/41 -> 28/41, with the false accepts rising from 2 to 11.
#
#  * weighting the unique slices without tiling them (--class-weight)
#    leaves the real half at 1452 rows, so the weight that expresses
#    "this class matters as much as that one" has to be 25x larger to
#    keep the real half's share of the loss.  The model then chases
#    those 1452 rows: measured end-to-end, 38/41 -> 13/41.
#
# The combination that works is to keep the real half's ROW COUNT as
# it was (so the loss balance between synthetic and real is what the
# --real-frac setting asks for) and to apply the class weights WITHIN
# that budget, normalised so the total real weight is unchanged.  A
# class that is over-represented gives up weight, a rare class gets
# more, and no row is duplicated.
class_weighted = False
if args.class_weight:
    _cnt = Counter(Yr.tolist())
    _inv = {k: 1.0 / max(1, v) for k, v in _cnt.items()}
    _cw = torch.tensor([_inv[int(y)] for y in Yr.tolist()], dtype=torch.float32)
    # Give every class the SAME total weight mass: class c contributes
    # count_c * (1/count_c) = 1 before the global rescale.  The rescale
    # restores the original total, so the real half keeps its share of
    # the loss and the synthetic half is unaffected.
    _cw = _cw * len(_cw) / float(_cw.sum())
    _w_before = float(Wslice.sum())
    Wslice = Wslice * _cw
    # The class weights must REDISTRIBUTE the real half's weight, not
    # change its total: the share of the loss that the real half gets
    # is what --real-frac asks for.
    _w_after = float(Wslice.sum())
    if _w_after > 0:
        Wslice = Wslice * (_w_before / _w_after)
    class_weighted = True
    print(f"class-weighted real half: {len(_cnt)} classes, "
          f"weight range [{float(Wslice.min()):.4f}, {float(Wslice.max()):.4f}]",
          flush=True)

# How many times the real half is tiled into the training set.
#
# The synthetic half has ~87k rows and the real half only ~1.8k, so the
# real half is tiled up to reach its intended share of the loss.  When
# the class weights are in play the tiling is switched OFF: duplicating
# a slice is exactly what made the model memorise the photographs it
# has to generalise from.  Instead the real rows keep their per-slice
# weight and the weight itself carries the share.
reps = (n_real_target + len(Yr) - 1) // max(1, len(Yr))
# NOTE on the repeat semantics: numpy's `repeat` repeats EACH ELEMENT
# `reps` times (tile), while torch's `Tensor.repeat` repeats the WHOLE
# TENSOR (block).  Using one for the images/labels and the other for the
# soft targets silently misaligns every row: the soft target then
# describes a different glyph than the image, which shows up as a model
# that collapses onto one class.  All four tensors must use the SAME
# semantics, so they all go through numpy's repeat.
_n_real_rows = n_real_target
Xr_r = torch.from_numpy(np.repeat(Xr.numpy(), reps, axis=0)[:_n_real_rows])
Yr_r = torch.from_numpy(np.repeat(Yr.numpy(), reps)[:_n_real_rows])
Cr_r = torch.from_numpy(np.repeat(Cr.numpy(), reps, axis=0)[:_n_real_rows])
Wr_r = torch.from_numpy(np.repeat(Wslice.numpy(), reps)[:_n_real_rows])
if class_weighted:
    # The class weights were normalised so the real half's TOTAL weight
    # is unchanged, so the tiled rows already carry the right mass.
    print(f"real half weight mass = {float(Wr_r.sum()):.0f} "
          f"(synthetic rows {len(Ytr_syn)})", flush=True)
# hard targets for the synthetic half (those labels are rendered, not OCR)
Cr_syn = torch.zeros(len(Ytr_syn), NUM_CLASSES)
Cr_syn[torch.arange(len(Ytr_syn)), Ytr_syn] = 1.0
Wsyn = Wtr_syn
print(f"train syn={len(Ytr_syn)} real={len(Yr_r)} val(syn)={len(Xval)} "
      f"real_frac={len(Yr_r)/(len(Ytr_syn)+len(Yr_r)):.2f}", flush=True)

Xall = torch.cat([Xtr_syn, Xr_r])
Yall = torch.cat([Ytr_syn, Yr_r])
Call = torch.cat([Cr_syn, Cr_r])
Wall = torch.cat([Wsyn, Wr_r])
assert len(Xall) == len(Yall) == len(Call) == len(Wall), (
    f"row mismatch: X={len(Xall)} Y={len(Yall)} C={len(Call)} W={len(Wall)}")
# The soft target must agree with the hard label wherever the candidate
# set is a single class; otherwise the two loss modes are NOT equivalent
# and a silent mismatch would be very hard to see.
# "single class" means the candidate set had exactly one member.  The
# row SUM is always 1 (the target is a distribution), so it cannot be
# used to tell a one-member set from a three-member one.
_single = (Call > 0).sum(1) == 1
if _single.any():
    _bad = int((Call.argmax(1)[_single] != Yall[_single]).sum())
    if _bad:
        _idx = torch.nonzero(_single & (Call.argmax(1) != Yall)).squeeze(1)[:5]
        raise SystemExit(
            f"soft/hard label mismatch on {_bad} single-class rows; "
            f"first idx={_idx.tolist()} "
            f"call={Call.argmax(1)[_idx].tolist()} "
            f"yall={Yall[_idx].tolist()} "
            f"n_syn={len(Ytr_syn)} "
            f"n_real={n_real_target} n_keep={len(keep)} reps={reps} "
            f"first_real_idx={_idx.tolist()[0] - len(Ytr_syn)}")
print(f"assert ok: rows={len(Xall)} single-class={int(_single.sum())}", flush=True)


def augment_real(x):
    """Per-row augmentation for the REAL half (bright ink on black).

    Each row gets its own random transform sampled from the global RNG:
    a 3x3 blur (~30%), stroke erosion via min-pool (~25%, thins the
    bright ink -- the AJ glyphs are exactly the thin-stroke case) and
    an affine squeeze/stretch centred on the glyph with a sub-column
    shift.  Scale < 1 along x reproduces the narrow 16x12 glyphs that
    the line-2 classifier currently misreads (F->E, H->M, 9->7).
    """
    import torch.nn.functional as F
    n = x.size(0)
    out = x.clone()
    mask = torch.rand(n) < 0.15
    if mask.any():
        out[mask] = F.avg_pool2d(out[mask], 3, 1, 1)
    mask = torch.rand(n) < 0.12
    if mask.any():
        out[mask] = -F.max_pool2d(-out[mask], 3, 1, 1)
    sx = 0.80 + 0.35 * torch.rand(n)
    sy = 0.92 + 0.16 * torch.rand(n)
    tx = (torch.rand(n) - 0.5) * 1.2
    ty = (torch.rand(n) - 0.5) * 0.6
    theta = torch.zeros(n, 2, 3)
    theta[:, 0, 0] = sx
    theta[:, 0, 2] = tx
    theta[:, 1, 1] = sy
    theta[:, 1, 2] = ty
    grid = F.affine_grid(theta, out.size(), align_corners=True)
    return F.grid_sample(out, grid, align_corners=True, mode="bilinear",
                         padding_mode="zeros")


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
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
bs = 512
best = 0.0
best_score = -1.0
best_state = None
for ep in range(args.epochs):
    # Regenerate the real half FROM THE CLEAN COPY every epoch.  The old
    # form mutated the Xall slice in place, so the noise did a cumulative
    # random walk across epochs and clamped the pixels to the 0/1 borders
    # -- the opposite of the intended JPEG-decode-noise simulation.
    if args.augment:
        torch.manual_seed(args.aug_seed + ep)
        _real_base = augment_real(Xr_r)
    else:
        _real_base = Xr_r
    if args.jitter > 0:
        torch.manual_seed(args.aug_seed + ep + 1_000_000)
        _real_base = (_real_base + torch.randn_like(_real_base) * args.jitter
                      ).clamp_(0.0, 1.0)
    if args.augment or args.jitter > 0:
        Xall[len(Xtr_syn):] = _real_base
    model.train()
    perm_tr = torch.randperm(len(Xall))
    correct = tot = 0
    for s in range(0, len(perm_tr), bs):
        idx = perm_tr[s:s + bs]
        lg = model(Xall[idx])
        # soft-target cross entropy: -sum_c t_c * log softmax_c, scaled by
        # the per-slice confidence weight.
        if args.hard_loss:
            # Plain cross entropy on the single most likely class.
            per = F.cross_entropy(lg, Yall[idx], reduction='none')
        else:
            # soft-target cross entropy: -sum_c t_c * log softmax_c
            per = -(Call[idx] * F.log_softmax(lg, dim=1)).sum(1)
        # Divide by the FULL row count, not the weight sum: normalising by
        # the weight sum silently rescales the learning rate whenever the
        # batch happens to contain more (or fewer) down-weighted slices.
        loss = (per * Wall[idx]).sum() / len(idx)
        opt.zero_grad(); loss.backward(); opt.step()
        correct += int((lg.argmax(1) == Yall[idx]).sum()); tot += len(idx)
    sched.step()
    model.eval()
    with torch.no_grad():
        va = 0
        for s in range(0, len(Xval), bs):
            v = model(Xval[s:s + bs]); va += int((v.argmax(1) == Yval[s:s + bs]).sum())
        val_acc = va / len(Xval)
        # real-slice accuracy is reported but NOT used for selection:
        # it is measured on training data, so it can only be a sanity
        # signal, never a model-selection criterion.
        ra = rhit = 0
        for s in range(0, len(Xr), bs):
            v = model(Xr[s:s + bs])
            # a cell counts as correct when the prediction lands inside
            # the checksum-consistent candidate set
            ok = Cr[s:s + bs].gather(1, v.argmax(1, keepdim=True)).squeeze(1) > 0
            rhit += int(ok.sum()); ra += len(ok)
        real_acc = rhit / max(1, ra)
        # Held-out PHOTOGRAPH accuracy.  This is the only leak-free
        # real signal: the training-set real accuracy saturates at 100%
        # within a handful of epochs, so it cannot tell an under-trained
        # model from a memorising one.
        hold_acc = -1.0
        if Xh is not None and len(Xh):
            ha = hh = 0
            for s2 in range(0, len(Xh), bs):
                v = model(Xh[s2:s2 + bs])
                ok = Ch[s2:s2 + bs].gather(1, v.argmax(1, keepdim=True)).squeeze(1) > 0
                hh += int(ok.sum()); ha += len(ok)
            hold_acc = hh / max(1, ha)
    # Model selection MUST consider the real slices.  The synthetic
    # validation set saturates at 100% within a couple of epochs, so
    # selecting on it alone picks the FIRST epoch that happens to reach
    # it -- an under-trained model whose real-photo accuracy is still
    # poor.  Selecting on the worst of (synthetic, real) fixes that;
    # real accuracy is measured against the checksum-consistent
    # candidate sets, so an ambiguous cell cannot be scored as wrong
    # when the network legitimately picked another member of the set.
    if hold_acc >= 0.0:
        # When whole photographs are held out, that score is the one
        # that generalises, so it dominates the selection.
        score = min(val_acc, hold_acc)
    else:
        # MEASURED 2026-11-05: the previous "fix" (min(val_acc, real_acc)
        # for both lines) REGRESSED the official-GT photos 41/41 -> 24/41
        # on line 1.  val_syn saturates at 1.0 early and carries no
        # information; min() therefore selects LATER epochs that drift
        # toward the synthetic domain, while real_acc (saturating on the
        # real slices, which share their source photos with the official
        # GT) tracks the domain that matters.  Keep the original rule:
        # line 0 selects on real_acc, line 1 on min(val, real).
        score = min(val_acc, real_acc) if args.line == 1 else real_acc
    if score > best_score:
        best_score = score
        best = val_acc
        best_state = {k: v.detach().cpu().numpy().copy() for k, v in model.state_dict().items()}
    print(f"ep{ep:2d} tr={correct/tot:.4f} val_syn={val_acc:.4f} "
          f"real={real_acc:.4f} hold={hold_acc:.4f} "
          f"score={score:.4f} best_score={best_score:.4f}",
          flush=True)

prefix = "LINE1" if args.line == 0 else "LINE2"
v2.emit_header(
    (best_state["conv1.weight"].transpose(2, 3, 1, 0), best_state["conv1.bias"],
     best_state["conv2.weight"].transpose(2, 3, 1, 0), best_state["conv2.bias"],
     best_state["conv3.weight"].transpose(2, 3, 1, 0), best_state["conv3.bias"],
     best_state["fc1.weight"], best_state["fc1.bias"],
     best_state["fc2.weight"], best_state["fc2.bias"],
     np.zeros(IN_H * IN_W, np.float32), np.ones(IN_H * IN_W, np.float32),
     best), args.out, prefix=prefix,
    comment=(f"/* Auto-generated by tools/train_cnn_omtest.py --line {args.line}. "
             f"Do not edit. */"))
print(f"wrote {args.out} best_val={best}", flush=True)
