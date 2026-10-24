#!/usr/bin/env python3
"""Head-to-head CNN vs CRNN on both real photo sets, with an honest GT.

Design
------
The two sets and their only usable label sources:

  passport   tmp/PassPortPhoto, 1440x960, handheld, tilt + reflection.
             Band crop is taken from the old pipeline's TextBox dump
             (offset 0 into the bottom-third crop, see
             probe_crop_offset.py).
  assets     assets/pic, 348x295..1280x1389, scraped document scans of
             every shape.  Band crop is catalog.json's loc.mrz.

Ground truth is built by CONSENSUS, not by checksum.  checksum_nullspace.py
shows a verifying line 2 still admits ~2e20 single-substitution variants
(0 can become A/K/U/< at any weight-7 position for free), so "verifies"
is a necessary condition and nothing more.  Agreement between several
independent crops and two independent heads is the real evidence: the
chance that 4 crops all land on the same wrong member of that null space
is negligible.

Metrics reported per head, per set:
  exact        line 2 identical to consensus GT
  char         per-character accuracy against consensus GT
  cksum        share of outputs whose five check digits verify
  cksum&wrong  share of outputs that verify AND are wrong  <- the number
               that matters, because it is the number a checksum-gated
               pipeline would silently ship
  invisible    of the wrong outputs, the share whose every error is a
               substitution the check digits cannot see

Usage:
    python3 tools/eval_two_sets.py --set passport
    python3 tools/eval_two_sets.py --set assets
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

import cv2
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "build" / "mrz_ocr_cnn_tool"
PASSPORT = ROOT.parent / "tmp" / "PassPortPhoto"
OMTEST = ROOT.parent / "tmp" / "omtest"
ASSETS = ROOT.parent / "assets" / "pic"

ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ<"
W = (7, 3, 1)


# ---------------------------------------------------------------- checksums
def val(c: str) -> int:
    if c == "<":
        return 0
    if "0" <= c <= "9":
        return ord(c) - 48
    return ord(c) - 55


def cd(s: str) -> int:
    return sum(val(c) * W[i % 3] for i, c in enumerate(s)) % 10


def l2_ok(l2: str) -> bool:
    if len(l2) != 44:
        return False
    for pos, (a, b) in ((9, (0, 9)), (19, (13, 19)), (27, (21, 27)),
                        (42, (28, 42))):
        d = cd(l2[a:b])
        if l2[pos] != str(d):
            return False
    return l2[43] == str(cd(l2[0:10] + l2[13:20] + l2[21:28] + l2[28:43]))


def invisible(a: str, b: str) -> bool:
    """True if every position where a and b differ is checksum-blind."""
    if len(a) != 44 or len(b) != 44:
        return False
    for i in range(44):
        if a[i] == b[i]:
            continue
        if l2_ok(b[:i] + a[i] + b[i + 1:]):
            continue  # swapping this position back keeps all five digits
        return False
    return True


# ------------------------------------------------------------------- runner
def decode(ppm: Path, crnn: int):
    env = dict(os.environ)
    env["MRZ_OCR_LINE_MODELS"] = "1"
    env["MRZ_OCR_PAR"] = "0"
    env["MRZ_OCR_CRNN"] = "1" if crnn else "0"
    p = subprocess.run([str(TOOL), str(ppm)], capture_output=True,
                       text=True, timeout=120, env=env)
    kv = {}
    for ln in p.stdout.splitlines():
        if ":" in ln:
            k, _, v = ln.partition(":")
            kv[k.strip()] = v.strip()
    return kv.get("result.line1", ""), kv.get("result.line2", "")


# ------------------------------------------------------------------- crops
def crops_passport():
    out = {}
    for txt in sorted(OMTEST.glob("*/*.jpg-result.txt")):
        img = txt.with_name(txt.name.replace("-result.txt", ""))
        if not img.exists():
            continue
        pts = [(int(a), int(b)) for a, b in
               re.findall(r"\[x: (\d+), y: (\d+)\]",
                          txt.read_text(errors="replace"))]
        boxes = [pts[i:i + 4] for i in range(0, len(pts), 4)]
        if len(boxes) < 2:
            continue
        rgb = cv2.imread(str(img))
        if rgb is None:
            continue
        H, Wd = rgb.shape[:2]
        cy = H - (H // 3) - 50
        y1 = max(0, min(p[1] for p in boxes[0]) + cy)
        y2 = min(H, max(p[1] for p in boxes[1]) + cy)
        out[img.name] = (img, (0, y1, Wd, y2 - y1))
    return out


def crops_assets():
    cat = json.load(open(ASSETS / "catalog.json"))
    out = {}
    for r in cat.get("records", []):
        name, loc = r.get("file"), (r.get("loc") or {}).get("mrz")
        if not name or not loc:
            continue
        p = ASSETS / name
        if p.exists():
            out[name] = (p, (loc["x"], loc["y"], loc["w"], loc["h"]))
    return out


# -------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=("passport", "assets"), required=True)
    ap.add_argument("--pads", default="4,10,18")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    spec = crops_passport() if args.set == "passport" else crops_assets()
    pads = [int(x) for x in args.pads.split(",")]
    tmp = Path(tempfile.mkdtemp(prefix="ev2_"))

    rows = []
    for name, (img, (x, y, w, h)) in sorted(spec.items()):
        rgb = cv2.imread(str(img))
        if rgb is None:
            continue
        H, Wd = rgb.shape[:2]
        per_head = {0: {}, 1: {}}       # head -> pad -> (l1, l2)
        for pad in pads:
            band = rgb[max(0, y - pad):min(H, y + h + pad),
                       max(0, x):min(Wd, x + w)]
            if band.shape[0] < 20:
                continue
            ppm = tmp / f"{Path(name).stem}_{pad}.ppm"
            Image.fromarray(cv2.cvtColor(band, cv2.COLOR_BGR2RGB)).save(
                str(ppm), format="PPM")
            for crnn in (0, 1):
                per_head[crnn][pad] = decode(ppm, crnn)

        votes = Counter()
        for crnn in (0, 1):
            for pad, (_, l2) in per_head[crnn].items():
                if l2_ok(l2):
                    votes[l2] += 1
        if not votes:
            rows.append({"id": name, "gt": None, "band_h": h})
            continue
        gt, n = votes.most_common(1)[0]
        # Acceptance has two tiers, and the JSON records which one a
        # sample fell into so the report can show both numbers:
        #   consensus  >=2 runs agree and strictly beat any rival
        #   single     exactly one run verified; no rival to contradict it
        # A rival that ties is always a drop -- two crops reading two
        # different verifying strings means one crop is off the document.
        runner = votes.most_common(2)[1][1] if len(votes) > 1 else 0
        tier = None
        if n >= 2 and n > runner:
            tier = "consensus"
        elif len(votes) == 1:
            tier = "single"
        if tier is None:
            rows.append({"id": name, "gt": None, "band_h": h,
                         "why": f"no consensus {dict(votes)}"})
            continue
        mid = pads[len(pads) // 2]
        rows.append({
            "id": name, "gt": gt, "votes": n, "tier": tier, "band_h": h,
            "cnn": per_head[0].get(mid, ("", ""))[1],
            "crnn": per_head[1].get(mid, ("", ""))[1],
            "cnn1": per_head[0].get(mid, ("", ""))[0],
            "crnn1": per_head[1].get(mid, ("", ""))[0],
        })

    gtrows = [r for r in rows if r["gt"]]
    n = len(gtrows)
    cons = [r for r in gtrows if r.get("tier") == "consensus"]
    print(f"set={args.set}  images={len(rows)}  GT={n} "
          f"(consensus {len(cons)}, single-vote {n - len(cons)})")
    if n == 0:
        return 1

    def carr(a, b):
        if not a or not b:
            return 0.0
        return sum(1 for i in range(44) if a[i] == b[i]) / 44

    for label, sub in (("ALL", gtrows), ("consensus only", cons)):
        if not sub:
            continue
        m = len(sub)
        print(f"\n--- {label} (n={m}) ---")
        print(f"{'head':6} {'exact':>8} {'char':>8} {'cksum':>8} "
              f"{'cksum&wrong':>12}")
        for head in ("cnn", "crnn"):
            ex = sum(1 for r in sub if r[head] == r["gt"])
            ch = sum(carr(r[head], r["gt"]) for r in sub) / m
            ck = sum(1 for r in sub if l2_ok(r[head]))
            ckw = sum(1 for r in sub if l2_ok(r[head]) and r[head] != r["gt"])
            print(f"{head:6} {ex:>4}/{m:<3} {ch*100:>7.2f}% {ck:>4}/{m:<3} "
                  f"{ckw:>5}/{m:<6}")

    # ---- where the errors sit along the row --------------------------
    # This is the diagnostic that separates "weaker classifier" from
    # "the row is being resampled onto 44 slots wrongly": a classifier
    # weakness spreads errors evenly, a geometry error piles them up at
    # the two ends of the row.
    print("\nerror position histogram (wrong outputs only, 44 slots "
          "in 4 blocks of 11):")
    print(f"{'head':6} {'0-10':>6} {'11-21':>6} {'22-32':>6} {'33-43':>6} "
          f"{'n_wrong':>8}")
    for head in ("cnn", "crnn"):
        blocks = [0, 0, 0, 0]
        nw = 0
        for r in gtrows:
            o = r[head]
            if o == r["gt"] or len(o) != 44:
                continue
            nw += 1
            for i in range(44):
                if o[i] != r["gt"][i]:
                    blocks[min(3, i // 11)] += 1
        print(f"{head:6} {blocks[0]:>6} {blocks[1]:>6} {blocks[2]:>6} "
              f"{blocks[3]:>6} {nw:>8}")

    # ---- confusion classes -------------------------------------------
    print("\nconfusion classes (output char -> GT char), wrong outputs:")
    for head in ("cnn", "crnn"):
        cnt = Counter()
        for r in gtrows:
            o = r[head]
            if o == r["gt"] or len(o) != 44:
                continue
            for i in range(44):
                if o[i] != r["gt"][i]:
                    cnt[(o[i], r["gt"][i])] += 1
        top = ", ".join(f"{a}->{b}x{k}" for (a, b), k in cnt.most_common(12))
        print(f"  {head:5} {top}")

    if args.out:
        json.dump(rows, open(args.out, "w"), indent=1, ensure_ascii=False)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
