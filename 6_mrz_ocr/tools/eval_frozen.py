#!/usr/bin/env python3
"""Score the CNN and the CRNN against a FROZEN ground truth.

The reference strings come from data/gt_frozen_*.json, which was built
by tools/build_frozen_gt.py and does NOT depend on the model being
scored.  That is the whole point: a retrain can no longer move its own
reference and call it an improvement.

Reports, per head:
  exact     line 2 identical to the frozen GT
  char      per-character accuracy against the frozen GT
  cksum     share of outputs whose five check digits verify
  cksum&wrong  outputs that verify AND differ from the GT -- the number
                a checksum-gated pipeline would silently ship
  p50/p95   wall-clock latency of the native tool

Usage:
    python3 tools/eval_frozen.py --set passport
    python3 tools/eval_frozen.py --set assets --out /tmp/x.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import tempfile
import time
from pathlib import Path

import cv2
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "build" / "mrz_ocr_cnn_tool"
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent))
from weight_fingerprint import verify_crnn_fingerprint  # noqa: E402
OMTEST = ROOT.parent / "tmp" / "omtest"
ASSETS = ROOT.parent / "assets" / "pic"
HIRES = ROOT.parent / "assets" / "pic_hires"

W = (7, 3, 1)


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
        if l2[pos] != str(cd(l2[a:b])):
            return False
    return l2[43] == str(cd(l2[0:10] + l2[13:20] + l2[21:28] + l2[28:43]))


def decode(ppm: Path, mode: str):
    env = dict(os.environ)
    env["MRZ_OCR_LINE_MODELS"] = "1"
    env["MRZ_OCR_PAR"] = "0"
    env["MRZ_OCR_CRNN"] = "1" if mode in ("crnn", "fuse") else "0"
    env["MRZ_OCR_FUSE"] = "1" if mode == "fuse" else "0"
    t0 = time.perf_counter()
    p = subprocess.run([str(TOOL), str(ppm)], capture_output=True,
                       text=True, timeout=120, env=env)
    ms = (time.perf_counter() - t0) * 1000.0
    kv = {}
    for ln in p.stdout.splitlines():
        if ":" in ln:
            k, _, v = ln.partition(":")
            kv[k.strip()] = v.strip()
    return (kv.get("result.line1", ""), kv.get("result.line2", ""),
            int(kv.get("result.conf1", "0") or 0),
            int(kv.get("result.conf2", "0") or 0), ms)


def load_gt(which: str):
    p = ROOT / "data" / f"gt_frozen_{which}.json"
    return json.loads(p.read_text())["records"]


def image_for(which: str, rec):
    name = rec["id"]
    if which in ("passport", "passport_corrected"):
        for d in sorted(OMTEST.iterdir()):
            if not d.is_dir():
                continue
            for jpg in d.glob("*.jpg"):
                if jpg.stem == Path(name).stem:
                    return jpg
    elif which == "hires":
        p = HIRES / name
        if p.exists():
            return p
    else:
        p = ASSETS / name
        if p.exists():
            return p
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=("passport", "assets", "passport_corrected", "hires"), required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--pads", default="")
    args = ap.parse_args()

    # CRNN/fuse rows are only meaningful while the sequence head matches
    # the tool's conv features (see tools/weight_fingerprint.py).
    verify_crnn_fingerprint(TOOL)

    recs = load_gt(args.set)
    tmp = Path(tempfile.mkdtemp(prefix="evf_"))
    pads = ([int(x) for x in args.pads.split(",")] if args.pads
            else sorted({int(r.get("pad", 10)) for r in recs}))

    rows = []
    for rec in recs:
        img = image_for(args.set, rec)
        if img is None:
            continue
        rgb = cv2.imread(str(img))
        if rgb is None:
            continue
        H, Wd = rgb.shape[:2]
        x, y, w, h = rec["crop"]
        per = {m: {} for m in ("cnn", "crnn", "fuse")}
        for pad in pads:
            band = rgb[max(0, y - pad):min(H, y + h + pad),
                       max(0, x):min(Wd, x + w)]
            if band.shape[0] < 20:
                continue
            ppm = tmp / f"{Path(rec['id']).stem}_{pad}.ppm"
            Image.fromarray(cv2.cvtColor(band, cv2.COLOR_BGR2RGB)).save(
                str(ppm), format="PPM")
            for m in per:
                per[m][pad] = decode(ppm, m)
        if not per["cnn"] or not per["crnn"]:
            continue
        mid = pads[len(pads) // 2]
        rows.append({
            "id": rec["id"], "gt": rec["line2"],
            "cnn": per["cnn"][mid], "crnn": per["crnn"][mid],
            "fuse": per["fuse"].get(mid),
        })

    n = len(rows)
    if n == 0:
        print("no samples")
        return 1
    print(f"set={args.set}  frozen-GT images={n}  pads={pads}")
    print(f"{'head':6} {'exact':>9} {'char':>9} {'cksum':>9} {'cksum&wrong':>13} "
          f"{'p50':>8} {'p95':>8}")
    out = {}
    for head in ("cnn", "crnn", "fuse"):
        exact = char = cksum = ckw = 0
        ms = []
        for r in rows:
            if r.get(head) is None:
                continue
            l1, l2, c1, c2, t = r[head]
            ms.append(t)
            if l2 == r["gt"]:
                exact += 1
            char += sum(1 for i in range(44)
                        if i < len(l2) and l2[i] == r["gt"][i]) / 44
            if l2_ok(l2):
                cksum += 1
                if l2 != r["gt"]:
                    ckw += 1
        mss = sorted(ms)
        out[head] = {
            "n": n, "exact": exact, "char_pct": char / n * 100,
            "cksum": cksum, "cksum_wrong": ckw,
            "p50": statistics.median(mss), "p95": mss[int(len(mss) * 0.95)],
        }
        print(f"{head:6} {exact:>4}/{n:<4} {char/n*100:>8.2f}% {cksum:>4}/{n:<4} "
              f"{ckw:>6}/{n:<6} {statistics.median(mss):>7.2f} "
              f"{mss[int(len(mss)*0.95)]:>7.2f}")

    if out["crnn"]["exact"] != out["cnn"]["exact"] or \
       abs(out["crnn"]["char_pct"] - out["cnn"]["char_pct"]) > 0.01:
        print(f"delta(crnn-cnn): exact={out['crnn']['exact']-out['cnn']['exact']:+d}  "
              f"char={out['crnn']['char_pct']-out['cnn']['char_pct']:+.2f}pp  "
              f"p50={out['crnn']['p50']-out['cnn']['p50']:+.2f}ms")

    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=1))
        print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
