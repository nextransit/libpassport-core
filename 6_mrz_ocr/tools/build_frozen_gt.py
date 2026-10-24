#!/usr/bin/env python3
"""Build a FROZEN ground truth for the two real photo sets.

WHY THIS FILE EXISTS
--------------------
tools/eval_two_sets.py derives the "consensus" GT from the checksum-valid
outputs of the very models it is asked to score:

    votes[l2] += 1        for every crop x head that verifies
    gt = votes.most_common(1)[0]

That makes the metric non-reproducible across model changes: a model
that produces MORE checksum-valid readings changes its own reference
string, so a retrain can look like an improvement even when the photos
were read worse.  Measured directly: the same 41 photos yielded an
18-image GT set with the shipped weights and a 20-image GT set with a
retrained head -- the sets are not even the same population.

This script builds the GT ONCE, from sources that do not depend on the
model under test:

  1. the BOARD's own detector output (tmp/omtest/*.jpg-result.txt),
     which is a PaddleOCR CRNN running on the device -- an independent
     recogniser;
  2. the board's checksum-repaired reading (fixmrz), which only ever
     rewrites the five check-digit positions (verified: 35/35 rewrites
     were at check-digit positions, 0 elsewhere);
  3. a multi-crop consensus over a SET OF READERS that is fixed for the
     whole comparison (the two heads x three pads), aggregated once and
     then written to disk.

The result is a JSON file that every later evaluation reads.  A model
change can no longer move the reference.

Usage:
    python3 tools/build_frozen_gt.py --set passport --out data/gt_frozen_passport.json
    python3 tools/build_frozen_gt.py --set assets   --out data/gt_frozen_assets.json
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
OMTEST = ROOT.parent / "tmp" / "omtest"
ASSETS = ROOT.parent / "assets" / "pic"

ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ<"
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


def board_labels():
    """The board's independent reading of the machine-position photos."""
    out = {}
    for txt in sorted(OMTEST.glob("*/*.jpg-result.txt")):
        t = txt.read_text(errors="replace")
        ls = {}
        for m in re.finditer(r"^textLine\[(\d+)\]\((.*?)\)\s*$", t, re.M):
            ls[int(m.group(1))] = m.group(2).strip()
        mf = re.search(r"-----MRZ fixed \(fixmrz\)-----\s*\n(.+)\n(.+)", t)
        key = f"{txt.parent.name}/{txt.name.replace('-result.txt', '').replace('.jpg', '')}"
        out[key] = {
            "det": (ls.get(0, ""), ls.get(1, "")),
            "fix": (mf.group(1).strip(), mf.group(2).strip()) if mf else None,
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=("passport", "assets"), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pads", default="4,10,18")
    args = ap.parse_args()

    spec = crops_passport() if args.set == "passport" else crops_assets()
    pads = [int(x) for x in args.pads.split(",")]
    board = board_labels()
    tmp = Path(tempfile.mkdtemp(prefix="fgt_"))

    records = []
    for name, (img, (x, y, w, h)) in sorted(spec.items()):
        rgb = cv2.imread(str(img))
        if rgb is None:
            continue
        H, Wd = rgb.shape[:2]
        per_head = {0: {}, 1: {}}
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

        # --- vote over every independent reading we have -------------
        votes = Counter()
        for crnn in (0, 1):
            for pad, (_, l2) in per_head[crnn].items():
                if l2_ok(l2):
                    votes[l2] += 1
        bk = f"{Path(name).stem}"
        binfo = board.get(bk) or board.get(f"{Path(name).parent.name}/{bk}")
        bfix = binfo["fix"][1] if binfo and binfo["fix"] else ""
        bdet = binfo["det"][1] if binfo else ""
        if bfix and l2_ok(bfix):
            votes[bfix] += 2          # the board is an independent reader
        elif bdet and l2_ok(bdet):
            votes[bdet] += 2

        if not votes:
            continue
        gt, n = votes.most_common(1)[0]
        runner = votes.most_common(2)[1][1] if len(votes) > 1 else 0
        if n < 2 or n <= runner:
            continue
        mid = pads[len(pads) // 2]
        records.append({
            "id": name,
            "line1": "", "line2": gt,
            "votes": n, "band_h": h, "tier": "consensus",
            "crop": [int(x), int(y), int(w), int(h)],
            "pad": mid,
        })
        # Keep the crop parameters so a later evaluation can rebuild the
        # exact same inputs without re-deriving them.
        records[-1]["crop"] = [int(x), int(y), int(w), int(h)]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"set": args.set, "records": records},
                              indent=1, ensure_ascii=False))
    print(f"wrote {out}  images={len(records)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
