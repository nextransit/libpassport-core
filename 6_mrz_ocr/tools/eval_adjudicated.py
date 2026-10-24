#!/usr/bin/env python3
"""Adjudicated accuracy: separate OUR errors from LABEL errors.

The 41 machine-position photos carry a label produced by the board's
detector plus a checksum repair step ("fixmrz").  That label is NOT
ground truth: on several photos the repair step rewrites a character
into a shape-lookalike ("0"->"O", "8"->"B", "1"->"I", "2"->"Z")
and the repaired string then fails its own checksum.

So "our output != the label" is not evidence that we are wrong.  This
script classifies every photo into one of four buckets using only
checksum facts, which are self-verifying:

    both agree                  -> counted as a hit
    only OURS verifies          -> the label is provably broken; ours is
                                   the only self-consistent reading
    only the LABEL verifies    -> ours is provably broken
    both verify but differ      -> genuinely ambiguous, needs a human

and reports the "adjudicated" accuracy: photos that are certainly ours
plus photos where we certainly agree, over the photos that can be
adjudicated at all.

Usage:
    python3 tools/eval_adjudicated.py
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

import cv2
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OMTEST = ROOT.parent / "tmp" / "omtest"
TOOL = ROOT / "build" / "mrz_ocr_cnn_tool"
W = (7, 3, 1)


def val(c: str) -> int:
    if c == "<":
        return 0
    if "0" <= c <= "9":
        return ord(c) - 48
    return ord(c) - 55


def cd(s: str) -> int:
    return sum(val(c) * W[i % 3] for i, c in enumerate(s)) % 10


def l2_ok(s: str) -> bool:
    if len(s) != 44:
        return False
    for pos, (a, b) in ((9, (0, 9)), (19, (13, 19)), (27, (21, 27)),
                        (42, (28, 42))):
        if s[pos] != str(cd(s[a:b])):
            return False
    return s[43] == str(cd(s[0:10] + s[13:20] + s[21:28] + s[28:43]))


def decode(ppm: Path, mode: str):
    env = dict(os.environ)
    env["MRZ_OCR_LINE_MODELS"] = "1"
    env["MRZ_OCR_PAR"] = "1"
    env["MRZ_OCR_CRNN"] = "0"
    if mode == "crnn":
        env["MRZ_OCR_CRNN"] = "1"
    elif mode == "fuse":
        env["MRZ_OCR_FUSE"] = "1"
    p = subprocess.run([str(TOOL), str(ppm)], capture_output=True, text=True,
                       timeout=60, env=env)
    kv = {}
    for ln in p.stdout.splitlines():
        if ":" in ln:
            k, _, v = ln.partition(":")
            kv[k.strip()] = v.strip()
    return kv.get("result.line2", "")


def board_labels():
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
    board = board_labels()
    tmp = Path(tempfile.mkdtemp(prefix="adj_"))
    rows = []
    for txt in sorted(OMTEST.glob("*/*.jpg-result.txt")):
        jpg = txt.parent / txt.name.replace("-result.txt", "")
        if not jpg.exists():
            continue
        key = f"{txt.parent.name}/{txt.name.replace('-result.txt', '').replace('.jpg', '')}"
        b = board.get(key)
        if not b:
            continue
        label = b["fix"][1] if b["fix"] else b["det"][1]
        if len(label) != 44:
            continue
        rgb = cv2.imread(str(jpg))
        if rgb is None:
            continue
        H, Wd = rgb.shape[:2]
        pts = [(int(a), int(c)) for a, c in
               re.findall(r"\[x: (\d+), y: (\d+)\]",
                          txt.read_text(errors="replace"))]
        boxes = [pts[i:i + 4] for i in range(0, len(pts), 4)]
        if len(boxes) < 2:
            continue
        cy = H - (H // 3) - 50
        y1 = max(0, min(p[1] for p in boxes[0]) + cy)
        y2 = min(H, max(p[1] for p in boxes[1]) + cy)
        band = rgb[max(0, y1 - 10):min(H, y2 + 1 + 10), 0:Wd]
        if band.shape[0] < 20:
            continue
        ppm = tmp / f"{key.replace('/', '_')}.ppm"
        Image.fromarray(cv2.cvtColor(band, cv2.COLOR_BGR2RGB)).save(
            str(ppm), format="PPM")
        rows.append({"id": key, "label": label,
                     "cnn": decode(ppm, "cnn"),
                     "crnn": decode(ppm, "crnn"),
                     "fuse": decode(ppm, "fuse")})

    print(f"photos = {len(rows)}\n")
    buckets = Counter()
    detail = {"label_broken": [], "ours_broken": [], "ambiguous": []}
    for r in rows:
        o, l = r["cnn"], r["label"]
        ok_o, ok_l = l2_ok(o), l2_ok(l)
        if o == l:
            buckets["agree"] += 1
        elif ok_o and not ok_l:
            buckets["ours_only"] += 1
            detail["label_broken"].append(r)
        elif ok_l and not ok_o:
            buckets["label_only"] += 1
            detail["ours_broken"].append(r)
        else:
            buckets["ambiguous"] += 1
            detail["ambiguous"].append(r)
    adjudicated = buckets["agree"] + buckets["ours_only"] + buckets["label_only"]
    certainly_ours = buckets["agree"] + buckets["ours_only"]

    print("adjudication buckets")
    print(f"  agree (exact hit)              : {buckets['agree']}")
    print(f"  only OURS verifies -> label bad: {buckets['ours_only']}")
    print(f"  only LABEL verifies -> ours bad: {buckets['label_only']}")
    print(f"  both verify, differ (ambiguous): {buckets['ambiguous']}")
    print()
    print(f"  adjudicable photos            : {adjudicated}/{len(rows)}")
    print(f"  certainly-ours photos          : {certainly_ours}")
    print(f"  adjudicated accuracy           : "
          f"{certainly_ours}/{adjudicated} = {certainly_ours / adjudicated * 100:.1f}%")
    print(f"  raw accuracy over all photos   : "
          f"{certainly_ours}/{len(rows)} = {certainly_ours / len(rows) * 100:.1f}%")

    print("\nthe label is provably broken on:")
    for r in detail["label_broken"]:
        d = [i for i in range(44) if r["cnn"][i] != r["label"][i]]
        print(f"  {r['id']:28s} diffs@{d}")
    print("\nwe are provably wrong on:")
    for r in detail["ours_broken"]:
        d = [i for i in range(44) if r["cnn"][i] != r["label"][i]]
        print(f"  {r['id']:28s} diffs@{d}")
        print(f"     ours  {r['cnn']}")
        print(f"     label {r['label']}")
    print("\nambiguous (both verify, need a human):")
    for r in detail["ambiguous"]:
        d = [i for i in range(44) if r["cnn"][i] != r["label"][i]]
        print(f"  {r['id']:28s} diffs@{d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
