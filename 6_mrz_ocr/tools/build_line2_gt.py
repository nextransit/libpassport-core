#!/usr/bin/env python3
"""Build an objective line-2 ground truth for a photo set.

Why this exists
---------------
Neither photo set ships a trustworthy text label:

  * tmp/PassPortPhoto  only has the OLD pipeline's own output
    (*.jpg-result.txt).  Using that as GT measures the new decoder
    against the old decoder, not against the truth.
  * assets/pic has catalog.json with a per-image MRZ bbox but the
    stored `mrz` text is garbage (U<IUTUTIITIIITIIIUTTHTUNB...).
  * a hand transcription of a contact sheet is not reliable either:
    measured on 17 samples, only 2/17 survived structural + check
    digit validation (0/O, 8/B, 5/S and 1-character offsets are
    everywhere once the glyph is ~20 px tall).

TD3 line 2, however, carries FIVE check digits covering 39 of its 44
characters (doc number, DOB, expiry, personal number, and an overall
digit over all of them).  A random 44-char string passes by accident
with probability ~1e-5, so a line 2 that verifies is correct for all
practical purposes.  That makes line 2 objectively measurable without
any human labelling -- and it is exactly the half of the MRZ where
CRNN currently does *better* than CNN, so it is the right half to
nail down first.

Method
------
For every image, decode it several times with deliberately different
geometry (2 heads x 3 vertical pads).  Keep only candidates whose five
check digits all verify.  If the surviving candidates agree, that
string becomes the GT; disagreement means the crop is bad and the
sample is dropped rather than guessed.

Usage
-----
    python3 tools/build_line2_gt.py --set passport --out /tmp/gt_l2_passport.json
    python3 tools/build_line2_gt.py --set assets   --out /tmp/gt_l2_assets.json
"""
from __future__ import annotations

import argparse
import json
import os
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

W = (7, 3, 1)


# ---------------------------------------------------------------- checksums
def check_digit(seg: str) -> int:
    t = 0
    for i, c in enumerate(seg):
        if c == "<":
            v = 0
        elif "0" <= c <= "9":
            v = ord(c) - 48
        elif "A" <= c <= "Z":
            v = ord(c) - 55
        else:
            return -1
        t += v * W[i % 3]
    return t % 10


def l2_ok(l2: str) -> bool:
    """All five TD3 line-2 check digits verify."""
    if len(l2) != 44:
        return False
    for pos, (a, b) in ((9, (0, 9)), (19, (13, 19)), (27, (21, 27)), (42, (28, 42))):
        d = check_digit(l2[a:b])
        if d < 0 or l2[pos] != str(d):
            return False
    d = check_digit(l2[0:10] + l2[13:20] + l2[21:28] + l2[28:43])
    return d >= 0 and l2[43] == str(d)


# ------------------------------------------------------------------- runner
def decode(ppm: Path, crnn: int) -> tuple[str, str, int, str]:
    env = dict(os.environ)
    env["MRZ_OCR_LINE_MODELS"] = "1"
    env["MRZ_OCR_PAR"] = "0"
    env["MRZ_OCR_SEGDBG"] = "1"
    env["MRZ_OCR_CRNN"] = "1" if crnn else "0"
    p = subprocess.run([str(TOOL), str(ppm)], capture_output=True,
                       text=True, timeout=120, env=env)
    kv = {}
    for ln in p.stdout.splitlines():
        if ":" in ln:
            k, _, v = ln.partition(":")
            kv[k.strip()] = v.strip()
    return (kv.get("result.line1", ""), kv.get("result.line2", ""),
            kv.get("result.conf2", ""), p.stderr)


# ------------------------------------------------------------------- crops
def crop_assets() -> dict[str, tuple[Path, tuple[int, int, int, int]]]:
    """assets/pic: bbox from catalog.json's loc.mrz."""
    cat = json.load(open(ASSETS / "catalog.json"))
    recs = cat if isinstance(cat, list) else cat.get("records", [])
    out = {}
    for r in recs:
        name = r.get("file") or r.get("name") or r.get("filename")
        loc = (r.get("loc") or {}).get("mrz")
        if not name or not loc:
            continue
        if isinstance(loc, dict):
            x, y, w, h = loc.get("x"), loc.get("y"), loc.get("w"), loc.get("h")
        else:
            x, y, w, h = loc[:4]
        if None in (x, y, w, h):
            continue
        p = ASSETS / name
        if p.exists():
            out[name] = (p, (int(x), int(y), int(w), int(h)))
    return out


def crop_passport() -> dict[str, tuple[Path, tuple[int, int, int, int]]]:
    """tmp/PassPortPhoto: bbox from the old pipeline's TextBox dump.

    The result txt coordinates live in a +50 px padded frame
    (ScaleParam sw = W+100, sh = H+100) on top of a crop that starts
    at H - H//3 - 50.  Both offsets have to be undone.
    """
    import re
    out = {}
    for txt in sorted(OMTEST.glob("*/*.jpg-result.txt")):
        img = txt.with_name(txt.name.replace("-result.txt", ""))
        if not img.exists():
            continue
        s = txt.read_text(errors="replace")
        pts = [(int(a), int(b)) for a, b in
               re.findall(r"\[x: (\d+), y: (\d+)\]", s)]
        boxes = [pts[i:i + 4] for i in range(0, len(pts), 4)]
        if len(boxes) < 2:
            continue
        rgb = cv2.imread(str(img))
        if rgb is None:
            continue
        H, Wd = rgb.shape[:2]
        crop_y = H - (H // 3) - 50
        # Offset 0, not -50: probed empirically (probe_crop_offset.py).
        # Offsets -25..+25 all still return two 44-char rows because the
        # band keeps a few px of slack, but -50 shifts the crop half a
        # line up and 0/12 samples decode.  So the logged TextBox y is
        # already in the cropped frame.
        y1 = max(0, min(p[1] for p in boxes[0]) + crop_y)
        y2b = min(H, max(p[1] for p in boxes[1]) + crop_y)
        out[img.name] = (img, (0, y1, Wd, y2b - y1))
    return out


# -------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=("passport", "assets"), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pads", default="4,10,18",
                    help="comma separated vertical pads added to the bbox")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    spec = crop_passport() if args.set == "passport" else crop_assets()
    items = sorted(spec.items())
    if args.limit:
        items = items[:args.limit]
    pads = [int(x) for x in args.pads.split(",")]

    tmp = Path(tempfile.mkdtemp(prefix="gtl2_"))
    gt: dict[str, dict] = {}
    dropped: dict[str, str] = {}

    for name, (img, (x, y, w, h)) in items:
        rgb = cv2.imread(str(img))
        if rgb is None:
            dropped[name] = "unreadable"
            continue
        H, Wd = rgb.shape[:2]
        votes: list[tuple[str, str]] = []   # (line2, "head/pad")
        for pad in pads:
            y0 = max(0, y - pad)
            y1 = min(H, y + h + pad)
            band = rgb[y0:y1, max(0, x):min(Wd, x + w)]
            if band.size == 0 or band.shape[0] < 20:
                continue
            ppm = tmp / f"{Path(name).stem}_{pad}.ppm"
            Image.fromarray(cv2.cvtColor(band, cv2.COLOR_BGR2RGB)).save(
                str(ppm), format="PPM")
            for crnn in (0, 1):
                l1, l2, conf, _ = decode(ppm, crnn)
                if l2_ok(l2):
                    votes.append((l2, f"{'crnn' if crnn else 'cnn'}/p{pad}"))
        if not votes:
            dropped[name] = "no checksum-valid line2 in any crop"
            continue
        cnt = Counter(v for v, _ in votes)
        best, n = cnt.most_common(1)[0]
        # A verifying line 2 is proof enough on its own (p ~ 1e-5 by
        # chance), so a single vote is accepted.  What is NOT accepted
        # is two verifying crops disagreeing -- that means one of them
        # is a different document and the crop is untrustworthy.
        if n != len(votes):
            dropped[name] = f"verifying crops disagree ({dict(cnt)})"
            continue
        gt[name] = {
            "l2": best,
            "votes": n,
            "tried": len(votes),
            "by": sorted({k for v, k in votes if v == best}),
            "band_h": h,
            "img_w": Wd,
            "img_h": H,
        }

    json.dump(gt, open(args.out, "w"), indent=1, ensure_ascii=False)
    print(f"set={args.set}  images={len(items)}  "
          f"gt={len(gt)}  dropped={len(dropped)}")
    for k, v in list(dropped.items()):
        print(f"  drop {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
