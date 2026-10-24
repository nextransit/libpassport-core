#!/usr/bin/env python3
"""Build line-2 ground truth by multi-crop / multi-scale consensus.

Why this and not the checksum
-----------------------------
checksum_nullspace.py measures ~2e20 single-substitution variants of any
verifying line 2 (0 -> A/K/U/< is free at every weight-7 column, and
0 <-> O is free in the two weight-10 columns taken together).  "It
verifies" is therefore a necessary condition and nothing more, and it is
*not even that* when the decoder's own beam search is allowed to run: the
beam walks the least-confident columns until the check digits close, so a
verifying output can be manufactured.

This tool therefore reads with MRZ_OCR_NOBEAM=1 -- the honest greedy
decode -- over a grid of independent rasterizations, and treats
AGREEMENT between them as the evidence:

    pads     vertical padding around loc.mrz   (0, 6, 12)
    shifts   horizontal offset in px           (-2, 0, +2)
    scales   target px per character           (18, 24, 30, 36)

Every one of those changes the rasterization the CNN sees: different
margin, different grid phase, different resampling filter.  Two readings
that agree after all that, and verify, are almost certainly the same
underlying text; two readings that agree only because the same
preprocessing produced the same systematic error do not survive the
grid.

Tiers
    A  cluster of >=3 agreeing verifying readings
    B  cluster of 2
    C  a single verifying reading, no corroboration
    -  nothing verifies

Usage
    python3 tools/build_gt.py --set assets
    python3 tools/build_gt.py --set passport
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import cv2
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "build" / "mrz_ocr_cnn_tool"
ASSETS = ROOT.parent / "assets" / "pic"
OMTEST = ROOT.parent / "tmp" / "omtest"
CORPUS = ROOT / "data" / "corpus_passport_photo"

PADS = (0, 6, 12)
SHIFTS = (-2, 0, 2)
SCALES = (18, 24, 30, 36)
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


def l2_ok(l: str) -> bool:
    if not l or len(l) != 44:
        return False
    try:
        for pos, (a, b) in ((9, (0, 9)), (19, (13, 19)), (27, (21, 27)),
                            (42, (28, 42))):
            if l[pos] != str(cd(l[a:b])):
                return False
        return l[43] == str(cd(l[0:10] + l[13:20] + l[21:28] + l[28:43]))
    except Exception:
        return False


def ham(a: str, b: str) -> int:
    if len(a) != 44 or len(b) != 44:
        return 44
    return sum(1 for x, y in zip(a, b) if x != y)


# ------------------------------------------------------------------- runner
ENV = dict(os.environ)
ENV.update(MRZ_OCR_LINE_MODELS="1", MRZ_OCR_PAR="0", MRZ_OCR_CRNN="0",
           MRZ_OCR_NOBEAM="1")


def read(ppm: Path) -> str:
    p = subprocess.run([str(TOOL), str(ppm)], capture_output=True, text=True,
                       env=ENV, timeout=120)
    for ln in p.stdout.splitlines():
        if ln.startswith("result.line2"):
            return ln.partition(":")[2].strip()
    return ""


def crop_variants(band):
    """Yield (tag, image) for every rasterization in the grid."""
    H, Wd = band.shape[:2]
    for pad in PADS:
        y0, y1 = pad, H - pad
        if y1 - y0 < 16:
            continue
        for sh in SHIFTS:
            x0, x1 = max(0, sh), min(Wd, Wd + sh)
            sub = band[y0:y1, x0:x1]
            if sub.shape[1] < 44:
                continue
            for t in SCALES:
                ppc = sub.shape[1] / 44.0
                s = t / ppc
                if abs(s - 1.0) < 1e-3:
                    img = sub
                else:
                    img = cv2.resize(sub, None, fx=s, fy=s,
                                     interpolation=cv2.INTER_CUBIC)
                yield f"p{pad}s{sh}t{t}", img


# -------------------------------------------------------------------- crops
def crops_assets():
    cat = json.load(open(ASSETS / "catalog.json"))
    out = []
    for r in cat.get("records", []):
        name, loc = r.get("file"), (r.get("loc") or {}).get("mrz")
        if not name or not loc:
            continue
        p = ASSETS / name
        img = cv2.imread(str(p)) if p.exists() else None
        if img is None:
            continue
        H, Wd = img.shape[:2]
        y, h = max(0, loc["y"]), loc["h"]
        x, w = max(0, loc["x"]), loc["w"]
        band = img[y:min(H, y + h), x:min(Wd, x + w)]
        if band.shape[0] < 16 or band.shape[1] < 44:
            continue
        out.append({"id": p.stem, "file": name, "band": band})
    return out


def crops_passport():
    """Same crop as eval_two_sets.py: TextBox dump, offset 0."""
    out = []
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
        band = rgb[y1:y2, 0:Wd]
        if band.shape[0] < 16:
            continue
        out.append({"id": img.stem, "file": img.name, "band": band})
    return out


# --------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=("assets", "passport"), required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    items = crops_assets() if args.set == "assets" else crops_passport()
    if args.limit:
        items = items[:args.limit]
    tmp = Path(tempfile.mkdtemp(prefix="gt_"))
    out_path = Path(args.out) if args.out else ROOT / "data" / f"gt_{args.set}.json"

    rows = []
    for i, it in enumerate(items):
        reads = []
        for tag, img in crop_variants(it["band"]):
            ppm = tmp / f"{it['id']}_{tag}.ppm"
            Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).save(
                str(ppm), format="PPM")
            reads.append((tag, read(ppm)))
        ver = [(t, r) for t, r in reads if l2_ok(r)]

        # Cluster the verifying readings by Hamming <= 2 and take the
        # biggest cluster.  Single-linkage is enough at this size and does
        # not need a metric embedding.
        clusters = []            # list of lists of (tag, str)
        for t, r in ver:
            hit = None
            for c in clusters:
                if any(ham(r, s) <= 2 for _, s in c):
                    hit = c
                    break
            if hit is None:
                clusters.append([(t, r)])
            else:
                hit.append((t, r))
        gt, support, tier = None, 0, "-"
        if clusters:
            clusters.sort(key=len, reverse=True)
            best = clusters[0]
            support = len(best) - 1
            # medoid of the winning cluster
            uniq = list(dict.fromkeys(s for _, s in best))
            gt = min(uniq, key=lambda a: sum(ham(a, b) for b in uniq))
            tier = "A" if len(best) >= 3 else ("B" if len(best) == 2 else "C")
        rows.append({
            "id": it["id"], "file": it["file"], "gt": gt, "tier": tier,
            "support": support, "n_reads": len(reads),
            "n_verifying": len(ver), "n_distinct": len(set(r for _, r in reads)),
            "rival": len(clusters) - 1 if clusters else 0,
        })
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(items)}", flush=True)

    json.dump({"set": args.set, "records": rows}, open(out_path, "w"), indent=1)
    n = len(rows)
    tiers = Counter(r["tier"] for r in rows)
    print(f"\nset={args.set} images={n}  " +
          "  ".join(f"{k}:{v}" for k, v in sorted(tiers.items())))
    good = [r for r in rows if r["tier"] in ("A", "B")]
    print(f"usable GT (A+B) = {len(good)}/{n}")
    print("rival clusters among tier A/B:",
          Counter(r["rival"] for r in good))
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
