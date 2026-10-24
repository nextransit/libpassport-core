#!/usr/bin/env python3
"""Candidate-level fusion across scales (the correct way to combine runs).

Why not per-character voting: the five ICAO check digits are a JOINT
constraint over the whole line.  Mixing column 5 from scale A with column 40
from scale B almost always lands in a region that satisfies neither.  Measured
on assets/pic: per-column majority 6/84 vs 30/84 for a single scale -- voting
actively destroys the result.

So fuse at the CANDIDATE level: each scale proposes whole lines, keep the
checksum-valid ones, and choose among them.

Selection rules
  single26 : one pass at 26 px/char (the validated normalization)
  cascade  : 26 first; on checksum failure try 20, 32, 16, 38 in order
  consens  : modal verifying candidate; ties -> closest to 26 px/char
  medoid   : verifying candidate with the smallest total Hamming distance to
             the other verifying candidates (robust to null-space freedom:
             two readings that differ only where the checksum is blind are
             still the same underlying reading)

Support of a choice = how many other verifying candidates sit within Hamming
distance 2 of it.  That is the only zero-cost corroboration available without
text GT.
"""
import json
import os
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

import cv2
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
TOOL = ROOT / "build" / "mrz_ocr_cnn_tool"
ASSETS = Path("/Users/zhouyong/gitlab/passport/assets/pic")
CORPUS = ROOT / "data" / "corpus_passport_photo"

W = (7, 3, 1)
SCALES = (16, 20, 26, 32, 38)
PREF = 26


def val(c):
    return 0 if c == "<" else (ord(c) - 48 if c.isdigit() else ord(c) - 55)


def cd(s):
    return sum(val(c) * W[i % 3] for i, c in enumerate(s)) % 10


def l2_ok(l):
    if not l or len(l) != 44:
        return False
    try:
        for p, (a, b) in ((9, (0, 9)), (19, (13, 19)), (27, (21, 27)), (42, (28, 42))):
            if l[p] != str(cd(l[a:b])):
                return False
        return l[43] == str(cd(l[0:10] + l[13:20] + l[21:28] + l[28:43]))
    except Exception:
        return False


def ham(a, b):
    return sum(1 for x, y in zip(a, b) if x != y) if len(a) == len(b) == 44 else 44


def run(ppm):
    env = dict(os.environ)
    env.update(MRZ_OCR_LINE_MODELS="1", MRZ_OCR_PAR="0", MRZ_OCR_CRNN="0")
    q = subprocess.run([str(TOOL), str(ppm)], capture_output=True, text=True, env=env)
    kv = {}
    for ln in q.stdout.splitlines():
        if ":" in ln:
            k, _, v = ln.partition(":")
            kv[k.strip()] = v.strip()
    return kv.get("result.line2", "")


def at_scale(band, t, path):
    ppc = band.shape[1] / 44.0
    s = max(1.0, t / ppc)
    img = band if abs(s - 1.0) < 1e-6 else cv2.resize(
        band, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
    Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).save(str(path), format="PPM")
    return path


def ink_rows(box):
    g = cv2.cvtColor(box, cv2.COLOR_BGR2GRAY) if box.ndim == 3 else box
    _, bw = cv2.threshold(g, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    prof = bw.mean(axis=1)
    thr = max(0.02, prof.mean() * 0.5)
    rows = prof > thr
    n = i = 0
    while i < len(rows):
        if rows[i]:
            n += 1
            while i < len(rows) and rows[i]:
                i += 1
        else:
            i += 1
    return n


def load():
    items = {"passport": [], "assets": []}
    for r in json.load(open(CORPUS / "corpus.json"))["records"]:
        b = cv2.imread(str(CORPUS / r["file"]))
        if b is not None:
            items["passport"].append((Path(r["file"]).stem, b, r["line2"], 2))
    for r in json.load(open(ASSETS / "catalog.json"))["records"]:
        loc = (r.get("loc") or {}).get("mrz")
        p = ASSETS / r["file"]
        if not loc or not p.exists():
            continue
        rgb = cv2.imread(str(p))
        if rgb is None:
            continue
        H, Wd = rgb.shape[:2]
        pad = 12
        b = rgb[max(0, loc["y"] - pad):min(H, loc["y"] + loc["h"] + pad),
                max(0, loc["x"]):min(Wd, loc["x"] + loc["w"])]
        if b.shape[0] < 16 or b.shape[1] < 44:
            continue
        items["assets"].append((p.stem, b, None, ink_rows(b)))
    return items


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "assets"
    tmp = Path(tempfile.mkdtemp())
    lst = load()[src]
    res = []
    for stem, band, gt, rows in lst:
        outs = {t: run(at_scale(band, t, tmp / f"{stem}_{t}.ppm")) for t in SCALES}
        verifying = [o for o in outs.values() if l2_ok(o)]
        single = outs[PREF]

        cascade = single
        if not l2_ok(cascade):
            for t in (20, 32, 16, 38):
                if l2_ok(outs[t]):
                    cascade = outs[t]
                    break

        consens = single
        if verifying:
            consens = Counter(verifying).most_common(1)[0][0]

        medoid, support = single, 0
        if verifying:
            uniq = list(dict.fromkeys(verifying))
            best, bestd = None, None
            for a in uniq:
                d = sum(ham(a, b) for b in uniq if b is not a)
                if bestd is None or d < bestd:
                    best, bestd = a, d
            medoid = best
            support = sum(1 for b in uniq if b is not best and ham(best, b) <= 2)

        res.append({
            "id": stem, "rows": rows, "gt": gt,
            "n_verifying": len(verifying), "n_distinct": len(set(outs.values())),
            "n_uniq_verifying": len(set(verifying)),
            "single": single, "cascade": cascade,
            "consens": consens, "medoid": medoid, "support": support,
        })

    n = len(res)
    print(f"\n=== {src} n={n} ===")
    if src == "passport":
        print(f"{'rule':10} {'exact':>9} {'char':>8}")
        for k in ("single", "cascade", "consens", "medoid"):
            ex = sum(1 for r in res if r[k] == r["gt"])
            ch = sum(sum(1 for i in range(44) if len(r[k]) == 44 and r[k][i] == r["gt"][i]) / 44
                     for r in res) / n
            print(f"{k:10} {ex:>4}/{n:<4} {ch * 100:>7.2f}%")
    else:
        print(f"{'rule':10} {'cksum':>9} {'supported':>10}")
        for k in ("single", "cascade", "consens", "medoid"):
            v = sum(1 for r in res if l2_ok(r[k]))
            print(f"{k:10} {v:>4}/{n:<4} "
                  f"{sum(1 for r in res if l2_ok(r[k]) and r['support'] >= 1):>10}")
        print("\nby ink-row count inside loc.mrz:")
        print(f"{'rows':>5} {'n':>4} {'single':>8} {'cascade':>8} {'medoid':>8}")
        for k in sorted({r["rows"] for r in res}):
            sub = [r for r in res if r["rows"] == k]
            print(f"{k:>5} {len(sub):>4} "
                  f"{sum(1 for r in sub if l2_ok(r['single'])):>8} "
                  f"{sum(1 for r in sub if l2_ok(r['cascade'])):>8} "
                  f"{sum(1 for r in sub if l2_ok(r['medoid'])):>8}")

    # scale stability: how many distinct line-2 strings does the model emit?
    dist = Counter(r["n_distinct"] for r in res)
    print(f"\nscale stability (distinct line-2 strings across {len(SCALES)} scales):")
    for k in sorted(dist):
        sub = [r for r in res if r["n_distinct"] == k]
        hit = sum(1 for r in sub if (r["single"] == r["gt"]) if src == "passport") \
            if src == "passport" else sum(1 for r in sub if l2_ok(r["single"]))
        print(f"  {k} distinct: {len(sub):>3} images, single-pass hits {hit}")
    stable = sum(1 for r in res if r["n_distinct"] == 1)
    print(f"  -> identical at every scale: {stable}/{n} ({stable / n * 100:.0f}%)")

    json.dump(res, open(f"/tmp/fuse_{src}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
