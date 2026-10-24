#!/usr/bin/env python3
"""Multi-scale posterior fusion vs single-pass, on both real corpora.

Motivation (literature): Bulatov 2019 (DOI 10.14529/mmp190307) shows that
combining several recognition results *with per-character alternatives*
beats any single run.  Our beam only explores top-3 alternatives of ONE
segmentation.  The cheap version of that idea is to decode the same band
at several px-per-character scales and fuse per column.

Measures, on the same images:
  base      : one pass at the native scale
  norm26    : one pass after upscaling to 26 px/char (the rule we validated)
  vote      : per-column majority over {native, 20, 26, 32} px/char
  cascade   : norm26, but retry the other scales only if checksum fails
  tta       : norm26 with MRZ_OCR_TTA=1

For PassPortPhoto we have objective line-2 GT, so exact/char are reported.
For assets/pic there is no text GT: only the checksum pass rate, which is a
lower bound on correctness (see report section 2 -- it is a rejector, not an
acceptor) plus the CNN/CRNN agreement rate.
"""
import json
import os
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
TOOL = ROOT / "build" / "mrz_ocr_cnn_tool"
ASSETS = Path("/Users/zhouyong/gitlab/passport/assets/pic")
CORPUS = ROOT / "data" / "corpus_passport_photo"

W = (7, 3, 1)


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


def run(ppm, crnn=0, tta=False):
    env = dict(os.environ)
    env.update(MRZ_OCR_LINE_MODELS="1", MRZ_OCR_PAR="0", MRZ_OCR_CRNN=str(crnn))
    if tta:
        env["MRZ_OCR_TTA"] = "1"
    else:
        env.pop("MRZ_OCR_TTA", None)
    q = subprocess.run([str(TOOL), str(ppm)], capture_output=True, text=True, env=env)
    kv = {}
    for ln in q.stdout.splitlines():
        if ":" in ln:
            k, _, v = ln.partition(":")
            kv[k.strip()] = v.strip()
    return kv.get("result.line1", ""), kv.get("result.line2", "")


def write_ppm(img, path):
    Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).save(str(path), format="PPM")
    return path


def scaled(band, target_ppc):
    """Resize so that band_w/44 == target_ppc.  Never downscale."""
    ppc = band.shape[1] / 44.0
    s = max(1.0, target_ppc / ppc)
    if abs(s - 1.0) < 1e-6:
        return band, 1.0
    return cv2.resize(band, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC), s


def vote(outs, prefer_idx):
    """Per-column majority over the 44-char outputs; tie -> prefer_idx."""
    ok = [o for o in outs if len(o) == 44]
    if not ok:
        return ""
    out = []
    for i in range(44):
        cnt = Counter(o[i] for o in ok)
        best, n = cnt.most_common(1)[0]
        if list(cnt.values()).count(n) > 1:
            best = outs[prefer_idx][i] if len(outs[prefer_idx]) == 44 else best
        out.append(best)
    return "".join(out)


def load_items():
    items = {"passport": [], "assets": []}
    recs = json.load(open(CORPUS / "corpus.json"))["records"]
    for r in recs:
        band = cv2.imread(str(CORPUS / r["file"]))
        if band is not None:
            items["passport"].append((Path(r["file"]).stem, band, r["line2"]))
    cat = json.load(open(ASSETS / "catalog.json"))
    for r in cat["records"]:
        loc = (r.get("loc") or {}).get("mrz")
        p = ASSETS / r["file"]
        if not loc or not p.exists():
            continue
        rgb = cv2.imread(str(p))
        if rgb is None:
            continue
        H, Wd = rgb.shape[:2]
        pad = 12
        y0 = max(0, loc["y"] - pad)
        y1 = min(H, loc["y"] + loc["h"] + pad)
        x0 = max(0, loc["x"])
        x1 = min(Wd, loc["x"] + loc["w"])
        band = rgb[y0:y1, x0:x1]
        if band.shape[0] < 16 or band.shape[1] < 44:
            continue
        items["assets"].append((p.stem, band, None))
    return items


SCALES = (20, 26, 32)


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else None
    tmp = Path(tempfile.mkdtemp())
    items = load_items()
    for src in ("passport", "assets"):
        if only and src != only:
            continue
        lst = items[src]
        acc = {k: [0, 0.0] for k in ("base", "norm26", "vote", "cascade", "tta")}
        n = 0
        for stem, band, gt in lst:
            n += 1
            native = write_ppm(band, tmp / f"{stem}_nat.ppm")
            per_scale = {}
            for t in SCALES:
                img, _ = scaled(band, t)
                per_scale[t] = write_ppm(img, tmp / f"{stem}_{t}.ppm")

            b_l2 = run(native)[1]
            n26_l2 = run(per_scale[26])[1]
            tta_l2 = run(per_scale[26], tta=True)[1]
            outs = [b_l2] + [run(per_scale[t])[1] for t in SCALES]
            v_l2 = vote(outs, prefer_idx=2)  # index 2 == the 26 px/char run

            # cascade: keep norm26 if it verifies, else first verifying scale
            cas = n26_l2
            if not l2_ok(cas):
                for t in SCALES:
                    o = run(per_scale[t])[1]
                    if l2_ok(o):
                        cas = o
                        break

            for key, o in (("base", b_l2), ("norm26", n26_l2), ("vote", v_l2),
                           ("cascade", cas), ("tta", tta_l2)):
                if gt is None:
                    acc[key][0] += l2_ok(o)
                else:
                    acc[key][0] += (o == gt)
                    acc[key][1] += sum(1 for i in range(44) if len(o) == 44 and o[i] == gt[i]) / 44.0

        print(f"\n=== {src}  n={n} " + ("(objective line-2 GT: exact / char)" if src == "passport" else "(checksum pass rate)") + " ===")
        for k, (hit, ch) in acc.items():
            if src == "passport":
                print(f"  {k:8} {hit:>3}/{n:<3} exact  {ch / n * 100:>6.2f}% char")
            else:
                print(f"  {k:8} {hit:>3}/{n:<3} cksum   {hit / n * 100:>6.2f}%")


if __name__ == "__main__":
    main()
