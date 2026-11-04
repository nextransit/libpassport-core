#!/usr/bin/env python3
"""Does glyph WIDTH separate the classes the CNN confuses?

Motivation
----------
On the real-photo eval set two columns of `2026-9-30-AC` are read wrong with
softmax probability 1.000 -- `O`->`0` and `7`->`1`.  Both models (the generic
one and the line-specific pair) agree, and TTA does not move them, so the
information is missing from the 16x12 patch rather than being noisy.

OCR-B draws `O` and `0` as ellipses of *different width*, and `7` as a wide
top bar over a diagonal while `1` is a narrow flag over a stem.  All of that
is a horizontal-extent difference, which a 12x16 resample preserves but a
per-glyph bbox normalisation would destroy.

This script measures the horizontal ink extent of every cell in the corpus
and reports, per ground-truth character, the width distribution.  If the
classes the CNN confuses occupy disjoint width ranges, width is a usable
tie-breaker; if they overlap, the loss is in the resample and only a higher
input resolution (or more data) can recover it.

Usage:
    python3 tools/diag_glyph_width.py [--omtest DIR] [--tag NAME]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import eval_prod_path as EP  # noqa: E402
import eval_omtest_e2e as EE  # noqa: E402

TOOL = ROOT / "build" / "mrz_ocr_cnn_tool"


def read_ppm_p6(path: Path):
    d = path.read_bytes()
    if not d.startswith(b"P6"):
        raise ValueError(f"not P6: {path}")
    # header: P6 <w> <h> <max>\n
    fields = []
    i = 2
    while len(fields) < 3:
        while i < len(d) and d[i : i + 1].isspace():
            i += 1
        if d[i : i + 1] == b"#":
            while d[i : i + 1] != b"\n":
                i += 1
            continue
        j = i
        while j < len(d) and not d[j : j + 1].isspace():
            j += 1
        fields.append(int(d[i:j]))
        i = j
    i += 1  # single whitespace after maxval
    w, h, _ = fields
    px = d[i:]
    if len(px) < w * h * 3:
        raise ValueError(f"short pixel data in {path}")
    return w, h, px


def ink_extent(path: Path):
    """Return (first_col, last_col, width, ink_total) of a 16x12 dump."""
    w, h, px = read_ppm_p6(path)
    cols = []
    total = 0.0
    for x in range(w):
        s = 0
        for y in range(h):
            s += px[(y * w + x) * 3]
        cols.append(s)
        total += s
    mx = max(cols) if cols else 0
    thr = mx * 0.25
    used = [x for x, s in enumerate(cols) if s > thr]
    if not used:
        return 0, -1, 0, 0.0
    return used[0], used[-1], used[-1] - used[0] + 1, total / 255.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--omtest", default=str(EP.DEFAULT_OMTEST))
    ap.add_argument("--tag", default="gw")
    ap.add_argument("--dump-dir", default="/tmp/mrz_glyphw")
    args = ap.parse_args()

    om = Path(args.omtest)
    dump = Path(args.dump_dir)
    dump.mkdir(parents=True, exist_ok=True)
    for f in dump.glob("*.ppm"):
        f.unlink()

    gt = EP.load_corrected_truth()
    tmp = Path("/tmp/mrz_eval_prod")
    tmp.mkdir(parents=True, exist_ok=True)

    # (gt_char, line_idx) -> list of widths
    widths: dict[tuple[str, int], list[int]] = defaultdict(list)
    # per sample: list of (col, gt_char, pred_char, width, p0)
    details = []

    n = 0
    for d in sorted(p for p in om.iterdir() if p.is_dir()):
        for txt in sorted((om / d.name).glob("*.jpg-result.txt")):
            img = txt.with_name(txt.name.replace("-result.txt", ""))
            if not img.exists():
                continue
            truth2 = gt.get(img.name) or gt.get(img.stem)
            if not truth2 or len(truth2) != 44:
                continue
            band = tmp / (img.stem + "_band.ppm")
            if not band.exists():
                continue
            n += 1
            tag = f"{args.tag}{n:03d}"
            env = dict(os.environ)
            env["MRZ_OCR_LINE_MODELS"] = "1"
            env["MRZ_OCR_PAR"] = "1"
            env["MRZ_OCR_DUMP"] = tag
            env["MRZ_OCR_DUMP_DIR"] = str(dump)
            env["MRZ_OCR_TOPK"] = "1"
            r = subprocess.run([str(TOOL), str(band)], capture_output=True,
                               text=True, env=env, timeout=120)
            line2 = ""
            for ln in r.stdout.splitlines():
                if ln.startswith("result.line2"):
                    line2 = ln.partition(":")[2].strip()
            if len(line2) != 44:
                continue
            # per-column top-1 probability from TOPK lines of line 1 (idx 1)
            p0 = {}
            for ln in r.stderr.splitlines():
                if not ln.startswith("TOPK L1 c"):
                    continue
                try:
                    col = int(ln.split("c")[1].split(":")[0])
                    first = ln.split(":", 1)[1].strip().split()[0]
                    p0[col] = float(first[2:-1])
                except Exception:
                    pass
            for c in range(44):
                pat = list(dump.glob(f"mrz_dump_{tag}_l1_c{c:02d}_*.ppm"))
                if not pat:
                    continue
                _, _, wid, _ = ink_extent(pat[0])
                g = truth2[c]
                widths[(g, 1)].append(wid)
                details.append((f"{d.name}/{img.name}", c, g, line2[c], wid,
                                p0.get(c, -1.0)))
            print(f"  {d.name}/{img.name}  cols={len(details)}", file=sys.stderr)

    print(f"\n=== samples with band+truth: {n} ===")
    print("\n=== horizontal ink extent (of 16) by ground-truth char, line 2 ===")
    print(f"{'char':>4} {'n':>4} {'min':>4} {'p10':>4} {'med':>4} {'p90':>4} {'max':>4}  histogram")
    for ch in sorted({k[0] for k in widths if k[1] == 1}):
        v = sorted(widths[(ch, 1)])
        if not v:
            continue
        q = lambda f: v[min(len(v) - 1, int(len(v) * f))]
        hist = defaultdict(int)
        for x in v:
            hist[x] += 1
        hs = " ".join(f"{k}:{hist[k]}" for k in sorted(hist))
        print(f"{ch:>4} {len(v):>4} {v[0]:>4} {q(0.1):>4} {q(0.5):>4} {q(0.9):>4} {v[-1]:>4}  {hs}")

    print("\n=== confusable pairs: width overlap ===")
    pairs = [("O", "0"), ("1", "7"), ("I", "1"), ("T", "1"), ("B", "8"),
             ("S", "5"), ("Z", "2"), ("G", "6"), ("D", "0"), ("Q", "0")]
    for a, b in pairs:
        va = sorted(widths.get((a, 1), []))
        vb = sorted(widths.get((b, 1), []))
        if not va or not vb:
            print(f"  {a}/{b}: insufficient data ({len(va)}/{len(vb)})")
            continue
        qa = lambda f: va[min(len(va) - 1, int(len(va) * f))]
        qb = lambda f: vb[min(len(vb) - 1, int(len(vb) * f))]
        # best single threshold separating them
        allv = sorted(set(va + vb))
        best = (0.0, None)
        for t in allv:
            ca = sum(1 for x in va if x <= t) / len(va)
            cb = sum(1 for x in vb if x > t) / len(vb)
            acc = (ca * len(va) + cb * len(vb)) / (len(va) + len(vb))
            if acc > best[0]:
                best = (acc, t)
        print(f"  {a}/{b}: {a} med={qa(0.5)} n={len(va)}   {b} med={qb(0.5)} n={len(vb)}"
              f"   best-split={best[1]} acc={best[0]*100:.1f}%")

    print("\n=== wrong columns on the eval set ===")
    bad = [x for x in details if x[2] != x[3]]
    for name, c, g, p, wid, pp in bad:
        print(f"  {name:<36} c{c:02d} gt={g} pred={p} width={wid} p0={pp:.3f}")
    print(f"  total wrong cells: {len(bad)} / {len(details)}")

    out = Path("/tmp/mrz_glyph_width.json")
    out.write_text(json.dumps(
        [{"sample": a, "col": b, "gt": c, "pred": d, "width": e, "p0": f}
         for a, b, c, d, e, f in details], indent=1))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
