#!/usr/bin/env python3
"""A/B the MRZ_OCR_CELL window-width mode over every exported product band.

The C side chooses the resampling window for a character cell:
  * default   -> cell = medw + 8 (the median INK-RUN width plus padding)
  * pitch     -> cell = the fitted grid pitch L
  * pitch090  -> 90% of the pitch

`medw` is the median width of the ink RUNS, and on a photo whose glyphs
touch, several runs merge and the median inflates: on Photo/3 the report
says medw=34 while the glyph ink is only ~15 px wide, so the 42 px window
holds one glyph plus a slice of its neighbour.  The pitch mode uses the
grid spacing instead, which is the physically correct cell width.

This script runs the C tool on every band in /tmp/product_export with each
mode and reports the exact-match count against the ground truth.
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys
from pathlib import Path

ROOT = Path("/Users/zhouyong/Desktop/work/Decard/gitlab/passport")
TOOL = ROOT / "6_mrz_ocr/build/mrz_ocr_cnn_tool"


def gt_map():
    g = {}
    for r in json.load(open(ROOT / "10_ocr_research/photo_direct_bench.json"))["records"]:
        if r.get("gt"):
            g[Path(r["path"]).name] = r["gt"]
    auto = json.load(open(ROOT / "tests/photo_auto_gt.json"))
    auto = auto.get("gt", auto)
    for k, v in auto.items():
        if k in g:
            continue
        if len(v.get("l1", "")) == 44 and len(v.get("l2", "")) == 44:
            g[k] = {"line1": v["l1"], "line2": v["l2"]}
    return g


def run(tool, ppm, cell_mode, extra_env=None):
    env = dict(os.environ)
    env.update({"MRZ_OCR_LINE_MODELS": "1", "MRZ_OCR_CRNN": "0", "MRZ_OCR_FUSE": "0"})
    if cell_mode:
        env["MRZ_OCR_CELL"] = cell_mode
    else:
        env.pop("MRZ_OCR_CELL", None)
    if extra_env:
        env.update(extra_env)
    r = subprocess.run([str(tool), str(ppm)], capture_output=True, text=True, env=env)
    kv = {}
    for l in r.stdout.splitlines():
        if ":" in l:
            k, _, v = l.partition(":")
            kv[k.strip()] = v.strip()
    return kv.get("result.line1", ""), kv.get("result.line2", "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bands", default="/tmp/product_export")
    ap.add_argument("--tool", default=str(TOOL))
    ap.add_argument("--modes", default="default,pitch,pitch090")
    ap.add_argument("--engines", default="cnn")
    args = ap.parse_args()

    gts = gt_map()
    bands = sorted(Path(args.bands).glob("*.ppm"))
    results = {}
    for mode in args.modes.split(","):
        cm = "" if mode == "default" else mode
        rows = {}
        for p in bands:
            l1, l2 = run(args.tool, p, cm)
            rows[p.stem] = (l1, l2)
        results[mode] = rows

    names = list(results)
    # summary
    for mode in names:
        rows = results[mode]
        n = ex = 0
        photo_rows = []
        for stem, (l1, l2) in rows.items():
            gt = gts.get(stem + ".jpg") or gts.get(stem)
            if not gt:
                continue
            g1, g2 = gt.get("line1", ""), gt.get("line2", "")
            if len(g1) != 44 or len(g2) != 44:
                continue
            n += 1
            ok = (l1 == g1 and l2 == g2)
            ex += int(ok)
            if not ok:
                photo_rows.append((stem, l1, l2, g1, g2))
        print(f"MRZ_OCR_CELL={mode:9s} exact={ex}/{n}")

    if len(names) >= 2:
        base = names[0]
        for mode in names[1:]:
            fixed, broke = [], []
            for stem, (l1, l2) in results[mode].items():
                gt = gts.get(stem + ".jpg") or gts.get(stem)
                if not gt:
                    continue
                g1, g2 = gt.get("line1", ""), gt.get("line2", "")
                if len(g1) != 44 or len(g2) != 44:
                    continue
                ob = (l1 == g1 and l2 == g2)
                b1, b2 = results[base][stem]
                on = (b1 == g1 and b2 == g2)
                if ob and not on:
                    fixed.append(stem)
                if not ob and on:
                    broke.append(stem)
            print(f"\n[{base} -> {mode}] fixed={len(fixed)} broke={len(broke)}")
            for s in fixed:
                print("   +", s)
            for s in broke:
                print("   -", s)

    # detail for the last mode on the Photo/ set
    print("\n--- detail (last mode) ---")
    last = names[-1]
    for stem, (l1, l2) in sorted(results[last].items()):
        if not stem.startswith("IMG"):
            continue
        gt = gts.get(stem + ".jpg")
        if not gt:
            continue
        g1, g2 = gt.get("line1", ""), gt.get("line2", "")
        if len(g1) != 44:
            continue
        m1 = "".join(" " if a == b else "^" for a, b in zip(g1, l1)) if len(l1) == 44 else ""
        m2 = "".join(" " if a == b else "^" for a, b in zip(g2, l2)) if len(l2) == 44 else ""
        print(f"{stem}")
        print(f"  gt1={g1}")
        print(f"  o1={l1}")
        print(f"     {m1}")
        print(f"  gt2={g2}")
        print(f"  o2={l2}")
        print(f"     {m2}")


if __name__ == "__main__":
    main()
