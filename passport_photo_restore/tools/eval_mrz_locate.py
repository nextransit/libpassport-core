#!/usr/bin/env python3
"""Evaluate MRZ row localisation on assets/pic against an INDEPENDENT truth.

Why this tool exists
--------------------
`assets/pic/catalog.json` carries a `loc.mrz` box, but its evidence field
says it outright:

    "bottom band: 82 dense dark rows (row projection > 15% width)"

That is the same row-projection heuristic the locator itself uses, so
scoring the locator against it is circular.  The box in `catalog.json`
cannot referee a change to the row-projection rule.

The independent reference used here is `catalog.json -> ocr.crop`: the
region the ON-BOARD pipeline actually cropped and fed to its OCR, kept
only for the samples where the board produced two valid MRZ lines
(`ocr.ok == true`).  That crop comes from the board's own pipeline, not
from anything in this repository's locator, so it can referee a change
to the locator.

What is measured
----------------
For each sample the locator returns two MRZ rows.  A sample counts as a
hit when BOTH rows start inside the board's crop (8 px of slack for the
row-projection edge).  We report HEAD vs the working tree so a change
can be judged on the samples it fixes and the samples it breaks.

Usage
-----
    python3 tools/eval_mrz_locate.py
    python3 tools/eval_mrz_locate.py --out /tmp/locate.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys

import cv2
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "passport_photo_restore" / "src"))

CATALOG = ROOT / "8_android" / "app" / "src" / "main" / "assets" / "pic" / "catalog.json"
PIC_DIR = ROOT / "8_android" / "app" / "src" / "main" / "assets" / "pic"


def load_restore(path: pathlib.Path, name: str):
    """Import a restore.py under a private module name."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def locate(mod, rgb) -> tuple | None:
    """Run the locator and return the two rows in SOURCE coordinates."""
    work, scale = mod.load(rgb)
    st = mod.detect_structure(mod._gray(work))
    pair = st.get("mrz_pair")
    if not pair:
        return None
    return tuple((int(a / scale), int(b / scale)) for a, b in pair)


def evaluate(mod, samples, slack: int = 8):
    hits = 0
    rows = []
    for rec in samples:
        pair = locate(mod, rec["rgb"])
        crop = rec["crop"]
        y0, y1 = crop["y"], crop["y"] + crop["h"]
        inside = False
        if pair:
            inside = (y0 - slack <= pair[0][0] <= y1 and
                      y0 - slack <= pair[1][0] <= y1)
        hits += inside
        rows.append({
            "file": rec["file"],
            "crop": [y0, y1],
            "rows": [list(p) for p in pair] if pair else None,
            "hit": bool(inside),
        })
    return hits, rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="")
    ap.add_argument("--slack", type=int, default=8)
    args = ap.parse_args()

    cat = json.loads(CATALOG.read_text())
    samples = []
    for rec in cat["records"]:
        ocr = rec.get("ocr") or {}
        crop = ocr.get("crop")
        # Only the samples the board itself read successfully: there the
        # crop is a known-good MRZ region rather than a guess.
        if not ocr.get("ok") or not crop:
            continue
        path = PIC_DIR / rec["file"]
        if not path.exists():
            continue
        rgb = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        samples.append({"file": rec["file"], "rgb": rgb, "crop": crop})

    if not samples:
        print("no samples with a board-verified MRZ crop", file=sys.stderr)
        return 2

    variants = {
        "working-tree": load_restore(
            ROOT / "passport_photo_restore" / "src" / "restore.py", "rst_work"),
    }
    head_path = ROOT / "passport_photo_restore" / "tools" / "restore_head.py"
    if head_path.exists():
        variants["HEAD"] = load_restore(head_path, "rst_head")

    results = {}
    for name, mod in variants.items():
        hits, rows = evaluate(mod, samples, args.slack)
        results[name] = {"hits": hits, "total": len(samples), "rows": rows}
        print(f"{name:<14} {hits}/{len(samples)} "
              f"({hits * 100 // len(samples)}%) 两行都落在板端 crop 内")

    names = list(results)
    if len(names) == 2:
        a = {r["file"]: r["hit"] for r in results[names[0]]["rows"]}
        b = {r["file"]: r["hit"] for r in results[names[1]]["rows"]}
        both = sum(1 for f in a if a[f] and b[f])
        only_a = sum(1 for f in a if a[f] and not b[f])
        only_b = sum(1 for f in a if b[f] and not a[f])
        neither = len(a) - both - only_a - only_b
        print(f"\n{names[0]} 与 {names[1]} 对比: 都中={both} "
              f"仅{names[0]}={only_a} 仅{names[1]}={only_b} 都错={neither}")

    if args.out:
        pathlib.Path(args.out).write_text(
            json.dumps({k: {"hits": v["hits"], "total": v["total"]}
                        for k, v in results.items()},
                       ensure_ascii=False, indent=1))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
