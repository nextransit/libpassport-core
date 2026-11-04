#!/usr/bin/env python3
"""Evaluate a recogniser binary against the FROZEN ground truth.

Why this exists next to eval_two_sets.py: that script derives its ground
truth by majority vote of the very tool it is measuring (`votes[l2] += 1`
inside its own decode loop).  Comparing two weight sets with it is
circular -- the assets/pic binary, which has no CRNN of its own, collapses
to a single reading repeated six times, reaches "consensus" on 37 of 65
images and scores a perfect 37/37.  That number measures nothing.

This script reads tests/gt_assets_pic.json or tests/gt_photo_passport.json,
which were frozen from multi-engine votes plus human review, and never
looks at the tool's output when deciding what the right answer is.

Band framing is deliberately identical to tools/export_photo_train.py
(fast_pair / adaptive_xpad / native_band, pad=10): the weights were trained
on glyphs resampled through exactly that crop, so evaluating through a
different crop would measure a train/inference mismatch instead of the
model.

Usage:
  python3 tools/eval_weight_sets.py --gt tests/gt_assets_pic.json \
      --tool build/mrz_ocr_cnn_tool_pic [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import mrz_band  # noqa: E402

ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ<"
PAD = 10


def parse_kv(out: str) -> dict:
    kv = {}
    for ln in out.splitlines():
        if ":" in ln:
            k, _, v = ln.partition(":")
            kv[k.strip()] = v.strip()
    return kv


def run(tool: Path, ppm: Path) -> dict:
    env = dict(os.environ)
    env["MRZ_OCR_LINE_MODELS"] = "1"
    env["MRZ_OCR_PAR"] = "1"
    env["MRZ_OCR_CRNN"] = "0"
    p = subprocess.run([str(tool), str(ppm)], capture_output=True,
                       text=True, timeout=120, env=env)
    kv = parse_kv(p.stdout)
    return {
        "ok": "OK" in kv.get("result.ok", ""),
        "line1": kv.get("result.line1", ""),
        "line2": kv.get("result.line2", ""),
        "conf1": int(kv.get("result.conf1", 0) or 0),
        "conf2": int(kv.get("result.conf2", 0) or 0),
        "layout": kv.get("result.layoutname", "?"),
        "cksum": int(kv.get("result.cksum", 0) or 0),
    }


def tier_ok(rec: dict, line: int) -> bool:
    """A/B (or untiered) labels are trusted; tier C is not."""
    t = rec.get("l1_tier" if line == 1 else "l2_tier") or rec.get("tier") or ""
    return t in ("A", "B", "")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--tool", required=True)
    ap.add_argument("--json", default="")
    ap.add_argument("--label", default="")
    # Photo-level holdout support.  train_cnn_omtest.py --holdout-photos
    # removes whole photographs, so the only honest generalisation number
    # is measured on exactly those photographs and nothing else.  The file
    # is the JSON list of basenames the trainer printed.
    ap.add_argument("--only-photos", default="",
                    help="JSON list of photo basenames to restrict to")
    ap.add_argument("--pad-map", default="tests/photo_pad_map.json",
                    help="JSON file mapping image basename (or label) to crop padding")
    args = ap.parse_args()

    pad_map = {}
    pm_path = ROOT / args.pad_map
    if args.pad_map and pm_path.exists():
        pad_map = {k: int(v) for k, v in json.loads(pm_path.read_text()).items()}

    only = None
    if args.only_photos:
        only = [Path(s).as_posix() for s in json.loads(Path(args.only_photos).read_text())]

    tool = Path(args.tool).resolve()
    if not tool.exists():
        print(f"tool not found: {tool}", file=sys.stderr)
        return 2

    d = json.loads((ROOT / args.gt).read_text())
    gt = d["gt"] if isinstance(d, dict) and "gt" in d else d
    tmp = Path(tempfile.mkdtemp(prefix="wset_"))

    rows = []
    skipped_band = 0
    for name, rec in sorted(gt.items()):
        label = rec.get("label") or ""
        l1, l2 = rec.get("l1") or "", rec.get("l2") or ""
        img_path = ROOT / label
        if not img_path.exists():
            continue
        if only is not None and not any(label.endswith(o) for o in only):
            continue
        try:
            rgb = Image.open(img_path).convert("RGB")
        except Exception:
            continue
        cands = mrz_band.band_candidates(rgb, PAD,
                                         pad_map.get(name, pad_map.get(label)))
        if not cands:
            skipped_band += 1
            rows.append({"id": name, "located": False})
            continue
        scored = []
        for i, (_kind, rect) in enumerate(cands):
            ppm = tmp / (Path(name).stem + f"_c{i}.ppm")
            rgb.crop(rect).save(str(ppm), format="PPM")
            scored.append(mrz_band.run_tool(tool, ppm, {"MRZ_OCR_PAR": "1"}))
        pick = mrz_band.pick_candidate(scored, [k for k, _ in cands])
        r = dict(scored[pick])
        r.update(id=name, located=True, gt1=l1, gt2=l2,
                 band=cands[pick][0], rect=list(cands[pick][1]),
                 l1_ok=tier_ok(rec, 1), l2_ok=tier_ok(rec, 2))
        rows.append(r)

    def acc(sel, pred_k, gt_k):
        ok = tot = 0
        for r in sel:
            g = r.get(gt_k) or ""
            p = r.get(pred_k) or ""
            if len(g) != 44:
                continue
            tot += 44
            if len(p) != 44:
                continue
            for a, b in zip(g, p):
                ok += (a == b)
        return ok, tot

    def exact(sel, pred_k, gt_k):
        return sum(1 for r in sel
                   if (r.get(gt_k) or "") == (r.get(pred_k) or "")
                   and len(r.get(gt_k) or "") == 44)

    sel = [r for r in rows if r.get("located")]
    l1_sel = [r for r in sel if r.get("l1_ok")]
    l2_sel = [r for r in sel if r.get("l2_ok")]

    o1, t1 = acc(l1_sel, "line1", "gt1")
    o2, t2 = acc(l2_sel, "line2", "gt2")
    n_empty = sum(1 for r in l1_sel
                  if len(r.get("line1") or "") != 44)
    print(f"tool = {tool.name}   gt = {args.gt}")
    print(f"records {len(rows)}  located {len(sel)}  "
          f"band-fail {skipped_band}  empty-output {n_empty}")
    print(f"line1: exact {exact(l1_sel, 'line1', 'gt1')}/{len(l1_sel)}  "
          f"char {o1}/{t1} = {o1/t1 if t1 else 0:.4f}  "
          f"({o1}/{len(l1_sel)*44} strict = {o1/(len(l1_sel)*44) if l1_sel else 0:.4f})")
    print(f"line2: exact {exact(l2_sel, 'line2', 'gt2')}/{len(l2_sel)}  "
          f"char {o2}/{t2} = {o2/t2 if t2 else 0:.4f}  "
          f"({o2}/{len(l2_sel)*44} strict = {o2/(len(l2_sel)*44) if l2_sel else 0:.4f})")
    both = [r for r in sel
            if r.get("l1_ok") and r.get("l2_ok")
            and len(r.get("gt1") or "") == 44 and len(r.get("gt2") or "") == 44]
    both_ok = sum(1 for r in both
                  if r["line1"] == r["gt1"] and r["line2"] == r["gt2"])
    print(f"both : exact {both_ok}/{len(both)}")
    layouts = {}
    for r in sel:
        layouts[r.get("layout", "?")] = layouts.get(r.get("layout", "?"), 0) + 1
    print(f"layout: {layouts}")

    if args.json:
        Path(args.json).write_text(json.dumps(rows, ensure_ascii=False, indent=1))
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
