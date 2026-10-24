#!/usr/bin/env python3
"""Automated labeling pipeline: annotate -> OCR -> flag FALSE -> manual QA.

Covers the three photo corpora:

  tmp/PassPortPhoto (44)   real handheld photos, machine-position
  Photo (6)                passport-scanner captures, NO ground truth yet
  assets/pic (87 files)    scraped web scans, dirty

For each image the pipeline:
  1. renders the MRZ band the recogniser used,
  2. runs the engines (CNN + CRNN + fused),
  3. compares against the best available reference:
       - PassPortPhoto : gt_frozen_passport_corrected.json (41 entries)
       - assets/pic    : tests/real_gt.json manual labels (5 entries)
       - Photo         : no reference yet
  4. emits a FALSE list plus a band-crop context sheet for the human.

Output:
  /tmp/auto_label/<corpus>_report.json      per-photo verdicts
  /tmp/auto_label/<corpus>_false.json        FALSE list
  /tmp/auto_label/context_<corpus>/          band crops for manual QA

Usage:
  python3 tools/auto_label_pipeline.py --corpus passport|photo|assets|all
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

import cv2
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "build" / "mrz_ocr_cnn_tool"
OMTEST = ROOT.parent / "tmp" / "omtest"
PHOTO = ROOT.parent / "Photo"
ASSETS = ROOT.parent / "assets" / "pic"
OUT = Path("/tmp/auto_label")

W = (7, 3, 1)


def val(c: str) -> int:
    if c == "<":
        return 0
    if c.isdigit():
        return int(c)
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


def l1_ok(s: str) -> bool:
    return len(s) == 44


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
    return {"l1": kv.get("result.line1", ""), "l2": kv.get("result.line2", ""),
            "conf1": int(kv.get("result.conf1", "0") or 0),
            "conf2": int(kv.get("result.conf2", "0") or 0),
            "ok": kv.get("result.ok", "")}


def crop_passport(jpg: Path, log: Path | None):
    if log is None:
        return None
    pts = [(int(a), int(b)) for a, b in
           re.findall(r"\[x: (\d+), y: (\d+)\]",
                      log.read_text(errors="replace"))]
    boxes = [pts[i:i + 4] for i in range(0, len(pts), 4)]
    if len(boxes) < 2:
        return None
    rgb = cv2.imread(str(jpg))
    if rgb is None:
        return None
    H, Wd = rgb.shape[:2]
    cy = H - (H // 3) - 50
    y1 = max(0, min(p[1] for p in boxes[0]) + cy)
    y2 = min(H, max(p[1] for p in boxes[1]) + cy)
    return rgb[max(0, y1 - 10):min(H, y2 + 1 + 10), 0:Wd]


def auto_crop(jpg: Path):
    rgb = cv2.imread(str(jpg))
    if rgb is None:
        return None
    H, W = rgb.shape[:2]
    y1 = max(0, H - H // 3 - 20)
    return rgb[y1:H, 0:W]


def load_ref_passport():
    ref = {}
    p = ROOT / "data" / "gt_frozen_passport_corrected.json"
    if p.exists():
        for r in json.loads(p.read_text())["records"]:
            ref[r["id"].replace(".jpg", "")] = {"l1": r.get("line1", ""),
                                                "l2": r["line2"]}
    return ref


def load_ref_assets():
    ref = {}
    p = ROOT.parent / "tests" / "real_gt.json"
    if p.exists():
        for name, g in json.loads(p.read_text())["gt"].items():
            ref[name] = {"l1": g.get("line1") or "", "l2": g.get("line2") or ""}
    return ref


def run_corpus(name, items, crop_fn, ref, out_sub, max_show=60):
    """items: list of (jpg, log|None, tag)"""
    false, verdicts = [], []
    out_sub.mkdir(parents=True, exist_ok=True)
    for jpg, log, tag in items:
        band = crop_fn(jpg, log)
        if band is None:
            verdicts.append({"id": tag, "err": "no band"})
            continue
        ppm = Path(tempfile.gettempdir()) / f"al_{name}_{tag.replace('/','_')}.ppm"
        Image.fromarray(cv2.cvtColor(band, cv2.COLOR_BGR2RGB)).save(
            str(ppm), format="PPM")
        outs = {m: decode(ppm, m) for m in ("cnn", "crnn", "fuse")}
        gt = ref.get(tag) or ref.get(tag.split("/")[-1])
        rec = {"id": tag,
               "gt_l1": gt["l1"] if gt else None,
               "gt_l2": gt["l2"] if gt else None,
               "outs": outs,
               "l2_ok": {m: l2_ok(o["l2"]) for m, o in outs.items()},
               "l1_ok": {m: l1_ok(o["l1"]) for m, o in outs.items()}}
        f = None
        if gt:
            for m, o in outs.items():
                l2_exact = (o["l2"] == gt["l2"])
                l1_exact = (not gt["l1"]) or (o["l1"] == gt["l1"])
                exact = l2_exact and l1_exact
                if not exact or not l2_ok(o["l2"]):
                    f = f or {"id": tag, "mode": m, "gt_l1": gt["l1"],
                              "gt_l2": gt["l2"], "pred_l1": o["l1"],
                              "pred_l2": o["l2"], "exact": exact}
        else:
            if not any(l2_ok(o["l2"]) for o in outs.values()):
                f = {"id": tag, "mode": "unverified_l2",
                     "l2_ok": rec["l2_ok"], "l1_ok": rec["l1_ok"],
                     "pred_l2": outs["cnn"]["l2"]}
        if f:
            false.append(f)
        verdicts.append(rec)
        Image.fromarray(cv2.cvtColor(band, cv2.COLOR_BGR2RGB)).save(
            str(out_sub / f"{tag.replace('/','_')}.png"), format="PNG")
    (OUT / f"{name}_report.json").write_text(
        json.dumps({"corpus": name, "n": len(items), "false": len(false),
                    "verdicts": verdicts}, indent=1, ensure_ascii=False))
    (OUT / f"{name}_false.json").write_text(
        json.dumps(false, indent=1, ensure_ascii=False))
    print(f"[{name}] images={len(items)} reference={len(ref)} false={len(false)}")
    for f in false[:max_show]:
        if "gt_l2" in f:
            print(f"  FALSE {f['id']} mode={f['mode']} exact={f['exact']}")
            print(f"    GT   {f['gt_l2']}")
            print(f"    pred {f['pred_l2']}")
        else:
            print(f"  UNLABELED {f['id']} cnn_l2_ok={f['l2_ok']['cnn']} "
                  f"pred={f['pred_l2'][:24]}...")
    return false


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", choices=("passport", "photo", "assets", "all"),
                    default="all")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    if args.corpus in ("passport", "all"):
        ref = load_ref_passport()
        items = []
        for txt in sorted(OMTEST.glob("*/*.jpg-result.txt")):
            jpg = txt.parent / txt.name.replace("-result.txt", "")
            if jpg.exists():
                items.append((jpg, txt, txt.parent.name + "/" + jpg.stem))
        run_corpus("passport", items, crop_passport, ref,
                   OUT / "context_passport")

    if args.corpus in ("photo", "all"):
        items = [(p, None, p.parent.name + "/" + p.name)
                 for p in sorted(PHOTO.glob("*/*.jpg"))]
        run_corpus("photo", items, lambda j, l: auto_crop(j), {},
                   OUT / "context_photo")

    if args.corpus in ("assets", "all"):
        ref = load_ref_assets()
        items = [(p, None, p.name) for p in sorted(ASSETS.glob("*"))
                 if p.is_file()]
        run_corpus("assets", items, lambda j, l: auto_crop(j), ref,
                   OUT / "context_assets")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
