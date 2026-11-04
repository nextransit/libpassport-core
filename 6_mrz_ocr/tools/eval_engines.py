#!/usr/bin/env python3
"""Per-engine evaluation: cnn / crnn / fuse, each measured on its own.

Why a separate script: `eval_weight_sets.py` always evaluates the plain CNN
head (it forces MRZ_OCR_CRNN=0), so "the model" in its report is only ever
the CNN.  The product path ships three recognisers and the acceptance
numbers have to name the one they came from:

    cnn   -- the per-character CNN head only
    crnn  -- the CRNN sequence head only (needs a CRNN weight set)
    fuse  -- CNN + CRNN combined in log space (needs a CRNN weight set)

The pic build ships no CRNN of its own (`crnn_weights_available()` returns 0
under MRZ_WEIGHTS_SET_PIC, because the CRNN head is trained on the CNN
features of ONE weight set).  For that build the crnn/fuse rows are
degenerate -- the tool silently falls back to the CNN head -- so the report
says so explicitly instead of reporting the CNN twice under two names.

Metrics per engine, on the frozen ground truth only:
    l1 exact  -- line 1 identical to the GT
    l2 exact  -- line 2 identical to the GT
    exact     -- both lines identical
    char      -- per-character accuracy over the same rows
tier C rows (truncated / non 44-character rows) are not scored.

Usage:
  python3 6_mrz_ocr/tools/eval_engines.py \
      --gt tests/gt_assets_pic_official.json \
      --tool 6_mrz_ocr/build/mrz_ocr_cnn_tool_pic \
      --engines cnn,crnn,fuse [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import mrz_band  # noqa: E402

PAD = 10

# Environment that selects the head inside the C recogniser:
#   MRZ_OCR_CRNN=1 -> CRNN sequence head (ignored when the build has none)
#   MRZ_OCR_FUSE=1 -> CNN + CRNN fused in log space (fuse wins over CRNN)
ENGINE_ENV = {
    "cnn": {"MRZ_OCR_CRNN": "0", "MRZ_OCR_FUSE": "0"},
    "crnn": {"MRZ_OCR_CRNN": "1", "MRZ_OCR_FUSE": "0"},
    "fuse": {"MRZ_OCR_CRNN": "0", "MRZ_OCR_FUSE": "1"},
}


def tier_ok(rec: dict, line: int) -> bool:
    t = rec.get("l1_tier" if line == 1 else "l2_tier") or rec.get("tier") or ""
    return t in ("A", "B", "")


def run_engine(tool: Path, rgb, cands, tmp: Path, name: str,
               env_extra: dict) -> dict:
    scored = []
    for i, (_kind, rect) in enumerate(cands):
        ppm = tmp / (Path(name).stem + f"_c{i}.ppm")
        rgb.crop(rect).save(str(ppm), format="PPM")
        scored.append(mrz_band.run_tool(tool, ppm, env_extra))
    pick = mrz_band.pick_candidate(scored, [k for k, _ in cands])
    out = dict(scored[pick])
    out["band"] = cands[pick][0]
    out["rect"] = list(cands[pick][1])
    return out


def metrics(rows: list) -> dict:
    sel = [r for r in rows if r.get("located")]
    l1_sel = [r for r in sel if r.get("l1_ok")]
    l2_sel = [r for r in sel if r.get("l2_ok")]

    def char_acc(sub, pk, gk):
        ok = tot = 0
        for r in sub:
            g = r.get(gk) or ""
            p = r.get(pk) or ""
            if len(g) != 44:
                continue
            tot += 44
            if len(p) != 44:
                continue
            ok += sum(1 for a, b in zip(g, p) if a == b)
        return ok, tot

    def exact(sub, pk, gk):
        return sum(1 for r in sub
                   if (r.get(gk) or "") == (r.get(pk) or "")
                   and len(r.get(gk) or "") == 44)

    both = [r for r in sel
            if r.get("l1_ok") and r.get("l2_ok")
            and len(r.get("gt1") or "") == 44 and len(r.get("gt2") or "") == 44]
    o1, t1 = char_acc(l1_sel, "line1", "gt1")
    o2, t2 = char_acc(l2_sel, "line2", "gt2")
    layouts = {}
    for r in sel:
        layouts[r.get("layout", "?")] = layouts.get(r.get("layout", "?"), 0) + 1
    return {
        "records": len(rows),
        "located": len(sel),
        "decode_ok": sum(1 for r in sel if r.get("ok")),
        "l1_rows": len(l1_sel),
        "l1_exact": exact(l1_sel, "line1", "gt1"),
        "l1_char_ok": o1,
        "l1_char_tot": t1,
        "l2_rows": len(l2_sel),
        "l2_exact": exact(l2_sel, "line2", "gt2"),
        "l2_char_ok": o2,
        "l2_char_tot": t2,
        "both_rows": len(both),
        "both_exact": sum(1 for r in both
                          if r["line1"] == r["gt1"] and r["line2"] == r["gt2"]),
        "layouts": layouts,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--tool", required=True)
    ap.add_argument("--engines", default="cnn,crnn,fuse")
    ap.add_argument("--pad-map", default="tests/photo_pad_map.json")
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    pad_map = {}
    pm_path = ROOT / args.pad_map
    if args.pad_map and pm_path.exists():
        pad_map = {k: int(v) for k, v in json.loads(pm_path.read_text()).items()}

    tool = Path(args.tool).resolve()
    if not tool.exists():
        print(f"tool not found: {tool}", file=sys.stderr)
        return 2

    d = json.loads((ROOT / args.gt).read_text())
    gt = d["gt"] if isinstance(d, dict) and "gt" in d else d
    tmp = Path(tempfile.mkdtemp(prefix="eveng_"))

    engines = [e.strip() for e in args.engines.split(",") if e.strip()]
    per_engine = {}
    for eng in engines:
        env_extra = dict(ENGINE_ENV.get(eng, ENGINE_ENV["cnn"]))
        env_extra["MRZ_OCR_PAR"] = "1"
        rows = []
        for name, rec in sorted(gt.items()):
            label = rec.get("label") or ""
            l1, l2 = rec.get("l1") or "", rec.get("l2") or ""
            img_path = ROOT / label
            if not img_path.exists():
                continue
            try:
                rgb = Image.open(img_path).convert("RGB")
            except Exception:
                continue
            cands = mrz_band.band_candidates(
                rgb, PAD, pad_map.get(name, pad_map.get(label)))
            if not cands:
                rows.append({"id": name, "located": False})
                continue
            r = run_engine(tool, rgb, cands, tmp, name, env_extra)
            r.update(id=name, located=True, gt1=l1, gt2=l2,
                     l1_ok=tier_ok(rec, 1), l2_ok=tier_ok(rec, 2))
            rows.append(r)
        per_engine[eng] = {"rows": rows, "metrics": metrics(rows)}

    # A build without a CRNN head silently runs the CNN instead; detect it
    # so the report does not present the same numbers twice.
    notes = []
    ref = per_engine.get("cnn")
    for eng in engines:
        if eng == "cnn" or ref is None:
            continue
        same = all(
            (a.get("line1"), a.get("line2")) == (b.get("line1"), b.get("line2"))
            for a, b in zip(ref["rows"], per_engine[eng]["rows"]))
        if same:
            notes.append(
                f"{eng}: 该构建不含 CRNN 权重（MRZ_WEIGHTS_SET_PIC），"
                f"实际退化为 cnn，下面的数值与 cnn 完全相同")

    print(f"tool = {tool.name}   gt = {args.gt}")
    print()
    hdr = (f"{'engine':7s} {'l1 exact':>12s} {'l1 char':>12s} "
           f"{'l2 exact':>12s} {'l2 char':>12s} {'exact(both)':>13s} "
           f"{'decode_ok':>10s}")
    print(hdr)
    print("-" * len(hdr))
    for eng in engines:
        m = per_engine[eng]["metrics"]
        l1c = (m["l1_char_ok"] / m["l1_char_tot"]) if m["l1_char_tot"] else 0.0
        l2c = (m["l2_char_ok"] / m["l2_char_tot"]) if m["l2_char_tot"] else 0.0
        print(f"{eng:7s} {m['l1_exact']:>5d}/{m['l1_rows']:<6d} "
              f"{l1c:>6.2%} ({m['l1_char_ok']}/{m['l1_char_tot']}) "
              f"{m['l2_exact']:>5d}/{m['l2_rows']:<6d} "
              f"{l2c:>6.2%} ({m['l2_char_ok']}/{m['l2_char_tot']}) "
              f"{m['both_exact']:>6d}/{m['both_rows']:<6d} "
              f"{m['decode_ok']:>6d}/{m['located']:<4d}")
    print()
    for eng in engines:
        m = per_engine[eng]["metrics"]
        print(f"{eng:7s} records {m['records']}  located {m['located']}  "
              f"layout {m['layouts']}")
    for n in notes:
        print("NOTE " + n)

    if args.json:
        out = {eng: per_engine[eng]["metrics"] for eng in engines}
        Path(args.json).write_text(json.dumps(
            {"tool": str(tool), "gt": args.gt, "metrics": out,
             "notes": notes,
             "rows": {eng: per_engine[eng]["rows"] for eng in engines}},
            ensure_ascii=False, indent=1))
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
