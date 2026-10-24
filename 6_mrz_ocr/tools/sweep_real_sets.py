#!/usr/bin/env python3
"""End-to-end sweep of both real-photo sets with the current pipeline.

Neither set has reliable text ground truth:
  * tmp/PassPortPhoto is the same 44 machine photos as tmp/omtest, whose
    labels are the board's fixmrz hypothesis (only ~14/41 are even
    checksum-consistent, see outputs/crnn_real_decode_review_2026-10-22.md);
  * assets/pic has no text labels at all -- catalog.json only carries an
    earlier, mostly-failed OCR attempt.

So this script reports only GT-free signals that cannot be faked:
  * band located / geometry sane
  * two lines of exactly 44 chars decoded
  * ICAO line-2 checksum passes
  * CNN and CRNN agreement (disagreement means at least one is wrong)
  * seg alignment score

Usage:
  /opt/homebrew/bin/python3 tools/sweep_real_sets.py [--set passport|assets|both]
"""
from __future__ import annotations
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "build" / "mrz_ocr_cnn_tool"
sys.path.insert(0, str(ROOT / "tools"))
from cmp_crop_crnn import l2_ok  # noqa: E402


def to_ppm(src: Path, dst: Path) -> bool:
    try:
        im = Image.open(src)
        im = im.convert("RGB")
        im.save(str(dst), format="PPM")
        return True
    except Exception:
        return False


def run(img: Path, crnn: int):
    env = dict(os.environ)
    env["MRZ_OCR_LINE_MODELS"] = "1"
    env["MRZ_OCR_PAR"] = "0"
    env["MRZ_OCR_SEGDBG"] = "1"
    env["MRZ_OCR_CRNN"] = "1" if crnn else "0"
    p = subprocess.run([str(TOOL), str(img)], capture_output=True, text=True,
                       timeout=120, env=env)
    kv = {}
    for ln in p.stdout.splitlines():
        if ":" in ln:
            k, _, v = ln.partition(":")
            kv[k.strip()] = v.strip()
    sc = [int(x) for x in re.findall(r"score=(\d+)", p.stderr)]
    band = kv.get("band.x band.y band.w band.h", "")
    return {
        "ok": kv.get("result.ok", "?"),
        "l1": kv.get("result.line1", ""),
        "l2": kv.get("result.line2", ""),
        "c1": kv.get("result.conf1", "0"),
        "c2": kv.get("result.conf2", "0"),
        "seg": min(sc) if sc else -1,
        "band": band,
    }


def summarise(rows, name):
    n = len(rows)
    if not n:
        return
    ok = sum(1 for r in rows if r["ok"] == "OK")
    two = sum(1 for r in rows if len(r["c"]["l1"]) == 44 and len(r["c"]["l2"]) == 44)
    ck_c = sum(1 for r in rows if l2_ok(r["c"]["l2"]))
    ck_r = sum(1 for r in rows if l2_ok(r["r"]["l2"]))
    agree = sum(1 for r in rows if r["c"]["l2"] == r["r"]["l2"] and r["c"]["l2"])
    print(f"\n== {name}  n={n}")
    print(f"   ok={ok}  two_lines_44={two}  cksum: cnn={ck_c} crnn={ck_r}")
    print(f"   cnn==crnn line2: {agree}/{n}")
    print(f"   ok-reason: {dict(Counter(r['ok'] for r in rows))}")
    seg = [r["c"]["seg"] for r in rows if r["c"]["seg"] >= 0]
    if seg:
        print(f"   seg: min={min(seg)} med={sorted(seg)[len(seg)//2]} max={max(seg)}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="both", choices=["passport", "assets", "both"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    sets = []
    if args.set in ("passport", "both"):
        sets.append(("PassPortPhoto", ROOT.parent / "tmp" / "PassPortPhoto"))
    if args.set in ("assets", "both"):
        sets.append(("assets/pic", ROOT.parent / "assets" / "pic"))

    tmp = Path(tempfile.mkdtemp(prefix="sweep_"))
    allrows = {}
    for name, d in sets:
        files = [f for f in sorted(d.iterdir())
                 if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")]
        if args.limit:
            files = files[: args.limit]
        rows = []
        for f in files:
            ppm = tmp / (f.stem + ".ppm")
            if not to_ppm(f, ppm):
                continue
            c = run(ppm, 0)
            r = run(ppm, 1)
            rows.append({"file": f.name, "c": c, "r": r, "ok": c["ok"]})
            print(f"{f.name:36s} cnn={c['ok']:16s} seg={c['seg']:3d} "
                  f"ck_c={int(l2_ok(c['l2']))} ck_r={int(l2_ok(r['l2']))} "
                  f"band={c['band']}")
        allrows[name] = rows
        summarise(rows, name)

    if args.out:
        Path(args.out).write_text(json.dumps(allrows, indent=1, ensure_ascii=False))
        print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
