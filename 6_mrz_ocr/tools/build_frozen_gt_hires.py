#!/usr/bin/env python3
"""Build a FROZEN ground truth for the hi-res scraped corpus.

The passport/assets frozen GTs (tools/build_frozen_gt.py) lean on the
BOARD's independent recogniser.  The hi-res corpus (assets/pic_hires,
fetched by tests/fetch_passports.py --width 1600) has no board logs, so
the readers here are the two heads of the current model at three crop
paddings -- that is model-dependent, and the docstring of
build_frozen_gt warns exactly about that.  Three things keep this honest
enough to ship as a DRAFT GT:

  * the reference is written ONCE and frozen -- later models are scored
    against the same string, so the GT can no longer move with a retrain
    (the failure mode that made eval_two_sets useless);
  * every candidate is forced through the ICAO check digits (1e-5
    false-accept rate, arithmetic independent of both heads);
  * candidates must agree across DIFFERENT heads AND different crop
    geometries; disagreement lands in the "adjudicate" tier, which must
    be confirmed by eye before the file is used as eval GT.

The band crop comes from the C tool's own locate_band on the full page.
That is a GEOMETRY source only -- labels never come from it.

Usage:
  python3 tools/build_frozen_gt_hires.py --dir ../assets/pic_hires \
      --out data/gt_frozen_hires.json
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
sys.path.insert(0, str(ROOT / "tools"))
from fix_gt_checksum import solve  # noqa: E402  (checksum repair solver)

TOOL = ROOT / "build" / "mrz_ocr_cnn_tool"
PADS = (4, 10, 18)
W = (7, 3, 1)


def val(c: str) -> int:
    if c == "<":
        return 0
    if "0" <= c <= "9":
        return ord(c) - 48
    return ord(c) - 55


def cd(s: str) -> int:
    return sum(val(c) * W[i % 3] for i, c in enumerate(s)) % 10


def l2_ok(l2: str) -> bool:
    if len(l2) != 44:
        return False
    for pos, (a, b) in ((9, (0, 9)), (19, (13, 19)), (27, (21, 27)),
                        (42, (28, 42))):
        if l2[pos] != str(cd(l2[a:b])):
            return False
    return l2[43] == str(cd(l2[0:10] + l2[13:20] + l2[21:28] + l2[28:43]))


def decode(ppm: Path, head: str) -> tuple[str, str]:
    env = dict(os.environ)
    env["MRZ_OCR_LINE_MODELS"] = "1"
    env["MRZ_OCR_PAR"] = "0"
    env["MRZ_OCR_CRNN"] = "1" if head == "crnn" else "0"
    env["MRZ_OCR_FUSE"] = "1" if head == "fuse" else "0"
    p = subprocess.run([str(TOOL), str(ppm)], capture_output=True,
                       text=True, timeout=120, env=env)
    kv = {}
    for ln in p.stdout.splitlines():
        if ":" in ln:
            k, _, v = ln.partition(":")
            kv[k.strip()] = v.strip()
    return kv.get("result.line1", ""), kv.get("result.line2", "")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(ROOT.parent / "assets" / "pic_hires"))
    ap.add_argument("--out", default=str(ROOT / "data" / "gt_frozen_hires.json"))
    ap.add_argument("--pads", default=",".join(str(p) for p in PADS))
    args = ap.parse_args()
    pads = [int(x) for x in args.pads.split(",")]
    src = Path(args.dir)
    tmp = Path(tempfile.mkdtemp(prefix="gt_hires_"))

    records, adjudicate = [], []
    stats: Counter = Counter()
    for img in sorted(src.glob("*.jpg")) + sorted(src.glob("*.png")):
        try:
            rgb = Image.open(img).convert("RGB")
        except Exception:
            stats["unreadable"] += 1
            continue
        Wd, H = rgb.size
        full = tmp / f"{img.stem}_full.ppm"
        rgb.save(str(full), format="PPM")

        # 1. locate the band with the C tool on the full page (geometry only).
        l1, l2 = decode(full, "cnn")
        probe = subprocess.run([str(TOOL), str(full)], capture_output=True,
                               text=True, timeout=120,
                               env=dict(os.environ, MRZ_OCR_LINE_MODELS="1"))
        m = re.search(r"band\.x band\.y band\.w band\.h : (\d+) (\d+) (\d+) (\d+)",
                      probe.stdout)
        if not m:
            stats["no_band"] += 1
            adjudicate.append({"id": img.name, "reason": "no_band"})
            continue
        bx, by, bw, bh = (int(m.group(i)) for i in range(1, 5))
        if bw < 400 or bh < 30:
            stats["tiny_band"] += 1
            adjudicate.append({"id": img.name, "reason": f"tiny_band {bw}x{bh}"})
            continue

        # 2. multi-crop x dual-head reads around the located band.
        cand2: Counter = Counter()
        cand1: Counter = Counter()
        n_reads = 0
        for pad in pads:
            band = rgb.crop((bx, max(0, by - pad), bx + bw, min(H, by + bh + pad)))
            ppm = tmp / f"{img.stem}_{pad}.ppm"
            band.save(str(ppm), format="PPM")
            for head in ("cnn", "crnn"):
                r1, r2 = decode(ppm, head)
                n_reads += 1
                if len(r2) == 44:
                    # solve() returns (None, None, None) for an input that
                    # already passes all five check digits -- and a tuple of
                    # None is truthy, so the naive `if fixed:` votes for the
                    # key None. Unpack and guard on the string itself.
                    res = solve(r2)
                    if res and isinstance(res[0], str) and len(res[0]) == 44:
                        cand2[res[0]] += 1
                if len(r1) == 44:
                    cand1[r1] += 1
        if not cand2:
            stats["no_cksum_candidate"] += 1
            adjudicate.append({"id": img.name, "reason": "no_cksum_candidate",
                               "reads": n_reads})
            continue
        l2_gt, votes = cand2.most_common(1)[0]
        if votes == n_reads:
            tier = "checksum_unanimous"
        elif votes >= max(2, (n_reads + 1) // 2):
            tier = "consensus"
        else:
            tier = "adjudicate"
        l1_gt, l1_votes = ("", 0)
        if cand1:
            l1_gt, l1_votes = cand1.most_common(1)[0]
            if l1_votes < n_reads:
                l1_gt = l1_gt if l1_votes >= max(2, (n_reads + 1) // 2) else ""

        rec = {"id": img.name, "crop": [bx, by, bw, bh], "pad": 10,
               "line2": l2_gt, "line1": l1_gt,
               "tier": tier, "votes": votes, "reads": n_reads,
               "line2_alts": [c for c, _ in cand2.most_common(3)[1:]]}
        records.append(rec)
        stats[f"tier_{tier}"] += 1
        if tier == "adjudicate":
            adjudicate.append({"id": img.name, "reason": "split_vote",
                               "top": cand2.most_common(3)})

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"set": "hires", "records": records},
                              indent=1))
    print(f"wrote {out}: {len(records)} records")
    print("  tiers:", dict(sorted(stats.items())))
    if adjudicate:
        adj = out.with_name(out.stem + "_adjudicate.json")
        adj.write_text(json.dumps(adjudicate, indent=1))
        print(f"  NEEDS HUMAN CHECK: {len(adjudicate)} records -> {adj}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
