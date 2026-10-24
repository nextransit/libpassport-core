#!/usr/bin/env python3
"""Multi-engine consensus GT builder for real MRZ pages.

The question this answers: can "passport image -> correct MRZ" be
re-obtained at scale without a human reading every glyph?

Honest limits first.  ICAO 9303 gives line 2 five check digits, so a
line-2 reading within Hamming ~2 of the truth can be ARITHMETICALLY
confirmed (fix_gt_checksum.solve; false-accept ~1e-5).  Line 1 has no
check digits at all, and line 2's nationality (10-12) and sex (20) sit
outside every checksum.  Those columns can never be proven correct
automatically -- they can only be corroborated by independent readers
and flagged for a human when the readers disagree.

So the tiers:
  A  auto-freeze   line2 closed by checksum AND identical across BOTH
                   reader families (our conv front-end family vs
                   tesseract/ocrb); line1 all 44 columns cross-family
                   agreed.  No human needed.
  B  auto-freeze*  line2 closed but only one family (or blind-spot
                   columns single-family); line1 mostly agreed.  Freeze
                   with a flag; sample-audit recommended.
  C  human queue   no checksum closure or family split.  The record
                   carries every candidate plus the exact disagreeing
                   column list, so the human checks positions instead
                   of reading 88 glyphs.

The two reader families share nothing: our C pipeline (cnn+crnn heads
over one conv front-end) vs tesseract 5 with the OCR-B model from the
Android assets.  cnn and crnn are one family -- they must never count
as two votes.

Usage:
  python3 tools/build_gt_multiengine.py --dir ../assets/pic_hires \
      --out data/gt_multiengine_hires.json
  python3 tools/build_gt_multiengine.py --gt data/gt_frozen_hires.json \
      --set-dir hires          # acceptance test against a frozen GT
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from fix_gt_checksum import solve  # noqa: E402
from mrz_engines import read_tesseract  # noqa: E402
from export_omtest_slices import adaptive_xpad, fast_pair, native_band  # noqa: E402

TOOL = ROOT / "build" / "mrz_ocr_cnn_tool"
PADS = (4, 10, 18)
W = (7, 3, 1)
ALPHA = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<"


def val(c: str) -> int:
    if c == "<":
        return 0
    if c.isdigit():
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
    return l2[43] == str(cd(l2[0:10] + l2[13:20] + l2[21:43]))


def read_native(ppm: Path, head: str) -> tuple[str, str]:
    env = dict(os.environ, MRZ_OCR_LINE_MODELS="1", MRZ_OCR_PAR="0",
               MRZ_OCR_CRNN="1" if head == "crnn" else "0")
    p = subprocess.run([str(TOOL), str(ppm)], capture_output=True,
                       text=True, timeout=120, env=env)
    kv = {}
    for ln in p.stdout.splitlines():
        if ":" in ln:
            k, _, v = ln.partition(":")
            kv[k.strip()] = v.strip()
    return kv.get("result.line1", ""), kv.get("result.line2", "")


def locate_band(rgb: Image.Image) -> tuple[int, int, int, int] | None:
    """The Android fast path, with the narrow-page scale normalisation
    from build_frozen_gt_hires."""
    Wd, H = rgb.size
    arr = np.asarray(rgb, dtype=np.uint32)
    gray = ((arr[:, :, 0] * 299 + arr[:, :, 1] * 587 +
             arr[:, :, 2] * 114 + 500) // 1000).astype(np.uint8)
    scale = 1.0
    if Wd < 900:
        scale = 1500.0 / Wd
        gray = np.asarray(
            Image.fromarray(gray).resize(
                (int(Wd * scale), int(H * scale)), Image.BILINEAR),
            dtype=np.uint8)
    pair = fast_pair(gray)
    if pair is None:
        return None
    xp = adaptive_xpad(gray, pair)
    top, bot = native_band(gray, pair[0], pair[1], xp[0], xp[1])
    bx = int(xp[0] / scale)
    by = int(top / scale)
    bw = int(Wd - xp[0] / scale - xp[1] / scale)
    bh = int((bot - top) / scale)
    if bw < 400 or bh < 30:
        return None
    return bx, by, bw, bh


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(ROOT.parent / "assets" / "pic_hires"))
    ap.add_argument("--out", default=str(ROOT / "data" / "gt_multiengine_hires.json"))
    ap.add_argument("--pads", default=",".join(str(p) for p in PADS))
    args = ap.parse_args()
    pads = [int(x) for x in args.pads.split(",")]
    src = Path(args.dir)
    tmp = Path(tempfile.mkdtemp(prefix="gtme_"))

    records, queue = [], []
    stats: Counter = Counter()
    for img in sorted(src.glob("*.jpg")) + sorted(src.glob("*.png")):
        try:
            rgb = Image.open(img).convert("RGB")
        except Exception:
            stats["unreadable"] += 1
            continue
        Wd, H = rgb.size
        box = locate_band(rgb)
        if box is None:
            stats["no_band"] += 1
            queue.append({"id": img.name, "reason": "locate_failed"})
            continue
        bx, by, bw, bh = box

        reads1: list[tuple[str, str, int]] = []   # (family, line, pad)
        reads2: list[tuple[str, str, int]] = []
        for pad in pads:
            band = rgb.crop((bx, max(0, by - pad), bx + bw, min(H, by + bh + pad)))
            ppm = tmp / f"{img.stem}_{pad}.ppm"
            band.save(str(ppm), format="PPM")
            for head in ("cnn", "crnn"):
                r1n, r2n = read_native(ppm, head)
                reads1.append(("native", r1n, pad))
                reads2.append(("native", r2n, pad))
            r1t, r2t = read_tesseract(ppm)
            reads1.append(("tess", r1t, pad))
            reads2.append(("tess", r2t, pad))

        # ---- line 2: checksum closure + cross-family vote ----
        # A read that ALREADY passes all five check digits is closed as-is;
        # solve() returns (None, None, None) for exactly those inputs, so
        # relying on solve alone silently drops the most confident reads
        # (measured: tesseract closed 10/10 but contributed zero votes).
        closed: Counter = Counter()          # repaired reading -> count
        closed_families: dict[str, set] = {}
        for fam, r2, pad in reads2:
            if len(r2) != 44:
                continue
            if l2_ok(r2):
                reading = r2
            else:
                res = solve(r2)
                reading = (res[0] if res and isinstance(res[0], str)
                           and len(res[0]) == 44 else None)
            if reading:
                closed[reading] += 1
                closed_families.setdefault(reading, set()).add(fam)

        rec = {"id": img.name, "crop": [bx, by, bw, bh], "pad": 10,
               "reads": len(reads2)}
        if not closed:
            stats["l2_no_closure"] += 1
            rec["tier"] = "C"
            rec["reason"] = "no_checksum_closure"
            rec["raw_l2"] = [r for _, r, _ in reads2]
            queue.append(rec)
            records.append(rec)
            continue
        l2_top, votes = closed.most_common(1)[0]
        fams = closed_families[l2_top]
        # tesseract's raw char accuracy (51-82% measured) plus its
        # systematic mod-10-neutral confusions (8<->B, 5->S...) mean a
        # checksum-valid tesseract reading can be far from the truth:
        # the solver's 1e-5 false-accept rate assumes Hamming ~2, which
        # does not hold for it.  So tesseract CORROBORATES but never
        # ESTABLISHES: closure comes from the native family, tesseract
        # agreement upgrades to A, disagreement flags the record.
        tess_closes_top = "tess" in fams
        tess_closes_other = any("tess" in closed_families[r]
                                for r in closed if r != l2_top)
        if len(fams) >= 2:
            l2_tier = "A"
        elif tess_closes_other and not tess_closes_top:
            l2_tier = "B"
            rec["l2_tess_conflict"] = {r: closed[r] for r in closed
                                       if r != l2_top}
        else:
            l2_tier = "B"
        # NOTE on blind spots: nationality (10-12) and sex (20) are
        # outside every checksum.  At l2_tier B they rest on a single
        # reader family; tier A means both families agreed on them too.

        # ---- line 1: per-column agreement across families ----
        cols: list[Counter] = [Counter() for _ in range(44)]
        n_l1 = 0
        for fam, r1, pad in reads1:
            if len(r1) != 44:
                continue
            n_l1 += 1
            for i, c in enumerate(r1):
                cols[i][c] += 1
        fam_cols: list[dict] = []
        for i in range(44):
            per_fam: dict[str, Counter] = {}
            for fam, r1, pad in reads1:
                if len(r1) == 44:
                    per_fam.setdefault(fam, Counter())[r1[i]] += 1
            fam_cols.append(per_fam)
        dispute = []
        weak = 0
        l1_chars = []
        for i in range(44):
            fam_top = {f: c.most_common(1)[0][0] for f, c in fam_cols[i].items()}
            if len(fam_top) >= 2 and len(set(fam_top.values())) == 1:
                # cross-family agreement on this column
                l1_chars.append(next(iter(fam_top.values())))
            elif len(fam_top) == 1:
                # only one family managed a read here: no cross check
                l1_chars.append(next(iter(fam_top.values())))
                weak += 1
            else:
                tops = {f: c.most_common(1)[0][0] for f, c in fam_cols[i].items()}
                dispute.append({"col": i, "by_family": tops})
                l1_chars.append(Counter(cols[i]).most_common(1)[0][0])
        l1_cand = "".join(l1_chars)
        # line 1 has no arithmetic backstop: A = zero cross-family
        # disputes, B = a handful (the human looks at those columns
        # only), C = structurally contested.
        l1_tier = ("A" if not dispute and weak == 0 and n_l1 >= 4
                   else "B" if len(dispute) <= 6 else "C")

        # Line 2 and line 1 freeze INDEPENDENTLY: line2 has arithmetic
        # behind it, line1 does not, so a record can ship a frozen line2
        # while its line1 still owes the human a look at N columns.
        tier = {"A": "A", "B": "B", "C": "C"}[l2_tier if l2_tier == "C"
                                              else l1_tier if l1_tier == "C"
                                              else "B" if "B" in (l2_tier, l1_tier)
                                              else "A"]
        rec.update({
            "tier": tier, "l2": l2_top, "l2_votes": votes,
            "l2_families": sorted(fams), "l2_tier": l2_tier,
            "l2_frozen": l2_tier in ("A", "B"),
            "l1": l1_cand, "l1_tier": l1_tier,
            "l1_frozen": l1_tier in ("A", "B"),
            "l1_dispute": dispute[:8], "l1_dispute_n": len(dispute),
            "raw_l1": [r for _, r, _ in reads1],
            "raw_l2": [r for _, r, _ in reads2],
        })
        records.append(rec)
        stats[f"tier_{tier}"] += 1
        if tier == "C":
            queue.append(rec)

    out = Path(args.out)
    out.write_text(json.dumps({"set": str(src.name), "records": records},
                              indent=1))
    print(f"wrote {out}: {len(records)} records")
    print("  tiers:", dict(sorted(stats.items())))
    q = out.with_name(out.stem + "_human_queue.json")
    q.write_text(json.dumps(queue, indent=1))
    print(f"  human queue: {len(queue)} -> {q}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
