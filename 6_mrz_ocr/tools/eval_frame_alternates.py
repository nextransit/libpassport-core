#!/usr/bin/env python3
"""Measure the band-framing ALTERNATES that the Android product path now tries.

The Kotlin product path (PhotoMrzPipeline.processFast) re-frames the SAME
restore pair with a scaled vertical pad (0.70 = shipped, plus the 1.35 /
0.45 alternates) and keeps the first candidate whose raw line2 passes
decode + shape + the five ICAO check digits.  This script replicates that
exact loop on the host so the design can be gated before it ships.

Unlike tools/eval_weight_sets.py (which selects among the full
mrz_band candidate zoo with a line2 check-digit preference), this measures
ONLY the pad alternates of one framing family -- i.e. exactly what the
device can do without new native code.

Usage:
  python3 tools/eval_frame_alternates.py --gt ../tests/gt_assets_pic_official.json \
      --tool build/mrz_ocr_cnn_tool [--alts 0.70,1.35,0.45]
"""

from __future__ import annotations
import argparse, json, os, subprocess, sys, tempfile
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REPO = ROOT.parent
sys.path.insert(0, str(HERE))
import mrz_band  # noqa: E402

W7 = [7, 3, 1]


def cd(seg: str) -> int:
    t = 0
    for i, c in enumerate(seg):
        v = (ord(c) - 48) if c.isdigit() else ((ord(c) - 55) if 'A' <= c <= 'Z' else 0)
        t += v * W7[i % 3]
    return t % 10


def line2_ok(l2: str) -> bool:
    if len(l2) != 44:
        return False
    try:
        return (cd(l2[0:9]) == int(l2[9]) and cd(l2[13:19]) == int(l2[19])
                and cd(l2[21:27]) == int(l2[27]) and cd(l2[28:42]) == int(l2[42])
                and cd(l2[0:10] + l2[13:20] + l2[21:43]) == int(l2[43]))
    except (ValueError, IndexError):
        return False


def run_tool(tool: Path, ppm: Path) -> tuple[str, str]:
    env = dict(os.environ)
    env["MRZ_OCR_PAR"] = "1"
    env["MRZ_OCR_LINE_MODELS"] = "1"
    p = subprocess.run([str(tool), str(ppm)], capture_output=True, text=True,
                       timeout=120, env=env)
    l1 = l2 = ""
    for line in p.stdout.splitlines():
        if line.startswith("result.line1"):
            l1 = line.split(":", 1)[1].strip()
        elif line.startswith("result.line2"):
            l2 = line.split(":", 1)[1].strip()
    return l1, l2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", default=str(REPO / "tests" / "gt_assets_pic_official.json"))
    ap.add_argument("--tool", default=str(ROOT / "build" / "mrz_ocr_cnn_tool"))
    ap.add_argument("--alts", default="0.70,1.35,0.45")
    args = ap.parse_args()

    tool = Path(args.tool).resolve()
    alts = [float(x) for x in args.alts.split(",")]
    gt_path = Path(args.gt)
    gt = json.loads(gt_path.read_text())
    gt = gt["gt"] if isinstance(gt, dict) and "gt" in gt else gt
    tmp = Path(tempfile.mkdtemp(prefix="framealt_"))

    stats = {"n": 0, "primary_ok": 0, "any_ok": 0, "exact_primary": 0,
             "exact_any": 0, "no_restore": 0, "rescued": [], "regressed": []}
    for name, rec in sorted(gt.items()):
        label = rec.get("label") or ""
        g1, g2 = rec.get("l1") or "", rec.get("l2") or ""
        img_path = REPO / label
        if not img_path.exists() or not g2:
            continue
        rgb = Image.open(img_path).convert("RGB")
        rows = mrz_band.restore_pair_rows(rgb)
        if not rows:
            stats["no_restore"] += 1
            continue
        y0a, _y1a, _y0b, y1b = rows
        W, H = rgb.size
        row_h = max(1, (y1b - y0a))
        # Kotlin nativeBandFast: sharedPad = clamp(8*k, 25*k, rowH*k)
        results = []
        for k in alts:
            pad = max(int(8 * k), min(int(25 * k), int(row_h * k)))
            rect = (0, max(0, y0a - pad), W, min(H, y1b + pad))
            ppm = tmp / f"{Path(name).stem}_p{int(k*100)}.ppm"
            rgb.crop(rect).save(str(ppm), format="PPM")
            l1, l2 = run_tool(tool, ppm)
            results.append((k, l1, l2))
        stats["n"] += 1
        p_ok = line2_ok(results[0][2])
        a_ok = any(line2_ok(r[2]) for r in results)
        stats["primary_ok"] += 1 if p_ok else 0
        stats["any_ok"] += 1 if a_ok else 0
        p_exact = results[0][2] == g2
        a_exact = any(r[2] == g2 for r in results)
        stats["exact_primary"] += 1 if p_exact else 0
        stats["exact_any"] += 1 if a_exact else 0
        if a_exact and not p_exact:
            stats["rescued"].append(name)
        if p_exact and not a_exact:
            stats["regressed"].append(name)

    n = max(stats["n"], 1)
    print(f"photos with a restore pair: {stats['n']} (no restore: {stats['no_restore']})")
    print(f"  line2 ICAO-valid   primary {stats['primary_ok']}/{n}   "
          f"any-alternate {stats['any_ok']}/{n}  (+{stats['any_ok']-stats['primary_ok']})")
    print(f"  line2 EXACT        primary {stats['exact_primary']}/{n}   "
          f"any-alternate {stats['exact_any']}/{n}  (+{stats['exact_any']-stats['exact_primary']})")
    if stats["rescued"]:
        print(f"  rescued: {stats['rescued']}")
    if stats["regressed"]:
        print(f"  REGRESSED (primary was right, alternates lost it): {stats['regressed']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
