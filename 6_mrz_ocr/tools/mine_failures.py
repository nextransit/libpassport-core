"""Failure miner for the MRZ OCR fix loop.

Reads failures_cnn.json / failures_traditional.json (written by
bench.py) and clusters the failures so the next iteration is driven by
evidence instead of guesswork:

  * by rendering condition (scale / noise / rot / profile buckets)
  * by MRZ column (which slot of the 44 fails)
  * by confusion pair (gt -> got)

Prints a report and, with --suggest, concrete next-loop actions
(corpus buckets to expand, hard pairs to weight).
"""
from __future__ import annotations
import argparse, json, sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def bucket(rec):
    rot = abs(float(rec.get("rot", 0.0) or 0.0))
    rot_b = "rot0" if rot < 0.5 else ("rot<2" if rot < 2.0 else "rot>=2")
    noise_b = ("n0" if rec["noise"] == 0 else
               "n<0.03" if rec["noise"] < 0.03 else "n>=0.03")
    return f"{rec['profile']}/s{rec['scale']}/{noise_b}/{rot_b}"


def load(method):
    p = ROOT / f"failures_{method}.json"
    if not p.exists():
        return []
    return json.loads(p.read_text())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="cnn")
    ap.add_argument("--suggest", action="store_true",
                    help="append concrete next-loop suggestions")
    a = ap.parse_args()

    fails = load(a.method)
    if not fails:
        print(f"no failures recorded for {a.method} -- nothing to mine")
        return 0

    print(f"== {a.method}: {len(fails)} failing images ==")

    by_bucket = Counter(bucket(r) for r in fails)
    print("\n-- by rendering condition --")
    for b, n in by_bucket.most_common():
        print(f"  {b:32s} {n}")

    by_col = Counter()
    for r in fails:
        for i in r.get("l1_err", []): by_col[f"L1:{i}"] += 1
        for i in r.get("l2_err", []): by_col[f"L2:{i}"] += 1
    if by_col:
        print("\n-- by column (top 15) --")
        for c, n in by_col.most_common(15):
            print(f"  {c:8s} {n}")

    pairs = Counter()
    for r in fails:
        for i in r.get("l1_err", []):
            g = r["gt1"][i] if i < len(r["gt1"]) else "_"
            h = r["line1"][i] if i < len(r["line1"]) else "_"
            pairs[f"{g}->{h}"] += 1
        for i in r.get("l2_err", []):
            g = r["gt2"][i] if i < len(r["gt2"]) else "_"
            h = r["line2"][i] if i < len(r["line2"]) else "_"
            pairs[f"{g}->{h}"] += 1
    if pairs:
        print("\n-- top confusion pairs --")
        for p, n in pairs.most_common(15):
            print(f"  {p:8s} {n}")

    n_line_fail = sum(1 for r in fails if not r.get("ok", True))
    print(f"\n-- pipeline failures (no output): {n_line_fail}")

    if a.suggest:
        print("\n== suggested next-loop actions ==")
        for b, n in by_bucket.most_common(3):
            if n >= 3:
                print(f"  * expand corpus bucket '{b}' ({n} failures): "
                      f"add variants with these params to the TRAIN corpus")
        for p, n in pairs.most_common(5):
            if n >= 3:
                g, h = p.split("->")
                print(f"  * confusion {p} x{n}: ensure both glyphs are in "
                      f"HARD_PAIRS with extra samples; consider TTA")
        if n_line_fail:
            print(f"  * {n_line_fail} pipeline failures: inspect "
                  f"segmentation (MRZ_OCR_DUMP=1) before touching weights")
    return 0


if __name__ == "__main__":
    sys.exit(main())
