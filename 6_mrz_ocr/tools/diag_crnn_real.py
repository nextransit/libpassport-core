#!/usr/bin/env python3
"""Diagnose CRNN line decoding on the real-photo corpus.

Answers three questions the aggregate accuracy numbers cannot:

  1. WHERE does CRNN lose characters that the per-character CNN gets
     right?  (per-column diff map)
  2. Does the error look like a segmentation fault (a glyph window off
     by one cell, which drags the whole tail) or an isolated classifier
     mistake?  Measured as the length of consecutive-error runs.
  3. Does the sequence head fail on the same photos the segmenter
     already flags (seg score), i.e. is the gate threshold right?

Both backends run the identical C front-end; only the head differs, so
a column-level diff is attributable to the head.

Usage:
  python3 tools/diag_crnn_real.py [--corpus data/corpus_real] [--limit N]
"""
from __future__ import annotations
import argparse
import json
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "build" / "mrz_ocr_cnn_tool"


def run(img: Path, crnn: int):
    env = dict(os.environ)
    env["MRZ_OCR_LINE_MODELS"] = "1"
    env["MRZ_OCR_PAR"] = "0"
    env["MRZ_OCR_SEGDBG"] = "1"
    env["MRZ_OCR_CRNN"] = "1" if crnn else "0"
    p = subprocess.run([str(TOOL), str(img)], capture_output=True,
                       text=True, timeout=60, env=env)
    kv = {}
    for ln in p.stdout.splitlines():
        if ":" not in ln:
            continue
        k, _, v = ln.partition(":")
        kv[k.strip()] = v.strip()
    scores = [int(s) for s in re.findall(r"score=(\d+)", p.stderr)]
    return {
        "ok": kv.get("result.ok", "?"),
        "line1": kv.get("result.line1", ""),
        "line2": kv.get("result.line2", ""),
        "conf1": kv.get("result.conf1", ""),
        "conf2": kv.get("result.conf2", ""),
        "amb": kv.get("result.ambiguous2", "0"),
        "margin": kv.get("result.altmargin2", "0"),
        "seg": min(scores) if scores else -1,
    }


def colerr(a: str, b: str) -> list[int]:
    """Column indices where a and b disagree (length-normalised to 44)."""
    a = (a + "<" * 44)[:44]
    b = (b + "<" * 44)[:44]
    return [i for i in range(44) if a[i] != b[i]]


def runs_of(idxs: list[int]) -> list[int]:
    """Lengths of consecutive-index runs."""
    if not idxs:
        return []
    out, cur = [], 1
    for i in range(1, len(idxs)):
        if idxs[i] == idxs[i - 1] + 1:
            cur += 1
        else:
            out.append(cur)
            cur = 1
    out.append(cur)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/corpus_real")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    corpus = ROOT / args.corpus if not args.corpus.startswith("/") else Path(args.corpus)
    recs = json.loads((corpus / "corpus.json").read_text())["records"]
    if args.limit:
        recs = recs[: args.limit]

    rows = []
    for rec in recs:
        img = corpus / rec["image"]
        c = run(img, 0)
        r = run(img, 1)
        e1c = colerr(rec["line1"], c["line1"])
        e2c = colerr(rec["line2"], c["line2"])
        e1r = colerr(rec["line1"], r["line1"])
        e2r = colerr(rec["line2"], r["line2"])
        rows.append({"id": rec["id"], "gt1": rec["line1"], "gt2": rec["line2"],
                     "cnn": c, "crnn": r,
                     "e1c": e1c, "e2c": e2c, "e1r": e1r, "e2r": e2r})

    n = len(rows)
    print(f"corpus={args.corpus} n={n}\n")

    # ---- 1. aggregate ----
    def acc(key, line, errs):
        tot = 44 * n
        bad = sum(len(x[errs]) for x in rows)
        return 1.0 - bad / tot

    print(f"{'':6} {'l1_acc':>8} {'l2_acc':>8} {'l1_full':>8} {'l2_full':>8}")
    for name, k, e1, e2 in (("cnn", "cnn", "e1c", "e2c"), ("crnn", "crnn", "e1r", "e2r")):
        f1 = sum(1 for x in rows if not x[e1])
        f2 = sum(1 for x in rows if not x[e2])
        print(f"{name:6} {acc(k,1,e1)*100:7.2f}% {acc(k,2,e2)*100:7.2f}% "
              f"{f1:5d}/{n} {f2:5d}/{n}")

    # ---- 2. error propagation: run lengths ----
    print("\n-- consecutive-error run lengths (line2) --")
    for name, e in (("cnn", "e2c"), ("crnn", "e2r")):
        allruns = []
        for x in rows:
            allruns += runs_of(x[e])
        if not allruns:
            print(f"{name}: no errors")
            continue
        hist = Counter(allruns)
        iso = sum(v for k, v in hist.items() if k == 1)
        print(f"{name}: errors={sum(allruns)} runs={len(allruns)} "
              f"mean_run={sum(allruns)/len(allruns):.2f} "
              f"isolated={iso/len(allruns)*100:.0f}% "
              f"max_run={max(allruns)} hist={dict(sorted(hist.items()))}")

    # ---- 3. per-column error histogram (line2) ----
    print("\n-- per-column error count (line2, 44 cols) --")
    for name, e in (("cnn", "e2c"), ("crnn", "e2r")):
        cnt = Counter()
        for x in rows:
            for c in x[e]:
                cnt[c] += 1
        print(f"{name}: " + " ".join(f"{cnt.get(i,0)}" for i in range(44)))
    print("      " + " ".join(f"{i//10}" for i in range(44)))
    print("      " + " ".join(f"{i%10}" for i in range(44)))

    # ---- 4. head-to-head on line2 ----
    both = sum(1 for x in rows if x["e2c"] and x["e2r"])
    only_cnn = sum(1 for x in rows if not x["e2c"] and x["e2r"])   # CNN exact, CRNN not
    only_crnn = sum(1 for x in rows if x["e2c"] and not x["e2r"])   # CRNN exact, CNN not
    neither = sum(1 for x in rows if not x["e2c"] and not x["e2r"])
    print(f"\n-- line2 head-to-head --\n"
          f"both exact={neither}  both wrong={both}  "
          f"CNN only={only_cnn}  CRNN only={only_crnn}")

    # ---- 5. regressions: CNN right, CRNN wrong ----
    print("\n-- line2 regressions (CNN exact, CRNN wrong) --")
    bad = 0
    for x in rows:
        if x["e2c"] or not x["e2r"]:
            continue
        bad += 1
        seg = x["crnn"]["seg"]
        print(f"[{x['id']}] seg={seg} conf={x['crnn']['conf2']} amb={x['crnn']['amb']}")
        print(f"  gt  : {x['gt2']}")
        print(f"  cnn : {x['cnn']['line2']}")
        print(f"  crnn: {x['crnn']['line2']}")
        pos = x["e2r"]
        print(f"  diff cols={pos} -> " +
              " ".join(f"{c}:{x['gt2'][c]}->{x['crnn']['line2'][c]}" for c in pos))
    print(f"  total regressions = {bad}")

    # ---- 6. wins: CNN wrong, CRNN right ----
    print("\n-- line2 wins (CRNN exact, CNN wrong) --")
    w = 0
    for x in rows:
        if x["e2r"] or not x["e2c"]:
            continue
        w += 1
        print(f"[{x['id']}] seg={x['crnn']['seg']} diff cols={x['e2c']}")
    print(f"  total wins = {w}")

    # ---- 7. seg score vs line2 exactness ----
    print("\n-- seg score vs line2 exactness --")
    for lo, hi in ((0, 69), (70, 100)):
        sub = [x for x in rows if lo <= x["crnn"]["seg"] <= hi]
        if not sub:
            print(f"seg {lo}-{hi}: n=0")
            continue
        okc = sum(1 for x in sub if not x["e2c"])
        okr = sum(1 for x in sub if not x["e2r"])
        print(f"seg {lo}-{hi}: n={len(sub)} cnn_exact={okc} crnn_exact={okr}")

    # ---- 8. crop height vs outcome ----
    # corpus_real is built by build_real_corpus.isolate_mrz(), which pads
    # the two MRZ rows by 6-7 px. When the observation-band detection
    # fails, the whole lower page is kept instead (500+ px tall), and the
    # band locator then has to find the rows inside a full page. Grouping
    # by crop height separates those two very different tasks.
    from PIL import Image
    print("\n-- crop height vs outcome (line2) --")
    buckets = {"tight (<=120)": [], "mid (121-250)": [], "page (>250)": []}
    for x, rec in zip(rows, recs):
        h = Image.open(corpus / rec["image"]).size[1]
        x["h"] = h
        key = ("tight (<=120)" if h <= 120 else
               "mid (121-250)" if h <= 250 else "page (>250)")
        buckets[key].append(x)
    for key, sub in buckets.items():
        if not sub:
            print(f"{key}: n=0")
            continue
        ec = sum(1 for x in sub if not x["e2c"])
        er = sum(1 for x in sub if not x["e2r"])
        bc = sum(len(x["e2c"]) for x in sub)
        br = sum(len(x["e2r"]) for x in sub)
        print(f"{key}: n={len(sub)} cnn_exact={ec} crnn_exact={er} "
              f"cnn_err_chars={bc} crnn_err_chars={br}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
