#!/usr/bin/env python3
"""MRZ decoder regression bench on corpus_eval + real photos.

Scoring per record (fixed-width ICAO lines, positional alignment):
    fmt   : both lines 44 chars, line1 doc-type shape, line2 alnum shape
    L1/L2 : per-character accuracy vs GT
    chk4  : all four local Mod-10 check digits of the decoded line2 pass
    chk5  : chk4 + composite check digit (pos 43) passes
    exact : decoded line1 AND line2 equal GT

Usage (v6 needs paddlex, use the paddle venv):
    /tmp/paddle_venv/bin/python tests/_phase3_bench.py --n 200
    /tmp/paddle_venv/bin/python tests/_phase3_bench.py --n 0      # all 1380
    python3 tests/_phase3_bench.py --n 100 --backend local        # tesseract
    /tmp/paddle_venv/bin/python tests/_phase3_bench.py --real     # assets/pic
"""
from __future__ import annotations
import argparse
import contextlib
import io
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "6_mrz_ocr" / "data" / "corpus_eval"
sys.path.insert(0, str(ROOT / "6_1_PaddleOCR"))

LINE_LEN = 44

# Composite check input: passport(0..9) + dob(13..19) + expiry(21..27)
# + personal(28..42) = 39 chars; cd at pos 43.
_COMPOSITE = ((0, 10), (13, 20), (21, 28), (28, 43))
_LOCAL_CHECKS = ((0, 9, 9), (13, 19, 19), (21, 27, 27), (28, 42, 42))


def _checks(line2):
    from mrz_checksum import verify_mod10
    if len(line2) != LINE_LEN:
        return 0, False
    n_ok = sum(verify_mod10(line2[a:b], line2[c]) for a, b, c in _LOCAL_CHECKS)
    comp = "".join(line2[a:b] for a, b in _COMPOSITE)
    return n_ok, verify_mod10(comp, line2[43])


def _fmt_ok(l1, l2):
    def real(s):
        return sum(1 for c in s if c != "<")
    return (len(l1) == LINE_LEN and len(l2) == LINE_LEN
            and l1[0] in "PIVACDR" and l1[1] == "<"
            and real(l1) >= 5 and real(l2) >= 5)


def stratified(recs, n):
    if not n or n >= len(recs):
        return recs
    groups = {}
    for r in recs:
        groups.setdefault(r.get("profile", "?"), []).append(r)
    out = []
    n_groups = len(groups)
    per = max(1, n // n_groups)
    for g in groups.values():
        step = max(1, len(g) // per)
        out.extend(g[::step][:per])
    return out[:n] if n else out


def load_records(n):
    recs = json.load(open(DATA / "corpus.json"))["records"]
    return stratified(recs, n)


def load_real():
    gt = json.load(open(ROOT / "tests" / "real_gt.json"))["gt"]
    base = ROOT / "assets" / "pic"
    out = []
    for name, v in gt.items():
        p = base / name
        if p.exists():
            out.append({"id": name, "image": str(p), "line1": v["line1"],
                        "line2": v["line2"], "abs": True})
    return out


def run_v6(recs, *_, **__):
    import paddle_v6
    rec, mode = paddle_v6._get_rec()
    if mode != "v6":
        raise SystemExit("paddlex unavailable: run with /tmp/paddle_venv/bin/python")
    out = []
    for r in recs:
        p = Path(r["image"]) if r.get("abs") else DATA / r["image"]
        ok, l1, l2, c1, c2, band, err, ms = paddle_v6._decode_one(rec, p)
        out.append((l1, l2, band, ms, err))
    return out


def run_local(recs, *_, **__):
    import paddle_local
    out = []
    for r in recs:
        p = Path(r["image"]) if r.get("abs") else DATA / r["image"]
        buf = io.StringIO()
        t0 = time.perf_counter()
        with contextlib.redirect_stdout(buf):
            paddle_local.main([sys.argv[0], str(p)])
        ms = (time.perf_counter() - t0) * 1000.0
        d = {}
        for ln in buf.getvalue().splitlines():
            if ": " in ln:
                k, _, v = ln.partition(": ")
                d[k.strip()] = v.strip()
        out.append((d.get("result.line1", ""), d.get("result.line2", ""),
                    d.get("band.x band.y band.w band.h", ""), ms, ""))
    return out


BACKENDS = {"v6": run_v6, "local": run_local}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200, help="sample size (0 = all)")
    ap.add_argument("--backend", default="v6", choices=sorted(BACKENDS))
    ap.add_argument("--real", action="store_true", help="use tests/real_gt.json")
    ap.add_argument("--json", default="", help="dump per-record metrics")
    args = ap.parse_args()

    recs = load_real() if args.real else load_records(args.n)
    if not recs:
        print("no records", file=sys.stderr)
        return 2

    results = BACKENDS[args.backend](recs)
    totals = dict(fmt=0, c1=0, c2=0, ex1=0, ex2=0, chk4=0, chk5=0)
    rows = []
    for r, (l1, l2, band, ms, err) in zip(recs, results):
        gt1, gt2 = r["line1"], r["line2"]
        fmt = _fmt_ok(l1, l2)
        cc1 = sum(a == b for a, b in zip(l1, gt1))
        cc2 = sum(a == b for a, b in zip(l2, gt2))
        n4, c5 = _checks(l2)
        totals["fmt"] += fmt
        totals["c1"] += cc1
        totals["c2"] += cc2
        totals["ex1"] += (l1 == gt1)
        totals["ex2"] += (l2 == gt2)
        totals["chk4"] += (n4 == 4)
        totals["chk5"] += (n4 == 4 and c5)
        rows.append(dict(id=r["id"], fmt=bool(fmt), l1=l1, l2=l2, cc1=cc1,
                         cc2=cc2, ex1=l1 == gt1, ex2=l2 == gt2, chk4=n4,
                         chk5=bool(c5), ms=round(ms, 1), err=err))

    n = len(recs)
    dt = sum(r["ms"] for r in rows) / 1000.0
    print(f"backend={args.backend}  n={n}  ({dt:.1f}s, {dt / n * 1000:.0f} ms/img)")
    print(f"  fmt  : {totals['fmt']:4d}/{n}")
    print(f"  L1   : {totals['c1'] / (n * LINE_LEN) * 100:5.2f}%  exact {totals['ex1']:4d}/{n}")
    print(f"  L2   : {totals['c2'] / (n * LINE_LEN) * 100:5.2f}%  exact {totals['ex2']:4d}/{n}")
    print(f"  chk4 : {totals['chk4']:4d}/{n} ({totals['chk4'] / n * 100:.1f}%)  "
          f"chk5: {totals['chk5']:4d}/{n} ({totals['chk5'] / n * 100:.1f}%)")
    if args.json:
        json.dump(rows, open(args.json, "w"), indent=1)
        print(f"  -> {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
