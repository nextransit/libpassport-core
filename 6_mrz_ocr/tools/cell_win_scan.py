#!/usr/bin/env python3
"""Scan the C-side cell-width factor over the whole corpus.

`MRZ_OCR_CELL_WIN` is a pure width factor applied to the fitted pitch:
`cell_use = factor * best_L`.  The default (unset) is `medw + 8`, which
is the width distribution the CNN was trained on.  This script runs the
REAL C tool over every photo in the corpus for a range of factors and
reports the exact-match count per factor, so the choice is made on the
shipped path rather than on a Python re-implementation.

The tool is driven through its `--serve` mode: one process per factor,
all photos streamed over stdin.  That keeps the scan to a couple of
seconds instead of 700 process starts.

Usage:
  python3 tools/cell_win_scan.py --bands /tmp/holdout_export2
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time
from pathlib import Path

ROOT = Path("/Users/zhouyong/Desktop/work/Decard/gitlab/passport")
TOOL = ROOT / "6_mrz_ocr/build/mrz_ocr_cnn_tool"


def run_serve(binary, env, paths):
    """Stream paths through one `--serve` process; yield (stdout, stderr)."""
    p = subprocess.Popen([str(binary), "--serve"], stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         text=True, bufsize=1, env=env)
    for path in paths:
        p.stdin.write(f"{path}\n")
        p.stdin.flush()
        out, err = [], []
        while True:
            line = p.stdout.readline()
            if not line:
                break
            out.append(line)
            if line.startswith("result.done"):
                break
        # drain stderr non-blocking-ish (the tool writes little)
        p.stdin.close()
        try:
            p.wait(timeout=5)
        except Exception:
            p.kill()
        yield "".join(out), "".join(err)


def run_batch(binary, env, paths):
    """One process per photo -- keeps stderr per photo, which carries the
    GRID diagnostics we want to correlate with the outcome."""
    res = []
    for path in paths:
        r = subprocess.run([str(binary), str(path)], capture_output=True,
                           text=True, env=env, timeout=60)
        res.append((r.stdout, r.stderr))
    return res


def parse(out):
    d = {}
    for line in out.splitlines():
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        d[k.strip()] = v.strip()
    return d.get("result.line1", ""), d.get("result.line2", "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bands", default="/tmp/holdout_export2")
    ap.add_argument("--gt", default=str(ROOT / "10_ocr_research/photo_direct_bench.json"))
    ap.add_argument("--factors", default="0.85,0.90,0.92,0.95,1.00,1.05,1.10,1.15,1.20")
    ap.add_argument("--extra-env", default="", help="k=v,k=v")
    args = ap.parse_args()

    gts = {}
    for r in json.load(open(args.gt))["records"]:
        if r.get("gt"):
            gts[Path(r["path"]).name] = r["gt"]

    bands = []
    for p in sorted(Path(args.bands).glob("*.ppm")):
        gt = gts.get(p.stem + ".jpg")
        if gt and len(gt["line1"]) == 44 and len(gt["line2"]) == 44:
            bands.append((p, gt))
    print(f"photos={len(bands)}")

    extra = {}
    for kv in args.extra_env.split(","):
        if "=" in kv:
            k, v = kv.split("=", 1)
            extra[k.strip()] = v.strip()

    factors = [f for f in args.factors.split(",") if f]
    results = {}
    for f in factors:
        env = dict(os.environ)
        env.update({"MRZ_OCR_LINE_MODELS": "1", "MRZ_OCR_CRNN": "0",
                    "MRZ_OCR_FUSE": "0"})
        env.update(extra)
        if f != "none":
            env["MRZ_OCR_CELL_WIN"] = f
        else:
            env.pop("MRZ_OCR_CELL_WIN", None)
        t0 = time.perf_counter()
        outs = run_batch(TOOL, env, [p for p, _ in bands])
        dt = time.perf_counter() - t0
        n_exact = 0
        l1ok = l2ok = 0
        per = {}
        for (p, gt), (out, err) in zip(bands, outs):
            a, b = parse(out)
            e1 = a == gt["line1"]
            e2 = b == gt["line2"]
            l1ok += e1
            l2ok += e2
            if e1 and e2:
                n_exact += 1
            per[p.stem] = (e1, e2, a, b)
        results[f] = dict(exact=n_exact, l1=l1ok, l2=l2ok, per=per,
                          sec=round(dt, 1))
        print(f"  factor={f:>5}  exact={n_exact:2d}/{len(bands)}  "
              f"l1={l1ok:2d} l2={l2ok:2d}  ({dt:.1f}s)")

    json.dump({f: {k: v for k, v in r.items() if k != "per"}
               for f, r in results.items()},
              open("/tmp/cell_win_scan.json", "w"))

    # which photos flip between the default and the best factor
    best = max(results, key=lambda f: results[f]["exact"])
    print(f"\nbest factor={best} exact={results[best]['exact']}/{len(bands)}")
    base = "none" if "none" in results else None
    if base and base != best:
        print(f"\nchanges vs default ({base}):")
        for p, _ in bands:
            a = results[base]["per"][p.stem]
            b = results[best]["per"][p.stem]
            if a[:2] != b[:2]:
                print(f"  {p.stem:30s} {a[:2]} -> {b[:2]}")
    print("\nbest-factor per-photo:")
    for p, _ in bands:
        a = results[best]["per"][p.stem]
        if not (a[0] and a[1]):
            print(f"  {p.stem:30s} l1={a[0]} l2={a[1]}")


if __name__ == "__main__":
    main()
