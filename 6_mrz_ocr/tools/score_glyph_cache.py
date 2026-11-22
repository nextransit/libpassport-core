#!/usr/bin/env python3
"""Score a CNN weight header on the cached product-path glyphs.

Reads the npz written by `tools/cache_glyphs.py`, runs the exact numpy
forward pass of `tools/cnn_offline_eval.py`, and reports:

  * exact      -- both MRZ lines of a photo match the GT character for
                  character (the user metric, pre post-processing)
  * line1/line2 exact
  * char accuracy
  * the confusions that cost the most

Two weight headers can be given at once, which is what an A/B needs:
the second one is scored against the same cache and printed next to the
first, so the delta is read off directly.

The score is a LOWER BOUND on the end-to-end result: it is the raw
classifier opinion, without the ICAO syntax mask or the checksum-constrained
decoding the C pipeline applies afterwards.  A candidate must still be
confirmed with `tools/band_sweep.py` and then the authoritative bench.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "ce", str(ROOT / "tools" / "cnn_offline_eval.py"))
ce = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ce)

GLYPHS = ce.GLYPHS
GI = {c: i for i, c in enumerate(GLYPHS)}


def load_cache(path):
    d = np.load(path, allow_pickle=True)
    X = d["X"].astype(np.float32)
    meta = [json.loads(s) for s in d["meta"]]
    Y = np.array([GI.get(m.get("ch", ""), -1) for m in meta], dtype=np.int64)
    return X, Y, meta


def score(X, Y, meta, P, line):
    """Return (exact, l1_exact, l2_exact, acc, per-file rows)."""
    keep = [i for i, m in enumerate(meta) if int(m["line"]) == line]
    if not keep:
        return None
    logits = ce.forward(X[keep], P)
    pred = logits.argmax(1)
    prob = ce.softmax(logits)
    # group by file
    byfile = {}
    for j, i in enumerate(keep):
        f = meta[i]["file"]
        byfile.setdefault(f, []).append((meta[i]["col"], pred[j], prob[j], i))
    exact = l1 = l2 = 0
    total = correct = 0
    rows = []
    for f, cells in byfile.items():
        cells.sort()
        s = ""
        ok = True
        for (c, p, pr, i) in cells:
            s += GLYPHS[p]
            if Y[i] >= 0:
                total += 1
                if GLYPHS[p] == GLYPHS[Y[i]]:
                    correct += 1
                else:
                    ok = False
        rows.append((f, s, ok, cells))
        if line == 0:
            l1 += 1 if ok else 0
        else:
            l2 += 1 if ok else 0
    return {"l1": l1, "l2": l2, "acc": correct / max(1, total),
            "rows": rows, "total": total, "correct": correct}


def report(tag, res, gt):
    if res is None:
        print(f"  {tag}: no data")
        return
    print(f"  {tag}: line1_exact={res['l1']} line2_exact={res['l2']} "
          f"char_acc={res['acc']*100:.2f}%")
    conf = Counter()
    for f, s, ok, cells in res["rows"]:
        g = gt.get(f)
        if not g:
            continue
        truth = g[res["line"]]
        for j, (c, p, pr, i) in enumerate(cells):
            if j < len(truth) and GLYPHS[p] != truth[j]:
                conf[(truth[j], GLYPHS[p])] += 1
    if conf:
        print(f"    top confusions (gt->pred): "
              + ", ".join(f"{a}->{b}:{n}" for (a, b), n in
                          conf.most_common(12)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/tmp/glyph_cache.npz")
    ap.add_argument("--weights", nargs="+", required=True,
                    help="weight headers, e.g. src/cnn_weights_line1.h")
    ap.add_argument("--prefix", nargs="+", default=None,
                    help="symbol prefix per header (default: infer)")
    ap.add_argument("--line", type=int, default=None,
                    help="0 or 1; default: run both")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    X, Y, meta = load_cache(args.cache)
    gts = {}
    for f in (ROOT.parent / "tests" / "gt_photo_passport.json",
              ROOT.parent / "tests" / "photo_auto_gt.json"):
        if not Path(f).exists():
            continue
        d = json.loads(Path(f).read_text())
        g = d.get("gt", d) if isinstance(d, dict) else d
        for k, v in g.items():
            if not isinstance(v, dict):
                continue
            l1 = v.get("l1") or v.get("line1")
            l2 = v.get("l2") or v.get("line2")
            if l1 and l2 and len(l1) == 44 and len(l2) == 44:
                gts[Path(k).stem] = (l1, l2)

    lines = [args.line] if args.line is not None else [0, 1]
    for wi, wpath in enumerate(args.weights):
        arrs = ce.parse_header(wpath)
        prefix = (args.prefix[wi] if args.prefix and wi < len(args.prefix)
                  else None)
        P = ce.pick(arrs, prefix or ce.infer_prefix(arrs))
        print(f"{Path(wpath).name}  (prefix={prefix or ce.infer_prefix(arrs)})")
        for li in lines:
            res = score(X, Y, meta, P, li)
            if res is None:
                continue
            res["line"] = li
            report(f"line{li+1}", res, gts)
            if args.verbose:
                for f, s, ok, cells in sorted(res["rows"]):
                    if not ok:
                        print(f"      {f}: {s}")


if __name__ == "__main__":
    main()
