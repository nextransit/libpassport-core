#!/usr/bin/env python3
"""Collect directed (gt -> predicted) character confusions from eval dumps.

Input: one or more eval dump JSONs whose records carry gt1/gt2 and the
engine predictions (cnn_l1/cnn_l2, crnn_l1/crn2_l2 ...), i.e. the files
written by eval_all_engines.py / bench dumps.

Output: data/confusions/<name>.json with
  { source, records, pairs: {"cnn": {"<gt><got>": count, ...},
                             "crnn": {...}},
    per_class: {"cnn": {"<gt>": {"n": total, "err": errors}}} }

The single source of truth for tools/gen_confusion_tables.py -- the
10-25 review showed the curated confusion tables had drifted from what
real photos actually confuse (corpus confusions != real confusions), so
the tables must be MEASURED, not guessed.
"""

from __future__ import annotations
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "10_ocr_research" / "confusions"


def diff_pairs(gt: str, got: str) -> list[tuple[str, str]]:
    out = []
    if len(gt) != len(got):
        return out
    for g, p in zip(gt, got):
        if g != p and (g.isalnum() or g == '<') and (p.isalnum() or p == '<'):
            out.append((g, p))
    return out


def collect(paths: list[Path]) -> dict:
    pairs = defaultdict(Counter)          # engine -> (gt,got) -> n
    per_class = defaultdict(lambda: defaultdict(lambda: {"n": 0, "err": 0}))
    records = 0
    sources = []
    for path in paths:
        d = json.loads(Path(path).read_text())
        recs = d["records"] if isinstance(d, dict) and "records" in d else d
        sources.append(Path(path).name)
        for rec in recs:
            if not rec.get("located", True):
                continue
            records += 1
            for li, (gt_key, pred_prefix) in enumerate(
                    (("gt1", "cnn"), ("gt2", "cnn"))):
                gt = rec.get(gt_key, "")
                pred = rec.get(f"{pred_prefix}_l{li+1}", "")
                for g, p in diff_pairs(gt, pred):
                    pairs["cnn"][f"{g}{p}"] += 1
            # per-class accuracy for cnn
            for gt_key, pred_key in (("gt1", "cnn_l1"), ("gt2", "cnn_l2")):
                gt, pred = rec.get(gt_key, ""), rec.get(pred_key, "")
                if len(gt) != len(pred) or not gt:
                    continue
                for g, p in zip(gt, pred):
                    st = per_class["cnn"][g]
                    st["n"] += 1
                    if g != p:
                        st["err"] += 1
            for gt_key, pred_key in (("gt1", "crnn_l1"), ("gt2", "crnn_l2")):
                gt, pred = rec.get(gt_key, ""), rec.get(pred_key, "")
                if len(gt) != len(pred) or not gt:
                    continue
                for g, p in zip(gt, pred):
                    st = per_class["crnn"][g]
                    st["n"] += 1
                    if g != p:
                        st["err"] += 1
                for g, p in diff_pairs(gt, pred):
                    pairs["crnn"][f"{g}{p}"] += 1
    return {
        "sources": sources,
        "records": records,
        "pairs": {e: dict(c) for e, c in pairs.items()},
        "per_class": {e: {k: dict(v) for k, v in m.items()}
                      for e, m in per_class.items()},
    }


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", action="append", required=True,
                    help="eval dump JSON (repeatable)")
    ap.add_argument("--out", required=True, help="output stats JSON")
    args = ap.parse_args()
    stats = collect([Path(p) for p in args.dump])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(stats, ensure_ascii=False, indent=1))
    print(f"records={stats['records']} -> {out}")
    for engine in ("cnn", "crnn"):
        top = sorted(stats["pairs"].get(engine, {}).items(),
                     key=lambda kv: -kv[1])[:15]
        print(f"[{engine}] top directed confusions:")
        for k, n in top:
            print(f"    {k[0]}->{k[1]}  {n}")
        hard = sorted(((k, v["err"] / v["n"], v["n"])
                       for k, v in stats["per_class"].get(engine, {}).items()
                       if v["n"] >= 20 and v["err"]),
                      key=lambda kv: -kv[1])[:10]
        if hard:
            print(f"[{engine}] hardest GT classes (n>=20):")
            for k, rate, n in hard:
                print(f"    {k}  err {rate*100:.1f}%  (n={n})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
