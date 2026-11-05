#!/usr/bin/env python3
"""Field-level accuracy panel (gray_training_outline_1105.md §7.3).

The outline asks for a four-dimensional attribution panel -- field × grid
column × character class (letter / digit / '<') × whether the check digit
closed -- so that "the overall CRR improved" cannot hide a field that did
not.  This module owns the field split; the boundaries are the same ones
the C decoder uses (check_digit_of over the same 0-based ranges), so a
field error here means the same thing as a check-digit failure in C.

Field split (outline §7.3, verbatim):

  line1  [0]        document type
         [2..4]     issuing state
         [5..43]    name (including the '<' separator)
  line2  [0..8]     document number    [9]      its check digit
         [10..12]   nationality        [13..18] date of birth  [19] its cd
         [20]       sex                [21..26] date of expiry [27] its cd
         [28..41]   personal number / optional data  [42] its cd
         [43]       composite check digit

Usage:
    python3 tools/field_accuracy.py --dump <bench dump.json> [--method cnn]
    python3 tools/field_accuracy.py --dump a.json --dump b.json --method fuse
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

FIELDS_L1 = [
    ("证件类型", 0, 1),
    ("签发国", 2, 5),
    ("姓名", 5, 44),
]
FIELDS_L2 = [
    ("证件号", 0, 9), ("证件号校验位", 9, 10),
    ("国籍", 10, 13),
    ("出生日期", 13, 19), ("出生校验位", 19, 20),
    ("性别", 20, 21),
    ("有效期", 21, 27), ("有效期校验位", 27, 28),
    ("个人号/可选区", 28, 42), ("个人号校验位", 42, 43),
    ("复合校验位", 43, 44),
]


def _cls(ch: str) -> str:
    if ch == "<":
        return "filler"
    if ch.isdigit():
        return "digit"
    return "letter"


def panel(records, method: str = "cnn") -> dict:
    """records: iterable of (gt1, gt2, pred1, pred2)."""
    stats: dict[str, dict] = {}
    by_class: dict[str, Counter] = {}
    tot_err = 0
    tot_cells = 0
    for gt1, gt2, p1, p2 in records:
        for fields, gt, pred in ((FIELDS_L1, gt1, p1),
                                 (FIELDS_L2, gt2, p2)):
            if not gt or not pred:
                continue
            for name, a, b in fields:
                if len(gt) < b or len(pred) < b:
                    continue
                g_s, p_s = gt[a:b], pred[a:b]
                st = stats.setdefault(name, {"n": 0, "err": 0, "cd": 0,
                                             "cd_ok": 0})
                for g, p in zip(g_s, p_s):
                    st["n"] += 1
                    by_class.setdefault(name, Counter())[_cls(g)] += 1
                    tot_cells += 1
                    if g != p:
                        st["err"] += 1
                        tot_err += 1
                        by_class.setdefault(name + "#err", Counter())[_cls(g)] += 1
                if "校验位" in name:
                    st["cd"] += 1
                    if g_s == p_s:
                        st["cd_ok"] += 1
    return {"stats": stats, "by_class": {k: dict(v) for k, v in by_class.items()},
            "tot_err": tot_err, "tot_cells": tot_cells}


def format_panel(p: dict, title: str = "") -> str:
    out = []
    if title:
        out.append(title)
    out.append(f"{'字段':<18s} {'格数':>6s} {'错误':>6s} {'错误率':>7s} "
               f"{'字母错':>7s} {'数字错':>7s} {'填充错':>7s}")
    for name, st in p["stats"].items():
        err = st["err"]
        n = max(1, st["n"])
        ecls = p["by_class"].get(name + "#err", {})
        out.append(f"{name:<18s} {st['n']:>6d} {err:>6d} {err/n*100:6.2f}% "
                   f"{ecls.get('letter',0):>7d} {ecls.get('digit',0):>7d} "
                   f"{ecls.get('filler',0):>7d}")
    out.append(f"总计: {p['tot_err']}/{p['tot_cells']} 格错 "
               f"({p['tot_err']/max(1,p['tot_cells'])*100:.2f}%)")
    return "\n".join(out)


def _from_dump(path: Path, method: str):
    d = json.loads(path.read_text())
    recs = d["records"] if isinstance(d, dict) and "records" in d else d
    for r in recs:
        g = r.get("gt") or {}
        g1, g2 = g.get("line1", ""), g.get("line2", "")
        raw = r.get("raw") or {}
        res = raw.get(method) or {}
        if not g1:
            continue
        yield g1, g2, res.get("line1", ""), res.get("line2", "")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", action="append", required=True)
    ap.add_argument("--method", default="cnn")
    args = ap.parse_args()
    rows = []
    for p in args.dump:
        rows.extend(_from_dump(Path(p), args.method))
    p = panel(rows)
    print(format_panel(p, f"method={args.method}  records={len(rows)}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
