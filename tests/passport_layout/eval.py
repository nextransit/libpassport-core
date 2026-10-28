"""eval: parse the human-reviewed xlsx into structured failure labels.

Used by benchmark to score the current locator and any new candidate.
The xlsx has 3 sheets: 审核表 (overview + status notes), 区域明细 (per
region), 缺失清单 (missing-by-region). We extract:

  * per-image: {ok_3region, missing: [..], notes: {region: note}, status_text}
  * aggregate: counts of each failure mode, useful for the
    "quantify the pain" step in research.md §1.
"""
from __future__ import annotations
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from openpyxl import load_workbook


@dataclass
class ImageVerdict:
    fname: str
    w: int | None
    h: int | None
    missing: list[str] = field(default_factory=list)
    notes: dict[str, str] = field(default_factory=dict)
    status_text: str = ""

    def io_categories(self) -> set[str]:
        s = set()
        if self.missing:
            s.add("missing")
        for r, note in self.notes.items():
            n = (note or "").lower()
            if "第2行" in n:
                s.add("mrz_line1_swallowed")
            if "覆盖" in n:
                s.add("data_covers_photo")
            if "无效证件" in n or "无效" in n:
                s.add("no_photo")
        return s


def parse_xlsx(path):
    wb = load_workbook(path, read_only=True)
    ws = wb["审核表"]
    rows = list(ws.iter_rows(values_only=True))
    out = []
    for r in rows[6:]:
        if not r or r[0] is None:
            continue
        fname = r[1]
        w = r[4] if r[4] else None
        h = r[5] if r[5] else None
        miss = r[21] or ""
        missing = [m.strip() for m in miss.replace("、", ",").split(",") if m.strip()]
        notes = {}
        for region, idx in (("photo", 26), ("data", 27), ("mrz", 28)):
            n = r[idx]
            if n:
                notes[region] = str(n)
        out.append(ImageVerdict(
            fname=str(fname), w=int(w) if w else None, h=int(h) if h else None,
            missing=missing, notes=notes, status_text=str(miss),
        ))
    return out


def aggregate(verdicts):
    miss_counter = Counter()
    note_counter = Counter()
    cat_counter = Counter()
    for v in verdicts:
        for m in v.missing:
            miss_counter[m] += 1
        for k, n in v.notes.items():
            note_counter[f"{k}::{n}"] += 1
        cat_counter.update(v.io_categories())
    return {
        "n": len(verdicts),
        "missing": dict(miss_counter),
        "notes": dict(note_counter.most_common(20)),
        "categories": dict(cat_counter),
    }


if __name__ == "__main__":
    import json
    p = Path("assets/pic/annotations_review.xlsx")
    vs = parse_xlsx(p)
    agg = aggregate(vs)
    print(json.dumps(agg, ensure_ascii=False, indent=2))
