#!/usr/bin/env python3
"""Split tests/photo_auto_gt.json into one GT file per training corpus.

The repository holds two corpora that must NOT be mixed when training:

  * tmp/PassPortPhoto  -- 44 phone photos of real passports, named
                          2026-9-29-*.jpg / 2026-9-30-*.jpg.  Their labels
                          come from the board app's -result.txt files, which
                          is what tools/export_omtest_slices.py consumes.
  * assets/pic         -- 65 scraped low-resolution passport images with
                          hash-like names.  No board labels exist for these,
                          so the only label source is photo_auto_gt.json.

Mixing them produces a model that is mediocre on both: the two sets differ in
resolution (PassPortPhoto is >=1500 px wide, assets/pic is often 600 px),
in font rendering, and in how much of the page is in frame.  This script
writes the split so each corpus can be exported and trained on its own.

Outputs (schema is preserved verbatim, only the record set is filtered):
  tests/gt_photo_passport.json
  tests/gt_assets_pic.json

Usage:
  python3 tools/split_gt_sets.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# prefix -> output file.  The prefix is matched against each record's `label`
# field, which is a repository-relative image path.
SETS = {
    "tmp/PassPortPhoto": "tests/gt_photo_passport.json",
    "assets/pic": "tests/gt_assets_pic.json",
}


def main() -> int:
    src = ROOT / "tests/photo_auto_gt.json"
    d = json.loads(src.read_text())
    gt = d["gt"] if isinstance(d, dict) and "gt" in d else d

    for prefix, out_rel in SETS.items():
        sub = {k: v for k, v in gt.items()
               if (v.get("label") or "").startswith(prefix)}
        missing = [k for k, v in sub.items()
                   if not (ROOT / (v.get("label") or "")).exists()]
        both44 = sum(1 for v in sub.values()
                     if len(v.get("l1") or "") == 44 and len(v.get("l2") or "") == 44)
        # Keep the wrapper shape so downstream loaders see the same schema.
        payload = dict(d) if isinstance(d, dict) else {}
        payload["gt"] = sub
        payload.setdefault("meta", {})
        if isinstance(payload["meta"], dict):
            payload["meta"] = dict(payload["meta"])
            payload["meta"]["split_from"] = "tests/photo_auto_gt.json"
            payload["meta"]["split_prefix"] = prefix
            payload["meta"]["records"] = len(sub)
        out = ROOT / out_rel
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=1))
        print(f"{out_rel}: {len(sub)} records ({both44} with both lines 44), "
              f"{len(missing)} unresolvable labels")
        for m in missing[:5]:
            print(f"    missing: {m}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
