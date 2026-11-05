#!/usr/bin/env python3
"""Named real-photo sets with document-level grouping.

Why a set definition and not just a directory
---------------------------------------------
The photo directories do NOT hold one photo per document:

    tmp/PassPortPhoto  44 photos / 40 documents  (3 documents shot twice
                                                  or three times)
    Photo/              6 photos /  1 document   (six camera shots of one
                                                  passport)
    assets/pic         64 photos / 57 documents

A train/test split made by file therefore leaks: the model has seen one
rendering of a document and is then asked to read another rendering of
the same document.  `doc_key` groups the photos of one document so a
split can keep them together, and `check_no_leakage` proves it did.

The ground truth is NOT duplicated here: it stays in
10_ocr_research/gt_review_loop.xlsx ("GT对比") and is read through
`gui._gt_map()`, so the sheet and the sets cannot drift apart.  A photo
without a ground truth becomes its own document group.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SETS_JSON = Path(__file__).resolve().parent / "photo_sets.json"
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".ppm")


def load_sets(path: Path | str = SETS_JSON) -> dict:
    """The set definitions (name -> dirs / weights / description)."""
    return json.loads(Path(path).read_text())


def _dir_entries(base: Path) -> list[Path]:
    """Photos in one directory.  Photo/ is a tree of camera shots; the
    other directories are flat.  Same rule as gui.iter_passport_photos."""
    if not base.is_dir():
        return []
    if base.name == "Photo":
        entries = sorted(base.glob("*/*"))
    else:
        entries = sorted(base.glob("*"))
    return [f for f in entries if f.suffix.lower() in IMAGE_SUFFIXES]


def resolve(name: str, sets: dict | None = None) -> list[tuple[str, Path]]:
    """(label, path) for every photo in a named set, in a stable order.

    The label keeps the directory prefix ('tmp/PassPortPhoto/x.jpg'), so
    a photo stays identifiable in the GUI list and in a report.
    """
    sets = sets or load_sets()
    if name not in sets["sets"]:
        raise KeyError(f"unknown photo set {name!r}; "
                       f"have: {', '.join(sorted(sets['sets']))}")
    out: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for d in sets["sets"][name]["dirs"]:
        base = ROOT / d
        for f in _dir_entries(base):
            key = str(f.resolve())
            if key in seen:
                continue
            seen.add(key)
            label = f"{d}/{f.relative_to(base)}"
            out.append((label, f))
    return out


def doc_key(label: str, gt_map: dict | None = None) -> str:
    """The document a photo belongs to.

    Derived from the ground truth's line 1 with the fillers removed, so
    every rendering of one document lands in the same group: 'P<CHN' and
    'P<CHN<<' are the same document, and the camera shots of Photo/ share
    one key.  A photo without a ground truth is its own group.
    """
    base = Path(label).name
    gt = (gt_map or {}).get(base)
    if not gt:
        return "no-gt:" + base
    line1 = gt[0] if isinstance(gt, (tuple, list)) else gt.get("l1", "")
    letters = "".join(ch for ch in str(line1).upper() if ch.isalpha())
    return letters or "no-gt:" + base


def group_by_document(photos: list[tuple[str, Path]],
                      gt_map: dict | None = None) -> dict[str, list[str]]:
    """document key -> photo labels."""
    groups: dict[str, list[str]] = {}
    for label, _p in photos:
        groups.setdefault(doc_key(label, gt_map), []).append(label)
    return groups


def split(name: str, gt_map: dict | None = None, holdout_docs: int = 0,
          seed: int = 1, sets: dict | None = None) -> dict:
    """Split a set into train/test at DOCUMENT level.

    `holdout_docs` documents (not photos) go to the test side.  With
    holdout_docs=0 everything is test material.
    """
    photos = resolve(name, sets)
    groups = group_by_document(photos, gt_map)
    keys = sorted(groups)
    if holdout_docs:
        rng = random.Random(seed)
        hold = set(rng.sample(keys, min(holdout_docs, len(keys))))
    else:
        hold = set()
    train = [label for k in keys if k not in hold
             for label in groups[k]]
    test = [label for k in keys if k in hold
            for label in groups[k]]
    return {"train": train, "test": test,
            "groups": groups, "n_docs": len(keys)}


def check_no_leakage(train: list[str], test: list[str],
                     gt_map: dict | None = None) -> list[str]:
    """Documents that appear on BOTH sides.  Empty list == clean split."""
    tr = {doc_key(l, gt_map) for l in train}
    te = {doc_key(l, gt_map) for l in test}
    return sorted(tr & te)


def describe(name: str, gt_map: dict | None = None,
             sets: dict | None = None) -> str:
    """One-line summary: photos, documents, GT coverage."""
    sets = sets or load_sets()
    photos = resolve(name, sets)
    groups = group_by_document(photos, gt_map)
    n_gt = sum(1 for label, _ in photos
               if (gt_map or {}).get(Path(label).name))
    return (f"{name}: {len(photos)} photos / {len(groups)} documents "
            f"/ {n_gt} with GT"
            + (f" -- {sets['sets'][name]['description']}"
               if name in sets["sets"] else ""))


if __name__ == "__main__":
    # Headless summary; the GT map needs the GUI module for the xlsx.
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "gui", ROOT / "tests" / "gui.py")
    gui = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gui)
    gui.PassportGUI.AUTO_WARM_ENGINES = False
    app = gui.PassportGUI()
    app.update()
    app.withdraw()
    gt = app._gt_map()
    sets = load_sets()
    for name in sorted(sets["sets"]):
        print(describe(name, gt, sets))
