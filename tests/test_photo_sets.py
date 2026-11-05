#!/usr/bin/env python3
"""Photo-set definitions: resolution, document grouping, leak-free split.

Run:  python3 tests/test_photo_sets.py
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

import photo_sets  # noqa: E402

FAILURES = []


def check(cond, msg):
    if not cond:
        FAILURES.append(msg)


def load_gui():
    spec = importlib.util.spec_from_file_location(
        "gui", ROOT / "tests" / "gui.py")
    gui = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gui)
    gui.PassportGUI.AUTO_WARM_ENGINES = False
    app = gui.PassportGUI()
    app.update()
    app.withdraw()
    return app


def main() -> int:
    sets = photo_sets.load_sets()
    check("sets" in sets, "photo_sets.json has no 'sets' block")

    app = load_gui()
    gt = app._gt_map()

    # 1. every set resolves, and the labels keep the directory prefix
    expected = {"passport": 44, "camshots": 6, "passport_union": 50,
                "pic": 64, "all": 114}
    for name, n in expected.items():
        photos = photo_sets.resolve(name, sets)
        check(len(photos) == n,
              f"set {name}: {len(photos)} photos, expected {n}")
        check(all("/" in l for l, _ in photos),
              f"set {name}: a label lost its directory prefix")

    # 2. document grouping: the numbers that make a file-level split leak
    g_passport = photo_sets.group_by_document(
        photo_sets.resolve("passport", sets), gt)
    g_cam = photo_sets.group_by_document(
        photo_sets.resolve("camshots", sets), gt)
    g_union = photo_sets.group_by_document(
        photo_sets.resolve("passport_union", sets), gt)
    g_pic = photo_sets.group_by_document(
        photo_sets.resolve("pic", sets), gt)
    check(len(g_passport) == 40,
          f"passport: {len(g_passport)} documents, expected 40")
    check(len(g_cam) == 1,
          f"camshots: {len(g_cam)} documents, expected 1")
    check(len(g_union) == 41,
          f"passport_union: {len(g_union)} documents, expected 41")
    check(len(g_pic) == 57,
          f"pic: {len(g_pic)} documents, expected 57")
    biggest = max(g_cam.values(), key=len)
    check(len(biggest) == 6,
          f"camshots: the largest document group has {len(biggest)} photos")

    # 3. a document-level split never leaks, for several hold-out sizes
    for hold in (1, 5, 12, 41):
        sp = photo_sets.split("passport_union", gt, holdout_docs=hold)
        leak = photo_sets.check_no_leakage(sp["train"], sp["test"], gt)
        check(not leak, f"holdout {hold}: documents on both sides: {leak}")
        check(len(sp["train"]) + len(sp["test"]) == 50,
              f"holdout {hold}: the split lost photos")

    # 4. the six camera shots must land on ONE side of any split
    sp = photo_sets.split("passport_union", gt, holdout_docs=8)
    cam_labels = {l for l, _ in photo_sets.resolve("camshots", sets)}
    in_train = cam_labels & set(sp["train"])
    in_test = cam_labels & set(sp["test"])
    check(not (in_train and in_test),
          "the six camera shots straddled the split")
    check(len(in_train) + len(in_test) == 6,
          "the camera shots went missing from the split")

    # 5. the multi-shot documents in PassPortPhoto must not straddle either
    multi = [k for k, v in g_passport.items() if len(v) > 1]
    check(len(multi) == 3,
          f"passport: {len(multi)} multi-shot documents, expected 3")
    for hold in (1, 7, 20):
        sp = photo_sets.split("passport_union", gt, holdout_docs=hold)
        for k in multi:
            labels = set(g_passport[k])
            side_tr = labels & set(sp["train"])
            side_te = labels & set(sp["test"])
            check(not (side_tr and side_te),
                  f"holdout {hold}: document {k[:16]} straddled the split")

    # 6. a photo without a ground truth is its own group (never leaks)
    no_gt = [l for l, _ in photo_sets.resolve("pic", sets)
             if not gt.get(Path(l).name)]
    for label in no_gt:
        key = photo_sets.doc_key(label, gt)
        check(key.startswith("no-gt:"),
              f"{label}: a photo without GT joined a shared group")

    print("\n".join(f"  {n}: {photo_sets.describe(n, gt, sets)}"
                    for n in sorted(sets["sets"])))
    if FAILURES:
        print(f"\n[TEST] {len(FAILURES)} failed")
        for f in FAILURES:
            print(" -", f)
        return 1
    print("\n[TEST] photo sets: all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
