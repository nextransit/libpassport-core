#!/usr/bin/env python3
"""Regenerate the three confusion tables from MEASURED statistics.

Tables and their consumers
--------------------------
  HARD_PAIRS   tools/train_cnn_v2.py         training: extra augmented
                                               samples per confusable pair
  CONF         tools/export_omtest_slices.py synthetic-label confusion
                                               alphabet (symmetric)
  confusionPairs  PhotoMrzPipeline.kt + eval_prod_path.py
                                            decode-time correction
                                               candidates (symmetric,
                                               digit pairs cost 1)

All three are REWRITTEN between BEGIN/END markers, merging:
  - the existing curated entries (kept verbatim, order preserved), and
  - measured pairs from tools/collect_confusions.py outputs above
    --min-count (default: 3 for training tables, 5 for decode tables).

'<'-involved pairs are EXCLUDED from the tables: a '<' <-> letter
confusion is almost always a segmentation (extra/missing character)
problem, not a glyph-shape problem, and the decode-side repair would
misfire on it.  They are printed for diagnostics instead.

The tables must be MEASURED, not guessed: the 10-25 review proved the
curated set had drifted from the real confusion surface (bench top pairs
4->9 etc. never made it into any table).
"""

from __future__ import annotations
import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REPO = HERE.parents[1]

TRAIN_PY = HERE / "train_cnn_v2.py"
SLICES_PY = HERE / "export_omtest_slices.py"
KT = REPO / "8_android/app/src/main/java/com/nextransit/mrzbench/photo/PhotoMrzPipeline.kt"
PROD_PY = HERE / "eval_prod_path.py"

BEGIN = "# BEGIN GENERATED CONFUSION PAIRS (tools/gen_confusion_tables.py -- do not edit by hand)"
END = "# END GENERATED CONFUSION PAIRS"
BEGIN_KT = "// BEGIN GENERATED CONFUSION PAIRS (tools/gen_confusion_tables.py -- do not edit by hand)"
END_KT = "// END GENERATED CONFUSION PAIRS"


def load_pairs(stats_paths: list[Path], engine: str) -> dict[str, int]:
    merged: dict[str, int] = {}
    for p in stats_paths:
        d = json.loads(Path(p).read_text())
        for k, n in d.get("pairs", {}).get(engine, {}).items():
            merged[k] = merged.get(k, 0) + n
    return merged


def classify(pairs: dict[str, int]):
    glyph, filler = [], []
    for k, n in pairs.items():
        (filler if k.count("<") else glyph).append((k[0], k[1], n))
    glyph.sort(key=lambda t: -t[2])
    filler.sort(key=lambda t: -t[2])
    return glyph, filler


def undirected(pairs):
    seen = {}
    for a, b, n in pairs:
        key = tuple(sorted((a, b)))
        seen[key] = seen.get(key, 0) + n
    return sorted(((a, b, n) for (a, b), n in seen.items()),
                  key=lambda t: -t[2])


def rewrite_block(text: str, begin: str, end: str, new_block: str,
                  path_for_err: str) -> str:
    if begin in text:
        pre, rest = text.split(begin, 1)
        _, post = rest.split(end, 1)
        return pre + begin + "\n" + new_block + "\n" + end + post
    raise SystemExit(f"marker block not found in {path_for_err}: add "
                     f"'{begin}' / '{end}' around the table first")


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", action="append", required=True,
                    help="confusion stats JSON (tools/collect_confusions.py)")
    ap.add_argument("--min-count-train", type=int, default=3)
    ap.add_argument("--min-count-decode", type=int, default=5)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    stats = [Path(p) for p in args.stats]
    pairs = load_pairs(stats, "cnn")
    glyph, filler = classify(pairs)
    und = undirected(glyph)

    curated_train = {frozenset(p) for p in
                     (("0","O"),("1","I"),("2","Z"),("3","D"),("4","A"),("5","S"),
                      ("6","O"),("7","Z"),("8","O"),("9","S"),("B","8"),("F","6"),
                      ("G","6"),("P","7"),("J","3"),("E","5"),("R","8"),("T","6"),
                      ("M","1"),("Q","O"),("<","0"),("U","V"),("N","M"),("K","X"))}
    curated_decode = {frozenset(p) for p in
                      (("0","O"),("0","Q"),("0","U"),("0","D"),("0","V"),("0","G"),
                       ("1","I"),("1","L"),("1","T"),("1","7"),("5","S"),("2","Z"),
                       ("8","B"),("8","R"),("8","X"),("8","3"),("6","G"),("4","W"),
                       ("4","A"),("W","H"),("W","A"),("W","U"),("W","M"),("H","N"),
                       ("H","M"),("M","N"),("R","B"),("L","C"),("K","B"),("K","X"),
                       ("D","U"),("D","O"),("U","V"),("U","O"),("V","Y"),("Q","O"),
                       ("T","I"),("T","7"),("X","K"))}
    new_train = [(a, b, n) for a, b, n in und
                 if n >= args.min_count_train and frozenset((a, b)) not in curated_train]
    new_decode = [(a, b, n) for a, b, n in und
                  if n >= args.min_count_decode and frozenset((a, b)) not in curated_decode]

    print("measured glyph-level pairs (undirected, count):")
    for a, b, n in und:
        tag = "TRAIN+DECODE" if n >= args.min_count_decode else (
            "train" if n >= args.min_count_train else "-")
        print(f"  {a}<->{b}  {n:4d}  {tag}")
    if filler:
        print("'<'-involved (segmentation-type, NOT tabled):")
        for a, b, n in filler[:10]:
            print(f"  {a}<->{b}  {n:4d}")

    # --- HARD_PAIRS (training) ---
    entries = [(a, b, n) for a, b, n in new_train]
    block = (
        "HARD_PAIRS = [\n"
        "    # curated base (kept from the original hand list)\n"
        "    (\"0\", \"O\"), (\"1\", \"I\"), (\"2\", \"Z\"), (\"3\", \"D\"), (\"4\", \"A\"),\n"
        "    (\"5\", \"S\"), (\"6\", \"O\"), (\"7\", \"Z\"), (\"8\", \"O\"), (\"9\", \"S\"),\n"
        "    (\"B\", \"8\"), (\"F\", \"6\"), (\"G\", \"6\"), (\"P\", \"7\"), (\"J\", \"3\"),\n"
        "    (\"E\", \"5\"), (\"R\", \"8\"), (\"T\", \"6\"), (\"M\", \"1\"), (\"Q\", \"O\"),\n"
        "    (\"<\", \"0\"), (\"U\", \"V\"), (\"N\", \"M\"), (\"K\", \"X\"),\n"
        "    # measured on real photos (tools/collect_confusions.py, official GTs)\n"
        + "".join(f"    (\"{a}\", \"{b}\"),  # n={n}\n" for a, b, n in entries)
        + "]")
    if not args.dry_run:
        text = TRAIN_PY.read_text()
        text = rewrite_block(text, BEGIN, END, block, str(TRAIN_PY))
        TRAIN_PY.write_text(text)

    # --- CONF (synthetic label alphabet, symmetric 2-char strings) ---
    conf_entries = []
    for a, b, n in new_train:
        if (a.isdigit() or b.isdigit()) and (a.isalpha() or b.isalpha()):
            conf_entries.append((a, b, n))
    conf_block = (
        "CONF = [\"0O\",\"0Q\",\"0U\",\"0D\",\"0V\",\"0G\",\"1I\",\"1L\",\"1T\",\"17\",\"5S\",\"2Z\",\"8B\",\"8R\",\"8X\",\n"
        "        \"83\",\"6G\",\"4W\",\"4A\",\"WH\",\"WA\",\"WU\",\"WM\",\"HN\",\"HM\",\"MN\",\"RB\",\"LC\",\"KB\",\"KX\",\n"
        "        \"DU\",\"DO\",\"UV\",\"UO\",\"VY\",\"QO\",\"TI\",\"T7\",\"XK\",\n"
        "        # measured on real photos (tools/collect_confusions.py)\n"
        + "".join(f"        \"{a}{b}\",  # n={n}\n" for a, b, n in conf_entries)
        + "       ]")
    if not args.dry_run:
        text = SLICES_PY.read_text()
        text = rewrite_block(text, BEGIN, END, conf_block, str(SLICES_PY))
        SLICES_PY.write_text(text)

    # --- confusionPairs (decode-time, Kotlin + eval_prod_path.py) ---
    def kt_entries(a, b, n):
        # Kotlin line comment: '#' is not a comment there (it breaks the
        # whole list literal -> the map's type inference fails).
        return f"        \"{a}\" to \"{b}\",  // n={n}"
    kt_block = (
        "    private val confusionPairs = listOf(\n"
        "        // curated base\n"
        "        \"0\" to \"O\", \"0\" to \"Q\", \"0\" to \"U\", \"0\" to \"D\",\n"
        "        \"0\" to \"V\", \"0\" to \"G\", \"1\" to \"I\", \"1\" to \"L\",\n"
        "        \"1\" to \"T\", \"1\" to \"7\", \"5\" to \"S\", \"2\" to \"Z\",\n"
        "        \"8\" to \"B\", \"8\" to \"R\", \"8\" to \"X\", \"8\" to \"3\",\n"
        "        \"6\" to \"G\", \"4\" to \"W\", \"4\" to \"A\", \"W\" to \"H\",\n"
        "        \"W\" to \"A\", \"W\" to \"U\", \"W\" to \"M\", \"H\" to \"N\",\n"
        "        \"H\" to \"M\", \"M\" to \"N\", \"R\" to \"B\", \"L\" to \"C\",\n"
        "        \"K\" to \"B\", \"K\" to \"X\", \"D\" to \"U\", \"D\" to \"O\",\n"
        "        \"U\" to \"V\", \"U\" to \"O\", \"V\" to \"Y\", \"Q\" to \"O\",\n"
        "        \"T\" to \"I\", \"T\" to \"7\", \"X\" to \"K\",\n"
        "        // measured on real photos (tools/collect_confusions.py)\n"
        + "\n".join(kt_entries(a, b, n) for a, b, n in new_decode)
        + "\n")
    py_block = (
        "_PAIRS = [\n"
        "    # curated base (same as PhotoMrzPipeline.confusionPairs)\n"
        "    (\"0\", \"O\"), (\"0\", \"Q\"), (\"0\", \"U\"), (\"0\", \"D\"),\n"
        "    (\"0\", \"V\"), (\"0\", \"G\"), (\"1\", \"I\"), (\"1\", \"L\"),\n"
        "    (\"1\", \"T\"), (\"1\", \"7\"), (\"5\", \"S\"), (\"2\", \"Z\"),\n"
        "    (\"8\", \"B\"), (\"8\", \"R\"), (\"8\", \"X\"), (\"8\", \"3\"),\n"
        "    (\"6\", \"G\"), (\"4\", \"W\"), (\"4\", \"A\"), (\"W\", \"H\"),\n"
        "    (\"W\", \"A\"), (\"W\", \"U\"), (\"W\", \"M\"), (\"H\", \"N\"),\n"
        "    (\"H\", \"M\"), (\"M\", \"N\"), (\"R\", \"B\"), (\"L\", \"C\"),\n"
        "    (\"K\", \"B\"), (\"K\", \"X\"), (\"D\", \"U\"), (\"D\", \"O\"),\n"
        "    (\"U\", \"V\"), (\"U\", \"O\"), (\"V\", \"Y\"), (\"Q\", \"O\"),\n"
        "    (\"T\", \"I\"), (\"T\", \"7\"), (\"X\", \"K\"),\n"
        "    # measured on real photos (tools/collect_confusions.py)\n"
        + "".join(f"    (\"{a}\", \"{b}\"),  # n={n}\n" for a, b, n in new_decode)
        + "]")
    if not args.dry_run:
        text = KT.read_text()
        text = rewrite_block(text, BEGIN_KT, END_KT, kt_block, str(KT))
        KT.write_text(text)
        text = PROD_PY.read_text()
        text = rewrite_block(text, BEGIN, END, py_block, str(PROD_PY))
        PROD_PY.write_text(text)

    print(("dry-run: nothing written" if args.dry_run
           else "updated: train_cnn_v2.py / export_omtest_slices.py / "
                "PhotoMrzPipeline.kt / eval_prod_path.py"))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
