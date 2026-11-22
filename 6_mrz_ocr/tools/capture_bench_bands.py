#!/usr/bin/env python3
"""Capture the band each engine actually reads, during the bench run.

The bench feeds the engines a band cut from the ORIGINAL photo
(`_rst_native_mrz_band` + `_ocr_pick_band`), while the CRNN weights were
trained on features exported from `/tmp/holdout_export2/*.ppm` slices.
Those two bands differ in size (bench ~2398x213 colour, holdout 1440x87),
which is the prime suspect for the CRNN's "high char accuracy, zero
exact" row: 28/47 on the holdout slices, 0/47 on the bench bands.

This hooks the resident-tool `run` so every band that reaches an engine
is copied to disk before the engine sees it, then prints the per-engine
verdict for the same band.

Usage:
  python3 tools/capture_bench_bands.py --limit 3
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests"))

OUT = Path("/tmp/bench_bands")
OUT.mkdir(parents=True, exist_ok=True)

import mrz_engine_pool  # noqa: E402

_orig_run = mrz_engine_pool.ResidentTool.run
_state = {"n": 0}


def _patched_run(self, path, *a, **kw):
    p = Path(path)
    try:
        _state["n"] += 1
        dst = OUT / f"{_state['n']:03d}_{p.name}"
        shutil.copyfile(p, dst)
    except Exception:
        pass
    return _orig_run(self, path, *a, **kw)


mrz_engine_pool.ResidentTool.run = _patched_run


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=3)
    ap.add_argument("--dirs", default="tmp/PassPortPhoto")
    args = ap.parse_args()

    import photo_direct_bench as P
    argv = ["--methods", "cnn,crnn,fuse", "--dirs", args.dirs,
            "--resident", "--limit", str(args.limit)]
    rc = P.main(argv)
    print(f"\ncaptured {_state['n']} bands into {OUT}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
