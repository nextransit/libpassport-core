#!/usr/bin/env python3
"""Install trained MRZ weights into the source tree and rebuild safely.

Two traps this script exists to avoid:

1. `cp -p` preserves the source mtime, so CMake considered the header
   unchanged and skipped recompiling -- the new weights never reached
   the binary, and every candidate evaluation was silently invalid.
2. The CNN has three weight headers.  Production runs with
   `MRZ_OCR_LINE_MODELS=1`, which only reads `LINE1_*` / `LINE2_*` from
   `cnn_weights_line1.h` / `cnn_weights_line2.h`.  The training scripts
   emit `DEFAULT_*` into `cnn_weights.h`, which production never uses.

Usage:
    python3 6_mrz_ocr/tools/install_weights.py --crnn /tmp/crnn_candidate.h
    python3 6_mrz_ocr/tools/install_weights.py --cnn-line1 a.h --cnn-line2 b.h
    python3 6_mrz_ocr/tools/install_weights.py --verify crnn
"""
from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "6_mrz_ocr" / "src"
BUILD = ROOT / "6_mrz_ocr" / "build"
TOOL = BUILD / "mrz_ocr_cnn_tool"

TARGETS = {
    "crnn": SRC / "crnn_weights.h",
    "cnn": SRC / "cnn_weights.h",
    "cnn-line1": SRC / "cnn_weights_line1.h",
    "cnn-line2": SRC / "cnn_weights_line2.h",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def install(src: Path, key: str) -> int:
    dst = TARGETS[key]
    before = sha(dst)
    # write_bytes() refreshes mtime; `cp -p` does not, which is the trap.
    dst.write_bytes(src.read_bytes())
    print(f"installed {src.name} -> {dst.name}  "
          f"sha {before[:12]} -> {sha(dst)[:12]}")
    return 0


def build() -> int:
    result = subprocess.run(["cmake", "--build", str(BUILD), "-j8"],
                            capture_output=True, text=True)
    out = result.stdout + result.stderr
    compiled = [line.strip() for line in out.splitlines()
                if "Building C object" in line]
    for line in compiled:
        print("  " + line)
    if not compiled:
        print("  !! no compile action -- weights may NOT be in the binary")
        print(f"  !! run: cmake --build {BUILD} --clean-first -j8")
        return 1
    return 0


def verify(target: str) -> int:
    """Smoke check: the CRNN-only output must differ from the old weights."""
    band = ROOT / "10_ocr_research" / "_diag_real_band.ppm"
    if not band.exists():
        print(f"no smoke band file: {band}")
        return 2
    env = dict(os.environ)
    env["MRZ_OCR_LINE_MODELS"] = "1"
    env["MRZ_OCR_CRNN"] = "1" if target.startswith("crnn") else "0"
    env.pop("MRZ_OCR_FUSE", None)
    proc = subprocess.run([str(TOOL), str(band)], text=True,
                          capture_output=True, env=env)
    for line in proc.stdout.splitlines():
        if "result.line" in line:
            print("  " + line.strip())
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--crnn")
    parser.add_argument("--cnn")
    parser.add_argument("--cnn-line1")
    parser.add_argument("--cnn-line2")
    parser.add_argument("--verify", choices=sorted(TARGETS))
    args = parser.parse_args()

    if args.verify:
        return verify(args.verify)

    changed = False
    for key, value in (("crnn", args.crnn), ("cnn", args.cnn),
                       ("cnn-line1", args.cnn_line1),
                       ("cnn-line2", args.cnn_line2)):
        if value:
            install(Path(value), key)
            changed = True
    if not changed:
        parser.error("nothing to install; pass --crnn/--cnn/--cnn-line1/--cnn-line2")
    return build()


if __name__ == "__main__":
    sys.exit(main())
