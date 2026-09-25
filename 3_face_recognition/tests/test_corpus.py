#!/usr/bin/env python3
"""Independent Python test runner for the face-recognition tool.

Generates pairs of synthetic images with `gen_sample`, runs them
through `face_tool` and asserts:
  * Same seed   -> high score (>= 95)
  * Different seed -> lower score (< 95 expected; usually < 90)
  * Wrong path  -> non-zero exit
"""
from __future__ import annotations
import json, os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GEN  = ROOT / "build" / "gen_sample"
TOOL = ROOT / "build" / "face_tool"
TMP  = Path("/tmp/face_corpus")
TMP.mkdir(exist_ok=True)

failures = []
def check(cond, msg):
    print(("OK   " if cond else "FAIL ") + msg)
    if not cond:
        failures.append(msg)

def gen(name, seed, w=128, h=128):
    p = TMP / f"{name}.ppm"
    subprocess.check_call([str(GEN), str(p), str(seed), str(w), str(h)])
    return str(p)

def compare(a, b, method):
    r = subprocess.run([str(TOOL), a, b, str(method)], text=True, capture_output=True)
    return r.returncode, r.stdout

def main():
    if not GEN.exists() or not TOOL.exists():
        print("tools not built -- run cmake --build build first", file=sys.stderr)
        return 1
    # 1. Same seed -> identical score 100
    a = gen("a", 1)
    b = gen("b", 1)
    rc, out = compare(a, b, 0)
    check(rc == 0, f"same seed aHash rc==0 (got {rc})")
    check("100/100" in out, f"same seed aHash score 100 (got: {out.strip()})")
    rc, out = compare(a, b, 1)
    check(rc == 0, f"same seed pHash rc==0 (got {rc})")
    check("100/100" in out, f"same seed pHash score 100 (got: {out.strip()})")

    # 2. Different seeds -> score should be < 100 (and often < 95)
    c = gen("c", 99)
    rc, out = compare(a, c, 0)
    check(rc in (0, 1), f"different seed aHash rc in (0,1) (got {rc})")
    score_line = [l for l in out.splitlines() if "score" in l][0]
    score = int(score_line.split(":")[1].split("/")[0].strip())
    check(score < 100, f"different seed aHash score < 100 (got {score})")

    # 3. Combined aHash + pHash on identical images -> 100
    rc, out = compare(a, b, 2)
    check("100/100" in out, f"combined method on identical -> 100 (got: {out.strip()})")

    # 4. Compare with missing file -> non-zero
    r = subprocess.run([str(TOOL), a, "/tmp/does-not-exist.ppm"], text=True, capture_output=True)
    check(r.returncode != 0, f"missing file -> non-zero exit (got {r.returncode})")

    # 5. Different image dimensions, same seed -> still high but not always 100
    big_a = gen("big_a", 1, 200, 200)
    big_b = gen("big_b", 1, 200, 200)
    rc, out = compare(big_a, big_b, 0)
    check("100/100" in out, f"same seed larger images aHash -> 100 (got: {out.strip()})")

    print(f"\n[Python] {len(failures)} failure(s)")
    return 0 if not failures else 1

if __name__ == "__main__":
    sys.exit(main())
