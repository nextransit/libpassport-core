#!/usr/bin/env python3
"""Independent Python test runner for the anti-counterfeit verifier.

Reads the test cases defined in data/test_cases.json, runs the C tool
`ac_tool` on each MRZ and checks the exit code + the report summary.
"""
from __future__ import annotations
import json, os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "build" / "ac_tool"
DATA = ROOT / "data" / "test_cases.json"

failures = []
def check(cond, msg):
    print(("OK   " if cond else "FAIL ") + msg)
    if not cond:
        failures.append(msg)

def run(mrz: str) -> tuple[int, str]:
    p = subprocess.run([str(TOOL), "-"], input=mrz, text=True,
                       capture_output=True)
    return p.returncode, p.stdout

def main():
    if not TOOL.exists():
        print("ac_tool not built -- run cmake --build build first", file=sys.stderr)
        return 1
    data = json.loads(DATA.read_text())
    for case in data["cases"]:
        rc, out = run(case["mrz"])
        expect = case["expected_severity"]
        if expect == "OK":
            check(rc == 0, f"{case['id']}: rc 0 expected")
            check("check_digits" in out and "OK" in out, f"{case['id']}: check_digits OK")
        elif expect == "FAIL":
            check(rc == 2, f"{case['id']}: rc 2 expected, got {rc}")
        elif expect == "WARN":
            check(rc in (1, 2), f"{case['id']}: rc 1 or 2, got {rc}")
        else:
            check(False, f"{case['id']}: unknown expected_severity {expect}")
    print(f"\n[Python] {len(failures)} failure(s)")
    return 0 if not failures else 1

if __name__ == "__main__":
    sys.exit(main())
