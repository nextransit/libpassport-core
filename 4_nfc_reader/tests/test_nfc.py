#!/usr/bin/env python3
"""End-to-end NFC mock reader test."""
import subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NFC = ROOT / "build" / "nfc_tool"

def run(script):
    r = subprocess.run([str(NFC), str(script)], text=True,
                       capture_output=True)
    return r.returncode, r.stdout, r.stderr

def main():
    rc, out, err = run(ROOT / "data" / "script_happy.json")
    print("=== happy path ===")
    print(out)
    print("=== failure path ===")
    rc, out, err = run(ROOT / "data" / "script_fail.json")
    print(out)
    return 0

if __name__ == "__main__":
    sys.exit(main())
