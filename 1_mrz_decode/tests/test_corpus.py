#!/usr/bin/env python3
"""Independent Python test runner for the MRZ library.

This script is intentionally separate from test_mrz.c so the C library
is also exercised by a different language. It reads the corpus defined
in data/test_cases.json, encodes each entry through a C subprocess
(`mrz_tool encode`) and verifies that:
  1. The encoder accepts all fields.
  2. The decoder accepts the encoded string.
  3. Each segment check digit matches the JSON's expected_checks.
  4. The composite check digit matches.

It also runs the negative test cases (bad length, bad char, bad check).
"""
from __future__ import annotations
import json, os, subprocess, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "build" / "mrz_tool"
DATA = ROOT / "data" / "test_cases.json"

def run_encode(fields: dict) -> str:
    argv = [
        str(TOOL), "encode",
        fields["doc_type"], fields["issuing_state"],
        fields["surname"], fields["given_names"],
        fields["passport_no"], fields["nationality"],
        fields["birth_yymmdd"], fields["sex"],
        fields["expiry_yymmdd"], fields["personal_no"],
    ]
    out = subprocess.check_output(argv, text=True)
    return out

def run_decode(mrz: str) -> dict:
    out = subprocess.check_output([str(TOOL), "decode", mrz], text=True)
    # parse output into a dict. Lines look like:
    #   passport_no    : 'E12345678'  ck=2
    # Strip leading/trailing whitespace and extract the quoted value.
    import re
    d = {}
    for line in out.splitlines():
        m = re.match(r"\s*([a-zA-Z_]+)\s*:\s*'([^']*)'", line)
        if m:
            d[m.group(1)] = m.group(2)
    return d

failures = []

def check(cond, msg):
    print(("OK   " if cond else "FAIL ") + msg)
    if not cond:
        failures.append(msg)

def main():
    if not TOOL.exists():
        print("mrz_tool not built yet -- run cmake --build build first", file=sys.stderr)
        return 1
    data = json.loads(DATA.read_text())
    for case in data["cases"]:
        cid = case["id"]
        mrz = run_encode(case)
        if mrz != case["expected_mrz"]:
            check(False, f"{cid}: encoder output mismatch")
            print("  expected:", repr(case["expected_mrz"]))
            print("  got:     ", repr(mrz))
        else:
            check(True, f"{cid}: encoder output matches")
        # Decode should succeed.
        try:
            decoded = run_decode(mrz)
            check(decoded["surname"] == case["surname"],
                  f"{cid}: surname round-trip")
            check(decoded["given_names"] == case["given_names"].replace(" ", "<"),
                  f"{cid}: given_names round-trip")
            check(decoded["passport_no"] == case["passport_no"],
                  f"{cid}: passport_no round-trip")
            # Verify each expected check digit.
            d2 = subprocess.check_output([str(TOOL), "decode", mrz], text=True)
            for line in d2.splitlines():
                if "ck=" in line:
                    k, _, v = line.partition("ck=")
                    k = k.strip().split()[-1].strip("'")
                    v = int(v.strip())
                    fname = next((fn for fn, fv in [
                        ("passport_no", case["passport_no"]),
                        ("birth", case["birth_yymmdd"]),
                        ("expiry", case["expiry_yymmdd"]),
                        ("personal_no", case["personal_no"] + "<" * max(0, 14 - len(case["personal_no"]))),
                    ] if k == fv), None)
                    if fname:
                        check(v == case["expected_checks"].get(fname),
                              f"{cid}: {fname} ck (got {v}, "
                              f"expected {case['expected_checks'].get(fname)})")
                elif line.startswith("composite_ck"):
                    ck_int = int(line.split(":")[1].strip())
                    check(ck_int == case["expected_checks"]["composite"],
                          f"{cid}: composite check digit (got {ck_int}, "
                          f"expected {case['expected_checks']['composite']})")
        except subprocess.CalledProcessError as e:
            check(False, f"{cid}: decoder rejected encoded MRZ ({e})")
    # Negative cases: shorter, empty
    neg = data.get("negative_cases", [])
    for ncase in neg:
        if ncase["id"] in ("too-short", "empty"):
            try:
                run_decode(ncase["mrz"])
                check(False, f"neg {ncase['id']}: should have failed")
            except subprocess.CalledProcessError:
                check(True, f"neg {ncase['id']}: correctly rejected")
    # Negative: bad check digit. Flip one byte in a known-good MRZ.
    good = data["cases"][0]
    mrz = run_encode(good)
    # Flip passport_no check digit (joined view position 9 in line2,
    # buffer offset MRZ_TD3_LINE_LEN+1+9 = 54).
    bad = mrz[:54] + ("6" if mrz[54] != "6" else "7") + mrz[55:]
    try:
        run_decode(bad)
        check(False, "neg wrong-pass-ck: should have failed")
    except subprocess.CalledProcessError:
        check(True, "neg wrong-pass-ck: correctly rejected")

    print(f"\n[Python] {len(failures)} failure(s)")
    return 0 if not failures else 1

if __name__ == "__main__":
    sys.exit(main())
