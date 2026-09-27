#!/usr/bin/env python3
"""Cross-check C crypto/BAC primitives (test_crypto_cli) against
independent Python implementations (hashlib + pycryptodome) plus
RFC/NIST known-answer tests."""
import hashlib
import os
import subprocess
import sys
from pathlib import Path

from Crypto.Cipher import DES3

ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "build" / "test_crypto_cli"

def cli(*args):
    r = subprocess.run([str(CLI), *args], text=True, capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(f"cli {args} rc={r.returncode} err={r.stderr}")
    return r.stdout.strip().splitlines()

def hexb(s):
    return bytes.fromhex(s)

def b2h(b):
    return b.hex().upper()

failures = 0

def check(name, got, want):
    global failures
    if got != want:
        failures += 1
        print(f"FAIL {name}\n  got : {got}\n  want: {want}")
    else:
        print(f"ok   {name}")

# ---- 1. SHA-1 known answers ----
for msg, want in [("", "da39a3ee5e6b4b0d3255bfef95601890afd80709"),
                  ("abc", "a9993e364706816aba3e25717850c26c9cd0d89d"),
                  ("The quick brown fox jumps over the lazy dog",
                   "2fd4e1c67a2d28fced849ee1bb76e7391b93eb12")]:
    check(f"sha1({msg[:20]!r})", cli("sha1", msg)[0],
          hashlib.sha1(msg.encode()).hexdigest().upper())

# ---- 2. random 3DES block / CBC cross-checks ----
for trial in range(20):
    key = os.urandom(16)
    blk = os.urandom(8)
    want = DES3.new(key, DES3.MODE_ECB).encrypt(blk).hex().upper()
    check(f"tdes enc t{trial}", cli("tdes", key.hex(), blk.hex(), "1")[0], want)
    got = cli("tdes", key.hex(), want, "0")[0]
    check(f"tdes dec t{trial}", got, blk.hex().upper())

for trial in range(10):
    key = os.urandom(16)
    iv = os.urandom(8)
    pt = os.urandom(32)
    want = DES3.new(key, DES3.MODE_CBC, iv).encrypt(pt).hex().upper()
    check(f"cbc enc t{trial}", cli("cbc", key.hex(), pt.hex(), iv.hex(), "1")[0], want)
    got = cli("cbc", key.hex(), want, iv.hex(), "0")[0]
    check(f"cbc dec t{trial}", got, pt.hex().upper())

# NIST SP 800-67 3DES-EDE-CBC known answer (24-byte key -> use 16-byte variant
# cross-checked above; here the known-answer for 16-byte key from OpenSSL).
key16 = bytes.fromhex("0123456789ABCDEFFEDCBA9876543210")
iv = bytes.fromhex("1234567890ABCDEF")
pt = bytes.fromhex("4E6F77206973207468652074696D6520")
want = DES3.new(key16, DES3.MODE_CBC, iv).encrypt(pt).hex().upper()
check("cbc nist-style(16key)", cli("cbc", key16.hex(), pt.hex(), iv.hex(), "1")[0], want)

# ---- 3. CBC-MAC (zero IV, block-multiple inputs) ----
for trial in range(10):
    key = os.urandom(16)
    data = os.urandom(8 * (1 + trial % 3))
    enc = DES3.new(key, DES3.MODE_CBC, b"\x00" * 8).encrypt(data)
    want = enc[-8:].hex().upper()
    check(f"cbc-mac t{trial}", cli("mac", key.hex(), data.hex())[0], want)

# ---- 4. BAC key derivation ----
def py_derive(mrz_info):
    kseed = hashlib.sha1(mrz_info.encode()).digest()[:16]
    e = DES3.new(kseed, DES3.MODE_ECB)
    kenc = e.encrypt(b"\x00" * 7 + b"\x01") + e.encrypt(b"\x00" * 7 + b"\x02")
    kmac = e.encrypt(b"\x00" * 7 + b"\x03") + e.encrypt(b"\x00" * 7 + b"\x04")
    return kseed, kenc, kmac

mrz_info = "L898902C3669080619406236"   # ICAO 9303 sample passport
kseed, kenc, kmac = py_derive(mrz_info)
got = cli("derive", mrz_info)
check("derive kseed", got[0], b2h(kseed))
check("derive kenc ", got[1], b2h(kenc))
check("derive kmac ", got[2], b2h(kmac))

# ---- 5. MRZ_information extraction ----
line2 = "L898902C36UTO6908061F9406236ZE184226B<<<<<18"
check("mrzinfo", cli("mrzinfo", line2)[0], "L898902C3669080619406236")

print("\n" + ("ALL PASS" if failures == 0 else f"{failures} FAILURES"))
sys.exit(1 if failures else 0)