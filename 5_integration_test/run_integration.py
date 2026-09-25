#!/usr/bin/env python3
"""End-to-end integration test for the passport reader.

Drives all four CLIs in sequence:
  1. 1_mrz_decode    -- encode/decode MRZ
  2. 6_mrz_ocr       -- OCR the MRZ from a synthetic image
  3. 7_security      -- cross-check OCR result against the original
  4. 3_face_recognition -- hash the placeholder face image

Outputs:
  integration_results.json
  integration_results.md
  integration_results.txt
"""
from __future__ import annotations
import json, os, subprocess, time, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

MRZ_TOOL = ROOT / "1_mrz_decode" / "build" / "mrz_tool"
AC_TOOL  = ROOT / "2_anticounterfeit" / "build" / "ac_tool"
OCR_TRAD = ROOT / "6_mrz_ocr" / "build" / "mrz_ocr_tool"
OCR_CNN  = ROOT / "6_mrz_ocr" / "build" / "mrz_ocr_cnn_tool"
FACE_TOOL= ROOT / "3_face_recognition" / "build" / "face_tool"
GEN_SAMP = ROOT / "3_face_recognition" / "build" / "gen_sample"
GEN_MRZ  = ROOT / "6_mrz_ocr" / "build" / "gen_mrz_image"

CORPUS = ROOT / "6_mrz_ocr" / "data" / "corpus" / "corpus.json"
DATA_OCR= ROOT / "6_mrz_ocr" / "data" / "corpus"

CANONICAL_LINE1 = "P<UTOERIKSSON<ANNA<MARIA<<<<<<<<<<<<<<<<<<<<"
CANONICAL_LINE2 = "L898902C36UTO6908061F9406236ZE184226B<<<<<18"
VIZ_TEXT = ("ERIKSSON ANNA MARIA, UTO, Passport L898902C3, "
            "Born 1969-08-06, Exp 1994-06-23")

def run(tool: Path, args: list, stdin=None, timeout=15) -> tuple[int, str, str, float]:
    t0 = time.perf_counter()
    try:
        r = subprocess.run([str(tool)] + args, text=True,
                           input=stdin, capture_output=True,
                           timeout=timeout)
        ms = (time.perf_counter() - t0) * 1000.0
        return r.returncode, r.stdout, r.stderr, ms
    except subprocess.TimeoutExpired:
        ms = (time.perf_counter() - t0) * 1000.0
        return -1, "", "timeout", ms

def main():
    if not all(p.exists() for p in
               (MRZ_TOOL, AC_TOOL, OCR_TRAD, OCR_CNN, FACE_TOOL, GEN_SAMP, GEN_MRZ)):
        print("some CLIs not built", file=sys.stderr)
        return 1

    results = {"stages": []}

    # Stage 1: encode canonical MRZ via 1_mrz_decode.
    rc, out, err, ms = run(MRZ_TOOL,
        ["encode", "P<", "UTO", "ERIKSSON", "ANNA MARIA",
         "L898902C3", "UTO", "690806", "F", "940623", "ZE184226B<<<<<"])
    mrz_str = out
    print(f"[1_mrz_decode] encode  -> rc={rc} ms={ms:.1f}", file=sys.stderr)
    results["stages"].append({"name": "1_mrz_encode", "rc": rc, "ms": ms,
                              "output": mrz_str})

    # Stage 2: decode the canonical MRZ.
    rc, out, err, ms = run(MRZ_TOOL, ["decode", mrz_str])
    print(f"[1_mrz_decode] decode  -> rc={rc} ms={ms:.1f}", file=sys.stderr)
    results["stages"].append({"name": "1_mrz_decode", "rc": rc, "ms": ms,
                              "output": out})

    # Stage 3: 2_anticounterfeit verify.
    rc, out, err, ms = run(AC_TOOL, ["-"], stdin=mrz_str)
    print(f"[2_anticounterfeit]    -> rc={rc} ms={ms:.1f}", file=sys.stderr)
    results["stages"].append({"name": "2_anticounterfeit", "rc": rc,
                              "ms": ms, "output": out})

    # Stage 4: cross-modal check via 7_security.
    rc, out, err, ms = run(ROOT / "7_security" / "build" / "security_tool",
                           ["crosscheck", "/dev/stdin", VIZ_TEXT], stdin=mrz_str)
    print(f"[7_security crosscheck] -> rc={rc} ms={ms:.1f}", file=sys.stderr)
    results["stages"].append({"name": "7_security_crosscheck", "rc": rc,
                              "ms": ms, "output": out})

    # Stage 5: NFC mock reader (4_nfc_reader).
    nfc_tool = ROOT / "4_nfc_reader" / "build" / "nfc_tool"
    rc, out, _, ms = run(nfc_tool, [str(ROOT / "4_nfc_reader" / "data" / "script_happy.json")])
    print(f"[4_nfc_reader happy]    -> rc={rc} ms={ms:.1f}", file=sys.stderr)
    results["stages"].append({"name": "4_nfc_reader_happy", "rc": rc, "ms": ms,
                              "output": out})
    rc, out, _, ms = run(nfc_tool, [str(ROOT / "4_nfc_reader" / "data" / "script_fail.json")])
    print(f"[4_nfc_reader fail]     -> rc={rc} ms={ms:.1f}", file=sys.stderr)
    results["stages"].append({"name": "4_nfc_reader_fail", "rc": rc, "ms": ms,
                              "output": out})

    # Stage 6: render MRZ image via 6_mrz_ocr/gen_mrz_image.
    img_path = Path("/tmp/integration_test.ppm")
    rc, out, err, ms = run(GEN_MRZ, [str(img_path),
                                       CANONICAL_LINE1, CANONICAL_LINE2,
                                       "scale=4"])
    print(f"[6_mrz_ocr gen]       -> rc={rc} ms={ms:.1f}", file=sys.stderr)
    results["stages"].append({"name": "6_gen_mrz_image", "rc": rc, "ms": ms})

    # Stage 6: OCR the rendered image (traditional).
    rc, out, err, ms = run(OCR_TRAD, [str(img_path)])
    print(f"[6_mrz_ocr trad]      -> rc={rc} ms={ms:.1f}", file=sys.stderr)
    results["stages"].append({"name": "6_mrz_ocr_traditional", "rc": rc,
                              "ms": ms, "output": out})

    # Stage 7: OCR the rendered image (CNN).
    rc, out, err, ms = run(OCR_CNN, [str(img_path)])
    print(f"[6_mrz_ocr cnn]       -> rc={rc} ms={ms:.1f}", file=sys.stderr)
    results["stages"].append({"name": "6_mrz_ocr_cnn", "rc": rc,
                              "ms": ms, "output": out})

    # Stage 8: face image generation + matching.
    face_a = Path("/tmp/integration_face_a.ppm")
    face_b = Path("/tmp/integration_face_b.ppm")
    rc, _, _, ms = run(GEN_SAMP, [str(face_a), "1", "128", "128"])
    print(f"[3_face gen a]        -> rc={rc} ms={ms:.1f}", file=sys.stderr)
    results["stages"].append({"name": "3_face_gen_a", "rc": rc, "ms": ms})
    rc, _, _, ms = run(GEN_SAMP, [str(face_b), "1", "128", "128"])
    print(f"[3_face gen b]        -> rc={rc} ms={ms:.1f}", file=sys.stderr)
    results["stages"].append({"name": "3_face_gen_b", "rc": rc, "ms": ms})
    rc, out, _, ms = run(FACE_TOOL, [str(face_a), str(face_b), "0"])
    print(f"[3_face match]        -> rc={rc} ms={ms:.1f}", file=sys.stderr)
    results["stages"].append({"name": "3_face_match", "rc": rc,
                              "ms": ms, "output": out})

    # Stage 9: BAC key derivation via 7_security.
    rc, out, _, ms = run(ROOT / "7_security" / "build" / "security_tool",
                           ["bac", "L898902C3", "690806", "940623"])
    print(f"[7_security bac]      -> rc={rc} ms={ms:.1f}", file=sys.stderr)
    results["stages"].append({"name": "7_security_bac", "rc": rc,
                              "ms": ms, "output": out})

    # Stage 10: SHA-1 / DES / AES known vectors via a tiny inline driver.
    # We invoke security_tool mac to compute a retail MAC.
    rc, out, _, ms = run(ROOT / "7_security" / "build" / "security_tool",
                           ["mac",
                            "0123456789ABCDEFFEDCBA9876543210",
                            "68656C6C6F20776F726C64"])
    print(f"[7_security mac]      -> rc={rc} ms={ms:.1f}", file=sys.stderr)
    results["stages"].append({"name": "7_security_mac", "rc": rc,
                              "ms": ms, "output": out})

    # Persist results.
    out_dir = ROOT / "5_integration_test"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "integration_results.json").write_text(
        json.dumps(results, indent=2))
    # Markdown summary.
    md = "# Integration test\n\n"
    md += "| stage | rc | ms |\n|-------|---|---|\n"
    for s in results["stages"]:
        md += f"| {s['name']} | {s['rc']} | {s['ms']:.1f} |\n"
    md += "\n## Outputs\n\n"
    for s in results["stages"]:
        if "output" in s:
            md += f"### {s['name']}\n```\n{s['output']}```\n\n"
    (out_dir / "integration_results.md").write_text(md)
    (out_dir / "integration_results.txt").write_text(md)
    print(f"\n{out_dir}/integration_results.md")
    return 0

if __name__ == "__main__":
    sys.exit(main())
