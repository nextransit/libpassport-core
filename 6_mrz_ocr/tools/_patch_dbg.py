#!/usr/bin/env python3
"""One-off patch: add a field-search trace to the C decoder."""
import sys

P = "src/mrz_ocr_cnn.c"
s = open(P, encoding="utf-8").read()

if "MRZ_OCR_FIELDDBG" in s:
    print("already patched")
    sys.exit(0)

old = """                if (nbad > 0) {
                    /* The composite target: the printed digit. */"""

new = """                if (nbad > 0) {
                    {
                        static int fdbg = -1;
                        if (fdbg < 0) { const char *e = getenv("MRZ_OCR_FIELDDBG"); fdbg = (e && e[0] && strcmp(e,"0")!=0) ? 1 : 0; }
                        if (fdbg) {
                            fprintf(stderr, "FLD nbad=%d dest=%s\\n", nbad, dest);
                            for (int f = 0; f < 4; ++f)
                                fprintf(stderr, "   f%d bad=%d nsol=%d\\n", f,
                                        mrz_field_cd_bad(dest, f), nsols[f]);
                        }
                    }
                    /* The composite target: the printed digit. */"""

assert old in s
s = s.replace(old, new, 1)

old2 = """                    if (best_r >= 0) {
                        char cand[45];"""
new2 = """                    {
                        static int fdbg2 = -1;
                        if (fdbg2 < 0) { const char *e = getenv("MRZ_OCR_FIELDDBG"); fdbg2 = (e && e[0] && strcmp(e,"0")!=0) ? 1 : 0; }
                        if (fdbg2)
                            fprintf(stderr, "   target=%d fixed=%d best_r=%d best=%.3f\\n",
                                    target, fixed_comp, best_r, best);
                    }
                    if (best_r >= 0) {
                        char cand[45];"""
assert old2 in s
s = s.replace(old2, new2, 1)

open(P, "w", encoding="utf-8").write(s)
print("patched")
