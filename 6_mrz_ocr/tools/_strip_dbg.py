#!/usr/bin/env python3
"""One-off patch: remove the field-search debug traces."""
import sys

P = "src/mrz_ocr_cnn.c"
s = open(P, encoding="utf-8").read()

before = len(s)

blocks = [
"""
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
""",
"""
                    {
                        static int fdbg2 = -1;
                        if (fdbg2 < 0) { const char *e = getenv("MRZ_OCR_FIELDDBG"); fdbg2 = (e && e[0] && strcmp(e,"0")!=0) ? 1 : 0; }
                        if (fdbg2)
                            fprintf(stderr, "   target=%d fixed=%d best_r=%d best=%.3f\\n",
                                    target, fixed_comp, best_r, best);
                    }
""",
"""                        double scf = mrz_candidate_logprob(cand, t3);
                        {
                            static int fd3 = -1;
                            if (fd3 < 0) { const char *e = getenv("MRZ_OCR_FIELDDBG"); fd3 = (e && e[0] && strcmp(e,"0")!=0) ? 1 : 0; }
                            if (fd3)
                                fprintf(stderr, "FLDCMP cand=%s ck=%d sc=%.4f\\n",
                                        cand, line2_checksum_ok(cand), scf);
                        }
                        if (line2_checksum_ok(cand)) {
                            double sc = scf;""",
"""                    {
                        static int fd4 = -1;
                        if (fd4 < 0) { const char *e = getenv("MRZ_OCR_FIELDDBG"); fd4 = (e && e[0] && strcmp(e,"0")!=0) ? 1 : 0; }
                        if (fd4)
                            fprintf(stderr, "FLDCMP beam_abs=%.4f fld=%.4f greedy_abs=%.4f\\n",
                                    beam_abs, fld_score, greedy_abs);
                    }
""",
]

repl = [
    "",
    "",
    """                        if (line2_checksum_ok(cand)) {
                            double sc = mrz_candidate_logprob(cand, t3);""",
    "",
]

for b, r in zip(blocks, repl):
    if b not in s:
        print("block not found:", b[:60].replace("\n", " "))
    s = s.replace(b, r, 1)

open(P, "w", encoding="utf-8").write(s)
print(f"stripped, {before} -> {len(s)} bytes")
