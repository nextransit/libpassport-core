#!/usr/bin/env python3
"""One-off patch: trace the field-search vs beam competition."""
import sys

P = "src/mrz_ocr_cnn.c"
s = open(P, encoding="utf-8").read()

if "FLDCMP" in s:
    print("already patched")
    sys.exit(0)

old = """                        if (line2_checksum_ok(cand)) {
                            double sc = mrz_candidate_logprob(cand, t3);
                            if (sc > fld_score) {
                                fld_score = sc;
                                memcpy(fld_best, cand, 45);
                                fld_have = 1;
                            }
                        }"""

new = """                        double scf = mrz_candidate_logprob(cand, t3);
                        {
                            static int fd3 = -1;
                            if (fd3 < 0) { const char *e = getenv("MRZ_OCR_FIELDDBG"); fd3 = (e && e[0] && strcmp(e,"0")!=0) ? 1 : 0; }
                            if (fd3)
                                fprintf(stderr, "FLDCMP cand=%s ck=%d sc=%.4f\\n",
                                        cand, line2_checksum_ok(cand), scf);
                        }
                        if (line2_checksum_ok(cand)) {
                            double sc = scf;
                            if (sc > fld_score) {
                                fld_score = sc;
                                memcpy(fld_best, cand, 45);
                                fld_have = 1;
                            }
                        }"""

assert old in s
s = s.replace(old, new, 1)

old2 = """                if (fld_have) {
                    double greedy_abs = mrz_candidate_logprob(pre_beam, t3);
                    double beam_abs = greedy_abs + best_score;
                    if (fld_score > beam_abs) {"""

new2 = """                if (fld_have) {
                    double greedy_abs = mrz_candidate_logprob(pre_beam, t3);
                    double beam_abs = greedy_abs + best_score;
                    {
                        static int fd4 = -1;
                        if (fd4 < 0) { const char *e = getenv("MRZ_OCR_FIELDDBG"); fd4 = (e && e[0] && strcmp(e,"0")!=0) ? 1 : 0; }
                        if (fd4)
                            fprintf(stderr, "FLDCMP beam_abs=%.4f fld=%.4f greedy_abs=%.4f\\n",
                                    beam_abs, fld_score, greedy_abs);
                    }
                    if (fld_score > beam_abs) {"""

assert old2 in s
s = s.replace(old2, new2, 1)

open(P, "w", encoding="utf-8").write(s)
print("patched")
