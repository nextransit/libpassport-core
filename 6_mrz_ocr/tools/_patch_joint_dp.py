#!/usr/bin/env python3
"""One-off patch: replace the sequential field repair with a JOINT DP.

The sequential version repaired one field at a time and re-validated
all five check digits after each repair, so a line with TWO bad fields
could never be repaired (the first repair leaves the second field bad,
so every candidate was rejected).

The joint version treats the fields as coupled through the composite
digit: the DP state is the accumulated composite residue (10 states).
"""
import sys

P = "src/mrz_ocr_cnn.c"

s = open(P, encoding="utf-8").read()

if "JOINT DP OVER THE FAILING FIELDS" in s:
    print("already patched")
    sys.exit(0)

i = s.index("                static mrz_field_sol sols[128];")
j = s.index("            int idxs[44]; float marg[44]; int picks = 0;")

new = '''                /* JOINT DP OVER THE FAILING FIELDS.
                 *
                 * The fields are coupled only through the composite
                 * digit, which covers all four field segments.  So the
                 * joint problem is a DP whose state is the accumulated
                 * composite residue -- 10 states.  For each failing
                 * field the exact per-field solutions (see
                 * mrz_field_solve) are combined, and only the
                 * combinations whose composite residue matches the
                 * printed composite digit are kept.
                 *
                 * The composite digit at column 43 is the CONSTRAINT,
                 * not a variable: it is what the document says, and the
                 * field bodies have to agree with it.  When it is not a
                 * digit there is no constraint and the derived value is
                 * written instead.
                 *
                 * The check digits are DERIVED from the field bodies,
                 * never enumerated, so the decoder only has to win on
                 * the FIELD, not on the individual character.  That is
                 * what lets it repair a cell the classifier is
                 * confidently wrong about -- which the margin-based
                 * beam can never do, because a confident error sorts
                 * to the very end of the beam's candidate list. */
                static mrz_field_sol sols[4][128];
                int nsols[4];
                int nbad = 0;
                for (int f = 0; f < 4; ++f) {
                    nsols[f] = 0;
                    if (!mrz_field_cd_bad(dest, f)) continue;
                    nsols[f] = mrz_field_solve(t3, f, sols[f], 128);
                    if (nsols[f] > 0) nbad++;
                }
                if (nbad > 0) {
                    /* The composite target: the printed digit. */
                    int target = (dest[43] >= '0' && dest[43] <= '9')
                               ? (dest[43] - '0') : -1;
                    /* The composite residue contributed by the fields
                     * that are NOT being repaired. */
                    int fixed_comp = 0;
                    for (int f = 0; f < 4; ++f)
                        if (nsols[f] == 0)
                            fixed_comp = (fixed_comp +
                                mrz_comp_seg(dest, MRZ_FLD_BODY[f][0],
                                            MRZ_FLD_BODY[f][1])) % 10;
                    /* DP over the accumulated composite residue.
                     * state[r] = best (score, choice per field) with
                     * accumulated residue r. */
                    double dscore[10];
                    int    dpick[10][4];
                    for (int r = 0; r < 10; ++r) dscore[r] = -1e300;
                    dscore[0] = 0.0;
                    for (int f = 0; f < 4; ++f) dpick[0][f] = -1;
                    for (int f = 0; f < 4; ++f) {
                        if (nsols[f] == 0) continue;
                        double ndscore[10];
                        int    ndpick[10][4];
                        for (int r = 0; r < 10; ++r) {
                            ndscore[r] = -1e300;
                            for (int q = 0; q < 4; ++q) ndpick[r][q] = -1;
                        }
                        for (int r = 0; r < 10; ++r) {
                            if (dscore[r] <= -1e299) continue;
                            for (int i = 0; i < nsols[f]; ++i) {
                                int nr = (r + sols[f][i].comp) % 10;
                                double sc = dscore[r] + sols[f][i].score;
                                if (sc > ndscore[nr]) {
                                    ndscore[nr] = sc;
                                    for (int q = 0; q < 4; ++q) ndpick[nr][q] = dpick[r][q];
                                    ndpick[nr][f] = i;
                                }
                            }
                        }
                        for (int r = 0; r < 10; ++r) {
                            dscore[r] = ndscore[r];
                            for (int q = 0; q < 4; ++q) dpick[r][q] = ndpick[r][q];
                        }
                    }
                    /* Pick the state that satisfies the composite. */
                    int best_r = -1;
                    double best = -1e300;
                    for (int r = 0; r < 10; ++r) {
                        if (dscore[r] <= -1e299) continue;
                        if (target >= 0 && (r + fixed_comp) % 10 != target) continue;
                        if (dscore[r] > best) { best = dscore[r]; best_r = r; }
                    }
                    if (best_r >= 0) {
                        char cand[45];
                        memcpy(cand, dest, 45);
                        for (int f = 0; f < 4; ++f) {
                            int i = dpick[best_r][f];
                            if (i < 0) continue;
                            int fstart = MRZ_FLD_BODY[f][0];
                            int fn = MRZ_FLD_BODY[f][1] - fstart;
                            memcpy(cand + fstart, sols[f][i].body, (size_t)fn);
                            cand[MRZ_FLD_CD[f]] = (char)('0' + sols[f][i].digit);
                        }
                        if (target < 0) {
                            /* No printed composite digit: derive it. */
                            char comp[41];
                            memcpy(comp, cand, 10);
                            memcpy(comp + 10, cand + 13, 7);
                            memcpy(comp + 17, cand + 21, 22);
                            comp[39] = 0;
                            cand[43] = (char)('0' + check_digit_of(comp, 39));
                        }
                        if (line2_checksum_ok(cand)) {
                            double sc = mrz_candidate_logprob(cand, t3);
                            if (sc > fld_score) {
                                fld_score = sc;
                                memcpy(fld_best, cand, 45);
                                fld_have = 1;
                            }
                        }
                    }
                }
            }

'''

s = s[:i] + new + s[j:]
open(P, "w", encoding="utf-8").write(s)
print("patched OK")
