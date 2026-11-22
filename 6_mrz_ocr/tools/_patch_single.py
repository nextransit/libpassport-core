#!/usr/bin/env python3
"""One-off patch: single-character repair constrained by the check digits.

WHY
---
The field search and the beam both score a candidate by its LOG
PROBABILITY.  That is the wrong criterion for the case that matters
here: the truth's characters have probability ~0.000-0.002, so a
checksum-consistent reading of the truth always loses to the greedy
reading on probability -- even when the greedy reading is WRONG.

What distinguishes them is the CHECK DIGITS.  A line whose characters
are right satisfies all five check digits; a line that misreads one
character usually does not, even when the misread happens to keep the
own check digit valid (the wrong characters simply weigh to the same
sum).

So this pass adds a criterion the other two do not have:

    for every column, for every alternative class:
        substitute it, DERIVE the check digits of the affected field,
        and keep the result if the WHOLE line then satisfies all five
            check digits;
        among the survivors, take the one with the highest probability.

The derivation is what makes this cheap: the affected field's check
digit is recomputed from the field body, so a substitution that makes
the field consistent is accepted without enumerating digits.

Cost: 44 columns x TOPK alternatives x O(1) check = a few hundred
operations, i.e. far below a millisecond.
"""
import sys

P = "src/mrz_ocr_cnn.c"
s = open(P, encoding="utf-8").read()

if "SINGLE-CHARACTER REPAIR" in s:
    print("already patched")
    sys.exit(0)

# --- 1. the helper -------------------------------------------------
anchor = "/* Composite contribution of a field SEGMENT (body + its own check"

helper = '''/* Which field does column c belong to, and where is its body?
 * Returns -1 for columns that are not inside a check-digit-protected
 * field (nationality, sex and the two tail check digits are not). */
static int mrz_field_of_col(int c) {
    for (int f = 0; f < 4; ++f)
        if (c >= MRZ_FLD_BODY[f][0] && c < MRZ_FLD_BODY[f][1]) return f;
    return -1;
}

/* Recompute a field's own check digit from its body and write it back.
 * The digit is DERIVED, never enumerated -- this is what lets the
 * repair accept a substitution without searching over digits. */
static void mrz_derive_cd(char *l2, int f) {
    char buf[16];
    int start = MRZ_FLD_BODY[f][0], end = MRZ_FLD_BODY[f][1];
    int n = end - start;
    if (n > 15) n = 15;
    memcpy(buf, l2 + start, (size_t)n); buf[n] = 0;
    l2[MRZ_FLD_CD[f]] = (char)('0' + check_digit_of(buf, n));
}

/* SINGLE-CHARACTER REPAIR.
 *
 * For every column, try every alternative class.  After the
 * substitution the affected field's check digit is DERIVED from the
 * field body, and the composite digit is derived as well; the result
 * is kept only if the whole line then satisfies ALL FIVE check digits.
 *
 * This is the criterion the probability-scored searches lack: the
 * truth's characters can have probability ~0.000 and still be the
 * only reading that makes the check digits close.
 *
 * `out` receives the best repair (highest log-probability among the
 * checksum-consistent ones).  Returns 1 when a repair was found. */
static int mrz_single_repair(const top3_t *t3, const char *base,
                             char *out, double *out_score) {
    int found = 0;
    double best = -1e300;
    char bestline[45];

    for (int c = 0; c < 44; ++c) {
        int f = mrz_field_of_col(c);
        if (f < 0) continue;
        for (int j = 0; j < mrz_topk(); ++j) {
            int g = t3[c].cand[j];
            if (g < 0) continue;
            char ch = MRZ_OCR_GLYPHS[g].ch;
            if (ch == base[c]) continue;
            char cand[45];
            memcpy(cand, base, 45);
            cand[c] = ch;
            /* Derive the affected field's check digit, and the
             * composite digit, from the modified bodies. */
            mrz_derive_cd(cand, f);
            if (cand[43] < '0' || cand[43] > '9') {
                char comp[41];
                memcpy(comp, cand, 10);
                memcpy(comp + 10, cand + 13, 7);
                memcpy(comp + 17, cand + 21, 22);
                comp[39] = 0;
                cand[43] = (char)('0' + check_digit_of(comp, 39));
            }
            if (!line2_checksum_ok(cand)) continue;
            double sc = mrz_candidate_logprob(cand, t3);
            if (sc > best) {
                best = sc;
                memcpy(bestline, cand, 45);
                found = 1;
            }
        }
    }
    if (found) {
        memcpy(out, bestline, 45);
        *out_score = best;
    }
    return found;
}

'''

assert anchor in s
s = s.replace(anchor, helper + anchor, 1)

# --- 2. wire it in, competing with the other two searches ---------
old = """                if (nbad > 0) {"""

new = """                if (nbad > 0) {"""

# insert the single-character pass right after the field-search block,
# i.e. just before the beam's column selection
old2 = """            int idxs[44]; float marg[44]; int picks = 0;"""

new2 = """            /* SINGLE-CHARACTER REPAIR, competing with the field
             * search and the beam on the full-line log-probability.
             *
             * It is the only one of the three that can use the check
             * digits as a FILTER rather than a score, which is what
             * lets it find a reading whose characters have near-zero
             * probability but whose check digits close. */
            {
                static int sr_enabled = -1;
                if (sr_enabled < 0) {
                    const char *e = getenv("MRZ_OCR_SINGLE_REPAIR");
                    sr_enabled = !(e && e[0] == '1' && e[1] == 0);
                }
                if (sr_enabled && !fld_have) {
                    char sr[45];
                    double sr_score = -1e300;
                    if (mrz_single_repair(t3, dest, sr, &sr_score)) {
                        if (sr_score > fld_score) {
                            fld_score = sr_score;
                            memcpy(fld_best, sr, 45);
                            fld_have = 1;
                        }
                    }
                }
            }

            int idxs[44]; float marg[44]; int picks = 0;"""

assert old2 in s
s = s.replace(old2, new2, 1)

open(P, "w", encoding="utf-8").write(s)
print("patched")
