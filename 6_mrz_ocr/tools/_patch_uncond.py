#!/usr/bin/env python3
"""One-off patch: make the field search UNCONDITIONAL.

Why
---
The field search only fired when a field's OWN check digit failed.  That
is too narrow: a field can be misread and still satisfy its own check
digit (the wrong characters happen to weigh to the same sum), and the
reading is then WRONG but "valid".  Measured on the real corpus, the
line-2 failures of 2026-9-29-L / -W / 2026-9-30-AJ are exactly of that
kind: the personal number is misread in three to five columns and the
check digit still verifies.

The fix is to solve every field unconditionally and let the FULL-line
log-probability decide.  The field search can only produce a line that
satisfies all five check digits, and it only wins when it is more
likely than the incumbent, so the change is safe by construction:
the decoder can never become worse, only occasionally better.
"""
import sys

P = "src/mrz_ocr_cnn.c"
s = open(P, encoding="utf-8").read()

if "UNCONDITIONAL FIELD SEARCH" in s:
    print("already patched")
    sys.exit(0)

old = """                for (int f = 0; f < 4; ++f) {
                    nsols[f] = 0;
                    if (!mrz_field_cd_bad(dest, f)) continue;
                    nsols[f] = mrz_field_solve(t3, f, sols[f], 128);
                    if (nsols[f] > 0) nbad++;
                }"""

new = """                for (int f = 0; f < 4; ++f) {
                    nsols[f] = 0;
                    /* UNCONDITIONAL FIELD SEARCH.
                     *
                     * Solving every field (not only the ones whose
                     * check digit fails) is what catches a field that
                     * is misread yet still satisfies its own check
                     * digit -- the wrong characters simply happen to
                     * weigh to the same sum.  The result is still safe:
                     * the candidate must satisfy ALL five check digits
                     * and must be MORE LIKELY than the incumbent, so
                     * the decoder can only get better.
                     *
                     * The cost is bounded: four fields x 100 DP states
                     * x the candidate width. */
                    nsols[f] = mrz_field_solve(t3, f, sols[f], 128);
                    if (nsols[f] > 0) nbad++;
                }"""

assert old in s
s = s.replace(old, new, 1)

# The joint DP now covers ALL four fields, so the composite constraint
# has to be applied to the SUM of the four segments -- the fixed_comp
# term (the fields that are not repaired) is now empty.
old2 = """                    int fixed_comp = 0;
                    for (int f = 0; f < 4; ++f)
                        if (nsols[f] == 0)
                            fixed_comp = (fixed_comp +
                                mrz_comp_seg(dest, MRZ_FLD_BODY[f][0],
                                            MRZ_FLD_BODY[f][1])) % 10;"""
new2 = """                    int fixed_comp = 0;"""
assert old2 in s
s = s.replace(old2, new2, 1)

open(P, "w", encoding="utf-8").write(s)
print("patched")
