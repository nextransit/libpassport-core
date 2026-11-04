/* Unit tests for the CNN pipeline's post-processing internals.
 *
 * Includes src/mrz_ocr_cnn.c directly so the static helpers
 * (check_digit_of, fill_allow, top3_of, line2_checksum_ok) can be
 * exercised. The mrz_ocr archive member is not pulled into the link
 * because every symbol it provides is already defined here. */
#include "../src/mrz_ocr_cnn.c"

#include <stdio.h>
#include <stdlib.h>

static int g_fail = 0, g_total = 0;
#define EXPECT(c) do {                                                  \
    g_total++;                                                          \
    if (!(c)) {                                                         \
        g_fail++;                                                       \
        fprintf(stderr, "[FAIL] %s:%d : %s\n", __FILE__, __LINE__, #c); \
    }                                                                   \
} while (0)
#define EXPECT_EQ(a, b) do {                                            \
    long _a = (long)(a), _b = (long)(b);                                \
    g_total++;                                                          \
    if (_a != _b) {                                                     \
        g_fail++;                                                       \
        fprintf(stderr, "[FAIL] %s:%d : got %ld want %ld\n",            \
                __FILE__, __LINE__, _a, _b);                            \
    }                                                                   \
} while (0)
#define RUN(fn) do { fprintf(stderr, "%s\n", #fn); fn(); } while (0)

static const char *ERIKSSON_L2 =
    "L898902C36UTO6908061F9406236ZE184226B<<<<<18";

static void test_check_digit(void) {
    /* ICAO example: passport number L898902C3 -> check digit 6. */
    EXPECT_EQ(check_digit_of("L898902C3", 9), 6);
    /* Birth date 690806 -> 1; expiry 940623 -> 6;
     * personal ZE184226B<<<<< -> 1; composite over 43 chars -> 8. */
    EXPECT_EQ(check_digit_of("690806", 6), 1);
    EXPECT_EQ(check_digit_of("940623", 6), 6);
    EXPECT_EQ(check_digit_of("ZE184226B<<<<<", 14), 1);
    EXPECT_EQ(check_digit_of(ERIKSSON_L2, 43), 8);
    /* '<' counts as 0. */
    EXPECT_EQ(check_digit_of("<<<", 3), 0);
}

static void test_line2_checksum_ok(void) {
    char buf[64];
    strcpy(buf, ERIKSSON_L2);
    EXPECT(line2_checksum_ok(buf) == 1);
    /* The five check-digit positions must each be enforced. */
    const int check_cols[] = {9, 19, 27, 42, 43};
    for (int ci = 0; ci < 5; ++ci) {
        int i = check_cols[ci];
        char orig = buf[i];
        buf[i] = (orig == '5') ? '7' : '5';
        EXPECT(line2_checksum_ok(buf) == 0);
        buf[i] = orig;
    }
    /* Field corruption is caught whenever the weighted delta is not a
     * multiple of 10 (ICAO weights are mod 10, so e.g. F->5 at a
     * weight-1 column is undetectable BY DESIGN). */
    buf[0] = '8';   /* L(21)->8: delta 13*7 */
    EXPECT(line2_checksum_ok(buf) == 0);
    buf[0] = 'L';
    buf[15] = '3';  /* birth date 690806 -> 693806 */
    EXPECT(line2_checksum_ok(buf) == 0);
    buf[15] = '0';
}

static void test_fill_allow_strict(void) {
    int allow[37];
    setenv("MRZ_OCR_STRICT_ICAO", "1", 1);

    fill_allow(allow, 0, 1);
    /* col1: A-Z + '<' (relaxed for non-standard doc subtypes like PD/PM) */
    for (int i = 0; i < 37; ++i)
        EXPECT_EQ(allow[i], (i >= mrz_ocr_glyph_index('A') &&
                             i <  mrz_ocr_glyph_index('A') + 26) ||
                            i == mrz_ocr_glyph_index('<'));

    fill_allow(allow, 0, 3);   /* issuer: alpha only */
    EXPECT(allow[mrz_ocr_glyph_index('D')] == 1);
    EXPECT(allow[mrz_ocr_glyph_index('0')] == 0);

    fill_allow(allow, 0, 20);  /* name field: alpha + '<' */
    EXPECT(allow[mrz_ocr_glyph_index('K')] == 1);
    EXPECT(allow[mrz_ocr_glyph_index('<')] == 1);
    EXPECT(allow[mrz_ocr_glyph_index('7')] == 0);

    fill_allow(allow, 1, 19);  /* check digit: numeric only */
    EXPECT(allow[mrz_ocr_glyph_index('5')] == 1);
    EXPECT(allow[mrz_ocr_glyph_index('A')] == 0);
    EXPECT(allow[mrz_ocr_glyph_index('<')] == 0);

    fill_allow(allow, 1, 20);  /* sex: alnum + '<' (relaxed for specimen) */
    EXPECT(allow[mrz_ocr_glyph_index('M')] == 1);
    EXPECT(allow[mrz_ocr_glyph_index('F')] == 1);
    EXPECT(allow[mrz_ocr_glyph_index('<')] == 1);
    EXPECT(allow[mrz_ocr_glyph_index('K')] == 1);
    EXPECT(allow[mrz_ocr_glyph_index('0')] == 1);

    fill_allow(allow, 1, 14);  /* birth date: digits + '<' + A-Z (relaxed) */
    EXPECT(allow[mrz_ocr_glyph_index('4')] == 1);
    EXPECT(allow[mrz_ocr_glyph_index('<')] == 1);
    EXPECT(allow[mrz_ocr_glyph_index('K')] == 1);

    fill_allow(allow, 1, 11);  /* nationality: alpha + '<' */
    EXPECT(allow[mrz_ocr_glyph_index('U')] == 1);
    EXPECT(allow[mrz_ocr_glyph_index('<')] == 1);
    EXPECT(allow[mrz_ocr_glyph_index('0')] == 0);

    fill_allow(allow, 1, 30);  /* personal number: alnum + '<' */
    EXPECT(allow[mrz_ocr_glyph_index('B')] == 1);
    EXPECT(allow[mrz_ocr_glyph_index('3')] == 1);
    EXPECT(allow[mrz_ocr_glyph_index('<')] == 1);
}

static void test_fill_allow_loose(void) {
    int allow[37];
    setenv("MRZ_OCR_STRICT_ICAO", "0", 1);
    fill_allow(allow, 0, 20);  /* legacy: name field unrestricted */
    EXPECT(allow[mrz_ocr_glyph_index('7')] == 1);
    EXPECT(allow[mrz_ocr_glyph_index('K')] == 1);
    fill_allow(allow, 1, 14);  /* legacy: dates unrestricted */
    EXPECT(allow[mrz_ocr_glyph_index('K')] == 1);
    fill_allow(allow, 0, 1);
    EXPECT(allow[mrz_ocr_glyph_index('<')] == 1);
    EXPECT(allow[mrz_ocr_glyph_index('A')] == 0);
    setenv("MRZ_OCR_STRICT_ICAO", "1", 1);
}

static void test_top3(void) {
    float probs[37];
    for (int i = 0; i < 37; ++i) probs[i] = 0.0f;
    probs[5] = 0.5f; probs[20] = 0.3f; probs[36] = 0.2f;
    int allow[37];
    for (int i = 0; i < 37; ++i) allow[i] = 1;
    top3_t t;
    top3_of(probs, allow, &t);
    EXPECT_EQ(t.cand[0], 5);
    EXPECT_EQ(t.cand[1], 20);
    EXPECT_EQ(t.cand[2], 36);
    /* mask out the winner -> it must not appear */
    allow[5] = 0;
    top3_of(probs, allow, &t);
    EXPECT_EQ(t.cand[0], 20);
    /* everything masked -> fallback '<' */
    for (int i = 0; i < 37; ++i) allow[i] = 0;
    top3_of(probs, allow, &t);
    EXPECT_EQ(t.cand[0], mrz_ocr_glyph_index('<'));
}

/* The checksum beam must rank candidates by likelihood, not by the
 * order in which the search happens to reach them.  Two candidates can
 * both satisfy all five ICAO check digits; only their probabilities
 * tell them apart.  The regression this guards against is the original
 * implementation, which accepted the FIRST checksum-valid row it
 * found while walking the least-confident columns outward and so
 * locked in a 0.02-probability glyph when a 0.14-probability glyph
 * elsewhere would have satisfied the same digit. */
static void test_candidate_logprob(void) {
    top3_t t3[44];
    float probs[37];
    for (int c = 0; c < 44; ++c) {
        for (int i = 0; i < 37; ++i) probs[i] = 0.0f;
        /* every column: 'A' is the top candidate */
        probs[mrz_ocr_glyph_index('A')] = 0.5f;
        probs[mrz_ocr_glyph_index('B')] = 0.3f;
        probs[mrz_ocr_glyph_index('C')] = 0.2f;
        int allow[37];
        for (int i = 0; i < 37; ++i) allow[i] = 1;
        top3_of(probs, allow, &t3[c]);
    }
    char hi[45], lo[45];
    memset(hi, 'A', 44); hi[44] = 0;
    memcpy(lo, hi, 45);
    /* column 10 becomes 'B' (more likely) in `hi` and 'C' (less likely)
     * in `lo` */
    hi[10] = 'B';
    lo[10] = 'C';
    double s_hi = mrz_candidate_logprob(hi, t3);
    double s_lo = mrz_candidate_logprob(lo, t3);
    /* log(0.3) > log(0.2) -> the 'B' row must score higher */
    EXPECT(s_hi > s_lo);
    /* a character that is not among the candidates scores as the
     * epsilon floor, so it must lose against any real candidate */
    char unknown[45];
    memcpy(unknown, hi, 45);
    unknown[10] = 'Z';
    EXPECT(mrz_candidate_logprob(unknown, t3) < s_hi);
}


/* ---- deep checksum beam -------------------------------------------
 *
 * The beam is the part of the decode that turns "the network is not
 * sure" into "the network is not sure AND the checksum agrees".
 * Two properties matter and both are regression-tested here:
 *
 *  1. it must FIND a checksum-valid reading even when several
 *     positions have to be corrected at once (the depth-2 search
 *     could not do that -- measured on the 41-photo corpus, three of
 *     the four residual failures need three or four corrections);
 *  2. it must not silently degrade into returning something that
 *     fails the checksum: when no valid reading exists the caller has
 *     to be able to tell.
 */
static void make_top3(top3_t *t3, const char *s) {
    for (int c = 0; c < 44; ++c) {
        for (int k = 0; k < TOPK; ++k) {
            t3[c].cand[k] = mrz_ocr_glyph_index(s[c]);
            t3[c].p[k] = (k == 0) ? 1.0f : 0.0f;
        }
    }
}

static void test_beam_finds_multiposition_fix(void) {
    /* Build a line that is wrong in FOUR positions at once and whose
     * corrections make it checksum-valid.  The greedy reading must
     * fail the checksum; the beam must recover the valid one. */
    char good[45];
    strcpy(good, ERIKSSON_L2);

    /* Corrupt four positions, each in a way the syntax mask allows:
     * col 3 '9'->'8' (doc no), col 17 '8'->'3' (birth), col 30 '2'->
     * '7' (personal no), col 40 '<'->'B'.  The check digits are left
     * alone, so the corrupt line does NOT satisfy the checksum. */
    char bad[45];
    strcpy(bad, good);
    bad[3]  = '8';
    bad[17] = '3';
    bad[30] = '7';
    bad[40] = 'B';
    EXPECT(line2_checksum_ok(bad) == 0);

    top3_t t3[44];
    make_top3(t3, bad);
    /* Give every corrupted column its correct value as the SECOND
     * candidate, with a slightly lower probability. */
    const int cols[4] = {3, 17, 30, 40};
    for (int i = 0; i < 4; ++i) {
        int c = cols[i];
        t3[c].cand[1] = mrz_ocr_glyph_index(good[c]);
        t3[c].p[1] = 0.4f;
        t3[c].p[0] = 0.6f;
        t3[c].cand[2] = t3[c].cand[1];
        t3[c].p[2] = 0.0f;
    }
    /* The candidate columns the beam may touch: the four corrupted
     * ones, made the least confident. */
    int idxs[44]; float marg[44]; int picks = 0;
    for (int c = 0; c < 44; ++c) {
        if (t3[c].cand[0] != t3[c].cand[1]) {
            idxs[picks] = c; marg[picks] = t3[c].p[0] - t3[c].p[1]; picks++;
        }
    }
    for (int a = 1; a < picks; ++a) {
        int ki = idxs[a]; float km = marg[a];
        int j = a - 1;
        while (j >= 0 && marg[j] > km) { idxs[j+1] = idxs[j]; marg[j+1] = marg[j]; j--; }
        idxs[j+1] = ki; marg[j+1] = km;
    }
    char best[45];
    memcpy(best, bad, 45);
    double score = -1e300;
    char alt[45]; alt[0] = 0;
    double alt_score = -1e300;
    mrz_beam_search(bad, t3, idxs, picks, best, &score, alt, &alt_score, 0.0);
    EXPECT(score > -1e299);
    EXPECT(strcmp(best, good) == 0);
}

static void test_beam_reports_no_solution(void) {
    /* No checksum-valid reading exists among the candidates: the beam
     * must say so (leave the score at the sentinel) instead of
     * returning an invalid line as if it had found one. */
    char bad[45];
    strcpy(bad, ERIKSSON_L2);
    bad[9] = '0';                    /* break the doc-number check digit */
    top3_t t3[44];
    make_top3(t3, bad);
    int idxs[44]; float marg[44]; int picks = 0;
    for (int c = 0; c < 44; ++c) {
        if (t3[c].cand[0] != t3[c].cand[1]) {
            idxs[picks] = c; marg[picks] = t3[c].p[0] - t3[c].p[1]; picks++;
        }
    }
    char best[45];
    memcpy(best, bad, 45);
    double score = -1e300;
    char alt[45]; alt[0] = 0;
    double alt_score = -1e300;
    mrz_beam_search(bad, t3, idxs, picks, best, &score, alt, &alt_score, 0.0);
    EXPECT(score <= -1e299);
    EXPECT(alt_score <= -1e299);
}


/* ---- ambiguity reporting -----------------------------------------
 *
 * The decoder has to tell the caller when a checksum-valid reading is
 * not the only plausible one.  A wrong-but-checksum-valid line is
 * indistinguishable from a correct one downstream, so the flag is the
 * only way a caller can decide to ask for a retake.
 */
static void test_alt_diff_count(void) {
    char a[45], b[45];
    strcpy(a, ERIKSSON_L2);
    strcpy(b, ERIKSSON_L2);
    EXPECT_EQ(mrz_alt_diff_count(a, b), 0);
    b[3] = 'X';
    EXPECT_EQ(mrz_alt_diff_count(a, b), 1);
    b[3] = ERIKSSON_L2[3];
    b[10] = 'X';
    EXPECT_EQ(mrz_alt_diff_count(a, b), 1);
}

static void test_alt_materiality(void) {
    char a[45], b[45];
    strcpy(a, ERIKSSON_L2);
    strcpy(b, ERIKSSON_L2);
    /* A difference on a check-digit column is not a risk: those columns
     * are determined by the checksum. */
    b[9] = '0';
    EXPECT_EQ(mrz_alt_is_material(a, b), 0);
    b[9] = ERIKSSON_L2[9];
    /* Nor on the nationality or the sex column. */
    b[10] = 'X';
    b[20] = 'X';
    EXPECT_EQ(mrz_alt_is_material(a, b), 0);
    b[10] = ERIKSSON_L2[10];
    b[20] = ERIKSSON_L2[20];
    /* A difference on the passport number IS material: the check digit
     * covers it, and both readings satisfied it. */
    b[3] = 'X';
    EXPECT_EQ(mrz_alt_is_material(a, b), 1);
}

/* Structural veto + nationality advisory (mrz_ocr.h).
 *
 * The check digits are satisfied by these strings; the point of the
 * test is that the veto fires anyway.  Every case below was taken from
 * the assets corpus, where the beam closed all five digits over noise. */
static void test_struct_veto(void) {
    char buf[64];

    /* A real device reading. */
    strcpy(buf, "E0132B7855CHN0110141M3105179GB682050108G0B88");
    EXPECT_EQ(mrz_ocr_line2_struct_ok(buf), 1);
    EXPECT_EQ(mrz_ocr_line2_nat_suspect(buf), 0);

    /* The ICAO specimen: valid.  UTO is the code ICAO itself uses in the
     * specimen documents ("P<UTOERIKSSON<<ANNA<MARIA"), and it is what
     * most MRZ parser libraries emit for "unknown".  Flagging it as an
     * impossible nationality was wrong: a document that a standards
     * body prints is by definition a document that exists.  It is now
     * in the accepted table. */
    strcpy(buf, ERIKSSON_L2);
    EXPECT_EQ(mrz_ocr_line2_struct_ok(buf), 1);
    EXPECT_EQ(mrz_ocr_line2_nat_suspect(buf), 0);

    /* Month 91 in the birth date (year 01, month 91). */
    strcpy(buf, "E0132B7855CHN0191141M3105179GB682050108G0B88");
    EXPECT_EQ(mrz_ocr_line2_struct_ok(buf), 0);

    /* Day 32 (year 01, month 32). */
    strcpy(buf, "E0132B7855CHN0132141M3105179GB682050108G0B88");
    EXPECT_EQ(mrz_ocr_line2_struct_ok(buf), 0);

    /* Expiry (1988) before birth (1990). */
    strcpy(buf, "E0132B7855CHN9001011M8801019GB682050108G0B88");
    EXPECT_EQ(mrz_ocr_line2_struct_ok(buf), 0);

    /* Anything that is not 44 characters is never structurally OK. */
    EXPECT_EQ(mrz_ocr_line2_struct_ok("E0132B7855CHN0110141M3105179GB68"), 0);
    EXPECT_EQ(mrz_ocr_line2_struct_ok(NULL), 0);

    /* A nationality no state issues: the beam closed the checksum on
     * this one on the assets corpus. */
    strcpy(buf, "ZC39917886HOL8101028M2909185<<<<<<<<<<<<<<04");
    EXPECT_EQ(mrz_ocr_line2_struct_ok(buf), 1);
    EXPECT_EQ(mrz_ocr_line2_nat_suspect(buf), 1);

    /* A field containing '<' is never flagged: Germany prints "D<<" and
     * several states print a partial code rather than the full alpha-3. */
    strcpy(buf, "L898902C36D<<6908061F9406236ZE184226B<<<<<18");
    EXPECT_EQ(mrz_ocr_line2_nat_suspect(buf), 0);
}

/* Cross-line consistency: issuing state (line 1 cols 2..4) vs
 * nationality (line 2 cols 10..12).
 *
 * This is the check that caught the single-character line-1 errors on
 * the 41-photo phone corpus -- the confidence did not (one wrong line 1
 * still scored 98) and neither did the line-2 ambiguity flag.
 *
 * The four failing cases below are the actual readings that the check
 * caught on that corpus:
 *   29-U  line1 read "HUG"  while line 2 said CHN
 *   29-W  line1 read "CHY"  while line 2 said CHN
 *   30-F  line1 read "CNN"  while line 2 said CHN
 *   30-H  line1 read "ATA"  while line 2 said CHN
 */
static void test_country_crosscheck(void) {
    const char *l2 = "E0132B7855CHN0110141M3105179GB682050108G0B88";
    char l1[64];

    /* Agrees -> no mismatch. */
    strcpy(l1, "P<CHNYUN<<YAOYU<<<<<<<<<<<<<<<<<<<<<<<<<<<<");
    EXPECT_EQ(mrz_ocr_line1_country_mismatch(l1, l2), 0);

    /* The four real cases: one wrong character in line 1. */
    strcpy(l1, "P<HUG<AC<<P<H<HH<<<<<<<<<<<<<<<<<<<<<<<<<<<<");
    EXPECT_EQ(mrz_ocr_line1_country_mismatch(l1, l2), 1);
    strcpy(l1, "P<CHYYHN<<PUYING<<<<<<<<<<<<<<<<<<<<<<<<<<<<");
    EXPECT_EQ(mrz_ocr_line1_country_mismatch(l1, l2), 1);
    strcpy(l1, "P<CNNIAN<<ZHK<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<");
    EXPECT_EQ(mrz_ocr_line1_country_mismatch(l1, l2), 1);
    strcpy(l1, "P<ATAAMJAM<<ZHEA<<<<<<<<<<<<<<<<<<<<<<<<<<<<");
    EXPECT_EQ(mrz_ocr_line1_country_mismatch(l1, l2), 1);

    /* A field containing '<' means the comparison does not apply:
     * Germany prints "D<<", a refugee document may legitimately differ. */
    strcpy(l1, "P<D<<YUN<<YAOYU<<<<<<<<<<<<<<<<<<<<<<<<<<<<<");
    EXPECT_EQ(mrz_ocr_line1_country_mismatch(l1, l2), 0);
    strcpy(l1, "P<CHNYUN<<YAOYU<<<<<<<<<<<<<<<<<<<<<<<<<<<<");
    EXPECT_EQ(mrz_ocr_line1_country_mismatch(
        l1, "E0132B7855D<<0110141M3105179GB682050108G0B88"), 0);

    /* Wrong length or NULL: the check must not fire. */
    strcpy(l1, "P<CHNYUN<<YAOYU<<<<<<<<<<<<<<<<<<<<<<<<<<<<");
    EXPECT_EQ(mrz_ocr_line1_country_mismatch(l1, "E0132B7855"), 0);
    EXPECT_EQ(mrz_ocr_line1_country_mismatch("P<CH", l2), 0);
    EXPECT_EQ(mrz_ocr_line1_country_mismatch(NULL, l2), 0);
    EXPECT_EQ(mrz_ocr_line1_country_mismatch(l1, NULL), 0);
}

/* The issuing state in line 1 gets the same content test the
 * nationality in line 2 has always had.  Line 1 carries no check
 * digits, so it is the row that most needs one. */
static void test_issuer_code_check(void) {
    char l1[64];

    /* A real alpha-3 code -> not suspicious. */
    strcpy(l1, "P<CHNYUN<<YAOYU<<<<<<<<<<<<<<<<<<<<<<<<<<<<");
    EXPECT_EQ(mrz_ocr_line1_issuer_suspect(l1), 0);
    strcpy(l1, "P<AUSSMITH<<JOHN<<<<<<<<<<<<<<<<<<<<<<<<<<<<<");
    EXPECT_EQ(mrz_ocr_line1_issuer_suspect(l1), 0);

    /* The ICAO special codes are legitimate. */
    strcpy(l1, "P<XXXYUN<<YAOYU<<<<<<<<<<<<<<<<<<<<<<<<<<<<");
    EXPECT_EQ(mrz_ocr_line1_issuer_suspect(l1), 0);
    strcpy(l1, "P<UNOYUN<<YAOYU<<<<<<<<<<<<<<<<<<<<<<<<<<<<");
    EXPECT_EQ(mrz_ocr_line1_issuer_suspect(l1), 0);
    /* These were MISSING from the table until this test existed, and a
     * missing entry means a real document gets flagged as impossible:
     * Kosovo, the EU laissez-passer, an Interpol travel document, a
     * Sovereign Military Order of Malta passport, a UN organisation
     * passport, and the British nationality categories. */
    strcpy(l1, "P<RKSYUN<<YAOYU<<<<<<<<<<<<<<<<<<<<<<<<<<<<");
    EXPECT_EQ(mrz_ocr_line1_issuer_suspect(l1), 0);
    strcpy(l1, "P<EUEYUN<<YAOYU<<<<<<<<<<<<<<<<<<<<<<<<<<<<");
    EXPECT_EQ(mrz_ocr_line1_issuer_suspect(l1), 0);
    strcpy(l1, "P<XPOYUN<<YAOYU<<<<<<<<<<<<<<<<<<<<<<<<<<<<");
    EXPECT_EQ(mrz_ocr_line1_issuer_suspect(l1), 0);
    strcpy(l1, "P<XOMYUN<<YAOYU<<<<<<<<<<<<<<<<<<<<<<<<<<<<");
    EXPECT_EQ(mrz_ocr_line1_issuer_suspect(l1), 0);
    strcpy(l1, "P<UNAYUN<<YAOYU<<<<<<<<<<<<<<<<<<<<<<<<<<<<");
    EXPECT_EQ(mrz_ocr_line1_issuer_suspect(l1), 0);
    strcpy(l1, "P<UNKYUN<<YAOYU<<<<<<<<<<<<<<<<<<<<<<<<<<<<");
    EXPECT_EQ(mrz_ocr_line1_issuer_suspect(l1), 0);
    strcpy(l1, "P<GBNYUN<<YAOYU<<<<<<<<<<<<<<<<<<<<<<<<<<<<");
    EXPECT_EQ(mrz_ocr_line1_issuer_suspect(l1), 0);
    strcpy(l1, "P<GBOYUN<<YAOYU<<<<<<<<<<<<<<<<<<<<<<<<<<<<");
    EXPECT_EQ(mrz_ocr_line1_issuer_suspect(l1), 0);
    strcpy(l1, "P<GBPYUN<<YAOYU<<<<<<<<<<<<<<<<<<<<<<<<<<<<");
    EXPECT_EQ(mrz_ocr_line1_issuer_suspect(l1), 0);
    strcpy(l1, "P<GBSYUN<<YAOYU<<<<<<<<<<<<<<<<<<<<<<<<<<<<");
    EXPECT_EQ(mrz_ocr_line1_issuer_suspect(l1), 0);
    strcpy(l1, "P<GBDYUN<<YAOYU<<<<<<<<<<<<<<<<<<<<<<<<<<<<");
    EXPECT_EQ(mrz_ocr_line1_issuer_suspect(l1), 0);
    strcpy(l1, "P<XKKYUN<<YAOYU<<<<<<<<<<<<<<<<<<<<<<<<<<<<");
    EXPECT_EQ(mrz_ocr_line1_issuer_suspect(l1), 0);
    strcpy(l1, "P<XKXYUN<<YAOYU<<<<<<<<<<<<<<<<<<<<<<<<<<<<");
    EXPECT_EQ(mrz_ocr_line1_issuer_suspect(l1), 0);
    /* The nationality check must accept the same codes. */
    EXPECT_EQ(mrz_ocr_line2_nat_suspect(
        "E0132B7855RKS0110141M3105179GB682050108G0B88"), 0);
    EXPECT_EQ(mrz_ocr_line2_nat_suspect(
        "E0132B7855XXB0110141M3105179GB682050108G0B88"), 0);
    EXPECT_EQ(mrz_ocr_line2_nat_suspect(
        "E0132B7855UTO0110141M3105179GB682050108G0B88"), 0);

    /* The four fabricated codes the recogniser actually produced on the
     * 41-photo corpus -- each one is a line 1 that was wrong. */
    strcpy(l1, "P<HUG<AC<<P<H<HH<<<<<<<<<<<<<<<<<<<<<<<<<<<<");
    EXPECT_EQ(mrz_ocr_line1_issuer_suspect(l1), 1);
    strcpy(l1, "P<CHYYHN<<PUYING<<<<<<<<<<<<<<<<<<<<<<<<<<<<");
    EXPECT_EQ(mrz_ocr_line1_issuer_suspect(l1), 1);
    strcpy(l1, "P<CNNIAN<<ZHK<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<");
    EXPECT_EQ(mrz_ocr_line1_issuer_suspect(l1), 1);
    strcpy(l1, "P<ATAAMJAM<<ZHEA<<<<<<<<<<<<<<<<<<<<<<<<<<<<");
    EXPECT_EQ(mrz_ocr_line1_issuer_suspect(l1), 1);

    /* A field containing '<' means the check does not apply: Germany
     * prints "D<<" and several states print a partial code. */
    strcpy(l1, "P<D<<YUN<<YAOYU<<<<<<<<<<<<<<<<<<<<<<<<<<<<<");
    EXPECT_EQ(mrz_ocr_line1_issuer_suspect(l1), 0);
    strcpy(l1, "P<C<<YUN<<YAOYU<<<<<<<<<<<<<<<<<<<<<<<<<<<<<");
    EXPECT_EQ(mrz_ocr_line1_issuer_suspect(l1), 0);

    /* Wrong length or NULL: must not fire. */
    EXPECT_EQ(mrz_ocr_line1_issuer_suspect("P<CHN"), 0);
    EXPECT_EQ(mrz_ocr_line1_issuer_suspect(NULL), 0);
}

int main(void) {
    setenv("MRZ_OCR_STRICT_ICAO", "1", 1);
    RUN(test_check_digit);
    RUN(test_line2_checksum_ok);
    RUN(test_fill_allow_strict);
    RUN(test_fill_allow_loose);
    RUN(test_top3);
    RUN(test_candidate_logprob);
    RUN(test_beam_finds_multiposition_fix);
    RUN(test_beam_reports_no_solution);
    RUN(test_alt_diff_count);
    RUN(test_alt_materiality);
    RUN(test_struct_veto);
    RUN(test_country_crosscheck);
    RUN(test_issuer_code_check);
    fprintf(stderr, "\n[TEST] %d total, %d failed\n", g_total, g_fail);
    return g_fail == 0 ? 0 : 1;
}
