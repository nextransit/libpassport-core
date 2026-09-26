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
    for (int i = 0; i < 37; ++i)
        EXPECT_EQ(allow[i], i == mrz_ocr_glyph_index('<'));  /* col1 = filler */

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

    fill_allow(allow, 1, 20);  /* sex: M / F / '<' */
    EXPECT(allow[mrz_ocr_glyph_index('M')] == 1);
    EXPECT(allow[mrz_ocr_glyph_index('F')] == 1);
    EXPECT(allow[mrz_ocr_glyph_index('<')] == 1);
    EXPECT(allow[mrz_ocr_glyph_index('K')] == 0);
    EXPECT(allow[mrz_ocr_glyph_index('0')] == 0);

    fill_allow(allow, 1, 14);  /* birth date: digits + '<' */
    EXPECT(allow[mrz_ocr_glyph_index('4')] == 1);
    EXPECT(allow[mrz_ocr_glyph_index('<')] == 1);
    EXPECT(allow[mrz_ocr_glyph_index('K')] == 0);

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

int main(void) {
    setenv("MRZ_OCR_STRICT_ICAO", "1", 1);
    RUN(test_check_digit);
    RUN(test_line2_checksum_ok);
    RUN(test_fill_allow_strict);
    RUN(test_fill_allow_loose);
    RUN(test_top3);
    fprintf(stderr, "\n[TEST] %d total, %d failed\n", g_total, g_fail);
    return g_fail == 0 ? 0 : 1;
}
