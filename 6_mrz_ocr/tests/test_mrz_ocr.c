/* Unit tests for the OCR-B template bank + simple sanity check
 * on the OCR pipeline (run on the canonical ERIKSSON synthetic image). */
#include "mrz_ocr.h"
#include "face.h"
#include "template.h"

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

static int g_fail = 0, g_total = 0;
static const char *g_cur = "";
#define EXPECT(c) do {                                                  \
    g_total++;                                                          \
    if (!(c)) {                                                         \
        g_fail++;                                                       \
        fprintf(stderr, "[FAIL] %s:%d in %s : %s\n",                    \
                __FILE__, __LINE__, g_cur, #c);                         \
    }                                                                   \
} while (0)
#define EXPECT_EQ_INT(a, b) do {                                        \
    long _a = (long)(a), _b = (long)(b);                                \
    g_total++;                                                          \
    if (_a != _b) {                                                     \
        g_fail++;                                                       \
        fprintf(stderr, "[FAIL] %s:%d in %s : got %ld want %ld\n",       \
                __FILE__, __LINE__, g_cur, _a, _b);                     \
    }                                                                   \
} while (0)
#define RUN(fn) do { g_cur = #fn; fn(); } while (0)

static void test_glyph_bank_complete(void) {
    EXPECT_EQ_INT(MRZ_OCR_GLYPH_COUNT, 37);
    /* All glyphs should be non-empty rows (sum > 0). */
    int non_empty = 0;
    for (int i = 0; i < MRZ_OCR_GLYPH_COUNT; ++i) {
        int sum = 0;
        for (int y = 0; y < MRZ_OCR_GLYPH_H; ++y)
            sum += __builtin_popcount(MRZ_OCR_GLYPHS[i].rows[y]);
        if (sum > 0) non_empty++;
    }
    EXPECT_EQ_INT(non_empty, 37);
}

static void test_glyph_index(void) {
    /* Digits -> 0..9. */
    for (int c = '0'; c <= '9'; ++c)
        EXPECT_EQ_INT(mrz_ocr_glyph_index(c), c - '0');
    /* Uppercase -> 10..35. */
    for (int c = 'A'; c <= 'Z'; ++c)
        EXPECT_EQ_INT(mrz_ocr_glyph_index(c), 10 + (c - 'A'));
    EXPECT_EQ_INT(mrz_ocr_glyph_index('<'), 36);
    /* Anything else -> -1. */
    EXPECT_EQ_INT(mrz_ocr_glyph_index('a'), -1);
    EXPECT_EQ_INT(mrz_ocr_glyph_index(' '), -1);
    EXPECT_EQ_INT(mrz_ocr_glyph_index('!'), -1);
}

static void test_glyph_chars_unique(void) {
    /* No duplicate characters in the bank. */
    for (int i = 0; i < MRZ_OCR_GLYPH_COUNT; ++i)
        for (int j = i + 1; j < MRZ_OCR_GLYPH_COUNT; ++j)
            EXPECT(MRZ_OCR_GLYPHS[i].ch != MRZ_OCR_GLYPHS[j].ch);
}

static void test_recognise_canonical(void) {
    /* Try to load the canonical corpus image (if present). We do not
     * require it -- if missing, this test is a no-op. */
    const char *path = "data/corpus/img_0001_p0_v0.ppm";
    face_image_t *img = face_image_load_ppm(path);
    if (!img) { fprintf(stderr, "(skip) no corpus image yet\n"); return; }
    mrz_ocr_init();
    mrz_ocr_result_t r;
    mrz_ocr_status_t s = mrz_ocr_recognise(img, &r);
    EXPECT_EQ_INT(s, (long)MRZ_OCR_OK);
    EXPECT(r.line1_len > 0);
    EXPECT(r.line2_len > 0);
    EXPECT(r.band_w > 0);
    face_image_free(img);
}

int main(void) {
    RUN(test_glyph_bank_complete);
    RUN(test_glyph_index);
    RUN(test_glyph_chars_unique);
    RUN(test_recognise_canonical);
    fprintf(stderr, "\n[TEST] %d total, %d failed\n", g_total, g_fail);
    return g_fail == 0 ? 0 : 1;
}
